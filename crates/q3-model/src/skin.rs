//! Skinning glTF — extraction des joints, poids, et animations.
//!
//! Les armes modernes en GLB peuvent contenir des animations squelettiques
//! (reload, idle sway, inspect) via le systeme skin/joints de glTF 2.0.
//! Ce module parse les donnees de skinning et fournit une API pour
//! evaluer les poses a un instant donne.
//!
//! # Architecture
//!
//! Le pipeline d'animation suit le schema classique :
//!
//! 1. **Parse** (`SkinData::from_gltf`) — extrait la hierarchie de joints,
//!    les inverse-bind matrices, les canaux d'animation (TRS keyframes),
//!    et les attributs JOINTS_0 / WEIGHTS_0 par vertex.
//!
//! 2. **Evaluate** (`SkinData::evaluate`) — a un instant `t` donne,
//!    interpole les keyframes (lerp pour T/S, slerp pour R), reconstruit
//!    les transforms locaux, propage la hierarchie parent->enfant, et
//!    applique les inverse-bind matrices pour produire un tableau de
//!    matrices "joint" pretes pour le GPU.
//!
//! 3. **Blend** (`SkinData::evaluate_blend`) — melange lineaire entre
//!    deux poses evaluees (transition douce entre "idle" et "reload"
//!    par exemple).
//!
//! 4. **GPU upload** — le tableau `EvaluatedPose::joint_matrices` est
//!    envoye dans un storage buffer et consomme par le compute shader
//!    `skinning.wgsl` qui transforme les vertices.
//!
//! # Conventions axes
//!
//! Les donnees sont PRESERVEES en espace glTF natif (Y-up).  La conversion
//! Y-up -> Z-up (conventions Q3) se fait au moment du rendu, cote engine,
//! via les matrices de vue/modele — exactement comme le GLB loader statique
//! (`glb.rs`).  Convertir ici casserait les orientations viewmodel tunees.
//!
//! # Limites
//!
//! * Maximum 64 joints par mesh (suffisant pour des armes FPS ; un
//!   personnage complet demanderait 128+).
//! * Interpolation lineaire (STEP) et cubique (CUBICSPLINE) non supportees —
//!   on utilise LINEAR pour tout.  C'est le mode par defaut de la majorite
//!   des exporteurs (Blender, Maya).
//! * Morph targets ignores (pas utilises par les armes modernes).

use q3_math::{Mat4, Quat, Vec3, Vec4};
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Constantes
// ---------------------------------------------------------------------------

/// Maximum joints supportes par mesh (correspond a la taille du buffer GPU
/// dans `skinning.wgsl` : `array<mat4x4<f32>, 64>`).
pub const MAX_JOINTS: usize = 64;

// ---------------------------------------------------------------------------
// Structures de donnees
// ---------------------------------------------------------------------------

/// Un joint dans la hierarchie squelettique.
///
/// Chaque joint a un index unique, un parent optionnel (None pour les racines),
/// une inverse-bind matrix (IBR) precomputee par l'exporteur, et sa transform
/// locale en pose de repos (rest pose).
#[derive(Debug, Clone)]
pub struct Joint {
    pub name: String,
    pub index: usize,
    /// Index du joint parent dans le tableau `SkinData::joints`.
    /// `None` pour les joints racine (typiquement 1 seul pour une arme).
    pub parent: Option<usize>,
    /// Matrice inverse-bind : transforme un vertex de l'espace modele vers
    /// l'espace local du joint en pose de repos.  Fournie directement par
    /// glTF (`skin.inverse_bind_matrices`).
    pub inverse_bind_matrix: Mat4,
    /// Transform locale en pose de repos (decomposee depuis le node glTF).
    /// Sert de fallback quand une animation ne cible pas ce joint.
    pub local_transform: Mat4,
}

/// Un keyframe d'animation — instant `time` avec les composantes TRS optionnelles.
///
/// glTF separe les canaux par propriete (translation, rotation, scale), donc
/// on reconstruit des keyframes "unifies" en fusionnant les canaux qui
/// partagent le meme joint.  Les composantes absentes sont `None` et on
/// utilise la valeur de la pose de repos a la place.
#[derive(Debug, Clone)]
pub struct Keyframe {
    pub time: f32,
    pub translation: Option<Vec3>,
    /// Quaternion (x, y, z, w) — stocke en Vec4 pour compatibilite avec
    /// les buffers, mais interpole via `Quat` (slerp).
    pub rotation: Option<Vec4>,
    pub scale: Option<Vec3>,
}

/// Un canal d'animation ciblant un joint specifique.
///
/// Les keyframes sont tries par temps croissant (garantie du loader).
/// L'interpolation se fait entre les deux keyframes encadrant `t`.
#[derive(Debug, Clone)]
pub struct AnimChannel {
    pub joint_index: usize,
    pub keyframes: Vec<Keyframe>,
}

/// Une animation nommee (ex: "reload", "idle", "fire", "inspect").
///
/// `duration` est le temps du dernier keyframe (pas forcement la fin
/// logique — certaines anims ont un keyframe de "retour" a t=duration).
/// `looping` est determine heuristiquement : les animations nommees
/// "idle", "walk", "run" sont marquees looping ; les autres non.
#[derive(Debug, Clone)]
pub struct Animation {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimChannel>,
    pub looping: bool,
}

/// Donnees de skinning completes d'un mesh GLB.
///
/// Contient tout ce qu'il faut pour animer le mesh cote CPU (evaluation
/// de pose) et alimenter le compute shader GPU (vertex_joints/weights).
#[derive(Debug, Clone)]
pub struct SkinData {
    /// Hierarchie de joints ordonnee par index (0..N).
    pub joints: Vec<Joint>,
    /// Animations disponibles (peut etre vide si le GLB a un skin mais
    /// pas d'animations — dans ce cas on reste en pose de repos).
    pub animations: Vec<Animation>,
    /// Indices de joints par vertex — 4 influences max (JOINTS_0 attribute).
    /// Longueur = nombre de vertices du mesh.
    pub vertex_joints: Vec<[u16; 4]>,
    /// Poids de skinning par vertex — 4 poids normalises (WEIGHTS_0 attribute).
    /// `sum(weights[i]) == 1.0` pour chaque vertex.
    pub vertex_weights: Vec<[f32; 4]>,
}

/// Pose evaluee — tableau de matrices joint pretes pour le GPU.
///
/// Chaque matrice transforme un vertex de l'espace modele (rest pose) vers
/// l'espace modele (pose animee).  Formule :
/// `joint_matrices[j] = world_transform[j] * inverse_bind_matrix[j]`
///
/// Le compute shader `skinning.wgsl` consomme ce tableau directement.
#[derive(Debug, Clone)]
pub struct EvaluatedPose {
    /// Une matrice par joint, indexee par `joint_index`.
    /// Taille = `SkinData::joints.len()`, padde a MAX_JOINTS pour le GPU
    /// (les entries au-dela de joints.len() sont Mat4::IDENTITY).
    pub joint_matrices: Vec<Mat4>,
}

impl EvaluatedPose {
    /// Retourne le tableau de matrices padde a MAX_JOINTS, pret pour
    /// l'upload dans le storage buffer GPU.
    pub fn gpu_matrices(&self) -> Vec<Mat4> {
        let mut out = self.joint_matrices.clone();
        out.resize(MAX_JOINTS, Mat4::IDENTITY);
        out
    }
}

// ---------------------------------------------------------------------------
// Parsing glTF -> SkinData
// ---------------------------------------------------------------------------

impl SkinData {
    /// Parse les donnees de skinning depuis un document glTF + ses buffers.
    ///
    /// Retourne `None` si le document n'a aucun skin (mesh statique pur).
    /// Prend le PREMIER skin et le PREMIER mesh trouves — suffisant pour
    /// les armes GLB qui sont des assets a un seul mesh + un seul skeleton.
    ///
    /// # Etapes de parsing
    ///
    /// 1. Lecture du skin glTF -> inverse-bind matrices + liste de joints (nodes)
    /// 2. Construction de la hierarchie parent/enfant via le scene graph
    /// 3. Lecture des attributs JOINTS_0 et WEIGHTS_0 sur les primitives du mesh
    /// 4. Lecture de toutes les animations et de leurs canaux (samplers)
    pub fn from_gltf(document: &gltf::Document, buffers: &[gltf::buffer::Data]) -> Option<Self> {
        // --- Etape 1 : trouver le premier skin ---
        let skin = document.skins().next()?;
        let reader = skin.reader(|b| Some(&buffers[b.index()]));

        // Les inverse-bind matrices sont fournies par le skin.  Si absentes
        // (cas rare mais legal en glTF), on utilise Mat4::IDENTITY.
        let ibms: Vec<Mat4> = reader
            .read_inverse_bind_matrices()
            .map(|iter| {
                iter.map(|cols| Mat4::from_cols_array_2d(&cols))
                    .collect()
            })
            .unwrap_or_else(|| vec![Mat4::IDENTITY; skin.joints().len()]);

        // --- Etape 2 : construire la hierarchie ---
        // On mappe chaque node index glTF -> index dans notre tableau de joints.
        let joint_nodes: Vec<gltf::Node<'_>> = skin.joints().collect();
        let joint_count = joint_nodes.len();

        if joint_count == 0 {
            debug!("skin: skin present mais 0 joints — ignore");
            return None;
        }
        if joint_count > MAX_JOINTS {
            warn!(
                "skin: {} joints depasse la limite MAX_JOINTS={} — tronque",
                joint_count, MAX_JOINTS
            );
        }
        let joint_count = joint_count.min(MAX_JOINTS);

        // Map : gltf node index -> notre joint index (0..joint_count)
        let mut node_to_joint: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::with_capacity(joint_count);
        for (j, node) in joint_nodes.iter().enumerate().take(joint_count) {
            node_to_joint.insert(node.index(), j);
        }

        // Construire les joints avec parent lookup.
        // Le parent d'un joint est le premier ancetre dans le scene graph
        // qui est aussi un joint du meme skin.  On cherche recursivement
        // dans les enfants de chaque node pour etablir la relation.
        //
        // Approche : pour chaque joint node, on parcourt les children du
        // scene graph et on cherche lequel des autres joints est un parent
        // direct.  Alternative plus simple : on itere tous les nodes du
        // document et on cherche les liens parent.
        let parent_map = build_parent_map(document, &node_to_joint);

        let mut joints = Vec::with_capacity(joint_count);
        for (j, node) in joint_nodes.iter().enumerate().take(joint_count) {
            let (t, r, s) = node.transform().decomposed();
            let local_transform = compose_trs(
                Vec3::from(t),
                Vec4::new(r[0], r[1], r[2], r[3]),
                Vec3::from(s),
            );
            joints.push(Joint {
                name: node.name().unwrap_or("").to_string(),
                index: j,
                parent: parent_map.get(&j).copied(),
                inverse_bind_matrix: ibms.get(j).copied().unwrap_or(Mat4::IDENTITY),
                local_transform,
            });
        }

        debug!(
            "skin: {} joints parses (racines: {})",
            joints.len(),
            joints.iter().filter(|j| j.parent.is_none()).count()
        );

        // --- Etape 3 : lire JOINTS_0 / WEIGHTS_0 sur le mesh ---
        let mut vertex_joints: Vec<[u16; 4]> = Vec::new();
        let mut vertex_weights: Vec<[f32; 4]> = Vec::new();

        if let Some(mesh) = document.meshes().next() {
            for prim in mesh.primitives() {
                let prim_reader = prim.reader(|b| Some(&buffers[b.index()]));

                // JOINTS_0 : indices de joints par vertex (u8 ou u16)
                if let Some(joints_iter) = prim_reader.read_joints(0) {
                    let prim_joints: Vec<[u16; 4]> = joints_iter.into_u16().collect();
                    vertex_joints.extend_from_slice(&prim_joints);
                }

                // WEIGHTS_0 : poids normalises par vertex
                if let Some(weights_iter) = prim_reader.read_weights(0) {
                    let prim_weights: Vec<[f32; 4]> = weights_iter.into_f32().collect();
                    vertex_weights.extend_from_slice(&prim_weights);
                }
            }
        }

        // Verification coherence : joints et weights doivent avoir meme longueur
        if vertex_joints.len() != vertex_weights.len() {
            warn!(
                "skin: JOINTS_0 ({}) et WEIGHTS_0 ({}) n'ont pas la meme longueur — ignore skin",
                vertex_joints.len(),
                vertex_weights.len()
            );
            return None;
        }

        debug!(
            "skin: {} vertices avec donnees de skinning",
            vertex_joints.len()
        );

        // --- Etape 4 : parser les animations ---
        let animations = parse_animations(document, buffers, &node_to_joint);
        debug!("skin: {} animations parsees", animations.len());
        for anim in &animations {
            debug!(
                "  - '{}' : {:.2}s, {} canaux, looping={}",
                anim.name,
                anim.duration,
                anim.channels.len(),
                anim.looping
            );
        }

        Some(SkinData {
            joints,
            animations,
            vertex_joints,
            vertex_weights,
        })
    }

    /// Evalue la pose a un instant donne pour une animation nommee.
    ///
    /// Si l'animation est `looping`, le temps wrappe automatiquement.
    /// Si l'animation n'est pas trouvee, retourne la pose de repos.
    ///
    /// # Algorithme
    ///
    /// 1. Pour chaque canal de l'animation, interpole les keyframes a `time`
    ///    (lerp pour translation/scale, slerp pour rotation).
    /// 2. Compose les transforms locaux (T*R*S) pour chaque joint.
    /// 3. Propage la hierarchie : `world[j] = world[parent[j]] * local[j]`.
    /// 4. Applique l'inverse-bind : `joint_matrix[j] = world[j] * ibm[j]`.
    pub fn evaluate(&self, anim_name: &str, time: f32) -> EvaluatedPose {
        let joint_count = self.joints.len();

        // Chercher l'animation par nom
        let anim = match self.animations.iter().find(|a| a.name == anim_name) {
            Some(a) => a,
            None => {
                // Animation non trouvee — retourne la pose de repos
                // (chaque joint_matrix = world_rest * ibm)
                return self.rest_pose();
            }
        };

        // Wrap du temps pour les animations looping
        let t = if anim.looping && anim.duration > 0.0 {
            time % anim.duration
        } else {
            time.min(anim.duration)
        };

        // Collecter les transforms locaux interpoles.
        // On part de la pose de repos et on override avec les valeurs animees.
        let mut local_transforms: Vec<(Vec3, Quat, Vec3)> = self
            .joints
            .iter()
            .map(|j| decompose_mat4(j.local_transform))
            .collect();

        // Appliquer chaque canal d'animation
        for channel in &anim.channels {
            if channel.joint_index >= joint_count {
                continue;
            }
            let (ref_t, ref_r, ref_s) = &mut local_transforms[channel.joint_index];

            // Interpole les keyframes a l'instant t
            let (interp_t, interp_r, interp_s) = interpolate_keyframes(&channel.keyframes, t);

            if let Some(translation) = interp_t {
                *ref_t = translation;
            }
            if let Some(rotation) = interp_r {
                *ref_r = rotation;
            }
            if let Some(scale) = interp_s {
                *ref_s = scale;
            }
        }

        // Recomposer les matrices locales depuis TRS
        let local_matrices: Vec<Mat4> = local_transforms
            .iter()
            .map(|(t, r, s)| {
                compose_trs(
                    *t,
                    Vec4::new(r.x, r.y, r.z, r.w),
                    *s,
                )
            })
            .collect();

        // Propager la hierarchie parent -> enfant pour obtenir les world transforms
        let world_transforms = propagate_hierarchy(&self.joints, &local_matrices);

        // Calculer les joint matrices finales : world * inverse_bind
        let joint_matrices: Vec<Mat4> = self
            .joints
            .iter()
            .enumerate()
            .map(|(i, joint)| world_transforms[i] * joint.inverse_bind_matrix)
            .collect();

        EvaluatedPose { joint_matrices }
    }

    /// Evalue un blend entre deux animations (pour transitions douces).
    ///
    /// `blend` dans [0, 1] : 0 = 100% anim_a, 1 = 100% anim_b.
    /// Le blend se fait par interpolation lineaire des matrices resultantes.
    ///
    /// Note : un vrai blend devrait interpoler les quaternions de rotation
    /// separement (slerp) pour eviter les artefacts de scale.  Mais pour
    /// des transitions courtes (0.2-0.3s) sur des armes, le lerp matriciel
    /// est visuellement acceptable et beaucoup plus simple.
    pub fn evaluate_blend(
        &self,
        anim_a: &str,
        time_a: f32,
        anim_b: &str,
        time_b: f32,
        blend: f32,
    ) -> EvaluatedPose {
        let pose_a = self.evaluate(anim_a, time_a);
        let pose_b = self.evaluate(anim_b, time_b);
        let t = blend.clamp(0.0, 1.0);

        // Lerp lineaire de chaque matrice joint
        let joint_matrices: Vec<Mat4> = pose_a
            .joint_matrices
            .iter()
            .zip(pose_b.joint_matrices.iter())
            .map(|(a, b)| mat4_lerp(*a, *b, t))
            .collect();

        EvaluatedPose { joint_matrices }
    }

    /// Retourne la liste des noms d'animations disponibles.
    pub fn animation_names(&self) -> Vec<&str> {
        self.animations.iter().map(|a| a.name.as_str()).collect()
    }

    /// Retourne true si le mesh a des donnees de skinning utilisables.
    pub fn has_skin(&self) -> bool {
        !self.joints.is_empty()
    }

    /// Retourne la duree d'une animation par nom.  `None` si non trouvee.
    pub fn animation_duration(&self, name: &str) -> Option<f32> {
        self.animations
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.duration)
    }

    /// Pose de repos (rest pose) — toutes les animations ignorees,
    /// chaque joint est dans sa position par defaut du scene graph.
    fn rest_pose(&self) -> EvaluatedPose {
        let local_matrices: Vec<Mat4> =
            self.joints.iter().map(|j| j.local_transform).collect();
        let world_transforms = propagate_hierarchy(&self.joints, &local_matrices);

        let joint_matrices: Vec<Mat4> = self
            .joints
            .iter()
            .enumerate()
            .map(|(i, joint)| world_transforms[i] * joint.inverse_bind_matrix)
            .collect();

        EvaluatedPose { joint_matrices }
    }
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

/// Construit une map joint_index -> parent_joint_index en parcourant
/// le scene graph du document glTF.
///
/// Pour chaque node qui est un joint, on cherche son noeud parent dans
/// le scene graph.  Si ce parent est aussi un joint du meme skin,
/// on enregistre la relation.
fn build_parent_map(
    document: &gltf::Document,
    node_to_joint: &std::collections::HashMap<usize, usize>,
) -> std::collections::HashMap<usize, usize> {
    let mut parent_map = std::collections::HashMap::new();

    // Parcourir tous les nodes et construire une map node_index -> parent_node_index
    let mut node_parent: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();

    for node in document.nodes() {
        for child in node.children() {
            node_parent.insert(child.index(), node.index());
        }
    }

    // Pour chaque joint, remonter la chaine de parents dans le scene graph
    // jusqu'a trouver un parent qui est aussi un joint
    for (&node_idx, &joint_idx) in node_to_joint.iter() {
        let mut current = node_idx;
        while let Some(&parent_node) = node_parent.get(&current) {
            if let Some(&parent_joint) = node_to_joint.get(&parent_node) {
                // Trouve : ce joint a un parent dans le skin
                parent_map.insert(joint_idx, parent_joint);
                break;
            }
            // Remonter d'un cran
            current = parent_node;
        }
        // Si on sort de la boucle sans trouver, le joint est une racine (pas de parent)
    }

    parent_map
}

/// Parse toutes les animations du document glTF et les convertit en
/// notre format `Animation`.
///
/// Chaque animation glTF contient des canaux (channel) qui ciblent
/// un node (= un joint) et une propriete (translation, rotation, scale).
/// On fusionne les canaux par joint pour creer nos `AnimChannel` avec
/// des `Keyframe` unifies.
fn parse_animations(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    node_to_joint: &std::collections::HashMap<usize, usize>,
) -> Vec<Animation> {
    let mut animations = Vec::new();

    for anim in document.animations() {
        let name = anim.name().unwrap_or("unnamed").to_string();

        // Collecte temporaire : joint_index -> (times, translations, rotations, scales)
        // On accumule les donnees brutes par joint avant de fusionner en keyframes.
        let mut channel_data: std::collections::HashMap<
            usize,
            ChannelAccumulator,
        > = std::collections::HashMap::new();

        for channel in anim.channels() {
            let target = channel.target();
            let node_idx = target.node().index();

            // Ignorer les canaux qui ne ciblent pas un joint de notre skin
            let joint_idx = match node_to_joint.get(&node_idx) {
                Some(&j) => j,
                None => continue,
            };

            let reader = channel.reader(|b| Some(&buffers[b.index()]));

            // Lire les timestamps (inputs du sampler)
            let times: Vec<f32> = match reader.read_inputs() {
                Some(iter) => iter.collect(),
                None => continue,
            };

            let acc = channel_data
                .entry(joint_idx)
                .or_insert_with(ChannelAccumulator::new);

            // Lire les outputs selon la propriete ciblee
            match reader.read_outputs() {
                Some(gltf::animation::util::ReadOutputs::Translations(iter)) => {
                    let values: Vec<Vec3> = iter.map(Vec3::from).collect();
                    acc.add_translations(&times, &values);
                }
                Some(gltf::animation::util::ReadOutputs::Rotations(iter)) => {
                    // Rotations en quaternion [x, y, z, w] normalises
                    let values: Vec<Vec4> =
                        iter.into_f32().map(|r| Vec4::new(r[0], r[1], r[2], r[3])).collect();
                    acc.add_rotations(&times, &values);
                }
                Some(gltf::animation::util::ReadOutputs::Scales(iter)) => {
                    let values: Vec<Vec3> = iter.map(Vec3::from).collect();
                    acc.add_scales(&times, &values);
                }
                Some(gltf::animation::util::ReadOutputs::MorphTargetWeights(_)) => {
                    // Morph targets ignores — pas utilises par les armes
                }
                None => {}
            }
        }

        // Fusionner les donnees accumulees en AnimChannels avec Keyframes unifies
        let mut channels = Vec::new();
        let mut max_duration: f32 = 0.0;

        for (joint_idx, acc) in channel_data {
            let keyframes = acc.build_keyframes();
            if let Some(last) = keyframes.last() {
                max_duration = max_duration.max(last.time);
            }
            channels.push(AnimChannel {
                joint_index: joint_idx,
                keyframes,
            });
        }

        // Heuristique looping : les animations dont le nom contient "idle",
        // "walk", "run", "sway" sont typiquement en boucle.  Les autres
        // ("reload", "fire", "inspect", "draw") sont one-shot.
        let looping = {
            let lower = name.to_lowercase();
            lower.contains("idle")
                || lower.contains("walk")
                || lower.contains("run")
                || lower.contains("sway")
                || lower.contains("loop")
        };

        if !channels.is_empty() {
            animations.push(Animation {
                name,
                duration: max_duration,
                channels,
                looping,
            });
        }
    }

    animations
}

/// Accumulateur temporaire pour fusionner les canaux TRS par joint.
///
/// glTF separe translation, rotation et scale en canaux distincts avec
/// potentiellement des timestamps differents.  On collecte tout puis on
/// fusionne sur une timeline unifiee.
struct ChannelAccumulator {
    translations: Vec<(f32, Vec3)>,
    rotations: Vec<(f32, Vec4)>,
    scales: Vec<(f32, Vec3)>,
}

impl ChannelAccumulator {
    fn new() -> Self {
        Self {
            translations: Vec::new(),
            rotations: Vec::new(),
            scales: Vec::new(),
        }
    }

    fn add_translations(&mut self, times: &[f32], values: &[Vec3]) {
        for (t, v) in times.iter().zip(values.iter()) {
            self.translations.push((*t, *v));
        }
    }

    fn add_rotations(&mut self, times: &[f32], values: &[Vec4]) {
        for (t, v) in times.iter().zip(values.iter()) {
            self.rotations.push((*t, *v));
        }
    }

    fn add_scales(&mut self, times: &[f32], values: &[Vec3]) {
        for (t, v) in times.iter().zip(values.iter()) {
            self.scales.push((*t, *v));
        }
    }

    /// Fusionne les canaux TRS en keyframes unifies.
    ///
    /// Strategie : on collecte tous les timestamps uniques de tous les canaux,
    /// puis pour chaque timestamp on sample (interpole) chaque propriete.
    /// Si un canal n'a pas de donnees pour ce timestamp, le keyframe
    /// marque la propriete comme `None`.
    fn build_keyframes(&self) -> Vec<Keyframe> {
        // Collecter tous les timestamps uniques
        let mut all_times: Vec<f32> = Vec::new();
        for (t, _) in &self.translations {
            all_times.push(*t);
        }
        for (t, _) in &self.rotations {
            all_times.push(*t);
        }
        for (t, _) in &self.scales {
            all_times.push(*t);
        }

        // Deduplicate et trier
        all_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        all_times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

        let mut keyframes = Vec::with_capacity(all_times.len());
        for time in &all_times {
            keyframes.push(Keyframe {
                time: *time,
                translation: if self.translations.is_empty() {
                    None
                } else {
                    Some(sample_vec3(&self.translations, *time))
                },
                rotation: if self.rotations.is_empty() {
                    None
                } else {
                    Some(sample_rotation(&self.rotations, *time))
                },
                scale: if self.scales.is_empty() {
                    None
                } else {
                    Some(sample_vec3(&self.scales, *time))
                },
            });
        }

        keyframes
    }
}

/// Interpole lineairement dans une liste (time, Vec3) triee par temps.
fn sample_vec3(data: &[(f32, Vec3)], time: f32) -> Vec3 {
    if data.is_empty() {
        return Vec3::ZERO;
    }
    if data.len() == 1 || time <= data[0].0 {
        return data[0].1;
    }
    if time >= data.last().unwrap().0 {
        return data.last().unwrap().1;
    }
    // Trouver l'intervalle encadrant
    for window in data.windows(2) {
        let (t0, v0) = window[0];
        let (t1, v1) = window[1];
        if time >= t0 && time <= t1 {
            let dt = t1 - t0;
            if dt < 1e-9 {
                return v0;
            }
            let alpha = (time - t0) / dt;
            return v0.lerp(v1, alpha);
        }
    }
    data.last().unwrap().1
}

/// Interpole par slerp dans une liste (time, Vec4/quaternion) triee par temps.
fn sample_rotation(data: &[(f32, Vec4)], time: f32) -> Vec4 {
    if data.is_empty() {
        return Vec4::new(0.0, 0.0, 0.0, 1.0); // identite
    }
    if data.len() == 1 || time <= data[0].0 {
        return data[0].1;
    }
    if time >= data.last().unwrap().0 {
        return data.last().unwrap().1;
    }
    for window in data.windows(2) {
        let (t0, q0) = window[0];
        let (t1, q1) = window[1];
        if time >= t0 && time <= t1 {
            let dt = t1 - t0;
            if dt < 1e-9 {
                return q0;
            }
            let alpha = (time - t0) / dt;
            return quat_slerp(q0, q1, alpha);
        }
    }
    data.last().unwrap().1
}

// ---------------------------------------------------------------------------
// Evaluation helpers
// ---------------------------------------------------------------------------

/// Interpole les keyframes d'un canal a l'instant `t`.
///
/// Retourne les composantes TRS interpolees (None si le canal ne fournit
/// pas cette propriete).
fn interpolate_keyframes(keyframes: &[Keyframe], t: f32) -> (Option<Vec3>, Option<Quat>, Option<Vec3>) {
    if keyframes.is_empty() {
        return (None, None, None);
    }

    // Cas limite : avant le premier keyframe
    if t <= keyframes[0].time {
        let kf = &keyframes[0];
        return (
            kf.translation,
            kf.rotation.map(|v| Quat::from_xyzw(v.x, v.y, v.z, v.w)),
            kf.scale,
        );
    }

    // Cas limite : apres le dernier keyframe
    if t >= keyframes.last().unwrap().time {
        let kf = keyframes.last().unwrap();
        return (
            kf.translation,
            kf.rotation.map(|v| Quat::from_xyzw(v.x, v.y, v.z, v.w)),
            kf.scale,
        );
    }

    // Trouver l'intervalle encadrant [kf_a, kf_b]
    for window in keyframes.windows(2) {
        let kf_a = &window[0];
        let kf_b = &window[1];

        if t >= kf_a.time && t <= kf_b.time {
            let dt = kf_b.time - kf_a.time;
            let alpha = if dt < 1e-9 { 0.0 } else { (t - kf_a.time) / dt };

            // Translation : lerp
            let translation = match (kf_a.translation, kf_b.translation) {
                (Some(a), Some(b)) => Some(a.lerp(b, alpha)),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };

            // Rotation : slerp (via Quat de glam pour la precision)
            let rotation = match (kf_a.rotation, kf_b.rotation) {
                (Some(a), Some(b)) => {
                    let qa = Quat::from_xyzw(a.x, a.y, a.z, a.w).normalize();
                    let qb = Quat::from_xyzw(b.x, b.y, b.z, b.w).normalize();
                    // Slerp avec chemin le plus court (dot negatif -> flip)
                    let result = if qa.dot(qb) < 0.0 {
                        qa.slerp(-qb, alpha)
                    } else {
                        qa.slerp(qb, alpha)
                    };
                    Some(result)
                }
                (Some(a), None) => Some(Quat::from_xyzw(a.x, a.y, a.z, a.w)),
                (None, Some(b)) => Some(Quat::from_xyzw(b.x, b.y, b.z, b.w)),
                (None, None) => None,
            };

            // Scale : lerp
            let scale = match (kf_a.scale, kf_b.scale) {
                (Some(a), Some(b)) => Some(a.lerp(b, alpha)),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };

            return (translation, rotation, scale);
        }
    }

    // Fallback (ne devrait pas arriver)
    let kf = keyframes.last().unwrap();
    (
        kf.translation,
        kf.rotation.map(|v| Quat::from_xyzw(v.x, v.y, v.z, v.w)),
        kf.scale,
    )
}

/// Propage la hierarchie de joints pour calculer les world transforms.
///
/// Chaque joint world = parent_world * local.
/// Les joints doivent etre ordonnes de sorte que le parent apparaisse
/// AVANT ses enfants dans le tableau.  C'est generalement le cas car
/// glTF ordonne les joints du skin du parent vers les feuilles.
///
/// Si ce n'est pas le cas (glTF ne le garantit pas strictement), on fait
/// plusieurs passes jusqu'a convergence — maximum `joints.len()` passes
/// pour etre safe (en pratique 1-2 suffisent).
fn propagate_hierarchy(joints: &[Joint], local_matrices: &[Mat4]) -> Vec<Mat4> {
    let count = joints.len();
    let mut world = vec![Mat4::IDENTITY; count];
    let mut resolved = vec![false; count];
    let mut remaining = count;

    // Iterative resolution : d'abord les racines, puis les enfants
    // quand leur parent est resolu.  Maximum `count` iterations pour
    // gerer n'importe quel ordre de joints.
    for _pass in 0..count {
        if remaining == 0 {
            break;
        }
        for i in 0..count {
            if resolved[i] {
                continue;
            }
            match joints[i].parent {
                None => {
                    // Joint racine : world = local
                    world[i] = local_matrices[i];
                    resolved[i] = true;
                    remaining -= 1;
                }
                Some(parent_idx) => {
                    if resolved[parent_idx] {
                        world[i] = world[parent_idx] * local_matrices[i];
                        resolved[i] = true;
                        remaining -= 1;
                    }
                    // Sinon on attend la prochaine passe
                }
            }
        }
    }

    // Si des joints n'ont pas ete resolus (cycle dans la hierarchie?),
    // on les laisse a IDENTITY et on avertit.
    if remaining > 0 {
        warn!(
            "skin: {} joints non resolus apres propagation (cycle dans la hierarchie?)",
            remaining
        );
    }

    world
}

// ---------------------------------------------------------------------------
// Quaternion / TRS helpers
// ---------------------------------------------------------------------------

/// Slerp entre deux quaternions stockes en Vec4 (x, y, z, w).
///
/// Utilise le chemin le plus court (flip si dot negatif).
/// Fallback vers lerp normalise pour les angles tres petits (evite
/// la division par zero quand sin(theta) -> 0).
fn quat_slerp(a: Vec4, b: Vec4, t: f32) -> Vec4 {
    let mut dot = a.dot(b);

    // Chemin le plus court : si dot negatif, on flippe b
    let b_adj = if dot < 0.0 {
        dot = -dot;
        -b
    } else {
        b
    };

    // Pour des angles tres petits, slerp degenere — on utilise lerp normalise
    if dot > 0.9995 {
        let result = a + (b_adj - a) * t;
        return result.normalize();
    }

    // Slerp standard : sin-based interpolation
    let theta = dot.clamp(-1.0, 1.0).acos();
    let sin_theta = theta.sin();

    if sin_theta.abs() < 1e-9 {
        // Angle ~0 ou ~pi : lerp normalise comme fallback
        let result = a + (b_adj - a) * t;
        return result.normalize();
    }

    let factor_a = ((1.0 - t) * theta).sin() / sin_theta;
    let factor_b = (t * theta).sin() / sin_theta;

    a * factor_a + b_adj * factor_b
}

/// Convertit un quaternion Vec4 (x, y, z, w) en Mat4 de rotation.
///
/// Formule standard de conversion quaternion -> matrice de rotation 4x4.
/// Pas de normalisation prealable — le quaternion doit deja etre unitaire.
fn quat_to_mat4(q: Vec4) -> Mat4 {
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;

    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    // Colonnes de la matrice de rotation (column-major, convention glam)
    Mat4::from_cols(
        Vec4::new(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy), 0.0),
        Vec4::new(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx), 0.0),
        Vec4::new(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy), 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    )
}

/// Compose une matrice TRS (Translation * Rotation * Scale) depuis les
/// composantes separees.
///
/// `rotation` est un quaternion (x, y, z, w) en Vec4.
/// L'ordre est celui de glTF : T * R * S.
fn compose_trs(translation: Vec3, rotation: Vec4, scale: Vec3) -> Mat4 {
    let t = Mat4::from_translation(translation);
    let r = quat_to_mat4(rotation);
    let s = Mat4::from_scale(scale);
    t * r * s
}

/// Decompose une Mat4 en (translation, rotation_quat, scale).
///
/// Suppose que la matrice est une transform TRS valide (pas de shear).
/// Extraction : translation = colonne 3, scale = longueur des colonnes 0-2,
/// rotation = matrice 3x3 normalisee -> quaternion.
fn decompose_mat4(m: Mat4) -> (Vec3, Quat, Vec3) {
    let translation = Vec3::new(m.col(3).x, m.col(3).y, m.col(3).z);

    let col0 = Vec3::new(m.col(0).x, m.col(0).y, m.col(0).z);
    let col1 = Vec3::new(m.col(1).x, m.col(1).y, m.col(1).z);
    let col2 = Vec3::new(m.col(2).x, m.col(2).y, m.col(2).z);

    let scale_x = col0.length();
    let scale_y = col1.length();
    let scale_z = col2.length();
    let scale = Vec3::new(scale_x, scale_y, scale_z);

    // Matrice de rotation normalisee (diviser chaque colonne par son scale)
    let rot_mat = if scale_x > 1e-9 && scale_y > 1e-9 && scale_z > 1e-9 {
        q3_math::Mat3::from_cols(col0 / scale_x, col1 / scale_y, col2 / scale_z)
    } else {
        q3_math::Mat3::IDENTITY
    };

    let rotation = Quat::from_mat3(&rot_mat);

    (translation, rotation, scale)
}

/// Interpolation lineaire de deux Mat4 composante par composante.
///
/// Utilisee pour le blend entre deux poses.  Ce n'est pas mathematiquement
/// "correct" (on devrait decomposer en TRS et interpoler separement),
/// mais pour des transitions courtes c'est visuellement acceptable.
fn mat4_lerp(a: Mat4, b: Mat4, t: f32) -> Mat4 {
    // glam Mat4 ne fournit pas de lerp directement — on interpole les colonnes
    Mat4::from_cols(
        a.col(0).lerp(b.col(0), t),
        a.col(1).lerp(b.col(1), t),
        a.col(2).lerp(b.col(2), t),
        a.col(3).lerp(b.col(3), t),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Tests quaternion slerp ---

    #[test]
    fn slerp_identity_at_t0() {
        let a = Vec4::new(0.0, 0.0, 0.0, 1.0); // identite
        let b = Vec4::new(0.0, 0.7071, 0.0, 0.7071); // 90 deg autour de Y
        let result = quat_slerp(a, b, 0.0);
        assert!((result - a).length() < 1e-4, "result = {:?}", result);
    }

    #[test]
    fn slerp_target_at_t1() {
        let a = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let b = Vec4::new(0.0, 0.7071, 0.0, 0.7071);
        let result = quat_slerp(a, b, 1.0);
        assert!((result - b).length() < 1e-3, "result = {:?}", result);
    }

    #[test]
    fn slerp_midpoint_is_unit() {
        let a = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let b = Vec4::new(0.0, 0.7071, 0.0, 0.7071);
        let mid = quat_slerp(a, b, 0.5);
        assert!(
            (mid.length() - 1.0).abs() < 1e-4,
            "mid length = {}",
            mid.length()
        );
    }

    #[test]
    fn slerp_shortest_path_with_negative_dot() {
        // Quaternions opposes representent la meme rotation mais le dot est negatif.
        // Le slerp doit choisir le chemin le plus court.
        let a = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let b = Vec4::new(0.0, 0.0, 0.0, -1.0); // meme rotation, signe oppose
        let mid = quat_slerp(a, b, 0.5);
        // Le resultat doit etre proche de a (ou -a) car c'est la meme rotation
        let qa = Quat::from_xyzw(a.x, a.y, a.z, a.w);
        let qr = Quat::from_xyzw(mid.x, mid.y, mid.z, mid.w);
        assert!(
            qa.dot(qr).abs() > 0.999,
            "devrait etre quasi-identite, got dot = {}",
            qa.dot(qr)
        );
    }

    // --- Tests compose_trs ---

    #[test]
    fn compose_trs_identity() {
        let m = compose_trs(
            Vec3::ZERO,
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec3::ONE,
        );
        let diff = (m - Mat4::IDENTITY).abs_diff_eq(Mat4::ZERO, 1e-5);
        assert!(diff, "expected identity, got {:?}", m);
    }

    #[test]
    fn compose_trs_translation_only() {
        let m = compose_trs(
            Vec3::new(1.0, 2.0, 3.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec3::ONE,
        );
        let expected = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        assert!(
            m.abs_diff_eq(expected, 1e-5),
            "expected {:?}, got {:?}",
            expected,
            m
        );
    }

    #[test]
    fn compose_trs_scale_only() {
        let m = compose_trs(
            Vec3::ZERO,
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec3::new(2.0, 3.0, 4.0),
        );
        let expected = Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0));
        assert!(
            m.abs_diff_eq(expected, 1e-5),
            "expected {:?}, got {:?}",
            expected,
            m
        );
    }

    // --- Tests decompose_mat4 ---

    #[test]
    fn decompose_identity() {
        let (t, r, s) = decompose_mat4(Mat4::IDENTITY);
        assert!(t.length() < 1e-5);
        assert!((r.dot(Quat::IDENTITY)).abs() > 0.999);
        assert!((s - Vec3::ONE).length() < 1e-5);
    }

    #[test]
    fn decompose_roundtrip() {
        let original_t = Vec3::new(1.0, 2.0, 3.0);
        let original_r = Vec4::new(0.0, 0.3827, 0.0, 0.9239); // ~45 deg Y
        let original_s = Vec3::new(1.5, 2.0, 0.5);

        let m = compose_trs(original_t, original_r, original_s);
        let (t, r, s) = decompose_mat4(m);

        assert!(
            (t - original_t).length() < 1e-3,
            "translation: expected {:?}, got {:?}",
            original_t,
            t
        );
        assert!(
            (s - original_s).length() < 1e-3,
            "scale: expected {:?}, got {:?}",
            original_s,
            s
        );

        let orig_q = Quat::from_xyzw(original_r.x, original_r.y, original_r.z, original_r.w)
            .normalize();
        assert!(
            r.dot(orig_q).abs() > 0.999,
            "rotation: expected {:?}, got {:?}",
            orig_q,
            r
        );
    }

    // --- Tests mat4_lerp ---

    #[test]
    fn mat4_lerp_at_zero_is_a() {
        let a = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let b = Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0));
        let result = mat4_lerp(a, b, 0.0);
        assert!(result.abs_diff_eq(a, 1e-5));
    }

    #[test]
    fn mat4_lerp_at_one_is_b() {
        let a = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let b = Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0));
        let result = mat4_lerp(a, b, 1.0);
        assert!(result.abs_diff_eq(b, 1e-5));
    }

    #[test]
    fn mat4_lerp_midpoint() {
        let a = Mat4::from_translation(Vec3::ZERO);
        let b = Mat4::from_translation(Vec3::new(10.0, 20.0, 30.0));
        let mid = mat4_lerp(a, b, 0.5);
        let expected = Mat4::from_translation(Vec3::new(5.0, 10.0, 15.0));
        assert!(mid.abs_diff_eq(expected, 1e-5));
    }

    // --- Tests quat_to_mat4 ---

    #[test]
    fn quat_to_mat4_identity() {
        let m = quat_to_mat4(Vec4::new(0.0, 0.0, 0.0, 1.0));
        assert!(
            m.abs_diff_eq(Mat4::IDENTITY, 1e-5),
            "expected identity, got {:?}",
            m
        );
    }

    #[test]
    fn quat_to_mat4_90deg_z() {
        // 90 deg autour de Z : quat = (0, 0, sin(45), cos(45)) = (0, 0, 0.7071, 0.7071)
        let q = Vec4::new(0.0, 0.0, std::f32::consts::FRAC_PI_4.sin(), std::f32::consts::FRAC_PI_4.cos());
        let m = quat_to_mat4(q);
        let expected = Mat4::from_quat(Quat::from_xyzw(q.x, q.y, q.z, q.w));
        assert!(
            m.abs_diff_eq(expected, 1e-4),
            "expected {:?}, got {:?}",
            expected,
            m
        );
    }

    // --- Tests propagate_hierarchy ---

    #[test]
    fn propagate_single_root() {
        let joints = vec![Joint {
            name: "root".to_string(),
            index: 0,
            parent: None,
            inverse_bind_matrix: Mat4::IDENTITY,
            local_transform: Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        }];
        let locals = vec![Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))];
        let world = propagate_hierarchy(&joints, &locals);
        assert!(world[0].abs_diff_eq(locals[0], 1e-5));
    }

    #[test]
    fn propagate_parent_child_chain() {
        let joints = vec![
            Joint {
                name: "root".to_string(),
                index: 0,
                parent: None,
                inverse_bind_matrix: Mat4::IDENTITY,
                local_transform: Mat4::IDENTITY,
            },
            Joint {
                name: "child".to_string(),
                index: 1,
                parent: Some(0),
                inverse_bind_matrix: Mat4::IDENTITY,
                local_transform: Mat4::IDENTITY,
            },
        ];
        let locals = vec![
            Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0)),
        ];
        let world = propagate_hierarchy(&joints, &locals);

        // root world = (1, 0, 0)
        assert!(world[0].abs_diff_eq(locals[0], 1e-5));

        // child world = root * child = translate(1,0,0) * translate(0,2,0) = translate(1,2,0)
        let expected = Mat4::from_translation(Vec3::new(1.0, 2.0, 0.0));
        assert!(
            world[1].abs_diff_eq(expected, 1e-5),
            "expected {:?}, got {:?}",
            expected,
            world[1]
        );
    }

    #[test]
    fn propagate_unordered_joints() {
        // Test que la propagation fonctionne meme si l'enfant est avant le parent
        // dans le tableau (ordre inverse)
        let joints = vec![
            Joint {
                name: "child".to_string(),
                index: 0,
                parent: Some(1), // parent est a l'index 1 (apres nous)
                inverse_bind_matrix: Mat4::IDENTITY,
                local_transform: Mat4::IDENTITY,
            },
            Joint {
                name: "root".to_string(),
                index: 1,
                parent: None,
                inverse_bind_matrix: Mat4::IDENTITY,
                local_transform: Mat4::IDENTITY,
            },
        ];
        let locals = vec![
            Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0)), // child
            Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)), // root
        ];
        let world = propagate_hierarchy(&joints, &locals);

        // root (index 1) : world = (1, 0, 0)
        let expected_root = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        assert!(world[1].abs_diff_eq(expected_root, 1e-5));

        // child (index 0) : world = root * child = translate(1, 2, 0)
        let expected_child = Mat4::from_translation(Vec3::new(1.0, 2.0, 0.0));
        assert!(
            world[0].abs_diff_eq(expected_child, 1e-5),
            "expected {:?}, got {:?}",
            expected_child,
            world[0]
        );
    }

    // --- Tests SkinData methods ---

    #[test]
    fn skin_data_has_skin_empty() {
        let skin = SkinData {
            joints: Vec::new(),
            animations: Vec::new(),
            vertex_joints: Vec::new(),
            vertex_weights: Vec::new(),
        };
        assert!(!skin.has_skin());
    }

    #[test]
    fn skin_data_animation_names() {
        let skin = SkinData {
            joints: vec![Joint {
                name: "root".to_string(),
                index: 0,
                parent: None,
                inverse_bind_matrix: Mat4::IDENTITY,
                local_transform: Mat4::IDENTITY,
            }],
            animations: vec![
                Animation {
                    name: "idle".to_string(),
                    duration: 2.0,
                    channels: Vec::new(),
                    looping: true,
                },
                Animation {
                    name: "fire".to_string(),
                    duration: 0.5,
                    channels: Vec::new(),
                    looping: false,
                },
            ],
            vertex_joints: Vec::new(),
            vertex_weights: Vec::new(),
        };
        let names = skin.animation_names();
        assert_eq!(names, vec!["idle", "fire"]);
        assert!(skin.has_skin());
        assert_eq!(skin.animation_duration("idle"), Some(2.0));
        assert_eq!(skin.animation_duration("fire"), Some(0.5));
        assert_eq!(skin.animation_duration("nonexistent"), None);
    }

    #[test]
    fn evaluated_pose_gpu_matrices_padded() {
        let pose = EvaluatedPose {
            joint_matrices: vec![Mat4::IDENTITY; 3],
        };
        let gpu = pose.gpu_matrices();
        assert_eq!(gpu.len(), MAX_JOINTS);
        for m in &gpu {
            assert!(m.abs_diff_eq(Mat4::IDENTITY, 1e-5));
        }
    }

    // --- Tests interpolation ---

    #[test]
    fn sample_vec3_single_value() {
        let data = vec![(0.0, Vec3::new(1.0, 2.0, 3.0))];
        let v = sample_vec3(&data, 0.5);
        assert!((v - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
    }

    #[test]
    fn sample_vec3_interpolation() {
        let data = vec![
            (0.0, Vec3::ZERO),
            (1.0, Vec3::new(10.0, 20.0, 30.0)),
        ];
        let v = sample_vec3(&data, 0.5);
        assert!((v - Vec3::new(5.0, 10.0, 15.0)).length() < 1e-4);
    }

    #[test]
    fn sample_vec3_clamp_before() {
        let data = vec![
            (1.0, Vec3::new(1.0, 0.0, 0.0)),
            (2.0, Vec3::new(2.0, 0.0, 0.0)),
        ];
        let v = sample_vec3(&data, 0.0); // avant le premier keyframe
        assert!((v - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn sample_vec3_clamp_after() {
        let data = vec![
            (0.0, Vec3::new(1.0, 0.0, 0.0)),
            (1.0, Vec3::new(2.0, 0.0, 0.0)),
        ];
        let v = sample_vec3(&data, 5.0); // apres le dernier keyframe
        assert!((v - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-5);
    }

    // --- Test rest pose ---

    #[test]
    fn rest_pose_identity_joints() {
        let skin = SkinData {
            joints: vec![
                Joint {
                    name: "root".to_string(),
                    index: 0,
                    parent: None,
                    inverse_bind_matrix: Mat4::IDENTITY,
                    local_transform: Mat4::IDENTITY,
                },
            ],
            animations: Vec::new(),
            vertex_joints: Vec::new(),
            vertex_weights: Vec::new(),
        };
        let pose = skin.rest_pose();
        // world = IDENTITY, ibm = IDENTITY => joint_matrix = IDENTITY
        assert!(pose.joint_matrices[0].abs_diff_eq(Mat4::IDENTITY, 1e-5));
    }

    #[test]
    fn evaluate_missing_anim_returns_rest_pose() {
        let skin = SkinData {
            joints: vec![Joint {
                name: "root".to_string(),
                index: 0,
                parent: None,
                inverse_bind_matrix: Mat4::IDENTITY,
                local_transform: Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            }],
            animations: Vec::new(),
            vertex_joints: Vec::new(),
            vertex_weights: Vec::new(),
        };
        let pose = skin.evaluate("nonexistent", 0.0);
        // rest_pose : world = translate(1,0,0), ibm = IDENTITY => joint = translate(1,0,0)
        let expected = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        assert!(pose.joint_matrices[0].abs_diff_eq(expected, 1e-5));
    }

    // --- Test blend ---

    #[test]
    fn blend_at_zero_is_anim_a() {
        let skin = SkinData {
            joints: vec![Joint {
                name: "root".to_string(),
                index: 0,
                parent: None,
                inverse_bind_matrix: Mat4::IDENTITY,
                local_transform: Mat4::IDENTITY,
            }],
            animations: vec![
                Animation {
                    name: "a".to_string(),
                    duration: 1.0,
                    channels: vec![AnimChannel {
                        joint_index: 0,
                        keyframes: vec![Keyframe {
                            time: 0.0,
                            translation: Some(Vec3::new(1.0, 0.0, 0.0)),
                            rotation: Some(Vec4::new(0.0, 0.0, 0.0, 1.0)),
                            scale: Some(Vec3::ONE),
                        }],
                    }],
                    looping: false,
                },
                Animation {
                    name: "b".to_string(),
                    duration: 1.0,
                    channels: vec![AnimChannel {
                        joint_index: 0,
                        keyframes: vec![Keyframe {
                            time: 0.0,
                            translation: Some(Vec3::new(10.0, 0.0, 0.0)),
                            rotation: Some(Vec4::new(0.0, 0.0, 0.0, 1.0)),
                            scale: Some(Vec3::ONE),
                        }],
                    }],
                    looping: false,
                },
            ],
            vertex_joints: Vec::new(),
            vertex_weights: Vec::new(),
        };

        let blended = skin.evaluate_blend("a", 0.0, "b", 0.0, 0.0);
        let pos = Vec3::new(
            blended.joint_matrices[0].col(3).x,
            blended.joint_matrices[0].col(3).y,
            blended.joint_matrices[0].col(3).z,
        );
        // blend=0 => 100% anim_a => translation (1, 0, 0)
        assert!(
            (pos - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-3,
            "pos = {:?}",
            pos
        );
    }
}
