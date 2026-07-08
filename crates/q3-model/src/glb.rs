//! Loader **glTF / GLB binaire** — minimal, géométrie statique uniquement.
//!
//! Charge un fichier `.glb` (ou `.gltf` + buffers/textures séparés) et
//! aplatit toutes les meshes/primitives en un seul `GlbMesh` avec
//! vertices `(pos, normal, uv)` + indices u32.
//!
//! # Limitations actuelles (v0.9.5)
//!
//! * **Pas d'animations** : le squelette + poses sont ignorés. Le modèle
//!   est dessiné en pose-of-rest (frame 0). Pour les drones survol BR,
//!   c'est suffisant — ils tournent dans le ciel via une transform
//!   monde animée côté engine.
//! * **Pas de PBR** : couleurs/textures du matériau glTF ignorées —
//!   le caller passe une teinte uniforme. Les drones BR se rendent en
//!   un seul color tint blanc/bleu pâle.
//! * **Pas de morph targets**.
//! * **Multi-primitives concaténés** : si un mesh glTF a plusieurs
//!   primitives (par exemple corps + ailes séparées), elles sont
//!   merged en un seul vertex/index buffer. Index offset géré
//!   automatiquement.
//!
//! Pour aller plus loin (PBR/skinning), il faudra refaire le pipeline
//! côté `q3-renderer` — ce loader retourne déjà les UVs et indices,
//! l'extension est facile.

use bytemuck::{Pod, Zeroable};
use thiserror::Error as ThisError;
use tracing::{debug, info};

#[derive(Debug, ThisError)]
pub enum GlbError {
    #[error("gltf: {0}")]
    Gltf(#[from] gltf::Error),
    #[error("glb: aucun mesh trouvé")]
    NoMesh,
    #[error("glb: primitive sans positions")]
    NoPositions,
    #[error("glb: {0}")]
    Other(String),
}

/// Vertex GPU layout (48 octets, std140 sur les drivers stricts).
///
/// **v0.9.5++** : ajout du champ `uv` (vec2) pour le sampling de la
/// `baseColorTexture` glTF. Stride passé de 32 → 48 octets.
///
/// **v0.9.6** : ajout d'un champ `color` (RGBA8) qui réutilise le
/// padding existant.  Permet de baker le `base_color_factor` de
/// chaque primitive glTF en vertex color, donc supporter des assets
/// multi-material où chaque sous-mesh a sa propre couleur (railgun
/// avec orange + steel + cyan, par exemple).  Le shader multiplie
/// vertex.color par la base color globale.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GlbVertex {
    pub pos: [f32; 3],
    pub _pad0: f32,
    pub normal: [f32; 3],
    pub _pad1: f32,
    pub uv: [f32; 2],
    /// Vertex color RGBA8 (0..255).  255 = blanc neutre (=> no-op).
    pub color: [u8; 4],
    pub _pad2: f32,
}

impl GlbVertex {
    pub const STRIDE_BYTES: usize = std::mem::size_of::<Self>();
}

/// Texture extracted from glTF — stocked as raw RGBA8 + dimensions
/// pour upload wgpu. Chaque mesh a au plus 1 baseColorTexture (multi-
/// material non supporté pour l'instant — un GLB multi-prim avec
/// matériaux distincts mergera en un seul mesh + dernière texture).
#[derive(Debug, Clone)]
pub struct GlbTexture {
    pub width: u32,
    pub height: u32,
    /// Données RGBA8 row-major, top-down (image crate convention).
    pub data: Vec<u8>,
}

/// **glTF KHR_lights_punctual** — light extraite des nodes du scene
/// graph. Position en local mesh space, à transformer en world au
/// moment du spawn de l'instance.  Range = 0 ou None signifie « pas
/// de cap distance » (point light infinie en théorie ; on cap à 500u
/// au render side).
#[derive(Debug, Clone, Copy)]
pub struct GlbLight {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    /// Distance d'extinction en unités mesh.  0 = pas de range
    /// définie côté glTF (lumière infinie).
    pub range: f32,
    pub kind: GlbLightKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlbLightKind {
    Point,
    Directional,
    Spot,
}

/// Mesh statique chargé depuis un GLB. Tous les buffers sont consolidés
/// — `vertices` et `indices` sont prêts à pousser vers wgpu via
/// `create_buffer_init`.
pub struct GlbMesh {
    pub vertices: Vec<GlbVertex>,
    pub indices: Vec<u32>,
    /// Bounds approximatives (min, max) pour le frustum culling — calculées au load.
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    /// Texture baseColor extraite du matériau glTF — `None` si absent.
    pub base_color_texture: Option<GlbTexture>,
    /// Couleur baseColorFactor du matériau glTF (multipliée avec la texture).
    pub base_color_factor: [f32; 4],
    /// **PBR maps** (v0.9.5++) — normal map (espace tangent),
    /// metallicRoughness combinée (G=roughness, B=metallic per glTF spec).
    /// `None` si absente du matériau (asset PBR partiel).
    pub normal_texture: Option<GlbTexture>,
    pub metallic_roughness_texture: Option<GlbTexture>,
    /// Facteurs scalaires multipliés avec les textures (ou utilisés
    /// seuls si textures absentes).
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    /// **Lights** (v0.9.5++) — extraites via KHR_lights_punctual.
    /// Vide si l'asset n'a pas de lumières définies dans son scene graph.
    pub lights: Vec<GlbLight>,
}

impl GlbMesh {
    /// Parse un GLB depuis ses bytes en mémoire (extraction du
    /// chunk binary inclus dans le `.glb` standard).
    pub fn from_glb_bytes(bytes: &[u8]) -> Result<Self, GlbError> {
        // **Détection Git LFS pointer** (v0.9.5++ safety) — si le
        // fichier fait <1 KB et commence par `version https://git-lfs`,
        // c'est un pointer LFS qui n'a pas été fetché.  On renvoie une
        // erreur très explicite pour que l'utilisateur sache quoi
        // faire (ça arrive après `git lfs migrate` sans `git lfs pull`).
        if bytes.len() < 1024 && bytes.starts_with(b"version https://git-lfs") {
            return Err(GlbError::Other(
                "Fichier GLB est un pointer Git LFS non résolu — \
                 lance `git lfs pull` à la racine du repo pour \
                 télécharger les vrais binaires (~232 MB)".to_string(),
            ));
        }
        // **Validation magic glTF** — un GLB valide commence par
        // `glTF` (0x46546C67 little-endian).  Avant le parser
        // gltf::import_slice, on vérifie pour donner un message clair
        // si l'asset est tronqué ou corrompu.
        if bytes.len() < 4 || &bytes[..4] != b"glTF" {
            return Err(GlbError::Other(format!(
                "Fichier non-GLB (magic = {:?} au lieu de \"glTF\")",
                &bytes[..4.min(bytes.len())]
            )));
        }
        let (doc, buffers, _images) = gltf::import_slice(bytes)?;
        Self::from_document(&doc, &buffers)
    }

    /// Construit le mesh depuis un document gltf déjà parsé + ses
    /// buffers binaires (séparés ou inline pour GLB).
    fn from_document(
        doc: &gltf::Document,
        buffers: &[gltf::buffer::Data],
    ) -> Result<Self, GlbError> {
        let mut all_verts: Vec<GlbVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        let mut bounds_min = [f32::INFINITY; 3];
        let mut bounds_max = [f32::NEG_INFINITY; 3];
        let mut base_color_texture: Option<GlbTexture> = None;
        let mut base_color_factor: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        let mut normal_texture: Option<GlbTexture> = None;
        let mut metallic_roughness_texture: Option<GlbTexture> = None;
        let mut metallic_factor: f32 = 0.0;
        let mut roughness_factor: f32 = 1.0;

        let mut prim_count = 0usize;

        // **Scene graph traversal** (v0.9.6) — on parcourt récursivement
        // les nodes de la scène par défaut en accumulant la world matrix
        // (parent_world * local).  Chaque mesh accroché à un node voit
        // ses vertex positions transformées par cette matrice.  Sans
        // ça, des assets exportés de Blender (multi-mesh parentés à un
        // empty `Gun_Root`) verraient leurs sous-meshes s'effondrer à
        // l'origine puisqu'ils ne sont pas eux-mêmes au monde origin.
        //
        // Convention : on **bake** la world matrix dans les vertices.
        // Les axes glTF natifs Y-up sont préservés tels quels — la
        // conversion Y-up→Z-up reste appliquée à l'instance par les
        // matrices côté engine (cf. commentaire historique).

        // Collecte récursive : pour chaque mesh_idx on
        // mémorise toutes les world matrices qui l'utilisent.
        // (Un même mesh peut être instancié par plusieurs nodes,
        //  cas rare mais on le supporte.)
        let mut mesh_instances: hashbrown::HashMap<usize, Vec<glam::Mat4>> =
            hashbrown::HashMap::new();
        // Trouve la scène par défaut (ou la première).
        let scene = doc.default_scene().or_else(|| doc.scenes().next());
        if let Some(scene) = scene {
            for node in scene.nodes() {
                collect_node_meshes(&node, glam::Mat4::IDENTITY, &mut mesh_instances);
            }
        } else {
            // Fallback : pas de scène — ajoute toutes les meshes avec identity.
            for mesh in doc.meshes() {
                mesh_instances.entry(mesh.index()).or_default().push(glam::Mat4::IDENTITY);
            }
        }

        for mesh in doc.meshes() {
            let world_matrices = match mesh_instances.get(&mesh.index()) {
                Some(v) if !v.is_empty() => v.clone(),
                _ => continue, // mesh orphelin (pas dans la scène)
            };

            for prim in mesh.primitives() {
                // Material : extrait factor + textures par primitive.
                let material = prim.material();
                let pbr = material.pbr_metallic_roughness();
                let prim_color = pbr.base_color_factor();
                base_color_factor = [1.0, 1.0, 1.0, 1.0]; // global = neutre, le vrai code va dans vertex color
                metallic_factor = pbr.metallic_factor();
                roughness_factor = pbr.roughness_factor();
                if base_color_texture.is_none() {
                    if let Some(info) = pbr.base_color_texture() {
                        let img = info.texture().source();
                        if let Some(extracted) = extract_texture(&img, buffers) {
                            base_color_texture = Some(extracted);
                        }
                    }
                }
                if metallic_roughness_texture.is_none() {
                    if let Some(info) = pbr.metallic_roughness_texture() {
                        let img = info.texture().source();
                        if let Some(extracted) = extract_texture(&img, buffers) {
                            metallic_roughness_texture = Some(extracted);
                        }
                    }
                }
                if normal_texture.is_none() {
                    if let Some(info) = material.normal_texture() {
                        let img = info.texture().source();
                        if let Some(extracted) = extract_texture(&img, buffers) {
                            normal_texture = Some(extracted);
                        }
                    }
                }

                let reader = prim.reader(|b| Some(&buffers[b.index()]));
                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .ok_or(GlbError::NoPositions)?
                    .collect();
                let normals: Vec<[f32; 3]> = reader
                    .read_normals()
                    .map(|it| it.collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0, 1.0]; positions.len()]);
                let uvs: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .map(|tc| tc.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

                // Pack base_color_factor en RGBA8.
                let prim_color8 = [
                    (prim_color[0].clamp(0.0, 1.0) * 255.0) as u8,
                    (prim_color[1].clamp(0.0, 1.0) * 255.0) as u8,
                    (prim_color[2].clamp(0.0, 1.0) * 255.0) as u8,
                    (prim_color[3].clamp(0.0, 1.0) * 255.0) as u8,
                ];

                // Pour chaque world matrix instanciant ce mesh, ajoute
                // une copie des vertices transformées.
                for &world in &world_matrices {
                    let normal_mat = glam::Mat3::from_mat4(world);
                    // Inversion-transpose pour les normals (compensation
                    // d'éventuels scales non-uniformes).
                    let normal_mat = normal_mat.inverse().transpose();

                    let base = all_verts.len() as u32;

                    for ((p, n), uv) in
                        positions.iter().zip(normals.iter()).zip(uvs.iter())
                    {
                        let p4 = world * glam::Vec4::new(p[0], p[1], p[2], 1.0);
                        let pw = [p4.x, p4.y, p4.z];
                        let nw = (normal_mat * glam::Vec3::new(n[0], n[1], n[2]))
                            .normalize_or_zero();
                        for k in 0..3 {
                            if pw[k] < bounds_min[k] { bounds_min[k] = pw[k]; }
                            if pw[k] > bounds_max[k] { bounds_max[k] = pw[k]; }
                        }
                        all_verts.push(GlbVertex {
                            pos: pw,
                            _pad0: 0.0,
                            normal: [nw.x, nw.y, nw.z],
                            _pad1: 0.0,
                            uv: *uv,
                            color: prim_color8,
                            _pad2: 0.0,
                        });
                    }

                    if let Some(idx_iter) = reader.clone().read_indices() {
                        for i in idx_iter.into_u32() {
                            all_indices.push(base + i);
                        }
                    } else {
                        for i in 0..(positions.len() as u32) {
                            all_indices.push(base + i);
                        }
                    }
                }
                let _ = &world_matrices; // borrow check
                prim_count += 1;
            }
        }

        if all_verts.is_empty() {
            return Err(GlbError::NoMesh);
        }

        info!(
            "glb: {} verts / {} indices / {} primitives, baseColor texture {}",
            all_verts.len(),
            all_indices.len(),
            prim_count,
            if base_color_texture.is_some() { "PRESENT" } else { "absent" }
        );
        debug!("glb bounds : min {:?} max {:?}", bounds_min, bounds_max);

        // **Extraction lights KHR_lights_punctual** — on parcourt
        // les nodes du scene graph (root scene), récupère leur world
        // matrix accumulée, et collecte chaque light attachée à un node.
        let mut lights: Vec<GlbLight> = Vec::new();
        for scene in doc.scenes() {
            for node in scene.nodes() {
                collect_lights_recursive(&node, glam::Mat4::IDENTITY, &mut lights);
            }
        }
        if !lights.is_empty() {
            info!("glb: {} lights extraites (KHR_lights_punctual)", lights.len());
        }

        Ok(GlbMesh {
            vertices: all_verts,
            indices: all_indices,
            bounds_min,
            bounds_max,
            base_color_texture,
            base_color_factor,
            normal_texture,
            metallic_roughness_texture,
            metallic_factor,
            roughness_factor,
            lights,
        })
    }

    /// Centre des bounds — utile pour positionner le drone autour du
    /// pivot mesh.
    pub fn center(&self) -> [f32; 3] {
        [
            (self.bounds_min[0] + self.bounds_max[0]) * 0.5,
            (self.bounds_min[1] + self.bounds_max[1]) * 0.5,
            (self.bounds_min[2] + self.bounds_max[2]) * 0.5,
        ]
    }

    /// Rayon englobant max pour culling sphérique.
    pub fn radius(&self) -> f32 {
        let c = self.center();
        let dx = (self.bounds_max[0] - c[0]).abs();
        let dy = (self.bounds_max[1] - c[1]).abs();
        let dz = (self.bounds_max[2] - c[2]).abs();
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// **v0.9.6** — parcourt récursivement le scene graph et accumule les
/// world matrices des nodes ayant un mesh attaché.  Permet au loader
/// de baker les transforms des nodes parents dans les vertex positions
/// (corrige les GLB multi-mesh exportés depuis Blender / DCC où chaque
/// sous-objet est parenté à un empty avec sa propre translation).
fn collect_node_meshes(
    node: &gltf::Node<'_>,
    parent_world: glam::Mat4,
    out: &mut hashbrown::HashMap<usize, Vec<glam::Mat4>>,
) {
    let local = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_world * local;
    if let Some(mesh) = node.mesh() {
        out.entry(mesh.index()).or_default().push(world);
    }
    for child in node.children() {
        collect_node_meshes(&child, world, out);
    }
}

/// Parcourt récursivement un node + ses enfants et collecte les
/// lights attachées (KHR_lights_punctual).  La world matrix accumulée
/// est utilisée pour transformer la position de la light en local
/// space mesh (le caller la transforme à nouveau au moment du spawn
/// de l'instance dans le monde).
fn collect_lights_recursive(
    node: &gltf::Node<'_>,
    parent_world: glam::Mat4,
    out: &mut Vec<GlbLight>,
) {
    let local = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent_world * local;
    if let Some(light) = node.light() {
        let pos = world.col(3).truncate();
        let color = light.color();
        let intensity = light.intensity();
        let range = light.range().unwrap_or(0.0);
        let kind = match light.kind() {
            gltf::khr_lights_punctual::Kind::Point => GlbLightKind::Point,
            gltf::khr_lights_punctual::Kind::Directional => GlbLightKind::Directional,
            gltf::khr_lights_punctual::Kind::Spot { .. } => GlbLightKind::Spot,
        };
        out.push(GlbLight {
            position: [pos.x, pos.y, pos.z],
            color,
            intensity,
            range,
            kind,
        });
    }
    for child in node.children() {
        collect_lights_recursive(&child, world, out);
    }
}

/// Décode une image glTF (URI/View) en RGBA8 pour upload GPU.
/// Retourne None si l'image n'est pas embedded ou ne peut pas être
/// décodée. Supporte PNG/JPEG (formats standard glTF).
fn extract_texture(
    img: &gltf::Image<'_>,
    buffers: &[gltf::buffer::Data],
) -> Option<GlbTexture> {
    use gltf::image::Source;
    let source = img.source();
    let (data, mime) = match source {
        Source::View { view, mime_type } => {
            // **Bornes sur fichier hostile** (v0.10.3) — l'index de buffer
            // et l'intervalle offset..offset+length viennent de l'en-tête
            // glTF sans garantie de cohérence avec la taille réelle du
            // buffer.  Un GLB forgé (index invalide ou length débordante)
            // faisait paniquer l'indexation.  On valide tout et on skip la
            // texture proprement (`None`) au lieu de crasher.
            let buf = buffers.get(view.buffer().index())?;
            let start = view.offset();
            let end = start.checked_add(view.length())?;
            let slice = buf.get(start..end)?;
            (slice.to_vec(), Some(mime_type))
        }
        Source::Uri { .. } => {
            // Les URIs externes ne sont pas supportées pour les GLB
            // packagés en `.glb` binary (toutes les images sont
            // embedded en buffer view).
            return None;
        }
    };
    // Décode via la crate `image` — auto-detect le format si mime
    // est PNG/JPEG.
    let format = match mime {
        Some("image/png") => Some(image::ImageFormat::Png),
        Some("image/jpeg") => Some(image::ImageFormat::Jpeg),
        _ => None,
    };
    let decoded = match format {
        Some(f) => image::load_from_memory_with_format(&data, f).ok()?,
        None => image::load_from_memory(&data).ok()?,
    };
    let rgba = decoded.to_rgba8();
    let (width, height) = rgba.dimensions();
    Some(GlbTexture {
        width,
        height,
        data: rgba.into_raw(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_stride_is_48_bytes() {
        // Critique pour le layout WGSL côté renderer (v0.9.5++ ajout UV).
        assert_eq!(GlbVertex::STRIDE_BYTES, 48);
    }

    #[test]
    fn empty_input_errors() {
        assert!(GlbMesh::from_glb_bytes(&[]).is_err());
    }
}
