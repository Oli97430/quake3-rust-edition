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
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GlbVertex {
    pub pos: [f32; 3],
    pub _pad0: f32,
    pub normal: [f32; 3],
    pub _pad1: f32,
    pub uv: [f32; 2],
    pub _pad2: [f32; 2],
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
        for mesh in doc.meshes() {
            for prim in mesh.primitives() {
                let reader = prim.reader(|b| Some(&buffers[b.index()]));
                let positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .ok_or(GlbError::NoPositions)?
                    .collect();
                let normals: Vec<[f32; 3]> = reader
                    .read_normals()
                    .map(|it| it.collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0, 1.0]; positions.len()]);
                // **UVs** (v0.9.5++) — set 0 utilisé par baseColor.
                // Si absent : (0, 0) par défaut (texture sample = pixel coin).
                let uvs: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .map(|tc| tc.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

                let base = all_verts.len() as u32;
                // **Note conventions axes GLB** : on PRÉSERVE les coords
                // glTF natives (Y-up) en local mesh.  Les matrices de
                // rendu côté engine (`queue_prop`, `queue_viewmodel`,
                // `spawn_glb_lights_for_prop`) appliquent la conversion
                // Y-up→Z-up via leurs `basis` vecteurs au moment de
                // poser l'instance.  Convertir ici casserait toutes
                // les orientations viewmodels tunées (plasma 180° Z,
                // MG -90° Y, etc.).  Voir le commentaire dans
                // `app.rs::queue_viewmodel` pour la convention.
                for ((p, n), uv) in positions.iter().zip(normals.iter()).zip(uvs.iter()) {
                    for k in 0..3 {
                        if p[k] < bounds_min[k] { bounds_min[k] = p[k]; }
                        if p[k] > bounds_max[k] { bounds_max[k] = p[k]; }
                    }
                    all_verts.push(GlbVertex {
                        pos: *p,
                        _pad0: 0.0,
                        normal: *n,
                        _pad1: 0.0,
                        uv: *uv,
                        _pad2: [0.0; 2],
                    });
                }

                if let Some(idx_iter) = reader.read_indices() {
                    for i in idx_iter.into_u32() {
                        all_indices.push(base + i);
                    }
                } else {
                    for i in 0..(positions.len() as u32) {
                        all_indices.push(base + i);
                    }
                }

                // **Material extraction** — premier prim qui a un
                // baseColorTexture la fournit pour tout le mesh.
                // Multi-material non supporté (asset moderne typique
                // a un material global).
                let material = prim.material();
                let pbr = material.pbr_metallic_roughness();
                base_color_factor = pbr.base_color_factor();
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
            let buf = &buffers[view.buffer().index()];
            let start = view.offset();
            let end = start + view.length();
            (buf[start..end].to_vec(), Some(mime_type))
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
