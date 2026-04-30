//! Structures binaires brutes, `#[repr(C)]`, telles que stockées dans le
//! fichier BSP. Toutes `bytemuck::Pod` → cast zéro-copie.
//!
//! Les tailles ici sont **load-bearing** : si l'un de ces structs change de
//! taille, le parseur lit des données décalées.

use bytemuck::{Pod, Zeroable};

/// Entrée de la table des lumps dans l'en-tête.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct DLump {
    pub file_ofs: i32,
    pub file_len: i32,
}

/// En-tête : magic + version + 17 lumps.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DHeader {
    pub magic: [u8; 4],
    pub version: i32,
    pub lumps: [DLump; 17],
}

/// Shader (= texture + surface flags + content flags).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DShader {
    pub shader: [u8; 64],
    pub surface_flags: i32,
    pub content_flags: i32,
}

impl DShader {
    /// Nom du shader, sans le padding 00.
    pub fn name(&self) -> &str {
        null_terminated_str(&self.shader)
    }
}

/// Plan de l'arbre BSP.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DPlane {
    pub normal: [f32; 3],
    pub dist: f32,
}

/// Nœud interne de l'arbre BSP.
///
/// `children[i]` : si positif, index de node ; si négatif, `-(leaf_index) - 1`
/// (les leafs sont encodés en négatif pour permettre le partage du même entier
/// signé).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DNode {
    pub plane_num: i32,
    pub children: [i32; 2],
    pub mins: [i32; 3],
    pub maxs: [i32; 3],
}

/// Feuille de l'arbre BSP. Contient les listes de surfaces et de brushes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DLeaf {
    pub cluster: i32,
    pub area: i32,
    pub mins: [i32; 3],
    pub maxs: [i32; 3],
    pub first_leaf_surface: i32,
    pub num_leaf_surfaces: i32,
    pub first_leaf_brush: i32,
    pub num_leaf_brushes: i32,
}

/// Submodel (modèle 0 = géométrie du monde ; > 0 = brushmodels, portes, lifts, etc.).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DModel {
    pub mins: [f32; 3],
    pub maxs: [f32; 3],
    pub first_surface: i32,
    pub num_surfaces: i32,
    pub first_brush: i32,
    pub num_brushes: i32,
}

/// Brush convexe (ensemble de plans pour la collision).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DBrush {
    pub first_side: i32,
    pub num_sides: i32,
    pub shader_num: i32,
}

/// Face d'un brush (référence un plan + un shader).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DBrushSide {
    pub plane_num: i32,
    pub shader_num: i32,
}

/// Vertex de draw (géométrie rendue).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DrawVert {
    pub xyz: [f32; 3],
    /// UV de la texture diffuse.
    pub st: [f32; 2],
    /// UV de la lightmap.
    pub lightmap: [f32; 2],
    pub normal: [f32; 3],
    /// RGBA 8-bit (alpha souvent inutilisé).
    pub color: [u8; 4],
}

/// Volume de brouillard / eau / lava défini par un brush.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DFog {
    pub shader: [u8; 64],
    pub brush_num: i32,
    pub visible_side: i32,
}

impl DFog {
    pub fn name(&self) -> &str {
        null_terminated_str(&self.shader)
    }
}

/// Type d'une surface de draw.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceType {
    Bad = 0,
    /// Face plane simple (triangle soup déjà triangulée).
    Planar = 1,
    /// Patch de Bézier bicubique 3×3 (nécessite une tessellation runtime).
    Patch = 2,
    /// Soupe de triangles (mesh de modèle).
    TriangleSoup = 3,
    /// Flare (billboard lumineux).
    Flare = 4,
}

impl SurfaceType {
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::Planar,
            2 => Self::Patch,
            3 => Self::TriangleSoup,
            4 => Self::Flare,
            _ => Self::Bad,
        }
    }
}

/// Surface de draw. Une map est un tableau de surfaces, chacune référant
/// une tranche de `draw_verts` et de `draw_indexes`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DSurface {
    pub shader_num: i32,
    pub fog_num: i32,
    /// Voir `SurfaceType`.
    pub surface_type: i32,
    pub first_vert: i32,
    pub num_verts: i32,
    pub first_index: i32,
    pub num_indexes: i32,
    pub lightmap_num: i32,
    pub lightmap_x: i32,
    pub lightmap_y: i32,
    pub lightmap_width: i32,
    pub lightmap_height: i32,
    pub lightmap_origin: [f32; 3],
    pub lightmap_vecs: [[f32; 3]; 3],
    pub patch_width: i32,
    pub patch_height: i32,
}

impl DSurface {
    pub fn kind(&self) -> SurfaceType {
        SurfaceType::from_i32(self.surface_type)
    }
}

fn null_terminated_str(bytes: &[u8]) -> &str {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end]).unwrap_or("")
}
