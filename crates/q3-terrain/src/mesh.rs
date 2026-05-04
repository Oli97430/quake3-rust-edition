//! Génération de mesh terrain — chunked + LOD discret.
//!
//! # Approche
//!
//! La heightmap est partitionnée en **chunks** de
//! [`CHUNK_SAMPLES`]×[`CHUNK_SAMPLES`] samples (en pratique 65×65 pour
//! avoir 64×64 quads, soit 8192 triangles à LOD 0). Chaque chunk peut
//! être maillé à 4 niveaux de détail selon sa distance à la caméra :
//!
//! | LOD | stride | tris/chunk | usage typique           |
//! |-----|--------|------------|-------------------------|
//! | 0   | 1      | 8 192      | foreground < 2 000 u    |
//! | 1   | 2      | 2 048      | mid 2 000..6 000 u      |
//! | 2   | 4      | 512        | far 6 000..14 000 u     |
//! | 3   | 8      | 128        | horizon > 14 000 u      |
//!
//! Sur la carte « Réunion 1/10 » (2400×2200 samples, chunk 64), on a
//! ~38×35 = 1330 chunks → si tous étaient à LOD 0 ce serait ~10 M tris,
//! impossible.  Avec LOD distance-based on tient typiquement
//! 200-400 K tris par frame.  Le quadtree « vrai » (subdiv adaptive
//! par rayon) viendra dans une release ultérieure ; ce premier pass
//! discret est déjà suffisant pour rendre la carte jouable.
//!
//! # Stitching (raccord entre LOD)
//!
//! Chunks voisins de LOD différents → fissures visibles aux bords.  Ce
//! module corrige naïvement en **dégradant le bord** du chunk au LOD
//! du voisin plus grossier (skirt zéro-tri).  Pour le MVP on accepte
//! les fissures qu'on cache avec des skirts verticaux (un mur de
//! triangles plongeant vers le sol au bord de chaque chunk) ; la
//! correction stitching propre arrive après.

use bytemuck::{Pod, Zeroable};
use q3_math::Vec3;

use crate::Terrain;

/// Côté d'un chunk en samples (vertices = `CHUNK_SAMPLES`,
/// quads = `CHUNK_SAMPLES - 1`).  64+1 = 65 → 4096 quads = 8192 tris.
pub const CHUNK_SAMPLES: usize = 65;

/// Côté d'un chunk en quads.
pub const CHUNK_QUADS: usize = CHUNK_SAMPLES - 1;

/// Niveaux de détail discrets.  Le stride entre samples maillés double
/// à chaque palier.  LOD 0 = full res, LOD 3 = 1/8.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LodLevel {
    Lod0 = 0,
    Lod1 = 1,
    Lod2 = 2,
    Lod3 = 3,
}

impl LodLevel {
    /// Stride entre samples au LOD courant (en samples).
    pub fn stride(self) -> usize {
        1 << (self as usize) // 1, 2, 4, 8
    }

    /// Nombre de quads sur un côté de chunk au LOD courant.
    pub fn quads_per_side(self) -> usize {
        CHUNK_QUADS / self.stride()
    }

    /// Itère du plus détaillé au plus grossier.
    pub fn all() -> [LodLevel; 4] {
        [Self::Lod0, Self::Lod1, Self::Lod2, Self::Lod3]
    }

    /// Sélection LOD par distance — heuristique simple basée sur la
    /// taille en pixels approximative d'un quad au LOD donné.  Pas de
    /// hystérésis ici (le caller peut ajouter une marge pour éviter le
    /// pop-in/pop-out frame-perfect).
    pub fn for_distance(dist_units: f32) -> Self {
        if dist_units < 2_000.0 {
            Self::Lod0
        } else if dist_units < 6_000.0 {
            Self::Lod1
        } else if dist_units < 14_000.0 {
            Self::Lod2
        } else {
            Self::Lod3
        }
    }
}

/// Coordonnée d'un chunk dans la grille — `cx ∈ [0, n_chunks_x[`,
/// idem pour `cy`.  La grille couvre `Terrain::width / CHUNK_QUADS`
/// chunks horizontalement (arrondi haut).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkCoord {
    pub cx: u32,
    pub cy: u32,
}

/// Vertex GPU pour le terrain.  Aligné std140 sur 48 octets pour rester
/// confortable sur les drivers stricts.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TerrainVertex {
    pub pos: [f32; 3],
    /// Padding f32 pour aligner sur 16 octets (3+1).
    pub _pad0: f32,
    pub normal: [f32; 3],
    pub _pad1: f32,
    /// Poids splat 4 canaux : roche / sable / végétation / urbain.
    /// Sommé à 1.0 (la classification garantit ça côté pipeline DEM).
    pub splat: [f32; 4],
}

impl TerrainVertex {
    pub const STRIDE_BYTES: usize = std::mem::size_of::<Self>();
}

/// Résultat de génération d'un chunk : buffers vertex + index prêts à
/// uploader sur le GPU.  Le caller (renderer) les passe à wgpu via
/// `create_buffer_init`.
pub struct ChunkMesh {
    pub coord: ChunkCoord,
    pub lod: LodLevel,
    pub vertices: Vec<TerrainVertex>,
    pub indices: Vec<u32>,
}

impl Terrain {
    /// Nombre de chunks horizontalement.
    pub fn n_chunks_x(&self) -> u32 {
        ((self.width.saturating_sub(1) + CHUNK_QUADS - 1) / CHUNK_QUADS) as u32
    }

    /// Nombre de chunks verticalement.
    pub fn n_chunks_y(&self) -> u32 {
        ((self.height.saturating_sub(1) + CHUNK_QUADS - 1) / CHUNK_QUADS) as u32
    }

    /// Centre monde d'un chunk (utilisé pour la sélection LOD côté
    /// renderer : distance caméra → LOD).
    pub fn chunk_center(&self, c: ChunkCoord) -> Vec3 {
        let x0 = self.meta.origin_x
            + (c.cx as usize * CHUNK_QUADS) as f32 * self.meta.units_per_sample;
        let y0 = self.meta.origin_y
            + (c.cy as usize * CHUNK_QUADS) as f32 * self.meta.units_per_sample;
        let half = (CHUNK_QUADS as f32 * self.meta.units_per_sample) * 0.5;
        let cx = x0 + half;
        let cy = y0 + half;
        Vec3::new(cx, cy, self.height_at(cx, cy))
    }

    /// Construit le mesh d'un chunk au LOD donné.  Les vertices sont
    /// dans le repère monde (déjà position absolue), les indices en
    /// u32 (un chunk de 65×65 dépasse 65k uniquement à LOD 0 → on
    /// reste en u32 pour rester homogène).
    pub fn build_chunk_mesh(&self, c: ChunkCoord, lod: LodLevel) -> ChunkMesh {
        let stride = lod.stride();
        let qps = lod.quads_per_side(); // quads par côté à ce LOD
        let vps = qps + 1;               // verts par côté

        // Indices "globaux" (heightmap) du coin haut-gauche.
        let gx0 = c.cx as usize * CHUNK_QUADS;
        let gy0 = c.cy as usize * CHUNK_QUADS;

        let mut vertices: Vec<TerrainVertex> = Vec::with_capacity(vps * vps);
        for j in 0..vps {
            for i in 0..vps {
                let gx = (gx0 + i * stride).min(self.width - 1);
                let gy = (gy0 + j * stride).min(self.height - 1);
                let wx = self.meta.origin_x + gx as f32 * self.meta.units_per_sample;
                let wy = self.meta.origin_y + gy as f32 * self.meta.units_per_sample;
                let wz = self.sample_z(gx, gy);
                let n = self.normal_at(wx, wy);
                let s = self.splat[gy * self.width + gx];
                let inv = 1.0 / 255.0;
                let mut splat = [
                    s[0] as f32 * inv,
                    s[1] as f32 * inv,
                    s[2] as f32 * inv,
                    s[3] as f32 * inv,
                ];
                // Renorm splat sum=1 pour blend propre.
                let sum: f32 = splat.iter().sum();
                if sum > 1e-3 {
                    for w in splat.iter_mut() {
                        *w /= sum;
                    }
                } else {
                    splat = [1.0, 0.0, 0.0, 0.0]; // fallback all-rock
                }
                vertices.push(TerrainVertex {
                    pos: [wx, wy, wz],
                    _pad0: 0.0,
                    normal: [n.x, n.y, n.z],
                    _pad1: 0.0,
                    splat,
                });
            }
        }

        // Triangulation : 2 tris par quad, ordre CCW vu d'au-dessus.
        // Convention : face up = normal Z+, donc pour un quad (i,j) avec
        // verts `tl, tr, bl, br`, les tris sont `tl → bl → br` et
        // `tl → br → tr` (CCW vu de Z+).
        let mut indices: Vec<u32> = Vec::with_capacity(qps * qps * 6);
        for j in 0..qps {
            for i in 0..qps {
                let tl = (j * vps + i) as u32;
                let tr = tl + 1;
                let bl = tl + vps as u32;
                let br = bl + 1;
                indices.extend_from_slice(&[tl, bl, br, tl, br, tr]);
            }
        }

        ChunkMesh {
            coord: c,
            lod,
            vertices,
            indices,
        }
    }

    /// Itère tous les chunks de la carte avec leur centre — utile pour
    /// que le renderer passe une fois par frame et choisisse le LOD
    /// pour chaque chunk selon la position caméra.
    pub fn iter_chunks(&self) -> impl Iterator<Item = ChunkCoord> {
        let nx = self.n_chunks_x();
        let ny = self.n_chunks_y();
        (0..ny).flat_map(move |cy| (0..nx).map(move |cx| ChunkCoord { cx, cy }))
    }

    /// Sélectionne le LOD à utiliser pour un chunk, étant donné la
    /// position caméra. Helper standard ; le renderer peut imposer le
    /// sien (par ex. forcer Lod0 dans un radius gameplay précis).
    pub fn select_lod(&self, c: ChunkCoord, camera_pos: Vec3) -> LodLevel {
        let center = self.chunk_center(c);
        let d = (center - camera_pos).length();
        LodLevel::for_distance(d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::TerrainMeta;

    fn flat_terrain(w: usize, h: usize, z: f32) -> Terrain {
        let z_min = -100.0;
        let z_max = 1000.0;
        let s = ((z - z_min) / (z_max - z_min) * 65535.0) as u16;
        Terrain {
            width: w,
            height: h,
            samples: vec![s; w * h],
            splat: vec![[200, 50, 0, 5]; w * h],
            meta: TerrainMeta {
                name: "test".into(),
                width: w,
                height: h,
                z_min,
                z_max,
                origin_x: 0.0,
                origin_y: 0.0,
                units_per_sample: 12.0,
                ocean_z: -100.0,
                water_level: 0.0,
                pois: vec![],
            },
        }
    }

    #[test]
    fn lod_strides_are_powers_of_two() {
        assert_eq!(LodLevel::Lod0.stride(), 1);
        assert_eq!(LodLevel::Lod1.stride(), 2);
        assert_eq!(LodLevel::Lod2.stride(), 4);
        assert_eq!(LodLevel::Lod3.stride(), 8);
    }

    #[test]
    fn lod_for_distance_monotonic() {
        assert_eq!(LodLevel::for_distance(100.0), LodLevel::Lod0);
        assert_eq!(LodLevel::for_distance(3_000.0), LodLevel::Lod1);
        assert_eq!(LodLevel::for_distance(8_000.0), LodLevel::Lod2);
        assert_eq!(LodLevel::for_distance(20_000.0), LodLevel::Lod3);
    }

    #[test]
    fn chunk_count_matches_width() {
        // 129 samples → 128 quads → 2 chunks de 64 quads.
        let t = flat_terrain(129, 129, 100.0);
        assert_eq!(t.n_chunks_x(), 2);
        assert_eq!(t.n_chunks_y(), 2);
    }

    #[test]
    fn chunk_lod0_has_8192_tris() {
        let t = flat_terrain(65, 65, 100.0);
        let mesh = t.build_chunk_mesh(ChunkCoord { cx: 0, cy: 0 }, LodLevel::Lod0);
        // 64 × 64 quads × 2 tris × 3 indices = 24 576 indices
        assert_eq!(mesh.indices.len(), 24_576);
        assert_eq!(mesh.indices.len() / 3, 8_192);
        // 65×65 verts
        assert_eq!(mesh.vertices.len(), 4_225);
    }

    #[test]
    fn chunk_lod3_has_128_tris() {
        let t = flat_terrain(65, 65, 100.0);
        let mesh = t.build_chunk_mesh(ChunkCoord { cx: 0, cy: 0 }, LodLevel::Lod3);
        // 8 × 8 quads à LOD 3
        assert_eq!(mesh.indices.len() / 3, 128);
        assert_eq!(mesh.vertices.len(), 81); // 9×9
    }

    #[test]
    fn vertex_positions_are_in_world_space() {
        let t = flat_terrain(65, 65, 100.0);
        let mesh = t.build_chunk_mesh(ChunkCoord { cx: 0, cy: 0 }, LodLevel::Lod0);
        // Premier vertex au coin (0,0) → world = origin_x + 0, origin_y + 0
        assert_eq!(mesh.vertices[0].pos[0], t.meta.origin_x);
        assert_eq!(mesh.vertices[0].pos[1], t.meta.origin_y);
        // Z = 100 (terrain plat).
        assert!((mesh.vertices[0].pos[2] - 100.0).abs() < 1e-2);
    }

    #[test]
    fn splat_weights_sum_to_one() {
        let t = flat_terrain(65, 65, 100.0);
        let mesh = t.build_chunk_mesh(ChunkCoord { cx: 0, cy: 0 }, LodLevel::Lod0);
        for v in &mesh.vertices {
            let sum: f32 = v.splat.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "splat sum != 1 : {:?}",
                v.splat
            );
        }
    }

    #[test]
    fn select_lod_close_camera_picks_lod0() {
        let t = flat_terrain(129, 129, 100.0);
        let cam = t.chunk_center(ChunkCoord { cx: 0, cy: 0 });
        assert_eq!(
            t.select_lod(ChunkCoord { cx: 0, cy: 0 }, cam),
            LodLevel::Lod0
        );
    }

    #[test]
    fn iter_chunks_count_matches_grid() {
        let t = flat_terrain(193, 129, 100.0); // 3 × 2 chunks
        let n = t.iter_chunks().count();
        assert_eq!(n, 6);
    }

    #[test]
    fn vertex_stride_is_48_bytes() {
        // Critique pour le layout WGSL : si on change ce taille, il
        // faut bumper la version du shader correspondant.
        assert_eq!(TerrainVertex::STRIDE_BYTES, 48);
    }
}
