//! Parseur de maps Quake 3 (**IBSP v46**).
//!
//! Format documenté par Kekoa Proudfoot (1999) et par les headers originaux
//! `qfiles.h` d'id Software. Les 17 lumps sont tous lus en zero-copy quand
//! c'est possible (via `bytemuck::cast_slice`).
//!
//! # Usage
//!
//! ```no_run
//! # use q3_bsp::Bsp;
//! let bytes = std::fs::read("maps/q3dm1.bsp").unwrap();
//! let bsp = Bsp::parse(&bytes).unwrap();
//! println!("vertices: {}", bsp.draw_verts.len());
//! ```

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

pub mod lumps;
pub mod patch;
pub mod raw;

use bytemuck::from_bytes;
use q3_common::{Error, Result};
use q3_math::{Aabb, Vec3};
use tracing::{debug, trace};

use self::raw::*;

/// Magic attendu dans l'en-tête.
pub const BSP_MAGIC: [u8; 4] = *b"IBSP";
/// Version Quake 3 Arena (Q3A = 46, ET = 47, RTCW = 47, QL = 46).
pub const BSP_VERSION: i32 = 46;

/// Nombre de lumps dans l'en-tête.
pub const NUM_LUMPS: usize = 17;

#[repr(usize)]
#[derive(Debug, Clone, Copy)]
pub enum LumpId {
    Entities = 0,
    Shaders = 1,
    Planes = 2,
    Nodes = 3,
    Leafs = 4,
    LeafSurfaces = 5,
    LeafBrushes = 6,
    Models = 7,
    Brushes = 8,
    BrushSides = 9,
    DrawVerts = 10,
    DrawIndexes = 11,
    Fogs = 12,
    Surfaces = 13,
    Lightmaps = 14,
    LightGrid = 15,
    Visibility = 16,
}

/// Map BSP parsée. Les données des lumps de taille variable (lightmaps,
/// entities, visibility) sont stockées telles quelles ; les lumps
/// tableaux sont stockés comme `Vec<T>` pour un accès direct.
#[derive(Debug, Clone)]
pub struct Bsp {
    pub entities: String,
    pub shaders: Vec<DShader>,
    pub planes: Vec<DPlane>,
    pub nodes: Vec<DNode>,
    pub leafs: Vec<DLeaf>,
    pub leaf_surfaces: Vec<i32>,
    pub leaf_brushes: Vec<i32>,
    pub models: Vec<DModel>,
    pub brushes: Vec<DBrush>,
    pub brush_sides: Vec<DBrushSide>,
    pub draw_verts: Vec<DrawVert>,
    pub draw_indexes: Vec<i32>,
    pub fogs: Vec<DFog>,
    pub surfaces: Vec<DSurface>,
    /// Lightmaps brutes, 128×128 RGB, concaténées. Utiliser `lightmaps()` pour
    /// un découpage par index.
    pub lightmap_bytes: Vec<u8>,
    /// Lightgrid : samples de 8 bytes chacun (ambient[3] + directed[3] + lat/long[2]).
    pub lightgrid_bytes: Vec<u8>,
    pub visibility: Visibility,
}

/// Taille d'une lightmap Q3 en pixels.
pub const LIGHTMAP_SIZE: usize = 128;
/// Nombre de bytes par lightmap (128*128*3).
pub const LIGHTMAP_BYTES: usize = LIGHTMAP_SIZE * LIGHTMAP_SIZE * 3;

/// Lump de visibilité (PVS).
#[derive(Debug, Clone, Default)]
pub struct Visibility {
    pub num_clusters: i32,
    pub cluster_bytes: i32,
    pub data: Vec<u8>,
}

impl Bsp {
    /// Parse une map BSP à partir de ses bytes bruts.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < std::mem::size_of::<DHeader>() {
            return Err(Error::bsp(format!(
                "fichier trop petit ({} bytes)",
                bytes.len()
            )));
        }

        let header: &DHeader = from_bytes(&bytes[..std::mem::size_of::<DHeader>()]);

        if header.magic != BSP_MAGIC {
            return Err(Error::bsp(format!(
                "mauvais magic : {:?}",
                std::str::from_utf8(&header.magic).unwrap_or("<non-utf8>")
            )));
        }
        if header.version != BSP_VERSION {
            return Err(Error::bsp(format!(
                "version {} non supportée (attendu {BSP_VERSION})",
                header.version
            )));
        }

        debug!(
            "bsp: magic=IBSP version={} taille={} bytes",
            header.version,
            bytes.len()
        );

        // Helper : extraire le slice d'un lump avec validation de bornes.
        let lump_bytes = |id: LumpId| -> Result<&[u8]> {
            let l = &header.lumps[id as usize];
            let start = l.file_ofs as usize;
            let len = l.file_len as usize;
            let end = start.checked_add(len).ok_or_else(|| {
                Error::bsp(format!("lump {id:?} overflow ({start} + {len})"))
            })?;
            if end > bytes.len() {
                return Err(Error::bsp(format!(
                    "lump {id:?} hors limites ({end} > {})",
                    bytes.len()
                )));
            }
            trace!("bsp: lump {id:?} offset={start} len={len}");
            Ok(&bytes[start..end])
        };

        let entities = {
            let b = lump_bytes(LumpId::Entities)?;
            // .trim_end_matches('\0') retire le padding 00
            std::str::from_utf8(b)
                .map_err(|e| Error::bsp(format!("entities UTF-8: {e}")))?
                .trim_end_matches('\0')
                .to_string()
        };

        let shaders = parse_pod_vec::<DShader>(lump_bytes(LumpId::Shaders)?, "shaders")?;
        let planes = parse_pod_vec::<DPlane>(lump_bytes(LumpId::Planes)?, "planes")?;
        let nodes = parse_pod_vec::<DNode>(lump_bytes(LumpId::Nodes)?, "nodes")?;
        let leafs = parse_pod_vec::<DLeaf>(lump_bytes(LumpId::Leafs)?, "leafs")?;
        let leaf_surfaces = parse_pod_vec::<i32>(lump_bytes(LumpId::LeafSurfaces)?, "leafSurfaces")?;
        let leaf_brushes = parse_pod_vec::<i32>(lump_bytes(LumpId::LeafBrushes)?, "leafBrushes")?;
        let models = parse_pod_vec::<DModel>(lump_bytes(LumpId::Models)?, "models")?;
        let brushes = parse_pod_vec::<DBrush>(lump_bytes(LumpId::Brushes)?, "brushes")?;
        let brush_sides = parse_pod_vec::<DBrushSide>(lump_bytes(LumpId::BrushSides)?, "brushSides")?;
        let draw_verts = parse_pod_vec::<DrawVert>(lump_bytes(LumpId::DrawVerts)?, "drawVerts")?;
        let draw_indexes = parse_pod_vec::<i32>(lump_bytes(LumpId::DrawIndexes)?, "drawIndexes")?;
        let fogs = parse_pod_vec::<DFog>(lump_bytes(LumpId::Fogs)?, "fogs")?;
        let surfaces = parse_pod_vec::<DSurface>(lump_bytes(LumpId::Surfaces)?, "surfaces")?;

        let lightmap_bytes = lump_bytes(LumpId::Lightmaps)?.to_vec();
        if lightmap_bytes.len() % LIGHTMAP_BYTES != 0 {
            return Err(Error::bsp(format!(
                "lightmaps : taille {} pas multiple de {LIGHTMAP_BYTES}",
                lightmap_bytes.len()
            )));
        }

        let lightgrid_bytes = lump_bytes(LumpId::LightGrid)?.to_vec();

        let visibility = {
            let b = lump_bytes(LumpId::Visibility)?;
            if b.len() >= 8 {
                let num_clusters = i32::from_le_bytes(b[0..4].try_into().unwrap());
                let cluster_bytes = i32::from_le_bytes(b[4..8].try_into().unwrap());
                Visibility {
                    num_clusters,
                    cluster_bytes,
                    data: b[8..].to_vec(),
                }
            } else {
                Visibility::default()
            }
        };

        debug!(
            "bsp: shaders={} planes={} nodes={} leafs={} verts={} indexes={} surfs={} lms={}",
            shaders.len(),
            planes.len(),
            nodes.len(),
            leafs.len(),
            draw_verts.len(),
            draw_indexes.len(),
            surfaces.len(),
            lightmap_bytes.len() / LIGHTMAP_BYTES,
        );

        Ok(Self {
            entities,
            shaders,
            planes,
            nodes,
            leafs,
            leaf_surfaces,
            leaf_brushes,
            models,
            brushes,
            brush_sides,
            draw_verts,
            draw_indexes,
            fogs,
            surfaces,
            lightmap_bytes,
            lightgrid_bytes,
            visibility,
        })
    }

    /// Nombre de lightmaps 128×128 embarquées.
    pub fn num_lightmaps(&self) -> usize {
        self.lightmap_bytes.len() / LIGHTMAP_BYTES
    }

    /// Retourne les bytes RGB (128*128*3) de la lightmap `i`.
    pub fn lightmap(&self, i: usize) -> Option<&[u8]> {
        let start = i.checked_mul(LIGHTMAP_BYTES)?;
        self.lightmap_bytes.get(start..start + LIGHTMAP_BYTES)
    }

    /// AABB axis-aligned d'un brush, reconstruit à partir de ses six plans
    /// axis-aligned `±x`, `±y`, `±z`. Q3 garantit que tous les brushes ont
    /// ces six sides (même pour les brushes non-axis-aligned, les sides
    /// « bevel » axis-aligned sont ajoutés au compile).
    ///
    /// Renvoie `None` si `brush_idx` est hors bornes, ou si un axe n'a pas
    /// ses deux faces opposées (brush mal formé — ne devrait pas arriver
    /// sur une map compilée correctement).
    pub fn brush_aabb(&self, brush_idx: usize) -> Option<Aabb> {
        let brush = self.brushes.get(brush_idx)?;
        let first = brush.first_side as usize;
        let last = first.checked_add(brush.num_sides as usize)?;
        let mut maxs = [0.0f32; 3];
        let mut mins = [0.0f32; 3];
        let mut have = [(false, false); 3];
        for side_i in first..last {
            let side = self.brush_sides.get(side_i)?;
            let p = self.planes.get(side.plane_num as usize)?;
            for axis in 0..3 {
                if (p.normal[axis] - 1.0).abs() < 1e-4 {
                    maxs[axis] = p.dist;
                    have[axis].0 = true;
                } else if (p.normal[axis] + 1.0).abs() < 1e-4 {
                    mins[axis] = -p.dist;
                    have[axis].1 = true;
                }
            }
        }
        if have.iter().all(|(h1, h2)| *h1 && *h2) {
            Some(Aabb::new(Vec3::from_array(mins), Vec3::from_array(maxs)))
        } else {
            None
        }
    }

    /// Vérifie si un cluster `from` peut voir un cluster `to` (PVS).
    pub fn cluster_visible(&self, from: i32, to: i32) -> bool {
        if from < 0 || to < 0 || self.visibility.cluster_bytes == 0 {
            return true;
        }
        let row = (from as usize) * (self.visibility.cluster_bytes as usize);
        let byte = row + (to as usize / 8);
        let bit = 1u8 << (to as usize % 8);
        self.visibility.data.get(byte).is_some_and(|&b| b & bit != 0)
    }
}

/// Convertit un slice de bytes en `Vec<T>` via bytemuck. Vérifie l'alignement
/// et la taille multiple.
fn parse_pod_vec<T: bytemuck::Pod>(bytes: &[u8], name: &str) -> Result<Vec<T>> {
    let size = std::mem::size_of::<T>();
    if bytes.len() % size != 0 {
        return Err(Error::bsp(format!(
            "lump {name} : {} bytes pas multiple de {size}",
            bytes.len()
        )));
    }
    // bytemuck::try_cast_slice vérifie l'alignement.
    let slice: &[T] = bytemuck::try_cast_slice(bytes).map_err(|e| {
        Error::bsp(format!("lump {name} cast: {e}"))
    })?;
    Ok(slice.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_magic() {
        let mut buf = vec![0u8; std::mem::size_of::<DHeader>()];
        buf[..4].copy_from_slice(b"FAIL");
        assert!(Bsp::parse(&buf).is_err());
    }

    #[test]
    fn rejects_bad_version() {
        let mut buf = vec![0u8; std::mem::size_of::<DHeader>()];
        buf[..4].copy_from_slice(&BSP_MAGIC);
        buf[4..8].copy_from_slice(&42i32.to_le_bytes());
        assert!(Bsp::parse(&buf).is_err());
    }

    #[test]
    fn empty_lumps_ok() {
        // Une map minimaliste avec IBSP+v46 mais tous lumps vides.
        let mut buf = vec![0u8; std::mem::size_of::<DHeader>()];
        buf[..4].copy_from_slice(&BSP_MAGIC);
        buf[4..8].copy_from_slice(&BSP_VERSION.to_le_bytes());
        let bsp = Bsp::parse(&buf).unwrap();
        assert_eq!(bsp.shaders.len(), 0);
        assert_eq!(bsp.num_lightmaps(), 0);
    }

    #[test]
    fn struct_sizes_match_spec() {
        // Tailles canoniques du format IBSP.
        assert_eq!(std::mem::size_of::<DLump>(), 8);
        assert_eq!(std::mem::size_of::<DHeader>(), 8 + 17 * 8);
        assert_eq!(std::mem::size_of::<DShader>(), 72);
        assert_eq!(std::mem::size_of::<DPlane>(), 16);
        assert_eq!(std::mem::size_of::<DNode>(), 36);
        assert_eq!(std::mem::size_of::<DLeaf>(), 48);
        assert_eq!(std::mem::size_of::<DModel>(), 40);
        assert_eq!(std::mem::size_of::<DBrush>(), 12);
        assert_eq!(std::mem::size_of::<DBrushSide>(), 8);
        assert_eq!(std::mem::size_of::<DrawVert>(), 44);
        assert_eq!(std::mem::size_of::<DFog>(), 72);
        assert_eq!(std::mem::size_of::<DSurface>(), 104);
    }
}
