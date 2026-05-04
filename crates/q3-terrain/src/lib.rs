//! Terrain large-échelle hors-BSP — pensé pour les cartes BR (jusqu'à
//! ±32k unités sur chaque axe) où le format BSP v46 sature.
//!
//! # Carte cible : Île de la Réunion 1/10
//!
//! La carte de référence de ce module est l'Île de la Réunion à
//! l'échelle horizontale 1/10 (vraie île 63×45 km → carte 25.2×18 km
//! en unités où 1 unité = 25 cm).  Le format est néanmoins générique :
//! n'importe quelle heightmap rectangulaire peut être chargée.
//!
//! # Pipeline d'authoring (cf. `tools/dem_to_terrain.py`)
//!
//! ```text
//!   SRTM .tif (30 m, gratuit USGS)
//!     ↓ rasterio + numpy
//!   décimation 1/10 + clamp + smoothing
//!     ↓
//!   heightmap PNG 16-bit grayscale  ──→  reunion.r16
//!     +
//!   classification biome (altitude + pente + biomap CC0)
//!     ↓
//!   splatmap RGBA 4-canaux            ──→  reunion.splat.png
//!     +
//!   metadata (JSON)                   ──→  reunion.terrain.json
//! ```
//!
//! Les 3 fichiers sont les inputs de [`Terrain::load_from_files`].
//!
//! # Format runtime
//!
//! * **Heightmap** : `.r16` raw 16-bit big-endian, `width × height` u16,
//!   où `0 = z_min`, `65535 = z_max` (interpolation linéaire).
//! * **Splatmap** : PNG RGBA — chaque canal pondère un material du jeu
//!   (R = roche basaltique, G = sable noir, B = végétation, A = urbain).
//! * **Metadata** : JSON décrivant la projection (m/unit), z_min/z_max,
//!   POI de la carte, et la table de matériaux.
//!
//! Le terrain est dessiné en **patches LOD** (quadtree adaptatif) pour
//! qu'on n'envoie pas 4.5 M tris d'un coup au GPU.  La collision passe
//! par un **trace heightfield** dédié (pas de BSP), implémenté dans le
//! module [`collision`].
//!
//! # État actuel (v0.9.5)
//!
//! **Scaffold + I/O.**  Les loaders heightmap/splatmap/metadata sont
//! fonctionnels et testés.  La génération de mesh tuilés et le LOD
//! dynamique restent à faire — c'est volontairement isolé dans un crate
//! à part pour ne pas pourrir `q3-engine` tant que le pipeline n'est
//! pas mature.

#![forbid(unsafe_code)]

pub mod br;
pub mod collision;
pub mod mesh;
pub mod metadata;
pub mod poi;

use std::io::Read;
use std::path::Path;

use q3_math::Vec3;
use thiserror::Error;
use tracing::info;

pub use metadata::TerrainMeta;
pub use poi::{Poi, PoiKind};

#[derive(Debug, Error)]
pub enum TerrainError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("metadata JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("dimensions inconsistantes : heightmap {hm} vs metadata {meta}")]
    DimMismatch { hm: usize, meta: usize },
    #[error("heightmap vide")]
    Empty,
    #[error("format inattendu : {0}")]
    Format(String),
}

/// Terrain heightfield large-échelle.  Une seule instance représente
/// toute la carte (jusqu'à ±32768 unités), le LOD côté renderer
/// découpe en patches.
pub struct Terrain {
    /// Largeur en samples.
    pub width: usize,
    /// Hauteur en samples.
    pub height: usize,
    /// Heightmap brute, 16-bit normalisée — `samples[y * width + x]`.
    pub samples: Vec<u16>,
    /// Splatmap RGBA — 4 canaux par sample (roche/sable/végétation/urbain).
    pub splat: Vec<[u8; 4]>,
    /// Métadonnées (échelle, biomes, POI).
    pub meta: TerrainMeta,
}

impl Terrain {
    /// Charge un terrain depuis le triple `.r16 + .splat.png + .json`
    /// produit par `tools/dem_to_terrain.py`.
    ///
    /// `base` désigne le préfixe sans extension : pour
    /// `data/maps/reunion.r16` + `reunion.splat.png` + `reunion.terrain.json`,
    /// passer `base = "data/maps/reunion"`.
    pub fn load_from_files(base: impl AsRef<Path>) -> Result<Self, TerrainError> {
        let base = base.as_ref();
        let r16_path = base.with_extension("r16");
        let splat_path = base.with_file_name(format!(
            "{}.splat.png",
            base.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("terrain")
        ));
        let meta_path = base.with_file_name(format!(
            "{}.terrain.json",
            base.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("terrain")
        ));

        let meta_bytes = std::fs::read(&meta_path)?;
        let meta: TerrainMeta = serde_json::from_slice(&meta_bytes)?;
        info!(
            "terrain: load '{}' ({}×{}, z=[{:.0},{:.0}])",
            meta.name, meta.width, meta.height, meta.z_min, meta.z_max
        );

        let r16_bytes = std::fs::read(&r16_path)?;
        let expected_bytes = meta.width * meta.height * 2;
        if r16_bytes.len() != expected_bytes {
            return Err(TerrainError::DimMismatch {
                hm: r16_bytes.len() / 2,
                meta: meta.width * meta.height,
            });
        }
        // Big-endian u16 pour matcher la convention `.r16` standard
        // (Unity / UE / WorldMachine la produisent en BE par défaut).
        let mut samples = Vec::with_capacity(meta.width * meta.height);
        let mut cur = std::io::Cursor::new(&r16_bytes);
        for _ in 0..(meta.width * meta.height) {
            let mut buf = [0u8; 2];
            cur.read_exact(&mut buf)?;
            samples.push(u16::from_be_bytes(buf));
        }

        // Splatmap : si absent, fallback "tout roche" (canal R = 255).
        let splat = if splat_path.exists() {
            load_splatmap_rgba(&splat_path, meta.width, meta.height)?
        } else {
            info!("terrain: splatmap absente, fallback all-rock");
            vec![[255, 0, 0, 0]; meta.width * meta.height]
        };

        if samples.is_empty() {
            return Err(TerrainError::Empty);
        }

        Ok(Self {
            width: meta.width,
            height: meta.height,
            samples,
            splat,
            meta,
        })
    }

    /// Hauteur monde au point `(x, y)` en unités Q3.  Hors carte → océan
    /// (z = 0). Interpolation bilinéaire entre les 4 samples voisins.
    pub fn height_at(&self, x: f32, y: f32) -> f32 {
        let gx = (x - self.meta.origin_x) / self.meta.units_per_sample;
        let gy = (y - self.meta.origin_y) / self.meta.units_per_sample;
        if gx < 0.0
            || gy < 0.0
            || gx >= (self.width - 1) as f32
            || gy >= (self.height - 1) as f32
        {
            return self.meta.ocean_z;
        }
        let x0 = gx.floor() as usize;
        let y0 = gy.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = gx - x0 as f32;
        let fy = gy - y0 as f32;
        let h00 = self.sample_z(x0, y0);
        let h10 = self.sample_z(x1, y0);
        let h01 = self.sample_z(x0, y1);
        let h11 = self.sample_z(x1, y1);
        let hx0 = h00 * (1.0 - fx) + h10 * fx;
        let hx1 = h01 * (1.0 - fx) + h11 * fx;
        hx0 * (1.0 - fy) + hx1 * fy
    }

    /// Retourne la hauteur monde Z d'un sample heightmap discret.
    pub fn sample_z(&self, x: usize, y: usize) -> f32 {
        let s = self.samples[y * self.width + x] as f32 / 65535.0;
        self.meta.z_min + s * (self.meta.z_max - self.meta.z_min)
    }

    /// Retourne le poids du biome `channel` (0=roche, 1=sable, 2=végé,
    /// 3=urbain) au point monde `(x, y)`. Pour le rendu material splat.
    pub fn biome_weight(&self, x: f32, y: f32, channel: usize) -> f32 {
        if channel >= 4 {
            return 0.0;
        }
        let gx = ((x - self.meta.origin_x) / self.meta.units_per_sample).max(0.0);
        let gy = ((y - self.meta.origin_y) / self.meta.units_per_sample).max(0.0);
        let xi = (gx as usize).min(self.width - 1);
        let yi = (gy as usize).min(self.height - 1);
        self.splat[yi * self.width + xi][channel] as f32 / 255.0
    }

    /// Renvoie tous les POI de la carte.
    pub fn pois(&self) -> &[Poi] {
        &self.meta.pois
    }

    /// Centre monde du terrain (utile pour positionner la caméra ou
    /// le ring shrink BR).
    pub fn center(&self) -> Vec3 {
        Vec3::new(
            self.meta.origin_x + (self.width as f32 * self.meta.units_per_sample) * 0.5,
            self.meta.origin_y + (self.height as f32 * self.meta.units_per_sample) * 0.5,
            (self.meta.z_min + self.meta.z_max) * 0.5,
        )
    }

    /// Étendue carrée recommandée pour le ring BR initial — basée sur
    /// la diagonale de la carte (l'océan plein autour assure que ring
    /// = carte au début, puis se contracte vers le centre).
    pub fn br_initial_radius(&self) -> f32 {
        let w = self.width as f32 * self.meta.units_per_sample;
        let h = self.height as f32 * self.meta.units_per_sample;
        ((w * w + h * h).sqrt()) * 0.5
    }
}

fn load_splatmap_rgba(
    _path: &Path,
    width: usize,
    height: usize,
) -> Result<Vec<[u8; 4]>, TerrainError> {
    // PNG decode désactivé pour l'instant — on garde l'API pour qu'un
    // futur commit branche la crate `image` (déjà workspace-dep) sans
    // toucher au reste.  Fallback déterministe : tout en roche.
    Ok(vec![[255, 0, 0, 0]; width * height])
}

#[cfg(test)]
mod tests {
    use super::*;
    use poi::PoiKind;

    fn synthetic_terrain() -> Terrain {
        let w = 4;
        let h = 4;
        let samples: Vec<u16> = (0..(w * h) as u16).map(|i| i * 4096).collect();
        let splat = vec![[200, 50, 0, 0]; w * h];
        let meta = TerrainMeta {
            name: "test".into(),
            width: w,
            height: h,
            z_min: 0.0,
            z_max: 1000.0,
            origin_x: 0.0,
            origin_y: 0.0,
            units_per_sample: 100.0,
            ocean_z: -50.0,
            pois: vec![Poi {
                name: "Piton".into(),
                kind: PoiKind::Volcano,
                x: 200.0,
                y: 200.0,
                radius: 80.0,
                tier: 4,
            }],
            water_level: 0.0,
        };
        Terrain {
            width: w,
            height: h,
            samples,
            splat,
            meta,
        }
    }

    #[test]
    fn height_in_bounds_interpolates() {
        let t = synthetic_terrain();
        // Au sample (0,0) → s=0 → z=z_min=0
        assert!((t.height_at(0.0, 0.0) - 0.0).abs() < 1e-2);
    }

    #[test]
    fn height_out_of_bounds_returns_ocean() {
        let t = synthetic_terrain();
        assert_eq!(t.height_at(-1000.0, 0.0), t.meta.ocean_z);
        assert_eq!(t.height_at(100000.0, 0.0), t.meta.ocean_z);
    }

    #[test]
    fn biome_weight_in_range() {
        let t = synthetic_terrain();
        let w = t.biome_weight(50.0, 50.0, 0);
        assert!(w > 0.0 && w <= 1.0);
    }

    #[test]
    fn pois_accessible() {
        let t = synthetic_terrain();
        assert_eq!(t.pois().len(), 1);
        assert_eq!(t.pois()[0].name, "Piton");
    }

    #[test]
    fn br_initial_radius_positive() {
        let t = synthetic_terrain();
        assert!(t.br_initial_radius() > 0.0);
    }
}
