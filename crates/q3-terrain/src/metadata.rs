//! Métadonnées d'un terrain — sérialisée en JSON à côté du `.r16`.

use serde::{Deserialize, Serialize};

use crate::poi::Poi;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainMeta {
    /// Nom court (utilisé en log + comme clé dans le menu).
    pub name: String,
    /// Largeur du heightmap en samples.
    pub width: usize,
    /// Hauteur du heightmap en samples.
    pub height: usize,
    /// Hauteur monde correspondant à `samples == 0`.
    pub z_min: f32,
    /// Hauteur monde correspondant à `samples == 65535`.
    pub z_max: f32,
    /// Coordonnée X monde du sample (0,0).  Convention : centre carte
    /// au monde (0,0), origine = `-(width/2) * units_per_sample`.
    pub origin_x: f32,
    /// Coordonnée Y monde du sample (0,0).
    pub origin_y: f32,
    /// Combien d'unités Q3 sépare deux samples consécutifs.  Pour la
    /// Réunion 1/10 avec 1 unité = 25 cm et SRTM 30 m → 30 m / 0.25 m
    /// / 10 = **12 unités/sample**.  Carte 25 200×18 000 = 2100×1500
    /// samples → 4.5 M tris non-LOD.
    pub units_per_sample: f32,
    /// Hauteur monde de l'océan (terrain hors-carte renvoie cette
    /// valeur). Typiquement 0 ou légèrement négative pour que les
    /// plages plongent visuellement dans la mer.
    pub ocean_z: f32,
    /// Hauteur monde du niveau d'eau (lagons, lacs, rivières) — au-
    /// dessus, terrain visible ; en dessous, biome aquatique.
    pub water_level: f32,
    /// POI géolocalisés sur la carte (capitale, volcan, plages...).
    pub pois: Vec<Poi>,
}

impl TerrainMeta {
    /// Constructeur de la carte canonique « Réunion 1/10 ».  Chargé
    /// par défaut quand on lance `\map br_reunion`.  Toutes les
    /// constantes ici reflètent l'île réelle après échelle 1/10 et
    /// projection en unités Q3 (1 unité = 25 cm).
    pub fn reunion_default() -> Self {
        // L'île réelle fait ~70×50 km (bounds lat -21.40..-20.87, lon
        // 55.20..55.84) — un poil plus que les 63×45 km grand axe car
        // on ajoute de l'eau autour pour le ring BR initial.  Échelle
        // 1/10 horizontale × 1 unité = 25 cm → 0.4 unité/m.
        //   70 km × 0.4 = 28 000 unités → on prend 2400 samples × 12 u
        //                = 28 800 unités (marge océan)
        //   50 km × 0.4 = 20 000 unités → on prend 1800 samples × 12 u
        //                = 21 600 unités (marge océan)
        // SRTM 30 m → après /10 → 3 m / sample → 12 unités/sample reste cohérent.
        // 2400 × 1800 = 4.32 M samples → quadtree LOD obligatoire.
        let units_per_sample = 12.0;
        let width = 2400;
        // 2200 samples × 12 u = 26 400 unités → couvre les ~30 km
        // d'extension N-S réels avec marge océan, et fait rentrer
        // Saint-Joseph (~-12 000 u) et Saint-Denis (~+10 500 u) sans
        // tronquer.
        let height = 2200;
        let map_w = width as f32 * units_per_sample;
        let map_h = height as f32 * units_per_sample;
        Self {
            name: "br_reunion".into(),
            width,
            height,
            // Z : océan -200, Piton des Neiges 3071 m → /10 → 307 m
            // → 1228 unités à 1 unité = 25 cm. On garde 1228.
            z_min: -200.0,
            z_max: 1228.0,
            origin_x: -map_w * 0.5,
            origin_y: -map_h * 0.5,
            units_per_sample,
            ocean_z: 0.0,
            water_level: 4.0, // niveau lagon, légèrement au-dessus de l'océan
            pois: crate::poi::reunion_pois(),
        }
    }
}
