//! Points d'intérêt — POI réels de la Réunion mappés en coordonnées
//! monde.  Coordonnées géographiques converties via projection
//! Mercator simple centrée sur l'île (lat -21.115°, lon 55.536°),
//! puis échelle 1/10.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoiKind {
    /// Capitale / grande ville — combat urbain dense, top-tier loot.
    City,
    /// Petite ville / village — mid-tier loot.
    Town,
    /// Plage / lagon — open ground.
    Beach,
    /// Cratère volcanique actif — danger zone, top-tier loot.
    Volcano,
    /// Sommet montagneux — vue panoramique, sniping.
    Peak,
    /// Cirque (Mafate / Cilaos / Salazie) — encaissé, mid-tier.
    Cirque,
    /// Forêt dense — couvert végétal.
    Forest,
    /// Zone industrielle / portuaire.
    Industrial,
    /// Aéroport.
    Airport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Poi {
    pub name: String,
    pub kind: PoiKind,
    /// Coordonnée X monde (unités Q3).
    pub x: f32,
    /// Coordonnée Y monde.
    pub y: f32,
    /// Rayon de la zone d'intérêt — sert au mini-map et au ring shrink
    /// pour ne pas couper un POI en deux (le ring final tombe sur un
    /// seul POI).
    pub radius: f32,
    /// Tier de loot (1 = pauvre, 4 = top-tier).  Donne au pipeline
    /// item-spawn la densité d'items à placer dans la zone.
    pub tier: u8,
}

/// POI canoniques de l'Île de la Réunion à l'échelle 1/10 — coords
/// dérivées de positions GPS réelles + Mercator local + scale 1/10.
///
/// Convention : (0,0) = centre île (lat -21.115°, lon 55.536°), X+ =
/// est, Y+ = nord. 1 unité = 25 cm, donc ~12 unités = 30 m réels après
/// 1/10. Pour une coord GPS (lat, lon) :
///
/// ```text
///   Δlat_m  = (lat + 21.115) * 111_320
///   Δlon_m  = (lon - 55.536) * 111_320 * cos(-21.115°)
///   y_unit  = Δlat_m * 100 / 25     // 1/10 + 0.25 m/unit
///   x_unit  = Δlon_m * 100 / 25
/// ```
///
/// Soit en pratique : **1 mètre réel ≈ 0.4 unités** après échelle.
pub fn reunion_pois() -> Vec<Poi> {
    // Helper inline : (lat°, lon°) → (x, y) unités carte.
    fn project(lat: f64, lon: f64) -> (f32, f32) {
        const CENTER_LAT: f64 = -21.115;
        const CENTER_LON: f64 = 55.536;
        const EARTH_M_PER_DEG: f64 = 111_320.0;
        let dlat_m = (lat - CENTER_LAT) * EARTH_M_PER_DEG;
        let dlon_m = (lon - CENTER_LON) * EARTH_M_PER_DEG
            * (CENTER_LAT.to_radians()).cos();
        // 1 m réel × (1/10 scale) × (1 unit / 0.25 m) = 0.4 unit/m
        let scale: f64 = 0.4;
        ((dlon_m * scale) as f32, (dlat_m * scale) as f32)
    }

    let mk = |name: &str, kind, lat, lon, radius, tier| {
        let (x, y) = project(lat, lon);
        Poi {
            name: name.into(),
            kind,
            x,
            y,
            radius,
            tier,
        }
    };

    vec![
        // Capitale et grandes villes
        mk("Saint-Denis", PoiKind::City, -20.879, 55.448, 600.0, 4),
        mk("Saint-Pierre", PoiKind::City, -21.342, 55.478, 500.0, 4),
        mk("Saint-Paul", PoiKind::City, -21.009, 55.270, 450.0, 3),
        mk("Le Tampon", PoiKind::Town, -21.278, 55.515, 380.0, 3),
        mk("Saint-Benoit", PoiKind::Town, -21.034, 55.713, 380.0, 3),
        mk("Saint-Louis", PoiKind::Town, -21.286, 55.412, 320.0, 3),
        mk("Saint-Joseph", PoiKind::Town, -21.382, 55.617, 300.0, 2),
        mk("Saint-André", PoiKind::Town, -20.961, 55.652, 300.0, 2),
        mk("Saint-Leu", PoiKind::Town, -21.171, 55.292, 280.0, 2),
        // Stations balnéaires / lagons
        mk("Saint-Gilles-les-Bains", PoiKind::Beach, -21.060, 55.222, 400.0, 4),
        mk("L'Hermitage", PoiKind::Beach, -21.083, 55.230, 300.0, 3),
        mk("Boucan Canot", PoiKind::Beach, -21.043, 55.227, 220.0, 3),
        mk("L'Étang-Salé", PoiKind::Beach, -21.265, 55.345, 280.0, 3),
        // Volcans / sommets
        mk("Piton de la Fournaise", PoiKind::Volcano, -21.244, 55.708, 450.0, 4),
        mk("Piton des Neiges", PoiKind::Peak, -21.099, 55.477, 350.0, 4),
        mk("Maïdo", PoiKind::Peak, -21.072, 55.380, 280.0, 3),
        mk("Le Dimitile", PoiKind::Peak, -21.220, 55.503, 200.0, 2),
        // Cirques (intérieur des terres)
        mk("Mafate", PoiKind::Cirque, -21.071, 55.422, 600.0, 4),
        mk("Cilaos", PoiKind::Cirque, -21.135, 55.471, 480.0, 3),
        mk("Salazie", PoiKind::Cirque, -21.030, 55.539, 500.0, 3),
        mk("Hell-Bourg", PoiKind::Town, -21.064, 55.521, 200.0, 2),
        // Forêts / zones naturelles
        mk("Forêt de Bélouve", PoiKind::Forest, -21.072, 55.555, 400.0, 2),
        mk("Forêt de Bébour", PoiKind::Forest, -21.116, 55.548, 380.0, 2),
        mk("Plaine des Cafres", PoiKind::Forest, -21.230, 55.580, 350.0, 2),
        mk("Plaine des Palmistes", PoiKind::Forest, -21.135, 55.658, 320.0, 2),
        // Zones industrielles / infra
        mk("Le Port", PoiKind::Industrial, -20.940, 55.291, 380.0, 4),
        mk("Aéroport Roland-Garros", PoiKind::Airport, -20.886, 55.510, 350.0, 4),
        mk("Aéroport de Pierrefonds", PoiKind::Airport, -21.323, 55.428, 220.0, 3),
        // Iconiques touristiques
        mk("Cap Méchant", PoiKind::Beach, -21.396, 55.654, 180.0, 2),
        mk("Anse des Cascades", PoiKind::Beach, -21.198, 55.832, 200.0, 2),
        mk("Grand Bassin", PoiKind::Cirque, -21.213, 55.544, 200.0, 2),
        mk("Trou de Fer", PoiKind::Peak, -21.083, 55.583, 220.0, 3),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reunion_has_30plus_pois() {
        let pois = reunion_pois();
        assert!(pois.len() >= 30, "found {} POIs", pois.len());
    }

    #[test]
    fn capitals_are_top_tier() {
        let pois = reunion_pois();
        let st_denis = pois.iter().find(|p| p.name == "Saint-Denis").unwrap();
        assert_eq!(st_denis.tier, 4);
        assert_eq!(st_denis.kind, PoiKind::City);
    }

    #[test]
    fn pois_are_within_map_bounds() {
        // Carte par défaut Réunion 1/10 = 28 800 × 26 400 unités
        // (cf. `TerrainMeta::reunion_default`).  Demi-bounds avec
        // marge ocean = ±14 400 / ±13 200.
        let pois = reunion_pois();
        for p in &pois {
            assert!(
                p.x.abs() < 14_400.0 && p.y.abs() < 13_200.0,
                "POI '{}' hors carte : ({}, {})",
                p.name,
                p.x,
                p.y
            );
        }
    }

    #[test]
    fn volcano_is_east_of_center() {
        // Le Piton de la Fournaise est dans la moitié est de l'île.
        let pois = reunion_pois();
        let pdf = pois
            .iter()
            .find(|p| p.kind == PoiKind::Volcano)
            .unwrap();
        assert!(pdf.x > 0.0, "Piton de la Fournaise devrait être à l'est : x={}", pdf.x);
    }
}
