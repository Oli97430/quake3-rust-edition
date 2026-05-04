#!/usr/bin/env python3
"""
DEM → terrain pipeline pour Q3 Rust Edition.

Convertit un raster d'élévation (SRTM .tif / .hgt / GeoTIFF) en triple
de fichiers consommé par `q3-terrain` :

    <output>.r16             → heightmap brute u16 big-endian
    <output>.splat.png       → splatmap RGBA 4-canaux (roche/sable/végé/urbain)
    <output>.terrain.json    → métadonnées (TerrainMeta sérialisée)

# Pipeline Réunion 1/10

  1. Récupérer le SRTM 30 m couvrant l'île :
     https://earthexplorer.usgs.gov/
     -> chercher "SRTM 1 Arc-Second Global"
     -> tiles s22e055.hgt (la Réunion tient dans une seule tile 1°×1°)

  2. Convertir HGT → GeoTIFF si besoin (gdal_translate).

  3. Lancer ce script :

         python tools/dem_to_terrain.py \
             --input data/srtm/s22e055.tif \
             --output assets/maps/br_reunion \
             --bbox-lat -21.40 -20.85 \
             --bbox-lon 55.20 55.85 \
             --target-width 2100 \
             --target-height 1500 \
             --units-per-sample 12 \
             --z-min -200 \
             --z-max 1228 \
             --preset reunion

  4. Vérifier le `.json` produit, lancer le jeu : `\map br_reunion`.

# Dépendances Python

    pip install rasterio numpy Pillow

# État

Script de référence — pour le moment du **scaffolding documenté** : il
contient l'ossature du traitement. Les étapes I/O réelles (rasterio,
classification biome, splat output) sont commentées comme `TODO` dans
les passages où l'asset DEM doit être effectivement présent. Lance le
script avec un .tif valide pour le voir tourner.
"""

import argparse
import json
import math
import os
import struct
import sys
from dataclasses import dataclass, field
from typing import List, Tuple


# -----------------------------------------------------------------------------
#   POI canoniques de la Réunion — synchronisés avec `crates/q3-terrain/src/poi.rs`
# -----------------------------------------------------------------------------

@dataclass
class PoiSpec:
    name: str
    kind: str  # "City" | "Town" | "Beach" | "Volcano" | "Peak" | "Cirque" | "Forest" | "Industrial" | "Airport"
    lat: float
    lon: float
    radius: float
    tier: int


REUNION_POIS: List[PoiSpec] = [
    PoiSpec("Saint-Denis",            "City",       -20.879, 55.448, 600.0, 4),
    PoiSpec("Saint-Pierre",           "City",       -21.342, 55.478, 500.0, 4),
    PoiSpec("Saint-Paul",             "City",       -21.009, 55.270, 450.0, 3),
    PoiSpec("Le Tampon",              "Town",       -21.278, 55.515, 380.0, 3),
    PoiSpec("Saint-Benoit",           "Town",       -21.034, 55.713, 380.0, 3),
    PoiSpec("Saint-Louis",            "Town",       -21.286, 55.412, 320.0, 3),
    PoiSpec("Saint-Joseph",           "Town",       -21.382, 55.617, 300.0, 2),
    PoiSpec("Saint-André",            "Town",       -20.961, 55.652, 300.0, 2),
    PoiSpec("Saint-Leu",              "Town",       -21.171, 55.292, 280.0, 2),
    PoiSpec("Saint-Gilles-les-Bains", "Beach",      -21.060, 55.222, 400.0, 4),
    PoiSpec("L'Hermitage",            "Beach",      -21.083, 55.230, 300.0, 3),
    PoiSpec("Boucan Canot",           "Beach",      -21.043, 55.227, 220.0, 3),
    PoiSpec("L'Étang-Salé",           "Beach",      -21.265, 55.345, 280.0, 3),
    PoiSpec("Piton de la Fournaise",  "Volcano",    -21.244, 55.708, 450.0, 4),
    PoiSpec("Piton des Neiges",       "Peak",       -21.099, 55.477, 350.0, 4),
    PoiSpec("Maïdo",                  "Peak",       -21.072, 55.380, 280.0, 3),
    PoiSpec("Le Dimitile",            "Peak",       -21.220, 55.503, 200.0, 2),
    PoiSpec("Mafate",                 "Cirque",     -21.071, 55.422, 600.0, 4),
    PoiSpec("Cilaos",                 "Cirque",     -21.135, 55.471, 480.0, 3),
    PoiSpec("Salazie",                "Cirque",     -21.030, 55.539, 500.0, 3),
    PoiSpec("Hell-Bourg",             "Town",       -21.064, 55.521, 200.0, 2),
    PoiSpec("Forêt de Bélouve",       "Forest",     -21.072, 55.555, 400.0, 2),
    PoiSpec("Forêt de Bébour",        "Forest",     -21.116, 55.548, 380.0, 2),
    PoiSpec("Plaine des Cafres",      "Forest",     -21.230, 55.580, 350.0, 2),
    PoiSpec("Plaine des Palmistes",   "Forest",     -21.135, 55.658, 320.0, 2),
    PoiSpec("Le Port",                "Industrial", -20.940, 55.291, 380.0, 4),
    PoiSpec("Aéroport Roland-Garros", "Airport",    -20.886, 55.510, 350.0, 4),
    PoiSpec("Aéroport de Pierrefonds","Airport",    -21.323, 55.428, 220.0, 3),
    PoiSpec("Cap Méchant",            "Beach",      -21.396, 55.654, 180.0, 2),
    PoiSpec("Anse des Cascades",      "Beach",      -21.198, 55.832, 200.0, 2),
    PoiSpec("Grand Bassin",           "Cirque",     -21.213, 55.544, 200.0, 2),
    PoiSpec("Trou de Fer",            "Peak",       -21.083, 55.583, 220.0, 3),
]


# -----------------------------------------------------------------------------
#   Projection Mercator locale → coordonnées monde (unités Q3)
# -----------------------------------------------------------------------------

def project_lat_lon_to_world(lat: float, lon: float, center_lat: float,
                              center_lon: float, scale: float) -> Tuple[float, float]:
    """
    Mercator local centré sur (center_lat, center_lon) à `scale` unités/m.

    Retourne (x, y) en unités Q3, X+ = est, Y+ = nord.
    Pour la Réunion 1/10 + 1 unit = 25 cm → scale = 0.4 unités/m.
    """
    EARTH_M_PER_DEG = 111_320.0
    dlat_m = (lat - center_lat) * EARTH_M_PER_DEG
    dlon_m = (lon - center_lon) * EARTH_M_PER_DEG * math.cos(math.radians(center_lat))
    return (dlon_m * scale, dlat_m * scale)


# -----------------------------------------------------------------------------
#   Splat classification — biome par altitude + pente + zone urbaine
# -----------------------------------------------------------------------------

def classify_biome(z: float, slope: float, urban_mask: float) -> Tuple[int, int, int, int]:
    """
    Retourne (rock, sand, vegetation, urban) en 0..255.

    Heuristique pour la Réunion :
      * < 5 m altitude → sable + un peu d'eau (lagons)
      * 5..200 m faible pente → végétation tropicale + un peu d'urbain
      * 200..1500 m + faible pente → forêt / plaine
      * > 1500 m → roche basaltique / sommet
      * pente forte > 30° → roche pure
      * urban_mask injecté depuis OSM ou shapefiles villes (optionnel)

    Renvoie 4 poids 0..255 dont la somme idéalement ~255.
    """
    rock, sand, veg, urban = 0, 0, 0, 0

    # Pente forte → roche pure.
    if slope > 0.6:  # ~31°
        return (255, 0, 0, 0)

    if z < 5.0:
        sand = 200
        veg = 30
    elif z < 200.0:
        veg = 200
        rock = 30
        sand = 10
    elif z < 1500.0:
        veg = 220
        rock = 30
    elif z < 2500.0:
        rock = 200
        veg = 50
    else:  # sommets
        rock = 255

    # Urban override (depuis OSM mask, si fourni)
    if urban_mask > 0.5:
        urban = 220
        veg = max(0, veg - 100)

    # Normalise vers 255 max.
    total = rock + sand + veg + urban
    if total > 0:
        scale = 255.0 / max(255, total)
        rock, sand, veg, urban = (int(c * scale) for c in (rock, sand, veg, urban))
    return (rock, sand, veg, urban)


# -----------------------------------------------------------------------------
#   Pipeline principal
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="DEM → q3-terrain pipeline")
    p.add_argument("--input", required=True, help="DEM GeoTIFF / HGT input path")
    p.add_argument("--output", required=True, help="Output base path (no extension)")
    p.add_argument("--bbox-lat", nargs=2, type=float, required=True,
                   metavar=("LAT_S", "LAT_N"),
                   help="Bounding box lat (sud, nord). Ex: -21.40 -20.85")
    p.add_argument("--bbox-lon", nargs=2, type=float, required=True,
                   metavar=("LON_W", "LON_E"),
                   help="Bounding box lon (ouest, est). Ex: 55.20 55.85")
    p.add_argument("--target-width", type=int, default=2100)
    p.add_argument("--target-height", type=int, default=1500)
    p.add_argument("--units-per-sample", type=float, default=12.0)
    p.add_argument("--z-min", type=float, default=-200.0)
    p.add_argument("--z-max", type=float, default=1228.0)
    p.add_argument("--preset", choices=["reunion", "generic"], default="reunion")
    p.add_argument("--name", default="br_reunion")
    args = p.parse_args()

    print(f"[dem_to_terrain] input  = {args.input}")
    print(f"[dem_to_terrain] output = {args.output}.{{r16,splat.png,terrain.json}}")
    print(f"[dem_to_terrain] target = {args.target_width} × {args.target_height} samples")

    # ========================================================================
    #   Étape 1 — chargement DEM
    # ========================================================================
    try:
        import rasterio
        import numpy as np
    except ImportError:
        print("\nERREUR : rasterio + numpy requis. Install :", file=sys.stderr)
        print("    pip install rasterio numpy Pillow\n", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(args.input):
        print(f"ERREUR : DEM input introuvable : {args.input}", file=sys.stderr)
        print("Récupère SRTM s22e055 sur https://earthexplorer.usgs.gov/", file=sys.stderr)
        sys.exit(2)

    with rasterio.open(args.input) as src:
        print(f"[dem_to_terrain] DEM source : {src.width}×{src.height}, CRS={src.crs}")
        # Crop sur la bounding box demandée.
        from rasterio.windows import from_bounds
        win = from_bounds(args.bbox_lon[0], args.bbox_lat[0],
                           args.bbox_lon[1], args.bbox_lat[1], src.transform)
        dem = src.read(1, window=win).astype(np.float32)
        print(f"[dem_to_terrain] DEM crop   : {dem.shape[1]}×{dem.shape[0]}")

    # ========================================================================
    #   Étape 2 — décimation vers la résolution cible
    # ========================================================================
    from PIL import Image
    img = Image.fromarray(dem, mode="F")
    img = img.resize((args.target_width, args.target_height), resample=Image.BILINEAR)
    dem_resized = np.array(img, dtype=np.float32)
    # Z out-of-ocean → 0 (DEM SRTM met -32768 hors-data parfois).
    dem_resized[dem_resized < -1000] = 0.0
    print(f"[dem_to_terrain] resized    : {dem_resized.shape[1]}×{dem_resized.shape[0]} "
          f"min={dem_resized.min():.0f} max={dem_resized.max():.0f}")

    # ========================================================================
    #   Étape 3 — DEM mètres → Q3 unités → u16 normalisé
    # ========================================================================
    # 1 m réel × 1/10 × 1/0.25 = 0.4 unités Q3
    # Carte Z : z_min .. z_max en unités. SRTM est en mètres réels.
    # Pour préset "reunion" : alt max ~3071 m → /10 → 307 m → 1228 unités.
    z_scale_m_to_unit = 0.4  # = (1/10 scale) × (1/0.25 m per unit)
    dem_units = dem_resized * z_scale_m_to_unit
    z_min_u = float(args.z_min)
    z_max_u = float(args.z_max)
    norm = (dem_units - z_min_u) / (z_max_u - z_min_u)
    norm = np.clip(norm, 0.0, 1.0)
    samples_u16 = (norm * 65535).astype(np.uint16)

    # Écriture .r16 big-endian
    r16_path = f"{args.output}.r16"
    os.makedirs(os.path.dirname(r16_path) or ".", exist_ok=True)
    with open(r16_path, "wb") as f:
        # numpy `tobytes()` est little-endian par défaut → flip.
        be = samples_u16.byteswap().tobytes() if samples_u16.dtype.byteorder == "<" \
             else samples_u16.tobytes()
        # Plus portable : on écrit explicitement en BE.
        for v in samples_u16.flatten():
            f.write(struct.pack(">H", int(v)))
    print(f"[dem_to_terrain] r16 écrit  : {r16_path} ({os.path.getsize(r16_path)} bytes)")

    # ========================================================================
    #   Étape 4 — splatmap (4 canaux : rock/sand/veg/urban)
    # ========================================================================
    # Calcule pente local par diff finite.
    h, w = dem_units.shape
    dz_dx = np.gradient(dem_units, axis=1)
    dz_dy = np.gradient(dem_units, axis=0)
    slope = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    # Normalise pente sur [0, 1] : ~50 unités de delta sur sample 12u → ratio 4.
    slope_n = np.clip(slope / 8.0, 0.0, 1.0)

    # Urban mask : pour MVP, on tag les samples PROCHE des coords de villes
    # (Saint-Denis / Pierre / etc.). Plus tard : OSM raster.
    urban_mask = np.zeros_like(dem_units)
    sw_x = (args.bbox_lon[1] - args.bbox_lon[0]) / float(w)
    sw_y = (args.bbox_lat[1] - args.bbox_lat[0]) / float(h)
    for poi in REUNION_POIS:
        if poi.kind not in ("City", "Town", "Industrial", "Airport"):
            continue
        # Convertit lat/lon en index pixel dans la heightmap.
        px = int((poi.lon - args.bbox_lon[0]) / sw_x)
        py = int((args.bbox_lat[1] - poi.lat) / sw_y)
        # Rayon en pixels — heuristique 80 m → ~ poi.radius * meters_per_pixel.
        # ~ 30 m / sample en SRTM raw ; ici les samples sont les target ones,
        # donc ~bbox_lon range / w * 111320 * cos(lat) ≈ ... simplifié :
        rad_pix = max(1, int(poi.radius / 30.0))  # rough
        for dy in range(-rad_pix, rad_pix + 1):
            for dx in range(-rad_pix, rad_pix + 1):
                if dx * dx + dy * dy <= rad_pix * rad_pix:
                    yy = py + dy
                    xx = px + dx
                    if 0 <= xx < w and 0 <= yy < h:
                        urban_mask[yy, xx] = 1.0

    # Classification.
    splat = np.zeros((h, w, 4), dtype=np.uint8)
    for yy in range(h):
        for xx in range(w):
            r, s, v, u = classify_biome(
                float(dem_units[yy, xx]),
                float(slope_n[yy, xx]),
                float(urban_mask[yy, xx]),
            )
            splat[yy, xx] = (r, s, v, u)
    splat_path = f"{args.output}.splat.png"
    Image.fromarray(splat, "RGBA").save(splat_path)
    print(f"[dem_to_terrain] splat OK   : {splat_path}")

    # ========================================================================
    #   Étape 5 — JSON metadata
    # ========================================================================
    if args.preset == "reunion":
        center_lat = (args.bbox_lat[0] + args.bbox_lat[1]) * 0.5
        center_lon = (args.bbox_lon[0] + args.bbox_lon[1]) * 0.5
    else:
        center_lat = (args.bbox_lat[0] + args.bbox_lat[1]) * 0.5
        center_lon = (args.bbox_lon[0] + args.bbox_lon[1]) * 0.5

    pois_json = []
    for poi in REUNION_POIS if args.preset == "reunion" else []:
        x, y = project_lat_lon_to_world(
            poi.lat, poi.lon, center_lat, center_lon, scale=z_scale_m_to_unit
        )
        pois_json.append({
            "name": poi.name,
            "kind": poi.kind,
            "x": x,
            "y": y,
            "radius": poi.radius,
            "tier": poi.tier,
        })

    map_w = args.target_width * args.units_per_sample
    map_h = args.target_height * args.units_per_sample
    meta = {
        "name": args.name,
        "width": args.target_width,
        "height": args.target_height,
        "z_min": z_min_u,
        "z_max": z_max_u,
        "origin_x": -map_w / 2.0,
        "origin_y": -map_h / 2.0,
        "units_per_sample": args.units_per_sample,
        "ocean_z": 0.0,
        "water_level": 4.0,
        "pois": pois_json,
    }
    json_path = f"{args.output}.terrain.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[dem_to_terrain] meta OK    : {json_path} ({len(pois_json)} POI)")

    print("\n[dem_to_terrain] terrain prêt — lance `\\map br_reunion` côté engine.")


if __name__ == "__main__":
    main()
