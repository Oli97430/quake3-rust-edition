# Branding — Quake 3 RUST EDITION

Assets vectoriels du projet. Tout est en SVG pour rester scalable infini
(GitHub README, splash, icone OS, page web) sans dépendre d'une chaîne
de génération d'image.

## Fichiers

| Fichier | Format | Usage |
|---|---|---|
| `banner.svg` | 1280×320 | Bandeau du README, header GitHub release notes |
| `logo.svg` | 256×256 | Icone carré (avatar org, splash, exe icon source) |

## Génération de l'icone Windows (`.ico`)

L'exécutable Windows embarque l'icone via `winres` (cf. `crates/q3-engine/build.rs`).
Le `.ico` doit contenir plusieurs résolutions empilées (16, 32, 48, 256).

Méthode recommandée — Inkscape + ImageMagick :

```bash
# 1. Exporte le SVG en plusieurs PNG
inkscape logo.svg --export-filename=icon-16.png  -w 16  -h 16
inkscape logo.svg --export-filename=icon-32.png  -w 32  -h 32
inkscape logo.svg --export-filename=icon-48.png  -w 48  -h 48
inkscape logo.svg --export-filename=icon-256.png -w 256 -h 256

# 2. Empile dans un .ico
magick icon-16.png icon-32.png icon-48.png icon-256.png ../icon.ico

# 3. Build : `winres` détecte automatiquement assets/icon.ico
cargo build --release -p q3-engine
```

## Palette officielle

| Couleur | Hex | Usage |
|---|---|---|
| Anthracite fond | `#04050b` → `#1a1216` | Backdrop panneaux HUD |
| Halo orange chaud | `#3a1908` | Glow vignette |
| Cyan accent | `#26a8f0` | Liserés, séparateurs |
| Crème titre | `#ffe28a` → `#d96a1c` | Texte "Q3" |
| Orange rouille | `#f29c4d` → `#7a2f0b` | Texte "RUST" |
| Cyan clair | `#cfe6ff` | Sous-titres, EDITION |

Source : palette extraite du HUD in-game (`hud_helpers.rs`) — cohérence
visuelle entre l'app et son branding externe.
