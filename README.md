<div align="center">

<img src="assets/branding/banner.svg" alt="Quake 3 RUST EDITION" width="100%"/>

#### *id Tech 3 reimagined in Rust — modern engine, classic gameplay*

[![License](https://img.shields.io/badge/License-GPL--2.0-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.78+-orange)](https://rustup.rs/)
[![Status](https://img.shields.io/badge/Status-v0.9.5-green)]()

</div>

---

Réécriture complète de **Quake III Arena** en Rust moderne — pipeline `wgpu` (Vulkan/DX12/Metal), assets glTF, anti-cheat serveur, netcode lag-compensation, audio spatial, post-FX HDR.

> **État** : v0.9.5 — moteur complet, jouable solo + bots IA, mode BR exploration, **9/9 armes en GLB**, anti-cheat actif, map downloader intégré.

## ✨ Highlights

- **Renderer wgpu HDR** : pipeline scene buffer `Rgba16Float` + ACES Narkowicz + multi-mip bloom + SSAO depth-based + CSM (PCF 3×3) + TAA Halton(2,3) + SSR water raymarch + god rays + volumetric fog (HG phase)
- **9/9 armes en GLB** moderne (pickup au sol + viewmodel main joueur) avec orientation tunée par arme
- **Animations bots Q3 fidèles** : parsing `animation.cfg` runtime, 25+ ranges (LEGS_*, TORSO_*, BOTH_*), TORSO_PAIN1, phase rebase per anim change
- **Netcode lag-compensation** : ring buffer `LagSample { origin, velocity, view_angles, crouching }` 30 samples × 50 ms, hit-center ajusté crouch (24u stand / 14u crouch)
- **Anti-cheat serveur** : cap angulaire 720°/s post-budget, teleport detection 2400 u/s, dt budget cumulatif, comparaisons signées (anti i32→u32 wrap exploit)
- **Map downloader intégré** : catalogue + DL HTTP background (ureq + rustls) + SHA256 + magic ZIP/IBSP check + cap 100 MB anti-DoS
- **Battle Royale Réunion** mode exploration : terrain procédural, POI tier 2-4, ring shrink optionnel
- **Lecteur audio universel** : mp3/wav/ogg/flac, scan récursif `~/Music`, `~/Downloads`, OneDrive, dossiers custom via `s_musicpath`
- **Punch angles recoil** par arme (BFG 6°, RL 5°, SG 4.5°, MG 1.4° + jitter ±0.5°)
- **Railgun sniper scope** : RMB tenu = zoom FOV÷3 + sens÷3 + crosshair de précision + mil-dot

## 🔫 Armes (9/9 GLB)

| Arme | GLB | Tir alt |
|------|-----|---------|
| Gauntlet | ✅ | Lunge dash + range 96u |
| Machine Gun | ✅ | Burst précision 18 dmg, no spread |
| Shotgun | ✅ | Slug AP 80 dmg, cooldown 1.5s |
| Grenade Launcher | ✅ | Airburst flat shot 110 splash |
| Rocket Launcher | ✅ | Lock-on cône 30°/1500u |
| Lightning Gun | ✅ | Shock blast 55 dmg |
| Railgun | ✅ | **Sniper zoom 3× + ricochet** |
| Plasma Gun | ✅ | Plasma orb gros splash 96u |
| BFG10K | ✅ | Death zone 250u splash, 160 dmg |

## 🛡️ Items

| Catégorie | Status |
|-----------|--------|
| Munitions (9 types) | 9/9 ✅ GLB |
| Armures (3 tiers) | 3/3 ✅ (shard 5 / combat 50 / body 100) |
| Health (4 tiers) | ✅ partagé via `health_pack.glb` |
| Powerups (Quad, Regen) | 2/6 ✅ GLB |
| Holdables (Medkit) | 1/2 ✅ |

## 🏗️ Architecture

```
crates/
├── q3-engine/      # binaire q3.exe + main loop + app.rs (~22k lines)
│   ├── src/app.rs       # state machine + render loop + input
│   ├── src/menu.rs      # menu UI (Root/Play/Options/Audio/MapDownloader)
│   ├── src/net/         # client + server + snapshots delta
│   ├── src/map_dl.rs    # HTTP map downloader (ureq + sha256 + zip)
│   └── src/vr.rs        # VR scaffolding (OpenXR partial)
├── q3-renderer/    # wgpu pipelines (BSP, MD3, GLB, terrain, sky, post)
├── q3-bsp/         # parseur IBSP v46 (zero-copy bytemuck)
├── q3-model/       # MD3 + glTF/GLB loader
├── q3-bot/         # IA bots (FSM + animation ranges)
├── q3-game/        # physique mouvement (strafe-jump, wall-jump, mantling)
├── q3-collision/   # trace BSP + bbox vs world
├── q3-terrain/     # heightmap BR + ring shrink + POI
├── q3-net/         # protocole snapshots + UserCmd quantification
├── q3-sound/       # rodio wrapper + spatial 3D
├── q3-image/       # decoder TGA/JPG/PNG + ImageCache
├── q3-shader/      # parseur Q3 shader scripts
├── q3-filesystem/  # VFS pak0+mods+assets/ avec cycle protection symlinks
├── q3-math/        # glam wrappers + Q3 Z-up conventions
└── q3-common/      # cvar registry + log + errors
```

## 🚀 Build

Pré-requis :
- Rust 1.78+ (workspace edition 2021)
- Drivers GPU compatibles wgpu (Vulkan / DX12 / Metal)
- Quake 3 Arena installé (Steam ou autre — pour les `pak0.pk3` originaux)

```bash
cargo build --release
./target/release/q3
```

Le moteur auto-détecte l'install Steam Q3. Override : `--base "C:\path\to\Quake 3 Arena"`.

## 🎵 Lecteur audio

```bash
# Console in-game (touche `~`)
seta s_musicpath "D:\Musique;E:\Spotify\Export"
music list
music play "C:\Users\You\Music\track.mp3"
```

Formats : WAV, OGG, MP3, FLAC. Scan récursif jusqu'à 4 niveaux.

## 🗺️ Map Downloader

```
Menu → OPTIONS → MAP DOWNLOADER
```

Catalogue inclus : Aerowalk, Cure, ZTN3DM2, Pukka3Tourney2, Lost World. DL HTTP background avec progression live, SHA256, magic ZIP+IBSP check, cap 100 MB. PK3 placés dans `baseq3/`.

Console alternative : `mapdl list` / `mapdl get <id>` / `mapdl status`.

## 🛠️ Cvars notables

| Cvar | Default | Description |
|------|---------|-------------|
| `cg_fov` | 90 | FOV horizontal à 4:3 (Q3 standard) |
| `cg_fovaspect` | 0 | 0 = Hor+ (Quake/arena), 1 = Vert- (CS/Apex) |
| `r_skybox` | "env/skybox_clouds" | Override skybox custom global |
| `r_hdr` | 0 | HDR10 surface (en cours) |
| `s_musicpath` | "" | Dossiers audio supplémentaires (`;` séparé Win, `:` Unix) |
| `g_godmode` | 0 | Joueur invincible vs bots (test) |
| `br_bots` | 0 | BR : 0 = exploration vide, 1 = match avec bots |

## 🎯 Anti-cheat (server-side)

- **Angular rate cap** : 720°/s yaw+pitch post-budget — anti-aimbot soft snap
- **Teleport detection** : 2400 u/s max — revert origin + freeze velocity, log warn
- **dt budget cumulatif** : ≤ 1 s/s wall-clock (anti-speedhack)
- **Lag-comp window** : 250 ms max rewind, refuse target dans le futur (anti clock-skew forgery)
- **Saturating arithmetic** : `saturating_sub` ammo, comparaisons signées (pas de wrap)
- **Magic check downloads** : SHA256 + ZIP magic + IBSP magic

## 📊 Performance

| Map | FPS avg | 1% low | 0.1% low |
|-----|---------|--------|----------|
| q3dm6 | 280 | 240 | 195 |
| q3dm17 | 320 | 290 | 260 |
| q3tourney2 | 350 | 310 | 280 |
| br_reunion | 180 | 145 | 120 |

*RTX 3090 @ 1920×1080, ULTRA*

Optims clés :
- God rays / volumetric fog early-out
- SSAO kernel précomputé `var<array>`
- Drone scratch buffer (0 alloc heap par frame)
- TAA Halton jitter (supersampling temporel)

## 🐛 Polish v0.9.5

~14 bugs critiques fixés sur 2 passes audit :
- Lag-comp u32 underflow guard + crouching hit-center
- MapDownloader race + DoS cap (100 MB) + BSP magic check
- VFS symlink cycle protection (Windows junctions)
- MD3 normal decoding (lat = [0, π], pas [0, 2π])
- Sound cache leak (`unload()` + `clear_cache()`)
- Ammo i32→u32 cast exploit (tir infini)
- Cap angulaire post-budget (anti lag-spike flick)
- Fire flags reset au switch d'arme
- NaN guard homing rocket
- Endianness assert q3-bsp + q3-model

## 🤝 Contribuer

Code commenté en français — chaque section explique le **pourquoi**, pas juste le **quoi**.

```bash
cargo fmt
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

## 📜 Licence

GPL-2.0-or-later (héritage Q3 id Software).

⚠️ Les `pak0.pk3` originaux Q3 NE SONT PAS redistribués — il faut posséder une copie légale du jeu (Steam, GOG, CD).

## 🙏 Crédits

- **id Software** — Quake III Arena (1999)
- **Kekoa Proudfoot** — doc BSP IBSP v46
- **wgpu / naga** — pipeline graphique cross-platform
- **rodio** — audio
- **glam** — math SIMD
- **gltf** — parser glTF 2.0
- **ureq + rustls** — HTTP pure Rust pour map downloader
