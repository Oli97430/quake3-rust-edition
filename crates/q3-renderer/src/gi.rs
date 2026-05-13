//! **Global Illumination** — système de light probes par Spherical Harmonics.
//!
//! Remplace l'ambient hémisphérique hardcodé (`hemi * 0.25` dans les shaders)
//! par un éclairage indirect échantillonné depuis une grille 3D de sondes
//! (probes) réparties dans le volume de la map.  Chaque sonde stocke 9
//! coefficients SH (bandes L0, L1, L2) par canal RGB, soit 27 floats.
//!
//! # Pourquoi les Spherical Harmonics ?
//!
//! L'éclairage indirect varie lentement dans l'espace — c'est par définition
//! de la lumière qui a rebondi au moins une fois et qui est donc fortement
//! diffusée.  Les SH L2 capturent cette variation directionnelle douce (5
//! lobes) tout en restant ultra-compactes (9 coefficients au lieu de centaines
//! de texels d'une lightprobe cubemap).  Évaluer l'irradiance pour une
//! direction `N` coûte 9 multiplications + 9 additions → quasi gratuit en
//! fragment shader.
//!
//! # Pipeline
//!
//! 1. **Placement** : grille régulière avec spacing configurable (défaut 256
//!    unités Q3 ≈ 6.5 m).  Les probes sont placées entre `bounds_min` et
//!    `bounds_max` (AABB de la map).
//! 2. **Baking** : pour chaque probe, on projette analytiquement :
//!    - la contribution du soleil (directional light)
//!    - un sky hémisphérique (bleu en haut, brun en bas)
//!    - les lightmaps BSP (couleur moyenne par couche → contribution omnidirectionnelle)
//! 3. **Upload** : les données SH sont packées dans un `wgpu::Buffer`
//!    (storage) et exposées via un bind group.
//! 4. **Sampling shader** : `gi.wgsl` fournit `sample_gi(world_pos, normal)`
//!    qui interpole trilinéairement entre les 8 probes les plus proches et
//!    évalue les SH dans la direction de la normale.

use bytemuck::{Pod, Zeroable};
use q3_math::Vec3;
use std::sync::Arc;
use tracing::debug;

// ─── Constantes SH ─────────────────────────────────────────────────────
//
// Fonctions de base des harmoniques sphériques réelles pour les bandes
// L=0, L=1, L=2. On utilise la normalisation "SH standard" (cf. Ramamoorthi
// & Hanrahan 2001, Sloan 2008).
//
// Convention : `dir` = direction unitaire (nx, ny, nz) dans le repère monde
// Q3 (X-forward, Y-left, Z-up).

const SH_C0: f32 = 0.282_094_8; // 1 / (2√π)
const SH_C1: f32 = 0.488_602_5; // √(3) / (2√π)
const SH_C2_0: f32 = 1.092_548_4; // √(15) / (2√π)
const SH_C2_1: f32 = 0.315_391_6; // √(5) / (4√π)
const SH_C2_2: f32 = 0.546_274_2; // √(15) / (4√π)

/// Évalue les 9 fonctions de base SH L0..L2 pour une direction unitaire.
///
/// Retourne `[Y_00, Y_1-1, Y_10, Y_11, Y_2-2, Y_2-1, Y_20, Y_21, Y_22]`.
///
/// On garde l'ordre "WGSL-friendly" (l croissant, m croissant) pour que
/// l'index dans le tableau corresponde directement à l'index dans le
/// storage buffer GPU.
pub fn sh_project_direction(dir: Vec3) -> [f32; 9] {
    let (x, y, z) = (dir.x, dir.y, dir.z);
    [
        // L=0, m=0
        SH_C0,
        // L=1
        SH_C1 * y,   // m=-1
        SH_C1 * z,   // m= 0
        SH_C1 * x,   // m=+1
        // L=2
        SH_C2_0 * x * y,                         // m=-2
        SH_C2_0 * y * z,                         // m=-1
        SH_C2_1 * (3.0 * z * z - 1.0),           // m= 0
        SH_C2_0 * x * z,                         // m=+1
        SH_C2_2 * (x * x - y * y),               // m=+2
    ]
}

/// Reconstruit la couleur d'irradiance à partir de coefficients SH L2 et
/// d'une direction.  `coeffs[i]` = `[r, g, b]` du i-ème coefficient.
pub fn sh_evaluate(coeffs: &[[f32; 3]; 9], dir: Vec3) -> Vec3 {
    let basis = sh_project_direction(dir);
    let mut r = 0.0f32;
    let mut g = 0.0f32;
    let mut b = 0.0f32;
    for i in 0..9 {
        r += basis[i] * coeffs[i][0];
        g += basis[i] * coeffs[i][1];
        b += basis[i] * coeffs[i][2];
    }
    // Clamp — l'irradiance reconstruite peut être légèrement négative
    // à cause des lobes SH ; on coupe pour éviter des artefacts visuels.
    Vec3::new(r.max(0.0), g.max(0.0), b.max(0.0))
}

// ─── Light Probe ───────────────────────────────────────────────────────

/// Sonde d'éclairage indirect : position monde + 9 coefficients SH RGB.
///
/// Le layout mémoire est pensé pour le GPU : 28 floats consécutifs (3 pos +
/// 1 pad + 9×3 SH = 27 + padding), packés dans 7 `vec4<f32>` côté WGSL.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LightProbe {
    pub position: Vec3,
    /// 9 coefficients SH L2, chacun étant un triplet (R, G, B).
    /// Ordre : L0m0, L1m-1, L1m0, L1m+1, L2m-2, L2m-1, L2m0, L2m+1, L2m+2.
    pub sh: [[f32; 3]; 9],
}

impl Default for LightProbe {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            sh: [[0.0; 3]; 9],
        }
    }
}

impl LightProbe {
    /// Ajoute la contribution d'une lumière directionnelle (soleil, etc.)
    /// projetée sur les SH.
    ///
    /// `dir` = direction VERS la lumière (normalisée), `color` = radiance RGB.
    /// On projette la radiance incidente dans chaque bande SH via le produit
    /// scalaire `color * basis_i(dir)`.
    pub fn add_directional(&mut self, dir: Vec3, color: Vec3) {
        let basis = sh_project_direction(dir);
        for (i, &b) in basis.iter().enumerate() {
            self.sh[i][0] += b * color.x;
            self.sh[i][1] += b * color.y;
            self.sh[i][2] += b * color.z;
        }
    }

    /// Ajoute un hémisphère analytique : `top_color` au-dessus (Z+),
    /// `bottom_color` en dessous (Z-).  On intègre sur la sphère unité
    /// avec un schéma simple : 6 directions cardinales + 8 diagonales.
    ///
    /// Pourquoi pas une intégrale fermée ?  Pour L2, le blend linéaire
    /// en Z des deux couleurs ne donne pas de forme analytique simple
    /// pour les bandes m != 0.  Le schéma discret à 14 directions est
    /// suffisamment précis pour un gradient doux haut/bas.
    pub fn add_hemisphere(&mut self, top_color: Vec3, bottom_color: Vec3) {
        // Échantillons uniformément répartis sur la sphère (14 dirs).
        let dirs: [(Vec3, f32); 14] = [
            // 6 cardinaux
            (Vec3::new(1.0, 0.0, 0.0), 0.0),
            (Vec3::new(-1.0, 0.0, 0.0), 0.0),
            (Vec3::new(0.0, 1.0, 0.0), 0.0),
            (Vec3::new(0.0, -1.0, 0.0), 0.0),
            (Vec3::new(0.0, 0.0, 1.0), 1.0),
            (Vec3::new(0.0, 0.0, -1.0), -1.0),
            // 8 diagonaux
            (Vec3::new(1.0, 1.0, 1.0).normalize(), 1.0),
            (Vec3::new(1.0, -1.0, 1.0).normalize(), 1.0),
            (Vec3::new(-1.0, 1.0, 1.0).normalize(), 1.0),
            (Vec3::new(-1.0, -1.0, 1.0).normalize(), 1.0),
            (Vec3::new(1.0, 1.0, -1.0).normalize(), -1.0),
            (Vec3::new(1.0, -1.0, -1.0).normalize(), -1.0),
            (Vec3::new(-1.0, 1.0, -1.0).normalize(), -1.0),
            (Vec3::new(-1.0, -1.0, -1.0).normalize(), -1.0),
        ];
        // Poids uniforme : la sphère a une surface de 4π, divisée par 14
        // échantillons.  Le facteur π supplémentaire vient de la convolution
        // SH de l'irradiance (cosine-weighted hemispherical integral).
        let weight = std::f32::consts::PI * 4.0 / 14.0;
        for &(dir, z_sign) in &dirs {
            // Interpolation linéaire selon Z : 0 = milieu, +1 = top, -1 = bottom.
            let blend = z_sign * 0.5 + 0.5; // remappage [-1,1] → [0,1]
            let color = bottom_color.lerp(top_color, blend);
            let basis = sh_project_direction(dir);
            for (i, &b) in basis.iter().enumerate() {
                self.sh[i][0] += weight * b * color.x;
                self.sh[i][1] += weight * b * color.y;
                self.sh[i][2] += weight * b * color.z;
            }
        }
    }
}

// ─── GPU-side probe data ───────────────────────────────────────────────
//
// 9 coefficients × 3 canaux = 27 floats.  On les packe dans 7 vec4 (28
// floats) avec 1 float de padding terminal.  Le shader lit le buffer
// comme un `array<vec4<f32>, 7>` par probe.

/// Données SH d'une probe telles qu'envoyées au GPU.  28 floats = 7 vec4.
///
/// Packing :
///   vec4[0] = sh[0].rgb, sh[1].r
///   vec4[1] = sh[1].gb,  sh[2].rg
///   vec4[2] = sh[2].b,   sh[3].rgb
///   vec4[3] = sh[4].rgb, sh[5].r
///   vec4[4] = sh[5].gb,  sh[6].rg
///   vec4[5] = sh[6].b,   sh[7].rgb
///   vec4[6] = sh[8].rgb, 0.0 (pad)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuProbeData {
    pub data: [[f32; 4]; 7],
}

impl GpuProbeData {
    /// Packe les 9 coefficients SH RGB d'une probe dans 7 vec4.
    pub fn from_sh(sh: &[[f32; 3]; 9]) -> Self {
        // Aplatir les 27 floats dans un vecteur de travail.
        let mut flat = [0.0f32; 28];
        for (i, coeff) in sh.iter().enumerate() {
            flat[i * 3] = coeff[0];
            flat[i * 3 + 1] = coeff[1];
            flat[i * 3 + 2] = coeff[2];
        }
        // flat[27] = 0.0 — padding déjà à zéro.
        let mut data = [[0.0f32; 4]; 7];
        for (v, chunk) in data.iter_mut().zip(flat.chunks_exact(4)) {
            v[0] = chunk[0];
            v[1] = chunk[1];
            v[2] = chunk[2];
            v[3] = chunk[3];
        }
        Self { data }
    }
}

/// En-tête de la grille envoyée au GPU — précède le tableau de probes.
///
/// Layout std430 (storage buffer) :
///   vec4<f32> bounds_min   (xyz + pad)
///   vec4<f32> bounds_max   (xyz + pad)
///   vec4<u32> grid_size    (nx, ny, nz + pad)
///   vec4<f32> spacing_inv  (1/sx, 1/sy, 1/sz + pad)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuGridHeader {
    pub bounds_min: [f32; 4],
    pub bounds_max: [f32; 4],
    pub grid_size: [u32; 4],
    pub spacing_inv: [f32; 4],
}

// ─── Probe Grid ────────────────────────────────────────────────────────

/// Grille 3D de light probes couvrant le volume de la map.
///
/// Le spacing par défaut (256 unités ≈ 6.5 m) est un compromis entre
/// précision de l'éclairage indirect et budget mémoire.  Une map Q3
/// typique (4096×4096×2048 unités) produit ≈ 16×16×8 = 2048 probes,
/// soit ~56 Ko de données SH — négligeable.
pub struct ProbeGrid {
    _device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    probes: Vec<LightProbe>,
    /// Dimensions de la grille (nombre de probes par axe).
    nx: u32,
    ny: u32,
    nz: u32,
    /// Bornes monde de la grille.
    bounds_min: Vec3,
    bounds_max: Vec3,
    /// Espacement entre probes sur chaque axe.
    spacing: Vec3,
    /// Buffer GPU contenant l'en-tête + les probes.
    buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl ProbeGrid {
    /// Crée une grille de probes couvrant le volume `[bounds_min, bounds_max]`
    /// avec un espacement `spacing` (en unités Q3 par axe).
    ///
    /// Les probes sont initialisées à zéro — il faut appeler `bake()` puis
    /// `upload()` pour les remplir et les envoyer au GPU.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        bounds_min: Vec3,
        bounds_max: Vec3,
        spacing: f32,
    ) -> Self {
        // Nombre de probes par axe : on veut au moins 2 pour pouvoir
        // interpoler (une probe à chaque extrémité).
        let extent = bounds_max - bounds_min;
        let nx = ((extent.x / spacing).ceil() as u32 + 1).max(2);
        let ny = ((extent.y / spacing).ceil() as u32 + 1).max(2);
        let nz = ((extent.z / spacing).ceil() as u32 + 1).max(2);

        // Espacement réel (peut différer légèrement du spacing demandé
        // pour couvrir exactement l'AABB avec un nombre entier de cellules).
        let sx = if nx > 1 { extent.x / (nx - 1) as f32 } else { extent.x };
        let sy = if ny > 1 { extent.y / (ny - 1) as f32 } else { extent.y };
        let sz = if nz > 1 { extent.z / (nz - 1) as f32 } else { extent.z };
        let spacing_vec = Vec3::new(sx.max(1.0), sy.max(1.0), sz.max(1.0));

        let total = (nx * ny * nz) as usize;
        let mut probes = Vec::with_capacity(total);

        // Placement : les probes sont indexées en ordre row-major (X varie
        // le plus vite, puis Y, puis Z) — cohérent avec le shader qui
        // calcule `idx = iz * ny * nx + iy * nx + ix`.
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let pos = bounds_min
                        + Vec3::new(
                            ix as f32 * spacing_vec.x,
                            iy as f32 * spacing_vec.y,
                            iz as f32 * spacing_vec.z,
                        );
                    probes.push(LightProbe { position: pos, ..LightProbe::default() });
                }
            }
        }

        debug!(
            "GI probe grid : {}×{}×{} = {} probes, spacing ~{:.0}u, bounds [{:.0}..{:.0}]",
            nx,
            ny,
            nz,
            total,
            spacing,
            bounds_min,
            bounds_max,
        );

        // Crée un buffer GPU initial (on le remplira lors de `upload`).
        let header_size = std::mem::size_of::<GpuGridHeader>() as u64;
        let probe_size = std::mem::size_of::<GpuProbeData>() as u64;
        let buf_size = header_size + probe_size * total as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi-probe-storage"),
            size: buf_size.max(64), // wgpu demande >= 1 byte
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gi-probe-bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // Lu par le fragment shader pour évaluer l'irradiance.
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi-probe-bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Self {
            _device: device,
            queue,
            probes,
            nx,
            ny,
            nz,
            bounds_min,
            bounds_max,
            spacing: spacing_vec,
            buffer,
            bind_group_layout,
            bind_group,
        }
    }

    // ── Accessors ──────────────────────────────────────────────────────

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    pub fn grid_dims(&self) -> (u32, u32, u32) {
        (self.nx, self.ny, self.nz)
    }

    // ── Baking ─────────────────────────────────────────────────────────

    /// Calcule les coefficients SH de chaque probe à partir :
    ///
    /// 1. Des lightmaps BSP — on extrait la couleur moyenne de chaque
    ///    lightmap et on la projette comme contribution omnidirectionnelle
    ///    (bande L0 uniquement, car les lightmaps ne portent pas d'info
    ///    directionnelle).
    /// 2. Du soleil — lumière directionnelle projetée sur toutes les bandes.
    /// 3. Du ciel — hémisphère analytique (sky bleu / ground brun).
    ///
    /// # Pourquoi pas du ray tracing ?
    ///
    /// Le ray tracing sur la BSP donnerait des résultats plus fidèles (GI
    /// exacte), mais nécessite un accélérateur de rayons (BVH ou k-d tree)
    /// et un gros budget de samples par probe.  L'approche analytique + LM
    /// moyenne est 3 ordres de grandeur plus rapide et donne un résultat
    /// visuellement convaincant pour le style Q3 (maps relativement
    /// ouvertes, éclairage baked dominé par le soleil + sky).
    pub fn bake(
        &mut self,
        bsp_lightmaps: &[&[u8]], // Tableau des lightmaps RGB brutes (128×128×3 chacune).
        sun_dir: Vec3,
        sun_color: Vec3,
        sky_color: Vec3,
    ) {
        // Couleur moyenne de toutes les lightmaps combinées (une approximation
        // grossière de l'irradiance ambiante baked dans la map).
        let lm_avg = lightmap_average(bsp_lightmaps);

        // Couleur du "sol" pour l'hémisphère : un brun-terre sombre,
        // typique du bounce de lumière renvoyé par le terrain Q3.
        let ground_color = Vec3::new(0.18, 0.14, 0.10);

        for probe in &mut self.probes {
            // Remet à zéro avant d'accumuler (au cas où on rebake).
            probe.sh = [[0.0; 3]; 9];

            // 1. Contribution du ciel hémisphérique.
            probe.add_hemisphere(sky_color, ground_color);

            // 2. Contribution du soleil.
            // On atténue légèrement (×0.4) car le soleil est déjà
            // partiellement comptabilisé dans les lightmaps baked.
            // Sans cette atténuation, les surfaces soleil-face seraient
            // sur-éclairées (double comptage).
            probe.add_directional(sun_dir.normalize(), sun_color * 0.4);

            // 3. Contribution des lightmaps (irradiance ambiante baked).
            // Projetée en L0 uniquement (omnidirectionnel), pondérée par
            // la constante SH L0 pour que l'énergie intégrée soit correcte.
            let basis_l0 = SH_C0;
            // Facteur de normalisation : pour que l'intégrale sur la sphère
            // de L0² × irradiance donne `lm_avg`, on multiplie par
            // 4π × Y00 = 4π × SH_C0 ≈ 3.5449.
            let l0_weight = 4.0 * std::f32::consts::PI * basis_l0;
            probe.sh[0][0] += lm_avg.x * l0_weight;
            probe.sh[0][1] += lm_avg.y * l0_weight;
            probe.sh[0][2] += lm_avg.z * l0_weight;
        }

        debug!(
            "GI bake terminé : {} probes, lm_avg=({:.3}, {:.3}, {:.3}), sun={:?}",
            self.probes.len(),
            lm_avg.x,
            lm_avg.y,
            lm_avg.z,
            sun_dir,
        );
    }

    /// Envoie les données de probes vers le buffer GPU.
    ///
    /// Doit être appelé après `bake()` et chaque fois que les probes sont
    /// modifiées (ex : changement de niveau, re-bake dynamique).
    pub fn upload(&self) {
        let header = GpuGridHeader {
            bounds_min: [self.bounds_min.x, self.bounds_min.y, self.bounds_min.z, 0.0],
            bounds_max: [self.bounds_max.x, self.bounds_max.y, self.bounds_max.z, 0.0],
            grid_size: [self.nx, self.ny, self.nz, 0],
            spacing_inv: [
                1.0 / self.spacing.x,
                1.0 / self.spacing.y,
                1.0 / self.spacing.z,
                0.0,
            ],
        };

        // Sérialise l'en-tête.
        let header_bytes = bytemuck::bytes_of(&header);
        self.queue.write_buffer(&self.buffer, 0, header_bytes);

        // Sérialise les probes.
        let header_offset = std::mem::size_of::<GpuGridHeader>() as u64;
        let gpu_probes: Vec<GpuProbeData> =
            self.probes.iter().map(|p| GpuProbeData::from_sh(&p.sh)).collect();
        let probe_bytes = bytemuck::cast_slice::<GpuProbeData, u8>(&gpu_probes);
        self.queue.write_buffer(&self.buffer, header_offset, probe_bytes);
    }

    /// Construit une grille "fallback" minimale (1 probe) qui encode un
    /// simple hémisphère sky/ground.  Utilisée quand le BSP n'a pas de
    /// lightmaps ou quand on veut un éclairage indirect par défaut sans
    /// passer par le baking complet.
    pub fn new_fallback(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let bounds_min = Vec3::new(-4096.0, -4096.0, -1024.0);
        let bounds_max = Vec3::new(4096.0, 4096.0, 2048.0);
        let mut grid = Self::new(
            device,
            queue,
            bounds_min,
            bounds_max,
            // Spacing énorme → une seule probe au centre, l'interpolation
            // retourne toujours cette probe unique.
            16384.0,
        );
        let sky = Vec3::new(0.40, 0.55, 0.70);
        let ground = Vec3::new(0.18, 0.14, 0.10);
        for probe in &mut grid.probes {
            probe.add_hemisphere(sky, ground);
        }
        grid.upload();
        debug!("GI fallback : {} probe(s) hemisphere-only", grid.probes.len());
        grid
    }
}

// ─── Utilitaires ───────────────────────────────────────────────────────

/// Calcule la couleur moyenne de toutes les lightmaps fournies.
///
/// Les lightmaps Q3 sont en RGB 128×128. On prend la moyenne arithmétique
/// de tous les texels de toutes les couches.  L'overbright Q3 (×2) est
/// appliqué ici pour être cohérent avec le rendu.
fn lightmap_average(lightmaps: &[&[u8]]) -> Vec3 {
    if lightmaps.is_empty() {
        // Pas de lightmaps → on retourne un gris moyen comme approximation
        // d'un éclairage ambient décent.
        return Vec3::new(0.25, 0.25, 0.25);
    }
    let mut sum_r = 0.0f64;
    let mut sum_g = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut count = 0u64;
    for lm in lightmaps {
        // Chaque lightmap = 128×128 texels × 3 bytes (RGB).
        for chunk in lm.chunks_exact(3) {
            // Overbright ×2 (même convention que lightmap.rs::overbright).
            let r = (chunk[0] as u16).saturating_mul(2).min(255);
            let g = (chunk[1] as u16).saturating_mul(2).min(255);
            let b = (chunk[2] as u16).saturating_mul(2).min(255);
            sum_r += r as f64 / 255.0;
            sum_g += g as f64 / 255.0;
            sum_b += b as f64 / 255.0;
            count += 1;
        }
    }
    let inv = 1.0 / count as f64;
    Vec3::new(
        (sum_r * inv) as f32,
        (sum_g * inv) as f32,
        (sum_b * inv) as f32,
    )
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Les fonctions de base SH doivent être orthonormales.  Vérification
    /// rapide : la direction (0,0,1) (Z-up) doit activer L=1 m=0 (la bande
    /// "Z") et laisser les bandes X/Y à zéro.
    #[test]
    fn sh_basis_z_up() {
        let b = sh_project_direction(Vec3::new(0.0, 0.0, 1.0));
        // L0 m0 = constante
        assert!((b[0] - SH_C0).abs() < 1e-5);
        // L1 m-1 (Y) = 0
        assert!(b[1].abs() < 1e-5);
        // L1 m0  (Z) = SH_C1
        assert!((b[2] - SH_C1).abs() < 1e-5);
        // L1 m+1 (X) = 0
        assert!(b[3].abs() < 1e-5);
    }

    /// L'évaluation SH d'un probe avec uniquement L0 doit retourner la
    /// même couleur dans toutes les directions (omnidirectionnel).
    #[test]
    fn sh_evaluate_l0_is_omnidirectional() {
        let mut coeffs = [[0.0f32; 3]; 9];
        // Encode une couleur (1, 0.5, 0) en L0.
        coeffs[0] = [1.0 / SH_C0, 0.5 / SH_C0, 0.0];
        let dirs = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 0.0),
        ];
        for d in dirs {
            let c = sh_evaluate(&coeffs, d);
            assert!((c.x - 1.0).abs() < 1e-4, "R devrait être ~1.0, obtenu {}", c.x);
            assert!((c.y - 0.5).abs() < 1e-4, "G devrait être ~0.5, obtenu {}", c.y);
        }
    }

    /// `GpuProbeData::from_sh` doit packe 27 floats + 1 pad dans 7 vec4.
    #[test]
    fn gpu_probe_data_packing() {
        let mut sh = [[0.0f32; 3]; 9];
        for (i, c) in sh.iter_mut().enumerate() {
            c[0] = (i * 3) as f32;
            c[1] = (i * 3 + 1) as f32;
            c[2] = (i * 3 + 2) as f32;
        }
        let gpu = GpuProbeData::from_sh(&sh);
        // Flatten et vérifier les 27 premières valeurs.
        let flat: Vec<f32> = gpu.data.iter().flat_map(|v| v.iter().copied()).collect();
        for i in 0..27 {
            assert_eq!(flat[i], i as f32, "mismatch à flat[{}]", i);
        }
        // Le 28e float doit être le pad (0.0).
        assert_eq!(flat[27], 0.0, "padding devrait être 0.0");
    }

    /// `GpuGridHeader` doit faire 64 bytes (4 × vec4<f32> = 4 × 16).
    #[test]
    fn gpu_grid_header_size() {
        assert_eq!(std::mem::size_of::<GpuGridHeader>(), 64);
    }

    /// `GpuProbeData` doit faire 112 bytes (7 × vec4 = 7 × 16).
    #[test]
    fn gpu_probe_data_size() {
        assert_eq!(std::mem::size_of::<GpuProbeData>(), 112);
    }

    /// La moyenne de lightmaps vides doit retourner un gris par défaut.
    #[test]
    fn lightmap_average_empty() {
        let avg = lightmap_average(&[]);
        assert!((avg.x - 0.25).abs() < 1e-5);
    }

    /// La moyenne d'une lightmap 1×1 (3 bytes) doit retourner la valeur
    /// overbright correspondante.
    #[test]
    fn lightmap_average_single_texel() {
        // 1 texel RGB = [100, 200, 50] → overbright ×2 → [200, 255, 100]
        // (200 clamp ok, 400 clamp 255, 100 clamp ok)
        let lm: &[u8] = &[100, 200, 50];
        let avg = lightmap_average(&[lm]);
        let expected_r = 200.0 / 255.0;
        let expected_g = 255.0 / 255.0; // 400 clampé à 255
        let expected_b = 100.0 / 255.0;
        assert!((avg.x - expected_r).abs() < 1e-3);
        assert!((avg.y - expected_g).abs() < 1e-3);
        assert!((avg.z - expected_b).abs() < 1e-3);
    }
}
