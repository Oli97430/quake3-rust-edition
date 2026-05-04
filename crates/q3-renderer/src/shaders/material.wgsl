// Shader multi-matériau : diffuse × lightmap × color, + dynamic lights,
// + fog volumétrique quand l'œil est à l'intérieur d'un brush fog.
//
// Bind groups :
//   0 : camera uniform
//   1 : lightmap texture_2d_array + sampler (partagé par tout le monde)
//   2 : diffuse texture_2d + sampler (par matériau)
//   3 : dynamic lights uniform buffer (partagé, rempli par DlightSet)
//   4 : fog uniform (partagé, actif seulement si caméra dans un volume)

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    inv_view_proj_rot: mat4x4<f32>,
    // time_info.x = horloge applicative (secondes). Utilisée par les
    // tcmod animés (scroll / rotate) du VS.
    time_info: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var lightmap_tex: texture_2d_array<f32>;
@group(1) @binding(1) var lightmap_samp: sampler;

@group(2) @binding(0) var diffuse_tex: texture_2d<f32>;
@group(2) @binding(1) var diffuse_samp: sampler;

// Paramètres d'animation par-matériau.
//   anim         : tcmod (xy=scroll UV/s, z=rotate rad/s, w=pad)
//   rgb_info.x   : mode rgbgen (0=default, 1=wave, 2=const)
//   rgb_wave     : waveform pour `rgbgen wave` (base, amp, phase, freq)
//   rgb_const    : couleur pour `rgbgen const` (rgb, pad)
//   wave_kind.x  : discriminant WaveFunc (0=sin, 1=tri, 2=sq, 3=saw, 4=invsaw, 5=noise)
//   deform_info  : (x=mode 0=none/1=wave/2=move, y=spread, _, _)
//   deform_dir   : (dx, dy, dz, _) direction pour mode Move
//   deform_wave  : waveform pour la deform (base, amp, phase, freq)
//   deform_kind.x: discriminant WaveFunc pour la deform
struct MaterialParams {
    anim: vec4<f32>,
    rgb_info: vec4<f32>,
    rgb_wave: vec4<f32>,
    rgb_const: vec4<f32>,
    wave_kind: vec4<u32>,
    deform_info: vec4<f32>,
    deform_dir: vec4<f32>,
    deform_wave: vec4<f32>,
    deform_kind: vec4<u32>,
    /// `.x` bit 0 = is_water (active distortion + Fresnel + caustics
    /// + tinte aqua dans `fs_main`). Reste réservé pour futurs flags
    /// (lava, slime, mirror, portal).
    flags: vec4<u32>,
};
@group(2) @binding(2) var<uniform> material_params: MaterialParams;

// IMPORTANT : MAX_DLIGHTS doit matcher la constante Rust dans `dlight.rs`.
const MAX_DLIGHTS: u32 = 16u;

struct GpuDlight {
    pos_radius: vec4<f32>,        // xyz = centre monde, w = rayon
    color_intensity: vec4<f32>,   // rgb = couleur, a = intensité
};

// On combine `count` et son padding en un seul vec4 pour éviter le piège
// std140 : `vec3<u32>` a une alignement 16, donc un layout
// `{ count: u32, _pad: vec3<u32>, lights }` insère 12 bytes de padding
// invisibles entre `count` et `_pad`, puis 4 bytes entre `_pad` et
// `lights` (qui doit être aligné sur 16).  Total WGSL = 544 bytes, mais
// le `#[repr(C)]` Rust ne les insère pas → mismatch.  Un simple vec4 tête
// d'enregistrement résout tout : offset 0 / size 16 / align 16, et
// `lights` suit à offset 16.  `count_pad.x` porte la valeur.
struct DlightBuffer {
    count_pad: vec4<u32>,
    lights: array<GpuDlight, MAX_DLIGHTS>,
};

@group(3) @binding(0) var<uniform> dlights: DlightBuffer;

// --- Fog volumétrique (bind group 4) -----------------------------------
//
// `color_distance.rgb` = couleur du brouillard (linéaire), `color_distance.a`
// = distance « opaque » au sens Q3 (à cette distance, la visibilité est
// quasi nulle).  `enabled == 1` quand la caméra est *à l'intérieur* d'un
// brush fog du BSP ; sinon 0 et le shader fait un no-op.
// (Le champ s'appelait historiquement `active` mais c'est un mot réservé
// WGSL qui fait planter naga — on le renomme en `enabled` côté GPU tout
// en gardant `active` côté Rust via `#[repr(C)]`.)
//
// Formule : densité = 1 / distance, facteur de visibilité = exp(-d*density).
// C'est une approximation exponentielle plus douce que le linéaire Q3 mais
// plus fidèle à la façon dont la lumière s'atténue physiquement dans un
// milieu diffusant (Beer-Lambert sans absorption sélective).
// Même ruse qu'au-dessus : on packe `enabled` + padding dans un vec4 pour
// que le layout WGSL std matche exactement les 32 bytes que Rust envoie
// (`[f32;4]` + `u32` + `[u32;3]`).  Sans ça, le `vec3<u32>` force un saut
// d'alignement de 4 bytes invisibles côté Rust, et le struct WGSL pèse
// 48 bytes — mismatch de binding.  `state.x` porte la valeur.
struct FogState {
    color_distance: vec4<f32>,
    state: vec4<u32>,
};

@group(4) @binding(0) var<uniform> fog: FogState;

// ─── Group 5 — SSR (#10) ────────────────────────────────────────────
// Lu UNIQUEMENT par water_shade pour faire du Screen-Space Reflection.
// `prev_hdr` = copie de la frame précédente (1-frame lag invisible
// sur l'eau qui n'a pas de motion vectors).  `depth_tex` = depth
// buffer courant pour le raymarch hit test.  Sampler Linear filtrant
// pour le HDR (interp).  Branches non-water ignorent ce groupe.
@group(5) @binding(0) var prev_hdr: texture_2d<f32>;
@group(5) @binding(1) var ssr_depth: texture_depth_2d;
@group(5) @binding(2) var ssr_samp: sampler;

// ─── Group 6 — CSM Shadow Map (#7) ─────────────────────────────────
// Shadow map projetée depuis le POV du soleil.  Pour chaque fragment
// world_pos on calcule sa position en clip space soleil, on convertit
// en UV [0..1], et on compare la depth attendue à celle stockée dans
// `shadow_tex` via le sampler comparison (hardware PCF).  PCF 5×5 fait
// dans le code pour des bordures douces.
struct SunU { view_proj: mat4x4<f32> };
@group(6) @binding(0) var<uniform> sun: SunU;
@group(6) @binding(1) var shadow_tex: texture_depth_2d;
@group(6) @binding(2) var shadow_cmp: sampler_comparison;

/// Calcule la fraction d'éclairage solaire visible (1.0 = full lit,
/// 0.0 = ombre complète) pour la position monde donnée.  Algorithme :
/// 1. Project world_pos via sun.view_proj → clip_sun
/// 2. NDC → UV (avec Y-flip texture)
/// 3. PCF 3×3 : 9 textureSampleCompare avec offsets pixel
/// 4. Bias proportionnel à l'angle normal·sun pour éviter shadow acne
fn shadow_factor(world_pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    let clip = sun.view_proj * vec4<f32>(world_pos, 1.0);
    if (clip.w <= 0.0) { return 1.0; }
    let ndc = clip.xyz / clip.w;
    // Hors frustum sun → on considère "éclairé" (pas d'info, pas d'ombre).
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0
        || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    // Bias proportionnel à l'angle normal·sun (Peter Panning prevention).
    let sun_dir_world = normalize(vec3<f32>(-0.4, -0.3, 0.86));
    let n_dot_l = clamp(dot(normalize(normal), sun_dir_world), 0.0, 1.0);
    let bias = mix(0.0008, 0.0001, n_dot_l);
    let ref_depth = ndc.z - bias;
    // PCF 3×3 — 9 samples avec offset 1 texel.
    let texel = 1.0 / 2048.0;
    var lit = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let off = vec2<f32>(f32(dx), f32(dy)) * texel;
            lit = lit + textureSampleCompare(shadow_tex, shadow_cmp, uv + off, ref_depth);
        }
    }
    return lit / 9.0;
}

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_uv: vec2<f32>,
    @location(3) lightmap_uv: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) lightmap_layer: u32,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_uv: vec2<f32>,
    @location(3) lightmap_uv: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) @interpolate(flat) lightmap_layer: u32,
};

// Applique `tcmod scroll` puis `tcmod rotate` (autour du centre (0.5, 0.5))
// à un UV de base.  Les deux animations sont cumulées dans le même uniform
// pour simplifier — l'ordre exact Q3 n'est pas respecté, mais pour scroll
// seul ou rotate seul (99 % des cas), le résultat est identique.
fn animate_uv(uv: vec2<f32>) -> vec2<f32> {
    let t = camera.time_info.x;
    let scrolled = uv + material_params.anim.xy * t;
    let angle = material_params.anim.z * t;
    if (angle == 0.0) {
        return scrolled;
    }
    let c = cos(angle);
    let s = sin(angle);
    let centered = scrolled - vec2<f32>(0.5, 0.5);
    let rotated = vec2<f32>(c * centered.x - s * centered.y,
                            s * centered.x + c * centered.y);
    return rotated + vec2<f32>(0.5, 0.5);
}

// Évalue une Waveform Q3 à l'instant `t`.  Redéclarée avant le VS pour que
// `deform_position` puisse l'appeler ; réutilisée aussi par `compute_rgb`.
//   base + amp * wave((t + phase) * freq)
// `noise` n'est pas implémenté (retourne 0).
fn eval_wave(wave: vec4<f32>, kind: u32, t: f32) -> f32 {
    let base = wave.x;
    let amp = wave.y;
    let phase = wave.z;
    let freq = wave.w;
    let u = fract((t + phase) * freq);
    var v = 0.0;
    if (kind == 0u) {
        v = sin(u * 6.28318530718);
    } else if (kind == 1u) {
        if (u < 0.5) {
            v = u * 4.0 - 1.0;
        } else {
            v = 3.0 - u * 4.0;
        }
    } else if (kind == 2u) {
        if (u < 0.5) { v = -1.0; } else { v = 1.0; }
    } else if (kind == 3u) {
        v = u;
    } else if (kind == 4u) {
        v = 1.0 - u;
    } else {
        v = 0.0;
    }
    return base + amp * v;
}

// Applique `deformVertexes wave` ou `move` à la position d'un vertex.
//   Wave : `offset = eval_wave(t + spread*(x+y+z))`, déplacement le long
//          de la normale.  Donne cet aspect « flamme qui ondule ».
//   Move : `offset = eval_wave(t)`, déplacement selon `deform_dir`.  Utilisé
//          par les bannières qui oscillent.
// En mode NONE on renvoie la position telle quelle (no-op).
fn deform_position(pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let mode = material_params.deform_info.x;
    if (mode < 0.5) {
        return pos;
    }
    let t = camera.time_info.x;
    if (mode < 1.5) {
        // Wave : déphase par position monde pour que chaque vertex oscille
        // indépendamment → effet « surface qui respire ».
        let spread = material_params.deform_info.y;
        let phase = spread * (pos.x + pos.y + pos.z);
        let offset = eval_wave(
            material_params.deform_wave,
            material_params.deform_kind.x,
            t + phase,
        );
        return pos + normal * offset;
    }
    // Move : déplacement uniforme selon deform_dir.
    let offset = eval_wave(
        material_params.deform_wave,
        material_params.deform_kind.x,
        t,
    );
    return pos + material_params.deform_dir.xyz * offset;
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let deformed = deform_position(in.position, in.normal);
    out.clip_pos = camera.view_proj * vec4<f32>(deformed, 1.0);
    out.world_pos = deformed;
    out.normal = in.normal;
    out.tex_uv = animate_uv(in.tex_uv);
    out.lightmap_uv = in.lightmap_uv;
    out.color = in.color;
    out.lightmap_layer = in.lightmap_layer;
    return out;
}

// Contribution additive des dlights actifs au point `world_pos` avec
// normale `n`.  Falloff quadratique `(1 - d/r)^2` clampé à [0, 1], pondéré
// par `max(0, n·l)` pour que les surfaces face-à-la-lumière s'éclairent
// plus fort que celles de dos (effet Lambert simple, sans spéculaire).
fn dlight_contribution(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var accum = vec3<f32>(0.0);
    let nrm = normalize(n);
    let count = min(dlights.count_pad.x, MAX_DLIGHTS);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let d = dlights.lights[i];
        let to = d.pos_radius.xyz - world_pos;
        let dist = length(to);
        let r = max(d.pos_radius.w, 1.0);
        let t = 1.0 - dist / r;
        if (t <= 0.0) {
            continue;
        }
        let falloff = t * t;
        let ndotl = max(dot(nrm, to / max(dist, 1e-3)), 0.25);
        accum = accum + d.color_intensity.rgb * d.color_intensity.a * falloff * ndotl;
    }
    return accum;
}

// Calcule la couleur de vertex effective selon `rgbgen`.  En mode
// DEFAULT on retourne la couleur baked ; en WAVE on retourne un glow
// scalaire luminance (r=g=b=wave) ; en CONST on retourne la couleur fixe.
fn compute_rgb(vertex_color: vec4<f32>) -> vec4<f32> {
    let mode = material_params.rgb_info.x;
    if (mode < 0.5) {
        return vertex_color;
    }
    if (mode < 1.5) {
        // Wave : évalue le glow scalaire, clamp [0, 1], l'applique aux 3 canaux.
        let glow = clamp(
            eval_wave(material_params.rgb_wave, material_params.wave_kind.x, camera.time_info.x),
            0.0, 1.0,
        );
        return vec4<f32>(glow, glow, glow, vertex_color.a);
    }
    // Const : remplace la couleur de vertex par la constante shader.
    return vec4<f32>(material_params.rgb_const.rgb, vertex_color.a);
}

// Applique le brouillard actif au `color` éclairé, en fonction de la distance
// oeil → fragment.  Quand `fog.state.x == 0` la fonction est un pass-through.
fn apply_fog(color: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    if (fog.state.x == 0u) {
        return color;
    }
    let eye = camera.view_pos.xyz;
    let d = length(world_pos - eye);
    // `distance` (w) peut être quasi-nul dans un shader mal compilé — on
    // clampe pour éviter une division par zéro qui donnerait `inf`.
    let fog_dist = max(fog.color_distance.a, 1.0);
    let density = 1.0 / fog_dist;
    let visibility = clamp(exp(-d * density), 0.0, 1.0);
    return mix(fog.color_distance.rgb, color, visibility);
}

// ─── **Volumetric fog** (v0.9.5++) ────────────────────────────────
//
// Raymarch per-pixel depuis la caméra jusqu'au fragment world_pos.
// À chaque step on évalue une densité (FBM noise + height decay) et on
// accumule :
//   * **extinction**   = exp(-density × dt) — atténue la couleur scène
//   * **in-scattering** = lumière solaire dispersée vers la caméra
//     (phase Henyey-Greenstein g=0.4 → forward-scattering modéré)
//
// Le résultat est `scene * transmittance + scatter`.  Donne :
//   - Atmospheric depth fade (fond fade vers la couleur du fog)
//   - Light shafts visibles dans la direction du soleil (god rays
//     volumétriques, complémentaires aux god rays screen-space de
//     post.rs qui sont 2D)
//   - Réagit dynamiquement à la position du joueur + direction du soleil
//
// 12 steps : compromis qualité/perf.  Density basée sur world_pos.xy
// (pas de noise 3D pour rester cheap), modulée par altitude.

const VOL_FOG_STEPS: i32 = 12;
const VOL_FOG_DENSITY_BASE: f32 = 0.00018;
const VOL_FOG_HEIGHT_REF: f32 = 200.0;   // altitude de référence
const VOL_FOG_HEIGHT_DECAY: f32 = 0.0008; // décroissance verticale

fn vol_fog_density(p: vec3<f32>) -> f32 {
    // FBM 2 octaves sur XY pour casser l'uniformité.
    let n1 = sin(p.x * 0.0023 + p.y * 0.0019) * 0.5 + 0.5;
    let n2 = sin(p.x * 0.0061 - p.y * 0.0073) * 0.5 + 0.5;
    let noise = n1 * 0.65 + n2 * 0.35;
    // Décroissance exponentielle avec l'altitude (fog accumulé en bas).
    let h = (p.z - VOL_FOG_HEIGHT_REF) * VOL_FOG_HEIGHT_DECAY;
    let height_factor = exp(-max(h, 0.0));
    return VOL_FOG_DENSITY_BASE * (0.6 + 0.4 * noise) * height_factor;
}

fn vol_fog_phase_hg(cos_theta: f32, g: f32) -> f32 {
    // Phase Henyey-Greenstein : forward (g>0) ou backward (g<0) scattering.
    let g2 = g * g;
    let denom = pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-3), 1.5);
    return (1.0 - g2) / (12.566370614 * denom); // 4π
}

fn apply_volumetric_fog(scene: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let eye = camera.view_pos.xyz;
    let to_eye = eye - world_pos;
    let dist = length(to_eye);
    if (dist < 1.0) { return scene; }
    // **Early-out perf** (v0.9.5++ polish) — skip le raymarch 12-step
    // au-delà de 4000u (la transmittance est déjà ~0.05, contribution
    // visuelle négligeable mais coût FPS élevé).  Pour les longs tirs
    // RG / le sky, ça économise ~15% de fragment cost.
    if (dist > 4000.0) { return scene; }
    let dir_to_eye = to_eye / dist;
    // Sun direction monde (cohérente avec sky/terrain/CSM).
    let sun_dir = normalize(vec3<f32>(-0.4, -0.3, 0.86));
    // Phase HG : on regarde dans la direction -dir_to_eye (du fragment
    // vers la caméra), le soleil est dans sun_dir.  cos(angle) entre
    // direction de la lumière et direction de view (back-scattering).
    let cos_theta = dot(-dir_to_eye, -sun_dir);
    let phase = vol_fog_phase_hg(cos_theta, 0.40);
    // Couleur de scattering — bleu-cyan léger pour évoquer Rayleigh +
    // teinte chaude au crépuscule (ici neutre, modulable par sun_color).
    let scatter_color = vec3<f32>(0.55, 0.70, 0.95);
    // Couleur du sun (scatter vers la caméra modulé par phase).
    let sun_color = vec3<f32>(1.00, 0.95, 0.85);
    let n_steps = VOL_FOG_STEPS;
    let step_len = dist / f32(n_steps);
    var transmittance = 1.0;
    var scattered = vec3<f32>(0.0);
    // March DEPUIS le fragment vers la caméra (équivalent intégrale).
    for (var i = 0; i < 12; i = i + 1) {
        let t = (f32(i) + 0.5) * step_len;
        let p = world_pos + dir_to_eye * t;
        let density = vol_fog_density(p);
        let extinction = density * step_len;
        // In-scattering pondéré par le phase (lumière du soleil
        // dispersée vers le pixel) + ambient scatter (lumière diffuse
        // bleue/blanc-gris du ciel).
        let in_scatter = (sun_color * phase + scatter_color * 0.05)
                       * density * step_len;
        scattered = scattered + in_scatter * transmittance;
        transmittance = transmittance * exp(-extinction);
    }
    return scene * transmittance + scattered;
}

// Tonemap + grading final — appliqué à la toute dernière étape de
// `shade`.  L'objectif est de sortir du rendu "plat en aplat Q3 1999" et
// d'obtenir quelque chose qui se rapproche des shooters modernes sans
// alourdir la chaîne (pas de post-process full-screen, tout est inline).
//
// 1. Saturation : on amplifie la distance au gris → couleurs plus vives.
//    Coeff 1.10, subtil mais ça fait respirer les dominantes jaune/bleu
//    des maps Q3 qui sont souvent désaturées à cause du baking lightmap.
// 2. Contraste en shadows : petite élévation (gamma 0.95 sur le canal
//    luminance) → les zones sombres gagnent en détail au lieu d'être
//    écrasées vers le noir.
// 3. Rolloff des highlights : remplace un simple `clamp(x,0,1)` par une
//    courbe douce `x / (x + 0.85) * 1.85` qui préserve la transition
//    proche-blanc au lieu de tout saturer brutalement.  Ça adoucit les
//    flashs de muzzle, les flares et les surfaces plein soleil.
// **`tonemap_grade` SUPPRIMÉE** (v0.9.5++ polish) — fonction
// obsolète, plus aucun appelant.  Le tonemap final est appliqué par
// `post.rs::fs_compose` (ACES Narkowicz + color grading + vignette).

// ─── Water shader effects ─────────────────────────────────────────────
//
// Activé quand `material_params.flags.x` a le bit 0 set. Combine 4
// techniques pour donner l'illusion d'une vraie surface d'eau :
//
// 1. **UV turb multi-octave** : on ajoute des sinusoïdes à plusieurs
//    fréquences/amplitudes (octaves 1..3) sur les UV du diffuse. Effet
//    "ondes qui dansent" plus riche qu'une seule sinusoïde — plus
//    proche de l'eau réelle qui n'a pas une seule fréquence dominante.
// 2. **Fresnel approximé** : fade entre une teinte cyan profonde
//    (face perpendiculaire à l'œil = on regarde en plongée → on voit
//    le fond bleu) et une teinte plus pâle réfléchie (face presque
//    rasante = surface miroir).  Donne le look "lac sombre quand
//    on regarde droit dedans, brillant quand on regarde au ras".
// 3. **Caustics simulés** : pattern noise multi-octave qui module la
//    luminosité — simule les reflets de lumière qui dansent au fond.
// 4. **Sparkle highlights** : sur les crêtes d'ondes (où l'amplitude
//    de la sinusoïde dépasse un seuil) on ajoute un highlight blanc-
//    bleu pour évoquer le scintillement de l'écume.

fn water_turb_uv(uv: vec2<f32>, world_pos: vec3<f32>, t: f32) -> vec2<f32> {
    // Multi-octave : l'amplitude diminue géométriquement avec la
    // fréquence (Brownian fractional). Phases différentes par octave
    // pour éviter les patterns trop évidents.
    let p = world_pos.xy * 0.05;
    let o1 = vec2<f32>(
        sin(p.x * 1.0 + t * 0.6),
        sin(p.y * 1.0 + t * 0.7 + 1.3),
    ) * 0.012;
    let o2 = vec2<f32>(
        sin(p.x * 2.3 + t * 1.4 + 0.7),
        sin(p.y * 2.7 + t * 1.6 + 2.1),
    ) * 0.006;
    let o3 = vec2<f32>(
        sin(p.x * 4.5 + t * 2.5 + 1.9),
        sin(p.y * 5.1 + t * 2.8),
    ) * 0.003;
    return uv + o1 + o2 + o3;
}

// Bruit pseudo-aléatoire 2D simple (hash sur UV). Pas un vrai Perlin
// mais suffisant pour le shimmer caustic sans coûter cher.
fn water_hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn water_caustics(world_pos: vec3<f32>, t: f32) -> f32 {
    // **Caustics enrichies** (v0.9.5++ #35) — pattern multi-octave
    // animé.  3 strates à fréquences/vitesses différentes, sommées et
    // seuillées pour produire des "rayures lumineuses" qui dansent.
    let p = world_pos.xy;
    // Octave 1 : large échelle, lente.
    let n1 = water_hash(floor(p * 0.04 + vec2<f32>(t * 0.30, t * 0.20)));
    // Octave 2 : moyen échelle, rotation différente.
    let n2 = water_hash(floor(p * 0.08 + vec2<f32>(-t * 0.40, t * 0.55)));
    // Octave 3 : fine, très rapide → effet "scintillement".
    let n3 = water_hash(floor(p * 0.16 + vec2<f32>(t * 0.65, -t * 0.42)));
    // Sin-modulated pour adoucir les hash discrets.
    let modulator = 0.5 + 0.5 * sin(p.x * 0.05 + p.y * 0.07 + t * 1.4);
    let mix_n = (n1 * 0.5 + n2 * 0.35 + n3 * 0.15) * (0.7 + modulator * 0.5);
    // Threshold + courbe pour "bandes" lumineuses.
    let lum = pow(mix_n, 5.0) * 3.5;
    return clamp(lum, 0.0, 1.0);
}

/// **Foam crests** (v0.9.5++ #35) — détecte les pics d'onde et
/// retourne un alpha [0..1] pour la mousse blanche.  Pic = combinaison
/// de 3 sinusoïdes orthogonales à phases différentes ; on prend la
/// crête du max et applique un seuil pointu.
fn water_foam(world_pos: vec3<f32>, t: f32) -> f32 {
    let s1 = sin(world_pos.x * 0.07 + t * 1.5);
    let s2 = sin(world_pos.y * 0.09 + t * 1.7);
    let s3 = sin((world_pos.x + world_pos.y) * 0.05 + t * 1.2);
    let combined = (s1 + s2 + s3) / 3.0;
    return smoothstep(0.55, 0.85, combined);
}

fn water_shade(
    in: VsOut,
    diffuse_base: vec4<f32>,
    lm: vec4<f32>,
) -> vec4<f32> {
    let t = camera.time_info.x;
    // 1. UV turbulent → resample diffuse pour casser la régularité.
    let turb_uv = water_turb_uv(in.tex_uv, in.world_pos, t);
    let diff_turb = textureSample(diffuse_tex, diffuse_samp, turb_uv);
    // 2. Fresnel : produit scalaire view·normal. À 0 (perpendiculaire)
    // on est en plongée → teinte sombre. À 1 (rasant) on a la surface
    // brillante.
    let view = normalize(camera.view_pos.xyz - in.world_pos);
    let n = normalize(in.normal);
    let fres = 1.0 - clamp(abs(dot(view, n)), 0.0, 1.0);
    // Courbe Schlick avec exposant 4 (vs 3 avant) — réflexion plus
    // physiquement plausible : surface plus mate en plongée, plus
    // miroir au ras (cf. Schlick 1994 : F = F0 + (1-F0)*(1-cos)^5).
    let fres_sharp = pow(fres, 4.0);
    // 3. Couleurs : profondeur cyan-bleu vs surface bleu pâle.
    // Deep tint éclairci (0.05/0.20/0.30 → 0.08/0.28/0.40) pour que
    // l'eau ne soit jamais quasi-noire en plongée verticale.
    let deep = vec3<f32>(0.08, 0.28, 0.40);
    let shallow = vec3<f32>(0.45, 0.75, 0.95);
    let surf_color = mix(deep, shallow, fres_sharp);
    // 4. Caustics : module l'éclairage du fond avec une pattern noise.
    // Multiplicateur réduit 0.7 → 0.45 pour éviter sur-illumination
    // qui faisait briller l'eau comme un projecteur disco.
    let cau = water_caustics(in.world_pos, t) * 0.45;
    // 5. Mix base diffuse (terrain sous l'eau) + teinte eau + caustics.
    // Mix 55 % → 60 % vers surf_color : eau plus "présente" en surface,
    // moins révélatrice du terrain sous-jacent (qui pouvait dominer
    // visuellement avec ses textures de sable / rocher).
    var rgb = diff_turb.rgb * (lm.rgb + vec3<f32>(cau));
    rgb = mix(rgb, surf_color, 0.60);
    // 6. Sparkle sur les crêtes : on échantillone une sinusoïde haute
    // fréquence sur la position pour trouver les "pointes" d'onde.
    // Magnitude 1.5 → 0.7 — avant ça créait des "flashs" blancs trop
    // intenses qui se mélangeaient avec le bloom post.
    let crest = sin(in.world_pos.x * 0.18 + t * 2.3)
              * sin(in.world_pos.y * 0.21 + t * 2.7);
    let sparkle = pow(max(crest, 0.0), 12.0) * 0.7;
    rgb = rgb + vec3<f32>(0.6, 0.85, 1.0) * sparkle;
    // **Foam crests** (v0.9.5++ #35) — mousse blanche sur les vraies
    // crêtes d'onde.  Mix 0.55 → 0.35 (mousse plus subtile, moins
    // de patches blancs lisibles).
    let foam = water_foam(in.world_pos, t);
    rgb = mix(rgb, vec3<f32>(0.95, 0.97, 1.0), foam * 0.35);
    // **Tropical glow** — lagons réunionnais ont un teint cyan-vert
    // sous-jacent. On ajoute un peu de turquoise dans les zones peu
    // profondes (peu de Fresnel = vue plongeante).
    rgb = rgb + vec3<f32>(0.05, 0.18, 0.22) * (1.0 - fres_sharp) * 0.4;
    // 7. Dlights toujours appliqués pour qu'une roquette qui passe
    // au-dessus illumine bien l'eau.
    rgb = rgb + dlight_contribution(in.world_pos, in.normal) * 0.4;

    // ─── **SSR — Screen Space Reflection** (v0.9.5++ #10) ──────────
    // Algorithme : raymarch en clip-space dans la direction de la
    // réflexion view·normal, jusqu'à ce que la profondeur reconstruite
    // dépasse la profondeur du depth buffer à l'UV courante.
    //
    // Pipeline :
    //   1. Reconstruct view-space ray (origin = world_pos, dir = reflect(view_dir))
    //   2. Step le ray dans le world, project chaque step en clip
    //   3. Compare clip.z / clip.w au depth buffer à clip.xy/clip.w
    //   4. Premier hit → sample prev_hdr à cet UV → c'est la réflexion
    //
    // Lecture depuis prev_hdr (frame N-1) — 1-frame lag invisible sur
    // une surface aussi peu fréquente que l'eau.
    let view_dir = normalize(in.world_pos - camera.view_pos.xyz);
    // Réflexion : l'eau a une normale +Z forte, mais les ondelettes
    // perturbent légèrement.  On utilise une normale approximative.
    let water_normal = normalize(in.normal + vec3<f32>(
        sin(in.world_pos.x * 0.05 + t * 1.2) * 0.05,
        sin(in.world_pos.y * 0.06 + t * 1.4) * 0.05,
        0.0,
    ));
    let refl_dir = reflect(view_dir, water_normal);
    var ssr_color = vec3<f32>(0.0);
    var ssr_alpha = 0.0;
    if (refl_dir.z > 0.05) {
        // On ne tente le SSR que si la réflexion va vers le HAUT —
        // sinon ça traverserait le sol, valeurs garbage.
        let max_steps = 32;
        let step_size = 24.0; // unités Q3 par step
        var march_pos = in.world_pos + water_normal * 2.0; // léger offset
        for (var i = 0; i < 32; i = i + 1) {
            march_pos = march_pos + refl_dir * step_size;
            // Project en clip space.
            let clip = camera.view_proj * vec4<f32>(march_pos, 1.0);
            if (clip.w <= 0.0) { break; }
            let ndc = clip.xyz / clip.w;
            // Hors frustum → on bail.
            if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0) {
                break;
            }
            let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
            let scene_depth = textureSampleLevel(ssr_depth, ssr_samp, uv, 0.0);
            let ray_depth = ndc.z;
            // Hit test : ray_depth a dépassé scene_depth → la
            // surface réelle est plus proche que la position
            // marchée → on a frappé quelque chose.
            if (ray_depth > scene_depth && ray_depth - scene_depth < 0.02) {
                ssr_color = textureSampleLevel(prev_hdr, ssr_samp, uv, 0.0).rgb;
                // Fade aux bords d'écran pour éviter coupure sèche.
                let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
                let edge_fade = smoothstep(0.0, 0.15, edge);
                ssr_alpha = edge_fade;
                break;
            }
        }
    }
    // Mélange : Fresnel détermine l'intensité de la réflexion.
    // À surface rasante (fres_sharp ~1) → forte réflexion.
    // À plongée verticale (fres_sharp ~0) → on voit le fond, peu de SSR.
    let ssr_strength = fres_sharp * ssr_alpha * 0.85;
    rgb = mix(rgb, ssr_color, ssr_strength);

    rgb = apply_fog(rgb, in.world_pos);
    // **Pas de tonemap_grade ici** (v0.9.5++ fix) — le pipeline est HDR
    // (Rgba16Float scene buffer), `post.rs::fs_compose` applique déjà
    // ACES Narkowicz + color grading + edge AO + vignette en final.
    // Avant on avait DOUBLE tonemap (Reinhard ici puis ACES post-pass)
    // qui écrasait les highlights de l'eau et lui donnait un look
    // "plat-saturé".  Maintenant l'eau écrit en HDR linéaire et le
    // tonemap unique de post-pass gère la compression dynamique.
    // Alpha : Fresnel + base. Plus on regarde rasant, plus on est
    // opaque (réflexion). Plus on regarde droit, plus on voit le fond.
    let alpha = mix(0.65, 0.95, fres_sharp);
    return vec4<f32>(rgb, alpha * diffuse_base.a * in.color.a);
}

fn shade(in: VsOut) -> vec4<f32> {
    let diffuse = textureSample(diffuse_tex, diffuse_samp, in.tex_uv);
    let lm = textureSample(
        lightmap_tex,
        lightmap_samp,
        in.lightmap_uv,
        i32(in.lightmap_layer),
    );
    // Branche eau : si le bit FLAG_WATER (bit 0 de flags.x) est set,
    // on bascule sur le pipeline visuel dédié.
    if ((material_params.flags.x & 1u) != 0u) {
        return water_shade(in, diffuse, lm);
    }
    // Couleur de vertex effective : selon le rgbgen du shader, on garde
    // la couleur baked ou on la remplace par un glow (wave) ou une const.
    let vc = compute_rgb(in.color);
    // couleur baked × lightmap × diffuse. Le sampler lightmap applique déjà
    // l'overbright CPU-side en écrivant en Rgba8Unorm non-sRGB.
    var rgb = vc.rgb * lm.rgb * diffuse.rgb;
    // **CSM shadow factor** (v0.9.5++ #7) — on échantillonne la shadow
    // map du soleil et on assombrit la composante "lightmap directe".
    // Le lightmap Q3 inclut déjà la lumière indirecte (radiosity baked)
    // donc on n'écrase pas tout : on module à 65 % max (35 % de
    // l'éclairage reste comme ambient via radiosity).
    let shadow = shadow_factor(in.world_pos, in.normal);
    let shadow_mix = mix(0.35, 1.0, shadow); // 0.35 ambiant min
    rgb = rgb * shadow_mix;
    // Ajout additif des dlights (halos rockets / muzzle flashes) —
    // PAS modulé par shadow car ce sont des lights locales (point
    // lights) qui ne sont pas occluded par la shadow map solaire.
    rgb = rgb + dlight_contribution(in.world_pos, in.normal) * diffuse.rgb;
    // Fog Q3 (par-volume, no-op quand caméra hors volume).
    rgb = apply_fog(rgb, in.world_pos);
    // **Volumetric fog** (v0.9.5++) — raymarch per-pixel + sun
    // in-scattering.  Donne atmospheric depth fade + god rays 3D
    // visibles dans la direction du soleil.  Toujours actif (pas
    // conditionné aux fog volumes Q3).
    rgb = apply_volumetric_fog(rgb, in.world_pos);
    // **Pas de tonemap_grade ici** (v0.9.5++ fix) — le pipeline est HDR
    // (Rgba16Float scene buffer), `post.rs::fs_compose` applique déjà
    // ACES Narkowicz + color grading + edge AO + vignette en final.
    // Avant on avait DOUBLE tonemap (Reinhard ici puis ACES post-pass)
    // qui écrasait les midtones.  Le path eau était déjà fixé, on aligne
    // ici le path opaque pour cohérence HDR end-to-end.
    return vec4<f32>(rgb, diffuse.a * in.color.a);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return shade(in);
}

@fragment
fn fs_main_alphatest(in: VsOut) -> @location(0) vec4<f32> {
    let c = shade(in);
    // alphaFunc GT0 — discard les pixels totalement transparents.
    if (c.a < 0.5) {
        discard;
    }
    return c;
}
