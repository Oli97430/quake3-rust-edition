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
fn tonemap_grade(rgb: vec3<f32>) -> vec3<f32> {
    // Saturation — on travaille par rapport à la luminance perceptuelle
    // BT.709 (le monde Q3 est déjà en sRGB clair, donc c'est OK).
    let lum = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let satured = mix(vec3<f32>(lum), rgb, 1.10);
    // Soft highlight rolloff — Reinhard modifié avec boost x1.85 pour
    // compenser la compression en haut de gamme.
    let mapped = satured / (satured + vec3<f32>(0.85));
    let highlights = mapped * 1.85;
    // Shadow lift — gamma 0.95 légère pour remonter les détails sombres.
    return pow(max(highlights, vec3<f32>(0.0)), vec3<f32>(0.95));
}

fn shade(in: VsOut) -> vec4<f32> {
    let diffuse = textureSample(diffuse_tex, diffuse_samp, in.tex_uv);
    let lm = textureSample(
        lightmap_tex,
        lightmap_samp,
        in.lightmap_uv,
        i32(in.lightmap_layer),
    );
    // Couleur de vertex effective : selon le rgbgen du shader, on garde
    // la couleur baked ou on la remplace par un glow (wave) ou une const.
    let vc = compute_rgb(in.color);
    // couleur baked × lightmap × diffuse. Le sampler lightmap applique déjà
    // l'overbright CPU-side en écrivant en Rgba8Unorm non-sRGB.
    var rgb = vc.rgb * lm.rgb * diffuse.rgb;
    // Ajout additif des dlights (halos rockets / muzzle flashes).
    rgb = rgb + dlight_contribution(in.world_pos, in.normal) * diffuse.rgb;
    // Fog volumétrique (no-op quand caméra hors volume).
    rgb = apply_fog(rgb, in.world_pos);
    // Dernier pas : tonemap + grading pour pousser le contraste/saturation.
    rgb = tonemap_grade(rgb);
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
