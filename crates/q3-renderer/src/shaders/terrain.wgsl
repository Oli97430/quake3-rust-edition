// Terrain BR (Réunion) — pipeline shader.
//
// Vertex layout match `q3_terrain::mesh::TerrainVertex` (48 octets):
//   pos:    vec3<f32>  + 1 pad
//   normal: vec3<f32>  + 1 pad
//   splat:  vec4<f32>
//
// **v0.9.5 polish** — gros pass de qualité visuelle :
//   * detail noise haute fréquence dans le fragment pour casser
//     l'aspect "carpet" plat de 4 couleurs uniformes
//   * slope-based blending : les faces verticales (falaises) prennent
//     la couleur roche même si le splat dit "végétation", ce qui
//     produit des cliffs gris au flanc des collines vertes
//   * rim lighting : un fresnel léger à grand angle qui souligne les
//     crêtes — donne du relief volumétrique
//   * atmospheric haze : fog plus chaud à l'horizon, plus froid en
//     altitude, accumulation linéaire avec densité
//   * macro-noise variation par biome : la végétation n'est pas un
//     vert plat mais varie en clairières plus claires/sombres

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    inv_view_proj_rot: mat4x4<f32>,
    time_info: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct TerrainParams {
    sun_dir: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>, // x=near y=far z=density w=water_level
};
@group(1) @binding(0) var<uniform> tparams: TerrainParams;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) splat: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) splat: vec4<f32>,
    @location(3) view_dist: f32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(in.pos, 1.0);
    out.world_pos = in.pos;
    out.normal = normalize(in.normal);
    out.splat = in.splat;
    out.view_dist = length(in.pos - camera.view_pos.xyz);
    return out;
}

// ─── Procedural noise helpers ──────────────────────────────────────
// Hash 2D simple — déterministe, bon enough pour le détail visuel.
fn hash2(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

// Value noise interpolé — 1 octave.
fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let a = hash2(i);
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    let u = f * f * (3.0 - 2.0 * f); // smoothstep
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// FBM 3 octaves — donne un grain de surface satisfaisant.
fn fbm(p: vec2<f32>) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0; i < 3; i = i + 1) {
        sum = sum + vnoise(p * freq) * amp;
        amp = amp * 0.5;
        freq = freq * 2.0;
    }
    return sum;
}

// ─── Couleurs de biome canoniques pour la Réunion ───────────────────
// 2 nuances par biome → la zone n'est pas un aplat, mais un bruit
// entre les deux (fbm mod sur les coords monde).
const ROCK_DARK:  vec3<f32> = vec3<f32>(0.42, 0.36, 0.32);
const ROCK_LIGHT: vec3<f32> = vec3<f32>(0.68, 0.62, 0.55);
const SAND_DARK:  vec3<f32> = vec3<f32>(0.30, 0.26, 0.22);
const SAND_LIGHT: vec3<f32> = vec3<f32>(0.62, 0.55, 0.42);
const VEG_DARK:   vec3<f32> = vec3<f32>(0.18, 0.45, 0.15);
const VEG_LIGHT:  vec3<f32> = vec3<f32>(0.45, 0.78, 0.30);
const URBAN_DARK: vec3<f32> = vec3<f32>(0.55, 0.52, 0.48);
const URBAN_LIGHT:vec3<f32> = vec3<f32>(0.85, 0.82, 0.78);

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);

    // **Slope-based override** : si la pente est forte (n.z petit), on
    // bascule vers la couleur roche peu importe le splat. Donne des
    // falaises grises naturelles au flanc des collines vertes.
    let slope = 1.0 - clamp(n.z, 0.0, 1.0); // 0 = sol plat, 1 = mur
    let cliff_factor = smoothstep(0.30, 0.55, slope);

    // **Macro variation** : un fbm basse fréquence sur la position
    // monde donne des "patches" de couleur plus claire/sombre par
    // zone — la végétation n'est plus un aplat.
    let macro_variation = fbm(in.world_pos.xy * 0.0008);
    // **Detail noise** : un grain haute fréquence pour le rendu de
    // près. À grande distance ça moyenne en uniforme grâce au fog.
    let detail = fbm(in.world_pos.xy * 0.05);

    // Mix dark↔light par biome basé sur macro+detail.
    let t = clamp(macro_variation * 0.7 + detail * 0.3, 0.0, 1.0);
    let rock  = mix(ROCK_DARK,  ROCK_LIGHT,  t);
    let sand  = mix(SAND_DARK,  SAND_LIGHT,  t);
    let veg   = mix(VEG_DARK,   VEG_LIGHT,   t);
    let urban = mix(URBAN_DARK, URBAN_LIGHT, t);

    // Splat blend 4-canaux sommés à 1.0 (CPU garantit).
    var albedo =
        in.splat.x * rock
      + in.splat.y * sand
      + in.splat.z * veg
      + in.splat.w * urban;

    // Slope override : injecte du rock proportionnel au cliff_factor.
    albedo = mix(albedo, rock, cliff_factor);

    // **Lighting** : Lambert + ambient hemisphère + rim léger.
    let l = normalize(tparams.sun_dir.xyz);
    let n_dot_l = max(dot(n, l), 0.0);
    // Hémisphère ambient : sky-up bleu pâle, ground-down marron — donne
    // un ambient occlusion gratuit pour les faces vers le bas.
    let sky_amb = vec3<f32>(0.40, 0.55, 0.70);
    let ground_amb = vec3<f32>(0.18, 0.14, 0.10);
    let hemi = mix(ground_amb, sky_amb, clamp(n.z * 0.5 + 0.5, 0.0, 1.0));
    var lit = albedo * (hemi * 0.45 + vec3<f32>(1.0, 0.95, 0.82) * n_dot_l * 0.85)
           * tparams.sun_dir.w;

    // **Rim light** — fresnel grand angle qui souligne les crêtes
    // contre le ciel. Plus visible quand on regarde un sommet à
    // contre-jour.
    let to_view = normalize(camera.view_pos.xyz - in.world_pos);
    let fresnel = pow(1.0 - max(dot(n, to_view), 0.0), 3.0);
    let rim_col = vec3<f32>(1.0, 0.85, 0.55) * 0.25 * fresnel;
    lit = lit + rim_col;

    // **Specular subtil sur rock** — un highlight Blinn-Phong étroit
    // pour faire briller les surfaces de roche humides ou polies.
    let half_v = normalize(l + to_view);
    let spec = pow(max(dot(n, half_v), 0.0), 32.0);
    let spec_strength = (in.splat.x + cliff_factor) * 0.15;
    lit = lit + vec3<f32>(1.0, 0.95, 0.85) * spec * spec_strength;

    // **Atmospheric haze** : fog distance non linéaire (squared) pour
    // que les zones < 2000u soient nettes mais que l'horizon se fonde
    // dans la couleur ciel. Couleur fog modulée par altitude (plus
    // chaud près du sol, plus froid en altitude).
    let fog_t_lin = clamp(
        (in.view_dist - tparams.fog_params.x)
            / max(tparams.fog_params.y - tparams.fog_params.x, 1.0),
        0.0,
        1.0,
    );
    let fog_t = pow(fog_t_lin, 1.6) * tparams.fog_params.z;
    // Couleur fog : mix turquoise océan (sol) → bleu froid (altitude).
    let fog_low = tparams.fog_color.rgb;
    let fog_high = vec3<f32>(0.30, 0.50, 0.75);
    let alt_t = clamp(in.world_pos.z / 1200.0, 0.0, 1.0);
    let fog_col = mix(fog_low, fog_high, alt_t);
    let final_color = mix(lit, fog_col, fog_t);

    return vec4<f32>(final_color, 1.0);
}
