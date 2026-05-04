// Ciel procédural — gradient bleu → horizon ocre, avec un léger halo
// "soleil" dans la direction fixée par `sun_dir`. Dessiné après le monde
// avec depth test = LessEqual et z = 1.0 (dernier plan).
//
// On réutilise le Camera group pour récupérer view + projection, afin de
// reconstruire la direction monde par pixel.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    inv_view_proj_rot: mat4x4<f32>,
    time_info: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VOut {
    // Plein écran en 3 vertices : (-1,-1), (3,-1), (-1,3).
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[idx];
    var out: VOut;
    // z = 1.0 = far plane → passe LessEqual.
    out.clip_pos = vec4<f32>(p, 1.0, 1.0);
    out.ndc = p;
    return out;
}

// Reconstruit la direction monde à partir d'un NDC via inv(proj × view_rot_only).
// Le résultat est en coordonnées Q3 : X=fwd, Y=left, Z=up.
fn world_dir_from_ndc(ndc: vec2<f32>) -> vec3<f32> {
    // WGPU : NDC z = 0 (near) .. 1 (far). z=1 → point à l'infini.
    let clip = vec4<f32>(ndc, 1.0, 1.0);
    let w = camera.inv_view_proj_rot * clip;
    return normalize(w.xyz / max(w.w, 1e-6));
}

// **FBM noise pour les nuages volumétriques** — variantes 2D
// échantillonnées sur la projection ciel.  Pas de noise lib, donc
// hash + value noise classique.
fn hash22(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}
fn vnoise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash22(i);
    let b = hash22(i + vec2<f32>(1.0, 0.0));
    let c = hash22(i + vec2<f32>(0.0, 1.0));
    let d = hash22(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn cloud_fbm(p: vec2<f32>, t: f32) -> f32 {
    // Drift lent du ciel : les nuages avancent vers l'est.
    let q = p + vec2<f32>(t * 0.015, 0.0);
    var s = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0; i < 5; i = i + 1) {
        s = s + vnoise2(q * freq) * amp;
        amp = amp * 0.55;
        freq = freq * 2.1;
    }
    return s;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let dir = world_dir_from_ndc(in.ndc);
    let t = clamp(dir.z * 0.5 + 0.5, 0.0, 1.0);

    // **Atmospheric scattering simplifié Rayleigh + Mie** (v0.9.5++)
    // — approximation analytique : la couleur ciel dépend de l'angle
    // entre la direction de vue et le soleil. Quand on regarde près
    // du soleil, le ciel est plus chaud (rouge/jaune par dispersion
    // Mie) ; loin, il est plus froid (bleu Rayleigh).
    let sun_dir_local = normalize(vec3<f32>(-0.4, -0.3, 0.86));
    let dir_dot_sun = clamp(dot(dir, sun_dir_local), -1.0, 1.0);
    // Rayleigh phase function (1 + cos²) approximée.
    let rayleigh_phase = 0.75 * (1.0 + dir_dot_sun * dir_dot_sun);
    // Mie phase function — Henyey-Greenstein g=0.76 (forward scatter).
    let g = 0.76;
    let g2 = g * g;
    let mie_phase = (1.0 - g2) /
        max(pow(1.0 + g2 - 2.0 * g * dir_dot_sun, 1.5), 0.001) * 0.05;
    // Gradient base ciel tropical : du turquoise horizon au bleu profond zénith.
    let horizon_base = vec3<f32>(0.45, 0.70, 0.90);
    let zenith_base  = vec3<f32>(0.04, 0.14, 0.40);
    let altitude_t = smoothstep(0.0, 0.85, t);
    let sky_base = mix(horizon_base, zenith_base, altitude_t);
    // Rayleigh (bleu) + Mie (chaud près du soleil).
    let rayleigh_col = vec3<f32>(0.55, 0.75, 1.00) * rayleigh_phase * 0.4;
    let mie_col = vec3<f32>(1.00, 0.85, 0.55) * mie_phase;
    let sky_grad = sky_base + rayleigh_col + mie_col;

    // **Volumetric haze sur l'horizon** (v0.9.5++) — la ligne sol-ciel
    // est noyée dans une brume tropicale (humidité haute Réunion).
    // Plus dense près de l'horizon (z ≈ 0), s'atténue vers le zénith.
    let haze_band = pow(1.0 - abs(dir.z), 8.0); // pic à l'équateur
    let haze_col = vec3<f32>(0.85, 0.82, 0.72); // brume blanc-or
    let sky_with_haze = mix(sky_grad, haze_col, haze_band * 0.55);

    // **Sun disk + halo + god rays** — réutilise sun_dir_local défini plus haut.
    let sun_dir = sun_dir_local;
    let sun_dot = max(dot(dir, sun_dir), 0.0);
    let glow_disk = pow(sun_dot, 512.0) * 2.0;   // disque solaire dur
    let glow_inner = pow(sun_dot, 64.0) * 0.55;  // halo proche
    let glow_outer = pow(sun_dot, 8.0) * 0.18;   // halo lointain (god rays)
    let sun_col = vec3<f32>(1.0, 0.95, 0.78);
    let halo_col = vec3<f32>(1.0, 0.75, 0.40);
    let sun = sun_col * glow_disk + halo_col * glow_inner
            + halo_col * glow_outer * (1.0 - haze_band * 0.6);

    // **God rays radiaux** depuis la direction soleil — bandes claires
    // qui rayonnent dans la haze. On utilise un noise rotatif simple
    // pour briser la régularité (sans noise lib, on hash le sun_dot).
    let radial_phase = sin(sun_dot * 80.0 + camera.time_info.x * 0.4) * 0.5 + 0.5;
    let god_ray_strength = pow(sun_dot, 4.0) * radial_phase * haze_band * 0.25;
    let god_col = vec3<f32>(1.0, 0.85, 0.55);
    let god_rays = god_col * god_ray_strength;

    var sky_col = sky_with_haze + sun + god_rays;

    // **Volumetric clouds** (v0.9.5++ #34) — FBM 2D projeté sur une
    // sphère imaginaire à altitude fixe.  On échantillonne uniquement
    // dans la moitié supérieure du ciel (z > 0.05) pour ne pas mettre
    // de nuages dans l'horizon ou l'océan.  La projection plane sur
    // dir.xy (puis scale) donne l'effet "perspective" : nuages plus
    // serrés près de l'horizon, plus écartés au zénith.
    let cloud_alt = 0.15;
    if dir.z > cloud_alt {
        let proj_factor = cloud_alt / max(dir.z, 0.001);
        let cloud_uv = dir.xy * proj_factor * 4.0;
        let density = cloud_fbm(cloud_uv, camera.time_info.x);
        // Threshold + smooth pour des nuages avec bord nets.
        let cloud_alpha = smoothstep(0.45, 0.78, density);
        // Couleur nuage : blanc en plein soleil, gris foncé sous-côté.
        let cloud_lit = mix(0.6, 1.0, max(dot(dir, sun_dir), 0.0));
        let cloud_color = vec3<f32>(0.95, 0.92, 0.88) * cloud_lit;
        // Atténuation par altitude (cumulus, pas overhead massif).
        let alt_atten = smoothstep(cloud_alt, 0.55, dir.z);
        let final_alpha = cloud_alpha * alt_atten;
        sky_col = mix(sky_col, cloud_color, final_alpha);
    }

    return vec4<f32>(sky_col, 1.0);
}
