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

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // Z de la direction monde = composante "up" en Q3.
    let dir = world_dir_from_ndc(in.ndc);
    let t = clamp(dir.z * 0.5 + 0.5, 0.0, 1.0);
    let horizon = vec3<f32>(0.72, 0.58, 0.42);
    let zenith  = vec3<f32>(0.10, 0.20, 0.45);
    let sky = mix(horizon, zenith, pow(t, 0.7));
    // Halo soleil fixe (direction arbitraire dans le quadrant NE).
    let sun_dir = normalize(vec3<f32>(0.4, -0.3, 0.6));
    let glow = pow(max(dot(dir, sun_dir), 0.0), 64.0) * 0.6;
    let sun_col = vec3<f32>(1.0, 0.88, 0.55);
    return vec4<f32>(sky + sun_col * glow, 1.0);
}
