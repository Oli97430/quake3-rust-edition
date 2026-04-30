// Flares / coronas : billboards additifs sur les sources lumineuses
// embarquées dans le BSP (lump surfaces, type = Flare).
//
// Le rendu se fait sur des quads caméra-facing construits CPU-side comme
// pour les particules, mais avec blending additif et un masque "corona" :
// cœur très brillant qui s'estompe en un halo doux.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    inv_view_proj_rot: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Distance normalisée au centre du quad, 0 (centre) → 1 (bord).
    let d = length(in.uv - vec2<f32>(0.5, 0.5)) * 2.0;
    // Corona = cœur dense + halo large qui fade quadratiquement.
    // `core`  : petite zone brillante au centre (≈ 20 % du rayon).
    // `halo`  : grand glow doux jusqu'au bord.
    let core = 1.0 - smoothstep(0.0, 0.25, d);
    let halo = pow(1.0 - clamp(d, 0.0, 1.0), 2.0);
    let intensity = core + 0.35 * halo;
    // Blending additif — on boost `color.rgb` par l'intensité et on garde
    // l'alpha à `intensity` aussi, utile si le pipeline passe en alpha
    // (ici on est en additif donc l'alpha est ignoré, mais ça reste propre).
    return vec4<f32>(in.color.rgb * intensity, intensity);
}
