// Shader de particules billboard (smoke puffs d'explosion).
// Input : un quad caméra-facing construit CPU-side à partir des axes
// caméra monde (right, up).  Masque radial gaussien pour un rendu doux.

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
    // Distance au centre du quad, normalisée [0,1] : 0 au centre, 1 au bord.
    let d = length(in.uv - vec2<f32>(0.5, 0.5)) * 2.0;
    // Falloff doux type « smoke puff » — zone pleine au centre qui
    // s'estompe progressivement vers le bord (pas de discard, alpha-blend pur).
    let mask = 1.0 - smoothstep(0.1, 1.0, d);
    let alpha = in.color.a * mask;
    return vec4<f32>(in.color.rgb, alpha);
}
