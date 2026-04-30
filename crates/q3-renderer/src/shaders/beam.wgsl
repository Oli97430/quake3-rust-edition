// Beam (LineList) pipeline — faisceaux type Lightning Gun / trail Railgun.
//
// Vertices en coordonnées monde Q3 (X=fwd, Y=left, Z=up). La couleur est
// passée par-vertex pour permettre un fade le long du faisceau. Dessiné
// avec blending additif, après les MD3, depth test enabled mais pas de
// write (un beam ne cache pas la géométrie derrière lui).

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    inv_view_proj_rot: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) color:    vec4<f32>,
};

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    out.clip_pos = camera.view_proj * vec4<f32>(v.position, 1.0);
    out.color = v.color;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return in.color;
}
