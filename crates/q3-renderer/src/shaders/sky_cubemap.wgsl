// Skybox cubemap — plein écran avec z=1.0, depth LessEqual.
//
// Reconstruit la direction monde par pixel via inv(proj × view_rot_only),
// puis convertit Q3 (X=fwd, Y=left, Z=up) vers WGPU cube (X=right, Y=up,
// Z=fwd) avant de sampler la texture cubique.
//
// Bind groups :
//   0 : camera uniform
//   1 : cubemap (texture_cube + sampler)

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    inv_view_proj_rot: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var sky_tex:  texture_cube<f32>;
@group(1) @binding(1) var sky_samp: sampler;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[idx];
    var out: VOut;
    out.clip_pos = vec4<f32>(p, 1.0, 1.0);
    out.ndc = p;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // Reconstruit la direction monde au pixel (coordonnées Q3).
    let clip = vec4<f32>(in.ndc, 1.0, 1.0);
    let w = camera.inv_view_proj_rot * clip;
    let dir_q3 = normalize(w.xyz / max(w.w, 1e-6));
    // Convertit Q3 (X=fwd, Y=left, Z=up) → WGPU cube (X=right, Y=up, Z=fwd).
    //   cube.x = -q3.y  (right = -left)
    //   cube.y =  q3.z  (up    =  up)
    //   cube.z =  q3.x  (fwd   =  fwd)
    let dir = vec3<f32>(-dir_q3.y, dir_q3.z, dir_q3.x);
    return textureSample(sky_tex, sky_samp, dir);
}
