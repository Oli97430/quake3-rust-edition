// Shader monde — sample la lightmap depuis un texture_2d_array.
//
// En attendant le shader system complet (textures/.shader), on rend
// uniquement la lightmap × couleur baked, ce qui donne déjà un rendu
// fidèle au « r_lightmap 1 » du jeu original.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    inv_view_proj_rot: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var lightmap_tex: texture_2d_array<f32>;
@group(1) @binding(1) var lightmap_samp: sampler;

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

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_pos = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.normal = in.normal;
    out.tex_uv = in.tex_uv;
    out.lightmap_uv = in.lightmap_uv;
    out.color = in.color;
    out.lightmap_layer = in.lightmap_layer;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let lm = textureSample(lightmap_tex, lightmap_samp, in.lightmap_uv, i32(in.lightmap_layer));
    // couleur baked × lightmap — approximation raisonnable tant qu'on n'a
    // pas le shader system + textures diffuses.
    let col = in.color.rgb * lm.rgb;
    return vec4<f32>(col, 1.0);
}
