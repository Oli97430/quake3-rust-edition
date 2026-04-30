// Decals : quads orientés sur une surface, visibles alpha-blend.
//
// Vertices en coordonnées monde — chaque décale est déjà tessellée en 6
// vertices (2 triangles) côté CPU avec la position calculée dans le plan
// tangent à la surface.  Un attribut `uv` 0..1 permet au fragment de
// masquer les bords avec un disque anti-aliasé.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    inv_view_proj_rot: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) uv:       vec2<f32>,
    @location(2) color:    vec4<f32>,
};

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv:    vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    out.clip_pos = camera.view_proj * vec4<f32>(v.position, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // Masque de disque centré en (0.5, 0.5). `d` = distance normalisée au
    // centre (0 au centre, 1 au bord du quad).  On laisse pleine opacité
    // jusqu'à r=0.45, puis fade linéaire jusqu'à 0.5 pour anti-aliaser le
    // bord sans texture.
    let d = length(in.uv - vec2<f32>(0.5, 0.5)) * 2.0;
    let mask = 1.0 - smoothstep(0.9, 1.0, d);
    if mask <= 0.0 {
        discard;
    }
    return vec4<f32>(in.color.rgb, in.color.a * mask);
}
