// Drone / GLB props — pipeline instanced **PBR Cook-Torrance**.
//
// Vertex layout (buffer 0) match `q3_model::glb::GlbVertex` (48 octets) :
//   pos:    vec3 + 1 pad
//   normal: vec3 + 1 pad
//   uv:     vec2 + 2 pad
//
// Buffer 1 = instance data : mat4 model + vec4 tint (80 octets).
//
// Group 1 :
//   binding 0 : baseColor texture
//   binding 1 : sampler partagé (linear, repeat)
//   binding 2 : normalMap (espace tangent — 0,1 plage RG, swizzled)
//   binding 3 : metallicRoughness texture (G=roughness, B=metallic)
//   binding 4 : material factors uniform (vec4 base_color_factor,
//                vec4 metallic_roughness_factors)
//
// Tangent space dérivé via cotangent frame (Mikkelsen) à partir de
// dpdx/dpdy + dudx/dudy — pas besoin de vertex tangents.  Acceptable
// pour des assets sans skinning ; pour du PBR de prod il faudrait
// passer les tangents en vertex attribute, mais c'est suffisant
// pour des props static au look correct.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    inv_view_proj_rot: mat4x4<f32>,
    time_info: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

@group(1) @binding(0) var base_color_tex: texture_2d<f32>;
@group(1) @binding(1) var base_color_samp: sampler;
@group(1) @binding(2) var normal_tex: texture_2d<f32>;
@group(1) @binding(3) var mr_tex: texture_2d<f32>;

struct MaterialFactors {
    base_color: vec4<f32>,
    /// .x = metallicFactor, .y = roughnessFactor, .z = unused, .w = unused
    mr: vec4<f32>,
};
@group(1) @binding(4) var<uniform> material: MaterialFactors;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) m0: vec4<f32>,
    @location(4) m1: vec4<f32>,
    @location(5) m2: vec4<f32>,
    @location(6) m3: vec4<f32>,
    @location(7) tint: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) tint: vec4<f32>,
    @location(3) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let model = mat4x4<f32>(in.m0, in.m1, in.m2, in.m3);
    let world_pos4 = model * vec4<f32>(in.pos, 1.0);
    out.world_pos = world_pos4.xyz;
    out.clip_pos = camera.view_proj * world_pos4;
    let n4 = model * vec4<f32>(in.normal, 0.0);
    out.world_normal = normalize(n4.xyz);
    out.tint = in.tint;
    out.uv = in.uv;
    return out;
}

// **Cotangent frame** (Mikkelsen 2010) — dérive le repère tangent à
// partir de dpdx/dpdy + duvdx/duvdy.  Permet d'appliquer une normalmap
// sans avoir le vertex tangent en attribute.
fn cotangent_frame(n: vec3<f32>, p: vec3<f32>, uv: vec2<f32>) -> mat3x3<f32> {
    let dp1 = dpdx(p);
    let dp2 = dpdy(p);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;
    let invmax = 1.0 / sqrt(max(dot(t, t), dot(b, b)));
    return mat3x3<f32>(t * invmax, b * invmax, n);
}

// **GGX/Trowbridge-Reitz NDF** — distribution des microfacets.
fn ggx_ndf(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom);
}

// **Smith geometry** — masking-shadowing (correlated approximation).
fn smith_g(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = r * r / 8.0;
    let g_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return g_v * g_l;
}

// **Schlick Fresnel** — réflectance vue selon l'angle.
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Sample baseColor.
    let base = textureSample(base_color_tex, base_color_samp, in.uv).rgb
             * material.base_color.rgb
             * in.tint.rgb;

    // Sample normalMap (tangent space) puis transforme en world.
    let n_local = normalize(in.world_normal);
    let n_tangent = textureSample(normal_tex, base_color_samp, in.uv).rgb * 2.0 - 1.0;
    let tbn = cotangent_frame(n_local, in.world_pos, in.uv);
    let n = normalize(tbn * n_tangent);

    // Sample metallicRoughness — convention glTF : G=roughness, B=metallic.
    let mr_sample = textureSample(mr_tex, base_color_samp, in.uv);
    let metallic = mr_sample.b * material.mr.x;
    let roughness = max(mr_sample.g * material.mr.y, 0.04);

    // Direction lumière (soleil tropical) + view.
    let l = normalize(vec3<f32>(-0.4, -0.3, 0.86));
    let v = normalize(camera.view_pos.xyz - in.world_pos);
    let h = normalize(l + v);

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0001);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    // F0 : 0.04 dielectric, baseColor pour metallic.
    let f0_dielectric = vec3<f32>(0.04);
    let f0 = mix(f0_dielectric, base, metallic);

    let d = ggx_ndf(n_dot_h, roughness);
    let g = smith_g(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(v_dot_h, f0);

    // Cook-Torrance specular.
    let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 0.001);

    // Diffuse Lambertian, atténuée par l'energy conservation
    // (kd = 1 - F) × (1 - metallic) — métaux n'ont pas de diffuse.
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * base / 3.14159265;

    // Sun illuminance approximée (lit blanche, intensité 3 stops).
    let sun_color = vec3<f32>(1.0, 0.97, 0.88) * 3.0;

    let direct = (diffuse + specular) * sun_color * n_dot_l;

    // Ambient hémisphère léger.
    let sky_amb = vec3<f32>(0.40, 0.55, 0.70);
    let ground_amb = vec3<f32>(0.18, 0.14, 0.10);
    let hemi = mix(ground_amb, sky_amb, clamp(n.z * 0.5 + 0.5, 0.0, 1.0));
    let ambient = base * hemi * 0.25;

    // Rim léger pour la silhouette.
    let rim = pow(1.0 - n_dot_v, 3.0);
    let rim_col = vec3<f32>(0.55, 0.85, 1.0) * 0.15 * rim;

    let final_color = direct + ambient + rim_col;
    return vec4<f32>(final_color, in.tint.a * material.base_color.a);
}
