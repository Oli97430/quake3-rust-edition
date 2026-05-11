// HDR10 output — PQ (ST.2084) transfer function + Rec.2020 gamut.
//
// Quand le display supporte HDR10, on bypass l'ACES tonemap SDR et on
// envoie les valeurs HDR lineaires via la courbe PQ.  Le resultat
// preserve la plage dynamique reelle du moteur (muzzle flash vraiment
// brillants, ombres profondes) sans compression [0,1].
//
// Pipeline :
//   hdr_input (Rgba16Float, lineaire) → exposure → bloom add →
//   soft rolloff → Rec.709→Rec.2020 → linear→PQ → Rgba16Float surface
//
// Le surface format doit etre Rgba16Float ou Rgb10a2Unorm pour que le
// display interprete les valeurs PQ correctement.

struct HdrParams {
    exposure: f32,
    peak_nits: f32,
    paper_white: f32,
    bloom_intensity: f32,
};

@group(0) @binding(0) var hdr_input: texture_2d<f32>;
@group(0) @binding(1) var bloom_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> params: HdrParams;

// Full-screen triangle via vertex_index — pas de vertex buffer.
// Index 0 → (-1,-1), Index 1 → (3,-1), Index 2 → (-1,3)
// Couvre tout l'ecran en un seul triangle (plus efficace que 2 triangles).
@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

// --- Rec.709 → Rec.2020 conversion ---
// Matrice de conversion du gamut BT.709 (sRGB) vers BT.2020 (HDR).
fn bt709_to_bt2020(color: vec3<f32>) -> vec3<f32> {
    // Matrice row-major multipliee en column-major WGSL.
    let r = 0.6274 * color.r + 0.3293 * color.g + 0.0433 * color.b;
    let g = 0.0691 * color.r + 0.9195 * color.g + 0.0114 * color.b;
    let b = 0.0164 * color.r + 0.0880 * color.g + 0.8956 * color.b;
    return vec3<f32>(r, g, b);
}

// --- PQ (Perceptual Quantizer) EOTF inverse ---
// ST.2084 : convertit luminance lineaire [0, 10000 nits] en signal PQ [0, 1].
fn linear_to_pq_channel(y: f32) -> f32 {
    let m1: f32 = 0.1593017578125;     // 2610/16384
    let m2: f32 = 78.84375;            // 2523/32 * 128
    let c1: f32 = 0.8359375;           // 3424/4096
    let c2: f32 = 18.8515625;          // 2413/128
    let c3: f32 = 18.6875;             // 2392/128

    let y_norm = clamp(y / 10000.0, 0.0, 1.0);
    let ym1 = pow(y_norm, m1);
    return pow((c1 + c2 * ym1) / (1.0 + c3 * ym1), m2);
}

fn linear_to_pq(color: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        linear_to_pq_channel(color.r),
        linear_to_pq_channel(color.g),
        linear_to_pq_channel(color.b)
    );
}

// --- Soft rolloff ---
// Compression douce pres du peak brightness pour eviter le hard-clipping.
// Courbe Reinhard modifiee : x * (1 + x/peak^2) / (1 + x)
fn soft_rolloff(color: vec3<f32>, peak: f32) -> vec3<f32> {
    let peak2 = peak * peak;
    return color * (vec3(1.0) + color / peak2) / (vec3(1.0) + color);
}

@fragment
fn fs_hdr10(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(hdr_input));
    let uv = frag_coord.xy / dims;

    // Echantillonner scene HDR + bloom
    var color = textureSample(hdr_input, samp, uv).rgb;
    let bloom = textureSample(bloom_tex, samp, uv).rgb;
    color += bloom * params.bloom_intensity;

    // Exposure
    color *= params.exposure;

    // Convertir en nits absolus (1.0 lineaire = paper_white nits)
    color *= params.paper_white;

    // Soft rolloff pres du peak brightness
    color = soft_rolloff(color, params.peak_nits);

    // Rec.709 → Rec.2020 (gamut plus large pour HDR)
    color = bt709_to_bt2020(color);

    // Lineaire → PQ (courbe perceptuelle ST.2084)
    color = linear_to_pq(color);

    return vec4<f32>(color, 1.0);
}
