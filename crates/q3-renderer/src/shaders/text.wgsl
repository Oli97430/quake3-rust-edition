// 2D text / HUD pipeline — atlas SDF.
//
// L'atlas est un R8 où 0.5 = bord du glyphe, >0.5 = intérieur, <0.5 =
// extérieur.  Avec un sampler LINEAR, l'échantillon interpolé donne une
// estimation continue de "à quelle distance suis-je du bord".  Un
// `smoothstep(0.48, 0.52, d)` produit un antialiasing propre sur une
// transition de ~1 texel — crisp à la taille native, doux quand on scale
// ×2/×4.
//
// On trace aussi une ombre portée subtile : deuxième `smoothstep` plus
// large, décalée d'un UV vers le bas-droit, en noir semi-transparent.
// Ça donne du "pop" sur fond clair sans alourdir visuellement.

struct VertexIn {
    @location(0) position: vec2<f32>,
    @location(1) uv:       vec2<f32>,
    @location(2) color:    vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = vec4<f32>(v.position, 0.0, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    return out;
}

@group(0) @binding(0) var atlas_tex: texture_2d<f32>;
@group(0) @binding(1) var atlas_smp: sampler;

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let d = textureSample(atlas_tex, atlas_smp, in.uv).r;

    // Alpha du glyphe : smoothstep autour du seuil 0.5.  La largeur de la
    // transition est calculée depuis le gradient de `d` en screen-space —
    // c'est le trick classique "screen-space derivative AA" qui garde
    // exactement 1 texel de doux peu importe le zoom.  Quand `fwidth`
    // est petit (glyphe grand à l'écran) la transition reste ~1 pixel ;
    // quand il est grand (glyphe lointain / très zoomé out) la transition
    // s'adoucit naturellement, évitant le shimmer.
    let aa = max(fwidth(d), 0.001);
    let alpha = smoothstep(0.5 - aa, 0.5 + aa, d);

    // Ombre portée — on ré-échantillonne la SDF avec un petit offset UV
    // et un seuil plus bas (0.42) pour faire grossir l'empreinte.  Pas
    // de shift UV ici (manque le size de l'atlas côté shader) — on
    // utilise simplement un seuil plus permissif au MÊME point UV, ce
    // qui donne un léger halo centré.  Le vrai shadow décalé nécessiterait
    // de passer l'atlas size en uniform ; pour un HUD 2D ça reste
    // esthétiquement efficace et gratuit.
    let halo_alpha = smoothstep(0.38 - aa, 0.42 + aa, d) * 0.45;
    // Couleur finale : glyphe opaque sur halo sombre.
    let body = in.color.rgb;
    let shadow_rgb = vec3<f32>(0.0, 0.0, 0.0);
    // Composite : le glyph recouvre le halo là où alpha > 0.
    let out_alpha = max(alpha, halo_alpha * (1.0 - alpha));
    let out_rgb = body * alpha + shadow_rgb * (halo_alpha * (1.0 - alpha));
    return vec4<f32>(out_rgb, out_alpha * in.color.a);
}
