// ============================================================================
// FSR — FidelityFX Super Resolution / Enhanced Temporal Upscaling
//
// Trois passes fullscreen :
//   1. EASU  — Edge Adaptive Spatial Upsampling (Lanczos directionnel)
//   2. Temporal Accumulate — blend avec history via motion vectors + YCoCg clamp
//   3. RCAS  — Robust Contrast Adaptive Sharpening
//
// Vertex partagé : fullscreen triangle (vid 0,1,2).
// ============================================================================

// ─── Vertex output commun ────────────────────────────────────────────
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = -1.0 + f32((vid & 1u) << 2u);
    let y = -1.0 + f32((vid & 2u) << 1u);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv  = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

// ─── Bind groups ─────────────────────────────────────────────────────
//
// Group 0 — pass-specific input texture + sampler + uniforms
@group(0) @binding(0) var input_tex:  texture_2d<f32>;
@group(0) @binding(1) var samp:       sampler;

// Uniform partagé par les trois passes. Chaque passe lit les champs
// qui la concernent ; les autres sont ignorés sans coût.
struct FsrParams {
    // EASU : taille source (interne) et destination (native)
    src_size:    vec2<f32>,  // largeur, hauteur de la texture source
    dst_size:    vec2<f32>,  // largeur, hauteur de la texture destination
    // RCAS : intensité du sharpening [0..1]
    sharpness:   f32,
    // Temporal : blend factor (fraction du current frame, ex. 0.10)
    blend_alpha: f32,
    _pad:        vec2<f32>,
};
@group(0) @binding(2) var<uniform> params: FsrParams;

// Group 1 — uniquement pour la passe temporal (history + motion vectors)
@group(1) @binding(0) var history_tex:  texture_2d<f32>;
@group(1) @binding(1) var motion_tex:   texture_2d<f32>;


// =====================================================================
// Utilitaires
// =====================================================================

// Luminance perceptive ITU BT.709.
fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Conversion RGB <-> YCoCg pour le neighborhood clamping temporal.
// YCoCg sépare la luminance de la chrominance — le clamping dans cet
// espace produit moins de désaturation que dans RGB direct.
fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    let y  = dot(rgb, vec3<f32>( 0.25, 0.50,  0.25));
    let co = dot(rgb, vec3<f32>( 0.50, 0.00, -0.50));
    let cg = dot(rgb, vec3<f32>(-0.25, 0.50, -0.25));
    return vec3<f32>(y, co, cg);
}

fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    let y  = ycocg.x;
    let co = ycocg.y;
    let cg = ycocg.z;
    let r = y + co - cg;
    let g = y      + cg;
    let b = y - co - cg;
    return vec3<f32>(r, g, b);
}

// Lanczos2 kernel — fenêtre sinc × sinc(x/2). Approximation rapide
// via fit polynomiale pour éviter les sin() coûteux sur mobile/Intel.
fn lanczos2(x: f32) -> f32 {
    let ax = abs(x);
    if (ax >= 2.0) { return 0.0; }
    if (ax < 1.0e-5) { return 1.0; }
    // Fit quadratique : w = (2/pi)^2 * sin(pi*x)*sin(pi*x/2) / (x^2)
    // Approximation : 1 - ax^2 * (pi^2/4 - pi^2/16 * ax^2) ... simplifié :
    let ax2 = ax * ax;
    // Exact Lanczos via trig — on accepte le coût sur desktop GPU.
    let pi_x = 3.14159265 * ax;
    let pi_x_half = pi_x * 0.5;
    return (sin(pi_x) / pi_x) * (sin(pi_x_half) / pi_x_half);
}


// =====================================================================
// Passe 1 — EASU : Edge Adaptive Spatial Upsampling
// =====================================================================
//
// Algorithme inspiré de AMD FSR 1.0 EASU :
//   - Echantillonne 12 voisins dans un pattern en croix autour du texel
//     source correspondant au pixel destination.
//   - Calcule le gradient de luminance dans 4 directions (H, V, D1, D2)
//     pour détecter l'orientation des contours.
//   - Applique un filtre Lanczos2 directionnel le long du contour
//     détecté — lisse parallèlement au bord, préserve perpendiculairement.
//   - Résultat : upscale net sur les contours, pas de ringing visible.
//
@fragment
fn fs_easu(in: VsOut) -> @location(0) vec4<f32> {
    let src_texel = 1.0 / params.src_size;
    // Position en coordonnées source (sub-texel) correspondant au pixel
    // destination courant.  `in.uv` est en [0,1] sur la destination.
    let src_uv = in.uv;

    // --- Echantillonnage 12 voisins (pattern en croix 5x5 sans coins) ---
    //
    //     .  N  .
    //    NW NC NE
    //   W  CW C CE E
    //    SW SC SE
    //     .  S  .
    //
    // On sample les 4 cardinaux proches, 4 diagonaux, et 4 cardinaux lointains.
    let c  = textureSample(input_tex, samp, src_uv).rgb;
    let n  = textureSample(input_tex, samp, src_uv + vec2<f32>( 0.0, -1.0) * src_texel).rgb;
    let s  = textureSample(input_tex, samp, src_uv + vec2<f32>( 0.0,  1.0) * src_texel).rgb;
    let e  = textureSample(input_tex, samp, src_uv + vec2<f32>( 1.0,  0.0) * src_texel).rgb;
    let w  = textureSample(input_tex, samp, src_uv + vec2<f32>(-1.0,  0.0) * src_texel).rgb;
    let ne = textureSample(input_tex, samp, src_uv + vec2<f32>( 1.0, -1.0) * src_texel).rgb;
    let nw = textureSample(input_tex, samp, src_uv + vec2<f32>(-1.0, -1.0) * src_texel).rgb;
    let se = textureSample(input_tex, samp, src_uv + vec2<f32>( 1.0,  1.0) * src_texel).rgb;
    let sw = textureSample(input_tex, samp, src_uv + vec2<f32>(-1.0,  1.0) * src_texel).rgb;
    // Cardinaux lointains (2 texels de distance) pour un gradient fiable.
    let n2 = textureSample(input_tex, samp, src_uv + vec2<f32>( 0.0, -2.0) * src_texel).rgb;
    let s2 = textureSample(input_tex, samp, src_uv + vec2<f32>( 0.0,  2.0) * src_texel).rgb;
    let e2 = textureSample(input_tex, samp, src_uv + vec2<f32>( 2.0,  0.0) * src_texel).rgb;
    let w2 = textureSample(input_tex, samp, src_uv + vec2<f32>(-2.0,  0.0) * src_texel).rgb;

    // --- Détection de bords via gradients de luminance ---
    let lc  = luminance(c);
    let ln  = luminance(n);
    let ls  = luminance(s);
    let le  = luminance(e);
    let lw  = luminance(w);
    let lne = luminance(ne);
    let lnw = luminance(nw);
    let lse = luminance(se);
    let lsw = luminance(sw);

    // Gradients dans les 4 axes (Sobel-like)
    let grad_h = abs(lw  - le)  + abs(lnw - lne) * 0.5 + abs(lsw - lse) * 0.5;
    let grad_v = abs(ln  - ls)  + abs(lnw - lsw) * 0.5 + abs(lne - lse) * 0.5;
    let grad_d1 = abs(lnw - lse) + abs(lw - ls) * 0.5 + abs(ln - le) * 0.5; // diag /
    let grad_d2 = abs(lne - lsw) + abs(le - ls) * 0.5 + abs(ln - lw) * 0.5; // diag backslash

    // Direction du bord : perpendiculaire au gradient dominant.
    // On combine les gradients en un vecteur directionnel.
    let edge_h = grad_v;  // bord horizontal = gradient vertical fort
    let edge_v = grad_h;  // bord vertical = gradient horizontal fort

    // Normalisation douce de la direction du bord — évite la division
    // par zéro dans les zones uniformes (pas de bord → isotrope).
    let edge_len = max(sqrt(edge_h * edge_h + edge_v * edge_v), 1.0e-6);
    let dir = vec2<f32>(edge_h, edge_v) / edge_len;

    // --- Filtre Lanczos2 directionnel ---
    // On reconstruit le pixel en pondérant les voisins par un kernel
    // Lanczos2 étiré le long de la direction du bord détecté.
    // `stretch` contrôle l'élongation : plus le gradient est fort,
    // plus le filtre s'étire le long du bord pour lisser le jagged.
    let edge_strength = clamp(edge_len * 4.0, 0.0, 1.0);

    // Offset sub-texel du pixel destination dans la grille source.
    let src_pos = src_uv * params.src_size - 0.5;
    let src_floor = floor(src_pos);
    let frac = src_pos - src_floor;

    // On évalue le kernel sur les 12 samples + le centre.
    // Pour chaque sample, on calcule sa distance projetée sur la
    // direction du bord (dimension parallèle) et perpendiculaire.
    // Le kernel est un Lanczos2 anisotrope : parallèle au bord
    // = fenêtre large (lisse), perpendiculaire = fenêtre étroite
    // (préserve le contraste).
    var total_weight = 0.0;
    var total_color  = vec3<f32>(0.0);

    // Macro inline : on traite chaque sample individuellement.
    // L'offset est en texels source par rapport au centre `frac`.
    let stretch = mix(1.0, 2.0, edge_strength);

    // Offsets des 13 samples par rapport au texel centre (0,0)
    var offsets: array<vec2<f32>, 13>;
    offsets[0]  = vec2<f32>( 0.0,  0.0); // centre
    offsets[1]  = vec2<f32>( 0.0, -1.0); // N
    offsets[2]  = vec2<f32>( 0.0,  1.0); // S
    offsets[3]  = vec2<f32>( 1.0,  0.0); // E
    offsets[4]  = vec2<f32>(-1.0,  0.0); // W
    offsets[5]  = vec2<f32>( 1.0, -1.0); // NE
    offsets[6]  = vec2<f32>(-1.0, -1.0); // NW
    offsets[7]  = vec2<f32>( 1.0,  1.0); // SE
    offsets[8]  = vec2<f32>(-1.0,  1.0); // SW
    offsets[9]  = vec2<f32>( 0.0, -2.0); // N2
    offsets[10] = vec2<f32>( 0.0,  2.0); // S2
    offsets[11] = vec2<f32>( 2.0,  0.0); // E2
    offsets[12] = vec2<f32>(-2.0,  0.0); // W2

    var colors: array<vec3<f32>, 13>;
    colors[0]  = c;
    colors[1]  = n;
    colors[2]  = s;
    colors[3]  = e;
    colors[4]  = w;
    colors[5]  = ne;
    colors[6]  = nw;
    colors[7]  = se;
    colors[8]  = sw;
    colors[9]  = n2;
    colors[10] = s2;
    colors[11] = e2;
    colors[12] = w2;

    for (var i = 0; i < 13; i = i + 1) {
        // Distance du sample au point d'interpolation en espace texel.
        let delta = offsets[i] - frac;
        // Projections parallèle et perpendiculaire au bord.
        let para = dot(delta, dir);
        let perp = length(delta - dir * para);
        // Kernel anisotrope : Lanczos2 avec fenêtre étirée parallèlement
        // au bord pour lisser les jaggies, et étroite perpendiculairement
        // pour conserver le contraste du contour.
        let w_para = lanczos2(para / stretch);
        let w_perp = lanczos2(perp);
        let weight = w_para * w_perp;
        total_weight = total_weight + weight;
        total_color  = total_color + colors[i] * weight;
    }

    // Fallback : dans les zones totalement plates (poids ≈ 0) on retourne
    // le sample bilinéaire du centre pour éviter les NaN.
    if (total_weight < 1.0e-6) {
        return vec4<f32>(c, 1.0);
    }

    let result = total_color / total_weight;
    return vec4<f32>(max(result, vec3<f32>(0.0)), 1.0);
}


// =====================================================================
// Passe 2 — Temporal Accumulate (blend avec history + motion vectors)
// =====================================================================
//
// Algorithme :
//   1. Lire le pixel courant (déjà upscalé par EASU, à résolution native).
//   2. Lire les motion vectors pour déterminer la position correspondante
//      dans la frame précédente (reprojection).
//   3. Sampler l'history texture à la position reprojetée.
//   4. Neighborhood clamp en YCoCg sur un kernel 3x3 du current pour
//      contraindre l'history dans un range plausible → anti-ghosting.
//   5. Blend exponentiel : ~90% history + ~10% current. En cas de
//      disocclusion (UV hors écran, ou clamp trop violent), on augmente
//      la contribution du current.
//
@fragment
fn fs_temporal(in: VsOut) -> @location(0) vec4<f32> {
    let dims = textureDimensions(input_tex);
    let texel = vec2<f32>(1.0 / f32(dims.x), 1.0 / f32(dims.y));

    // Pixel courant (sortie EASU).
    let current_rgb = textureSample(input_tex, samp, in.uv).rgb;
    let current_ycocg = rgb_to_ycocg(current_rgb);

    // --- Motion vector (RG = delta UV en screen space) ---
    // Convention : motion_tex stocke le déplacement (uv_prev - uv_curr)
    // c.-à-d. la direction vers l'ancienne position. Si aucun motion
    // vector n'est disponible (texture noire), la reprojection tombe
    // sur la même position = TAA sans mouvement, identique à l'existant.
    let mv = textureSample(motion_tex, samp, in.uv).rg;
    let history_uv = in.uv + mv;

    // --- Neighborhood clamp 3x3 en YCoCg ---
    var ycocg_min = current_ycocg;
    var ycocg_max = current_ycocg;
    // Accumulation moments pour variance clipping (Karis 2014 amélioré).
    var moment1 = current_ycocg;
    var moment2 = current_ycocg * current_ycocg;
    var sample_count = 1.0;

    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) { continue; }
            let off = vec2<f32>(f32(dx), f32(dy)) * texel;
            let neighbor = textureSample(input_tex, samp, in.uv + off).rgb;
            let n_ycocg = rgb_to_ycocg(neighbor);
            ycocg_min = min(ycocg_min, n_ycocg);
            ycocg_max = max(ycocg_max, n_ycocg);
            moment1 = moment1 + n_ycocg;
            moment2 = moment2 + n_ycocg * n_ycocg;
            sample_count = sample_count + 1.0;
        }
    }

    // Variance clipping : box centrée sur la moyenne ± gamma × stddev.
    // Plus robuste que le simple min/max : tolère les outliers légers
    // (particles, muzzle flash) sans rejeter tout l'history.
    let mu    = moment1 / sample_count;
    let sigma = sqrt(max(moment2 / sample_count - mu * mu, vec3<f32>(0.0)));
    let gamma = 1.25; // facteur de tolérance
    let var_min = mu - sigma * gamma;
    let var_max = mu + sigma * gamma;
    // Intersection min/max AABB et variance box pour un clamp hybride.
    let clip_min = max(ycocg_min, var_min);
    let clip_max = min(ycocg_max, var_max);

    // --- Sample history à la position reprojetée ---
    var history_rgb = textureSample(history_tex, samp, history_uv).rgb;
    var history_ycocg = rgb_to_ycocg(history_rgb);

    // Clamp history dans la box du voisinage courant.
    let clamped_ycocg = clamp(history_ycocg, clip_min, clip_max);

    // Mesure de disocclusion : si le clamp a beaucoup bougé l'history,
    // ou si les UV reprojetés sortent de l'écran, on favorise le current.
    let clamp_dist = length(clamped_ycocg - history_ycocg);
    let uv_valid = step(0.0, history_uv.x) * step(history_uv.x, 1.0)
                 * step(0.0, history_uv.y) * step(history_uv.y, 1.0);
    // Plus le clamp est violent, plus on donne de poids au current.
    let disocclusion = clamp(clamp_dist * 8.0, 0.0, 1.0);
    let effective_alpha = mix(params.blend_alpha, 1.0, max(disocclusion, 1.0 - uv_valid));

    let clamped_rgb = ycocg_to_rgb(clamped_ycocg);

    // Blend exponentiel : history dominant pour accumuler, current
    // pour rafraîchir les zones en mouvement / disocclusion.
    let blended = mix(clamped_rgb, current_rgb, effective_alpha);

    return vec4<f32>(max(blended, vec3<f32>(0.0)), 1.0);
}


// =====================================================================
// Passe 3 — RCAS : Robust Contrast Adaptive Sharpening
// =====================================================================
//
// Inspiré de AMD FidelityFX CAS / FSR RCAS :
//   - Pattern 5-tap en croix (centre + 4 cardinaux).
//   - Calcule le contraste local (max - min luminance).
//   - Applique un lobe négatif (unsharp mask) proportionnel au contraste
//     et inversement proportionnel au bruit local.
//   - Résultat : les contours nets sont renforcés, le bruit et les
//     surfaces lisses ne sont pas amplifiés.
//   - `sharpness` 0.0 = pass-through, 1.0 = sharpening maximal.
//
@fragment
fn fs_rcas(in: VsOut) -> @location(0) vec4<f32> {
    let dims = textureDimensions(input_tex);
    let texel = vec2<f32>(1.0 / f32(dims.x), 1.0 / f32(dims.y));

    // 5-tap cross : centre + 4 voisins cardinaux.
    let c = textureSample(input_tex, samp, in.uv).rgb;
    let n = textureSample(input_tex, samp, in.uv + vec2<f32>( 0.0, -1.0) * texel).rgb;
    let s = textureSample(input_tex, samp, in.uv + vec2<f32>( 0.0,  1.0) * texel).rgb;
    let e = textureSample(input_tex, samp, in.uv + vec2<f32>( 1.0,  0.0) * texel).rgb;
    let w = textureSample(input_tex, samp, in.uv + vec2<f32>(-1.0,  0.0) * texel).rgb;

    // Luminances pour le calcul de contraste.
    let lc = luminance(c);
    let ln = luminance(n);
    let ls = luminance(s);
    let le = luminance(e);
    let lw = luminance(w);

    // Contraste local = écart entre la luminance min et max des 5 taps.
    let l_min = min(lc, min(min(ln, ls), min(le, lw)));
    let l_max = max(lc, max(max(ln, ls), max(le, lw)));

    // Facteur de sharpening adaptatif : plus le contraste est fort
    // (vrai contour), plus on sharpen. Dans les zones plates ou
    // bruitées (contraste faible), le facteur est quasi nul.
    // La formule RCAS originale : w = -1 / (4 * max_neg_lobe + 1)
    // avec max_neg_lobe = min(contraste / max_lum, k).
    //
    // Simplification : on utilise le ratio min/max comme proxy de la
    // confiance qu'il s'agit d'un vrai edge et pas de bruit.
    let range = l_max - l_min;
    let peak = l_max;
    // Évite div/0 dans les zones totalement noires.
    let contrast_ratio = range / max(peak, 0.05);

    // Poids du lobe négatif : proportionnel au contraste, plafonné.
    // Plus sharpness est haut, plus le lobe est agressif.
    let max_neg = -0.125 * params.sharpness; // AMD recommande -1/8 max
    let neg_weight = max_neg * clamp(contrast_ratio, 0.0, 1.0);

    // Appliquer le filtre unsharp : centre renforcé, voisins soustraits.
    // Kernel = [0, neg_w, 0; neg_w, 1+4*|neg_w|, neg_w; 0, neg_w, 0]
    // Normalisation : somme des poids = 1 (pour préserver l'énergie).
    let pos_weight = 1.0 - 4.0 * neg_weight; // > 1 quand neg_weight < 0
    let sharpened = c * pos_weight + (n + s + e + w) * neg_weight;

    // Clamp pour éviter les valeurs négatives dues au lobe (ringing).
    // On ne clamp pas au-dessus pour préserver les highlights HDR.
    let result = max(sharpened, vec3<f32>(0.0));

    return vec4<f32>(result, 1.0);
}
