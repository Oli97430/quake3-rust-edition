// ─── Screen-Space Raytracing (SSR++) — réflexions + ombres de contact ───
//
// Compute shader dispatché en workgroups 8x8.  Trois passes :
//   1. generate_hi_z   — construit la pyramide de profondeur (min-depth
//      par mip) pour accélérer le raymarch hiérarchique.
//   2. trace_reflections — SSR hi-z avec 64 steps coarse + 8 binary
//      refinements, jitter temporel (blue noise), Fresnel, cone tracing
//      pour roughness, fade aux bords d'écran.
//   3. trace_contact_shadows — 16 steps vers le soleil, pénombre douce
//      basée sur la distance à l'occluder.
//
// Toutes les coordonnées "screen" sont en pixels half-res (la texture
// de sortie est à demi-résolution pour la perf).

// ─── Uniforms ────────────────────────────────────────────────────────
struct RtParams {
    // Matrice inverse de view_proj — reconstruit world pos depuis depth.
    inv_view_proj: mat4x4<f32>,
    // Matrice view_proj — projette world pos en clip space.
    view_proj: mat4x4<f32>,
    // Position caméra monde (xyz), w = time (secondes).
    camera_pos: vec4<f32>,
    // Direction du soleil normalisée (xyz), w = frame_index (u32 cast f32).
    sun_dir: vec4<f32>,
    // Dimensions texture pleine résolution (xy), half-res (zw).
    resolution: vec4<f32>,
    // Near/far plan (xy), max shadow distance (z), roughness multiplier (w).
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> rt: RtParams;

// ─── Textures d'entrée ───────────────────────────────────────────────
// Depth buffer pleine résolution (Depth32Float, lu comme float).
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
// Couleur scene HDR (Rgba16Float, pleine résolution).
@group(0) @binding(2) var color_tex: texture_2d<f32>;
// Blue noise 128x128 (Rg8Unorm) — jitter temporel.
@group(0) @binding(3) var noise_tex: texture_2d<f32>;

// ─── Hi-Z pyramid (lecture) ──────────────────────────────────────────
// Mip 0 = pleine résolution, mips 1..N = chaque niveau /2.
// Chaque texel = min(4 parents) — conservative pour le raymarching.
@group(0) @binding(4) var hiz_tex: texture_2d<f32>;

// ─── Textures de sortie (storage, half-res) ──────────────────────────
@group(0) @binding(5) var reflection_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var shadow_out: texture_storage_2d<r8unorm, write>;

// ─── Hi-Z generation (écriture dans le mip courant) ──────────────────
// Bind group séparé pour la passe hi-z (un mip source + un mip dest).
@group(1) @binding(0) var hiz_src: texture_2d<f32>;
@group(1) @binding(1) var hiz_dst: texture_storage_2d<r32float, write>;


// ═══════════════════════════════════════════════════════════════════════
// Utilitaires
// ═══════════════════════════════════════════════════════════════════════

// Reconstruit la position monde depuis les coordonnées pixel et la depth.
fn reconstruct_world_pos(pixel: vec2<f32>, depth: f32, inv_vp: mat4x4<f32>, full_res: vec2<f32>) -> vec3<f32> {
    // UV [0..1] → NDC [-1..1] avec Y-flip (convention wgpu).
    let uv = pixel / full_res;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0), depth, 1.0);
    let world_h = inv_vp * ndc;
    return world_h.xyz / world_h.w;
}

// Projette une position monde en screen-space UV [0..1].
// Retourne (uv.x, uv.y, ndc.z, clip.w).  clip.w <= 0 = derrière la caméra.
fn project_to_screen(world_pos: vec3<f32>, vp: mat4x4<f32>) -> vec4<f32> {
    let clip = vp * vec4<f32>(world_pos, 1.0);
    if (clip.w <= 0.0) {
        return vec4<f32>(-1.0, -1.0, -1.0, clip.w);
    }
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    return vec4<f32>(uv, ndc.z, clip.w);
}

// Dérive la normale depuis le depth buffer (cross des gradients voisins).
// Approche rapide : échantillonne 3 points voisins, calcule le cross product.
fn derive_normal_from_depth(pixel: vec2<i32>, full_res: vec2<f32>) -> vec3<f32> {
    let d_c = textureLoad(depth_tex, pixel, 0).r;
    let d_r = textureLoad(depth_tex, pixel + vec2<i32>(1, 0), 0).r;
    let d_d = textureLoad(depth_tex, pixel + vec2<i32>(0, 1), 0).r;

    let p_c = reconstruct_world_pos(vec2<f32>(pixel) + vec2<f32>(0.5), d_c, rt.inv_view_proj, full_res);
    let p_r = reconstruct_world_pos(vec2<f32>(pixel) + vec2<f32>(1.5, 0.5), d_r, rt.inv_view_proj, full_res);
    let p_d = reconstruct_world_pos(vec2<f32>(pixel) + vec2<f32>(0.5, 1.5), d_d, rt.inv_view_proj, full_res);

    return normalize(cross(p_r - p_c, p_d - p_c));
}

// Fresnel-Schlick : F0 = 0.04 (diélectrique standard).
fn fresnel_schlick(n_dot_v: f32) -> f32 {
    let f0 = 0.04;
    return f0 + (1.0 - f0) * pow(clamp(1.0 - n_dot_v, 0.0, 1.0), 5.0);
}

// Fade aux bords d'écran — évite les artefacts de coupure dure.
fn screen_edge_fade(uv: vec2<f32>) -> f32 {
    let edge_x = min(uv.x, 1.0 - uv.x);
    let edge_y = min(uv.y, 1.0 - uv.y);
    let edge = min(edge_x, edge_y);
    return smoothstep(0.0, 0.08, edge);
}

// Lecture du blue noise pour jitter.  Tile 128x128 avec offset temporel.
fn blue_noise_sample(pixel: vec2<u32>) -> vec2<f32> {
    let frame = u32(rt.sun_dir.w);
    let coord = vec2<i32>((pixel + vec2<u32>(frame * 7u, frame * 11u)) % vec2<u32>(128u));
    return textureLoad(noise_tex, coord, 0).rg;
}


// ═══════════════════════════════════════════════════════════════════════
// Passe 1 : Hi-Z mipmap generation
// ═══════════════════════════════════════════════════════════════════════
//
// Chaque texel du mip destination = min des 4 texels parents.
// Dispatché par mip successivement (mip 1 depuis mip 0, etc.).
// Le min conservatif garantit qu'un ray ne peut pas "traverser" une
// surface sans la détecter au niveau coarse.

@compute @workgroup_size(8, 8)
fn generate_hi_z(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(hiz_dst);
    if (gid.x >= dst_size.x || gid.y >= dst_size.y) {
        return;
    }

    // Coordonnées du bloc 2x2 dans le mip source.
    let src_base = vec2<i32>(gid.xy) * 2;
    let s00 = textureLoad(hiz_src, src_base + vec2<i32>(0, 0), 0).r;
    let s10 = textureLoad(hiz_src, src_base + vec2<i32>(1, 0), 0).r;
    let s01 = textureLoad(hiz_src, src_base + vec2<i32>(0, 1), 0).r;
    let s11 = textureLoad(hiz_src, src_base + vec2<i32>(1, 1), 0).r;

    // Min = plus proche de la caméra (en reversed-Z, max serait correct ;
    // en depth standard wgpu 0..1 near..far, min = plus proche).
    let min_depth = min(min(s00, s10), min(s01, s11));

    textureStore(hiz_dst, vec2<i32>(gid.xy), vec4<f32>(min_depth, 0.0, 0.0, 1.0));
}


// ═══════════════════════════════════════════════════════════════════════
// Passe 2 : SSR — réflexions screen-space (Hi-Z accelerated)
// ═══════════════════════════════════════════════════════════════════════
//
// Algorithme :
//   1. Reconstruit la position monde et la normale depuis le depth buffer
//   2. Calcule la direction de réflexion (reflect -view_dir sur normal)
//   3. Projette origin + direction en screen-space
//   4. Raymarch hiérarchique : 64 coarse steps sur la pyramide hi-z
//      (saute les zones vides aux mips grossiers, raffine aux mips fins)
//   5. 8 steps de raffinement binaire sur le dernier intervalle trouvé
//   6. Sample la couleur scene à l'UV de hit
//   7. Pondère par Fresnel × edge_fade × confidence
//
// La sortie est half-res (dispatch = half_width/8 × half_height/8).

// Nombre max de mip levels pour le hi-z traversal.
const HIZ_MAX_LEVEL: i32 = 6;

@compute @workgroup_size(8, 8)
fn trace_reflections(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_res = vec2<u32>(rt.resolution.zw);
    if (gid.x >= half_res.x || gid.y >= half_res.y) {
        return;
    }

    let full_res = rt.resolution.xy;
    // Coordonnées pleine résolution correspondantes (centre du pixel half-res).
    let full_pixel = vec2<f32>(gid.xy) * 2.0 + vec2<f32>(1.0);
    let full_pixel_i = vec2<i32>(full_pixel);

    // Lecture depth — skip le sky (depth >= 0.999).
    let depth = textureLoad(depth_tex, full_pixel_i, 0).r;
    if (depth >= 0.999) {
        textureStore(reflection_out, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    // Reconstruit position monde et normale.
    let world_pos = reconstruct_world_pos(full_pixel, depth, rt.inv_view_proj, full_res);
    let normal = derive_normal_from_depth(full_pixel_i, full_res);

    // Direction de vue et direction de réflexion.
    let view_dir = normalize(world_pos - rt.camera_pos.xyz);
    let refl_dir = reflect(view_dir, normal);

    // Fresnel — pondère le blend final.
    let n_dot_v = max(dot(normal, -view_dir), 0.0);
    let fresnel = fresnel_schlick(n_dot_v);

    // Si la réflexion pointe vers la caméra (derrière la surface visible),
    // le SSR ne peut rien trouver → bail early.
    let refl_proj = project_to_screen(world_pos + refl_dir * 10.0, rt.view_proj);
    if (refl_proj.w <= 0.0) {
        textureStore(reflection_out, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    // ─── Jitter temporel (blue noise) pour anti-aliasing temporel ────
    let noise = blue_noise_sample(gid.xy);
    let jitter_offset = noise.x * 0.5; // demi-step de jitter

    // ─── Projection du ray en screen space ───────────────────────────
    let origin_screen = project_to_screen(world_pos + normal * 0.5, rt.view_proj);
    // Point loin sur le ray pour définir la direction screen.
    let far_point = world_pos + refl_dir * rt.params.y; // far plane distance
    let far_screen = project_to_screen(far_point, rt.view_proj);

    // Direction en UV-space.
    let origin_uv = origin_screen.xy;
    let dir_uv = far_screen.xy - origin_uv;
    let dir_len = length(dir_uv * full_res);
    if (dir_len < 1.0) {
        textureStore(reflection_out, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    // Normaliser le pas pour ~1 pixel par step au mip 0.
    let step_uv = dir_uv / dir_len;
    // Depth (NDC Z) interpolé linéairement le long du ray.
    let origin_z = origin_screen.z;
    let far_z = far_screen.z;
    let step_z = (far_z - origin_z) / dir_len;

    // ─── Coarse ray march (64 steps, hi-z accelerated) ───────────────
    let max_steps = 64;
    var ray_uv = origin_uv + step_uv * jitter_offset;
    var ray_z = origin_z + step_z * jitter_offset;
    var stride = 2.0; // Commence à 2 pixels pour couvrir plus de distance.
    var hit = false;
    var hit_uv = vec2<f32>(0.0);
    var hit_z = 0.0;
    var prev_ray_uv = ray_uv;
    var prev_ray_z = ray_z;

    for (var i = 0; i < 64; i = i + 1) {
        prev_ray_uv = ray_uv;
        prev_ray_z = ray_z;

        ray_uv = ray_uv + step_uv * stride;
        ray_z = ray_z + step_z * stride;

        // Hors écran → abandon.
        if (ray_uv.x < 0.0 || ray_uv.x > 1.0 || ray_uv.y < 0.0 || ray_uv.y > 1.0) {
            break;
        }

        // Sample hi-z au mip correspondant au stride actuel.
        // mip = log2(stride) clampé à [0, HIZ_MAX_LEVEL].
        let mip_level = clamp(i32(log2(stride)), 0, HIZ_MAX_LEVEL);
        let sample_pixel = vec2<i32>(ray_uv * full_res) >> vec2<u32>(u32(mip_level));
        let scene_z = textureLoad(hiz_tex, sample_pixel, mip_level).r;

        // Hit test : le ray a dépassé la surface du depth buffer.
        if (ray_z > scene_z) {
            // Vérifier que l'épaisseur n'est pas trop grande (discontinuité
            // de profondeur = on a "sauté" derrière un objet → faux positif).
            let thickness = ray_z - scene_z;
            if (thickness < 0.05) {
                hit = true;
                hit_uv = ray_uv;
                hit_z = ray_z;
                break;
            }
            // Épaisseur trop grande : on réduit le stride et on revient
            // pour affiner (descend d'un mip level implicitement).
            stride = max(stride * 0.5, 1.0);
            ray_uv = prev_ray_uv;
            ray_z = prev_ray_z;
        } else {
            // Pas de hit : on peut agrandir le stride pour aller plus vite.
            stride = min(stride * 1.2, 32.0);
        }
    }

    // ─── Raffinement binaire (8 steps) ───────────────────────────────
    // Converge sur le point de contact exact entre le dernier "miss" et
    // le premier "hit".
    if (hit) {
        var lo_uv = prev_ray_uv;
        var lo_z = prev_ray_z;
        var hi_uv = hit_uv;
        var hi_z = hit_z;

        for (var b = 0; b < 8; b = b + 1) {
            let mid_uv = (lo_uv + hi_uv) * 0.5;
            let mid_z = (lo_z + hi_z) * 0.5;
            let mid_pixel = vec2<i32>(mid_uv * full_res);
            let scene_z = textureLoad(depth_tex, mid_pixel, 0).r;
            if (mid_z > scene_z) {
                hi_uv = mid_uv;
                hi_z = mid_z;
            } else {
                lo_uv = mid_uv;
                lo_z = mid_z;
            }
        }
        hit_uv = (lo_uv + hi_uv) * 0.5;
    }

    // ─── Écriture du résultat ────────────────────────────────────────
    if (!hit) {
        textureStore(reflection_out, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    // Clamp final des UV de hit.
    let clamped_uv = clamp(hit_uv, vec2<f32>(0.001), vec2<f32>(0.999));
    let hit_pixel = vec2<i32>(clamped_uv * full_res);
    let refl_color = textureLoad(color_tex, hit_pixel, 0).rgb;

    // Pondération finale : Fresnel × edge fade × roughness cone (TODO).
    let edge_f = screen_edge_fade(clamped_uv);
    // Confiance basée sur l'angle de la réflexion par rapport à la surface.
    let confidence = clamp(dot(refl_dir, normal), 0.0, 1.0);
    let weight = fresnel * edge_f * (0.5 + 0.5 * confidence);

    textureStore(reflection_out, vec2<i32>(gid.xy), vec4<f32>(refl_color * weight, weight));
}


// ═══════════════════════════════════════════════════════════════════════
// Passe 3 : Contact shadows (ombres de contact screen-space)
// ═══════════════════════════════════════════════════════════════════════
//
// Pour chaque pixel, on marche en screen-space dans la direction du soleil.
// Si un sample occulte le ray (depth < ray_depth), on est dans l'ombre.
// La pénombre est douce : plus l'occluder est loin, plus l'ombre est claire
// (penumbra angle estimé).
//
// 16 steps, 200 unités Q3 max distance.  Résultat en R8Unorm :
//   0.0 = pleine ombre de contact
//   1.0 = aucune occlusion

const SHADOW_STEPS: i32 = 16;
const SHADOW_MAX_DIST: f32 = 200.0;

@compute @workgroup_size(8, 8)
fn trace_contact_shadows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_res = vec2<u32>(rt.resolution.zw);
    if (gid.x >= half_res.x || gid.y >= half_res.y) {
        return;
    }

    let full_res = rt.resolution.xy;
    let full_pixel = vec2<f32>(gid.xy) * 2.0 + vec2<f32>(1.0);
    let full_pixel_i = vec2<i32>(full_pixel);

    // Lecture depth — skip sky.
    let depth = textureLoad(depth_tex, full_pixel_i, 0).r;
    if (depth >= 0.999) {
        textureStore(shadow_out, vec2<i32>(gid.xy), vec4<f32>(1.0, 0.0, 0.0, 1.0));
        return;
    }

    // Reconstruit la position monde.
    let world_pos = reconstruct_world_pos(full_pixel, depth, rt.inv_view_proj, full_res);

    // Direction vers le soleil (normalisée).
    let sun = normalize(rt.sun_dir.xyz);

    // Jitter le point de départ pour réduire le banding.
    let noise = blue_noise_sample(gid.xy);
    let jitter = noise.x;

    // Step length dans le monde.
    let step_world = SHADOW_MAX_DIST / f32(SHADOW_STEPS);
    var shadow = 1.0; // 1 = plein soleil
    var min_penumbra = 1.0;

    for (var i = 0; i < 16; i = i + 1) {
        let t = (f32(i) + jitter) * step_world;
        let sample_pos = world_pos + sun * t;

        // Projeter en screen space.
        let proj = project_to_screen(sample_pos, rt.view_proj);
        if (proj.w <= 0.0) { continue; }

        let sample_uv = proj.xy;
        // Hors écran → pas d'info, on skip.
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 ||
            sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        let sample_pixel = vec2<i32>(sample_uv * full_res);
        let scene_depth = textureLoad(depth_tex, sample_pixel, 0).r;

        // Occlusion : si la surface dans le depth buffer est plus proche
        // que le point projeté, quelque chose bloque la lumière.
        let ray_depth = proj.z;
        let delta = ray_depth - scene_depth;

        if (delta > 0.0005 && delta < 0.03) {
            // Penumbra : plus le step est proche de l'origine (i petit),
            // plus l'ombre est dure.  Plus c'est loin, plus c'est doux.
            let penumbra = smoothstep(0.0, SHADOW_MAX_DIST * 0.5, t);
            // L'occluder le plus proche donne l'ombre la plus dure.
            min_penumbra = min(min_penumbra, 0.1 + penumbra * 0.9);
            shadow = min(shadow, min_penumbra);
        }
    }

    // Clamp : jamais full black (ambient light existe toujours dans Q3).
    shadow = max(shadow, 0.15);

    textureStore(shadow_out, vec2<i32>(gid.xy), vec4<f32>(shadow, 0.0, 0.0, 1.0));
}
