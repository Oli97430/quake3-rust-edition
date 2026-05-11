// Water shader haute qualite -- reflets, refraction, caustiques, mousse.
//
// Separe de material.wgsl pour permettre un pipeline dedie avec :
// - Refraction via distortion du depth buffer
// - Reflets SSR haute qualite (64 steps)
// - Caustiques volumetriques animes
// - Mousse dynamique aux bords (depth-based)
// - Absorption exponentielle (Beer-Lambert)
// - Flowmap optionnel pour courants
//
// Ce shader cible le format SCENE_HDR_FORMAT (Rgba16Float) -- les valeurs
// de sortie sont en HDR lineaire, le tonemap est applique par post.rs.

// -----------------------------------------------------------------------
// Bind group 0 -- camera uniform (partage avec le pipeline principal)
// -----------------------------------------------------------------------
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    inv_view_proj_rot: mat4x4<f32>,
    time_info: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

// -----------------------------------------------------------------------
// Bind group 1 -- parametres specifiques eau + textures scene
// -----------------------------------------------------------------------
struct WaterParams {
    // .x = wave_speed, .y = wave_amplitude, .z = refraction_strength, .w = foam_threshold
    params0: vec4<f32>,
    // .x = absorption_r, .y = absorption_g, .z = absorption_b, .w = max_depth
    absorption: vec4<f32>,
    // .xy = flow_direction, .z = flow_speed, .w = caustic_intensity
    flow: vec4<f32>,
    // .x = time (seconds) -- miroir de camera.time_info.x pour coherence
    time: vec4<f32>,
};
@group(1) @binding(0) var<uniform> water: WaterParams;
@group(1) @binding(1) var depth_tex: texture_2d<f32>;
@group(1) @binding(2) var color_tex: texture_2d<f32>;   // scene sans eau
@group(1) @binding(3) var water_samp: sampler;

// -----------------------------------------------------------------------
// Constantes physiques
// -----------------------------------------------------------------------

// Indice de refraction de l'eau (IOR ~1.33), utilise pour Schlick F0
const WATER_IOR: f32 = 1.33;
// F0 pour l'eau : ((n1-n2)/(n1+n2))^2 = ((1.0-1.33)/(1.0+1.33))^2 ~ 0.02
const WATER_F0: f32 = 0.02;

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;

// -----------------------------------------------------------------------
// Gerstner wave -- deplacement physiquement base de la surface
// -----------------------------------------------------------------------
// Une vague Gerstner deplace les vertex a la fois horizontalement
// (trochoidal, via steepness Q) et verticalement (sinusoidal).
// Cela produit des cretes pointues et des creux arrondis -- bien plus
// realiste qu'un simple sin(x).
//
// Parametres :
//   pos       -- position XY du vertex sur le plan d'eau
//   time      -- horloge (secondes)
//   dir       -- direction normalisee de propagation
//   steepness -- Q dans [0,1], 0 = pur sinus, 1 = cretes maximales
//   wavelength -- distance (unites Q3) entre deux cretes
//
// Retourne un vec3 : (dx, dy, dz) deplacement a ajouter a la position.
fn gerstner_wave(pos: vec2<f32>, time: f32, dir: vec2<f32>, steepness: f32, wavelength: f32) -> vec3<f32> {
    let k = TAU / max(wavelength, 0.01);       // nombre d'onde
    let c = sqrt(9.81 / k);                     // vitesse de phase (gravite)
    let d = normalize(dir);
    let f = k * (dot(d, pos) - c * time);       // phase
    let a = steepness / k;                       // amplitude derivee de Q

    // Deplacement horizontal (trochoidal) + vertical
    return vec3<f32>(
        d.x * a * cos(f),
        d.y * a * cos(f),
        a * sin(f),
    );
}

// -----------------------------------------------------------------------
// Multi-octave Gerstner -- superposition de 4 vagues
// -----------------------------------------------------------------------
// On empile 4 ondes Gerstner a des frequences, directions et amplitudes
// differentes pour briser la repetition et creer une surface complexe.
// Les parametres sont choisis pour evoquer un plan d'eau Q3 (lacs, puits
// de lave reconvertis) : ondes assez courtes, amplitude moderee.
fn water_displacement(pos: vec2<f32>, time: f32) -> vec3<f32> {
    let speed = water.params0.x;
    let amp = water.params0.y;
    let t = time * speed;

    // Application du flowmap optionnel (courants) -- decale la position
    // d'evaluation dans le temps pour simuler un deplacement lateral.
    let flow_offset = water.flow.xy * water.flow.z * time;
    let p = pos + flow_offset;

    var d = vec3<f32>(0.0);
    // Onde 1 : dominante, longue, vers +X+Y
    d += gerstner_wave(p, t, vec2<f32>(0.8, 0.6), 0.35 * amp, 120.0);
    // Onde 2 : secondaire, diagonale opposee
    d += gerstner_wave(p, t, vec2<f32>(-0.5, 0.86), 0.25 * amp, 80.0);
    // Onde 3 : haute frequence pour detail
    d += gerstner_wave(p, t, vec2<f32>(0.3, -0.95), 0.18 * amp, 45.0);
    // Onde 4 : detail fin, rapide
    d += gerstner_wave(p, t, vec2<f32>(-0.9, -0.4), 0.12 * amp, 25.0);

    return d;
}

// -----------------------------------------------------------------------
// Normale a partir du deplacement (differences finies)
// -----------------------------------------------------------------------
// On evalue le deplacement en 3 points voisins et on reconstruit la
// normale par produit vectoriel des tangentes.  Epsilon = 0.5 unites Q3
// donne un bon compromis resolution/bruit.
fn water_normal(pos: vec2<f32>, time: f32) -> vec3<f32> {
    let eps = 0.5;
    let d0 = water_displacement(pos, time);
    let dx = water_displacement(pos + vec2<f32>(eps, 0.0), time);
    let dy = water_displacement(pos + vec2<f32>(0.0, eps), time);

    // Tangentes dans l'espace monde (plan d'eau = XY, hauteur = Z)
    let tx = vec3<f32>(eps, 0.0, dx.z - d0.z);
    let ty = vec3<f32>(0.0, eps, dy.z - d0.z);

    return normalize(cross(ty, tx));
}

// -----------------------------------------------------------------------
// Caustiques volumetriques
// -----------------------------------------------------------------------
// Simule la convergence de la lumiere solaire a travers la surface ondulee.
// Technique : on evalue la densite de deformation de la surface (jacobien
// de la projection) via differences finies.  Zones ou la surface concentre
// la lumiere = caustiques brillants.  Pattern multi-octave pour le shimmer.
fn caustics(world_pos: vec3<f32>, time: f32) -> f32 {
    let intensity = water.flow.w;
    if (intensity <= 0.0) {
        return 0.0;
    }

    let p = world_pos.xy;
    let t = time;

    // Octave 1 : large scale, lente
    let eps = 1.0;
    let c0 = water_displacement(p, t).xy;
    let cx = water_displacement(p + vec2<f32>(eps, 0.0), t).xy;
    let cy = water_displacement(p + vec2<f32>(0.0, eps), t).xy;

    // Jacobien 2D du deplacement horizontal -- quand le jacobien < 1,
    // la lumiere converge (= caustique).
    let j00 = 1.0 + (cx.x - c0.x) / eps;
    let j01 = (cy.x - c0.x) / eps;
    let j10 = (cx.y - c0.y) / eps;
    let j11 = 1.0 + (cy.y - c0.y) / eps;
    let det = abs(j00 * j11 - j01 * j10);

    // Inversion + seuillage : det petit = forte convergence = brillant
    let caust = pow(clamp(1.0 / max(det, 0.05), 0.0, 5.0), 2.0) - 1.0;

    // Seconde octave pour enrichir le pattern (frequence 2x, dephasage)
    let p2 = p * 1.7 + vec2<f32>(t * 0.3, -t * 0.2);
    let c0b = water_displacement(p2, t * 1.3).xy;
    let cxb = water_displacement(p2 + vec2<f32>(eps, 0.0), t * 1.3).xy;
    let cyb = water_displacement(p2 + vec2<f32>(0.0, eps), t * 1.3).xy;
    let jb00 = 1.0 + (cxb.x - c0b.x) / eps;
    let jb01 = (cyb.x - c0b.x) / eps;
    let jb10 = (cxb.y - c0b.y) / eps;
    let jb11 = 1.0 + (cyb.y - c0b.y) / eps;
    let det2 = abs(jb00 * jb11 - jb01 * jb10);
    let caust2 = pow(clamp(1.0 / max(det2, 0.05), 0.0, 5.0), 2.0) - 1.0;

    return clamp((caust * 0.6 + caust2 * 0.4) * intensity, 0.0, 1.0);
}

// -----------------------------------------------------------------------
// Mousse dynamique aux bords (depth-based)
// -----------------------------------------------------------------------
// La mousse apparait quand la difference de profondeur entre la surface
// d'eau et la geometrie sous-jacente est faible (= bord, plage, rocher
// affleurant).  Un noise anime empoche la mousse d'etre un contour net.
fn foam(depth_diff: f32, world_pos: vec2<f32>, time: f32) -> f32 {
    let threshold = water.params0.w;
    if (threshold <= 0.0) {
        return 0.0;
    }

    // Noise multi-octave simple (sin-based, pas de Perlin pour le cout)
    let n1 = sin(world_pos.x * 0.12 + time * 1.3) * 0.5 + 0.5;
    let n2 = sin(world_pos.y * 0.15 + time * 1.7 + 2.1) * 0.5 + 0.5;
    let n3 = sin((world_pos.x + world_pos.y) * 0.08 + time * 0.9) * 0.5 + 0.5;
    let noise = n1 * 0.5 + n2 * 0.3 + n3 * 0.2;

    // Plus la profondeur est faible, plus la mousse est intense
    let edge_factor = 1.0 - smoothstep(0.0, threshold, depth_diff);

    // Seuillage du noise pour creer des patches discontinus
    let foam_mask = smoothstep(0.3, 0.6, noise);

    return edge_factor * foam_mask;
}

// -----------------------------------------------------------------------
// Beer-Lambert absorption
// -----------------------------------------------------------------------
// L'eau absorbe la lumiere exponentiellement avec la profondeur.
// Chaque canal RGB a son propre coefficient d'absorption -- le rouge
// est absorbe bien plus vite que le bleu, d'ou la teinte bleue en
// profondeur.  C'est le modele physique standard (Beer-Lambert).
fn beer_lambert(depth: f32, scene_color: vec3<f32>) -> vec3<f32> {
    let coeffs = water.absorption.xyz;
    let max_d = water.absorption.w;
    let d = clamp(depth, 0.0, max_d);
    let transmittance = exp(-coeffs * d);
    return scene_color * transmittance;
}

// -----------------------------------------------------------------------
// Schlick Fresnel
// -----------------------------------------------------------------------
fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    let t = clamp(1.0 - cos_theta, 0.0, 1.0);
    return f0 + (1.0 - f0) * t * t * t * t * t;
}

// -----------------------------------------------------------------------
// Hash pseudo-aleatoire 2D (pour le bruit SSR)
// -----------------------------------------------------------------------
fn hash2d(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

// -----------------------------------------------------------------------
// Vertex shader -- deplace les vertices du plan d'eau via Gerstner
// -----------------------------------------------------------------------
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
    @location(3) screen_pos: vec4<f32>,
};

@vertex
fn vs_water(in: VsIn) -> VsOut {
    let time = water.time.x;
    let disp = water_displacement(in.position.xy, time);

    // Deplace le vertex : XY trochoidal + Z vertical
    var world = in.position;
    world.x += disp.x;
    world.y += disp.y;
    world.z += disp.z;

    let clip = camera.view_proj * vec4<f32>(world, 1.0);
    let n = water_normal(in.position.xy, time);

    var out: VsOut;
    out.clip_pos = clip;
    out.world_pos = world;
    out.normal = n;
    out.tex_uv = in.tex_uv;
    out.screen_pos = clip;
    return out;
}

// -----------------------------------------------------------------------
// Fragment shader
// -----------------------------------------------------------------------
// Pipeline complet :
//   1. Normale Gerstner multi-octave
//   2. Refraction screen-space (distortion UV par normale)
//   3. Beer-Lambert sur la couleur refractee
//   4. Caustiques volumetriques
//   5. SSR 64-step (reflection)
//   6. Fresnel Schlick blend refraction/reflection
//   7. Mousse depth-based
//   8. Sortie HDR lineaire
@fragment
fn fs_water(in: VsOut) -> @location(0) vec4<f32> {
    let time = water.time.x;
    let dims = vec2<f32>(textureDimensions(color_tex));

    // UV ecran [0,1] depuis clip position
    let ndc = in.screen_pos.xyz / in.screen_pos.w;
    let screen_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    // ---- 1. Normale eau (deja interpolee depuis le VS mais on la
    //         recalcule au fragment pour plus de precision) ----
    let n = normalize(in.normal);
    let view_dir = normalize(camera.view_pos.xyz - in.world_pos);

    // ---- 2. Refraction screen-space ----
    // On decale les UV ecran proportionnellement a la composante XY de
    // la normale (perturbation de surface vue de dessus).  Le facteur
    // refraction_strength controle l'intensite de la distortion.
    let refr_strength = water.params0.z;
    let distortion = n.xy * refr_strength;
    let refr_uv = clamp(screen_uv + distortion, vec2<f32>(0.001), vec2<f32>(0.999));

    // Lecture profondeur scene au point refracte
    let scene_depth_raw = textureSampleLevel(depth_tex, water_samp, refr_uv, 0.0).r;
    let water_depth_raw = ndc.z;

    // Difference de profondeur en unites lineaires approximatives.
    // On utilise un linearize simplifie (perspective reverse-Z) :
    // les valeurs proches de 1.0 sont proches de la camera.
    // depth_diff > 0 quand le fond est PLUS LOIN que la surface.
    let z_near = 4.0;   // coherent avec camera.rs DEFAULT_Z_NEAR
    let z_far = 16384.0; // coherent avec camera.rs DEFAULT_Z_FAR
    let lin_scene = z_near * z_far / (z_far - scene_depth_raw * (z_far - z_near));
    let lin_water = z_near * z_far / (z_far - water_depth_raw * (z_far - z_near));
    let depth_diff = max(lin_scene - lin_water, 0.0);

    // Si la refraction pointe DEVANT la surface (artefact de distortion
    // trop forte), on fallback aux UV non-distordues.
    var final_refr_uv = refr_uv;
    if (depth_diff < 0.1) {
        final_refr_uv = screen_uv;
    }

    // ---- 3. Couleur refractee + Beer-Lambert ----
    let refracted_raw = textureSampleLevel(color_tex, water_samp, final_refr_uv, 0.0).rgb;
    let refracted = beer_lambert(depth_diff, refracted_raw);

    // ---- 4. Caustiques volumetriques ----
    let caust = caustics(in.world_pos, time);
    // Les caustiques eclairent la scene SOUS l'eau -- on les module par
    // la profondeur (pas de caustiques a la surface meme).
    let caust_depth_fade = smoothstep(0.0, 50.0, depth_diff);
    let refracted_lit = refracted + refracted * caust * caust_depth_fade * 0.8;

    // ---- 5. SSR -- Screen Space Reflection (64 steps) ----
    // Raymarch en clip-space depuis la position du fragment dans la
    // direction de reflexion.  On cherche le premier point ou la
    // profondeur du ray depasse celle du depth buffer (= hit geometry).
    let refl_dir = reflect(-view_dir, n);
    var ssr_color = vec3<f32>(0.0);
    var ssr_hit = 0.0;

    // On tente le SSR seulement si la reflexion va vers le HAUT
    // (sinon on traverserait le sol -- valeurs garbage).
    if (refl_dir.z > 0.02) {
        let step_size = 16.0;   // unites Q3 par step
        var march_pos = in.world_pos + n * 1.5;  // offset pour eviter self-hit

        for (var i = 0; i < 64; i = i + 1) {
            march_pos = march_pos + refl_dir * step_size;

            let clip = camera.view_proj * vec4<f32>(march_pos, 1.0);
            if (clip.w <= 0.0) { break; }

            let ray_ndc = clip.xyz / clip.w;
            // Hors frustum => bail
            if (ray_ndc.x < -1.0 || ray_ndc.x > 1.0 ||
                ray_ndc.y < -1.0 || ray_ndc.y > 1.0) {
                break;
            }

            let ray_uv = vec2<f32>(
                ray_ndc.x * 0.5 + 0.5,
                -ray_ndc.y * 0.5 + 0.5,
            );

            let sampled_depth = textureSampleLevel(depth_tex, water_samp, ray_uv, 0.0).r;
            let ray_depth = ray_ndc.z;

            // Hit test : le ray a depasse la geometrie dans le depth buffer
            if (ray_depth > sampled_depth && ray_depth - sampled_depth < 0.015) {
                ssr_color = textureSampleLevel(color_tex, water_samp, ray_uv, 0.0).rgb;

                // Fade aux bords d'ecran pour eviter la coupure seche
                let edge = min(
                    min(ray_uv.x, 1.0 - ray_uv.x),
                    min(ray_uv.y, 1.0 - ray_uv.y),
                );
                let edge_fade = smoothstep(0.0, 0.12, edge);

                // Fade avec la distance de marche (reflexions lointaines
                // moins fiables -- bruit, resolution insuffisante)
                let dist_fade = 1.0 - f32(i) / 64.0;

                ssr_hit = edge_fade * dist_fade;
                break;
            }
        }
    }

    // Fallback : si pas de hit SSR, couleur du ciel/ambiant
    let sky_color = vec3<f32>(0.35, 0.55, 0.85);
    let reflected = mix(sky_color, ssr_color, ssr_hit);

    // ---- 6. Fresnel Schlick ----
    let cos_theta = max(dot(view_dir, n), 0.0);
    let fres = fresnel_schlick(cos_theta, WATER_F0);

    // Blend : a incidence rasante (fres ~1) on voit la reflexion,
    // en plongee (fres ~0.02) on voit le fond refracte.
    var color = mix(refracted_lit, reflected, fres);

    // ---- 7. Mousse depth-based ----
    let foam_val = foam(depth_diff, in.world_pos.xy, time);
    // La mousse est blanche avec une legere teinte cyan
    let foam_color = vec3<f32>(0.92, 0.96, 1.0);
    color = mix(color, foam_color, foam_val * 0.55);

    // ---- 8. Sparkle highlights sur les cretes ----
    // Les cretes Gerstner ont une composante Z elevee du displacement ;
    // on utilise la derivee de la hauteur comme proxy pour les cretes.
    let crest = water_displacement(in.world_pos.xy, time).z;
    let sparkle_mask = pow(max(crest / max(water.params0.y, 0.01), 0.0), 6.0);
    let sparkle = sparkle_mask * 0.5 * max(dot(n, normalize(vec3<f32>(-0.4, -0.3, 0.86))), 0.0);
    color += vec3<f32>(0.7, 0.85, 1.0) * sparkle;

    // ---- 9. Sortie HDR ----
    // Alpha : opaque au ras (Fresnel -> reflexion forte), transparent
    // en plongee pour laisser entrevoir le fond via la refraction.
    let alpha = mix(0.6, 0.98, fres);

    return vec4<f32>(color, alpha);
}
