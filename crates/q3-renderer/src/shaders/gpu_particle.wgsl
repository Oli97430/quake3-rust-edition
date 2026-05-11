// Systeme de particules GPU compute+render — remplace le CPU billboard
// pour de bien plus hauts counts (65 536) avec physique integree
// (gravite, drag, turbulence).
//
// Architecture ping-pong : le compute shader lit un buffer source et
// ecrit dans un buffer destination. Le render shader lit le buffer
// destination pour dessiner les billboards camera-facing.
//
// Types de particules :
//   0 = Blood    — rouge, gravite forte, drag moyen
//   1 = Spark    — jaune->blanc, gravite legere, drag faible
//   2 = Smoke    — gris, gravite negative (monte), drag fort, turbulence
//   3 = Debris   — brun, gravite forte, drag faible
//   4 = Fire     — orange->rouge, gravite negative, turbulence moyenne
//   5 = PlasmaTrail — bleu-violet, pas de gravite, drag moyen
//   6 = RocketTrail — orange fumee, gravite legere, drag moyen

// ─── Camera uniform (group 0) — layout identique a tous les pipelines ───
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    inv_view_proj_rot: mat4x4<f32>,
    time_info: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

// ─── Particle data layout (GPU-side) ───
struct Particle {
    position: vec3<f32>,
    life: f32,
    velocity: vec3<f32>,
    max_life: f32,
    color: vec4<f32>,
    size: f32,
    ptype: u32,
    _pad0: f32,
    _pad1: f32,
};

// ─── Simulation parameters ───
struct SimParams {
    dt: f32,
    time: f32,
    max_particles: u32,
    _pad: u32,
};

// ─── Indirect draw args (pour le rendu instancie) ───
struct IndirectArgs {
    vertex_count: atomic<u32>,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

// ─── Compute bindings (group 1) ───
@group(1) @binding(0) var<storage, read>       particles_src: array<Particle>;
@group(1) @binding(1) var<storage, read_write> particles_dst: array<Particle>;
@group(1) @binding(2) var<uniform>             sim: SimParams;
@group(1) @binding(3) var<storage, read_write> indirect: IndirectArgs;

// ─── Noise helpers pour la turbulence fumee ───
// Hash pseudo-aleatoire rapide — suffisant pour un jitter visuel.
fn hash31(p: vec3<f32>) -> f32 {
    let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453);
}

fn hash33(p: vec3<f32>) -> vec3<f32> {
    let x = hash31(p);
    let y = hash31(p + vec3<f32>(37.0, 17.0, 59.0));
    let z = hash31(p + vec3<f32>(71.0, 53.0, 97.0));
    return vec3<f32>(x, y, z) * 2.0 - vec3<f32>(1.0);
}

// ─── Parametres par type de particule ───
// Gravite Q3 : Z-up, acceleration en unites/s^2.
fn gravity_for_type(ptype: u32) -> vec3<f32> {
    switch ptype {
        // Blood — chute rapide comme du liquide
        case 0u: { return vec3<f32>(0.0, 0.0, -800.0); }
        // Spark — chute legere
        case 1u: { return vec3<f32>(0.0, 0.0, -400.0); }
        // Smoke — monte lentement (convection)
        case 2u: { return vec3<f32>(0.0, 0.0, 120.0); }
        // Debris — lourds, chute rapide
        case 3u: { return vec3<f32>(0.0, 0.0, -800.0); }
        // Fire — monte doucement
        case 4u: { return vec3<f32>(0.0, 0.0, 80.0); }
        // PlasmaTrail — zero G (projectile plasma)
        case 5u: { return vec3<f32>(0.0, 0.0, 0.0); }
        // RocketTrail — legere montee (chaleur)
        case 6u: { return vec3<f32>(0.0, 0.0, 40.0); }
        default: { return vec3<f32>(0.0, 0.0, -800.0); }
    }
}

fn drag_for_type(ptype: u32) -> f32 {
    switch ptype {
        case 0u: { return 0.97; }  // Blood — drag moyen
        case 1u: { return 0.995; } // Spark — presque pas de drag
        case 2u: { return 0.92; }  // Smoke — drag fort (air epais)
        case 3u: { return 0.99; }  // Debris — drag faible
        case 4u: { return 0.94; }  // Fire — drag moyen-fort
        case 5u: { return 0.96; }  // PlasmaTrail — drag moyen
        case 6u: { return 0.95; }  // RocketTrail — drag moyen
        default: { return 0.98; }
    }
}

fn turbulence_strength(ptype: u32) -> f32 {
    switch ptype {
        case 2u: { return 180.0; } // Smoke — forte turbulence
        case 4u: { return 100.0; } // Fire — turbulence moyenne
        case 6u: { return 60.0; }  // RocketTrail — turbulence legere
        default: { return 0.0; }   // Les autres n'en ont pas
    }
}

// ════════════════════════════════════════════════════════════════════
//  COMPUTE : mise a jour physique de chaque particule
// ════════════════════════════════════════════════════════════════════
@compute @workgroup_size(256)
fn cs_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= sim.max_particles {
        return;
    }

    var p = particles_src[idx];

    // Particule morte — on la copie telle quelle (life <= 0).
    if p.life <= 0.0 {
        particles_dst[idx] = p;
        return;
    }

    let dt = sim.dt;
    let ptype = p.ptype;

    // ── Gravite ──
    let grav = gravity_for_type(ptype);
    p.velocity = p.velocity + grav * dt;

    // ── Drag (friction aerodynamique) — applique par frame ──
    // Pour un dt variable on approche le drag exponentiel :
    // drag_frame = drag^(dt/0.016) pour normaliser a ~60fps.
    let base_drag = drag_for_type(ptype);
    let drag_frame = pow(base_drag, dt / 0.016);
    p.velocity = p.velocity * drag_frame;

    // ── Turbulence (noise-based, pour smoke/fire/trail) ──
    let turb_str = turbulence_strength(ptype);
    if turb_str > 0.0 {
        // Seed spatio-temporel pour varier le jitter entre particules
        // et entre frames. On utilise position + temps pour decoreller.
        let seed = p.position * 0.02 + vec3<f32>(sim.time * 1.3);
        let noise_disp = hash33(seed) * turb_str * dt;
        p.velocity = p.velocity + noise_disp;
    }

    // ── Integration position ──
    p.position = p.position + p.velocity * dt;

    // ── Vie ──
    p.life = p.life - dt;
    if p.life <= 0.0 {
        p.life = 0.0;
    }

    // ── Ecriture destination ──
    particles_dst[idx] = p;

    // ── Compteur atomique de particules vivantes (pour indirect draw) ──
    if p.life > 0.0 {
        atomicAdd(&indirect.vertex_count, 4u);
    }
}

// ════════════════════════════════════════════════════════════════════
//  RENDER : vertex shader — billboard camera-facing
// ════════════════════════════════════════════════════════════════════

// On lit le buffer particules destination (le plus a jour apres compute).
// group(1) en render = le particle buffer (read-only).
@group(1) @binding(0) var<storage, read> render_particles: array<Particle>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

// Couleur de base par type — le fragment shader la module avec le life ratio.
fn base_color_for_type(ptype: u32) -> vec3<f32> {
    switch ptype {
        case 0u: { return vec3<f32>(0.7, 0.02, 0.02); }  // Blood — rouge sombre
        case 1u: { return vec3<f32>(1.0, 0.85, 0.3); }    // Spark — jaune chaud
        case 2u: { return vec3<f32>(0.45, 0.42, 0.38); }  // Smoke — gris
        case 3u: { return vec3<f32>(0.5, 0.35, 0.15); }   // Debris — brun
        case 4u: { return vec3<f32>(1.0, 0.5, 0.08); }    // Fire — orange vif
        case 5u: { return vec3<f32>(0.3, 0.4, 1.0); }     // PlasmaTrail — bleu
        case 6u: { return vec3<f32>(1.0, 0.6, 0.2); }     // RocketTrail — orange fumee
        default: { return vec3<f32>(1.0, 1.0, 1.0); }
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VsOut {
    var out: VsOut;

    let p = render_particles[instance_index];

    // Particule morte — degenerer le triangle (vertex a zero).
    if p.life <= 0.0 {
        out.clip_pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.uv = vec2<f32>(0.0);
        out.color = vec4<f32>(0.0);
        return out;
    }

    // ── Billboard camera-facing ──
    // On extrait les axes right/up de la view matrix via view_proj.
    // Pour un billboard parfait ecran-aligne, on lit les colonnes de
    // l'inverse de la view (= transposee de la partie rotation de VP).
    // Heuristique plus simple : la view_proj inverse est deja dispo
    // via inv_view_proj_rot. On approxime en lisant les lignes de VP
    // comme axes camera (valable quand la projection n'est pas trop
    // deformee, ce qui est le cas en pratique avec FOV < 130).
    let vp = camera.view_proj;
    // Colonne 0 de view_proj^T = ligne 0 de VP = axe right en clip
    // On veut les axes camera MONDE, pas clip. Approximation :
    // on normalise les 3 premieres composantes des lignes 0 et 1.
    let cam_right = normalize(vec3<f32>(vp[0][0], vp[1][0], vp[2][0]));
    let cam_up    = normalize(vec3<f32>(vp[0][1], vp[1][1], vp[2][1]));

    // Life ratio [0..1] — 1 = vient de naitre, 0 = mort
    let life_ratio = clamp(p.life / max(p.max_life, 0.001), 0.0, 1.0);

    // Taille evolue avec la vie : les particules grossissent en mourant
    // (simulation d'expansion fumee/feu).
    let size_mult = mix(1.5, 1.0, life_ratio);
    let half_size = p.size * size_mult * 0.5;

    // 4 vertices par particule (triangle strip) :
    //   0: bottom-left, 1: bottom-right, 2: top-left, 3: top-right
    let corner_id = vertex_index % 4u;
    var offset_x: f32;
    var offset_y: f32;
    var uv: vec2<f32>;
    switch corner_id {
        case 0u: { offset_x = -1.0; offset_y = -1.0; uv = vec2<f32>(0.0, 0.0); }
        case 1u: { offset_x =  1.0; offset_y = -1.0; uv = vec2<f32>(1.0, 0.0); }
        case 2u: { offset_x = -1.0; offset_y =  1.0; uv = vec2<f32>(0.0, 1.0); }
        case 3u: { offset_x =  1.0; offset_y =  1.0; uv = vec2<f32>(1.0, 1.0); }
        default: { offset_x = 0.0; offset_y = 0.0; uv = vec2<f32>(0.0); }
    }

    let world_pos = p.position
        + cam_right * (offset_x * half_size)
        + cam_up    * (offset_y * half_size);

    out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = uv;

    // ── Couleur + alpha fade ──
    var base_col = base_color_for_type(p.ptype);

    // Spark : transition jaune chaud -> blanc vif en debut de vie
    if p.ptype == 1u {
        base_col = mix(vec3<f32>(1.0, 1.0, 0.9), base_col, life_ratio);
    }
    // Fire : transition orange -> rouge sombre en fin de vie
    if p.ptype == 4u {
        base_col = mix(vec3<f32>(0.3, 0.05, 0.01), base_col, life_ratio);
    }

    // Alpha : plein au debut, fade-out sur les derniers 40 % de la vie.
    // Ceci empeche le « pop » de disparition brusque.
    let alpha_fade = smoothstep(0.0, 0.4, life_ratio);
    // Modulation supplementaire par la couleur stockee (permet au caller
    // de varier l'opacite initiale par emitter).
    let final_alpha = p.color.a * alpha_fade;

    out.color = vec4<f32>(base_col * p.color.rgb, final_alpha);
    return out;
}

// ════════════════════════════════════════════════════════════════════
//  RENDER : fragment shader — masque circulaire doux
// ════════════════════════════════════════════════════════════════════
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Distance au centre du quad, normalisee [0,1] : 0 centre, 1 bord.
    let d = length(in.uv - vec2<f32>(0.5, 0.5)) * 2.0;

    // Masque radial doux — zone pleine au centre, falloff progressif.
    // Plus doux que le CPU particle pour un rendu « volumetrique ».
    let mask = 1.0 - smoothstep(0.0, 1.0, d);

    let alpha = in.color.a * mask;

    // Discard les fragments quasi-transparents pour eviter les
    // ecritures inutiles dans le framebuffer (gain bandwidth).
    if alpha < 0.004 {
        discard;
    }

    return vec4<f32>(in.color.rgb, alpha);
}
