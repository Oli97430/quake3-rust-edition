// Global Illumination — evaluation des light probes SH L2 dans le shader.
//
// Ce fichier est inclu dans les shaders material/drone via le systeme de
// bind groups. Il fournit `sample_gi(world_pos, normal)` qui remplace
// l'ambient hemispherique hardcode par un eclairage indirect interpole
// depuis une grille 3D de probes SH.
//
// Les 9 coefficients SH × 3 canaux RGB = 27 floats sont packes dans
// 7 vec4 (28 floats, le .w du dernier est padding).
//
// Convention axes : Z-up (Q3).

// --- Structures ---

struct ProbeData {
    // sh[0] = (R_Y00, R_Y1n1, R_Y10, R_Y11)
    // sh[1] = (R_Y2n2, R_Y2n1, R_Y20, R_Y21)
    // sh[2] = (R_Y22, G_Y00, G_Y1n1, G_Y10)
    // sh[3] = (G_Y11, G_Y2n2, G_Y2n1, G_Y20)
    // sh[4] = (G_Y21, G_Y22, B_Y00, B_Y1n1)
    // sh[5] = (B_Y10, B_Y11, B_Y2n2, B_Y2n1)
    // sh[6] = (B_Y20, B_Y21, B_Y22, pad)
    sh: array<vec4<f32>, 7>,
};

struct ProbeGridInfo {
    bounds_min: vec4<f32>,   // xyz = min, w = spacing
    bounds_max: vec4<f32>,   // xyz = max, w = pad
    grid_size: vec4<u32>,    // xyz = count per axis, w = total
};

@group(7) @binding(0) var<uniform> gi_grid: ProbeGridInfo;
@group(7) @binding(1) var<storage, read> gi_probes: array<ProbeData>;

// --- Constantes SH ---

const SH_C0: f32 = 0.282094792;  // 1/(2*sqrt(pi))
const SH_C1: f32 = 0.488602512;  // sqrt(3)/(2*sqrt(pi))
const SH_C2_0: f32 = 1.092548431; // sqrt(15)/(2*sqrt(pi))
const SH_C2_1: f32 = 0.315391565; // sqrt(5)/(4*sqrt(pi))
const SH_C2_2: f32 = 0.546274215; // sqrt(15)/(4*sqrt(pi))

// --- Evaluation SH L2 ---

fn sh_irradiance(probe: ProbeData, n: vec3<f32>) -> vec3<f32> {
    // Evaluer les 9 fonctions de base SH pour la direction n
    var basis: array<f32, 9>;
    basis[0] = SH_C0;                          // Y00
    basis[1] = SH_C1 * n.y;                    // Y1,-1
    basis[2] = SH_C1 * n.z;                    // Y1,0
    basis[3] = SH_C1 * n.x;                    // Y1,1
    basis[4] = SH_C2_0 * n.x * n.y;            // Y2,-2
    basis[5] = SH_C2_0 * n.y * n.z;            // Y2,-1
    basis[6] = SH_C2_1 * (3.0 * n.z * n.z - 1.0); // Y2,0
    basis[7] = SH_C2_0 * n.x * n.z;            // Y2,1
    basis[8] = SH_C2_2 * (n.x * n.x - n.y * n.y); // Y2,2

    // Depack les coefficients depuis les 7 vec4
    var r = 0.0;
    var g = 0.0;
    var b = 0.0;

    // Canal R : sh[0].xyzw = R(0..3), sh[1].xyzw = R(4..7), sh[2].x = R(8)
    r += probe.sh[0].x * basis[0];
    r += probe.sh[0].y * basis[1];
    r += probe.sh[0].z * basis[2];
    r += probe.sh[0].w * basis[3];
    r += probe.sh[1].x * basis[4];
    r += probe.sh[1].y * basis[5];
    r += probe.sh[1].z * basis[6];
    r += probe.sh[1].w * basis[7];
    r += probe.sh[2].x * basis[8];

    // Canal G : sh[2].yzw = G(0..2), sh[3].xyzw = G(3..6), sh[4].xy = G(7..8)
    g += probe.sh[2].y * basis[0];
    g += probe.sh[2].z * basis[1];
    g += probe.sh[2].w * basis[2];
    g += probe.sh[3].x * basis[3];
    g += probe.sh[3].y * basis[4];
    g += probe.sh[3].z * basis[5];
    g += probe.sh[3].w * basis[6];
    g += probe.sh[4].x * basis[7];
    g += probe.sh[4].y * basis[8];

    // Canal B : sh[4].zw = B(0..1), sh[5].xyzw = B(2..5), sh[6].xyz = B(6..8)
    b += probe.sh[4].z * basis[0];
    b += probe.sh[4].w * basis[1];
    b += probe.sh[5].x * basis[2];
    b += probe.sh[5].y * basis[3];
    b += probe.sh[5].z * basis[4];
    b += probe.sh[5].w * basis[5];
    b += probe.sh[6].x * basis[6];
    b += probe.sh[6].y * basis[7];
    b += probe.sh[6].z * basis[8];

    return max(vec3(r, g, b), vec3(0.0));
}

// --- Interpolation trilineaire de la grille ---

fn sample_gi(world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let spacing = gi_grid.bounds_min.w;
    if spacing <= 0.0 {
        // Pas de grille — fallback hemisphere
        let sky = vec3(0.40, 0.55, 0.70);
        let ground = vec3(0.18, 0.14, 0.10);
        let hemi = mix(ground, sky, clamp(normal.z * 0.5 + 0.5, 0.0, 1.0));
        return hemi * 0.25;
    }

    // Position normalisee dans la grille [0..grid_size-1]
    let rel = (world_pos - gi_grid.bounds_min.xyz) / spacing;
    let grid = vec3<f32>(f32(gi_grid.grid_size.x), f32(gi_grid.grid_size.y), f32(gi_grid.grid_size.z));
    let clamped = clamp(rel, vec3(0.0), grid - vec3(1.001));

    // Indices entiers et fraction pour trilinear
    let i = vec3<u32>(u32(clamped.x), u32(clamped.y), u32(clamped.z));
    let f = fract(clamped);

    let sx = gi_grid.grid_size.x;
    let sy = gi_grid.grid_size.y;

    // 8 probes autour du point (clamp si au bord)
    let i0 = i;
    let i1 = vec3<u32>(min(i.x + 1u, sx - 1u), min(i.y + 1u, sy - 1u), min(i.z + 1u, gi_grid.grid_size.z - 1u));

    let idx000 = i0.z * sy * sx + i0.y * sx + i0.x;
    let idx100 = i0.z * sy * sx + i0.y * sx + i1.x;
    let idx010 = i0.z * sy * sx + i1.y * sx + i0.x;
    let idx110 = i0.z * sy * sx + i1.y * sx + i1.x;
    let idx001 = i1.z * sy * sx + i0.y * sx + i0.x;
    let idx101 = i1.z * sy * sx + i0.y * sx + i1.x;
    let idx011 = i1.z * sy * sx + i1.y * sx + i0.x;
    let idx111 = i1.z * sy * sx + i1.y * sx + i1.x;

    // Evaluer SH pour chaque probe
    let c000 = sh_irradiance(gi_probes[idx000], normal);
    let c100 = sh_irradiance(gi_probes[idx100], normal);
    let c010 = sh_irradiance(gi_probes[idx010], normal);
    let c110 = sh_irradiance(gi_probes[idx110], normal);
    let c001 = sh_irradiance(gi_probes[idx001], normal);
    let c101 = sh_irradiance(gi_probes[idx101], normal);
    let c011 = sh_irradiance(gi_probes[idx011], normal);
    let c111 = sh_irradiance(gi_probes[idx111], normal);

    // Interpolation trilineaire
    let c00 = mix(c000, c100, f.x);
    let c10 = mix(c010, c110, f.x);
    let c01 = mix(c001, c101, f.x);
    let c11 = mix(c011, c111, f.x);
    let c0 = mix(c00, c10, f.y);
    let c1 = mix(c01, c11, f.y);
    let result = mix(c0, c1, f.z);

    return result;
}
