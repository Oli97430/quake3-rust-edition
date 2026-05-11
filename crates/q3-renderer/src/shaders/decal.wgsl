// Decals dynamiques — marques de tir procedurales par type d'arme.
//
// Chaque decal = quad oriente sur la surface impactee.  Le type d'arme
// determine le pattern procedural (impact balle, brulure rocket, sang,
// plasma, rail, lightning).  Aucune texture externe — tout est genere
// dans le fragment shader pour un rendu net a toute distance.
//
// Layout vertex : position (vec3) + uv (vec2) + color (vec4) + decal_type (u32).
// Le `color.a` encode le timestamp de spawn pour le vieillissement.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>,
    inv_view_proj_rot: mat4x4<f32>,
    time_info: vec4<f32>,
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
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    out.clip_pos = camera.view_proj * vec4<f32>(v.position, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    out.world_pos = v.position;
    return out;
}

// --- Procedural noise ---

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let k = vec2<f32>(
        hash21(p),
        hash21(p + vec2<f32>(127.1, 311.7))
    );
    return k;
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
        mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

fn fbm(p: vec2<f32>, octaves: i32) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var pp = p;
    for (var i = 0; i < octaves; i++) {
        v += a * noise(pp);
        pp *= 2.0;
        a *= 0.5;
    }
    return v;
}

// --- Decal patterns par type d'arme ---

// Type 0 : impact balle (Machinegun, Shotgun) — petit trou + fissures radiales
fn bullet_hole(uv: vec2<f32>) -> vec4<f32> {
    let d = length(uv - 0.5);
    let hole = smoothstep(0.08, 0.04, d);
    let ring = (smoothstep(0.20, 0.10, d) - hole) * 0.7;
    let angle = atan2(uv.y - 0.5, uv.x - 0.5);
    let cracks = step(0.9, sin(angle * 7.0 + hash21(floor(vec2(angle * 2.0, 0.0))) * 6.28))
               * smoothstep(0.45, 0.12, d) * 0.3;
    let alpha = max(max(hole, ring), cracks) * smoothstep(0.5, 0.3, d);
    return vec4<f32>(0.02, 0.02, 0.02, alpha);
}

// Type 1 : brulure (Rocket, Grenade) — cercle noir charbon + bord orange
fn scorch_mark(uv: vec2<f32>, age: f32) -> vec4<f32> {
    let d = length(uv - 0.5);
    let n = fbm((uv - 0.5) * 6.0, 3);
    let dd = d + n * 0.1;
    let char_f = smoothstep(0.35, 0.12, dd);
    let edge = smoothstep(0.35, 0.22, dd) - smoothstep(0.22, 0.12, dd);
    let glow_fade = max(0.0, 1.0 - age * 0.3);
    let color = mix(vec3<f32>(0.03, 0.02, 0.01), vec3<f32>(0.8, 0.3, 0.05), edge * glow_fade);
    let alpha = char_f * smoothstep(0.5, 0.25, dd);
    return vec4<f32>(color, alpha);
}

// Type 2 : eclaboussure de sang (Gauntlet, degats) — forme organique
fn blood_splat(uv: vec2<f32>) -> vec4<f32> {
    let c = uv - 0.5;
    let d = length(c);
    let angle = atan2(c.y, c.x);
    let organic_r = 0.3 + fbm(vec2(angle * 3.0, d * 2.0), 2) * 0.15;
    let splat = smoothstep(organic_r, organic_r - 0.05, d);
    let color = vec3<f32>(0.4, 0.01, 0.01);
    return vec4<f32>(color, splat);
}

// Type 3 : brulure plasma (Plasma Gun) — halo bleu-violet
fn plasma_burn(uv: vec2<f32>, age: f32) -> vec4<f32> {
    let d = length(uv - 0.5);
    let n = fbm((uv - 0.5) * 4.0, 2);
    let glow = exp(-d * d * 20.0) * max(0.0, 1.0 - age * 0.5);
    let ring = smoothstep(0.22, 0.16, d + n * 0.05) * 0.8;
    let color = mix(vec3<f32>(0.1, 0.0, 0.3), vec3<f32>(0.3, 0.5, 1.0), glow);
    let alpha = max(ring, glow) * smoothstep(0.4, 0.2, d);
    return vec4<f32>(color, alpha);
}

// Type 4 : marque railgun — cicatrice allongee avec residus energetiques
fn rail_slug(uv: vec2<f32>, age: f32) -> vec4<f32> {
    let c = uv - 0.5;
    let d = length(c * vec2(1.0, 3.0));
    let scar = smoothstep(0.3, 0.22, d);
    let energy = exp(-d * d * 15.0) * max(0.0, 1.0 - age * 0.4);
    let color = mix(vec3(0.05, 0.05, 0.05), vec3(0.2, 0.4, 1.0), energy);
    let alpha = scar * smoothstep(0.4, 0.15, d);
    return vec4<f32>(color, alpha);
}

// Type 5 : brulure lightning — pattern fractal branchu
fn lightning_burn(uv: vec2<f32>) -> vec4<f32> {
    let d = length(uv - 0.5);
    let branches = fbm((uv - 0.5) * 10.0, 4);
    let pattern = step(0.6, branches) * smoothstep(0.45, 0.08, d);
    let color = vec3(0.5, 0.7, 1.0) * pattern + vec3(0.02, 0.02, 0.02) * (1.0 - pattern);
    let alpha = max(pattern * 0.8, smoothstep(0.12, 0.04, d) * 0.4) * smoothstep(0.5, 0.25, d);
    return vec4<f32>(color, alpha);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // Masque de base circulaire (comme avant) — fallback si pas de type
    let d = length(in.uv - vec2<f32>(0.5, 0.5)) * 2.0;
    let mask = 1.0 - smoothstep(0.9, 1.0, d);
    if mask <= 0.0 {
        discard;
    }

    // Decal type encode dans le canal G de la couleur vertex (hack compact :
    // le DecalRenderer encode le type dans color[1] comme f32).
    // Types : 0=bullet, 1=scorch, 2=blood, 3=plasma, 4=rail, 5=lightning
    let dtype = u32(in.color.g * 255.0 + 0.5);
    // Age depuis le spawn (pour les effets qui se refroidissent)
    let age = max(0.0, camera.time_info.x - in.color.b * 1000.0);

    var result: vec4<f32>;
    if dtype == 1u {
        result = scorch_mark(in.uv, age);
    } else if dtype == 2u {
        result = blood_splat(in.uv);
    } else if dtype == 3u {
        result = plasma_burn(in.uv, age);
    } else if dtype == 4u {
        result = rail_slug(in.uv, age);
    } else if dtype == 5u {
        result = lightning_burn(in.uv);
    } else {
        result = bullet_hole(in.uv);
    }

    // Modulation par la couleur vertex (alpha = fade temporel du DecalRenderer)
    result = vec4<f32>(result.rgb, result.a * in.color.a * mask);

    if result.a < 0.01 {
        discard;
    }

    return result;
}
