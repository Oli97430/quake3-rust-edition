// Skinning compute shader — transforme les vertices d'un mesh par les
// matrices joints evaluees CPU-side.
//
// Dispatch: ceil(vertex_count / 256) workgroups.
// Chaque thread transforme un vertex par ses 4 joints ponderes.
//
// Le systeme est decouple du render : on ecrit dans un buffer dst qui
// sera directement utilise comme vertex buffer par le draw suivant.
// Cela permet de skinned n'importe quel mesh (arme, personnage) sans
// modifier le pipeline de rendu existant.

struct SkinnedVertex {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    pad0: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    pad1: f32,
    uv_x: f32,
    uv_y: f32,
    pad2: f32,
    pad3: f32,
};

struct SkinInfluence {
    joint0: u32,
    joint1: u32,
    joint2: u32,
    joint3: u32,
    weight0: f32,
    weight1: f32,
    weight2: f32,
    weight3: f32,
};

struct Params {
    vertex_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> src_vertices: array<SkinnedVertex>;
@group(0) @binding(1) var<storage, read> skin_influences: array<SkinInfluence>;
@group(0) @binding(2) var<storage, read> joint_matrices: array<mat4x4<f32>, 64>;
@group(0) @binding(3) var<storage, read_write> dst_vertices: array<SkinnedVertex>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn cs_skin(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.vertex_count {
        return;
    }

    let src = src_vertices[idx];
    let influence = skin_influences[idx];

    let src_pos = vec3<f32>(src.pos_x, src.pos_y, src.pos_z);
    let src_normal = vec3<f32>(src.normal_x, src.normal_y, src.normal_z);

    var skinned_pos = vec3<f32>(0.0, 0.0, 0.0);
    var skinned_normal = vec3<f32>(0.0, 0.0, 0.0);

    // Joint 0
    if influence.weight0 > 0.0 {
        let m = joint_matrices[influence.joint0];
        skinned_pos += (m * vec4<f32>(src_pos, 1.0)).xyz * influence.weight0;
        skinned_normal += (m * vec4<f32>(src_normal, 0.0)).xyz * influence.weight0;
    }
    // Joint 1
    if influence.weight1 > 0.0 {
        let m = joint_matrices[influence.joint1];
        skinned_pos += (m * vec4<f32>(src_pos, 1.0)).xyz * influence.weight1;
        skinned_normal += (m * vec4<f32>(src_normal, 0.0)).xyz * influence.weight1;
    }
    // Joint 2
    if influence.weight2 > 0.0 {
        let m = joint_matrices[influence.joint2];
        skinned_pos += (m * vec4<f32>(src_pos, 1.0)).xyz * influence.weight2;
        skinned_normal += (m * vec4<f32>(src_normal, 0.0)).xyz * influence.weight2;
    }
    // Joint 3
    if influence.weight3 > 0.0 {
        let m = joint_matrices[influence.joint3];
        skinned_pos += (m * vec4<f32>(src_pos, 1.0)).xyz * influence.weight3;
        skinned_normal += (m * vec4<f32>(src_normal, 0.0)).xyz * influence.weight3;
    }

    let final_normal = normalize(skinned_normal);

    var out: SkinnedVertex;
    out.pos_x = skinned_pos.x;
    out.pos_y = skinned_pos.y;
    out.pos_z = skinned_pos.z;
    out.pad0 = 0.0;
    out.normal_x = final_normal.x;
    out.normal_y = final_normal.y;
    out.normal_z = final_normal.z;
    out.pad1 = 0.0;
    out.uv_x = src.uv_x;
    out.uv_y = src.uv_y;
    out.pad2 = 0.0;
    out.pad3 = 0.0;

    dst_vertices[idx] = out;
}
