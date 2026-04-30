//! Structures binaires brutes MD3. Toutes `#[repr(C)]` et `bytemuck::Pod`.

use bytemuck::{Pod, Zeroable};

/// Magic bytes `IDP3`.
pub const MD3_IDENT: [u8; 4] = *b"IDP3";
/// Version officielle du format.
pub const MD3_VERSION: i32 = 15;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Md3Header {
    pub ident: [u8; 4],
    pub version: i32,
    pub name: [u8; 64],
    pub flags: i32,
    pub num_frames: i32,
    pub num_tags: i32,
    pub num_surfaces: i32,
    pub num_skins: i32,
    pub ofs_frames: i32,
    pub ofs_tags: i32,
    pub ofs_surfaces: i32,
    pub ofs_eof: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Md3Frame {
    pub mins: [f32; 3],
    pub maxs: [f32; 3],
    pub local_origin: [f32; 3],
    pub radius: f32,
    pub name: [u8; 16],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Md3Tag {
    pub name: [u8; 64],
    pub origin: [f32; 3],
    pub axis: [[f32; 3]; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Md3Surface {
    pub ident: [u8; 4],
    pub name: [u8; 64],
    pub flags: i32,
    pub num_frames: i32,
    pub num_shaders: i32,
    pub num_verts: i32,
    pub num_triangles: i32,
    pub ofs_triangles: i32,
    pub ofs_shaders: i32,
    pub ofs_st: i32,
    pub ofs_xyz_normal: i32,
    pub ofs_end: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Md3Shader {
    pub name: [u8; 64],
    pub shader_index: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Md3Triangle {
    pub indexes: [i32; 3],
}

/// Position `xyz` en unités Q3 × 64 (pour tenir en `i16`), normale packée
/// en 2 octets `(lat, lng)` sur 256 valeurs chacune.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Md3XyzNormal {
    pub xyz: [i16; 3],
    pub normal: u16,
}
