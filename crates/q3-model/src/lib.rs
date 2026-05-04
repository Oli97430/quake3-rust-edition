//! Chargeur de modèles **MD3** (Quake 3 format).
//!
//! Un MD3 est un modèle skinné *par frame* (morph target animation) composé de
//! plusieurs *surfaces*, chacune avec son propre shader. Utilisé pour les
//! joueurs, armes, items et décors animés.
//!
//! # Endianness
//!
//! Les fichiers MD3 sont **little-endian** (Q3 est sorti sur x86).  Le
//! parseur utilise `bytemuck::cast_slice` qui hérite de l'endianness host
//! — on assert compile-time que la cible est LE.
//!
//! # Structure d'un fichier MD3
//!
//! ```text
//! MD3Header
//!   frames[num_frames]      — bounding box + position pour chaque frame
//!   tags[num_frames * num_tags] — points d'attachement (arme→main, tête→cou…)
//!   surfaces[num_surfaces]  — chaque surface contient :
//!     - shaders[num_shaders]
//!     - triangles[num_triangles]
//!     - st[num_verts]                — UVs (constants)
//!     - xyz_normals[num_frames * num_verts] — positions+normales par frame
//! ```
//!
//! Les positions sont stockées en `i16` avec facteur `1/64` (unités Q3).
//! Les normales sont packées en lat/lon sur 2 octets.
//!
//! # Améliorations vs C original
//!
//! * **Parse zero-copy** via `bytemuck` — pas de `memcpy` manuel.
//! * **Bounds-checked** : tout accès hors du buffer retourne `Err` au lieu
//!   d'un segfault.
//! * **API idiomatique** : `Md3::parse(&bytes) -> Result<Md3>`.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

const _: () = assert!(
    cfg!(target_endian = "little"),
    "q3-model: target host doit être little-endian (Q3 MD3 = LE)"
);

pub mod raw;
pub mod glb;
pub use glb::{GlbMesh, GlbVertex, GlbError};

use bytemuck::try_cast_slice;
use q3_common::{Error, Result};
use q3_math::{Mat4, Vec3, Vec4};
use raw::{
    Md3Frame, Md3Header, Md3Shader, Md3Surface, Md3Tag, Md3Triangle, Md3XyzNormal, MD3_IDENT,
    MD3_VERSION,
};
use tracing::debug;

/// Facteur d'échelle des vertex XYZ (stockés en `i16`).
pub const MD3_XYZ_SCALE: f32 = 1.0 / 64.0;

/// Un modèle MD3 entièrement chargé en mémoire.
#[derive(Debug, Clone)]
pub struct Md3 {
    pub name: String,
    pub flags: i32,
    pub frames: Vec<Frame>,
    /// `tags[frame_index * num_tags + tag_index]`.
    pub tags: Vec<Tag>,
    pub num_tags: usize,
    pub surfaces: Vec<Surface>,
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub mins: Vec3,
    pub maxs: Vec3,
    pub local_origin: Vec3,
    pub radius: f32,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct Tag {
    pub name: String,
    pub origin: Vec3,
    /// Matrice 3×3 `[forward, left, up]`.
    pub axis: [Vec3; 3],
}

#[derive(Debug, Clone)]
pub struct Surface {
    pub name: String,
    pub shaders: Vec<String>,
    pub triangles: Vec<[u32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    /// `xyz_normals[frame * num_verts + vert]`.
    pub xyz_normals: Vec<Md3XyzNormal>,
    pub num_verts: usize,
    pub num_frames: usize,
}

impl Surface {
    /// Retourne la position décodée d'un vertex pour une frame donnée.
    #[inline]
    pub fn vertex_position(&self, frame: usize, vert: usize) -> Vec3 {
        let i = frame * self.num_verts + vert;
        let raw = &self.xyz_normals[i];
        Vec3::new(
            raw.xyz[0] as f32 * MD3_XYZ_SCALE,
            raw.xyz[1] as f32 * MD3_XYZ_SCALE,
            raw.xyz[2] as f32 * MD3_XYZ_SCALE,
        )
    }

    /// Retourne la normale décodée (lat/lon packés sur 16 bits).
    #[inline]
    pub fn vertex_normal(&self, frame: usize, vert: usize) -> Vec3 {
        let i = frame * self.num_verts + vert;
        decode_normal(self.xyz_normals[i].normal)
    }
}

impl Md3 {
    /// Parse un buffer MD3 complet.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let header: &Md3Header = bytes
            .get(..core::mem::size_of::<Md3Header>())
            .and_then(|s| bytemuck::try_from_bytes(s).ok())
            .ok_or_else(|| Error::Parse("MD3: header tronqué".into()))?;

        if header.ident != MD3_IDENT {
            return Err(Error::Parse(format!(
                "MD3: magic invalide {:?}",
                header.ident
            )));
        }
        if header.version != MD3_VERSION {
            return Err(Error::Parse(format!(
                "MD3: version {} non supportée (attendu {})",
                header.version, MD3_VERSION
            )));
        }

        let name = cstr(&header.name).to_string();
        let num_frames = header.num_frames as usize;
        let num_tags = header.num_tags as usize;
        let num_surfaces = header.num_surfaces as usize;

        // Frames
        let frames_bytes = slice_at::<Md3Frame>(
            bytes,
            header.ofs_frames as usize,
            num_frames,
            "frames",
        )?;
        let frames = frames_bytes
            .iter()
            .map(|f| Frame {
                mins: Vec3::from_array(f.mins),
                maxs: Vec3::from_array(f.maxs),
                local_origin: Vec3::from_array(f.local_origin),
                radius: f.radius,
                name: cstr(&f.name).to_string(),
            })
            .collect();

        // Tags : num_frames × num_tags
        let tags_bytes = slice_at::<Md3Tag>(
            bytes,
            header.ofs_tags as usize,
            num_frames * num_tags,
            "tags",
        )?;
        let tags = tags_bytes
            .iter()
            .map(|t| Tag {
                name: cstr(&t.name).to_string(),
                origin: Vec3::from_array(t.origin),
                axis: [
                    Vec3::from_array(t.axis[0]),
                    Vec3::from_array(t.axis[1]),
                    Vec3::from_array(t.axis[2]),
                ],
            })
            .collect();

        // Surfaces — chaînées par `ofs_end` depuis `ofs_surfaces`.
        let mut surfaces = Vec::with_capacity(num_surfaces);
        let mut surf_ofs = header.ofs_surfaces as usize;
        for _ in 0..num_surfaces {
            let surf_bytes = bytes
                .get(surf_ofs..)
                .ok_or_else(|| Error::Parse("MD3: offset surface invalide".into()))?;
            let surf_header: &Md3Surface = surf_bytes
                .get(..core::mem::size_of::<Md3Surface>())
                .and_then(|s| bytemuck::try_from_bytes(s).ok())
                .ok_or_else(|| Error::Parse("MD3: surface header tronqué".into()))?;
            if surf_header.ident != MD3_IDENT {
                return Err(Error::Parse(format!(
                    "MD3: surface magic invalide {:?}",
                    surf_header.ident
                )));
            }

            let s_num_frames = surf_header.num_frames as usize;
            let s_num_shaders = surf_header.num_shaders as usize;
            let s_num_verts = surf_header.num_verts as usize;
            let s_num_tris = surf_header.num_triangles as usize;

            // Les offsets internes sont relatifs au début de la surface.
            let shaders_abs = surf_ofs + surf_header.ofs_shaders as usize;
            let tris_abs = surf_ofs + surf_header.ofs_triangles as usize;
            let st_abs = surf_ofs + surf_header.ofs_st as usize;
            let xyz_abs = surf_ofs + surf_header.ofs_xyz_normal as usize;

            let shaders_raw = slice_at::<Md3Shader>(bytes, shaders_abs, s_num_shaders, "shaders")?;
            let shaders: Vec<String> =
                shaders_raw.iter().map(|s| cstr(&s.name).to_string()).collect();

            let tris_raw =
                slice_at::<Md3Triangle>(bytes, tris_abs, s_num_tris, "triangles")?;
            let triangles: Vec<[u32; 3]> = tris_raw
                .iter()
                .map(|t| [t.indexes[0] as u32, t.indexes[1] as u32, t.indexes[2] as u32])
                .collect();

            // UVs — stockés comme `[f32;2]`.
            let st_slice = slice_at_bytes(
                bytes,
                st_abs,
                s_num_verts * core::mem::size_of::<[f32; 2]>(),
                "st",
            )?;
            let st_cast: &[[f32; 2]] = try_cast_slice(st_slice)
                .map_err(|_| Error::Parse("MD3: st slice mal aligné".into()))?;
            let uvs = st_cast.to_vec();

            // Xyz/normals — `num_frames * num_verts`.
            let xyz_raw = slice_at::<Md3XyzNormal>(
                bytes,
                xyz_abs,
                s_num_frames * s_num_verts,
                "xyz_normals",
            )?;

            surfaces.push(Surface {
                name: cstr(&surf_header.name).to_string(),
                shaders,
                triangles,
                uvs,
                xyz_normals: xyz_raw.to_vec(),
                num_verts: s_num_verts,
                num_frames: s_num_frames,
            });

            surf_ofs += surf_header.ofs_end as usize;
        }

        debug!(
            "MD3 '{}' : {} frames, {} tags, {} surfaces",
            name, num_frames, num_tags, num_surfaces
        );

        Ok(Self {
            name,
            flags: header.flags,
            frames,
            tags,
            num_tags,
            surfaces,
        })
    }

    /// Trouve un tag par nom dans une frame donnée.
    pub fn tag(&self, frame: usize, name: &str) -> Option<&Tag> {
        if self.num_tags == 0 {
            return None;
        }
        let start = frame * self.num_tags;
        let end = start + self.num_tags;
        self.tags.get(start..end)?.iter().find(|t| t.name == name)
    }

    /// Matrice de transformation locale d'un tag, interpolée linéairement
    /// entre `frame_a` et `frame_b` avec le coefficient `lerp` (clampé dans
    /// `[0, 1]`).
    ///
    /// Colonnes : 0 = forward, 1 = left, 2 = up, 3 = translation.
    /// Retourne `None` si le tag est absent d'une des deux frames.
    pub fn tag_transform(
        &self,
        frame_a: usize,
        frame_b: usize,
        lerp: f32,
        name: &str,
    ) -> Option<Mat4> {
        let a = self.tag(frame_a, name)?;
        let b = self.tag(frame_b, name)?;
        let t = lerp.clamp(0.0, 1.0);
        let origin = lerp_vec3(a.origin, b.origin, t);
        let fwd = safe_normalize(lerp_vec3(a.axis[0], b.axis[0], t), Vec3::X);
        let left = safe_normalize(lerp_vec3(a.axis[1], b.axis[1], t), Vec3::Y);
        let up = safe_normalize(lerp_vec3(a.axis[2], b.axis[2], t), Vec3::Z);
        Some(Mat4::from_cols(
            Vec4::new(fwd.x, fwd.y, fwd.z, 0.0),
            Vec4::new(left.x, left.y, left.z, 0.0),
            Vec4::new(up.x, up.y, up.z, 0.0),
            Vec4::new(origin.x, origin.y, origin.z, 1.0),
        ))
    }
}

#[inline]
fn lerp_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t
}

#[inline]
fn safe_normalize(v: Vec3, fallback: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq > 1e-12 {
        v / len_sq.sqrt()
    } else {
        fallback
    }
}

fn slice_at<'a, T: bytemuck::Pod>(
    bytes: &'a [u8],
    offset: usize,
    count: usize,
    what: &'static str,
) -> Result<&'a [T]> {
    let size = core::mem::size_of::<T>();
    let end = offset
        .checked_add(size.checked_mul(count).ok_or_else(|| {
            Error::Parse(format!("MD3: débordement sur {} ({}*{})", what, size, count))
        })?)
        .ok_or_else(|| Error::Parse(format!("MD3: offset débordé pour {}", what)))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| Error::Parse(format!("MD3: {} hors-borne", what)))?;
    try_cast_slice(slice).map_err(|e| Error::Parse(format!("MD3: {} mal aligné ({})", what, e)))
}

fn slice_at_bytes<'a>(
    bytes: &'a [u8],
    offset: usize,
    len: usize,
    what: &'static str,
) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| Error::Parse(format!("MD3: offset débordé pour {}", what)))?;
    bytes
        .get(offset..end)
        .ok_or_else(|| Error::Parse(format!("MD3: {} hors-borne", what)))
}

fn cstr(bytes: &[u8]) -> &str {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end]).unwrap_or("")
}

/// Décode une normale MD3 packée en `u16` (format spec Q3) :
/// * **`lat`** (octet bas, byte 0) ∈ [0, 255] → angle zenith ∈ [0, π].
///   `lat = 0` → +Z (haut), `lat = 128` ≈ horizon, `lat = 255` ≈ -Z.
/// * **`lng`** (octet haut, byte 1) ∈ [0, 255] → azimuth ∈ [0, 2π].
///
/// Reconstruction sphérique : `(cos(lng)·sin(lat), sin(lng)·sin(lat), cos(lat))`.
///
/// Avant v0.9.5++ on traitait les deux comme [0, 2π] ce qui distordait les
/// normales (`lat=0` ne donnait pas `+Z` strict ; les valeurs ~horizon
/// étaient projetées sur le mauvais hémisphère).
fn decode_normal(packed: u16) -> Vec3 {
    let lat = (packed & 0xFF) as f32 * (core::f32::consts::PI / 255.0);
    let lng = ((packed >> 8) & 0xFF) as f32 * (core::f32::consts::TAU / 255.0);
    let (sl, cl) = lat.sin_cos();
    let (sg, cg) = lng.sin_cos();
    Vec3::new(cg * sl, sg * sl, cl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_normal_roundtrip_zenith() {
        // packed = 0 → zenith pure (+Z dans Q3).
        let n = decode_normal(0);
        assert!((n - Vec3::Z).length() < 1e-3, "n = {:?}", n);
    }

    #[test]
    fn decode_normal_horizon_ranges() {
        // lat ≈ π/2 (octet 128) → vecteur ~horizontal, |z| petit, longueur 1.
        let n = decode_normal(128);
        assert!(n.z.abs() < 0.05, "horizon |z| = {}", n.z);
        assert!((n.length() - 1.0).abs() < 1e-3, "len = {}", n.length());
    }

    #[test]
    fn decode_normal_unit_length_for_random_packings() {
        for &packed in &[0u16, 0xFFFFu16, 0x4080, 0x80C0, 0x1234] {
            let n = decode_normal(packed);
            assert!((n.length() - 1.0).abs() < 1e-3, "packed=0x{:04x} len={}", packed, n.length());
        }
    }

    #[test]
    fn header_size_is_108() {
        assert_eq!(core::mem::size_of::<Md3Header>(), 108);
    }

    #[test]
    fn frame_size_is_56() {
        assert_eq!(core::mem::size_of::<Md3Frame>(), 56);
    }

    #[test]
    fn tag_size_is_112() {
        assert_eq!(core::mem::size_of::<Md3Tag>(), 112);
    }

    #[test]
    fn surface_size_is_108() {
        assert_eq!(core::mem::size_of::<Md3Surface>(), 108);
    }

    #[test]
    fn shader_size_is_68() {
        assert_eq!(core::mem::size_of::<Md3Shader>(), 68);
    }

    #[test]
    fn triangle_size_is_12() {
        assert_eq!(core::mem::size_of::<Md3Triangle>(), 12);
    }

    #[test]
    fn xyz_normal_size_is_8() {
        assert_eq!(core::mem::size_of::<Md3XyzNormal>(), 8);
    }

    #[test]
    fn parse_rejects_bad_magic() {
        let bytes = vec![0u8; core::mem::size_of::<Md3Header>()];
        assert!(Md3::parse(&bytes).is_err());
    }

    /// Construit un Md3 synthétique avec 2 frames et un unique tag
    /// `tag_weapon` — uniquement pour les tests de `tag_transform`.
    fn fake_md3_two_frames(o0: Vec3, o1: Vec3) -> Md3 {
        let dummy = Frame {
            mins: Vec3::ZERO,
            maxs: Vec3::ZERO,
            local_origin: Vec3::ZERO,
            radius: 0.0,
            name: String::new(),
        };
        let t0 = Tag {
            name: "tag_weapon".into(),
            origin: o0,
            axis: [Vec3::X, Vec3::Y, Vec3::Z],
        };
        let t1 = Tag {
            name: "tag_weapon".into(),
            origin: o1,
            axis: [Vec3::X, Vec3::Y, Vec3::Z],
        };
        Md3 {
            name: "synthetic".into(),
            flags: 0,
            frames: vec![dummy.clone(), dummy],
            tags: vec![t0, t1],
            num_tags: 1,
            surfaces: Vec::new(),
        }
    }

    #[test]
    fn tag_transform_at_lerp_zero_is_frame_a() {
        let m = fake_md3_two_frames(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(100.0, 200.0, 300.0),
        );
        let mat = m.tag_transform(0, 1, 0.0, "tag_weapon").unwrap();
        let origin = mat.col(3).truncate();
        assert!((origin - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
    }

    #[test]
    fn tag_transform_lerp_half_is_midpoint() {
        let m = fake_md3_two_frames(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 20.0, 30.0));
        let mat = m.tag_transform(0, 1, 0.5, "tag_weapon").unwrap();
        let origin = mat.col(3).truncate();
        assert!((origin - Vec3::new(5.0, 10.0, 15.0)).length() < 1e-5);
    }

    #[test]
    fn tag_transform_missing_returns_none() {
        let m = fake_md3_two_frames(Vec3::ZERO, Vec3::ZERO);
        assert!(m.tag_transform(0, 1, 0.0, "tag_nonexistent").is_none());
    }

    #[test]
    fn tag_transform_axes_stay_orthonormal_after_lerp() {
        let m = fake_md3_two_frames(Vec3::ZERO, Vec3::ZERO);
        let mat = m.tag_transform(0, 1, 0.5, "tag_weapon").unwrap();
        let fwd = mat.col(0).truncate();
        let left = mat.col(1).truncate();
        let up = mat.col(2).truncate();
        assert!((fwd.length() - 1.0).abs() < 1e-5);
        assert!((left.length() - 1.0).abs() < 1e-5);
        assert!((up.length() - 1.0).abs() < 1e-5);
        assert!(fwd.dot(left).abs() < 1e-5);
        assert!(fwd.dot(up).abs() < 1e-5);
    }

    #[test]
    fn safe_normalize_zero_uses_fallback() {
        assert_eq!(safe_normalize(Vec3::ZERO, Vec3::Z), Vec3::Z);
    }

    #[test]
    fn safe_normalize_nonzero_unitises() {
        let n = safe_normalize(Vec3::new(5.0, 0.0, 0.0), Vec3::Y);
        assert!((n - Vec3::X).length() < 1e-6);
    }
}
