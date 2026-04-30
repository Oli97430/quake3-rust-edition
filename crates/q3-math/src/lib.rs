//! Math primitives et conventions Quake 3.
//!
//! Conventions de coordonnées Q3 (différent de la plupart des moteurs modernes) :
//!
//! * **X = avant (forward)**
//! * **Y = gauche (left)**
//! * **Z = haut (up)**
//!
//! Les angles sont stockés en **degrés** dans l'ordre `[pitch, yaw, roll]` :
//!
//! * `pitch` — rotation autour de Y (haut/bas, positif = regarder vers le bas)
//! * `yaw`   — rotation autour de Z (gauche/droite)
//! * `roll`  — rotation autour de X (inclinaison latérale)
//!
//! Unité de longueur : 1 unité = ~2.54 cm (approx, stackable 64 unités = 1 player height).

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

pub use glam::{
    BVec3, IVec2, IVec3, Mat3, Mat4, Quat, UVec2, UVec3, Vec2, Vec3, Vec3A, Vec4,
};

/// Facteur de conversion degré → radian.
pub const DEG2RAD: f32 = std::f32::consts::PI / 180.0;
/// Facteur de conversion radian → degré.
pub const RAD2DEG: f32 = 180.0 / std::f32::consts::PI;

/// Triplet `[pitch, yaw, roll]` en degrés.
///
/// Équivalent du `vec3_t angles` en C, mais explicitement typé pour éviter
/// de le confondre avec un `Vec3` de position.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Angles {
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
}

impl Angles {
    pub const ZERO: Self = Self { pitch: 0.0, yaw: 0.0, roll: 0.0 };

    #[inline]
    pub const fn new(pitch: f32, yaw: f32, roll: f32) -> Self {
        Self { pitch, yaw, roll }
    }

    /// Convertit ces angles en trois vecteurs de base orthonormés :
    /// `(forward, right, up)`.
    ///
    /// Reproduction fidèle de `AngleVectors()` de `q_math.c` du jeu original.
    pub fn to_vectors(self) -> Basis {
        let (sy, cy) = (self.yaw * DEG2RAD).sin_cos();
        let (sp, cp) = (self.pitch * DEG2RAD).sin_cos();
        let (sr, cr) = (self.roll * DEG2RAD).sin_cos();

        let forward = Vec3::new(cp * cy, cp * sy, -sp);
        let right = Vec3::new(
            -sr * sp * cy + cr * sy,
            -sr * sp * sy - cr * cy,
            -sr * cp,
        );
        let up = Vec3::new(
            cr * sp * cy + sr * sy,
            cr * sp * sy - sr * cy,
            cr * cp,
        );
        Basis { forward, right, up }
    }

    /// Matrice de rotation correspondant à ces angles (row-major, colonne = axe).
    pub fn to_mat3(self) -> Mat3 {
        let b = self.to_vectors();
        Mat3::from_cols(b.forward, b.right, b.up)
    }

    /// Quaternion équivalent (utile pour interpoler sans lock-gimbal).
    pub fn to_quat(self) -> Quat {
        Quat::from_euler(
            glam::EulerRot::ZYX,
            self.yaw * DEG2RAD,
            self.pitch * DEG2RAD,
            self.roll * DEG2RAD,
        )
    }

    /// Normalise chaque composante dans `[-180, 180]`.
    pub fn normalized(self) -> Self {
        Self {
            pitch: normalize_180(self.pitch),
            yaw: normalize_180(self.yaw),
            roll: normalize_180(self.roll),
        }
    }
}

/// Base orthonormée issue d'un triplet d'angles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Basis {
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}

/// Normalise un angle dans `[-180, 180]` degrés.
#[inline]
pub fn normalize_180(mut a: f32) -> f32 {
    a %= 360.0;
    if a > 180.0 {
        a -= 360.0;
    } else if a < -180.0 {
        a += 360.0;
    }
    a
}

/// Normalise un angle dans `[0, 360)` degrés.
#[inline]
pub fn normalize_360(mut a: f32) -> f32 {
    a %= 360.0;
    if a < 0.0 {
        a += 360.0;
    }
    a
}

/// Différence d'angle signée la plus courte `b - a`, dans `[-180, 180]`.
#[inline]
pub fn angle_subtract(a: f32, b: f32) -> f32 {
    normalize_180(b - a)
}

/// `dst = src + scale * dir` — équivalent de `VectorMA`.
#[inline]
pub fn vector_ma(src: Vec3, scale: f32, dir: Vec3) -> Vec3 {
    src + dir * scale
}

/// Boîte axis-aligned (AABB). Utilisée pour les bounds BSP et les entités.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Aabb {
    pub mins: Vec3,
    pub maxs: Vec3,
}

impl Aabb {
    pub const EMPTY: Self = Self {
        mins: Vec3::splat(f32::INFINITY),
        maxs: Vec3::splat(f32::NEG_INFINITY),
    };

    pub const fn new(mins: Vec3, maxs: Vec3) -> Self {
        Self { mins, maxs }
    }

    #[inline]
    pub fn center(self) -> Vec3 {
        (self.mins + self.maxs) * 0.5
    }

    #[inline]
    pub fn size(self) -> Vec3 {
        self.maxs - self.mins
    }

    #[inline]
    pub fn contains(self, p: Vec3) -> bool {
        p.cmpge(self.mins).all() && p.cmple(self.maxs).all()
    }

    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        self.maxs.cmpge(other.mins).all() && self.mins.cmple(other.maxs).all()
    }

    /// Étend la boîte pour inclure le point `p`.
    pub fn expand_to(&mut self, p: Vec3) {
        self.mins = self.mins.min(p);
        self.maxs = self.maxs.max(p);
    }

    /// Union de deux AABB.
    pub fn union(self, other: Self) -> Self {
        Self {
            mins: self.mins.min(other.mins),
            maxs: self.maxs.max(other.maxs),
        }
    }
}

impl Default for Aabb {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Plan défini par sa normale et sa distance à l'origine.
///
/// Équation : `dot(normal, p) = dist`. Un point `p` est devant le plan si
/// `dot(normal, p) - dist > 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Plane {
    pub normal: Vec3,
    pub dist: f32,
}

impl Plane {
    #[inline]
    pub const fn new(normal: Vec3, dist: f32) -> Self {
        Self { normal, dist }
    }

    /// Distance signée du point `p` au plan.
    #[inline]
    pub fn distance(self, p: Vec3) -> f32 {
        self.normal.dot(p) - self.dist
    }

    /// Côté sur lequel se trouve la boîte : `1` devant, `-1` derrière, `0` à cheval.
    pub fn box_on_side(self, aabb: Aabb) -> i32 {
        let c = aabb.center();
        let e = aabb.size() * 0.5;
        let r = e.x * self.normal.x.abs()
            + e.y * self.normal.y.abs()
            + e.z * self.normal.z.abs();
        let d = self.distance(c);
        if d - r > 0.0 {
            1
        } else if d + r < 0.0 {
            -1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_vectors_identity() {
        let b = Angles::ZERO.to_vectors();
        assert!((b.forward - Vec3::X).length() < 1e-5);
        assert!((b.right - (-Vec3::Y)).length() < 1e-5);
        assert!((b.up - Vec3::Z).length() < 1e-5);
    }

    #[test]
    fn angle_vectors_yaw_90() {
        // yaw 90° : on regarde vers +Y (gauche dans Q3 donc... vers "left axis")
        let b = Angles::new(0.0, 90.0, 0.0).to_vectors();
        assert!((b.forward - Vec3::Y).length() < 1e-5);
    }

    #[test]
    fn normalize_180_wraps() {
        assert!((normalize_180(270.0) - -90.0).abs() < 1e-4);
        assert!((normalize_180(-270.0) - 90.0).abs() < 1e-4);
        assert!((normalize_180(180.0) - 180.0).abs() < 1e-4);
    }

    #[test]
    fn aabb_contains() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        assert!(b.contains(Vec3::ZERO));
        assert!(!b.contains(Vec3::splat(2.0)));
    }

    #[test]
    fn plane_distance() {
        let p = Plane::new(Vec3::Z, 0.0);
        assert_eq!(p.distance(Vec3::new(0.0, 0.0, 5.0)), 5.0);
        assert_eq!(p.distance(Vec3::new(0.0, 0.0, -3.0)), -3.0);
    }
}
