//! Types de sortie et d'état pour le trace de collision.

use super::Contents;
use q3_math::{Aabb, Vec3};

/// Résultat d'une trace.
///
/// * `fraction = 1.0` → pas d'impact, la trace atteint `end`.
/// * `fraction < 1.0` → impact à `start + (end - start) * fraction`, plan
///   d'impact dans `plane_normal` / `plane_dist`, brush dans `brush_index`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trace {
    pub fraction: f32,
    pub end_pos: Vec3,
    pub plane_normal: Vec3,
    pub plane_dist: f32,
    pub contents: Contents,
    pub brush_index: Option<u32>,
    /// `true` si `start` était déjà à l'intérieur d'un brush solide.
    pub start_solid: bool,
    /// `true` si tout le segment `start..end` était dans un solide.
    pub all_solid: bool,
}

impl Trace {
    /// Trace vierge arrivant à destination sans impact.
    pub const fn miss(end: Vec3) -> Self {
        Self {
            fraction: 1.0,
            end_pos: end,
            plane_normal: Vec3::ZERO,
            plane_dist: 0.0,
            contents: Contents::empty(),
            brush_index: None,
            start_solid: false,
            all_solid: false,
        }
    }

    #[inline]
    pub fn hit(&self) -> bool {
        self.fraction < 1.0
    }
}

/// Boîte englobant la trace, relative au point tracé (typiquement un player
/// hull). `mins` et `maxs` sont des **offsets** par rapport à `start`/`end`.
///
/// Pour une trace de rayon, utiliser [`TraceBox::POINT`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceBox {
    pub mins: Vec3,
    pub maxs: Vec3,
}

impl TraceBox {
    /// Boîte dégénérée — équivalent à un rayon ponctuel.
    pub const POINT: Self = Self {
        mins: Vec3::ZERO,
        maxs: Vec3::ZERO,
    };

    #[inline]
    pub const fn new(mins: Vec3, maxs: Vec3) -> Self {
        Self { mins, maxs }
    }

    /// Boîte symétrique de demi-taille `half` autour du point tracé.
    #[inline]
    pub fn symmetric(half: Vec3) -> Self {
        Self {
            mins: -half,
            maxs: half,
        }
    }

    /// Extension (toujours positive ou nulle) du hull dans la direction
    /// **opposée** à `normal`. Utilisée pour décaler le plan de manière
    /// à tester le trace comme s'il s'agissait d'un point dont la
    /// position a été « shiftée » vers le coin le plus PROCHE du brush.
    ///
    /// **BUG FIX critique** : la version originale prenait le coin
    /// `{maxs si n>=0 sinon mins}` (= max-along-+n), ce qui donnait
    /// l'extension du hull dans la direction sortante du brush. Or
    /// l'algorithme Q3 (cf. `CM_TraceThroughBrush` dans `cm_trace.c`)
    /// utilise le coin **anti-sign** = `{mins si n>=0 sinon maxs}` qui
    /// est le coin du hull le plus proche du brush. Pour un hull
    /// asymétrique (player Q3 : mins.z=-24, maxs.z=32) la différence
    /// est SIGNIFICATIVE — le bug rendait le `update_ground` faux et
    /// gelait tous les sauts (le push-out échouait infiniment).
    pub fn offset_for_plane(&self, normal: Vec3) -> f32 {
        let nx = if normal.x >= 0.0 { self.mins.x } else { self.maxs.x };
        let ny = if normal.y >= 0.0 { self.mins.y } else { self.maxs.y };
        let nz = if normal.z >= 0.0 { self.mins.z } else { self.maxs.z };
        -(normal.x * nx + normal.y * ny + normal.z * nz)
    }

    #[inline]
    pub fn is_point(&self) -> bool {
        self.mins == Vec3::ZERO && self.maxs == Vec3::ZERO
    }

    /// Centre de la boîte (utile pour les boîtes non symétriques).
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.mins + self.maxs) * 0.5
    }
}

impl Default for TraceBox {
    fn default() -> Self {
        Self::POINT
    }
}

/// État mutable d'un trace en cours — on le passe en `&mut` pendant la
/// descente récursive de l'arbre BSP.
pub(crate) struct TraceWork {
    pub start: Vec3,
    pub end: Vec3,
    pub bounds: TraceBox,
    pub mask: Contents,
    pub trace: Trace,
}

impl TraceWork {
    /// Early-out : est-ce que la boîte du trace intersecte la boîte du brush ?
    /// On forme l'AABB du segment `start..end` étendu par `bounds`, et on la
    /// teste contre `brush`.
    pub fn bounds_box_overlaps(&self, brush: &Aabb) -> bool {
        let seg_min = self.start.min(self.end) + self.bounds.mins;
        let seg_max = self.start.max(self.end) + self.bounds.maxs;
        seg_max.cmpge(brush.mins).all() && seg_min.cmple(brush.maxs).all()
    }
}
