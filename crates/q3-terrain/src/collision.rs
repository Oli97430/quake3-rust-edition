//! Trace heightfield — équivalent moral de `q3_collision::trace_ray` mais
//! sur un terrain heightmap.  Permet aux balles, projectiles, joueurs
//! et bots de collider sur le terrain BR sans BSP.
//!
//! # Algorithme
//!
//! On marche le long du rayon par incréments de `units_per_sample / 2`
//! (Nyquist sur la heightmap) et on compare la hauteur du rayon avec
//! la hauteur du terrain au point. Premier point où `ray_z < terrain_z`
//! → impact, on raffine en bisection sur le segment précédent pour
//! une précision sub-sample.
//!
//! C'est O(distance / step_size), donc une trace de 8192 unités sur
//! `units_per_sample = 12` fait ~1300 itérations — acceptable côté
//! engine pour un tir hitscan, mais on évitera de l'utiliser pour la
//! détection de tous les pas du joueur (où le BSP collide reste ok).

use q3_math::Vec3;

use crate::Terrain;

/// Résultat d'une trace heightfield.  Compatible structurellement avec
/// `q3_collision::Trace` pour qu'on puisse les unifier côté engine
/// (la fraction et la normale sont les seuls champs vraiment
/// consommés ailleurs).
#[derive(Debug, Clone, Copy)]
pub struct TerrainTrace {
    /// `[0..1]` — 1.0 = pas d'impact (rayon traverse l'air complètement).
    pub fraction: f32,
    /// Normale au point d'impact (unit vector).  `Z+` quand le tir
    /// vient du ciel, ~horizontale sur une falaise.  Pour un miss,
    /// vaut `Vec3::Z` (normale par défaut).
    pub plane_normal: Vec3,
    /// Position monde de l'impact.
    pub end_pos: Vec3,
}

impl TerrainTrace {
    pub fn miss(end: Vec3) -> Self {
        Self {
            fraction: 1.0,
            plane_normal: Vec3::Z,
            end_pos: end,
        }
    }
}

impl Terrain {
    /// Trace un rayon `start → end` contre la heightmap. Retourne le
    /// premier point sub-terrain. Si le rayon ne descend jamais sous
    /// la surface, retourne un miss.
    pub fn trace_ray(&self, start: Vec3, end: Vec3) -> TerrainTrace {
        let dir = end - start;
        let len = dir.length();
        if len < 1e-3 {
            return TerrainTrace::miss(end);
        }
        // Step taille = demi-sample pour respecter Nyquist horizontal.
        let step = (self.meta.units_per_sample * 0.5).max(1.0);
        let n_steps = (len / step).ceil() as usize;
        let n_steps = n_steps.max(1).min(8192); // cap dur de sécurité

        let mut prev_t = 0.0_f32;
        let mut prev_above = self.is_above_terrain(start);
        for i in 1..=n_steps {
            let t = (i as f32 / n_steps as f32).min(1.0);
            let p = start + dir * t;
            let above = self.is_above_terrain(p);
            if prev_above && !above {
                // Bisection sur [prev_t, t] pour préciser le point.
                let mut lo = prev_t;
                let mut hi = t;
                for _ in 0..16 {
                    let mid = 0.5 * (lo + hi);
                    let pmid = start + dir * mid;
                    if self.is_above_terrain(pmid) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let frac = 0.5 * (lo + hi);
                let hit = start + dir * frac;
                return TerrainTrace {
                    fraction: frac,
                    plane_normal: self.normal_at(hit.x, hit.y),
                    end_pos: hit,
                };
            }
            prev_t = t;
            prev_above = above;
        }
        TerrainTrace::miss(end)
    }

    fn is_above_terrain(&self, p: Vec3) -> bool {
        p.z >= self.height_at(p.x, p.y)
    }

    /// Normale du terrain au point monde (x, y) — calculée par sobel
    /// 3×3 sur la heightmap.  Propre et O(1).
    pub fn normal_at(&self, x: f32, y: f32) -> Vec3 {
        let s = self.meta.units_per_sample;
        let dz_dx = (self.height_at(x + s, y) - self.height_at(x - s, y)) / (2.0 * s);
        let dz_dy = (self.height_at(x, y + s) - self.height_at(x, y - s)) / (2.0 * s);
        // Normale = (-dz/dx, -dz/dy, 1) puis renorm.
        let n = Vec3::new(-dz_dx, -dz_dy, 1.0);
        let len = n.length().max(1e-6);
        n / len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::TerrainMeta;
    use crate::Terrain;

    fn flat_terrain(z: f32) -> Terrain {
        let w = 16;
        let h = 16;
        // Heightmap "tout plat" : on calcule la valeur u16 qui correspond à z.
        let z_min = -100.0;
        let z_max = 1000.0;
        let s = ((z - z_min) / (z_max - z_min) * 65535.0) as u16;
        Terrain {
            width: w,
            height: h,
            samples: vec![s; w * h],
            splat: vec![[255, 0, 0, 0]; w * h],
            meta: TerrainMeta {
                name: "flat".into(),
                width: w,
                height: h,
                z_min,
                z_max,
                origin_x: -800.0,
                origin_y: -800.0,
                units_per_sample: 100.0,
                ocean_z: -100.0,
                water_level: 0.0,
                pois: vec![],
            },
        }
    }

    #[test]
    fn trace_horizontal_ray_above_terrain_misses() {
        let t = flat_terrain(0.0);
        let tr = t.trace_ray(Vec3::new(-500.0, 0.0, 200.0), Vec3::new(500.0, 0.0, 200.0));
        assert!((tr.fraction - 1.0).abs() < 1e-3, "fraction={}", tr.fraction);
    }

    #[test]
    fn trace_falling_ray_hits_terrain() {
        let t = flat_terrain(50.0);
        let tr = t.trace_ray(Vec3::new(0.0, 0.0, 500.0), Vec3::new(0.0, 0.0, -500.0));
        assert!(tr.fraction < 1.0, "fraction={}", tr.fraction);
        // Doit toucher autour de z=50.
        assert!((tr.end_pos.z - 50.0).abs() < 5.0);
    }

    #[test]
    fn flat_terrain_normal_is_up() {
        let t = flat_terrain(100.0);
        let n = t.normal_at(0.0, 0.0);
        assert!((n.z - 1.0).abs() < 1e-3);
        assert!(n.x.abs() < 1e-3);
        assert!(n.y.abs() < 1e-3);
    }
}
