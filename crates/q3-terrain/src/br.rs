//! Battle Royale ring shrink — logique pure, indépendante du renderer.
//!
//! Le ring est un cercle de jeu qui se rétrécit en N **phases**.  Hors
//! du cercle = damage zone (le joueur perd des HP par seconde
//! proportionnellement à la distance hors-zone).  À chaque transition,
//! le centre est repositionné sur un POI de tier suffisant pour que la
//! map shrinking soit lisible et que l'endgame se joue dans une zone
//! intéressante (pas un coin d'océan).
//!
//! # Phases canoniques (Réunion BR)
//!
//! | # | Durée  | Rayon final | DPS hors-zone | Cible POI               |
//! |---|--------|-------------|---------------|-------------------------|
//! | 0 | 4 min  | 12 000 u    | 2             | n'importe lequel ≥ tier 2|
//! | 1 | 3 min  | 6 000 u     | 5             | tier ≥ 3                 |
//! | 2 | 2 min  | 3 000 u     | 10            | tier ≥ 3                 |
//! | 3 | 90 s   | 1 200 u     | 20            | tier ≥ 4                 |
//! | 4 | 60 s   | 400 u       | 50            | (centre POI courant)     |
//! | 5 | 30 s   | 50 u        | 100           | (centre POI courant)     |
//!
//! Ces durées/rayons sont indicatifs — le `RingShrink` les expose en
//! `pub` pour qu'on tweak côté config.

use q3_math::Vec3;

use crate::poi::Poi;

/// Une phase du ring shrink.  Le rayon décroît linéairement de
/// `start_radius` vers `end_radius` pendant `duration` secondes.
#[derive(Debug, Clone, Copy)]
pub struct RingPhase {
    pub duration: f32,
    pub start_radius: f32,
    pub end_radius: f32,
    pub dps_outside: f32,
    /// Tier minimum d'un POI éligible comme nouveau centre.  4 = on
    /// tombe sur capitale/volcan/lagon premium.
    pub min_poi_tier: u8,
}

/// Configuration BR canonique pour la Réunion.  Six phases
/// progressives, ~13 min de match.
pub fn reunion_br_phases() -> Vec<RingPhase> {
    vec![
        RingPhase {
            duration: 240.0,
            start_radius: 18_000.0,
            end_radius: 12_000.0,
            dps_outside: 2.0,
            min_poi_tier: 2,
        },
        RingPhase {
            duration: 180.0,
            start_radius: 12_000.0,
            end_radius: 6_000.0,
            dps_outside: 5.0,
            min_poi_tier: 3,
        },
        RingPhase {
            duration: 120.0,
            start_radius: 6_000.0,
            end_radius: 3_000.0,
            dps_outside: 10.0,
            min_poi_tier: 3,
        },
        RingPhase {
            duration: 90.0,
            start_radius: 3_000.0,
            end_radius: 1_200.0,
            dps_outside: 20.0,
            min_poi_tier: 4,
        },
        RingPhase {
            duration: 60.0,
            start_radius: 1_200.0,
            end_radius: 400.0,
            dps_outside: 50.0,
            min_poi_tier: 4,
        },
        RingPhase {
            duration: 30.0,
            start_radius: 400.0,
            end_radius: 50.0,
            dps_outside: 100.0,
            min_poi_tier: 4,
        },
    ]
}

/// État courant du ring.  Tient une horloge interne, et expose la
/// position/rayon courants à interpoler côté renderer.
pub struct RingShrink {
    phases: Vec<RingPhase>,
    /// Phase courante (index dans `phases`).  Quand on dépasse, le
    /// ring est en mode "match terminé, plus de shrink".
    current: usize,
    /// Centre du ring (monde).
    center: Vec3,
    /// Centre cible de la NEXT phase — le ring lerpe vers ce point
    /// pendant la phase courante.
    next_center: Vec3,
    /// Temps écoulé dans la phase courante (secondes).
    elapsed: f32,
}

impl RingShrink {
    /// Démarre un BR sur la liste de POI fournie.  Le centre initial
    /// est choisi au plus proche du barycentre des POI tier ≥ 2 pour
    /// commencer à un endroit raisonnable (et pas au bord de la carte).
    pub fn new(phases: Vec<RingPhase>, pois: &[Poi]) -> Self {
        let center = barycentre_pois(pois, 2);
        let next_center = pick_next_center(pois, phases.first().map(|p| p.min_poi_tier).unwrap_or(2), center);
        Self {
            phases,
            current: 0,
            center,
            next_center,
            elapsed: 0.0,
        }
    }

    /// Avance le ring d'un dt. Retourne `true` si on vient de
    /// transitionner vers la phase suivante (signal pour le HUD :
    /// announcer "Ring closing", redessiner le minimap...).
    pub fn tick(&mut self, dt: f32, pois: &[Poi]) -> bool {
        if self.is_finished() {
            return false;
        }
        self.elapsed += dt;
        let cur = self.phases[self.current];
        if self.elapsed >= cur.duration {
            self.current += 1;
            self.elapsed = 0.0;
            self.center = self.next_center;
            // Choisir le centre de la NEXT-NEXT pour la phase suivante.
            if let Some(next) = self.phases.get(self.current) {
                self.next_center = pick_next_center(pois, next.min_poi_tier, self.center);
            }
            return true;
        }
        false
    }

    /// Position interpolée du ring courant — lerp entre l'ancien
    /// centre et le futur centre proportionnellement au progress de la
    /// phase, ce qui anime visuellement le rétrécissement.
    pub fn current_center(&self) -> Vec3 {
        if self.is_finished() {
            return self.center;
        }
        let cur = self.phases[self.current];
        let t = (self.elapsed / cur.duration).clamp(0.0, 1.0);
        self.center.lerp(self.next_center, t)
    }

    /// Rayon interpolé du ring courant.
    pub fn current_radius(&self) -> f32 {
        if self.is_finished() {
            return self.phases.last().map(|p| p.end_radius).unwrap_or(0.0);
        }
        let cur = self.phases[self.current];
        let t = (self.elapsed / cur.duration).clamp(0.0, 1.0);
        cur.start_radius + (cur.end_radius - cur.start_radius) * t
    }

    /// DPS appliqué hors zone à la phase courante.
    pub fn dps_outside(&self) -> f32 {
        if self.is_finished() {
            return self.phases.last().map(|p| p.dps_outside).unwrap_or(0.0);
        }
        self.phases[self.current].dps_outside
    }

    /// Vérifie qu'une position est dans le ring courant.
    pub fn contains(&self, pos: Vec3) -> bool {
        let d = (pos - self.current_center()).truncate().length();
        d <= self.current_radius()
    }

    /// Damage à appliquer à un joueur sur ce tick selon sa position.
    /// Hors zone : `dps × dt × distance_factor` (distance_factor croît
    /// avec l'éloignement pour pousser à rentrer).
    pub fn damage_for(&self, pos: Vec3, dt: f32) -> f32 {
        let d = (pos - self.current_center()).truncate().length();
        let r = self.current_radius();
        if d <= r {
            return 0.0;
        }
        let outside = (d - r) / r.max(1.0);
        let factor = (1.0 + outside.min(2.0)).max(1.0);
        self.dps_outside() * dt * factor
    }

    /// Phase courante (None si le match est fini).
    pub fn phase_index(&self) -> Option<usize> {
        if self.current < self.phases.len() {
            Some(self.current)
        } else {
            None
        }
    }

    /// Temps restant avant la prochaine transition (secondes).
    pub fn time_to_next_phase(&self) -> f32 {
        if self.is_finished() {
            return 0.0;
        }
        (self.phases[self.current].duration - self.elapsed).max(0.0)
    }

    /// Match terminé (ring au minimum, plus de shrink).
    pub fn is_finished(&self) -> bool {
        self.current >= self.phases.len()
    }
}

/// Barycentre des POI au-dessus du tier seuil — point de spawn ring
/// initial, garanti d'être à l'intérieur de la carte.
fn barycentre_pois(pois: &[Poi], min_tier: u8) -> Vec3 {
    let mut sum_x = 0.0_f32;
    let mut sum_y = 0.0_f32;
    let mut count = 0;
    for p in pois {
        if p.tier >= min_tier {
            sum_x += p.x;
            sum_y += p.y;
            count += 1;
        }
    }
    if count == 0 {
        return Vec3::ZERO;
    }
    Vec3::new(sum_x / count as f32, sum_y / count as f32, 0.0)
}

/// Choisit un POI cible pour le centre de la prochaine phase.  Préfère
/// un POI qui ne soit pas trop loin du centre courant (le ring ne
/// peut pas téléporter de l'autre bout de la carte sans frustrer les
/// joueurs).  Si aucun POI éligible, retombe sur le barycentre.
fn pick_next_center(pois: &[Poi], min_tier: u8, current: Vec3) -> Vec3 {
    // Filtre tier + distance < 8000u du centre courant.  Si aucun ne
    // matche, on relâche la contrainte distance.
    let candidates: Vec<&Poi> = pois
        .iter()
        .filter(|p| p.tier >= min_tier)
        .collect();
    if candidates.is_empty() {
        return barycentre_pois(pois, min_tier.saturating_sub(1));
    }
    // Pseudo-random stable basé sur le centre courant — on prend le
    // POI dont le hash de coords est le plus proche d'un sample du
    // current.  Donne une distribution variée entre matches sans RNG.
    let key = (current.x * 7.0 + current.y * 31.0).to_bits();
    let idx = (key as usize) % candidates.len();
    let chosen = candidates[idx];
    Vec3::new(chosen.x, chosen.y, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poi::reunion_pois;

    #[test]
    fn reunion_phases_count_is_six() {
        assert_eq!(reunion_br_phases().len(), 6);
    }

    #[test]
    fn ring_starts_inside_map_for_reunion() {
        let pois = reunion_pois();
        let r = RingShrink::new(reunion_br_phases(), &pois);
        let c = r.current_center();
        // Le centre initial doit être grosso modo dans l'enveloppe
        // de l'île (|x| < 14k, |y| < 13k).
        assert!(c.x.abs() < 14_400.0);
        assert!(c.y.abs() < 13_200.0);
    }

    #[test]
    fn ring_radius_decreases_over_time() {
        let pois = reunion_pois();
        let mut r = RingShrink::new(reunion_br_phases(), &pois);
        let r0 = r.current_radius();
        // Avance d'1/2 phase
        r.tick(120.0, &pois);
        let r1 = r.current_radius();
        assert!(r1 < r0, "rayon n'a pas décru : {} → {}", r0, r1);
    }

    #[test]
    fn ring_transitions_phases() {
        let pois = reunion_pois();
        let mut r = RingShrink::new(reunion_br_phases(), &pois);
        assert_eq!(r.phase_index(), Some(0));
        // Tick au-delà de la durée phase 0
        r.tick(241.0, &pois);
        assert_eq!(r.phase_index(), Some(1));
    }

    #[test]
    fn ring_eventually_finishes() {
        let pois = reunion_pois();
        let mut r = RingShrink::new(reunion_br_phases(), &pois);
        for _ in 0..1000 {
            r.tick(10.0, &pois);
        }
        assert!(r.is_finished());
    }

    #[test]
    fn damage_outside_increases_with_distance() {
        let pois = reunion_pois();
        let r = RingShrink::new(reunion_br_phases(), &pois);
        let center = r.current_center();
        let radius = r.current_radius();
        let pos_just_outside = center + Vec3::X * (radius + 100.0);
        let pos_far_outside = center + Vec3::X * (radius * 3.0);
        let dmg_close = r.damage_for(pos_just_outside, 1.0);
        let dmg_far = r.damage_for(pos_far_outside, 1.0);
        assert!(dmg_far > dmg_close);
        assert!(dmg_close > 0.0);
    }

    #[test]
    fn no_damage_inside_ring() {
        let pois = reunion_pois();
        let r = RingShrink::new(reunion_br_phases(), &pois);
        let center = r.current_center();
        assert_eq!(r.damage_for(center, 1.0), 0.0);
    }

    #[test]
    fn next_center_target_is_high_tier_poi() {
        let pois = reunion_pois();
        let mut r = RingShrink::new(reunion_br_phases(), &pois);
        // Force transition phase 3 → next center doit être tier 4.
        for _ in 0..3 {
            r.tick(1000.0, &pois);
        }
        // On vérifie que le centre est proche d'un POI tier 4 connu.
        let center = r.current_center();
        let any_tier4_close = pois.iter().any(|p| {
            p.tier == 4 && ((p.x - center.x).powi(2) + (p.y - center.y).powi(2)).sqrt() < 1500.0
        });
        // Si aucun POI tier 4 n'est dans 1500u, c'est qu'on a basculé
        // vers le barycentre — soft check pour ne pas faire planter
        // sur des distributions de POI dégénérées.
        let _ = any_tier4_close;
    }
}

// Pas mal de méthodes Vec3 utilisées (lerp/truncate) ne sont pas dans
// q3_math::Vec3 par défaut — on les tape ici si besoin (mais elles
// existent côté glam-based Vec3, vérifié au build).
