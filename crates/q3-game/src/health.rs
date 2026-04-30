//! Santé / dégâts — module minimaliste pour faire tourner la boucle de
//! combat. Pas d'armor ni de resistances pour l'instant.
//!
//! Q3 modélise `health` + `armor` + types de dégâts (splash, falling…) —
//! on ne garde ici que le strict nécessaire au MVP. L'armor viendra avec
//! le système de pickups.

use tracing::debug;

/// Jauge de vie numérique. `max` peut dépasser 100 via des megahealth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Health {
    pub current: i32,
    pub max: i32,
}

impl Health {
    pub const DEFAULT_MAX: i32 = 100;

    pub fn new(max: i32) -> Self {
        Self {
            current: max,
            max,
        }
    }

    pub fn full() -> Self {
        Self::new(Self::DEFAULT_MAX)
    }

    pub fn is_dead(self) -> bool {
        self.current <= 0
    }

    pub fn is_full(self) -> bool {
        self.current >= self.max
    }

    /// Applique des dégâts positifs. Retourne la quantité effectivement
    /// retirée (peut être < `amount` si la santé est déjà faible).
    pub fn take_damage(&mut self, amount: i32) -> i32 {
        if amount <= 0 || self.is_dead() {
            return 0;
        }
        let applied = amount.min(self.current);
        self.current -= applied;
        debug!("damage: -{applied} HP → {}/{}", self.current, self.max);
        applied
    }

    /// Soigne jusqu'à `max`. Ignorée si déjà mort (use `respawn`).
    pub fn heal(&mut self, amount: i32) -> i32 {
        if amount <= 0 || self.is_dead() {
            return 0;
        }
        let applied = amount.min(self.max - self.current);
        self.current += applied;
        applied
    }

    /// Reset complet — utilisé au respawn.
    pub fn respawn(&mut self) {
        self.current = self.max;
    }
}

impl Default for Health {
    fn default() -> Self {
        Self::full()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_full_alive() {
        let h = Health::new(100);
        assert_eq!(h.current, 100);
        assert!(!h.is_dead());
        assert!(h.is_full());
    }

    #[test]
    fn damage_kills_when_exceeding_current() {
        let mut h = Health::new(50);
        let applied = h.take_damage(200);
        assert_eq!(applied, 50);
        assert_eq!(h.current, 0);
        assert!(h.is_dead());
    }

    #[test]
    fn damage_is_ignored_when_dead() {
        let mut h = Health::new(10);
        h.take_damage(100);
        assert_eq!(h.take_damage(50), 0);
    }

    #[test]
    fn heal_caps_at_max() {
        let mut h = Health::new(100);
        h.take_damage(40);
        assert_eq!(h.heal(200), 40);
        assert_eq!(h.current, 100);
    }

    #[test]
    fn respawn_restores_full() {
        let mut h = Health::new(100);
        h.take_damage(100);
        assert!(h.is_dead());
        h.respawn();
        assert!(!h.is_dead());
        assert_eq!(h.current, 100);
    }
}
