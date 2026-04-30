//! IA des bots — squelette.
//!
//! Le Q3 original utilise l'**AAS** (Area Awareness System) : un graphe de
//! zones pré-calculées par `bspc` qui permet au bot de planifier un chemin
//! en O(log n). Compiler bspc en Rust est un projet à part ; ce crate
//! fournit donc pour l'instant :
//!
//! * une FSM à 3 états (`Idle`, `Roam`, `Combat`),
//! * un générateur de commandes (`UserCmd`-like) à consommer par le
//!   système de mouvement,
//! * un champ de vision / ligne de tir via [`q3_collision::CollisionWorld`].
//!
//! La navigation réelle sera ajoutée quand l'AAS sera portée.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

use q3_collision::{CollisionWorld, Contents};
use q3_math::{Angles, Vec3};
use smallvec::SmallVec;
use tracing::trace;

/// Commande produite par le bot, consommable par `q3_game::PlayerMove`.
#[derive(Debug, Clone, Copy, Default)]
pub struct BotCmd {
    /// Axes de mouvement : `forward` en +x, `right` en +y, `up` pour jump.
    pub forward_move: f32,
    pub right_move: f32,
    pub up_move: f32,
    /// Angles désirés (la caméra suivra).
    pub view_angles: Angles,
    /// `true` si le bot tire ce tick.
    pub fire: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotState {
    Idle,
    Roam,
    Combat,
}

/// Niveau de difficulté du bot — tuning Q3 historique (`bot_minplayers`
/// + `g_spSkill`).  Chaque niveau module **la précision du tir**, **le
/// temps de réaction** à une nouvelle cible, et **le cooldown** entre
/// deux tirs.  `III` = équilibre de référence, `I` = "easy" (manque
/// beaucoup), `V` = "nightmare" (aim presque parfait).
///
/// On ne touche PAS à la vitesse de mouvement — la diff vient
/// exclusivement du combat pour rester lisible côté joueur.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotSkill {
    I,
    II,
    III,
    IV,
    V,
}

impl BotSkill {
    /// Parse depuis un entier 1..5 — utilisé par la commande
    /// console `addbot <name> <skill>` et la cvar `bot_skill`.
    /// Retourne `III` pour tout ce qui est hors plage (défaut raisonnable).
    pub fn from_int(n: i32) -> Self {
        match n {
            1 => Self::I,
            2 => Self::II,
            3 => Self::III,
            4 => Self::IV,
            5 => Self::V,
            _ => Self::III,
        }
    }

    /// Valeur 1..5 — pour affichage / logs / sérialisation.
    pub fn to_int(self) -> i32 {
        match self {
            Self::I => 1,
            Self::II => 2,
            Self::III => 3,
            Self::IV => 4,
            Self::V => 5,
        }
    }

    /// Dispersion ajoutée aux angles de visée, en degrés (pic gaussien).
    /// Plus c'est grand, plus le bot "rate".  Appliqué par tick de tir
    /// côté engine.  Valeurs calées pour qu'à 5 le bot ait un aim de
    /// sniper (~1° d'écart max), et à 1 il spray dans les 10° — miss
    /// reste fréquent mais le joueur sait qu'il est visé.
    pub fn aim_error_deg(self) -> f32 {
        match self {
            Self::I => 10.0,
            Self::II => 6.0,
            Self::III => 3.5,
            Self::IV => 1.8,
            Self::V => 0.8,
        }
    }

    /// Délai entre apercevoir l'ennemi et commencer à tirer, en
    /// secondes.  Simule le temps de réaction humain — un bot V réagit
    /// à 100 ms, un bot I met presque une demi-seconde.
    pub fn reaction_time_sec(self) -> f32 {
        match self {
            Self::I => 0.45,
            Self::II => 0.30,
            Self::III => 0.20,
            Self::IV => 0.13,
            Self::V => 0.08,
        }
    }

    /// Multiplicateur appliqué au cooldown hitscan.  Un bot I est plus
    /// lent à relâcher un tir (mult > 1.0), un bot V est plus agressif
    /// (mult < 1.0).  On garde la fenêtre [0.7, 1.5] pour que ça reste
    /// perceptible sans casser le feel (un bot qui spamme à 4× est
    /// frustrant, pas difficile).
    pub fn fire_cooldown_mult(self) -> f32 {
        match self {
            Self::I => 1.50,
            Self::II => 1.20,
            Self::III => 1.00,
            Self::IV => 0.85,
            Self::V => 0.70,
        }
    }
}

impl Default for BotSkill {
    fn default() -> Self {
        Self::III
    }
}

#[derive(Debug, Clone)]
pub struct Bot {
    pub name: String,
    pub state: BotState,
    pub position: Vec3,
    pub view_angles: Angles,
    /// Cibles connues (derniers points d'intérêt).
    pub waypoints: SmallVec<[Vec3; 8]>,
    pub target_enemy: Option<Vec3>,
    /// Accumulateur d'orientation : on tourne la tête vers l'ennemi à cette
    /// vitesse max (deg/s).
    pub turn_rate: f32,
    /// Niveau de difficulté — affecte précision, réaction, cooldown.
    /// Consommé côté engine dans `tick_bots`.
    pub skill: BotSkill,
}

impl Bot {
    pub fn new(name: impl Into<String>, position: Vec3) -> Self {
        Self::with_skill(name, position, BotSkill::default())
    }

    pub fn with_skill(
        name: impl Into<String>,
        position: Vec3,
        skill: BotSkill,
    ) -> Self {
        Self {
            name: name.into(),
            state: BotState::Idle,
            position,
            view_angles: Angles::ZERO,
            waypoints: SmallVec::new(),
            target_enemy: None,
            turn_rate: 540.0,
            skill,
        }
    }

    pub fn push_waypoint(&mut self, p: Vec3) {
        self.waypoints.push(p);
    }

    /// Avance d'un tick : met à jour l'état FSM et retourne la commande de
    /// ce frame. `dt` en secondes.
    pub fn tick(&mut self, dt: f32, world: &CollisionWorld) -> BotCmd {
        // Transition d'état simple.
        self.state = match (self.state, self.target_enemy, self.waypoints.is_empty()) {
            (_, Some(_), _) => BotState::Combat,
            (_, None, false) => BotState::Roam,
            _ => BotState::Idle,
        };
        trace!("bot {} state={:?}", self.name, self.state);

        match self.state {
            BotState::Idle => BotCmd::default(),
            BotState::Roam => self.tick_roam(dt),
            BotState::Combat => self.tick_combat(dt, world),
        }
    }

    fn tick_roam(&mut self, dt: f32) -> BotCmd {
        let Some(&wp) = self.waypoints.first() else {
            return BotCmd::default();
        };
        let to = wp - self.position;
        if to.length_squared() < (32.0f32).powi(2) {
            self.waypoints.remove(0);
            return BotCmd::default();
        }
        let desired_yaw = to.y.atan2(to.x).to_degrees();
        self.view_angles.yaw = turn_toward(self.view_angles.yaw, desired_yaw, self.turn_rate * dt);
        BotCmd {
            forward_move: 1.0,
            right_move: 0.0,
            up_move: 0.0,
            view_angles: self.view_angles,
            fire: false,
        }
    }

    fn tick_combat(&mut self, dt: f32, world: &CollisionWorld) -> BotCmd {
        let Some(enemy) = self.target_enemy else {
            return BotCmd::default();
        };
        let to = enemy - self.position;
        // Garde contre `atan2(0, 0)` si l'ennemi est exactement sur la
        // position du bot (collision corps-à-corps, téléport collé).
        // Le résultat serait 0° techniquement, mais on sort plutôt pour
        // ne pas claquer des commandes dégénérées (et on laissera le
        // prochain tick avec un ennemi qui aura bougé).
        if to.length_squared() < 1e-6 {
            return BotCmd::default();
        }
        let desired_yaw = to.y.atan2(to.x).to_degrees();
        let desired_pitch = (-to.z).atan2(to.truncate().length()).to_degrees();
        self.view_angles.yaw = turn_toward(self.view_angles.yaw, desired_yaw, self.turn_rate * dt);
        self.view_angles.pitch =
            turn_toward(self.view_angles.pitch, desired_pitch, self.turn_rate * dt);

        // Tir si LOS — trace ray vers l'ennemi, si fraction == 1 alors libre.
        let trace = world.trace_ray(self.position, enemy, Contents::MASK_SHOT);
        let fire = trace.fraction >= 0.999;

        BotCmd {
            forward_move: 0.0,
            right_move: strafe_direction(dt),
            up_move: 0.0,
            view_angles: self.view_angles,
            fire,
        }
    }
}

/// Oscillation gauche/droite pour faire du "circle strafing" basique.
fn strafe_direction(dt: f32) -> f32 {
    // Un sinus pseudo-temporel — pour un vrai bot on ferait un RNG seedé par
    // tick, mais ce stub suffit à démontrer l'intégration.
    let _ = dt;
    if fastrand_bit() { 1.0 } else { -1.0 }
}

fn fastrand_bit() -> bool {
    // mini LCG — juste pour produire qqch de pseudo-aléatoire sans dépendance.
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u32> = const { Cell::new(0xDEADBEEF) };
    }
    STATE.with(|s| {
        let mut v = s.get();
        v = v.wrapping_mul(1664525).wrapping_add(1013904223);
        s.set(v);
        v & 1 == 1
    })
}

/// Tourne `current` vers `target` en respectant `max_step` (en degrés).
pub fn turn_toward(current: f32, target: f32, max_step: f32) -> f32 {
    let diff = q3_math::angle_subtract(current, target);
    let step = diff.clamp(-max_step, max_step);
    q3_math::normalize_180(current + step)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_converges() {
        // à 10°/step, on met 9 steps pour passer de 0 à 90.
        let mut a = 0.0;
        for _ in 0..9 {
            a = turn_toward(a, 90.0, 10.0);
        }
        assert!((a - 90.0).abs() < 1e-4, "a = {a}");
    }

    #[test]
    fn bot_starts_idle() {
        let b = Bot::new("test", Vec3::ZERO);
        assert_eq!(b.state, BotState::Idle);
    }

    #[test]
    fn skill_roundtrip() {
        for n in 1..=5 {
            assert_eq!(BotSkill::from_int(n).to_int(), n);
        }
        // Hors plage → III (défaut).
        assert_eq!(BotSkill::from_int(99), BotSkill::III);
        assert_eq!(BotSkill::from_int(0), BotSkill::III);
        assert_eq!(BotSkill::from_int(-1), BotSkill::III);
    }

    #[test]
    fn skill_aim_error_monotonic() {
        // Plus on monte en difficulté, plus l'aim est précis.
        assert!(BotSkill::I.aim_error_deg() > BotSkill::II.aim_error_deg());
        assert!(BotSkill::II.aim_error_deg() > BotSkill::III.aim_error_deg());
        assert!(BotSkill::III.aim_error_deg() > BotSkill::IV.aim_error_deg());
        assert!(BotSkill::IV.aim_error_deg() > BotSkill::V.aim_error_deg());
    }

    #[test]
    fn skill_cooldown_mult_monotonic() {
        // Plus on monte en difficulté, plus le bot relâche vite.
        assert!(BotSkill::I.fire_cooldown_mult() > BotSkill::V.fire_cooldown_mult());
    }

    #[test]
    fn skill_reaction_time_monotonic() {
        // Plus on monte, plus le bot réagit vite.
        assert!(BotSkill::I.reaction_time_sec() > BotSkill::V.reaction_time_sec());
    }
}
