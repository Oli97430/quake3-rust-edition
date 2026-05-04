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

/// **Visibility / LOS abstraction** (v0.9.5) — découple le bot du
/// système de collision sous-jacent. Permet aux bots de tourner aussi
/// bien dans une map BSP classique que sur un terrain BR (heightmap).
///
/// Le seul service requis : un trace ray qui retourne `true` si le
/// segment `start → end` est clair (pas d'obstacle solide entre les
/// deux), `false` sinon.
///
/// Implémentations fournies :
/// * `&CollisionWorld` — utilise `trace_ray(MASK_SHOT)` du BSP
/// * Adapter terrain (côté engine) — utilise `Terrain::trace_ray`
pub trait LosWorld {
    /// Retourne `true` si la ligne de visée `start → end` n'est pas
    /// occultée par la géométrie monde.
    fn is_clear(&self, start: Vec3, end: Vec3) -> bool;
}

impl LosWorld for CollisionWorld {
    fn is_clear(&self, start: Vec3, end: Vec3) -> bool {
        let tr = self.trace_ray(start, end, Contents::MASK_SHOT);
        tr.fraction >= 0.999
    }
}

/// Permet de passer `&CollisionWorld` directement quand l'API attend
/// `&dyn LosWorld` sans wrapper supplémentaire.
impl<T: LosWorld + ?Sized> LosWorld for &T {
    fn is_clear(&self, start: Vec3, end: Vec3) -> bool {
        (**self).is_clear(start, end)
    }
}

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
    /// **Anti-stuck** — accumulateur de temps depuis le dernier
    /// déplacement significatif. Quand > [`STUCK_THRESHOLD_SEC`], le
    /// bot suspecte qu'il est bloqué contre un mur/obstacle et adopte
    /// une stratégie d'évasion (saut + strafe + abandon waypoint).
    pub stuck_timer: f32,
    /// Position au dernier check anti-stuck. Pour calculer le delta.
    pub last_check_pos: Vec3,
    /// Phase de strafe pour l'anti-stuck — alterne G/D pour se
    /// décoller d'un coin (pure période fixe, pas de besoin de RNG).
    pub stuck_strafe_phase: f32,
    /// **Strafe-jump phase** (G4a) — accumulateur de temps pour le
    /// pattern strafe-jump bot. Period ~0.8s pour le strafe G/D.
    pub strafe_phase: f32,
    /// Accumulateur jump périodique pendant le strafe-jump (0.6s).
    pub jump_phase: f32,
    /// **Combat strafe phase** (v0.9.5) — accumulateur pour alterner
    /// G/D en combat de manière cohérente (période ~0.7s) au lieu d'un
    /// coin-flip par frame qui produisait du jitter visuel et zéro
    /// déplacement net.
    pub combat_strafe_phase: f32,
    /// **Dodge** (v0.9.5) — fenêtre pendant laquelle le bot esquive en
    /// strafe-jump dur après avoir reçu un hit. Set par
    /// `notify_damage_taken()`. Pendant la fenêtre, le bot strafe à
    /// pleine amplitude perpendiculairement à l'ennemi + tente un
    /// jump → mimic l'esquive humaine après damage.
    pub dodge_until: f32,
    /// Direction du dodge courant (+1 droite / -1 gauche). Choisi à
    /// l'activation du dodge.
    pub dodge_dir: f32,
    /// Horloge interne du bot — incrémentée chaque tick.  Utilisée
    /// par les phases (strafe combat, dodge) qui ont besoin d'un t
    /// monotone sans dépendre de l'horloge engine.
    pub clock: f32,
    /// **Personnalité** (v0.9.5) — distance d'engagement préférée
    /// dérivée de [`BotPersonality::preferred_distance`].  Modifie le
    /// seuil close/mid/far du combat tick.
    pub preferred_distance: f32,
}

/// Temps sans avancer (secondes) au-delà duquel le bot considère qu'il
/// est bloqué et déclenche son comportement d'évasion.
pub const STUCK_THRESHOLD_SEC: f32 = 0.6;
/// Distance minimum (unités) pour considérer que le bot a "bougé"
/// dans la fenêtre du stuck timer.
pub const STUCK_MIN_TRAVEL: f32 = 16.0;
/// Durée du comportement d'évasion (secondes). Pendant cette fenêtre,
/// le bot saute + strafe au lieu d'avancer tout droit.
pub const UNSTUCK_DURATION: f32 = 0.7;

/// Durée d'un dodge déclenché par un hit reçu. Court par design : assez
/// long pour produire un mouvement net (~0.5 m de strafe), assez court
/// pour ne pas immobiliser le bot loin de la zone de combat.
pub const DODGE_DURATION: f32 = 0.45;

/// **Personnalités bot** (v0.9.5) — biais de comportement persistant
/// par bot, qui module la distance d'engagement préférée.  L'engine
/// passe la valeur via [`Bot::set_preferred_distance`] au spawn.
///
/// * Rusher : 220u — charge close-range, switch SG/RL si possible
/// * Sniper : 900u — garde la distance, prefère le Railgun
/// * Camper : 500u — tient sa zone, recule rarement
/// * Balanced : 400u — comportement par défaut
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotPersonality {
    Rusher,
    Sniper,
    Camper,
    Balanced,
}

impl BotPersonality {
    pub fn preferred_distance(self) -> f32 {
        match self {
            Self::Rusher => 220.0,
            Self::Sniper => 900.0,
            Self::Camper => 500.0,
            Self::Balanced => 400.0,
        }
    }
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
            stuck_timer: 0.0,
            last_check_pos: position,
            stuck_strafe_phase: 0.0,
            strafe_phase: 0.0,
            jump_phase: 0.0,
            combat_strafe_phase: 0.0,
            dodge_until: f32::NEG_INFINITY,
            dodge_dir: 1.0,
            clock: 0.0,
            preferred_distance: BotPersonality::Balanced.preferred_distance(),
        }
    }

    /// Setter pour la personnalité — appelé par l'engine au spawn pour
    /// distribuer les biais de combat.
    pub fn set_personality(&mut self, p: BotPersonality) {
        self.preferred_distance = p.preferred_distance();
    }

    /// **API engine** : à appeler quand le bot vient de prendre un hit.
    /// Active une fenêtre de dodge `DODGE_DURATION` pendant laquelle
    /// le bot strafe perpendiculairement + saute. Sans cette réaction,
    /// les bots restaient « plantés » sous le feu et étaient triviaux
    /// à laser.  La direction du dodge est alternée à chaque hit pour
    /// rendre l'esquive moins prévisible.
    pub fn notify_damage_taken(&mut self) {
        self.dodge_until = self.clock + DODGE_DURATION;
        // Flip la direction à chaque hit pour zigzaguer sous le feu
        // soutenu.
        self.dodge_dir = -self.dodge_dir;
        if self.dodge_dir.abs() < 0.5 {
            self.dodge_dir = 1.0;
        }
    }

    pub fn push_waypoint(&mut self, p: Vec3) {
        self.waypoints.push(p);
    }

    /// Avance d'un tick : met à jour l'état FSM et retourne la commande de
    /// ce frame. `dt` en secondes.  `world` est l'environnement LOS —
    /// passe `&CollisionWorld` (BSP) ou un adapter terrain.
    pub fn tick(&mut self, dt: f32, world: &dyn LosWorld) -> BotCmd {
        // Clock interne — utilisée par les phases de dodge / strafe combat.
        self.clock += dt;
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
            self.stuck_timer = 0.0;
            return BotCmd::default();
        }
        let desired_yaw = to.y.atan2(to.x).to_degrees();
        self.view_angles.yaw = turn_toward(self.view_angles.yaw, desired_yaw, self.turn_rate * dt);

        // **Bot strafe-jump** (G4a) — quand le bot a un long chemin
        // (>200u), il alterne strafe G/D toutes les 0.4s + jump
        // périodique pour gagner ~20-25% de vitesse via la mécanique
        // air-accel Q3. Mimic basique du strafe-jump joueur — pas
        // optimal mais visible et améliore le dynamisme des bots.
        let dist_to_wp = to.length();
        if dist_to_wp > 200.0 {
            // Phase de strafe : sin(t * pi / 0.4) → période 0.8s
            // (un cycle complet G→D→G).
            self.strafe_phase += dt;
            if self.strafe_phase > 10.0 {
                self.strafe_phase -= 10.0; // évite drift float long match
            }
            let side = (self.strafe_phase * std::f32::consts::PI / 0.4).sin();
            // Jump périodique toutes les 0.6s. `up_move > 0` → engine
            // déclenche le saut (le pmove gère le on_ground).
            self.jump_phase += dt;
            let do_jump = if self.jump_phase >= 0.6 {
                self.jump_phase = 0.0;
                true
            } else {
                false
            };
            return BotCmd {
                forward_move: 1.0,
                right_move: side * 0.6, // strafe partiel pour rester orienté wp
                up_move: if do_jump { 1.0 } else { 0.0 },
                view_angles: self.view_angles,
                fire: false,
            };
        }

        // **Anti-stuck** : check le déplacement effectué depuis le
        // dernier sample. Si le bot stagne (< STUCK_MIN_TRAVEL en
        // STUCK_THRESHOLD_SEC), on entre en mode évasion : saut +
        // strafe latéral + skip waypoint. Évite le bug "bot collé au
        // mur indéfiniment" caractéristique d'une IA waypoint naïve
        // sans navmesh.
        self.stuck_timer += dt;
        if self.stuck_timer >= STUCK_THRESHOLD_SEC {
            let traveled = (self.position - self.last_check_pos).length();
            if traveled < STUCK_MIN_TRAVEL {
                // Bot bloqué → drop le waypoint courant + active phase
                // évasion. Sans drop, il retournera taper le même mur
                // dès la fin de l'évasion.
                self.waypoints.remove(0);
                self.stuck_strafe_phase = UNSTUCK_DURATION;
                trace!(
                    "bot {} STUCK detected (traveled {traveled:.1}u) → drop wp + escape",
                    self.name
                );
            }
            self.stuck_timer = 0.0;
            self.last_check_pos = self.position;
        }

        // Mode évasion en cours ?
        if self.stuck_strafe_phase > 0.0 {
            self.stuck_strafe_phase = (self.stuck_strafe_phase - dt).max(0.0);
            // Strafe latéral alterné — sign du time pour avoir un
            // pattern G-D-G qui décolle d'un coin rentrant.
            let strafe = if self.stuck_strafe_phase > UNSTUCK_DURATION * 0.5 {
                1.0
            } else {
                -1.0
            };
            return BotCmd {
                forward_move: 0.5, // continue à pousser
                right_move: strafe,
                up_move: 1.0,      // jump pour décoller d'une marche
                view_angles: self.view_angles,
                fire: false,
            };
        }

        BotCmd {
            forward_move: 1.0,
            right_move: 0.0,
            up_move: 0.0,
            view_angles: self.view_angles,
            fire: false,
        }
    }

    fn tick_combat(&mut self, dt: f32, world: &dyn LosWorld) -> BotCmd {
        let Some(enemy) = self.target_enemy else {
            return BotCmd::default();
        };
        let to = enemy - self.position;
        if to.length_squared() < 1e-6 {
            return BotCmd::default();
        }
        let dist = to.length();
        let desired_yaw = to.y.atan2(to.x).to_degrees();
        let desired_pitch = (-to.z).atan2(to.truncate().length()).to_degrees();
        self.view_angles.yaw = turn_toward(self.view_angles.yaw, desired_yaw, self.turn_rate * dt);
        self.view_angles.pitch =
            turn_toward(self.view_angles.pitch, desired_pitch, self.turn_rate * dt);

        // Tir si LOS — trace ray vers l'ennemi, si fraction == 1 alors libre.
        let fire = world.is_clear(self.position, enemy);

        // **Avance / recule selon distance ET personnalité** (v0.9.5)
        // — la zone optimale est centrée sur `preferred_distance` :
        //   * dist > pd × 1.5 : avance pour fermer
        //   * dist < pd × 0.6 : recule pour préserver la range
        //   * sinon : strafe pur
        // Un Rusher (pd=220) reste agressif close, un Sniper (pd=900)
        // recule dès qu'on s'approche, etc.
        let pd = self.preferred_distance.max(50.0);
        let forward: f32 = if dist > pd * 1.5 {
            1.0
        } else if dist < pd * 0.6 {
            -0.6
        } else {
            0.0
        };

        // **Anti-stuck en combat** : même mécanisme que tick_roam.
        // Sans ça, un bot qui voit le joueur derrière un mur reste
        // collé au mur en strafant indéfiniment.
        self.stuck_timer += dt;
        if self.stuck_timer >= STUCK_THRESHOLD_SEC {
            let traveled = (self.position - self.last_check_pos).length();
            if traveled < STUCK_MIN_TRAVEL && forward.abs() > 0.1 {
                // Bloqué pendant un mouvement → escape: jump + strafe
                self.stuck_strafe_phase = UNSTUCK_DURATION;
                trace!(
                    "bot {} COMBAT-STUCK (traveled {traveled:.1}u) → escape",
                    self.name
                );
            }
            self.stuck_timer = 0.0;
            self.last_check_pos = self.position;
        }
        if self.stuck_strafe_phase > 0.0 {
            self.stuck_strafe_phase = (self.stuck_strafe_phase - dt).max(0.0);
            return BotCmd {
                forward_move: 1.0,
                right_move: if (self.stuck_strafe_phase / UNSTUCK_DURATION) > 0.5 {
                    1.0
                } else {
                    -1.0
                },
                up_move: 1.0,
                view_angles: self.view_angles,
                fire: false,
            };
        }

        // **Dodge** (v0.9.5) — si le bot a été touché récemment, il
        // strafe perpendiculairement à la ligne de tir + saute. Tire
        // toujours pendant le dodge si LOS clair → garde la pression
        // offensive. Sans cette réaction, les bots faisaient cible.
        if self.clock < self.dodge_until {
            return BotCmd {
                forward_move: forward * 0.5, // mouvement avant atténué pendant le dodge
                right_move: self.dodge_dir,  // strafe pleine amplitude
                up_move: 1.0,                // jump pour casser la mire
                view_angles: self.view_angles,
                fire,
            };
        }

        // **Combat strafe cohérent** — phase périodique 0.7s qui
        // alterne G/D au lieu d'un coin-flip par frame.  Avant v0.9.5,
        // `strafe_direction()` retournait un bit aléatoire chaque tick
        // → le bot tremblait sur place sans déplacement net. Avec une
        // phase, il fait des allers-retours visibles, donc plus dur à
        // toucher au railgun.
        self.combat_strafe_phase += dt;
        if self.combat_strafe_phase > 1.4 {
            self.combat_strafe_phase -= 1.4;
        }
        let strafe = if self.combat_strafe_phase < 0.7 { 1.0 } else { -1.0 };

        BotCmd {
            forward_move: forward,
            right_move: strafe,
            up_move: 0.0,
            view_angles: self.view_angles,
            fire,
        }
    }
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
