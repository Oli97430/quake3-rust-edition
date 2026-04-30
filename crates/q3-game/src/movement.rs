//! Physique joueur — port simplifié de `bg_pmove.c`.
//!
//! Reproduit les **frictions** et **accélérations** de Quake 3 (qui
//! permettent le strafe-jump, rocket-jump, etc. — c'est la sensation de
//! mouvement qui fait la signature du jeu), puis clip le mouvement contre
//! la BSP via [`CollisionWorld::trace_box`], avec slide + step-up.

use q3_collision::{CollisionWorld, Contents, TraceBox};
use q3_math::{Angles, Vec3};

/// Paramètres physiques, identiques aux valeurs par défaut du jeu original.
#[derive(Debug, Clone, Copy)]
pub struct PhysicsParams {
    /// Accélération au sol (cvar `pm_accelerate`).
    pub accelerate: f32,
    /// Accélération en l'air (cvar `pm_airaccelerate`).
    pub air_accelerate: f32,
    /// Friction au sol (cvar `pm_friction`).
    pub friction: f32,
    /// Vitesse max de déplacement normale.
    pub max_speed: f32,
    /// Vitesse max en marche "walk" (Shift maintenu) — silencieuse,
    /// ~ `pm_walkspeed` dans Q3 mais on utilise une valeur qui reste
    /// « dynamique » en combat (160 est un compromis : inaudible à la
    /// cadence de pas mais on avance).
    pub walk_speed: f32,
    /// Vitesse max en crouch (Ctrl maintenu) — `pm_crouchspeed` = 100.
    pub crouch_speed: f32,
    /// Puissance du saut (delta Z instantané).
    pub jump_velocity: f32,
    /// Gravité (unités/s²).
    pub gravity: f32,
    /// Seuil de vitesse sous lequel la friction tue le mouvement d'un coup.
    pub stop_speed: f32,
    /// Multiplicateur de friction appliqué quand le joueur relâche WASD
    /// (aucun wish horizontal) sur le sol. La friction Q3 de base (6) laisse
    /// glisser ~350 ms entre `max_speed` et 0 — sensation « savon » pour un
    /// shooter compétitif moderne. En boostant la friction d'un facteur 3
    /// seulement quand l'input est nul, on conserve la courbe de
    /// décélération classique pendant le mouvement (clé du strafe-jump) tout
    /// en arrêtant net le joueur dès qu'il relâche les touches.
    pub stop_friction_mult: f32,
}

impl Default for PhysicsParams {
    fn default() -> Self {
        Self {
            accelerate: 10.0,
            air_accelerate: 1.0,
            friction: 6.0,
            max_speed: 320.0,
            walk_speed: 160.0,
            crouch_speed: 100.0,
            jump_velocity: 270.0,
            gravity: 800.0,
            stop_speed: 100.0,
            stop_friction_mult: 3.0,
        }
    }
}

/// État de mouvement d'un joueur.
#[derive(Debug, Clone, Copy)]
pub struct PlayerMove {
    pub origin: Vec3,
    pub velocity: Vec3,
    pub view_angles: Angles,
    pub on_ground: bool,
    /// `true` = joueur accroupi — hull raccourci en Z (maxs.z = 16 au
    /// lieu de 32). État "sticky" : on ne se relève que si le hull debout
    /// tient dans l'espace au-dessus, d'où la nécessité de le stocker sur
    /// le PlayerMove et non de le recalculer du cmd à chaque tick.
    pub crouching: bool,
}

impl PlayerMove {
    pub fn new(origin: Vec3) -> Self {
        Self {
            origin,
            velocity: Vec3::ZERO,
            view_angles: Angles::ZERO,
            on_ground: true,
            crouching: false,
        }
    }
}

/// Input utilisateur pour une frame (valeurs dans [-1, 1]).
#[derive(Debug, Clone, Copy, Default)]
pub struct MoveCmd {
    pub forward: f32,
    pub side: f32,
    pub up: f32,
    pub jump: bool,
    /// Ctrl maintenu : accroupi. Hull raccourci + vitesse max baissée.
    pub crouch: bool,
    /// Shift maintenu : marche lente — cap la vitesse à `walk_speed`, ce
    /// qui rend les pas silencieux dans le système footsteps de l'engine.
    pub walk: bool,
    pub delta_time: f32,
}

/// Applique la friction de Q3 (exponentielle par tick) avec un coefficient
/// explicite. L'appelant choisit `friction` = `params.friction` pendant un
/// mouvement actif (strafe-jump friendly) ou `params.friction * stop_mult`
/// quand le joueur relâche WASD pour couper la glisse résiduelle.
fn apply_friction(vel: Vec3, dt: f32, params: PhysicsParams, friction: f32) -> Vec3 {
    let speed = vel.length();
    if speed < 0.001 {
        return Vec3::ZERO;
    }
    let control = speed.max(params.stop_speed);
    let drop = control * friction * dt;
    let new_speed = (speed - drop).max(0.0);
    vel * (new_speed / speed)
}

/// Accélération "Quake" — clé du strafe-jump : on ne limite que la
/// projection de la vitesse actuelle sur la direction de wish, pas le module
/// total → on peut dépasser `max_speed` en combinant strafe + souris.
fn accelerate(vel: Vec3, wish_dir: Vec3, wish_speed: f32, accel: f32, dt: f32) -> Vec3 {
    let current = vel.dot(wish_dir);
    let add_speed = wish_speed - current;
    if add_speed <= 0.0 {
        return vel;
    }
    let accel_speed = (accel * dt * wish_speed).min(add_speed);
    vel + wish_dir * accel_speed
}

/// Demi-taille du player hull Q3 en unités de map (mins = -half, maxs = +half
/// sauf pour Z où le joueur "regarde" vers le haut : mins.z = -24, maxs.z = 32).
const HULL_MINS: Vec3 = Vec3::new(-15.0, -15.0, -24.0);
const HULL_MAXS: Vec3 = Vec3::new(15.0, 15.0, 32.0);
/// Hull accroupi — `maxs.z` passe de 32 à 16 (taille totale 40 vs 56).
/// Valeurs issues de `CROUCH_MAXS_2` dans `bg_public.h`.
const HULL_MAXS_CROUCH: Vec3 = Vec3::new(15.0, 15.0, 16.0);

/// Hauteur max d'une marche qu'on accepte de monter sans saut.
const STEP_HEIGHT: f32 = 18.0;
/// Pente minimum pour considérer un plan comme du "sol" (cos(45°) ≈ 0.7).
const MIN_GROUND_NORMAL_Z: f32 = 0.7;
/// Profondeur du trace vers le bas pour détecter le sol sous les pieds.
/// Valeur Q3 canonique (`bg_pmove.c`) — le bias de `SURFACE_CLIP_EPSILON`
/// dans notre `trace_box` reste inférieur à ça, donc on détecte bien le
/// sol même quand l'origine est posée à `fraction * end` (epsilon près).
const GROUND_CHECK_DEPTH: f32 = 0.25;
/// `(1 + overclip)` → on pousse légèrement au-delà du plan pour éviter les
/// recontacts éternels avec la même surface.
const OVERCLIP: f32 = 1.001;

/// Player hull Q3 — variante debout ou accroupie.
pub fn player_hull() -> TraceBox {
    TraceBox::new(HULL_MINS, HULL_MAXS)
}

/// Hull effectif pour un état donné. Utilisé par la collision : un joueur
/// accroupi passe sous des conduits de 40u de haut, ce qui serait
/// impossible avec le hull debout de 56u.
pub fn player_hull_for(crouching: bool) -> TraceBox {
    let maxs = if crouching { HULL_MAXS_CROUCH } else { HULL_MAXS };
    TraceBox::new(HULL_MINS, maxs)
}

/// Projette `v` sur le plan `normal`, en multipliant légèrement par
/// [`OVERCLIP`] pour éviter les collisions répétées contre la même surface.
fn clip_velocity(v: Vec3, normal: Vec3) -> Vec3 {
    let backoff = v.dot(normal) * OVERCLIP;
    if backoff < 0.0 {
        v - normal * backoff
    } else {
        // On ne clippe que si on se dirige VERS le plan (dot < 0).
        v
    }
}

impl PlayerMove {
    /// Un tick de physique **sans collision** — pour les tests unitaires ou
    /// les caméras free-fly.
    pub fn tick(&mut self, cmd: MoveCmd, params: PhysicsParams) {
        self.integrate_velocity(cmd, params);
        self.origin += self.velocity * cmd.delta_time;
    }

    /// Un tick de physique complet avec collision BSP — slide + step-up.
    pub fn tick_collide(
        &mut self,
        cmd: MoveCmd,
        params: PhysicsParams,
        world: &CollisionWorld,
    ) {
        let mask = Contents::MASK_PLAYERSOLID;

        // Transitions crouch : presse → accroupit immédiatement. Relâche →
        // tente de se relever si le hull debout tient dans l'espace
        // au-dessus (un bon vieux trick des maps Q3 : un conduit bas te
        // force à rester accroupi).
        if cmd.crouch {
            self.crouching = true;
        } else if self.crouching {
            let stand = player_hull_for(false);
            let tr = world.trace_box(self.origin, self.origin, stand, mask);
            if !tr.all_solid && tr.fraction >= 1.0 {
                self.crouching = false;
            }
        }
        let hull = player_hull_for(self.crouching);

        // Mise à jour de on_ground : petit trace vers le bas.
        self.update_ground(world, hull, mask);

        self.integrate_velocity(cmd, params);

        // Slide move : jusqu'à 4 itérations (si la vitesse se réoriente).
        let dt = cmd.delta_time;
        let mut time_left = dt;
        let mut hit_wall = false;
        let mut planes: smallvec::SmallVec<[Vec3; 5]> = smallvec::SmallVec::new();
        // **Step-up bug fix** : on capture l'état AVANT la boucle slide
        // pour le repasser à `try_step_up`. Sinon le step-up part de la
        // position post-collision (collée au riser de la marche) avec
        // une vélocité déjà clippée tangente au mur ⇒ le re-slide à
        // hauteur de marche n'avance plus, l'heuristique
        // `step_d > orig_d` rejette toujours, et le joueur est obligé
        // de sauter pour franchir une simple marche. Q3 vanilla appelle
        // ça `primal_velocity` dans `PM_SlideMove`.
        let primal_origin = self.origin;
        let primal_velocity = self.velocity;

        for _ in 0..4 {
            if time_left <= 0.0 {
                break;
            }
            let step = self.velocity * time_left;
            if step.length_squared() < 1e-8 {
                break;
            }
            let end = self.origin + step;
            let tr = world.trace_box(self.origin, end, hull, mask);

            if tr.all_solid {
                // Hull embedded dans un brush solide : classiquement le tick
                // suivant un atterrissage (l'epsilon de pénétration fait que
                // `start` est vu comme solide). Q3 original (bg_pmove.c
                // `PM_SlideMove`) n'annule que la vitesse verticale pour ne
                // pas accumuler la gravité ni les dégâts de chute, mais laisse
                // la vitesse horizontale intacte pour que le prochain tick
                // — après `update_ground` qui pousse hors du brush — puisse
                // bouger. Tuer toute la velocity ici clouait le joueur au sol.
                self.velocity.z = 0.0;
                break;
            }

            if tr.fraction > 0.0 {
                self.origin = tr.end_pos;
            }
            if tr.fraction >= 1.0 {
                break;
            }

            hit_wall = true;
            time_left *= 1.0 - tr.fraction;

            // Duplicate-plane : si on re-touche un plan déjà rencontré,
            // on évite de diverger en projetant sur lui.
            if planes.iter().any(|n| n.dot(tr.plane_normal) > 0.99) {
                self.velocity += tr.plane_normal;
                continue;
            }
            planes.push(tr.plane_normal);

            // Clip vitesse contre tous les plans touchés jusqu'ici.
            let mut clipped = self.velocity;
            for n in &planes {
                clipped = clip_velocity(clipped, *n);
            }
            self.velocity = clipped;

            // Vérifie qu'on ne « gratte » pas entre 2 plans
            // (coincé dans un coin).
            if planes.len() >= 2 {
                let mut still_blocked = false;
                for (i, n1) in planes.iter().enumerate() {
                    for n2 in planes.iter().skip(i + 1) {
                        if self.velocity.dot(*n1) < 0.0 && self.velocity.dot(*n2) < 0.0 {
                            // coin rentrant : stop.
                            self.velocity = Vec3::ZERO;
                            still_blocked = true;
                            break;
                        }
                    }
                    if still_blocked {
                        break;
                    }
                }
                if still_blocked {
                    break;
                }
            }
        }

        // Step-up : si on a tapé un mur pendant un mouvement horizontal, tente
        // de monter la marche et re-slider depuis là — en repartant de la
        // position et de la vélocité **avant** le slide pour préserver
        // l'élan vers la marche (cf. `primal_*`).
        if hit_wall && self.on_ground {
            self.try_step_up(
                cmd,
                params,
                world,
                hull,
                mask,
                primal_origin,
                primal_velocity,
            );
        }

        // Re-check ground après le move.
        self.update_ground(world, hull, mask);
    }

    fn integrate_velocity(&mut self, cmd: MoveCmd, params: PhysicsParams) {
        let basis = self.view_angles.to_vectors();

        // Wish velocity = intention du joueur dans le plan horizontal.
        // `cmd.forward` / `cmd.side` sont des fractions dans [-1, 1]
        // (bouton enfoncé = 1.0), qu'on scale par `max_speed` pour
        // obtenir une vitesse absolue cible. Sans ce scale, le `min(cap)`
        // ci-dessous résolvait à ~1 u/s (la magnitude brute de wish) et
        // le joueur n'avançait quasiment pas — cf. PM_CmdScale /
        // PM_ClipVelocity dans `bg_pmove.c`.
        let mut wish = basis.forward * cmd.forward + basis.right * cmd.side;
        wish.z = 0.0;
        wish *= params.max_speed;
        // Plafond de vitesse wish selon l'état : crouch (100) < walk (160)
        // < run (320). Crouch l'emporte sur walk — c'est le même ordre
        // de priorité que Q3. Jumping ne fige pas la vitesse wish car
        // l'accel air est déjà faible : le plafond de wish_speed pendant
        // l'envol cap la projection, donc un bunny-hop en crouch gardera
        // sa momentum mais n'accélérera qu'à 100 u/s.
        let cap = if self.crouching {
            params.crouch_speed
        } else if cmd.walk {
            params.walk_speed
        } else {
            params.max_speed
        };
        let wish_speed = wish.length().min(cap);
        let wish_dir = if wish.length() > 0.0001 {
            wish.normalize()
        } else {
            Vec3::ZERO
        };

        if self.on_ground {
            // Friction dynamique : boostée quand aucune touche de
            // déplacement n'est enfoncée — fait passer le temps d'arrêt
            // depuis `max_speed` d'environ 350 ms à ~110 ms tout en
            // préservant la courbe Q3 classique pendant un strafe-jump.
            let no_wish = wish_dir.length_squared() < 1e-6;
            let friction = if no_wish {
                params.friction * params.stop_friction_mult
            } else {
                params.friction
            };
            self.velocity = apply_friction(self.velocity, cmd.delta_time, params, friction);
            self.velocity = accelerate(
                self.velocity,
                wish_dir,
                wish_speed,
                params.accelerate,
                cmd.delta_time,
            );
            // Saut désactivé pendant le crouch — Q3 permet techniquement
            // le "duck jump" mais le résultat est une hauteur ridicule
            // et ça casse la lisibilité du mouvement. On aligne sur le
            // comportement par défaut de la plupart des mods compétitifs.
            if cmd.jump && !self.crouching {
                self.velocity.z = params.jump_velocity;
                self.on_ground = false;
            }
        } else {
            // Air control : même formule mais accel bien plus faible.
            self.velocity = accelerate(
                self.velocity,
                wish_dir,
                wish_speed,
                params.air_accelerate,
                cmd.delta_time,
            );
            self.velocity.z -= params.gravity * cmd.delta_time;
        }
    }

    fn update_ground(&mut self, world: &CollisionWorld, hull: TraceBox, mask: Contents) {
        let start = self.origin;
        let end = start - Vec3::Z * GROUND_CHECK_DEPTH;
        let tr = world.trace_box(start, end, hull, mask);

        // Cas pathologique : le hull chevauche déjà un brush solide
        // (`all_solid` ou `start_solid`). En pratique ça arrive systéma-
        // tiquement après un atterrissage, parce que le slide move final
        // dépose l'origine à un epsilon **sous** la surface (la fraction
        // retournée n'est jamais parfaitement 1.0, on pénètre de quelques
        // 0.0001u). Sans correction, `on_ground` restait à `false` forever
        // → le joueur passait en branche "air" dans `integrate_velocity`,
        // où `air_accelerate = 1.0` donne une poussée ridicule, et le slide
        // suivant re-voyait `all_solid` et annulait la vitesse. Résultat :
        // joueur cloué au sol, WASD inerte.
        //
        // Q3 original résout ça avec `PM_CorrectAllSolid` qui pousse le
        // joueur sur 1u dans différentes directions jusqu'à trouver une
        // position libre (bg_pmove.c). On fait la même chose ici, version
        // simple : on probe vers le haut par paliers de 0.5/1/2/4u.
        if tr.all_solid || tr.start_solid {
            // Pas de saut : on veut juste sortir de l'embedding d'epsilon.
            // Étapes fines (0.05u) d'abord, plus grossières ensuite pour
            // les cas où le joueur spawne réellement dans une poutre.
            for &up in &[0.05_f32, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0] {
                let probe_start = start + Vec3::Z * up;
                let probe_end = probe_start - Vec3::Z * GROUND_CHECK_DEPTH;
                let tr2 = world.trace_box(probe_start, probe_end, hull, mask);
                if !tr2.all_solid && !tr2.start_solid {
                    self.origin = probe_start;
                    self.on_ground = tr2.fraction < 1.0
                        && tr2.plane_normal.z >= MIN_GROUND_NORMAL_Z;
                    if self.on_ground && self.velocity.z < 0.0 {
                        self.velocity.z = 0.0;
                    }
                    return;
                }
            }
            // Push-out a échoué (coincé profond) — on considère quand même
            // qu'on est au sol pour permettre le mouvement horizontal au
            // tick suivant, plutôt que de rester stuck en air-mode.
            self.on_ground = true;
            if self.velocity.z < 0.0 {
                self.velocity.z = 0.0;
            }
            return;
        }

        self.on_ground =
            tr.fraction < 1.0 && tr.plane_normal.z >= MIN_GROUND_NORMAL_Z;
        // Si on est collé au sol et la vitesse verticale pointe vers le bas,
        // on la tue (sinon on accumule de la gravité pendant qu'on est posé).
        if self.on_ground && self.velocity.z < 0.0 {
            self.velocity.z = 0.0;
        }
    }

    fn try_step_up(
        &mut self,
        cmd: MoveCmd,
        params: PhysicsParams,
        world: &CollisionWorld,
        hull: TraceBox,
        mask: Contents,
        // État avant la boucle slide — la position et la vélocité que le
        // joueur avait à l'instant `T` quand il a tapé la marche, pas
        // après que la collision ait clippé sa course.
        primal_origin: Vec3,
        primal_velocity: Vec3,
    ) {
        let post_slide_origin = self.origin;
        let start_origin = primal_origin;
        let start_vel = primal_velocity;

        // 1) Trace UP de STEP_HEIGHT.
        let up = start_origin + Vec3::Z * STEP_HEIGHT;
        let up_tr = world.trace_box(start_origin, up, hull, mask);
        let up_origin = up_tr.end_pos;

        // 2) Ré-intègre le slide depuis cette position surélevée.
        let mut stepped = PlayerMove {
            origin: up_origin,
            velocity: start_vel,
            view_angles: self.view_angles,
            on_ground: false,
            crouching: self.crouching,
        };
        // On refait juste le slide (sans ré-intégrer la vitesse), en réutilisant
        // tick_collide… mais tick_collide ré-appliquerait la vitesse. On fait
        // donc un inline simplifié.
        let _ = params;
        let _ = cmd;
        let dt = cmd.delta_time;
        let mut time_left = dt;
        let mut planes: smallvec::SmallVec<[Vec3; 5]> = smallvec::SmallVec::new();
        for _ in 0..4 {
            if time_left <= 0.0 {
                break;
            }
            let step_vec = stepped.velocity * time_left;
            if step_vec.length_squared() < 1e-8 {
                break;
            }
            let end = stepped.origin + step_vec;
            let tr = world.trace_box(stepped.origin, end, hull, mask);
            if tr.all_solid {
                return; // step-up échoué : on garde le résultat du slide initial.
            }
            if tr.fraction > 0.0 {
                stepped.origin = tr.end_pos;
            }
            if tr.fraction >= 1.0 {
                break;
            }
            time_left *= 1.0 - tr.fraction;
            if planes.iter().any(|n| n.dot(tr.plane_normal) > 0.99) {
                stepped.velocity += tr.plane_normal;
                continue;
            }
            planes.push(tr.plane_normal);
            let mut v = stepped.velocity;
            for n in &planes {
                v = clip_velocity(v, *n);
            }
            stepped.velocity = v;
        }

        // 3) Trace DOWN pour reposer le joueur sur la marche.
        let down = stepped.origin - Vec3::Z * STEP_HEIGHT;
        let down_tr = world.trace_box(stepped.origin, down, hull, mask);
        if down_tr.plane_normal.z < MIN_GROUND_NORMAL_Z {
            // Pas de sol praticable là-haut : on garde le slide original.
            return;
        }
        stepped.origin = down_tr.end_pos;

        // Accepte le step-up seulement s'il nous fait avancer plus loin
        // que ce que la simple boucle slide a obtenu. On compare en
        // distance HORIZONTALE (XY) — sinon une montée verticale pure
        // gagnerait artificiellement contre un slide horizontal qui
        // glissait le long du mur.
        let post_xy = (post_slide_origin - start_origin).truncate();
        let step_xy = (stepped.origin - start_origin).truncate();
        if step_xy.length_squared() > post_xy.length_squared() {
            self.origin = stepped.origin;
            self.velocity = stepped.velocity;
        } else {
            // Step-up rejeté : on garde le résultat post-slide.
            self.origin = post_slide_origin;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_pulls_down_in_air() {
        let mut pm = PlayerMove::new(Vec3::new(0.0, 0.0, 100.0));
        pm.on_ground = false;
        let params = PhysicsParams::default();
        for _ in 0..10 {
            pm.tick(MoveCmd { delta_time: 0.1, ..Default::default() }, params);
        }
        assert!(pm.velocity.z < 0.0);
        assert!(pm.origin.z < 100.0);
    }

    #[test]
    fn friction_stops_on_ground() {
        let mut pm = PlayerMove::new(Vec3::ZERO);
        pm.velocity = Vec3::new(200.0, 0.0, 0.0);
        pm.on_ground = true;
        let params = PhysicsParams::default();
        for _ in 0..200 {
            pm.tick(MoveCmd { delta_time: 0.05, ..Default::default() }, params);
        }
        assert!(pm.velocity.length() < 0.1);
    }

    /// Régression : quand WASD est relâché, le joueur doit s'arrêter presque
    /// instantanément (< 150 ms) depuis la vitesse max.  Sans le boost de
    /// friction no-input, la décélération classique de Q3 traîne ~350 ms,
    /// ce qui donne une sensation « glacée » incompatible avec les
    /// standards modernes.
    #[test]
    fn stop_is_immediate_when_no_input() {
        let mut pm = PlayerMove::new(Vec3::ZERO);
        pm.on_ground = true;
        pm.velocity = Vec3::new(320.0, 0.0, 0.0);
        let params = PhysicsParams::default();
        // Simule 9 ticks de 1/60s = 150 ms sans aucun input.
        for _ in 0..9 {
            pm.tick(
                MoveCmd { delta_time: 1.0 / 60.0, ..Default::default() },
                params,
            );
        }
        assert!(
            pm.velocity.length() < 1.0,
            "joueur doit être à l'arrêt après 150 ms (actuel = {})",
            pm.velocity.length()
        );
    }

    /// Régression : le boost de friction no-input ne doit pas saboter le
    /// strafe-jump.  Pendant un déplacement actif, la décélération reste
    /// sur la courbe Q3 classique (friction = 6) → après 100 ms de
    /// strafe contre la direction de vitesse, on garde encore de la
    /// vitesse résiduelle (le mouvement perdurerait plusieurs centaines
    /// de ms dans Q3 original).
    #[test]
    fn strafe_preserves_classic_friction_curve() {
        let mut pm = PlayerMove::new(Vec3::ZERO);
        pm.on_ground = true;
        pm.velocity = Vec3::new(320.0, 0.0, 0.0);
        let params = PhysicsParams::default();
        // Input actif (forward = 1.0) — wish non-nul → friction classique.
        for _ in 0..6 {
            pm.tick(
                MoveCmd {
                    forward: 1.0,
                    delta_time: 1.0 / 60.0,
                    ..Default::default()
                },
                params,
            );
        }
        // Après ~100 ms d'input actif, on conserve > 250 u/s — la courbe
        // standard Q3 n'aurait pas eu le temps de freiner autant.
        assert!(
            pm.velocity.length() > 250.0,
            "strafe-jump curve cassée : vitesse tombée à {}",
            pm.velocity.length()
        );
    }

    /// Construit un collision world minimal : un cube solide de -16..16
    /// centré à l'origine. Copié du helper de q3-collision (pas exposé).
    fn cube_world() -> CollisionWorld {
        use q3_bsp::raw::{
            DBrush, DBrushSide, DLeaf, DModel, DNode, DPlane, DShader, DSurface, DrawVert,
        };
        let bsp = q3_bsp::Bsp {
            entities: String::new(),
            shaders: vec![DShader {
                shader: [0; 64],
                surface_flags: 0,
                content_flags: Contents::SOLID.bits() as i32,
            }],
            planes: vec![
                DPlane { normal: [1.0, 0.0, 0.0], dist: 16.0 },
                DPlane { normal: [-1.0, 0.0, 0.0], dist: 16.0 },
                DPlane { normal: [0.0, 1.0, 0.0], dist: 16.0 },
                DPlane { normal: [0.0, -1.0, 0.0], dist: 16.0 },
                DPlane { normal: [0.0, 0.0, 1.0], dist: 16.0 },
                DPlane { normal: [0.0, 0.0, -1.0], dist: 16.0 },
            ],
            nodes: vec![DNode {
                plane_num: 0,
                children: [-1, -1],
                mins: [-16, -16, -16],
                maxs: [16, 16, 16],
            }],
            leafs: vec![DLeaf {
                cluster: 0,
                area: 0,
                mins: [-16, -16, -16],
                maxs: [16, 16, 16],
                first_leaf_surface: 0,
                num_leaf_surfaces: 0,
                first_leaf_brush: 0,
                num_leaf_brushes: 1,
            }],
            leaf_surfaces: vec![],
            leaf_brushes: vec![0],
            models: vec![DModel {
                mins: [-16.0; 3],
                maxs: [16.0; 3],
                first_surface: 0,
                num_surfaces: 0,
                first_brush: 0,
                num_brushes: 1,
            }],
            brushes: vec![DBrush {
                first_side: 0,
                num_sides: 6,
                shader_num: 0,
            }],
            brush_sides: (0..6)
                .map(|i| DBrushSide {
                    plane_num: i,
                    shader_num: 0,
                })
                .collect(),
            draw_verts: Vec::<DrawVert>::new(),
            draw_indexes: vec![],
            fogs: vec![],
            surfaces: Vec::<DSurface>::new(),
            lightmap_bytes: vec![],
            lightgrid_bytes: vec![],
            visibility: q3_bsp::Visibility::default(),
        };
        CollisionWorld::new(bsp)
    }

    #[test]
    fn slide_stops_at_wall() {
        let world = cube_world();
        // On démarre à l'ouest du cube, et on court vers l'est.
        let mut pm = PlayerMove::new(Vec3::new(-80.0, 0.0, 40.0));
        pm.velocity = Vec3::new(400.0, 0.0, 0.0);
        pm.on_ground = false;
        let params = PhysicsParams::default();
        for _ in 0..60 {
            pm.tick_collide(
                MoveCmd { delta_time: 1.0 / 60.0, ..Default::default() },
                params,
                &world,
            );
        }
        // Le joueur ne doit pas avoir traversé le cube.
        // Hull maxs.x = 15, cube mins.x = -16 → centre joueur ≤ -31.
        assert!(pm.origin.x < -30.0, "origin.x = {}", pm.origin.x);
    }

    #[test]
    fn crouch_caps_ground_speed() {
        // Sans collision : on teste juste l'intégration. Le cap crouch
        // doit empêcher de dépasser ~100 u/s dans la direction de wish.
        // `cmd.forward = 1.0` représente "bouton enfoncé à fond" —
        // l'engine le scale ensuite par max_speed pour obtenir la vitesse
        // cible avant le plafond crouch/walk.
        let mut pm = PlayerMove::new(Vec3::ZERO);
        pm.crouching = true;
        pm.on_ground = true;
        pm.view_angles = Angles::new(0.0, 0.0, 0.0);
        let params = PhysicsParams::default();
        for _ in 0..300 {
            pm.tick(
                MoveCmd {
                    forward: 1.0,
                    delta_time: 0.01,
                    ..Default::default()
                },
                params,
            );
        }
        let horiz = Vec3::new(pm.velocity.x, pm.velocity.y, 0.0).length();
        assert!(
            horiz <= params.crouch_speed + 1.0,
            "crouch speed should cap at {}, got {horiz}",
            params.crouch_speed
        );
        assert!(
            horiz > params.crouch_speed * 0.8,
            "crouch speed should near cap, got {horiz}"
        );
    }

    #[test]
    fn walk_caps_ground_speed() {
        let mut pm = PlayerMove::new(Vec3::ZERO);
        pm.on_ground = true;
        pm.view_angles = Angles::new(0.0, 0.0, 0.0);
        let params = PhysicsParams::default();
        for _ in 0..300 {
            pm.tick(
                MoveCmd {
                    forward: 1.0,
                    walk: true,
                    delta_time: 0.01,
                    ..Default::default()
                },
                params,
            );
        }
        let horiz = Vec3::new(pm.velocity.x, pm.velocity.y, 0.0).length();
        assert!(
            horiz <= params.walk_speed + 1.0,
            "walk speed should cap at {}, got {horiz}",
            params.walk_speed
        );
        assert!(
            horiz > params.walk_speed * 0.8,
            "walk speed should near cap, got {horiz}"
        );
    }

    /// Régression : en course normale, la vitesse cap bien à `max_speed`.
    /// Confirme qu'on n'a pas cassé la cinématique de base en ajoutant
    /// les modes crouch/walk.
    #[test]
    fn run_caps_ground_speed_at_max() {
        let mut pm = PlayerMove::new(Vec3::ZERO);
        pm.on_ground = true;
        pm.view_angles = Angles::new(0.0, 0.0, 0.0);
        let params = PhysicsParams::default();
        for _ in 0..300 {
            pm.tick(
                MoveCmd {
                    forward: 1.0,
                    delta_time: 0.01,
                    ..Default::default()
                },
                params,
            );
        }
        let horiz = Vec3::new(pm.velocity.x, pm.velocity.y, 0.0).length();
        assert!(
            horiz <= params.max_speed + 1.0,
            "run speed should cap at {}, got {horiz}",
            params.max_speed
        );
        assert!(
            horiz > params.max_speed * 0.8,
            "run speed should near cap, got {horiz}"
        );
    }

    #[test]
    fn crouch_state_sticky_released_when_clear() {
        let world = cube_world();
        let mut pm = PlayerMove::new(Vec3::new(-100.0, 0.0, 40.0));
        pm.on_ground = false;
        let params = PhysicsParams::default();

        // Crouch d'abord.
        pm.tick_collide(
            MoveCmd {
                crouch: true,
                delta_time: 1.0 / 60.0,
                ..Default::default()
            },
            params,
            &world,
        );
        assert!(pm.crouching, "should enter crouch on press");

        // Relâcher crouch en espace libre → stand back up.
        pm.tick_collide(
            MoveCmd { delta_time: 1.0 / 60.0, ..Default::default() },
            params,
            &world,
        );
        assert!(
            !pm.crouching,
            "should stand back up when clearance is available"
        );
    }

    #[test]
    fn crouched_jump_is_ignored() {
        let mut pm = PlayerMove::new(Vec3::ZERO);
        pm.on_ground = true;
        pm.crouching = true;
        let params = PhysicsParams::default();
        pm.tick(
            MoveCmd { jump: true, delta_time: 0.01, ..Default::default() },
            params,
        );
        // Le saut est désactivé en crouch : vitesse Z reste nulle
        // (crouch_speed = 100, pas de composante verticale applicable).
        assert_eq!(pm.velocity.z, 0.0);
    }
}
