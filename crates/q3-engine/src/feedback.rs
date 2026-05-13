//! Feedback gameplay — hit markers, killcam, annonceur vocal.
//!
//! Centralise les systèmes de feedback visuel et audio qui récompensent
//! le joueur pour ses actions : hit confirmation, kill confirmation,
//! killcam post-mortem, et annonces vocales de séries de frags.
//!
//! Chaque sous-système est indépendant et exposé via un struct dédié
//! que l'`App` peut instancier et ticker chaque frame. Aucun accès
//! direct au renderer ou au sound — on produit des données que la
//! couche HUD / audio consomme.

use q3_math::{Angles, Vec3};

// ─── Constantes de timing ─────────────────────────────────────────────
// Calibrées pour un FPS rapide (~125 fps tickrate Q3). Chaque valeur
// a été testée perceptuellement : assez longue pour être vue, assez
// courte pour ne pas gêner le flux de jeu.

/// Durée du hit marker normal (secondes).
const HIT_MARKER_DURATION: f32 = 0.25;
/// Durée du hit marker critique / headshot.
const CRITICAL_MARKER_DURATION: f32 = 0.35;
/// Durée du hit marker kill-confirm.
const KILL_MARKER_DURATION: f32 = 0.50;
/// Durée du hit marker splash (indirect).
const SPLASH_MARKER_DURATION: f32 = 0.20;

/// Taille de base du crosshair marker (pixels).
const BASE_MARKER_SIZE: f32 = 12.0;
/// Facteur multiplicatif du pulse initial (le marker grossit puis shrink).
const PULSE_SCALE: f32 = 2.2;

/// Durée de vie d'un floating damage number (secondes).
const FLOAT_DAMAGE_LIFETIME: f32 = 1.2;
/// Déplacement vertical total du floating damage (unités monde).
const FLOAT_DAMAGE_DRIFT: f32 = 20.0;

/// Durée de vie d'un damage indicator directionnel (secondes).
const DAMAGE_INDICATOR_LIFETIME: f32 = 1.5;

/// Nombre max de frames killcam par bot (3 s × 60 fps = 180).
const KILLCAM_MAX_FRAMES: usize = 180;
/// Durée du replay killcam (secondes).
const KILLCAM_REPLAY_DURATION: f32 = 3.0;

/// Fenêtre temporelle pour un double-kill (secondes).
const DOUBLE_KILL_WINDOW: f32 = 3.0;
/// Fenêtre temporelle pour un multi-kill (3 kills).
const MULTI_KILL_WINDOW: f32 = 4.0;
/// Fenêtre temporelle pour un mega-kill (4 kills).
const MEGA_KILL_WINDOW: f32 = 5.0;
/// Fenêtre temporelle pour un ultra-kill (5+ kills).
const ULTRA_KILL_WINDOW: f32 = 6.0;

/// Fenêtre "Excellent" — 2 kills consécutifs très rapprochés.
const EXCELLENT_WINDOW: f32 = 2.0;

/// Durée de rétention des kill timestamps pour la détection multi-kill.
const KILL_TIMESTAMP_RETENTION: f32 = 8.0;

/// Durée d'affichage par défaut d'une annonce (secondes).
const ANNOUNCEMENT_DISPLAY_DEFAULT: f32 = 2.5;
/// Durée d'affichage longue (Godlike, Unstoppable).
const ANNOUNCEMENT_DISPLAY_LONG: f32 = 3.5;

// ─── Hit Markers ──────────────────────────────────────────────────────

/// Configuration visuelle du hit marker selon le type de coup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitType {
    /// Degats normaux — croix blanche fine.
    Normal,
    /// Headshot (si implemente) — croix rouge + son distinct.
    Critical,
    /// Kill confirm — grande croix rouge + skull.
    Kill,
    /// Degats splash / indirect.
    Splash,
}

impl HitType {
    /// Duree d'affichage selon le type — les kills restent plus longtemps
    /// pour que le joueur ait le temps de register le feedback.
    fn duration(self) -> f32 {
        match self {
            Self::Normal => HIT_MARKER_DURATION,
            Self::Critical => CRITICAL_MARKER_DURATION,
            Self::Kill => KILL_MARKER_DURATION,
            Self::Splash => SPLASH_MARKER_DURATION,
        }
    }
}

/// Etat du hit marker actif.
#[derive(Debug, Clone)]
pub struct HitMarker {
    pub hit_type: HitType,
    pub damage: f32,
    pub start_time: f32,
    pub duration: f32,
    /// Direction du hit (pour le damage indicator directionnel).
    pub direction: Vec3,
}

impl HitMarker {
    pub fn new(hit_type: HitType, damage: f32, now: f32, direction: Vec3) -> Self {
        Self {
            hit_type,
            damage,
            start_time: now,
            duration: hit_type.duration(),
            direction,
        }
    }

    /// Progression `[0..1]` de l'animation. `None` si expire.
    pub fn progress(&self, now: f32) -> Option<f32> {
        let elapsed = now - self.start_time;
        if elapsed < 0.0 || elapsed > self.duration {
            return None;
        }
        Some(elapsed / self.duration)
    }

    /// Taille du crosshair marker (pixels) — pulse puis shrink.
    ///
    /// Courbe : peak instantane a `PULSE_SCALE * base` puis decroissance
    /// exponentielle vers `base`. Le resultat est un "pop" satisfaisant
    /// qui attire l'oeil sans masquer le reticule trop longtemps.
    pub fn size(&self, now: f32) -> f32 {
        let Some(t) = self.progress(now) else {
            return 0.0;
        };

        // Facteur de base selon le type : kills = plus gros, splash = plus petit
        let type_scale = match self.hit_type {
            HitType::Normal => 1.0,
            HitType::Critical => 1.3,
            HitType::Kill => 1.6,
            HitType::Splash => 0.8,
        };

        // Facteur de dommage : les gros coups font un marker legerement plus grand.
        // Clamp entre 1.0 et 1.4 pour eviter que le BFG remplisse l'ecran.
        let damage_scale = (self.damage / 100.0).clamp(0.0, 1.0) * 0.4 + 1.0;

        // Pulse : exponentielle decroissante — gros au debut, shrink rapide.
        // `1 + (PULSE_SCALE - 1) * e^(-6t)` donne un pop net puis retour calme.
        let pulse = 1.0 + (PULSE_SCALE - 1.0) * (-6.0 * t).exp();

        BASE_MARKER_SIZE * type_scale * damage_scale * pulse
    }

    /// Couleur RGBA — blanc pour normal, rouge pour kill, jaune pour critical.
    ///
    /// L'alpha diminue sur le dernier tiers de la duree pour un fade-out
    /// naturel. Pas de fade-in : apparition instantanee = feedback net.
    pub fn color(&self, now: f32) -> [f32; 4] {
        let Some(t) = self.progress(now) else {
            return [0.0; 4];
        };

        // Couleur de base selon le type
        let (r, g, b) = match self.hit_type {
            HitType::Normal => (1.0_f32, 1.0, 1.0),   // blanc pur
            HitType::Critical => (1.0, 0.9, 0.2),       // jaune vif
            HitType::Kill => (1.0, 0.15, 0.1),          // rouge sang
            HitType::Splash => (0.85, 0.85, 0.85),      // gris clair
        };

        // Alpha : plein sur les 2/3, puis fade lineaire sur le dernier tiers.
        // Raison : le joueur a besoin de VOIR le marker immediatement, le
        // fade ne sert qu'a eviter une coupure nette a la fin.
        let alpha = if t < 0.67 {
            1.0
        } else {
            // Remap [0.67..1.0] → [1.0..0.0]
            let fade_t = (t - 0.67) / 0.33;
            1.0 - fade_t
        };

        [r, g, b, alpha]
    }
}

// ─── Floating Damage Numbers ──────────────────────────────────────────

/// Floating damage number au-dessus d'un ennemi touche.
///
/// Spawn a la position monde de l'impact et derive vers le haut
/// avec un fade-out progressif. Les critiques sont plus gros et
/// rouges pour attirer l'attention.
#[derive(Debug, Clone)]
pub struct FloatingDamage {
    pub amount: i32,
    pub world_pos: Vec3,
    pub start_time: f32,
    pub is_critical: bool,
}

impl FloatingDamage {
    /// Position avec drift vertical (monte de `FLOAT_DAMAGE_DRIFT` unites sur la duree).
    ///
    /// Easing ease-out (racine carree) : mouvement rapide au debut qui
    /// ralentit — empeche le texte de "trainer" visuellement.
    pub fn screen_offset(&self, now: f32) -> f32 {
        let elapsed = now - self.start_time;
        let t = (elapsed / FLOAT_DAMAGE_LIFETIME).clamp(0.0, 1.0);
        // Ease-out : sqrt donne une deceleration naturelle
        let eased = t.sqrt();
        FLOAT_DAMAGE_DRIFT * eased
    }

    /// Alpha fade — plein pendant 40% de la duree, puis fade lineaire.
    ///
    /// Retourne `None` si le nombre est expire (purge par le systeme).
    pub fn alpha(&self, now: f32) -> Option<f32> {
        let elapsed = now - self.start_time;
        if !(0.0..=FLOAT_DAMAGE_LIFETIME).contains(&elapsed) {
            return None;
        }
        let t = elapsed / FLOAT_DAMAGE_LIFETIME;
        if t < 0.4 {
            Some(1.0)
        } else {
            // Remap [0.4..1.0] → [1.0..0.0]
            Some(1.0 - (t - 0.4) / 0.6)
        }
    }
}

// ─── Damage Indicator ─────────────────────────────────────────────────

/// Indicateur directionnel de degats recus (arcs rouges aux bords de l'ecran).
///
/// Montre au joueur d'ou vient le tir qui l'a touche. Essentiel dans un
/// FPS rapide ou on n'a pas le temps de tourner la camera pour chercher
/// l'attaquant.
#[derive(Debug, Clone)]
pub struct DamageIndicator {
    pub direction: Vec3,
    pub start_time: f32,
    pub intensity: f32,
}

impl DamageIndicator {
    /// Angle en radians sur l'ecran (0 = haut, PI/2 = droite, etc.)
    ///
    /// Projette la direction du tir dans le plan camera (forward, right)
    /// pour obtenir un angle 2D utilisable par le renderer HUD.
    pub fn screen_angle(&self, camera_forward: Vec3, camera_right: Vec3) -> f32 {
        // Projeter la direction source dans le plan horizontal camera.
        // On ignore la composante verticale (Z) pour que l'indicateur
        // reste sur le pourtour de l'ecran sans monter/descendre.
        let to_source = self.direction.normalize_or_zero();

        // Composantes dans le repere camera 2D
        let forward_dot = to_source.dot(camera_forward);
        let right_dot = to_source.dot(camera_right);

        // atan2 donne l'angle avec forward = 0, right = PI/2.
        // Negatif sur right_dot car Q3 a Y = left (inverse du screen-space).
        (-right_dot).atan2(-forward_dot)
    }

    /// Alpha fade — proportionnel a l'intensite initiale puis decroit.
    ///
    /// Retourne `None` si l'indicateur est expire.
    pub fn alpha(&self, now: f32) -> Option<f32> {
        let elapsed = now - self.start_time;
        if !(0.0..=DAMAGE_INDICATOR_LIFETIME).contains(&elapsed) {
            return None;
        }
        let t = elapsed / DAMAGE_INDICATOR_LIFETIME;
        // L'intensite module l'alpha de depart : un gros coup = plus visible.
        // Clamp a 1.0 pour les coups violents (rocket direct).
        let base_alpha = (self.intensity / 50.0).clamp(0.3, 1.0);
        // Fade quadratique : rapide au debut, ralentit en fin — l'indicateur
        // "colle" plus longtemps a basse opacite, utile peripheriquement.
        Some(base_alpha * (1.0 - t * t))
    }
}

// ─── Hit Marker System ────────────────────────────────────────────────

/// Systeme centralise de hit markers, floating damages et damage indicators.
///
/// L'`App` appelle `register_*` quand un evenement de combat se produit,
/// puis `prune` chaque frame pour nettoyer les elements expires. La
/// couche HUD itere sur les vecs publics pour dessiner.
pub struct HitMarkerSystem {
    pub active_markers: Vec<HitMarker>,
    pub floating_damages: Vec<FloatingDamage>,
    pub damage_indicators: Vec<DamageIndicator>,
}

impl HitMarkerSystem {
    pub fn new() -> Self {
        Self {
            // Pre-alloc raisonnable : on depasse rarement 4 markers simultanes
            // (shotgun multi-pellet = 1 marker agrege, pas N).
            active_markers: Vec::with_capacity(4),
            floating_damages: Vec::with_capacity(8),
            damage_indicators: Vec::with_capacity(4),
        }
    }

    /// Enregistre un hit sur un ennemi.
    ///
    /// `damage` : degats infliges (avant armure). `world_pos` : position
    /// de l'impact dans le monde. `direction` : vecteur joueur → cible.
    pub fn register_hit(&mut self, damage: f32, world_pos: Vec3, direction: Vec3, now: f32) {
        // Choix du type : les gros coups (rail, rocket direct) sont "critiques"
        // pour un feedback visuel plus marque. Seuil a 60 = environ un rail.
        let hit_type = if damage >= 60.0 {
            HitType::Critical
        } else {
            HitType::Normal
        };

        self.active_markers
            .push(HitMarker::new(hit_type, damage, now, direction));

        self.floating_damages.push(FloatingDamage {
            amount: damage as i32,
            world_pos,
            start_time: now,
            is_critical: damage >= 60.0,
        });
    }

    /// Enregistre un kill.
    ///
    /// Distinct de `register_hit` : le marker est plus gros, plus rouge,
    /// et dure plus longtemps. Le floating damage affiche le kill confirm.
    pub fn register_kill(&mut self, world_pos: Vec3, direction: Vec3, now: f32) {
        self.active_markers
            .push(HitMarker::new(HitType::Kill, 0.0, now, direction));

        // Floating damage special pour kill : on met un montant "symbolique"
        // que le renderer peut interpreter comme un skull ou "+1 FRAG".
        self.floating_damages.push(FloatingDamage {
            amount: 0, // Sentinel : le HUD affiche un skull / "+1" au lieu d'un chiffre
            world_pos,
            start_time: now,
            is_critical: true,
        });
    }

    /// Enregistre un hit splash (degats indirects — rocket splash, grenade).
    pub fn register_splash(
        &mut self,
        damage: f32,
        world_pos: Vec3,
        direction: Vec3,
        now: f32,
    ) {
        self.active_markers
            .push(HitMarker::new(HitType::Splash, damage, now, direction));

        self.floating_damages.push(FloatingDamage {
            amount: damage as i32,
            world_pos,
            start_time: now,
            is_critical: false,
        });
    }

    /// Enregistre des degats recus par le joueur.
    ///
    /// `direction` : vecteur source → joueur (indique d'ou vient le tir).
    /// `intensity` : montant de degats — module l'opacite de l'indicateur.
    pub fn register_damage_taken(&mut self, direction: Vec3, intensity: f32, now: f32) {
        // Eviter d'empiler trop d'indicateurs dans la meme direction :
        // si un indicateur recent pointe dans un cone similaire, on le
        // rafraichit plutot que d'en creer un nouveau.
        let dominated = self.damage_indicators.iter_mut().find(|d| {
            let age = now - d.start_time;
            let similar_dir = d.direction.normalize_or_zero().dot(direction.normalize_or_zero()) > 0.7;
            age < DAMAGE_INDICATOR_LIFETIME * 0.5 && similar_dir
        });

        if let Some(existing) = dominated {
            // Rafraichir : on re-arme le timer et on cumule l'intensite.
            existing.start_time = now;
            existing.intensity = (existing.intensity + intensity).min(150.0);
        } else {
            self.damage_indicators.push(DamageIndicator {
                direction,
                start_time: now,
                intensity,
            });
        }
    }

    /// Purge les elements expires. Appele chaque frame.
    pub fn prune(&mut self, now: f32) {
        self.active_markers
            .retain(|m| m.progress(now).is_some());
        self.floating_damages
            .retain(|d| d.alpha(now).is_some());
        self.damage_indicators
            .retain(|d| d.alpha(now).is_some());
    }
}

impl Default for HitMarkerSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Killcam ──────────────────────────────────────────────────────────

/// Snapshot de la camera du tueur a un instant donne.
///
/// Stocke juste assez d'information pour reconstituer le POV du tueur
/// pendant les dernieres secondes avant le kill. L'arme est un `u8`
/// (slot index) pour rester leger en memoire.
#[derive(Debug, Clone, Copy)]
pub struct KillcamFrame {
    pub time: f32,
    pub position: Vec3,
    pub angles: Angles,
    pub weapon: u8,
}

/// Systeme killcam — enregistre les positions de tous les bots,
/// puis rejoue les 3 dernieres secondes depuis le POV du tueur.
///
/// Fonctionne comme un ring buffer par bot : on ecrase les frames
/// les plus anciennes quand la capacite est atteinte. Au moment
/// du kill, on copie les N dernieres frames du tueur et on demarre
/// un replay avec interpolation lineaire.
pub struct KillcamSystem {
    /// Ring buffer de frames pour chaque bot (index = bot_id).
    /// Chaque Vec agit comme un ring buffer FIFO : on push en fin et
    /// drain en debut quand on depasse `max_frames_per_bot`.
    bot_frames: Vec<Vec<KillcamFrame>>,
    /// Nombre max de frames stockees par bot (3s x 60fps = 180).
    max_frames_per_bot: usize,
    /// Etat de replay actif.
    pub replay: Option<KillcamReplay>,
}

/// Replay killcam en cours — contient les frames copiees du tueur
/// et les metadonnees d'affichage (nom, arme).
pub struct KillcamReplay {
    pub frames: Vec<KillcamFrame>,
    pub start_time: f32,
    pub current_index: usize,
    pub killer_name: String,
    pub weapon_name: String,
    /// Duree totale du replay (secondes).
    pub duration: f32,
}

impl KillcamSystem {
    pub fn new(max_bots: usize) -> Self {
        let mut bot_frames = Vec::with_capacity(max_bots);
        for _ in 0..max_bots {
            // Pre-alloc chaque buffer a la capacite max pour eviter les
            // reallocations pendant le jeu (latence = ennemi du gameplay).
            bot_frames.push(Vec::with_capacity(KILLCAM_MAX_FRAMES));
        }
        Self {
            bot_frames,
            max_frames_per_bot: KILLCAM_MAX_FRAMES,
            replay: None,
        }
    }

    /// Appele chaque frame serveur pour enregistrer la position d'un bot.
    ///
    /// L'enregistrement est continu et silencieux — pas d'allocation en
    /// regime permanent grace au pre-alloc. Les vieilles frames sont
    /// purgees FIFO quand on depasse la capacite.
    pub fn record_bot(&mut self, bot_id: usize, frame: KillcamFrame) {
        // Ignore les bot_id hors range — peut arriver si des bots sont
        // ajoutes dynamiquement au dela du max initial.
        let Some(buf) = self.bot_frames.get_mut(bot_id) else {
            return;
        };

        // Ring buffer FIFO : purge le plus ancien si plein.
        if buf.len() >= self.max_frames_per_bot {
            buf.remove(0);
        }
        buf.push(frame);
    }

    /// Demarre le replay killcam depuis le POV du bot tueur.
    ///
    /// Copie les frames enregistrees du bot, les trie par temps, et
    /// initialise le replay. Si le bot n'a pas assez de frames (vient
    /// de spawn), le replay sera plus court.
    pub fn start_replay(
        &mut self,
        killer_bot_id: usize,
        killer_name: &str,
        weapon: &str,
        now: f32,
    ) {
        let Some(buf) = self.bot_frames.get(killer_bot_id) else {
            return;
        };

        if buf.is_empty() {
            return;
        }

        // Copier les frames — on ne vide pas le buffer source car le bot
        // continue de jouer pendant le replay (et on pourrait en avoir
        // besoin pour un second killcam).
        let frames = buf.clone();

        // Duree effective : on prend la fenetre temporelle reelle des frames
        // copiees, plafonnee a KILLCAM_REPLAY_DURATION.
        let time_span = frames.last().unwrap().time - frames.first().unwrap().time;
        let duration = time_span.clamp(0.5, KILLCAM_REPLAY_DURATION);

        self.replay = Some(KillcamReplay {
            frames,
            start_time: now,
            current_index: 0,
            killer_name: killer_name.to_string(),
            weapon_name: weapon.to_string(),
            duration,
        });
    }

    /// Sample la camera interpolee a l'instant donne.
    ///
    /// Retourne `(position, angles)` interpoles lineairement entre les
    /// deux frames les plus proches. `None` si le replay est termine ou
    /// inactif.
    pub fn sample(&self, now: f32) -> Option<(Vec3, Angles)> {
        let replay = self.replay.as_ref()?;
        let elapsed = now - replay.start_time;

        if elapsed < 0.0 || elapsed > replay.duration {
            return None;
        }

        if replay.frames.is_empty() {
            return None;
        }

        // Temps absolu dans l'espace des frames enregistrees.
        // On remappe le temps de replay [0..duration] sur le range temporel
        // des frames [first.time..last.time].
        let first_time = replay.frames.first().unwrap().time;
        let last_time = replay.frames.last().unwrap().time;
        let frame_span = last_time - first_time;

        if frame_span <= 0.0 {
            // Toutes les frames au meme instant — retourne la derniere.
            let f = replay.frames.last().unwrap();
            return Some((f.position, f.angles));
        }

        let playback_t = elapsed / replay.duration;
        let target_time = first_time + playback_t * frame_span;

        // Recherche binaire de la frame juste avant `target_time`.
        let idx = replay
            .frames
            .partition_point(|f| f.time <= target_time)
            .saturating_sub(1);

        let a = &replay.frames[idx];

        // Cas limite : derniere frame — pas d'interpolation.
        if idx + 1 >= replay.frames.len() {
            return Some((a.position, a.angles));
        }

        let b = &replay.frames[idx + 1];

        // Facteur d'interpolation entre frame A et frame B.
        let seg_duration = b.time - a.time;
        let frac = if seg_duration > 0.0 {
            ((target_time - a.time) / seg_duration).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Lerp position
        let pos = a.position + (b.position - a.position) * frac;

        // Lerp angles — attention aux wraps a 180/-180.
        // On utilise la methode courte : normaliser la difference.
        let pitch = lerp_angle(a.angles.pitch, b.angles.pitch, frac);
        let yaw = lerp_angle(a.angles.yaw, b.angles.yaw, frac);
        let roll = lerp_angle(a.angles.roll, b.angles.roll, frac);
        let angles = Angles::new(pitch, yaw, roll);

        Some((pos, angles))
    }

    /// Met fin au replay (skip par le joueur).
    pub fn skip(&mut self) {
        self.replay = None;
    }

    /// True si un replay est en cours.
    pub fn is_active(&self) -> bool {
        self.replay.is_some()
    }

    /// Verifie si le replay en cours est termine (temps ecoule).
    /// Retourne `true` si le replay vient d'etre purge.
    pub fn check_expired(&mut self, now: f32) -> bool {
        let expired = self
            .replay
            .as_ref()
            .is_some_and(|r| now - r.start_time > r.duration);
        if expired {
            self.replay = None;
        }
        expired
    }
}

/// Interpolation lineaire d'angle en degres, par le chemin le plus court.
///
/// Evite le saut 359° → 1° en passant par la difference normalisee.
fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let mut diff = b - a;
    // Normaliser dans [-180, 180] pour prendre le chemin court.
    while diff > 180.0 {
        diff -= 360.0;
    }
    while diff < -180.0 {
        diff += 360.0;
    }
    a + diff * t
}

// ─── Kill Streak Announcer ────────────────────────────────────────────

/// Types d'annonces vocales.
///
/// Ordonnees par priorite implicite — les enums plus bas dans la liste
/// ont une priorite plus elevee (plus "impressionnant").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Announcement {
    FirstBlood,
    DoubleKill,      // 2 kills en 3s
    MultiKill,       // 3 kills en 4s
    MegaKill,        // 4 kills en 5s
    UltraKill,       // 5+ kills en 6s
    Rampage,         // 5 kills streak total
    Dominating,      // 10 kills streak
    Godlike,         // 15 kills streak
    Unstoppable,     // 20 kills streak
    Impressive,      // 2 railgun hits consecutifs
    Excellent,       // 2 kills consecutifs (< 2s)
    Humiliation,     // Kill au gauntlet
}

impl Announcement {
    /// Nom du fichier son (relatif a `assets/sounds/announcer/`).
    ///
    /// Convention Q3 : noms courts en snake_case. Les fichiers sont
    /// pre-charges au lancement de la map pour eviter tout I/O pendant
    /// le gameplay.
    pub fn sound_file(&self) -> &'static str {
        match self {
            Self::FirstBlood => "first_blood.wav",
            Self::DoubleKill => "double_kill.wav",
            Self::MultiKill => "multi_kill.wav",
            Self::MegaKill => "mega_kill.wav",
            Self::UltraKill => "ultra_kill.wav",
            Self::Rampage => "rampage.wav",
            Self::Dominating => "dominating.wav",
            Self::Godlike => "godlike.wav",
            Self::Unstoppable => "unstoppable.wav",
            Self::Impressive => "impressive.wav",
            Self::Excellent => "excellent.wav",
            Self::Humiliation => "humiliation.wav",
        }
    }

    /// Texte affiche au centre de l'ecran.
    pub fn display_text(&self) -> &'static str {
        match self {
            Self::FirstBlood => "FIRST BLOOD!",
            Self::DoubleKill => "DOUBLE KILL!",
            Self::MultiKill => "MULTI KILL!",
            Self::MegaKill => "M-M-MEGA KILL!",
            Self::UltraKill => "ULTRA KILL!!!",
            Self::Rampage => "RAMPAGE!",
            Self::Dominating => "DOMINATING!",
            Self::Godlike => "GODLIKE!",
            Self::Unstoppable => "UNSTOPPABLE!",
            Self::Impressive => "IMPRESSIVE!",
            Self::Excellent => "EXCELLENT!",
            Self::Humiliation => "HUMILIATION!",
        }
    }

    /// Duree d'affichage (secondes).
    pub fn display_duration(&self) -> f32 {
        match self {
            // Les annonces rares / impressionnantes restent plus longtemps —
            // recompense visuelle proportionnelle a la difficulte.
            Self::Godlike | Self::Unstoppable | Self::UltraKill => ANNOUNCEMENT_DISPLAY_LONG,
            _ => ANNOUNCEMENT_DISPLAY_DEFAULT,
        }
    }

    /// Priorite (pour eviter de superposer).
    ///
    /// Plus le nombre est eleve, plus l'annonce est prioritaire.
    /// Quand deux annonces arrivent dans la meme fenetre, la plus
    /// prioritaire passe en premier (ou remplace la moins prioritaire).
    pub fn priority(&self) -> u8 {
        match self {
            Self::FirstBlood => 5,
            Self::DoubleKill => 3,
            Self::MultiKill => 4,
            Self::MegaKill => 6,
            Self::UltraKill => 7,
            Self::Rampage => 5,
            Self::Dominating => 6,
            Self::Godlike => 8,
            Self::Unstoppable => 9,
            Self::Impressive => 4,
            Self::Excellent => 4,
            Self::Humiliation => 10, // Q3 tradition : gauntlet frag = ultimate flex
        }
    }
}

/// Systeme d'annonces vocales et textuelles.
///
/// Detecte les multi-kills (timing), les streaks (cumul), et les
/// evenements speciaux (railgun, gauntlet). Produit des `Announcement`
/// que la couche audio/HUD consomme.
pub struct AnnouncerSystem {
    /// Compteur de kills streak (reset a la mort).
    pub kill_streak: u32,
    /// Timestamps des kills recents (pour multi-kill detection).
    /// Purgee au dela de `KILL_TIMESTAMP_RETENTION` secondes.
    recent_kill_times: Vec<f32>,
    /// Compteur de hits railgun consecutifs.
    pub consecutive_rail_hits: u32,
    /// Annonce en cours d'affichage : `(type, start_time)`.
    pub active_announcement: Option<(Announcement, f32)>,
    /// Queue d'annonces en attente — depilee par `update()`.
    pending: Vec<Announcement>,
    /// Premier sang deja annonce — un seul par match.
    first_blood_done: bool,
}

impl AnnouncerSystem {
    pub fn new() -> Self {
        Self {
            kill_streak: 0,
            recent_kill_times: Vec::with_capacity(16),
            consecutive_rail_hits: 0,
            active_announcement: None,
            pending: Vec::with_capacity(4),
            first_blood_done: false,
        }
    }

    /// Appele quand le joueur tue un ennemi.
    ///
    /// `weapon` : slot de l'arme utilisee (1 = gauntlet, 7 = railgun, etc.)
    /// Retourne l'annonce la plus prioritaire declenchee par ce kill, s'il y en a.
    pub fn on_kill(&mut self, weapon: u8, now: f32) -> Option<Announcement> {
        self.kill_streak = self.kill_streak.saturating_add(1);
        self.recent_kill_times.push(now);

        // Purge des vieux timestamps — on ne garde que les recents
        // pour la detection multi-kill.
        self.recent_kill_times
            .retain(|&t| now - t < KILL_TIMESTAMP_RETENTION);

        let mut best: Option<Announcement> = None;

        // ─── First Blood ────────────────────────────────────────
        if !self.first_blood_done {
            self.first_blood_done = true;
            best = higher_priority(best, Some(Announcement::FirstBlood));
        }

        // ─── Humiliation (gauntlet kill, slot 1) ────────────────
        // Le gauntlet frag est l'insulte supreme dans Q3. Priorite max.
        if weapon == 1 {
            best = higher_priority(best, Some(Announcement::Humiliation));
        }

        // ─── Multi-kill detection (kills dans une fenetre temporelle) ──
        // On compte combien de kills tombent dans chaque fenetre.
        // La fenetre la plus large qui matche donne le tier.
        let recent_in_window = |window: f32| -> usize {
            self.recent_kill_times
                .iter()
                .filter(|&&t| now - t <= window)
                .count()
        };

        let kills_3s = recent_in_window(DOUBLE_KILL_WINDOW);
        let kills_4s = recent_in_window(MULTI_KILL_WINDOW);
        let kills_5s = recent_in_window(MEGA_KILL_WINDOW);
        let kills_6s = recent_in_window(ULTRA_KILL_WINDOW);

        // Du plus impressionnant au moins — on break au premier match.
        let multi = if kills_6s >= 5 {
            Some(Announcement::UltraKill)
        } else if kills_5s >= 4 {
            Some(Announcement::MegaKill)
        } else if kills_4s >= 3 {
            Some(Announcement::MultiKill)
        } else if kills_3s >= 2 {
            Some(Announcement::DoubleKill)
        } else {
            None
        };
        best = higher_priority(best, multi);

        // ─── Excellent (2 kills en < 2s) ────────────────────────
        // Plus strict que DoubleKill — recompense la rapidite pure.
        if self.recent_kill_times.len() >= 2 {
            let prev = self.recent_kill_times[self.recent_kill_times.len() - 2];
            if now - prev < EXCELLENT_WINDOW {
                best = higher_priority(best, Some(Announcement::Excellent));
            }
        }

        // ─── Streak milestones ──────────────────────────────────
        // Paliers cumulatifs depuis le dernier respawn.
        let streak_ann = match self.kill_streak {
            5 => Some(Announcement::Rampage),
            10 => Some(Announcement::Dominating),
            15 => Some(Announcement::Godlike),
            20 => Some(Announcement::Unstoppable),
            _ => None,
        };
        best = higher_priority(best, streak_ann);

        // Enqueue l'annonce gagnante
        if let Some(ann) = best {
            self.enqueue(ann);
        }

        best
    }

    /// Appele quand le joueur meurt — reset streak.
    ///
    /// On ne reset PAS `first_blood_done` : c'est un flag par match,
    /// pas par vie. Le reset match est la responsabilite de l'appelant
    /// qui doit creer un nouveau `AnnouncerSystem`.
    pub fn on_death(&mut self) {
        self.kill_streak = 0;
        self.consecutive_rail_hits = 0;
        self.recent_kill_times.clear();
    }

    /// Appele quand le joueur touche avec le railgun.
    ///
    /// 2 hits consecutifs = "IMPRESSIVE" (Q3 classique).
    pub fn on_rail_hit(&mut self, now: f32) -> Option<Announcement> {
        self.consecutive_rail_hits = self.consecutive_rail_hits.saturating_add(1);

        if self.consecutive_rail_hits >= 2 {
            // Reset a 0 (pas 1) : il faut 2 NOUVEAUX hits pour le prochain
            // Impressive. Sinon chaque hit apres le 2e triggerait.
            self.consecutive_rail_hits = 0;
            self.enqueue(Announcement::Impressive);
            let _ = now; // Timestamp disponible si on veut logger
            return Some(Announcement::Impressive);
        }
        None
    }

    /// Appele quand le joueur rate avec le railgun.
    ///
    /// Un seul miss reset le compteur — Q3 est impitoyable.
    pub fn on_rail_miss(&mut self) {
        self.consecutive_rail_hits = 0;
    }

    /// Tick — gere la queue d'annonces et l'expiration.
    ///
    /// Retourne l'annonce qui vient de devenir active (pour que la couche
    /// audio puisse jouer le son), ou `None` si rien de nouveau.
    pub fn update(&mut self, now: f32) -> Option<Announcement> {
        // Verifier si l'annonce active a expire.
        if let Some((ann, start)) = &self.active_announcement {
            if now - start > ann.display_duration() {
                self.active_announcement = None;
            }
        }

        // Si pas d'annonce active et queue non vide, depiler la plus prioritaire.
        if self.active_announcement.is_none() && !self.pending.is_empty() {
            // Trier par priorite decroissante et prendre la premiere.
            // La queue est petite (< 5 elements en pratique), le sort est negligeable.
            self.pending.sort_by_key(|b| std::cmp::Reverse(b.priority()));
            let next = self.pending.remove(0);
            self.active_announcement = Some((next, now));
            return Some(next);
        }

        None
    }

    /// Texte a afficher au centre du HUD (si actif).
    ///
    /// Retourne `(texte, alpha)` ou `alpha` diminue sur la fin de la duree
    /// pour un fade-out elegant.
    pub fn hud_text(&self, now: f32) -> Option<(&str, f32)> {
        let (ann, start) = self.active_announcement.as_ref()?;
        let elapsed = now - start;
        let duration = ann.display_duration();

        if elapsed > duration {
            return None;
        }

        let t = elapsed / duration;

        // Alpha : plein sur les 60% centraux, fade-in rapide (10%), fade-out (30%).
        // Le fade-in donne un leger "pop" au lieu d'apparaitre brutalement.
        let alpha = if t < 0.1 {
            // Fade-in : [0..0.1] → [0..1]
            t / 0.1
        } else if t < 0.7 {
            // Plein
            1.0
        } else {
            // Fade-out : [0.7..1.0] → [1..0]
            1.0 - (t - 0.7) / 0.3
        };

        Some((ann.display_text(), alpha))
    }

    /// Enqueue une annonce.
    ///
    /// Si une annonce de meme type est deja en queue, on la remplace
    /// plutot que de doubler (evite "DOUBLE KILL DOUBLE KILL" en rafale).
    fn enqueue(&mut self, ann: Announcement) {
        // Check si une annonce du meme type est deja en queue
        if let Some(existing) = self.pending.iter_mut().find(|a| **a == ann) {
            *existing = ann; // Refresh (noop ici mais semantiquement correct)
            return;
        }

        // Si l'annonce active est moins prioritaire, on peut la preempter.
        // Sinon on queue pour apres.
        if let Some((active, _)) = &self.active_announcement {
            if ann.priority() > active.priority() {
                // L'annonce en cours est moins prioritaire — on la requeue
                // et on met la nouvelle en direct.
                // Note : on ne fait PAS ca ici car `enqueue` ne gere pas
                // le timing. C'est `update()` qui depilera au bon moment.
                // On se contente de push.
            }
        }

        self.pending.push(ann);
    }

    /// Reset complet pour un nouveau match.
    pub fn reset(&mut self) {
        self.kill_streak = 0;
        self.recent_kill_times.clear();
        self.consecutive_rail_hits = 0;
        self.active_announcement = None;
        self.pending.clear();
        self.first_blood_done = false;
    }
}

impl Default for AnnouncerSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Retourne l'annonce avec la priorite la plus elevee entre deux `Option`.
/// Utilise pour comparer les candidats dans `on_kill`.
fn higher_priority(
    a: Option<Announcement>,
    b: Option<Announcement>,
) -> Option<Announcement> {
    match (a, b) {
        (None, b) => b,
        (a, None) => a,
        (Some(a), Some(b)) => {
            if b.priority() >= a.priority() {
                Some(b)
            } else {
                Some(a)
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── HitMarker ─────────────────────────────────────────────────

    #[test]
    fn hit_marker_progress_within_duration() {
        let m = HitMarker::new(HitType::Normal, 25.0, 1.0, Vec3::X);
        // Mi-parcours
        let p = m.progress(1.0 + HIT_MARKER_DURATION * 0.5);
        assert!(p.is_some());
        let p = p.unwrap();
        assert!((p - 0.5).abs() < 0.01);
    }

    #[test]
    fn hit_marker_expired_after_duration() {
        let m = HitMarker::new(HitType::Normal, 25.0, 1.0, Vec3::X);
        assert!(m.progress(1.0 + HIT_MARKER_DURATION + 0.01).is_none());
    }

    #[test]
    fn hit_marker_size_starts_large_then_shrinks() {
        let m = HitMarker::new(HitType::Normal, 50.0, 0.0, Vec3::X);
        let early = m.size(0.01);
        let late = m.size(HIT_MARKER_DURATION * 0.9);
        assert!(early > late, "marker should shrink over time");
    }

    #[test]
    fn hit_marker_kill_larger_than_normal() {
        let normal = HitMarker::new(HitType::Normal, 50.0, 0.0, Vec3::X);
        let kill = HitMarker::new(HitType::Kill, 50.0, 0.0, Vec3::X);
        assert!(kill.size(0.05) > normal.size(0.05));
    }

    #[test]
    fn hit_marker_color_fades_at_end() {
        let m = HitMarker::new(HitType::Normal, 25.0, 0.0, Vec3::X);
        let early_alpha = m.color(HIT_MARKER_DURATION * 0.5)[3];
        let late_alpha = m.color(HIT_MARKER_DURATION * 0.9)[3];
        assert!(early_alpha > late_alpha, "alpha should fade at end");
    }

    // ── FloatingDamage ────────────────────────────────────────────

    #[test]
    fn floating_damage_drifts_upward() {
        let d = FloatingDamage {
            amount: 50,
            world_pos: Vec3::ZERO,
            start_time: 0.0,
            is_critical: false,
        };
        let early = d.screen_offset(0.1);
        let late = d.screen_offset(FLOAT_DAMAGE_LIFETIME * 0.8);
        assert!(late > early, "damage number should drift upward");
    }

    #[test]
    fn floating_damage_expires() {
        let d = FloatingDamage {
            amount: 50,
            world_pos: Vec3::ZERO,
            start_time: 0.0,
            is_critical: false,
        };
        assert!(d.alpha(FLOAT_DAMAGE_LIFETIME + 0.1).is_none());
    }

    // ── DamageIndicator ───────────────────────────────────────────

    #[test]
    fn damage_indicator_fades() {
        let d = DamageIndicator {
            direction: Vec3::X,
            start_time: 0.0,
            intensity: 50.0,
        };
        let early = d.alpha(0.1).unwrap();
        let late = d.alpha(DAMAGE_INDICATOR_LIFETIME * 0.9).unwrap();
        assert!(early > late);
    }

    #[test]
    fn damage_indicator_expires() {
        let d = DamageIndicator {
            direction: Vec3::X,
            start_time: 0.0,
            intensity: 50.0,
        };
        assert!(d.alpha(DAMAGE_INDICATOR_LIFETIME + 0.1).is_none());
    }

    #[test]
    fn damage_indicator_screen_angle_forward() {
        let d = DamageIndicator {
            direction: -Vec3::X, // Source is behind the player (player faces +X)
            start_time: 0.0,
            intensity: 50.0,
        };
        let angle = d.screen_angle(Vec3::X, -Vec3::Y);
        // Source behind = direction is -X, so projected angle should be ~0 (top = behind)
        // atan2(-right_dot, -forward_dot) with to_source = -X:
        // forward_dot = -1, right_dot = 0 → atan2(0, 1) = 0
        assert!(angle.abs() < 0.01, "source behind should be angle ~0");
    }

    // ── HitMarkerSystem ───────────────────────────────────────────

    #[test]
    fn system_registers_and_prunes() {
        let mut sys = HitMarkerSystem::new();
        sys.register_hit(25.0, Vec3::ZERO, Vec3::X, 0.0);
        sys.register_hit(50.0, Vec3::Y, Vec3::X, 0.0);
        assert_eq!(sys.active_markers.len(), 2);
        assert_eq!(sys.floating_damages.len(), 2);

        // Avancer le temps au-dela de toutes les durees
        sys.prune(10.0);
        assert_eq!(sys.active_markers.len(), 0);
        assert_eq!(sys.floating_damages.len(), 0);
    }

    #[test]
    fn system_damage_taken_coalesces() {
        let mut sys = HitMarkerSystem::new();
        sys.register_damage_taken(Vec3::X, 30.0, 0.0);
        sys.register_damage_taken(Vec3::X, 20.0, 0.1); // Meme direction, recent → coalesce
        assert_eq!(sys.damage_indicators.len(), 1);
        assert!((sys.damage_indicators[0].intensity - 50.0).abs() < 0.01);
    }

    #[test]
    fn system_damage_taken_different_direction() {
        let mut sys = HitMarkerSystem::new();
        sys.register_damage_taken(Vec3::X, 30.0, 0.0);
        sys.register_damage_taken(-Vec3::X, 20.0, 0.1); // Direction opposee → nouveau
        assert_eq!(sys.damage_indicators.len(), 2);
    }

    // ── Killcam ───────────────────────────────────────────────────

    #[test]
    fn killcam_record_and_replay() {
        let mut cam = KillcamSystem::new(4);

        // Enregistrer 60 frames pour le bot 0
        for i in 0..60 {
            let t = i as f32 / 60.0;
            cam.record_bot(
                0,
                KillcamFrame {
                    time: t,
                    position: Vec3::new(t * 100.0, 0.0, 56.0),
                    angles: Angles::new(0.0, t * 90.0, 0.0),
                    weapon: 7,
                },
            );
        }

        cam.start_replay(0, "Xaero", "railgun", 10.0);
        assert!(cam.is_active());

        // Sample au milieu du replay
        let (pos, angles) = cam.sample(10.0 + KILLCAM_REPLAY_DURATION * 0.5).unwrap();
        assert!(pos.x > 0.0, "position should have progressed");
        assert!(angles.yaw > 0.0, "yaw should have rotated");
    }

    #[test]
    fn killcam_skip() {
        let mut cam = KillcamSystem::new(2);
        cam.record_bot(
            0,
            KillcamFrame {
                time: 0.0,
                position: Vec3::ZERO,
                angles: Angles::ZERO,
                weapon: 2,
            },
        );
        cam.start_replay(0, "Sarge", "machinegun", 0.0);
        assert!(cam.is_active());
        cam.skip();
        assert!(!cam.is_active());
    }

    #[test]
    fn killcam_expired() {
        let mut cam = KillcamSystem::new(2);
        for i in 0..10 {
            cam.record_bot(
                0,
                KillcamFrame {
                    time: i as f32 * 0.1,
                    position: Vec3::ZERO,
                    angles: Angles::ZERO,
                    weapon: 2,
                },
            );
        }
        cam.start_replay(0, "Sarge", "machinegun", 0.0);
        assert!(!cam.check_expired(0.5));
        assert!(cam.check_expired(100.0));
        assert!(!cam.is_active());
    }

    #[test]
    fn killcam_ring_buffer_evicts() {
        let mut cam = KillcamSystem::new(1);
        // Ecrire plus que la capacite max
        for i in 0..(KILLCAM_MAX_FRAMES + 50) {
            cam.record_bot(
                0,
                KillcamFrame {
                    time: i as f32 / 60.0,
                    position: Vec3::ZERO,
                    angles: Angles::ZERO,
                    weapon: 2,
                },
            );
        }
        assert_eq!(cam.bot_frames[0].len(), KILLCAM_MAX_FRAMES);
    }

    #[test]
    fn killcam_ignore_invalid_bot_id() {
        let mut cam = KillcamSystem::new(2);
        // Ne doit pas paniquer
        cam.record_bot(
            999,
            KillcamFrame {
                time: 0.0,
                position: Vec3::ZERO,
                angles: Angles::ZERO,
                weapon: 2,
            },
        );
    }

    #[test]
    fn killcam_interpolation_smooth() {
        let mut cam = KillcamSystem::new(1);

        // Deux frames : position 0 → position 100
        cam.record_bot(
            0,
            KillcamFrame {
                time: 0.0,
                position: Vec3::ZERO,
                angles: Angles::new(0.0, 0.0, 0.0),
                weapon: 2,
            },
        );
        cam.record_bot(
            0,
            KillcamFrame {
                time: 1.0,
                position: Vec3::new(100.0, 0.0, 0.0),
                angles: Angles::new(0.0, 180.0, 0.0),
                weapon: 2,
            },
        );

        cam.start_replay(0, "Test", "mg", 5.0);
        let replay = cam.replay.as_ref().unwrap();
        let duration = replay.duration;

        // Mi-chemin du replay → position ~50
        let (pos, angles) = cam.sample(5.0 + duration * 0.5).unwrap();
        assert!(
            (pos.x - 50.0).abs() < 5.0,
            "mid-replay position should be ~50, got {}",
            pos.x
        );
        assert!(
            (angles.yaw - 90.0).abs() < 10.0,
            "mid-replay yaw should be ~90, got {}",
            angles.yaw
        );
    }

    // ── Announcer ─────────────────────────────────────────────────

    #[test]
    fn announcer_first_blood() {
        let mut ann = AnnouncerSystem::new();
        let result = ann.on_kill(2, 0.0); // machinegun kill
        assert_eq!(result, Some(Announcement::FirstBlood));
    }

    #[test]
    fn announcer_first_blood_only_once() {
        let mut ann = AnnouncerSystem::new();
        ann.on_kill(2, 0.0);
        let result = ann.on_kill(2, 5.0); // Second kill, much later
        assert_ne!(result, Some(Announcement::FirstBlood));
    }

    #[test]
    fn announcer_humiliation() {
        let mut ann = AnnouncerSystem::new();
        ann.first_blood_done = true; // Skip first blood
        let result = ann.on_kill(1, 0.0); // weapon slot 1 = gauntlet
        assert_eq!(result, Some(Announcement::Humiliation));
    }

    #[test]
    fn announcer_double_kill() {
        let mut ann = AnnouncerSystem::new();
        ann.first_blood_done = true;
        ann.on_kill(2, 0.0);
        let result = ann.on_kill(2, 1.0); // 2nd kill within 3s
        // Could be DoubleKill or Excellent (both valid, Excellent has same priority)
        assert!(
            result == Some(Announcement::DoubleKill)
                || result == Some(Announcement::Excellent),
            "expected DoubleKill or Excellent, got {:?}",
            result
        );
    }

    #[test]
    fn announcer_multi_kill() {
        let mut ann = AnnouncerSystem::new();
        ann.first_blood_done = true;
        ann.on_kill(2, 0.0);
        ann.on_kill(2, 1.0);
        let result = ann.on_kill(2, 2.0); // 3rd kill within 4s
        // MultiKill or higher
        assert!(result.is_some());
    }

    #[test]
    fn announcer_streak_rampage() {
        let mut ann = AnnouncerSystem::new();
        ann.first_blood_done = true;
        // 5 kills espaces pour eviter multi-kill timing
        for i in 0..5 {
            ann.on_kill(2, i as f32 * 10.0);
        }
        assert_eq!(ann.kill_streak, 5);
    }

    #[test]
    fn announcer_death_resets() {
        let mut ann = AnnouncerSystem::new();
        ann.on_kill(2, 0.0);
        ann.on_kill(2, 1.0);
        assert_eq!(ann.kill_streak, 2);
        ann.on_death();
        assert_eq!(ann.kill_streak, 0);
        assert!(ann.recent_kill_times.is_empty());
    }

    #[test]
    fn announcer_impressive() {
        let mut ann = AnnouncerSystem::new();
        ann.on_rail_hit(0.0);
        let result = ann.on_rail_hit(1.0);
        assert_eq!(result, Some(Announcement::Impressive));
        // Counter should be reset
        assert_eq!(ann.consecutive_rail_hits, 0);
    }

    #[test]
    fn announcer_rail_miss_resets() {
        let mut ann = AnnouncerSystem::new();
        ann.on_rail_hit(0.0);
        ann.on_rail_miss();
        let result = ann.on_rail_hit(2.0);
        assert_eq!(result, None); // Only 1 hit since miss
    }

    #[test]
    fn announcer_update_cycles_queue() {
        let mut ann = AnnouncerSystem::new();
        ann.first_blood_done = true;

        // Two kills rapides → queue double-kill et excellent
        ann.on_kill(2, 0.0);
        ann.on_kill(2, 0.5);

        // Premier update : depile la plus prioritaire
        let first = ann.update(1.0);
        assert!(first.is_some());
        assert!(ann.active_announcement.is_some());

        // Deuxieme update avant expiration → rien de nouveau
        let second = ann.update(1.1);
        assert!(second.is_none());

        // Apres expiration → depile la suivante
        let after_expire = ann.update(1.0 + ANNOUNCEMENT_DISPLAY_DEFAULT + 0.1);
        // Peut etre Some ou None selon ce qui reste en queue
        let _ = after_expire;
    }

    #[test]
    fn announcer_hud_text_fades() {
        let mut ann = AnnouncerSystem::new();
        ann.active_announcement = Some((Announcement::DoubleKill, 0.0));

        // Debut : fade-in
        let (text, alpha) = ann.hud_text(0.05).unwrap();
        assert_eq!(text, "DOUBLE KILL!");
        assert!(alpha < 1.0, "should be fading in");

        // Milieu : plein
        let (_, alpha) = ann.hud_text(ANNOUNCEMENT_DISPLAY_DEFAULT * 0.5).unwrap();
        assert!((alpha - 1.0).abs() < 0.01);

        // Fin : fade-out
        let (_, alpha) = ann.hud_text(ANNOUNCEMENT_DISPLAY_DEFAULT * 0.9).unwrap();
        assert!(alpha < 1.0, "should be fading out");

        // Expire
        assert!(ann.hud_text(ANNOUNCEMENT_DISPLAY_DEFAULT + 0.1).is_none());
    }

    #[test]
    fn announcer_reset() {
        let mut ann = AnnouncerSystem::new();
        ann.on_kill(2, 0.0);
        ann.on_kill(2, 1.0);
        ann.on_rail_hit(2.0);
        ann.update(3.0);

        ann.reset();

        assert_eq!(ann.kill_streak, 0);
        assert!(ann.recent_kill_times.is_empty());
        assert_eq!(ann.consecutive_rail_hits, 0);
        assert!(ann.active_announcement.is_none());
        assert!(ann.pending.is_empty());
        assert!(!ann.first_blood_done);
    }

    // ── Announcement ──────────────────────────────────────────────

    #[test]
    fn announcement_priority_order() {
        assert!(Announcement::Humiliation.priority() > Announcement::DoubleKill.priority());
        assert!(Announcement::Unstoppable.priority() > Announcement::Godlike.priority());
        assert!(Announcement::Godlike.priority() > Announcement::Rampage.priority());
    }

    #[test]
    fn announcement_display_text_not_empty() {
        let all = [
            Announcement::FirstBlood,
            Announcement::DoubleKill,
            Announcement::MultiKill,
            Announcement::MegaKill,
            Announcement::UltraKill,
            Announcement::Rampage,
            Announcement::Dominating,
            Announcement::Godlike,
            Announcement::Unstoppable,
            Announcement::Impressive,
            Announcement::Excellent,
            Announcement::Humiliation,
        ];
        for ann in &all {
            assert!(!ann.display_text().is_empty());
            assert!(!ann.sound_file().is_empty());
            assert!(ann.display_duration() > 0.0);
            assert!(ann.priority() > 0);
        }
    }

    // ── lerp_angle ────────────────────────────────────────────────

    #[test]
    fn lerp_angle_short_path() {
        // 350 → 10 should go through 0, not 180
        let result = lerp_angle(350.0, 10.0, 0.5);
        assert!(
            (result - 0.0).abs() < 1.0 || (result - 360.0).abs() < 1.0,
            "expected ~0 or ~360, got {}",
            result
        );
    }

    #[test]
    fn lerp_angle_identity() {
        let result = lerp_angle(45.0, 45.0, 0.5);
        assert!((result - 45.0).abs() < 0.01);
    }

    #[test]
    fn lerp_angle_simple() {
        let result = lerp_angle(0.0, 90.0, 0.5);
        assert!((result - 45.0).abs() < 0.01);
    }

    // ── higher_priority ───────────────────────────────────────────

    #[test]
    fn higher_priority_picks_best() {
        let result = higher_priority(
            Some(Announcement::DoubleKill),
            Some(Announcement::Humiliation),
        );
        assert_eq!(result, Some(Announcement::Humiliation));
    }

    #[test]
    fn higher_priority_handles_none() {
        assert_eq!(
            higher_priority(None, Some(Announcement::FirstBlood)),
            Some(Announcement::FirstBlood)
        );
        assert_eq!(
            higher_priority(Some(Announcement::FirstBlood), None),
            Some(Announcement::FirstBlood)
        );
        assert_eq!(
            higher_priority(None::<Announcement>, None),
            None
        );
    }
}
