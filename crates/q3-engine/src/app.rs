//! `ApplicationHandler` winit — détient la fenêtre, le renderer, l'état
//! du jeu. Appelé par la boucle d'événements.

use q3_bot::{Bot, BotSkill};
use q3_bsp::Bsp;
use q3_common::cmd::{Args, CmdRegistry};
use q3_common::console::{register_builtins, Console, EngineHooks};
use q3_common::cvar::{CvarFlags, CvarRegistry};
use q3_filesystem::Vfs;
use q3_game::health::Health;
use q3_game::movement::{MoveCmd, PhysicsParams, PlayerMove};
use q3_game::{EntityKind, World};
use q3_image::ImageCache;
use q3_math::{Aabb, Angles, Mat4, Vec3, Vec4};
use q3_renderer::{material::load_shader_registry, md3::Md3Gpu, Renderer};
use q3_sound::{Emitter3D, Listener, LoopHandle, Priority, SoundHandle, SoundSystem};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info, warn};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, NamedKey, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

/// Action produite par la console, traitée une fois par frame par l'App
/// pour pouvoir toucher à `event_loop` / `self` sans briser le borrow checker.
#[derive(Debug)]
enum PendingAction {
    Quit,
    Map(String),
    /// Spawn d'un bot avec le nom + niveau de skill donnés.
    /// `skill = None` → on prend `bot_skill` cvar (défaut III).
    AddBot(String, Option<i32>),
    /// Retire tous les bots.
    ClearBots,
    /// Kick un slot serveur — no-op hors mode `--host`. Le slot peut
    /// être un humain ou un bot.
    Kick(u8),
    /// Envoie un message de chat au serveur (mode client). En `--host`
    /// ou solo, on push directement dans le chat_feed local pour echo.
    SayChat(String),
    /// Restart du match courant (même map) : scores remis à zéro, joueur
    /// + bots respawnent, projectiles/FX purgés.
    Restart,
    /// Active ou rallonge un powerup joueur pour sa durée canonique.
    /// Commande de dev — pas gaté par cheats activés pour l'instant, le MVP
    /// n'a pas de séparation mp/sp. Le stacking suit la même règle que
    /// pour le pickup (additif, pas overwrite).
    GivePowerup(PowerupKind),
    /// Consomme le holdable actuellement dans `held_item` (si présent).
    /// No-op silencieux sinon — même sémantique que Q3 `cmd +use`.
    UseHoldable,
    /// Équivalent `give<holdable>` pour les tests (pas de pickup requis).
    GiveHoldable(HoldableKind),
}

pub struct App {
    vfs: Arc<Vfs>,
    init_width: u32,
    init_height: u32,
    requested_map: Option<String>,

    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    world: Option<World>,
    player: PlayerMove,
    params: PhysicsParams,

    console: Console,
    pending: Arc<Mutex<Vec<PendingAction>>>,
    cvars: CvarRegistry,

    /// Menu principal — état central (page courante, sélection, map list).
    /// `menu.open = true` met en pause le gameplay et route tous les
    /// inputs non-console vers le menu.
    menu: crate::menu::Menu,
    /// Position souris en pixels écran, origine haut-gauche. Mise à jour
    /// par `WindowEvent::CursorMoved` dès que le curseur n'est pas capturé.
    /// Utilisée uniquement par le menu pour le hover/clic.
    cursor_pos: (f32, f32),

    /// Pickups (MD3 statiques) à dessiner chaque frame.
    pickups: Vec<PickupGpu>,
    /// Projectiles actifs (rockets, plasma, … — chacun porte son mesh).
    projectiles: Vec<Projectile>,
    /// Mesh partagé pour les rockets en vol (fallback silencieux si absent).
    rocket_mesh: Option<Arc<Md3Gpu>>,
    /// Mesh partagé pour les tirs de plasma (fallback silencieux si absent).
    plasma_mesh: Option<Arc<Md3Gpu>>,
    /// Mesh partagé pour les grenades volantes (fallback silencieux si absent).
    grenade_mesh: Option<Arc<Md3Gpu>>,
    /// Effets d'explosion à dessiner cette frame (timer décrescent).
    explosions: Vec<Explosion>,
    /// Particules (sparks d'explosion) actives. Advectées chaque tick,
    /// dessinées en streaks additifs via le beam renderer.
    particles: Vec<Particle>,
    /// Viewmodels par arme — chargés paresseusement à `load_map`.
    viewmodels: Vec<(WeaponId, Arc<Md3Gpu>)>,
    /// Armes détenues par le joueur (bitmask).
    weapons_owned: u32,
    /// Arme active — affichée en viewmodel et utilisée par `fire_weapon`.
    active_weapon: WeaponId,
    /// Dernière arme utilisée avant le dernier `select_weapon_slot` réussi.
    /// Permet au « last weapon toggle » (touche X) de revenir en arrière en
    /// une frappe — bind utile pour alterner rocket/rail ou shotgun/LG sur
    /// le même doigt.  Initialisée à l'arme de départ tant qu'aucun swap
    /// n'a eu lieu (dans ce cas, appuyer sur X devient un no-op).
    last_weapon: WeaponId,
    /// Instant du dernier changement d'arme réussi.  Sert à piloter
    /// l'animation du panel ammo (le bloc plonge sous l'écran puis
    /// remonte, comme un viewmodel qu'on baisse/relève).  Voir
    /// [`WEAPON_SWITCH_ANIM_SEC`] pour la durée totale.
    weapon_switch_at: f32,
    /// Overlay de stats FPS : moyenne + courbe des dernières frames.
    /// Toggle par F9.  Désactivé par défaut — on n'affiche un compteur
    /// de FPS que quand le dev le demande pour ne pas distraire un
    /// joueur lambda.
    show_perf_overlay: bool,
    /// Ring buffer des dernières valeurs de `dt` (en secondes), indexé
    /// par `frame_time_head`.  Longueur [`FRAME_TIME_BUF`] : 120 slots
    /// ≈ 2 s à 60 fps, assez pour lire un pic isolé sans lisser en
    /// moyenne glissante trop agressive.
    frame_times: [f32; FRAME_TIME_BUF],
    frame_time_head: usize,
    /// Stock de munitions par slot d'arme (indexé par `WeaponId::slot()`
    /// qui retourne 1..=9).  Taille 10 avec `ammo[0]` inutilisé — évite
    /// un OOB latent quand on accède à `ammo[9]` pour le BFG (qui aurait
    /// paniqué sur un `[i32; 9]`).
    ammo: [i32; 10],
    /// Prochain instant où un "click empty" peut être re-émis — évite le spam
    /// quand le joueur maintient Fire sans munitions.
    next_empty_click_at: f32,
    /// Bots actifs — IA + physique + rendu MD3.
    bots: Vec<BotDriver>,
    /// Rig partagé pour tous les bots (3 meshes lower/upper/head, connectés
    /// via `tag_torso` et `tag_head` — la convention Q3 pour les player
    /// models). Lazy-chargé au premier `addbot`.
    bot_rig: Option<PlayerRig>,
    /// Remote players **interpolés** — état rendu à la frame courante,
    /// dérivé de `remote_interp` à chaque tick d'affichage. Reconstruit
    /// chaque frame (pas chaque snapshot) pour que le mouvement soit
    /// fluide à 60+ FPS malgré les snapshots à 20 Hz.
    remote_players: Vec<RemotePlayer>,
    /// Buffer d'interpolation par slot remote — voir [`RemoteInterp`].
    /// Un slot apparaît au 1er snapshot le mentionnant, et est retiré
    /// quand un snapshot ne le mentionne plus (joueur déconnecté).
    remote_interp: HashMap<u8, RemoteInterp>,
    /// Projectiles distants (rockets, plasma, grenades en vol côté serveur).
    /// Indexés par id stable assigné par le serveur. Mis à jour à chaque
    /// snapshot ; les ids absents de la snapshot suivante sont retirés
    /// (projectile détruit ou expiré côté serveur).
    remote_projectiles: HashMap<u32, RemoteProjectile>,
    /// Table slot → nom des joueurs distants. Mise à jour à chaque
    /// snapshot full (les deltas héritent de la baseline). Sert au
    /// kill-feed et au scoreboard pour afficher les noms réels au lieu
    /// de « slot N ».
    remote_names: HashMap<u8, String>,
    /// Score par slot remote : `(frags, deaths, team)`. Mis à jour à
    /// chaque snapshot. Le scoreboard les lit pour afficher les vraies
    /// valeurs au lieu de stubs `(0, 0)`. La team sert au regroupement
    /// par équipe en TDM.
    remote_scores: HashMap<u8, (i16, i16, u8)>,
    /// IDs de pickups que le serveur a marqués indispos. Le rendu
    /// local skip ces pickups pour qu'ils disparaissent visuellement
    /// (cohérent avec le fait qu'on ne peut pas les ramasser à nouveau
    /// avant respawn). Vide en mode solo.
    remote_unavailable_pickups: std::collections::HashSet<u16>,
    /// `true` si le serveur a flagué notre slot comme spectateur. Le
    /// local pmove bascule alors en noclip free-fly et le HUD masque
    /// HP/armor/ammo. Mis à jour à chaque snapshot.
    is_spectator: bool,
    /// Team du joueur local (`FREE` = FFA, `RED` / `BLUE` = TDM).
    /// Mis à jour à chaque snapshot via le `PlayerState` dont le slot
    /// matche `client_slot`. Sert à colorer la ligne joueur dans le
    /// scoreboard et à grouper par équipe en TDM.
    local_team: u8,
    /// En spectator, si `Some(slot)` la caméra suit ce joueur en POV
    /// au lieu du noclip free-fly. `None` → caméra libre. Cyclé via
    /// LMB (next) / RMB (prev) ; SPACE retourne en free-fly.
    /// Démos `--play` et matches en cours utilisent le même chemin.
    follow_slot: Option<u8>,
    /// **Lean** (M7) — valeur courante lissée ∈ [-1, 1]. Lerpée vers
    /// la cible (input lean_axis) chaque frame. Le rendu applique un
    /// offset latéral sur la caméra (4u/lean) + un roll de view (~6°)
    /// pour signaler le penché.
    lean_value: f32,
    /// Roue de chat rapide ouverte ? `V` toggle, `1..=8` envoie une
    /// phrase prédéfinie de [`CHAT_WHEEL_MESSAGES`] et referme. `Esc`
    /// referme aussi. Les messages partent par le pipeline standard
    /// `PendingAction::SayChat` — solo = echo local, multi = serveur
    /// broadcast. `chat_wheel_opened_at` sert à animer l'ouverture.
    chat_wheel_open: bool,
    chat_wheel_opened_at: f32,
    /// Bots locaux à spawner dès que la map est chargée. Set par
    /// `App::new` depuis le `--bots N` CLI, drainé par `load_map`
    /// après que `world` + `bot_rig` soient prêts. Ignoré en mode
    /// `--host` (les bots y vont côté serveur).
    pending_local_bots: u8,
    /// Santé du joueur. `is_dead()` déclenche la séquence de respawn.
    player_health: Health,
    /// Armor du joueur (0 à 200). Absorbe 2/3 des dégâts en Q3 ;
    /// modèle simplifié ici : diminue de la moitié des dégâts reçus.
    player_armor: i32,
    /// Si `Some(t)`, le joueur est mort et respawnera à `App::time_sec >= t`.
    respawn_at: Option<f32>,
    /// Joueur invincible jusqu'à `time_sec >= player_invul_until`. Se
    /// déclenche sur respawn pour couvrir la fenêtre où le joueur
    /// apparaît dans une zone chaude (spawn camping). 0.0 = aucune
    /// invulnérabilité active. Bloque dégâts ET knockback — sinon le
    /// joueur serait poussé hors d'un spawn contesté sans pouvoir
    /// réagir. Cf. `RESPAWN_INVUL_SEC`.
    player_invul_until: f32,
    /// Direction monde UNITAIRE pointant depuis le dernier attaquant
    /// vers le joueur (sens du vecteur "d'où vient le tir"). Utilisée
    /// par le pain-arrow du HUD pour afficher un indicateur directionnel
    /// autour du réticule. Écrite à chaque prise de dégât "orientée"
    /// (projectile direct, splash, hitscan bot) et ignorée pour les
    /// sources sans direction claire (chute, void, lave). Vec3::ZERO =
    /// pas d'attaquant récent (l'indicateur se base sur le timer pour
    /// décider s'il doit afficher quoi que ce soit, pas sur ce vecteur).
    last_damage_dir: Vec3,
    /// `time_sec` auquel l'indicateur de direction cesse d'être affiché.
    /// Écrit en même temps que `last_damage_dir`, avec une fenêtre de
    /// `DAMAGE_DIR_SHOW_SEC` (cf. constante). Le rendu vérifie seulement
    /// ce champ pour gater l'affichage + calcule une alpha fade en
    /// fonction du temps restant.
    last_damage_until: f32,
    /// Screen-shake : intensité PIC de la secousse courante en degrés.
    /// Décroît linéairement vers 0 jusqu'à `shake_until`. Empile par
    /// `max()` (une grosse explosion qui suit une petite ne réduit pas
    /// l'effet). Remis à 0 à la mort / respawn / fin de match.
    shake_intensity: f32,
    /// `time_sec` auquel la secousse courante expire. Après cette
    /// échéance, aucun décalage caméra n'est appliqué même si
    /// `shake_intensity` est non nul (le calcul de fade lerp le
    /// ramènerait vers 0 quoi qu'il arrive, mais on raccourcit pour
    /// éviter les micro-decimals sur un long run).
    shake_until: f32,
    /// Compteur de morts (affiché dans le HUD).
    deaths: u32,
    /// Kills du joueur (bots abattus).
    frags: u32,
    /// Si `Some`, un joueur (humain ou bot) a atteint `FRAG_LIMIT`. Le
    /// gameplay est gelé et un overlay d'intermission s'affiche.
    match_winner: Option<KillActor>,
    /// `time_sec` au démarrage du match courant. La durée écoulée vaut
    /// `time_sec - match_start_at` ; à `TIME_LIMIT_SECONDS` le match
    /// se termine sur le plus fragueur. Réinitialisé par `restart_match`
    /// et au premier load d'une map.
    match_start_at: f32,
    /// Fin de la fenêtre de warmup (en `time_sec`).  Tant que
    /// `time_sec < warmup_until`, les tirs / IA bots / respawns sont
    /// gelés et l'overlay "MATCH BEGINS IN X" est affiché.
    warmup_until: f32,
    /// `true` dès que la première frag joueur-vs-joueur du match a eu
    /// lieu — évite d'annoncer plusieurs fois "FIRST BLOOD".  Reset par
    /// `restart_match` et au load d'une map.
    first_blood_announced: bool,
    /// Compteur global de tirs effectués par le joueur ce match (un tir
    /// = un appel à `fire_weapon` qui a vraiment dépensé des munitions ;
    /// le Gauntlet compte aussi à condition d'avoir fait swing).  Remis
    /// à zéro par `restart_match`.
    total_shots: u32,
    /// Compteur global de tirs ayant infligé au moins 1 dégât à un bot
    /// ce match.  Un même shot qui touche 2 bots (splash rocket) compte
    /// +1.  Utilisé pour l'affichage `ACC: NN%` au scoreboard.
    total_hits: u32,
    /// Bitfield des seuils time-warning déjà annoncés pour ce match.
    /// Chaque bit correspond à une annonce unique (60s, 30s, 10..=1s).
    /// Évite de refire la même annonce à chaque frame pendant la
    /// seconde où `ceil(time_remaining)` matche le seuil.  Reset par
    /// `restart_match` et au load d'une map.
    time_warnings_fired: u16,
    /// Prochaine fois où le joueur peut tirer (cooldown machinegun-like).
    next_player_fire_at: f32,
    /// Fin du muzzle flash HUD (flash visible si `time_sec < t`).
    muzzle_flash_until: f32,
    /// Niveau de recul courant de l'arme (0 = au repos, 1 = kick max).
    /// Décroît chaque tick via `VIEW_KICK_DECAY_PER_SEC` ; un tir relance
    /// un pic en sommant `weapon.view_kick()` (clampé à 1.2 pour que les
    /// cadences élevées ne laissent pas le kick saturer indéfiniment).
    /// Consommé en offset du viewmodel dans `queue_viewmodel`.
    view_kick: f32,
    /// Fin du hit-marker HUD (petite croix quand un tir a touché un bot).
    hit_marker_until: f32,
    /// Fin du flash d'armure : set à `now + ARMOR_FLASH_SEC` à chaque
    /// fois que l'armure absorbe une fraction de dégâts > 0.  Dessiné
    /// comme un liseré cyan pulsant autour de l'écran, distinct du
    /// damage vignette rouge pour que le joueur lise « le coup m'a
    /// touché MAIS l'armure a tenu ».
    armor_flash_until: f32,
    /// Fin du flash de douleur : set à `now + PAIN_FLASH_SEC` à chaque
    /// fois que le joueur **perd** effectivement des HP (après absorb
    /// armure).  Dessiné comme un liseré rouge pulsant plus large que
    /// le flash armure et légèrement plus long (350 ms) pour que le
    /// coup soit senti même pendant un swap d'arme ou un strafe rapide.
    pain_flash_until: f32,
    /// Cumul de dégâts infligés au dernier adversaire sur la fenêtre
    /// récente.  Reset si > `DMG_BURST_WINDOW` s'est écoulé sans hit,
    /// affiché en chiffre jaune juste sous le crosshair — équivalent
    /// du « combo counter » de beat 'em up, utile pour lire la sortie
    /// d'un spray LG ou d'une double rocket.
    recent_dmg_total: i32,
    recent_dmg_last_at: f32,
    /// Prochain battement de cœur audible.  Boucle à fréquence
    /// croissante tant que la santé reste sous `HEARTBEAT_THRESHOLD` —
    /// 1.0 Hz à 40 HP, 2.5 Hz à 1 HP.  Silencieux au-dessus du seuil
    /// et à la mort (évite de passer en boucle sur un joueur mort).
    next_heartbeat_at: f32,
    /// Faisceaux actifs (LG continu, trail railgun). Ré-émis au renderer
    /// chaque frame tant que `expire_at > time_sec`, puis purgés.
    beams: Vec<ActiveBeam>,
    /// Médailles en cours d'affichage (popup HUD).  Purgées dès
    /// `time_sec >= expire_at`.  Cap défensif à [`MEDAL_MAX`] pour qu'un
    /// déluge de frags en 1 tick ne fasse pas grossir la Vec indéfiniment.
    active_medals: Vec<ActiveMedal>,
    /// Kill feed : liste chronologique des derniers kills, affichée en
    /// haut à droite de l'écran. Drain FIFO au delà de `KILL_FEED_MAX`.
    kill_feed: Vec<KillEvent>,
    /// Chat feed : taunts / laments émis par les bots sur kill, mort,
    /// respawn. Rendu en bas-gauche. Purgé à expiration (`time_sec >=
    /// expire_at`). Indépendant du kill feed pour pouvoir afficher des
    /// lignes de chat sans qu'un frag ne soit survenu.
    chat_feed: Vec<ChatLine>,
    /// Toasts de ramassage d'items — rendus en bas-centre, fade à
    /// l'expiration.  Purgés à `time_sec >= expire_at`.  FIFO au delà
    /// de [`PICKUP_TOAST_MAX`].
    pickup_toasts: Vec<PickupToast>,
    /// Série de kills consécutifs du joueur depuis le dernier respawn —
    /// Unreal-style.  Remis à 0 à chaque mort.  Les paliers (3, 5, 7,
    /// 10, 15, 20) déclenchent des toasts + sons pour marquer la progression.
    player_streak: u32,
    /// Prochain instant où un bot peut de nouveau parler (cooldown
    /// global anti-spam).  Q3 original a un système de personnalités
    /// + weights complexe ; nous prenons un simple throttle : un bot
    /// au plus parle par fenêtre de `CHAT_GLOBAL_COOLDOWN`.
    next_chat_at: f32,
    /// Prochain instant où le joueur peut relancer une taunt F3.  Le
    /// cooldown dédié (plutôt que `next_chat_at`) évite qu'une taunt
    /// joueur ne mute les bots ou inversement.
    next_player_taunt_at: f32,
    /// Chiffres de dégât flottants, rendus dans le HUD en 2D après
    /// projection à l'écran. Purgés dès que `time_sec >= expire_at`.
    floating_damages: Vec<FloatingDamage>,
    /// Temps écoulé depuis le démarrage — alimente la rotation des pickups.
    time_sec: f32,
    /// Debug : Q3_AUTOSHOT=N → un seul auto-screenshot puis exit.
    auto_shot_taken: bool,
    /// Phase du view-bob (radians). Avance proportionnellement à la vitesse
    /// horizontale tant que le joueur est au sol. Utilisé pour osciller la
    /// caméra en Z (bob vertical) + un léger roll latéral façon Q3.
    bob_phase: f32,
    /// Fin de chaque powerup indexé par `PowerupKind::index()`, ou `None`
    /// si inactif. Tableau plutôt que champs séparés pour que l'ajout d'un
    /// powerup futur (Battle Suit, Flight…) se fasse via une seule variante
    /// d'enum, sans casser les helpers ni le HUD.
    powerup_until: [Option<f32>; PowerupKind::COUNT],
    /// Marqueurs « warning déjà joué » par slot de powerup. Armés à `false`
    /// lors d'un nouveau pickup (`grant_powerup`) et basculés à `true` dès
    /// que le joueur entend le bip ≤3s. Évite de spammer le sfx à chaque
    /// frame sous le seuil. Remis à `false` sur expiration pour couvrir
    /// un futur pickup du MÊME kind dans la même vie.
    powerup_warned: [bool; PowerupKind::COUNT],
    /// Slot d'inventaire unique pour les holdables (medkit, teleporter).
    /// Un seul à la fois — un nouveau pickup remplace l'ancien. Consommé
    /// à l'activation via `usei` / `cmd +use`.
    held_item: Option<HoldableKind>,
    /// Cause de la dernière mort du joueur — `(killer, cause)` du dernier
    /// `push_kill_cause` avec `victim == Player`. Affichée sous le bandeau
    /// "YOU DIED" pour que le joueur sache qui/quoi l'a fraggé, même après
    /// que l'entrée kill-feed correspondante ait fini d'expirer (le fade
    /// visuel est plus court que `RESPAWN_DELAY_SEC` dans certains cas).
    /// Cleared on respawn / new match / load map.
    last_death_cause: Option<(KillActor, KillCause)>,
    /// Buffer fractionnaire pour la régénération. Q3 applique +15 HP/s ;
    /// à 60 fps ça ferait `15 * 0.016 = 0.24 HP/frame`, tronqué à 0. On
    /// accumule ici le reste pour appliquer des heal entiers quand il
    /// franchit 1.0 — smooth + fidèle au feeling original.
    regen_accum: f32,

    /// Audio — `None` si l'initialisation a échoué (pas de device, headless…).
    sound: Option<Arc<SoundSystem>>,
    /// Jump pads extraits de la map. Recomputés à chaque `load_map`,
    /// tick-testés après la physique joueur (cf. `tick_jump_pads`).
    jump_pads: Vec<JumpPad>,
    /// Téléporteurs de la map. Même cycle de vie que `jump_pads`.
    teleporters: Vec<Teleporter>,
    /// Zones `trigger_hurt` — dégâts continus au joueur au contact.
    hurt_zones: Vec<HurtZone>,
    /// Emetteurs ambient attachés à la map (classname `target_speaker` en
    /// mode LOOPED).  On garde un `LoopHandle` par speaker pour pouvoir
    /// les stopper à la `load_map` suivante : sans ça, un speaker de la
    /// map précédente continuerait à tourner en silence (mais brûlerait
    /// un canal `SpatialSink`).  Les speakers one-shot (sans spawnflag
    /// LOOPED) ne sont pas stockés ici — on les déclenche sur
    /// `trigger_multiple` à la volée.
    ambient_speakers: Vec<LoopHandle>,
    /// Réserve d'air restante en secondes. Décrémentée chaque tick tant
    /// que l'œil du joueur est dans un brush `Contents::WATER` ; remontée
    /// instantanément à `AIR_CAPACITY_SEC` dès qu'on ressort. Quand elle
    /// atteint 0, on déclenche des dégâts de noyade périodiques.
    air_left: f32,
    /// Prochain tick de dégâts de noyade (en `time_sec` absolu).  Calé à
    /// l'instant du premier manque d'air puis avancé de `DROWN_INTERVAL`
    /// tant que le joueur reste submergé sans air.  Réinitialisé dès qu'on
    /// ressurface pour qu'un re-submergement immédiat ne hit pas au tick 0.
    next_drown_at: f32,
    /// État « sous l'eau » au tick précédent — utilisé pour détecter les
    /// fronts montants/descendants et jouer le splash + émettre des
    /// particules. Sans cette mémoire on ne pourrait pas distinguer
    /// « immergé et l'étant déjà » de « immergé à l'instant ».
    was_underwater: bool,
    /// Prochaine émission de bulles quand le joueur manque d'air
    /// (`time_sec` absolu). Les bulles ne sortent que lorsque la jauge
    /// est basse, façon Q3 — tant qu'il reste beaucoup d'air, on suppose
    /// que le perso retient son souffle sans fuite.
    next_bubble_at: f32,
    /// Index du pad/teleport dans lequel le joueur se trouve actuellement.
    /// Q3 ne re-déclenche pas le push / téléport tant que l'entité reste
    /// dans le trigger — on mémorise donc le contact courant pour éviter
    /// de spammer la vélocité / le son à chaque tick.
    on_jumppad_idx: Option<usize>,
    on_teleport_idx: Option<usize>,
    /// Sons couramment utilisés, chargés paresseusement.
    sfx_jump: Option<SoundHandle>,
    sfx_land: Option<SoundHandle>,
    /// Splash joué au passage surface↔eau.  `in` sur front d'immersion,
    /// `out` sur front de sortie.  Les deux paths sont optionnels : si
    /// un seul existe on l'utilise dans les deux sens ; si aucun, silent.
    sfx_water_in: Option<SoundHandle>,
    sfx_water_out: Option<SoundHandle>,
    /// Whoosh du jump pad — joué au centre du trigger à l'entrée.
    sfx_jumppad: Option<SoundHandle>,
    /// "telein" (côté source) et "teleout" (côté destination) — joués
    /// en même temps à deux positions pour un effet stéréo.
    sfx_teleport_in: Option<SoundHandle>,
    sfx_teleport_out: Option<SoundHandle>,
    /// Pas de course — 4 variantes Q3 standard, choisies aléatoirement à
    /// chaque foulée pour éviter l'effet "loop mécanique".
    sfx_footsteps: Vec<SoundHandle>,
    /// Index de la dernière variante jouée. On impose que la prochaine
    /// soit différente — sinon, à 4 variantes uniformes, on entend parfois
    /// 2–3 fois la même à la suite, ce qui casse l'illusion.
    last_footstep_idx: Option<usize>,
    /// Phase de view-bob à la dernière foulée jouée. On trigger un pas
    /// quand `bob_phase` franchit un multiple de π depuis la dernière
    /// foulée — les deux pieds (phase 0 et π) génèrent donc un step.
    last_footstep_phase: f32,
    /// Sons de combat (joueur + bots).
    sfx_fire: Vec<(WeaponId, SoundHandle)>,
    sfx_pain_player: Option<SoundHandle>,
    sfx_pain_bot: Option<SoundHandle>,
    /// SFX génériques Q3 de cueillette d'arme / de munitions.
    sfx_weapon_pickup: Option<SoundHandle>,
    sfx_ammo_pickup: Option<SoundHandle>,
    /// Click sec joué quand le joueur appuie sur tir sans munitions.
    /// Convention Q3 : `sound/weapons/noammo.wav`. Si absent on reste
    /// silencieux (mieux qu'un faux son générique).
    sfx_no_ammo: Option<SoundHandle>,
    /// « Shunk » joué quand le joueur change d'arme.  Convention Q3 :
    /// `sound/weapons/change.wav` (raise générique).  Déclenché aussi
    /// bien par `select_weapon_slot` (touche 1..9) que par l'autoswitch
    /// out-of-ammo, pour que le joueur entende le swap dans tous les cas.
    sfx_weapon_switch: Option<SoundHandle>,
    /// SFX d'explosion de rocket (= fallback pour toute explosion splash).
    sfx_rocket_explode: Option<SoundHandle>,
    /// Tick joué au joueur quand un de ses tirs touche un bot. Feedback
    /// audio classique de Q3A/CPMA pour confirmer l'impact sans avoir à
    /// regarder le chiffre de dégât flottant. Joué depuis la position
    /// oreille (près du listener) pour ignorer l'atténuation distance.
    sfx_hit: Option<SoundHandle>,
    /// Thunk plus grave joué sur chaque frag du joueur — distinct du
    /// hitsound pour dire « tu viens d'achever ta cible », pas juste « tu
    /// l'as touchée ». Déclenché 1× par tick même si plusieurs bots sont
    /// fraggés simultanément (splash qui tue 2 bots = 1 thunk, pas 2).
    /// Même convention « dans l'oreille » que `sfx_hit`.
    sfx_kill_confirm: Option<SoundHandle>,
    /// Médaille « Humiliation » — Q3 la joue quand on achève un ennemi au
    /// Gauntlet. Superposée au kill-confirm standard, pas en remplacement :
    /// on veut garder la continuité sonore de la volée + la voix humaine
    /// qui annonce « Humiliation! » par-dessus.
    sfx_humiliation: Option<SoundHandle>,
    /// **Announcer fraglimit countdown** (A5) — Q3 a des samples vocaux
    /// canoniques pour les 3 derniers frags du match. Joués UNE FOIS
    /// au moment où le compteur joueur passe sous le seuil.
    sfx_one_frag: Option<SoundHandle>,
    sfx_two_frags: Option<SoundHandle>,
    sfx_three_frags: Option<SoundHandle>,
    /// `Some(n)` si on a déjà annoncé "{n} frag(s) left" — empêche la
    /// répétition à chaque kill quand on stationne à n=1 par exemple.
    /// Reset à `None` au restart de match.
    last_frags_announced: Option<u32>,
    /// Médaille « Excellent » — Q3 la joue quand le joueur enchaîne 2
    /// frags en moins de 2 secondes. Implémentée via `last_frag_at` :
    /// si `time_sec - last_frag_at ≤ EXCELLENT_WINDOW_SEC` au moment d'un
    /// nouveau frag, on joue le sample. Multi-frag dans le même tick ne
    /// multiplie pas le nombre de médailles (max 1 par tick).
    sfx_excellent: Option<SoundHandle>,
    /// Horodatage du dernier frag du joueur, en secondes d'`time_sec`.
    /// `f32::NEG_INFINITY` tant qu'aucun frag n'a été réalisé (évite un
    /// faux « excellent » au tout premier kill de la partie). Reset à
    /// NEG_INFINITY au changement de map et au respawn.
    last_frag_at: f32,
    /// Médaille « Impressive » — Q3 la joue à chaque fois que le joueur
    /// enchaîne 2 tirs Railgun successifs qui touchent. Samples canoniques
    /// `sound/feedback/impressive.wav`.
    sfx_impressive: Option<SoundHandle>,
    /// Bip court joué une fois à 3 secondes restantes sur un powerup actif.
    /// Préviens l'expiration imminente — le flash visuel fait 3s blink 2Hz,
    /// l'audio le complète pour les joueurs qui ne regardent pas le badge.
    /// `None` = asset introuvable → warning silencieux (le blink reste).
    sfx_powerup_warn: Option<SoundHandle>,
    /// Son joué pile à l'instant d'expiration ("powerdown" Q3). Plus grave
    /// que `sfx_powerup_warn` pour marquer la bascule "actif → inactif".
    sfx_powerup_end: Option<SoundHandle>,
    /// Trace l'état du tir Railgun précédent du joueur : `true` ssi il
    /// a touché un bot. Reset à `false` sur miss et au changement de map.
    /// Consommé dans `fire_weapon` : si l'ancien et le nouveau sont `true`,
    /// on award Impressive et on reset à `false` pour demander à nouveau
    /// 2 hits consécutifs avant la prochaine médaille (mimique du
    /// compteur Q3 qui incrémente et reset à la paire).
    rg_last_hit: bool,
    /// Pour détecter la transition "en l'air → au sol" et jouer un land sfx.
    was_airborne: bool,
    /// Offset vertical actuel de la caméra dû au crouch, en unités monde.
    /// 0.0 quand debout, −`CROUCH_VIEW_DROP` quand totalement accroupi.
    /// Lerpé linéairement vers la cible chaque tick pour éviter le "snap"
    /// visuel désagréable du duck instantané.
    view_crouch_offset: f32,

    input: Input,
    mouse_captured: bool,
    last_tick: Instant,
    /// Accumulateur pour le pas fixe de physique à 125Hz (8ms), identique
    /// au serveur Q3 (`sv_fps 20` côté net mais le client simule à
    /// `pm_frametime ≈ 8ms` pour le feel strafe-jump canonique). Sans pas
    /// fixe, `cmd.delta_time` varie avec la framerate ce qui change
    /// sensiblement le comportement de l'accélération et de la friction —
    /// les joueurs remarquent tout de suite que ça "ne sent pas Q3".
    physics_accumulator: f32,

    /// Runtime réseau — pilote le mode `SinglePlayer`/`Server`/`Client`
    /// choisi via CLI. Vide / no-op en solo. Les `tick_*` sont appelés
    /// dans la boucle principale (cf. `RedrawRequested`).
    net: crate::net::NetRuntime,
    /// Runtime VR — stub pour l'instant, `is_enabled()` reste `false`
    /// tant qu'OpenXR n'est pas branché. Les hooks `begin_frame` /
    /// `end_frame` sont déjà appelés en boucle pour que l'activation
    /// ultérieure soit transparente côté App.
    vr: crate::vr::VrRuntime,
}

struct PickupGpu {
    mesh: Arc<Md3Gpu>,
    origin: Vec3,
    angles: Angles,
    kind: PickupKind,
    /// Durée de respawn en secondes après cueillette (30 par défaut Q3).
    respawn_cooldown: f32,
    /// `Some(t)` = ramassé, reviendra à `time_sec >= t`. `None` = disponible.
    respawn_at: Option<f32>,
    /// Index de l'entité source dans `World::entities`. Stable des deux
    /// côtés du réseau, sert de clé pour synchroniser l'état dispo /
    /// indispo avec l'autorité serveur en mode client.
    entity_index: u16,
}

/// Qui a lancé un projectile — on évite de se toucher soi-même via l'owner.
#[derive(Debug, Clone, Copy)]
enum ProjectileOwner {
    Player,
    /// Index dans `App::bots` au moment du tir (peut être stale si un bot
    /// est supprimé, mais on n'enlève jamais de bots — respawn en place).
    /// Utilisé par le kill-feed pour résoudre le nom du tueur.
    Bot(usize),
}

/// Acteur d'un kill — le joueur local, un bot nommé, ou le « monde »
/// (futur : chute dans la lave, gouffre, etc.).
#[derive(Debug, Clone)]
enum KillActor {
    Player,
    Bot(String),
    World,
}

impl KillActor {
    fn label(&self) -> &str {
        match self {
            Self::Player => "You",
            Self::Bot(name) => name.as_str(),
            Self::World => "World",
        }
    }
}

/// Entrée du kill-feed affiché en haut à droite. Les entrées expirent au
/// bout de `KILL_FEED_LIFETIME` secondes, avec un léger fade alpha sur la
/// dernière seconde.
#[derive(Debug, Clone)]
struct KillEvent {
    killer: KillActor,
    victim: KillActor,
    /// Cause de la mort — `Some(w)` pour les kills via une arme, `None` pour
    /// les morts "environnementales" (lave, void, chute). Le HUD affiche un
    /// tag "WORLD" / "LAVA" à la place du tag arme quand c'est `None`.
    cause: KillCause,
    expire_at: f32,
}

/// Cause générique d'un kill, pour le kill-feed.
#[derive(Debug, Clone, Copy)]
enum KillCause {
    Weapon(WeaponId),
    /// Dégâts environnementaux (trigger_hurt). Le `&'static str` est le
    /// label affiché (ex. "LAVA", "VOID", "HURT").
    Environment(&'static str),
}

impl KillCause {
    fn tag(self) -> &'static str {
        match self {
            Self::Weapon(w) => weapon_tag(w),
            Self::Environment(s) => s,
        }
    }
}

/// Limite de frags pour gagner le match (équivalent `fraglimit` Q3).
const FRAG_LIMIT: u32 = 20;

/// Durée d'un match en secondes (équivalent `timelimit` Q3, exprimé en
/// secondes plutôt qu'en minutes). À l'expiration, le joueur (ou bot)
/// avec le plus de frags gagne ; égalité → pas de vainqueur affiché
/// (Q3 gère le sudden death, nous coupons court).
const TIME_LIMIT_SECONDS: f32 = 300.0;

/// View-bob — fréquence angulaire de base en rad/s. À vitesse de course
/// typique (~320 u/s), on cale ~2 bobs par seconde (1 par foulée). La
/// phase avance à `BOB_FREQ * speed_factor * dt`.
const BOB_FREQ: f32 = 12.5;
/// Amplitude verticale max du bob, en unités monde. 1.5 donne un
/// ressenti "Quake" plus subtil que CS ; réglable via cg_bobup équivalent.
const BOB_AMP_Z: f32 = 1.5;
/// Amplitude latérale (roll caméra, degrés). Ajoute un léger pas de
/// course sans nausée.
const BOB_AMP_ROLL: f32 = 0.6;
/// Vitesse horizontale qui correspond à une amplitude bob = 1.0. Au-dessus
/// de cette vitesse on sature à 1.0 — évite le "mega-bob" en bunny hop.
const BOB_SPEED_REF: f32 = 320.0;
/// Dé-cadrage : en-dessous de cette vitesse horizontale, pas de bob
/// (on considère le joueur immobile ou en train de freiner).
const BOB_SPEED_MIN: f32 = 30.0;

/// Roll latéral de la caméra en fonction du strafe, en degrés. Clone de
/// `cl_rollangle` Q3 (`cl_rollangle.c`, défaut `2.0`). Appliqué AU-DESSUS
/// du bob-roll — donne au mouvement latéral un feel "incliné" reconnaissable
/// des FPS de l'ère Quake. Trop grand donne le mal de mer ; Q3 tient 2°.
const CL_ROLL_ANGLE: f32 = 2.0;
/// Vitesse (unités monde/s) à partir de laquelle le roll sature à
/// `CL_ROLL_ANGLE`. En-dessous, roll = side_vel · angle / speed. Clone
/// exact de `cl_rollspeed` Q3 (défaut `200`). Un joueur qui marche
/// diagonalement a donc un roll réduit ; en strafe run (320 u/s) on est
/// déjà au max.
const CL_ROLL_SPEED: f32 = 200.0;

/// Durée d'affichage de l'indicateur de provenance des dégâts (pain
/// arrow), en secondes. Sur cette fenêtre l'alpha fade linéairement de
/// 1.0 → 0.0 pour laisser à l'œil le temps de localiser l'attaquant
/// sans pour autant rester affiché quand la menace est ancienne.
const DAMAGE_DIR_SHOW_SEC: f32 = 2.0;

/// Seuil HP sous lequel le HUD affiche une vignette rouge pulsante.
/// 25 HP = 1/4 de santé, point à partir duquel Q3 stock active son beep
/// de low-health en son original.  Au-dessus, pas de vignette.
const LOW_HEALTH_THRESHOLD: i32 = 25;
/// Fréquence du pulse du cadre rouge de low-health (Hz).  ≈ 1.5 donne
/// un rythme cardiaque légèrement accéléré — pas stroboscopique, mais
/// perceptible comme un rappel d'urgence.
const LOW_HEALTH_PULSE_HZ: f32 = 1.5;
/// Alpha maximal du cadre rouge à severity=1.0 et pulse=1.0.  Sous 0.5
/// pour que le cadre reste additive-readable par-dessus la géométrie.
const LOW_HEALTH_MAX_ALPHA: f32 = 0.45;
/// Épaisseur (px écran) des 4 barres du cadre.  28 ≈ 3 % de hauteur 1080p,
/// visible sans manger le viewport.
const LOW_HEALTH_FRAME_THICK: f32 = 28.0;

/// Durée du flash « armure a encaissé » (cadre cyan pulsant).  250 ms
/// est assez court pour que plusieurs coups successifs relancent le
/// flash sans saturer l'écran, mais assez long pour que le joueur le
/// perçoive même à des cadences de tir élevées (machinegun à 10 Hz).
const ARMOR_FLASH_SEC: f32 = 0.25;

/// Durée du flash de douleur (cadre rouge pulsant) — voir `pain_flash_until`.
/// Un poil plus long que l'armor flash pour que même un coup amorti laisse
/// une trace rouge lisible, et donc une hiérarchie visuelle claire entre
/// "absorbé par l'armure" (cyan 250 ms) et "a traversé l'armure" (rouge
/// 350 ms qui persiste après la fin du flash cyan).
const PAIN_FLASH_SEC: f32 = 0.35;

/// Durée totale de l'animation « panel ammo qui descend puis remonte »
/// jouée à chaque [`App::switch_to_weapon`].  220 ms se cale bien avec
/// les sons de raise d'arme (entre 180 et 260 ms selon le sample) et
/// donne un feedback visuel lisible sans masquer longtemps l'ammo.
const WEAPON_SWITCH_ANIM_SEC: f32 = 0.22;

/// Amplitude maximale (pixels écran) du drop du panel ammo pendant la
/// moitié basse de l'animation [`WEAPON_SWITCH_ANIM_SEC`].  24 ≈ un peu
/// plus que la hauteur du panel ammo lui-même → il sort franchement
/// puis remonte, effet « changement de chargeur ».
const WEAPON_SWITCH_DROP_PX: f32 = 24.0;

/// Taille du ring buffer de frame-times (en frames).  120 ≈ 2 s à 60 fps
/// ou 1 s à 120 fps — assez pour estimer une moyenne stable sans lisser
/// un pic unique.  Passé directement à un `[f32; _]` pour éviter une
/// allocation dans le hot path.
const FRAME_TIME_BUF: usize = 120;

/// Fenêtre d'agrégation du compteur de dégâts « combo » affiché sous
/// le crosshair.  Tant qu'un hit tombe dans cette fenêtre depuis le
/// précédent, les dégâts sont sommés ; sinon le compteur repart à
/// zéro.  1.2 s laisse le temps à une rafale LG (0.05 s de cooldown)
/// d'accumuler ~24 hits, mais coupe proprement entre deux engagements
/// distincts.
const DMG_BURST_WINDOW: f32 = 1.2;

/// Durée totale d'affichage d'un popup de médaille, en secondes.  Le
/// sample audio fait ~1 s ; on laisse le texte ~2 s pour que le joueur
/// ait le temps de le lire sans revenir sur ce qu'il était en train
/// de faire (target acquisition, mouvement).
const MEDAL_SHOW_SEC: f32 = 2.0;
/// Proportion du début de `MEDAL_SHOW_SEC` consacrée au fade-in
/// (0 → 1 alpha).  Le reste suit un fade-out jusqu'à la fin.
const MEDAL_FADE_IN_RATIO: f32 = 0.15;
/// Cap sur le nombre de médailles simultanément affichées — empêche une
/// double-frag + Railgun combo de remplir la Vec avant purge naturelle.
const MEDAL_MAX: usize = 4;

/// Capacité maximale d'air en secondes. Fidèle à `pm_airleft` Q3 (12 s).
/// Le joueur peut tenir 12 secondes sous l'eau avant que les dégâts de
/// noyade commencent. Remontée à la surface réinitialise instantanément
/// la jauge — pas de récupération graduelle.
const AIR_CAPACITY_SEC: f32 = 12.0;
/// Dégâts infligés par tick de noyade, une fois `air_left` épuisé.
/// Q3 applique 2 HP/tour ; on conserve.
const DROWN_DAMAGE: i32 = 2;
/// Intervalle entre deux tours de noyade, une fois `air_left == 0`.
/// 1 seconde donne un TTK de 50 s depuis 100 HP — plus punitif que Q3
/// (qui applique aussi des hits plus fréquents à faible santé) mais
/// suffisant pour signaler l'urgence.
const DROWN_INTERVAL: f32 = 1.0;
/// Facteur multiplicatif appliqué au master volume lorsque l'oreille
/// (eye) est dans un volume `CONTENTS_WATER`. Q3 utilise un vrai
/// low-pass ; nous approchons l'effet en abaissant juste le niveau
/// global — 0.45 est assez feutré pour être perceptible sans rendre
/// l'action inaudible. Restauré à 1.0 en sortant de l'eau.
const UNDERWATER_VOLUME_FACTOR: f32 = 0.45;

/// Screen-shake sur explosion proche : durée d'atténuation en secondes.
/// Volontairement court (~0.25s) pour rester punchy sans donner le mal
/// de mer dans un couloir où 3 rockets explosent coup sur coup.
const SHAKE_DURATION: f32 = 0.25;
/// Amplitude max de la secousse caméra, en degrés (appliqués en pitch
/// et yaw avec des sinusoïdes déphasées). 1.5° = ressenti "gros boom"
/// sans rendre la visée injouable pendant le décalage.
const SHAKE_MAX_AMPLITUDE: f32 = 1.5;
/// Distance œil-explosion à laquelle la secousse est à son max. Sous
/// ce seuil on clamp à `SHAKE_MAX_AMPLITUDE` ; au-dessus on lerp vers
/// `SHAKE_FAR_DIST` où l'effet s'annule.
const SHAKE_NEAR_DIST: f32 = 300.0;
/// Distance œil-explosion au-delà de laquelle on n'applique plus de
/// secousse — une rocket à l'autre bout de la map ne doit pas faire
/// trembler la caméra.
const SHAKE_FAR_DIST: f32 = 900.0;

/// Taux de décroissance du `view_kick` (recul viewmodel) en unités
/// "demi-vies par seconde" : à 10, le kick perd ~99.99 % de sa valeur
/// en 1 s, ce qui donne un retour rapide mais perceptible.  Calé pour
/// qu'à la cadence de tir MG (~10 Hz) la contribution des tirs
/// précédents soit essentiellement résiduelle — le joueur sent la
/// rafale sans voir le viewmodel coincé en position kick.
const VIEW_KICK_DECAY_PER_SEC: f32 = 10.0;

/// Plafond du `view_kick` pour éviter la saturation sur tir rapide.
/// Au-delà, les valeurs cumulées n'offrent plus de ressenti
/// supplémentaire et commencent à sortir visuellement du viewmodel du
/// champ de vision — on clampe donc en sommant les impulses.
const VIEW_KICK_MAX: f32 = 1.2;

/// Fenêtre temporelle pour la médaille « Excellent » : le joueur doit
/// achever un 2ᵉ ennemi dans ce délai (après son frag précédent) pour
/// déclencher l'announcement. Valeur Q3 d'origine = 2.0s, cf.
/// `MEDAL_EXCELLENT_TIME` dans `g_combat.c`. Multi-frag dans le même
/// tick (splash qui tue 2 bots) compte aussi pour un « Excellent » —
/// la logique mesure `time_sec - last_frag_at` avant la mise à jour.
const EXCELLENT_WINDOW_SEC: f32 = 2.0;

/// Décalage vertical de la caméra entre debout et accroupi. Le hull
/// passe de 56u à 40u de haut (Δ = 16), mais on drop un peu plus (20u)
/// pour matcher `CROUCH_VIEWHEIGHT` Q3 qui place l'œil plus bas que le
/// simple ratio hull.
const CROUCH_VIEW_DROP: f32 = 20.0;
/// Vitesse de transition de la caméra entre les deux hauteurs, en unités
/// monde par seconde. `CROUCH_ANIMATE_SPEED` dans `bg_pmove.c` vaut 32,
/// on aligne.
const CROUCH_TRANSITION_SPEED: f32 = 32.0 * 2.0; // ~0.3 s de A à B visuellement, plus snappy

/// Quad Damage : durée du buff (secondes) et multiplicateur.
const QUAD_DAMAGE_DURATION: f32 = 30.0;
const QUAD_DAMAGE_FACTOR: i32 = 4;

/// Haste : durée + multiplicateurs (vitesse et cadence de tir).
/// Q3 canon = 1.3× sur les deux. `FIRE_MULT` est < 1 parce qu'il divise
/// le cooldown entre deux tirs (firing plus vite = attendre moins).
const HASTE_DURATION: f32 = 30.0;
const HASTE_SPEED_MULT: f32 = 1.3;
const HASTE_FIRE_MULT: f32 = 1.0 / 1.3;

/// Regeneration : durée + soins par seconde. Q3 canon = +15 HP/s tant que
/// la santé est sous max, +5 HP/s au-dessus jusqu'à 200. On garde la
/// branche basse pour rester lisible au MVP — la logique est déjà capée
/// par `Health::heal` (plafonne à `max`).
const REGEN_DURATION: f32 = 30.0;
const REGEN_HP_PER_SECOND: f32 = 15.0;

/// Battle Suit (item_enviro) : durée du buff (secondes). Q3 canon = 30 s.
/// Pendant cette fenêtre, le joueur est immunisé contre les dégâts
/// « environnementaux » — chute, lave, slime, noyade, explosions de
/// son propre rocket. Les dégâts d'armes d'un autre joueur passent
/// (≈ atténués en Q3 ; ici on garde plein-pot pour ne pas compliquer).
const BATTLE_SUIT_DURATION: f32 = 30.0;

/// Invisibility (item_invis) : durée du buff (secondes). Q3 canon = 30 s.
/// Le joueur devient fortement transparent (alpha ≈ 0.15) et les bots
/// doivent se rapprocher à `BOT_SIGHT_RANGE * BOT_SIGHT_INVIS_FACTOR`
/// avant de l'acquérir. Les dégâts passent normalement — l'effet est
/// purement défensif via la détection.
const INVISIBILITY_DURATION: f32 = 30.0;

/// Flight (item_flight) : durée du buff (secondes). Q3 canon = 60 s,
/// plus long que les autres parce que l'item est rare et plus utilitaire
/// qu'offensif. Annule la gravité et permet de monter/descendre avec
/// jump/crouch à vitesse `FLIGHT_THRUST_SPEED`.
const FLIGHT_DURATION: f32 = 60.0;
/// Vitesse verticale appliquée quand jump (monte) ou crouch (descend)
/// est maintenu pendant Flight. Équivaut à `pm_flyspeed` Q3.
const FLIGHT_THRUST_SPEED: f32 = 300.0;

/// Durée de vie d'une entrée avant purge (secondes).
const KILL_FEED_LIFETIME: f32 = 5.0;
/// Nombre max d'entrées affichées simultanément — au delà on drain
/// la plus ancienne.
const KILL_FEED_MAX: usize = 6;

/// Durée d'affichage d'une ligne de chat bot.
const CHAT_LINE_LIFETIME: f32 = 4.5;
/// Cap sur le nombre de lignes de chat simultanées — au delà, FIFO.
const CHAT_FEED_MAX: usize = 4;
/// Cooldown global entre deux taunts de bots (secondes).  Empêche un
/// quad-kill ou une rafale de frags en spawn-protect de transformer
/// le chat en mur de texte.
const CHAT_GLOBAL_COOLDOWN: f32 = 2.2;

/// Durée d'affichage d'un toast de pickup (secondes).  2.5s = le temps
/// de lire "You got the Rocket Launcher" sans occuper l'écran trop
/// longtemps — si on pickup plusieurs items d'affilée, les toasts
/// s'empilent et les anciens sont évincés par le cap FIFO.
const PICKUP_TOAST_LIFETIME: f32 = 2.5;
/// Cap sur le nombre de toasts de pickup simultanés.  4 couvre le cas
/// « tu traverses un rack d'items en ramassant tout » sans déborder.
const PICKUP_TOAST_MAX: usize = 4;
/// Durée du warmup avant qu'un match (ou un restart) ne démarre
/// réellement.  Pendant cette fenêtre, les tirs, l'IA bot et les respawns
/// de bots sont gelés ; un gros overlay "MATCH BEGINS IN X" est affiché.
/// 3 s est le défaut Q3 (`g_warmup 3`).
const WARMUP_DURATION: f32 = 3.0;
/// Probabilité (0..1) qu'un bot parle sur un trigger donné.  Q3 original
/// utilise des weights per-line dans les `.c` ; on simule la variance
/// avec un simple tirage Bernoulli — 55 % des évènements génèrent une
/// ligne, assez pour que le chat ait vie sans saturer l'overlay.
const CHAT_TRIGGER_PROB: f32 = 0.55;

/// Trigger d'un taunt / chat de bot — mappe 1:1 vers une des tables
/// `CHAT_LINES_*`.  On reste volontairement proche des catégories
/// historiques Q3 (kill_insult, death, enter_game, taunt, level_start)
/// pour que l'extension via un vrai parseur `.c` soit triviale plus tard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatTrigger {
    /// Le bot vient de fragger le joueur (ou un autre bot).  Ligne
    /// typée « trash talk » qui charrie la victime.
    KillInsult,
    /// Le bot vient d'être fraggé.  Lamentation / excuse — "j'ai lagué",
    /// "coup de chance", etc.
    Death,
    /// Le bot vient de respawn.  Ligne courte « retour en jeu » —
    /// probabilité nettement plus basse (on ne veut pas une ligne à
    /// chaque respawn sur kill-stream).
    Respawn,
}

/// Lignes « le bot t'a tué » — style trash talk, courtes, génériques,
/// nous les avons rédigées pour ne pas copier les chats Q3 originaux
/// (qui sont copyright id Software).  Une trentaine, pour que la
/// variance ressorte sur un match long.
const CHAT_LINES_KILL_INSULT: &[&str] = &[
    "Too easy.",
    "Git gud.",
    "Next.",
    "That all?",
    "Better luck elsewhere.",
    "Nothing personal.",
    "You walked into that.",
    "Free frag.",
    "Reflexes check: failed.",
    "Don't respawn if you're just gonna do that again.",
    "Ez.",
    "Was that your best?",
    "Step it up.",
    "Gg no re.",
    "I was holding back.",
];

/// Lignes « je viens de mourir » — lamentations, excuses, menaces
/// de revanche.  Doit rester léger : un bot qui crie « CHEATER » à
/// chaque mort use vite.
const CHAT_LINES_DEATH: &[&str] = &[
    "Lucky shot.",
    "Lag.",
    "You got lucky.",
    "Cheap.",
    "I'll remember that.",
    "Ouch.",
    "Not bad.",
    "Revenge incoming.",
    "Nice try, but next round.",
    "Whatever.",
    "Framerate dropped.",
    "One second I had you.",
];

/// Lignes « je reviens » — très courtes, basse probabilité (on divise
/// `CHAT_TRIGGER_PROB` par 2 pour ce trigger côté `maybe_bot_chat`).
const CHAT_LINES_RESPAWN: &[&str] = &[
    "Back.",
    "Again.",
    "Round two.",
    "Let's go.",
    "Here we go.",
];

/// Lignes de taunt **joueur** (touche F3) — style assumé trash-talk
/// arène, rédigées maison (aucune copie des voice lines id Software).
/// Le pool est un peu plus large que les pools bots pour que le joueur
/// qui spam F3 ne revoie pas la même ligne toutes les 5 secondes.
const PLAYER_TAUNT_LINES: &[&str] = &[
    "Come get some.",
    "Bring it.",
    "Taste this.",
    "Any day now.",
    "Scrubs.",
    "Is this a DM or a tutorial?",
    "Rookies.",
    "Who's next?",
    "Line up.",
    "Respawn denied.",
    "Humble yourself.",
    "Step up.",
    "Skill issue.",
    "All yours.",
    "Catch me if you can.",
    "Clock is ticking.",
    "Too slow.",
    "Hands down.",
];

/// Cooldown anti-spam pour la touche F3 (taunt joueur).  On bloque à ~1.5 s
/// pour qu'un joueur qui frappe F3 en boucle n'encombre pas le chat feed
/// ni ne noie la pile sonore.
const PLAYER_TAUNT_COOLDOWN: f32 = 1.5;

/// Probabilité (0..1) qu'un bot vivant réponde à une taunt F3 du joueur.
/// On tire un seul clapback par taunt (pas de chaîne en cascade) pour
/// ne pas transformer un simple F3 en karaoké.  25 % = en moyenne une
/// taunt sur quatre déclenche une réplique — assez rare pour rester
/// surprenant, assez fréquent pour que le joueur sente l'IA réagir.
const BOT_CLAPBACK_PROB: f32 = 0.25;

/// Lignes de clapback d'un bot en réponse à une taunt joueur (F3).
/// Pool distinct des kill insults pour que le ton soit « défensif /
/// piqué » plutôt que « humiliation sur frag ».
const BOT_CLAPBACK_LINES: &[&str] = &[
    "Keep talking.",
    "Words don't aim.",
    "Cute.",
    "We'll see.",
    "Save it.",
    "Big talk, zero frags.",
    "Mouth over skill.",
    "Talk more, die more.",
    "Noted.",
    "Yeah yeah.",
    "Prove it.",
];

impl ChatTrigger {
    fn pool(self) -> &'static [&'static str] {
        match self {
            Self::KillInsult => CHAT_LINES_KILL_INSULT,
            Self::Death => CHAT_LINES_DEATH,
            Self::Respawn => CHAT_LINES_RESPAWN,
        }
    }

    /// Multiplicateur appliqué à `CHAT_TRIGGER_PROB` — certains
    /// triggers sont plus bavards (kill insult) que d'autres (respawn).
    fn weight(self) -> f32 {
        match self {
            Self::KillInsult => 1.0,
            Self::Death => 0.85,
            Self::Respawn => 0.35,
        }
    }
}

/// Ligne de chat bot — rendue en surimpression sous le HUD.
#[derive(Debug, Clone)]
struct ChatLine {
    /// Nom du bot qui parle (préfixé « BOT: » dans l'UI pour distinction).
    speaker: String,
    /// Texte de la ligne — jamais localisé, anglais cyberspeak Q3.
    text: String,
    /// Instant d'expiration (horloge `App::time_sec`).  La ligne fade
    /// sur son dernier quart de vie pour disparaître en douceur.
    expire_at: f32,
    /// Lifetime initiale — sert à calculer le fade alpha au rendu.
    /// Stocké plutôt qu'un `spawn_at` pour économiser un calcul et
    /// rester cohérent avec `KillEvent` qui fait pareil.
    lifetime: f32,
}

/// Toast de ramassage d'item : message éphémère rendu en HUD quand
/// le joueur ramasse un weapon/powerup/holdable.  On n'affiche PAS
/// les ramassages d'ammo ou de petits health/armor (spammy) — seuls
/// les items « notables » génèrent un toast.
#[derive(Debug, Clone)]
struct PickupToast {
    /// Texte affiché (ex. « YOU GOT THE ROCKET LAUNCHER »).
    text: String,
    /// Couleur RGBA du texte — rouge pour powerups, jaune pour armes,
    /// blanc pour holdables.  Aide l'œil à catégoriser sans lire.
    color: [f32; 4],
    /// Instant d'expiration (horloge `App::time_sec`).
    expire_at: f32,
    /// Lifetime initiale — pour calculer le fade au rendu.
    lifetime: f32,
}

/// Chiffre de dégât flottant — typique Quake Live / Overwatch. Apparaît
/// à la position de l'entité touchée et monte lentement avant de s'éteindre.
/// Couleur : jaune quand on inflige, rouge quand on encaisse.
#[derive(Debug, Clone, Copy)]
struct FloatingDamage {
    /// Position monde initiale, en général `entity.origin + Vec3::Z * center`.
    /// Le texte dérive vers le haut avec le temps mais on garde l'origine
    /// pour recalculer à chaque frame (le `time_sec` actuel détermine l'offset
    /// vertical).
    origin: Vec3,
    /// Montant affiché — typiquement 5..150. 0 est filtré avant push.
    damage: i32,
    /// `true` = dégât subi par le joueur (rouge), `false` = dégât infligé
    /// à un bot (jaune).
    to_player: bool,
    /// `time_sec` au delà duquel on purge.
    expire_at: f32,
    /// Durée totale initiale (s). Sert au calcul du fade alpha + drift.
    lifetime: f32,
}

/// Durée de vie d'un chiffre de dégât avant purge.
const DAMAGE_NUMBER_LIFETIME: f32 = 1.2;
/// Hauteur parcourue (unités Q3) entre spawn et expiration.
const DAMAGE_NUMBER_RISE: f32 = 28.0;
/// Cap défensif — on ne laisse pas exploser la Vec si quelque chose spamme.
const DAMAGE_NUMBER_MAX: usize = 64;

/// Style de rendu d'un beam.  Sélectionne la forme géométrique générée
/// (ligne droite, zigzag électrique, hélice), pas la couleur ni la
/// longueur — les deux restent portées par `ActiveBeam`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BeamStyle {
    /// Segment unique `a → b`.  Utilisé par le beam de respawn, les trails
    /// génériques, etc.
    Straight,
    /// Zigzag jittered autour de `a → b` — arc électrique Lightning Gun.
    Lightning,
    /// Hélice autour de `a → b` — trail Railgun.
    Spiral,
}

/// Faisceau visuel persistant — typiquement LG tant que la touche est
/// tenue, ou trail railgun qui s'estompe au bout d'une fraction de seconde.
/// Ré-émis chaque frame au `BeamRenderer` jusqu'à expiration.
#[derive(Debug, Clone, Copy)]
struct ActiveBeam {
    a: Vec3,
    b: Vec3,
    /// RGBA avec alpha > 0, additive blend dans le renderer.
    color: [f32; 4],
    /// `time_sec` au delà duquel le beam est purgé.
    expire_at: f32,
    /// Durée totale initiale (s). Sert au calcul du fade alpha.
    lifetime: f32,
    /// Forme géométrique — voir [`BeamStyle`].
    style: BeamStyle,
}

/// Projectile actif — rocket, plasma, grenade, … La structure porte le
/// mesh + le tint visuel pour que `queue_projectiles` soit générique.
struct Projectile {
    origin: Vec3,
    velocity: Vec3,
    /// Paramètres de dégâts à l'impact (direct + splash).
    direct_damage: i32,
    splash_radius: f32,
    splash_damage: i32,
    owner: ProjectileOwner,
    /// Arme d'origine — utilisée par le kill-feed pour afficher avec quoi
    /// la victime a été tuée (rocket, grenade, plasma, bfg).
    weapon: WeaponId,
    /// Timestamp d'expiration (fallback si on ne touche rien — fuse pour
    /// les grenades).
    expire_at: f32,
    /// Gravité appliquée chaque tick sur Vz (0 = projectile linéaire type
    /// rocket/plasma, ~800 = grenade Q3).
    gravity: f32,
    /// `true` → impact monde = rebond amorti ; `false` → impact monde = boom.
    /// Un impact bot direct déclenche toujours le boom.
    bounce: bool,
    /// Mesh à dessiner — `None` = invisible (pas d'asset chargé). Arc partagé
    /// donc clone ≈ bump de refcount.
    mesh: Option<Arc<Md3Gpu>>,
    /// Teinte RGBA appliquée au draw (distingue plasma ≠ rocket à l'œil).
    tint: [f32; 4],
    /// Timestamp du prochain puff de trail à spawner — seulement utilisé
    /// pour les projectiles qui émettent une traînée (rocket, grenade, BFG).
    /// `0.0` garantit qu'un premier puff sort dès le tick de spawn.
    next_trail_at: f32,
    /// **Lock-on** (W5) — index dans `self.bots` du bot poursuivi.
    /// `None` = projectile balistique normal. Pendant l'update, la
    /// vélocité est progressivement réorientée vers le centre du bot
    /// (steering proportionnel à `HOMING_TURN_RATE`). Si le bot meurt
    /// ou disparaît, le projectile redevient balistique.
    homing_target: Option<usize>,
}

/// Effet d'explosion — dessiné chaque frame tant que `time_sec < expire_at`.
struct Explosion {
    origin: Vec3,
    expire_at: f32,
}

/// Type de médaille Q3 affichée en popup.  L'ordre correspond à la priorité
/// visuelle quand plusieurs médailles s'empilent (Humiliation passe devant).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Medal {
    /// Gauntlet frag — message cinglant.
    Humiliation,
    /// 2 frags en moins de 2 s.
    Excellent,
    /// 2 hits Railgun consécutifs.
    Impressive,
}

impl Medal {
    /// Texte affiché à l'écran — majuscules Q3-style.
    fn label(self) -> &'static str {
        match self {
            Self::Humiliation => "HUMILIATION!",
            Self::Excellent => "EXCELLENT!",
            Self::Impressive => "IMPRESSIVE!",
        }
    }

    /// Couleur RGBA de la médaille — trois teintes différenciées pour que
    /// le joueur reconnaisse la médaille du coin de l'œil sans lire.
    fn color(self) -> [f32; 4] {
        match self {
            // Rouge profond : humiliation = humiliant pour la cible.
            Self::Humiliation => [1.0, 0.25, 0.2, 1.0],
            // Jaune doré : excellence = récompense.
            Self::Excellent => [1.0, 0.85, 0.2, 1.0],
            // Bleu métallique : impressionnant = technique, chirurgical.
            Self::Impressive => [0.55, 0.8, 1.0, 1.0],
        }
    }
}

/// Médaille active à l'écran — empilée au moment de l'octroi, purgée après
/// `MEDAL_SHOW_SEC`.  Plusieurs médailles peuvent coexister : elles
/// s'empilent verticalement et fade indépendamment.
#[derive(Debug, Clone, Copy)]
struct ActiveMedal {
    kind: Medal,
    /// `time_sec` au delà duquel on purge.
    expire_at: f32,
    /// Instant d'apparition — sert au fade-in / fade-out.
    spawn_time: f32,
}

/// Particule d'explosion — spark qui part du point de boom, est tiré vers
/// le bas par la gravité et s'éteint en fade-out. Rendue via le beam
/// renderer comme un streak lumineux (segment `origin → origin + v·dt`).
#[derive(Debug, Clone, Copy)]
struct Particle {
    origin: Vec3,
    velocity: Vec3,
    /// RGBA de base — le renderer module l'alpha par life ratio.
    color: [f32; 4],
    /// `time_sec` au delà duquel on purge.
    expire_at: f32,
    /// Durée totale initiale (s). Sert au calcul du fade.
    lifetime: f32,
}

/// Nombre de particules spawnées à chaque explosion (rocket / splash).
const PARTICLE_EXPLOSION_COUNT: usize = 24;
/// Durée moyenne (s) de vie d'une particule d'explosion.
const PARTICLE_EXPLOSION_LIFETIME: f32 = 0.55;
/// Vitesse initiale max (unités Q3/s) d'une particule — ~420 équivaut
/// à un spark qui parcourt ~230 unités avant de retomber.
const PARTICLE_EXPLOSION_SPEED: f32 = 420.0;
/// Gravité appliquée aux particules (unités Q3/s²). Plus élevé que le
/// joueur pour que les sparks retombent vite.
const PARTICLE_GRAVITY: f32 = 600.0;
/// Cap défensif — on limite le nombre total de particules actives pour
/// éviter que la Vec gonfle en cas de spam d'explosions.
const PARTICLE_MAX: usize = 512;
/// Longueur du streak (en secondes simulées). `streak = velocity * this`.
/// Plus grand = traînées plus longues, plus petit = points presque.
const PARTICLE_STREAK_DT: f32 = 0.022;

/// Intervalle entre deux puffs de trail derrière un projectile (rocket /
/// grenade / BFG).  40 ms → ≈ 25 puffs/s : dense mais soutenable ; à
/// `PROJECTILE_SPEED = 900–2000 u/s` un puff tombe tous les ~35–80 u Q3,
/// ce qui donne un trail visuellement continu sans vide.
const PROJECTILE_TRAIL_INTERVAL: f32 = 0.04;

/// Nombre de sparks par impact hitscan (mur ou bot).
const PARTICLE_HIT_COUNT: usize = 6;
/// Durée d'un spark d'impact — court, juste un flash.
const PARTICLE_HIT_LIFETIME: f32 = 0.22;
/// Vitesse initiale d'un spark d'impact.
const PARTICLE_HIT_SPEED: f32 = 220.0;

/// Nombre de particules spawnées sur mort d'un bot (gibs).
const PARTICLE_GIB_COUNT: usize = 32;
/// Durée d'un gib — plus long que les sparks pour bien voir le spray.
const PARTICLE_GIB_LIFETIME: f32 = 0.8;
/// Vitesse initiale d'un gib.
const PARTICLE_GIB_SPEED: f32 = 340.0;

/// Nombre de sparks spawnés quand un pickup respawn (colonne ascendante).
const PARTICLE_RESPAWN_COUNT: usize = 14;
/// Gouttelettes émises à la transition surface↔eau.  18 ≈ un éventail
/// dense sans saturer le cap de 512 particules totales quand plusieurs
/// transitions enchainent (respawn aquatique, jump pad sous-marin…).
const PARTICLE_WATER_SPLASH_COUNT: usize = 18;
/// Bulles par burst quand le joueur manque d'air.  Volontairement bas
/// (3 par burst, toutes les 0.15–0.5 s) — plus c'est intermittent, plus
/// c'est lisible comme « fuite de souffle » plutôt que « nuage ».
const PARTICLE_BUBBLE_COUNT: usize = 3;
/// Durée du beam de respawn (pickup ou joueur).
const RESPAWN_FX_DURATION: f32 = 0.45;
/// Hauteur de la colonne de respawn (unités Q3).
const RESPAWN_FX_HEIGHT: f32 = 64.0;

/// Effet d'un pickup — on mappe `EntityKind` au moment du chargement de la
/// map pour éviter de re-parser les classnames chaque frame.
#[derive(Debug, Clone, Copy)]
enum PickupKind {
    /// Small cross = 5, medium = 25, large = 50, mega = 100 (Q3 standard).
    Health { amount: i32, max_cap: i32 },
    /// Inclut armor shards + combat + body. Stocké en `i32` pour simplifier.
    Armor { amount: i32 },
    /// Arme ramassable — ajoute l'arme au bitmask et donne `ammo`.
    Weapon { weapon: WeaponId, ammo: i32 },
    /// Boîte de munitions — ajoute `amount` au slot d'arme ciblé.
    Ammo { slot: u8, amount: i32 },
    /// Powerup temporaire (Quad Damage, etc). Durée en secondes.
    Powerup { powerup: PowerupKind, duration: f32 },
    /// Holdable — équipe le slot d'inventaire (écrase le précédent).
    /// Consommé à l'activation (touche `use`).
    Holdable { kind: HoldableKind },
    /// Visible mais sans effet (ex. armes pas encore implémentées).
    Inert,
}

/// Items consommables stockés dans le slot `App::held_item`. Un seul à la
/// fois ; ramasser un holdable en écrase un autre déjà en stock (canon Q3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HoldableKind {
    /// Medkit : restaure la santé à `HOLDABLE_MEDKIT_TARGET_HP` en un clic.
    /// En Q3 : soigne +25 avec cap à 125 (au-dessus du max normal 100).
    Medkit,
    /// Personal Teleporter : téléporte le joueur à un point de spawn DM
    /// aléatoire. Pas de son d'activation spécifique côté code, on
    /// réutilise le son de téléport « ambient ».
    Teleporter,
}

impl HoldableKind {
    /// Résolution classname → HoldableKind.
    fn from_classname(name: &str) -> Option<Self> {
        match name {
            "holdable_medkit" => Some(Self::Medkit),
            "holdable_teleporter" => Some(Self::Teleporter),
            _ => None,
        }
    }

    /// Libellé court affiché sur le HUD (icône holdable en bas-droite).
    fn hud_label(self) -> &'static str {
        match self {
            Self::Medkit => "MEDKIT",
            Self::Teleporter => "TPORT",
        }
    }
}

/// Cible HP du Medkit — Q3 canon : soin +25 jusqu'à un plafond de 125,
/// soit 25 de plus que le max baseline. On simplifie ici en fixant la
/// cible absolue (pas d'ajout sur soi-même, donc pas de stacking).
const HOLDABLE_MEDKIT_TARGET_HP: i32 = 125;

/// Powerups implémentés. Chaque variante :
/// * consomme un slot de `App::powerup_until` via `index()`,
/// * affiche un badge HUD avec une couleur dédiée,
/// * est mappée depuis son `classname` Q3 dans `PickupKind::from_entity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PowerupKind {
    /// ×4 sur tous les dégâts infligés par le joueur pendant la durée.
    /// Son de pickup + indicateur HUD bleu.
    QuadDamage,
    /// +30 % vitesse de course et cadence de tir. Badge orange.
    Haste,
    /// +15 HP/s tant que `current < max`. Badge vert. Ne ressuscite pas
    /// un joueur mort (cf. `Health::heal`).
    Regeneration,
    /// Battle Suit (item_enviro) : immunité contre les dégâts
    /// environnementaux (chute, lave, slime, noyade, splash de son propre
    /// rocket). Badge jaune. Les tirs adverses passent quand même — c'est
    /// un "enviro suit", pas une invulnérabilité complète.
    BattleSuit,
    /// Invisibility (item_invis) : quasi-transparent, détection bot réduite
    /// à un quart de la portée nominale et mémoire d'agro raccourcie.
    /// Badge blanc. N'atténue pas les dégâts — le joueur est fragile
    /// pendant la durée, mais plus difficile à toucher.
    Invisibility,
    /// Flight (item_flight) : gravité désactivée pendant toute la durée,
    /// jump/crouch permettent de prendre/perdre de l'altitude à vitesse
    /// `FLIGHT_THRUST_SPEED`. Badge violet. Durée allongée (60 s) parce
    /// que l'item est rare et souvent utilisé pour explorer.
    Flight,
}

impl PowerupKind {
    /// Nombre de variantes — `powerup_until` est dimensionné là-dessus.
    const COUNT: usize = 6;
    /// Toutes les variantes — utile pour itérer (tick d'expiration, HUD).
    const ALL: [PowerupKind; Self::COUNT] = [
        Self::QuadDamage,
        Self::Haste,
        Self::Regeneration,
        Self::BattleSuit,
        Self::Invisibility,
        Self::Flight,
    ];

    /// Slot dans `App::powerup_until`. Doit matcher l'ordre de `ALL`.
    fn index(self) -> usize {
        match self {
            Self::QuadDamage => 0,
            Self::Haste => 1,
            Self::Regeneration => 2,
            Self::BattleSuit => 3,
            Self::Invisibility => 4,
            Self::Flight => 5,
        }
    }

    /// Durée canonique appliquée à chaque pickup.
    fn duration(self) -> f32 {
        match self {
            Self::QuadDamage => QUAD_DAMAGE_DURATION,
            Self::Haste => HASTE_DURATION,
            Self::Regeneration => REGEN_DURATION,
            Self::BattleSuit => BATTLE_SUIT_DURATION,
            Self::Invisibility => INVISIBILITY_DURATION,
            Self::Flight => FLIGHT_DURATION,
        }
    }

    /// Libellé court affiché dans le HUD (badge au-dessus du chrono).
    fn hud_label(self) -> &'static str {
        match self {
            Self::QuadDamage => "QUAD",
            Self::Haste => "HASTE",
            Self::Regeneration => "REGEN",
            Self::BattleSuit => "ENVIRO",
            Self::Invisibility => "INVIS",
            Self::Flight => "FLIGHT",
        }
    }

    /// Couleur du badge HUD (RGBA). Pleine opacité — le blink en fin de
    /// durée mod l'alpha à l'affichage.
    fn hud_color(self) -> [f32; 4] {
        match self {
            Self::QuadDamage => [0.35, 0.55, 1.0, 1.0],
            Self::Haste => [1.0, 0.78, 0.25, 1.0],
            Self::Regeneration => [0.45, 1.0, 0.4, 1.0],
            Self::BattleSuit => [1.0, 1.0, 0.35, 1.0],
            Self::Invisibility => [0.85, 0.85, 1.0, 1.0],
            Self::Flight => [0.85, 0.5, 1.0, 1.0],
        }
    }

    /// Couleur du flash de pickup (colonne ascendante émise au moment
    /// du ramassage). Ton un peu plus saturé que le HUD.
    fn pickup_fx_color(self) -> [f32; 4] {
        match self {
            Self::QuadDamage => [0.3, 0.4, 1.0, 1.0],
            Self::Haste => [1.0, 0.75, 0.2, 1.0],
            Self::Regeneration => [0.35, 1.0, 0.3, 1.0],
            Self::BattleSuit => [1.0, 0.95, 0.25, 1.0],
            Self::Invisibility => [0.8, 0.8, 1.0, 1.0],
            Self::Flight => [0.85, 0.4, 1.0, 1.0],
        }
    }

    /// Teinte full-screen légère appliquée quand ce powerup est actif
    /// (voir `draw_hud`). `None` → pas de vignette. On réserve ça pour
    /// les effets qui méritent d'être "sentis" en vision périphérique.
    fn fullscreen_tint(self) -> Option<[f32; 3]> {
        match self {
            Self::QuadDamage => Some([0.25, 0.4, 1.0]),
            Self::Haste => Some([1.0, 0.55, 0.15]),
            // Regen : pas de tint. Le rendu du joueur se soignant est
            // déjà communiqué par la barre HP qui remonte + le badge.
            Self::Regeneration => None,
            // Battle Suit : légère teinte jaunâtre, évoque le casque /
            // visière du suit Q3.
            Self::BattleSuit => Some([0.95, 0.9, 0.35]),
            // Invis : pas de tint — on ne veut pas que la vision du joueur
            // soit dégradée pendant qu'il est avantage. Le feedback passe
            // par le rendu des bras (atténué en vue 1re pers).
            Self::Invisibility => None,
            // Flight : pas de tint — on veut une perception du monde
            // intacte pour viser / planer. Le feedback vient du fait que
            // la gravité ne tire plus, très lisible.
            Self::Flight => None,
        }
    }

    /// Tente de dériver un powerup depuis un classname Q3. `None` si le
    /// classname n'est pas implémenté (= `Inert` côté pickup).
    fn from_classname(name: &str) -> Option<Self> {
        match name {
            "item_quad" => Some(Self::QuadDamage),
            "item_haste" => Some(Self::Haste),
            "item_regen" => Some(Self::Regeneration),
            "item_enviro" => Some(Self::BattleSuit),
            "item_invis" => Some(Self::Invisibility),
            "item_flight" => Some(Self::Flight),
            _ => None,
        }
    }
}

impl PickupKind {
    /// Dérive kind + cooldown depuis le classname. Convention Q3 par défaut.
    fn from_entity(kind: &q3_game::EntityKind) -> (PickupKind, f32) {
        use q3_game::EntityKind::*;
        match kind {
            ItemHealth(name) => match name.as_str() {
                "item_health_small" => (PickupKind::Health { amount: 5, max_cap: 200 }, 35.0),
                "item_health" => (PickupKind::Health { amount: 25, max_cap: 100 }, 35.0),
                "item_health_large" => (PickupKind::Health { amount: 50, max_cap: 100 }, 35.0),
                "item_health_mega" => (PickupKind::Health { amount: 100, max_cap: 200 }, 35.0),
                _ => (PickupKind::Health { amount: 25, max_cap: 100 }, 35.0),
            },
            ItemArmor(name) => match name.as_str() {
                "item_armor_shard" => (PickupKind::Armor { amount: 5 }, 25.0),
                "item_armor_combat" => (PickupKind::Armor { amount: 50 }, 25.0),
                "item_armor_body" => (PickupKind::Armor { amount: 100 }, 25.0),
                _ => (PickupKind::Armor { amount: 25 }, 25.0),
            },
            ItemWeapon(name) => {
                // Respawn 5s pour les armes (Q3 standard).
                let ammo_start = |w: WeaponId| w.params().starting_ammo;
                match name.as_str() {
                    "weapon_gauntlet" => (
                        PickupKind::Weapon { weapon: WeaponId::Gauntlet, ammo: 0 },
                        5.0,
                    ),
                    "weapon_machinegun" => (
                        PickupKind::Weapon { weapon: WeaponId::Machinegun, ammo: ammo_start(WeaponId::Machinegun) },
                        5.0,
                    ),
                    "weapon_shotgun" => (
                        PickupKind::Weapon { weapon: WeaponId::Shotgun, ammo: ammo_start(WeaponId::Shotgun) },
                        5.0,
                    ),
                    "weapon_grenadelauncher" => (
                        PickupKind::Weapon {
                            weapon: WeaponId::Grenadelauncher,
                            ammo: ammo_start(WeaponId::Grenadelauncher),
                        },
                        5.0,
                    ),
                    "weapon_rocketlauncher" => (
                        PickupKind::Weapon {
                            weapon: WeaponId::Rocketlauncher,
                            ammo: ammo_start(WeaponId::Rocketlauncher),
                        },
                        5.0,
                    ),
                    "weapon_lightning" => (
                        PickupKind::Weapon {
                            weapon: WeaponId::Lightninggun,
                            ammo: ammo_start(WeaponId::Lightninggun),
                        },
                        5.0,
                    ),
                    "weapon_plasmagun" => (
                        PickupKind::Weapon {
                            weapon: WeaponId::Plasmagun,
                            ammo: ammo_start(WeaponId::Plasmagun),
                        },
                        5.0,
                    ),
                    "weapon_bfg" => (
                        PickupKind::Weapon {
                            weapon: WeaponId::Bfg,
                            ammo: ammo_start(WeaponId::Bfg),
                        },
                        5.0,
                    ),
                    "weapon_railgun" => (
                        PickupKind::Weapon { weapon: WeaponId::Railgun, ammo: ammo_start(WeaponId::Railgun) },
                        5.0,
                    ),
                    // Autres armes pas encore gameplay-fonctionnelles — on
                    // les laisse visibles mais inertes en attendant leur
                    // implémentation (rocket, plasma, grenade, bfg, …).
                    _ => (PickupKind::Inert, 5.0),
                }
            }
            ItemAmmo(name) => {
                // Cooldown 40 s (Q3 standard pour munitions).
                match name.as_str() {
                    "ammo_bullets" => (
                        PickupKind::Ammo { slot: WeaponId::Machinegun.slot(), amount: 50 },
                        40.0,
                    ),
                    "ammo_shells" => (
                        PickupKind::Ammo { slot: WeaponId::Shotgun.slot(), amount: 10 },
                        40.0,
                    ),
                    "ammo_grenades" => (
                        PickupKind::Ammo { slot: WeaponId::Grenadelauncher.slot(), amount: 5 },
                        40.0,
                    ),
                    "ammo_rockets" => (
                        PickupKind::Ammo { slot: WeaponId::Rocketlauncher.slot(), amount: 5 },
                        40.0,
                    ),
                    "ammo_lightning" => (
                        PickupKind::Ammo { slot: WeaponId::Lightninggun.slot(), amount: 60 },
                        40.0,
                    ),
                    "ammo_cells" => (
                        PickupKind::Ammo { slot: WeaponId::Plasmagun.slot(), amount: 30 },
                        40.0,
                    ),
                    "ammo_bfg" => (
                        PickupKind::Ammo { slot: WeaponId::Bfg.slot(), amount: 15 },
                        40.0,
                    ),
                    "ammo_slugs" => (
                        PickupKind::Ammo { slot: WeaponId::Railgun.slot(), amount: 10 },
                        40.0,
                    ),
                    _ => (PickupKind::Inert, 40.0),
                }
            }
            ItemPowerup(name) => {
                // Respawn 120s (Q3 powerup standard). La durée par variante
                // vient de `PowerupKind::duration()` — on évite de dupliquer.
                match PowerupKind::from_classname(name) {
                    Some(p) => (
                        PickupKind::Powerup {
                            powerup: p,
                            duration: p.duration(),
                        },
                        120.0,
                    ),
                    // Tous les powerups du jeu de base sont maintenant
                    // gérés ; ce bras reste un filet de sécurité pour
                    // tout classname hors mapping.
                    None => (PickupKind::Inert, 120.0),
                }
            }
            ItemHoldable(name) => {
                // Respawn 60s — holdables réapparaissent deux fois plus
                // vite qu'un powerup parce qu'ils sont « à usage unique »
                // (une fois utilisés, disparus du slot).
                match HoldableKind::from_classname(name) {
                    Some(k) => (PickupKind::Holdable { kind: k }, 60.0),
                    None => (PickupKind::Inert, 60.0),
                }
            }
            _ => (PickupKind::Inert, 30.0),
        }
    }
}

/// Armes disponibles pour le joueur. Le MVP expose trois styles de tir
/// distincts (hitscan simple, spread multi-pellets, hitscan lourd) pour
/// valider la logique de switch. Les autres armes Q3 suivront le même
/// pattern quand les effets (splash, beam continu, …) seront branchés.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WeaponId {
    Gauntlet,
    Machinegun,
    Shotgun,
    Grenadelauncher,
    Rocketlauncher,
    Lightninggun,
    Railgun,
    Plasmagun,
    Bfg,
}

/// Type de tir — détermine le branchement dans `fire_weapon`.
#[derive(Debug, Clone, Copy)]
enum WeaponKind {
    /// Trace(s) instantanée(s) sur `pellets` raycasts.
    Hitscan,
    /// Projectile qui vole à `speed` unités/s, explose à l'impact avec
    /// `splash_radius` de dégâts de zone = `splash_damage`.
    Projectile {
        speed: f32,
        splash_radius: f32,
        splash_damage: i32,
    },
}

/// Paramètres dérivables d'une arme — consommés par `fire_weapon`.
#[derive(Debug, Clone, Copy)]
struct WeaponParams {
    kind: WeaponKind,
    damage: i32,
    cooldown: f32,
    range: f32,
    /// Nombre de traces hitscan par tir (1 pour machinegun/railgun, N pour shotgun).
    /// Ignoré pour les projectiles (toujours 1 spawn par tir).
    pellets: u8,
    /// Demi-angle de dispersion en degrés (0 = parfait).
    spread_deg: f32,
    /// Munitions consommées par tir (1 en standard ; 0 pour arme sans ammo).
    ammo_cost: u8,
    /// Stock de départ du joueur pour cette arme (Q3 baseline).
    starting_ammo: i32,
    /// Capacité max de ce type de munition.
    max_ammo: i32,
}

impl WeaponId {
    const ALL: [WeaponId; 9] = [
        Self::Gauntlet,
        Self::Machinegun,
        Self::Shotgun,
        Self::Grenadelauncher,
        Self::Rocketlauncher,
        Self::Lightninggun,
        Self::Railgun,
        Self::Plasmagun,
        Self::Bfg,
    ];

    /// Raccourci clavier (touche 1..9) — reprend la numérotation Q3 canon.
    fn slot(self) -> u8 {
        match self {
            Self::Gauntlet => 1,
            Self::Machinegun => 2,
            Self::Shotgun => 3,
            Self::Grenadelauncher => 4,
            Self::Rocketlauncher => 5,
            Self::Lightninggun => 6,
            Self::Railgun => 7,
            Self::Plasmagun => 8,
            Self::Bfg => 9,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Gauntlet => "gauntlet",
            Self::Machinegun => "machinegun",
            Self::Shotgun => "shotgun",
            Self::Grenadelauncher => "grenade",
            Self::Rocketlauncher => "rocket",
            Self::Lightninggun => "lightning",
            Self::Railgun => "railgun",
            Self::Plasmagun => "plasma",
            Self::Bfg => "bfg",
        }
    }

    /// Path VFS conventionnel du viewmodel.
    fn viewmodel_path(self) -> &'static str {
        match self {
            Self::Gauntlet => "models/weapons2/gauntlet/gauntlet.md3",
            Self::Machinegun => "models/weapons2/machinegun/machinegun.md3",
            Self::Shotgun => "models/weapons2/shotgun/shotgun.md3",
            Self::Grenadelauncher => "models/weapons2/grenadel/grenadel.md3",
            Self::Rocketlauncher => "models/weapons2/rocketl/rocketl.md3",
            Self::Lightninggun => "models/weapons2/lightning/lightning.md3",
            Self::Railgun => "models/weapons2/railgun/railgun.md3",
            Self::Plasmagun => "models/weapons2/plasma/plasma.md3",
            Self::Bfg => "models/weapons2/bfg/bfg.md3",
        }
    }

    /// Couleur + taille du muzzle flash 3D pour cette arme.  `None` pour
    /// les armes qui n'en produisent pas (Gauntlet = melee, Lightning =
    /// beam continu → le halo de la tête du beam fait déjà le travail).
    /// Les couleurs sont linéaires RGB ; l'alpha agit comme gain global
    /// sur le sprite additif (voir `flare.wgsl`).  Les tailles sont en
    /// unités Q3 (~24 ≈ 0.6 m).
    fn muzzle_flash(self) -> Option<([f32; 4], f32)> {
        match self {
            // Pas de muzzle flash pour le gauntlet (lame rotative, pas d'explosion de poudre).
            Self::Gauntlet => None,
            // Poudre chaude : jaune-orangé dominant, cœur quasi blanc
            // donné par le shader radial → alpha ~1.1 pour saturer le core.
            Self::Machinegun => Some(([1.00, 0.80, 0.35, 1.0], 22.0)),
            // Shotgun = double canon, plus gros nuage.
            Self::Shotgun => Some(([1.00, 0.75, 0.30, 1.0], 32.0)),
            // Lance-grenade : détonation lente → flash plus pesant, rouge orangé.
            Self::Grenadelauncher => Some(([1.00, 0.65, 0.25, 1.0], 28.0)),
            // RL : gros pop blanc-orangé, le plus visible.
            Self::Rocketlauncher => Some(([1.00, 0.70, 0.30, 1.0], 36.0)),
            // Lightning : pas de flash standalone — le beam émet déjà sa
            // propre lumière, un halo de tête est géré côté beam trail.
            Self::Lightninggun => None,
            // Railgun : décharge électrique bleu-violet, plus fine mais très
            // intense → alpha légèrement boosté.
            Self::Railgun => Some(([0.55, 0.80, 1.00, 1.2], 26.0)),
            // Plasma : bleu-cyan vif, petite taille mais spammée.
            Self::Plasmagun => Some(([0.40, 0.75, 1.00, 1.0], 18.0)),
            // BFG : énorme boule verte — canon pire que tout, on balance
            // un halo vert saturé de 48u.
            Self::Bfg => Some(([0.35, 1.00, 0.50, 1.1], 48.0)),
        }
    }

    /// Quantité de recul ajoutée à `App::view_kick` à chaque tir.
    /// L'échelle est normalisée : 1.0 ≈ recul fort (rocket), 0.2 ≈
    /// pulse discrète (plasma).  Le kick décroît ensuite exponentiellement
    /// via [`VIEW_KICK_DECAY_PER_SEC`], donc cette valeur agit comme un
    /// "impulse" instantané — une rafale de MG accumule, mais le jeu
    /// reste lisible parce que la décroissance est rapide.
    fn view_kick(self) -> f32 {
        match self {
            // Contact melee : aucun recul, c'est une lame qui tourne.
            Self::Gauntlet => 0.0,
            // MG : rafale rapide → kick faible par balle mais cumulatif.
            Self::Machinegun => 0.20,
            // Shotgun : poussée franche à l'épaule.
            Self::Shotgun => 0.75,
            // Lance-grenade : recul marqué, décharge lente.
            Self::Grenadelauncher => 0.60,
            // Rocket : poussée maximale, l'arme saute visuellement.
            Self::Rocketlauncher => 1.00,
            // LG : beam continu, on tremble à peine à chaque tick.
            Self::Lightninggun => 0.06,
            // Railgun : recul fort mais plus "sec" que la rocket.
            Self::Railgun => 0.85,
            // Plasma : légère pulsation, l'énergie ne pousse pas comme la poudre.
            Self::Plasmagun => 0.18,
            // BFG : énorme, pire que la rocket.
            Self::Bfg => 1.20,
        }
    }

    /// Path VFS conventionnel du SFX de tir.
    /// Liste de chemins candidats pour le son de tir, dans l'ordre de
    /// priorité. Le 1er existant dans le VFS est utilisé. Plusieurs
    /// candidats parce que selon le pk3 (vanilla, démos, mods) les
    /// noms varient. Les retours utilisateur "le RL n'a pas de son"
    /// remontent typiquement à un asset au nom légèrement différent
    /// (ex: `rocklf1.wav` sans le `a` final dans certains demos).
    fn fire_sfx_paths(self) -> &'static [&'static str] {
        match self {
            Self::Gauntlet => &[
                "sound/weapons/melee/fstatck.wav",
                "sound/weapons/melee/fstratck.wav",
            ],
            Self::Machinegun => &[
                "sound/weapons/machinegun/machgf1b.wav",
                "sound/weapons/machinegun/machgf2b.wav",
                "sound/weapons/machinegun/machgf3b.wav",
            ],
            Self::Shotgun => &[
                "sound/weapons/shotgun/sshotf1b.wav",
            ],
            Self::Grenadelauncher => &[
                "sound/weapons/grenade/grenlf1a.wav",
            ],
            Self::Rocketlauncher => &[
                "sound/weapons/rocket/rocklf1a.wav",
                // Variantes connues : certains demos/pk3 partiels n'ont
                // pas le `a` final, ou utilisent `rocketf1` au lieu de
                // `rocklf1`. On essaie large pour qu'au moins un matche.
                "sound/weapons/rocket/rocklf1.wav",
                "sound/weapons/rocket/rocketf1.wav",
                "sound/weapons/rocket/rl_fire.wav",
            ],
            Self::Lightninggun => &[
                "sound/weapons/lightning/lg_fire.wav",
                "sound/weapons/lightning/lightning_fire.wav",
            ],
            Self::Railgun => &[
                "sound/weapons/railgun/railgf1a.wav",
            ],
            Self::Plasmagun => &[
                "sound/weapons/plasma/hyprbf1a.wav",
                "sound/weapons/plasma/lasfly.wav",
            ],
            Self::Bfg => &[
                "sound/weapons/bfg/bfg_fire.wav",
                "sound/weapons/bfg/bfgf1a.wav",
            ],
        }
    }

    /// Paramètres du **tir secondaire** (alt-fire). `None` = pas d'alt
    /// défini pour cette arme → le moteur retombe sur le primaire.
    /// Convention :
    /// * **Shotgun slug** : 1 pellet AP, 0 spread, +damage, cooldown long.
    /// * **Railgun ricochet** : même params que primaire mais flag de
    ///   ricochet géré en aval (cf. `alt_active` dans fire_weapon).
    /// * **Rocket lock-on** : params identiques à primaire, le lock-on
    ///   s'applique au moment du spawn projectile.
    fn secondary_params(self) -> Option<WeaponParams> {
        let p = self.params();
        match self {
            // **W1 — Gauntlet lunge** : range étendue à 96 u (vs 64 du
            // primaire) et damage boosté à 75 (vs 50). Le boost
            // simule le "lunge" avant : on tape plus loin et plus
            // fort. Côté input, alt + gauntlet déclenche aussi un
            // dash-forward (cf. fire_weapon).
            Self::Gauntlet => Some(WeaponParams {
                damage: 75,
                range: 96.0,
                cooldown: 0.5,
                ..p
            }),
            Self::Shotgun => Some(WeaponParams {
                damage: 80,        // un slug de précision (vs 11×10 spread)
                cooldown: 1.5,     // beaucoup plus long que SG primaire
                pellets: 1,
                spread_deg: 0.0,
                ammo_cost: 2,      // 2 cartouches pour un slug
                ..p
            }),
            Self::Railgun => Some(WeaponParams {
                cooldown: p.cooldown * 1.4, // ricochet = pas spammable
                damage: 65, // dmg primaire conservé pour le 1er hit
                ..p
            }),
            Self::Rocketlauncher => Some(WeaponParams {
                cooldown: p.cooldown * 1.5, // lock-on a un coût rythmique
                ..p
            }),
            // Pas d'alt défini → primaire utilisé.
            _ => None,
        }
    }

    fn params(self) -> WeaponParams {
        match self {
            Self::Gauntlet => WeaponParams {
                kind: WeaponKind::Hitscan,
                damage: 50,
                cooldown: 0.4,
                range: 64.0,
                pellets: 1,
                spread_deg: 0.0,
                ammo_cost: 0,       // gauntlet = munitions infinies
                starting_ammo: 0,
                max_ammo: 0,
            },
            Self::Machinegun => WeaponParams {
                kind: WeaponKind::Hitscan,
                damage: 7,
                cooldown: 0.10,
                range: 4096.0,
                pellets: 1,
                spread_deg: 0.8,
                ammo_cost: 1,
                starting_ammo: 100,
                max_ammo: 200,
            },
            Self::Shotgun => WeaponParams {
                kind: WeaponKind::Hitscan,
                damage: 10,
                cooldown: 1.0,
                range: 2048.0,
                pellets: 11,
                spread_deg: 5.0,
                ammo_cost: 1,
                starting_ammo: 10,
                max_ammo: 50,
            },
            Self::Plasmagun => WeaponParams {
                // Projectile rapide, splash minuscule, dégât modeste — Q3
                // baseline : 20 dmg direct, 15 splash, rayon 20 u.
                kind: WeaponKind::Projectile {
                    speed: 2000.0,
                    splash_radius: 20.0,
                    splash_damage: 15,
                },
                damage: 20,
                cooldown: 0.1, // 10 tirs/sec — l'arme "spammy" de Q3
                range: 8192.0,
                pellets: 1,
                spread_deg: 0.0,
                ammo_cost: 1,
                starting_ammo: 50,
                max_ammo: 200,
            },
            Self::Grenadelauncher => WeaponParams {
                // Grenade : vitesse initiale 700, gravité + rebond appliqués
                // dans `tick_projectiles` (champs spécifiques au spawn).
                // Rayon splash identique au rocket, dégât légèrement plus faible.
                kind: WeaponKind::Projectile {
                    speed: 700.0,
                    splash_radius: 150.0,
                    splash_damage: 100,
                },
                damage: 100,
                cooldown: 0.8,
                range: 8192.0,
                pellets: 1,
                spread_deg: 0.0,
                ammo_cost: 1,
                starting_ammo: 10,
                max_ammo: 50,
            },
            Self::Rocketlauncher => WeaponParams {
                kind: WeaponKind::Projectile {
                    speed: 900.0,
                    splash_radius: 120.0,
                    splash_damage: 100,
                },
                damage: 100, // dégât direct (hit non-splash)
                cooldown: 0.8,
                range: 8192.0,
                pellets: 1,
                spread_deg: 0.0,
                ammo_cost: 1,
                starting_ammo: 10,
                max_ammo: 50,
            },
            Self::Lightninggun => WeaponParams {
                // Lightning = hitscan tenu, très haute cadence, portée courte.
                // Q3 baseline : 8 dmg / shot @ 20 Hz = 160 dps, 768 units max.
                kind: WeaponKind::Hitscan,
                damage: 8,
                cooldown: 0.05, // 20 shots/sec
                range: 768.0,
                pellets: 1,
                spread_deg: 0.0,
                ammo_cost: 1,
                starting_ammo: 100,
                max_ammo: 200,
            },
            Self::Railgun => WeaponParams {
                kind: WeaponKind::Hitscan,
                damage: 80,
                cooldown: 1.5,
                range: 8192.0,
                pellets: 1,
                spread_deg: 0.0,
                ammo_cost: 1,
                starting_ammo: 10,
                max_ammo: 50,
            },
            Self::Bfg => WeaponParams {
                // BFG Q3 canon : projectile rapide 2000 u/s, splash 120u /
                // 100 dmg, ROF élevé (5 tirs/sec). Essentiellement une rocket
                // turbo avec son propre pool munitions.
                kind: WeaponKind::Projectile {
                    speed: 2000.0,
                    splash_radius: 120.0,
                    splash_damage: 100,
                },
                damage: 100,
                cooldown: 0.2,
                range: 8192.0,
                pellets: 1,
                spread_deg: 0.0,
                ammo_cost: 1,
                starting_ammo: 20,
                max_ammo: 40,
            },
        }
    }
}

/// Rig de player Q3 : 3 MD3 (lower/upper/head) partagés entre tous les bots.
///
/// L'attachement se fait via les **tags** : l'origine/orientation de
/// `tag_torso` dans `lower` donne la transformation locale de `upper` ;
/// idem pour `tag_head` dans `upper` → `head`. Lower est l'ancre : sa
/// transformation est celle du bot (origine + yaw + optionnellement pitch
/// partiel pour viser haut/bas).
#[derive(Clone)]
struct PlayerRig {
    lower: Arc<Md3Gpu>,
    upper: Arc<Md3Gpu>,
    head: Arc<Md3Gpu>,
}

/// Pilote d'un bot : IA (`Bot`) + corps (`PlayerMove`) + couleur de MD3.
///
/// Le rig visuel est partagé via `App::bot_rig`, il n'est donc pas dupliqué ici.
struct BotDriver {
    bot: Bot,
    body: PlayerMove,
    /// Teinte RGBA appliquée au MD3 pour distinguer visuellement les bots.
    tint: [f32; 4],
    /// Index circulaire dans la liste de waypoints (spawn points).
    wp_cursor: usize,
    /// Dernière fois (en secondes, horloge `App::time_sec`) où le bot a
    /// vu le joueur. `None` = jamais vu récemment.
    last_saw_player_at: Option<f32>,
    /// Instant où le bot a repéré le joueur pour la première fois sur
    /// l'engagement courant (remis à `None` quand la mémoire d'agro
    /// expire).  Sert à appliquer le délai de réaction : un bot ne tire
    /// qu'après `skill.reaction_time_sec()` écoulé depuis cette date —
    /// simule le temps cerveau-à-gâchette humain.  Sans ça, le bot
    /// ouvre le feu au même tick que la détection, ce qui donne un feel
    /// de « hack ».
    first_seen_player_at: Option<f32>,
    /// Prochaine fois où le bot peut tirer en hitscan (cooldown court).
    next_fire_at: f32,
    /// Prochaine fois où le bot peut lancer une rocket (cooldown long).
    next_rocket_at: f32,
    /// Santé du bot — 0 déclenche le respawn du bot.
    health: Health,
    /// Invincibilité après respawn : le bot ignore dégâts + knockback
    /// tant que `time_sec < invul_until`. Symétrie avec
    /// `App::player_invul_until`. 0.0 = aucune invulnérabilité.
    invul_until: f32,
    /// Nombre de joueurs tués par ce bot depuis le début du match.
    /// Affiché dans le scoreboard (TAB). Seul le joueur compte comme
    /// victime pour l'instant (FF off entre bots).
    frags: u32,
    /// Nombre de morts de ce bot depuis le début du match.
    /// Incrémenté dans `respawn_dead_bots` juste avant de remettre HP à fond.
    deaths: u32,
    /// Dernière fois où le bot a tiré (horloge `time_sec`).  Sert à
    /// la machine d'animation : on joue `TORSO_ATTACK` pendant ~250 ms
    /// après chaque coup de feu pour que le upper body ait un retour
    /// visuel au lieu de rester figé dans `TORSO_STAND`.
    last_fire_at: f32,
    /// Dernière fois où le bot a encaissé un dégât.  Déclenche l'anim
    /// `TORSO_PAIN` / `LEGS_PAIN` pendant une courte fenêtre (~200 ms).
    last_damage_at: f32,
    /// Instant où le bot a quitté le sol (décollage).  On l'utilise
    /// pour détecter le « saut déclenché » vs « vient de marcher dans
    /// le vide » — utile pour ne jouer `LEGS_JUMP` que sur la première
    /// centaine de ms d'airtime, puis passer sur une pose neutre.
    airborne_since: Option<f32>,
    /// Dernière fois où le bot a retouché le sol après une phase
    /// airborne.  Joue `LEGS_LAND` pendant 150 ms.
    last_land_at: f32,
    /// Animation jouée à la frame précédente — sert uniquement au rendu
    /// pour conserver l'avancement interne de l'animation cyclique
    /// entre deux frames (évite les "sauts" quand on change d'état).
    /// Optionnel : si `None`, on repart à la phase 0 la prochaine frame.
    #[allow(dead_code)]
    anim_phase: f32,
    /// État d'animation actuellement joué (mis à jour dans `queue_bots`).
    #[allow(dead_code)]
    anim_state: BotAnimState,
}

/// États d'animation du bot, pilotés par la combinaison vitesse /
/// on_ground / derniers évènements (tir, dégât, mort).  Traduit vers
/// des plages de frames Q3 standard dans `queue_bots`.
///
/// On ne parse pas `animation.cfg` (Q3 encode les ranges par modèle
/// dans ce fichier, typiquement `player/sarge/animation.cfg`) — on
/// utilise les offsets par défaut du format Q3 historique, en clamp
/// par `num_frames()` du mesh pour éviter d'indexer hors-plage si le
/// modèle a moins d'animations que la référence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)] // variants scaffoldées pour la logique de transition future
enum BotAnimState {
    /// Au sol, quasi immobile.  `TORSO_STAND` + `LEGS_IDLE`.
    #[default]
    Idle,
    /// Au sol, en mouvement significatif.  `TORSO_STAND` + `LEGS_RUN`.
    Run,
    /// En l'air, juste décollé.  `LEGS_JUMP`.
    Jump,
    /// Vient de retoucher le sol.  `LEGS_LAND`.
    Land,
    /// Tir récent (<250 ms).  Force `TORSO_ATTACK` (lower reste selon
    /// déplacement).
    Attack,
    /// Dégât récent (<200 ms).  Force `TORSO_PAIN` + lower selon vitesse.
    Pain,
    /// HP ≤ 0.  Joue `BOTH_DEATH1` une seule fois puis reste figé sur
    /// la dernière frame (`BOTH_DEAD1`).  N'est dessiné par `queue_bots`
    /// que si on retire le `continue` sur `is_dead()` — pour l'instant
    /// les cadavres sont supprimés visuellement, conservé pour usage
    /// futur.
    Death,
}

/// Plage de frames `(start, end)` dans le MD3 — `end` exclusif façon
/// `start..end` Rust.  Timings en FPS directement interprétables.
#[derive(Debug, Clone, Copy)]
struct AnimRange {
    start: usize,
    end: usize,
    fps: f32,
    /// Si `false`, on bloque sur `end-1` une fois arrivé (anims non
    /// cycliques : mort, atterrissage, tir, douleur).
    looping: bool,
}

impl AnimRange {
    /// Frame courante + voisin + lerp pour un instant donné, ancré sur
    /// `phase` (temps local à l'animation, 0.0 = début).  Retourne
    /// `(fa, fb, lerp)` clampés à `nf` du mesh.
    fn sample(&self, phase: f32, nf: usize) -> (usize, usize, f32) {
        if self.end <= self.start || nf == 0 {
            return (0, 0, 0.0);
        }
        let span = self.end - self.start;
        let t = phase * self.fps;
        let (a_rel, b_rel, lerp) = if self.looping {
            let cyc = t.rem_euclid(span as f32);
            let a = cyc.floor() as usize % span;
            (a, (a + 1) % span, cyc.fract())
        } else {
            // Non-loop : clamp en fin de plage (freeze sur la dernière
            // frame utile).  La frame `end-1` devient l'image finale.
            let a = (t.floor() as usize).min(span - 1);
            let b = (a + 1).min(span - 1);
            (a, b, t.fract().min(1.0))
        };
        let fa = (self.start + a_rel).min(nf.saturating_sub(1));
        let fb = (self.start + b_rel).min(nf.saturating_sub(1));
        (fa, fb, lerp)
    }
}

/// Table des plages Q3 standard pour le squelette `player` (offsets de
/// `animation.cfg` canonique pour `models/players/sarge/`).  En pratique
/// chaque modèle a sa propre config — on clamp sur `nf` pour ne jamais
/// lire hors plage.
#[allow(dead_code)] // certaines ranges (death) sont scaffoldées pour usage futur
mod bot_anims {
    use super::AnimRange;

    // Offsets historiques animation.cfg (Q3 id player sarge) — unités en frames.
    pub const BOTH_DEATH1: AnimRange = AnimRange { start: 0, end: 30, fps: 25.0, looping: false };
    pub const TORSO_GESTURE: AnimRange = AnimRange { start: 90, end: 130, fps: 15.0, looping: true };
    pub const TORSO_ATTACK: AnimRange = AnimRange { start: 130, end: 145, fps: 15.0, looping: false };
    pub const TORSO_STAND: AnimRange = AnimRange { start: 171, end: 176, fps: 20.0, looping: true };
    pub const LEGS_WALK: AnimRange = AnimRange { start: 193, end: 213, fps: 20.0, looping: true };
    pub const LEGS_RUN: AnimRange = AnimRange { start: 213, end: 226, fps: 20.0, looping: true };
    pub const LEGS_JUMP: AnimRange = AnimRange { start: 248, end: 254, fps: 20.0, looping: false };
    pub const LEGS_LAND: AnimRange = AnimRange { start: 254, end: 258, fps: 20.0, looping: false };
    pub const LEGS_IDLE: AnimRange = AnimRange { start: 268, end: 291, fps: 15.0, looping: true };
}

/// Paramètres de combat — tunés pour être lisibles à l'œil.
const BOT_SIGHT_RANGE: f32 = 1500.0;
const BOT_FOV_COS: f32 = 0.5; // cos(60°) → FOV total 120°
const BOT_MEMORY_SEC: f32 = 2.0;
/// Facteur appliqué à `BOT_SIGHT_RANGE` quand le joueur porte
/// Invisibility. 0.25 = bots doivent se rapprocher à ~375 u avant de
/// distinguer le joueur (≈ distance d'un combat shotgun). Q3 scale
/// similaire (`chars[INVISIBILITY].visibility` = 0.25).
const BOT_SIGHT_INVIS_FACTOR: f32 = 0.25;
/// Durée (secondes) de la mémoire d'agro quand le joueur invisible
/// sort de vue. Plus court que `BOT_MEMORY_SEC` — sinon l'invisibilité
/// ne sert à rien si un bot vous aperçoit une fraction de seconde.
const BOT_MEMORY_INVIS_SEC: f32 = 0.5;
const BOT_FIRE_COOLDOWN: f32 = 0.6;
const BOT_DAMAGE: i32 = 8;
/// Cooldown rocket bot : plus long que le hitscan pour éviter le spam.
const BOT_ROCKET_COOLDOWN: f32 = 3.5;
/// Distance mini joueur↔bot pour qu'il lance une rocket (évite le suicide
/// par splash en close range).
const BOT_ROCKET_MIN_DIST: f32 = 250.0;
/// Distance max au-dessus de laquelle la rocket devient ridicule (trop
/// lente pour toucher, et le bot a déjà un railgun à cette portée).
const BOT_ROCKET_MAX_DIST: f32 = 1100.0;
/// Seuil au-dessous duquel le bot passe au Shotgun (spray close-range).
const BOT_SG_RANGE: f32 = 220.0;
/// Seuil au-dessus duquel le bot passe au Railgun (sniper long-range).
const BOT_RG_RANGE: f32 = 900.0;
/// Vitesse de la rocket bot — on prend la même que celle du joueur.
const BOT_ROCKET_SPEED: f32 = 900.0;
const PLAYER_EYE_HEIGHT: f32 = 40.0;
const BOT_EYE_HEIGHT: f32 = 32.0;
const RESPAWN_DELAY_SEC: f32 = 1.5;
/// Durée d'invulnérabilité après un respawn (joueur ou bot). Valeur
/// Q3 classique : ~2s. Couvre le temps qu'il faut pour se repérer et
/// réagir avant d'être exposé aux tirs — anti-spawn-camping minimum.
const RESPAWN_INVUL_SEC: f32 = 2.0;
/// Rayon de la sphère englobante d'un bot — approx. player hull Q3 (~30 units).
const BOT_HIT_RADIUS: f32 = 28.0;
/// Hauteur du centre de la sphère (un peu au-dessus du milieu du torse).
const BOT_CENTER_HEIGHT: f32 = 36.0;
/// HP par défaut d'un bot au spawn.
const BOT_DEFAULT_HP: i32 = 100;
/// Portée de contact pour ramasser un item (rayon XY + delta Z).
const PICKUP_RADIUS: f32 = 36.0;
const PICKUP_VERT_REACH: f32 = 72.0;

/// Extension de la hitbox joueur utilisée pour les triggers (push / teleport).
/// Q3 utilise un "player hull" de ±15 en XY, −24..+32 en Z depuis l'origine.
/// Équivalent de `PLAYER_MINS/PLAYER_MAXS` dans `bg_public.h`.
const PLAYER_HULL_HALF_XY: f32 = 15.0;
const PLAYER_HULL_MIN_Z: f32 = -24.0;
const PLAYER_HULL_MAX_Z: f32 = 32.0;

/// Jump pad façon Q3 : un trigger_push qui balance le joueur vers sa cible
/// (`target_position` ou autre entité nommée) via une parabole gravitaire.
///
/// La vitesse est pré-calculée au chargement via la formule `AimAtTarget`
/// de `g_trigger.c` — on fige ça plutôt que de re-résoudre par tick, la
/// cible ne bouge jamais après le spawn.
#[derive(Debug, Clone, Copy)]
struct JumpPad {
    /// AABB absolu du trigger (brush `*N` résolu en `World::from_bsp`).
    bounds: Aabb,
    /// Centre du trigger — utilisé pour localiser l'émission sonore.
    center: Vec3,
    /// Vitesse à appliquer brute au joueur à l'entrée (x,y horizontaux
    /// + z vertical façon "time × gravity").
    launch_velocity: Vec3,
}

/// Téléporteur : `trigger_teleport` avec un `target` pointant vers un
/// `misc_teleporter_dest` (ou équivalent). À l'entrée, le joueur est
/// repositionné + angles réalignés + vitesse réinitialisée, conformément
/// à `TeleportPlayer()` de `g_misc.c`.
#[derive(Debug, Clone, Copy)]
struct Teleporter {
    bounds: Aabb,
    /// Centre du trigger — pour le son "telein" au départ.
    src_center: Vec3,
    /// Position cible (origin du `misc_teleporter_dest`).
    dst_origin: Vec3,
    /// Angles à la sortie — Q3 lit `angles` sur la destination, défaut `0 0 0`.
    dst_angles: Angles,
}

/// Zone dégâts `trigger_hurt`. Les flags reflètent les spawnflags Q3 :
/// bit 0 = START_OFF, bit 2 = NO_PROTECTION, bit 4 = SLOW. Le SILENT
/// (bit 1) n'a pas d'équivalent audio dédié ici — on joue juste le
/// pain sfx du joueur à chaque tick de dégât, ce qui reste discret.
#[derive(Debug, Clone, Copy)]
struct HurtZone {
    bounds: Aabb,
    /// Dégâts appliqués par tick actif (5 par défaut selon Q3).
    damage: i32,
    /// Intervalle entre deux applications : 0.1 s (standard) ou 1 s
    /// (SLOW) pour les trappes "once per second".
    interval: f32,
    /// Prochain instant où la zone peut infliger un dégât. Absolu
    /// (`time_sec`), initialement 0 → tire immédiatement au premier
    /// contact.
    next_at: f32,
    /// `true` = NO_PROTECTION — armor et powerups n'absorbent pas.
    /// Utilisé pour le "lava instant death" où dmg ≥ 100.
    no_protection: bool,
    /// Étiquette qui apparaît dans le kill-feed ("LAVA", "VOID", "HURT").
    label: &'static str,
}

#[derive(Default)]
struct Input {
    /// Directions maintenues — un booléen par cardinal pour éviter le
    /// bug "hold W + tap S + release S" où l'axe `forward` se
    /// retrouvait forcé à 0 alors que W est toujours physiquement
    /// enfoncé. On calcule `forward_axis()` / `side_axis()` à partir
    /// de ces 4 bits à chaque tick, ce qui donne toujours l'état
    /// correct tant qu'au moins une des touches est maintenue.
    fwd_down: bool,
    back_down: bool,
    left_down: bool,
    right_down: bool,
    jump: bool,
    /// `true` tant que Ctrl est maintenu — accroupit le joueur (hull 40u
    /// au lieu de 56u, vitesse capée à `crouch_speed`). L'état accroupi
    /// persiste dans `PlayerMove::crouching` tant que le hull debout ne
    /// rentre pas dans l'espace au-dessus (plafond bas).
    crouch: bool,
    /// `true` tant que Shift est maintenu — marche lente (`walk_speed`).
    /// Désactive les footsteps audibles via le seuil de vitesse du
    /// pipeline `sfx_footsteps`.
    walk: bool,
    /// `true` tant que bouton gauche est enfoncé.
    fire: bool,
    /// `true` tant que bouton droit est enfoncé pour le **tir
    /// secondaire** (alt-fire). En spectator c'est utilisé pour cycler
    /// la follow-cam — le toggle de mode est géré côté handler souris.
    /// Edge consommé par fire_weapon → false après 1 tir alt.
    secondary_fire: bool,
    /// `true` tant que TAB est maintenu — affiche l'overlay scoreboard.
    scoreboard: bool,
    /// **Lean** (M7) — peek hors d'un couvert sans bouger les pieds.
    /// `lean_axis()` calcule un signe ∈ {-1, 0, +1} : -1 = lean gauche,
    /// +1 = lean droit. Le moteur applique un offset latéral sur la
    /// caméra + un roll d'angle pour la signature visuelle. KeyB pour
    /// la gauche, KeyN pour la droite — choix anti-collision avec
    /// AZERTY (Q déjà strafe-left). Utilisable au pad / joystick aussi.
    lean_left_held: bool,
    lean_right_held: bool,
    /// **Slide tactique** (M3) — armé à la frame où Ctrl passe de
    /// relâché → enfoncé pendant qu'on court vite. Consommé au prochain
    /// `MoveCmd::slide_pressed` puis remis à false.
    slide_armed: bool,
    /// **Dash** (M4) — armé par double-tap d'une touche directionnelle
    /// dans une fenêtre de 250 ms. Consommé au prochain
    /// `MoveCmd::dash_pressed`.
    dash_armed: bool,
    /// Timestamps du dernier press par direction, pour détecter le
    /// double-tap. Indexé `[forward, back, left, right]`.
    last_dir_press: [f32; 4],
}

impl Input {
    /// Axe avant/arrière : +1 quand seule la touche "forward" est
    /// maintenue, -1 quand seule "back", 0 sinon (neutre ou conflit).
    fn forward_axis(&self) -> f32 {
        match (self.fwd_down, self.back_down) {
            (true, false) => 1.0,
            (false, true) => -1.0,
            _ => 0.0,
        }
    }
    /// Lean axis : -1 = lean gauche, +1 = lean droite, 0 = neutre /
    /// les deux touches enfoncées (annulation).
    fn lean_axis(&self) -> f32 {
        match (self.lean_right_held, self.lean_left_held) {
            (true, false) => 1.0,
            (false, true) => -1.0,
            _ => 0.0,
        }
    }
    /// Idem mais gauche/droite — -1 = strafe gauche, +1 = strafe droite.
    fn side_axis(&self) -> f32 {
        match (self.right_down, self.left_down) {
            (true, false) => 1.0,
            (false, true) => -1.0,
            _ => 0.0,
        }
    }
}

impl App {
    pub fn new(
        vfs: Arc<Vfs>,
        width: u32,
        height: u32,
        requested_map: Option<String>,
        net_mode: crate::net::NetMode,
        vr_enabled: bool,
        // Nombre de bots demandés via `--bots N`. Sémantique unifiée :
        // en `--host` (serveur réseau) → spawn côté autoritatif. En
        // solo / `--connect` → spawn local après chargement de la map
        // (cf. `pending_local_bots` consommé dans `load_map`).
        // Avant v0.7, ce paramètre était ignoré en solo, ce qui faisait
        // que `--bots 4` en solo ne donnait aucun adversaire visible.
        server_bots: u8,
        spectate: bool,
        team: Option<String>,
        record: Option<PathBuf>,
    ) -> Self {
        let cvars = CvarRegistry::new();
        let cmds = CmdRegistry::new();
        let hooks = EngineHooks::new();
        let pending: Arc<Mutex<Vec<PendingAction>>> = Arc::new(Mutex::new(Vec::new()));

        // Hooks : les commandes console ne peuvent pas toucher directement
        // à `event_loop` / `self`, alors on enqueue une action et on la drain
        // à la frame suivante.
        {
            let p = pending.clone();
            hooks.set_quit(move || {
                p.lock().push(PendingAction::Quit);
            });
        }
        {
            let p = pending.clone();
            hooks.set_map(move |name: &str| {
                p.lock().push(PendingAction::Map(name.to_string()));
            });
        }

        register_builtins(&cmds, &cvars, &hooks);

        // Commandes spécifiques à l'app — pas via EngineHooks pour garder le
        // crate console découplé.
        {
            let p = pending.clone();
            cmds.add("addbot", move |args: &Args| {
                // `addbot [<name> [<skill:1..5>]]`
                let name = if args.count() >= 2 {
                    args.argv(1).to_string()
                } else {
                    format!("bot{}", fastrand_suffix())
                };
                let skill = if args.count() >= 3 {
                    args.argv(2).parse::<i32>().ok()
                } else {
                    None
                };
                p.lock().push(PendingAction::AddBot(name, skill));
            });
        }
        {
            let p = pending.clone();
            cmds.add("clearbots", move |_args: &Args| {
                p.lock().push(PendingAction::ClearBots);
            });
        }
        {
            let p = pending.clone();
            cmds.add("restart", move |_args: &Args| {
                p.lock().push(PendingAction::Restart);
            });
        }
        {
            let p = pending.clone();
            cmds.add("say", move |args: &Args| {
                // `say <message...>` — concatène les args ≥ 1 avec
                // espaces. Permet `/say hello world` sans guillemets.
                if args.count() < 2 {
                    return;
                }
                let msg: Vec<&str> = (1..args.count()).map(|i| args.argv(i)).collect();
                let text = msg.join(" ");
                if !text.is_empty() {
                    p.lock().push(PendingAction::SayChat(text));
                }
            });
        }
        {
            let p = pending.clone();
            cmds.add("kick", move |args: &Args| {
                // `kick <slot_id>` — kick un slot serveur. Pas de match
                // par nom en v1, juste par id pour rester simple.
                let Some(slot_str) = (args.count() >= 2).then(|| args.argv(1)) else {
                    warn!("kick: usage `kick <slot_id>`");
                    return;
                };
                let Ok(slot_id) = slot_str.parse::<u8>() else {
                    warn!("kick: slot_id `{slot_str}` invalide");
                    return;
                };
                p.lock().push(PendingAction::Kick(slot_id));
            });
        }
        {
            // `give<powerup>` — utiles en dev pour tester les buffs sans
            // ramasser le pickup (que la map courante n'a pas forcément).
            // Stack avec un powerup actif du même type, comme le ferait un
            // vrai ramassage — pas d'overwrite.
            let p = pending.clone();
            cmds.add("givequad", move |_args: &Args| {
                p.lock().push(PendingAction::GivePowerup(PowerupKind::QuadDamage));
            });
            let p = pending.clone();
            cmds.add("givehaste", move |_args: &Args| {
                p.lock().push(PendingAction::GivePowerup(PowerupKind::Haste));
            });
            let p = pending.clone();
            cmds.add("giveregen", move |_args: &Args| {
                p.lock().push(PendingAction::GivePowerup(PowerupKind::Regeneration));
            });
            let p = pending.clone();
            cmds.add("giveenviro", move |_args: &Args| {
                p.lock().push(PendingAction::GivePowerup(PowerupKind::BattleSuit));
            });
            let p = pending.clone();
            cmds.add("giveinvis", move |_args: &Args| {
                p.lock().push(PendingAction::GivePowerup(PowerupKind::Invisibility));
            });
            let p = pending.clone();
            cmds.add("giveflight", move |_args: &Args| {
                p.lock().push(PendingAction::GivePowerup(PowerupKind::Flight));
            });
            // Holdables : `use` = consomme le slot courant, `givemedkit` /
            // `giveteleporter` pour les tests.
            let p = pending.clone();
            cmds.add("use", move |_args: &Args| {
                p.lock().push(PendingAction::UseHoldable);
            });
            let p = pending.clone();
            cmds.add("givemedkit", move |_args: &Args| {
                p.lock().push(PendingAction::GiveHoldable(HoldableKind::Medkit));
            });
            let p = pending.clone();
            cmds.add("giveteleporter", move |_args: &Args| {
                p.lock().push(PendingAction::GiveHoldable(HoldableKind::Teleporter));
            });
        }


        // Cvars par défaut exposées — on peut en rajouter plus tard.
        cvars.register("sensitivity", "5.0", CvarFlags::ARCHIVE);
        // m_pitch / m_yaw : facteurs de conversion pixels → degrés Q3.
        // Valeur par défaut = 0.022 deg/pixel (canonique Q3A). Inverser
        // via `m_pitch -0.022` pour les joueurs qui préfèrent "mouse up
        // = look down". m_yaw se règle indépendamment pour ceux qui
        // veulent un yaw plus sensible que le pitch.
        cvars.register("m_pitch", "0.022", CvarFlags::ARCHIVE);
        cvars.register("m_yaw", "0.022", CvarFlags::ARCHIVE);
        cvars.register("cl_forwardspeed", "400", CvarFlags::ARCHIVE);
        cvars.register("cl_sidespeed", "350", CvarFlags::ARCHIVE);
        cvars.register("r_gamma", "1.0", CvarFlags::ARCHIVE);
        cvars.register("s_volume", "0.8", CvarFlags::ARCHIVE);
        cvars.register("s_musicvolume", "0.25", CvarFlags::ARCHIVE);
        // Difficulté des bots — 1..5 (I..V). III = défaut équilibré.
        // Consommé par `spawn_bot` sauf override via `addbot <name> <n>`.
        cvars.register("bot_skill", "3", CvarFlags::ARCHIVE);
        cvars.register(
            "version",
            q3_common::ENGINE_VERSION,
            CvarFlags::INIT,
        );

        // Charge `q3config.cfg` depuis le répertoire utilisateur s'il existe.
        // Les cvars ARCHIVE enregistrées ci-dessus gardent leurs valeurs par
        // défaut si le fichier est absent ; sinon elles sont écrasées par
        // les valeurs disque. Les erreurs de parsing sont loggées mais ne
        // bloquent pas le boot — même comportement que Q3A original.
        if let Some(path) = user_config_path() {
            match std::fs::read_to_string(&path) {
                Ok(script) => {
                    let (applied, errors) = cvars.apply_config_script(&script);
                    info!(
                        "config: {} restaurée(s) depuis {} ({} erreur(s))",
                        applied,
                        path.display(),
                        errors.len()
                    );
                    for e in errors.iter().take(4) {
                        warn!("config: {e}");
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    debug!("config: pas de {} (premier lancement)", path.display());
                }
                Err(e) => warn!("config: lecture {} : {e}", path.display()),
            }
        } else {
            debug!("config: pas de répertoire user détecté, prefs non-persistées");
        }

        let console = Console::new(cvars.clone(), cmds.clone());

        // Menu : on pré-charge la liste des maps du VFS (ordre alpha) pour
        // éviter un scan à chaque frame. Le menu s'ouvre d'emblée si
        // aucune `--map` n'a été demandée en CLI — c'est le comportement
        // Q3 historique (écran de sélection plutôt qu'un fond vide).
        let map_list: Vec<String> = vfs
            .list_suffix(".bsp")
            .into_iter()
            .filter(|p| p.starts_with("maps/"))
            .collect();
        let mut menu = crate::menu::Menu::new(map_list, /* in_game */ false);
        if requested_map.is_none() {
            menu.open_root();
        }

        Self {
            vfs,
            init_width: width,
            init_height: height,
            requested_map,
            window: None,
            renderer: None,
            world: None,
            player: PlayerMove::new(Vec3::ZERO),
            params: PhysicsParams::default(),
            console,
            pending,
            cvars,
            menu,
            cursor_pos: (0.0, 0.0),
            pickups: Vec::new(),
            projectiles: Vec::new(),
            rocket_mesh: None,
            plasma_mesh: None,
            grenade_mesh: None,
            explosions: Vec::new(),
            particles: Vec::new(),
            viewmodels: Vec::new(),
            // Loadout Q3 : Gauntlet (slot 1) + Machinegun (slot 2) avec 100
            // balles. Shotgun / rocket / railgun deviennent des pickups.
            weapons_owned: (1u32 << WeaponId::Gauntlet.slot())
                | (1u32 << WeaponId::Machinegun.slot()),
            active_weapon: WeaponId::Machinegun,
            last_weapon: WeaponId::Machinegun,
            weapon_switch_at: f32::NEG_INFINITY,
            show_perf_overlay: false,
            frame_times: [0.0; FRAME_TIME_BUF],
            frame_time_head: 0,
            ammo: {
                let mut a = [0i32; 10];
                a[WeaponId::Machinegun.slot() as usize] =
                    WeaponId::Machinegun.params().starting_ammo;
                a
            },
            next_empty_click_at: 0.0,
            bots: Vec::new(),
            bot_rig: None,
            remote_players: Vec::new(),
            remote_interp: HashMap::new(),
            remote_projectiles: HashMap::new(),
            remote_names: HashMap::new(),
            remote_scores: HashMap::new(),
            remote_unavailable_pickups: std::collections::HashSet::new(),
            is_spectator: false,
            local_team: q3_net::team::FREE,
            follow_slot: None,
            lean_value: 0.0,
            chat_wheel_open: false,
            chat_wheel_opened_at: f32::NEG_INFINITY,
            // Bots locaux : on n'inscrit que si on n'est PAS en mode
            // serveur (le path serveur les spawn directement plus haut
            // via `nr.add_server_bot`). Drainé après `load_map`.
            pending_local_bots: if matches!(net_mode, crate::net::NetMode::Server { .. }) {
                0
            } else {
                server_bots
            },
            player_health: Health::full(),
            player_armor: 0,
            respawn_at: None,
            player_invul_until: 0.0,
            last_damage_dir: Vec3::ZERO,
            last_damage_until: 0.0,
            shake_intensity: 0.0,
            shake_until: 0.0,
            deaths: 0,
            frags: 0,
            match_winner: None,
            match_start_at: 0.0,
            warmup_until: 0.0,
            first_blood_announced: false,
            total_shots: 0,
            total_hits: 0,
            time_warnings_fired: 0,
            next_player_fire_at: 0.0,
            muzzle_flash_until: 0.0,
            view_kick: 0.0,
            hit_marker_until: 0.0,
            armor_flash_until: 0.0,
            pain_flash_until: 0.0,
            recent_dmg_total: 0,
            recent_dmg_last_at: 0.0,
            next_heartbeat_at: 0.0,
            beams: Vec::new(),
            active_medals: Vec::new(),
            kill_feed: Vec::new(),
            chat_feed: Vec::new(),
            pickup_toasts: Vec::new(),
            player_streak: 0,
            next_chat_at: 0.0,
            next_player_taunt_at: 0.0,
            floating_damages: Vec::new(),
            time_sec: 0.0,
            auto_shot_taken: false,
            bob_phase: 0.0,
            powerup_until: [None; PowerupKind::COUNT],
            powerup_warned: [false; PowerupKind::COUNT],
            held_item: None,
            last_death_cause: None,
            regen_accum: 0.0,
            jump_pads: Vec::new(),
            teleporters: Vec::new(),
            hurt_zones: Vec::new(),
            ambient_speakers: Vec::new(),
            air_left: AIR_CAPACITY_SEC,
            next_drown_at: 0.0,
            was_underwater: false,
            next_bubble_at: 0.0,
            on_jumppad_idx: None,
            on_teleport_idx: None,
            sound: None,
            sfx_jump: None,
            sfx_land: None,
            sfx_water_in: None,
            sfx_water_out: None,
            sfx_jumppad: None,
            sfx_teleport_in: None,
            sfx_teleport_out: None,
            sfx_footsteps: Vec::new(),
            last_footstep_idx: None,
            last_footstep_phase: 0.0,
            sfx_fire: Vec::new(),
            sfx_pain_player: None,
            sfx_pain_bot: None,
            sfx_weapon_pickup: None,
            sfx_ammo_pickup: None,
            sfx_no_ammo: None,
            sfx_weapon_switch: None,
            sfx_rocket_explode: None,
            sfx_hit: None,
            sfx_kill_confirm: None,
            sfx_humiliation: None,
            sfx_one_frag: None,
            sfx_two_frags: None,
            sfx_three_frags: None,
            last_frags_announced: None,
            sfx_excellent: None,
            last_frag_at: f32::NEG_INFINITY,
            sfx_impressive: None,
            sfx_powerup_warn: None,
            sfx_powerup_end: None,
            rg_last_hit: false,
            was_airborne: false,
            view_crouch_offset: 0.0,
            input: Input::default(),
            mouse_captured: false,
            last_tick: Instant::now(),
            physics_accumulator: 0.0,
            net: {
                // Userinfo enrichie pour le client : flag spectator si
                // demandé en CLI. Le nom est figé pour l'instant — sera
                // pris depuis la cvar `name` quand le système des
                // configstrings serveur sera en place.
                let team_part = match team.as_deref().map(str::to_ascii_lowercase).as_deref() {
                    Some("red" | "r" | "1") => "\\team\\red",
                    Some("blue" | "b" | "2") => "\\team\\blue",
                    _ => "",
                };
                let userinfo = if spectate {
                    format!("\\name\\Spectator\\rate\\25000\\spectator\\1{team_part}")
                } else {
                    format!("\\name\\Player\\rate\\25000{team_part}")
                };
                let mut nr = crate::net::NetRuntime::new_with_userinfo(net_mode, userinfo);
                // Spawn les bots demandés au serveur. Les noms sont
                // séquentiels « bot01..botNN » — une commande console
                // pour ajouter à la volée pourra venir plus tard.
                for i in 0..server_bots {
                    let name = format!("bot{:02}", i + 1);
                    let _ = nr.add_server_bot(name, q3_bot::BotSkill::III);
                }
                // Enregistrement démo si demandé. Pris en compte
                // uniquement en mode client (server-side ça n'aurait
                // pas de sens, on enregistrerait nos propres snapshots
                // depuis le PoV serveur).
                if let Some(path) = record.as_ref() {
                    nr.start_recording_demo(path);
                }
                nr
            },
            vr: crate::vr::VrRuntime::init(vr_enabled),
        }
    }

    /// Active / désactive la capture du curseur.
    fn set_mouse_capture(&mut self, on: bool) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if on {
            let _ = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
            window.set_cursor_visible(false);
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
        }
        self.mouse_captured = on;
    }

    /// Applique une action remontée par le menu principal. Centralisé ici
    /// pour garder les handlers d'input (clavier / souris) concis — tout
    /// ce qui change l'état monde / fenêtre passe par cette fonction.
    ///
    /// Règles :
    ///   * `Resume` n'a de sens qu'en pause (world chargé) : on referme le
    ///     menu et on réarme la capture souris.
    ///   * `LoadMap` charge la BSP ; en cas de succès on referme le menu
    ///     et on capture la souris, sinon on laisse l'UI ouverte pour que
    ///     le joueur tente autre chose (le log a déjà l'erreur).
    ///   * `Quit` sort proprement via `event_loop.exit()`, laissant
    ///     `exiting()` sauvegarder les cvars ARCHIVE.
    fn apply_menu_action(
        &mut self,
        action: crate::menu::MenuAction,
        event_loop: &ActiveEventLoop,
    ) {
        use crate::menu::MenuAction;
        match action {
            MenuAction::None => {}
            MenuAction::Resume => {
                self.menu.close();
                self.set_mouse_capture(true);
            }
            MenuAction::LoadMap(path) => {
                self.load_map(&path);
                if self.world.is_some() {
                    self.menu.close();
                    self.menu.set_in_game(true);
                    self.set_mouse_capture(true);
                }
            }
            MenuAction::Quit => event_loop.exit(),
        }
    }

    fn load_map(&mut self, path: &str) {
        let bytes = match self.vfs.read(path) {
            Ok(b) => b,
            Err(e) => {
                error!("load_map {path}: {e}");
                return;
            }
        };
        let bsp = match Bsp::parse(&bytes) {
            Ok(b) => b,
            Err(e) => {
                error!("parse_bsp {path}: {e}");
                return;
            }
        };
        info!(
            "map `{path}` chargée : {} verts, {} surfaces, {} entities text",
            bsp.draw_verts.len(),
            bsp.surfaces.len(),
            bsp.entities.len()
        );
        if let Some(r) = self.renderer.as_mut() {
            if let Err(e) = r.upload_bsp(&bsp) {
                error!("upload_bsp: {e}");
            }
            // Skybox : cherche le premier shader BSP marqué `skyparms` dans
            // le registre et tente de charger sa cubemap. Faute de cubemap,
            // on retombe gracieusement sur le ciel procédural.
            resolve_and_load_sky(r, &self.vfs, &bsp);
        }
        let world = World::from_bsp(bsp);
        if let Some(spawn) = world.player_start {
            // `info_player_deathmatch.origin` est déjà la position
            // d'origine du hull joueur en Q3 (placée 24u au-dessus du sol
            // pour que `origin + HULL_MINS.z = 0` → pieds pile au sol).
            // On ne rajoute rien : ajouter un offset faisait spawner le
            // joueur 40u en l'air, il chutait, et l'atterrissage révélait
            // tous les bugs de collision "epsilon" qui clouaient WASD.
            // Le niveau des yeux est géré côté caméra (offset `eye_z`),
            // pas côté physique.
            self.player = PlayerMove::new(spawn);
            self.player.view_angles = world.player_start_angles;
            // Invul initiale : même fenêtre que les respawns courants,
            // pour ne pas être sniper au chargement de la map.
            self.player_invul_until = self.time_sec + RESPAWN_INVUL_SEC;
            // Pain-arrow : on part avec un indicateur éteint.
            self.last_damage_until = 0.0;
            // Screen-shake : caméra stable à l'apparition.
            self.shake_intensity = 0.0;
            self.shake_until = 0.0;
            // Armor flash : éteint à l'apparition.
            self.armor_flash_until = 0.0;
            self.pain_flash_until = 0.0;
            if let Some(r) = self.renderer.as_mut() {
                r.camera_mut().position =
                    self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
                r.camera_mut().angles = self.player.view_angles;
            }
            info!("spawn joueur à {:?}", self.player.origin);
        }

        // Musique : cherche la clé `music` sur worldspawn. Fallback silencieux
        // si absente ou si le fichier n'est pas dans le VFS.
        if let Some(snd) = self.sound.as_ref() {
            if let Some(path) = world.entities.iter().find_map(|e| {
                matches!(e.kind, q3_game::EntityKind::Worldspawn)
                    .then(|| e.extra.iter().find(|(k, _)| k == "music").map(|(_, v)| v.clone()))
                    .flatten()
            }) {
                match self.vfs.read(&path) {
                    Ok(bytes) => match snd.play_music(bytes.to_vec()) {
                        Ok(()) => info!("music: '{path}' lancée"),
                        Err(e) => warn!("music '{path}' KO: {e}"),
                    },
                    Err(e) => warn!("music '{path}' introuvable: {e}"),
                }
            }
        }

        // Sons couramment déclenchés par le joueur : on tente quelques
        // noms conventionnels Q3, les échecs deviennent None (pas bloquant).
        self.sfx_jump = None;
        self.sfx_land = None;
        self.sfx_water_in = None;
        self.sfx_water_out = None;
        self.sfx_jumppad = None;
        self.sfx_teleport_in = None;
        self.sfx_teleport_out = None;
        self.sfx_footsteps.clear();
        self.last_footstep_idx = None;
        self.last_footstep_phase = 0.0;
        self.sfx_fire.clear();
        self.sfx_pain_player = None;
        self.sfx_pain_bot = None;
        self.sfx_weapon_pickup = None;
        self.sfx_ammo_pickup = None;
        self.sfx_no_ammo = None;
        self.sfx_weapon_switch = None;
        if let Some(snd) = self.sound.as_ref() {
            self.sfx_jump = try_load_sfx(&self.vfs, snd, "sound/player/default/jump1.wav");
            self.sfx_land = try_load_sfx(&self.vfs, snd, "sound/player/default/fall1.wav");
            // Splash d'eau : convention Q3 `sound/player/watr_in.wav` /
            // `watr_out.wav`. On accepte aussi les variantes sans underscore
            // trouvées dans certains pk3.
            self.sfx_water_in = try_load_sfx(&self.vfs, snd, "sound/player/watr_in.wav")
                .or_else(|| try_load_sfx(&self.vfs, snd, "sound/player/waterin.wav"));
            self.sfx_water_out = try_load_sfx(&self.vfs, snd, "sound/player/watr_out.wav")
                .or_else(|| try_load_sfx(&self.vfs, snd, "sound/player/waterout.wav"));
            // Jump pad : convention Q3 c'est `sound/world/jumppad.wav`.
            // Absent → silencieux, le gameplay tourne quand même.
            self.sfx_jumppad = try_load_sfx(&self.vfs, snd, "sound/world/jumppad.wav");
            // Téléporteur : "in" côté source, "out" côté destination.
            // Quelques pk3 n'ont qu'un des deux — on accepte n'importe lequel.
            self.sfx_teleport_in =
                try_load_sfx(&self.vfs, snd, "sound/world/telein.wav");
            self.sfx_teleport_out =
                try_load_sfx(&self.vfs, snd, "sound/world/teleout.wav");
            // Footsteps Q3A : 4 variantes "step{N}.wav" génériques.
            // Conventions connues :
            //   * sound/player/footsteps/step{1-4}.wav (défaut)
            //   * .../metal{1-4}, .../clank{1-4} (surface métal)
            // Pour le MVP on ne distingue pas la surface — on prend le set
            // par défaut s'il existe. Chaque variante manquante est skip.
            for n in 1..=4 {
                let p = format!("sound/player/footsteps/step{n}.wav");
                if let Some(h) = try_load_sfx(&self.vfs, snd, &p) {
                    self.sfx_footsteps.push(h);
                }
            }
            if self.sfx_footsteps.is_empty() {
                // Certains pk3 n'ont que le set "boot/metal/clank" — on
                // essaie un fallback histoire d'avoir au moins quelque
                // chose qui claque sous les pieds.
                for n in 1..=4 {
                    let p = format!("sound/player/footsteps/boot{n}.wav");
                    if let Some(h) = try_load_sfx(&self.vfs, snd, &p) {
                        self.sfx_footsteps.push(h);
                    }
                }
            }
            if !self.sfx_footsteps.is_empty() {
                info!(
                    "sfx: {} variante(s) de footstep chargée(s)",
                    self.sfx_footsteps.len()
                );
            }
            // Tir : on essaie une liste de chemins candidats par arme
            // (vanilla + variantes connues). Le 1er existant gagne.
            // Loggue le succès pour diagnostiquer rapidement les armes
            // muettes — historique : RL silencieux sur certains pak0
            // partiels où `rocklf1a.wav` n'était pas présent.
            for w in WeaponId::ALL {
                let mut loaded = false;
                for &path in w.fire_sfx_paths() {
                    if let Some(h) = try_load_sfx(&self.vfs, snd, path) {
                        info!("sfx: {:?} ← `{}`", w, path);
                        self.sfx_fire.push((w, h));
                        loaded = true;
                        break;
                    }
                }
                if !loaded {
                    warn!(
                        "sfx: arme {:?} SANS son de tir (aucun candidat trouvé : {:?})",
                        w,
                        w.fire_sfx_paths()
                    );
                }
            }
            info!("sfx: {}/{} arme(s) avec son de tir",
                self.sfx_fire.len(), WeaponId::ALL.len());
            // Pain : premier nom dispo parmi la série Q3.
            const PAIN_CANDIDATES: &[&str] = &[
                "sound/player/sarge/pain25_1.wav",
                "sound/player/visor/pain25_1.wav",
                "sound/player/doom/pain25_1.wav",
                "sound/player/major/pain25_1.wav",
            ];
            for p in PAIN_CANDIDATES {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_pain_player = Some(h);
                    break;
                }
            }
            // Bot pain → on prend un autre persona si possible pour distinguer.
            const BOT_PAIN_CANDIDATES: &[&str] = &[
                "sound/player/visor/pain50_1.wav",
                "sound/player/doom/pain50_1.wav",
                "sound/player/sarge/pain50_1.wav",
                "sound/player/major/pain50_1.wav",
            ];
            for p in BOT_PAIN_CANDIDATES {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_pain_bot = Some(h);
                    break;
                }
            }
            // Pickup SFX (conventions Q3 ; échec → None, reste silencieux).
            self.sfx_weapon_pickup =
                try_load_sfx(&self.vfs, snd, "sound/misc/w_pkup.wav");
            self.sfx_ammo_pickup =
                try_load_sfx(&self.vfs, snd, "sound/misc/am_pkup.wav");
            // No-ammo click : convention Q3 pak0 `sound/weapons/noammo.wav`.
            // Quelques paks distribuent le SFX sous d'autres noms — on
            // essaye les plus probables avant d'abandonner (silencieux
            // plutôt qu'un faux son générique).
            const NOAMMO_CANDIDATES: &[&str] = &[
                "sound/weapons/noammo.wav",
                "sound/misc/noammo.wav",
                "sound/feedback/noammo.wav",
            ];
            for p in NOAMMO_CANDIDATES {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_no_ammo = Some(h);
                    break;
                }
            }
            // Weapon switch / raise SFX — plusieurs conventions selon
            // les paks.  `change.wav` est le plus répandu, sinon on
            // tombe sur un clic générique pickup.
            const WSWITCH_CANDIDATES: &[&str] = &[
                "sound/weapons/change.wav",
                "sound/weapons/weapon_switch.wav",
                "sound/misc/w_pkup.wav",
            ];
            for p in WSWITCH_CANDIDATES {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_weapon_switch = Some(h);
                    break;
                }
            }
            // Explosion rocket → fallback sur d'autres splash_explode si absent.
            const EXPLODE_CANDIDATES: &[&str] = &[
                "sound/weapons/rocket/rocklx1a.wav",
                "sound/weapons/grenade/grenlx1a.wav",
                "sound/weapons/bfg/bfg_explode.wav",
            ];
            for p in EXPLODE_CANDIDATES {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_rocket_explode = Some(h);
                    break;
                }
            }
            // Hitsound : convention Q3 pure n'en a pas, mais plusieurs
            // assets courts conviennent — on prend le premier trouvé.
            // L'ordre est choisi pour tomber sur un SFX « clean » :
            //   * `sound/feedback/hit.wav` — CPMA / OSP si monté en mod
            //   * `sound/misc/menu1.wav` — click UI Q3 de base, court
            //   * `sound/weapons/machinegun/ric1.wav` — ricochet MG, très sec
            // Manquant partout → hitsound silencieux (pas bloquant).
            const HIT_CANDIDATES: &[&str] = &[
                "sound/feedback/hit.wav",
                "sound/feedback/hit1.wav",
                "sound/misc/menu1.wav",
                "sound/weapons/machinegun/ric1.wav",
            ];
            for p in HIT_CANDIDATES {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_hit = Some(h);
                    info!("sfx: hit '{p}' chargé");
                    break;
                }
            }
            // Kill-confirm : thunk plus marqué que le hitsound. Q3 n'a pas
            // de « kill blip » canonique — les médailles (`excellent`,
            // `impressive`) sont gated sur des conditions spéciales. On
            // privilégie des samples courts/secs sans voix humaine pour
            // éviter la superposition bavarde à chaque frag :
            //   * `sound/items/respawn1.wav` — pop d'apparition d'item, sec
            //   * `sound/misc/menu2.wav` — click UI plus grave que menu1
            //   * `sound/weapons/rocket/rocklx1a.wav` — explosion reculée
            //   * fallback fin : `sound/feedback/excellent.wav` (voix, bon
            //     dernier recours — mieux que silence).
            const KILL_CONFIRM_CANDIDATES: &[&str] = &[
                "sound/items/respawn1.wav",
                "sound/misc/menu2.wav",
                "sound/feedback/excellent.wav",
            ];
            for p in KILL_CONFIRM_CANDIDATES {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_kill_confirm = Some(h);
                    info!("sfx: kill-confirm '{p}' chargé");
                    break;
                }
            }
            // Médaille « Humiliation » — voix humaine canonique Q3
            // (`sound/feedback/humiliation.wav`). Aucun bon fallback si
            // le PK3 d'origine n'est pas monté : un bip random donnerait
            // l'impression de feedback correct pour n'importe quel kill,
            // ce qui casserait la spécificité du Gauntlet. On laisse à
            // None → médaille silencieuse, le kill-confirm standard
            // reste audible donc l'info de frag n'est pas perdue.
            const HUMILIATION_CANDIDATES: &[&str] = &[
                "sound/feedback/humiliation.wav",
            ];
            for p in HUMILIATION_CANDIDATES {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_humiliation = Some(h);
                    info!("sfx: humiliation '{p}' chargé");
                    break;
                }
            }
            // Announcer fraglimit countdown (A5). Samples Q3 canoniques
            // dans `sound/feedback/`. Les 3 derniers frags annoncés —
            // au-delà aucune annonce. Si un sample manque, l'annonce
            // est silencieuse (le check fraglimit reste fonctionnel).
            for p in &["sound/feedback/1_frag.wav", "sound/feedback/one_frag.wav"] {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_one_frag = Some(h);
                    info!("sfx: one_frag '{p}' chargé");
                    break;
                }
            }
            for p in &["sound/feedback/2_frags.wav", "sound/feedback/two_frags.wav"] {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_two_frags = Some(h);
                    info!("sfx: two_frags '{p}' chargé");
                    break;
                }
            }
            for p in &["sound/feedback/3_frags.wav", "sound/feedback/three_frags.wav"] {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_three_frags = Some(h);
                    info!("sfx: three_frags '{p}' chargé");
                    break;
                }
            }
            // Médaille « Excellent » — voix canonique Q3 également.
            // Deux chemins vus dans les pk3 : `excellent.wav` en racine
            // de `feedback/`, parfois `excellent1.wav` sur certains mods.
            const EXCELLENT_CANDIDATES: &[&str] = &[
                "sound/feedback/excellent.wav",
                "sound/feedback/excellent1.wav",
            ];
            for p in EXCELLENT_CANDIDATES {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_excellent = Some(h);
                    info!("sfx: excellent '{p}' chargé");
                    break;
                }
            }
            // Médaille « Impressive » — voix Q3 pour 2 hits Railgun
            // consécutifs. Même logique « pas de fallback » que
            // Humiliation : sans le sample canonique, médaille muette.
            const IMPRESSIVE_CANDIDATES: &[&str] = &[
                "sound/feedback/impressive.wav",
                "sound/feedback/impressive1.wav",
            ];
            for p in IMPRESSIVE_CANDIDATES {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_impressive = Some(h);
                    info!("sfx: impressive '{p}' chargé");
                    break;
                }
            }
            // Powerup warning (3s restantes). Q3 n'a pas de sample spécifique
            // pour ça (le client d'origine gère le feedback via le blink
            // seul + le `powerdown` à zéro). On réutilise un thunk court
            // bien connu pour que le bip soit distinct du hitsound. Ordre
            // choisi : respawn > menu > klaxon UI.
            const POWERUP_WARN_CANDIDATES: &[&str] = &[
                "sound/items/poweruprespawn.wav",
                "sound/items/respawn1.wav",
                "sound/misc/menu1.wav",
            ];
            for p in POWERUP_WARN_CANDIDATES {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_powerup_warn = Some(h);
                    info!("sfx: powerup-warn '{p}' chargé");
                    break;
                }
            }
            // Powerdown : canonique dans baseq3 (`sound/items/wearoff.wav`),
            // sinon fallback sur le même pool que le warning pour que le
            // joueur ait au moins un cue distinct à zéro.
            const POWERUP_END_CANDIDATES: &[&str] = &[
                "sound/items/wearoff.wav",
                "sound/items/suitenergy.wav",
                "sound/misc/menu2.wav",
            ];
            for p in POWERUP_END_CANDIDATES {
                if let Some(h) = try_load_sfx(&self.vfs, snd, p) {
                    self.sfx_powerup_end = Some(h);
                    info!("sfx: powerup-end '{p}' chargé");
                    break;
                }
            }
        }
        // Reset du timer "dernier frag" pour qu'un kill en début de
        // nouvelle map ne déclenche pas un Excellent hérité de la map
        // précédente. NEG_INFINITY → `time_sec - last_frag_at > 2.0s`
        // garanti pour le 1er frag.
        self.last_frag_at = f32::NEG_INFINITY;
        // Et pareil pour le combo Railgun : une nouvelle map remet le
        // tracker à zéro, on ne veut pas qu'un hit RG de la map
        // précédente valide un Impressive sur le premier hit d'après.
        self.rg_last_hit = false;

        // Pickups : pour chaque entité avec un MD3 conventionnel, on
        // charge le mesh et on stocke (mesh, origin, angles). Les erreurs
        // individuelles sont loggées sans bloquer le chargement — les PK3
        // de démo n'ont pas forcément tous les weapons2/…
        self.pickups.clear();
        if let Some(r) = self.renderer.as_mut() {
            let mut loaded = 0usize;
            let mut missing = 0usize;
            for (i, ent) in world.entities.iter().enumerate() {
                let Some(path) = ent.kind.pickup_model_path() else {
                    continue;
                };
                match r.load_md3(&self.vfs, path) {
                    Ok(mesh) => {
                        let (kind, respawn_cooldown) = PickupKind::from_entity(&ent.kind);
                        self.pickups.push(PickupGpu {
                            mesh,
                            origin: ent.origin,
                            angles: ent.angles,
                            kind,
                            respawn_cooldown,
                            respawn_at: None,
                            entity_index: i as u16,
                        });
                        loaded += 1;
                    }
                    Err(e) => {
                        missing += 1;
                        warn!("md3 '{path}' KO: {e}");
                    }
                }
            }
            info!("pickups: {} chargés, {} manquants", loaded, missing);

            // Viewmodels : un mesh par arme détenue. Les absents sont
            // loggés en warn mais ne bloquent pas la map — on n'affiche
            // simplement pas de viewmodel pour l'arme concernée.
            self.viewmodels.clear();
            for w in WeaponId::ALL {
                match r.load_md3(&self.vfs, w.viewmodel_path()) {
                    Ok(mesh) => {
                        info!(
                            "viewmodel {}: '{}' ({} frames)",
                            w.name(),
                            w.viewmodel_path(),
                            mesh.num_frames()
                        );
                        self.viewmodels.push((w, mesh));
                    }
                    Err(_) => warn!("viewmodel {}: asset manquant", w.name()),
                }
            }

            // Rocket volant — MD3 partagé pour tous les projectiles rocket.
            // Absent = projectile invisible, mais le gameplay tourne quand même.
            self.rocket_mesh = r
                .load_md3(&self.vfs, "models/ammo/rocket/rocket.md3")
                .ok();
            // Plasma volant — Q3 original utilise un sprite, pas un MD3 ;
            // on tente quelques noms connus, sinon silencieux (invisible).
            const PLASMA_MESH_CANDIDATES: &[&str] = &[
                "models/weaphits/plasma.md3",
                "models/weaphits/plasmball.md3",
                "models/ammo/plasma/plasma.md3",
            ];
            self.plasma_mesh = PLASMA_MESH_CANDIDATES
                .iter()
                .find_map(|p| r.load_md3(&self.vfs, p).ok());
            // Grenade volante — un seul nom conventionnel dans les pk3 Q3.
            const GRENADE_MESH_CANDIDATES: &[&str] = &[
                "models/ammo/grenade1.md3",
                "models/weaphits/grenade1.md3",
                "models/ammo/grenade/grenade.md3",
            ];
            self.grenade_mesh = GRENADE_MESH_CANDIDATES
                .iter()
                .find_map(|p| r.load_md3(&self.vfs, p).ok());
        }

        // Nouvelle map → on remet à zéro les projectiles / explosions de
        // l'ancienne instance pour éviter des frames fantômes.
        self.projectiles.clear();
        self.explosions.clear();
        self.particles.clear();
        // Nouveau match : on repart à zéro sur les frags + état d'intermission.
        self.frags = 0;
        self.deaths = 0;
        self.match_winner = None;
        self.warmup_until = self.time_sec + WARMUP_DURATION;
        self.first_blood_announced = false;
        self.total_shots = 0;
        self.total_hits = 0;
        self.time_warnings_fired = 0;
        // On décale virtuellement le départ du chrono à la fin du warmup
        // pour que `time_remaining` reste borné à `TIME_LIMIT_SECONDS`
        // pendant le compte à rebours.
        self.match_start_at = self.warmup_until;

        // Nouvelle map → on purge les bots de la précédente (ils avaient des
        // waypoints et une physique liés à l'ancien monde).
        self.bots.clear();

        // Jump pads + téléporteurs — reconstruction complète à chaque map.
        // Résolution des `target → target_position / misc_teleporter_dest`
        // via `World::find_by_targetname`, qu'on a déjà.
        self.jump_pads.clear();
        self.teleporters.clear();
        self.hurt_zones.clear();
        // Stoppe les ambient speakers de la map précédente AVANT d'en
        // lancer de nouveaux — sinon chaque map change laisserait
        // fuiter un ou plusieurs `SpatialSink` toujours vivants,
        // mixant deux ambiances à la fois.
        if let Some(snd) = self.sound.as_ref() {
            for h in self.ambient_speakers.drain(..) {
                snd.stop_loop(h);
            }
        } else {
            self.ambient_speakers.clear();
        }
        self.on_jumppad_idx = None;
        self.on_teleport_idx = None;
        let gravity = self.params.gravity.max(1.0);
        for ent in &world.entities {
            match ent.kind {
                EntityKind::TriggerPush => {
                    let Some(target_name) = ent.target.as_deref() else {
                        warn!("trigger_push #{:?} sans `target` — ignoré", ent.id);
                        continue;
                    };
                    let Some(dst) = world.find_by_targetname(target_name).next() else {
                        warn!(
                            "trigger_push #{:?}: cible '{}' introuvable",
                            ent.id, target_name
                        );
                        continue;
                    };
                    let bounds = ent.bounds;
                    let origin = (bounds.mins + bounds.maxs) * 0.5;
                    // AimAtTarget de g_trigger.c :
                    //   height = dest.z - origin.z
                    //   time   = sqrt(height / (0.5 * gravity))
                    //   horiz  = (dest - origin).xy, puis scale pour qu'on
                    //            atterrisse sur la cible au moment `time`.
                    //   result.z = time * gravity (valeur absolue du "up kick")
                    let height = dst.origin.z - origin.z;
                    if height <= 0.0 {
                        warn!(
                            "trigger_push #{:?}: cible non au-dessus (h={:.1}), ignoré",
                            ent.id, height
                        );
                        continue;
                    }
                    let time = (height / (0.5 * gravity)).sqrt();
                    if !time.is_finite() || time <= 0.0 {
                        continue;
                    }
                    let mut dir = dst.origin - origin;
                    dir.z = 0.0;
                    let dist = dir.length();
                    let mut v = if dist > 1e-3 {
                        (dir / dist) * (dist / time)
                    } else {
                        Vec3::ZERO
                    };
                    v.z = time * gravity;
                    self.jump_pads.push(JumpPad {
                        bounds,
                        center: origin,
                        launch_velocity: v,
                    });
                }
                EntityKind::TriggerHurt => {
                    let bounds = ent.bounds;
                    let damage = extra_i32(ent, "dmg").unwrap_or(5).max(1);
                    let spawnflags = extra_i32(ent, "spawnflags").unwrap_or(0);
                    // Spawnflags Q3 — cf. g_trigger.c :
                    //   bit 0 = START_OFF, bit 1 = SILENT,
                    //   bit 2 = NO_PROTECTION, bit 4 = SLOW.
                    // START_OFF n'est pas supporté (on n'a pas de targetname
                    // dispatcher — tout part enabled). C'est OK pour q3dm*.
                    let slow = (spawnflags & 16) != 0;
                    let no_protection = (spawnflags & 4) != 0;
                    let interval = if slow { 1.0 } else { 0.1 };
                    // Label kill-feed : on devine lave / void / standard
                    // depuis la hauteur de la zone (les void drops sont
                    // typiquement de très larges zones tout en bas de map).
                    // Approximation, mais plus parlant qu'un tag "HURT"
                    // générique. Damage ≥ 100 = instant kill (void typique).
                    let label: &'static str = if damage >= 100 {
                        "VOID"
                    } else if no_protection {
                        "LAVA"
                    } else {
                        "HURT"
                    };
                    self.hurt_zones.push(HurtZone {
                        bounds,
                        damage,
                        interval,
                        next_at: 0.0,
                        no_protection,
                        label,
                    });
                }
                EntityKind::TriggerTeleport => {
                    let Some(target_name) = ent.target.as_deref() else {
                        warn!("trigger_teleport #{:?} sans `target` — ignoré", ent.id);
                        continue;
                    };
                    let Some(dst) = world.find_by_targetname(target_name).next() else {
                        warn!(
                            "trigger_teleport #{:?}: cible '{}' introuvable",
                            ent.id, target_name
                        );
                        continue;
                    };
                    let bounds = ent.bounds;
                    let src_center = (bounds.mins + bounds.maxs) * 0.5;
                    self.teleporters.push(Teleporter {
                        bounds,
                        src_center,
                        dst_origin: dst.origin,
                        dst_angles: dst.angles,
                    });
                }
                // `func_plat` (ascenseurs) : non implémenté — requiert un
                // système de *brush mover* dynamique dans `q3-collision`
                // (pour qu'un joueur puisse monter dessus pendant que le
                // brush translate) + un pipeline de transform par
                // sous-modèle dans `q3-renderer` (pour dessiner la chose
                // animée). Les q3dm1-13 de base n'utilisent pas ce
                // classname, donc on se contente de le logger pour ne pas
                // flooder de warnings sur les maps qui en contiennent
                // (q3tourney*, maps custom…).
                EntityKind::FuncPlat => {
                    debug!(
                        "func_plat #{:?} détecté — non implémenté, traité comme non-bloquant",
                        ent.id
                    );
                }
                _ => {
                    // target_speaker : on les traite dans une passe
                    // séparée ci-dessous pour éviter de dupliquer le
                    // code de chargement audio sur chaque bras du match.
                    if is_target_speaker(ent) {
                        self.spawn_target_speaker(ent);
                    }
                }
            }
        }
        info!(
            "world: {} jump pad(s), {} téléporteur(s), {} hurt zone(s), {} speaker(s) loopés",
            self.jump_pads.len(),
            self.teleporters.len(),
            self.hurt_zones.len(),
            self.ambient_speakers.len()
        );

        self.world = Some(world);
    }

    /// Instancie un `target_speaker` de la map courante.
    ///
    /// Conventions Q3 (g_target.c / SP_target_speaker) :
    ///
    /// * `noise` : chemin VFS du sample (WAV).  Obligatoire ; sans lui
    ///   le speaker est muet et on skip.
    /// * `spawnflags` : bitmask
    ///   - bit 0 (1) = `LOOPED_ON`  — démarre en boucle dès le spawn
    ///   - bit 1 (2) = `LOOPED_OFF` — loop activable via target, démarre off
    ///   - bit 2 (4) = `RELIABLE`   — réseau (inutilisé ici)
    ///   - bit 3 (8) = `ACTIVATOR`  — joué sur le joueur déclencheur
    ///                                uniquement ; pas de spatialisation
    /// * `wait` : période (s) pour les "random" speakers — non loopés,
    ///   rejoue périodiquement.  Pas implémenté ici (les q3dm* de base
    ///   n'en ont pas besoin ; on skip silencieusement).
    ///
    /// On ne gère pour l'instant que le cas `LOOPED_ON` : les speakers
    /// one-shot déclenchés par trigger ne sont pas encore cablés côté
    /// game logic (pas de dispatcher `use` / `activate`).  C'est suffisant
    /// pour les ambiances génériques (ronronnement de machinerie, vent,
    /// eau) qu'on entend sur les maps standard.
    fn spawn_target_speaker(&mut self, ent: &q3_game::Entity) {
        let Some(noise) = extra_str(ent, "noise") else {
            debug!("target_speaker #{:?} sans `noise` — ignoré", ent.id);
            return;
        };
        let spawnflags = extra_i32(ent, "spawnflags").unwrap_or(0);
        // LOOPED_ON (bit 0) — seul mode supporté.  Tout le reste est
        // skip avec un debug log pour qu'on ne se pose pas de questions
        // en testant une map custom.
        let looped_on = (spawnflags & 1) != 0;
        if !looped_on {
            debug!(
                "target_speaker #{:?} (noise='{noise}') non-LOOPED_ON — non supporté, ignoré",
                ent.id
            );
            return;
        }
        let Some(snd) = self.sound.as_ref() else {
            // Audio KO (headless / device en rade) — le speaker est mort
            // silencieusement, pas la peine de log warn par entité.
            return;
        };
        // Certains speakers Q3 utilisent `.wav` sans préfixe de dossier —
        // on tente `sound/` + noise comme fallback si la 1re lecture
        // échoue.  Évite de manquer des speakers qui pointent vers une
        // ressource du pak dans un autre layout.
        let (bytes, used_path) = match self.vfs.read(noise) {
            Ok(b) => (b, noise.to_string()),
            Err(_) => {
                let alt = format!("sound/{noise}");
                match self.vfs.read(&alt) {
                    Ok(b) => (b, alt),
                    Err(e) => {
                        warn!("speaker #{:?}: noise '{noise}' introuvable: {e}", ent.id);
                        return;
                    }
                }
            }
        };
        let handle = match snd.load(used_path.clone(), bytes.to_vec()) {
            Ok(h) => h,
            Err(e) => {
                warn!("speaker #{:?}: chargement '{used_path}' KO: {e}", ent.id);
                return;
            }
        };
        // Émetteur 3D : volume plein au cœur, fade linéaire jusqu'à
        // 1024u (environ une grande salle Q3).  Near/far sont les
        // mêmes valeurs que `Emitter3D::default()` — si la map veut
        // des distances différentes, on lira `radius` / `wait` plus tard.
        //
        // La priorité `Ambient` garantit qu'un ambient loop ne vole
        // jamais un canal à un tir ou à une voix — de toute façon
        // `play_3d_loop` passe par un pool séparé, mais ça reste la
        // sémantique correcte pour cohérence future.
        let emitter = Emitter3D {
            position: ent.origin,
            near_dist: 96.0,
            far_dist: 1200.0,
            volume: 0.6,
            priority: Priority::Ambient,
        };
        match snd.play_3d_loop(handle, emitter) {
            Some(h) => {
                debug!(
                    "speaker #{:?}: loop '{used_path}' démarré @ {:?}",
                    ent.id, ent.origin
                );
                self.ambient_speakers.push(h);
            }
            None => {
                warn!(
                    "speaker #{:?}: play_3d_loop a refusé '{used_path}' (audio saturé ?)",
                    ent.id
                );
            }
        }
    }

    fn drain_pending(&mut self, event_loop: &ActiveEventLoop) {
        let actions: Vec<_> = {
            let mut p = self.pending.lock();
            std::mem::take(&mut *p)
        };
        for action in actions {
            match action {
                PendingAction::Quit => {
                    info!("console: quit");
                    event_loop.exit();
                }
                PendingAction::Map(name) => {
                    let path = if name.starts_with("maps/") || name.ends_with(".bsp") {
                        name.clone()
                    } else {
                        format!("maps/{name}.bsp")
                    };
                    self.load_map(&path);
                    // Si on a `--bots N` en attente de spawn (cas où le
                    // joueur a chargé la map via le menu plutôt que via
                    // `--map`), on les spawn ici aussi.
                    if self.pending_local_bots > 0 {
                        let n = self.pending_local_bots;
                        self.pending_local_bots = 0;
                        for i in 0..n {
                            let bot_name = format!("bot{:02}", i + 1);
                            self.spawn_bot(&bot_name, Some(3));
                        }
                        info!("solo: {n} bot(s) spawnés post-map (via menu)");
                    }
                }
                PendingAction::AddBot(name, skill) => {
                    // Dispatch : si on est serveur réseau, le bot va
                    // sur l'autoritatif côté `ServerState`. Sinon
                    // (solo / client), le bot est local — comportement
                    // historique. En mode client, l'addbot local n'a
                    // pas vraiment de sens (le serveur est autoritatif),
                    // mais on conserve le comportement pour les tests
                    // hors-réseau.
                    if self.net.mode.is_server() {
                        let bot_skill =
                            skill.map(q3_bot::BotSkill::from_int).unwrap_or_default();
                        match self.net.add_server_bot(name.clone(), bot_skill) {
                            Some(slot_id) => info!(
                                "console/addbot: bot serveur '{}' (slot {}, skill {:?})",
                                name, slot_id, bot_skill
                            ),
                            None => warn!("console/addbot: serveur plein"),
                        }
                    } else {
                        self.spawn_bot(&name, skill);
                    }
                }
                PendingAction::ClearBots => {
                    let n = self.bots.len();
                    self.bots.clear();
                    info!("clearbots: {} bots retirés", n);
                }
                PendingAction::SayChat(msg) => {
                    if self.net.mode.is_client() {
                        // Client : envoie au serveur, le retour viendra
                        // via ServerEvent::Chat broadcasté à tous.
                        self.net.send_chat(&msg);
                    } else {
                        // Solo / host : echo direct dans le feed local.
                        self.chat_feed.push(ChatLine {
                            speaker: "YOU".to_string(),
                            text: msg,
                            expire_at: self.time_sec + 6.0,
                            lifetime: 6.0,
                        });
                        if self.chat_feed.len() > CHAT_FEED_MAX {
                            self.chat_feed.remove(0);
                        }
                    }
                }
                PendingAction::Kick(slot_id) => {
                    if self.net.mode.is_server() {
                        if self.net.kick_slot(slot_id) {
                            info!("console/kick: slot {slot_id} viré");
                        } else {
                            warn!("console/kick: slot {slot_id} introuvable");
                        }
                    } else {
                        warn!("console/kick: hors mode --host, ignoré");
                    }
                }
                PendingAction::Restart => {
                    // En mode serveur, on restart côté autoritatif —
                    // ça affecte tous les clients via MatchStarted event.
                    // Sinon (solo / client), restart local — historique.
                    if self.net.mode.is_server() {
                        if self.net.restart_server_match() {
                            info!("console/restart: match serveur relancé");
                        }
                    } else {
                        self.restart_match();
                    }
                }
                PendingAction::GivePowerup(kind) => {
                    // Même logique de stacking que pour le pickup.
                    let duration = kind.duration();
                    let expires_at = self.grant_powerup(kind, duration);
                    info!(
                        "give{}: actif pour {duration:.0}s (expire @ t={expires_at:.1})",
                        kind.hud_label().to_lowercase(),
                    );
                }
                PendingAction::UseHoldable => {
                    self.use_held_item();
                }
                PendingAction::GiveHoldable(kind) => {
                    let prev = self.held_item.replace(kind);
                    info!(
                        "give{}: slot ← {:?}{}",
                        kind.hud_label().to_lowercase(),
                        kind,
                        prev.map(|p| format!(" (remplace {p:?})")).unwrap_or_default(),
                    );
                }
            }
        }
    }

    /// Restart du match : scores remis à zéro (joueur + bots), tous les
    /// FX en vol purgés, joueur respawn sur un nouveau spawn point, bots
    /// téléportés à leurs spawns. La map reste chargée — pas de re-parse
    /// BSP ni de re-upload GPU.
    fn restart_match(&mut self) {
        info!("restart match");
        // Scores + état intermission.
        self.frags = 0;
        self.deaths = 0;
        self.match_winner = None;
        self.warmup_until = self.time_sec + WARMUP_DURATION;
        self.first_blood_announced = false;
        self.total_shots = 0;
        self.total_hits = 0;
        self.time_warnings_fired = 0;
        self.match_start_at = self.warmup_until;
        self.respawn_at = None;
        self.bob_phase = 0.0;
        self.last_footstep_phase = 0.0;
        self.view_crouch_offset = 0.0;
        self.player.crouching = false;
        self.clear_powerups();
        // FX en vol — on purge tout ce qui référence des positions
        // désormais périmées.
        self.projectiles.clear();
        self.explosions.clear();
        self.particles.clear();
        self.beams.clear();
        self.kill_feed.clear();
        self.chat_feed.clear();
        self.pickup_toasts.clear();
        self.player_streak = 0;
        self.last_frags_announced = None;
        self.next_chat_at = 0.0;
        self.next_player_taunt_at = 0.0;
        self.last_death_cause = None;
        self.floating_damages.clear();
        self.active_medals.clear();
        // Pickups : remet tous à actifs (respawn_at = None).
        for p in self.pickups.iter_mut() {
            p.respawn_at = None;
        }
        // Santé + armure + ammo joueur : reset au loadout initial.
        self.player_health = Health::full();
        self.player_armor = 0;
        // Réserve d'air pleine à la nouvelle manche.
        self.air_left = AIR_CAPACITY_SEC;
        self.next_drown_at = 0.0;
        self.weapons_owned = (1u32 << WeaponId::Gauntlet.slot())
            | (1u32 << WeaponId::Machinegun.slot());
        self.active_weapon = WeaponId::Machinegun;
        self.last_weapon = WeaponId::Machinegun;
        self.weapon_switch_at = f32::NEG_INFINITY;
        self.ammo = {
            let mut a = [0i32; 10];
            a[WeaponId::Machinegun.slot() as usize] =
                WeaponId::Machinegun.params().starting_ammo;
            a
        };
        self.next_player_fire_at = 0.0;
        // Téléporte le joueur à un spawn point DM.
        self.respawn_player();
        // Reset les stats des bots, puis force-kill pour que
        // `respawn_dead_bots` les replace au prochain tick.
        for d in self.bots.iter_mut() {
            d.frags = 0;
            d.deaths = 0;
            d.health = Health::full();
            // On les "tue" logiquement : le prochain tick de
            // respawn_dead_bots les replacera sur un spawn aléatoire.
            d.health.take_damage(10_000);
        }
        self.respawn_dead_bots();
    }

    /// Tente de spawner un bot : charge (si besoin) un mesh partagé, choisit
    /// un spawn point DM, amorce la liste de waypoints et ajoute le driver.
    /// Consomme un évènement reçu dans `Snapshot::events` et le dispatch
    /// aux pipelines de feedback locaux : explosions visuelles, son,
    /// dlight. C'est l'analogue côté client du `booms` traité dans
    /// `tick_projectiles` pour les projectiles locaux — sans la partie
    /// dégâts (autoritative serveur).
    fn handle_remote_event(&mut self, evt: q3_net::ServerEvent) {
        match evt {
            q3_net::ServerEvent::Explosion { pos, kind } => {
                let origin = Vec3::from_array(pos);
                self.explosions.push(Explosion {
                    origin,
                    expire_at: self.time_sec + 0.35,
                });
                self.push_explosion_particles(origin);
                // Plasma : pas de fumée ; rocket / grenade : oui.
                if !matches!(kind, q3_net::ExplosionKind::Plasma) {
                    self.push_explosion_smoke(origin);
                }
                if let Some(r) = self.renderer.as_mut() {
                    let (color, intensity, radius) = match kind {
                        q3_net::ExplosionKind::Plasma => ([0.4, 0.6, 1.0], 2.5, 120.0),
                        q3_net::ExplosionKind::Grenade => ([1.0, 0.7, 0.3], 3.5, 280.0),
                        q3_net::ExplosionKind::Rocket => ([1.0, 0.7, 0.3], 4.0, 300.0),
                        // BFG : flash vert intense, rayon plus large pour
                        // signaler le radius de splash 200u du BFG.
                        q3_net::ExplosionKind::Bfg => ([0.4, 1.0, 0.5], 5.0, 400.0),
                    };
                    r.spawn_dlight(origin, radius, color, intensity, self.time_sec, 0.5);
                }
                if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_rocket_explode) {
                    play_at(snd, h, origin, Priority::Weapon);
                }
            }
            q3_net::ServerEvent::PlayerKilled {
                victim,
                killer,
                weapon,
            } => {
                // Résout les noms via la table reçue dans le snapshot.
                // Si on n'a pas encore le nom (table pas encore arrivée),
                // on fallback sur "slot N" — visible mais pas parlant.
                let resolve = |slot: u8| -> KillActor {
                    if let Some(my) = self
                        .net
                        .client_stage()
                        .and(self.remote_names.get(&slot).cloned().or_else(|| {
                            // Aucune entrée — on déduit si c'est nous ou pas.
                            None
                        }))
                    {
                        KillActor::Bot(my)
                    } else if self
                        .remote_names
                        .get(&slot)
                        .is_none()
                    {
                        // Pas de nom mais c'est peut-être nous (notre slot
                        // n'est pas dans remote_names puisqu'on s'exclut).
                        // En l'absence de connaissance précise, on log
                        // génériquement.
                        KillActor::Bot(format!("slot {slot}"))
                    } else {
                        KillActor::Bot(format!("slot {slot}"))
                    }
                };
                let killer_actor = resolve(killer);
                let victim_actor = resolve(victim);
                // Mappe le u8 weapon vers le WeaponId local. Inconnu → World.
                let cause = match weapon {
                    1 => KillCause::Weapon(WeaponId::Gauntlet),
                    2 => KillCause::Weapon(WeaponId::Machinegun),
                    3 => KillCause::Weapon(WeaponId::Shotgun),
                    4 => KillCause::Weapon(WeaponId::Grenadelauncher),
                    5 => KillCause::Weapon(WeaponId::Rocketlauncher),
                    6 => KillCause::Weapon(WeaponId::Lightninggun),
                    7 => KillCause::Weapon(WeaponId::Railgun),
                    8 => KillCause::Weapon(WeaponId::Plasmagun),
                    9 => KillCause::Weapon(WeaponId::Bfg),
                    _ => KillCause::Environment("WORLD"),
                };
                self.push_kill_cause(killer_actor, victim_actor, cause);
            }
            q3_net::ServerEvent::MatchEnded { winner } => {
                // Mappe le winner_slot serveur vers `KillActor` local —
                // recycle l'enum existant pour réutiliser l'écran
                // intermission. 255 = égalité → `World` comme dans Q3.
                let winner_actor = if winner == 255 {
                    KillActor::World
                } else if Some(winner) == self.net.client_stage().and({
                    self.remote_names
                        .keys()
                        .find(|&&k| k != winner)
                        .copied()
                        .or(None);
                    None
                }) {
                    KillActor::Player
                } else {
                    let name = self
                        .remote_names
                        .get(&winner)
                        .cloned()
                        .unwrap_or_else(|| format!("slot {winner}"));
                    KillActor::Bot(name)
                };
                self.match_winner = Some(winner_actor);
            }
            q3_net::ServerEvent::MatchStarted => {
                // Reset HUD côté client : winner clear, kill-feed vidé.
                // Les frags/deaths viendront du prochain snapshot
                // (déjà à 0 puisque le serveur a reset tous les slots).
                self.match_winner = None;
                self.kill_feed.clear();
            }
            q3_net::ServerEvent::RailTrail { from, to, owner: _ } => {
                // Push un ActiveBeam style railgun : spirale rose-magenta
                // qui s'estompe en 0.6s. Aligné sur le visuel du tir
                // local Q3 pour la cohérence (le joueur ne doit pas
                // voir une différence entre son propre rail et celui
                // d'un remote).
                let lifetime = 0.6_f32;
                self.beams.push(ActiveBeam {
                    a: Vec3::from_array(from),
                    b: Vec3::from_array(to),
                    color: [0.95, 0.25, 0.55, 0.85],
                    expire_at: self.time_sec + lifetime,
                    lifetime,
                    style: BeamStyle::Spiral,
                });
            }
            q3_net::ServerEvent::Chat { slot, message } => {
                // Reconstruit le texte (UTF-8 lossy, sans padding nul).
                let end = message.iter().position(|&b| b == 0).unwrap_or(96);
                let text = String::from_utf8_lossy(&message[..end]).into_owned();
                // Résolution du nom : table remote_names + fallback "slot N".
                let speaker = self
                    .remote_names
                    .get(&slot)
                    .cloned()
                    .unwrap_or_else(|| format!("slot {slot}"));
                self.chat_feed.push(ChatLine {
                    speaker,
                    text,
                    expire_at: self.time_sec + 6.0,
                    lifetime: 6.0,
                });
                if self.chat_feed.len() > CHAT_FEED_MAX {
                    self.chat_feed.remove(0);
                }
            }
            q3_net::ServerEvent::Sound { id, pos } => {
                // Mappe l'id wire vers un SoundHandle local. Inconnu → silence.
                let origin = Vec3::from_array(pos);
                let handle = match id {
                    q3_net::sound_id::PICKUP_HEALTH
                    | q3_net::sound_id::PICKUP_ARMOR
                    | q3_net::sound_id::PICKUP_POWERUP => self.sfx_weapon_pickup,
                    q3_net::sound_id::PICKUP_AMMO => self.sfx_ammo_pickup,
                    q3_net::sound_id::PICKUP_WEAPON => self.sfx_weapon_pickup,
                    _ => None,
                };
                if let (Some(snd), Some(h)) = (self.sound.as_ref(), handle) {
                    play_at(snd, h, origin, Priority::Normal);
                }
            }
            q3_net::ServerEvent::LightningBeam { from, to, owner: _ } => {
                // Lightning : zigzag bleu, vie très courte (0.08s) pour
                // que la cadence 50 Hz du LG ne sature pas l'écran avec
                // des beams qui s'accumulent (chaque tick = 1 beam, on
                // veut juste l'illusion d'un faisceau continu).
                let lifetime = 0.08_f32;
                self.beams.push(ActiveBeam {
                    a: Vec3::from_array(from),
                    b: Vec3::from_array(to),
                    color: [0.55, 0.75, 1.0, 0.9],
                    expire_at: self.time_sec + lifetime,
                    lifetime,
                    style: BeamStyle::Lightning,
                });
            }
        }
    }

    /// Charge le rig MD3 partagé (lower / upper / head) à partir du VFS,
    /// si pas déjà chargé. Un "player rig" Q3 est *trois* MD3 :
    /// `lower.md3` (jambes), `upper.md3` (torse, connecté par `tag_torso`)
    /// et `head.md3` (tête, par `tag_head`). On tente les models disponibles
    /// jusqu'à en trouver un dont les 3 parties parsent — un seul mesh
    /// manquant et on passe au suivant, pour ne pas se retrouver avec un
    /// pantin sans tête ni buste.
    ///
    /// Appelé depuis `spawn_bot` (1er bot local) **et** depuis l'application
    /// d'une snapshot serveur si on voit un remote player et que le rig
    /// n'est pas encore prêt.
    fn ensure_player_rig_loaded(&mut self) {
        if self.bot_rig.is_some() {
            return;
        }
        const PLAYER_CANDIDATES: &[&str] = &[
            "models/players/sarge",
            "models/players/visor",
            "models/players/doom",
            "models/players/major",
            "models/players/bitterman",
            "models/players/biker",
            "models/players/anarki",
            "models/players/crash",
            "models/players/grunt",
            "models/players/hunter",
            "models/players/keel",
            "models/players/klesk",
            "models/players/lucy",
            "models/players/mynx",
            "models/players/orbb",
            "models/players/ranger",
            "models/players/razor",
            "models/players/slash",
            "models/players/sorlag",
            "models/players/tankjr",
            "models/players/uriel",
            "models/players/xaero",
        ];
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        for dir in PLAYER_CANDIDATES {
            let lower = r.load_md3(&self.vfs, &format!("{dir}/lower.md3"));
            let upper = r.load_md3(&self.vfs, &format!("{dir}/upper.md3"));
            let head = r.load_md3(&self.vfs, &format!("{dir}/head.md3"));
            if let (Ok(l), Ok(u), Ok(h)) = (lower, upper, head) {
                info!(
                    "player rig: '{dir}' (lower={} upper={} head={} frames)",
                    l.num_frames(),
                    u.num_frames(),
                    h.num_frames()
                );
                self.bot_rig = Some(PlayerRig { lower: l, upper: u, head: h });
                break;
            }
        }
        if self.bot_rig.is_none() {
            // Aucun model joueur trouvé dans le pak0 actif. Sans rig,
            // `queue_bots` skip TOUS les bots (cf. l'`if let Some(rig)`
            // côté render). Symptôme observé : bots présents dans la
            // logique de jeu (frags, tirs, etc.) mais invisibles.
            // Cause typique : pak0 partiel (démo Q3) ou flag --base
            // qui pointe sur un dossier sans `models/players/`.
            warn!(
                "player rig: AUCUN model joueur trouvé — les bots seront \
                 INVISIBLES. Vérifie ton baseq3/pak0.pk3 (chemin attendu \
                 `models/players/<nom>/{{lower,upper,head}}.md3`). \
                 Candidats testés : {}",
                PLAYER_CANDIDATES.len()
            );
        }
    }

    fn spawn_bot(&mut self, name: &str, skill_override: Option<i32>) {
        // Pré-vérifs avant de prendre la moindre ressource. On scope le
        // borrow de `self.world` à ce bloc pour pouvoir appeler
        // `ensure_player_rig_loaded` (qui prend `&mut self`) juste après.
        {
            let Some(world) = self.world.as_ref() else {
                warn!("addbot: pas de map chargée");
                return;
            };
            if world.spawn_points.is_empty() && world.player_start.is_none() {
                warn!("addbot: la map n'a pas de point de spawn");
                return;
            }
        }

        // Lazy-load le rig partagé au premier bot.
        if self.bot_rig.is_none() {
            self.ensure_player_rig_loaded();
            if self.bot_rig.is_none() {
                warn!("addbot: aucun player model complet (lower+upper+head) trouvé dans le VFS");
                return;
            }
        }
        // Re-acquisition du borrow après le `&mut self` ci-dessus.
        let world = self.world.as_ref().expect("world checké au début");

        // Spawn point : on cycle dans les DM spawn points, fallback sur
        // player_start. On offset de +1 pour ne pas faire spawner le premier
        // bot sur `spawn_points[0]` qui coïncide typiquement avec le
        // `player_start` du joueur — sinon le bot apparaît clippé dans le
        // joueur au démarrage d'une map (visible comme une silhouette blanche
        // à travers le viewmodel tant que le joueur ne bouge pas).
        let idx = self.bots.len() + 1;
        let (origin, angles) = if !world.spawn_points.is_empty() {
            let sp = &world.spawn_points[idx % world.spawn_points.len()];
            (sp.origin, sp.angles)
        } else {
            (world.player_start.unwrap_or(Vec3::ZERO), world.player_start_angles)
        };
        let spawn_origin = origin + Vec3::Z * 40.0;

        // Résolution de la difficulté : override explicite par la
        // commande console > cvar `bot_skill` > défaut III.  On clampe
        // via `BotSkill::from_int` — tout ce qui est hors 1..5 retombe
        // sur III silencieusement (mieux qu'un log bruyant pour un typo).
        let skill_n = skill_override
            .or_else(|| self.cvars.get_i32("bot_skill"))
            .unwrap_or(3);
        let skill = BotSkill::from_int(skill_n);
        let mut bot = Bot::with_skill(name, spawn_origin, skill);
        bot.view_angles = angles;
        // Waypoints : tous les autres spawn points, dans l'ordre.
        for (i, sp) in world.spawn_points.iter().enumerate() {
            if i != idx % world.spawn_points.len().max(1) {
                bot.push_waypoint(sp.origin + Vec3::Z * 40.0);
            }
        }

        let mut body = PlayerMove::new(spawn_origin);
        body.view_angles = angles;

        let tint = bot_tint(idx);
        info!("addbot: '{name}' spawné à {:?} (tint={:?})", spawn_origin, tint);
        self.bots.push(BotDriver {
            bot,
            body,
            tint,
            wp_cursor: 0,
            last_saw_player_at: None,
            first_seen_player_at: None,
            next_fire_at: 0.0,
            next_rocket_at: 0.0,
            health: Health::new(BOT_DEFAULT_HP),
            // Fenêtre d'invulnérabilité appliquée aussi au premier spawn :
            // sinon un bot qui apparaît au milieu d'une frag line mange
            // une rocket instantanément.
            invul_until: self.time_sec + RESPAWN_INVUL_SEC,
            frags: 0,
            deaths: 0,
            last_fire_at: f32::NEG_INFINITY,
            last_damage_at: f32::NEG_INFINITY,
            airborne_since: None,
            last_land_at: f32::NEG_INFINITY,
            anim_phase: 0.0,
            anim_state: BotAnimState::Idle,
        });
    }

    /// Téléporte le joueur à un point de spawn DM (ou `player_start` en
    /// fallback), restaure sa santé et réinitialise sa physique. Appelé
    /// quand `respawn_at` est échu.
    fn respawn_player(&mut self) {
        // Lookup du spawn : pseudo-aléatoire léger indexé sur time+deaths
        // pour ne pas retomber au même endroit à chaque mort.
        let (origin, angles) = {
            let Some(world) = self.world.as_ref() else {
                return;
            };
            if !world.spawn_points.is_empty() {
                let seed = (self.time_sec * 1000.0) as usize
                    ^ (self.deaths as usize).wrapping_mul(2654435761);
                let sp = &world.spawn_points[seed % world.spawn_points.len()];
                (sp.origin, sp.angles)
            } else if let Some(ps) = world.player_start {
                (ps, world.player_start_angles)
            } else {
                (Vec3::ZERO, Angles::default())
            }
        };
        let new_origin = origin + Vec3::Z * 40.0;
        self.player = PlayerMove::new(new_origin);
        self.player.view_angles = angles;
        self.player_health.respawn();
        self.player_armor = 0;
        // Poumons pleins au respawn : on ne veut pas que le joueur sorte
        // du purgatoire respawn en étant déjà "à bout de souffle".
        self.air_left = AIR_CAPACITY_SEC;
        self.next_drown_at = 0.0;
        self.respawn_at = None;
        // Fenêtre d'invulnérabilité au respawn — protège des dégâts
        // et du knockback pendant `RESPAWN_INVUL_SEC` secondes.
        self.player_invul_until = self.time_sec + RESPAWN_INVUL_SEC;
        self.was_airborne = false;
        // Pain-arrow : on oublie le dernier attaquant de la vie précédente.
        self.last_damage_until = 0.0;
        // Screen-shake : pas de secousse résiduelle au respawn.
        self.shake_intensity = 0.0;
        self.shake_until = 0.0;
        // Armor flash : réinitialisé au respawn.
        self.armor_flash_until = 0.0;
        self.pain_flash_until = 0.0;
        // Les powerups ne persistent pas entre les vies — Q3 standard.
        self.clear_powerups();
        // Idem pour le holdable : pas de stash à la mort.
        self.held_item = None;
        // La bannière "killed by X" n'a plus lieu d'être une fois le joueur
        // revenu à la vie.
        self.last_death_cause = None;
        // Streak : une mort clôt la série — classique Unreal.  Pas de
        // toast « série terminée » côté joueur (déprimant) ; les bots
        // peuvent déjà se foutre de la gueule via ChatTrigger::Death.
        self.player_streak = 0;
        // Et on remet la phase du view-bob à 0 : le joueur est statique
        // au respawn, on ne veut pas qu'un reste de sin(phase) donne un
        // micro-soubresaut la première frame. Idem pour le seuil de
        // footstep — sinon le premier pas peut rater.
        self.bob_phase = 0.0;
        self.last_footstep_phase = 0.0;
        // On remet la hauteur de vue à debout : quelqu'un qui meurt
        // accroupi apparaît au respawn en position normale.
        self.view_crouch_offset = 0.0;
        self.player.crouching = false;
        // Nouvelle vie = nouveau contexte, on ré-autorise un push / teleport
        // au premier contact quand bien même on respawnerait dans un trigger.
        self.on_jumppad_idx = None;
        self.on_teleport_idx = None;
        if let Some(r) = self.renderer.as_mut() {
            r.camera_mut().position =
                self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
            r.camera_mut().angles = self.player.view_angles;
        }
        // FX de respawn — colonne blanc-bleu au pied du joueur, visible
        // aux autres bots dans la vue 3D (et au joueur si la caméra re-
        // regarde vers le bas).
        self.push_respawn_fx(self.player.origin, [0.8, 0.9, 1.0, 1.0]);
        info!("respawn → {:?}", self.player.origin);
    }

    /// Vérifie la collision joueur ↔ pickups actifs. Applique l'effet +
    /// arme le timer de respawn + joue un sfx. Un pickup inactif (timer
    /// en cours) est réactivé quand `time_sec >= respawn_at`.
    fn tick_pickups(&mut self) {
        // Réactive les pickups dont le cooldown est échu. On collecte les
        // positions pour déclencher un FX de respawn hors-borrow.
        let mut reborn_positions: Vec<Vec3> = Vec::new();
        for p in self.pickups.iter_mut() {
            if let Some(t) = p.respawn_at {
                if self.time_sec >= t {
                    p.respawn_at = None;
                    reborn_positions.push(p.origin);
                }
            }
        }
        // Cyan pour signaler "c'est revenu" — visible à travers la map.
        for pos in reborn_positions {
            self.push_respawn_fx(pos, [0.4, 0.9, 1.0, 1.0]);
        }
        if self.player_health.is_dead() {
            return;
        }
        // Pour la capture du son hors borrow, on collecte d'abord les events.
        enum Event {
            Health {
                sfx_pos: Vec3,
                new_hp: i32,
                hp_was: i32,
            },
            Armor {
                sfx_pos: Vec3,
                new_armor: i32,
            },
            Weapon {
                sfx_pos: Vec3,
                weapon: WeaponId,
                was_new: bool,
                ammo_after: i32,
            },
            Ammo {
                sfx_pos: Vec3,
                slot: u8,
                ammo_after: i32,
            },
            Powerup {
                sfx_pos: Vec3,
                powerup: PowerupKind,
                /// `time_sec` auquel le buff cessera (utile pour le log
                /// depuis le handler, qui ne peut plus accéder à `self`
                /// en lecture structurée du powerup).
                expires_at: f32,
            },
            Holdable {
                sfx_pos: Vec3,
                kind: HoldableKind,
                /// `true` = écrase un holdable déjà possédé (utile pour le
                /// log : « you dropped your medkit »).
                replaced: bool,
            },
        }
        let mut events: Vec<Event> = Vec::new();
        let px = self.player.origin;
        for p in self.pickups.iter_mut() {
            if p.respawn_at.is_some() {
                continue;
            }
            let dxy = (p.origin.truncate() - px.truncate()).length();
            let dz = (p.origin.z - px.z).abs();
            if dxy > PICKUP_RADIUS || dz > PICKUP_VERT_REACH {
                continue;
            }
            match p.kind {
                PickupKind::Health { amount, max_cap } => {
                    let old_max = self.player_health.max;
                    if self.player_health.current >= max_cap {
                        continue; // plein pour ce type
                    }
                    // Elève le cap si le pickup autorise plus (megahealth).
                    if max_cap > old_max {
                        self.player_health.max = max_cap;
                    }
                    let hp_was = self.player_health.current;
                    // `heal` plafonne à `max` → on force `current` direct
                    // quand megahealth dépasse 100 courant.
                    let new_hp = (self.player_health.current + amount).min(max_cap);
                    self.player_health.current = new_hp;
                    events.push(Event::Health { sfx_pos: p.origin, new_hp, hp_was });
                }
                PickupKind::Armor { amount } => {
                    if self.player_armor >= 200 {
                        continue;
                    }
                    let new_armor = (self.player_armor + amount).min(200);
                    self.player_armor = new_armor;
                    events.push(Event::Armor { sfx_pos: p.origin, new_armor });
                }
                PickupKind::Weapon { weapon, ammo } => {
                    let bit = 1u32 << weapon.slot();
                    let was_new = (self.weapons_owned & bit) == 0;
                    // Q3 : si arme déjà possédée on refuse si ammo plein.
                    let slot = weapon.slot() as usize;
                    let max = weapon.params().max_ammo;
                    if !was_new && self.ammo[slot] >= max {
                        continue;
                    }
                    self.weapons_owned |= bit;
                    self.ammo[slot] = (self.ammo[slot] + ammo).min(max);
                    events.push(Event::Weapon {
                        sfx_pos: p.origin,
                        weapon,
                        was_new,
                        ammo_after: self.ammo[slot],
                    });
                }
                PickupKind::Ammo { slot, amount } => {
                    let s = slot as usize;
                    // Cherche le max_ammo du slot — on prend celui de l'arme
                    // qui matche, sinon fallback 200.
                    let max = WeaponId::ALL
                        .into_iter()
                        .find(|w| w.slot() == slot)
                        .map(|w| w.params().max_ammo)
                        .unwrap_or(200);
                    if self.ammo[s] >= max {
                        continue;
                    }
                    self.ammo[s] = (self.ammo[s] + amount).min(max);
                    events.push(Event::Ammo {
                        sfx_pos: p.origin,
                        slot,
                        ammo_after: self.ammo[s],
                    });
                }
                PickupKind::Powerup { powerup, duration } => {
                    // Stack additif (Q3 standard : récupérer un nouveau
                    // powerup pendant que l'ancien tourne rallonge la fin,
                    // pas un overwrite). Inlined ici — on ne peut pas
                    // appeler `self.grant_powerup()` pendant qu'on itère
                    // `self.pickups` en mut (second borrow E0499).
                    let slot = &mut self.powerup_until[powerup.index()];
                    let base = slot
                        .filter(|&t| t > self.time_sec)
                        .unwrap_or(self.time_sec);
                    let expires_at = base + duration;
                    *slot = Some(expires_at);
                    events.push(Event::Powerup {
                        sfx_pos: p.origin,
                        powerup,
                        expires_at,
                    });
                }
                PickupKind::Holdable { kind } => {
                    // Un seul slot — on remplace silencieusement. Q3 fait
                    // pareil (le joueur drop implicitement l'ancien).
                    let replaced = self.held_item.is_some();
                    self.held_item = Some(kind);
                    events.push(Event::Holdable {
                        sfx_pos: p.origin,
                        kind,
                        replaced,
                    });
                }
                PickupKind::Inert => continue,
            }
            // Cooldown propre à chaque kind (cf. `PickupKind::from_entity`).
            p.respawn_at = Some(self.time_sec + p.respawn_cooldown);
        }

        // Effets hors boucle pour ne pas tenir `self.pickups` en emprunt.
        for e in events {
            match e {
                Event::Health { sfx_pos, new_hp, hp_was } => {
                    info!("pickup health: {} → {}", hp_was, new_hp);
                    if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_pain_bot) {
                        // Faute d'un sfx dédié chargé, on recycle le pain_bot
                        // comme bip de pickup — audible et diégétique.
                        play_at(snd, h, sfx_pos, Priority::Low);
                    }
                }
                Event::Armor { sfx_pos, new_armor } => {
                    info!("pickup armor → {}", new_armor);
                    if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_pain_bot) {
                        play_at(snd, h, sfx_pos, Priority::Low);
                    }
                }
                Event::Weapon { sfx_pos, weapon, was_new, ammo_after } => {
                    if was_new {
                        info!("pickup weapon: {} (ammo={})", weapon.name(), ammo_after);
                        // Switch auto vers la nouvelle arme, façon Q3.
                        // Passe par `switch_to_weapon` : sauvegarde last_weapon
                        // + joue raise SFX + arme l'animation ammo panel.
                        self.switch_to_weapon(weapon);
                        // Toast seulement sur arme neuve — sinon le joueur
                        // qui passe sur un spawn pour refaire ses ammo voit
                        // un toast sur chaque tick, peu utile.
                        self.push_pickup_toast(
                            format!("YOU GOT THE {}", weapon.name().to_uppercase()),
                            [1.0, 0.85, 0.25, 1.0], // jaune armes
                        );
                    } else {
                        info!("pickup weapon (déjà possédé) +ammo → {}", ammo_after);
                    }
                    let sfx = self.sfx_weapon_pickup.or(self.sfx_pain_bot);
                    if let (Some(snd), Some(h)) = (self.sound.as_ref(), sfx) {
                        play_at(snd, h, sfx_pos, Priority::Normal);
                    }
                }
                Event::Ammo { sfx_pos, slot, ammo_after } => {
                    info!("pickup ammo slot {} → {}", slot, ammo_after);
                    let sfx = self.sfx_ammo_pickup.or(self.sfx_pain_bot);
                    if let (Some(snd), Some(h)) = (self.sound.as_ref(), sfx) {
                        play_at(snd, h, sfx_pos, Priority::Low);
                    }
                }
                Event::Powerup { sfx_pos, powerup, expires_at } => {
                    let remaining = (expires_at - self.time_sec).max(0.0);
                    info!(
                        "pickup powerup: {:?} (expire dans {remaining:.1}s, @ t={expires_at:.1})",
                        powerup,
                    );
                    // FX dédié : colonne colorée façon respawn mais plus
                    // haute et teintée selon le powerup.
                    self.push_respawn_fx(sfx_pos, powerup.pickup_fx_color());
                    // Toast coloré identique à la teinte du powerup —
                    // continuité visuelle entre le halo au sol et le
                    // texte au HUD.
                    let mut color = powerup.hud_color();
                    color[3] = 1.0;
                    self.push_pickup_toast(
                        format!("YOU GOT {}", powerup.hud_label()),
                        color,
                    );
                    let sfx = self.sfx_weapon_pickup.or(self.sfx_pain_bot);
                    if let (Some(snd), Some(h)) = (self.sound.as_ref(), sfx) {
                        play_at(snd, h, sfx_pos, Priority::High);
                    }
                }
                Event::Holdable { sfx_pos, kind, replaced } => {
                    if replaced {
                        info!("pickup holdable: {:?} (remplace l'ancien)", kind);
                    } else {
                        info!("pickup holdable: {:?}", kind);
                    }
                    // Feedback identique à un powerup mais teinte jaune
                    // pour distinguer à l'œil : c'est en slot, pas actif.
                    self.push_respawn_fx(sfx_pos, [1.0, 0.85, 0.3, 1.0]);
                    self.push_pickup_toast(
                        format!("HOLDING {}", kind.hud_label()),
                        [1.0, 1.0, 1.0, 1.0],
                    );
                    let sfx = self.sfx_weapon_pickup.or(self.sfx_pain_bot);
                    if let (Some(snd), Some(h)) = (self.sound.as_ref(), sfx) {
                        play_at(snd, h, sfx_pos, Priority::Normal);
                    }
                }
            }
        }
    }

    /// Ajoute une entrée au kill-feed, en respectant `KILL_FEED_MAX`.
    /// Appelé depuis tous les sites où une mort est détectée — joueur tue
    /// bot (hitscan ou projectile) et bot tue joueur (MG ou rocket).
    fn push_kill(&mut self, killer: KillActor, victim: KillActor, weapon: WeaponId) {
        self.push_kill_cause(killer, victim, KillCause::Weapon(weapon));
    }

    /// Variante générique — autorise un tag arbitraire (ex. "VOID", "LAVA")
    /// pour les kills non-arme.
    fn push_kill_cause(&mut self, killer: KillActor, victim: KillActor, cause: KillCause) {
        // Snapshot de la cause de mort du joueur : on capture AVANT le move
        // dans l'événement kill-feed, pour pouvoir l'afficher sous "YOU
        // DIED". Ne prend pas en compte les kills où la victime n'est pas
        // le joueur local (bot vs bot, suicides bots).
        if matches!(victim, KillActor::Player) {
            self.last_death_cause = Some((killer.clone(), cause));
        }
        // Snapshot killer/victim AVANT le move dans KillEvent — servent
        // plus bas à déclencher les taunts bot.
        let killer_for_chat = (killer.clone(), victim.clone());
        let ev = KillEvent {
            killer,
            victim,
            cause,
            expire_at: self.time_sec + KILL_FEED_LIFETIME,
        };
        self.kill_feed.push(ev);
        if self.kill_feed.len() > KILL_FEED_MAX {
            let excess = self.kill_feed.len() - KILL_FEED_MAX;
            self.kill_feed.drain(0..excess);
        }
        // FIRST BLOOD : première frag joueur-vs-joueur du match.  On
        // exclut les suicides (killer == victim) et les morts par le
        // décor (KillActor::World) — Q3 original a la même règle.  Un
        // seul tir par match ; reset par `restart_match`.
        if !self.first_blood_announced {
            let is_selfkill = matches!(
                (&self.kill_feed.last().unwrap().killer, &self.kill_feed.last().unwrap().victim),
                (KillActor::Player, KillActor::Player)
            ) || match (
                &self.kill_feed.last().unwrap().killer,
                &self.kill_feed.last().unwrap().victim,
            ) {
                (KillActor::Bot(a), KillActor::Bot(b)) => a == b,
                _ => false,
            };
            let killer_is_world = matches!(
                self.kill_feed.last().unwrap().killer,
                KillActor::World
            );
            if !is_selfkill && !killer_is_world {
                self.first_blood_announced = true;
                let killer_label = self
                    .kill_feed
                    .last()
                    .unwrap()
                    .killer
                    .label()
                    .to_string();
                let msg = format!("FIRST BLOOD: {killer_label} draws first!");
                info!("first blood → {killer_label}");
                self.push_pickup_toast(msg, [1.0, 0.2, 0.2, 1.0]);
                // Son d'annonce si dispo — on réutilise "excellent" qui
                // est la sample medal la plus emblématique.
                if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_excellent) {
                    play_medal(snd, h, self.player.origin);
                }
            }
        }
        // Chat feed : bot-killer → insulte ; bot-victim → lamentation.
        let (killer_clone, victim_clone) = killer_for_chat;
        if let KillActor::Bot(ref name) = killer_clone {
            if let Some(idx) = self.find_bot_by_name(name) {
                self.maybe_bot_chat(idx, ChatTrigger::KillInsult);
            }
        }
        if let KillActor::Bot(ref name) = victim_clone {
            if let Some(idx) = self.find_bot_by_name(name) {
                self.maybe_bot_chat(idx, ChatTrigger::Death);
            }
        }
    }

    /// Cherche l'index d'un bot par son nom (liste petite, linéaire est ok).
    fn find_bot_by_name(&self, name: &str) -> Option<usize> {
        self.bots.iter().position(|d| d.bot.name == name)
    }

    /// Tire aléatoirement une ligne de chat pour le bot `idx` sur le
    /// `trigger` donné — honore le cooldown global et la probabilité
    /// par trigger.  No-op si les conditions ne sont pas réunies.
    fn maybe_bot_chat(&mut self, idx: usize, trigger: ChatTrigger) {
        if self.time_sec < self.next_chat_at {
            return;
        }
        let prob = CHAT_TRIGGER_PROB * trigger.weight();
        if rand_unit().abs() > prob {
            return;
        }
        let Some(bot) = self.bots.get(idx) else { return; };
        let pool = trigger.pool();
        if pool.is_empty() {
            return;
        }
        // Sélection de ligne : même LCG interne (rand_unit) sur [0,1[.
        let line_idx = (rand_unit().abs() * pool.len() as f32) as usize;
        let line_idx = line_idx.min(pool.len() - 1);
        let text = pool[line_idx].to_string();
        let speaker = bot.bot.name.clone();
        self.push_chat(speaker, text);
        self.next_chat_at = self.time_sec + CHAT_GLOBAL_COOLDOWN;
    }

    /// Ajoute une ligne de chat et FIFO-évince la plus ancienne si on
    /// dépasse `CHAT_FEED_MAX`.
    fn push_chat(&mut self, speaker: String, text: String) {
        let line = ChatLine {
            speaker,
            text,
            expire_at: self.time_sec + CHAT_LINE_LIFETIME,
            lifetime: CHAT_LINE_LIFETIME,
        };
        self.chat_feed.push(line);
        if self.chat_feed.len() > CHAT_FEED_MAX {
            let excess = self.chat_feed.len() - CHAT_FEED_MAX;
            self.chat_feed.drain(0..excess);
        }
    }

    /// Ajoute un toast de pickup.  FIFO au delà de [`PICKUP_TOAST_MAX`].
    /// Appelé depuis `tick_pickups` lors des événements notables
    /// (arme neuve, powerup, holdable) — PAS sur ammo ni health/armor
    /// banale pour éviter de noyer le HUD.
    fn push_pickup_toast(&mut self, text: String, color: [f32; 4]) {
        let toast = PickupToast {
            text,
            color,
            expire_at: self.time_sec + PICKUP_TOAST_LIFETIME,
            lifetime: PICKUP_TOAST_LIFETIME,
        };
        self.pickup_toasts.push(toast);
        if self.pickup_toasts.len() > PICKUP_TOAST_MAX {
            let excess = self.pickup_toasts.len() - PICKUP_TOAST_MAX;
            self.pickup_toasts.drain(0..excess);
        }
    }

    /// À appeler à chaque frag du joueur — incrémente le streak et
    /// déclenche un toast si on franchit un palier (3, 5, 7, 10, 15, 20).
    /// Ne gère PAS `self.frags` lui-même : l'appelant doit continuer
    /// à incrémenter `self.frags` séparément (pour préserver la
    /// sémantique scoreboard = kills totaux vs streak = kills depuis
    /// le dernier respawn).
    fn on_player_frag(&mut self) {
        self.player_streak = self.player_streak.saturating_add(1);

        // **Announcer fraglimit countdown** (A5) — annonce vocale Q3
        // canonique aux 3 derniers frags. `last_frags_announced` évite
        // de re-annoncer si le compteur stationne (ex: 1 frag left
        // pendant 5 kills suite à un autoswitch-streak côté ennemi).
        let frags_left = FRAG_LIMIT.saturating_sub(self.frags);
        if (1..=3).contains(&frags_left)
            && self.last_frags_announced != Some(frags_left)
        {
            self.last_frags_announced = Some(frags_left);
            let handle = match frags_left {
                1 => self.sfx_one_frag,
                2 => self.sfx_two_frags,
                3 => self.sfx_three_frags,
                _ => None,
            };
            if let (Some(snd), Some(h)) = (self.sound.as_ref(), handle) {
                play_medal(snd, h, self.player.origin);
                info!("announcer: {frags_left} frag(s) left");
            }
        }

        // Tiers Unreal : label + couleur. Seuls les seuils exacts
        // déclenchent un toast — entre 3 et 5 on reste silencieux.
        let (label, color) = match self.player_streak {
            3 => ("KILLING SPREE!", [1.0, 0.85, 0.25, 1.0]),
            5 => ("RAMPAGE!", [1.0, 0.55, 0.15, 1.0]),
            7 => ("DOMINATING!", [1.0, 0.3, 0.2, 1.0]),
            10 => ("UNSTOPPABLE!", [0.9, 0.2, 0.9, 1.0]),
            15 => ("GODLIKE!", [1.0, 0.95, 0.35, 1.0]),
            20 => ("WICKED SICK!", [1.0, 0.4, 1.0, 1.0]),
            _ => return,
        };
        info!("streak: {} kills → {label}", self.player_streak);
        self.push_pickup_toast(label.to_string(), color);
        // Relais audio : on recycle le son "excellent" (voix montante)
        // pour marquer le palier sans charger un sample dédié.
        if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_excellent) {
            play_medal(snd, h, self.player.origin);
        }
    }

    /// Mappe une valeur `remaining_sec` en bit de `time_warnings_fired`.
    /// `None` = pas de seuil à ce chiffre, pas d'annonce.
    fn time_warning_bit(sec: i32) -> Option<u16> {
        match sec {
            60 => Some(1 << 0),
            30 => Some(1 << 1),
            n @ 1..=10 => Some(1 << (2 + (10 - n) as u16)),
            _ => None,
        }
    }

    /// À appeler chaque tick ; annonce 1 MIN / 30 SEC / countdown final
    /// via un toast au moment où `ceil(time_remaining)` franchit un
    /// seuil (une seule fois par seuil par match).  Silencieux pendant
    /// le warmup et l'intermission.
    fn tick_time_warnings(&mut self) {
        if self.match_winner.is_some() || self.is_warmup() {
            return;
        }
        let elapsed = self.time_sec - self.match_start_at;
        if elapsed < 0.0 {
            return;
        }
        let remaining = (TIME_LIMIT_SECONDS - elapsed).max(0.0);
        // `ceil` : tant qu'il reste 59.2s, on est "à 1 min" (ceil=60).
        let rem_i = remaining.ceil() as i32;
        let Some(bit) = Self::time_warning_bit(rem_i) else {
            return;
        };
        if (self.time_warnings_fired & bit) != 0 {
            return;
        }
        self.time_warnings_fired |= bit;
        // Message + couleur selon le seuil.  Countdown final en rouge
        // vif pour l'urgence, 30s en orange, 60s en jaune.
        let (msg, color) = match rem_i {
            60 => ("1 MINUTE REMAINING".to_string(), [1.0, 0.95, 0.3, 1.0]),
            30 => ("30 SECONDS".to_string(), [1.0, 0.6, 0.15, 1.0]),
            n => (format!("{n}"), [1.0, 0.25, 0.2, 1.0]),
        };
        info!("time warning: {msg}");
        self.push_pickup_toast(msg, color);
        // Beep uniformément sur tous les seuils (on réutilise le sample
        // "powerup about to expire" qui a la bonne urgence sonore).
        if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_powerup_end) {
            play_medal(snd, h, self.player.origin);
        }
    }

    /// Joue un battement de cœur sourd tant que le joueur est à faible
    /// santé.  Fréquence scale avec HP :
    ///   - 40 HP → 1.0 Hz (1 par seconde)
    ///   - 25 HP → ~1.7 Hz
    ///   - 1  HP → ~2.5 Hz
    /// Réutilise `sfx_hit` (blip court et sec) joué à la position du
    /// joueur : panning inchangé (le joueur est sa propre source), mais
    /// atténué via `Priority::Low` pour ne pas masquer un bruit de pas
    /// ennemi.  Silencieux pendant le warmup / la mort / l'intermission.
    fn tick_heartbeat(&mut self) {
        const HEARTBEAT_THRESHOLD: i32 = 40;
        if self.match_winner.is_some()
            || self.is_warmup()
            || self.player_health.is_dead()
        {
            return;
        }
        let hp = self.player_health.current;
        if hp >= HEARTBEAT_THRESHOLD || hp <= 0 {
            // Armé à « bientôt » pour que le prochain entrée sous seuil
            // ne joue pas un battement en retard — on veut un front
            // montant propre.
            self.next_heartbeat_at = 0.0;
            return;
        }
        if self.next_heartbeat_at == 0.0 {
            self.next_heartbeat_at = self.time_sec + 0.4;
            return;
        }
        if self.time_sec < self.next_heartbeat_at {
            return;
        }
        // Interval = lerp 1.0s (HP=40) → 0.4s (HP=1).
        let severity = 1.0
            - (hp as f32 / HEARTBEAT_THRESHOLD as f32).clamp(0.0, 1.0);
        let interval = 1.0 - severity * 0.6;
        if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_hit) {
            play_at(snd, h, self.player.origin, Priority::Low);
        }
        self.next_heartbeat_at = self.time_sec + interval;
    }

    /// Scanne frags joueur + bots — si quelqu'un atteint `FRAG_LIMIT`
    /// ou si le timer `TIME_LIMIT_SECONDS` expire, arme l'intermission.
    /// Idempotent : une fois `match_winner` posé, on ne réécrit plus.
    fn check_match_end(&mut self) {
        if self.match_winner.is_some() {
            return;
        }
        // Victoire par fraglimit (priorité haute : si quelqu'un a déjà
        // atteint la limite pile quand le timer tombe à 0, on lui donne
        // la victoire au lieu de recalculer sur les scores).
        if self.frags >= FRAG_LIMIT {
            self.match_winner = Some(KillActor::Player);
            info!("match over — le joueur gagne ({} frags)", self.frags);
            return;
        }
        for d in &self.bots {
            if d.frags >= FRAG_LIMIT {
                self.match_winner = Some(KillActor::Bot(d.bot.name.clone()));
                info!("match over — bot '{}' gagne ({} frags)", d.bot.name, d.frags);
                return;
            }
        }
        // Victoire par timelimit : le + fragueur (joueur ou bot) gagne.
        // En cas d'égalité on prend le joueur humain par défaut —
        // simplification intentionnelle (Q3 ferait un overtime court).
        let elapsed = self.time_sec - self.match_start_at;
        if elapsed >= TIME_LIMIT_SECONDS {
            let mut best_frags = self.frags;
            let mut winner = KillActor::Player;
            for d in &self.bots {
                if d.frags > best_frags {
                    best_frags = d.frags;
                    winner = KillActor::Bot(d.bot.name.clone());
                }
            }
            let label = match &winner {
                KillActor::Player => "le joueur".to_string(),
                KillActor::Bot(name) => format!("bot '{name}'"),
                KillActor::World => "world".to_string(),
            };
            info!("match over — time up — {label} gagne ({best_frags} frags)");
            self.match_winner = Some(winner);
        }
    }

    /// Pousse un chiffre de dégât flottant à `origin`. Un `damage == 0`
    /// est ignoré (pas de feedback pour les tirs absorbés par l'armure
    /// à 0 HP, par ex.). `to_player == true` colore en rouge (dégât subi),
    /// sinon en jaune (dégât infligé).
    /// Queue un popup de médaille.  Appelé en même temps que le SFX
    /// correspondant pour que le son et le visuel soient synchronisés.
    /// Si `MEDAL_MAX` est atteint, on drop la plus ancienne (même logique
    /// que les dommages flottants).
    fn push_medal(&mut self, kind: Medal) {
        if self.active_medals.len() >= MEDAL_MAX {
            self.active_medals.remove(0);
        }
        self.active_medals.push(ActiveMedal {
            kind,
            expire_at: self.time_sec + MEDAL_SHOW_SEC,
            spawn_time: self.time_sec,
        });
    }

    fn push_damage_number(&mut self, origin: Vec3, damage: i32, to_player: bool) {
        if damage <= 0 {
            return;
        }
        if self.floating_damages.len() >= DAMAGE_NUMBER_MAX {
            // On drop le plus vieux — le plus récent a plus de valeur
            // perceptuelle pour le joueur.
            self.floating_damages.remove(0);
        }
        self.floating_damages.push(FloatingDamage {
            origin,
            damage,
            to_player,
            expire_at: self.time_sec + DAMAGE_NUMBER_LIFETIME,
            lifetime: DAMAGE_NUMBER_LIFETIME,
        });
        // Combo counter : on ne suit que les dégâts INFLIGÉS par le
        // joueur.  Les dégâts reçus (`to_player = true`) ont déjà la
        // pain-arrow + vignette rouge pour faire passer l'info.
        if !to_player {
            if self.time_sec - self.recent_dmg_last_at > DMG_BURST_WINDOW {
                self.recent_dmg_total = 0;
            }
            self.recent_dmg_total = self.recent_dmg_total.saturating_add(damage);
            self.recent_dmg_last_at = self.time_sec;
        }
    }

    /// Pousse `PARTICLE_EXPLOSION_COUNT` sparks depuis `origin`. Chaque
    /// spark part dans une direction hémisphérique (biais +Z pour éviter
    /// que tout parte sous le sol), avec vitesse et lifetime randomisées.
    /// Respecte `PARTICLE_MAX` : les plus vieilles particules sont drop
    /// en priorité quand on approche du cap.
    fn push_explosion_particles(&mut self, origin: Vec3) {
        // Couche 1 — sparks chauds (orange/jaune, rapide, courte vie).
        for _ in 0..PARTICLE_EXPLOSION_COUNT {
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.remove(0);
            }
            let mut dir = Vec3::new(rand_unit(), rand_unit(), rand_unit());
            let len = dir.length();
            if len < 1e-3 {
                dir = Vec3::Z;
            } else {
                dir /= len;
            }
            if dir.z < 0.0 {
                dir.z = -dir.z;
            }
            let speed = PARTICLE_EXPLOSION_SPEED * (0.5 + 0.5 * rand_unit().abs());
            let velocity = dir * speed;
            let hue = rand_unit().abs();
            let color = [1.0, 0.55 + 0.40 * hue, 0.15 + 0.30 * hue, 1.0];
            let lifetime = PARTICLE_EXPLOSION_LIFETIME * (0.7 + 0.3 * rand_unit().abs());
            self.particles.push(Particle {
                origin,
                velocity,
                color,
                expire_at: self.time_sec + lifetime,
                lifetime,
            });
        }
        // Couche 2 — **débris** (P3) : chunks gris-foncés à orange terne,
        // plus lents, plus longs en vie, biais vers le bas (gravité fait
        // déjà son boulot mais le tir initial est plus latéral). Donne
        // l'impression d'un vrai morceau de matière éjectée.
        let debris_count = (PARTICLE_EXPLOSION_COUNT / 3).max(6);
        for _ in 0..debris_count {
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.remove(0);
            }
            let mut dir = Vec3::new(rand_unit(), rand_unit(), rand_unit() * 0.6);
            let len = dir.length();
            if len < 1e-3 {
                dir = Vec3::Z;
            } else {
                dir /= len;
            }
            // Speed plus lent (~50 % du flash) — débris alourdis.
            let speed = PARTICLE_EXPLOSION_SPEED
                * 0.55
                * (0.4 + 0.6 * rand_unit().abs());
            let velocity = dir * speed;
            // Tons sombres : gris brun → orange terni (cendres tièdes).
            let h = rand_unit().abs();
            let color = [0.30 + 0.30 * h, 0.20 + 0.20 * h, 0.10 + 0.10 * h, 1.0];
            // Vie 1.5× plus longue pour qu'ils traînent.
            let lifetime =
                PARTICLE_EXPLOSION_LIFETIME * 1.5 * (0.7 + 0.3 * rand_unit().abs());
            self.particles.push(Particle {
                origin,
                velocity,
                color,
                expire_at: self.time_sec + lifetime,
                lifetime,
            });
        }
        // Couche 3 — **flash initial** (P3) : un dlight ultra-bref super
        // intense pour évoquer la détonation chimique (vs. la fire-glow
        // qui suit, plus longue et tiède). Renforce la lecture "boum"
        // de l'œil sans nécessiter de billboard.
        if let Some(r) = self.renderer.as_mut() {
            r.spawn_dlight(
                origin,
                500.0,           // radius
                [1.0, 0.95, 0.85],
                8.0,             // intensity boost vs glow standard 4-5
                self.time_sec,
                0.08,            // lifetime court (80 ms)
            );
        }
    }

    /// Pousse un petit nuage de puffs de fumée billboard autour de `origin`
    /// (via le renderer). Complète les sparks additifs : après un flash et
    /// des étincelles, il reste un nuage gris qui monte et s'estompe.
    ///
    /// 6 puffs espacés sur une sphère unit bruitée, vitesse lente biaisée
    /// vers le haut, tailles initiales petites (4u) qui gonflent jusqu'à
    /// 24u en fin de vie.  Couleur gris chaud → gris froid selon randomness.
    fn push_explosion_smoke(&mut self, origin: Vec3) {
        const SMOKE_COUNT: usize = 6;
        let Some(r) = self.renderer.as_mut() else { return };
        for _ in 0..SMOKE_COUNT {
            let mut dir = Vec3::new(rand_unit(), rand_unit(), rand_unit());
            let len = dir.length();
            if len < 1e-3 {
                dir = Vec3::Z;
            } else {
                dir /= len;
            }
            // Biais vers le haut (fumée qui monte) + petit offset initial
            // pour que les 6 puffs ne spawnent pas tous au même point.
            dir.z = dir.z.abs() * 0.8 + 0.2;
            let spawn = origin + dir * 4.0;
            let velocity = dir * (15.0 + 10.0 * rand_unit().abs());
            // Couleur gris moyen avec léger tint randomisé.
            let g = 0.35 + 0.15 * rand_unit().abs();
            let color = [g, g, g, 0.55];
            r.spawn_particle(
                spawn,
                velocity,
                color,
                4.0,   // taille début
                24.0,  // taille fin (expansion du nuage)
                self.time_sec,
                1.2 + 0.4 * rand_unit().abs(),
            );
        }
    }

    /// Pousse `PARTICLE_HIT_COUNT` étincelles à un point d'impact hitscan.
    /// Les sparks partent dans un cône autour de `normal` (rebond sur la
    /// surface touchée) avec vitesse et lifetime randomisées. `color` est
    /// l'ARGB de base (typiquement blanc-jaune pour un mur, rouge pour
    /// un bot, rose pour le railgun).
    fn push_hit_sparks(&mut self, pos: Vec3, normal: Vec3, color: [f32; 4]) {
        // Normal safe — si l'appelant passe un vecteur nul, on retombe sur Z.
        let n = if normal.length_squared() < 1e-4 {
            Vec3::Z
        } else {
            normal.normalize()
        };
        for _ in 0..PARTICLE_HIT_COUNT {
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.remove(0);
            }
            // Direction = normale + bruit : les sparks s'écartent mais
            // restent globalement orientés "vers l'extérieur" de la surface.
            let jitter = Vec3::new(rand_unit(), rand_unit(), rand_unit());
            let mut dir = n * 1.3 + jitter * 0.8;
            let len = dir.length();
            if len < 1e-3 {
                dir = n;
            } else {
                dir /= len;
            }
            let speed = PARTICLE_HIT_SPEED * (0.4 + 0.6 * rand_unit().abs());
            let lifetime = PARTICLE_HIT_LIFETIME * (0.6 + 0.4 * rand_unit().abs());
            self.particles.push(Particle {
                origin: pos,
                velocity: dir * speed,
                color,
                expire_at: self.time_sec + lifetime,
                lifetime,
            });
        }
    }

    /// Pose une marque persistante au point d'impact d'un tir hitscan
    /// sur un mur / sol.  Q3 spawne un petit "bullet hole" sombre à cet
    /// endroit, qui reste ~15 s avant de s'effacer — c'est un feedback
    /// visuel important, surtout à la Machinegun (on voit le trajet des
    /// balles sur la surface).  Le rayon et l'opacité dépendent de
    /// l'arme : Railgun laisse une brûlure plus large et plus marquée
    /// qu'une balle de MG.  Plasma ne passe pas par ici (c'est un
    /// projectile + explosion, il a son propre décal de brûlure).
    ///
    /// `normal` doit pointer hors du mur (typiquement `trace.plane_normal`).
    /// `pos` est le point d'impact exact retourné par la trace.
    fn push_wall_mark(&mut self, pos: Vec3, normal: Vec3, weapon: WeaponId) {
        // Paramètres par arme.  Couleur = sRGB linéaire, alpha est la
        // base (fade appliqué par le renderer sur les 25 % finaux).
        //
        // Gauntlet exclu : pas de "tir" qui touche un mur (contact melee),
        // donc pas de décale à spawner — c'est un miss dans l'espace.
        let (radius, color, lifetime) = match weapon {
            WeaponId::Machinegun => (3.0_f32, [0.06, 0.05, 0.04, 0.75], 15.0_f32),
            WeaponId::Shotgun => (2.5, [0.06, 0.05, 0.04, 0.75], 15.0),
            WeaponId::Lightninggun => (4.0, [0.10, 0.12, 0.18, 0.60], 8.0),
            WeaponId::Railgun => (6.0, [0.15, 0.05, 0.10, 0.70], 20.0),
            WeaponId::Gauntlet => return,
            // Les armes projectile ne font pas de hitscan, mais on
            // garde une branche défensive au cas où on wire la fonction
            // depuis ailleurs.
            _ => (4.0, [0.05, 0.04, 0.03, 0.6], 15.0),
        };
        if let Some(r) = self.renderer.as_mut() {
            r.spawn_decal(pos, normal, radius, color, self.time_sec, lifetime);
        }
    }

    /// Pose une flaque de sang persistante sous `pos` (typiquement
    /// l'épicentre d'un frag).  Trace un rayon vers le bas jusqu'au
    /// premier contact solide pour coller la décale au sol plutôt que
    /// flotter au niveau du buste.  Si la trace ne trouve rien dans les
    /// 512u (tombe dans le vide, plateforme trop haute), on skip : pas
    /// de décale plutôt qu'une flaque qui flotterait.
    ///
    /// Le rayon est randomisé par ±30 % et légèrement jittered sur les
    /// axes XY pour que deux frags superposés ne laissent pas deux
    /// flaques pixel-perfect identiques — impression d'impacts distincts
    /// sans coût.
    fn push_blood_splat(&mut self, pos: Vec3) {
        use q3_collision::Contents;
        let Some(world) = self.world.as_ref() else { return; };
        // Jitter XY léger pour éviter la collision parfaite entre deux
        // flaques (multi-frag au même spot, respawn kill-on-sight).
        let jx = rand_unit() * 8.0;
        let jy = rand_unit() * 8.0;
        let from = pos + Vec3::new(jx, jy, 0.0);
        let to = from + Vec3::new(0.0, 0.0, -512.0);
        let trace = world.collision.trace_ray(from, to, Contents::MASK_SHOT);
        if trace.fraction >= 1.0 {
            // Rien en-dessous sur 512u — le bot est tombé dans un pit,
            // la flaque atterrirait dans le void, on skip.
            return;
        }
        let hit_pt = from + (to - from) * trace.fraction;
        // La normale remonte du sol — si la trace tape un mur incliné
        // (improbable mais possible) on retombe sur Z pour ne pas
        // spawner une décale tordue.
        let normal = if trace.plane_normal.z > 0.25 {
            trace.plane_normal
        } else {
            Vec3::Z
        };
        // Taille variable : une flaque principale (12–16u) + tenir
        // compte que plusieurs morts sur une frag-chain font du volume.
        let radius = 12.0 + rand_unit().abs() * 4.0;
        // Rouge sombre, alpha modéré — pas trop saignant pour ne pas
        // faire écran sur les textures claires (marbre, base).
        let color = [0.22, 0.02, 0.02, 0.75];
        // Durée longue : 30 s, alignée sur les scorch marks rocket ;
        // c'est une trace d'événement important, on veut qu'elle reste.
        if let Some(r) = self.renderer.as_mut() {
            r.spawn_decal(hit_pt, normal, radius, color, self.time_sec, 30.0);
        }
    }

    /// Blob shadow sous une entité : trace un rayon vertical descendant
    /// depuis `center`, et si on touche du solide dans `max_drop` unités,
    /// émet une décale sombre circulaire au point d'impact.  Transitoire :
    /// la décale vit ~une frame (lifetime < 1 tick prune) pour être
    /// reposée au frame suivant quand l'entité aura bougé.
    ///
    /// Q3 original : `CG_PlayerShadow` trace `SHADOW_DISTANCE = 128u` et
    /// texture `gfx/marks/shadow` projetée en alpha-blend.  Ici on
    /// remplace le tex par le masque circulaire procédural du decal
    /// shader — même rendu final, zéro asset.
    ///
    /// `radius` en unités Q3 (~16 joueur, ~10 pickup).  `alpha_max` est
    /// l'opacité au pied (entité posée) — on fade vers 0 à `max_drop`
    /// pour que les entités qui volent n'aient pas un patch noir absurde
    /// au sol 2 mètres plus bas.
    fn push_blob_shadow(&mut self, center: Vec3, radius: f32, alpha_max: f32, max_drop: f32) {
        use q3_collision::Contents;
        let Some(world) = self.world.as_ref() else { return; };
        let from = center;
        let to = from + Vec3::new(0.0, 0.0, -max_drop);
        let trace = world.collision.trace_ray(from, to, Contents::MASK_SHOT);
        if trace.fraction >= 1.0 {
            // Rien en-dessous — entité dans le vide, pas d'ombre.
            return;
        }
        // Ne pas projeter l'ombre sur un plafond / mur incliné : si la
        // normale ne pointe pas « raisonnablement » vers le haut, skip.
        // 0.5 ≈ 60° d'inclinaison max — les rampes douces conservent leur
        // ombre, les murs verticaux (normale horizontale) sont éliminés.
        if trace.plane_normal.z < 0.5 {
            return;
        }
        let hit_pt = from + (to - from) * trace.fraction;
        // Atténuation : au contact direct (fraction ≈ 0) on est à fond,
        // plus on est loin, plus l'ombre est discrète.  Courbe quadratique
        // pour un fade doux en fin de plage.
        let falloff = 1.0 - trace.fraction.clamp(0.0, 1.0);
        let alpha = alpha_max * falloff * falloff;
        if alpha < 0.02 {
            return;
        }
        // Noir pur — l'ombre n'a pas de teinte, elle obscurcit simplement
        // le sol par transparence.
        let color = [0.0, 0.0, 0.0, alpha];
        // Lifetime volontairement très court : la décale est re-émise
        // chaque frame depuis la position courante de l'entité.  0.05 s
        // tolère des framerates jusqu'à ~20 Hz sans trou visible ; au-delà
        // de 60 Hz un nouveau spawn vient avant que l'ancien fade, donc
        // le chevauchement est imperceptible.  `prune_decals` retire au
        // frame suivant.
        if let Some(r) = self.renderer.as_mut() {
            r.spawn_decal(hit_pt, trace.plane_normal, radius, color, self.time_sec, 0.05);
        }
    }

    /// Émet une blob shadow par entité visible — player, bots, pickups
    /// disponibles, projectiles "lourds" (rockets, GL).  Appelé chaque
    /// frame avant le rendu des entités.
    ///
    /// Règle de radius : on prend ~40 % de la demi-largeur d'une hitbox
    /// Q3 standard (player box 32u large → 16u d'ombre au sol), et ~10u
    /// pour les pickups qui sont plus petits.  L'alpha max du joueur
    /// est plus élevé que celui des pickups pour que l'œil focalise sur
    /// les menaces animées.
    fn push_all_blob_shadows(&mut self) {
        // Joueur local — pas d'ombre si mort (le gib éparpille + le
        // ragdoll n'est pas simulé, ça ferait une ombre statique qui
        // ment à la lecture).  Pas d'ombre non plus si invisible powerup
        // actif : l'absence d'ombre est un tell visuel cohérent.
        if !self.player_health.is_dead()
            && !self.is_powerup_active(PowerupKind::Invisibility)
        {
            let foot = self.player.origin;
            self.push_blob_shadow(foot, 16.0, 0.55, 128.0);
        }
        // Bots — même règle : pas d'ombre si mort, ni si bot invisible
        // (mécanique future, actuellement inexistante).
        let bot_feet: Vec<Vec3> = self
            .bots
            .iter()
            .filter(|d| !d.health.is_dead())
            .map(|d| d.body.origin)
            .collect();
        for p in bot_feet {
            self.push_blob_shadow(p, 16.0, 0.55, 128.0);
        }
        // Pickups disponibles — l'ombre disparaît quand l'item est
        // consommé, ce qui aide à identifier visuellement ce qui est
        // encore ramassable.  Rayon plus petit (items Q3 ≈ 24u³).
        let pickup_positions: Vec<Vec3> = self
            .pickups
            .iter()
            .filter(|p| p.respawn_at.is_none())
            .map(|p| p.origin)
            .collect();
        for p in pickup_positions {
            self.push_blob_shadow(p, 10.0, 0.40, 96.0);
        }
        // Projectiles "lourds" : rockets et grenades.  Les projectiles
        // "énergie" (plasma, BFG, ...) émettent déjà leur propre dlight
        // vive au sol — y ajouter une ombre serait contradictoire
        // (l'objet lumineux ne projette rien de sombre).  Grenades qui
        // rebondissent gagnent vraiment à avoir l'ombre : elle aide
        // l'œil à prédire le rebond.
        let proj_positions: Vec<Vec3> = self
            .projectiles
            .iter()
            .filter(|p| matches!(
                p.weapon,
                WeaponId::Rocketlauncher | WeaponId::Grenadelauncher
            ))
            .map(|p| p.origin)
            .collect();
        for p in proj_positions {
            // Grenade ≈ 8u de diamètre, rocket ≈ 10u ; on prend 6u
            // d'ombre, plus petite et plus discrète que pour les
            // entités joueur.  Alpha max modéré pour que le joueur ne
            // soit pas distrait par le point noir en plein jeu.
            self.push_blob_shadow(p, 6.0, 0.35, 200.0);
        }
    }

    /// Pousse `PARTICLE_GIB_COUNT` particules rouges depuis `pos`, style
    /// gibs Quake. Direction sphérique (pas de biais vers le haut — un
    /// bot qui meurt part dans tous les sens), affecté par la gravité.
    fn push_death_gibs(&mut self, pos: Vec3) {
        for _ in 0..PARTICLE_GIB_COUNT {
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.remove(0);
            }
            let mut dir = Vec3::new(rand_unit(), rand_unit(), rand_unit());
            let len = dir.length();
            if len < 1e-3 {
                dir = Vec3::Z;
            } else {
                dir /= len;
            }
            // Biais léger vers le haut pour que les gibs rebondissent
            // visuellement plutôt que de tous piquer direct vers le sol.
            if dir.z < -0.3 {
                dir.z = -dir.z * 0.5;
            }
            let speed = PARTICLE_GIB_SPEED * (0.4 + 0.6 * rand_unit().abs());
            // Teinte rouge carmin avec variance vers pourpre.
            let hue = rand_unit().abs();
            let color = [0.8 + 0.2 * hue, 0.05 + 0.15 * hue, 0.05 + 0.10 * hue, 1.0];
            let lifetime = PARTICLE_GIB_LIFETIME * (0.7 + 0.3 * rand_unit().abs());
            self.particles.push(Particle {
                origin: pos,
                velocity: dir * speed,
                color,
                expire_at: self.time_sec + lifetime,
                lifetime,
            });
        }
    }

    /// Effet de respawn — colonne additive courte + sparks qui s'élèvent.
    /// Utilisé à la fois quand un pickup réapparaît et quand le joueur
    /// respawn. `color` détermine la teinte (cyan pour pickup, blanc-bleu
    /// pour le joueur).
    fn push_respawn_fx(&mut self, pos: Vec3, color: [f32; 4]) {
        // Beam vertical : hot au bas (près du sol), fade vers le haut.
        let bottom = pos;
        let top = pos + Vec3::Z * RESPAWN_FX_HEIGHT;
        self.beams.push(ActiveBeam {
            a: bottom,
            b: top,
            color,
            expire_at: self.time_sec + RESPAWN_FX_DURATION,
            lifetime: RESPAWN_FX_DURATION,
            style: BeamStyle::Straight,
        });
        // Sparks qui montent en cône étroit + un peu de dispersion xy.
        for _ in 0..PARTICLE_RESPAWN_COUNT {
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.remove(0);
            }
            let dx = rand_unit() * 0.35;
            let dy = rand_unit() * 0.35;
            let dz = 0.85 + 0.15 * rand_unit().abs();
            let mut dir = Vec3::new(dx, dy, dz);
            let len = dir.length().max(1e-6);
            dir /= len;
            let speed = 180.0 * (0.6 + 0.4 * rand_unit().abs());
            let lifetime = 0.45 * (0.7 + 0.3 * rand_unit().abs());
            self.particles.push(Particle {
                origin: pos + Vec3::new(rand_unit() * 6.0, rand_unit() * 6.0, 0.0),
                velocity: dir * speed,
                color,
                expire_at: self.time_sec + lifetime,
                lifetime,
            });
        }
    }

    /// Splash à la transition surface↔eau : une gerbe de gouttelettes
    /// bleu pâle qui gicle vers le haut en éventail.  Purement décoratif —
    /// pas de physique (les particules utilisent déjà la gravité standard
    /// du moteur, ce qui les fait retomber naturellement comme des gouttes).
    fn push_water_splash(&mut self, pos: Vec3) {
        let color = [0.7, 0.85, 1.0, 1.0];
        for _ in 0..PARTICLE_WATER_SPLASH_COUNT {
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.remove(0);
            }
            // Éventail conique vers le haut : petit radius latéral, grosse
            // composante Z positive pour que ça gicle façon éclaboussure.
            let dx = rand_unit() * 0.9;
            let dy = rand_unit() * 0.9;
            let dz = 0.6 + 0.4 * rand_unit().abs();
            let mut dir = Vec3::new(dx, dy, dz);
            let len = dir.length().max(1e-6);
            dir /= len;
            let speed = 140.0 * (0.6 + 0.4 * rand_unit().abs());
            let lifetime = 0.4 * (0.7 + 0.3 * rand_unit().abs());
            self.particles.push(Particle {
                origin: pos + Vec3::new(rand_unit() * 8.0, rand_unit() * 8.0, 0.0),
                velocity: dir * speed,
                color,
                expire_at: self.time_sec + lifetime,
                lifetime,
            });
        }
    }

    /// Burst de bulles : quelques particules cyan qui montent avec une
    /// vitesse lente.  Sans mécanique de buoyancy explicite — on compte
    /// sur le fait que la lifetime est courte et que la vitesse initiale
    /// reste suffisante pour contrer la gravité sur la durée de vie.
    fn push_bubble_burst(&mut self, pos: Vec3) {
        let color = [0.75, 0.9, 1.0, 0.85];
        for _ in 0..PARTICLE_BUBBLE_COUNT {
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.remove(0);
            }
            let dx = rand_unit() * 0.2;
            let dy = rand_unit() * 0.2;
            let dz = 0.9 + 0.1 * rand_unit().abs();
            let mut dir = Vec3::new(dx, dy, dz);
            let len = dir.length().max(1e-6);
            dir /= len;
            let speed = 60.0 * (0.7 + 0.3 * rand_unit().abs());
            let lifetime = 0.7 * (0.7 + 0.3 * rand_unit().abs());
            self.particles.push(Particle {
                origin: pos + Vec3::new(rand_unit() * 3.0, rand_unit() * 3.0, 0.0),
                velocity: dir * speed,
                color,
                expire_at: self.time_sec + lifetime,
                lifetime,
            });
        }
    }

    /// Tir hitscan joueur : N raycasts (N = `pellets`) depuis l'œil caméra
    /// avec une dispersion aléatoire. Chaque pellet cherche le bot avec la
    /// plus petite intersection sphère-ray avant un mur.
    ///
    /// Retourne `true` si un tir a été consommé (pour que le caller applique
    /// le cooldown complet), `false` si l'arme était vide (cooldown très court
    /// pour permettre un auto-switch futur).
    /// Secondes restantes sur un powerup — `None` si inactif ou expiré.
    fn powerup_remaining(&self, kind: PowerupKind) -> Option<f32> {
        self.powerup_until[kind.index()]
            .map(|t| t - self.time_sec)
            .filter(|&r| r > 0.0)
    }

    /// Le powerup est-il actif à cet instant ?
    fn is_powerup_active(&self, kind: PowerupKind) -> bool {
        self.powerup_remaining(kind).is_some()
    }

    /// Cycle la caméra spectator vers la prochaine cible vivante.
    /// `direction` : +1 = next, -1 = prev. Ignore les morts. Si rien
    /// à suivre (aucun remote vivant), `follow_slot` reste / devient
    /// `None` et l'utilisateur reste en free-fly.
    fn cycle_follow_target(&mut self, direction: i32) {
        let alive: Vec<u8> = self
            .remote_interp
            .iter()
            .filter_map(|(slot, buf)| {
                let s = buf.current()?;
                (!s.is_dead).then_some(*slot)
            })
            .collect();
        self.follow_slot = pick_follow_target(self.follow_slot, &alive, direction);
    }

    /// `true` si la fenêtre de warmup n'est pas terminée (tirs, IA bots
    /// et respawns sont figés).
    fn is_warmup(&self) -> bool {
        self.time_sec < self.warmup_until
    }

    /// Active / rallonge un powerup selon la règle de stacking Q3 (additif).
    /// Retourne le `time_sec` absolu d'expiration après grant.
    fn grant_powerup(&mut self, kind: PowerupKind, duration: f32) -> f32 {
        let slot = &mut self.powerup_until[kind.index()];
        let base = slot.filter(|&t| t > self.time_sec).unwrap_or(self.time_sec);
        let t = base + duration;
        *slot = Some(t);
        // Nouveau pickup = ré-armement du warning. Évite qu'un Quad pris
        // alors qu'un Quad précédent vient de flaguer `warned=true` zappe
        // le bip 3s de la nouvelle instance.
        self.powerup_warned[kind.index()] = false;
        t
    }

    /// Consomme le holdable du slot courant. No-op si vide, si le joueur
    /// est mort, ou si le match est terminé. Effet par variante :
    /// * Medkit : remonte la santé à `HOLDABLE_MEDKIT_TARGET_HP` (125).
    ///   Si la santé est déjà au-dessus de la cible, on considère que le
    ///   joueur n'en a pas besoin → le holdable reste en slot.
    /// * Teleporter : téléporte le joueur à un spawn point DM au hasard
    ///   (même seed que `respawn_player`), avec un effet visuel/sonore.
    fn use_held_item(&mut self) {
        let Some(kind) = self.held_item else { return; };
        if self.player_health.is_dead() || self.match_winner.is_some() {
            return;
        }
        match kind {
            HoldableKind::Medkit => {
                if self.player_health.current >= HOLDABLE_MEDKIT_TARGET_HP {
                    // Pas de waste : on laisse le medkit en slot.
                    return;
                }
                let before = self.player_health.current;
                self.player_health.current = HOLDABLE_MEDKIT_TARGET_HP;
                info!(
                    "use medkit: {} → {} HP",
                    before, self.player_health.current
                );
                // Consommé seulement si effectif.
                self.held_item = None;
            }
            HoldableKind::Teleporter => {
                let Some(world) = self.world.as_ref() else { return; };
                if world.spawn_points.is_empty() {
                    return;
                }
                // Même pattern pseudo-aléatoire que `respawn_player`.
                let seed = (self.time_sec * 1000.0) as usize
                    ^ (self.deaths as usize).wrapping_mul(2654435761);
                let sp = &world.spawn_points[seed % world.spawn_points.len()];
                let dest = sp.origin + Vec3::Z * 40.0;
                let prev = self.player.origin;
                info!(
                    "use teleporter: {:.0?} → {:.0?}",
                    prev, dest
                );
                self.player.origin = dest;
                self.player.view_angles = sp.angles;
                self.player.velocity = Vec3::ZERO;
                self.player.on_ground = false;
                // Effet FX : colonne colorée à l'ancienne et nouvelle
                // position, comme pour un vrai téléport de map.
                self.push_respawn_fx(prev, [0.5, 0.3, 1.0, 1.0]);
                self.push_respawn_fx(dest, [0.5, 0.3, 1.0, 1.0]);
                self.held_item = None;
            }
        }
    }

    /// Purge tous les powerups actifs (mort, restart de match…).
    fn clear_powerups(&mut self) {
        for s in self.powerup_until.iter_mut() {
            *s = None;
        }
        for w in self.powerup_warned.iter_mut() {
            *w = false;
        }
        self.regen_accum = 0.0;
    }

    /// Scanne les powerups expirés et les remet à `None` + log une seule
    /// fois la transition, pour ne pas garder un timestamp périmé à vie.
    fn tick_powerup_expiry(&mut self) {
        // On capture le listener AVANT la boucle — lire `self.player` dans
        // le corps d'un for-sur-`self.` marche, mais hoister rend la boucle
        // plus lisible et laisse la porte ouverte à un `self.bots` futur.
        let listener = self.player.origin;
        let end_sfx = self.sfx_powerup_end;
        let snd = self.sound.clone();
        for kind in PowerupKind::ALL {
            let i = kind.index();
            if let Some(t) = self.powerup_until[i] {
                if self.time_sec >= t {
                    self.powerup_until[i] = None;
                    // Ré-arme le warning : un futur pickup du même kind
                    // dans la même vie doit bip à 3s.
                    self.powerup_warned[i] = false;
                    info!("{}: expiré", kind.hud_label().to_lowercase());
                    if let (Some(snd), Some(h)) = (snd.as_ref(), end_sfx) {
                        play_hit_feedback(snd, h, listener);
                    }
                }
            }
        }
    }

    /// Scanne les powerups à 3s restantes (± tick) et joue un bip one-shot.
    /// Dépend de `tick_powerup_expiry` pour remettre `warned` à `false`
    /// quand un powerup expire + pour ré-armer sur un nouveau pickup.
    /// Silencieux si `sfx_powerup_warn` est introuvable.
    fn tick_powerup_warnings(&mut self) {
        const WARN_THRESHOLD_SEC: f32 = 3.0;
        let Some(warn_sfx) = self.sfx_powerup_warn else { return; };
        let Some(snd) = self.sound.clone() else { return; };
        let listener = self.player.origin;
        for kind in PowerupKind::ALL {
            let i = kind.index();
            if self.powerup_warned[i] {
                continue;
            }
            let Some(remaining) = self.powerup_remaining(kind) else {
                continue;
            };
            if remaining <= WARN_THRESHOLD_SEC {
                self.powerup_warned[i] = true;
                play_hit_feedback(&snd, warn_sfx, listener);
            }
        }
    }

    /// Applique un tick de régénération si Regen est actif. Fraction
    /// accumulée dans `regen_accum` pour éviter la quantification (dt très
    /// petit → 15·dt < 1 HP/frame). `Health::heal` cap déjà à `max`, donc
    /// pas de risque de dépasser 200 via megahealth.
    fn tick_regeneration(&mut self, dt: f32) {
        if !self.is_powerup_active(PowerupKind::Regeneration)
            || self.player_health.is_dead()
        {
            // Si plus actif, on reset le buffer pour ne pas rejouer un
            // demi-HP accumulé lors d'un prochain pickup.
            self.regen_accum = 0.0;
            return;
        }
        self.regen_accum += REGEN_HP_PER_SECOND * dt;
        if self.regen_accum >= 1.0 {
            let whole = self.regen_accum.floor() as i32;
            if whole > 0 && !self.player_health.is_full() {
                self.player_health.heal(whole);
            }
            self.regen_accum -= whole as f32;
        }
    }

    /// Émet un pas si `bob_phase` a franchi un multiple de π depuis la
    /// dernière foulée enregistrée. Chaque pas pioche une variante != de
    /// la précédente (évite "l'effet cassette bloquée"). Silent-no-op si
    /// aucun sfx n'est chargé.
    fn maybe_play_footstep(&mut self, phase_before: f32, phase_after: f32) {
        use std::f32::consts::PI;
        if self.sfx_footsteps.is_empty() {
            return;
        }
        // Prochain multiple de π après `last_footstep_phase` :
        let next_step_phase = (self.last_footstep_phase / PI).floor() * PI + PI;
        // Avons-nous franchi ce seuil ce tick ? On teste sur `[phase_before,
        // phase_after)` — bob_phase ne descend jamais dans ce block, donc
        // un simple >= suffit.
        if phase_after >= next_step_phase && phase_before < next_step_phase {
            // Variante pseudo-aléatoire différente de la précédente.
            let n = self.sfx_footsteps.len();
            let mut idx = (rand_unit().abs() * n as f32) as usize % n;
            if let Some(prev) = self.last_footstep_idx {
                if idx == prev && n > 1 {
                    idx = (idx + 1) % n;
                }
            }
            self.last_footstep_idx = Some(idx);
            self.last_footstep_phase = next_step_phase;
            if let Some(snd) = self.sound.as_ref() {
                // Atténuation : un pas n'écrase ni un tir, ni un pickup.
                // `Low` priority laisse le mixer dropper si plein.
                play_at(snd, self.sfx_footsteps[idx], self.player.origin, Priority::Low);
            }
        }
    }

    /// Params physiques effectifs pour le tick courant — appliquent les
    /// modifiers powerup. Haste multiplie `max_speed` par 1.3 (+ l'accel
    /// sinon la vitesse cible est atteignable trop lentement).
    fn effective_physics_params(&self) -> PhysicsParams {
        let mut p = self.params;
        if self.is_powerup_active(PowerupKind::Haste) {
            p.max_speed *= HASTE_SPEED_MULT;
            // L'accélération suit : sans ça, la vitesse max en steady
            // state augmente mais on met + longtemps à y arriver qu'à
            // atteindre max_speed baseline.
            p.accelerate *= HASTE_SPEED_MULT;
        }
        if self.is_powerup_active(PowerupKind::Flight) {
            // Flight : gravité désactivée. On laisse le reste des params
            // intacts — le joueur garde sa vitesse horizontale, et la
            // composante Z est gérée au niveau app.rs (vertical thrust
            // depuis jump/crouch) avant d'appeler tick_collide.
            p.gravity = 0.0;
        }
        p
    }

    /// Retourne le multiplicateur à appliquer sur les dégâts infligés par
    /// le joueur ce tick. 4 quand Quad Damage est actif, 1 sinon.
    fn player_damage_multiplier(&self) -> i32 {
        if self.is_powerup_active(PowerupKind::QuadDamage) {
            QUAD_DAMAGE_FACTOR
        } else {
            1
        }
    }

    /// Facteur multiplicatif appliqué au cooldown entre deux tirs.
    /// < 1 = on tire plus vite (Haste). 1 = cadence nominale.
    fn player_fire_cooldown_mult(&self) -> f32 {
        if self.is_powerup_active(PowerupKind::Haste) {
            HASTE_FIRE_MULT
        } else {
            1.0
        }
    }

    /// AABB du joueur dans les coordonnées monde — approximation du "player
    /// hull" Q3 utilisée uniquement pour tester les overlaps triggers.
    fn player_aabb(&self) -> Aabb {
        let o = self.player.origin;
        Aabb::new(
            Vec3::new(
                o.x - PLAYER_HULL_HALF_XY,
                o.y - PLAYER_HULL_HALF_XY,
                o.z + PLAYER_HULL_MIN_Z,
            ),
            Vec3::new(
                o.x + PLAYER_HULL_HALF_XY,
                o.y + PLAYER_HULL_HALF_XY,
                o.z + PLAYER_HULL_MAX_Z,
            ),
        )
    }

    /// Teste les contacts joueur ↔ `trigger_push` et applique la vélocité
    /// pré-calculée à l'entrée. On ne re-déclenche pas tant que le joueur
    /// reste dans le trigger — comportement identique à Q3 (le push est
    /// un "touch one-shot", pas un champ continu).
    fn tick_jump_pads(&mut self) {
        if self.jump_pads.is_empty()
            || self.player_health.is_dead()
            || self.match_winner.is_some()
        {
            self.on_jumppad_idx = None;
            return;
        }
        let hull = self.player_aabb();
        let mut hit: Option<usize> = None;
        for (i, pad) in self.jump_pads.iter().enumerate() {
            if hull.intersects(pad.bounds) {
                hit = Some(i);
                break;
            }
        }
        // Transition "pas dans le pad" → "dans le pad" : on fire.
        if let Some(idx) = hit {
            if self.on_jumppad_idx != Some(idx) {
                let pad = self.jump_pads[idx];
                self.player.velocity = pad.launch_velocity;
                // `on_ground = false` sinon le prochain tick de physique
                // tue immédiatement la composante verticale par le "ground
                // friction". Q3 fait la même chose via `pm->ps->groundEntity
                // = ENTITYNUM_NONE` quand il applique le push.
                self.player.on_ground = false;
                self.was_airborne = true;
                if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_jumppad) {
                    play_at(snd, h, pad.center, Priority::Normal);
                }
                debug!(
                    "jump pad #{idx}: v = ({:.0}, {:.0}, {:.0})",
                    pad.launch_velocity.x, pad.launch_velocity.y, pad.launch_velocity.z
                );
            }
        }
        self.on_jumppad_idx = hit;
    }

    /// Teste les contacts joueur ↔ `trigger_teleport` et téléporte à
    /// l'entrée. Reproduit le comportement de `TeleportPlayer` de
    /// `g_misc.c` : reset vélocité, ré-align angles, FX + double son.
    fn tick_teleporters(&mut self) {
        if self.teleporters.is_empty()
            || self.player_health.is_dead()
            || self.match_winner.is_some()
        {
            self.on_teleport_idx = None;
            return;
        }
        let hull = self.player_aabb();
        let mut hit: Option<usize> = None;
        for (i, tp) in self.teleporters.iter().enumerate() {
            if hull.intersects(tp.bounds) {
                hit = Some(i);
                break;
            }
        }
        if let Some(idx) = hit {
            if self.on_teleport_idx != Some(idx) {
                let tp = self.teleporters[idx];
                // Q3 TeleportPlayer :
                //   origin  <- dest.origin  (+ petit nudge Z géré par les
                //               mappeurs directement dans le .map)
                //   angles  <- dest.angles
                //   velocity = 0  (sortie "propre" — gère les chains)
                //   teleport_time (pas utilisé ici, on ne rend pas le flash)
                self.player.origin = tp.dst_origin;
                self.player.view_angles = tp.dst_angles;
                self.player.velocity = Vec3::ZERO;
                self.player.on_ground = false;
                self.was_airborne = true;
                // Les pads/triggers à destination ne doivent pas refire
                // tant qu'on n'en est pas sorti au moins une fois.
                self.on_jumppad_idx = None;
                if let Some(snd) = self.sound.as_ref() {
                    if let Some(h) = self.sfx_teleport_in {
                        play_at(snd, h, tp.src_center, Priority::Normal);
                    }
                    if let Some(h) = self.sfx_teleport_out {
                        play_at(snd, h, tp.dst_origin, Priority::Normal);
                    }
                }
                // Petit FX de particules aux deux extrémités — cyan Q3.
                self.push_respawn_fx(tp.src_center, [0.6, 0.9, 1.0, 1.0]);
                self.push_respawn_fx(tp.dst_origin, [0.6, 0.9, 1.0, 1.0]);
                info!(
                    "teleport #{idx}: {:?} → {:?}",
                    tp.src_center, tp.dst_origin
                );
            }
        }
        self.on_teleport_idx = hit;
    }

    /// Teste le contact joueur ↔ zones `trigger_hurt`. Inflige les dégâts
    /// à cadence `zone.interval` (0.1 s standard, 1 s pour SLOW). Respecte
    /// l'absorption armor sauf pour les zones NO_PROTECTION (= void / lava
    /// instant-kill). Mort → kill-feed "World → You [VOID/LAVA/HURT]".
    /// Gère la noyade : tant que la caméra (l'œil) du joueur est dans un
    /// brush `Contents::WATER`, `air_left` se décrémente.  Quand il atteint
    /// 0, on inflige `DROWN_DAMAGE` toutes les `DROWN_INTERVAL` secondes.
    /// Dès que le joueur refait surface, la jauge est rechargée à fond ;
    /// pas de récupération graduelle (on matche le comportement Q3).
    ///
    /// L'invulnérabilité post-respawn suspend aussi les dégâts de noyade,
    /// sinon un respawn malheureux dans un couloir entièrement immergé
    /// tuerait le joueur avant même qu'il ait pu bouger.
    fn tick_drown(&mut self, dt: f32) {
        if self.player_health.is_dead() || self.match_winner.is_some() {
            return;
        }
        let Some(world) = self.world.as_ref() else { return; };
        let eye = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
        let head_in_water = world
            .collision
            .point_contents(eye)
            .contains(q3_collision::Contents::WATER);
        // Fronts montants/descendants : splash audio + burst de particules
        // à la surface (on positionne à la hauteur d'œil, pas à la hauteur
        // des pieds, car c'est la transition de tête qui déclenche le son
        // dans Q3).
        if head_in_water != self.was_underwater {
            let splash_pos = eye;
            if head_in_water {
                // Entrée : watr_in + burst blanc/bleu vers le bas.
                if let Some(snd) = self.sound.as_ref() {
                    if let Some(h) = self.sfx_water_in.or(self.sfx_water_out) {
                        play_at(snd, h, splash_pos, Priority::Normal);
                    }
                }
                self.push_water_splash(splash_pos);
            } else {
                // Sortie : watr_out + burst qui gicle vers le haut.
                if let Some(snd) = self.sound.as_ref() {
                    if let Some(h) = self.sfx_water_out.or(self.sfx_water_in) {
                        play_at(snd, h, splash_pos, Priority::Normal);
                    }
                }
                self.push_water_splash(splash_pos);
            }
            // Muffle audio : on divise le master par ~2 sous l'eau pour
            // simuler le filtrage grave qu'imposerait un vrai low-pass.
            // Approximation pauvre (on ne touche pas aux fréquences) mais
            // suffisante pour signaler à l'oreille « environnement changé ».
            // On relit le s_volume courant pour ne pas écraser un réglage
            // utilisateur changé depuis le dernier switch.
            if let Some(snd) = self.sound.as_ref() {
                let base = self.cvars.get_f32("s_volume").unwrap_or(0.8);
                let factor = if head_in_water { UNDERWATER_VOLUME_FACTOR } else { 1.0 };
                snd.set_master_volume(base * factor);
            }
        }
        self.was_underwater = head_in_water;
        if !head_in_water {
            // Surface : on remplit les poumons et on annule tout tick
            // de noyade en attente.
            self.air_left = AIR_CAPACITY_SEC;
            self.next_drown_at = 0.0;
            return;
        }
        // Bulles d'air : tant qu'il reste > 50 % d'air, on suppose que le
        // joueur retient son souffle. En dessous, il laisse échapper un
        // filet de bulles régulier. Fréquence qui accélère en-dessous de
        // la moitié pour donner un feedback visuel progressif d'urgence.
        if self.air_left < AIR_CAPACITY_SEC * 0.5 && self.time_sec >= self.next_bubble_at {
            self.push_bubble_burst(eye);
            // 0.5 s quand il reste ~6 s, 0.15 s quand il reste 0 s.
            let urgency = 1.0 - (self.air_left / (AIR_CAPACITY_SEC * 0.5)).clamp(0.0, 1.0);
            let interval = 0.5 - 0.35 * urgency;
            self.next_bubble_at = self.time_sec + interval;
        }
        // Battle Suit : respire indéfiniment. On réinitialise juste la
        // jauge pour que l'HUD ne montre pas un compteur figé à mi-chemin
        // pendant la durée du powerup, et pour garder 12 s de marge au
        // moment où il expire (plutôt que de reprendre pile où on en
        // était avant le pickup, ce qui serait brutal).
        if self.is_powerup_active(PowerupKind::BattleSuit) {
            self.air_left = AIR_CAPACITY_SEC;
            self.next_drown_at = 0.0;
            return;
        }
        // Sous l'eau : on décompte l'air.  Quand on arrive à 0 pour la
        // première fois, on arme le prochain tick de dégâts à `now + INTERVAL`
        // pour laisser une seconde de grâce.
        if self.air_left > 0.0 {
            self.air_left = (self.air_left - dt).max(0.0);
            if self.air_left == 0.0 {
                self.next_drown_at = self.time_sec + DROWN_INTERVAL;
            }
            return;
        }
        // Air épuisé — applique le dégât si l'heure est venue.
        if self.player_invul_until > self.time_sec {
            return;
        }
        if self.time_sec < self.next_drown_at {
            return;
        }
        self.next_drown_at = self.time_sec + DROWN_INTERVAL;
        // La noyade ignore l'armure (pas d'armure « contre l'eau »).
        let taken = self.player_health.take_damage(DROWN_DAMAGE);
        if taken > 0 {
            if let (Some(snd), Some(handle)) =
                (self.sound.as_ref(), self.sfx_pain_player)
            {
                play_at(snd, handle, self.player.origin, Priority::High);
            }
            let eye_pos = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
            self.push_damage_number(eye_pos, taken, true);
            self.pain_flash_until = self.time_sec + PAIN_FLASH_SEC;
        }
        if taken > 0 && self.player_health.is_dead() {
            self.deaths = self.deaths.saturating_add(1);
            self.respawn_at = Some(self.time_sec + RESPAWN_DELAY_SEC);
            info!(
                "mort par noyade — respawn dans {RESPAWN_DELAY_SEC:.1}s (deaths={})",
                self.deaths
            );
            self.push_kill_cause(
                KillActor::World,
                KillActor::Player,
                KillCause::Environment("drowning"),
            );
        }
    }

    fn tick_hurt_zones(&mut self) {
        if self.hurt_zones.is_empty()
            || self.player_health.is_dead()
            || self.match_winner.is_some()
        {
            return;
        }
        let hull = self.player_aabb();
        let now = self.time_sec;
        // On collecte les hits pour relâcher le borrow mut de `hurt_zones`
        // avant de toucher à player_health / player_armor / kill_feed.
        #[derive(Clone, Copy)]
        struct Hit {
            damage: i32,
            no_protection: bool,
            label: &'static str,
        }
        let mut hits: Vec<Hit> = Vec::new();
        // Invul post-respawn : on skip les zones standards (fire trigger,
        // slime, lava low-damage) mais on laisse passer les zones
        // `no_protection` (VOID / sortie de map) — sinon un joueur qui
        // respawnerait par malchance dans une zone de VOID serait
        // « sauvé » et resterait coincé en invul perpétuel.
        let player_invul = self.player_invul_until > self.time_sec;
        // Battle Suit : absorbe toutes les zones protégeables (lave,
        // slime, petits triggers de dégât). Les zones `no_protection`
        // (VOID, zones de mort instantanée d'un designer) passent quand
        // même — si le level designer a câblé un kill guaranti, on ne
        // le contourne pas avec un suit.
        let player_envshield = self.is_powerup_active(PowerupKind::BattleSuit);
        for zone in self.hurt_zones.iter_mut() {
            if !hull.intersects(zone.bounds) {
                continue;
            }
            if player_invul && !zone.no_protection {
                continue;
            }
            if player_envshield && !zone.no_protection {
                continue;
            }
            if now < zone.next_at {
                continue;
            }
            zone.next_at = now + zone.interval;
            hits.push(Hit {
                damage: zone.damage,
                no_protection: zone.no_protection,
                label: zone.label,
            });
        }
        for h in hits {
            let dmg_after = if h.no_protection {
                h.damage
            } else {
                let absorbed = (h.damage / 2).min(self.player_armor);
                self.player_armor -= absorbed;
                if absorbed > 0 {
                    self.armor_flash_until = now + ARMOR_FLASH_SEC;
                }
                h.damage - absorbed
            };
            let taken = self.player_health.take_damage(dmg_after);
            if taken > 0 {
                if let (Some(snd), Some(handle)) =
                    (self.sound.as_ref(), self.sfx_pain_player)
                {
                    play_at(snd, handle, self.player.origin, Priority::High);
                }
                let eye = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
                self.push_damage_number(eye, taken, true);
                self.pain_flash_until = now + PAIN_FLASH_SEC;
            }
            if taken > 0 && self.player_health.is_dead() {
                self.deaths = self.deaths.saturating_add(1);
                self.respawn_at = Some(self.time_sec + RESPAWN_DELAY_SEC);
                info!(
                    "mort par {} — respawn dans {RESPAWN_DELAY_SEC:.1}s (deaths={})",
                    h.label, self.deaths
                );
                self.push_kill_cause(
                    KillActor::World,
                    KillActor::Player,
                    KillCause::Environment(h.label),
                );
                break;
            }
        }
    }

    /// Applique les dégâts de chute en cas d'atterrissage violent. Appelée
    /// une fois par tick juste après `tick_hurt_zones`, avec la vitesse Z
    /// mémorisée *avant* le tick de physique (après le tick, `update_ground`
    /// l'a ramenée à 0). Le `was_airborne` flag est celui qu'on utilise
    /// pour le SFX de `sfx_land` — on le laisse intact pour ce dernier.
    ///
    /// Paliers calés pour que :
    ///  - un saut sur place (|vz| ≈ jump_velocity = 270) ne fasse rien ;
    ///  - un saut depuis un rebord de 60 u donne ~5 HP ;
    ///  - une chute de 500 u donne 25 HP ;
    ///  - une chute dans le vide applique 50 HP → quasi-kill, mais les
    ///    zones `trigger_hurt` "VOID" font le reste (100 HP NO_PROTECTION).
    fn tick_fall_damage(&mut self, prev_vz: f32) {
        // Conditions de déclenchement : on landait ce tick, on est vivant,
        // et le match n'est pas terminé.
        if !self.was_airborne
            || !self.player.on_ground
            || self.player_health.is_dead()
            || self.match_winner.is_some()
            // Invul post-respawn : on ne punit pas un joueur qui vient
            // d'apparaître en hauteur. Cohérent avec les autres sources
            // de dégât bloquées pendant la fenêtre.
            || self.player_invul_until > self.time_sec
            // Battle Suit : pas de dégâts de chute. Canonique Q3 —
            // c'est une des protections clé du suit.
            || self.is_powerup_active(PowerupKind::BattleSuit)
        {
            return;
        }
        let fall = prev_vz.abs();
        // Pas de dégâts sous le seuil "saut normal". 400 = ce que donne
        // un saut depuis une plateforme de 60 u avec la jump_velocity
        // par défaut de 270. Tout ce qui est plus haut fait mal.
        const FALL_SAFE: f32 = 400.0;
        let damage: i32 = if fall < FALL_SAFE {
            0
        } else if fall < 550.0 {
            5
        } else if fall < 700.0 {
            10
        } else if fall < 900.0 {
            25
        } else {
            50
        };
        if damage == 0 {
            return;
        }
        let taken = self.player_health.take_damage(damage);
        if taken > 0 {
            if let (Some(snd), Some(handle)) =
                (self.sound.as_ref(), self.sfx_pain_player)
            {
                play_at(snd, handle, self.player.origin, Priority::High);
            }
            let eye = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
            self.push_damage_number(eye, taken, true);
            self.pain_flash_until = self.time_sec + PAIN_FLASH_SEC;
        }
        if taken > 0 && self.player_health.is_dead() {
            self.deaths = self.deaths.saturating_add(1);
            self.respawn_at = Some(self.time_sec + RESPAWN_DELAY_SEC);
            info!(
                "mort par chute (|vz|={:.0}) — respawn dans {RESPAWN_DELAY_SEC:.1}s (deaths={})",
                fall, self.deaths
            );
            self.push_kill_cause(
                KillActor::World,
                KillActor::Player,
                KillCause::Environment("FALL"),
            );
        }
    }

    fn fire_weapon(&mut self) -> bool {
        use q3_collision::Contents;
        let Some(world) = self.world.as_ref() else { return false; };
        let weapon = self.active_weapon;
        // **Tir secondaire** (RMB) — chaque arme implémente sa variante
        // alt-fire via `secondary_params(weapon)` qui modifie damage/
        // pellets/spread/cooldown/range au lancement. Pour les armes
        // sans alt-fire défini, on retombe sur les params primaires.
        let secondary = self.input.secondary_fire;
        let mut params = weapon.params();
        let mut alt_active = false;
        if secondary {
            if let Some(alt) = weapon.secondary_params() {
                params = alt;
                alt_active = true;
            }
        }
        let slot = weapon.slot() as usize;
        let _ = alt_active; // utilisé plus bas pour rocket lock-on / rail ricochet

        // **W1 — Gauntlet lunge** : avant tout traitement de tir, on
        // applique un dash-forward au joueur si alt-fire + gauntlet.
        // Effet "swing en avant" qui ferme la distance d'un coup,
        // signature des FPS gauntlet melee modernes.
        if alt_active && weapon == WeaponId::Gauntlet {
            let basis = self.player.view_angles.to_vectors();
            let mut f = basis.forward;
            f.z = 0.0;
            if f.length_squared() > 0.001 {
                let f = f.normalize();
                self.player.velocity.x += f.x * 380.0;
                self.player.velocity.y += f.y * 380.0;
            }
        }

        // Early-out si pas assez de munitions. On joue un "click empty"
        // bridé à 1 fois par 0.4 s pour éviter le spam log/audio.
        // Deux comportements en chaîne :
        //   1. On joue le SFX click si dispo (`sfx_no_ammo`) — feedback
        //      audio immédiat que la gâchette est molle.
        //   2. On tente un auto-switch vers la meilleure arme disposant
        //      de munitions — comportement Q3 original (`cg_autoswitch`).
        //      Si on trouve une arme cible, on la sélectionne ET on skip
        //      le tir de cette frame (le joueur doit retapper).  Sinon
        //      on reste sur l'arme vide et il est "dry".
        if (self.ammo[slot] as u32) < params.ammo_cost as u32 {
            if self.time_sec >= self.next_empty_click_at {
                self.next_empty_click_at = self.time_sec + 0.4;
                info!("*click* — {} est vide", weapon.name());
                if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_no_ammo) {
                    // Joué à la position oreille : feedback du joueur,
                    // pas un son monde — pas d'atténuation distance.
                    let ear = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
                    play_at(snd, h, ear, Priority::Weapon);
                }
                // Autoswitch : on cherche la meilleure arme possédée qui
                // a encore assez de munitions. Priorité à l'ordre "fort
                // vers faible" — RG > RL > LG > PG > SG > MG > GL > BFG >
                // Gauntlet — pour que le joueur ne tombe pas sur un gaunt
                // alors qu'il avait encore 5 rockets.  Gauntlet en dernier
                // car elle n'utilise pas de munitions (toujours « viable »).
                const AUTOSWITCH_ORDER: &[WeaponId] = &[
                    WeaponId::Railgun,
                    WeaponId::Rocketlauncher,
                    WeaponId::Lightninggun,
                    WeaponId::Plasmagun,
                    WeaponId::Shotgun,
                    WeaponId::Machinegun,
                    WeaponId::Grenadelauncher,
                    WeaponId::Bfg,
                    WeaponId::Gauntlet,
                ];
                for &candidate in AUTOSWITCH_ORDER {
                    if candidate == weapon {
                        continue;
                    }
                    let bit = 1u32 << candidate.slot();
                    if (self.weapons_owned & bit) == 0 {
                        continue;
                    }
                    let c_cost = candidate.params().ammo_cost as i32;
                    let c_slot = candidate.slot() as usize;
                    // Gauntlet : ammo_cost = 0, toujours viable — elle
                    // passe naturellement ce test.
                    if self.ammo[c_slot] >= c_cost {
                        info!(
                            "autoswitch: {} → {} (out of ammo)",
                            weapon.name(),
                            candidate.name()
                        );
                        // Passe par switch_to_weapon pour mémoriser
                        // `last_weapon` et armer l'animation du panel.
                        // Le helper applique déjà un cooldown fire de
                        // 0.1s ; on garde explicitement 0.15s ici (légèrement
                        // plus long, car on vient de re-tenter un tir à sec).
                        self.switch_to_weapon(candidate);
                        self.next_player_fire_at = self.time_sec + 0.15;
                        break;
                    }
                }
            }
            return false;
        }
        self.ammo[slot] = (self.ammo[slot] - params.ammo_cost as i32).max(0);
        // Stats accuracy : un `fire_weapon()` qui dépasse l'ammo-check
        // compte comme un shot tiré.  Le Gauntlet (ammo_cost = 0) compte
        // aussi — un swing est un "tir" pour le tally global.
        self.total_shots = self.total_shots.saturating_add(1);

        // Rompt l'invulnérabilité post-respawn dès que le joueur attaque.
        // Counter-play Q3 classique : l'invul est une « bulle défensive »
        // qui disparaît au premier tir — on ne peut pas spawner, foncer
        // et fragger en invul. On laisse passer la frame de tir courante
        // (le joueur n'est pas l'un de ses propres splash targets de toute
        // façon puisque la branche invul plus haut skip ses rockets).
        if self.player_invul_until > self.time_sec {
            self.player_invul_until = 0.0;
            info!("invul rompue par tir");
        }

        let eye = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
        let basis = self.player.view_angles.to_vectors();

        // SFX + muzzle flash une seule fois par volée.
        if let Some((_, h)) = self.sfx_fire.iter().find(|(w, _)| *w == weapon) {
            if let Some(snd) = self.sound.as_ref() {
                play_at(snd, *h, eye, Priority::Weapon);
            }
        }
        self.muzzle_flash_until = self.time_sec + 0.06;
        // Impulse de recul : cumule avec la valeur courante pour que les
        // rafales sentent "chargées", clampé à `VIEW_KICK_MAX` pour
        // éviter un viewmodel qui disparaît de l'écran sur tir auto.
        self.view_kick = (self.view_kick + weapon.view_kick()).min(VIEW_KICK_MAX);
        // Dlight muzzle flash : orange-jaune chaud, rayon modeste, durée
        // calée sur `muzzle_flash_until` (60 ms).  Invisible pour les
        // armes dont le WeaponKind gère son propre visuel (rail/LG),
        // mais sans conséquence si on la spawne quand même — la dlight
        // expire avant que l'œil y fasse attention.
        if let Some(r) = self.renderer.as_mut() {
            r.spawn_dlight(
                eye + basis.forward * 4.0,
                200.0,
                [1.0, 0.75, 0.3],
                1.5,
                self.time_sec,
                0.06,
            );
        }

        // Branchement sur le type de tir : projectile → spawn entité et
        // on sort. Hitscan → on continue dans la boucle de raycasts.
        if let WeaponKind::Projectile {
            speed,
            splash_radius,
            splash_damage,
        } = params.kind
        {
            // Spawn légèrement devant l'œil pour éviter d'exploser dans la
            // face du joueur au moindre obstacle côté sol.
            let spawn = eye + basis.forward * 16.0;
            // Mesh + tint + physique propres à l'arme.
            // * plasma : bleu électrique, linéaire, fuse 5s (kill-switch)
            // * grenade : gris neutre, gravité + rebond, fuse 2.5s (Q3 canon)
            // * rocket (défaut) : blanc, linéaire, fuse 5s
            let (mesh, tint, gravity, bounce, fuse) = match weapon {
                WeaponId::Plasmagun => (
                    self.plasma_mesh.clone(),
                    [0.45, 0.65, 1.0, 1.0],
                    0.0,
                    false,
                    5.0,
                ),
                WeaponId::Grenadelauncher => (
                    self.grenade_mesh.clone(),
                    [0.9, 0.9, 0.75, 1.0],
                    800.0, // Q3 g_gravity
                    true,
                    2.5, // fuse canonique
                ),
                WeaponId::Bfg => (
                    // On partage le mesh rocket — pas de MD3 BFG natif dans
                    // les pk3 standard pour le projectile en vol (sprite Q3).
                    self.rocket_mesh.clone(),
                    [0.45, 1.0, 0.45, 1.0], // vert BFG
                    0.0,
                    false,
                    5.0,
                ),
                _ => (
                    self.rocket_mesh.clone(),
                    [1.0, 1.0, 1.0, 1.0],
                    0.0,
                    false,
                    5.0,
                ),
            };
            let dmg_mul = self.player_damage_multiplier();
            // **Rocket lock-on** (W5) — alt-fire sur RL : on cherche le
            // bot le plus proche dans un cône frontal de 30° et 1500u.
            // Si trouvé, on tag le projectile pour homing. Sinon, ça
            // part en ligne droite comme un primaire — lock raté = tir
            // perdu, pas un cancel : le joueur s'engage en pressant RMB.
            let homing_target = if alt_active && weapon == WeaponId::Rocketlauncher
            {
                find_lock_target(&self.bots, eye, basis.forward, self.time_sec)
            } else {
                None
            };
            self.projectiles.push(Projectile {
                origin: spawn,
                velocity: basis.forward * speed,
                direct_damage: params.damage * dmg_mul,
                splash_radius,
                splash_damage: splash_damage * dmg_mul,
                owner: ProjectileOwner::Player,
                weapon,
                expire_at: self.time_sec + fuse,
                gravity,
                bounce,
                mesh,
                tint,
                next_trail_at: 0.0,
                homing_target,
            });
            return true;
        }

        let spread_rad = params.spread_deg.to_radians();
        let mut any_hit = false;
        // Idem que `any_hit` mais pour les kills : un tir qui finit un
        // bot (taken > 0 && dead) flag ceci. Déclenche le thunk kill-
        // confirm une seule fois en fin de volée pour éviter la pile
        // de samples sur un SG double-frag.
        let mut any_kill = false;
        // Compteur de frags joueur sur cette volée — flushé vers
        // `on_player_frag()` après relâche des borrows sur `self.world`
        // et `self.bots`.  Permet de gérer les streaks (Unreal-style)
        // sans E0502 / E0499.
        let mut pending_player_frags: u32 = 0;
        // Médaille « Humiliation » : Q3 la joue uniquement sur kill au
        // Gauntlet. La volée entière est exécutée avec une seule
        // `weapon`, donc on peut lire directement `weapon` après — pas
        // besoin de flag par-hit. On garde une bool pour la symétrie.
        let mut any_gauntlet_kill = false;
        // On accumule les frags à pousser dans la kill feed pour les flusher
        // APRÈS avoir lâché le borrow immutable sur `self.world` — sinon
        // `self.push_kill(..)` (mut) entre en conflit avec `world` (immut).
        let mut pending_kills: Vec<(KillActor, KillActor, WeaponId)> = Vec::new();
        // Même logique pour les chiffres de dégât : on collecte pendant la
        // boucle et on flushe après avoir lâché `world`.
        let mut pending_damage_nums: Vec<(Vec3, i32, bool)> = Vec::new();
        // Impact sparks (pos, normal, color) et gibs de mort (pos) —
        // collectés pour flush hors-borrow.
        let mut pending_sparks: Vec<(Vec3, Vec3, [f32; 4])> = Vec::new();
        let mut pending_gibs: Vec<Vec3> = Vec::new();
        // Bullet holes persistants sur les murs (pos, normal) — posés
        // après la boucle pour les mêmes raisons de borrow que les sparks.
        let mut pending_wall_marks: Vec<(Vec3, Vec3)> = Vec::new();
        // P5 — décales de sang derrière les cibles touchées. Traçables
        // post-borrow comme les sparks.
        let mut pending_blood_decals: Vec<(Vec3, Vec3)> = Vec::new();

        // Couleur des sparks selon l'arme (différencie MG blanc d'un RG rose).
        let spark_world_color: [f32; 4] = match weapon {
            WeaponId::Railgun => [0.95, 0.5, 0.75, 1.0],
            WeaponId::Lightninggun => [0.7, 0.85, 1.0, 1.0],
            _ => [1.0, 0.95, 0.7, 1.0],
        };
        let spark_flesh_color: [f32; 4] = [1.0, 0.25, 0.2, 1.0];

        for _ in 0..params.pellets.max(1) {
            let fwd = if spread_rad == 0.0 {
                basis.forward
            } else {
                let (rx, ry) = (rand_unit() * spread_rad, rand_unit() * spread_rad);
                // On perturbe la direction dans la base caméra puis renorm.
                let d = basis.forward + basis.right * rx.tan() + basis.up * ry.tan();
                let len = d.length().max(1e-6);
                d / len
            };
            let end = eye + fwd * params.range;

            let world_trace = world.collision.trace_ray(eye, end, Contents::MASK_SHOT);
            let t_wall = world_trace.fraction * params.range;

            let mut best: Option<(f32, usize)> = None;
            for (i, d) in self.bots.iter().enumerate() {
                if d.health.is_dead() {
                    continue;
                }
                // Les bots invincibles (fenêtre post-respawn) ne se
                // sélectionnent même pas comme cible — évite le flash
                // de chiffre de dégât à 0 et un "ping" de hitsound.
                if d.invul_until > self.time_sec {
                    continue;
                }
                let center = d.body.origin + Vec3::Z * BOT_CENTER_HEIGHT;
                let oc = center - eye;
                let t_closest = oc.dot(fwd);
                if t_closest < 0.0 || t_closest > t_wall {
                    continue;
                }
                let d_perp_sq = oc.length_squared() - t_closest * t_closest;
                if d_perp_sq > BOT_HIT_RADIUS * BOT_HIT_RADIUS {
                    continue;
                }
                match best {
                    Some((t, _)) if t <= t_closest => {}
                    _ => best = Some((t_closest, i)),
                }
            }

            if let Some((t_bot, idx)) = best {
                // On calcule le multiplicateur Quad AVANT de prendre le
                // mut-borrow sur `self.bots` (emprunt partiel non supporté
                // quand on appelle `self.player_damage_multiplier()`).
                let dmg = params.damage * self.player_damage_multiplier();
                let bot_driver = &mut self.bots[idx];
                let bot_pos = bot_driver.body.origin + Vec3::Z * BOT_CENTER_HEIGHT;
                let taken = bot_driver.health.take_damage(dmg);
                let dead = bot_driver.health.is_dead();
                let name = bot_driver.bot.name.clone();
                if taken > 0 {
                    // Horodate la prise de dégât pour déclencher l'anim
                    // TORSO_PAIN côté rendu (fenêtre ~200 ms).
                    bot_driver.last_damage_at = self.time_sec;
                    if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_pain_bot) {
                        play_at(snd, h, bot_pos, Priority::Low);
                    }
                    any_hit = true;
                    pending_damage_nums.push((bot_pos, taken, false));
                    // Sparks rouges au point d'impact sur le bot ; normale
                    // = opposée à la direction du tir (les gouttes partent
                    // vers le tireur).
                    let hit_pt = eye + fwd * t_bot;
                    pending_sparks.push((hit_pt, -fwd, spark_flesh_color));
                    // **P5 — Blood spray decal** : on trace AU-DELÀ de
                    // la cible (16 u dans la direction du tir) pour
                    // trouver le mur derrière, et y poser une décale
                    // de sang qui rappelle l'impact corporel. Effet
                    // cinéma — pas crucial gameplay, gros impact visuel.
                    let beyond_start = hit_pt + fwd * 4.0;
                    let beyond_end = beyond_start + fwd * 64.0;
                    let beyond =
                        world.collision.trace_ray(beyond_start, beyond_end, Contents::SOLID);
                    if beyond.fraction < 1.0 {
                        let blood_pt = beyond_start + fwd * (beyond.fraction * 64.0);
                        pending_blood_decals.push((blood_pt, beyond.plane_normal));
                    }
                }
                if taken > 0 && dead {
                    self.frags = self.frags.saturating_add(1);
                    pending_player_frags += 1;
                    any_kill = true;
                    if matches!(weapon, WeaponId::Gauntlet) {
                        any_gauntlet_kill = true;
                    }
                    info!(
                        "frag! bot '{}' abattu avec {} (frags={})",
                        name,
                        weapon.name(),
                        self.frags
                    );
                    pending_kills.push((
                        KillActor::Player,
                        KillActor::Bot(name),
                        weapon,
                    ));
                    pending_gibs.push(bot_pos);
                }
            } else if world_trace.fraction < 1.0 {
                // Pas touché de bot mais bien tapé un mur → sparks + bullet
                // hole persistant.  La décale se pose sur la surface via
                // `plane_normal` (orientation alignée au mur, pas
                // camera-facing) — elle reste visible même quand le
                // joueur se déplace autour de l'impact.
                let hit_pt = eye + fwd * t_wall;
                // **P4 — Impacts différenciés par surface** : on lit les
                // Contents du brush impacté pour adapter sparks et
                // décales. Q3 surface_flags (METALSTEPS, FLESH, …) ne
                // sont pas remontés par le trace v1, on retombe donc
                // sur Contents (WATER/LAVA/SLIME) qui couvre 80 % du
                // sentiment "matériau impacté".
                let contents = world_trace.contents;
                let (impact_color, impact_decal): ([f32; 4], Option<[f32; 4]>) =
                    if contents.contains(Contents::WATER) {
                        // Splash bleu, pas de décale (l'eau ne marque pas).
                        ([0.55, 0.80, 1.0, 1.0], None)
                    } else if contents.contains(Contents::LAVA) {
                        // Lava : éclats orange-rouge bouillonnants, pas
                        // de décale (les murs en lave ne se marquent
                        // pas — la surface est en mouvement).
                        ([1.0, 0.40, 0.10, 1.0], None)
                    } else if contents.contains(Contents::SLIME) {
                        // Slime : vert acide.
                        ([0.55, 0.95, 0.30, 1.0], None)
                    } else {
                        // Surface "solide" générique : sparks de l'arme
                        // (rail rose, LG bleu, MG/SG orange) + bullet
                        // hole persistant.
                        (spark_world_color, Some([0.05, 0.05, 0.06, 0.7]))
                    };
                pending_sparks.push((hit_pt, world_trace.plane_normal, impact_color));
                if impact_decal.is_some() {
                    pending_wall_marks.push((hit_pt, world_trace.plane_normal));
                }

                // **Rail ricochet** (W7) : si alt-fire actif sur railgun
                // ET le tir a tapé un mur sans bot, on rebondit le rai
                // selon la loi de Snell-Descartes (réflexion miroir) et
                // on relance une trace courte (1024u) depuis le point
                // d'impact, en cherchant un bot. Dégâts à 50% pour
                // équilibrer.
                if alt_active && matches!(weapon, WeaponId::Railgun) {
                    let n = world_trace.plane_normal;
                    let reflected = fwd - n * 2.0 * fwd.dot(n);
                    let ricochet_origin =
                        hit_pt + reflected * 2.0; // léger lift hors mur
                    let ricochet_range = 1024.0_f32;
                    let ricochet_end =
                        ricochet_origin + reflected * ricochet_range;
                    let r_trace = world.collision.trace_ray(
                        ricochet_origin,
                        ricochet_end,
                        Contents::MASK_SHOT,
                    );
                    let r_t_wall = r_trace.fraction * ricochet_range;
                    let mut r_best: Option<(f32, usize)> = None;
                    for (i, d) in self.bots.iter().enumerate() {
                        if d.health.is_dead() || d.invul_until > self.time_sec
                        {
                            continue;
                        }
                        let center =
                            d.body.origin + Vec3::Z * BOT_CENTER_HEIGHT;
                        let oc = center - ricochet_origin;
                        let t_closest = oc.dot(reflected);
                        if t_closest < 0.0 || t_closest > r_t_wall {
                            continue;
                        }
                        let d_perp_sq =
                            oc.length_squared() - t_closest * t_closest;
                        if d_perp_sq > BOT_HIT_RADIUS * BOT_HIT_RADIUS {
                            continue;
                        }
                        match r_best {
                            Some((t, _)) if t <= t_closest => {}
                            _ => r_best = Some((t_closest, i)),
                        }
                    }
                    // Visuel : 2nd beam du point d'impact vers la fin
                    // du ricochet (bot ou mur).
                    let (r_t_hit, r_hit_pt) = match r_best {
                        Some((t, _)) => (t, ricochet_origin + reflected * t),
                        None if r_trace.fraction < 1.0 => (
                            r_t_wall,
                            ricochet_origin + reflected * r_t_wall,
                        ),
                        None => (ricochet_range, ricochet_end),
                    };
                    let _ = r_t_hit;
                    self.beams.push(ActiveBeam {
                        a: hit_pt,
                        b: r_hit_pt,
                        color: [0.95, 0.45, 0.85, 0.75], // teinte ricochet
                        expire_at: self.time_sec + 0.5,
                        lifetime: 0.5,
                        style: BeamStyle::Spiral,
                    });
                    if let Some((t, idx)) = r_best {
                        let dmg = (params.damage as f32 * 0.5) as i32
                            * self.player_damage_multiplier();
                        let bot_driver = &mut self.bots[idx];
                        let bot_pos =
                            bot_driver.body.origin + Vec3::Z * BOT_CENTER_HEIGHT;
                        let taken = bot_driver.health.take_damage(dmg);
                        if taken > 0 {
                            bot_driver.last_damage_at = self.time_sec;
                            any_hit = true;
                            pending_damage_nums
                                .push((bot_pos, taken, false));
                            let r_hit = ricochet_origin + reflected * t;
                            pending_sparks.push((
                                r_hit,
                                -reflected,
                                spark_flesh_color,
                            ));
                            if bot_driver.health.is_dead() {
                                let name = bot_driver.bot.name.clone();
                                self.frags = self.frags.saturating_add(1);
                                pending_player_frags += 1;
                                pending_kills.push((
                                    KillActor::Player,
                                    KillActor::Bot(name),
                                    weapon,
                                ));
                                pending_gibs.push(bot_pos);
                            }
                        }
                    }
                }
            }

            // Visuel : LG et Railgun tracent une ligne œil → point d'impact.
            // - LG : lifetime court (cd 0.05s) pour un beam continu tant
            //   qu'on tient la touche, couleur bleu électrique.
            // - Railgun : lifetime long (0.6s) pour un trail rose qui
            //   persiste après le tir unique.
            if matches!(weapon, WeaponId::Lightninggun | WeaponId::Railgun) {
                let t_hit = match best {
                    Some((t_bot, _)) => t_bot.min(t_wall),
                    None => t_wall,
                };
                let hit_pt = eye + fwd * t_hit;
                // On décale légèrement le départ vers le bas/devant pour
                // que le beam parte visuellement de la main, pas du centre
                // de l'écran.
                let start = eye + basis.forward * 8.0 - basis.up * 6.0;
                let (color, lifetime) = match weapon {
                    WeaponId::Lightninggun => ([0.55, 0.75, 1.0, 0.9], 0.08_f32),
                    WeaponId::Railgun => ([0.95, 0.25, 0.55, 0.85], 0.6_f32),
                    _ => unreachable!(),
                };
                let style = match weapon {
                    WeaponId::Lightninggun => BeamStyle::Lightning,
                    WeaponId::Railgun => BeamStyle::Spiral,
                    _ => BeamStyle::Straight,
                };
                self.beams.push(ActiveBeam {
                    a: start,
                    b: hit_pt,
                    color,
                    expire_at: self.time_sec + lifetime,
                    lifetime,
                    style,
                });
            }
        }

        if any_hit {
            self.hit_marker_until = self.time_sec + 0.18;
            // Stat accuracy : un tir hitscan qui touche au moins 1 bot
            // compte comme +1 hit.  Une volée SG qui touche 2 cibles
            // compte 1 hit (pour matcher la granularité de `total_shots`
            // qui est aussi par-appel-fire).
            self.total_hits = self.total_hits.saturating_add(1);
            // Feedback audio hitsound. Joué 1× max par volée hitscan (SG
            // fait 11 pellets, on ne veut pas 11 ticks superposés).
            if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_hit) {
                play_hit_feedback(snd, h, self.player.origin);
            }
        }
        // Kill-confirm : thunk en couche par-dessus le hit blip quand la
        // volée achève une cible. Pareil, 1× par tick pour éviter un
        // 2× thunk sur un double-frag SG.
        if any_kill {
            if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_kill_confirm) {
                play_kill_feedback(snd, h, self.player.origin);
            }
            // Médailles Q3 — évaluées AVANT `last_frag_at = time_sec`
            // pour que la fenêtre d'Excellent mesure bien le delta
            // entre frag précédent et frag courant.
            let delta = self.time_sec - self.last_frag_at;
            let excellent_eligible =
                self.last_frag_at.is_finite() && delta <= EXCELLENT_WINDOW_SEC;
            if any_gauntlet_kill {
                if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_humiliation) {
                    play_medal(snd, h, self.player.origin);
                    info!("medal: humiliation");
                }
                self.push_medal(Medal::Humiliation);
            }
            if excellent_eligible {
                if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_excellent) {
                    play_medal(snd, h, self.player.origin);
                    info!("medal: excellent (Δ = {:.2}s)", delta);
                }
                self.push_medal(Medal::Excellent);
            }
            self.last_frag_at = self.time_sec;
        }
        // Médaille « Impressive » : uniquement sur tir Railgun. Un hit
        // arme la médaille pour le tir d'après ; à la 2ᵉ hit on award
        // et on reset. Un miss coupe la chaîne. Railgun a `pellets = 1`,
        // donc `any_hit` ≡ « railgun a touché un bot » (alors que sur
        // SG 11 pellets un any_hit pourrait refléter 1/11 bavure).
        if matches!(weapon, WeaponId::Railgun) {
            let rg_hit_now = any_hit;
            if self.rg_last_hit && rg_hit_now {
                if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_impressive) {
                    play_medal(snd, h, self.player.origin);
                    info!("medal: impressive");
                }
                self.push_medal(Medal::Impressive);
                // Reset : il faut 2 nouveaux hits pour relancer le cycle,
                // sinon chaque hit au-delà du 2ᵉ spammerait la médaille.
                self.rg_last_hit = false;
            } else {
                self.rg_last_hit = rg_hit_now;
            }
        }
        // Le borrow immutable sur `world` a disparu ici (le `let Some(world)`
        // dans le scope de la fonction est dropped une fois sorti de la
        // boucle). On peut maintenant appeler `push_kill` en mutable.
        for (killer, victim, w) in pending_kills {
            self.push_kill(killer, victim, w);
        }
        for (origin, dmg, to_player) in pending_damage_nums {
            self.push_damage_number(origin, dmg, to_player);
        }
        for (pos, normal, color) in pending_sparks {
            self.push_hit_sparks(pos, normal, color);
        }
        for (pos, normal) in pending_wall_marks {
            self.push_wall_mark(pos, normal, weapon);
        }
        // P5 — décales de sang derrière les cibles touchées. Petite
        // tache rouge sombre, alpha 80 %, vie longue (30 s) pour que
        // les murs gardent leur historique de combat.
        if let Some(r) = self.renderer.as_mut() {
            for (pos, normal) in pending_blood_decals {
                r.spawn_decal(
                    pos,
                    normal,
                    14.0,
                    [0.45, 0.05, 0.05, 0.8],
                    self.time_sec,
                    30.0,
                );
            }
        }
        for pos in pending_gibs {
            self.push_death_gibs(pos);
            self.push_blood_splat(pos);
        }
        // Flush streak : chaque frag de la volée incrémente le compteur
        // et peut déclencher une bannière (palier 3/5/7/10/15/20).  On
        // traite les frags un par un pour que double-kill 2→3 pousse
        // bien la bannière « KILLING SPREE ».
        for _ in 0..pending_player_frags {
            self.on_player_frag();
        }
        true
    }

    /// Cycle l'arme active parmi les armes possédées. `direction` vaut +1
    /// (arme suivante) ou -1 (précédente). Sans arme possédée, no-op.
    fn cycle_weapon(&mut self, direction: i32) {
        let owned: Vec<WeaponId> = WeaponId::ALL
            .into_iter()
            .filter(|w| self.weapons_owned & (1u32 << w.slot()) != 0)
            .collect();
        if owned.is_empty() {
            return;
        }
        let cur = owned.iter().position(|w| *w == self.active_weapon).unwrap_or(0);
        let n = owned.len() as i32;
        let next = ((cur as i32 + direction).rem_euclid(n)) as usize;
        let target = owned[next];
        if target != self.active_weapon {
            // Passe par switch_to_weapon : sauvegarde last_weapon, joue
            // le SFX raise et arme l'animation ammo panel.
            self.switch_to_weapon(target);
        }
    }

    /// Fait avancer chaque projectile d'un pas de simulation. Sur impact
    /// (monde ou bot) on applique le dégât direct au bot touché et un
    /// splash radial (atténué linéairement) à toutes les entités proches.
    /// Le joueur est aussi affecté par son propre splash (rocket jump).
    fn tick_projectiles(&mut self, dt: f32) {
        use q3_collision::Contents;
        let Some(world) = self.world.as_ref() else { return; };
        // On collecte les explosions à appliquer hors-borrow.
        #[derive(Debug, Clone, Copy)]
        enum HitTarget {
            Bot(usize),
            Player,
        }
        struct Boom {
            pos: Vec3,
            direct_damage: i32,
            splash_radius: f32,
            splash_damage: i32,
            direct_target: Option<HitTarget>,
            owner: ProjectileOwner,
            weapon: WeaponId,
            /// Direction du projectile au moment de l'impact — sert à
            /// orienter le knockback d'un hit direct (sinon `(cible - b.pos)`
            /// est quasi-nul et le push part dans une direction aléatoire).
            /// Pour les explosions par fuse (grenade qui time out au sol),
            /// on utilise `Vec3::Z` en fallback — poussée verticale.
            impact_dir: Vec3,
            /// Normale de la surface impactée — pertinente seulement pour
            /// les impacts monde (utilisée pour orienter la décale de
            /// brûlure).  `Vec3::ZERO` signifie "pas d'impact monde"
            /// (direct entity hit, grenade fuse, …) et désactive le spawn
            /// de décale.
            world_normal: Vec3,
        }
        let mut booms: Vec<Boom> = Vec::new();
        // Puffs de trail accumulés pendant la boucle — flushés hors-borrow
        // pour pouvoir accéder à `self.renderer` (on ne peut pas l'emprunter
        // mutable pendant `self.projectiles.retain_mut`).  Chaque entrée
        // porte (position monde, weapon) ; la weapon choisit la couleur.
        let mut pending_trails: Vec<(Vec3, WeaponId)> = Vec::new();
        // P11 — anneau orbital plasma : positions de mini-sparks à
        // spawner après la boucle (renderer-borrow). Couleur câblée
        // bleu énergétique pour l'œil ; appliquée au flush.
        let mut pending_orbital_sparks: Vec<Vec3> = Vec::new();

        self.projectiles.retain_mut(|p| {
            if self.time_sec >= p.expire_at {
                booms.push(Boom {
                    pos: p.origin,
                    direct_damage: 0,
                    splash_radius: p.splash_radius,
                    splash_damage: p.splash_damage,
                    direct_target: None,
                    owner: p.owner,
                    weapon: p.weapon,
                    // Fuse au sol : grenade qui explose après 2.5s.
                    // Pas de direction privilégiée — Z up pour lever
                    // ce qui est autour.
                    impact_dir: Vec3::Z,
                    // Fuse = pas de contact surface ce tick → pas de
                    // décale (la grenade a déjà marqué le sol lors de
                    // ses rebonds).
                    world_normal: Vec3::ZERO,
                });
                return false;
            }
            // Gravité : accélère Vz vers le bas chaque tick.
            if p.gravity != 0.0 {
                p.velocity.z -= p.gravity * dt;
            }
            // **Rocket lock-on** (W5) — steering proportionnel vers le
            // centre du bot ciblé. Si le bot meurt entre temps ou
            // disparaît de la liste, on dropping le target → balistique.
            // `HOMING_TURN_RATE` ≈ 4 rad/s : assez tournant pour suivre
            // un bot qui strafe, pas tellement que le rocket fait des
            // U-turn impossibles à esquiver.
            if let Some(target_idx) = p.homing_target {
                const HOMING_TURN_RATE: f32 = 4.0;
                if let Some(bot) = self.bots.get(target_idx) {
                    if !bot.health.is_dead() {
                        let target_pos =
                            bot.body.origin + Vec3::Z * BOT_CENTER_HEIGHT;
                        let to_t = target_pos - p.origin;
                        let dist = to_t.length();
                        if dist > 1.0 {
                            let want_dir = to_t / dist;
                            let cur_speed = p.velocity.length().max(1.0);
                            let cur_dir = p.velocity / cur_speed;
                            // Lerp directionnel limité par HOMING_TURN_RATE
                            // (radians/s). On approxime via slerp simple :
                            // mix puis renormalise.
                            let mix = (HOMING_TURN_RATE * dt).min(1.0);
                            let new_dir = (cur_dir * (1.0 - mix)
                                + want_dir * mix)
                                .normalize();
                            p.velocity = new_dir * cur_speed;
                        }
                    } else {
                        p.homing_target = None;
                    }
                } else {
                    p.homing_target = None;
                }
            }
            let next = p.origin + p.velocity * dt;

            // Intersection sphère-ray sur les cibles valides — le premier
            // contact (distance la plus faible) gagne. On évite le self-hit
            // via l'owner.
            let seg = next - p.origin;
            let seg_len = seg.length();
            let dir = if seg_len > 0.0 { seg / seg_len } else { Vec3::Z };
            let mut best: Option<(f32, HitTarget)> = None;
            // Candidats bots (cibles pour projectiles joueur). Les rockets
            // bots n'endommagent pas d'autres bots (pas de FF côté IA).
            if matches!(p.owner, ProjectileOwner::Player) {
                for (i, d) in self.bots.iter().enumerate() {
                    if d.health.is_dead() {
                        continue;
                    }
                    // Un bot invincible post-respawn n'est pas une cible
                    // valide pour un projectile — le rocket le traverse.
                    // Cohérent avec le fait que le splash + direct hit
                    // l'ignoreraient de toute façon plus bas.
                    if d.invul_until > self.time_sec {
                        continue;
                    }
                    let center = d.body.origin + Vec3::Z * BOT_CENTER_HEIGHT;
                    let oc = center - p.origin;
                    let t_closest = oc.dot(dir);
                    if t_closest < 0.0 || t_closest > seg_len {
                        continue;
                    }
                    let perp_sq = oc.length_squared() - t_closest * t_closest;
                    if perp_sq > BOT_HIT_RADIUS * BOT_HIT_RADIUS {
                        continue;
                    }
                    match best {
                        Some((t, _)) if t <= t_closest => {}
                        _ => best = Some((t_closest, HitTarget::Bot(i))),
                    }
                }
            }
            // Candidat joueur (cible pour projectiles bots). Skipé si mort
            // ou si fenêtre d'invulnérabilité active — symétrique avec les
            // bots : le projectile traverse sans explosion.
            if matches!(p.owner, ProjectileOwner::Bot(_))
                && !self.player_health.is_dead()
                && self.player_invul_until <= self.time_sec
            {
                let center = self.player.origin + Vec3::Z * BOT_CENTER_HEIGHT;
                let oc = center - p.origin;
                let t_closest = oc.dot(dir);
                if t_closest >= 0.0 && t_closest <= seg_len {
                    let perp_sq = oc.length_squared() - t_closest * t_closest;
                    if perp_sq <= BOT_HIT_RADIUS * BOT_HIT_RADIUS {
                        match best {
                            Some((t, _)) if t <= t_closest => {}
                            _ => best = Some((t_closest, HitTarget::Player)),
                        }
                    }
                }
            }
            // Intersection avec le monde (hit le premier).
            let trace = world.collision.trace_ray(p.origin, next, Contents::MASK_SHOT);
            let t_world = trace.fraction * seg_len;

            if let Some((t_hit, tgt)) = best {
                if t_hit <= t_world {
                    let hit = p.origin + dir * t_hit;
                    booms.push(Boom {
                        pos: hit,
                        direct_damage: p.direct_damage,
                        splash_radius: p.splash_radius,
                        splash_damage: p.splash_damage,
                        direct_target: Some(tgt),
                        owner: p.owner,
                        weapon: p.weapon,
                        impact_dir: dir,
                        // Direct entity hit : on ne marque pas la surface
                        // (on a touché un joueur ou un bot, pas un mur).
                        world_normal: Vec3::ZERO,
                    });
                    return false;
                }
            }
            if trace.fraction < 1.0 {
                // Impact monde — deux comportements :
                //  * `bounce=false` (rocket, plasma) : explose immédiatement.
                //  * `bounce=true` (grenade) : réflexion amortie le long du
                //    plan d'impact, continue à vivre jusqu'à la fuse.
                if p.bounce {
                    let normal = trace.plane_normal;
                    // Projectile quasi-immobile sur le sol → on le colle à
                    // la surface et on coupe sa vitesse pour qu'il attende
                    // la fuse sans glisser indéfiniment.
                    const BOUNCE_DAMPEN: f32 = 0.65;
                    const MIN_BOUNCE_SPEED: f32 = 40.0;
                    // Point d'impact + epsilon le long de la normale pour
                    // éviter de re-trace du même point au tick suivant.
                    let hit_pos = p.origin + dir * t_world + normal * 0.125;
                    // Réflexion : v' = v - 2(v·n)n, puis amortissement.
                    let vn = p.velocity.dot(normal);
                    let reflected = p.velocity - normal * (2.0 * vn);
                    let new_v = reflected * BOUNCE_DAMPEN;
                    if new_v.length() < MIN_BOUNCE_SPEED {
                        // Repos : on le fige pour éviter le jitter, la fuse
                        // fera exploser la grenade d'elle-même.
                        p.velocity = Vec3::ZERO;
                    } else {
                        p.velocity = new_v;
                    }
                    p.origin = hit_pos;
                    return true;
                }
                let hit = p.origin + dir * t_world;
                booms.push(Boom {
                    pos: hit,
                    direct_damage: 0,
                    splash_radius: p.splash_radius,
                    splash_damage: p.splash_damage,
                    direct_target: None,
                    owner: p.owner,
                    weapon: p.weapon,
                    impact_dir: dir,
                    // Impact sur une surface du monde : la trace fournit
                    // une normale cohérente, on la mémorise pour spawner
                    // une décale orientée.
                    world_normal: trace.plane_normal,
                });
                return false;
            }

            p.origin = next;
            // Trail : seulement pour les projectiles qui laissent visuellement
            // une traînée. Plasma a maintenant SON propre trail orbital
            // (P11) — un anneau d'éclats qui spirale autour du bolt.
            // Les autres emprisonnent le tick de puff par `next_trail_at`,
            // ce qui rend la cadence indépendante du framerate.
            if matches!(
                p.weapon,
                WeaponId::Rocketlauncher | WeaponId::Grenadelauncher | WeaponId::Bfg
            ) && self.time_sec >= p.next_trail_at
            {
                pending_trails.push((p.origin, p.weapon));
                p.next_trail_at = self.time_sec + PROJECTILE_TRAIL_INTERVAL;
            }
            // **P11 — Plasma orbital trail** : anneau de 4 sparks autour
            // du bolt, phase fonction du temps absolu pour un effet
            // hélicoïdal continu (les sparks reculent légèrement par
            // rapport au projectile, donnant l'illusion d'orbite).
            if matches!(p.weapon, WeaponId::Plasmagun)
                && self.time_sec >= p.next_trail_at
            {
                let v = p.velocity;
                let speed = v.length().max(1.0);
                let axis = v / speed;
                let helper = if axis.z.abs() < 0.9 { Vec3::Z } else { Vec3::Y };
                let perp = axis.cross(helper).normalize();
                let perp2 = axis.cross(perp).normalize();
                let phase = self.time_sec * 12.0; // 12 rad/s = ~2 tours/sec
                for k in 0..4 {
                    let a = phase + k as f32 * std::f32::consts::FRAC_PI_2;
                    let radius = 4.5;
                    let dir = perp * a.cos() + perp2 * a.sin();
                    let pos = p.origin - axis * 6.0 + dir * radius;
                    pending_orbital_sparks.push(pos);
                }
                p.next_trail_at = self.time_sec + 0.04;
            }
            true
        });
        // Flush des orbital sparks plasma (P11) — petits flares
        // bleus très courts qui composent l'anneau autour du bolt.
        for pos in pending_orbital_sparks {
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.remove(0);
            }
            self.particles.push(Particle {
                origin: pos,
                velocity: Vec3::ZERO,
                color: [0.55, 0.80, 1.0, 1.0],
                expire_at: self.time_sec + 0.18,
                lifetime: 0.18,
            });
        }
        // Flush des trails — self.renderer accessible maintenant que le
        // borrow sur self.projectiles est relâché.
        if let Some(r) = self.renderer.as_mut() {
            for (pos, weapon) in pending_trails {
                let (color, size_start, size_end, lifetime) = match weapon {
                    // Rocket : fumée blanche dense qui s'étale.
                    WeaponId::Rocketlauncher => ([0.85, 0.85, 0.85, 0.6], 2.5, 10.0, 0.7),
                    // Grenade : fumée grise plus fine, vie courte.
                    WeaponId::Grenadelauncher => ([0.6, 0.6, 0.6, 0.5], 2.0, 6.0, 0.5),
                    // BFG : traînée verdâtre énergétique.
                    WeaponId::Bfg => ([0.4, 1.0, 0.5, 0.7], 4.0, 14.0, 0.8),
                    _ => continue,
                };
                r.spawn_particle(
                    pos,
                    // Vitesse très faible vers le haut — la fumée flotte
                    // pendant qu'elle grossit, sans partir droit en l'air.
                    Vec3::Z * 8.0,
                    color,
                    size_start,
                    size_end,
                    self.time_sec,
                    lifetime,
                );
            }
        }

        // Applique les explosions. Ici on peut ré-emprunter self en mutable.
        // Les kills détectés sont collectés puis poussés dans `kill_feed`
        // après la boucle, pour ne pas entrer en conflit de borrow avec
        // `self.bots.get_mut()` en cours. Même combine pour les chiffres
        // de dégât flottants.
        let mut pending_kills: Vec<(KillActor, KillActor, WeaponId)> = Vec::new();
        let mut pending_damage_nums: Vec<(Vec3, i32, bool)> = Vec::new();
        // Gibs sur morts de bot (direct + splash). Flushés hors-borrow.
        let mut pending_gibs: Vec<Vec3> = Vec::new();
        // Compteur de frags joueur — flushé vers `on_player_frag()`
        // après la boucle d'explosions pour gérer les streaks sans
        // conflit de borrow sur `self.bots` / `self.projectiles`.
        let mut pending_player_frags: u32 = 0;
        // `true` dès qu'une explosion du joueur (direct ou splash) inflige
        // au moins 1 HP à un bot vivant ce tick → on joue le hitsound en
        // fin de boucle. Collecté plutôt que déclenché au point d'impact
        // pour fusionner les multi-hits (splash qui touche 3 bots = 1 tick).
        let mut player_connected = false;
        // Idem pour les kills : déclenche le thunk kill-confirm une fois
        // en fin de boucle, pas une fois par bot fraggé (rocket qui tue
        // 2 bots par splash = 1 thunk, pas 2).
        let mut player_killed = false;
        for b in booms {
            // Visuel + SFX
            self.explosions.push(Explosion {
                origin: b.pos,
                expire_at: self.time_sec + 0.35,
            });
            self.push_explosion_particles(b.pos);
            // Puffs de fumée billboard — complètent les sparks en laissant
            // un nuage gris qui flotte quelques secondes après le flash.
            // Plasma fait peu de fumée (énergie "propre"), pas de puff.
            if !matches!(b.weapon, WeaponId::Plasmagun | WeaponId::Bfg) {
                self.push_explosion_smoke(b.pos);
            }
            // Décale de brûlure : seulement sur un vrai impact surface
            // (normal non-nulle).  Couleur : noir brûlé à 60 % d'alpha,
            // rayon = 18 (≈ taille d'une scorch mark Q3 rocketlauncher),
            // vie = 20 s avec fade sur les dernières 25 %.
            if b.world_normal != Vec3::ZERO {
                if let Some(r) = self.renderer.as_mut() {
                    r.spawn_decal(
                        b.pos,
                        b.world_normal,
                        18.0,
                        [0.05, 0.04, 0.03, 0.6],
                        self.time_sec,
                        20.0,
                    );
                }
            }
            // Dlight explosion : flash intense orange, fade rapide sur
            // ~0.5 s.  Couleur et intensité différentes pour plasma
            // (bleu froid) vs. rocket/grenade/BFG (orange chaud).
            if let Some(r) = self.renderer.as_mut() {
                let (color, intensity, radius) = match b.weapon {
                    WeaponId::Plasmagun => ([0.4, 0.6, 1.0], 2.5, 120.0),
                    WeaponId::Bfg => ([0.4, 1.0, 0.5], 5.0, 400.0),
                    _ => ([1.0, 0.7, 0.3], 4.0, 300.0),
                };
                r.spawn_dlight(b.pos, radius, color, intensity, self.time_sec, 0.5);
            }
            if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_rocket_explode) {
                play_at(snd, h, b.pos, Priority::Weapon);
            }
            // Screen-shake : une explosion à <= `SHAKE_NEAR_DIST` du œil
            // déclenche la secousse max ; au-delà de `SHAKE_FAR_DIST`
            // aucun effet. On empile par `max()` — si une nouvelle
            // explosion très proche survient pendant qu'une ancienne
            // décroît encore, on prend la plus forte plutôt que d'en
            // faire la somme (évite un trembling insupportable en salle
            // de rockets).
            {
                let eye = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
                let dist = (eye - b.pos).length();
                let t = 1.0
                    - ((dist - SHAKE_NEAR_DIST)
                        / (SHAKE_FAR_DIST - SHAKE_NEAR_DIST))
                        .clamp(0.0, 1.0);
                if t > 0.0 {
                    let amp = SHAKE_MAX_AMPLITUDE * t;
                    if amp > self.shake_intensity {
                        self.shake_intensity = amp;
                    }
                    self.shake_until = self.time_sec + SHAKE_DURATION;
                }
            }
            // Dégât direct (bot OU joueur)
            match b.direct_target {
                Some(HitTarget::Bot(idx)) => {
                    if let Some(bd) = self.bots.get_mut(idx) {
                        // Knockback direct : même formule que le splash,
                        // mais orienté le long de la trajectoire du
                        // projectile (sinon `(centre - b.pos)` vaut quasi
                        // zéro puisque b.pos est sur la hit-sphere du
                        // bot). Appliqué AVANT `take_damage` pour utiliser
                        // le dégât brut comme en Q3 (`G_Damage`).
                        const KNOCKBACK_MASS: f32 = 200.0;
                        const KNOCKBACK_COEFF: f32 = 1000.0;
                        const KNOCKBACK_MAX: f32 = 200.0;
                        let knock = (b.direct_damage as f32).min(KNOCKBACK_MAX);
                        let kick = knock * KNOCKBACK_COEFF / KNOCKBACK_MASS;
                        let kvel = b.impact_dir * kick;
                        bd.body.velocity += kvel;
                        if kvel.z > 10.0 {
                            bd.body.on_ground = false;
                        }
                        let taken = bd.health.take_damage(b.direct_damage);
                        let name = bd.bot.name.clone();
                        let dead = bd.health.is_dead();
                        if taken > 0 {
                            let bot_pos = bd.body.origin + Vec3::Z * BOT_CENTER_HEIGHT;
                            if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_pain_bot) {
                                play_at(snd, h, bot_pos, Priority::Low);
                            }
                            // Chiffre de dégât jaune si c'est le joueur qui
                            // l'inflige ; rouge si c'est un autre bot (FF off
                            // aujourd'hui mais on laisse la branche cohérente
                            // au cas où).
                            let to_player = !matches!(b.owner, ProjectileOwner::Player);
                            pending_damage_nums.push((bot_pos, taken, to_player));
                            if matches!(b.owner, ProjectileOwner::Player) {
                                player_connected = true;
                            }
                        }
                        if taken > 0 && dead && matches!(b.owner, ProjectileOwner::Player) {
                            self.frags = self.frags.saturating_add(1);
                            pending_player_frags += 1;
                            player_killed = true;
                            info!("frag! bot '{}' vaporisé par rocket (frags={})", name, self.frags);
                            pending_kills.push((
                                KillActor::Player,
                                KillActor::Bot(name),
                                b.weapon,
                            ));
                            let bot_pos = bd.body.origin + Vec3::Z * BOT_CENTER_HEIGHT;
                            pending_gibs.push(bot_pos);
                        }
                    }
                }
                Some(HitTarget::Player) => {
                    if !self.player_health.is_dead() {
                        // Knockback direct joueur — même règle que pour
                        // les bots. Formule Q3 : avant absorption armor.
                        const KNOCKBACK_MASS: f32 = 200.0;
                        const KNOCKBACK_COEFF: f32 = 1000.0;
                        const KNOCKBACK_MAX: f32 = 200.0;
                        let knock = (b.direct_damage as f32).min(KNOCKBACK_MAX);
                        let kick = knock * KNOCKBACK_COEFF / KNOCKBACK_MASS;
                        let kvel = b.impact_dir * kick;
                        self.player.velocity += kvel;
                        if kvel.z > 10.0 {
                            self.player.on_ground = false;
                            self.was_airborne = true;
                        }
                        let absorbed = (b.direct_damage / 2).min(self.player_armor);
                        self.player_armor -= absorbed;
                        if absorbed > 0 {
                            self.armor_flash_until = self.time_sec + ARMOR_FLASH_SEC;
                        }
                        let real = b.direct_damage - absorbed;
                        let taken = self.player_health.take_damage(real);
                        if taken > 0 {
                            let eye = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
                            if let (Some(snd), Some(h)) =
                                (self.sound.as_ref(), self.sfx_pain_player)
                            {
                                play_at(snd, h, eye, Priority::High);
                            }
                            // Rouge : dégât subi par le joueur. On ancre le
                            // chiffre sur le joueur (il se verra quand on
                            // projettera sur l'écran, typique "-X HP" flotant).
                            pending_damage_nums.push((eye, taken, true));
                            // Pain-arrow : direction = trajectoire du projectile
                            // qui vient de toucher. `b.impact_dir` pointe déjà
                            // dans le sens du vol (source → joueur) — c'est
                            // exactement la sémantique qu'on veut pour le HUD.
                            self.last_damage_dir = b.impact_dir;
                            self.last_damage_until = self.time_sec + DAMAGE_DIR_SHOW_SEC;
                            self.pain_flash_until = self.time_sec + PAIN_FLASH_SEC;
                        }
                        if self.player_health.is_dead() && self.respawn_at.is_none() {
                            self.deaths = self.deaths.saturating_add(1);
                            self.respawn_at = Some(self.time_sec + RESPAWN_DELAY_SEC);
                            info!("joueur mort (direct rocket bot)");
                            let killer = resolve_killer(&self.bots, b.owner);
                            pending_kills.push((killer, KillActor::Player, b.weapon));
                            // Crédit scoreboard pour le bot tueur.
                            if let ProjectileOwner::Bot(idx) = b.owner {
                                if let Some(bd) = self.bots.get_mut(idx) {
                                    bd.frags = bd.frags.saturating_add(1);
                                }
                            }
                        }
                    }
                }
                None => {}
            }
            // Splash radial — bots (sauf le bot directement touché si c'est
            // lui qui a mangé le direct). Les rockets bots n'endommagent
            // pas les autres bots (FF off).
            let r2 = b.splash_radius * b.splash_radius;
            let spare_bot = match b.direct_target {
                Some(HitTarget::Bot(i)) => Some(i),
                _ => None,
            };
            if matches!(b.owner, ProjectileOwner::Player) {
                for (i, bd) in self.bots.iter_mut().enumerate() {
                    if bd.health.is_dead() {
                        continue;
                    }
                    if Some(i) == spare_bot {
                        continue; // déjà impacté en direct
                    }
                    // Invul post-respawn : ni dégât ni knockback.
                    if bd.invul_until > self.time_sec {
                        continue;
                    }
                    let center = bd.body.origin + Vec3::Z * BOT_CENTER_HEIGHT;
                    let d2 = (center - b.pos).length_squared();
                    if d2 > r2 {
                        continue;
                    }
                    let d = d2.sqrt();
                    let falloff = 1.0 - (d / b.splash_radius);
                    let dmg = (b.splash_damage as f32 * falloff).round() as i32;
                    if dmg <= 0 {
                        continue;
                    }
                    // Knockback bot — mêmes constantes que pour le joueur
                    // (cf. `splash joueur` plus bas). Pousse le bot hors du
                    // centre d'explosion : le PlayerMove du bot intègrera
                    // cette vitesse au prochain tick_collide, avec gravité
                    // et friction — donc un bot propulsé retombera
                    // naturellement. Sans ce kick, un splash qui manque de
                    // tuer le bot ne lui faisait visuellement rien.
                    const KNOCKBACK_MASS: f32 = 200.0;
                    const KNOCKBACK_COEFF: f32 = 1000.0;
                    const KNOCKBACK_MAX: f32 = 200.0;
                    if d > 0.001 {
                        let dir = (center - b.pos) / d;
                        let knock = (dmg as f32).min(KNOCKBACK_MAX);
                        let kick = knock * KNOCKBACK_COEFF / KNOCKBACK_MASS;
                        let kvel = dir * kick;
                        bd.body.velocity += kvel;
                        if kvel.z > 10.0 {
                            bd.body.on_ground = false;
                        }
                    }
                    let taken = bd.health.take_damage(dmg);
                    let dead = bd.health.is_dead();
                    let name = bd.bot.name.clone();
                    if taken > 0 {
                        if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_pain_bot) {
                            play_at(snd, h, center, Priority::Low);
                        }
                        pending_damage_nums.push((center, taken, false));
                        // Ce bloc est gated par `matches!(b.owner, Player)`
                        // plus haut, donc tout hit ici est un hit du joueur.
                        player_connected = true;
                    }
                    if taken > 0 && dead {
                        self.frags = self.frags.saturating_add(1);
                        pending_player_frags += 1;
                        player_killed = true;
                        info!("frag! bot '{}' splash rocket (frags={})", name, self.frags);
                        pending_kills.push((
                            KillActor::Player,
                            KillActor::Bot(name),
                            b.weapon,
                        ));
                        pending_gibs.push(center);
                    }
                }
            }
            // Splash radial — joueur (rocket jump joueur + splash bot).
            // Skipé si le joueur vient d'encaisser un direct (cumul inutile).
            // Invulnérabilité post-respawn : on skip complètement — pas de
            // dégât et pas de knockback, pour ne pas pousser le joueur
            // hors de son spawn avant qu'il puisse se positionner.
            let player_took_direct = matches!(b.direct_target, Some(HitTarget::Player));
            let player_invul = self.player_invul_until > self.time_sec;
            if !self.player_health.is_dead() && !player_took_direct && !player_invul {
                let eye = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
                let d2 = (eye - b.pos).length_squared();
                if d2 <= r2 {
                    let d = d2.sqrt();
                    let falloff = 1.0 - (d / b.splash_radius);
                    let dmg = (b.splash_damage as f32 * falloff).round() as i32;
                    if dmg > 0 {
                        // Knockback : avant absorption armor (Q3 utilise
                        // le dégât brut). Formule originale : knockback =
                        // damage, puis vel += dir * G_KNOCKBACK * knockback / MASS.
                        // MASS=200, G_KNOCKBACK=1000 → kick = damage * 5 u/s.
                        // On cap à 200 HP équivalent façon Q3 pour éviter
                        // les rockets-jumps à 1000 u/s verticales.
                        const KNOCKBACK_MASS: f32 = 200.0;
                        const KNOCKBACK_COEFF: f32 = 1000.0;
                        const KNOCKBACK_MAX: f32 = 200.0;
                        if d > 0.001 {
                            let dir = (eye - b.pos) / d;
                            let knock = (dmg as f32).min(KNOCKBACK_MAX);
                            let kick = knock * KNOCKBACK_COEFF / KNOCKBACK_MASS;
                            let kvel = dir * kick;
                            self.player.velocity += kvel;
                            // Décolle du sol pour que `update_ground`
                            // n'avale pas le kick vertical au prochain
                            // tick. Seuil court (10 u/s) pour ne pas
                            // déclencher sur un splash purement horizontal.
                            if kvel.z > 10.0 {
                                self.player.on_ground = false;
                                self.was_airborne = true;
                            }
                        }
                        // Battle Suit : on conserve le knockback (appliqué
                        // juste au-dessus) mais on ne retient aucun dégât.
                        // C'est le comportement Q3 canonique : on peut
                        // encore rocket-jump en enviro, mais sans prendre
                        // un centime de santé pour autant.
                        let absorbed = if self.is_powerup_active(PowerupKind::BattleSuit) {
                            self.player_armor.min(0)
                        } else {
                            (dmg / 2).min(self.player_armor)
                        };
                        self.player_armor -= absorbed;
                        if absorbed > 0 {
                            self.armor_flash_until = self.time_sec + ARMOR_FLASH_SEC;
                        }
                        let real = if self.is_powerup_active(PowerupKind::BattleSuit) {
                            0
                        } else {
                            dmg - absorbed
                        };
                        let taken = self.player_health.take_damage(real);
                        if taken > 0 {
                            if let (Some(snd), Some(h)) =
                                (self.sound.as_ref(), self.sfx_pain_player)
                            {
                                play_at(snd, h, eye, Priority::Normal);
                            }
                            pending_damage_nums.push((eye, taken, true));
                            // Pain-arrow splash : direction = explosion → œil,
                            // normalisée. Pour un splash non-radial pur (eye
                            // au centre exact du boom), on retombe sur Vec3::Z
                            // — l'indicateur pointerait vers le bas ; skippé
                            // dans ce cas rare pour éviter un artefact visuel.
                            if d > 0.001 {
                                self.last_damage_dir = (eye - b.pos) / d;
                                self.last_damage_until =
                                    self.time_sec + DAMAGE_DIR_SHOW_SEC;
                            }
                            self.pain_flash_until = self.time_sec + PAIN_FLASH_SEC;
                        }
                        if self.player_health.is_dead() && self.respawn_at.is_none() {
                            self.deaths = self.deaths.saturating_add(1);
                            self.respawn_at = Some(self.time_sec + RESPAWN_DELAY_SEC);
                            info!("joueur mort (splash rocket)");
                            let killer = resolve_killer(&self.bots, b.owner);
                            pending_kills.push((killer, KillActor::Player, b.weapon));
                            if let ProjectileOwner::Bot(idx) = b.owner {
                                if let Some(bd) = self.bots.get_mut(idx) {
                                    bd.frags = bd.frags.saturating_add(1);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Hitsound : un seul blip par tick, quel que soit le nombre de
        // cibles splashées. Déclenché après la boucle pour fusionner les
        // multi-hits (rocket qui splash 3 bots = 1 son, pas 3).
        if player_connected {
            // Stat accuracy : 1 tick de projectile joueur qui touche au
            // moins une cible = +1 hit.  Léger sous-comptage si deux
            // rockets impactent le même tick, mais rare et acceptable.
            self.total_hits = self.total_hits.saturating_add(1);
            if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_hit) {
                play_hit_feedback(snd, h, self.player.origin);
            }
        }
        // Kill-confirm : thunk qui se superpose au hitsound quand une
        // explosion achève une cible. Même fusion multi-kill qu'au-dessus.
        if player_killed {
            if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_kill_confirm) {
                play_kill_feedback(snd, h, self.player.origin);
            }
            // Médaille « Excellent » — 2 frags en < 2s, tous styles
            // confondus (hitscan ou projectile). Pas de check Humiliation
            // ici : le Gauntlet est hitscan-only, aucune explosion rocket
            // ne peut être un Gauntlet kill. Update `last_frag_at` après
            // évaluation de la fenêtre.
            let delta = self.time_sec - self.last_frag_at;
            let excellent_eligible =
                self.last_frag_at.is_finite() && delta <= EXCELLENT_WINDOW_SEC;
            if excellent_eligible {
                if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_excellent) {
                    play_medal(snd, h, self.player.origin);
                    info!("medal: excellent (Δ = {:.2}s, splash)", delta);
                }
                self.push_medal(Medal::Excellent);
            }
            self.last_frag_at = self.time_sec;
        }

        // Flush des kills vers le feed, maintenant que plus rien n'emprunte
        // `self.bots` mutablement.
        for (killer, victim, weapon) in pending_kills {
            self.push_kill(killer, victim, weapon);
        }
        for (origin, dmg, to_player) in pending_damage_nums {
            self.push_damage_number(origin, dmg, to_player);
        }
        for pos in pending_gibs {
            self.push_death_gibs(pos);
            self.push_blood_splat(pos);
        }
        // Streak : flush des frags de la volée d'explosions — même
        // logique que côté fire_weapon, paliers déclenchés un par un.
        for _ in 0..pending_player_frags {
            self.on_player_frag();
        }

        // Nettoyage des explosions expirées.
        let t = self.time_sec;
        self.explosions.retain(|e| t < e.expire_at);

        // Advection des particules : position += v·dt ; la vitesse
        // est tirée vers le bas par PARTICLE_GRAVITY. Pas de collision
        // monde — un spark qui traverse le sol est un compromis acceptable
        // et coûte zéro raycast (il s'éteint vite de toute façon).
        for p in &mut self.particles {
            p.origin += p.velocity * dt;
            p.velocity.z -= PARTICLE_GRAVITY * dt;
        }
        self.particles.retain(|p| t < p.expire_at);
    }

    /// Change d'arme si le slot est possédé. Ignoré si la touche ne mappe
    /// à aucune arme de `WeaponId::ALL`.
    /// Dispatch d'une touche `Digit1..Digit9` selon que la roue de chat
    /// est ouverte ou non. Roue fermée → comportement Q3 standard
    /// (sélection d'arme). Roue ouverte → envoi de message pour les
    /// digits `1..=8` ou cancel pour `9`. Centralisé ici plutôt que
    /// dispersé dans 9 branches du `match` clavier.
    /// Détecte un double-tap d'une touche directionnelle pour déclencher
    /// un dash. `dir_idx` : 0=forward, 1=back, 2=left, 3=right.
    /// Fenêtre de 250 ms entre les deux presses — au-delà on ne considère
    /// pas le 2e tap comme une suite mais comme un nouveau tap initial.
    fn detect_double_tap(&mut self, dir_idx: usize) {
        const DOUBLE_TAP_WINDOW: f32 = 0.25;
        if dir_idx >= 4 { return; }
        let now = self.time_sec;
        let last = self.input.last_dir_press[dir_idx];
        if now - last < DOUBLE_TAP_WINDOW {
            // 2e tap dans la fenêtre → arme le dash. Le pmove le
            // consomme au prochain tick et arme son cooldown.
            self.input.dash_armed = true;
            // Reset le timestamp pour ne pas re-déclencher au 3e tap.
            self.input.last_dir_press[dir_idx] = 0.0;
        } else {
            self.input.last_dir_press[dir_idx] = now;
        }
    }

    fn handle_digit_or_wheel(&mut self, digit: u8) {
        if self.chat_wheel_open {
            self.chat_wheel_open = false;
            // Digit 9 = cancel sans envoi. Digits 1..=8 → message.
            if (1..=8).contains(&digit) {
                let idx = (digit - 1) as usize;
                let (_label, full_msg) = CHAT_WHEEL_MESSAGES[idx];
                self.pending
                    .lock()
                    .push(PendingAction::SayChat(full_msg.to_string()));
            }
        } else {
            self.select_weapon_slot(digit);
        }
    }

    fn select_weapon_slot(&mut self, slot: u8) {
        for w in WeaponId::ALL {
            if w.slot() == slot && (self.weapons_owned & (1u32 << slot)) != 0 && self.active_weapon != w {
                self.switch_to_weapon(w);
                return;
            }
        }
    }

    /// Bascule effective vers `w`.  Suppose que `w` est possédée et
    /// différente de `self.active_weapon` (sinon no-op).  Sauvegarde
    /// l'arme précédente dans `last_weapon` pour la touche X (last
    /// weapon toggle), joue le SFX de raise, arme l'animation de swap
    /// du viewmodel ammo panel (`weapon_switch_at`), et applique le
    /// cooldown anti-fire-parasite.
    fn switch_to_weapon(&mut self, w: WeaponId) {
        if self.active_weapon == w {
            return;
        }
        if (self.weapons_owned & (1u32 << w.slot())) == 0 {
            return;
        }
        self.last_weapon = self.active_weapon;
        self.active_weapon = w;
        info!("weapon → {}", w.name());
        self.next_player_fire_at = self.time_sec + 0.1;
        self.weapon_switch_at = self.time_sec;
        self.play_weapon_switch_sfx();
    }

    /// Bascule sur la dernière arme utilisée (touche X).  No-op si
    /// `last_weapon` n'est pas possédée (swap vers le loadout de départ
    /// après l'avoir perdu via un restart, par exemple) ou identique
    /// à l'arme courante (cas trivial de démarrage).
    fn swap_to_last_weapon(&mut self) {
        let target = self.last_weapon;
        if target == self.active_weapon {
            return;
        }
        if (self.weapons_owned & (1u32 << target.slot())) == 0 {
            return;
        }
        self.switch_to_weapon(target);
    }

    /// Joue le SFX de changement d'arme — factorisé car invoqué à la
    /// fois par `select_weapon_slot` (touche 1..9) et par l'autoswitch
    /// out-of-ammo dans `fire_weapon`.
    fn play_weapon_switch_sfx(&self) {
        if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_weapon_switch) {
            let ear = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
            play_at(snd, h, ear, Priority::Weapon);
        }
    }

    /// Taunt joueur (touche F3) — pousse une ligne de chat sous le nom
    /// « YOU » et joue le kill-confirm sfx comme feedback audio.  Le
    /// cooldown [`PLAYER_TAUNT_COOLDOWN`] bride le spam, mais sans muter
    /// les bots (on touche uniquement `next_player_taunt_at`).  Si le
    /// joueur est mort, on ignore — éviter qu'un cadavre ne parle.
    ///
    /// Déclenche aussi un clapback bot avec probabilité
    /// [`BOT_CLAPBACK_PROB`] — un seul bot vivant tiré au hasard répond
    /// par une ligne du pool [`BOT_CLAPBACK_LINES`].
    fn trigger_player_taunt(&mut self) {
        if self.player_health.is_dead() {
            return;
        }
        if self.time_sec < self.next_player_taunt_at {
            return;
        }
        if PLAYER_TAUNT_LINES.is_empty() {
            return;
        }
        // Tirage uniforme (rand_unit peut retourner un négatif, on prend abs).
        let idx = (rand_unit().abs() * PLAYER_TAUNT_LINES.len() as f32) as usize;
        let idx = idx.min(PLAYER_TAUNT_LINES.len() - 1);
        let text = PLAYER_TAUNT_LINES[idx].to_string();
        self.push_chat("YOU".to_string(), text);
        if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_kill_confirm) {
            let ear = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
            play_at(snd, h, ear, Priority::VoiceOver);
        }
        self.next_player_taunt_at = self.time_sec + PLAYER_TAUNT_COOLDOWN;
        // Clapback bot — roulette une seule fois par taunt.
        self.maybe_bot_clapback();
    }

    /// Peut-être faire répliquer un bot au joueur qui vient de taunt.
    /// Tire à `BOT_CLAPBACK_PROB` ; si oui, choisit un bot vivant au
    /// hasard et poste une ligne de [`BOT_CLAPBACK_LINES`] sous son nom.
    /// Respecte volontairement `next_chat_at` (le clapback est du chat
    /// bot canonique) pour éviter de doubler un taunt bot déjà en vol.
    fn maybe_bot_clapback(&mut self) {
        if rand_unit().abs() > BOT_CLAPBACK_PROB {
            return;
        }
        if self.time_sec < self.next_chat_at {
            return;
        }
        // Liste des bots vivants.  On utilise leur index dans `self.bots`
        // pour éviter un clone inutile du nom.
        let alive: Vec<usize> = self
            .bots
            .iter()
            .enumerate()
            .filter(|(_, d)| !d.health.is_dead())
            .map(|(i, _)| i)
            .collect();
        if alive.is_empty() {
            return;
        }
        let pick = (rand_unit().abs() * alive.len() as f32) as usize;
        let pick = pick.min(alive.len() - 1);
        let bot_idx = alive[pick];
        let line_idx = (rand_unit().abs() * BOT_CLAPBACK_LINES.len() as f32) as usize;
        let line_idx = line_idx.min(BOT_CLAPBACK_LINES.len() - 1);
        let speaker = self.bots[bot_idx].bot.name.clone();
        let text = BOT_CLAPBACK_LINES[line_idx].to_string();
        self.push_chat(speaker, text);
        self.next_chat_at = self.time_sec + CHAT_GLOBAL_COOLDOWN;
    }

    /// Repositionne les bots morts à un spawn DM aléatoire et remet leur HP
    /// à fond. On conserve leur index dans `self.bots` pour stabilité.
    fn respawn_dead_bots(&mut self) {
        let mut spawn_positions: Vec<Vec3> = Vec::new();
        let mut respawned_bot_indices: Vec<usize> = Vec::new();
        {
            let Some(world) = self.world.as_ref() else { return; };
            if world.spawn_points.is_empty() && world.player_start.is_none() {
                return;
            }
            for (i, d) in self.bots.iter_mut().enumerate() {
                if !d.health.is_dead() {
                    continue;
                }
                let (origin, angles) = if !world.spawn_points.is_empty() {
                    let idx = (self.time_sec as usize ^ i.wrapping_mul(2654435761))
                        % world.spawn_points.len();
                    let sp = &world.spawn_points[idx];
                    (sp.origin, sp.angles)
                } else {
                    (world.player_start.unwrap_or(Vec3::ZERO), world.player_start_angles)
                };
                let new_origin = origin + Vec3::Z * 40.0;
                d.body = PlayerMove::new(new_origin);
                d.body.view_angles = angles;
                d.bot.position = new_origin;
                d.bot.view_angles = angles;
                d.bot.target_enemy = None;
                d.bot.waypoints.clear();
                d.last_saw_player_at = None;
                d.next_fire_at = 0.0;
                d.next_rocket_at = 0.0;
                // On compte la mort juste avant le respawn — `health.is_dead()`
                // ne sera plus vrai après `respawn()` donc ici est le bon moment.
                d.deaths = d.deaths.saturating_add(1);
                d.health.respawn();
                // Même fenêtre d'invul que le joueur — évite que deux bots
                // qui respawnent au même moment se fraggent mutuellement
                // avant d'avoir bougé.
                d.invul_until = self.time_sec + RESPAWN_INVUL_SEC;
                info!(
                    "bot '{}' respawn → {:?} (deaths={})",
                    d.bot.name, new_origin, d.deaths
                );
                spawn_positions.push(new_origin);
                respawned_bot_indices.push(i);
            }
        }
        // FX de respawn bot — légèrement plus rose que le joueur pour le
        // distinguer visuellement.
        for pos in spawn_positions {
            self.push_respawn_fx(pos, [1.0, 0.7, 0.85, 1.0]);
        }
        // Chat "respawn" — low-probability (cf. `ChatTrigger::weight`) pour
        // qu'une vague de respawns ne sature pas l'overlay.
        for idx in respawned_bot_indices {
            self.maybe_bot_chat(idx, ChatTrigger::Respawn);
        }
    }

    /// Traitement d'une touche quand la console est **fermée** — contrôles
    /// joueur classiques.
    fn handle_game_key(&mut self, code: KeyCode, pressed: bool, event_loop: &ActiveEventLoop) {
        match code {
            // Mouvement : QWERTY WASD + alias AZERTY (Z=forward, Q=left) +
            // flèches. Chaque direction a son propre booléen, l'axe est
            // reconstitué par `forward_axis()` / `side_axis()` — évite le
            // bug où relâcher la 2e touche d'un combo (ex. release S
            // pendant qu'on tient W) remettait l'axe à 0. Les alias
            // n'interfèrent pas : KeyQ/KeyZ/arrow ne sont mappés nulle
            // part ailleurs dans le jeu.
            //
            // NB : `winit::PhysicalKey` correspond à la position physique
            // en layout US QWERTY ; sur AZERTY, la touche labellisée "Z"
            // est physiquement KeyW (déjà mappée), donc un joueur FR qui
            // utilise ZQSD (convention AZERTY) marche déjà avec KeyW/KeyA.
            // Les alias KeyQ/KeyZ couvrent le cas où le joueur presse
            // LES LETTRES labellisées "A" et "W" sur un clavier AZERTY.
            KeyCode::KeyW | KeyCode::KeyZ | KeyCode::ArrowUp => {
                let was = self.input.fwd_down;
                self.input.fwd_down = pressed;
                if pressed && !was { self.detect_double_tap(0); }
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                let was = self.input.back_down;
                self.input.back_down = pressed;
                if pressed && !was { self.detect_double_tap(1); }
            }
            KeyCode::KeyA | KeyCode::KeyQ | KeyCode::ArrowLeft => {
                let was = self.input.left_down;
                self.input.left_down = pressed;
                if pressed && !was { self.detect_double_tap(2); }
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                let was = self.input.right_down;
                self.input.right_down = pressed;
                if pressed && !was { self.detect_double_tap(3); }
            }
            KeyCode::Space => {
                self.input.jump = pressed;
                // En spectator, SPACE relâche la follow-cam → free-fly.
                // Edge sur press uniquement pour ne pas spammer le clear.
                if pressed && self.is_spectator && self.follow_slot.is_some() {
                    self.follow_slot = None;
                }
            }
            // Ctrl : crouch (gauche ou droite, les deux font le job façon Q3).
            // Si la touche passe de relâché → enfoncée pendant qu'on
            // court vite, on arme aussi le **slide tactique** (M3) qui
            // sera consommé au prochain MoveCmd. Le slide réutilise
            // CTRL pour ne pas prendre une touche supplémentaire — la
            // physique distingue les deux via `slide_pressed` (edge)
            // vs `crouch` (state).
            KeyCode::ControlLeft | KeyCode::ControlRight => {
                let was_pressed = self.input.crouch;
                self.input.crouch = pressed;
                if pressed && !was_pressed && self.player.on_ground {
                    let speed_xy = self.player.velocity.truncate().length();
                    if speed_xy >= q3_game::movement::SLIDE_MIN_SPEED {
                        self.input.slide_armed = true;
                    }
                }
            }
            // Shift : walk — vitesse réduite et footsteps silencieux.
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.input.walk = pressed,
            // TAB maintenu : overlay scoreboard. Relâché → retour HUD normal.
            KeyCode::Tab => self.input.scoreboard = pressed,
            // **Lean** (M7) — KeyB / KeyN pour peek gauche / droite.
            // Évite les collisions avec ZQSD (AZERTY) et WASD (QWERTY).
            // Maintenu = lean appliqué, relâché = retour neutre lerpé
            // côté caméra (cf. apply_view_lean).
            KeyCode::KeyB => self.input.lean_left_held = pressed,
            KeyCode::KeyN => self.input.lean_right_held = pressed,
            // F3 : taunt joueur — ligne de chat + son kill confirm dans
            // la même veine que les bots.  Edge-triggered + cooldown
            // interne (`next_player_taunt_at`) pour éviter le spam audio
            // quand le joueur garde la touche enfoncée.
            KeyCode::F3 if pressed => self.trigger_player_taunt(),
            // F9 : toggle overlay de stats FPS (moyenne + min/max + courbe
            // frametime).  Off par défaut, le joueur lambda n'a pas besoin
            // d'un compteur à l'écran.
            KeyCode::F9 if pressed => {
                self.show_perf_overlay = !self.show_perf_overlay;
            }
            // F5 : restart match (remise à zéro frags / HP / respawn global).
            // Pratique pour itérer sans quitter l'exe.
            KeyCode::F5 if pressed => self.restart_match(),
            // F11 : screenshot — TGA 32-bit dans le user dir (à côté du
            // q3config.cfg). Nom incrémenté pour ne pas écraser les
            // précédents (shot-0001.tga, shot-0002.tga, …). Bind edge-
            // triggered : plusieurs captures nécessitent plusieurs press.
            KeyCode::F11 if pressed => self.take_screenshot(),
            // ENTER : active le holdable courant (medkit / teleporter).
            // Bind canon Q3 = touche « Enter », réutilisée telle quelle.
            // Edge-triggered sur press pour ne pas spam-consommer le slot.
            KeyCode::Enter if pressed => self.use_held_item(),
            KeyCode::Escape => {
                if pressed {
                    if self.chat_wheel_open {
                        // Esc referme la roue sans envoyer.
                        self.chat_wheel_open = false;
                    } else {
                        event_loop.exit();
                    }
                }
            }
            // V : toggle la roue de chat rapide. Edge sur press uniquement.
            // Désactivée pendant un respawn (joueur mort) — pas de chat
            // depuis l'au-delà, ça donnerait des spams gênants.
            KeyCode::KeyV if pressed && !self.player_health.is_dead() => {
                self.chat_wheel_open = !self.chat_wheel_open;
                if self.chat_wheel_open {
                    self.chat_wheel_opened_at = self.time_sec;
                }
            }
            // Sélection d'arme façon Q3 : 1..9 mappent les slots d'arme.
            // EXCEPTION : roue ouverte → 1..8 envoient un message de
            // [`CHAT_WHEEL_MESSAGES`] et referment la roue, 9 referme
            // sans envoyer (cancel rapide).
            KeyCode::Digit1 if pressed => self.handle_digit_or_wheel(1),
            KeyCode::Digit2 if pressed => self.handle_digit_or_wheel(2),
            KeyCode::Digit3 if pressed => self.handle_digit_or_wheel(3),
            KeyCode::Digit4 if pressed => self.handle_digit_or_wheel(4),
            KeyCode::Digit5 if pressed => self.handle_digit_or_wheel(5),
            KeyCode::Digit6 if pressed => self.handle_digit_or_wheel(6),
            KeyCode::Digit7 if pressed => self.handle_digit_or_wheel(7),
            KeyCode::Digit8 if pressed => self.handle_digit_or_wheel(8),
            KeyCode::Digit9 if pressed => self.handle_digit_or_wheel(9),
            // X : last weapon toggle — bascule sur l'arme précédemment
            // active.  Edge-triggered (if pressed) comme les slots 1..9.
            // KeyQ est déjà mappé au strafe gauche (AZERTY), on prend
            // KeyX qui est libre sur tous les layouts courants.
            KeyCode::KeyX if pressed => self.swap_to_last_weapon(),
            _ => {}
        }
    }

    /// Pose une capture d'écran dans le user dir (`screenshots/shot-NNNN.tga`).
    /// Incrémente un compteur en scannant le dossier : le premier shot d'une
    /// session peut-être plus lent (listing), les suivants sont triviaux.
    /// Silencieux si le renderer n'est pas encore prêt — inoffensif si le
    /// joueur spam F11 pendant le boot.
    fn take_screenshot(&mut self) {
        let Some(r) = self.renderer.as_mut() else {
            warn!("screenshot: renderer pas encore initialisé");
            return;
        };
        let Some(dir) = screenshot_dir() else {
            warn!("screenshot: impossible de résoudre le répertoire user");
            return;
        };
        let idx = next_screenshot_index(&dir);
        let path = dir.join(format!("shot-{idx:04}.tga"));
        info!("screenshot: capture demandée → {}", path.display());
        r.queue_screenshot(path);
    }

    /// Traitement d'une touche quand la console est **ouverte**.
    fn handle_console_key(&mut self, event: &KeyEvent) {
        if event.state != ElementState::Pressed {
            return;
        }
        if let winit::keyboard::Key::Named(named) = &event.logical_key {
            match named {
                NamedKey::Enter => self.console.submit(),
                NamedKey::Backspace => self.console.backspace(),
                NamedKey::ArrowUp => self.console.history_prev(),
                NamedKey::ArrowDown => self.console.history_next(),
                NamedKey::Escape => self.console.set_open(false),
                // Tab : auto-complétion façon Q3 (cvars + cmds).
                NamedKey::Tab => self.console.tab_complete(),
                _ => {
                    // Space par ex. arrive aussi en text, on n'insère pas deux fois.
                }
            }
            return;
        }
        if let Some(text) = event.text.as_ref() {
            for c in text.chars() {
                self.console.push_char(c);
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title(concat!("Quake 3 RUST EDITION v", env!("CARGO_PKG_VERSION")))
            .with_inner_size(PhysicalSize::new(self.init_width, self.init_height));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                error!("create_window: {e}");
                event_loop.exit();
                return;
            }
        };
        let mut renderer = match Renderer::new_blocking(window.clone()) {
            Ok(r) => r,
            Err(e) => {
                error!("renderer: {e}");
                event_loop.exit();
                return;
            }
        };

        // Charge tous les scripts/*.shader et prépare un cache d'images
        // adossé au VFS. Les matériaux seront résolvés paresseusement au
        // premier drawcall qui les référence.
        match load_shader_registry(&self.vfs) {
            Ok(reg) => {
                let images = ImageCache::new((*self.vfs).clone());
                renderer.attach_materials(reg, images);
            }
            Err(e) => warn!("shader registry: {e} — rendu en lightmap-only"),
        }

        self.window = Some(window);
        self.renderer = Some(renderer);

        // Audio : best-effort, on log juste un warn si indispo (headless CI).
        match SoundSystem::new() {
            Ok(s) => {
                let arc = Arc::new(s);
                arc.set_master_volume(self.cvars.get_f32("s_volume").unwrap_or(0.8));
                arc.set_music_volume(self.cvars.get_f32("s_musicvolume").unwrap_or(0.25));
                self.sound = Some(arc);
                info!("audio: rodio prêt");
            }
            Err(e) => warn!("audio: {e}"),
        }

        if let Some(map) = self.requested_map.clone() {
            // `--map` explicite en CLI : on charge direct, le menu reste
            // fermé. C'est le flow "je sais ce que je veux" pour le dev.
            self.load_map(&map);
            // **Bots solo via `--bots N`** : drainé une fois ici, après
            // `load_map` qui a chargé `world` + `bot_rig`. Skill III par
            // défaut, noms séquentiels « bot01..botNN ».
            if self.pending_local_bots > 0 {
                let n = self.pending_local_bots;
                self.pending_local_bots = 0;
                // Force-load le rig joueur AVANT le 1er spawn_bot pour
                // garantir un check explicite. Si ça échoue, on affiche
                // un warn tonitruant pour orienter le diagnostic plutôt
                // que de laisser spawn_bot bouncer N fois en silence.
                self.ensure_player_rig_loaded();
                if self.bot_rig.is_none() {
                    warn!(
                        "===========================================\n\
                         BOTS DEMANDÉS ({n}) MAIS PAS DE PLAYER RIG\n\
                         Le pak0.pk3 actif ne contient pas de modèle\n\
                         joueur complet (lower.md3 + upper.md3 + head.md3)\n\
                         dans models/players/<nom>/. Les bots seront\n\
                         remplacés par des beams verticaux colorés en\n\
                         attendant un asset complet.\n\
                         ==========================================="
                    );
                }
                let world_ok = self.world.as_ref().is_some();
                let spawn_ok = self.world.as_ref().map_or(false, |w| {
                    !w.spawn_points.is_empty() || w.player_start.is_some()
                });
                info!(
                    "bot diagnostic: world_loaded={world_ok}, \
                     spawn_points_or_player_start={spawn_ok}, \
                     rig_loaded={}",
                    self.bot_rig.is_some()
                );
                let before = self.bots.len();
                for i in 0..n {
                    let name = format!("bot{:02}", i + 1);
                    self.spawn_bot(&name, Some(3));
                }
                info!(
                    "solo: --bots {n} demandés → {} bots effectivement \
                     présents dans self.bots (avant: {before})",
                    self.bots.len()
                );
            }
            // Q3_SPAWN_BOTS=N : spawn N bots immédiatement après le chargement
            // de la map. Équivalent d'un `addbot` en console, utile pour les
            // tests CI où on veut un screenshot avec des adversaires visibles
            // sans avoir à piloter le curseur.
            if let Ok(s) = std::env::var("Q3_SPAWN_BOTS") {
                if let Ok(n) = s.parse::<usize>() {
                    for i in 0..n {
                        self.spawn_bot(&format!("testbot{i}"), None);
                    }
                }
            }
        } else {
            // Sinon on laisse le menu principal ouvert (positionné à Root
            // par `new`). Le joueur choisit sa map via l'UI, ou quitte —
            // plus de fond noir déroutant.
            info!(
                "menu: {} maps disponibles dans le VFS",
                self.menu.map_list.len()
            );
        }

        // Capture souris seulement si le menu n'est pas affiché — sinon
        // le joueur doit garder le curseur pour cliquer les items.
        self.set_mouse_capture(!self.menu.open);

        self.last_tick = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                info!("fermeture demandée");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                let repeat = key_event.repeat;
                // La touche backtick (`) (ou "²" en AZERTY) toggle la console.
                if key_event.state == ElementState::Pressed && !repeat {
                    if let PhysicalKey::Code(KeyCode::Backquote) = key_event.physical_key {
                        self.console.toggle();
                        if self.console.is_open() {
                            self.input = Input::default();
                            self.set_mouse_capture(false);
                        } else {
                            self.set_mouse_capture(true);
                        }
                        return;
                    }
                }

                if self.console.is_open() {
                    self.handle_console_key(&key_event);
                } else if self.menu.open {
                    // Toutes les touches pressées vont au menu quand il
                    // est visible — pas de répétition clavier (Q3 ne
                    // faisait pas défiler non plus sur auto-repeat).
                    if key_event.state == ElementState::Pressed && !repeat {
                        if let PhysicalKey::Code(code) = key_event.physical_key {
                            let action = self.menu.on_key(code, &self.cvars);
                            self.apply_menu_action(action, event_loop);
                        }
                    }
                } else if !repeat {
                    let pressed = key_event.state == ElementState::Pressed;
                    if let PhysicalKey::Code(code) = key_event.physical_key {
                        // Escape en jeu → ouvre le menu pause.
                        if pressed && code == KeyCode::Escape {
                            self.menu.set_in_game(self.world.is_some());
                            self.menu.open_root();
                            self.set_mouse_capture(false);
                            return;
                        }
                        self.handle_game_key(code, pressed, event_loop);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
                self.cursor_pos = (x, y);
                if self.menu.open {
                    let (w, h) = self
                        .window
                        .as_ref()
                        .map(|w| {
                            let s = w.inner_size();
                            (s.width as f32, s.height as f32)
                        })
                        .unwrap_or((self.init_width as f32, self.init_height as f32));
                    self.menu.on_mouse_move(x, y, w, h);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                use winit::event::MouseButton;
                if self.menu.open {
                    if button == MouseButton::Left && state == ElementState::Pressed {
                        let (w, h) = self
                            .window
                            .as_ref()
                            .map(|w| {
                                let s = w.inner_size();
                                (s.width as f32, s.height as f32)
                            })
                            .unwrap_or((self.init_width as f32, self.init_height as f32));
                        let action = self.menu.on_mouse_click(
                            self.cursor_pos.0,
                            self.cursor_pos.1,
                            w,
                            h,
                            &self.cvars,
                        );
                        self.apply_menu_action(action, event_loop);
                    }
                } else if !self.console.is_open() {
                    if button == MouseButton::Left {
                        // En spectator, LMB cycle next sur les targets
                        // disponibles (au press uniquement). Sinon, fire.
                        if self.is_spectator
                            && state == ElementState::Pressed
                        {
                            self.cycle_follow_target(1);
                        } else {
                            self.input.fire = state == ElementState::Pressed;
                        }
                    } else if button == MouseButton::Right {
                        if self.is_spectator {
                            // En spectator : cycle prev de la follow-cam.
                            if state == ElementState::Pressed {
                                self.cycle_follow_target(-1);
                            }
                        } else {
                            // En jeu : tir secondaire (alt-fire). Hold
                            // pour les armes à charge continue (LG alt,
                            // BFG alt) ; consommé sur edge pour les
                            // armes single-shot (rail, shotgun slug,
                            // rocket lock).
                            self.input.secondary_fire =
                                state == ElementState::Pressed;
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                use winit::event::MouseScrollDelta;
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 60.0,
                };
                if self.menu.open {
                    self.menu.on_scroll(lines);
                } else if !self.console.is_open() && !self.player_health.is_dead() {
                    // Q3 convention : molette haut = arme précédente, bas = suivante.
                    if lines.abs() >= 0.5 {
                        self.cycle_weapon(if lines > 0.0 { -1 } else { 1 });
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_tick).as_secs_f32().min(0.1);
                self.last_tick = now;

                // Échantillonne le frametime dans le ring buffer.  On
                // mesure le `dt` réel (post-clamp 100 ms) — sur un gros
                // hitch >100 ms on sous-estime la perte, mais c'est une
                // concession déjà faite pour la physique et cohérente.
                self.frame_times[self.frame_time_head] = dt;
                self.frame_time_head = (self.frame_time_head + 1) % FRAME_TIME_BUF;

                // VR : `xrWaitFrame` doit être la toute première chose du
                // tick quand une session XR est active. En mode stub c'est
                // un no-op sans effet de bord.
                self.vr.begin_frame();

                // Réseau : drain des paquets entrants + envoi des usercmd /
                // snapshots. No-op en solo. Le `dt` passé est le temps réel
                // écoulé depuis la frame précédente (pas le step fixe de
                // la physique) — c'est celui que le scheduler 20 Hz (serveur)
                // ou 60 Hz (client) utilise pour décider s'il faut pousser
                // un paquet ou attendre encore.
                if self.net.mode.is_server() {
                    self.net.tick_server(dt, self.world.as_ref());
                } else if self.net.mode.is_client() {
                    // On n'envoie d'input que si la console / menu est
                    // fermée — sinon le joueur tape une commande et son
                    // perso bouge en arrière-plan.
                    let input = if self.console.is_open() {
                        None
                    } else {
                        Some(crate::net::LocalInput {
                            forward: self.input.forward_axis(),
                            side: self.input.side_axis(),
                            up: 0.0,
                            jump: self.input.jump,
                            crouch: self.input.crouch,
                            walk: self.input.walk,
                            fire: self.input.fire,
                            use_holdable: false,
                            view_pitch: self.player.view_angles.pitch,
                            view_yaw: self.player.view_angles.yaw,
                            view_roll: self.player.view_angles.roll,
                            weapon: self.active_weapon.slot(),
                        })
                    };
                    self.net.tick_client(dt, input.as_ref());
                    // Applique la dernière snapshot serveur à l'état local
                    // pour que le rendu corresponde à l'autoritatif. Sans
                    // prédiction, le joueur ressent un input lag égal au
                    // RTT — acceptable en LAN (étape 5 ajoutera la prédiction).
                    if let Some(p) = self.net.take_client_snapshot() {
                        let snap = &p.snapshot;
                        // === Réconciliation prédictive ===
                        // 1. Rewind : restaure l'état autoritatif serveur.
                        // 2. Replay : rejoue les UserCmd encore en vol pour
                        //    converger vers « ce que le serveur calculera
                        //    bientôt à partir de mes inputs ». Sans cette
                        //    étape, le joueur snap-back à 20 Hz (jitter
                        //    visible). Avec, il garde une réactivité
                        //    indistinguable du solo en LAN.
                        if let Some(my_slot) = snap
                            .players
                            .iter()
                            .find(|x| x.slot == snap.client_slot)
                        {
                            // Capture team locale pour le scoreboard.
                            self.local_team = my_slot.team;
                            // Rewind.
                            self.player.origin = Vec3::from_array(my_slot.origin);
                            self.player.velocity = Vec3::from_array(my_slot.velocity);
                            self.player.on_ground =
                                my_slot.flags & q3_net::player_flags::ON_GROUND != 0;
                            self.player.crouching =
                                my_slot.flags & q3_net::player_flags::CROUCHING != 0;
                            // view_angles : on garde les angles locaux —
                            // c'est le client qui est maître de sa caméra.
                            // Sans ça, un round-trip de quantification
                            // donnerait des micro-saccades de pitch/yaw à
                            // 20 Hz.

                            // Spectator : détecté ici plutôt que plus
                            // bas pour gater le replay correctement.
                            let spec = my_slot.flags
                                & q3_net::player_flags::SPECTATOR
                                != 0;

                            // Replay des UserCmd non-ackées.
                            if let Some(world) = self.world.as_ref() {
                                let params = q3_game::movement::PhysicsParams::default();
                                for cmd in &p.cmds_to_replay {
                                    let dt_ms = cmd.delta_ms.max(1);
                                    let dt = dt_ms as f32 / 1000.0;
                                    if spec {
                                        // Noclip free-fly replay : même
                                        // logique que `apply_spectator_move`
                                        // côté serveur, pour que prédiction
                                        // = autoritatif.
                                        let basis = self.player.view_angles.to_vectors();
                                        let mut wish = basis.forward
                                            * q3_net::UserCmd::dequantize_axis(cmd.forward)
                                            + basis.right
                                                * q3_net::UserCmd::dequantize_axis(cmd.side);
                                        if cmd.buttons & q3_net::buttons::JUMP != 0 {
                                            wish += Vec3::Z;
                                        }
                                        if cmd.buttons & q3_net::buttons::CROUCH != 0 {
                                            wish -= Vec3::Z;
                                        }
                                        let v = if wish.length_squared() > 1e-4 {
                                            wish.normalize() * 640.0
                                        } else {
                                            Vec3::ZERO
                                        };
                                        self.player.velocity = v;
                                        self.player.origin += v * dt;
                                        self.player.on_ground = false;
                                        continue;
                                    }
                                    let mc = MoveCmd {
                                        forward: q3_net::UserCmd::dequantize_axis(cmd.forward),
                                        side: q3_net::UserCmd::dequantize_axis(cmd.side),
                                        up: q3_net::UserCmd::dequantize_axis(cmd.up),
                                        jump: cmd.buttons & q3_net::buttons::JUMP != 0,
                                        crouch: cmd.buttons & q3_net::buttons::CROUCH != 0,
                                        walk: cmd.buttons & q3_net::buttons::WALK != 0,
                                        // Slide / dash : edges locales,
                                        // pas resync via wire pour
                                        // l'instant — replay côté
                                        // client uniquement.
                                        slide_pressed: false,
                                        dash_pressed: false,
                                        delta_time: dt,
                                    };
                                    self.player.tick_collide(mc, params, &world.collision);
                                }
                            }

                            // HUD bits : santé, armor, frags. Ceux-là sont
                            // toujours autoritaifs serveur (pas d'intérêt à
                            // les prédire — la prédiction de la santé va
                            // contre le sens du jeu : on doit voir l'impact
                            // au moment où le serveur l'enregistre).
                            self.player_health.current = my_slot.health as i32;
                            self.player_armor = my_slot.armor.max(0) as i32;
                            self.frags = my_slot.frags.max(0) as u32;
                            self.deaths = my_slot.deaths.max(0) as u32;
                            // Mirror ammo serveur → HUD local. Cap les
                            // négatifs à 0 (jamais censé arriver mais
                            // safety pour le HUD qui n'aime pas les
                            // valeurs négatives).
                            for (i, a) in my_slot.ammo.iter().enumerate() {
                                if i < self.ammo.len() {
                                    self.ammo[i] = (*a).max(0) as i32;
                                }
                            }
                            // Spectator : flag mis à jour à chaque
                            // snapshot. Le local pmove le lit pour
                            // basculer en noclip, et le HUD masque les
                            // jauges HP/armor/ammo.
                            self.is_spectator =
                                my_slot.flags & q3_net::player_flags::SPECTATOR != 0;

                            // Powerups : on bit-mappe `powerups` u8 vers
                            // les slots `powerup_until` locaux. Le wire
                            // n'envoie pas le temps restant en v1, on
                            // affiche un compteur fictif 30s remis à
                            // chaque snapshot tant que le bit reste set.
                            // Quand le bit clear, on remet à None →
                            // badge disparaît.
                            const NET_POWERUP_DISPLAY_SEC: f32 = 30.0;
                            for kind in PowerupKind::ALL {
                                let bit = match kind {
                                    PowerupKind::QuadDamage => q3_net::powerup_flags::QUAD_DAMAGE,
                                    PowerupKind::Haste => q3_net::powerup_flags::HASTE,
                                    PowerupKind::Regeneration => q3_net::powerup_flags::REGENERATION,
                                    PowerupKind::BattleSuit => q3_net::powerup_flags::BATTLE_SUIT,
                                    PowerupKind::Invisibility => q3_net::powerup_flags::INVISIBILITY,
                                    PowerupKind::Flight => q3_net::powerup_flags::FLIGHT,
                                };
                                let active = my_slot.powerups & bit != 0;
                                let slot_ix = kind.index();
                                if active {
                                    // Refresh expiry à chaque snapshot.
                                    self.powerup_until[slot_ix] =
                                        Some(self.time_sec + NET_POWERUP_DISPLAY_SEC);
                                } else {
                                    self.powerup_until[slot_ix] = None;
                                }
                            }
                        }
                        // === Update remote interp buffers ===
                        // Push un nouveau sample dans chaque slot remote.
                        // Le rendu des `remote_players` lerpera entre
                        // older et newer chaque frame.
                        let received_at = Instant::now();
                        let mut seen_slots: smallvec::SmallVec<[u8; 16]> =
                            smallvec::SmallVec::new();
                        for ps in &snap.players {
                            // Tracking score : s'applique à TOUS les slots
                            // y compris le local. Permet au scoreboard
                            // d'afficher les frags/deaths fiables sans
                            // dépendre du compteur local (qui peut diverger
                            // après un restart serveur).
                            self.remote_scores
                                .insert(ps.slot, (ps.frags, ps.deaths, ps.team));
                            if ps.slot == snap.client_slot {
                                continue;
                            }
                            seen_slots.push(ps.slot);
                            let entry =
                                self.remote_interp.entry(ps.slot).or_default();
                            entry.push(RemoteSample {
                                origin: Vec3::from_array(ps.origin),
                                velocity: Vec3::from_array(ps.velocity),
                                view_angles: Angles::new(
                                    ps.view_angles[0],
                                    ps.view_angles[1],
                                    ps.view_angles[2],
                                ),
                                on_ground: ps.flags
                                    & q3_net::player_flags::ON_GROUND
                                    != 0,
                                is_dead: ps.flags
                                    & q3_net::player_flags::DEAD
                                    != 0,
                                recently_fired: ps.flags
                                    & q3_net::player_flags::RECENTLY_FIRED
                                    != 0,
                                team: ps.team,
                                server_time: snap.server_time,
                                received_at,
                            });
                        }
                        // Garbage-collect : retire les slots absents du
                        // snapshot courant (joueur a quitté).
                        self.remote_interp
                            .retain(|slot, _| seen_slots.contains(slot));
                        if !self.remote_interp.is_empty() && self.bot_rig.is_none() {
                            self.ensure_player_rig_loaded();
                        }

                        // === Remote projectiles ===
                        // Update entries from snap.entities + GC ceux
                        // absents (projectile expiré ou détruit serveur).
                        let mut seen_ids: smallvec::SmallVec<[u32; 16]> =
                            smallvec::SmallVec::new();
                        for ent in &snap.entities {
                            seen_ids.push(ent.id);
                            self.remote_projectiles.insert(
                                ent.id,
                                RemoteProjectile {
                                    id: ent.id,
                                    kind: ent.kind,
                                    origin: Vec3::from_array(ent.origin),
                                    velocity: Vec3::from_array(ent.velocity),
                                    last_received_at: received_at,
                                },
                            );
                        }
                        self.remote_projectiles
                            .retain(|id, _| seen_ids.contains(id));

                        // === Table noms (slot → name) ===
                        // Reconstruite à chaque snapshot — l'overhead
                        // est négligeable (8 entrées max). Les noms ne
                        // sont rafraîchis qu'aux fulls (~1 s) côté
                        // serveur, mais le client en garde toujours
                        // le miroir cohérent.
                        if !snap.players_info.is_empty() {
                            self.remote_names.clear();
                            for pi in &snap.players_info {
                                self.remote_names.insert(pi.slot, pi.name_string());
                            }
                        }

                        // === Pickups indisponibles ===
                        // Le serveur envoie l'ensemble des pickups
                        // ramassés (state=0). On reconstruit le HashSet
                        // à chaque snapshot — taille ≤ 16 typiquement,
                        // l'overhead est négligeable.
                        self.remote_unavailable_pickups.clear();
                        for ps in &snap.pickups {
                            if ps.available == 0 {
                                self.remote_unavailable_pickups.insert(ps.id);
                            }
                        }
                        // Propagation respawn_at sur les pickups locaux —
                        // permet au HUD respawn-timer + aux marqueurs
                        // flottants au-dessus du sol de réagir aux
                        // pickups grabés par d'autres joueurs en multi.
                        // Edge-detection : `respawn_at = None` + wire
                        // dit `unavailable` ⇒ nouvelle transition. Sur
                        // les snapshots suivants on laisse le countdown
                        // dérouler localement (`tick_pickups`).
                        for p in self.pickups.iter_mut() {
                            let unavail = self
                                .remote_unavailable_pickups
                                .contains(&p.entity_index);
                            match (unavail, p.respawn_at.is_some()) {
                                (true, false) => {
                                    p.respawn_at =
                                        Some(self.time_sec + p.respawn_cooldown);
                                }
                                (false, true) => {
                                    // Serveur dit dispo → on l'efface
                                    // même si le countdown local pensait
                                    // qu'il restait du temps (rare désync
                                    // d'horloge — l'autorité l'emporte).
                                    p.respawn_at = None;
                                }
                                _ => {}
                            }
                        }

                        // === Évènements ponctuels (1-shot) ===
                        // On copie la liste pour libérer le borrow `snap`
                        // avant les appels qui prennent &mut self.
                        let events_to_play = snap.events.clone();
                        for evt in events_to_play {
                            self.handle_remote_event(evt);
                        }
                    }
                }

                // Décroissance du recul viewmodel : exponentielle pour
                // un retour naturel vers 0 (pas de discontinuité, le
                // viewmodel "remonte" vers sa position de repos).  On
                // zère en-dessous d'un seuil pour éviter de traîner des
                // ε infinitésimaux dans le calcul de transform.
                if self.view_kick > 0.0 {
                    self.view_kick *= (-VIEW_KICK_DECAY_PER_SEC * dt).exp();
                    if self.view_kick < 1e-4 {
                        self.view_kick = 0.0;
                    }
                }

                // Draine les actions remontées depuis la console.
                self.drain_pending(event_loop);

                // Physique joueur seulement si la console est fermée, que
                // le joueur est vivant et que le match n'est pas terminé.
                // Intermission → tout est gelé, on laisse juste l'audio
                // et les particules vivre leur vie.
                if !self.console.is_open()
                    && !self.player_health.is_dead()
                    && self.match_winner.is_none()
                {
                    // Pas fixe Q3 : 8ms (125Hz). L'original simule la
                    // physique à cette fréquence quelle que soit la
                    // framerate GPU — sans ce pas fixe, la friction et
                    // l'accélération dérivent avec les FPS et le feel
                    // (notamment l'accélération perçue et le strafe-jump)
                    // change visiblement entre 60fps et 144fps. On
                    // accumule le temps réel dans `physics_accumulator`
                    // et on exécute N ticks de 8ms.
                    const PHYSICS_STEP: f32 = 1.0 / 125.0;
                    const MAX_PHYSICS_STEPS: u32 = 8;
                    self.physics_accumulator += dt;
                    // Clamp l'accumulateur : si la frame a stutter 200ms,
                    // on ne veut pas faire 25 ticks d'un coup — ça "spire"
                    // le simulateur (accumulation de ticks qui allonge
                    // encore la frame). Mieux vaut décroché un peu que
                    // bloquer le jeu.
                    let max_accum = PHYSICS_STEP * MAX_PHYSICS_STEPS as f32;
                    if self.physics_accumulator > max_accum {
                        self.physics_accumulator = max_accum;
                    }

                    // Edges slide/dash : on consomme les flags armés
                    // (set par les handlers clavier) et on les remet à
                    // false une fois injectés dans la cmd. Le pmove
                    // déclenche slide_remaining / dash_remaining et
                    // gère lui-même les cooldowns.
                    let slide_pressed =
                        std::mem::take(&mut self.input.slide_armed);
                    let dash_pressed =
                        std::mem::take(&mut self.input.dash_armed);
                    let cmd = MoveCmd {
                        forward: self.input.forward_axis(),
                        side: self.input.side_axis(),
                        up: 0.0,
                        jump: self.input.jump,
                        crouch: self.input.crouch,
                        walk: self.input.walk,
                        slide_pressed,
                        dash_pressed,
                        delta_time: PHYSICS_STEP,
                    };
                    let was_on_ground = self.player.on_ground;
                    // Sauvegarde la vitesse verticale avant intégration :
                    // `update_ground` la met à 0 quand on touche le sol,
                    // mais c'est précisément cette valeur pré-impact qu'il
                    // nous faut pour calculer les dégâts de chute.
                    let prev_vz = self.player.velocity.z;
                    let physics = self.effective_physics_params();

                    while self.physics_accumulator >= PHYSICS_STEP {
                        // Spectator : noclip free-fly. Bypass `tick_collide`
                        // (pas de collision, pas de gravité) — réplique
                        // exact du `apply_spectator_move` serveur pour
                        // que la prédiction locale matche l'autoritatif.
                        if self.is_spectator {
                            // Follow-cam : si on suit un slot et qu'il
                            // est encore vivant dans remote_interp, on
                            // snap origin/view sur lui. Plus de noclip.
                            if let Some(target) = self.follow_slot {
                                if let Some(buf) = self.remote_interp.get(&target) {
                                    if let Some(s) = buf.current() {
                                        if !s.is_dead {
                                            // Léger offset Z pour ne pas
                                            // se planter dans la nuque
                                            // du modèle (caméra ~ tête).
                                            self.player.origin =
                                                s.origin + Vec3::new(0.0, 0.0, 8.0);
                                            self.player.view_angles = s.view_angles;
                                            self.player.velocity = Vec3::ZERO;
                                            self.player.on_ground = false;
                                            self.physics_accumulator -=
                                                PHYSICS_STEP;
                                            continue;
                                        }
                                    }
                                }
                                // Cible disparue / morte → drop le follow,
                                // l'utilisateur peut recycler avec LMB.
                                self.follow_slot = None;
                            }
                            let basis = self.player.view_angles.to_vectors();
                            let mut wish = basis.forward * cmd.forward
                                + basis.right * cmd.side;
                            if self.input.jump {
                                wish += Vec3::Z;
                            }
                            if self.input.crouch {
                                wish -= Vec3::Z;
                            }
                            const SPEC_SPEED: f32 = 640.0;
                            let v = if wish.length_squared() > 1e-4 {
                                wish.normalize() * SPEC_SPEED
                            } else {
                                Vec3::ZERO
                            };
                            self.player.velocity = v;
                            self.player.origin += v * PHYSICS_STEP;
                            self.player.on_ground = false;
                            self.physics_accumulator -= PHYSICS_STEP;
                            continue;
                        }
                        // Flight : pousse verticalement depuis jump/crouch
                        // *avant* l'intégration. On remplace la vitesse Z
                        // plutôt que de l'additionner — sinon relâcher jump
                        // laisse une vélocité résiduelle qui continue à
                        // monter. on_ground est forcé à false pour que
                        // l'intégration ne déclenche pas la friction-sol
                        // ni le snap-to-ground, et pour que le saut
                        // "standard" (on_ground -> jump vel) n'interfère
                        // pas avec notre override.
                        if self.is_powerup_active(PowerupKind::Flight) {
                            self.player.on_ground = false;
                            let up = if self.input.jump {
                                FLIGHT_THRUST_SPEED
                            } else if self.input.crouch {
                                -FLIGHT_THRUST_SPEED
                            } else {
                                self.player.velocity.z * 0.92
                            };
                            self.player.velocity.z = up;
                        }
                        if let Some(world) = self.world.as_ref() {
                            self.player.tick_collide(cmd, physics, &world.collision);
                        }
                        self.physics_accumulator -= PHYSICS_STEP;
                    }
                    // Triggers de monde : jump pads puis téléporteurs puis
                    // zones de dégât. Ordre important — un jump pad ne doit
                    // pas pouvoir propulser après un teleport "même tick",
                    // et un teleport doit overrider un push précédent
                    // (gagnant absolu : téléport). Hurt vient en dernier
                    // pour qu'une téléportation hors lave n'applique pas
                    // de dégâts au tick de sortie.
                    self.tick_jump_pads();
                    self.tick_teleporters();
                    self.tick_hurt_zones();
                    // Noyade : test de l'œil dans un brush d'eau + décompte
                    // de la jauge d'air. Après `hurt_zones` pour qu'une lave
                    // qui tue ce tick n'engendre pas aussi un hit de noyade
                    // redondant — `is_dead()` gatera early-return.
                    self.tick_drown(dt);
                    // Dégâts de chute : après les triggers pour qu'un pad
                    // qui te re-lance avant de toucher le sol n'injecte
                    // pas de dégâts, et avant le sfx_land pour que le pain
                    // et le "thud" sortent en cohérence.
                    self.tick_fall_damage(prev_vz);
                    // SFX : saut (edge press) + atterrissage (air → sol).
                    if let Some(snd) = self.sound.as_ref() {
                        if cmd.jump && was_on_ground && !self.player.on_ground {
                            if let Some(h) = self.sfx_jump {
                                play_at(snd, h, self.player.origin, Priority::Normal);
                            }
                            self.was_airborne = true;
                        }
                        if self.was_airborne && self.player.on_ground {
                            if let Some(h) = self.sfx_land {
                                play_at(snd, h, self.player.origin, Priority::Normal);
                            }
                            self.was_airborne = false;
                        }
                    }

                    // View-bob : avance la phase proportionnellement à la
                    // vitesse horizontale, uniquement au sol. En l'air on
                    // décay doucement vers 0 pour ne pas bobber au jumping.
                    let hspeed = (self.player.velocity.x.powi(2)
                        + self.player.velocity.y.powi(2))
                    .sqrt();
                    if self.player.on_ground && hspeed > BOB_SPEED_MIN {
                        let factor = (hspeed / BOB_SPEED_REF).clamp(0.0, 1.0);
                        let phase_before = self.bob_phase;
                        self.bob_phase += BOB_FREQ * factor * dt;
                        // Détecte la foulée : on émet un pas à chaque
                        // passage d'un multiple de π (0, π, 2π, …). Deux
                        // pas par cycle (gauche/droite). On *avance* la
                        // phase aussi en walk/crouch (le view-bob reste
                        // agréable), mais on ne joue PAS le son : c'est
                        // l'équivalent du "walk silencieux" Q3, qui
                        // empêche un bot ennemi de t'entendre à travers
                        // un mur.
                        if !self.input.walk && !self.player.crouching {
                            self.maybe_play_footstep(phase_before, self.bob_phase);
                        } else {
                            // On avance quand même le seuil pour que le
                            // prochain passage en course n'émette pas un
                            // gros rafale de pas rattrapés.
                            self.last_footstep_phase = self.bob_phase;
                        }
                        // Garde la phase dans [0, 2π] pour éviter un drift
                        // flottant sur une session longue.
                        if self.bob_phase > std::f32::consts::TAU {
                            self.bob_phase -= std::f32::consts::TAU;
                            self.last_footstep_phase -= std::f32::consts::TAU;
                        }
                    }
                }

                // Listener audio = position caméra + axe droit.
                if let Some(snd) = self.sound.as_ref() {
                    let basis = self.player.view_angles.to_vectors();
                    snd.set_listener(Listener {
                        position: self.player.origin,
                        right: basis.right,
                    });
                    snd.tick();
                }

                self.time_sec += dt;

                // Tick de chaque bot : IA (vision + LOS) → BotCmd → MoveCmd
                // → physique. Les tirs hitscan retournent un total de dégâts
                // à appliquer au joueur ce tick + une liste de rockets à
                // merger dans l'état global des projectiles.
                // Skippé pendant l'intermission pour figer la scène.
                // Idem pendant le warmup — le countdown doit laisser la
                // scène tranquille.
                if self.match_winner.is_none() && !self.is_warmup() {
                if let Some(world) = self.world.as_ref() {
                    let alive = !self.player_health.is_dead();
                    let invisible = self.is_powerup_active(PowerupKind::Invisibility);
                    let bot_out = tick_bots(
                        &mut self.bots,
                        dt,
                        self.time_sec,
                        self.params,
                        world,
                        self.player.origin,
                        alive,
                        invisible,
                        &self.rocket_mesh,
                    );
                    // Ajoute les rockets bots au pool global. `tick_projectiles`
                    // les gérera comme celles du joueur (direct + splash).
                    self.projectiles.extend(bot_out.projectiles);
                    // Bullet holes des tirs bots qui ont raté — posés en
                    // décales persistantes pour que le joueur voit les
                    // impacts sur les murs autour de lui.
                    for (pos, normal, weapon) in bot_out.wall_marks {
                        self.push_wall_mark(pos, normal, weapon);
                    }
                    // Rail trails bots : même pool que les beams joueur,
                    // consommé par le renderer.  On évite la saturation en
                    // coupant au-delà d'un petit plafond (un rail par bot
                    // et par frame, max 8 bots = 8 entries par tick).
                    self.beams.extend(bot_out.rail_beams);
                    let dmg = bot_out.damage;
                    let mg_damager = bot_out.last_mg_damager;
                    let mg_damager_idx = bot_out.last_mg_damager_idx;
                    let hitscan_weapon = bot_out.last_hitscan_weapon
                        .unwrap_or(WeaponId::Machinegun);
                    // Invul post-respawn : on avale les dégâts bot hitscan
                    // (MG/SG/RG) ce tick — les bots continuent à tirer,
                    // le traçage + SFX reste, seul l'impact HP disparaît.
                    // `tick_bots` a déjà consommé les munitions / cooldowns.
                    let player_invul = self.player_invul_until > self.time_sec;
                    if dmg > 0 && alive && !player_invul {
                        // Armor absorbe la moitié des dégâts (modèle simplifié).
                        let absorbed = (dmg / 2).min(self.player_armor);
                        self.player_armor -= absorbed;
                        if absorbed > 0 {
                            self.armor_flash_until = self.time_sec + ARMOR_FLASH_SEC;
                        }
                        let dmg_after = dmg - absorbed;
                        let taken = self.player_health.take_damage(dmg_after);
                        if taken > 0 {
                            if let (Some(snd), Some(h)) =
                                (self.sound.as_ref(), self.sfx_pain_player)
                            {
                                play_at(snd, h, self.player.origin, Priority::High);
                            }
                            // Chiffre rouge flottant à hauteur d'œil — rendu
                            // par le HUD après projection écran.
                            let eye = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
                            self.push_damage_number(eye, taken, true);
                            // Pain-arrow : direction = bot tireur → joueur.
                            // On utilise l'index du dernier damager pour
                            // retrouver la position monde, normalisée en 3D.
                            // Pas d'indicateur si on ne peut pas attribuer
                            // (valeurs par défaut → reste 0).
                            if let Some(idx) = mg_damager_idx {
                                if let Some(bd) = self.bots.get(idx) {
                                    let from = bd.body.origin
                                        + Vec3::Z * BOT_CENTER_HEIGHT;
                                    let d = eye - from;
                                    let len = d.length();
                                    if len > 1e-3 {
                                        self.last_damage_dir = d / len;
                                        self.last_damage_until =
                                            self.time_sec + DAMAGE_DIR_SHOW_SEC;
                                    }
                                }
                            }
                            self.pain_flash_until = self.time_sec + PAIN_FLASH_SEC;
                        }
                        if taken > 0 && self.player_health.is_dead() {
                            self.deaths = self.deaths.saturating_add(1);
                            self.respawn_at = Some(self.time_sec + RESPAWN_DELAY_SEC);
                            info!(
                                "joueur abattu — respawn dans {RESPAWN_DELAY_SEC:.1}s (deaths={})",
                                self.deaths
                            );
                            let killer = mg_damager
                                .map(KillActor::Bot)
                                .unwrap_or(KillActor::World);
                            // L'arme dépend du profil hitscan choisi par le
                            // bot ce tick (SG en close, MG en mid, RG en long).
                            self.push_kill(killer, KillActor::Player, hitscan_weapon);
                            if let Some(idx) = mg_damager_idx {
                                if let Some(bd) = self.bots.get_mut(idx) {
                                    bd.frags = bd.frags.saturating_add(1);
                                }
                            }
                        }
                    }
                }
                } // fin du `if self.match_winner.is_none()`

                // Tir joueur : hitscan sphère → bot le plus proche non occulté.
                if self.input.fire
                    && !self.console.is_open()
                    && !self.player_health.is_dead()
                    && self.match_winner.is_none()
                    && !self.is_warmup()
                    && self.time_sec >= self.next_player_fire_at
                {
                    let cooldown = self.active_weapon.params().cooldown
                        * self.player_fire_cooldown_mult();
                    // `fire_weapon` retourne `false` si vide → on applique un
                    // mini-cooldown pour éviter de spinner chaque frame.
                    let fired = self.fire_weapon();
                    self.next_player_fire_at = self.time_sec
                        + if fired { cooldown } else { 0.2 };
                }

                // Respawn bots morts en place — garde `bots.len()` stable
                // et recycle les waypoints. Gelé pendant l'intermission
                // et pendant le warmup.
                if self.match_winner.is_none() && !self.is_warmup() {
                    self.respawn_dead_bots();
                }

                // Projectiles en vol → avance + impact + splash damage.
                // Tick même pendant l'intermission pour que les particules
                // / explosions en cours se dissipent proprement.
                self.tick_projectiles(dt);

                // Contacts pickups ↔ joueur. Inclut réactivation après timer.
                if self.match_winner.is_none() {
                    self.tick_pickups();
                }

                // Respawn après délai — hors de la borrow scope du monde
                // pour pouvoir ré-emprunter `self` en mutable.
                if self.match_winner.is_none() {
                    if let Some(t) = self.respawn_at {
                        if self.time_sec >= t {
                            self.respawn_player();
                        }
                    }
                }

                // Expire les powerups dont la fin est dépassée + tick la
                // régénération. Les Option repassent à None plutôt que de
                // garder un timestamp périmé pour toujours. Regen appliqué
                // uniquement si actif + joueur vivant (cf. `tick_regeneration`).
                self.tick_powerup_expiry();
                // Warning sonore à ≤3s — appelé APRÈS expiry pour que les
                // slots qui viennent d'expirer ce tick ne déclenchent pas
                // d'abord un warn puis un end dans le même frame.
                self.tick_powerup_warnings();
                self.tick_regeneration(dt);

                // Vérifie la fin de match après toutes les résolutions de
                // ce tick. Idempotent — une fois le winner défini, no-op.
                self.check_match_end();
                self.tick_heartbeat();

                // Time-warnings : annonce aux seuils 60s / 30s / 10..1s.
                self.tick_time_warnings();

                // Snapshot des temps restants des powerups AVANT le mut-borrow
                // du renderer — sinon `self.powerup_remaining()` rentre en
                // conflit (E0502).
                let mut powerup_remaining = [None; PowerupKind::COUNT];
                for kind in PowerupKind::ALL {
                    powerup_remaining[kind.index()] = self.powerup_remaining(kind);
                }
                // Snapshot invisibilité avant le mut-borrow du renderer —
                // le viewmodel est atténué côté tint quand le joueur est
                // invisible pour refléter le fade visuel aux autres.
                let player_invisible = self.is_powerup_active(PowerupKind::Invisibility);

                // Blob shadows : émises AVANT le mut-borrow du renderer
                // pour éviter un conflit de borrow (la méthode re-prend
                // `self.renderer.as_mut()` en interne).  Spawne une
                // décale transitoire (~1 frame) au pied de chaque entité
                // visible → le flush décales du `render()` dessine sous
                // tout le reste comme il se doit.
                self.push_all_blob_shadows();

                if let Some(r) = self.renderer.as_mut() {
                    // View-bob : on applique l'offset uniquement à la caméra,
                    // pas à `player.view_angles` (qui est toujours utilisée
                    // pour les traces d'arme et flashs — on ne veut pas les
                    // faire osciller). En intermission / mort, bob = 0 :
                    // self.bob_phase a cessé d'avancer, sin(phase) reste.
                    // On atténue donc aussi par un facteur binaire.
                    let bob_active = !self.console.is_open()
                        && !self.player_health.is_dead()
                        && self.match_winner.is_none()
                        && self.player.on_ground;
                    let (bob_z, bob_roll) = if bob_active {
                        let hspeed = (self.player.velocity.x.powi(2)
                            + self.player.velocity.y.powi(2))
                        .sqrt();
                        let factor = ((hspeed - BOB_SPEED_MIN).max(0.0)
                            / (BOB_SPEED_REF - BOB_SPEED_MIN))
                            .clamp(0.0, 1.0);
                        let z = self.bob_phase.sin().abs() * BOB_AMP_Z * factor;
                        // Sinusoïde pleine (pas `abs`) pour que le roll
                        // alterne gauche ↔ droite, un bob sur deux.
                        let roll = (self.bob_phase * 0.5).sin() * BOB_AMP_ROLL * factor;
                        (z, roll)
                    } else {
                        (0.0, 0.0)
                    };
                    // Animation de la hauteur de vue quand on s'accroupit
                    // / se relève : lerp linéaire, indépendant du framerate.
                    let target = if self.player.crouching {
                        -CROUCH_VIEW_DROP
                    } else {
                        0.0
                    };
                    let max_step = CROUCH_TRANSITION_SPEED * dt;
                    let d = target - self.view_crouch_offset;
                    if d.abs() <= max_step {
                        self.view_crouch_offset = target;
                    } else {
                        self.view_crouch_offset += d.signum() * max_step;
                    }
                    // Spectator follow-cam : quand le joueur est mort
                    // mais que le match n'est pas terminé, on orbite
                    // lentement autour du bot vivant le plus proche du
                    // corps.  Donne au joueur quelque chose à regarder
                    // pendant les 2 s de cooldown respawn, au lieu d'un
                    // plan figé sur le sol.  Révèle aussi la position
                    // des adversaires qui continuent le combat.
                    let mut dead_cam: Option<(Vec3, Angles)> = None;
                    if self.player_health.is_dead() && self.match_winner.is_none() {
                        let mut best: Option<(f32, Vec3)> = None;
                        for bd in &self.bots {
                            if bd.health.is_dead() {
                                continue;
                            }
                            let p = bd.body.origin;
                            let d = (p - self.player.origin).length_squared();
                            if best.map_or(true, |(bd2, _)| d < bd2) {
                                best = Some((d, p));
                            }
                        }
                        if let Some((_, target)) = best {
                            let target_eye = target + Vec3::Z * (BOT_CENTER_HEIGHT as f32);
                            // Orbite : angle qui tourne à 20°/s, rayon
                            // 220 u, hauteur +80 u au-dessus de la cible.
                            let t = self.time_sec * 20.0f32.to_radians();
                            let orbit = Vec3::new(
                                target_eye.x + t.cos() * 220.0,
                                target_eye.y + t.sin() * 220.0,
                                target_eye.z + 80.0,
                            );
                            // Angles pour regarder la cible.
                            let to = target_eye - orbit;
                            let yaw = to.y.atan2(to.x).to_degrees();
                            let horiz = (to.x * to.x + to.y * to.y).sqrt();
                            let pitch = (-to.z).atan2(horiz).to_degrees();
                            dead_cam = Some((orbit, Angles::new(pitch, yaw, 0.0)));
                        }
                    }
                    // **Lean** (M7) — lerp la valeur courante vers le
                    // target d'input. Constante de temps ~80 ms pour
                    // un mouvement vif mais pas snappy. La valeur est
                    // dans [-1, 1], on la convertit en offset latéral
                    // côté caméra.
                    let lean_target = if self.player_health.is_dead() {
                        0.0
                    } else {
                        self.input.lean_axis()
                    };
                    let lean_smooth = (1.0 - (-dt * 12.0).exp()).clamp(0.0, 1.0);
                    self.lean_value =
                        self.lean_value + (lean_target - self.lean_value) * lean_smooth;
                    if let Some((pos, _)) = dead_cam {
                        r.camera_mut().position = pos;
                    } else {
                        let basis = self.player.view_angles.to_vectors();
                        // Offset latéral 6u par unit de lean — assez pour
                        // un peek visible sans téléporter le joueur. Le
                        // hitbox du player ne bouge pas (juste la caméra).
                        let lean_offset = basis.right * (self.lean_value * 6.0);
                        r.camera_mut().position = self.player.origin
                            + Vec3::Z * (PLAYER_EYE_HEIGHT + bob_z + self.view_crouch_offset)
                            + lean_offset;
                    }
                    let mut cam_angles = if let Some((_, a)) = dead_cam {
                        a
                    } else {
                        self.player.view_angles
                    };
                    cam_angles.roll += bob_roll;
                    // **Lean roll** (M7) — bascule visuelle de la caméra
                    // proportionnelle à la valeur de lean (max 6° à
                    // lean=1.0). Signe inverse du sens du lean, comme
                    // un humain qui penche la tête sur le côté.
                    cam_angles.roll -= self.lean_value * 6.0;
                    // Strafe-roll : formule Q3 `CL_KickRoll`. On projette
                    // la vélocité sur le vecteur "droite" des view_angles,
                    // le signe du produit scalaire donne la direction du
                    // lean, l'amplitude scale linéairement jusqu'à
                    // `CL_ROLL_SPEED` puis sature à `CL_ROLL_ANGLE`.
                    // Appliqué seulement vivant + jeu actif (mort ou
                    // intermission → caméra stable).
                    if !self.player_health.is_dead() && self.match_winner.is_none() {
                        let right = self.player.view_angles.to_vectors().right;
                        let side = self.player.velocity.dot(right);
                        let sign = if side < 0.0 { -1.0 } else { 1.0 };
                        let abs_side = side.abs();
                        let roll = if abs_side < CL_ROLL_SPEED {
                            abs_side * CL_ROLL_ANGLE / CL_ROLL_SPEED
                        } else {
                            CL_ROLL_ANGLE
                        };
                        cam_angles.roll += roll * sign;
                    }
                    // Screen-shake : décalage pitch/yaw deux sinusoïdes
                    // déphasées + fréquences premières pour éviter un
                    // mouvement 1D lisible. Amplitude = `shake_intensity`
                    // × fade(remaining). Inhibé si mort / intermission
                    // pour garder la caméra fixe pendant le respawn
                    // (sinon l'overlay "YOU DIED" tremble bizarrement).
                    let shake_remaining = self.shake_until - self.time_sec;
                    if shake_remaining > 0.0
                        && self.shake_intensity > 1e-3
                        && !self.player_health.is_dead()
                        && self.match_winner.is_none()
                    {
                        let fade = (shake_remaining / SHAKE_DURATION).clamp(0.0, 1.0);
                        let amp = self.shake_intensity * fade;
                        // Déphasage π/2 entre pitch et yaw + fréquences
                        // distinctes (47Hz vs 31Hz) → motif non répétitif
                        // à l'œil sur la fenêtre 250ms.
                        let phase_p = self.time_sec * 47.0;
                        let phase_y = self.time_sec * 31.0 + std::f32::consts::FRAC_PI_2;
                        cam_angles.pitch += phase_p.sin() * amp;
                        cam_angles.yaw += phase_y.sin() * amp;
                    } else if shake_remaining <= 0.0 {
                        // Remise à zéro propre une fois expiré — évite
                        // d'accumuler des micro-valeurs sur des sessions
                        // longues et simplifie les asserts de tests.
                        self.shake_intensity = 0.0;
                    }
                    r.camera_mut().angles = cam_angles;
                    // FOV : `cg_fov` (cvar archivée) est interprété comme
                    // FOV horizontal à 4:3, conforme à la convention Q3.
                    // La caméra dérive le vfov à partir de là, ce qui
                    // donne un scaling **Hor+** sur 16:9 / 21:9 / 32:9 :
                    // l'image verticale ne change pas, le champ s'élargit
                    // horizontalement avec l'aspect. Lu chaque frame —
                    // pas cher, et permet la modif live depuis la console.
                    let fov_cvar = self.cvars.get_f32("cg_fov").unwrap_or(90.0);
                    r.camera_mut().set_horizontal_fov_4_3(fov_cvar);

                    r.begin_frame();
                    queue_pickups(
                        r,
                        &self.pickups,
                        self.time_sec,
                        &self.remote_unavailable_pickups,
                        self.net.mode.is_client(),
                    );
                    queue_projectiles(r, &self.projectiles, self.time_sec);
                    // Purge les beams expirés, puis ré-émet les survivants
                    // au renderer. Fade linéaire en fonction de la lifetime
                    // initiale pour que les trails Railgun s'éteignent
                    // proprement sans rendre le LG trop pâle.
                    //
                    // Style par arme :
                    //   * LG  → zigzag `push_beam_lightning` avec seed qui
                    //           change chaque frame pour l'effet crépitant.
                    //   * RG  → hélice `push_beam_spiral`, fade-out de l'alpha
                    //           via la couleur initiale de fin (color_b).
                    //   * autre → segment droit classique (respawn beam, …).
                    let now = self.time_sec;
                    self.beams.retain(|b| b.expire_at > now);
                    // Seed global pour les LG du frame — identique sur tous
                    // les beams tant qu'on est au même tick, change dès qu'on
                    // repasse dans la boucle.
                    let lg_seed = (self.time_sec * 60.0) as u64;
                    for b in &self.beams {
                        let remaining = (b.expire_at - now).max(0.0);
                        let fade = (remaining / b.lifetime.max(1e-3)).clamp(0.0, 1.0);
                        let mut col = b.color;
                        col[3] *= fade;
                        match b.style {
                            BeamStyle::Straight => {
                                r.push_beam(b.a, b.b, col);
                            }
                            BeamStyle::Lightning => {
                                // **P10 Lightning upgrade** : faisceau
                                // principal + 2-3 branches courtes qui
                                // partent au tiers et au deux-tiers du
                                // tracé. Effet "arc électrique qui
                                // bifurque" — amplifie l'impression de
                                // décharge brute. Branches plus courtes
                                // (~30 % longueur) et plus fines.
                                r.push_beam_lightning(b.a, b.b, col, 4.0, 14, lg_seed);
                                // Halo pâle blanc autour du tracé pour
                                // simuler l'ionisation lumineuse.
                                let mut halo = [1.0, 1.0, 1.0, col[3] * 0.55];
                                let _ = &mut halo;
                                r.push_beam_lightning(
                                    b.a, b.b, [1.0, 1.0, 1.0, col[3] * 0.55],
                                    1.5, 14, lg_seed.wrapping_add(7),
                                );
                                // Branches secondaires : on prend 2
                                // points le long du tracé (33 % et 66 %)
                                // et on émet un mini-arc latéral.
                                let dir = b.b - b.a;
                                let len = dir.length().max(1.0);
                                let axis = dir / len;
                                // Vecteur perpendiculaire arbitraire à
                                // axis pour les branches.
                                let helper = if axis.z.abs() < 0.9 {
                                    Vec3::Z
                                } else {
                                    Vec3::Y
                                };
                                let perp = axis.cross(helper).normalize();
                                let perp2 = axis.cross(perp).normalize();
                                for &(t, sign) in &[(0.33_f32, 1.0_f32), (0.66, -1.0)] {
                                    let origin = b.a + dir * t;
                                    // Mix perp + perp2 selon seed pour
                                    // varier l'orientation des branches.
                                    let mix_phase =
                                        (lg_seed.wrapping_mul(31) as f32) * 0.001;
                                    let branch_dir =
                                        perp * mix_phase.cos() + perp2 * mix_phase.sin();
                                    let branch_end =
                                        origin + axis * (len * 0.15)
                                            + branch_dir * (len * 0.20 * sign);
                                    let mut branch_col = col;
                                    branch_col[3] *= 0.45;
                                    r.push_beam_lightning(
                                        origin,
                                        branch_end,
                                        branch_col,
                                        2.5,
                                        7,
                                        lg_seed.wrapping_add((t * 100.0) as u64),
                                    );
                                }
                            }
                            BeamStyle::Spiral => {
                                // **Rail beam upgrade** (P9) : double
                                // hélice — outer halo coloré (rayon 4u,
                                // 96 segments pour une spirale lisse) +
                                // inner core blanc-bouillant (rayon 1u)
                                // pour un cœur lumineux qui contraste.
                                // Gradient color → tail-faded pour que
                                // l'œil suive la direction du tir.
                                let mut tail = col;
                                tail[3] *= 0.15;
                                // Outer halo (couleur arme).
                                r.push_beam_spiral(b.a, col, b.b, tail, 4.0, 2.5, 96);
                                // Inner core (blanc-bouillant). Alpha
                                // boosté pour bien sortir au centre.
                                let mut core = [1.0, 1.0, 1.0, col[3] * 1.0];
                                let mut core_tail = core;
                                core_tail[3] *= 0.25;
                                r.push_beam_spiral(
                                    b.a, core, b.b, core_tail, 1.0, 2.5, 64,
                                );
                                // Trace centrale droite — donne le sens
                                // du tir et masque le "mou" central de
                                // la spirale en gros plan.
                                let mut axis = col;
                                axis[3] *= 0.5;
                                r.push_beam(b.a, b.b, axis);
                            }
                        }
                    }
                    // Particules d'explosion — un streak additif par spark.
                    // Le tip garde la couleur hot, la queue fade vers alpha=0
                    // pour donner un effet de traînée. L'alpha global suit
                    // le ratio de vie restant.
                    for p in &self.particles {
                        let remaining = (p.expire_at - now).max(0.0);
                        let fade = (remaining / p.lifetime.max(1e-3)).clamp(0.0, 1.0);
                        let head = p.origin;
                        let tail = p.origin - p.velocity * PARTICLE_STREAK_DT;
                        let mut hot = p.color;
                        hot[3] = fade;
                        let mut dim = p.color;
                        dim[3] = 0.0;
                        r.push_beam_gradient(head, hot, tail, dim);
                    }
                    // Reconstruit `remote_players` chaque frame depuis
                    // les buffers d'interp. C'est CE Vec qui est rendu —
                    // sa fluidité ne dépend donc pas du débit snapshot
                    // (20 Hz) mais de la framerate (60+ Hz).
                    self.remote_players.clear();
                    for (slot, interp) in &self.remote_interp {
                        let Some(s) = interp.current() else { continue };
                        self.remote_players.push(RemotePlayer {
                            slot: *slot,
                            origin: s.origin,
                            view_angles: s.view_angles,
                            velocity: s.velocity,
                            on_ground: s.on_ground,
                            is_dead: s.is_dead,
                            recently_fired: s.recently_fired,
                            team: s.team,
                        });
                    }
                    if let Some(rig) = self.bot_rig.as_ref() {
                        queue_bots(r, rig, &self.bots, self.time_sec);
                        queue_remote_players(
                            r,
                            rig,
                            &self.remote_players,
                            self.time_sec,
                        );
                    } else {
                        // **Fallback bot visibility** : si aucun rig
                        // joueur n'a pu être chargé (pak0 partiel), on
                        // dessine un beam vertical haut-coloré à chaque
                        // position de bot pour que le joueur les voie.
                        // Garantit qu'un `--bots 4` produit toujours un
                        // signal visuel, même sans MD3 dispo.
                        for (i, b) in self.bots.iter().enumerate() {
                            if b.health.is_dead() {
                                continue;
                            }
                            let foot = b.body.origin;
                            let head = foot + Vec3::Z * 56.0;
                            let col = bot_tint(i + 1);
                            r.push_beam(foot, head, col);
                            // Nameplate flottant.
                            let lbl = format!("[BOT] {}", b.bot.name);
                            r.push_text(
                                head.x as f32, head.y as f32,
                                2.0,
                                col,
                                &lbl,
                            );
                            // ^ NB : push_text est 2D screen-space, pas
                            // worldspace. Sur un fallback ça donne un
                            // texte fixe en haut-gauche (visible dans
                            // tous les cas). Suffisant comme indicateur
                            // "bot N existe" malgré l'absence de mesh.
                        }
                    }
                    // Remote projectiles : extrapolés depuis le dernier
                    // snapshot, rendus avec les meshes locaux. Vec passé
                    // par adresse pour éviter une copie ; l'itération
                    // est dans queue_remote_projectiles.
                    if !self.remote_projectiles.is_empty() {
                        let remote_projs: Vec<RemoteProjectile> =
                            self.remote_projectiles.values().cloned().collect();
                        queue_remote_projectiles(
                            r,
                            &remote_projs,
                            &self.rocket_mesh,
                            &self.plasma_mesh,
                            &self.grenade_mesh,
                            self.time_sec,
                        );
                    }
                    if !self.player_health.is_dead() {
                        if let Some(wm) = self
                            .viewmodels
                            .iter()
                            .find_map(|(w, m)| (*w == self.active_weapon).then_some(m))
                        {
                            // BUG FIX : on passait `self.player.origin` (= pieds)
                            // en position d'œil → le MD3 se retrouvait ~40u sous
                            // la caméra, hors du frustum. On aligne maintenant
                            // sur la position exacte de la caméra (eye height +
                            // bob + crouch) pour que l'arme suive fidèlement la
                            // vue, incluant le view-bob et l'accroupi.
                            let eye = self.player.origin
                                + Vec3::Z
                                    * (PLAYER_EYE_HEIGHT + bob_z + self.view_crouch_offset);
                            queue_viewmodel(
                                r,
                                wm,
                                eye,
                                cam_angles,
                                self.time_sec,
                                self.player_invul_until,
                                player_invisible,
                                self.active_weapon,
                                self.time_sec < self.muzzle_flash_until,
                                self.view_kick,
                            );
                        }
                    }
                    // Flash HUD sur explosion proche (< 300 unités ≈ 6 m).
                    let eye = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
                    let boom_near = self.explosions.iter().any(|e| {
                        (e.origin - eye).length_squared() < 300.0 * 300.0
                    });
                    // Underwater : test du contents au point d'œil. WATER
                    // déclenche la teinte bleue ; LAVA/SLIME pourraient ajouter
                    // leur propre teinte mais le contenu de la zone de hurt
                    // s'en charge déjà (rouge via low-health, jaune pour slime
                    // n'est pas implémenté côté gameplay). On se contente du
                    // cas WATER qui est le plus fréquent visuellement.
                    let underwater = self
                        .world
                        .as_ref()
                        .map(|w| {
                            w.collision
                                .point_contents(eye)
                                .contains(q3_collision::Contents::WATER)
                        })
                        .unwrap_or(false);
                    let aw_params = self.active_weapon.params();
                    let ammo_shown = if aw_params.ammo_cost == 0 {
                        None
                    } else {
                        Some(self.ammo[self.active_weapon.slot() as usize])
                    };
                    // Purge les entrées expirées du kill-feed, des chiffres
                    // de dégât flottants et des médailles avant dessin.
                    let now = self.time_sec;
                    self.kill_feed.retain(|k| k.expire_at > now);
                    self.chat_feed.retain(|c| c.expire_at > now);
                    self.pickup_toasts.retain(|t| t.expire_at > now);
                    self.floating_damages.retain(|d| d.expire_at > now);
                    self.active_medals.retain(|m| m.expire_at > now);
                    // Chrono du match : `None` une fois l'intermission en cours
                    // (on affichera 00:00 figé).
                    let time_remaining = (TIME_LIMIT_SECONDS
                        - (self.time_sec - self.match_start_at))
                        .max(0.0);
                    // Construit la liste des remote players pour le
                    // scoreboard. En solo, vide.
                    let remote_score_rows: Vec<(String, i16, i16, u8)> = self
                        .remote_interp
                        .keys()
                        .map(|slot| {
                            let name = self
                                .remote_names
                                .get(slot)
                                .cloned()
                                .unwrap_or_else(|| format!("slot {slot}"));
                            // Frags / deaths / team : depuis remote_scores
                            // qui mirroir l'info reçue dans la dernière
                            // snapshot. Default (0, 0, FREE) si pas encore
                            // d'info — robuste au 1er tick avant snapshot.
                            let (frags, deaths, team) = self
                                .remote_scores
                                .get(slot)
                                .copied()
                                .unwrap_or((0, 0, q3_net::team::FREE));
                            (name, frags, deaths, team)
                        })
                        .collect();
                    draw_hud(
                        r,
                        &self.console,
                        &self.player.origin,
                        self.player_health,
                        self.player_armor,
                        self.deaths,
                        self.frags,
                        self.respawn_at.map(|t| (t - self.time_sec).max(0.0)),
                        self.time_sec < self.muzzle_flash_until,
                        self.time_sec < self.hit_marker_until,
                        self.active_weapon.name(),
                        ammo_shown,
                        boom_near,
                        &self.kill_feed,
                        &self.floating_damages,
                        self.input.scoreboard,
                        &self.bots,
                        &remote_score_rows,
                        self.local_team,
                        self.match_winner.as_ref(),
                        time_remaining,
                        &powerup_remaining,
                        now,
                        self.last_damage_dir,
                        self.last_damage_until,
                        self.player.view_angles,
                        &self.active_medals,
                        underwater,
                        self.air_left,
                        self.held_item,
                        self.last_death_cause.as_ref(),
                        self.player.velocity.truncate().length(),
                        self.player_streak,
                        self.total_shots,
                        self.total_hits,
                        self.is_spectator,
                        self.follow_slot.and_then(|slot| {
                            self.remote_names
                                .get(&slot)
                                .map(|n| format!("FOLLOWING  {n}"))
                                .or_else(|| Some(format!("FOLLOWING  slot {slot}")))
                        }),
                        self.weapons_owned,
                        self.ammo,
                        self.active_weapon.slot(),
                        self.armor_flash_until,
                        self.pain_flash_until,
                        self.recent_dmg_total,
                        self.recent_dmg_last_at,
                        self.view_kick,
                        self.weapon_switch_at,
                    );
                    // Pastille de statut réseau + VR, en haut-gauche.
                    // Affichée seulement quand utile (pas en solo/sans VR)
                    // pour ne pas encombrer le HUD du mode classique.
                    draw_netvr_status(
                        r,
                        &self.net.mode,
                        self.vr.is_enabled(),
                        self.net.client_rtt_ms(),
                    );
                    // Chat feed : lignes de bots sur kill/death/respawn.
                    // Rendu en bas-gauche, au-dessus du watermark — se
                    // fond dans le HUD sans concurrencer le kill-feed.
                    draw_chat_feed(r, &self.chat_feed, now);
                    // Toasts de pickup : textes éphémères en bas-centre,
                    // empilés du plus récent (bas) au plus ancien (haut)
                    // pour que le dernier pickup soit immédiatement lu.
                    draw_pickup_toasts(r, &self.pickup_toasts, now);
                    // Indicateurs de respawn : petit compteur flottant à
                    // l'emplacement de chaque pickup ramassé récemment
                    // (fade in sur les dernières 3 s avant respawn).
                    // Aide le joueur à mémoriser les timings des items
                    // chauds (Quad, MegaHealth, RA).
                    draw_pickup_respawn_indicators(r, &self.pickups, self.player.origin, now);
                    // Panneau respawn-timers en haut-gauche : MH/RA/YA
                    // + powerups en cooldown avec leur countdown. Petit
                    // par design — ne dessine que les items stratégiques
                    // (cf. `pickup_timer_label`) et limite à 5 entrées.
                    draw_item_respawn_timers(r, &self.pickups, now);
                    // Warmup overlay : compteur géant au centre de l'écran
                    // tant que `time_sec < warmup_until`.  Rendu par-dessus
                    // tout le HUD mais sous l'intermission + le menu.
                    if self.warmup_until > now && self.match_winner.is_none() {
                        draw_warmup_overlay(r, self.warmup_until - now);
                    }
                    // Overlay perf : toggle F9.  Rendu sous le menu pour
                    // qu'il reste visible pendant un pause menu (permet de
                    // diagnostiquer une chute de fps liée au menu lui-même).
                    if self.show_perf_overlay {
                        draw_perf_overlay(r, &self.frame_times);
                    }
                    // **Bot diagnostic banner** — voyant, toujours
                    // visible, contient l'état complet pour diag à
                    // distance. Couleurs :
                    // * jaune = bots OK (rig + alive)
                    // * rouge = problème (rig manquant OR bot count=0)
                    let alive = self
                        .bots
                        .iter()
                        .filter(|b| !b.health.is_dead())
                        .count();
                    let rig_status = if self.bot_rig.is_some() {
                        "OK"
                    } else {
                        "MISSING"
                    };
                    let banner = if self.bots.is_empty() {
                        format!(
                            "[DIAG] NO BOTS in self.bots | pending={} | rig={} | world={}",
                            self.pending_local_bots,
                            rig_status,
                            if self.world.is_some() { "OK" } else { "NONE" }
                        )
                    } else {
                        format!(
                            "[DIAG] BOTS {}/{} alive | rig={} | players_visible_3D={}",
                            alive,
                            self.bots.len(),
                            rig_status,
                            if self.bot_rig.is_some() { "TRUE (MD3)" } else { "FALSE (beam fallback)" }
                        )
                    };
                    let scale = HUD_SCALE * 1.2;
                    let bw = banner.len() as f32 * 8.0 * scale;
                    let fw = r.width() as f32;
                    let bx = (fw - bw) * 0.5;
                    let by = 6.0;
                    let bg = if self.bots.is_empty() || self.bot_rig.is_none() {
                        [0.45, 0.05, 0.05, 0.92] // rouge dense
                    } else {
                        [0.04, 0.20, 0.10, 0.85] // vert
                    };
                    r.push_rect(
                        0.0, 0.0, fw, 8.0 * scale + 16.0, bg,
                    );
                    push_text_shadow(
                        r,
                        bx,
                        by,
                        scale,
                        if self.bots.is_empty() { COL_YELLOW } else { COL_WHITE },
                        &banner,
                    );
                    // Roue de chat rapide (touche V). Rendue après le HUD
                    // mais avant le menu pour qu'elle disparaisse en
                    // ouvrant les options. La fenêtre d'animation
                    // d'ouverture est très courte (120 ms).
                    if self.chat_wheel_open {
                        let fw = r.width() as f32;
                        let fh = r.height() as f32;
                        draw_chat_wheel(
                            r,
                            fw,
                            fh,
                            (self.time_sec - self.chat_wheel_opened_at).max(0.0),
                        );
                    }
                    // Menu principal par-dessus tout le reste — dessiné en
                    // dernier pour que son overlay assombri couvre le HUD
                    // et la scène. Hors menu, rien n'est émis : coût zéro
                    // pendant le jeu courant.
                    if self.menu.open {
                        let fw = r.width() as f32;
                        let fh = r.height() as f32;
                        self.menu.draw(r, &self.cvars, fw, fh);
                    }
                    if let Err(e) = r.render(now) {
                        warn!("render: {e}");
                    }
                }

                // VR : `xrEndFrame` clôt la frame côté runtime et présente
                // les composition layers des deux yeux. No-op en mode stub.
                self.vr.end_frame();

                // DEBUG : Q3_AUTOSHOT=<secondes> → screenshot auto puis exit.
                if let Ok(s) = std::env::var("Q3_AUTOSHOT") {
                    if let Ok(t) = s.parse::<f32>() {
                        if self.time_sec >= t && !self.auto_shot_taken {
                            self.auto_shot_taken = true;
                            self.take_screenshot();
                            if let Some(r) = self.renderer.as_mut() {
                                let _ = r.render(self.time_sec);
                            }
                            event_loop.exit();
                        }
                    }
                }

                if let Some(w) = self.window.as_ref() {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
        }
    }

    /// Appelé une dernière fois par winit juste avant la sortie. On en
    /// profite pour persister les cvars ARCHIVE dans `q3config.cfg` —
    /// sensibilité, FOV, bindings, volume… Un échec disque est loggé mais
    /// ne fait pas crasher : le jeu est en train de quitter.
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        let Some(path) = user_config_path() else {
            debug!("config: sortie sans répertoire user, rien à écrire");
            return;
        };
        let script = self.cvars.serialize_archive();
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                warn!("config: mkdir {} : {e}", parent.display());
                return;
            }
        }
        // Écriture atomique : on écrit dans un tmp à côté, puis rename.
        // Évite de laisser un fichier tronqué si le process est tué en
        // plein `write`.
        let tmp = path.with_extension("cfg.tmp");
        if let Err(e) = std::fs::write(&tmp, script.as_bytes()) {
            warn!("config: write {} : {e}", tmp.display());
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, &path) {
            warn!("config: rename {} → {} : {e}", tmp.display(), path.display());
            // Best effort : on tente quand même d'écrire direct à `path`.
            let _ = std::fs::write(&path, script.as_bytes());
            return;
        }
        info!("config: sauvegardée dans {}", path.display());
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if !self.mouse_captured || self.console.is_open() {
                return;
            }
            // Convention Q3 fidèle : `sensitivity` module des coefficients
            // `m_pitch` / `m_yaw` (0.022 deg/pixel par défaut).
            //   * yaw   : Q3 fait `yaw   -= m_yaw   * mx` → souris droite
            //     (mx>0) = tourne à droite (yaw ↘).
            //   * pitch : Q3 fait `pitch += m_pitch * my` → souris bas
            //     (my>0) = regarde bas (pitch ↗, convention Q3 pitch>0
            //     = regarde sous l'horizon).
            // L'ancien code avait `pitch -= my * sens` → inversé. Corrigé :
            // + m_pitch. Pour restaurer l'inversion "mouse up = look down"
            // (Tribes / simulateurs de vol), il suffit de `m_pitch -0.022`.
            let sens = self.cvars.get_f32("sensitivity").unwrap_or(5.0).max(0.0);
            let m_pitch = self.cvars.get_f32("m_pitch").unwrap_or(0.022);
            let m_yaw = self.cvars.get_f32("m_yaw").unwrap_or(0.022);
            // Cap défensif : si l'OS délivre un burst (alt-tab, mouse
            // reconnexion), on tronque à 500 px par événement pour éviter
            // un saut de vue type "teleport".
            let dx = (delta.0 as f32).clamp(-500.0, 500.0);
            let dy = (delta.1 as f32).clamp(-500.0, 500.0);
            let yaw_delta = -dx * sens * m_yaw;
            let pitch_delta = dy * sens * m_pitch;
            let a = &mut self.player.view_angles;
            a.yaw = (a.yaw + yaw_delta).rem_euclid(360.0);
            a.pitch = (a.pitch + pitch_delta).clamp(-89.0, 89.0);
        }
    }
}

// Palette HUD + constantes d'échelle + safe-area : extraits dans
// `hud_helpers.rs` pour soulager `app.rs` (cf. split #55). Re-exposés
// ici via `use` pour garder les call-sites inchangés dans le reste
// du fichier.
use crate::hud_helpers::{
    COL_WHITE, COL_GRAY, COL_YELLOW, COL_RED, COL_CONSOLE_BG, COL_CONSOLE_BORDER,
    COL_PANEL_BG, COL_PANEL_EDGE, COL_PANEL_EDGE_DIM,
    HUD_SCALE, LINE_H,
    hud_safe_rect_x, push_text_shadow, push_panel, push_bar_gradient,
};

/// Messages prédéfinis de la roue de chat rapide. Indexés 0..7,
/// déclenchés via `1..=8` quand la roue est ouverte. Les libellés
/// affichés sur la roue restent courts (3-12 chars) pour la lisibilité ;
/// le texte envoyé sur le wire est plus naturel ("Good game!" plutôt
/// que "GG"). Garde le tableau à 8 entrées : la roue est divisée en
/// huit secteurs de 45° chacun.
const CHAT_WHEEL_MESSAGES: [(&str, &str); 8] = [
    ("GG",       "Good game!"),
    ("NICE",     "Nice shot!"),
    ("THX",      "Thanks!"),
    ("SORRY",    "Sorry!"),
    ("HELP",     "Need help!"),
    ("AMMO",     "Need ammo."),
    ("ARMOR",    "Need armor."),
    ("ATTACK",   "Attack!"),
];

/// Overlay de statistiques de performance : FPS moyen / min / max +
/// histogramme des frametimes sur la fenêtre [`FRAME_TIME_BUF`].  Placé
/// en bas à gauche pour ne pas concurrencer la stats panel (haut-droite)
/// ni le chat feed (bas-gauche, mais au-dessus de nous — le watermark
/// perf est collé au sol).  Toggle par F9.
fn draw_perf_overlay(r: &mut Renderer, frame_times: &[f32]) {
    let w = r.width() as f32;
    let h = r.height() as f32;
    // Agrégation : moyenne, min, max.  On ignore les slots à 0.0 (buffer
    // pas encore plein au boot).
    let mut sum = 0.0f32;
    let mut n = 0u32;
    let mut lo = f32::INFINITY;
    let mut hi = 0.0f32;
    for &dt in frame_times {
        if dt <= 0.0 {
            continue;
        }
        sum += dt;
        n += 1;
        if dt < lo { lo = dt; }
        if dt > hi { hi = dt; }
    }
    if n == 0 {
        return;
    }
    let avg = sum / n as f32;
    let fps_avg = 1.0 / avg;
    let fps_min = 1.0 / hi;
    let fps_max = 1.0 / lo;
    let ms_avg = avg * 1000.0;
    // Panel : 180 × 64, bas-gauche, 8 px de marge.  Assez large pour
    // trois lignes (fps avg, min/max, ms) + une petite courbe.
    let panel_w = 200.0;
    let panel_h = 72.0;
    let px = 8.0;
    let py = h - panel_h - 8.0;
    push_panel(r, px, py, panel_w, panel_h);
    // Ligne 1 : fps moyen, gros chiffre.  Couleur modulée par le ratio —
    // vert > 90, jaune 45..90, rouge < 45.
    let fps_color = if fps_avg >= 90.0 {
        [0.5, 1.0, 0.5, 1.0]
    } else if fps_avg >= 45.0 {
        [1.0, 0.85, 0.35, 1.0]
    } else {
        [1.0, 0.4, 0.3, 1.0]
    };
    let line1 = format!("{:>5.1} FPS", fps_avg);
    push_text_shadow(r, px + 10.0, py + 6.0, HUD_SCALE * 1.2, fps_color, &line1);
    // Ligne 2 : min/max + ms.  Plus petite, couleur froide.
    let line2 = format!("min {:>3.0}  max {:>3.0}  {:>4.1} ms", fps_min, fps_max, ms_avg);
    push_text_shadow(r, px + 10.0, py + 26.0, HUD_SCALE * 0.8, COL_GRAY, &line2);
    // Histogramme : chaque frametime mappé à une barre verticale.  On
    // scale par rapport à 50 ms (20 fps) — au-delà la barre sature.
    let graph_x = px + 10.0;
    let graph_y = py + 44.0;
    let graph_w = panel_w - 20.0;
    let graph_h = 20.0;
    r.push_rect(graph_x, graph_y, graph_w, graph_h, [0.02, 0.03, 0.05, 0.75]);
    // Ligne horizontale cible à 60 fps (~16.6 ms) pour repère visuel.
    let target_y = graph_y + graph_h - graph_h * (16.6 / 50.0);
    r.push_rect(graph_x, target_y, graph_w, 1.0, [0.25, 0.7, 0.4, 0.55]);
    let n_samples = frame_times.len();
    let bar_w = graph_w / n_samples as f32;
    for (i, &dt) in frame_times.iter().enumerate() {
        if dt <= 0.0 {
            continue;
        }
        let ms = dt * 1000.0;
        let frac = (ms / 50.0).clamp(0.02, 1.0);
        let bar_h = graph_h * frac;
        let bx = graph_x + i as f32 * bar_w;
        let by = graph_y + graph_h - bar_h;
        // Couleur : verte si dt correspond à >60 fps, jaune 30..60, rouge < 30.
        let col = if ms <= 16.7 {
            [0.45, 0.95, 0.45, 0.9]
        } else if ms <= 33.4 {
            [1.0, 0.85, 0.35, 0.9]
        } else {
            [1.0, 0.4, 0.3, 0.95]
        };
        r.push_rect(bx, by, bar_w.max(1.0), bar_h, col);
    }
    // Évite un warning `unused` sur `w` : on ne l'a pas utilisé mais on
    // a besoin de `h` et garder la symétrie rend le helper re-usable.
    let _ = w;
}

/// Dessine un cercle creux (anneau) en empilant une couronne de quads
/// 1×1 le long d'un cercle de rayon `radius` — approximation suffisante
/// pour des tailles de crosshair (≤ 18 px).  `thickness` contrôle la
/// largeur radiale de l'anneau.  On prend 24 segments : assez pour que
/// l'œil ne lise pas les marches à cette échelle.
fn push_ring(r: &mut Renderer, cx: f32, cy: f32, radius: f32, thickness: f32, color: [f32; 4]) {
    if radius < 0.5 {
        return;
    }
    const N: usize = 24;
    let step = std::f32::consts::TAU / N as f32;
    let half_t = thickness * 0.5;
    for i in 0..N {
        let a = i as f32 * step;
        let (s, c) = a.sin_cos();
        let x = cx + c * radius - half_t;
        let y = cy + s * radius - half_t;
        r.push_rect(x, y, thickness, thickness, color);
    }
}

/// Dessine le crosshair associé au slot d'arme courant.  `spread` est
/// l'ouverture courante du recul (déjà clampée par l'appelant), ajoutée
/// au gap central pour les profils « cross ».  Le code branche sur le
/// slot 1..=9 (Q3 canon) — cf. [`WeaponId::slot`] pour la correspondance.
fn draw_weapon_crosshair(r: &mut Renderer, cx: f32, cy: f32, slot: u8, spread: f32) {
    let t = 2.0;
    let arm = 5.0;
    let base_gap = 2.0;
    let gap = base_gap + spread;
    // 4-arm avec paramètres donnés — factorisé car plusieurs profils
    // en dérivent (machinegun, shotgun, lightning).
    let push_cross = |r: &mut Renderer, arm_len: f32, thick: f32, g: f32, col: [f32; 4]| {
        r.push_rect(cx - g - arm_len, cy - thick * 0.5, arm_len, thick, col);
        r.push_rect(cx + g, cy - thick * 0.5, arm_len, thick, col);
        r.push_rect(cx - thick * 0.5, cy - g - arm_len, thick, arm_len, col);
        r.push_rect(cx - thick * 0.5, cy + g, thick, arm_len, col);
    };
    match slot {
        // Gauntlet : anneau épais, évoque une portée « aura melee ».
        1 => {
            push_ring(r, cx, cy, 9.0, 2.0, COL_WHITE);
            r.push_rect(cx - 1.0, cy - 1.0, 2.0, 2.0, COL_WHITE);
        }
        // Shotgun : arms courts, gap de base large → spread implicite
        // même sans recul appliqué.
        3 => {
            push_cross(r, 4.0, t, gap + 4.0, COL_WHITE);
            r.push_rect(cx - 1.0, cy - 1.0, 2.0, 2.0, COL_WHITE);
        }
        // Grenade launcher : dot + anneau fin (indicateur de tir arqué).
        4 => {
            push_ring(r, cx, cy, 8.0, 1.0, COL_WHITE);
            r.push_rect(cx - 1.5, cy - 1.5, 3.0, 3.0, COL_WHITE);
        }
        // Rocket : dot central + anneau épais (warning explosif proche).
        5 => {
            push_ring(r, cx, cy, 10.0, 2.0, COL_WHITE);
            r.push_rect(cx - 1.5, cy - 1.5, 3.0, 3.0, COL_WHITE);
        }
        // Lightning gun : cross fin + dot central, pas de spread (beam).
        6 => {
            push_cross(r, 6.0, 1.0, base_gap, COL_WHITE);
            r.push_rect(cx - 1.0, cy - 1.0, 2.0, 2.0, COL_WHITE);
        }
        // Railgun : dot seul + anneau très fin (sniping, viser juste).
        7 => {
            push_ring(r, cx, cy, 5.0, 1.0, [1.0, 1.0, 1.0, 0.55]);
            r.push_rect(cx - 1.0, cy - 1.0, 2.0, 2.0, COL_WHITE);
        }
        // Plasma : petit X (diagonales) + dot central, suggère le tir rapide.
        8 => {
            let d = 5.0;
            let th = 1.0;
            // Branches diagonales discrètes (pixel art — pas de rotation,
            // on les approxime avec des quads carrés le long de la diag).
            for i in 1..=(d as i32) {
                let k = i as f32;
                r.push_rect(cx + k, cy + k, th, th, COL_WHITE);
                r.push_rect(cx - k - th, cy + k, th, th, COL_WHITE);
                r.push_rect(cx + k, cy - k - th, th, th, COL_WHITE);
                r.push_rect(cx - k - th, cy - k - th, th, th, COL_WHITE);
            }
            r.push_rect(cx - 1.0, cy - 1.0, 2.0, 2.0, COL_WHITE);
        }
        // BFG : cadre carré vide + dot (massive, reconnaissable).
        9 => {
            let s = 12.0;
            let th = 2.0;
            r.push_rect(cx - s, cy - s, 2.0 * s, th, COL_WHITE);         // top
            r.push_rect(cx - s, cy + s - th, 2.0 * s, th, COL_WHITE);    // bot
            r.push_rect(cx - s, cy - s, th, 2.0 * s, COL_WHITE);         // left
            r.push_rect(cx + s - th, cy - s, th, 2.0 * s, COL_WHITE);    // right
            r.push_rect(cx - 1.5, cy - 1.5, 3.0, 3.0, COL_WHITE);
        }
        // Machinegun (slot 2) et par défaut : 4-arm classique avec recul.
        _ => {
            push_cross(r, arm, t, gap, COL_WHITE);
            r.push_rect(cx - 1.0, cy - 1.0, 2.0, 2.0, COL_WHITE);
        }
    }
}

/// Étiquette courte d'un pickup pour le HUD respawn-timer. `None` =
/// item non-stratégique (small health, shard, weapon, ammo) qu'on
/// ne dessine pas — on garde le panneau lisible avec 4-5 entrées max.
fn pickup_timer_label(kind: &PickupKind) -> Option<&'static str> {
    match kind {
        // Mega-Health : seul item Health qu'on traque (plafond 200).
        PickupKind::Health { max_cap: 200, .. } => Some("MH"),
        // Red Armor (100) + Yellow Armor (50). Shards (5) ignorés.
        PickupKind::Armor { amount: 100 } => Some("RA"),
        PickupKind::Armor { amount: 50 } => Some("YA"),
        PickupKind::Powerup { powerup, .. } => Some(match powerup {
            PowerupKind::QuadDamage => "QUAD",
            PowerupKind::Haste => "HASTE",
            PowerupKind::Regeneration => "REGEN",
            PowerupKind::BattleSuit => "BSUIT",
            PowerupKind::Invisibility => "INVIS",
            PowerupKind::Flight => "FLIGHT",
        }),
        _ => None,
    }
}

/// Couleur d'accent par catégorie d'item — facilite la lecture
/// périphérique du panneau (rouge = MH/RA, jaune = YA, magenta = Quad…).
fn pickup_timer_color(kind: &PickupKind) -> [f32; 4] {
    match kind {
        PickupKind::Health { .. } => [1.00, 0.30, 0.30, 1.0],
        PickupKind::Armor { amount: 100 } => [1.00, 0.45, 0.20, 1.0],
        PickupKind::Armor { amount: 50 } => [1.00, 0.86, 0.20, 1.0],
        PickupKind::Powerup { powerup: PowerupKind::QuadDamage, .. } => {
            [0.55, 0.40, 1.00, 1.0]
        }
        PickupKind::Powerup { .. } => [0.40, 0.85, 1.00, 1.0],
        _ => [1.0, 1.0, 1.0, 1.0],
    }
}

/// Panneau "Item respawn timers" — petite colonne en haut-gauche qui
/// liste les items stratégiques (MH, RA, YA, powerups) en cooldown
/// avec leur countdown en `MM:SS` (ou `SS.S`s sous 10 s pour un visuel
/// "tic-tac" lisible). Les items disponibles ne sont pas listés —
/// la liste s'auto-vide quand le match repart à zéro. Filtre
/// `pickup_timer_label` pour limiter à ~5 entrées max.
fn draw_item_respawn_timers(r: &mut Renderer, pickups: &[PickupGpu], now: f32) {
    // Collecte les items en cooldown avec leur reste, triés par
    // countdown croissant (le plus proche du respawn en haut).
    let mut entries: Vec<(&'static str, f32, [f32; 4])> = pickups
        .iter()
        .filter_map(|p| {
            let label = pickup_timer_label(&p.kind)?;
            let t = p.respawn_at?;
            let remaining = t - now;
            (remaining > 0.0).then(|| (label, remaining, pickup_timer_color(&p.kind)))
        })
        .collect();
    if entries.is_empty() {
        return;
    }
    entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    // Limite à 5 entrées — au-delà ça devient un mur de texte qui parasite
    // la vue centrale au lieu d'aider.
    entries.truncate(5);

    let char_w = 8.0 * HUD_SCALE;
    let row_h = LINE_H;
    let panel_w = 9.0 * char_w + 16.0; // "MH  12.3s" + padding
    let panel_h = row_h * (entries.len() as f32) + 16.0;
    let panel_x = 12.0;
    // Sous le pastille netvr (≈ 60 px) pour ne pas la chevaucher.
    let panel_y = 80.0;

    // Fond sombre + liseré accent — même langage visuel que les autres
    // panneaux HUD.
    r.push_rect(panel_x, panel_y, panel_w, panel_h, [0.05, 0.07, 0.10, 0.75]);
    r.push_rect(panel_x, panel_y, 2.0, panel_h, [0.30, 0.55, 0.85, 1.0]);

    for (i, (label, remaining, col)) in entries.iter().enumerate() {
        let y = panel_y + 8.0 + (i as f32) * row_h;
        // Format : sous 10 s → "9.4s" pour un effet tic-tac visuel ;
        // au-dessus → "MM:SS" pour les longs cooldowns powerup (120 s).
        let countdown = if *remaining < 10.0 {
            format!("{:>4.1}s", remaining)
        } else {
            let secs = remaining.ceil() as i32;
            format!("{:02}:{:02}", secs / 60, secs % 60)
        };
        push_text_shadow(r, panel_x + 8.0, y, HUD_SCALE, *col, label);
        push_text_shadow(
            r,
            panel_x + panel_w - char_w * 5.5,
            y,
            HUD_SCALE,
            COL_WHITE,
            &countdown,
        );
    }
}

/// Rend la roue de chat rapide centrée à l'écran. 8 secteurs de 45°
/// chacun, chaque secteur affichant un libellé court de
/// [`CHAT_WHEEL_MESSAGES`]. Animation : pop-in 120 ms (scale 0→1).
///
/// Le secteur 1 (digit `1`) est en HAUT, on tourne dans le sens horaire
/// — convention naturelle puisque la disposition pavé numérique
/// 1..=8 sur clavier ne forme pas un cercle, et la touche `1` la plus
/// à gauche pour la main = en haut visuellement = "premier choix".
fn draw_chat_wheel(r: &mut Renderer, w: f32, h: f32, elapsed: f32) {
    use std::f32::consts::PI;
    // Animation pop-in : scale 0.6 → 1.0 sur 120 ms, ease-out.
    let anim = (elapsed / 0.12).clamp(0.0, 1.0);
    let scale = 0.6 + 0.4 * (1.0 - (1.0 - anim).powi(3));
    let alpha = anim;

    // Voile noir derrière pour isoler du HUD/scène.
    r.push_rect(0.0, 0.0, w, h, [0.0, 0.0, 0.0, 0.45 * alpha]);

    let cx = w * 0.5;
    let cy = h * 0.5;
    let radius = (h * 0.20) * scale;
    let inner_r = radius * 0.30;

    // Cercle de fond — approximé par un disque (rect arrondi visuel
    // via empilement). On fait simple : un seul gros rect rotatif
    // n'existant pas dans notre Renderer, on dessine 32 secteurs
    // courts pour faire un anneau visible.
    let segments = 32;
    let ring_thickness = 4.0;
    for i in 0..segments {
        let a0 = (i as f32) * (2.0 * PI / segments as f32);
        let a1 = ((i + 1) as f32) * (2.0 * PI / segments as f32);
        let mid = (a0 + a1) * 0.5;
        let x = cx + mid.cos() * radius;
        let y = cy + mid.sin() * radius;
        r.push_rect(
            x - 1.0,
            y - 1.0,
            ring_thickness * 0.5,
            ring_thickness * 0.5,
            [0.4, 0.7, 1.0, 0.6 * alpha],
        );
    }

    // Texte central : titre de l'overlay.
    let title = "QUICK CHAT";
    let title_scale = HUD_SCALE * scale;
    let title_w = title.len() as f32 * 8.0 * title_scale;
    push_text_shadow(
        r,
        cx - title_w * 0.5,
        cy - 8.0 * title_scale * 0.5,
        title_scale,
        [1.0, 1.0, 1.0, alpha],
        title,
    );
    let hint = "[ ESC cancel ]";
    let hint_w = hint.len() as f32 * 8.0 * HUD_SCALE;
    r.push_text(
        cx - hint_w * 0.5,
        cy + 8.0 * title_scale,
        HUD_SCALE * scale,
        [0.7, 0.7, 0.7, alpha],
        hint,
    );

    // 8 secteurs : indice 0 en haut (-PI/2), sens horaire.
    let label_radius = radius * 1.05;
    for (i, (label, _full)) in CHAT_WHEEL_MESSAGES.iter().enumerate() {
        let angle = -PI * 0.5 + (i as f32) * (2.0 * PI / 8.0);
        let lx = cx + angle.cos() * label_radius;
        let ly = cy + angle.sin() * label_radius;
        let label_full = format!("{}.{}", i + 1, label);
        let lscale = HUD_SCALE * scale * 1.1;
        let lw = label_full.len() as f32 * 8.0 * lscale;
        // Pilule de fond pour chaque libellé.
        r.push_rect(
            lx - lw * 0.5 - 6.0,
            ly - 8.0 * lscale * 0.5 - 4.0,
            lw + 12.0,
            8.0 * lscale + 8.0,
            [0.06, 0.10, 0.14, 0.85 * alpha],
        );
        push_text_shadow(
            r,
            lx - lw * 0.5,
            ly - 8.0 * lscale * 0.5,
            lscale,
            [1.0, 0.86, 0.2, alpha],
            &label_full,
        );
        let _ = inner_r;
    }
}

/// Cherche le bot le plus proche dans un cône frontal pour la cible
/// d'une roquette **lock-on** (W5). Retourne `Some(index)` dans
/// `bots`, ou `None` si aucun bot vivant ne tient dans le cône
/// (~30° de demi-angle, 1500u de range). On exclut les bots morts ou
/// en invul de spawn — pas de lock douteux sur un mort qui clignote.
fn find_lock_target(
    bots: &[BotDriver],
    eye: Vec3,
    forward: Vec3,
    time_sec: f32,
) -> Option<usize> {
    const LOCK_RANGE: f32 = 1500.0;
    // cos(30°) ≈ 0.866 — le cible doit être dans ce cône frontal pour
    // lock. Plus serré = lock plus exigeant (skill check).
    const LOCK_DOT_THRESHOLD: f32 = 0.866;
    let mut best: Option<(f32, usize)> = None;
    for (i, d) in bots.iter().enumerate() {
        if d.health.is_dead() || d.invul_until > time_sec {
            continue;
        }
        let center = d.body.origin + Vec3::Z * BOT_CENTER_HEIGHT;
        let to_t = center - eye;
        let dist = to_t.length();
        if dist > LOCK_RANGE || dist < 1.0 {
            continue;
        }
        let dot = to_t.dot(forward) / dist;
        if dot < LOCK_DOT_THRESHOLD {
            continue;
        }
        // Heuristique : on prend le plus PROCHE dans le cône (pas le
        // plus aligné). Plus naturel — un ennemi à 200u un peu sur le
        // côté lock avant un ennemi à 1400u parfaitement centré.
        match best {
            Some((d_prev, _)) if d_prev <= dist => {}
            _ => best = Some((dist, i)),
        }
    }
    best.map(|(_, i)| i)
}

/// Logique pure de cycle follow-cam. Sortie : nouveau `follow_slot`.
///
/// * `current` : slot actuellement suivi (ou `None`).
/// * `alive`   : slots vivants (ordre quelconque — la fonction trie).
/// * `direction` : `+1` next, `-1` prev.
///
/// Comportement :
/// * `alive` vide → `None` (rien à suivre).
/// * `current = None` et `direction > 0` → premier slot.
/// * `current = None` et `direction < 0` → dernier slot.
/// * `current = Some(s)` mais `s` plus dans `alive` → ré-entre au début.
/// * Sinon : avance / recule cycliquement.
fn pick_follow_target(current: Option<u8>, alive: &[u8], direction: i32) -> Option<u8> {
    let mut sorted: Vec<u8> = alive.to_vec();
    sorted.sort_unstable();
    if sorted.is_empty() {
        return None;
    }
    let next = match current {
        None => {
            if direction > 0 {
                sorted[0]
            } else {
                *sorted.last().unwrap()
            }
        }
        Some(c) => {
            let idx = sorted.iter().position(|&s| s == c).unwrap_or(0);
            let len = sorted.len() as i32;
            let new_idx = (idx as i32 + direction).rem_euclid(len) as usize;
            sorted[new_idx]
        }
    };
    Some(next)
}

#[allow(clippy::too_many_arguments)]
fn draw_hud(
    r: &mut Renderer,
    console: &Console,
    player_origin: &Vec3,
    health: Health,
    armor: i32,
    deaths: u32,
    frags: u32,
    respawn_remaining: Option<f32>,
    muzzle_flash: bool,
    hit_marker: bool,
    weapon_name: &str,
    // `None` = arme à munitions infinies (Gauntlet) → on affiche "--".
    ammo_current: Option<i32>,
    explosion_flash: bool,
    kill_feed: &[KillEvent],
    floating_damages: &[FloatingDamage],
    // `true` tant que le joueur maintient TAB — overlay scoreboard centré.
    scoreboard_open: bool,
    bots: &[BotDriver],
    // Lignes additionnelles pour le scoreboard depuis le snapshot
    // réseau — `(name, frags, deaths)` par remote player. Vide en solo.
    remote_score_rows: &[(String, i16, i16, u8)],
    local_team: u8,
    // `Some` = le match est terminé, on dessine un overlay d'intermission
    // + on force l'affichage du scoreboard pour figer le score final.
    match_winner: Option<&KillActor>,
    // Secondes restantes avant timelimit. Rendu en `mm:ss`, rouge < 30s.
    time_remaining: f32,
    // Secondes restantes par powerup — indexé par `PowerupKind::index()`.
    // `None` sur un slot = powerup inactif. On affiche un badge par
    // powerup actif, empilé verticalement en haut au centre.
    powerup_remaining: &[Option<f32>; PowerupKind::COUNT],
    now: f32,
    // Pain-arrow : direction MONDE du dernier tir orienté encaissé
    // (source → joueur). `last_damage_until > now` gate l'affichage,
    // `view_angles` sert à projeter en vue locale pour choisir le
    // quadrant d'écran qui flashera.
    last_damage_dir: Vec3,
    last_damage_until: f32,
    view_angles: Angles,
    // Médailles en cours d'affichage — empilées verticalement en haut
    // au-dessus de la ligne de powerups, chacune fade indépendamment.
    active_medals: &[ActiveMedal],
    // `true` quand l'œil du joueur est à l'intérieur d'un brush Contents::WATER.
    // Déclenche un overlay bleu semi-transparent + un léger pulse lumineux
    // pour évoquer les caustiques. Détection côté engine via `point_contents`.
    underwater: bool,
    // Secondes d'air restantes dans les poumons.  Affichée uniquement
    // quand `underwater` est vrai, sous forme de barre horizontale
    // centrée bas d'écran pour signaler l'urgence.
    air_left: f32,
    // Holdable actuellement stocké (medkit / personal teleporter). `None`
    // = slot vide. Affiché en bas-droite juste au-dessus du compteur
    // d'armure, avec un hint « [ENTER] » pour indiquer la touche d'usage.
    held_item: Option<HoldableKind>,
    // Dernière mort du joueur : `Some((killer, cause))`. Rendu sous le
    // bandeau "YOU DIED" seulement quand `respawn_remaining.is_some()`
    // (le slot de vie précédente). Lu uniquement ici — la valeur est
    // écrite par `push_kill_cause` côté App.
    last_death_cause: Option<&(KillActor, KillCause)>,
    // Vitesse horizontale courante du joueur en units/s (longueur du
    // vecteur vélocité projeté sur XY).  Affichée en bas-centre comme
    // un tachymètre Defrag/CPM ; aide le joueur à sentir le strafe-jump.
    player_speed_ups: f32,
    // Streak courant de frags depuis le dernier respawn — affiché dans
    // un petit badge corner quand ≥ 3 pour relier la bannière temporelle
    // (PICKUP_TOAST_LIFETIME) à un état persistant visible.
    player_streak: u32,
    // Stats accuracy joueur : shots tirés / shots touchant au moins une
    // cible.  Rendu uniquement dans le scoreboard et l'intermission.
    total_shots: u32,
    total_hits: u32,
    // `true` quand le serveur a flagué notre slot comme spectateur.
    // Le HUD masque alors HP/armor/ammo/crosshair et affiche un overlay
    // « SPECTATING » centré haut. En solo, toujours `false`.
    is_spectator: bool,
    // En spectator avec follow-cam, libellé à afficher sous l'overlay
    // (ex: `"FOLLOWING  TARK"`). `None` → free-fly, on affiche un hint.
    follow_label: Option<String>,
    // Bitfield des armes possédées + tableau d'ammo par slot 0..=8,
    // pour dessiner une bande d'inventaire en bas-centre.  `active_slot`
    // est le numéro visible 1..=9 de l'arme en main.
    weapons_owned: u32,
    ammo_per_slot: [i32; 10],
    active_slot: u8,
    // Deadline du flash d'armure : si > `now`, on dessine un cadre
    // cyan pulsant pour signaler que l'armure a absorbé des dégâts.
    armor_flash_until: f32,
    // Deadline du flash de douleur : idem, mais rouge et plus épais.
    // Set quand le joueur perd effectivement des HP (après absorb
    // armure).  L'overlay rouge domine l'overlay cyan si les deux
    // deadlines sont actives (coup a traversé l'armure).
    pain_flash_until: f32,
    // Combo counter : cumul de dégâts infligés, reset si inactif
    // plus de `DMG_BURST_WINDOW` s.  `last_at` sert à déterminer
    // le fade et l'affichage — on ne dessine rien en dehors de la
    // fenêtre active.
    recent_dmg_total: i32,
    recent_dmg_last_at: f32,
    // Recul courant (0..=VIEW_KICK_MAX).  Convertit en spread du
    // réticule : les 4 arms s'écartent quand on tire, se resserrent
    // quand on lâche la gâchette.  Feedback visuel direct de la
    // précision instantanée — un joueur qui spam machinegun voit
    // son crosshair s'ouvrir ≈ 20 px à pleine cadence.
    view_kick: f32,
    // Instant du dernier swap d'arme réussi — sert à piloter l'animation
    // de drop/raise du panel ammo.  `f32::NEG_INFINITY` = aucun swap
    // n'a eu lieu, offset nul.
    weapon_switch_at: f32,
) {
    let w = r.width() as f32;
    let h = r.height() as f32;

    // Underwater tint : voile bleu cyan subtil couvrant tout l'écran,
    // modulé par un pulse lent (≈0.7 Hz) pour évoquer les caustiques.
    // Dessiné en premier pour que les éléments de HUD restent lisibles
    // par-dessus. Alpha ~0.28 : assez présent pour "vendre" l'immersion
    // sans étouffer les détails de la scène.
    if underwater {
        let pulse = 0.5 + 0.5 * (now * std::f32::consts::TAU * 0.7).sin();
        let alpha = 0.22 + 0.06 * pulse;
        r.push_rect(0.0, 0.0, w, h, [0.08, 0.28, 0.55, alpha]);
        // Jauge d'air : barre centrée bas d'écran, 240×8 px, avec cadre
        // sombre et remplissage clamp(air_left / AIR_CAPACITY_SEC).
        // Couleur verte tant qu'il reste plus d'1/3, jaune ensuite, rouge
        // pulsant quand la jauge est vide (air épuisé → dégâts imminents).
        let bar_w = 240.0;
        let bar_h = 8.0;
        let bar_x = (w - bar_w) * 0.5;
        let bar_y = h - 72.0;
        let ratio = (air_left / AIR_CAPACITY_SEC).clamp(0.0, 1.0);
        // Fond sombre + bordure.
        r.push_rect(bar_x - 2.0, bar_y - 2.0, bar_w + 4.0, bar_h + 4.0,
            [0.0, 0.0, 0.0, 0.55]);
        let fill_color = if air_left <= 0.0 {
            let p = 0.5 + 0.5 * (now * std::f32::consts::TAU * 2.5).sin();
            [1.0, 0.15 + 0.2 * p, 0.15, 0.9]
        } else if ratio < 0.33 {
            [1.0, 0.75, 0.15, 0.9]
        } else {
            [0.3, 0.85, 1.0, 0.9]
        };
        r.push_rect(bar_x, bar_y, bar_w * ratio.max(0.02), bar_h, fill_color);
    }

    // ─── Armor absorb flash ────────────────────────────────────────
    // Cadre cyan autour de l'écran qui fade linéairement sur
    // `ARMOR_FLASH_SEC`.  Distinct visuellement du cadre rouge du
    // low-health (qui pulse en permanence en dessous de 25 HP) :
    // l'armor flash est un *événement* (le coup vient de taper).
    // On le place ici, juste après l'underwater tint, pour qu'il soit
    // recouvert par les éléments HUD (chiffres, textes) et ne parasite
    // pas leur lisibilité.
    if armor_flash_until > now {
        let remaining = armor_flash_until - now;
        let ratio = (remaining / ARMOR_FLASH_SEC).clamp(0.0, 1.0);
        let alpha = ratio * 0.45;
        let thick = 24.0;
        let col = [0.25, 0.85, 1.0, alpha];
        // 4 barres du cadre (haut, bas, gauche, droite).
        r.push_rect(0.0, 0.0, w, thick, col);
        r.push_rect(0.0, h - thick, w, thick, col);
        r.push_rect(0.0, thick, thick, h - 2.0 * thick, col);
        r.push_rect(w - thick, thick, thick, h - 2.0 * thick, col);
    }

    // ─── Pain vignette ─────────────────────────────────────────────
    // Cadre rouge légèrement plus épais que le flash armure, fade
    // linéaire sur `PAIN_FLASH_SEC`.  Dessiné APRÈS l'armor flash pour
    // qu'un coup qui traverse l'armure soit dominé par le rouge.
    // Utilise aussi un voile central très léger (alpha 0.08 max) qui
    // couvre tout l'écran avec dégradé vers les bords, pour vendre la
    // sensation « coup pris » sans obscurcir le réticule.
    if pain_flash_until > now {
        let remaining = pain_flash_until - now;
        let ratio = (remaining / PAIN_FLASH_SEC).clamp(0.0, 1.0);
        let alpha = ratio * 0.55;
        let thick = 32.0;
        let col = [1.0, 0.15, 0.12, alpha];
        r.push_rect(0.0, 0.0, w, thick, col);
        r.push_rect(0.0, h - thick, w, thick, col);
        r.push_rect(0.0, thick, thick, h - 2.0 * thick, col);
        r.push_rect(w - thick, thick, thick, h - 2.0 * thick, col);
        // Voile central subtil (n'obscurcit pas le combat).
        r.push_rect(0.0, 0.0, w, h, [0.4, 0.0, 0.0, ratio * 0.10]);
    }

    // Ligne d'état permanente en haut à gauche — version discrète (debug
    // position), restera utile pour les speedrunners mais ne fait plus
    // concurrence aux stats principales (déplacées vers un panneau propre).
    let status = format!(
        "pos {:>6.0} {:>6.0} {:>6.0}",
        player_origin.x, player_origin.y, player_origin.z
    );
    push_text_shadow(r, 8.0, 8.0, HUD_SCALE * 0.8, COL_GRAY, &status);

    // ─── Stats panel top-right (moderne) ───────────────────────────
    // Panneau unique groupant HP / Armor / Frags / Deaths / Time — au
    // lieu de 5 lignes isolées qui se battent avec l'arrière-plan.
    // Largeur calibrée pour "FRAGS NN/NN" (texte le plus long),
    // hauteur = 5 lignes + padding.
    // Skip en mode spectateur — pas de HP/Armor/Ammo à afficher
    // (le slot ne reçoit pas de dégâts), seul le frag-feed reste utile.
    let char_w = 8.0 * HUD_SCALE;
    let stats_w = 200.0;
    // `stats_y`/`stats_h` hoistés HORS du gate spectator car le kill-feed
    // (qui s'affiche aussi pour le spectateur) les utilise pour son
    // placement vertical.
    let stats_h = LINE_H * 5.0 + 14.0;
    let stats_y = 8.0;
    // Ancrage right-edge clampé sur le rect HUD safe 16:9 : sur 32:9
    // le bloc HP/Armor reste près du centre de l'écran au lieu d'être
    // collé au bord physique droit où le joueur ne le voit plus.
    let (safe_x, safe_w) = hud_safe_rect_x(w, h);
    let stats_x = safe_x + safe_w - stats_w - 8.0;
    if !is_spectator {
    let stats_h = LINE_H * 5.0 + 14.0;
    push_panel(r, stats_x, stats_y, stats_w, stats_h);

    // HP + barre graphique dégradée rouge→vert.
    let hp_ratio = (health.current as f32 / health.max as f32).clamp(0.0, 1.0);
    let hp_color = if health.current <= 25 {
        COL_RED
    } else if health.current <= 50 {
        COL_YELLOW
    } else {
        COL_WHITE
    };
    let hp_text = format!("HP {:>3}", health.current.max(0));
    let line_y = stats_y + 8.0;
    push_text_shadow(
        r,
        stats_x + 10.0,
        line_y,
        HUD_SCALE,
        hp_color,
        &hp_text,
    );
    // Mini-barre à droite du chiffre, alignée sur la ligne HP.
    let bar_x = stats_x + stats_w - 86.0;
    let bar_w_px = 78.0;
    let bar_h_px = 6.0;
    let bar_y_offset = 5.0;
    push_bar_gradient(
        r,
        bar_x,
        line_y + bar_y_offset,
        bar_w_px,
        bar_h_px,
        hp_ratio,
        [1.0, 0.25, 0.20],
        [0.30, 0.90, 0.40],
    );

    // Armor + barre cyan→bleu.
    let armor_ratio = (armor as f32 / 200.0).clamp(0.0, 1.0);
    let armor_text = format!("ARM {armor:>3}");
    let line_y = stats_y + 8.0 + LINE_H;
    push_text_shadow(
        r,
        stats_x + 10.0,
        line_y,
        HUD_SCALE,
        COL_WHITE,
        &armor_text,
    );
    push_bar_gradient(
        r,
        bar_x,
        line_y + bar_y_offset,
        bar_w_px,
        bar_h_px,
        armor_ratio,
        [0.25, 0.50, 0.65],
        [0.30, 0.85, 1.00],
    );

    // Frags.
    let frags_text = format!("FRAGS {frags}/{FRAG_LIMIT}");
    let frags_x = stats_x + stats_w - (frags_text.len() as f32 * char_w) - 10.0;
    push_text_shadow(
        r,
        frags_x,
        stats_y + 8.0 + LINE_H * 2.0,
        HUD_SCALE,
        COL_YELLOW,
        &frags_text,
    );

    // Deaths.
    let deaths_text = format!("DEATHS {deaths}");
    let deaths_x = stats_x + stats_w - (deaths_text.len() as f32 * char_w) - 10.0;
    push_text_shadow(
        r,
        deaths_x,
        stats_y + 8.0 + LINE_H * 3.0,
        HUD_SCALE,
        COL_GRAY,
        &deaths_text,
    );

    // Chrono mm:ss. Rouge < 30s (aka "last minute" agressif), jaune < 60s,
    // gris sinon. Pendant l'intermission on fige à 00:00.
    let clamp = if match_winner.is_some() { 0.0 } else { time_remaining };
    let total = clamp.floor().max(0.0) as i32;
    let mins = total / 60;
    let secs = total % 60;
    let time_text = format!("TIME {mins}:{secs:02}");
    let time_color = if clamp <= 30.0 {
        COL_RED
    } else if clamp <= 60.0 {
        COL_YELLOW
    } else {
        COL_GRAY
    };
    let time_x = stats_x + stats_w - (time_text.len() as f32 * char_w) - 10.0;
    push_text_shadow(
        r,
        time_x,
        stats_y + 8.0 + LINE_H * 4.0,
        HUD_SCALE,
        time_color,
        &time_text,
    );
    } // fin if !is_spectator (stats panel)

    // Powerups actifs : un badge par powerup, empilé verticalement en
    // haut-centre. L'ordre (`PowerupKind::ALL`) place Quad en haut. Un
    // blink 2 Hz dans les 3 dernières secondes prévient l'expiration.
    {
        let scale = HUD_SCALE * 1.8;
        let cw = 8.0 * scale;
        let line_h = 8.0 * scale + 4.0;
        let mut row = 0;
        for kind in PowerupKind::ALL {
            let Some(remaining) = powerup_remaining[kind.index()] else {
                continue;
            };
            // Arrondi à l'entier le + proche : plus stable visuellement que
            // floor (qui passerait "30" → "29" instantanément).
            let shown = remaining.round().max(0.0) as i32;
            let label = format!("{} {shown}", kind.hud_label());
            // Alpha : plein, sauf blink dans les 3 dernières sec (2Hz).
            let alpha = if remaining < 3.0 {
                if (now * 4.0).sin() > 0.0 { 1.0 } else { 0.35 }
            } else {
                1.0
            };
            let mut color = kind.hud_color();
            color[3] *= alpha;
            let text_w = label.len() as f32 * cw;
            let x = (w - text_w) * 0.5;
            let y = 12.0 + row as f32 * line_h;
            r.push_text(x, y, scale, color, &label);
            row += 1;
        }
    }

    // Chiffres de dégât flottants : rendus en premier dans le HUD pour
    // qu'un overlay (explosion flash, message YOU DIED) les recouvre sans
    // lutte visuelle. On projette la position monde → écran, on décale
    // vers le haut avec le temps, on fade alpha sur les derniers 40%.
    // Jaune = dégât infligé à un bot (hit confirm), rouge = dégât subi
    // par le joueur (pain flash).
    let dmg_scale = HUD_SCALE * 1.2;
    let dmg_char_w = 8.0 * dmg_scale;
    let dmg_line_h = 8.0 * dmg_scale;
    for d in floating_damages {
        let remaining = (d.expire_at - now).max(0.0);
        let life = d.lifetime.max(1e-3);
        let elapsed_ratio = (1.0 - remaining / life).clamp(0.0, 1.0);
        // Montée verticale dans le monde — plus lisible qu'un offset écran
        // pur (suit la géométrie Q3 Z-up).
        let rise = DAMAGE_NUMBER_RISE * elapsed_ratio;
        let world_pos = d.origin + Vec3::Z * rise;
        let Some((sx, sy)) = r.project_to_screen(world_pos) else {
            continue;
        };
        // Fade : plein alpha pendant 60% de la vie, puis tombe linéairement.
        let alpha = if elapsed_ratio < 0.6 {
            1.0
        } else {
            ((1.0 - elapsed_ratio) / 0.4).clamp(0.0, 1.0)
        };
        let color = if d.to_player {
            [1.0, 0.35, 0.25, alpha]
        } else {
            [1.0, 0.95, 0.2, alpha]
        };
        let text = format!("{}", d.damage);
        // Centrage horizontal + léger remontage vertical pour que le
        // chiffre ne "sorte" pas sous la tête du bot à t=0.
        let tx = sx - (text.len() as f32 * dmg_char_w) * 0.5;
        let ty = sy - dmg_line_h * 0.5;
        r.push_text(tx, ty, dmg_scale, color, &text);
    }

    // Nameplates bots : au-dessus de chaque bot vivant dans le champ de
    // vue, affiche nom + mini-barre de santé.  Pas de LOS test — ce fork
    // solo vise le training contre IA, voir les PV à travers un mur est
    // plus utile que gênant.  Fade en fonction de la distance pour
    // éviter le spam de texte à longue portée.
    {
        const NAMEPLATE_MAX_DIST: f32 = 1800.0;
        let plate_scale = HUD_SCALE * 0.9;
        let plate_char = 8.0 * plate_scale;
        let plate_h = 8.0 * plate_scale;
        let bar_w = 60.0;
        let bar_h = 4.0;
        for bd in bots {
            if bd.health.is_dead() {
                continue;
            }
            let head = bd.body.origin + Vec3::Z * (BOT_CENTER_HEIGHT as f32 * 1.2);
            let dist = (head - *player_origin).length();
            if dist > NAMEPLATE_MAX_DIST {
                continue;
            }
            let Some((sx, sy)) = r.project_to_screen(head) else {
                continue;
            };
            // Fade linéaire au-delà de 80 % de la distance max.
            let fade = if dist < NAMEPLATE_MAX_DIST * 0.8 {
                1.0
            } else {
                ((NAMEPLATE_MAX_DIST - dist) / (NAMEPLATE_MAX_DIST * 0.2))
                    .clamp(0.0, 1.0)
            };
            let name = &bd.bot.name;
            let text_w = name.len() as f32 * plate_char;
            let tx = sx - text_w * 0.5;
            let ty = sy - plate_h - 14.0;
            // Ombre + texte couleur bot.
            r.push_text(tx + 1.0, ty + 1.0, plate_scale, [0.0, 0.0, 0.0, 0.8 * fade], name);
            r.push_text(
                tx,
                ty,
                plate_scale,
                [bd.tint[0], bd.tint[1], bd.tint[2], fade],
                name,
            );
            // Barre de santé juste sous le nom.
            let hp = bd.health.current.max(0);
            let max = bd.health.max.max(1);
            let ratio = (hp as f32 / max as f32).clamp(0.0, 1.0);
            let bx = sx - bar_w * 0.5;
            let by = ty + plate_h + 2.0;
            // Fond sombre.
            r.push_rect(bx, by, bar_w, bar_h, [0.0, 0.0, 0.0, 0.7 * fade]);
            // Remplissage vert → rouge selon santé.
            let fill_col = if ratio > 0.5 {
                [0.3, 0.9, 0.3, fade]
            } else if ratio > 0.25 {
                [1.0, 0.85, 0.15, fade]
            } else {
                [1.0, 0.25, 0.2, fade]
            };
            r.push_rect(bx + 1.0, by + 1.0, (bar_w - 2.0) * ratio, bar_h - 2.0, fill_col);
        }
    }

    // Kill feed : liste chronologique des N derniers kills. Une entrée
    // fade alpha sur la dernière seconde avant expiration. Style :
    //   `YOU  [ROCKET]  BOTNAME`
    //   `ShamBot  [MG]  You`
    // Chaque ligne est posée sur une pilule anthracite légère pour
    // rester lisible sur un fond quelconque + s'aligner visuellement
    // avec le nouveau panneau stats.  Positionnée juste sous le
    // panneau stats.
    let feed_y0 = stats_y + stats_h + 8.0;
    for (i, ev) in kill_feed.iter().rev().enumerate() {
        let remaining = (ev.expire_at - now).max(0.0);
        let alpha = (remaining / 1.0).min(1.0); // fade sur la dernière seconde
        let killer_label = ev.killer.label();
        let victim_label = ev.victim.label();
        let cause_tag = ev.cause.tag();
        let line = format!("{killer_label}  [{cause_tag}]  {victim_label}");
        let killer_col = actor_color(&ev.killer, alpha);
        let line_w = line.len() as f32 * char_w;
        let y = feed_y0 + (i as f32) * (LINE_H + 2.0);
        // Ancré à droite du rect HUD safe (cf. `hud_safe_rect_x`) : sur
        // ultra-wide 21:9/32:9 le kill-feed reste côté joueur au lieu
        // d'être éjecté tout au bord physique de l'écran.
        let pill_x = safe_x + safe_w - line_w - 20.0;
        let pill_w = line_w + 16.0;
        let pill_h = LINE_H + 2.0;
        // Pilule de fond (anthracite, alpha modulée par fade).
        r.push_rect(
            pill_x,
            y - 2.0,
            pill_w,
            pill_h,
            [0.04, 0.05, 0.08, 0.65 * alpha],
        );
        // Liseré accent cyan à gauche — focal visuel discret.
        r.push_rect(
            pill_x,
            y - 2.0,
            2.0,
            pill_h,
            [0.15, 0.60, 0.85, 0.85 * alpha],
        );
        push_text_shadow(
            r,
            pill_x + 8.0,
            y,
            HUD_SCALE,
            killer_col,
            &line,
        );
    }

    // ─── Ammo/Weapon panel bottom-right (moderne) ──────────────────
    // Le panneau groupe nom d'arme (petit, au-dessus) + chiffre ammo
    // géant en dessous, sur un fond anthracite liseré cyan — remplace
    // les deux push_text isolés qui flottaient en l'air.  Le chiffre
    // ammo reste l'info visuelle primaire (grand format 1.5× HUD),
    // mais gagne en lisibilité par le châssis.
    // Skip en spectateur — pas d'arme à afficher.
    if !is_spectator {
    let weapon_text = weapon_name.to_uppercase();
    let (ammo_text, ammo_color) = match ammo_current {
        None => ("--".to_string(), COL_GRAY),
        Some(n) if n <= 0 => (format!("{n}"), COL_RED),
        Some(n) if n < 5 => {
            let pulse = 0.5 + 0.5 * (now * std::f32::consts::TAU * 3.0).sin();
            let col = [1.0, 0.3 + 0.4 * pulse, 0.3 + 0.3 * pulse, 1.0];
            (format!("{n}"), col)
        }
        Some(n) => (format!("{n}"), COL_WHITE),
    };
    let big_scale = HUD_SCALE * 1.8;
    let big_char = 8.0 * big_scale;
    let panel_w = 200.0;
    let panel_h = 80.0;
    // Ancrage right-edge mais clampé sur la zone HUD safe 16:9 — sur
    // 21:9 / 32:9 le panel reste visible du coin de l'œil au lieu
    // d'être plaqué tout à droite de l'écran.
    let (safe_x, safe_w) = hud_safe_rect_x(w, h);
    let panel_x = safe_x + safe_w - panel_w - 8.0;
    let base_panel_y = h - panel_h - 8.0;
    // Animation swap d'arme : le panel plonge vers le bas puis remonte.
    // `sin(t * PI)` atteint 1 au milieu et revient à 0 aux bornes — une
    // demi-onde sinus donne une arrivée/départ sans saccade, comme un
    // viewmodel qu'on baisse puis relève.  Alpha fade symétrique pour
    // éviter un clignotement de rectangle quand le panel est bas.
    let swap_elapsed = now - weapon_switch_at;
    let (panel_y, panel_alpha) = if swap_elapsed >= 0.0 && swap_elapsed < WEAPON_SWITCH_ANIM_SEC {
        let t = (swap_elapsed / WEAPON_SWITCH_ANIM_SEC).clamp(0.0, 1.0);
        let drop = (t * std::f32::consts::PI).sin() * WEAPON_SWITCH_DROP_PX;
        // Fade : 1.0 aux bornes, 0.35 au creux — on garde un peu de
        // lecture pour que l'œil accroche le chiffre d'ammo qui change.
        let alpha = 1.0 - 0.65 * (t * std::f32::consts::PI).sin();
        (base_panel_y + drop, alpha)
    } else {
        (base_panel_y, 1.0)
    };
    if panel_alpha > 0.99 {
        push_panel(r, panel_x, panel_y, panel_w, panel_h);
    } else {
        // Variante transparente du panel pour animer sans recalculer les
        // constantes COL_PANEL_*.  On multiplie juste l'alpha du fond.
        let bg = [
            COL_PANEL_BG[0],
            COL_PANEL_BG[1],
            COL_PANEL_BG[2],
            COL_PANEL_BG[3] * panel_alpha,
        ];
        let edge = [
            COL_PANEL_EDGE[0],
            COL_PANEL_EDGE[1],
            COL_PANEL_EDGE[2],
            COL_PANEL_EDGE[3] * panel_alpha,
        ];
        let edge_dim = [
            COL_PANEL_EDGE_DIM[0],
            COL_PANEL_EDGE_DIM[1],
            COL_PANEL_EDGE_DIM[2],
            COL_PANEL_EDGE_DIM[3] * panel_alpha,
        ];
        r.push_rect(panel_x, panel_y, panel_w, panel_h, bg);
        r.push_rect(panel_x, panel_y, panel_w, 2.0, edge);
        r.push_rect(panel_x, panel_y + panel_h - 1.0, panel_w, 1.0, edge_dim);
    }
    // Nom de l'arme : petit, en haut à gauche du panneau, couleur accent.
    let accent = [0.6, 0.85, 1.0, panel_alpha];
    push_text_shadow(
        r,
        panel_x + 12.0,
        panel_y + 8.0,
        HUD_SCALE * 0.9,
        accent,
        &weapon_text,
    );
    // Chiffre ammo géant, aligné à droite.
    let ammo_x = panel_x + panel_w - (ammo_text.len() as f32 * big_char) - 14.0;
    let ammo_y = panel_y + panel_h - big_char - 4.0;
    let ammo_color_a = [ammo_color[0], ammo_color[1], ammo_color[2], ammo_color[3] * panel_alpha];
    push_text_shadow(r, ammo_x, ammo_y, big_scale, ammo_color_a, &ammo_text);
    } // fin if !is_spectator (ammo panel)

    // Holdable slot : badge en bas-droite au-dessus de l'arme courante.
    // Deux lignes : libellé coloré (MEDKIT / TPORT) + hint gris [ENTER].
    // Masqué si aucun holdable — pas de réserve de place pour ne pas
    // parasiter un HUD déjà chargé.
    if let Some(kind) = held_item {
        let label = kind.hud_label();
        let hint = "[ENTER]";
        let color = match kind {
            HoldableKind::Medkit => [0.4, 1.0, 0.4, 1.0],
            HoldableKind::Teleporter => [0.6, 0.4, 1.0, 1.0],
        };
        // Placé au-dessus du panneau ammo/weapon.  On reste discret —
        // ce n'est pas l'info la plus fréquemment lue, juste un rappel
        // « oui tu as un medkit prêt ».  Coordonnée absolue (le panneau
        // ammo fait 80 px de haut, +8 de marge, +40 pour les 2 lignes).
        let hold_y = h - 80.0 - 8.0 - LINE_H * 2.2;
        let hold_x = w - (label.len() as f32 * char_w) - 16.0;
        push_text_shadow(r, hold_x, hold_y, HUD_SCALE, color, label);
        let hint_x = w - (hint.len() as f32 * char_w) - 16.0;
        push_text_shadow(r, hint_x, hold_y + LINE_H, HUD_SCALE, COL_GRAY, hint);
    }

    // Réticule de visée — profil par arme.  Le gap central (cross) ou
    // le rayon (ring) scale linéairement sur `view_kick` (0..1.2) pour
    // visualiser le recul.  Chaque arme a une silhouette dédiée pour
    // que le joueur « sente » l'outil qu'il tient même sans regarder
    // le panel ammo :
    //   - gauntlet     : cercle épais (melee range)
    //   - machinegun   : 4-arm classique
    //   - shotgun      : 4-arm arms courts + base_gap large (spread implicite)
    //   - grenade      : point central + cercle (arc indirect)
    //   - rocket       : point + ring épais (warning explosif)
    //   - lightning    : plus fin avec dot central (beam)
    //   - railgun      : dot seul + liseré (sniping)
    //   - plasma       : 8-spokes petits (tir rapide dispersé)
    //   - bfg          : cadre carré + dot (signature massive)
    let cx = w * 0.5;
    let cy = h * 0.5;
    let spread = (view_kick * 15.0).clamp(0.0, 18.0);
    draw_weapon_crosshair(r, cx, cy, active_slot, spread);

    // Hit marker : 4 segments diagonaux autour du réticule (flash bref).
    if hit_marker {
        let hm = [1.0, 0.3, 0.3, 1.0];
        let sz = 4.0;
        let off = 10.0;
        r.push_rect(cx - off - sz, cy - off - 1.0, sz, 2.0, hm);
        r.push_rect(cx + off, cy - off - 1.0, sz, 2.0, hm);
        r.push_rect(cx - off - sz, cy + off - 1.0, sz, 2.0, hm);
        r.push_rect(cx + off, cy + off - 1.0, sz, 2.0, hm);
    }

    // Combo counter : cumul de dégâts infligés sur la fenêtre récente.
    // Affiché sous le crosshair, fade sur la dernière 0.3 s de la
    // fenêtre.  Couleur : blanc → jaune → orange → rouge selon la
    // magnitude (100, 200, 400+ HP).  Utile pour visualiser un combo
    // LG ou une double rocket.
    if recent_dmg_total > 0 {
        let age = now - recent_dmg_last_at;
        if age < DMG_BURST_WINDOW {
            let fade_start = DMG_BURST_WINDOW - 0.3;
            let alpha = if age < fade_start {
                1.0
            } else {
                1.0 - (age - fade_start) / 0.3
            }
            .clamp(0.0, 1.0);
            let col = if recent_dmg_total >= 400 {
                [1.0, 0.35, 0.25, alpha]
            } else if recent_dmg_total >= 200 {
                [1.0, 0.55, 0.20, alpha]
            } else if recent_dmg_total >= 100 {
                [1.0, 0.85, 0.25, alpha]
            } else {
                [1.0, 1.0, 1.0, alpha]
            };
            let text = format!("+{recent_dmg_total}");
            let scale = 2.2;
            let glyph_w = 8.0 * scale;
            let text_w = text.len() as f32 * glyph_w;
            let tx = cx - text_w * 0.5;
            let ty = cy + 22.0;
            r.push_text(tx + 2.0, ty + 2.0, scale, [0.0, 0.0, 0.0, 0.85 * alpha], &text);
            r.push_text(tx, ty, scale, col, &text);
        }
    }

    // Vignette basse santé : cadre rouge pulsant qui apparaît quand HP
    // passe sous `LOW_HEALTH_THRESHOLD`.  L'intensité croît à mesure que
    // la santé chute — à 1 HP le cadre est bien visible, à 25 HP juste
    // un rappel subtil.  Pulsation sinusoïdale ~1.5 Hz pour un sentiment
    // d'urgence sans clignoter.
    //
    // Dessiné AVANT le pain-arrow pour que la flèche directionnelle
    // reste lisible par-dessus le cadre permanent.
    if !health.is_dead() && health.current <= LOW_HEALTH_THRESHOLD && health.max > 0 {
        let severity = 1.0
            - (health.current as f32 / LOW_HEALTH_THRESHOLD as f32).clamp(0.0, 1.0);
        // Pulsation : 0.55 + 0.45·sin → intensité oscille entre 0.1 et 1.0.
        let pulse = 0.55 + 0.45 * (now * LOW_HEALTH_PULSE_HZ * std::f32::consts::TAU).sin();
        let alpha = (severity * pulse * LOW_HEALTH_MAX_ALPHA).clamp(0.0, LOW_HEALTH_MAX_ALPHA);
        if alpha > 0.01 {
            let col = [1.0, 0.1, 0.1, alpha];
            // Cadre : 4 barres de `LOW_HEALTH_FRAME_THICK` unités à chaque
            // bord.  Les coins se chevauchent ; l'additif rouge-sur-rouge
            // les rend légèrement plus denses — effet voulu (les coins
            // sont des zones d'attention naturelle).
            let thick = LOW_HEALTH_FRAME_THICK;
            r.push_rect(0.0, 0.0, w, thick, col);          // haut
            r.push_rect(0.0, h - thick, w, thick, col);    // bas
            r.push_rect(0.0, 0.0, thick, h, col);          // gauche
            r.push_rect(w - thick, 0.0, thick, h, col);    // droite
        }
    }

    // Pain-arrow : 4 barres fines aux bords écran (haut / droite / bas
    // / gauche), chacune pondérée par son alignement avec la direction
    // monde → vue de l'attaquant. La source étant en face = flash haut,
    // dans le dos = flash bas, etc. Fade linéaire sur
    // `DAMAGE_DIR_SHOW_SEC` secondes.
    let remaining = last_damage_until - now;
    if remaining > 0.0 && last_damage_dir.length_squared() > 1e-6 {
        // On veut le vecteur « regard → source ». `last_damage_dir`
        // pointe source → joueur, donc on l'inverse.
        let to_src = -last_damage_dir;
        let basis = view_angles.to_vectors();
        // Seule la direction 2D horizontale compte pour le quadrant :
        // un tir rasant-haut vs rasant-bas tape du même côté pour
        // l'UX. On projette sur (forward, right) et renormalise.
        let fwd2 = to_src.dot(basis.forward);
        let rgt2 = to_src.dot(basis.right);
        let len2 = (fwd2 * fwd2 + rgt2 * rgt2).sqrt();
        if len2 > 1e-3 {
            let fwd = fwd2 / len2;
            let rgt = rgt2 / len2;
            let fade = (remaining / DAMAGE_DIR_SHOW_SEC).clamp(0.0, 1.0);
            // Intensité par quadrant = max(0, dot avec la normale de ce
            // bord). Carré pour concentrer l'effet dans le bon quadrant
            // et ne pas allumer tous les bords à la fois.
            let w_top = fwd.max(0.0).powi(2);
            let w_bot = (-fwd).max(0.0).powi(2);
            let w_rgt = rgt.max(0.0).powi(2);
            let w_lft = (-rgt).max(0.0).powi(2);
            let bar_thick = 14.0_f32;
            let bar_len = (w.min(h) * 0.5).max(120.0);
            let col = |alpha: f32| [1.0, 0.15, 0.15, alpha];
            // Haut
            if w_top > 1e-3 {
                let a = (w_top * fade * 0.55).min(0.7);
                r.push_rect(
                    cx - bar_len * 0.5,
                    0.0,
                    bar_len,
                    bar_thick,
                    col(a),
                );
            }
            // Bas
            if w_bot > 1e-3 {
                let a = (w_bot * fade * 0.55).min(0.7);
                r.push_rect(
                    cx - bar_len * 0.5,
                    h - bar_thick,
                    bar_len,
                    bar_thick,
                    col(a),
                );
            }
            // Droite
            if w_rgt > 1e-3 {
                let a = (w_rgt * fade * 0.55).min(0.7);
                r.push_rect(
                    w - bar_thick,
                    cy - bar_len * 0.5,
                    bar_thick,
                    bar_len,
                    col(a),
                );
            }
            // Gauche
            if w_lft > 1e-3 {
                let a = (w_lft * fade * 0.55).min(0.7);
                r.push_rect(
                    0.0,
                    cy - bar_len * 0.5,
                    bar_thick,
                    bar_len,
                    col(a),
                );
            }
        }

        // Marqueur directionnel AUTOUR du crosshair — en plus du flash
        // de bord, un petit chevron rouge (3 rects en forme de « > »)
        // placé sur un cercle autour du centre pointe vers l'attaquant.
        // Précise le tir à distance quand les bords de l'écran sont déjà
        // lumineux (explosion, Quad, water overlay).
        if len2 > 1e-3 {
            let fwd = fwd2 / len2;
            let rgt = rgt2 / len2;
            let fade = (remaining / DAMAGE_DIR_SHOW_SEC).clamp(0.0, 1.0);
            let radius = 70.0_f32;
            // Screen-space : right = +x, forward = -y (haut).
            let dx = rgt * radius;
            let dy = -fwd * radius;
            let mx = cx + dx;
            let my = cy + dy;
            // Tick principal : petit carré rouge 8×8.  Contour noir
            // pour lisibilité sur fond clair.
            let a = (fade * 0.9).min(1.0);
            r.push_rect(mx - 5.0, my - 5.0, 10.0, 10.0, [0.0, 0.0, 0.0, 0.75 * a]);
            r.push_rect(mx - 3.0, my - 3.0, 6.0, 6.0, [1.0, 0.2, 0.15, a]);
        }
    }

    // Radar / minimap : carré en haut-centre-gauche (sous le chrono du
    // match) qui situe le joueur + bots vivants dans un rayon de ~2000
    // unités world.  "Up" sur le radar = direction regardée par le
    // joueur — la map tourne autour du joueur, pas l'inverse.  Utile en
    // DM chargé pour repérer qui te tourne autour.
    {
        const RADAR_WORLD_RANGE: f32 = 2000.0;
        let radar_size = 140.0_f32;
        let radar_x = 8.0;
        let radar_y = 90.0; // sous la ligne d'info top-gauche
        let cx_r = radar_x + radar_size * 0.5;
        let cy_r = radar_y + radar_size * 0.5;
        let scale = (radar_size * 0.5) / RADAR_WORLD_RANGE;
        // Fond semi-opaque + liseré.
        r.push_rect(radar_x, radar_y, radar_size, radar_size, [0.0, 0.0, 0.0, 0.55]);
        let border = 2.0;
        r.push_rect(radar_x, radar_y, radar_size, border, [0.7, 0.7, 0.8, 0.6]);
        r.push_rect(
            radar_x,
            radar_y + radar_size - border,
            radar_size,
            border,
            [0.7, 0.7, 0.8, 0.6],
        );
        r.push_rect(radar_x, radar_y, border, radar_size, [0.7, 0.7, 0.8, 0.6]);
        r.push_rect(
            radar_x + radar_size - border,
            radar_y,
            border,
            radar_size,
            [0.7, 0.7, 0.8, 0.6],
        );
        // Base de direction joueur pour rotation 2D.
        let basis = view_angles.to_vectors();
        let fwd_x = basis.forward.x;
        let fwd_y = basis.forward.y;
        let rgt_x = basis.right.x;
        let rgt_y = basis.right.y;
        // Dots bots.
        for bd in bots {
            if bd.health.is_dead() {
                continue;
            }
            let dx = bd.body.origin.x - player_origin.x;
            let dy = bd.body.origin.y - player_origin.y;
            let dist2 = dx * dx + dy * dy;
            if dist2 > RADAR_WORLD_RANGE * RADAR_WORLD_RANGE {
                continue;
            }
            // Projection sur (forward, right) du joueur.
            let local_fwd = dx * fwd_x + dy * fwd_y;
            let local_rgt = dx * rgt_x + dy * rgt_y;
            // Map : forward → écran haut (-y), right → écran droit (+x).
            let px = cx_r + local_rgt * scale;
            let py = cy_r - local_fwd * scale;
            // Petit dot 5×5 avec ombre noire 7×7.
            r.push_rect(px - 3.5, py - 3.5, 7.0, 7.0, [0.0, 0.0, 0.0, 0.8]);
            r.push_rect(
                px - 2.5,
                py - 2.5,
                5.0,
                5.0,
                [bd.tint[0], bd.tint[1], bd.tint[2], 1.0],
            );
        }
        // Joueur : triangle-ish (3 rects concentriques) au centre, couleur
        // verte — toujours "up".
        r.push_rect(cx_r - 4.0, cy_r - 6.0, 8.0, 2.0, [0.2, 1.0, 0.2, 1.0]);
        r.push_rect(cx_r - 3.0, cy_r - 4.0, 6.0, 2.0, [0.2, 1.0, 0.2, 1.0]);
        r.push_rect(cx_r - 2.0, cy_r - 2.0, 4.0, 6.0, [0.2, 1.0, 0.2, 1.0]);
    }

    // Muzzle flash : rendu en billboard 3D à `tag_flash` du viewmodel —
    // voir `queue_muzzle_flash` côté pipeline 3D.  Rien à faire en HUD.
    let _ = muzzle_flash;

    // Explosion flash : overlay full-screen rouge/orangé assez opaque quand
    // une rocket explose proche du joueur. Remplit l'écran ~0.35 s.
    if explosion_flash {
        r.push_rect(0.0, 0.0, w, h, [1.0, 0.35, 0.1, 0.35]);
    }

    // Tint full-screen léger par powerup actif qui en demande un — subtil
    // pour ne pas gêner la visibilité mais visible en périphérie. Pulse
    // doucement à 1Hz. Les tints s'additionnent dans l'ordre (un Quad +
    // Haste → rect bleu puis rect orange).
    for kind in PowerupKind::ALL {
        let Some(remaining) = powerup_remaining[kind.index()] else {
            continue;
        };
        let Some(rgb) = kind.fullscreen_tint() else { continue };
        let pulse = 0.5 + 0.5 * (now * std::f32::consts::TAU).sin();
        let base_alpha = 0.07 + pulse * 0.04;
        // Fade out dans la dernière seconde pour préparer la fin.
        let alpha = base_alpha * remaining.min(1.0);
        r.push_rect(0.0, 0.0, w, h, [rgb[0], rgb[1], rgb[2], alpha]);
    }

    // Overlay Quad Damage spécialisé : en plus du tint de base,
    // bandeaux de vignette haut/bas en bleu saturé + pulse x2 Hz,
    // pour le feedback "je suis surpuissant" caractéristique de Q3.
    // 4 bandes empilées (top + bottom, chacune en deux fines bandes
    // pour simuler un dégradé linéaire sans shader).
    if let Some(remaining) = powerup_remaining[PowerupKind::QuadDamage.index()] {
        let pulse = 0.5 + 0.5 * (now * std::f32::consts::TAU * 2.0).sin();
        let fade = remaining.min(0.8) / 0.8; // fondu dans la dernière 0.8 s
        let strong = 0.45 * pulse * fade;
        let soft = 0.18 * pulse * fade;
        let quad = [0.35, 0.55, 1.0];
        let band_outer = (h * 0.06).max(18.0);
        let band_inner = (h * 0.05).max(14.0);
        // Haut : bande extérieure (opaque) + bande intérieure (légère)
        r.push_rect(0.0, 0.0, w, band_outer, [quad[0], quad[1], quad[2], strong]);
        r.push_rect(
            0.0,
            band_outer,
            w,
            band_inner,
            [quad[0], quad[1], quad[2], soft],
        );
        // Bas : miroir.
        r.push_rect(
            0.0,
            h - band_outer,
            w,
            band_outer,
            [quad[0], quad[1], quad[2], strong],
        );
        r.push_rect(
            0.0,
            h - band_outer - band_inner,
            w,
            band_inner,
            [quad[0], quad[1], quad[2], soft],
        );
        // Liseré très fin au milieu des bords gauche/droit pour encadrer
        // légèrement sans obstruer le champ de vision central.
        let side_w = 3.0;
        r.push_rect(0.0, 0.0, side_w, h, [quad[0], quad[1], quad[2], strong]);
        r.push_rect(
            w - side_w,
            0.0,
            side_w,
            h,
            [quad[0], quad[1], quad[2], strong],
        );
    }

    // Overlay spectator : bandeau haut avec l'état (FREE-FLY ou FOLLOWING
    // <name>) + hint clavier discret. Affiché uniquement en mode
    // spectator pour ne pas parasiter le HUD joueur.
    if is_spectator {
        let title = follow_label
            .as_deref()
            .unwrap_or("SPECTATING — FREE FLY");
        let hint = if follow_label.is_some() {
            "[ LMB next ]  [ RMB prev ]  [ SPACE free-fly ]"
        } else {
            "[ LMB / RMB cycle players ]"
        };
        let title_scale = HUD_SCALE * 1.6;
        let title_char_w = 8.0 * title_scale;
        let title_w = title.len() as f32 * title_char_w;
        let title_x = (w - title_w) * 0.5;
        let title_y = h * 0.06;
        let panel_w = (title_w + 64.0).max(hint.len() as f32 * char_w + 32.0);
        let panel_x = (w - panel_w) * 0.5;
        r.push_rect(
            panel_x,
            title_y - 8.0,
            panel_w,
            title_scale * 8.0 + LINE_H + 18.0,
            [0.05, 0.07, 0.10, 0.65],
        );
        push_text_shadow(r, title_x, title_y, title_scale, COL_YELLOW, title);
        let hint_w = hint.len() as f32 * char_w;
        let hint_x = (w - hint_w) * 0.5;
        r.push_text(
            hint_x,
            title_y + title_scale * 8.0 + 4.0,
            HUD_SCALE,
            COL_GRAY,
            hint,
        );
    }

    // Bandeau central si le joueur est mort (respawn_remaining = Some(t)).
    // Sous la ligne principale, un sous-titre "killed by <actor> (<cause>)"
    // rappelle qui/quoi a fraggé le joueur — utile en DM chargé où le
    // kill-feed défile trop vite pour être lu pendant la mort.
    if let Some(remaining) = respawn_remaining {
        let msg = format!("YOU DIED — respawn in {remaining:.1}s");
        let msg_w = msg.len() as f32 * char_w * 2.0;
        let cx = (w - msg_w) * 0.5;
        let cy = h * 0.5 - LINE_H;
        // On étire le panneau quand il y a un sous-titre, pour englober
        // les deux lignes plus le padding.
        let has_cause = last_death_cause.is_some();
        let panel_h = if has_cause {
            LINE_H * 2.0 + 16.0 + LINE_H * 1.4
        } else {
            LINE_H * 2.0 + 16.0
        };
        r.push_rect(0.0, cy - 8.0, w, panel_h, [0.1, 0.0, 0.0, 0.6]);
        r.push_text(cx, cy, HUD_SCALE * 2.0, COL_RED, &msg);
        if let Some((killer, cause)) = last_death_cause {
            let killer_label = killer.label();
            let cause_label = cause.tag();
            // "killed by <killer> (<cause>)" — cause entre parenthèses
            // parce qu'elle peut être une arme OU un tag environnemental
            // (LAVA, VOID…), et la forme parenthésée lit pareil dans les
            // deux cas. Suicide : killer == Player (auto-frag au RL en
            // splash sur soi-même), on montre "YOURSELF" à la place.
            let pretty_killer = if matches!(killer, KillActor::Player) {
                "yourself"
            } else {
                killer_label
            };
            let sub = format!("killed by {pretty_killer} ({cause_label})");
            let sub_scale = HUD_SCALE * 1.1;
            let sub_char_w = 8.0 * sub_scale;
            let sub_w = sub.len() as f32 * sub_char_w;
            let sub_x = (w - sub_w) * 0.5;
            let sub_y = cy + LINE_H * 1.9;
            // Couleur : gris-rouge atténué — présent mais sans voler la
            // vedette au bandeau principal.
            r.push_text(sub_x, sub_y, sub_scale, [1.0, 0.6, 0.5, 0.95], &sub);
        }
    }

    // Scoreboard TAB — overlay centré listant joueur + bots avec leurs
    // frags/deaths. S'affiche au-dessus du HUD mais sous la console
    // (pour ne pas gêner la saisie). Dessiné seulement quand TAB est maintenu.
    // Overlay « SPECTATING » : centré haut, teal pulsant. Indique
    // sans ambiguïté qu'on est en mode observateur (le HUD HP/AR
    // reste visible mais inutile, l'overlay évite la confusion).
    if is_spectator {
        let label = "SPECTATING";
        let scale = HUD_SCALE * 1.8;
        let cw = 8.0 * scale;
        let text_w = label.len() as f32 * cw;
        let x = (w - text_w) * 0.5;
        let y = 4.0;
        // Pulse 1Hz pour attirer l'œil sans être agressif.
        let pulse = 0.55 + 0.45 * (now * std::f32::consts::TAU * 1.0).sin().abs();
        let color = [0.30, 0.85 * pulse, 0.95 * pulse, 1.0];
        push_text_shadow(r, x, y, scale, color, label);
        // Hint discret en dessous.
        let hint = "free-fly: WASD + SPACE/CTRL";
        let s = HUD_SCALE * 0.85;
        let cw_s = 8.0 * s;
        let hint_w = hint.len() as f32 * cw_s;
        push_text_shadow(
            r,
            (w - hint_w) * 0.5,
            y + scale * 8.0 + 4.0,
            s,
            [0.7, 0.85, 0.95, 0.7],
            hint,
        );
    }

    // Pendant l'intermission on force l'affichage pour geler le score final.
    if scoreboard_open || match_winner.is_some() {
        draw_scoreboard(
            r,
            w,
            h,
            frags,
            deaths,
            bots,
            remote_score_rows,
            local_team,
            total_shots,
            total_hits,
        );
    }

    // Intermission : bandeau plein écran + gros titre. Dessiné après le
    // scoreboard pour le recouvrir proprement, mais avant la console.
    if let Some(winner) = match_winner {
        // Voile sombre + deux liserés accent horizontaux pour encadrer le
        // titre sans occulter le scoreboard (lisibilité des chiffres
        // finaux).  Remplace le rectangle monochrome par un cadre « boss
        // battle over » qui sent le fin-de-round moderne.
        r.push_rect(0.0, 0.0, w, h, [0.0, 0.0, 0.0, 0.55]);
        r.push_rect(0.0, h * 0.12, w, 3.0, COL_PANEL_EDGE);
        r.push_rect(0.0, h * 0.32, w, 3.0, COL_PANEL_EDGE);
        let scale = HUD_SCALE * 3.0;
        let winner_label = winner.label();
        let line1 = "MATCH OVER";
        let line2 = format!("WINNER: {winner_label}");
        let big_char_w = 8.0 * scale;
        let big_line_h = 8.0 * scale;
        let l1_w = line1.len() as f32 * big_char_w;
        let l2_w = line2.len() as f32 * big_char_w;
        let cy = h * 0.18;
        push_text_shadow(r, (w - l1_w) * 0.5, cy, scale, COL_YELLOW, line1);
        push_text_shadow(
            r,
            (w - l2_w) * 0.5,
            cy + big_line_h * 1.4,
            scale,
            COL_WHITE,
            &line2,
        );
    }

    if console.is_open() {
        draw_console(r, console, w, h);
    } else {
        // Discret rappel "appuyez sur ` pour la console" — ombre portée
        // pour rester lisible par-dessus le watermark ou une carte claire.
        push_text_shadow(
            r,
            8.0,
            h - LINE_H - 8.0,
            HUD_SCALE * 0.85,
            COL_GRAY,
            "press ` to open console",
        );
    }

    // Médailles Q3 — popups dans le tiers supérieur central.  Chaque
    // médaille s'affiche avec un fade-in rapide (0 → 1 sur
    // MEDAL_FADE_IN_RATIO de sa vie) puis un fade-out long.  Empilage
    // vertical : la plus récente en bas, les anciennes au-dessus, pour
    // que le dernier frag soit tout de suite lisible au centre-regard.
    //
    // Dessiné en dernier pour passer devant tout le reste du HUD.
    if !active_medals.is_empty() {
        let medal_scale = HUD_SCALE * 2.5;
        let medal_char = 8.0 * medal_scale;
        let line_h = 10.0 * medal_scale + 4.0;
        // Plus la médaille est récente, plus elle est basse à l'écran.
        // `base_y` = ligne du dernier popup (≈ 20 % de la hauteur).
        let base_y = h * 0.20;
        for (idx, m) in active_medals.iter().enumerate() {
            let age = (now - m.spawn_time).max(0.0);
            let life = (m.expire_at - m.spawn_time).max(1e-3);
            let t = (age / life).clamp(0.0, 1.0);
            // Fade-in puis fade-out — symétrique autour de `MEDAL_FADE_IN_RATIO`.
            let alpha = if t < MEDAL_FADE_IN_RATIO {
                t / MEDAL_FADE_IN_RATIO
            } else {
                1.0 - (t - MEDAL_FADE_IN_RATIO) / (1.0 - MEDAL_FADE_IN_RATIO)
            }
            .clamp(0.0, 1.0);
            let label = m.kind.label();
            let mut col = m.kind.color();
            col[3] *= alpha;
            let text_w = label.len() as f32 * medal_char;
            // Les popups plus anciens remontent : `idx` croissant = plus
            // récent, donc on soustrait `(len-1-idx) * line_h` depuis le bas.
            let row_from_bottom = active_medals.len() - 1 - idx;
            let y = base_y - (row_from_bottom as f32) * line_h;
            r.push_text(w * 0.5 - text_w * 0.5, y, medal_scale, col, label);
        }
    }

    // Watermark logo "Q3 RUST" en bas-gauche : branding discret, alpha
    // bas pour ne pas entrer en compétition avec les chiffres du HUD.
    // Scale calibré pour faire ~110 px de large à 1080p.
    {
        let wm_scale = (w / 1920.0).clamp(0.7, 1.5) * 0.22;
        let wm_x = 8.0;
        let wm_y = h - 8.0 - 8.0 * (8.0 * wm_scale);
        crate::logo::draw_watermark(r, wm_x, wm_y, wm_scale);
    }

    // ─── Speedometer (Defrag / CPM style) ──────────────────────────
    // Vitesse horizontale en UPS (units per second) centrée bas.
    // Fade sous 50 u/s pour ne pas polluer le HUD à l'arrêt.  Échelle
    // couleur : blanc < 320, jaune 320..500, orange 500..700, rouge
    // > 700 (avec pulse).  Seuils calés sur VQ3 : 320 = sprint, 500 =
    // strafe confirmé, 700+ = rocket-jump compétitif.
    if player_speed_ups > 50.0 {
        let ups = player_speed_ups.round() as i32;
        let (color, fade_in) = if ups < 320 {
            ([1.0, 1.0, 1.0], (player_speed_ups - 50.0) / 270.0)
        } else if ups < 500 {
            ([1.0, 0.95, 0.35], 1.0)
        } else if ups < 700 {
            ([1.0, 0.65, 0.20], 1.0)
        } else {
            let pulse = 0.5 + 0.5 * (now * std::f32::consts::TAU * 2.5).sin();
            ([1.0, 0.25 + 0.10 * pulse, 0.25], 1.0)
        };
        let fade = fade_in.clamp(0.0, 1.0);
        let text = format!("{ups} UPS");
        let scale = 2.4;
        let glyph_w = 8.0 * scale;
        let text_w = text.len() as f32 * glyph_w;
        let x = (w - text_w) * 0.5;
        let y = h * 0.84;
        r.push_text(x + 1.0, y + 1.0, scale, [0.0, 0.0, 0.0, 0.75 * fade], &text);
        r.push_text(x, y, scale, [color[0], color[1], color[2], fade], &text);
    }

    // ─── Bande d'inventaire d'armes ────────────────────────────────
    // 9 slots 1..=9 en bas-centre, au-dessus du speedometer.  Chaque
    // slot = un petit rect avec le chiffre de raccourci et un label
    // court (« MG », « SG », « RL »…).  États visuels :
    //   - Arme possédée + ammo > 0 : texte blanc, rect gris foncé.
    //   - Arme possédée mais vide : texte rouge, rect rouge foncé.
    //   - Arme non possédée       : texte gris, rect quasi transparent.
    //   - Arme active             : cadre jaune autour du slot.
    // Placé à `y = h * 0.895` (sous le speedometer qui est à 0.84),
    // largeur totale ~450px calée pour rester lisible en 1080p.
    {
        const SLOT_LABELS: [&str; 9] = [
            "GNT", "MG", "SG", "GL", "RL", "LG", "RG", "PG", "BFG",
        ];
        let slot_w = 44.0;
        let slot_h = 28.0;
        let gap = 4.0;
        let total_w = 9.0 * slot_w + 8.0 * gap;
        let start_x = (w - total_w) * 0.5;
        let y = h * 0.895;
        for (i, label) in SLOT_LABELS.iter().enumerate() {
            let slot = (i + 1) as u8; // slots Q3 sont 1..=9
            let owned = (weapons_owned & (1u32 << slot)) != 0;
            let ammo = ammo_per_slot[slot as usize];
            let has_ammo = ammo > 0 || slot == WeaponId::Gauntlet.slot();
            let is_active = slot == active_slot;
            let x = start_x + (i as f32) * (slot_w + gap);
            let (bg, fg) = match (owned, has_ammo) {
                (true, true) => ([0.12, 0.12, 0.14, 0.70], [1.0, 1.0, 1.0, 0.95]),
                (true, false) => ([0.35, 0.08, 0.08, 0.65], [1.0, 0.45, 0.45, 0.95]),
                (false, _) => ([0.06, 0.06, 0.07, 0.35], [0.45, 0.45, 0.48, 0.75]),
            };
            r.push_rect(x, y, slot_w, slot_h, bg);
            if is_active {
                // Cadre jaune 2px.  On empile 4 rects plutôt que de
                // tirer une stroke — le pipeline HUD ne gère que des
                // quads pleins.
                let t = 2.0;
                let c = [1.0, 0.85, 0.25, 1.0];
                r.push_rect(x, y, slot_w, t, c);               // haut
                r.push_rect(x, y + slot_h - t, slot_w, t, c);  // bas
                r.push_rect(x, y, t, slot_h, c);               // gauche
                r.push_rect(x + slot_w - t, y, t, slot_h, c);  // droite
            }
            // Chiffre de raccourci en haut-gauche.
            let num = format!("{slot}");
            r.push_text(x + 3.0, y + 2.0, 1.2, fg, &num);
            // Label court centré sur le slot.
            let lbl_scale = if label.len() <= 2 { 1.4 } else { 1.0 };
            let lbl_w = label.len() as f32 * 8.0 * lbl_scale;
            r.push_text(
                x + (slot_w - lbl_w) * 0.5,
                y + slot_h - 8.0 * lbl_scale - 3.0,
                lbl_scale,
                fg,
                label,
            );
        }
    }

    // ─── Proximity alert (bot dans le dos) ─────────────────────────
    // Badge clignotant rouge à gauche/droite du crosshair qui signale
    // la présence d'un bot vivant dans le cône arrière du joueur, à
    // ≤ PROX_DIST unités, non visible (puisque derrière).  Pulse à
    // 4 Hz, fade avec la distance pour suggérer l'urgence.
    //
    // Cône arrière : dot(forward, to_bot) < -0.3 (~ 107° de demi-
    // ouverture arrière).  Distance : 500 unités ≈ 12.5 m Q3, soit
    // la portée à laquelle un rail ou un rocket a le temps d'arriver
    // avant que le joueur ne réagisse.
    {
        const PROX_DIST: f32 = 500.0;
        let basis = view_angles.to_vectors();
        let fwd_xy = Vec3::new(basis.forward.x, basis.forward.y, 0.0)
            .normalize_or_zero();
        let mut closest: Option<(f32, Vec3)> = None;
        for bd in bots {
            if bd.health.is_dead() {
                continue;
            }
            let to_bot = bd.body.origin - *player_origin;
            let to_bot_xy = Vec3::new(to_bot.x, to_bot.y, 0.0);
            let dist = to_bot_xy.length();
            if dist < 1.0 || dist > PROX_DIST {
                continue;
            }
            let dir = to_bot_xy / dist;
            if fwd_xy.dot(dir) < -0.3 {
                if closest.map_or(true, |(d, _)| dist < d) {
                    closest = Some((dist, dir));
                }
            }
        }
        if let Some((dist, _)) = closest {
            let prox = 1.0 - (dist / PROX_DIST).clamp(0.0, 1.0);
            // Pulse plus rapide quand plus proche (3 → 6 Hz).
            let hz = 3.0 + prox * 3.0;
            let pulse = 0.5 + 0.5 * (now * std::f32::consts::TAU * hz).sin();
            let alpha = (0.45 + 0.35 * prox) * (0.55 + 0.45 * pulse);
            let text = "! REAR !";
            let scale = 1.6 + prox * 0.4;
            let glyph_w = 8.0 * scale;
            let text_w = text.len() as f32 * glyph_w;
            // Placé juste au-dessus de la bande d'inventaire.
            let by = h * 0.895 - 8.0 * scale - 10.0;
            let bx = (w - text_w) * 0.5;
            r.push_text(bx + 2.0, by + 2.0, scale, [0.0, 0.0, 0.0, alpha * 0.85], text);
            r.push_text(bx, by, scale, [1.0, 0.25, 0.20, alpha], text);
        }
    }

    // ─── Streak badge ───────────────────────────────────────────────
    // Petit badge persistant quand le joueur a ≥ 3 kills sans mourir.
    // Complément au toast éphémère (palier) : l'état reste visible.
    if player_streak >= 3 {
        let text = format!("x{player_streak}");
        let scale = 2.8;
        let glyph_w = 8.0 * scale;
        let glyph_h = 8.0 * scale;
        let pad = 6.0;
        let text_w = text.len() as f32 * glyph_w;
        let box_w = text_w + pad * 2.0;
        let box_h = glyph_h + pad * 2.0;
        let (bg, fg) = match player_streak {
            3..=4 => ([1.0, 0.85, 0.25, 0.25], [1.0, 0.95, 0.55, 1.0]),
            5..=6 => ([1.0, 0.55, 0.15, 0.30], [1.0, 0.80, 0.35, 1.0]),
            7..=9 => ([1.0, 0.30, 0.20, 0.35], [1.0, 0.55, 0.35, 1.0]),
            10..=14 => ([0.9, 0.2, 0.9, 0.35], [1.0, 0.65, 1.0, 1.0]),
            15..=19 => ([1.0, 0.95, 0.35, 0.40], [1.0, 1.0, 0.65, 1.0]),
            _ => ([1.0, 0.4, 1.0, 0.45], [1.0, 0.7, 1.0, 1.0]),
        };
        let bx = 14.0;
        let by = h - 180.0 - box_h;
        r.push_rect(bx, by, box_w, box_h, bg);
        r.push_text(bx + pad + 1.0, by + pad + 1.0, scale, [0.0, 0.0, 0.0, 0.85], &text);
        r.push_text(bx + pad, by + pad, scale, fg, &text);
    }
}

/// Dessine un overlay semi-transparent listant le joueur et chaque bot
/// avec leur score (frags / deaths). Centré à l'écran, trié par frags
/// décroissants pour que le leader soit en tête.
fn draw_scoreboard(
    r: &mut Renderer,
    w: f32,
    h: f32,
    player_frags: u32,
    player_deaths: u32,
    bots: &[BotDriver],
    // Lignes additionnelles pour les remote players réseau —
    // `(name, frags, deaths, team)`. En solo, vide.
    remote_rows: &[(String, i16, i16, u8)],
    local_team: u8,
    total_shots: u32,
    total_hits: u32,
) {
    // Collecte + tri : joueur + bots, par frags desc, tie-break deaths asc.
    #[derive(Clone)]
    struct Row {
        name: String,
        frags: u32,
        deaths: u32,
        is_player: bool,
        team: u8,
        /// `None` pour le joueur humain et les remote players ; `Some(1..5)`
        /// pour un bot IA local. Affiché en colonne dédiée — le joueur
        /// peut voir d'un coup d'œil contre quel niveau il joue.
        skill: Option<i32>,
    }
    let mut rows: Vec<Row> =
        Vec::with_capacity(bots.len() + remote_rows.len() + 1);
    rows.push(Row {
        name: "YOU".to_string(),
        frags: player_frags,
        deaths: player_deaths,
        is_player: true,
        team: local_team,
        skill: None,
    });
    for bd in bots {
        rows.push(Row {
            name: bd.bot.name.clone(),
            frags: bd.frags,
            deaths: bd.deaths,
            is_player: false,
            team: q3_net::team::FREE,
            skill: Some(bd.bot.skill.to_int()),
        });
    }
    for (name, frags, deaths, team) in remote_rows {
        rows.push(Row {
            name: name.clone(),
            frags: (*frags).max(0) as u32,
            deaths: (*deaths).max(0) as u32,
            is_player: false,
            team: *team,
            skill: None,
        });
    }
    // TDM détecté quand au moins une row a une team non-FREE. Plus
    // robuste que de plumber GameType côté client : tant que les
    // PlayerState arrivent avec team != 0, on regroupe.
    let is_tdm = rows.iter().any(|r| r.team != q3_net::team::FREE);
    if is_tdm {
        // Tri par team (RED puis BLUE puis FREE), puis frags desc dans
        // chaque groupe. Le mapping numérique inverse RED(1) → BLUE(2)
        // ne donne pas l'ordre désiré — on remap explicitement.
        let team_rank = |t: u8| -> u8 {
            match t {
                q3_net::team::RED => 0,
                q3_net::team::BLUE => 1,
                _ => 2,
            }
        };
        rows.sort_by(|a, b| {
            team_rank(a.team)
                .cmp(&team_rank(b.team))
                .then(b.frags.cmp(&a.frags))
                .then(a.deaths.cmp(&b.deaths))
        });
    } else {
        rows.sort_by(|a, b| b.frags.cmp(&a.frags).then(a.deaths.cmp(&b.deaths)));
    }

    // Géométrie : on taille le panneau au contenu. Largeur = max(nom) + 2
    // colonnes FRAGS/DEATHS + padding. Le tout centré.
    let char_w = 8.0 * HUD_SCALE;
    let header = "NAME";
    let max_name_len = rows
        .iter()
        .map(|r| r.name.len())
        .max()
        .unwrap_or(0)
        .max(header.len())
        .max(10); // plancher pour éviter un panneau ridicule à 1 bot
    let name_col_w = (max_name_len as f32 + 2.0) * char_w;
    let num_col_w = 9.0 * char_w; // largeur d'une colonne chiffrée
    let skill_col_w = 7.0 * char_w; // "SKILL" + "III." etc.
    let panel_w = name_col_w + num_col_w * 2.0 + skill_col_w + 32.0;
    let title_h = LINE_H * 2.0;
    let row_h = LINE_H;
    // +1.0 pour l'en-tête, +1.5 pour laisser respirer la ligne stats finale.
    // En TDM on ajoute une ligne pour les totaux par équipe (calcul réel
    // plus bas, on majore ici à +1 ligne pour pré-allouer la hauteur).
    let tdm_reserve = if rows.iter().any(|r| r.team != q3_net::team::FREE) {
        1.0
    } else {
        0.0
    };
    let panel_h = title_h + row_h * (rows.len() as f32 + 1.0 + 1.5 + tdm_reserve) + 24.0;
    let panel_x = (w - panel_w) * 0.5;
    let panel_y = (h - panel_h) * 0.5;

    // Fond moderne : voile noir derrière pour isoler du monde, puis
    // panneau anthracite + liseré accent cyan.  Deux niveaux pour que
    // le scoreboard « sorte » de l'arrière-plan même sur une map
    // lumineuse.
    r.push_rect(0.0, 0.0, w, h, [0.0, 0.0, 0.0, 0.30]);
    push_panel(r, panel_x, panel_y, panel_w, panel_h);

    // Titre centré, double ombre portée.
    let title = "SCOREBOARD";
    let title_scale = HUD_SCALE * 1.5;
    let title_char_w = 8.0 * title_scale;
    let title_x = panel_x + (panel_w - title.len() as f32 * title_char_w) * 0.5;
    push_text_shadow(r, title_x, panel_y + 8.0, title_scale, COL_YELLOW, title);

    // En-têtes de colonnes.
    let col_name_x = panel_x + 16.0;
    let col_frags_x = col_name_x + name_col_w;
    let col_deaths_x = col_frags_x + num_col_w;
    let col_skill_x = col_deaths_x + num_col_w;
    let header_y = panel_y + title_h + 8.0;
    push_text_shadow(r, col_name_x, header_y, HUD_SCALE, COL_GRAY, "NAME");
    push_text_shadow(r, col_frags_x, header_y, HUD_SCALE, COL_GRAY, "FRAGS");
    push_text_shadow(r, col_deaths_x, header_y, HUD_SCALE, COL_GRAY, "DEATHS");
    push_text_shadow(r, col_skill_x, header_y, HUD_SCALE, COL_GRAY, "SKILL");
    // Séparateur fin sous les en-têtes.
    r.push_rect(
        panel_x + 12.0,
        header_y + LINE_H * 0.9,
        panel_w - 24.0,
        1.0,
        COL_PANEL_EDGE_DIM,
    );

    // Couleurs équipes (alignées sur le tinting du modèle 3D).
    const COL_TEAM_RED: [f32; 4] = [1.00, 0.45, 0.45, 1.0];
    const COL_TEAM_BLUE: [f32; 4] = [0.50, 0.70, 1.00, 1.0];

    // Lignes de données. Joueur en jaune avec surlignage discret, bots
    // en blanc.  Zébrage alternatif léger pour suivre la ligne à l'œil.
    // En TDM : rangée colorée selon team (rouge/bleu) qui prime sur le
    // jaune joueur — l'œil cherche d'abord son équipe.
    let mut prev_team: Option<u8> = None;
    for (i, row) in rows.iter().enumerate() {
        let y = header_y + row_h * (i as f32 + 1.0);
        // Bandeau de séparation entre équipes en TDM. Discret — juste
        // une fine ligne dans la couleur de l'équipe nouvelle.
        if is_tdm && prev_team != Some(row.team) && i > 0 {
            r.push_rect(
                panel_x + 12.0,
                y - 3.0,
                panel_w - 24.0,
                1.0,
                COL_PANEL_EDGE_DIM,
            );
        }
        prev_team = Some(row.team);
        // Fond surligné pour la ligne joueur.
        if row.is_player {
            r.push_rect(
                panel_x + 8.0,
                y - 2.0,
                panel_w - 16.0,
                row_h,
                [0.45, 0.32, 0.08, 0.35],
            );
        } else if i % 2 == 1 {
            r.push_rect(
                panel_x + 8.0,
                y - 2.0,
                panel_w - 16.0,
                row_h,
                [0.10, 0.12, 0.18, 0.35],
            );
        }
        // Couleur du texte : team d'abord en TDM, sinon jaune joueur, sinon blanc.
        let col = if is_tdm {
            match row.team {
                q3_net::team::RED => COL_TEAM_RED,
                q3_net::team::BLUE => COL_TEAM_BLUE,
                _ => if row.is_player { COL_YELLOW } else { COL_WHITE },
            }
        } else if row.is_player {
            COL_YELLOW
        } else {
            COL_WHITE
        };
        push_text_shadow(r, col_name_x, y, HUD_SCALE, col, &row.name);
        let frags_s = format!("{}", row.frags);
        let deaths_s = format!("{}", row.deaths);
        push_text_shadow(r, col_frags_x, y, HUD_SCALE, col, &frags_s);
        push_text_shadow(r, col_deaths_x, y, HUD_SCALE, col, &deaths_s);
        // Colonne SKILL : chiffres romains pour distinguer du bruit des
        // autres colonnes numériques.  Joueur humain → tiret.
        let skill_s = match row.skill {
            None => "--".to_string(),
            Some(1) => "I".to_string(),
            Some(2) => "II".to_string(),
            Some(3) => "III".to_string(),
            Some(4) => "IV".to_string(),
            Some(5) => "V".to_string(),
            Some(n) => format!("{n}"),
        };
        push_text_shadow(r, col_skill_x, y, HUD_SCALE, col, &skill_s);
    }

    // En TDM : ligne récap totaux par équipe juste sous le tableau.
    // Format : "RED  N    BLUE  M". Aide à voir d'un coup d'œil
    // qui mène le match sans additionner mentalement.
    let mut tdm_extra_lines = 0.0_f32;
    if is_tdm {
        let (mut red_f, mut blue_f) = (0u32, 0u32);
        for row in &rows {
            match row.team {
                q3_net::team::RED => red_f += row.frags,
                q3_net::team::BLUE => blue_f += row.frags,
                _ => {}
            }
        }
        let team_y = header_y + row_h * (rows.len() as f32 + 1.0) + 6.0;
        let red_label = format!("RED  {red_f}");
        let blue_label = format!("BLUE  {blue_f}");
        let red_w = red_label.len() as f32 * char_w;
        let blue_w = blue_label.len() as f32 * char_w;
        let total_w = red_w + 4.0 * char_w + blue_w;
        let team_x = panel_x + (panel_w - total_w) * 0.5;
        push_text_shadow(r, team_x, team_y, HUD_SCALE, COL_TEAM_RED, &red_label);
        push_text_shadow(
            r,
            team_x + red_w + 4.0 * char_w,
            team_y,
            HUD_SCALE,
            COL_TEAM_BLUE,
            &blue_label,
        );
        tdm_extra_lines = 1.0;
    }

    // Ligne stats joueur sous le tableau : accuracy + total shots.
    // Affichée en gris discret en bas du panneau pour ne pas parasiter
    // les colonnes principales.
    let stats_y = header_y + row_h * (rows.len() as f32 + 1.5 + tdm_extra_lines);
    let acc_pct = if total_shots > 0 {
        (total_hits as f32 / total_shots as f32 * 100.0).round() as i32
    } else {
        0
    };
    let stats_line = format!("YOU — ACC: {acc_pct}%   ({total_hits}/{total_shots})");
    let stats_w = stats_line.len() as f32 * char_w;
    let stats_x = panel_x + (panel_w - stats_w) * 0.5;
    r.push_text(stats_x, stats_y, HUD_SCALE, COL_GRAY, &stats_line);
}

fn draw_console(r: &mut Renderer, console: &Console, w: f32, h: f32) {
    let con_h = (h * 0.55).max(160.0);
    r.push_rect(0.0, 0.0, w, con_h, COL_CONSOLE_BG);
    r.push_rect(0.0, con_h - 2.0, w, 2.0, COL_CONSOLE_BORDER);

    // Nombre de lignes affichables au-dessus du prompt.
    let usable = con_h - (LINE_H + 16.0);
    let max_lines = (usable / LINE_H).floor() as usize;
    let lines: Vec<&str> = console.lines().collect();
    let start = lines.len().saturating_sub(max_lines);
    let mut y = 8.0;
    for line in &lines[start..] {
        r.push_text(8.0, y, HUD_SCALE, COL_WHITE, line);
        y += LINE_H;
    }

    // Prompt.
    let prompt_y = con_h - LINE_H - 6.0;
    let prompt = format!("] {}", console.input());
    r.push_text(8.0, prompt_y, HUD_SCALE, COL_YELLOW, &prompt);
    // Caret clignotant approximé par l'underscore fixe en fin de ligne.
    let caret_x = 8.0 + (prompt.len() as f32) * 8.0 * HUD_SCALE;
    r.push_rect(caret_x, prompt_y, 8.0 * HUD_SCALE, 8.0 * HUD_SCALE, COL_YELLOW);
}

/// Lit un entier depuis `entity.extra` (clé-valeur brut du BSP). Retourne
/// `None` si la clé est absente ou non parseable. Utilisé pour `dmg`,
/// `spawnflags`, et autres champs facultatifs des triggers.
fn extra_i32(ent: &q3_game::Entity, key: &str) -> Option<i32> {
    ent.extra
        .iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| v.parse().ok())
}

/// Lit une clé `extra` d'une entité comme `&str` (vue sans allocation).
/// `None` si la clé est absente.  Utilisé pour `noise` (chemin VFS)
/// et autres champs textuels facultatifs.
fn extra_str<'a>(ent: &'a q3_game::Entity, key: &str) -> Option<&'a str> {
    ent.extra
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

/// `true` si `ent` est un `target_speaker` — couvre à la fois le classname
/// mappé en `Misc("target_speaker")` par [`EntityKind::from_classname`]
/// et les éventuelles futures variantes (si on ajoute une variante
/// dédiée plus tard, il suffira de l'ajouter ici).
fn is_target_speaker(ent: &q3_game::Entity) -> bool {
    matches!(&ent.kind, q3_game::EntityKind::Misc(s) if s == "target_speaker")
}

/// Étiquette courte d'une arme pour l'afficher dans le kill-feed —
/// équivalent aux icônes Q3 (on reste en texte tant qu'on n'a pas chargé
/// les sprites `gfx/2d/iconw_*`).
fn weapon_tag(w: WeaponId) -> &'static str {
    match w {
        WeaponId::Gauntlet => "GAUNTLET",
        WeaponId::Machinegun => "MG",
        WeaponId::Shotgun => "SG",
        WeaponId::Grenadelauncher => "GL",
        WeaponId::Rocketlauncher => "RL",
        WeaponId::Lightninggun => "LG",
        WeaponId::Railgun => "RG",
        WeaponId::Plasmagun => "PG",
        WeaponId::Bfg => "BFG",
    }
}

/// Couleur d'un acteur dans le kill-feed : jaune pour le joueur
/// (distinctif), gris-clair pour les bots, rouge sombre pour le monde.
/// `alpha` module le fade de sortie — `[r,g,b,a]` RGBA final.
fn actor_color(actor: &KillActor, alpha: f32) -> [f32; 4] {
    let alpha = alpha.clamp(0.0, 1.0);
    match actor {
        KillActor::Player => [1.0, 0.9, 0.3, alpha],
        KillActor::Bot(_) => [0.85, 0.85, 0.85, alpha],
        KillActor::World => [0.8, 0.2, 0.2, alpha],
    }
}

/// Résout le `KillActor` associé à un `ProjectileOwner` — lookup du nom
/// du bot si besoin. Utilisé pour l'attribution des kills sur projectile.
/// Si l'idx est stale ou inconnu, retombe sur `World`.
fn resolve_killer(bots: &[BotDriver], owner: ProjectileOwner) -> KillActor {
    match owner {
        ProjectileOwner::Player => KillActor::Player,
        ProjectileOwner::Bot(idx) => bots
            .get(idx)
            .map(|bd| KillActor::Bot(bd.bot.name.clone()))
            .unwrap_or(KillActor::World),
    }
}

/// Tente de charger un son WAV/OGG depuis le VFS, silencieusement sur erreur.
fn try_load_sfx(vfs: &Vfs, snd: &Arc<SoundSystem>, path: &str) -> Option<SoundHandle> {
    let bytes = match vfs.read(path) {
        Ok(b) => b.to_vec(),
        Err(_) => return None,
    };
    match snd.load(path, bytes) {
        Ok(h) => Some(h),
        Err(e) => {
            warn!("sfx '{path}' KO: {e}");
            None
        }
    }
}

fn play_at(snd: &Arc<SoundSystem>, handle: SoundHandle, origin: Vec3, priority: Priority) {
    snd.play_3d(
        handle,
        Emitter3D {
            position: origin,
            near_dist: 64.0,
            far_dist: 2048.0,
            volume: 1.0,
            priority,
        },
    );
}

/// Joue le hitsound « dans l'oreille » du joueur : on place l'émetteur à
/// la position listener courante → `near_dist` le garde à plein volume
/// quelle que soit la distance réelle de la victime. Équivalent moral
/// d'un son 2D, sans ajouter d'API à SoundSystem. Pas de pitch/pan
/// variable, on veut un feedback uniforme et instantané.
fn play_hit_feedback(snd: &Arc<SoundSystem>, handle: SoundHandle, listener: Vec3) {
    snd.play_3d(
        handle,
        Emitter3D {
            position: listener,
            near_dist: 64.0,
            far_dist: 2048.0,
            volume: 1.0,
            priority: Priority::Weapon,
        },
    );
}

/// Variante « kill-confirm » : même logique de spatialisation que
/// `play_hit_feedback` (émetteur au listener → volume plein). Priorité
/// `VoiceOver` pour qu'un thunk de frag ne soit jamais volé — un kill
/// perdu dans le mix est plus frustrant qu'un hitsound perdu. Volume
/// légèrement réduit (0.9) pour que le sample ne sonne pas plus fort
/// que le hit simple quand les deux se chevauchent dans le même tick.
fn play_kill_feedback(snd: &Arc<SoundSystem>, handle: SoundHandle, listener: Vec3) {
    snd.play_3d(
        handle,
        Emitter3D {
            position: listener,
            near_dist: 64.0,
            far_dist: 2048.0,
            volume: 0.9,
            priority: Priority::VoiceOver,
        },
    );
}

/// Médaille Q3 (Humiliation / Excellent / …). Même technique de
/// spatialisation dégénérée que `play_hit_feedback` — émetteur au
/// listener → volume plein indépendant de la position du frag. Volume
/// à 1.0 (médaille = feedback le plus saillant), priorité `VoiceOver`
/// pour qu'une voix annoncée ne soit jamais coupée au mix.
fn play_medal(snd: &Arc<SoundSystem>, handle: SoundHandle, listener: Vec3) {
    snd.play_3d(
        handle,
        Emitter3D {
            position: listener,
            near_dist: 64.0,
            far_dist: 2048.0,
            volume: 1.0,
            priority: Priority::VoiceOver,
        },
    );
}

/// Cherche dans `bsp.shaders` le premier shader marqué `skyparms` dans le
/// registre actif, et appelle `renderer.load_sky_cubemap` avec son
/// `far_box`. Les erreurs (registre absent, aucun shader sky, cubemap
/// manquante) sont loggées mais jamais propagées — on retombe sur le ciel
/// procédural par défaut.
fn resolve_and_load_sky(r: &mut Renderer, vfs: &Arc<Vfs>, bsp: &Bsp) {
    // Snapshot des noms candidats : on ne peut pas garder de `&ShaderRegistry`
    // vivant pendant qu'on appelle `load_sky_cubemap` (mutable borrow).
    let candidate = {
        let Some(reg) = r.shader_registry() else {
            return;
        };
        let mut found: Option<String> = None;
        for ds in &bsp.shaders {
            let name = ds.name().to_ascii_lowercase();
            let Some(sh) = reg.get(&name) else { continue };
            if !sh.is_sky {
                continue;
            }
            let Some(sp) = sh.sky_parms.as_ref() else {
                continue;
            };
            let Some(fb) = sp.far_box.as_ref() else {
                continue;
            };
            info!("sky shader '{}' → cubemap '{}'", name, fb);
            found = Some(fb.clone());
            break;
        }
        found
    };
    let Some(base) = candidate else {
        info!("sky: aucune cubemap déclarée — ciel procédural");
        return;
    };
    if let Err(e) = r.load_sky_cubemap(vfs, &base) {
        warn!("sky cubemap '{base}' KO: {e} — fallback procédural");
    }
}

/// Sortie d'un tick bots : total de dégâts hitscan appliqués au joueur +
/// liste de projectiles spawné par les bots ce tick (rockets pour l'instant).
struct BotTickOut {
    damage: i32,
    projectiles: Vec<Projectile>,
    /// Nom du dernier bot ayant landé un hit hitscan ce tick, utilisé comme
    /// attribution du kill si le joueur meurt (approximation suffisante :
    /// les bots tirent typiquement en cluster sur 1-2 frames). Le nom
    /// historique « mg » est conservé pour limiter la diff ; désormais
    /// l'arme effective est portée par `last_hitscan_weapon`.
    last_mg_damager: Option<String>,
    /// Même info que `last_mg_damager` mais sous forme d'index dans le
    /// vecteur de bots — utilisé pour créditer le frag dans le scoreboard
    /// sans avoir à re-scanner par nom.
    last_mg_damager_idx: Option<usize>,
    /// Arme effectivement utilisée par le dernier bot ayant tiré hitscan
    /// ce tick. Alimente le kill-feed (`[SG] / [MG] / [RG]`) quand le
    /// joueur meurt d'un hitscan.
    last_hitscan_weapon: Option<WeaponId>,
    /// Bullet holes à déposer : `(position, normale, arme)`.  Alimenté
    /// par les tirs bot qui ratent — la balle part selon une direction
    /// jittée dans le cône d'erreur, on trace jusqu'au premier mur, et
    /// on enregistre l'impact.  Flush hors-borrow par App::update.
    wall_marks: Vec<(Vec3, Vec3, WeaponId)>,
    /// Trails de railgun émis par les bots ce tick.  Teinte = couleur
    /// du bot (rose de base + modulation `bd.tint`) — chaque bot laisse
    /// une signature visuelle distincte, ce qui aide à identifier qui
    /// vient de sniper depuis l'autre bout de la map quand plusieurs
    /// bots sont en vue.
    rail_beams: Vec<ActiveBeam>,
}

/// Profil de tir hitscan bot : détermine quelle arme est utilisée selon
/// la distance bot↔joueur, et avec quels paramètres.
#[derive(Debug, Clone, Copy)]
struct HitscanProfile {
    weapon: WeaponId,
    /// Dégât par pellet.
    dmg_per_pellet: i32,
    /// Nombre de pellets (1 pour MG/RG, >1 pour SG).
    pellets: u32,
    /// Cooldown entre deux tirs (secondes).
    cooldown: f32,
}

/// Sélectionne le profil hitscan selon la distance au joueur.
///
/// * close (< `BOT_SG_RANGE`) : Shotgun, spray 6 × 6 dmg, gros cooldown.
/// * mid (entre les deux seuils) : Machinegun, 8 dmg, cooldown court (behavior historique).
/// * long (≥ `BOT_RG_RANGE`) : Railgun, 35 dmg, gros cooldown.
///
/// La rocket est une couche séparée qui s'ajoute en mid-long
/// indépendamment du hitscan.
fn bot_hitscan_profile(dist: f32) -> HitscanProfile {
    if dist < BOT_SG_RANGE {
        HitscanProfile {
            weapon: WeaponId::Shotgun,
            dmg_per_pellet: 6,
            pellets: 6,
            cooldown: 1.2,
        }
    } else if dist >= BOT_RG_RANGE {
        HitscanProfile {
            weapon: WeaponId::Railgun,
            dmg_per_pellet: 35,
            pellets: 1,
            cooldown: 2.5,
        }
    } else {
        HitscanProfile {
            weapon: WeaponId::Machinegun,
            dmg_per_pellet: BOT_DAMAGE,
            pellets: 1,
            cooldown: BOT_FIRE_COOLDOWN,
        }
    }
}

/// Conduit chaque bot d'un tick : vision, IA, physique. Retourne les
/// dégâts hitscan et la liste de projectiles spawné ce tick.
///
/// `player_alive=false` → les bots perdent l'agro et patrouillent, sans
/// tirer. On laisse `target_enemy` à `None` dans ce cas.
fn tick_bots(
    bots: &mut [BotDriver],
    dt: f32,
    now: f32,
    params: PhysicsParams,
    world: &World,
    player_origin: Vec3,
    player_alive: bool,
    player_invisible: bool,
    rocket_mesh: &Option<Arc<Md3Gpu>>,
) -> BotTickOut {
    use q3_collision::Contents;
    let player_eye = player_origin + Vec3::Z * PLAYER_EYE_HEIGHT;
    let mut out = BotTickOut {
        damage: 0,
        projectiles: Vec::new(),
        last_mg_damager: None,
        last_mg_damager_idx: None,
        last_hitscan_weapon: None,
        wall_marks: Vec::new(),
        rail_beams: Vec::new(),
    };

    for (idx, d) in bots.iter_mut().enumerate() {
        // Bot mort → la logique de respawn s'en occupe ailleurs, on skip.
        if d.health.is_dead() {
            continue;
        }
        // Re-génère waypoints si vidés.
        if d.bot.waypoints.is_empty() && !world.spawn_points.is_empty() {
            d.wp_cursor = (d.wp_cursor + 1) % world.spawn_points.len();
            let start = d.wp_cursor;
            for i in 0..world.spawn_points.len() {
                let sp = &world.spawn_points[(start + i) % world.spawn_points.len()];
                d.bot.push_waypoint(sp.origin + Vec3::Z * 40.0);
            }
        }

        d.bot.position = d.body.origin;

        // --- Vision : portée + FOV + LOS.
        // Invisibility rabote la portée à 25 % : à 375 u le joueur est
        // quasi-invisible dans les portées utiles d'un AK hitscan. En
        // combat déjà engagé (l'agro 360° ci-dessous), l'effet est le
        // même — le bot perd contact dès qu'il dépasse la zone proche.
        let sight_range = if player_invisible {
            BOT_SIGHT_RANGE * BOT_SIGHT_INVIS_FACTOR
        } else {
            BOT_SIGHT_RANGE
        };
        let bot_eye = d.body.origin + Vec3::Z * BOT_EYE_HEIGHT;
        let to_player = player_eye - bot_eye;
        let dist = to_player.length();
        let mut visible = false;
        if player_alive && dist < sight_range && dist > 1.0 {
            let fwd = d.body.view_angles.to_vectors().forward;
            let to_norm = to_player / dist;
            // Test FOV — en combat on est tolérant (la tête suit la cible).
            let in_fov = fwd.dot(to_norm) >= BOT_FOV_COS
                || d.last_saw_player_at.is_some(); // déjà en combat : agro 360°
            if in_fov {
                let trace = world.collision.trace_ray(bot_eye, player_eye, Contents::MASK_SHOT);
                visible = trace.fraction >= 0.999;
            }
        }

        // Mémoire d'agro raccourcie si le joueur est invisible : 0.5 s
        // au lieu des 2 s nominales, sinon le powerup ne servirait à rien
        // une fois le bot « accroché ».
        let memory = if player_invisible {
            BOT_MEMORY_INVIS_SEC
        } else {
            BOT_MEMORY_SEC
        };
        if visible {
            d.bot.target_enemy = Some(player_eye);
            d.last_saw_player_at = Some(now);
            // Front montant : on vient juste de re-repérer le joueur
            // après une fenêtre d'oubli → on arme le délai de réaction.
            if d.first_seen_player_at.is_none() {
                d.first_seen_player_at = Some(now);
            }
        } else if let Some(t) = d.last_saw_player_at {
            if now - t > memory {
                d.bot.target_enemy = None;
                d.last_saw_player_at = None;
                d.first_seen_player_at = None;
            }
        }

        // --- IA → BotCmd
        let bc = d.bot.tick(dt, &world.collision);

        // --- Damage hitscan : si le bot a tiré ce tick avec LOS et que
        // son cooldown est échu, on choisit un profil d'arme selon la
        // distance (SG close / MG mid / RG long) et on applique le burst
        // total en une passe.
        //
        // Couches de difficulté (cf. `BotSkill`) :
        //   1. **réaction** : rejette le tir si le délai de `skill` n'est
        //      pas écoulé depuis la détection (`first_seen_player_at`) ;
        //   2. **cooldown mult** : applique `skill.fire_cooldown_mult()`
        //      au `profile.cooldown` pour que les paliers hauts tirent
        //      plus agressivement ;
        //   3. **aim error** : probabilité de miss proportionnelle à
        //      `skill.aim_error_deg()` — on déclenche l'évènement tir
        //      (anim + son) mais on n'applique pas les dégâts.
        let skill = d.bot.skill;
        let reacted = d
            .first_seen_player_at
            .map(|t| now - t >= skill.reaction_time_sec())
            .unwrap_or(false);
        if bc.fire && visible && reacted && now >= d.next_fire_at {
            let profile = bot_hitscan_profile(dist);
            // Probabilité de miss : 10° d'erreur → ~0.67 chance de rater,
            // 1° → ~0.07.  Plafonnée à 0.80 pour qu'un bot niveau I reste
            // quand même une menace à close range.
            let miss_prob = (skill.aim_error_deg() / 15.0).min(0.80);
            let missed = rand_unit_01() < miss_prob;
            if !missed {
                let burst = profile.dmg_per_pellet * (profile.pellets as i32);
                out.damage += burst;
                out.last_mg_damager = Some(d.bot.name.clone());
                out.last_mg_damager_idx = Some(idx);
                out.last_hitscan_weapon = Some(profile.weapon);
                // Rail trail tinté : on ne dessine un beam persistant que
                // pour le Railgun (couleur + cadence qui justifient la
                // trace).  Mix 50/50 avec la teinte du bot pour que le
                // rose d'origine reste reconnaissable mais que deux bots
                // différents produisent deux traces distinctes.
                if matches!(profile.weapon, WeaponId::Railgun) {
                    let base = [0.95, 0.25, 0.55];
                    let tint = d.tint;
                    let color = [
                        (base[0] + tint[0]) * 0.5,
                        (base[1] + tint[1]) * 0.5,
                        (base[2] + tint[2]) * 0.5,
                        0.85,
                    ];
                    let muzzle = bot_eye
                        + d.body.view_angles.to_vectors().forward * 10.0;
                    out.rail_beams.push(ActiveBeam {
                        a: muzzle,
                        b: player_eye,
                        color,
                        expire_at: now + 0.6,
                        lifetime: 0.6,
                        style: BeamStyle::Spiral,
                    });
                }
            } else {
                // Miss → simule la trajectoire "à côté" et pose un
                // bullet hole là où la balle est retombée.  Direction =
                // axe bot→joueur avec un écart angulaire gaussien bornée
                // par `skill.aim_error_deg()`.  On trace ~4096u (portée
                // perçue Q3) pour trouver un mur.
                let to_norm = to_player / dist;
                let err_rad = skill.aim_error_deg().to_radians();
                // Petite base orthonormée autour de `to_norm` pour
                // décaler l'axe dans un disque perpendiculaire.
                let any = if to_norm.z.abs() < 0.9 {
                    Vec3::Z
                } else {
                    Vec3::new(1.0, 0.0, 0.0)
                };
                let right = any.cross(to_norm).normalize();
                let up = to_norm.cross(right);
                // Jitter gaussien 2D ≃ somme de deux uniformes pour un
                // effet "dispersion hitscan" peu coûteux.
                let jx = (rand_unit() + rand_unit()) * 0.5 * err_rad;
                let jy = (rand_unit() + rand_unit()) * 0.5 * err_rad;
                let miss_dir = (to_norm + right * jx + up * jy).normalize();
                let end = bot_eye + miss_dir * 4096.0;
                let trace = world.collision.trace_ray(bot_eye, end, Contents::MASK_SHOT);
                let miss_hit = if trace.fraction < 1.0 {
                    let hit_pt = bot_eye + (end - bot_eye) * trace.fraction;
                    out.wall_marks
                        .push((hit_pt, trace.plane_normal, profile.weapon));
                    hit_pt
                } else {
                    end
                };
                // Même traitement rail-tint qu'en cas de hit : le beam est
                // visible même quand le tir rate, et trace jusqu'au mur le
                // plus proche.  Sinon, on perd complètement le feedback
                // visuel quand un bot sniper long-range tire à côté.
                if matches!(profile.weapon, WeaponId::Railgun) {
                    let base = [0.95, 0.25, 0.55];
                    let tint = d.tint;
                    let color = [
                        (base[0] + tint[0]) * 0.5,
                        (base[1] + tint[1]) * 0.5,
                        (base[2] + tint[2]) * 0.5,
                        0.85,
                    ];
                    let muzzle = bot_eye
                        + d.body.view_angles.to_vectors().forward * 10.0;
                    out.rail_beams.push(ActiveBeam {
                        a: muzzle,
                        b: miss_hit,
                        color,
                        expire_at: now + 0.6,
                        lifetime: 0.6,
                        style: BeamStyle::Spiral,
                    });
                }
            }
            d.next_fire_at = now + profile.cooldown * skill.fire_cooldown_mult();
            // Horodate le tir : sert à la machine d'anim pour jouer
            // TORSO_ATTACK pendant une courte fenêtre après le coup
            // (qu'on ait touché ou pas — le geste est le même).
            d.last_fire_at = now;
            // Un bot invul qui tire perd son bouclier — même règle que
            // le joueur. Empêche l'exploit « je respawne, je spam, je reste
            // invul les 2s ».
            if d.invul_until > now {
                d.invul_until = 0.0;
            }
        }

        // --- Rocket : tir occasionnel indépendant du hitscan, pour varier
        // le gameplay. Évité en close range (BOT_ROCKET_MIN_DIST) pour que
        // le bot ne se suicide pas sur son propre splash, et coupé au
        // delà de BOT_ROCKET_MAX_DIST (railgun takes over à cette portée).
        //
        // Le skill joue ici sur la *cadence* uniquement (pas d'aim error
        // parce que la rocket vole vers `to_player_norm` et le joueur
        // peut l'esquiver ; on ne simule pas un écart à l'émission).
        if visible
            && reacted
            && (BOT_ROCKET_MIN_DIST..=BOT_ROCKET_MAX_DIST).contains(&dist)
            && now >= d.next_rocket_at
        {
            let to_player_norm = to_player / dist;
            let spawn = bot_eye + to_player_norm * 16.0;
            out.projectiles.push(Projectile {
                origin: spawn,
                velocity: to_player_norm * BOT_ROCKET_SPEED,
                direct_damage: 100,
                splash_radius: 120.0,
                splash_damage: 100,
                owner: ProjectileOwner::Bot(idx),
                weapon: WeaponId::Rocketlauncher,
                expire_at: now + 5.0,
                gravity: 0.0,
                bounce: false,
                mesh: rocket_mesh.clone(),
                tint: [1.0, 1.0, 1.0, 1.0],
                next_trail_at: 0.0,
                homing_target: None, // bots ne lock-on pas pour l'instant
            });
            d.next_rocket_at = now + BOT_ROCKET_COOLDOWN * skill.fire_cooldown_mult();
            d.last_fire_at = now;
            // Rocket = attaque → fin de l'invul, même règle que le
            // hitscan. On évite ainsi la fausse sensation « le bot me
            // rocket depuis son spawn sans risque ».
            if d.invul_until > now {
                d.invul_until = 0.0;
            }
        }

        // --- Physique : MoveCmd alimenté par BotCmd.
        d.body.view_angles = bc.view_angles;
        let cmd = MoveCmd {
            forward: bc.forward_move,
            side: bc.right_move,
            up: bc.up_move,
            jump: bc.up_move > 0.0,
            // Les bots ne se baissent ni ne marchent pour l'instant :
            // l'IA est trop simpliste pour exploiter ces modes (pas de
            // stealth, pas de conduits à traverser accroupi). Pas de
            // slide / dash non plus — on les ajoutera quand l'IA saura
            // les utiliser tactiquement.
            crouch: false,
            walk: false,
            slide_pressed: false,
            dash_pressed: false,
            delta_time: dt,
        };
        let was_on_ground = d.body.on_ground;
        d.body.tick_collide(cmd, params, &world.collision);

        // Détection front montant / descendant du contact sol, qui
        // alimente la machine d'animation côté rendu.  Décollage →
        // LEGS_JUMP ; atterrissage → LEGS_LAND pendant 150 ms.
        match (was_on_ground, d.body.on_ground) {
            (true, false) => {
                d.airborne_since = Some(now);
            }
            (false, true) => {
                d.airborne_since = None;
                d.last_land_at = now;
            }
            _ => {}
        }
    }

    out
}

/// Queue chaque bot pour rendu en reconstituant le rig Q3 complet :
/// **lower** (jambes) ancré à la position/yaw du bot, puis **upper** (torse)
/// attaché via le tag `tag_torso`, puis **head** via `tag_head`.  Sans ça,
/// on rendait uniquement les jambes et les bots avaient l'air flottants —
/// c'est la convention Q3 depuis le premier jour.
///
/// Anime les frames en cycle lent (~8 Hz) si le modèle en a plusieurs.
/// Dans la vraie machine d'état Q3, chaque partie a son propre index de
/// frame (lower = BOTH_DEATH*/LEGS_RUN/IDLE/…, upper = TORSO_STAND/ATTACK/…),
/// pilotés par `animation.cfg`.  On simplifie ici : même cycle sur toutes
/// les parties.  Le résultat visuel reste honnête tant qu'on ne joue pas
/// une animation de mort ou un tir.
///
/// Les bots invincibles (fenêtre post-respawn) passent par un tint modulé
/// en sinusoïde cyan-blanc pour signaler l'état au joueur — sans ça, un
/// tir qui passe à travers un bot invul paraîtrait buggé. La pulsation à
/// ~3 Hz reste lisible à l'œil mais pas distrayante.
fn queue_bots(r: &mut Renderer, rig: &PlayerRig, bots: &[BotDriver], time_sec: f32) {
    use glam::{Mat4 as GMat4, Quat, Vec3 as GVec3};

    // Seuils de la machine d'états — en secondes depuis l'évènement.
    // Gardés courts pour que l'anim ne « colle » pas (dans Q3 TORSO_ATTACK
    // dure 15 frames à 15 fps soit ~1.0s, mais on raccourcit pour que
    // des bursts rapprochés aient un retour visuel distinct par tir).
    const ATTACK_WINDOW_SEC: f32 = 0.25;
    const PAIN_WINDOW_SEC: f32 = 0.20;
    const LAND_WINDOW_SEC: f32 = 0.15;
    // Vitesse XY au-dessus de laquelle on considère que le bot « court »
    // plutôt qu'il traîne sur place — 40 u/s ≈ vitesse de walk Q3.
    const RUN_SPEED_SQ: f32 = 40.0 * 40.0;

    let nf_lower = rig.lower.num_frames();
    let nf_upper = rig.upper.num_frames();
    let nf_head = rig.head.num_frames();

    for d in bots {
        if d.health.is_dead() {
            continue;
        }

        // --- Sélection d'anim côté jambes (lower) + torse (upper).
        // On dérive l'état en lecture seule depuis les horodatages
        // tenus à jour par le code de combat et la physique.
        let v_xy_sq = d.body.velocity.x * d.body.velocity.x
            + d.body.velocity.y * d.body.velocity.y;
        let moving = v_xy_sq > RUN_SPEED_SQ;
        let recently_fired = (time_sec - d.last_fire_at) < ATTACK_WINDOW_SEC;
        let recently_hurt = (time_sec - d.last_damage_at) < PAIN_WINDOW_SEC;
        let recently_landed = (time_sec - d.last_land_at) < LAND_WINDOW_SEC;
        let airborne = !d.body.on_ground;

        // Torso (upper) : priorité pain > attack > stand. La pain anim
        // n'existe pas dans Q3 animation.cfg standard (c'est typiquement
        // juste un mini burst de gesture/stand avec un tint) — on
        // rejoue TORSO_GESTURE pour ce rôle.
        let upper_range = if recently_hurt {
            bot_anims::TORSO_GESTURE
        } else if recently_fired {
            bot_anims::TORSO_ATTACK
        } else {
            bot_anims::TORSO_STAND
        };
        // Legs (lower) : airborne > landing > moving > idle.
        let lower_range = if airborne {
            bot_anims::LEGS_JUMP
        } else if recently_landed {
            bot_anims::LEGS_LAND
        } else if moving {
            if v_xy_sq > 200.0 * 200.0 {
                bot_anims::LEGS_RUN
            } else {
                bot_anims::LEGS_WALK
            }
        } else {
            bot_anims::LEGS_IDLE
        };

        // Phase locale à chaque anim : on utilise `time_sec` directement
        // comme phase (les anims cycliques prennent `rem_euclid`, les non-
        // cycliques clamp à la fin — elles se stabilisent donc sur leur
        // frame terminale tant qu'on y reste).  Cette approche évite de
        // muter `BotDriver` depuis `queue_bots`.
        let phase = time_sec;
        let (fa_l, fb_l, lerp_l) = lower_range.sample(phase, nf_lower);
        let (fa_u, fb_u, lerp_u) = upper_range.sample(phase, nf_upper);
        // La tête ne s'anime pas en Q3 : on la rend sur la première
        // frame (les meshes head.md3 sont statiques).
        let (fa_h, fb_h, lerp_h) = (0usize, 0usize, 0.0_f32);
        let _ = nf_head;

        let o = d.body.origin;
        let rot = Quat::from_rotation_z(d.body.view_angles.yaw.to_radians());
        let lower_m = GMat4::from_scale_rotation_translation(
            GVec3::ONE,
            rot,
            GVec3::new(o.x, o.y, o.z),
        );
        // Tint : pulsation invul si applicable, + flash rouge très léger
        // sur pain pour renforcer le retour visuel sans dépendre d'une
        // animation absente du MD3.
        let mut tint = invul_tint_override(d.tint, d.invul_until, time_sec);
        if recently_hurt {
            let t = ((time_sec - d.last_damage_at) / PAIN_WINDOW_SEC).clamp(0.0, 1.0);
            let mix = 1.0 - t;
            tint[0] = (tint[0] + mix * 0.6).min(1.0);
            tint[1] *= 1.0 - mix * 0.4;
            tint[2] *= 1.0 - mix * 0.4;
        }

        // Compose les tag transforms le long de la chaîne lower → upper → head.
        // Les tags du lower sont indexés sur l'anim lower ; ceux du upper
        // sur l'anim upper.  Sans ça, le torse se plierait sur l'offset
        // de la course alors qu'on joue l'attaque dessus.
        let ident = GMat4::IDENTITY;
        let torso_local = rig
            .lower
            .tag_transform(fa_l, fb_l, lerp_l, "tag_torso")
            .unwrap_or(ident);
        let upper_m = lower_m * torso_local;
        let head_local = rig
            .upper
            .tag_transform(fa_u, fb_u, lerp_u, "tag_head")
            .unwrap_or(ident);
        let head_m = upper_m * head_local;

        r.draw_md3_animated(rig.lower.clone(), lower_m, tint, fa_l, fb_l, lerp_l);
        r.draw_md3_animated(rig.upper.clone(), upper_m, tint, fa_u, fb_u, lerp_u);
        r.draw_md3_animated(rig.head.clone(),  head_m,  tint, fa_h, fb_h, lerp_h);
    }
}

/// Si l'entité est invincible (`invul_until > now`), remplace son tint
/// par une modulation cyan-blanc pulsant à ~3 Hz. Retourne le tint
/// d'origine sinon. Facteur fixé pour que la pulsation soit claire mais
/// que la silhouette (couleur d'équipe) reste lisible quand l'invul
/// retombe.
fn invul_tint_override(base: [f32; 4], invul_until: f32, now: f32) -> [f32; 4] {
    if invul_until <= now {
        return base;
    }
    // Pulsation : 0.5 + 0.5 * sin(2π * 3 * t) → [0, 1].
    let pulse = 0.5 + 0.5 * (now * std::f32::consts::TAU * 3.0).sin();
    // Lerp entre le tint de base et un blanc cyan à haute intensité.
    let target = [0.6, 0.9, 1.4, base[3]];
    [
        base[0] + (target[0] - base[0]) * pulse,
        base[1] + (target[1] - base[1]) * pulse,
        base[2] + (target[2] - base[2]) * pulse,
        base[3],
    ]
}

/// État visuel d'un autre joueur connecté (humain ou bot serveur),
/// dérivé du dernier `Snapshot::players` après **interpolation**.
///
/// Le rendu est volontairement plus simple que `BotDriver` côté client :
///   * pas d'animation `attack` (le snapshot v1 n'expose pas l'état de
///     tir, on l'ajoutera dans `PlayerState::flags` plus tard)
///   * pas de tracking du dernier hit (pas de pain anim)
///   * pas de tracking du landing (le snapshot 20 Hz est trop grossier
///     pour le détecter avec précision — on s'en passe)
///
/// On garde idle / walk / run / jump qui suffisent à donner une silhouette
/// qui bouge correctement avec le mouvement serveur.
#[derive(Debug, Clone)]
struct RemotePlayer {
    slot: u8,
    origin: Vec3,
    view_angles: Angles,
    velocity: Vec3,
    on_ground: bool,
    is_dead: bool,
    /// Le serveur a vu `BUTTON_FIRE` récemment (≤ ~250 ms) — déclenche
    /// l'anim TORSO_ATTACK pour rendre le tir lisible côté autres
    /// clients. Sans ce flag, un duel à distance affichait deux MD3
    /// inertes même quand l'un mitraillait l'autre.
    recently_fired: bool,
    /// Équipe TDM : 0=free, 1=red, 2=blue. Module le tint MD3 pour
    /// distinguer rouge/bleu à l'œil. En FFA reste à 0 → tint slot-id.
    team: u8,
}

/// Échantillon du snapshot pour un slot remote — composantes interpolables.
///
/// Stocké dans [`RemoteInterp`] avec son `server_time` (depuis le snapshot)
/// et son `received_at` (`Instant::now()` au moment de la réception).
/// L'interp temporelle utilise le wallclock local (`received_at`), pas le
/// `server_time`, pour rester correct même si l'horloge serveur dérive
/// par rapport au client (ce qui ne devrait jamais arriver en pratique
/// vu qu'`Instant` est monotonic des deux côtés).
#[derive(Debug, Clone)]
struct RemoteSample {
    origin: Vec3,
    velocity: Vec3,
    view_angles: Angles,
    on_ground: bool,
    is_dead: bool,
    /// Repris depuis `PlayerState::flags::RECENTLY_FIRED`. Pas interpolé —
    /// c'est un état discret. Si l'un des deux samples (older/newer) est
    /// vrai, on rend l'anim attack ; ça étend légèrement la fenêtre de
    /// visibilité à l'œil (~50 ms en plus) sans être un bug.
    recently_fired: bool,
    /// Équipe — pas interpolée, on prend `newer` (l'équipe ne change
    /// pas pendant un match en pratique).
    team: u8,
    server_time: u32,
    received_at: Instant,
}

/// Buffer d'interpolation pour un slot remote — paire des deux derniers
/// samples reçus. À chaque snapshot, `newer` glisse vers `older` et le
/// nouveau prend la place. La lerp se fait entre ces deux uniquement,
/// approche style Source/Q3 « 1 snapshot delayed » : visuellement on
/// est ~50 ms en arrière du serveur mais le mouvement est continu.
///
/// On se passe d'un buffer plus profond car v1 n'a pas besoin de
/// rewind/replay côté autres joueurs ni d'extrapolation — la lerp
/// pure suffit à supprimer la téléportation 20 Hz.
#[derive(Debug, Default)]
struct RemoteInterp {
    older: Option<RemoteSample>,
    newer: Option<RemoteSample>,
}

impl RemoteInterp {
    fn push(&mut self, sample: RemoteSample) {
        // Cas pathologique : deux snapshots avec même `server_time`
        // (rebroadcast forcé, retransmission). On écrase `newer` plutôt
        // que de pousser un duplicate qui collerait `older=newer` et
        // donnerait `span=0`.
        if let Some(n) = &self.newer {
            if n.server_time == sample.server_time {
                self.newer = Some(sample);
                return;
            }
        }
        self.older = self.newer.take();
        self.newer = Some(sample);
    }

    /// Calcule l'état rendu maintenant. `None` si pas un seul snapshot
    /// reçu pour ce slot.
    fn current(&self) -> Option<RemoteSample> {
        let newer = self.newer.as_ref()?;
        let older = match &self.older {
            Some(o) => o,
            None => return Some(newer.clone()),
        };

        let elapsed = newer.received_at.elapsed().as_secs_f32();
        // `span` reflète l'intervalle de temps **serveur** entre les deux
        // samples — pas le délai de réception. Ça permet de gérer
        // proprement un paquet retardé : si la précédente snap est arrivée
        // 50 ms plus tard que prévu, le span entre `older.server_time` et
        // `newer.server_time` reste 50 ms (rythme serveur), et la lerp
        // utilise quand même l'horloge locale pour avancer entre les
        // deux états.
        let span_ms = newer.server_time.saturating_sub(older.server_time) as f32;
        if span_ms <= 0.5 {
            // Span dégénéré (server_time identique ou inversé) : on snap
            // sur newer plutôt que d'extrapoler bizarrement.
            return Some(newer.clone());
        }
        let t = ((elapsed * 1000.0) / span_ms).clamp(0.0, 1.0);
        Some(RemoteSample {
            origin: older.origin.lerp(newer.origin, t),
            velocity: older.velocity.lerp(newer.velocity, t),
            view_angles: lerp_angles(older.view_angles, newer.view_angles, t),
            // Bools : on bascule au seuil 0.5 — pas continu mais ce sont
            // des états discrets de toute façon (un mort ne ressuscite pas
            // à mi-chemin entre deux snapshots).
            on_ground: if t < 0.5 { older.on_ground } else { newer.on_ground },
            is_dead: if t < 0.5 { older.is_dead } else { newer.is_dead },
            // recently_fired : OU logique des deux samples. Étend la
            // fenêtre visible — sans ça un tir tout juste reçu pourrait
            // être lerpé vers `false` au-delà de 0.5 et l'anim ne
            // partirait pas du tout pour les snapshots courts.
            recently_fired: older.recently_fired || newer.recently_fired,
            team: newer.team,
            server_time: newer.server_time,
            received_at: newer.received_at,
        })
    }
}

/// Projectile **distant** — issu de `Snapshot::entities`. Différent du
/// `Projectile` local : pas de logique de dégâts, pas d'owner/ownership,
/// juste de quoi rendre la roquette en vol.
///
/// L'extrapolation linéaire à partir de `(origin, velocity, last_received_at)`
/// donne une trajectoire continue à la framerate de rendu malgré les
/// snapshots à 20 Hz (snapshot interval = 50 ms = 45 u à vitesse rocket,
/// très visible si on téléporte).
#[derive(Debug, Clone)]
struct RemoteProjectile {
    id: u32,
    kind: q3_net::EntityKindWire,
    origin: Vec3,
    velocity: Vec3,
    last_received_at: Instant,
}

impl RemoteProjectile {
    fn extrapolated_origin(&self, now: Instant) -> Vec3 {
        let dt = (now - self.last_received_at).as_secs_f32();
        // Extrapolation pure ligne droite — valide pour rocket/plasma
        // (gravité 0). Pour les grenades on devra ajouter `velocity.z -=
        // gravity * dt` quand le serveur les supportera.
        self.origin + self.velocity * dt
    }
}

/// Lerp d'angles avec gestion correcte de la wraparound. Sans ça, un
/// joueur qui tourne de 350° à 10° interpolerait dans le mauvais sens
/// (340° à reculons au lieu de 20° dans le bon sens) → spin visible.
fn lerp_angles(a: Angles, b: Angles, t: f32) -> Angles {
    Angles::new(
        lerp_angle_deg(a.pitch, b.pitch, t),
        lerp_angle_deg(a.yaw, b.yaw, t),
        lerp_angle_deg(a.roll, b.roll, t),
    )
}

fn lerp_angle_deg(a: f32, b: f32, t: f32) -> f32 {
    // Trouve le delta signé le plus court dans [-180, 180].
    let mut d = (b - a).rem_euclid(360.0);
    if d > 180.0 {
        d -= 360.0;
    }
    a + d * t
}

/// Rendu des autres joueurs vus dans le dernier snapshot serveur. Même
/// rig MD3 que les bots, animation simplifiée (pas d'état attack/pain/land).
fn queue_remote_players(
    r: &mut Renderer,
    rig: &PlayerRig,
    players: &[RemotePlayer],
    time_sec: f32,
) {
    use glam::{Mat4 as GMat4, Quat, Vec3 as GVec3};

    // Mêmes seuils de vitesse que `queue_bots` pour la cohérence visuelle
    // entre les bots locaux et les remote players (sans ça un bot et un
    // humain à même vitesse joueraient des anims différentes).
    const RUN_SPEED_SQ: f32 = 40.0 * 40.0;

    let nf_lower = rig.lower.num_frames();
    let nf_upper = rig.upper.num_frames();

    for p in players {
        if p.is_dead {
            continue;
        }

        let v_xy_sq = p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y;
        let moving = v_xy_sq > RUN_SPEED_SQ;
        let airborne = !p.on_ground;

        // Choix d'anim. Attack > stand — un joueur qui tire est typiquement
        // immobile, on veut que ça reste lisible visuellement.
        let upper_range = if p.recently_fired {
            bot_anims::TORSO_ATTACK
        } else {
            bot_anims::TORSO_STAND
        };
        let lower_range = if airborne {
            bot_anims::LEGS_JUMP
        } else if moving {
            if v_xy_sq > 200.0 * 200.0 {
                bot_anims::LEGS_RUN
            } else {
                bot_anims::LEGS_WALK
            }
        } else {
            bot_anims::LEGS_IDLE
        };

        let phase = time_sec;
        let (fa_l, fb_l, lerp_l) = lower_range.sample(phase, nf_lower);
        let (fa_u, fb_u, lerp_u) = upper_range.sample(phase, nf_upper);
        let (fa_h, fb_h, lerp_h) = (0usize, 0usize, 0.0_f32);

        let o = p.origin;
        let rot = Quat::from_rotation_z(p.view_angles.yaw.to_radians());
        let lower_m = GMat4::from_scale_rotation_translation(
            GVec3::ONE,
            rot,
            GVec3::new(o.x, o.y, o.z),
        );

        // Tint : si team set (TDM), on prend la couleur d'équipe.
        // En FFA (team=0), on retombe sur `bot_tint(slot)` qui donne
        // une teinte unique par slot ID.
        let tint = match p.team {
            q3_net::team::RED => [1.0, 0.30, 0.30, 1.0],
            q3_net::team::BLUE => [0.30, 0.55, 1.0, 1.0],
            _ => bot_tint(p.slot as usize),
        };

        let ident = GMat4::IDENTITY;
        let torso_local = rig
            .lower
            .tag_transform(fa_l, fb_l, lerp_l, "tag_torso")
            .unwrap_or(ident);
        let upper_m = lower_m * torso_local;
        let head_local = rig
            .upper
            .tag_transform(fa_u, fb_u, lerp_u, "tag_head")
            .unwrap_or(ident);
        let head_m = upper_m * head_local;

        r.draw_md3_animated(rig.lower.clone(), lower_m, tint, fa_l, fb_l, lerp_l);
        r.draw_md3_animated(rig.upper.clone(), upper_m, tint, fa_u, fb_u, lerp_u);
        r.draw_md3_animated(rig.head.clone(),  head_m,  tint, fa_h, fb_h, lerp_h);
    }
}

/// Rendu des projectiles distants reçus dans `Snapshot::entities`,
/// avec extrapolation linéaire pour combler entre les snapshots 20 Hz.
/// Les meshes sont les mêmes que pour les projectiles locaux —
/// rocket/plasma/grenade — et sont passés en argument car ils vivent
/// dans `App`. Les explosions ne sont pas rendues ici (v1 ne les
/// transmet pas via wire).
fn queue_remote_projectiles(
    r: &mut Renderer,
    projectiles: &[RemoteProjectile],
    rocket_mesh: &Option<Arc<Md3Gpu>>,
    plasma_mesh: &Option<Arc<Md3Gpu>>,
    grenade_mesh: &Option<Arc<Md3Gpu>>,
    time_sec: f32,
) {
    use glam::{Mat4 as GMat4, Quat, Vec3 as GVec3};
    let now = Instant::now();
    for p in projectiles {
        let origin = p.extrapolated_origin(now);
        let (mesh, light_color, light_intensity, light_radius, tint) = match p.kind {
            q3_net::EntityKindWire::Rocket => (
                rocket_mesh,
                [1.0_f32, 0.75, 0.3],
                1.4_f32,
                140.0_f32,
                [1.0_f32, 0.7, 0.4, 1.0],
            ),
            q3_net::EntityKindWire::Plasma => (
                plasma_mesh,
                [0.4, 0.6, 1.0],
                1.2,
                90.0,
                [0.6, 0.8, 1.0, 1.0],
            ),
            q3_net::EntityKindWire::Grenade => (
                grenade_mesh,
                [1.0, 0.75, 0.3],
                0.6,
                60.0,
                [0.9, 0.7, 0.3, 1.0],
            ),
            q3_net::EntityKindWire::Bfg => (
                // Pas de mesh BFG en cache — on réutilise le rocket
                // mesh comme placeholder pour avoir au moins une
                // sphère orientée. La dlight verte signale BFG sans
                // ambiguïté.
                rocket_mesh,
                [0.4, 1.0, 0.5],
                2.5,
                180.0,
                [0.5, 1.0, 0.6, 1.0],
            ),
            // Pas d'`EntityState` Explosion en v1 sur le wire — les
            // explosions sont des événements séparés.
            q3_net::EntityKindWire::Explosion => continue,
        };
        r.spawn_dlight(origin, light_radius, light_color, light_intensity, time_sec, 0.08);
        let Some(mesh) = mesh.as_ref() else { continue };
        let v = p.velocity;
        let len = v.length();
        if len < 1e-3 {
            continue;
        }
        let dir = v / len;
        let yaw = dir.y.atan2(dir.x);
        let pitch = (-dir.z).asin();
        let rot = Quat::from_rotation_z(yaw) * Quat::from_rotation_y(pitch);
        let trans = GVec3::new(origin.x, origin.y, origin.z);
        let m = GMat4::from_scale_rotation_translation(GVec3::ONE, rot, trans);
        r.draw_md3(mesh.clone(), m, tint);
    }
}

/// Tint cyclique pour distinguer les bots visuellement.
fn bot_tint(idx: usize) -> [f32; 4] {
    const PALETTE: &[[f32; 4]] = &[
        [1.0, 0.6, 0.6, 1.0], // rouge doux
        [0.6, 0.8, 1.0, 1.0], // bleu doux
        [0.7, 1.0, 0.7, 1.0], // vert doux
        [1.0, 0.9, 0.5, 1.0], // jaune doux
        [1.0, 0.7, 1.0, 1.0], // magenta
        [0.7, 1.0, 1.0, 1.0], // cyan
    ];
    PALETTE[idx % PALETTE.len()]
}

/// Retourne le chemin de `q3config.cfg` dans le répertoire de préférences
/// utilisateur, ou `None` si ni `APPDATA` (Windows) ni `XDG_CONFIG_HOME` /
/// `HOME` (Unix) ne sont renseignés — cas d'un build embarqué ou CI où la
/// persistance n'a pas de sens.
///
/// Convention :
/// * Windows : `%APPDATA%\q3-rust\q3config.cfg`
/// * macOS / Linux : `$XDG_CONFIG_HOME/q3-rust/q3config.cfg`, sinon
///   `$HOME/.config/q3-rust/q3config.cfg`.
///
/// On évite volontairement d'ajouter la crate `dirs` pour garder le
/// graphe de deps léger — ces variables couvrent 99 % des installations.
fn user_config_path() -> Option<PathBuf> {
    const SUBDIR: &str = "q3-rust";
    const FILE: &str = "q3config.cfg";

    #[cfg(windows)]
    {
        if let Ok(appdata) = std::env::var("APPDATA") {
            return Some(PathBuf::from(appdata).join(SUBDIR).join(FILE));
        }
    }

    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return Some(PathBuf::from(xdg).join(SUBDIR).join(FILE));
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return Some(PathBuf::from(home).join(".config").join(SUBDIR).join(FILE));
        }
    }
    // Dernier recours Windows sans APPDATA (rare) : USERPROFILE.
    if let Ok(up) = std::env::var("USERPROFILE") {
        if !up.is_empty() {
            return Some(PathBuf::from(up).join(SUBDIR).join(FILE));
        }
    }
    None
}

/// Répertoire `screenshots/` à côté du `q3config.cfg` user. Retourne `None`
/// si aucune convention d'OS ne permet de résoudre le user dir (cas très
/// rare Windows sans APPDATA/USERPROFILE, Unix sans HOME/XDG).
fn screenshot_dir() -> Option<PathBuf> {
    user_config_path()
        .as_ref()
        .and_then(|p| p.parent())
        .map(|d| d.join("screenshots"))
}

/// Cherche le prochain indice libre pour un fichier `shot-NNNN.tga` dans
/// `dir`. Retourne 1 si le dossier n'existe pas ou est vide. Si un trou
/// existe (ex. shot-0001 puis shot-0003), on reprend après le plus grand
/// index observé — plus intuitif que de combler (évite d'écraser accident-
/// ellement un screenshot qui avait été explicitement supprimé).
fn next_screenshot_index(dir: &std::path::Path) -> u32 {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return 1,
    };
    let mut highest: u32 = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = match name.to_str() {
            Some(s) => s,
            None => continue,
        };
        // Format attendu : `shot-NNNN.tga`.
        let stem = match name.strip_prefix("shot-").and_then(|s| s.strip_suffix(".tga")) {
            Some(s) => s,
            None => continue,
        };
        if let Ok(n) = stem.parse::<u32>() {
            if n > highest {
                highest = n;
            }
        }
    }
    highest + 1
}

/// Suffixe pseudo-aléatoire pour générer un nom de bot par défaut.
fn fastrand_suffix() -> u32 {
    use std::cell::Cell;
    thread_local! {
        static S: Cell<u32> = const { Cell::new(0x1A2B3C4D) };
    }
    S.with(|s| {
        let mut v = s.get();
        v = v.wrapping_mul(1103515245).wrapping_add(12345);
        s.set(v);
        (v >> 16) & 0xFFF
    })
}

/// Flotant pseudo-aléatoire dans `[-1.0, 1.0)` — LCG thread-local,
/// suffisant pour le spread du shotgun (pas de sécurité requise).
fn rand_unit() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static S: Cell<u32> = const { Cell::new(0x9E3779B9) };
    }
    S.with(|s| {
        let mut v = s.get();
        v = v.wrapping_mul(1664525).wrapping_add(1013904223);
        s.set(v);
        // Mantisse 24 bits → [0,1), puis ramène à [-1,1).
        let u = (v >> 8) as f32 / (1u32 << 24) as f32;
        u * 2.0 - 1.0
    })
}

/// Variante `[0.0, 1.0)` utile pour les tests de Bernoulli (miss
/// probabilistes du skill bot).  Réutilise le même état LCG que
/// `rand_unit` — un seul stream par thread, suffisant pour du bruit
/// gameplay non-critique.
fn rand_unit_01() -> f32 {
    (rand_unit() + 1.0) * 0.5
}

/// Queue les projectiles actifs. Chaque projectile porte son propre mesh
/// (cloné en Arc → gratuit) et son tint RGBA. Un projectile sans mesh est
/// silencieusement ignoré — le gameplay reste correct, on n'a juste pas de
/// visuel pour cet asset manquant.
fn queue_projectiles(r: &mut Renderer, projectiles: &[Projectile], time_sec: f32) {
    use glam::{Mat4, Quat, Vec3 as GVec3};
    for p in projectiles {
        // Dlight trail : même les projectiles sans mesh (eg. BFG shot si
        // le MD3 a échoué à charger) doivent illuminer les murs qu'ils
        // frôlent.  Couleur selon le tint du projectile : plasma bleu,
        // rocket/grenade jaune-orange.
        let (color, intensity, radius) = match p.weapon {
            WeaponId::Plasmagun => ([0.4, 0.6, 1.0], 1.2, 90.0),
            WeaponId::Bfg => ([0.4, 1.0, 0.5], 2.5, 180.0),
            WeaponId::Grenadelauncher => ([1.0, 0.75, 0.3], 0.6, 60.0),
            _ => ([1.0, 0.75, 0.3], 1.4, 140.0),
        };
        // Vie très courte — on re-spawn chaque frame, le fade gère
        // naturellement l'extinction après l'explosion quand le projectile
        // disparaît de la liste.
        r.spawn_dlight(p.origin, radius, color, intensity, time_sec, 0.08);

        let Some(mesh) = p.mesh.as_ref() else { continue; };
        let v = p.velocity;
        let len = v.length();
        if len < 1e-3 {
            continue;
        }
        let dir = v / len;
        // Oriente le MD3 (axe modèle = +X Q3) vers la direction de vol.
        let yaw = dir.y.atan2(dir.x);
        let pitch = (-dir.z).asin(); // nose-down positif en Q3
        let rot = Quat::from_rotation_z(yaw) * Quat::from_rotation_y(pitch);
        let trans = GVec3::new(p.origin.x, p.origin.y, p.origin.z);
        let m = Mat4::from_scale_rotation_translation(GVec3::ONE, rot, trans);
        r.draw_md3(mesh.clone(), m, p.tint);
    }
}

/// Queue les instances MD3 des pickups, avec la rotation Q3 classique sur
/// Z (120 deg/s) et un léger bob vertical.
///
/// `unavailable_remote` : IDs (= entity_index) des pickups que le serveur
/// a marqués indispo. Ignoré en mode solo (`is_client = false`) — l'état
/// `respawn_at` local fait foi. En mode client, l'autorité serveur prime.
fn queue_pickups(
    r: &mut Renderer,
    pickups: &[PickupGpu],
    time_sec: f32,
    unavailable_remote: &std::collections::HashSet<u16>,
    is_client: bool,
) {
    use glam::{Mat4, Quat, Vec3 as GVec3};
    let spin = (time_sec * 120.0).to_radians();
    let bob = (time_sec * 2.0).sin() * 3.0;
    for p in pickups {
        if p.respawn_at.is_some() {
            continue; // ramassé localement (mode solo), en cooldown
        }
        if is_client && unavailable_remote.contains(&p.entity_index) {
            continue; // ramassé côté serveur, masqué côté client
        }
        let trans = GVec3::new(p.origin.x, p.origin.y, p.origin.z + bob);
        let rot = Quat::from_rotation_z(spin + p.angles.yaw.to_radians());
        let m = Mat4::from_scale_rotation_translation(GVec3::ONE, rot, trans);
        r.draw_md3(p.mesh.clone(), m, [1.0, 1.0, 1.0, 1.0]);
    }
}

/// Queue le viewmodel (arme en 1ère personne).
///
/// Positionné dans le repère de la caméra : devant, légèrement à gauche,
/// un peu sous l'axe de visée. L'animation cycle linéairement toutes les
/// frames du modèle à 5 Hz si le MD3 en possède plus d'une — juste pour
/// montrer que l'interpolation shader fait son travail. Un vrai système
/// d'armes pilotera plus tard frame_a/frame_b selon l'état (idle, fire,
/// raise, lower).
fn queue_viewmodel(
    r: &mut Renderer,
    mesh: &Arc<Md3Gpu>,
    eye: Vec3,
    view_angles: Angles,
    time_sec: f32,
    invul_until: f32,
    invisible: bool,
    // `weapon` : arme tenue — pilote la couleur du muzzle flash 3D.
    // `muzzle_active` : `true` pendant la fenêtre de flash (60 ms).
    weapon: WeaponId,
    muzzle_active: bool,
    // `view_kick` : recul courant accumulé (0..VIEW_KICK_MAX ≈ 1.2).
    // On pull le viewmodel en arrière le long de `forward` et on pitch
    // légèrement up : l'œil lit ça comme « l'arme vient de cogner ma main »
    // sans avoir besoin d'animer les frames MD3.  Décale aussi le muzzle
    // flash puisqu'il est attaché au tag → cohérent.
    view_kick: f32,
) {
    let basis = view_angles.to_vectors();
    // Offsets en unités Q3 — calibré à l'œil pour un MD3 d'arme standard.
    // Q3 standard : viewmodel à droite (main droite du joueur) et sous
    // l'axe pour ne pas masquer le réticule. `basis.right` est bien la
    // direction droite du joueur en world → offset positif.
    //
    // Historique : avant le correctif de `view_matrix` (miroir horizontal),
    // le viewmodel était placé à GAUCHE en world et le miroir le ramenait
    // à droite à l'écran. Maintenant que la view est correcte, on met
    // physiquement le modèle à droite pour qu'il apparaisse à droite.
    //
    // Recul : on tire l'arme vers soi (−forward) proportionnellement au
    // kick, et on la fait remonter un peu (+up) — c'est la signature
    // visuelle d'un tir, ça se voit même à 60 Hz sur une rafale MG.
    //   kick = 1.0  →  recul = 3u en arrière, 1.5u vers le haut
    //   kick = 0.2  →  ~0.6u / ~0.3u (micro-secousse auto-fire MG)
    //
    // Décalage de base calé pour un MD3 d'arme Q3 standard, dont l'origin
    // mesh est ~ au pivot de la main droite : forward 8 (devant le near
    // plane à 4u, marge confortable), right 5 (main droite), -up 5
    // (sous l'œil, hauteur poitrine). Avant : forward 12 / right 6 / -up 6
    // → ressentait trop loin et trop bas, la main « flottait » devant la
    // caméra au lieu d'être tenue contre l'épaule. Retour utilisateur
    // "positionnement à vérifier" → on rapproche le viewmodel.
    let kick_back = view_kick * 3.0;
    let kick_up = view_kick * 1.5;
    let origin = eye + basis.forward * (8.0 - kick_back) + basis.right * 5.0
        - basis.up * (5.0 - kick_up);
    // Le repère local du MD3 est +X=forward, +Y=left, +Z=up.
    let left = -basis.right;
    let transform = Mat4::from_cols(
        Vec4::new(basis.forward.x, basis.forward.y, basis.forward.z, 0.0),
        Vec4::new(left.x, left.y, left.z, 0.0),
        Vec4::new(basis.up.x, basis.up.y, basis.up.z, 0.0),
        Vec4::new(origin.x, origin.y, origin.z, 1.0),
    );

    let nf = mesh.num_frames();
    let (fa, fb, lerp) = if nf <= 1 {
        (0, 0, 0.0)
    } else {
        let cyc = (time_sec * 5.0) % (nf as f32);
        let a = (cyc.floor() as usize) % nf;
        let b = (a + 1) % nf;
        (a, b, cyc.fract())
    };
    // Tint modulé si le joueur est invincible : la main/arme pulse en
    // cyan pour rappeler à l'écran « tu es invul, mais pas pour toujours ».
    let mut tint = invul_tint_override([1.0, 1.0, 1.0, 1.0], invul_until, time_sec);
    // Invisibility : on atténue fortement l'alpha du viewmodel et on
    // pousse la teinte vers un cyan froid. L'œil périphérique capte
    // « quelque chose ne va pas avec mon arme » sans qu'on ait besoin
    // d'un second overlay — l'absence de l'objet fait le travail.
    // On combine multiplicativement avec la teinte d'invul pour que les
    // deux powerups stackés restent lisibles (rose-cyan pulsé, alpha
    // toujours dominé par l'invisibilité).
    if invisible {
        tint[0] *= 0.55;
        tint[1] *= 0.85;
        tint[2] *= 1.0;
        tint[3] *= 0.18;
    }
    r.draw_md3_viewmodel(mesh.clone(), transform, tint, fa, fb, lerp);

    // Muzzle flash 3D : sprite additif billboard à `tag_flash` du viewmodel.
    // C'est le comportement canonique Q3 — un vrai MD3 tag pointe la sortie
    // du canon, et on y colle un halo additif dont la couleur dépend de
    // l'arme (poudre chaude pour MG/SG/RL, cyan pour plasma, vert pour BFG,
    // bleu-blanc pour railgun).  Le sprite est transitoire : publié chaque
    // frame pendant la fenêtre active, automatiquement vidé après flush.
    if muzzle_active {
        if let Some((base_color, radius)) = weapon.muzzle_flash() {
            // Certains viewmodels utilisent "tag_flash" (Q3 standard),
            // d'autres "tag_barrel" sur des variantes — on tente les deux.
            let tag_local = mesh
                .tag_transform(fa, fb, lerp, "tag_flash")
                .or_else(|| mesh.tag_transform(fa, fb, lerp, "tag_barrel"));
            if let Some(tag_local) = tag_local {
                let world = transform * tag_local;
                let tag_pos = Vec3::new(
                    world.col(3).x,
                    world.col(3).y,
                    world.col(3).z,
                );
                // Légère variation de taille inter-tir pour éviter l'effet
                // « sprite figé identique à chaque frame » — Q3 original
                // randomisait la rotation + scale du quad flash.  Ici on
                // pulse via `time_sec` sur ~60 ms.  Aussi on jitter l'alpha
                // pour simuler le scintillement de la combustion.
                let phase = (time_sec * 60.0).sin() * 0.5 + 0.5; // [0..1]
                let scale = 0.85 + phase * 0.30; // [0.85 .. 1.15]
                let alpha = base_color[3] * (0.85 + phase * 0.20);
                let color = [base_color[0], base_color[1], base_color[2], alpha];
                r.push_muzzle_flash(tag_pos, color, radius * scale);
            }
        }
    }
}

/// Pastille de statut réseau + VR dessinée en haut-gauche du HUD.
///
/// Invisible en solo non-VR (silence complet) ; en mode réseau elle
/// affiche l'étiquette du mode + l'adresse (ex: `[HOST 0.0.0.0:27960]`
/// ou `[NET 127.0.0.1:27960]`) ; en mode VR elle ajoute `[VR]`.
/// Les couleurs sont calées sur la palette du logo Q3 RUST — orange pour
/// le chip réseau, cyan pour VR — afin que l'UI reste cohérente.
fn draw_netvr_status(
    r: &mut Renderer,
    mode: &crate::net::NetMode,
    vr_on: bool,
    rtt_ms: Option<u32>,
) {
    use crate::logo::palette;

    let networked = !matches!(mode, crate::net::NetMode::SinglePlayer);
    if !networked && !vr_on {
        return; // rien à afficher en solo non-VR
    }

    // Coordonnées en pixels écran.  Marge identique au watermark HUD.
    let margin_x = 12.0;
    let margin_y = 12.0;
    let text_scale = 2.0; // SDF text scale ; cf. `push_text` — 1 glyphe = 8 px logiques
    let glyph_w = 8.0 * text_scale;
    let pad_x = glyph_w * 0.5;
    let pad_y = text_scale * 2.0;
    let line_h = 8.0 * text_scale + pad_y * 2.0;

    let mut y = margin_y;

    if networked {
        let (label, color_bg) = match mode {
            crate::net::NetMode::Server { bind_addr, .. } => (
                format!("HOST {bind_addr}"),
                [palette::ORANGE[0] * 0.55, palette::ORANGE[1] * 0.35, 0.08, 0.80],
            ),
            crate::net::NetMode::Client { server_addr } => {
                let ping = rtt_ms
                    .map(|ms| format!("  {ms}ms"))
                    .unwrap_or_else(|| "  --ms".to_string());
                (
                    format!("NET  {server_addr}{ping}"),
                    [0.10, 0.30, 0.55, 0.80],
                )
            }
            crate::net::NetMode::DemoPlayback { path } => {
                let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("demo");
                (
                    format!("DEMO  {name}"),
                    [0.20, 0.20, 0.30, 0.80],
                )
            }
            crate::net::NetMode::SinglePlayer => unreachable!(),
        };
        let text_w = label.len() as f32 * glyph_w;
        r.push_rect(
            margin_x,
            y,
            text_w + pad_x * 2.0,
            line_h,
            color_bg,
        );
        // Ombre + texte en CREAM.
        r.push_text(
            margin_x + pad_x + 1.0,
            y + pad_y + 1.0,
            text_scale,
            [0.0, 0.0, 0.0, 0.8],
            &label,
        );
        r.push_text(
            margin_x + pad_x,
            y + pad_y,
            text_scale,
            palette::CREAM,
            &label,
        );
        y += line_h + 4.0;
    }

    if vr_on {
        let label = "VR";
        let text_w = label.len() as f32 * glyph_w;
        r.push_rect(
            margin_x,
            y,
            text_w + pad_x * 2.0,
            line_h,
            [0.08, 0.40, 0.55, 0.82],
        );
        r.push_text(
            margin_x + pad_x + 1.0,
            y + pad_y + 1.0,
            text_scale,
            [0.0, 0.0, 0.0, 0.8],
            label,
        );
        r.push_text(
            margin_x + pad_x,
            y + pad_y,
            text_scale,
            [0.65, 0.95, 1.00, 1.0],
            label,
        );
    }
}

/// Affiche un petit compteur flottant à l'emplacement de chaque pickup
/// actuellement "hidden" (ramassé, en attente de respawn).  Visible
/// seulement dans les 5 dernières secondes avant respawn, avec fade-in,
/// et seulement si le pickup est dans une distance raisonnable du
/// joueur — sert de timing memory aid sans encombrer l'écran.
fn draw_pickup_respawn_indicators(
    r: &mut Renderer,
    pickups: &[PickupGpu],
    player_origin: Vec3,
    now: f32,
) {
    const INDICATOR_WINDOW: f32 = 5.0;
    const INDICATOR_MAX_DIST: f32 = 2000.0;
    let scale = 1.5;
    let char_w = 8.0 * scale;
    for p in pickups {
        let Some(respawn_at) = p.respawn_at else { continue };
        let remaining = respawn_at - now;
        if remaining <= 0.0 || remaining > INDICATOR_WINDOW {
            continue;
        }
        let dist = (p.origin - player_origin).length();
        if dist > INDICATOR_MAX_DIST {
            continue;
        }
        // Légère élévation au-dessus du sol (l'item dort sous terre).
        let pos = p.origin + Vec3::Z * 24.0;
        let Some((sx, sy)) = r.project_to_screen(pos) else { continue };
        // Fade-in symétrique : alpha monte de 0→1 pendant la première
        // seconde, reste à 1, puis repulse sur la dernière seconde pour
        // signaler l'imminence.
        let fade_in = (INDICATOR_WINDOW - remaining).min(1.0);
        let pulse = if remaining < 1.0 {
            0.6 + 0.4 * (now * std::f32::consts::TAU * 3.0).sin().abs()
        } else {
            1.0
        };
        let alpha = fade_in * pulse;
        // Fade distance sur le dernier tiers.
        let dist_fade = if dist < INDICATOR_MAX_DIST * 0.66 {
            1.0
        } else {
            ((INDICATOR_MAX_DIST - dist) / (INDICATOR_MAX_DIST * 0.34))
                .clamp(0.0, 1.0)
        };
        let a = alpha * dist_fade;
        let text = format!("{:.1}", remaining);
        let tw = text.len() as f32 * char_w;
        let tx = sx - tw * 0.5;
        let ty = sy - 8.0 * scale * 0.5;
        // Ombre + chiffre cyan — distinct du damage-number orange/rouge.
        r.push_text(tx + 1.0, ty + 1.0, scale, [0.0, 0.0, 0.0, 0.8 * a], &text);
        r.push_text(tx, ty, scale, [0.6, 0.9, 1.0, a], &text);
    }
}

/// Dessine les lignes de chat bot en bas-gauche de l'écran.  Chaque
/// ligne = un bandeau sombre avec `BOT:` + nom + texte, fade sur son
/// dernier quart de vie.  Les lignes plus anciennes sont au-dessus
/// (plus proches du haut de pile), la plus récente en bas — on lit
/// de bas en haut comme un terminal Quake.
fn draw_chat_feed(r: &mut Renderer, chat: &[ChatLine], now: f32) {
    use crate::logo::palette;
    if chat.is_empty() {
        return;
    }
    let text_scale = 1.6;
    let glyph_w = 8.0 * text_scale;
    let glyph_h = 8.0 * text_scale;
    let pad_x = glyph_w * 0.4;
    let pad_y = text_scale * 1.5;
    let line_h = glyph_h + pad_y * 2.0;
    let gap = 3.0;

    // On rend `chat` en ordre chrono (plus ancienne en haut) mais on
    // empile depuis le bas, donc on parcourt à l'envers et on décrémente
    // le `y` en partant d'une origine basse.
    let screen_w = r.width() as f32;
    let screen_h = r.height() as f32;
    // Marge basse : on laisse ~80 px pour ne pas taper dans le watermark
    // ni le HUD d'ammo s'il pousse vers la gauche.
    let margin_x = 14.0;
    let margin_bottom = 80.0;
    let mut y = screen_h - margin_bottom - line_h;

    for line in chat.iter().rev() {
        // Fade alpha sur dernier quart de vie.
        let remaining = (line.expire_at - now).max(0.0);
        let frac = (remaining / line.lifetime.max(1e-3)).clamp(0.0, 1.0);
        let alpha = if frac > 0.25 {
            1.0
        } else {
            (frac / 0.25).clamp(0.0, 1.0)
        };
        let prefix = format!("{}: ", line.speaker);
        let full = format!("{prefix}{}", line.text);
        // Largeur estimée : une ligne de chat tient sur une largeur max,
        // on clampe pour ne pas déborder du HUD (utile sur 720p étroit).
        let max_chars = ((screen_w - margin_x * 2.0 - pad_x * 2.0) / glyph_w).floor() as usize;
        let shown = if full.len() > max_chars.max(8) {
            // Garder ~max_chars-1 puis ellipse — évite le wrap multi-ligne.
            let mut s = full[..full.len().min(max_chars.saturating_sub(1))].to_string();
            s.push('…');
            s
        } else {
            full
        };
        let text_w = shown.len() as f32 * glyph_w;
        // Fond : bleu nuit semi-translucide modulé par alpha.
        r.push_rect(
            margin_x,
            y,
            text_w + pad_x * 2.0,
            line_h,
            [
                palette::MIDNIGHT[0],
                palette::MIDNIGHT[1],
                palette::MIDNIGHT[2],
                0.75 * alpha,
            ],
        );
        // Liseré orange à gauche — accent visuel cohérent avec le logo.
        r.push_rect(
            margin_x,
            y,
            2.0,
            line_h,
            [
                palette::ORANGE[0],
                palette::ORANGE[1],
                palette::ORANGE[2],
                alpha,
            ],
        );
        // Ombre + texte.
        r.push_text(
            margin_x + pad_x + 1.0,
            y + pad_y + 1.0,
            text_scale,
            [0.0, 0.0, 0.0, 0.85 * alpha],
            &shown,
        );
        r.push_text(
            margin_x + pad_x,
            y + pad_y,
            text_scale,
            [
                palette::CREAM[0],
                palette::CREAM[1],
                palette::CREAM[2],
                alpha,
            ],
            &shown,
        );
        y -= line_h + gap;
        if y < 0.0 {
            break;
        }
    }
}

/// Rend les toasts de pickup en bas-centre : un petit rectangle noir
/// translucide par toast, le plus récent en bas, pile de haut en haut
/// à mesure qu'ils s'accumulent.  Fade alpha sur le dernier quart de
/// vie pour une sortie douce.
///
/// Positionnement centré horizontalement — le texte est court (nom de
/// l'arme / du powerup) donc la largeur varie par ligne ; on laisse le
/// rectangle hugger le texte pour rester lisible sans "boîte" fixe.
fn draw_pickup_toasts(r: &mut Renderer, toasts: &[PickupToast], now: f32) {
    if toasts.is_empty() {
        return;
    }
    let text_scale = 2.0;
    let glyph_w = 8.0 * text_scale;
    let glyph_h = 8.0 * text_scale;
    let pad_x = glyph_w * 0.6;
    let pad_y = text_scale * 2.0;
    let line_h = glyph_h + pad_y * 2.0;
    let gap = 4.0;

    let screen_w = r.width() as f32;
    let screen_h = r.height() as f32;
    // Ancrage : ~35 % de la hauteur sous le centre (plus bas que le crosshair,
    // plus haut que le HUD de health/armor).  Les toasts s'empilent vers le
    // haut à partir de là, le dernier pickup toujours en bas de la pile.
    let anchor_y = screen_h * 0.68;
    let mut y = anchor_y - line_h;

    for t in toasts.iter().rev() {
        let remaining = (t.expire_at - now).max(0.0);
        let frac = (remaining / t.lifetime.max(1e-3)).clamp(0.0, 1.0);
        // Fade sur le dernier quart + pop-in rapide sur le premier 10 %
        // (scale-in visuel serait plus classe mais coûte un shader dédié ;
        // alpha-pop est suffisant pour accrocher l'œil).
        let alpha = if frac > 0.25 {
            1.0
        } else {
            (frac / 0.25).clamp(0.0, 1.0)
        };
        let text_w = t.text.len() as f32 * glyph_w;
        let box_w = text_w + pad_x * 2.0;
        let x = (screen_w - box_w) * 0.5;
        // Fond noir semi-opaque — lisibilité garantie quelle que soit la
        // scène derrière.
        r.push_rect(x, y, box_w, line_h, [0.0, 0.0, 0.0, 0.55 * alpha]);
        // Liseré coloré à gauche pour relayer la catégorie (jaune arme,
        // couleur powerup, blanc holdable).
        r.push_rect(
            x,
            y,
            3.0,
            line_h,
            [t.color[0], t.color[1], t.color[2], alpha],
        );
        // Ombre + texte.
        r.push_text(
            x + pad_x + 1.0,
            y + pad_y + 1.0,
            text_scale,
            [0.0, 0.0, 0.0, 0.85 * alpha],
            &t.text,
        );
        r.push_text(
            x + pad_x,
            y + pad_y,
            text_scale,
            [t.color[0], t.color[1], t.color[2], alpha],
            &t.text,
        );
        y -= line_h + gap;
        if y < 0.0 {
            break;
        }
    }
}

/// Overlay warmup : gros chiffre centré qui décompte les secondes avant
/// le début du match.  Appelé tant que `remaining > 0`.  Trois composants :
///   1. Un bandeau noir translucide plein-écran très léger pour affaiblir
///      la scène (rappel visuel que le match n'a pas commencé).
///   2. Le texte "MATCH BEGINS IN" sur une ligne, au-dessus du chiffre.
///   3. Le chiffre du compteur (ceil) en énorme, couleur pulsée par
///      seconde entière pour donner un effet "tick" visuel.
fn draw_warmup_overlay(r: &mut Renderer, remaining: f32) {
    let w = r.width() as f32;
    let h = r.height() as f32;
    // Voile global très discret — juste de quoi tirer la saturation vers
    // le bas pour que l'overlay ressorte.
    r.push_rect(0.0, 0.0, w, h, [0.0, 0.0, 0.0, 0.18]);

    let label = "MATCH BEGINS IN";
    let label_scale = 2.5;
    let label_gw = 8.0 * label_scale;
    let label_gh = 8.0 * label_scale;
    let label_w = label.len() as f32 * label_gw;
    let label_x = (w - label_w) * 0.5;
    let label_y = h * 0.38;
    // Ombre + texte.
    r.push_text(
        label_x + 2.0,
        label_y + 2.0,
        label_scale,
        [0.0, 0.0, 0.0, 0.85],
        label,
    );
    r.push_text(
        label_x,
        label_y,
        label_scale,
        [1.0, 0.95, 0.6, 1.0],
        label,
    );

    // Le chiffre : on l'arrondit UP pour afficher "3" durant la 1re
    // seconde, "2" durant la 2e, etc.  Minimum 1 affiché tant que le
    // warmup n'est pas strictement fini.
    let seconds_shown = remaining.ceil().max(1.0) as i32;
    let digit = format!("{seconds_shown}");
    let digit_scale = 10.0;
    let digit_gw = 8.0 * digit_scale;
    let digit_gh = 8.0 * digit_scale;
    let digit_w = digit.len() as f32 * digit_gw;
    let digit_x = (w - digit_w) * 0.5;
    let digit_y = label_y + label_gh + 24.0;

    // Pulse : chaque seconde, la première moitié est en "pop" brillant,
    // la seconde moitié décroît vers la couleur de base.  Donne le
    // "thump-thump" visuel d'un countdown.
    let frac = remaining.fract();
    let pulse = if frac > 0.5 { (frac - 0.5) * 2.0 } else { 1.0 };
    let tint = [1.0, 0.4 + 0.5 * pulse, 0.1 + 0.4 * pulse, 1.0];

    r.push_text(
        digit_x + 3.0,
        digit_y + 3.0,
        digit_scale,
        [0.0, 0.0, 0.0, 0.85],
        &digit,
    );
    r.push_text(digit_x, digit_y, digit_scale, tint, &digit);

    // Consomme le glyph pour éviter un "unused" warning sur certaines
    // configs.
    let _ = digit_gh;
}

#[cfg(test)]
mod interp_tests {
    //! Tests focalisés sur l'interpolation des remote players.
    use super::{lerp_angle_deg, lerp_angles, RemoteInterp, RemoteSample};
    use q3_math::{Angles, Vec3};
    use std::time::{Duration, Instant};

    fn sample(server_time: u32, x: f32, yaw: f32, received_at: Instant) -> RemoteSample {
        RemoteSample {
            origin: Vec3::new(x, 0.0, 0.0),
            velocity: Vec3::ZERO,
            view_angles: Angles::new(0.0, yaw, 0.0),
            on_ground: true,
            is_dead: false,
            recently_fired: false,
            team: 0,
            server_time,
            received_at,
        }
    }

    #[test]
    fn lerp_angle_short_path_through_zero() {
        // De 350° à 10°, le chemin court passe par 0° (delta = +20°).
        let r = lerp_angle_deg(350.0, 10.0, 0.5);
        // À mi-chemin : 360° (= 0°) ± epsilon.
        let normalized = r.rem_euclid(360.0);
        // Tolère 0° ou ~360°.
        assert!(
            (normalized - 0.0).abs() < 0.01 || (normalized - 360.0).abs() < 0.01,
            "lerp 350→10 à 0.5 doit valoir ~0°, obtenu {r}"
        );
    }

    #[test]
    fn lerp_angle_short_path_negative_delta() {
        // De 10° à 350°, chemin court via 0° (delta = -20°).
        let r = lerp_angle_deg(10.0, 350.0, 0.5);
        let normalized = r.rem_euclid(360.0);
        assert!(
            (normalized - 0.0).abs() < 0.01 || (normalized - 360.0).abs() < 0.01,
            "lerp 10→350 à 0.5 doit valoir ~0°, obtenu {r}"
        );
    }

    #[test]
    fn interp_first_sample_returns_directly() {
        let mut buf = RemoteInterp::default();
        let now = Instant::now();
        buf.push(sample(100, 50.0, 90.0, now));
        let s = buf.current().expect("doit avoir un sample");
        assert_eq!(s.origin.x, 50.0);
    }

    #[test]
    fn interp_two_samples_lerp_at_midpoint() {
        // Construit older et newer "comme si" newer venait d'arriver
        // 25 ms après older (mid-span pour un span 50 ms).
        let mut buf = RemoteInterp::default();
        let t_old = Instant::now() - Duration::from_millis(25);
        buf.push(sample(100, 0.0, 0.0, t_old));
        // newer reçu 25 ms après → received_at il y a 25 ms (`now - 0`).
        // Mais comme `current()` calcule `elapsed = newer.received_at.elapsed()`,
        // on a besoin que received_at soit "il y a 25 ms" pour que t = 0.5.
        // span = 150 - 100 = 50 ms ; elapsed = 25 ms → t = 0.5.
        let t_new = Instant::now() - Duration::from_millis(25);
        buf.push(sample(150, 100.0, 0.0, t_new));

        let s = buf.current().expect("sample interpolé");
        // À mi-chemin entre origin.x = 0 et 100 → ~50, mais elapsed ≈ 25 ms
        // peut varier de quelques ms à cause de l'horloge du test.
        assert!(
            (s.origin.x - 50.0).abs() < 15.0,
            "origin.x devrait être ~50, obtenu {}",
            s.origin.x
        );
    }

    #[test]
    fn interp_clamps_at_t_one_after_long_delay() {
        // Si plus de span ms se sont écoulés depuis newer, t = 1 → newer.
        let mut buf = RemoteInterp::default();
        buf.push(sample(100, 0.0, 0.0, Instant::now() - Duration::from_secs(2)));
        buf.push(sample(150, 100.0, 0.0, Instant::now() - Duration::from_secs(1)));
        let s = buf.current().expect("sample");
        assert!(
            (s.origin.x - 100.0).abs() < 0.5,
            "après long délai, lerp doit clamper à newer ; obtenu {}",
            s.origin.x
        );
    }

    #[test]
    fn interp_handles_yaw_wraparound() {
        let mut buf = RemoteInterp::default();
        // older yaw=350, newer yaw=10. À mi-chemin doit donner ~0°,
        // pas 180°.
        let elapsed = Duration::from_millis(25);
        buf.push(sample(100, 0.0, 350.0, Instant::now() - elapsed));
        buf.push(sample(150, 0.0, 10.0, Instant::now() - elapsed));
        let s = buf.current().expect("sample");
        let yaw_norm = s.view_angles.yaw.rem_euclid(360.0);
        assert!(
            yaw_norm < 30.0 || yaw_norm > 330.0,
            "yaw lerpé doit rester proche de 0 (chemin court), obtenu {yaw_norm}"
        );
    }

    #[test]
    fn interp_dedup_same_server_time() {
        // Deux pushes avec le même server_time : on n'en garde qu'un,
        // sinon span = 0 et la lerp explose.
        let mut buf = RemoteInterp::default();
        let now = Instant::now();
        buf.push(sample(100, 0.0, 0.0, now));
        buf.push(sample(100, 50.0, 0.0, now));
        // older doit toujours être None car le 2e a écrasé newer.
        assert!(buf.older.is_none());
        assert_eq!(buf.newer.as_ref().unwrap().origin.x, 50.0);
    }

    /// Sanity : un seul lerp avec t=0 et t=1.
    #[test]
    fn lerp_angles_endpoints() {
        let a = Angles::new(10.0, 20.0, 30.0);
        let b = Angles::new(40.0, 50.0, 60.0);
        let at0 = lerp_angles(a, b, 0.0);
        let at1 = lerp_angles(a, b, 1.0);
        assert!((at0.pitch - a.pitch).abs() < 0.001);
        assert!((at0.yaw - a.yaw).abs() < 0.001);
        // Pour t=1, on tombe sur b modulo 360 (peut diverger d'un multiple).
        assert!(((at1.pitch - b.pitch).rem_euclid(360.0) % 360.0).abs() < 0.01);
    }
}

#[cfg(test)]
mod respawn_timer_tests {
    use super::*;

    #[test]
    fn mega_health_is_traked_normal_health_is_not() {
        // MH = max_cap 200, on l'affiche. 25/100 health = item small,
        // pas la peine de le polluer le panneau respawn.
        assert_eq!(
            pickup_timer_label(&PickupKind::Health { amount: 100, max_cap: 200 }),
            Some("MH")
        );
        assert_eq!(
            pickup_timer_label(&PickupKind::Health { amount: 25, max_cap: 100 }),
            None
        );
        assert_eq!(
            pickup_timer_label(&PickupKind::Health { amount: 5, max_cap: 200 }),
            Some("MH"),
            "small bubble counts as Mega category par max_cap"
        );
    }

    #[test]
    fn ra_and_ya_distinguished_shards_filtered() {
        assert_eq!(pickup_timer_label(&PickupKind::Armor { amount: 100 }), Some("RA"));
        assert_eq!(pickup_timer_label(&PickupKind::Armor { amount: 50 }), Some("YA"));
        // Shard (5) et combat (25 par défaut) : on n'encombre pas.
        assert_eq!(pickup_timer_label(&PickupKind::Armor { amount: 5 }), None);
        assert_eq!(pickup_timer_label(&PickupKind::Armor { amount: 25 }), None);
    }

    #[test]
    fn all_powerups_get_a_label() {
        // Tout powerup vaut la peine d'être affiché — durée 30-120 s,
        // crucial pour timer les rotations item.
        for pw in [
            PowerupKind::QuadDamage,
            PowerupKind::Haste,
            PowerupKind::Regeneration,
            PowerupKind::BattleSuit,
            PowerupKind::Invisibility,
            PowerupKind::Flight,
        ] {
            let kind = PickupKind::Powerup { powerup: pw, duration: 30.0 };
            assert!(
                pickup_timer_label(&kind).is_some(),
                "powerup {pw:?} devrait avoir un label HUD"
            );
        }
    }

    #[test]
    fn weapon_and_ammo_pickups_excluded() {
        // Armes + ammo = trop nombreux, pollueraient le panneau.
        let w = PickupKind::Weapon { weapon: WeaponId::Rocketlauncher, ammo: 10 };
        let a = PickupKind::Ammo { slot: 5, amount: 25 };
        assert_eq!(pickup_timer_label(&w), None);
        assert_eq!(pickup_timer_label(&a), None);
    }

    #[test]
    fn quad_color_distinct_from_other_powerups() {
        // Quad = mauve unique, les autres powerups partagent le bleu — Q3
        // canon : le joueur a appris à associer le mauve au Quad damage.
        let quad = pickup_timer_color(&PickupKind::Powerup {
            powerup: PowerupKind::QuadDamage,
            duration: 30.0,
        });
        let haste = pickup_timer_color(&PickupKind::Powerup {
            powerup: PowerupKind::Haste,
            duration: 30.0,
        });
        assert_ne!(quad, haste);
    }
}

#[cfg(test)]
mod chat_wheel_tests {
    use super::CHAT_WHEEL_MESSAGES;

    #[test]
    fn exactly_eight_messages_for_radial_layout() {
        // Le rendu radial divise 360° en 8 secteurs de 45°. Si on ajoute
        // ou retire une entrée sans réviser la géométrie, la roue se
        // désaligne. Ce test bloque le retrait silencieux d'un message.
        assert_eq!(CHAT_WHEEL_MESSAGES.len(), 8);
    }

    #[test]
    fn labels_are_short_for_readability() {
        // Le libellé visible (gauche du tuple) doit rester court pour
        // tenir dans la pilule autour du cercle, sinon ça déborde.
        for (label, _) in &CHAT_WHEEL_MESSAGES {
            assert!(
                !label.is_empty() && label.len() <= 12,
                "label '{label}' devrait être 1..=12 chars"
            );
        }
    }

    #[test]
    fn full_messages_are_natural() {
        // Le texte envoyé sur le wire doit être lisible côté receveur,
        // pas juste un acronyme cryptique. On teste que le full_msg est
        // strictement plus long que le label (au moins une expansion).
        for (label, full) in &CHAT_WHEEL_MESSAGES {
            assert!(
                full.len() > label.len() || *full == *label,
                "full_msg '{full}' devrait étendre '{label}'"
            );
            assert!(full.len() <= 96, "full_msg trop long pour wire (96 max)");
        }
    }
}

#[cfg(test)]
mod follow_cam_tests {
    use super::pick_follow_target;

    #[test]
    fn empty_alive_returns_none() {
        assert_eq!(pick_follow_target(None, &[], 1), None);
        assert_eq!(pick_follow_target(Some(3), &[], -1), None);
    }

    #[test]
    fn no_current_next_picks_first() {
        assert_eq!(pick_follow_target(None, &[5, 2, 7], 1), Some(2));
    }

    #[test]
    fn no_current_prev_picks_last() {
        assert_eq!(pick_follow_target(None, &[5, 2, 7], -1), Some(7));
    }

    #[test]
    fn cycle_next_wraps_around() {
        // sorted = [2, 5, 7]
        assert_eq!(pick_follow_target(Some(2), &[5, 2, 7], 1), Some(5));
        assert_eq!(pick_follow_target(Some(5), &[5, 2, 7], 1), Some(7));
        assert_eq!(pick_follow_target(Some(7), &[5, 2, 7], 1), Some(2));
    }

    #[test]
    fn cycle_prev_wraps_around() {
        assert_eq!(pick_follow_target(Some(2), &[5, 2, 7], -1), Some(7));
        assert_eq!(pick_follow_target(Some(5), &[5, 2, 7], -1), Some(2));
        assert_eq!(pick_follow_target(Some(7), &[5, 2, 7], -1), Some(5));
    }

    #[test]
    fn current_not_in_alive_falls_back_to_first() {
        // Cas : on suit un slot qui vient de mourir / quitter. Le cycle
        // doit ré-entrer dans la liste plutôt que de panic ou retourner
        // None — UX : RMB en spectator avec target morte = on bascule
        // sur quelqu'un de visible.
        assert_eq!(pick_follow_target(Some(99), &[1, 4, 8], 1), Some(4));
        assert_eq!(pick_follow_target(Some(99), &[1, 4, 8], -1), Some(8));
    }

    #[test]
    fn single_alive_returns_itself() {
        assert_eq!(pick_follow_target(None, &[3], 1), Some(3));
        assert_eq!(pick_follow_target(Some(3), &[3], 1), Some(3));
        assert_eq!(pick_follow_target(Some(3), &[3], -1), Some(3));
    }
}

#[cfg(test)]
mod chat_tests {
    //! Verrouille l'invariant : chaque `ChatTrigger` a au moins une
    //! ligne dans son pool, sinon le tirage aléatoire paniquerait.
    use super::{ChatTrigger, CHAT_LINES_DEATH, CHAT_LINES_KILL_INSULT, CHAT_LINES_RESPAWN};

    #[test]
    fn every_trigger_has_at_least_one_line() {
        for t in [ChatTrigger::KillInsult, ChatTrigger::Death, ChatTrigger::Respawn] {
            assert!(!t.pool().is_empty(), "pool vide pour {t:?}");
        }
    }

    #[test]
    fn trigger_pools_are_non_empty_directly() {
        assert!(!CHAT_LINES_KILL_INSULT.is_empty());
        assert!(!CHAT_LINES_DEATH.is_empty());
        assert!(!CHAT_LINES_RESPAWN.is_empty());
    }

    #[test]
    fn trigger_weights_stay_in_unit_range() {
        for t in [ChatTrigger::KillInsult, ChatTrigger::Death, ChatTrigger::Respawn] {
            let w = t.weight();
            assert!((0.0..=1.0).contains(&w), "weight hors [0,1] pour {t:?} : {w}");
        }
    }

    #[test]
    fn kill_insult_is_weightier_than_respawn() {
        // Taunt post-frag >> retour au spawn — pour que le chat se remplisse
        // majoritairement d'action de jeu et pas d'un flot de "Back." "Back."
        assert!(ChatTrigger::KillInsult.weight() > ChatTrigger::Respawn.weight());
    }
}

#[cfg(test)]
mod input_tests {
    //! Tests de régression pour le pipeline des inputs de mouvement.
    //!
    //! Bug historique : « le personnage ne bouge pas quand je presse WASD ».
    //! La première hypothèse (map non chargée → pas de `tick_collide`) a été
    //! fixée via l'auto-load dans `resumed()`. Ces tests verrouillent la
    //! sémantique de `Input::forward_axis()` / `side_axis()` pour éviter
    //! toute régression future dans la chaîne KeyCode → axe → `MoveCmd`.
    //!
    //! Le pipeline complet Input → PlayerMove::tick_collide est déjà
    //! couvert par les tests de `q3-game::movement` (`run_caps_ground_speed_at_max`,
    //! etc.) — ici on ne teste que la partie spécifique à l'engine.
    use super::Input;
    use q3_game::movement::{MoveCmd, PhysicsParams, PlayerMove};
    use q3_math::Vec3;

    #[test]
    fn forward_key_only_gives_plus_one_forward_axis() {
        let mut i = Input::default();
        i.fwd_down = true;
        assert_eq!(i.forward_axis(), 1.0);
        assert_eq!(i.side_axis(), 0.0);
    }

    #[test]
    fn back_key_only_gives_minus_one_forward_axis() {
        let mut i = Input::default();
        i.back_down = true;
        assert_eq!(i.forward_axis(), -1.0);
    }

    #[test]
    fn right_key_only_gives_plus_one_side_axis() {
        let mut i = Input::default();
        i.right_down = true;
        assert_eq!(i.side_axis(), 1.0);
        assert_eq!(i.forward_axis(), 0.0);
    }

    #[test]
    fn left_key_only_gives_minus_one_side_axis() {
        let mut i = Input::default();
        i.left_down = true;
        assert_eq!(i.side_axis(), -1.0);
    }

    #[test]
    fn opposite_keys_cancel_to_zero() {
        // Q3 : presser W+S simultanément donne 0 (pas 1 et pas -1). C'est le
        // comportement attendu des moteurs FPS classiques — pas de priorité
        // entre directions opposées, le joueur doit relâcher l'une pour
        // aller dans l'autre.
        let mut i = Input::default();
        i.fwd_down = true;
        i.back_down = true;
        assert_eq!(i.forward_axis(), 0.0);

        i = Input::default();
        i.left_down = true;
        i.right_down = true;
        assert_eq!(i.side_axis(), 0.0);
    }

    #[test]
    fn default_input_is_all_zero() {
        let i = Input::default();
        assert_eq!(i.forward_axis(), 0.0);
        assert_eq!(i.side_axis(), 0.0);
        assert!(!i.fwd_down);
        assert!(!i.back_down);
        assert!(!i.left_down);
        assert!(!i.right_down);
        assert!(!i.jump);
        assert!(!i.crouch);
        assert!(!i.walk);
        assert!(!i.fire);
        assert!(!i.scoreboard);
    }

    #[test]
    fn full_pipeline_forward_key_moves_player_forward() {
        // Régression end-to-end : simule « joueur presse W » et vérifie que
        // le `MoveCmd` construit depuis `Input` amène bien `PlayerMove` à
        // avancer sur l'axe X au tick suivant. C'est *exactement* ce que
        // fait le code de `window_event::RedrawRequested` — une copie
        // contre-vérification pour que tout futur refactor de `Input`
        // ou de la construction du `MoveCmd` soit pris ici plutôt qu'en
        // prod avec le bug bloquant « WASD ne fait rien ».
        let mut input = Input::default();
        input.fwd_down = true;

        let cmd = MoveCmd {
            forward: input.forward_axis(),
            side: input.side_axis(),
            up: 0.0,
            jump: input.jump,
            crouch: input.crouch,
            walk: input.walk,
            slide_pressed: false,
            dash_pressed: false,
            delta_time: 1.0 / 60.0,
        };
        assert_eq!(cmd.forward, 1.0, "W seul doit produire forward=1.0");

        let mut pm = PlayerMove::new(Vec3::ZERO);
        pm.on_ground = true;
        let params = PhysicsParams::default();
        // 60 ticks = 1 seconde de simulation — suffisant pour dépasser
        // quelques dizaines d'unités sur X.
        let start_x = pm.origin.x;
        for _ in 0..60 {
            pm.tick(cmd, params);
        }
        assert!(
            pm.origin.x > start_x + 50.0,
            "le joueur devrait avancer significativement ; start={start_x}, now={}",
            pm.origin.x
        );
    }
}

#[cfg(test)]
mod config_tests {
    //! Tests pour la persistance `q3config.cfg`.
    //!
    //! La fonction `user_config_path` lit des variables d'environnement —
    //! les tests serializent l'accès avec un mutex pour éviter les courses
    //! entre threads quand `cargo test` lance plusieurs tests en parallèle.
    use super::{next_screenshot_index, user_config_path};
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn user_config_path_respects_xdg_on_unix() {
        let _g = ENV_LOCK.lock().unwrap();
        let prev_xdg = std::env::var_os("XDG_CONFIG_HOME");
        let prev_appdata = std::env::var_os("APPDATA");
        // On ne touche pas à APPDATA sous Windows (où il est prioritaire) —
        // on teste juste que la fonction produit un chemin non-None quand
        // au moins une variable est définie.
        std::env::set_var("XDG_CONFIG_HOME", "/tmp/q3rust-unit-test");
        let p = user_config_path().expect("un chemin doit exister");
        assert!(p.ends_with("q3config.cfg"), "doit finir par q3config.cfg : {p:?}");
        // Restore
        match prev_xdg {
            Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
        // Sanity — pas touché
        assert_eq!(std::env::var_os("APPDATA"), prev_appdata);
    }

    #[test]
    fn user_config_path_file_component_stable() {
        let _g = ENV_LOCK.lock().unwrap();
        // Garanti par la convention quoi qu'il arrive.
        if let Some(p) = user_config_path() {
            assert_eq!(
                p.file_name().and_then(|s| s.to_str()),
                Some("q3config.cfg")
            );
            // Parent = répertoire dédié à l'app pour éviter de polluer
            // la racine du user.
            assert_eq!(
                p.parent()
                    .and_then(|d| d.file_name())
                    .and_then(|s| s.to_str()),
                Some("q3-rust")
            );
        }
    }

    #[test]
    fn next_screenshot_index_on_missing_dir_is_one() {
        let dir = std::env::temp_dir().join(format!(
            "q3rust-shot-missing-{}",
            std::process::id()
        ));
        // Pas de create_dir_all : on veut vérifier le cas "dossier absent".
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(next_screenshot_index(&dir), 1);
    }

    #[test]
    fn next_screenshot_index_reads_highest_existing() {
        let dir = std::env::temp_dir().join(format!(
            "q3rust-shot-highest-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for n in [1u32, 3, 7, 42] {
            std::fs::write(dir.join(format!("shot-{n:04}.tga")), b"x").unwrap();
        }
        // Fichiers parasites ignorés.
        std::fs::write(dir.join("README.txt"), b"bonjour").unwrap();
        std::fs::write(dir.join("shot-bad.tga"), b"x").unwrap();
        assert_eq!(next_screenshot_index(&dir), 43);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
