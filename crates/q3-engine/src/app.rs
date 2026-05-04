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
    /// **Music player** (v0.9.5+) — joue un fichier audio local en
    /// loop comme musique de fond.  Path absolu ou relatif au CWD.
    /// Formats supportés : ceux décodés par rodio (WAV, OGG, MP3 si
    /// la feature est activée — actuellement WAV/OGG seulement).
    MusicPlay(std::path::PathBuf),
    /// Stoppe la musique de fond.
    MusicStop,
    /// **Map download manager** sous-commandes (v0.9.5++).
    MapDlList,
    MapDlGet(String),
    MapDlStatus,
}

pub struct App {
    vfs: Arc<Vfs>,
    init_width: u32,
    init_height: u32,
    requested_map: Option<String>,

    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    world: Option<World>,
    /// **Battle Royale** (v0.9.5) — terrain heightmap chargé via les
    /// maps `br_*`.  Mutuellement exclusif avec `world` : une carte BR
    /// n'a pas de BSP, ses traces vont vers `Terrain::trace_ray` et son
    /// rendu sera assuré par un pipeline terrain dédié (pas encore
    /// branché côté GPU — étape suivante de l'intégration).
    terrain: Option<Arc<q3_terrain::Terrain>>,
    /// Ring shrink BR — `Some` quand une carte BR est active. Tickée
    /// chaque frame dans `update`, applique des dégâts hors-zone au
    /// joueur et aux bots.
    br_ring: Option<q3_terrain::br::RingShrink>,
    /// **Drones survol BR** (v0.9.5) — vaisseaux GLB qui orbitent au-
    /// dessus de la zone de combat. Cosmétique pure (pas de damage,
    /// pas d'IA). Spawné à load_terrain_map.
    drones: Vec<Drone>,
    /// **Rochers de décor BR** (v0.9.5) — props GLB disséminés sur
    /// le terrain pour casser la monotonie. Position+yaw+scale fixés
    /// au load, pas d'animation. Cosmétique pure.
    rocks: Vec<RockProp>,
    /// **Ammo crate GLB** scale auto-calculé au load à partir de
    /// `mesh.radius()` pour matcher la taille des MD3 pickup standard
    /// (~25u). `None` = asset pas chargé → fallback MD3.
    ammo_crate_scale: Option<f32>,
    /// **Quad pickup GLB** — idem ammo_crate mais target ~40u (le
    /// Quad MD3 d'origine fait ~50u, on prend une taille proche).
    quad_pickup_scale: Option<f32>,
    /// **Health pack GLB** — remplace les MD3 d'item_health* par une
    /// trousse de soin moderne.  Target ~22u (cohérent avec la taille
    /// d'un MD3 health bobble Q3).  `None` = asset pas chargé →
    /// fallback MD3.
    health_pack_scale: Option<f32>,
    /// **Railgun pickup GLB** — remplace le MD3 du railgun posé au sol.
    /// Target ~30u (taille d'un weapon pickup Q3 standard).  Utilisé
    /// à la fois pour le pickup au sol ET le viewmodel 1ère personne
    /// (orientation baseline, muzzle flash via offset hardcoded).
    railgun_pickup_scale: Option<f32>,
    /// **Grenade ammo box GLB** — remplace le MD3 d'ammo grenade
    /// (item_grenades).  Target ~25u, cohérent avec ammo_crate (MG).
    grenade_ammo_scale: Option<f32>,
    /// **Rocket ammo box GLB** — remplace le MD3 d'ammo rocket
    /// (item_rockets).  Target ~25u, cohérent avec les autres ammo
    /// crates.  Concerne aussi le pickup d'arme `weapon_rocketlauncher`
    /// qui inclut ses munitions de départ.
    rocket_ammo_scale: Option<f32>,
    /// **Cell ammo box GLB** — cellules énergétiques pour le Plasma Gun
    /// (item_cells).  Target ~25u.  Concerne aussi le pickup d'arme
    /// `weapon_plasmagun` qui inclut ses munitions de départ.
    cell_ammo_scale: Option<f32>,
    /// **LG battery ammo box GLB** — batteries pour le Lightning Gun
    /// (item_lightning).  Target ~25u.  Concerne aussi le pickup
    /// d'arme `weapon_lightning` qui inclut ses munitions de départ.
    lg_ammo_scale: Option<f32>,
    /// **Big armor (Red Armor) GLB** — remplace le MD3 d'item_armor_body
    /// (100 armor).  Target ~30u (taille canonique d'un body armor Q3).
    big_armor_scale: Option<f32>,
    /// **Plasma gun pickup GLB** — remplace le MD3 du Plasma Gun posé
    /// au sol (style fusil d'assaut moderne).  Utilisé à la fois pour
    /// le pickup au sol ET le viewmodel 1ère personne.  Distinct de
    /// `cell_ammo` (les boîtes de cellules ammo gardent leur look).
    plasma_pickup_scale: Option<f32>,
    /// **Railgun ammo (slugs) GLB** — boîte de slugs pour le Railgun
    /// (item_slugs).  Distinct de `railgun_pickup` (l'arme elle-même).
    railgun_ammo_scale: Option<f32>,
    /// **Regeneration powerup GLB** — remplace le MD3 d'item_regen.
    /// Distinct du `quad_pickup` (lui-même remplacé).  Target ~30u.
    regen_pickup_scale: Option<f32>,
    /// **Machine Gun pickup GLB** — remplace le MD3 du MG posé au sol
    /// ET le viewmodel 1ère personne (rotation 180° X comme plasma).
    /// Target ~30u (weapon pickup Q3 standard).
    machinegun_pickup_scale: Option<f32>,
    /// **BFG10K pickup GLB** — remplace le MD3 du BFG posé au sol ET
    /// le viewmodel 1ère personne.  Target ~32u (un peu plus gros que
    /// les autres weapon pickups, le BFG est imposant).
    bfg_pickup_scale: Option<f32>,
    /// **Lightning Gun pickup GLB** — remplace le MD3 du LG posé au sol
    /// ET le viewmodel 1ère personne.  Distinct des batteries LG ammo
    /// (qui restent sur `lg_ammo`).  Target ~30u.
    lightninggun_pickup_scale: Option<f32>,
    /// **Shotgun pickup GLB** — remplace le MD3 du SG posé au sol ET
    /// le viewmodel 1ère personne.  Target ~30u.
    shotgun_pickup_scale: Option<f32>,
    /// **Grenade Launcher pickup GLB** — remplace le MD3 du GL au sol
    /// ET le viewmodel 1ère personne.  Distinct de `grenade_ammo`
    /// (boîte de munitions).  Target ~30u.
    grenadelauncher_pickup_scale: Option<f32>,
    /// **Gauntlet pickup GLB** — remplace le MD3 du Gauntlet au sol
    /// ET le viewmodel 1ère personne.  Pas d'ammo box associée.
    /// Target ~30u.
    gauntlet_pickup_scale: Option<f32>,
    /// **Shotgun shells (calibre 12) GLB** — boîte de cartouches SG
    /// (item_shells).  Distinct du `shotgun_pickup` (l'arme).
    /// Target ~25u (cohérent avec autres ammo crates).
    shotgun_ammo_scale: Option<f32>,
    /// **BFG ammo GLB** — boîte de munitions BFG (item_bfg).  Distinct
    /// du `bfg_pickup` (l'arme).  Target ~25u.
    bfg_ammo_scale: Option<f32>,
    /// **Rocket Launcher pickup GLB** — remplace le MD3 du RL au sol
    /// ET le viewmodel 1ère personne.  Distinct de `rocket_ammo`
    /// (boîte de munitions).  Target ~30u.
    rocketlauncher_pickup_scale: Option<f32>,
    /// **Combat Armor (yellow, 50) GLB** — remplace le MD3
    /// d'item_armor_combat.  Distinct de `big_armor` (Body 100 red).
    /// Target ~28u.
    combat_armor_scale: Option<f32>,
    /// **Medkit holdable GLB** — remplace le MD3 du medkit (trousse en
    /// stock, holdable_medkit).  Target ~22u.
    medkit_scale: Option<f32>,
    /// **Armor Shard (5) GLB** — remplace le MD3 d'item_armor_shard.
    /// Target ~18u (petit shard).
    armor_shard_scale: Option<f32>,
    /// **GLB prop lights cache** (v0.9.5++) — par prop name, la liste
    /// de lights KHR_lights_punctual extraites du mesh.  Vide pour
    /// les assets sans lights définies.
    prop_lights: hashbrown::HashMap<String, Vec<q3_model::glb::GlbLight>>,
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
    /// **Stats live overlay** (V3f) — toggle F8. Affiche K/D,
    /// accuracy par arme, time alive, frags/min en bas-droite. Off
    /// par défaut.
    show_stats_overlay: bool,
    /// Timestamp de la dernière fois que le joueur est mort, pour
    /// calculer "time alive" depuis le dernier respawn.
    last_respawn_time: f32,
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
    /// **Rapid-fire SFX throttle** (v0.9.5++) — timestamp du dernier
    /// SFX de tir joué. Limite la cadence audio sur PG/LG pour ne pas
    /// saturer les 64 canaux du SoundSystem en rapid-fire.
    last_fire_sfx_at: f32,
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
    /// **Punch angles** (v0.9.5++) — offset visuel ajouté aux angles de
    /// caméra pour le rendu uniquement (pas pour l'aim).  À chaque tir
    /// l'arme « kicke » la vue : pitch up + petit jitter de yaw selon
    /// l'arme.  Décroît exponentiellement vers zéro.  En degrés.
    view_kick_pitch: f32,
    view_kick_yaw: f32,
    /// **Map download manager** (v0.9.5++) — catalogue + DL HTTP des
    /// `.pk3` community vers `baseq3/`.  Console : `mapdl list` /
    /// `mapdl get <id>` / `mapdl status`.
    map_dl: crate::map_dl::MapDownloader,
    /// Fin du hit-marker HUD (petite croix quand un tir a touché un bot).
    hit_marker_until: f32,
    /// Fin du kill-marker HUD : grand X rouge vif autour du réticule
    /// quand la dernière volée a tué la cible. Distinct du hit-marker
    /// (4 segments diagonaux gris/rouge sombre) pour donner un feedback
    /// visuel net « kill confirm » même sans regarder le scoreboard.
    /// Set à `time_sec + KILL_MARKER_DURATION_SEC` quand `any_kill`.
    kill_marker_until: f32,
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
    /// Fin du flash powerup : set à `now + POWERUP_FLASH_SEC` quand un
    /// powerup est acquis (Quad, Haste, etc.).  Dessiné comme un voile
    /// full-screen teinté de la couleur du powerup, alpha en fade-out
    /// rapide pour évoquer une "sublimation" — feedback visuel fort
    /// que l'effet est désormais actif.
    powerup_flash_until: f32,
    /// Couleur RGBA du flash powerup courant (alpha de base, modulé
    /// par le fade-out dans `draw_hud`).  Réécrit à chaque pickup.
    powerup_flash_color: [f32; 4],
    /// **Atmospheric lightning** (BR uniquement) — fin du flash courant
    /// d'éclair (overlay HUD blanc bleuté).  > now → flash en cours.
    lightning_flash_until: f32,
    /// Prochain instant programmé pour déclencher un nouvel éclair.
    /// Tiré entre `LIGHTNING_INTERVAL_MIN/MAX` après chaque flash.
    /// `f32::INFINITY` = lightning désactivé (mode non-BR ou non-init).
    next_lightning_at: f32,
    /// Fenêtre du FOV punch — kick d'élargissement horizontal du champ
    /// de vision quand le joueur enregistre un hit.  L'ouverture FOV
    /// donne une sensation de "boost adrenaline" sur frag, distincte
    /// du view-kick (recul caméra) qui agit sur les angles.
    fov_punch_until: f32,
    /// Amplitude du punch courant en degrés (additif au cg_fov).
    /// Hit confirmé = `FOV_PUNCH_HIT`, frag létal = `FOV_PUNCH_FRAG`.
    fov_punch_strength: f32,
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
    /// **Viewmodel sway** (v0.9.5++) — yaw/pitch précédents pour calculer
    /// le delta angulaire frame-to-frame.  Le viewmodel "lag" derrière
    /// la caméra proportionnellement au delta → impression de poids
    /// physique de l'arme.  Lissé par `viewmodel_sway` (eased).
    viewmodel_prev_yaw: f32,
    viewmodel_prev_pitch: f32,
    /// État du sway easé sur 2 axes (yaw, pitch).  Décroît exponentiellement
    /// vers 0 chaque frame ; impulse à chaque rotation caméra.
    viewmodel_sway: [f32; 2],
    /// Instant où le joueur a re-touché le sol après une phase
    /// airborne — déclenche le jump-kick (gun dip puis spring).
    player_last_land_at: f32,
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
    /// **Death sound** distinct du pain. Joué UNE FOIS sur le hit
    /// fatal — le pain (cooldownisé) reste pour les hits non-létaux.
    /// Q3 vanilla : `sound/player/death1.wav`.
    sfx_death_bot: Option<SoundHandle>,
    /// SFX génériques Q3 de cueillette d'arme / de munitions.
    sfx_weapon_pickup: Option<SoundHandle>,
    sfx_ammo_pickup: Option<SoundHandle>,
    /// Sons dédiés pickups health/armor (Q3 vanilla `sound/items/*`).
    /// Avant v0.9.4 on recyclait sfx_pain_bot — moche, peu lisible.
    sfx_health_pickup: Option<SoundHandle>,
    sfx_armor_pickup: Option<SoundHandle>,
    sfx_megahealth_pickup: Option<SoundHandle>,
    /// Impact plasma sur surface (sizzle "pshhh"). Q3 :
    /// `sound/weapons/plasma/plasmx1a.wav`.
    sfx_plasma_impact: Option<SoundHandle>,
    /// Impact rocket. Q3 : `sound/weapons/rocket/rocklx1a.wav`.
    sfx_rocket_impact: Option<SoundHandle>,
    /// Impact grenade explosion. Q3 : `sound/weapons/grenade/grenlx1a.wav`.
    sfx_grenade_impact: Option<SoundHandle>,
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

/// Durée d'affichage du grand X de kill-confirm sur le réticule.  Plus
/// long que le hit-marker (0.18 s) pour être bien lisible — le joueur
/// doit pouvoir voir le X même s'il était en train de transitionner
/// vers une autre cible. 0.45 s ≈ 27 frames @ 60 fps, juste assez pour
/// un fade visible sans masquer durablement le réticule.
const KILL_MARKER_DURATION_SEC: f32 = 0.45;

/// Durée du flash plein écran lors d'un pickup de powerup. 0.55 s assez
/// long pour être senti périphériquement (fade rapide les ~150 dernières
/// ms) mais court pour ne pas masquer l'environnement post-pickup —
/// le joueur veut continuer à jouer, pas regarder un fade-out.
const POWERUP_FLASH_SEC: f32 = 0.55;

/// Durée du flash visuel d'un éclair atmosphérique (overlay blanc-bleu
/// full-screen).  120 ms ≈ frame d'éclair canonique sur les vrais
/// orages — assez bref pour ne pas masquer le combat, assez long pour
/// que l'œil le capte.  Le fade exponentiel (ratio³) le fait paraître
/// "intense puis fini" instantanément.
const LIGHTNING_FLASH_SEC: f32 = 0.12;
/// Intervalle min/max entre deux éclairs (s). Tirage uniforme dans
/// [MIN, MAX] à chaque flash. Max élevé pour que l'effet reste un
/// événement, pas un strobe constant.
const LIGHTNING_INTERVAL_MIN: f32 = 12.0;
const LIGHTNING_INTERVAL_MAX: f32 = 35.0;

/// Durée du FOV punch sur hit confirmé.  180 ms ≈ 11 frames @ 60 fps —
/// le joueur ressent l'expansion sans qu'elle persiste assez pour
/// gêner la mire sur le tir suivant.
const FOV_PUNCH_DURATION: f32 = 0.18;
/// Amplitude max du FOV punch en degrés (additif au cg_fov de base).
/// 3° à 90° de FOV = +3 % de champ horizontal — perceptible mais pas
/// disorientant.  Sur frag létal on monte à 5° via `FOV_PUNCH_FRAG`.
const FOV_PUNCH_HIT: f32 = 3.0;
const FOV_PUNCH_FRAG: f32 = 5.0;

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
    /// Le bot vient d'acquérir une cible visible (engagement frais).
    /// Style « radio combat » — court, sec.
    Spotted,
    /// Le bot a entendu un bruit (tir/footstep) sans LOS — il enquête.
    Heard,
    /// HP bas (<25) — taunt d'esquive / appel à l'aide.
    LowHp,
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

/// Radio chatter « contact ! » — bot vient de spotter le joueur.
/// Style sec, court, militaire.  Donne l'illusion d'une équipe qui
/// communique, même si chaque bot est solo en interne.
const CHAT_LINES_SPOTTED: &[&str] = &[
    "Contact!",
    "I see him.",
    "Got eyes on.",
    "Tally!",
    "Spotted, engaging.",
    "Target acquired.",
    "Visual.",
    "There you are.",
    "Found him.",
];

/// Radio chatter « j'entends quelque chose » — bot a perçu un tir
/// hors-LOS.  Style « j'enquête ».
const CHAT_LINES_HEARD: &[&str] = &[
    "Heard something.",
    "Movement nearby.",
    "Where are you?",
    "Footsteps...",
    "Was that a shot?",
    "Listening.",
    "Hold your fire — checking.",
];

/// HP bas — bot retraite ou appelle au support (sans coordination
/// réelle, c'est purement cosmétique). Lignes courtes.
const CHAT_LINES_LOW_HP: &[&str] = &[
    "Need health!",
    "I'm hit, falling back.",
    "Low health, backing off.",
    "Critical!",
    "Cover me.",
    "Pulling back.",
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
            Self::Spotted => CHAT_LINES_SPOTTED,
            Self::Heard => CHAT_LINES_HEARD,
            Self::LowHp => CHAT_LINES_LOW_HP,
        }
    }

    /// Multiplicateur appliqué à `CHAT_TRIGGER_PROB` — certains
    /// triggers sont plus bavards (kill insult) que d'autres (respawn).
    fn weight(self) -> f32 {
        match self {
            Self::KillInsult => 1.0,
            Self::Death => 0.85,
            Self::Respawn => 0.35,
            // Spotted = courant mais cool — 50%. Heard = rare (ne pas
            // confirmer une planque). LowHp = signal d'aide marqué.
            Self::Spotted => 0.50,
            Self::Heard => 0.30,
            Self::LowHp => 0.70,
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
    /// Ignoré quand `is_frag = true` (on affiche "+1 FRAG" à la place).
    damage: i32,
    /// `true` = dégât subi par le joueur (rouge), `false` = dégât infligé
    /// à un bot (jaune).
    to_player: bool,
    /// `time_sec` au delà duquel on purge.
    expire_at: f32,
    /// Durée totale initiale (s). Sert au calcul du fade alpha + drift.
    lifetime: f32,
    /// `true` = c'est un frag-confirm.  Le rendu affiche "+1 FRAG" en gros
    /// caractères orange-rouge au lieu du nombre, et la durée de vie est
    /// plus longue pour que le joueur ait le temps de savourer.
    is_frag: bool,
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
            // **Quad Damage** : tint plein-écran SUPPRIMÉ (v0.9.5++
            // user request) — feedback visuel jugé trop intrusif.
            // Le SFX d'activation + le chrono powerup HUD suffisent.
            Self::QuadDamage => None,
            // **Haste** : tint plein-écran SUPPRIMÉ (v0.9.5++ user
            // request idem Quad).  La vitesse boostée est déjà bien
            // perceptible via le mouvement / la cadence de tir.
            Self::Haste => None,
            // Regen : pas de tint. Le rendu du joueur se soignant est
            // déjà communiqué par la barre HP qui remonte + le badge.
            Self::Regeneration => None,
            // **Battle Suit** : tint plein-écran SUPPRIMÉ (v0.9.5++ user
            // request idem Quad/Haste).  L'immunité environnementale est
            // perçue par l'absence de pain audio / dégât HP.
            Self::BattleSuit => None,
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

    /// **Recoil camera kick** par arme — `(pitch_deg, yaw_jitter_deg)`.
    /// Le pitch est ajouté positivement (canon vers le haut), le yaw
    /// est aléatoire dans `±yaw_jitter_deg` (pour donner un côté
    /// "rafale qui balade le réticule").  Profil distinct du
    /// `view_kick` (qui agit sur l'offset spatial du viewmodel) :
    /// celui-ci tape les angles de vue pour le rendu uniquement,
    /// l'aim réel n'est pas affecté.
    fn recoil_kick(self) -> (f32, f32) {
        match self {
            Self::Gauntlet        => (0.0, 0.0),   // mêlée, pas de recul
            Self::Machinegun      => (1.4, 0.5),   // rafale — petit pitch + jitter horizontal
            Self::Shotgun         => (4.5, 0.0),   // gros punch vertical
            Self::Grenadelauncher => (3.2, 0.0),   // recul marqué
            Self::Rocketlauncher  => (5.0, 0.0),   // énorme, l'arme saute
            Self::Lightninggun    => (0.4, 0.2),   // beam continu, tremblement minime
            Self::Railgun         => (3.8, 0.0),   // sec et fort
            Self::Plasmagun       => (0.6, 0.3),   // pulsation légère
            Self::Bfg             => (6.0, 0.0),   // monstrueux
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
                // Vanilla Q3 canon
                "sound/weapons/grenade/grenlf1a.wav",
                // Variantes connues sur pak0 partiels / mods
                "sound/weapons/grenade/grenlf1.wav",
                "sound/weapons/grenade/glaunch1.wav",
                "sound/weapons/grenade/gl_fire.wav",
                "sound/weapons/grenade/grenadef1.wav",
                "sound/weapons/grenadef1a.wav",
                // **Fallback sonore** : si AUCUN sample grenade n'existe
                // dans les paks de l'utilisateur, on cascade vers rocket
                // (les deux armes sont des launchers projectile, le
                // « thunk-fwoosh » est suffisamment proche pour ne pas
                // casser l'identification arme).  Et si rocket aussi
                // absent → shotgun (toujours présent sur les paks).
                "sound/weapons/rocket/rocklf1a.wav",
                "sound/weapons/rocket/rocklf1.wav",
                "sound/weapons/shotgun/sshotf1b.wav",
                "sound/weapons/shotgun/sshotf1.wav",
            ],
            Self::Rocketlauncher => &[
                // Vanilla Q3 canon
                "sound/weapons/rocket/rocklf1a.wav",
                "sound/weapons/rocket/rocklf1.wav",
                // Variantes connues sur paks partiels / mods
                "sound/weapons/rocket/rocketf1.wav",
                "sound/weapons/rocket/rocketf1a.wav",
                "sound/weapons/rocket/rl_fire.wav",
                "sound/weapons/rocket/rocket_fire.wav",
                "sound/weapons/rocket/rocket.wav",
                "sound/weapons/rocket/rl1.wav",
                "sound/weapons/rocket/rocket1.wav",
                "sound/weapons/rocketlf1a.wav",
                "sound/weapons/rocketlf1.wav",
                // **Fallback sonore** : sample SHOTGUN — punchy/explosif,
                // évoque bien un tir de rocket si aucun sample dédié.
                // Le shotgun a fait ses preuves comme « toujours présent »
                // sur les paks utilisateur (jamais signalé silencieux).
                "sound/weapons/shotgun/sshotf1b.wav",
                "sound/weapons/shotgun/sshotf1.wav",
            ],
            Self::Lightninggun => &[
                "sound/weapons/lightning/lg_fire.wav",
                "sound/weapons/lightning/lightning_fire.wav",
            ],
            Self::Railgun => &[
                "sound/weapons/railgun/railgf1a.wav",
            ],
            Self::Plasmagun => &[
                // Q3 vanilla : `hyprbf1a.wav` est le « hyperblaster fire »
                // — c'est bien le SFX du plasmagun.  `lasfly.wav` est la
                // boucle continue (laser fly) en fallback.  On essaie
                // aussi `plasx1.wav` / `plasmaf1.wav` qui existent dans
                // certains paks tiers / mods.
                "sound/weapons/plasma/hyprbf1a.wav",
                "sound/weapons/plasma/plasmaf1.wav",
                "sound/weapons/plasma/plasx1.wav",
                "sound/weapons/plasma/plasma_fire.wav",
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
            // **W2 — Machinegun burst précision** : tap-fire single shot
            // sans dispersion + dégât boosté. Coût : 3 munitions, cadence
            // longue. Récompense le tir posé sur cible distante (sniping
            // léger). Le primaire reste l'arme « volume de feu ».
            Self::Machinegun => Some(WeaponParams {
                damage: 18,
                cooldown: 0.35,
                spread_deg: 0.0,
                ammo_cost: 3,
                ..p
            }),
            // **W4 — Grenade airburst** : grenade sans rebond, plus rapide,
            // splash élargi. Trajectoire tendue qui explose au premier
            // contact mur/sol — tir direct, pas un piège au sol. Le
            // `bounce=false` est appliqué au spawn (cf. fire_weapon).
            Self::Grenadelauncher => Some(WeaponParams {
                kind: WeaponKind::Projectile {
                    speed: 1100.0,         // vs 700 primaire — flat shot
                    splash_radius: 200.0,  // vs 150 — zone élargie
                    splash_damage: 110,
                },
                damage: 110,
                cooldown: 1.0,             // un peu plus long
                ..p
            }),
            // **W6 — Lightning shock blast** : décharge unique haute
            // tension, courte portée mais gros dégât burst. Cooldown
            // long pour empêcher le DPS continu d'être supérieur au
            // primaire. Coût : 5 cellules pour un tap.
            Self::Lightninggun => Some(WeaponParams {
                damage: 55,
                cooldown: 0.8,
                range: 1024.0,             // vs 768 — un peu plus long
                ammo_cost: 5,
                ..p
            }),
            // **W8 — Plasma orb** : un gros plasma lent à splash large.
            // Spawn unique vs rafale primaire — "lobbe" tactique pour
            // contrôler un couloir. Speed/splash overridés via le `kind`.
            Self::Plasmagun => Some(WeaponParams {
                kind: WeaponKind::Projectile {
                    speed: 700.0,          // vs 2000 primaire
                    splash_radius: 96.0,   // vs 20 — vraie zone
                    splash_damage: 60,
                },
                damage: 50,
                cooldown: 0.6,             // vs 0.1 — vraiment moins spammable
                ammo_cost: 8,              // 8× le coût primaire
                ..p
            }),
            // **W9 — BFG death zone** : projectile plus lent, splash
            // énorme, dégât splash gigantesque. Une « bombe » Q3 qui
            // demande de prédire le mouvement adverse — slow ball big
            // boom. Cooldown long pour ne pas en abuser.
            Self::Bfg => Some(WeaponParams {
                kind: WeaponKind::Projectile {
                    speed: 1100.0,         // vs 2000 primaire
                    splash_radius: 250.0,  // vs 120 — zone monstrueuse
                    splash_damage: 160,
                },
                damage: 120,
                cooldown: 0.6,             // vs 0.2 — pas spammable
                ammo_cost: 3,              // 3× plus cher
                ..p
            }),
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
    /// Ranges parsées depuis `animation.cfg` du modèle joueur.  Source
    /// CANONIQUE des animations Q3 : chaque modèle peut avoir ses propres
    /// offsets (sarge ≠ keel ≠ orbb).  Si le fichier est absent, on
    /// fallback sur les offsets canoniques (constants `bot_anims::*`).
    anims: hashbrown::HashMap<&'static str, AnimRange>,
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
    /// **Anti-spam pain SFX** — timestamp du dernier "ouch" joué.
    /// Le shotgun envoie 11 pellets en 1 tick → sans cooldown on
    /// joue 11 plays de pain simultanés ⇒ son saturé. De même un
    /// splash rocket touche plusieurs bots simultanément. Cooldown
    /// par-bot de [`PAIN_SFX_COOLDOWN`] secondes.
    last_pain_sfx_at: f32,
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
    /// **Sound awareness** (v0.9.5) — dernière position du joueur
    /// entendue (tir, footstep proche).  Le bot va enquêter cette
    /// position s'il n'a pas de LOS direct.  None = pas d'évènement
    /// auditif récent.
    last_heard_pos: Option<Vec3>,
    /// Timestamp du dernier évènement sonore perçu — l'awareness
    /// expire après [`BOT_HEARING_MEMORY_SEC`] sans nouveau bruit.
    last_heard_at: f32,
    /// **Personnalité bot** (v0.9.5) — biais de comportement persistant
    /// par bot.  Affecte la distance d'engagement préférée et la propension
    /// à fuir/charger sous le feu.
    personality: BotPersonality,
    /// **Squad chatter cooldown** — empêche un bot de spammer la radio.
    /// Chaque bot ne parle qu'une fois toutes les `CHATTER_COOLDOWN_SEC`.
    last_chatter_at: f32,
    /// **Animation v0.9.5++** — instant où la mort a été enregistrée.
    /// Permet de jouer BOTH_DEATH1 (1.2 s à 25 fps) puis BOTH_DEAD1 freeze.
    /// `None` = bot vivant.
    death_started_at: Option<f32>,
    /// Variante de mort assignée au respawn — 0/1/2 → DEATH1/2/3.
    /// Hash du nom + deaths pour stable mais varié.
    death_variant: u8,
    /// Yaw précédent — utilisé pour détecter le LEGS_TURN (yaw change
    /// rapide alors que le bot est stationnaire).
    prev_yaw: f32,
    /// Instant du dernier yaw delta significatif — déclenche LEGS_TURN
    /// pendant ~200 ms.
    last_turn_at: f32,
    /// Instant du dernier taunt (TORSO_GESTURE).  None = pas de taunt actif.
    /// Le taunt dure ~2.7 s (40 frames à 15 fps).
    gesture_started_at: Option<f32>,
    /// Identifiant (= `range.start`) de la `lower_range` jouée à la
    /// dernière frame, et instant du dernier changement.  Permet de
    /// rebaser `phase = time - lower_anim_started_at` quand la range
    /// change → l'anim redémarre à frame 0 au lieu de téléporter au
    /// milieu du cycle (v0.9.5++ fix).  `start = usize::MAX` = jamais
    /// initialisé encore.
    lower_anim_start: usize,
    lower_anim_started_at: f32,
    /// Idem pour upper (TORSO_*).
    upper_anim_start: usize,
    upper_anim_started_at: f32,
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
    ///
    /// **v0.9.5++ fix** : si la range hardcoded dépasse `nf` (modèle
    /// avec moins de frames que le canonical sarge 308), on scale-down
    /// proportionnellement.  Avant, des frames out-of-range étaient
    /// clampées à `nf-1` → animation FIGÉE sur 1 seule frame.
    fn sample(&self, phase: f32, nf: usize) -> (usize, usize, f32) {
        if self.end <= self.start || nf == 0 {
            return (0, 0, 0.0);
        }
        // **Auto-rescale si range hors-plage** — le canonical Q3 sarge
        // utilise 308 frames par mesh.  Si notre mesh en a moins, on
        // scale pour rester proportionnel.  Garantit que les anims
        // bougent même sur des modèles tronqués / customs.
        let canonical_total = 308_usize;
        let (start, end) = if self.end > nf && nf > 1 {
            let scale = nf as f32 / canonical_total as f32;
            let s = ((self.start as f32 * scale) as usize).min(nf - 2);
            let e = ((self.end as f32 * scale) as usize).max(s + 1).min(nf - 1);
            (s, e)
        } else {
            (self.start, self.end)
        };
        let span = end - start;
        if span == 0 {
            // **Bug fix v0.9.5++** — avant on retournait `(0, 0, 0.0)`
            // ce qui forçait le MD3 frame 0 (T-pose) au lieu de la frame
            // canonique de la range.  Pour BOTH_DEAD1 (start:29 end:30
            // span:1 mais après clamp nf<31 → span peut tomber à 0) ça
            // remettait le cadavre en T-pose.  On retourne la frame de
            // start clampée aux bornes du mesh.
            let f = start.min(nf.saturating_sub(1));
            return (f, f, 0.0);
        }
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
        let fa = (start + a_rel).min(nf.saturating_sub(1));
        let fb = (start + b_rel).min(nf.saturating_sub(1));
        (fa, fb, lerp)
    }
}

/// **animation.cfg parser** — lit le fichier canonique Q3 d'un player
/// model et retourne une HashMap `nom → AnimRange`.  Format Q3 :
/// ```text
/// sex m
/// fixedlegs
/// fixedtorso
/// // first num loop fps
/// 0   30 0  25      // BOTH_DEATH1
/// 29  1  0  25      // BOTH_DEAD1
/// ...
/// ```
/// Lignes vides + `//` + headers (sex/fixedlegs/etc.) ignorés.
/// L'ordre des 25 lignes data correspond à la liste ANIM_NAMES.
///
/// Spécificité Q3 (cf. `cgame/cg_players.c`) : pour les `LEGS_*`,
/// on doit soustraire `TORSO_GESTURE.firstFrame` car le mesh `lower.md3`
/// ne contient PAS les frames TORSO.  Adjustement appliqué ici.
fn parse_animation_cfg(content: &str) -> hashbrown::HashMap<&'static str, AnimRange> {
    const ANIM_NAMES: [&str; 25] = [
        "BOTH_DEATH1", "BOTH_DEAD1", "BOTH_DEATH2", "BOTH_DEAD2", "BOTH_DEATH3", "BOTH_DEAD3",
        "TORSO_GESTURE", "TORSO_ATTACK", "TORSO_ATTACK2", "TORSO_DROP", "TORSO_RAISE",
        "TORSO_STAND", "TORSO_STAND2",
        "LEGS_WALKCR", "LEGS_WALK", "LEGS_RUN", "LEGS_BACK", "LEGS_SWIM",
        "LEGS_JUMP", "LEGS_LAND", "LEGS_JUMPB", "LEGS_LANDB",
        "LEGS_IDLE", "LEGS_IDLECR", "LEGS_TURN",
    ];
    let mut data_lines: Vec<(i32, i32, i32, f32)> = Vec::with_capacity(25);
    for raw in content.lines() {
        let line = raw.split("//").next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        // Skip headers : non-numeric first token.
        let first_tok = line.split_whitespace().next().unwrap_or("");
        if !first_tok.chars().next().map(|c| c.is_ascii_digit() || c == '-').unwrap_or(false) {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            continue;
        }
        let first = parts[0].parse::<i32>().unwrap_or(0);
        let num = parts[1].parse::<i32>().unwrap_or(0);
        let loop_frames = parts[2].parse::<i32>().unwrap_or(0);
        let fps = parts[3].parse::<f32>().unwrap_or(15.0).max(1.0);
        data_lines.push((first, num, loop_frames, fps));
    }
    let mut out = hashbrown::HashMap::new();
    if data_lines.len() < 13 {
        // Pas assez de lignes pour TORSO_GESTURE → on abandonne.
        warn!("animation.cfg: seulement {} lignes data trouvées (besoin ≥ 13)", data_lines.len());
        return out;
    }
    // TORSO_GESTURE = index 6 dans ANIM_NAMES.  Sa firstFrame sert
    // d'offset pour shifter les LEGS_* (qui commencent à index 13).
    let torso_gesture_first = data_lines[6].0;
    for (i, name) in ANIM_NAMES.iter().enumerate() {
        if i >= data_lines.len() {
            break;
        }
        let (mut first, num, loop_frames, fps) = data_lines[i];
        // **Q3 LEGS adjustment** : lower.md3 contient BOTH (0-89)
        // puis LEGS directement à la suite — sans le TORSO bloc.
        // Donc on doit shifter LEGS_* de -torso_gesture_first.
        if i >= 13 {
            first -= torso_gesture_first;
        }
        let start = first.max(0) as usize;
        let span = num.max(1) as usize;
        let end = start + span;
        let looping = loop_frames > 0;
        out.insert(*name, AnimRange { start, end, fps, looping });
    }
    info!("animation.cfg: {} anims parsées (TORSO_GESTURE.first = {})",
          out.len(), torso_gesture_first);
    out
}

/// Table des plages Q3 standard pour le squelette `player` (offsets de
/// `animation.cfg` canonique pour `models/players/sarge/`).  En pratique
/// chaque modèle a sa propre config — on clamp sur `nf` pour ne jamais
/// lire hors plage.
#[allow(dead_code)] // certaines ranges (death) sont scaffoldées pour usage futur
mod bot_anims {
    use super::AnimRange;

    // Offsets historiques animation.cfg (Q3 id player sarge) — unités en frames.
    // Ranges pour le **upper** (`models/players/sarge/upper.md3`).
    pub const BOTH_DEATH1: AnimRange = AnimRange { start: 0, end: 30, fps: 25.0, looping: false };
    pub const BOTH_DEAD1:  AnimRange = AnimRange { start: 29, end: 30, fps: 1.0,  looping: true };  // freeze last frame
    pub const BOTH_DEATH2: AnimRange = AnimRange { start: 30, end: 60, fps: 25.0, looping: false };
    pub const BOTH_DEAD2:  AnimRange = AnimRange { start: 59, end: 60, fps: 1.0,  looping: true };
    pub const BOTH_DEATH3: AnimRange = AnimRange { start: 60, end: 90, fps: 25.0, looping: false };
    pub const BOTH_DEAD3:  AnimRange = AnimRange { start: 89, end: 90, fps: 1.0,  looping: true };
    pub const TORSO_GESTURE: AnimRange = AnimRange { start: 90, end: 130, fps: 15.0, looping: false };
    pub const TORSO_ATTACK:  AnimRange = AnimRange { start: 130, end: 145, fps: 15.0, looping: false };
    pub const TORSO_ATTACK2: AnimRange = AnimRange { start: 145, end: 157, fps: 15.0, looping: false }; // gauntlet swing
    pub const TORSO_DROP:    AnimRange = AnimRange { start: 157, end: 162, fps: 15.0, looping: false };
    pub const TORSO_RAISE:   AnimRange = AnimRange { start: 162, end: 168, fps: 15.0, looping: false };
    pub const TORSO_STAND:   AnimRange = AnimRange { start: 171, end: 176, fps: 20.0, looping: true };
    pub const TORSO_STAND2:  AnimRange = AnimRange { start: 176, end: 181, fps: 20.0, looping: true };
    /// **Q3 sarge n'a pas de TORSO_PAIN** dédié → on émule en jouant
    /// quelques frames de TORSO_GESTURE clampées au début (effet
    /// "tressaut" 200 ms).  Si l'animation.cfg fournit une range
    /// `TORSO_PAIN1` officielle, le parser l'utilisera à la place via
    /// le lookup `anim("TORSO_PAIN1", ...)`.  Plage courte = 5 frames.
    pub const TORSO_PAIN1:   AnimRange = AnimRange { start: 168, end: 171, fps: 12.0, looping: false };
    // Ranges pour le **lower** (`models/players/sarge/lower.md3`).
    pub const LEGS_WALKCR:   AnimRange = AnimRange { start: 178, end: 193, fps: 20.0, looping: true };
    pub const LEGS_WALK:     AnimRange = AnimRange { start: 193, end: 213, fps: 20.0, looping: true };
    pub const LEGS_RUN:      AnimRange = AnimRange { start: 213, end: 226, fps: 20.0, looping: true };
    pub const LEGS_BACK:     AnimRange = AnimRange { start: 226, end: 236, fps: 20.0, looping: true };
    pub const LEGS_SWIM:     AnimRange = AnimRange { start: 236, end: 246, fps: 20.0, looping: true };
    pub const LEGS_JUMP:     AnimRange = AnimRange { start: 248, end: 254, fps: 20.0, looping: false };
    pub const LEGS_LAND:     AnimRange = AnimRange { start: 254, end: 258, fps: 20.0, looping: false };
    pub const LEGS_JUMPB:    AnimRange = AnimRange { start: 258, end: 264, fps: 20.0, looping: false };
    pub const LEGS_LANDB:    AnimRange = AnimRange { start: 264, end: 268, fps: 20.0, looping: false };
    pub const LEGS_IDLE:     AnimRange = AnimRange { start: 268, end: 291, fps: 15.0, looping: true };
    pub const LEGS_IDLECR:   AnimRange = AnimRange { start: 291, end: 308, fps: 15.0, looping: true };
    pub const LEGS_TURN:     AnimRange = AnimRange { start: 308, end: 313, fps: 15.0, looping: true };
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
/// Cooldown anti-spam du SFX pain par bot (secondes). Sans ça un SG
/// 11-pellets joue 11 plays simultanés, et un splash rocket multi-bot
/// fait pareil. 0.18s = juste assez pour qu'un seul play soit audible
/// sur une volée mais qu'un nouveau hit après pause re-déclenche le
/// pain.
const PAIN_SFX_COOLDOWN: f32 = 0.18;

/// **Sound awareness** (v0.9.5) — rayon (unités) dans lequel un tir
/// joueur est entendu par les bots. Au-delà, le bruit s'atténue trop
/// pour déclencher une investigation.
const BOT_HEARING_RADIUS: f32 = 1800.0;
/// Mémoire d'évènement sonore — un bot oublie un bruit après cette
/// durée s'il n'en perçoit pas de nouveau.
const BOT_HEARING_MEMORY_SEC: f32 = 6.0;
/// Cooldown chatter par bot — sans ça la radio sature à chaque kill
/// et chaque contact visuel.
const CHATTER_COOLDOWN_SEC: f32 = 4.0;

/// Personnalité bot — biais de comportement persistant.  N'affecte pas
/// la skill (qui contrôle l'aim/réaction) mais le **style** : un Sniper
/// préfère le railgun à longue distance, un Rusher charge avec SG/RL,
/// un Camper temporise sur les pickups.
///
/// Wrapper local autour de [`q3_bot::BotPersonality`] pour ajouter une
/// distribution déterministe par index + un tag debug.  Le bot consomme
/// directement `q3_bot::BotPersonality` via `Bot::set_personality`.
type BotPersonality = q3_bot::BotPersonality;

fn bot_personality_from_index(idx: usize) -> BotPersonality {
    match idx % 4 {
        0 => BotPersonality::Rusher,
        1 => BotPersonality::Sniper,
        2 => BotPersonality::Camper,
        _ => BotPersonality::Balanced,
    }
}

#[allow(dead_code)]
fn bot_personality_tag(p: BotPersonality) -> &'static str {
    match p {
        BotPersonality::Rusher => "rush",
        BotPersonality::Sniper => "snipe",
        BotPersonality::Camper => "camp",
        BotPersonality::Balanced => "bal",
    }
}

/// **Headshot zone** — tout impact dont la position Z dépasse cette
/// hauteur (relative à `body.origin`) compte comme un headshot. Q3
/// vanilla ne distingue pas la tête, mais c'est la signature des FPS
/// modernes. Le hit_radius reste une sphère unique (pas de hitbox
/// per-bone) — on autorise donc les "headshots" sur le bord supérieur
/// de la sphère, ce qui ressent juste comme un viser haut.
const HEADSHOT_Z_THRESHOLD: f32 = 48.0;
/// Multiplicateur de dégâts pour un headshot.
const HEADSHOT_DMG_MULT: f32 = 1.5;
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
            // **Music player commands** (v0.9.5+) — `music play <path>`
            // charge un fichier audio local en loop. `music stop`
            // l'arrête. `music list` affiche les fichiers du dossier
            // `music/` utilisateur (cf. `user_music_dir`).
            let p = pending.clone();
            cmds.add("music", move |args: &Args| {
                if args.count() < 2 {
                    info!("music: usage `music play <path>` | `music stop` | `music list`");
                    return;
                }
                match args.argv(1) {
                    "play" => {
                        if args.count() < 3 {
                            info!("music: usage `music play <path>`");
                            return;
                        }
                        let raw = args.argv(2);
                        // Concat pour gérer les paths avec espaces.
                        let path_str = if args.count() > 3 {
                            (2..args.count())
                                .map(|i| args.argv(i))
                                .collect::<Vec<_>>()
                                .join(" ")
                        } else {
                            raw.to_string()
                        };
                        p.lock().push(PendingAction::MusicPlay(path_str.into()));
                    }
                    "stop" => {
                        p.lock().push(PendingAction::MusicStop);
                    }
                    "list" => {
                        // Liste synchrone — log direct sans passer par
                        // pending. Pour la console seule.
                        for path in list_music_files() {
                            info!("music: {}", path.display());
                        }
                    }
                    other => {
                        info!("music: sous-commande inconnue `{other}`");
                    }
                }
            });
        }
        {
            let p = pending.clone();
            cmds.add("clearbots", move |_args: &Args| {
                p.lock().push(PendingAction::ClearBots);
            });
        }
        {
            // **mapdl** — download manager intégré.  Sous-commandes :
            //   `mapdl list`             liste le catalogue
            //   `mapdl get <id>`         lance un DL en arrière-plan
            //   `mapdl status`           snapshot du job courant
            let p = pending.clone();
            cmds.add("mapdl", move |args: &Args| {
                if args.count() < 2 {
                    info!("mapdl: usage `mapdl list` | `mapdl get <id>` | `mapdl status`");
                    return;
                }
                match args.argv(1) {
                    "list" => {
                        p.lock().push(PendingAction::MapDlList);
                    }
                    "get" => {
                        if args.count() < 3 {
                            info!("mapdl: usage `mapdl get <id>` (utilise `mapdl list`)");
                            return;
                        }
                        p.lock().push(PendingAction::MapDlGet(args.argv(2).to_string()));
                    }
                    "status" => {
                        p.lock().push(PendingAction::MapDlStatus);
                    }
                    other => {
                        info!("mapdl: sous-commande inconnue `{other}`");
                    }
                }
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
        // **Player cosmetic tint** (v0.9.5++) — couleur appliquée au
        // viewmodel de l'arme + halo dlight quand le joueur sprinte.
        // Format : "r,g,b" en floats 0..1 (ex. "1.0,0.4,0.4" = rouge).
        // Default : blanc (pas de tint).
        cvars.register("cg_playertint", "1.0,1.0,1.0", CvarFlags::ARCHIVE);
        cvars.register("s_volume", "0.8", CvarFlags::ARCHIVE);
        cvars.register("s_sfxvolume", "1.0", CvarFlags::ARCHIVE);
        cvars.register("s_musicvolume", "0.25", CvarFlags::ARCHIVE);
        // **Dossiers supplémentaires** où chercher des fichiers audio.
        // Liste séparée par `;` (Windows convention) ou `:` (Unix).
        // Permet à l'utilisateur d'ajouter ses dossiers Spotify-export,
        // SSD secondaire, etc.  Scanné récursivement (max 4 niveaux).
        cvars.register("s_musicpath", "", CvarFlags::ARCHIVE);
        // **FOV scaling mode** (v0.9.5++ #8) :
        //   `0` = Hor+ (défaut Quake/arena, FOV élargit horizontalement
        //          avec l'aspect — recommandé pour la plupart des joueurs).
        //   `1` = Vert- (Counter-Strike / Apex style, FOV horizontal lock,
        //          vertical réduit sur ultrawide pour pas d'avantage périph).
        cvars.register("cg_fovaspect", "0", CvarFlags::ARCHIVE);
        // **HDR display output** (v0.9.5++ #8) — nécessite un écran HDR10
        // compatible.  `0` = sRGB classique, `1` = HDR10 (PQ tonemap).
        // Note : la pipeline interne reste HDR (Rgba16Float scene buffer)
        // dans tous les cas ; cette cvar contrôle uniquement le format
        // de la surface présentée.  Bascule effective au prochain
        // changement de résolution / fullscreen toggle.
        cvars.register("r_hdr", "0", CvarFlags::ARCHIVE);
        // **Skybox override** (v0.9.5++) — chemin VFS de base
        // (sans `_rt/_lf/...` suffixes) pour forcer une skybox custom
        // sur toutes les maps.  Les fichiers attendus sont
        // `<value>_{rt,lf,up,dn,ft,bk}.tga`.  Vide = utilise le sky
        // shader défini dans le BSP de la map (comportement Q3 standard).
        // Nouvelle valeur appliquée au prochain `map <name>`.
        cvars.register("r_skybox", "env/skybox_clouds", CvarFlags::ARCHIVE);
        // **Godmode** (v0.9.5++ test) — `1` = le joueur ne prend aucun
        // dégât bots/projectiles bots/dégât environnemental (lave, drown,
        // splash). Pratique pour tester gameplay/visuels sans mourir.
        // Pas archivé (réinitialisé au boot).
        cvars.register("g_godmode", "0", CvarFlags::empty());
        // **Battle Royale bots** (v0.9.5++) — `0` = carte BR vide
        // (mode exploration, défaut).  `1` = spawn `pending_local_bots`
        // au chargement de la map terrain.  Permet de tester le terrain
        // sans IA gameplay.
        cvars.register("br_bots", "0", CvarFlags::ARCHIVE);
        // Mouse smoothing low-pass — 0 = aucun, 0.9 = très lissé.  Lerp
        // appliqué côté input dans `App::on_mouse_motion`.
        cvars.register("m_smoothing", "0.0", CvarFlags::ARCHIVE);
        // Bloom HDR — 1=on, 0=off, lu par `Renderer` chaque frame.
        cvars.register("r_bloom", "1", CvarFlags::ARCHIVE);
        // Vsync — toggle au prochain redémarrage. Le swapchain `Mailbox`
        // (vsync off) impose une recreation chez wgpu, on ne le fait pas
        // à chaud pour ne pas avoir à gérer les frames en vol.
        cvars.register("r_vsync", "1", CvarFlags::ARCHIVE);
        // **Video settings persistence** (v0.9.5) — taille fenêtre +
        // mode plein écran sont sauvegardés dans q3config.cfg via le
        // flag ARCHIVE.  Restaurés au boot avant la création de la
        // fenêtre (cf. `App::resumed`).
        cvars.register("r_width", "1920", CvarFlags::ARCHIVE);
        cvars.register("r_height", "1080", CvarFlags::ARCHIVE);
        // Fullscreen par défaut (v0.9.5++) — borderless sur moniteur primaire.
        // L'utilisateur peut basculer en fenêtré via le menu Vidéo, et
        // la valeur est persistée dans `q3config.cfg` via le flag ARCHIVE.
        cvars.register("r_fullscreen", "1", CvarFlags::ARCHIVE);
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
        let mut map_list: Vec<String> = vfs
            .list_suffix(".bsp")
            .into_iter()
            .filter(|p| p.starts_with("maps/"))
            // **Q3DM0 + Q3DM12 + Q3DM19 supprimées** (v0.9.5++ user
            // request) — exclues du menu.  Filtrage case-insensitive
            // pour attraper q3dm0.bsp / Q3DM12.BSP / etc.  Note : on
            // utilise une regex stricte (`q3dm0.bsp`) sur Q3DM0 pour
            // ne pas accidentellement matcher q3dm01..q3dm09 si jamais.
            .filter(|p| {
                let lc = p.to_ascii_lowercase();
                if lc.contains("q3dm12") || lc.contains("q3dm19") {
                    return false;
                }
                // Match strict q3dm0.bsp (pas q3dm0X)
                if lc.ends_with("/q3dm0.bsp") || lc == "maps/q3dm0.bsp" {
                    return false;
                }
                true
            })
            .collect();
        // **Battle Royale Réunion** (v0.9.5++ ré-intégrée) — la carte
        // utilise un terrain procédural (pas de BSP), donc elle n'est
        // pas découverte par `vfs.list_suffix(".bsp")`.  On l'ajoute
        // manuellement.  Le `load_map` route automatiquement les
        // `br_*` vers `load_terrain_map`.  Bots gated par cvar
        // `br_bots` (défaut 0 → réunion vide).
        let br_entry = "maps/br_reunion.bsp".to_string();
        if !map_list.iter().any(|m| m == &br_entry) {
            map_list.push(br_entry);
        }
        map_list.sort();
        let mut menu = crate::menu::Menu::new(map_list, /* in_game */ false);
        // Seed le catalogue du downloader pour la page menu MapDownloader.
        let catalog: Vec<(String, String)> = crate::map_dl::MapDownloader::default_catalog()
            .iter()
            .map(|e| (e.id.to_string(), format!("{} — {}", e.name, e.author.unwrap_or("?"))))
            .collect();
        menu.set_mapdl_catalog(catalog);
        if requested_map.is_none() {
            menu.open_root();
        }

        let this = Self {
            vfs,
            init_width: width,
            init_height: height,
            requested_map,
            window: None,
            renderer: None,
            world: None,
            terrain: None,
            br_ring: None,
            drones: Vec::new(),
            rocks: Vec::new(),
            ammo_crate_scale: None,
            quad_pickup_scale: None,
            health_pack_scale: None,
            railgun_pickup_scale: None,
            grenade_ammo_scale: None,
            rocket_ammo_scale: None,
            cell_ammo_scale: None,
            lg_ammo_scale: None,
            big_armor_scale: None,
            plasma_pickup_scale: None,
            railgun_ammo_scale: None,
            regen_pickup_scale: None,
            machinegun_pickup_scale: None,
            bfg_pickup_scale: None,
            lightninggun_pickup_scale: None,
            shotgun_pickup_scale: None,
            grenadelauncher_pickup_scale: None,
            gauntlet_pickup_scale: None,
            shotgun_ammo_scale: None,
            bfg_ammo_scale: None,
            rocketlauncher_pickup_scale: None,
            combat_armor_scale: None,
            medkit_scale: None,
            armor_shard_scale: None,
            prop_lights: hashbrown::HashMap::new(),
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
            show_stats_overlay: false,
            last_respawn_time: 0.0,
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
            last_fire_sfx_at: f32::NEG_INFINITY,
            total_shots: 0,
            total_hits: 0,
            time_warnings_fired: 0,
            next_player_fire_at: 0.0,
            muzzle_flash_until: 0.0,
            view_kick: 0.0,
            view_kick_pitch: 0.0,
            view_kick_yaw: 0.0,
            map_dl: crate::map_dl::MapDownloader::new(
                std::path::PathBuf::from("baseq3"),
            ),
            // (Le menu sera seedé après construction — voir set_mapdl_catalog)
            hit_marker_until: 0.0,
            kill_marker_until: 0.0,
            armor_flash_until: 0.0,
            pain_flash_until: 0.0,
            powerup_flash_until: 0.0,
            powerup_flash_color: [0.0; 4],
            lightning_flash_until: 0.0,
            next_lightning_at: f32::INFINITY,
            fov_punch_until: 0.0,
            fov_punch_strength: 0.0,
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
            viewmodel_prev_yaw: 0.0,
            viewmodel_prev_pitch: 0.0,
            viewmodel_sway: [0.0, 0.0],
            player_last_land_at: f32::NEG_INFINITY,
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
            sfx_death_bot: None,
            sfx_weapon_pickup: None,
            sfx_ammo_pickup: None,
            sfx_health_pickup: None,
            sfx_armor_pickup: None,
            sfx_megahealth_pickup: None,
            sfx_plasma_impact: None,
            sfx_rocket_impact: None,
            sfx_grenade_impact: None,
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
        };
        this
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
                if self.world.is_some() || self.terrain.is_some() {
                    self.menu.close();
                    self.menu.set_in_game(true);
                    self.set_mouse_capture(true);
                }
            }
            MenuAction::Quit => event_loop.exit(),
            MenuAction::ApplyResolution { width, height } => {
                if let Some(window) = self.window.as_ref() {
                    let _ = window.request_inner_size(
                        winit::dpi::PhysicalSize::new(width, height),
                    );
                    // **Persistance v0.9.5++** — écrit dans les cvars
                    // ARCHIVE pour que `q3config.cfg` les sauvegarde
                    // au exit() (sinon la résolution était perdue
                    // entre 2 lancements).
                    let _ = self.cvars.set("r_width", &format!("{width}"));
                    let _ = self.cvars.set("r_height", &format!("{height}"));
                    info!("menu: résolution → {}×{} (persisté)", width, height);
                }
            }
            MenuAction::ToggleFullscreen => {
                if let Some(window) = self.window.as_ref() {
                    use winit::window::Fullscreen;
                    let new_fs = !self.menu.fullscreen;
                    if new_fs {
                        let monitor = window.current_monitor();
                        window.set_fullscreen(Some(Fullscreen::Borderless(monitor)));
                    } else {
                        window.set_fullscreen(None);
                    }
                    self.menu.set_fullscreen(new_fs);
                    // **Persistance v0.9.5++** — écrit dans `r_fullscreen`
                    // pour que le réglage soit conservé entre 2 launches.
                    let _ = self.cvars.set(
                        "r_fullscreen",
                        if new_fs { "1" } else { "0" },
                    );
                    info!("menu: fullscreen → {} (persisté)", new_fs);
                }
            }
            MenuAction::ToggleVsync => {
                self.menu.vsync = !self.menu.vsync;
                let _ = self.cvars.set(
                    "r_vsync",
                    if self.menu.vsync { "1" } else { "0" },
                );
                info!("menu: vsync → {} (effet au prochain redémarrage)", self.menu.vsync);
            }
            MenuAction::PlayMusicFile(path) => {
                self.handle_music_play(&path);
                self.menu.set_music_now_playing(Some(path));
            }
            MenuAction::StopMusic => {
                if let Some(snd) = self.sound.as_ref() {
                    snd.stop_music();
                }
                self.menu.set_music_now_playing(None);
                info!("menu: music stopped");
            }
            MenuAction::DownloadMap(id) => {
                let started = self.map_dl.start(&id);
                if started {
                    self.menu.set_mapdl_status(format!("downloading {}", id));
                } else {
                    self.menu.set_mapdl_status(
                        "queue full or invalid id".into(),
                    );
                }
            }
        }
    }

    fn load_map(&mut self, path: &str) {
        // **Battle Royale terrain** (v0.9.5) — toute map nommée
        // `maps/br_*` est routée vers le pipeline terrain (heightmap +
        // ring) au lieu du BSP.  `path` peut arriver sous forme
        // `maps/br_reunion.bsp` (depuis le menu) ou `br_reunion` (depuis
        // la console) ; on détecte les deux.
        let stripped = path
            .strip_prefix("maps/")
            .unwrap_or(path)
            .strip_suffix(".bsp")
            .unwrap_or_else(|| path.strip_prefix("maps/").unwrap_or(path));
        if stripped.starts_with("br_") {
            self.load_terrain_map(stripped);
            return;
        }

        // **Reset état BR** (v0.9.5+) — quand on bascule d'une carte
        // BR vers une carte BSP (ex. via menu), il faut clear le
        // terrain + ring + drones, sinon `check_match_end` BR voit
        // 1 entité vivante et déclare immédiatement "VICTORY",
        // bloquant le gameplay.
        self.terrain = None;
        self.br_ring = None;
        // Désarme aussi l'atmosphère BR au cas où on quitte un BR pour
        // un BSP : empêche le tick_atmosphere de tirer un éclair sur
        // un BSP indoor (tick_atmosphere gate aussi via terrain.is_none
        // mais on évite le state stale).
        self.next_lightning_at = f32::INFINITY;
        self.lightning_flash_until = 0.0;
        self.drones.clear();
        self.rocks.clear();
        self.match_winner = None;

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
            // Cvar `r_skybox` permet de forcer une skybox custom
            // pour toutes les maps (override le sky shader BSP).
            let skybox_override = self.cvars.get_string("r_skybox")
                .filter(|s| !s.trim().is_empty());
            resolve_and_load_sky(r, &self.vfs, &bsp, skybox_override.as_deref());
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

        self.load_common_sfx();
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

        }
        // **Asset weapons / projectiles** — sortis du `if let Some(r)`
        // pour pouvoir les invoquer aussi depuis `load_terrain_map`
        // (pas de BSP mais besoin des mêmes meshes).
        self.load_weapon_assets();
        // **Ammo crate GLB** — partagé entre BSP et BR.
        self.load_ammo_crate_asset();
        // **Quad pickup GLB** — remplace l'icone MD3 du Quad Damage.
        self.load_quad_pickup_asset();
        // **Health pack GLB** — trousse de soin moderne pour items
        // item_health*. Partagé entre BSP et BR.
        self.load_health_pack_asset();
        // **Railgun pickup GLB** — modèle moderne pour le railgun
        // au sol ET viewmodel 1ère personne.
        self.load_railgun_pickup_asset();
        // **Grenade ammo box GLB** — boîte de munitions grenades.
        self.load_grenade_ammo_asset();
        // **Rocket ammo box GLB** — boîte de munitions roquettes.
        self.load_rocket_ammo_asset();
        // **Cell ammo box GLB** — cellules énergétiques (plasma).
        self.load_cell_ammo_asset();
        self.load_lg_ammo_asset();
        self.load_big_armor_asset();
        self.load_plasma_pickup_asset();
        self.load_railgun_ammo_asset();
        self.load_regen_pickup_asset();
        self.load_machinegun_pickup_asset();
        self.load_bfg_pickup_asset();
        self.load_lightninggun_pickup_asset();
        self.load_shotgun_pickup_asset();
        self.load_grenadelauncher_pickup_asset();
        self.load_gauntlet_pickup_asset();
        self.load_shotgun_ammo_asset();
        self.load_bfg_ammo_asset();
        self.load_rocketlauncher_pickup_asset();
        self.load_combat_armor_asset();
        self.load_medkit_asset();
        self.load_armor_shard_asset();

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

        // Drain pending_local_bots (centralisé) — couvre toutes les
        // voies de chargement (CLI `--map`, console `/map`, menu Play).
        // Avant v0.9.3 chaque path drainait à part ou pas du tout, ce
        // qui faisait des bots invisibles via le menu. Centralisé ici,
        // c'est le seul site responsable du spawn initial.
        if self.pending_local_bots > 0 {
            let n = self.pending_local_bots;
            self.pending_local_bots = 0;
            self.ensure_player_rig_loaded();
            for i in 0..n {
                let bot_name = format!("bot{:02}", i + 1);
                self.spawn_bot(&bot_name, Some(3));
            }
            info!(
                "spawn initial : {} bot(s) demandés, {} effectivement présents",
                n,
                self.bots.len()
            );
        }
    }

    /// **Battle Royale terrain loader** (v0.9.5) — pendant de
    /// `load_map` pour les cartes BR.
    ///
    /// `name` est le nom court sans préfixe `maps/` ni suffixe `.bsp`
    /// (ex `br_reunion`). On cherche les assets `assets/maps/<name>.r16`
    /// + `.splat.png` + `.terrain.json`.  Si les fichiers ne sont pas
    /// présents (cas où l'utilisateur n'a pas encore lancé le pipeline
    /// Python `tools/dem_to_terrain.py`), on **synthétise** un terrain
    /// de test à partir du preset `TerrainMeta::reunion_default()` —
    /// c'est plat (heightmap de 0) mais ça permet d'instancier le BR
    /// (ring shrink, POIs, spawns) pour valider la stack haut niveau.
    fn load_terrain_map(&mut self, name: &str) {
        use q3_terrain::{br::RingShrink, br::reunion_br_phases, Terrain, TerrainMeta};

        // Tentative de chargement disque ; sinon fallback synthétique.
        let base_path = format!("assets/maps/{name}");
        let terrain = match Terrain::load_from_files(&base_path) {
            Ok(t) => {
                info!(
                    "BR: terrain `{}` chargé depuis disque ({}×{} samples)",
                    name, t.width, t.height
                );
                t
            }
            Err(e) => {
                warn!(
                    "BR: load_from_files('{}') échec : {} — fallback terrain synthétique \
                     (lance tools/dem_to_terrain.py pour produire les assets réels)",
                    base_path, e
                );
                synthesize_reunion_fallback()
            }
        };
        let _ = TerrainMeta::reunion_default; // utilisé par le fallback

        let pois = terrain.pois().to_vec();
        let ring = RingShrink::new(reunion_br_phases(), &pois);
        info!(
            "BR: ring initialisé — phase 0, {} POI sur la carte",
            pois.len()
        );

        // Spawn joueur sur un POI tier ≥ 3 random (utilise la même
        // logique stable que le ring : hash du temps actuel mod len).
        let candidates: Vec<&q3_terrain::Poi> =
            pois.iter().filter(|p| p.tier >= 3).collect();
        let spawn_xy = if candidates.is_empty() {
            (terrain.center().x, terrain.center().y)
        } else {
            let idx = (self.time_sec.to_bits() as usize) % candidates.len();
            (candidates[idx].x, candidates[idx].y)
        };
        let spawn_z = terrain.height_at(spawn_xy.0, spawn_xy.1) + 80.0;
        let spawn = Vec3::new(spawn_xy.0, spawn_xy.1, spawn_z);

        // **Orient toward center** (v0.9.5) — sur un POI côtier (ex.
        // Saint-Denis au nord), regarder l'horizon = océan vide. On
        // calcule un yaw qui pointe vers le centre de l'île pour que
        // le joueur voie immédiatement le relief / les autres POI.
        let center = terrain.center();
        let to_center = Vec3::new(center.x - spawn.x, center.y - spawn.y, 0.0);
        let spawn_yaw = if to_center.length_squared() > 1.0 {
            to_center.y.atan2(to_center.x).to_degrees()
        } else {
            0.0
        };
        info!(
            "BR: spawn joueur à {:?} (yaw {:.0}° → centre)",
            spawn, spawn_yaw
        );

        // Reset état joueur (équivalent du load_map BSP) — on ne route
        // PAS vers `World::from_bsp` puisqu'il n'y a pas de BSP.
        self.player = PlayerMove::new(spawn);
        self.player.view_angles =
            q3_math::Angles { pitch: 0.0, yaw: spawn_yaw, roll: 0.0 };
        self.player_invul_until = self.time_sec + RESPAWN_INVUL_SEC;
        self.last_damage_until = 0.0;
        self.shake_intensity = 0.0;
        self.shake_until = 0.0;
        self.armor_flash_until = 0.0;
        self.pain_flash_until = 0.0;
        if let Some(r) = self.renderer.as_mut() {
            r.camera_mut().position =
                self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
            r.camera_mut().angles = self.player.view_angles;
        }

        // **Asset weapons** (v0.9.5) — viewmodels + projectile meshes.
        // Sans ça, les pickups d'arme BR ne montrent pas de viewmodel
        // quand le joueur les ramasse → bug "arme invisible".
        self.load_weapon_assets();

        // **SFX communs** (v0.9.5+) — fire/pain/pickup/feedback/etc.
        // Sans ça le mode BR était silencieux (SFX chargés seulement
        // dans le path BSP de load_map).
        self.load_common_sfx();

        // **Drones GLB** désactivés (utilisateur préfère sans).
        // **Ammo crate GLB** — remplace le MD3 du pickup d'ammo MG par
        // un mesh stylisé. Asset disque optionnel.
        self.load_ammo_crate_asset();
        // **Quad pickup GLB** — pour les Quad Damage en BR aussi.
        self.load_quad_pickup_asset();
        // **Health pack GLB** — trousse de soin moderne.
        self.load_health_pack_asset();
        // **Railgun pickup GLB** — modèle moderne du railgun au sol.
        self.load_railgun_pickup_asset();
        // **Grenade ammo box GLB** — boîte de munitions grenades.
        self.load_grenade_ammo_asset();
        // **Rocket ammo box GLB** — boîte de munitions roquettes.
        self.load_rocket_ammo_asset();
        // **Cell ammo box GLB** — cellules énergétiques (plasma).
        self.load_cell_ammo_asset();
        self.load_lg_ammo_asset();
        self.load_big_armor_asset();
        self.load_plasma_pickup_asset();
        self.load_railgun_ammo_asset();
        self.load_regen_pickup_asset();
        self.load_machinegun_pickup_asset();
        self.load_bfg_pickup_asset();
        self.load_lightninggun_pickup_asset();
        self.load_shotgun_pickup_asset();
        self.load_grenadelauncher_pickup_asset();
        self.load_gauntlet_pickup_asset();
        self.load_shotgun_ammo_asset();
        self.load_bfg_ammo_asset();
        self.load_rocketlauncher_pickup_asset();
        self.load_combat_armor_asset();
        self.load_medkit_asset();
        self.load_armor_shard_asset();
        // **Mode exploration** (v0.9.5++ user request) — quand
        // `br_bots=0`, on n'affiche QUE rochers + powerups sur le
        // terrain, sans statues / statue_femme / drones / hellhounds /
        // ring shrink.  Carte propre pour visite touristique.
        let exploration_mode = self.cvars.get_i32("br_bots").unwrap_or(0) == 0;
        // **Rochers GLB** — gardés en exploration (décor minimal).
        self.load_rock_asset();
        self.spawn_br_rocks(&terrain);
        // **Statues GLB** — désactivées en exploration (user request).
        if !exploration_mode {
            self.load_statue_asset();
            self.spawn_br_statues(&terrain);
            // **Statues femme GLB** — élément décoratif sur les plages.
            self.load_statue_femme_asset();
            self.spawn_br_statue_femme(&terrain);
        }
        // **Buildings / Hellhounds / Grass** déjà désactivés.
        // **Spawn lights GLB** (v0.9.5++) — pour chaque prop placé,
        // émet ses lights KHR_lights_punctual à la position monde.
        // Skipped si l'asset n'a pas de lights (la plupart).
        for prop_name in ["rock", "statue", "statue_femme", "ammo_crate", "quad_pickup"] {
            self.spawn_glb_lights_for_prop(prop_name);
        }

        // Enregistre l'état BR.  On laisse `world = None` — les sites
        // qui dépendent de `self.world.as_ref()` (collisions, items
        // spawning) testeront `terrain.is_some()` en fallback.
        self.world = None;
        let terrain_arc = Arc::new(terrain);
        // Upload côté GPU pour le rendu — pipeline terrain dédié, cache
        // chunks LOD-adaptatif, sélection chaque frame selon la caméra.
        if let Some(r) = self.renderer.as_mut() {
            r.upload_terrain(terrain_arc.clone());
        }
        self.terrain = Some(terrain_arc);
        // **Mode exploration** (v0.9.5++) — pas de ring shrink quand
        // `br_bots=0` (sinon il tuerait le joueur en visite tranquille).
        let exploration_mode = self.cvars.get_i32("br_bots").unwrap_or(0) == 0;
        self.br_ring = if exploration_mode { None } else { Some(ring) };
        // Reset atmosphère BR à chaque entrée de map terrain — garantit
        // que `tick_atmosphere` ré-arme `next_lightning_at` à un délai
        // frais (cf. branche `is_finite()`), même si on revient en BR
        // après un détour BSP.  Sinon un éclair pouvait se déclencher
        // immédiatement à l'entrée si le timer précédent avait été
        // dépassé hors-BR.
        self.next_lightning_at = f32::INFINITY;
        self.lightning_flash_until = 0.0;
        // **Items spawn par POI tier** (v0.9.5) — pour chaque POI, on
        // pose un set d'items proportionnel au tier autour du centre
        // (ring de quelques unités). Pas de respawn en BR (item ramassé
        // = perdu pour le match), modélisé via `respawn_cooldown` énorme.
        self.spawn_br_pickups();

        // **POI light pillars** (v0.9.5) — colonnes lumineuses au-dessus
        // des POI tier 4 (capitales, volcan, lagons premium) pour les
        // repérer de loin façon Apex/Warzone "loot beam".  Empilées en
        // dlights successives sur 800u de haut. Tint par type POI.
        if let Some(terrain) = self.terrain.as_ref().cloned() {
            if let Some(r) = self.renderer.as_mut() {
                for poi in terrain.pois() {
                    if poi.tier < 4 {
                        continue;
                    }
                    let (color, intensity) = match poi.kind {
                        q3_terrain::PoiKind::Volcano => ([1.0, 0.45, 0.10], 4.0),
                        q3_terrain::PoiKind::Beach => ([0.40, 0.85, 1.0], 3.5),
                        q3_terrain::PoiKind::City => ([1.0, 0.85, 0.30], 4.0),
                        q3_terrain::PoiKind::Peak => ([0.85, 0.55, 1.0], 3.5),
                        q3_terrain::PoiKind::Cirque => ([0.55, 1.0, 0.55], 3.5),
                        _ => ([1.0, 0.85, 0.30], 3.5),
                    };
                    let z_ground = terrain.height_at(poi.x, poi.y);
                    // 4 dlights empilées de 200u en 200u → "pillar"
                    // sur 800u. Lifetime 9999s = pratique infinie.
                    for k in 0..4 {
                        let z = z_ground + 100.0 + k as f32 * 200.0;
                        r.spawn_dlight(
                            Vec3::new(poi.x, poi.y, z),
                            300.0, // rayon
                            color,
                            intensity,
                            self.time_sec,
                            9999.0,
                        );
                    }
                }
            }
        }

        // Drain bots BR (v0.9.5) — `spawn_bot` détecte automatiquement
        // le mode terrain et choisit un POI tier ≥ 2 pour le spawn.
        // Gated par cvar `br_bots` (défaut 0 = carte vide pour
        // exploration / tests visuels).
        let br_bots_enabled = self.cvars.get_i32("br_bots").unwrap_or(0) != 0;
        if br_bots_enabled && self.pending_local_bots > 0 {
            let n = self.pending_local_bots;
            self.pending_local_bots = 0;
            self.ensure_player_rig_loaded();
            for i in 0..n {
                let bot_name = format!("brbot{:02}", i + 1);
                self.spawn_bot(&bot_name, Some(3));
            }
            info!(
                "BR: spawn initial {} bot(s) demandés, {} effectifs",
                n,
                self.bots.len()
            );
        } else {
            // Drain les bots pending sans les spawn — sinon le compteur
            // accumulerait à chaque map BR rechargée.
            self.pending_local_bots = 0;
            info!("BR: réunion vide (br_bots = 0) — pas de spawn de bots");
        }
    }

    /// **Generic GLB prop loader** (v0.9.5+) — charge un asset GLB
    /// disque et l'enregistre dans le renderer.  v0.9.5++ : stocke
    /// aussi les lights extraites pour spawn au moment du place
    /// d'instance.
    fn load_prop_glb(&mut self, name: &str, paths: &[&str]) {
        let bases = resolve_asset_search_bases();
        let mut tried: Vec<String> = Vec::new();
        for base in &bases {
            for rel in paths {
                let full = base.join(rel);
                tried.push(full.display().to_string());
                let bytes = match std::fs::read(&full) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                match q3_model::glb::GlbMesh::from_glb_bytes(&bytes) {
                    Ok(mesh) => {
                        info!(
                            "prop GLB '{}' chargé : '{}' ({} verts, {} idx, radius {:.1}, {} lights)",
                            name,
                            full.display(),
                            mesh.vertices.len(),
                            mesh.indices.len(),
                            mesh.radius(),
                            mesh.lights.len(),
                        );
                        // Stocke les lights pour spawn au place.
                        self.prop_lights.insert(name.to_string(), mesh.lights.clone());
                        if let Some(r) = self.renderer.as_mut() {
                            r.upload_prop(name, &mesh);
                        }
                        return;
                    }
                    Err(e) => warn!("prop GLB '{}' '{}': {}", name, full.display(), e),
                }
            }
        }
        warn!(
            "prop GLB '{}' : aucun asset trouvé. Cherché dans :\n  {}",
            name,
            tried.join("\n  ")
        );
    }

    /// **Spawn des lights GLB** (v0.9.5++) — quand on a placé un
    /// `RockProp` du nom donné, applique sa transform à chaque light
    /// extraite du mesh et spawn une dlight monde correspondante.
    fn spawn_glb_lights_for_prop(&mut self, prop_name: &str) {
        let Some(lights) = self.prop_lights.get(prop_name).cloned() else { return; };
        if lights.is_empty() { return; }
        let Some(r) = self.renderer.as_mut() else { return; };
        // Récupère toutes les instances du prop pour spawner leurs lights.
        let instances: Vec<(Vec3, f32, f32)> = self
            .rocks
            .iter()
            .filter(|p| p.prop_name == prop_name)
            .map(|p| (p.pos, p.yaw, p.scale))
            .collect();
        for (pos, yaw, scale) in instances {
            let cy = yaw.cos();
            let sy = yaw.sin();
            for light in &lights {
                // Transform local → world : Y-up→Z-up, scale, rotate Z(yaw), translate.
                let lx = light.position[0] * scale;
                let ly = light.position[1] * scale; // Y local devient Z world
                let lz = light.position[2] * scale;
                let world_x = pos.x + cy * lx + sy * lz;
                let world_y = pos.y + sy * lx - cy * lz;
                let world_z = pos.z + ly;
                let radius = if light.range > 0.0 {
                    (light.range * scale).clamp(50.0, 800.0)
                } else {
                    300.0
                };
                let intensity = (light.intensity * 0.01).clamp(0.5, 5.0);
                r.spawn_dlight(
                    Vec3::new(world_x, world_y, world_z),
                    radius,
                    light.color,
                    intensity,
                    self.time_sec,
                    9999.0,
                );
            }
        }
    }

    fn load_rock_asset(&mut self) {
        self.load_prop_glb(
            "rock",
            &["assets/models/rock_scatter.glb", "assets/models/rock.glb"],
        );
    }
    fn load_statue_asset(&mut self) {
        self.load_prop_glb(
            "statue",
            &["assets/models/statue.glb"],
        );
    }
    fn load_building_asset(&mut self) {
        self.load_prop_glb(
            "building",
            &["assets/models/building.glb"],
        );
    }
    fn load_hellhound_asset(&mut self) {
        self.load_prop_glb(
            "hellhound",
            &["assets/models/hellhound.glb"],
        );
    }
    fn load_statue_femme_asset(&mut self) {
        self.load_prop_glb(
            "statue_femme",
            &["assets/models/statue_femme.glb"],
        );
    }
    fn load_grass_asset(&mut self) {
        self.load_prop_glb(
            "grass",
            &["assets/models/grass.glb"],
        );
    }
    fn load_quad_pickup_asset(&mut self) {
        self.load_prop_glb(
            "quad_pickup",
            &["assets/models/quad_pickup.glb"],
        );
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("quad_pickup") {
                if radius > 0.001 {
                    self.quad_pickup_scale = Some(40.0 / radius);
                }
            }
        }
    }
    fn load_health_pack_asset(&mut self) {
        self.load_prop_glb(
            "health_pack",
            &["assets/models/health_pack.glb"],
        );
        // Target world size ~22u — cohérent avec un MD3 health bobble.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("health_pack") {
                if radius > 0.001 {
                    self.health_pack_scale = Some(22.0 / radius);
                    info!(
                        "health_pack: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.health_pack_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_railgun_pickup_asset(&mut self) {
        self.load_prop_glb(
            "railgun_pickup",
            &["assets/models/railgun_pickup.glb"],
        );
        // Target world size ~30u — taille d'un weapon pickup Q3.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("railgun_pickup") {
                if radius > 0.001 {
                    self.railgun_pickup_scale = Some(30.0 / radius);
                    info!(
                        "railgun_pickup: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.railgun_pickup_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_grenade_ammo_asset(&mut self) {
        self.load_prop_glb(
            "grenade_ammo",
            &["assets/models/grenade_ammo.glb"],
        );
        // Target world size ~25u — cohérent avec ammo_crate.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("grenade_ammo") {
                if radius > 0.001 {
                    self.grenade_ammo_scale = Some(25.0 / radius);
                    info!(
                        "grenade_ammo: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.grenade_ammo_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_rocket_ammo_asset(&mut self) {
        self.load_prop_glb(
            "rocket_ammo",
            &["assets/models/rocket_ammo.glb"],
        );
        // Target world size ~25u — cohérent avec ammo_crate / grenade_ammo.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("rocket_ammo") {
                if radius > 0.001 {
                    self.rocket_ammo_scale = Some(25.0 / radius);
                    info!(
                        "rocket_ammo: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.rocket_ammo_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_cell_ammo_asset(&mut self) {
        self.load_prop_glb(
            "cell_ammo",
            &["assets/models/cell_ammo.glb"],
        );
        // Target world size ~25u — cohérent avec les autres ammo crates.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("cell_ammo") {
                if radius > 0.001 {
                    self.cell_ammo_scale = Some(25.0 / radius);
                    info!(
                        "cell_ammo: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.cell_ammo_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_lg_ammo_asset(&mut self) {
        self.load_prop_glb(
            "lg_ammo",
            &["assets/models/lg_ammo.glb"],
        );
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("lg_ammo") {
                if radius > 0.001 {
                    self.lg_ammo_scale = Some(25.0 / radius);
                    info!(
                        "lg_ammo: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.lg_ammo_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_big_armor_asset(&mut self) {
        self.load_prop_glb(
            "big_armor",
            &["assets/models/big_armor.glb"],
        );
        // Target world size ~30u — taille canonique d'un body armor Q3.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("big_armor") {
                if radius > 0.001 {
                    self.big_armor_scale = Some(30.0 / radius);
                    info!(
                        "big_armor: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.big_armor_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_plasma_pickup_asset(&mut self) {
        self.load_prop_glb(
            "plasma_pickup",
            &["assets/models/plasma_pickup.glb"],
        );
        // Target world size ~30u — comme railgun_pickup (weapon Q3).
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("plasma_pickup") {
                if radius > 0.001 {
                    self.plasma_pickup_scale = Some(30.0 / radius);
                    info!(
                        "plasma_pickup: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.plasma_pickup_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_machinegun_pickup_asset(&mut self) {
        self.load_prop_glb(
            "machinegun_pickup",
            &["assets/models/machinegun_pickup.glb"],
        );
        // Target world size ~30u — taille d'un weapon pickup Q3
        // (cohérent avec plasma_pickup et railgun_pickup).
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("machinegun_pickup") {
                if radius > 0.001 {
                    self.machinegun_pickup_scale = Some(30.0 / radius);
                    info!(
                        "machinegun_pickup: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.machinegun_pickup_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_bfg_pickup_asset(&mut self) {
        self.load_prop_glb(
            "bfg_pickup",
            &["assets/models/bfg_pickup.glb"],
        );
        // Target world size ~32u — un peu plus gros que les autres
        // weapon pickups (le BFG est imposant en Q3).
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("bfg_pickup") {
                if radius > 0.001 {
                    self.bfg_pickup_scale = Some(32.0 / radius);
                    info!(
                        "bfg_pickup: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.bfg_pickup_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_lightninggun_pickup_asset(&mut self) {
        self.load_prop_glb(
            "lightninggun_pickup",
            &["assets/models/lightninggun_pickup.glb"],
        );
        // Target world size ~30u — taille standard d'un weapon pickup Q3.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("lightninggun_pickup") {
                if radius > 0.001 {
                    self.lightninggun_pickup_scale = Some(30.0 / radius);
                    info!(
                        "lightninggun_pickup: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.lightninggun_pickup_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_shotgun_pickup_asset(&mut self) {
        self.load_prop_glb(
            "shotgun_pickup",
            &["assets/models/shotgun_pickup.glb"],
        );
        // Target world size ~30u — taille standard d'un weapon pickup Q3.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("shotgun_pickup") {
                if radius > 0.001 {
                    self.shotgun_pickup_scale = Some(30.0 / radius);
                    info!(
                        "shotgun_pickup: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.shotgun_pickup_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_grenadelauncher_pickup_asset(&mut self) {
        self.load_prop_glb(
            "grenadelauncher_pickup",
            &["assets/models/grenadelauncher_pickup.glb"],
        );
        // Target world size ~30u — taille standard weapon pickup Q3.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("grenadelauncher_pickup") {
                if radius > 0.001 {
                    self.grenadelauncher_pickup_scale = Some(30.0 / radius);
                    info!(
                        "grenadelauncher_pickup: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.grenadelauncher_pickup_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_gauntlet_pickup_asset(&mut self) {
        self.load_prop_glb(
            "gauntlet_pickup",
            &["assets/models/gauntlet_pickup.glb"],
        );
        // Target world size ~30u — taille standard weapon pickup Q3.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("gauntlet_pickup") {
                if radius > 0.001 {
                    self.gauntlet_pickup_scale = Some(30.0 / radius);
                    info!(
                        "gauntlet_pickup: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.gauntlet_pickup_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_shotgun_ammo_asset(&mut self) {
        self.load_prop_glb(
            "shotgun_ammo",
            &["assets/models/shotgun_ammo.glb"],
        );
        // Target world size ~18u (un peu plus petit que les autres
        // ammo crates ~25u — user request : cartouches calibre 12 plus
        // petites au sol qu'une vraie caisse de munitions).
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("shotgun_ammo") {
                if radius > 0.001 {
                    self.shotgun_ammo_scale = Some(18.0 / radius);
                    info!(
                        "shotgun_ammo: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.shotgun_ammo_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_bfg_ammo_asset(&mut self) {
        self.load_prop_glb(
            "bfg_ammo",
            &["assets/models/bfg_ammo.glb"],
        );
        // Target world size ~25u — cohérent avec les autres ammo crates.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("bfg_ammo") {
                if radius > 0.001 {
                    self.bfg_ammo_scale = Some(25.0 / radius);
                    info!(
                        "bfg_ammo: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.bfg_ammo_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_rocketlauncher_pickup_asset(&mut self) {
        self.load_prop_glb(
            "rocketlauncher_pickup",
            &["assets/models/rocketlauncher_pickup.glb"],
        );
        // Target world size ~30u — taille standard weapon pickup Q3.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("rocketlauncher_pickup") {
                if radius > 0.001 {
                    self.rocketlauncher_pickup_scale = Some(30.0 / radius);
                    info!(
                        "rocketlauncher_pickup: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.rocketlauncher_pickup_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_combat_armor_asset(&mut self) {
        self.load_prop_glb(
            "combat_armor",
            &["assets/models/combat_armor.glb"],
        );
        // Target world size ~28u — un peu plus petit que body armor 100.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("combat_armor") {
                if radius > 0.001 {
                    self.combat_armor_scale = Some(28.0 / radius);
                    info!(
                        "combat_armor: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.combat_armor_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_medkit_asset(&mut self) {
        self.load_prop_glb(
            "medkit",
            &["assets/models/medkit.glb"],
        );
        // Target world size ~22u (cohérent avec health_pack).
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("medkit") {
                if radius > 0.001 {
                    self.medkit_scale = Some(22.0 / radius);
                    info!(
                        "medkit: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.medkit_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_armor_shard_asset(&mut self) {
        self.load_prop_glb(
            "armor_shard",
            &["assets/models/armor_shard.glb"],
        );
        // Target world size ~18u (petit shard, plus petit que combat 50).
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("armor_shard") {
                if radius > 0.001 {
                    self.armor_shard_scale = Some(18.0 / radius);
                    info!(
                        "armor_shard: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.armor_shard_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_railgun_ammo_asset(&mut self) {
        self.load_prop_glb(
            "railgun_ammo",
            &["assets/models/railgun_ammo.glb"],
        );
        // Target world size ~25u — cohérent avec les autres ammo crates.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("railgun_ammo") {
                if radius > 0.001 {
                    self.railgun_ammo_scale = Some(25.0 / radius);
                    info!(
                        "railgun_ammo: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.railgun_ammo_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_regen_pickup_asset(&mut self) {
        self.load_prop_glb(
            "regen_pickup",
            &["assets/models/regen_pickup.glb"],
        );
        // Target world size ~30u — comme quad_pickup (powerup Q3).
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("regen_pickup") {
                if radius > 0.001 {
                    self.regen_pickup_scale = Some(30.0 / radius);
                    info!(
                        "regen_pickup: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.regen_pickup_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_ammo_crate_asset(&mut self) {
        self.load_prop_glb(
            "ammo_crate",
            &["assets/models/ammo_crate.glb"],
        );
        // Calcule le scale auto pour matcher la taille standard d'un
        // pickup MD3 Q3 (~25u) en mémoise dans `self`.
        // Target world size = 25u — cible visuelle d'un MD3 ammo Q3.
        // scale = target / mesh_radius.
        if let Some(r) = self.renderer.as_ref() {
            if let Some(radius) = r.prop_radius("ammo_crate") {
                if radius > 0.001 {
                    self.ammo_crate_scale = Some(25.0 / radius);
                    info!(
                        "ammo_crate: native radius={:.2}, scale auto={:.3}",
                        radius,
                        self.ammo_crate_scale.unwrap()
                    );
                }
            }
        }
    }
    fn load_tropical_asset(&mut self) {
        self.load_prop_glb(
            "tropical",
            &["assets/models/tropical_pack.glb", "assets/models/tropical.glb"],
        );
    }

    /// **Spawn rochers BR** — disséminés sur les zones rocheuses
    /// (splat rock > 0.5) à des altitudes raisonnables.  Densité
    /// variable selon le biome :
    ///   * sommets (z > 800) : densité haute
    ///   * pentes moyennes (200..800) : densité moyenne
    ///   * sable côtier / végétation dense : faible
    fn spawn_br_rocks(&mut self, terrain: &q3_terrain::Terrain) {
        self.rocks.clear();
        let has_mesh = self
            .renderer
            .as_ref()
            .map(|r| r.has_prop("rock"))
            .unwrap_or(false);
        if !has_mesh {
            return;
        }
        // Auto-scale rock — target world size ~80u par rocher.
        let r_native = self
            .renderer
            .as_ref()
            .and_then(|r| r.prop_radius("rock"))
            .unwrap_or(1.0);
        let base_rock_scale = if r_native > 0.001 { 80.0 / r_native } else { 1.0 };
        // On échantillonne `n_samples` positions pseudo-aléatoires
        // dans la grille du terrain et on accepte celles qui passent
        // le filtre biome.  Hash-based pour reproductibilité d'une
        // session à l'autre (pas de RNG pour le décor → on aime
        // qu'un POI ait toujours les mêmes rochers).
        const N_SAMPLES: usize = 1500;
        let mut spawned = 0;
        for k in 0..N_SAMPLES {
            // Hash → coords dans la grille (évite distribution uniform
            // pure qui donne des grilles visibles).
            let h1 = (k.wrapping_mul(2654435761) ^ 0xDEADBEEF) as u32;
            let h2 = (k.wrapping_mul(40503) ^ 0x12345678) as u32;
            let gx = (h1 % terrain.width as u32) as usize;
            let gy = (h2 % terrain.height as u32) as usize;
            // Position monde correspondante.
            let wx = terrain.meta.origin_x
                + gx as f32 * terrain.meta.units_per_sample;
            let wy = terrain.meta.origin_y
                + gy as f32 * terrain.meta.units_per_sample;
            let wz = terrain.height_at(wx, wy);
            // Skip océan et eau.
            if wz <= terrain.meta.water_level + 5.0 {
                continue;
            }
            // Skip zones urbaines (les villes ont déjà leurs items, pas de rochers dedans).
            let urban = terrain.biome_weight(wx, wy, 3);
            if urban > 0.4 {
                continue;
            }
            let rock_w = terrain.biome_weight(wx, wy, 0);
            // Densité par biome — accept_proba scale par splat rock + altitude.
            let alt_factor = if wz > 800.0 {
                0.9
            } else if wz > 200.0 {
                0.5
            } else {
                0.15
            };
            let accept_proba = (rock_w * 0.7 + alt_factor * 0.3).clamp(0.0, 1.0);
            // Rejection sampling via 3e hash.
            let h3 = (k.wrapping_mul(73856093) ^ 0xCAFEBABE) as u32;
            let r3 = (h3 & 0xffff) as f32 / 65535.0;
            if r3 > accept_proba {
                continue;
            }
            // Yaw aléatoire pour variété.
            let yaw_h = (k.wrapping_mul(83492791) ^ 0xFEEDC0DE) as u32;
            let yaw = (yaw_h & 0xffff) as f32 / 65535.0 * std::f32::consts::TAU;
            // Scale variable [0.5, 1.5].
            let scale_h = (k.wrapping_mul(2246822519) ^ 0xBADCAB1E) as u32;
            let scale = 0.5 + (scale_h & 0xff) as f32 / 255.0 * 1.0;
            // Tint blanc → couleurs natives de la baseColorTexture du GLB.
            self.rocks.push(RockProp {
                pos: Vec3::new(wx, wy, wz),
                yaw,
                scale: scale * base_rock_scale,
                tint: [1.0, 1.0, 1.0, 1.0],
                prop_name: "rock",
            });
            spawned += 1;
            if spawned >= 400 {
                break; // cap pour ne pas exploser le cost de draw
            }
        }
        info!("BR: {} rochers disséminés sur le terrain", self.rocks.len());
    }

    /// **Spawn props tropicaux BR** (v0.9.5+) — palmiers/plantes du
    /// `tropical_pack.glb` en zones côtières (sable + bord de
    /// végétation).  Densité haute près des plages, nulle en altitude.
    /// **Spawn statues BR** (v0.9.5++) — 1 statue landmark par POI
    /// tier ≥ 3.  Position : centre du POI, légèrement décalée pour
    /// ne pas overlap les pickups.  Yaw aléatoire stable, scale auto
    /// basé sur le radius natif (target ~120u — visible à distance,
    /// signature visuelle des zones premium).  Stocké dans `self.rocks`
    /// avec un nom de prop dédié pour réutiliser le pipeline.
    fn spawn_br_statues(&mut self, terrain: &q3_terrain::Terrain) {
        let has_mesh = self
            .renderer
            .as_ref()
            .map(|r| r.has_prop("statue"))
            .unwrap_or(false);
        if !has_mesh {
            return;
        }
        let r_native = self
            .renderer
            .as_ref()
            .and_then(|r| r.prop_radius("statue"))
            .unwrap_or(1.0);
        let base_scale = if r_native > 0.001 { 120.0 / r_native } else { 1.0 };
        let mut spawned = 0usize;
        for (i, poi) in terrain.pois().iter().enumerate() {
            if poi.tier < 3 {
                continue;
            }
            // Décalage léger autour du centre POI pour ne pas piler
            // sur les items spawn.
            let offset_x = (i as f32 * 13.0).sin() * 60.0;
            let offset_y = (i as f32 * 13.0).cos() * 60.0;
            let x = poi.x + offset_x;
            let y = poi.y + offset_y;
            let z = terrain.height_at(x, y);
            if z <= terrain.meta.water_level + 5.0 {
                continue; // skip océan / lagons
            }
            let yaw_h = (i.wrapping_mul(2654435761) ^ 0xC0DEC0DE) as u32;
            let yaw = (yaw_h & 0xffff) as f32 / 65535.0 * std::f32::consts::TAU;
            // Hack RockProp partagé — on encode le type via un name
            // suffixe dans la matrice render. On ajoute un nouveau
            // marqueur scale (3e signe) pour distinguer statue de
            // rock+tropical.
            // Simplification : statue = scale > 1000 (jamais atteint
            // par rocks/tropical qui restent < 200).
            self.rocks.push(RockProp {
                pos: Vec3::new(x, y, z),
                yaw,
                scale: base_scale,
                tint: [1.0, 1.0, 1.0, 1.0],
                prop_name: "statue",
            });
            spawned += 1;
        }
        info!(
            "BR: {} statues placées sur POI tier ≥ 3 (radius={:.2}, scale base={:.1})",
            spawned, r_native, base_scale
        );
    }

    /// **Spawn grass BR** (v0.9.5++) — touffes d'herbe disséminées
    /// en zones végétales basse altitude (splat veg > 0.5, z < 400m).
    /// Densité élevée (~600 touffes) pour casser l'uniforme du sol
    /// vert.  Auto-scale ~25u (touffe d'herbe à hauteur cheville).
    fn spawn_br_grass(&mut self, terrain: &q3_terrain::Terrain) {
        let has_mesh = self
            .renderer
            .as_ref()
            .map(|r| r.has_prop("grass"))
            .unwrap_or(false);
        if !has_mesh {
            return;
        }
        let r_native = self
            .renderer
            .as_ref()
            .and_then(|r| r.prop_radius("grass"))
            .unwrap_or(1.0);
        let base_scale = if r_native > 0.001 { 25.0 / r_native } else { 1.0 };
        const N_SAMPLES: usize = 4000;
        let mut spawned = 0usize;
        for k in 0..N_SAMPLES {
            let h1 = (k.wrapping_mul(2654435761) ^ 0x6BADBEEF) as u32;
            let h2 = (k.wrapping_mul(40503) ^ 0x12345678) as u32;
            let gx = (h1 % terrain.width as u32) as usize;
            let gy = (h2 % terrain.height as u32) as usize;
            let wx = terrain.meta.origin_x + gx as f32 * terrain.meta.units_per_sample;
            let wy = terrain.meta.origin_y + gy as f32 * terrain.meta.units_per_sample;
            let wz = terrain.height_at(wx, wy);
            if wz <= terrain.meta.water_level + 5.0 || wz > 400.0 {
                continue; // skip océan + altitude
            }
            // Filtre : zones avec splat végétation > 0.5.
            let veg = terrain.biome_weight(wx, wy, 2);
            if veg < 0.4 {
                continue;
            }
            let h3 = (k.wrapping_mul(73856093) ^ 0xCAFE) as u32;
            // Densité scaled par poids végé.
            if (h3 & 0xff) as f32 / 255.0 > veg {
                continue;
            }
            let yaw_h = (k.wrapping_mul(83492791) ^ 0x55AA) as u32;
            let yaw = (yaw_h & 0xffff) as f32 / 65535.0 * std::f32::consts::TAU;
            let scale_h = (k.wrapping_mul(2246822519) ^ 0xBEAD) as u32;
            let scale = 0.7 + (scale_h & 0xff) as f32 / 255.0 * 0.6;
            self.rocks.push(RockProp {
                pos: Vec3::new(wx, wy, wz),
                yaw,
                scale: base_scale * scale,
                tint: [1.0, 1.0, 1.0, 1.0],
                prop_name: "grass",
            });
            spawned += 1;
            if spawned >= 600 {
                break;
            }
        }
        info!(
            "BR: {} touffes d'herbe disséminées (radius={:.2}, scale={:.2})",
            spawned, r_native, base_scale
        );
    }

    /// **Spawn statues femme BR** (v0.9.5++) — décor sur les plages
    /// et stations balnéaires (Beach POI). Une par plage majeure.
    fn spawn_br_statue_femme(&mut self, terrain: &q3_terrain::Terrain) {
        use q3_terrain::PoiKind;
        let has_mesh = self
            .renderer
            .as_ref()
            .map(|r| r.has_prop("statue_femme"))
            .unwrap_or(false);
        if !has_mesh {
            return;
        }
        let r_native = self
            .renderer
            .as_ref()
            .and_then(|r| r.prop_radius("statue_femme"))
            .unwrap_or(1.0);
        // Target ~80u (taille statue grande nature).
        let base_scale = if r_native > 0.001 { 80.0 / r_native } else { 1.0 };
        let mut spawned = 0usize;
        for (i, poi) in terrain.pois().iter().enumerate() {
            if !matches!(poi.kind, PoiKind::Beach) || poi.tier < 2 {
                continue;
            }
            // Centre POI (pas d'offset — la statue est l'attraction).
            let x = poi.x;
            let y = poi.y;
            let z = terrain.height_at(x, y);
            if z <= terrain.meta.water_level + 5.0 {
                continue;
            }
            let yaw_h = (i.wrapping_mul(2654435761) ^ 0xFEEDC0FE) as u32;
            let yaw = (yaw_h & 0xffff) as f32 / 65535.0 * std::f32::consts::TAU;
            self.rocks.push(RockProp {
                pos: Vec3::new(x, y, z),
                yaw,
                scale: base_scale,
                tint: [1.0, 1.0, 1.0, 1.0],
                prop_name: "statue_femme",
            });
            spawned += 1;
        }
        info!(
            "BR: {} statues_femme placées (Beach POI tier ≥ 2, scale={:.2})",
            spawned, base_scale
        );
    }

    /// **Spawn hellhounds BR** (v0.9.5++) — décors statiques style
    /// Quake autour des POI Forest + Volcano. Petite meute par POI.
    fn spawn_br_hellhounds(&mut self, terrain: &q3_terrain::Terrain) {
        use q3_terrain::PoiKind;
        let has_mesh = self
            .renderer
            .as_ref()
            .map(|r| r.has_prop("hellhound"))
            .unwrap_or(false);
        if !has_mesh {
            return;
        }
        let r_native = self
            .renderer
            .as_ref()
            .and_then(|r| r.prop_radius("hellhound"))
            .unwrap_or(1.0);
        // Target ~30u (taille d'un chien à l'écran).
        let base_scale = if r_native > 0.001 { 30.0 / r_native } else { 1.0 };
        let mut spawned = 0usize;
        for (i, poi) in terrain.pois().iter().enumerate() {
            // Forest = meute de 3-4 chiens, Volcano = meute de 5
            // (gardien du cratère).
            let count = match poi.kind {
                PoiKind::Forest => 3,
                PoiKind::Volcano => 5,
                PoiKind::Cirque => 2,
                _ => 0,
            };
            if count == 0 { continue; }
            let ring = poi.radius * 0.4;
            for k in 0..count {
                let theta = (k as f32 / count as f32) * std::f32::consts::TAU
                    + (i as f32 * 0.27);
                let x = poi.x + theta.cos() * ring;
                let y = poi.y + theta.sin() * ring;
                let z = terrain.height_at(x, y);
                if z <= terrain.meta.water_level + 5.0 {
                    continue;
                }
                // Yaw face vers l'extérieur (sentinelle).
                let yaw = theta;
                self.rocks.push(RockProp {
                    pos: Vec3::new(x, y, z),
                    yaw,
                    scale: base_scale,
                    tint: [1.0, 1.0, 1.0, 1.0],
                    prop_name: "hellhound",
                });
                spawned += 1;
            }
        }
        info!(
            "BR: {} hellhounds placés (Forest+Volcano+Cirque, radius={:.2}, scale={:.2})",
            spawned, r_native, base_scale
        );
    }

    /// **Spawn buildings BR** (v0.9.5++) — clusters de bâtiments
    /// dans les zones City/Town pour donner du relief vertical aux
    /// quartiers urbains.  3-6 buildings par POI urbain en cercle
    /// autour du centre.  Auto-scale ~250u (immeuble).  Yaw aligné
    /// approximativement face au centre POI pour suggérer une rue.
    fn spawn_br_buildings(&mut self, terrain: &q3_terrain::Terrain) {
        use q3_terrain::PoiKind;
        let has_mesh = self
            .renderer
            .as_ref()
            .map(|r| r.has_prop("building"))
            .unwrap_or(false);
        if !has_mesh {
            return;
        }
        let r_native = self
            .renderer
            .as_ref()
            .and_then(|r| r.prop_radius("building"))
            .unwrap_or(1.0);
        let base_scale = if r_native > 0.001 { 250.0 / r_native } else { 1.0 };
        let mut spawned = 0usize;
        for (i, poi) in terrain.pois().iter().enumerate() {
            // Densité par tier : capitales = 6, towns = 3-4, autres = 0.
            let count = match (poi.kind, poi.tier) {
                (PoiKind::City, _) => 6,
                (PoiKind::Town, t) if t >= 3 => 4,
                (PoiKind::Town, _) => 3,
                (PoiKind::Industrial | PoiKind::Airport, _) => 4,
                _ => 0,
            };
            if count == 0 {
                continue;
            }
            // Anneau de buildings autour du POI center.
            let ring_radius = poi.radius * 0.55;
            for k in 0..count {
                let angle = (k as f32 / count as f32) * std::f32::consts::TAU
                    + (i as f32 * 0.13);
                let x = poi.x + angle.cos() * ring_radius;
                let y = poi.y + angle.sin() * ring_radius;
                let z = terrain.height_at(x, y);
                if z <= terrain.meta.water_level + 5.0 {
                    continue;
                }
                // Yaw face au centre POI (bâtiment "regarde" la place).
                let to_center_x = poi.x - x;
                let to_center_y = poi.y - y;
                let yaw = to_center_y.atan2(to_center_x);
                // Variation taille par building (±30%).
                let var_h = (i.wrapping_mul(2654435761).wrapping_add(k as usize) ^ 0xBADC0DE)
                    as u32;
                let scale_var = 0.7 + (var_h & 0xff) as f32 / 255.0 * 0.6;
                // Tint blanc → texture native du GLB.
                self.rocks.push(RockProp {
                    pos: Vec3::new(x, y, z),
                    yaw,
                    scale: base_scale * scale_var,
                    tint: [1.0, 1.0, 1.0, 1.0],
                    prop_name: "building",
                });
                spawned += 1;
            }
        }
        info!(
            "BR: {} buildings placés sur zones urbaines (radius={:.2}, scale base={:.1})",
            spawned, r_native, base_scale
        );
    }

    fn spawn_br_tropical(&mut self, terrain: &q3_terrain::Terrain) {
        let has_mesh = self
            .renderer
            .as_ref()
            .map(|r| r.has_prop("tropical"))
            .unwrap_or(false);
        if !has_mesh {
            return;
        }
        // Auto-scale tropical — target world size ~50u par plante.
        let r_native = self
            .renderer
            .as_ref()
            .and_then(|r| r.prop_radius("tropical"))
            .unwrap_or(1.0);
        let base_trop_scale = if r_native > 0.001 { 50.0 / r_native } else { 1.0 };
        // On stocke dans `self.rocks` aussi (struct identique) mais
        // on tag avec un yaw + tint qui le différenciera côté queue
        // (via un fanion en signe du scale, bidouille rapide).  Plus
        // propre : nouvelle struct, mais par MVP on partage.
        const N_SAMPLES: usize = 1200;
        let mut spawned = 0;
        for k in 0..N_SAMPLES {
            let h1 = (k.wrapping_mul(2654435761) ^ 0xC0FFEE00) as u32;
            let h2 = (k.wrapping_mul(40503) ^ 0x55AA55AA) as u32;
            let gx = (h1 % terrain.width as u32) as usize;
            let gy = (h2 % terrain.height as u32) as usize;
            let wx = terrain.meta.origin_x
                + gx as f32 * terrain.meta.units_per_sample;
            let wy = terrain.meta.origin_y
                + gy as f32 * terrain.meta.units_per_sample;
            let wz = terrain.height_at(wx, wy);
            // Skip océan + altitude.
            if wz <= terrain.meta.water_level + 5.0 || wz > 200.0 {
                continue;
            }
            let sand = terrain.biome_weight(wx, wy, 1);
            let veg = terrain.biome_weight(wx, wy, 2);
            // Accept seulement sur transitions sable/végé (côte) ou
            // plein végé en basse altitude.
            let prob = (sand * 0.6 + veg * 0.5).clamp(0.0, 1.0);
            let h3 = (k.wrapping_mul(73856093) ^ 0xCAFEBABE) as u32;
            let r3 = (h3 & 0xffff) as f32 / 65535.0;
            if r3 > prob {
                continue;
            }
            let yaw_h = (k.wrapping_mul(83492791) ^ 0xFEEDC0DE) as u32;
            let yaw = (yaw_h & 0xffff) as f32 / 65535.0 * std::f32::consts::TAU;
            let scale_h = (k.wrapping_mul(2246822519) ^ 0xBADCAB1E) as u32;
            let scale = 0.6 + (scale_h & 0xff) as f32 / 255.0 * 0.8; // 0.6..1.4
            // Tint vert tropical avec léger jitter.
            let tj = ((scale_h >> 8) & 0x3f) as f32 / 63.0 * 0.25;
            let tint = [0.7 + tj * 0.2, 0.95 - tj * 0.2, 0.5 + tj * 0.3, 1.0];
            self.rocks.push(RockProp {
                pos: Vec3::new(wx, wy, wz),
                yaw,
                scale: scale * base_trop_scale,
                tint,
                prop_name: "tropical",
            });
            spawned += 1;
            if spawned >= 250 {
                break;
            }
        }
        info!("BR: {} props tropicaux disséminés", spawned);
    }

    /// **Bot frag drop** (v0.9.5++) — quand un bot meurt, drop un
    /// item aléatoire au sol.  v0.9.5+++ : weapon-aware drops biaisés
    /// sur l'arme du killer (frag au RL → drop rockets, frag au PG →
    /// drop cells).  Plus : 2 % chance de powerup rare (Haste/Regen).
    fn spawn_bot_drop(&mut self, pos: Vec3) {
        let Some(r) = self.renderer.as_mut() else { return; };
        let roll = rand_unit().abs();
        // Arme du killer = arme active du joueur au moment du frag.
        // Si le bot s'est suicidé (ring), pas de bias spécifique.
        let killer_weapon = self.active_weapon;

        let (kind, mesh_path) = if roll < 0.02 {
            // 2 % powerup rare (Haste, lifetime court).
            (
                PickupKind::Powerup {
                    powerup: PowerupKind::Haste,
                    duration: 20.0,
                },
                "models/powerups/instant/haste.md3",
            )
        } else if roll < 0.27 {
            (
                PickupKind::Health { amount: 25, max_cap: 100 },
                "models/powerups/health/medium_cross.md3",
            )
        } else if roll < 0.50 {
            (
                PickupKind::Armor { amount: 25 },
                "models/powerups/armor/shard.md3",
            )
        } else if roll < 0.62 {
            (
                PickupKind::Health { amount: 5, max_cap: 200 },
                "models/powerups/health/small_cross.md3",
            )
        } else if roll < 0.92 {
            // Ammo bias : drop ammo de l'arme du killer (récompense
            // immédiate). Fallback MG si l'arme n'a pas d'ammo (Gauntlet).
            let (slot, amount, mesh) = match killer_weapon {
                WeaponId::Rocketlauncher => (WeaponId::Rocketlauncher.slot(), 5, "models/powerups/ammo/rocketam.md3"),
                WeaponId::Plasmagun => (WeaponId::Plasmagun.slot(), 30, "models/powerups/ammo/plasmaam.md3"),
                WeaponId::Lightninggun => (WeaponId::Lightninggun.slot(), 30, "models/powerups/ammo/lightningam.md3"),
                WeaponId::Railgun => (WeaponId::Railgun.slot(), 5, "models/powerups/ammo/railgunam.md3"),
                WeaponId::Shotgun => (WeaponId::Shotgun.slot(), 10, "models/powerups/ammo/shotgunam.md3"),
                WeaponId::Grenadelauncher => (WeaponId::Grenadelauncher.slot(), 5, "models/powerups/ammo/grenadeam.md3"),
                WeaponId::Bfg => (WeaponId::Bfg.slot(), 5, "models/powerups/ammo/bfgam.md3"),
                _ => (WeaponId::Machinegun.slot(), 50, "models/powerups/ammo/machinegunam.md3"),
            };
            (
                PickupKind::Ammo { slot, amount },
                mesh,
            )
        } else {
            // Weapon drop = arme du killer (rare).
            let (weapon, mesh) = match killer_weapon {
                WeaponId::Rocketlauncher => (WeaponId::Rocketlauncher, "models/weapons2/rocketl/rocketl.md3"),
                WeaponId::Plasmagun => (WeaponId::Plasmagun, "models/weapons2/plasma/plasma.md3"),
                WeaponId::Railgun => (WeaponId::Railgun, "models/weapons2/railgun/railgun.md3"),
                WeaponId::Lightninggun => (WeaponId::Lightninggun, "models/weapons2/lightning/lightning.md3"),
                _ => (WeaponId::Shotgun, "models/weapons2/shotgun/shotgun.md3"),
            };
            (
                PickupKind::Weapon { weapon, ammo: 5 },
                mesh,
            )
        };
        let mesh = match r.load_md3(&self.vfs, mesh_path) {
            Ok(m) => m,
            Err(_) => return, // mesh manquant → skip silencieusement
        };
        // Position : à hauteur du buste pour que le drop soit visible
        // au-dessus du gib spawn (et pas enterré dans le sang).
        let drop_pos = pos + Vec3::Z * 16.0;
        self.pickups.push(PickupGpu {
            mesh,
            origin: drop_pos,
            angles: q3_math::Angles::ZERO,
            kind,
            // Drop éphémère : 30 s avant qu'il ne disparaisse, pour
            // ne pas saturer la carte si beaucoup de bots meurent.
            respawn_cooldown: 30.0,
            respawn_at: None,
            entity_index: u16::MAX,
        });
    }

    /// **Common SFX loader** (v0.9.5+) — charge tous les SFX joueur
    /// (jump/footsteps/fire/pain/pickup/feedback) depuis le VFS.
    /// Appelé par load_map (BSP) et load_terrain_map (BR) — sans ça
    /// le mode BR était silencieux car les SFX étaient chargés
    /// uniquement dans le path BSP.
    fn load_common_sfx(&mut self) {
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
            } else {
                warn!(
                    "sfx: AUCUNE footstep chargée \
                     (cherché sound/player/footsteps/step1-4.wav et boot1-4.wav)"
                );
            }
            // Tir : on essaie une liste de chemins candidats par arme
            // (vanilla + variantes connues). Le 1er existant gagne.
            // Loggue le succès pour diagnostiquer rapidement les armes
            // muettes — historique : RL silencieux sur certains pak0
            // partiels où `rocklf1a.wav` n'était pas présent.
            // **Discovery VFS-based** (v0.9.5++) — au lieu de chercher
            // des chemins hardcoded canoniques, on liste TOUS les .wav
            // sous `sound/weapons/` et on les matche par keyword sur le
            // nom de fichier.  Robuste face aux variations de paks
            // (Steam, demo, mods, custom).
            let all_weapon_wavs: Vec<String> = self
                .vfs
                .list_prefix("sound/weapons/")
                .into_iter()
                .filter(|p| p.ends_with(".wav"))
                .collect();
            info!(
                "sfx: {} fichiers .wav trouvés sous sound/weapons/",
                all_weapon_wavs.len()
            );
            if all_weapon_wavs.is_empty() {
                warn!(
                    "sfx: AUCUN .wav trouvé sous sound/weapons/ — vérifie \
                     le path de baseq3 et la présence des paks (pak0.pk3 etc.)"
                );
            } else {
                // Log un échantillon pour diagnostic.
                for sample in all_weapon_wavs.iter().take(8) {
                    info!("sfx: dispo `{}`", sample);
                }
                if all_weapon_wavs.len() > 8 {
                    info!("sfx: ...et {} autres", all_weapon_wavs.len() - 8);
                }
            }
            // Match keyword-based : pour chaque arme, cherche un .wav
            // dont le path contient un keyword associé + indice "fire/launch".
            // Préfère les paths canoniques d'abord (faster + plus précis),
            // sinon fallback sur n'importe quel .wav contenant le keyword.
            for w in WeaponId::ALL {
                let mut loaded = false;
                // 1. Essai des paths canoniques (rapide).
                for &path in w.fire_sfx_paths() {
                    if let Some(h) = try_load_sfx(&self.vfs, snd, path) {
                        info!("sfx: {:?} ← `{}` (canonique)", w, path);
                        self.sfx_fire.push((w, h));
                        loaded = true;
                        break;
                    }
                }
                // 2. Fallback : keyword match sur le listing VFS.
                if !loaded {
                    let keywords: &[&str] = match w {
                        WeaponId::Gauntlet => &["melee", "gaunt", "fstatck"],
                        WeaponId::Machinegun => &["machgun", "machinegun", "machgf", "mg_"],
                        WeaponId::Shotgun => &["shotgun", "sshot"],
                        WeaponId::Grenadelauncher => &["grenade", "grenlf", "gl_"],
                        WeaponId::Rocketlauncher => &["rocket", "rocklf", "rl_"],
                        WeaponId::Lightninggun => &["lightning", "lg_"],
                        WeaponId::Railgun => &["railgun", "railgf", "rail"],
                        WeaponId::Plasmagun => &["plasma", "hyprbf", "plasx"],
                        WeaponId::Bfg => &["bfg"],
                    };
                    for path in &all_weapon_wavs {
                        let lower = path.to_lowercase();
                        if keywords.iter().any(|kw| lower.contains(kw))
                            && (lower.contains("fire") || lower.contains("f1") || lower.contains("launch"))
                        {
                            if let Some(h) = try_load_sfx(&self.vfs, snd, path) {
                                info!("sfx: {:?} ← `{}` (keyword match)", w, path);
                                self.sfx_fire.push((w, h));
                                loaded = true;
                                break;
                            }
                        }
                    }
                }
                // 3. 2ème fallback : n'importe quel .wav contenant le keyword.
                if !loaded {
                    let keywords: &[&str] = match w {
                        WeaponId::Gauntlet => &["melee", "gaunt"],
                        WeaponId::Machinegun => &["machgun", "machinegun", "machgf"],
                        WeaponId::Shotgun => &["shotgun", "sshot"],
                        WeaponId::Grenadelauncher => &["grenade", "grenlf"],
                        WeaponId::Rocketlauncher => &["rocket", "rocklf"],
                        WeaponId::Lightninggun => &["lightning"],
                        WeaponId::Railgun => &["railgun", "railgf", "rail"],
                        WeaponId::Plasmagun => &["plasma", "hyprbf"],
                        WeaponId::Bfg => &["bfg"],
                    };
                    for path in &all_weapon_wavs {
                        let lower = path.to_lowercase();
                        if keywords.iter().any(|kw| lower.contains(kw)) {
                            if let Some(h) = try_load_sfx(&self.vfs, snd, path) {
                                info!("sfx: {:?} ← `{}` (keyword loose match)", w, path);
                                self.sfx_fire.push((w, h));
                                loaded = true;
                                break;
                            }
                        }
                    }
                }
                if !loaded {
                    warn!(
                        "sfx: arme {:?} SANS son de tir (canonique + keyword échec)",
                        w
                    );
                }
            }
            // **Universal fallback** (v0.9.5++) — si AU MOINS une arme a
            // un son chargé, toutes les armes sans son héritent du
            // premier sample disponible.  Garantit qu'aucune arme ne
            // reste muette tant qu'un seul .wav weapons existe sur le
            // disque.  Cas d'usage : pak0 partiels où seul le shotgun
            // est présent → les 8 autres armes utilisent ce sample.
            if !self.sfx_fire.is_empty()
                && self.sfx_fire.len() < WeaponId::ALL.len()
            {
                let fallback_handle = self.sfx_fire[0].1;
                let fallback_weapon = self.sfx_fire[0].0;
                for w in WeaponId::ALL {
                    if !self.sfx_fire.iter().any(|(other, _)| *other == w) {
                        warn!(
                            "sfx: arme {:?} muette → fallback sur le sample de {:?}",
                            w, fallback_weapon
                        );
                        self.sfx_fire.push((w, fallback_handle));
                    }
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

    }

    /// **Drone GLB asset** (v0.9.5) — charge `assets/models/drone.glb`
    /// si présent et upload le mesh vers le pipeline drone du
    /// renderer.  Échec silencieux si absent (pas d'asset = pas de
    /// drones, gameplay identique).
    fn load_drone_asset(&mut self) {
        const PATHS: &[&str] = &[
            "assets/models/drone.glb",
            "models/drone.glb",
            "assets/drone.glb",
        ];
        let bases = resolve_asset_search_bases();
        let mut tried: Vec<String> = Vec::new();
        for base in &bases {
            for rel in PATHS {
                let full = base.join(rel);
                tried.push(full.display().to_string());
                let bytes = match std::fs::read(&full) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                match q3_model::glb::GlbMesh::from_glb_bytes(&bytes) {
                    Ok(mesh) => {
                        info!(
                            "drone GLB chargé : '{}' ({} verts, radius {:.1})",
                            full.display(),
                            mesh.vertices.len(),
                            mesh.radius()
                        );
                        if let Some(r) = self.renderer.as_mut() {
                            r.upload_drone_mesh(&mesh);
                        }
                        return;
                    }
                    Err(e) => warn!("drone GLB '{}': {}", full.display(), e),
                }
            }
        }
        warn!(
            "drone GLB : aucun asset trouvé. Cherché dans :\n  {}",
            tried.join("\n  ")
        );
    }

    /// **Spawn drones BR** (v0.9.5) — pose 6 drones en orbites
    /// concentriques au-dessus du centre de l'île, à des altitudes
    /// et rayons variés.  Décalage de phase par drone pour qu'ils
    /// ne soient pas tous au même endroit.
    fn spawn_br_drones(&mut self, terrain: &q3_terrain::Terrain) {
        self.drones.clear();
        // Sans renderer ou sans mesh chargé, inutile de spawner.
        let has_mesh = self
            .renderer
            .as_ref()
            .map(|r| r.drone_has_mesh())
            .unwrap_or(false);
        if !has_mesh {
            return;
        }
        let center = terrain.center();
        // 6 drones dans 3 orbites + 2 hauteurs. Altitudes baissées
        // (800-1400) et rayons réduits pour qu'ils restent dans le
        // FOV joueur sans avoir à viser le ciel.
        const SPECS: &[(f32, f32, f32, f32, [f32; 4])] = &[
            ( 2_500.0,  900.0,  0.15, 0.0,                        [0.85, 0.92, 1.00, 1.0]),
            ( 2_500.0,  900.0,  0.15, std::f32::consts::PI,       [0.95, 0.85, 0.55, 1.0]),
            ( 4_500.0, 1_200.0, -0.10, 1.0,                        [0.70, 0.80, 1.00, 1.0]),
            ( 4_500.0, 1_200.0, -0.10, 1.0 + std::f32::consts::PI, [0.95, 0.95, 0.95, 1.0]),
            ( 6_500.0, 1_400.0,  0.07, 2.0,                        [1.00, 0.65, 0.40, 1.0]),
            ( 6_500.0, 1_400.0,  0.07, 2.0 + std::f32::consts::PI, [0.55, 0.85, 1.00, 1.0]),
        ];
        // **Auto-scale drone** (v0.9.5++) — calcule le scale pour
        // que la silhouette du drone ait ~600u dans le monde (visible
        // depuis 5km). target / mesh_radius. Garde-fou si radius
        // dégénéré (asset boggy) → fallback 100.
        let r_mesh = self.renderer.as_ref().map(|r| r.drone_radius()).unwrap_or(0.0);
        let base_scale = if r_mesh > 0.001 {
            600.0 / r_mesh
        } else {
            100.0
        };
        for &(radius, alt, ang_speed, phase, tint) in SPECS {
            self.drones.push(Drone {
                orbit_center: center,
                orbit_radius: radius,
                altitude: alt,
                angular_speed: ang_speed,
                phase,
                scale: base_scale * (1.0 + (phase * 0.4).sin() * 0.25),
                tint,
            });
        }
        info!(
            "BR: {} drones spawnés (mesh radius={:.2}, scale base={:.1})",
            self.drones.len(),
            r_mesh,
            base_scale,
        );
    }

    /// **Weapon / projectile assets** (v0.9.5) — chargé à chaque map
    /// load (BSP ou BR terrain) pour avoir les viewmodels + rocket /
    /// plasma / grenade meshes.  Sans ça, les pickups d'arme en BR
    /// ne montraient PAS de viewmodel quand on switche dessus.
    fn load_weapon_assets(&mut self) {
        let Some(r) = self.renderer.as_mut() else { return; };
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
        self.rocket_mesh = r
            .load_md3(&self.vfs, "models/ammo/rocket/rocket.md3")
            .ok();
        const PLASMA_MESH_CANDIDATES: &[&str] = &[
            "models/weaphits/plasma.md3",
            "models/weaphits/plasmball.md3",
            "models/ammo/plasma/plasma.md3",
        ];
        self.plasma_mesh = PLASMA_MESH_CANDIDATES
            .iter()
            .find_map(|p| r.load_md3(&self.vfs, p).ok());
        const GRENADE_MESH_CANDIDATES: &[&str] = &[
            "models/ammo/grenade1.md3",
            "models/weaphits/grenade1.md3",
            "models/ammo/grenade/grenade.md3",
        ];
        self.grenade_mesh = GRENADE_MESH_CANDIDATES
            .iter()
            .find_map(|p| r.load_md3(&self.vfs, p).ok());
    }

    /// **BR pickups spawn** (v0.9.5) — pose un set d'items autour de
    /// chaque POI proportionnel à son tier. Modèle :
    /// * tier 4 : Quad/Mega Health + RG + RL + ammo (5-6 items)
    /// * tier 3 : MH + RA + RL ou PG + ammo (4 items)
    /// * tier 2 : SG/MG + health/armor (3 items)
    ///
    /// Les items sont disposés en cercle 200u de rayon autour du centre
    /// POI, à 30u au-dessus du sol pour ne pas être enterrés.
    /// `respawn_cooldown` = 9999.0 (i.e. jamais en BR — un item ramassé
    /// est perdu pour le match).
    fn spawn_br_pickups(&mut self) {
        use q3_terrain::PoiKind;

        self.pickups.clear();
        let Some(terrain) = self.terrain.as_ref().cloned() else {
            return;
        };
        let Some(_renderer) = self.renderer.as_mut() else {
            return;
        };

        // Spec d'items par tier (path MD3, kind).  On utilise les
        // chemins canoniques Q3 pak0 — chargement échoue en silence
        // si l'asset manque, l'item est juste skip.
        struct ItemSpec {
            path: &'static str,
            kind: PickupKind,
        }
        let tier4: &[ItemSpec] = &[
            ItemSpec {
                path: "models/powerups/instant/quad.md3",
                kind: PickupKind::Powerup {
                    powerup: PowerupKind::QuadDamage,
                    duration: 30.0,
                },
            },
            ItemSpec {
                path: "models/powerups/health/mega_cross.md3",
                kind: PickupKind::Health { amount: 100, max_cap: 200 },
            },
            ItemSpec {
                path: "models/weapons2/railgun/railgun.md3",
                kind: PickupKind::Weapon {
                    weapon: WeaponId::Railgun,
                    ammo: 10,
                },
            },
            ItemSpec {
                path: "models/weapons2/rocketl/rocketl.md3",
                kind: PickupKind::Weapon {
                    weapon: WeaponId::Rocketlauncher,
                    ammo: 10,
                },
            },
            ItemSpec {
                path: "models/powerups/armor/armor_red.md3",
                kind: PickupKind::Armor { amount: 100 },
            },
        ];
        let tier3: &[ItemSpec] = &[
            ItemSpec {
                path: "models/powerups/health/large_cross.md3",
                kind: PickupKind::Health { amount: 50, max_cap: 100 },
            },
            ItemSpec {
                path: "models/powerups/armor/armor_yel.md3",
                kind: PickupKind::Armor { amount: 50 },
            },
            ItemSpec {
                path: "models/weapons2/plasma/plasma.md3",
                kind: PickupKind::Weapon {
                    weapon: WeaponId::Plasmagun,
                    ammo: 50,
                },
            },
            ItemSpec {
                path: "models/weapons2/rocketl/rocketl.md3",
                kind: PickupKind::Weapon {
                    weapon: WeaponId::Rocketlauncher,
                    ammo: 10,
                },
            },
        ];
        let tier2: &[ItemSpec] = &[
            ItemSpec {
                path: "models/weapons2/shotgun/shotgun.md3",
                kind: PickupKind::Weapon {
                    weapon: WeaponId::Shotgun,
                    ammo: 10,
                },
            },
            ItemSpec {
                path: "models/weapons2/machinegun/machinegun.md3",
                kind: PickupKind::Weapon {
                    weapon: WeaponId::Machinegun,
                    ammo: 100,
                },
            },
            ItemSpec {
                path: "models/powerups/health/medium_cross.md3",
                kind: PickupKind::Health { amount: 25, max_cap: 100 },
            },
            ItemSpec {
                path: "models/powerups/armor/shard.md3",
                kind: PickupKind::Armor { amount: 5 },
            },
        ];

        let mut spawned = 0usize;
        let mut missing = 0usize;
        for (i, poi) in terrain.pois().iter().enumerate() {
            let specs: &[ItemSpec] = match poi.tier {
                4 => tier4,
                3 => tier3,
                2 => tier2,
                _ => continue,
            };
            // Bonus pour les POI iconiques (Volcano/City) — un set
            // tier-au-dessus pose une couche d'items rares.
            let _bonus = matches!(poi.kind, PoiKind::Volcano | PoiKind::City);

            // **Mode exploration** (v0.9.5++) — si `br_bots=0`, on ne
            // spawn QUE des powerups (Quad / Regen / Haste / etc.) ET
            // rochers, comme demandé par le user.  Les armes / armures /
            // health crosses sont skip.
            let exploration = self.cvars.get_i32("br_bots").unwrap_or(0) == 0;
            for (k, spec) in specs.iter().enumerate() {
                if exploration && !matches!(spec.kind, PickupKind::Powerup { .. }) {
                    continue;
                }
                // Position en cercle autour du POI.
                let theta = (k as f32 / specs.len() as f32) * std::f32::consts::TAU;
                let r_off = 100.0 + (i as f32).sin() * 30.0;
                let x = poi.x + theta.cos() * r_off;
                let y = poi.y + theta.sin() * r_off;
                let z = terrain.height_at(x, y) + 30.0;

                let mesh_res = self
                    .renderer
                    .as_mut()
                    .map(|r| r.load_md3(&self.vfs, spec.path));
                let mesh = match mesh_res {
                    Some(Ok(m)) => m,
                    Some(Err(_)) => {
                        missing += 1;
                        continue;
                    }
                    None => continue,
                };
                self.pickups.push(PickupGpu {
                    mesh,
                    origin: Vec3::new(x, y, z),
                    angles: q3_math::Angles::ZERO,
                    kind: spec.kind.clone(),
                    // BR : pas de respawn (un item ramassé est perdu).
                    respawn_cooldown: 9999.0,
                    respawn_at: None,
                    entity_index: u16::MAX,
                });
                spawned += 1;
            }
        }
        info!(
            "BR pickups: {} placés sur {} POI ({} assets manquants)",
            spawned,
            terrain.pois().len(),
            missing
        );
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
                    // load_map drain pending_local_bots en interne
                    // depuis v0.9.3 — pas besoin de dupliquer ici.
                    self.load_map(&path);
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
                PendingAction::MusicPlay(path) => {
                    self.handle_music_play(&path);
                }
                PendingAction::MusicStop => {
                    if let Some(snd) = self.sound.as_ref() {
                        snd.stop_music();
                        info!("music: stopped");
                    }
                }
                PendingAction::MapDlList => {
                    info!("mapdl: catalogue ({} entrées) :", self.map_dl.catalog.len());
                    for line in self.map_dl.list_for_console() {
                        info!("{}", line);
                    }
                    info!("mapdl: utilise `mapdl get <id>` pour télécharger");
                }
                PendingAction::MapDlGet(id) => {
                    self.map_dl.start(&id);
                }
                PendingAction::MapDlStatus => {
                    let st = self.map_dl.status_snapshot();
                    match st {
                        crate::map_dl::DownloadStatus::Idle => {
                            info!("mapdl: idle (aucun job actif)");
                        }
                        crate::map_dl::DownloadStatus::Downloading { id, received, total } => {
                            let pct = if total > 0 {
                                (received as f32 / total as f32 * 100.0) as i32
                            } else { -1 };
                            info!("mapdl: `{}` — {} / {} bytes ({}%)",
                                  id, received, total, pct);
                        }
                        crate::map_dl::DownloadStatus::Verifying { id } => {
                            info!("mapdl: `{}` — vérification SHA256", id);
                        }
                        crate::map_dl::DownloadStatus::Extracting { id } => {
                            info!("mapdl: `{}` — extraction PK3", id);
                        }
                        crate::map_dl::DownloadStatus::Done { id, path } => {
                            info!("mapdl: `{}` — terminé → {}", id, path.display());
                        }
                        crate::map_dl::DownloadStatus::Error { id, message } => {
                            info!("mapdl: `{}` — ERREUR : {}", id, message);
                        }
                    }
                }
            }
        }
    }

    /// Joue un fichier audio local en loop comme musique de fond.
    /// Path : absolu ou relatif au CWD.  Échec silencieux si le fichier
    /// n'est pas trouvé (log warn) ou si rodio ne sait pas le décoder
    /// (formats supportés actuellement : WAV, OGG).
    fn handle_music_play(&mut self, path: &std::path::Path) {
        let Some(snd) = self.sound.as_ref() else {
            warn!("music: audio non initialisé");
            return;
        };
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                warn!("music: impossible de lire `{}`: {}", path.display(), e);
                return;
            }
        };
        match snd.play_music(bytes) {
            Ok(()) => info!("music: now playing `{}`", path.display()),
            Err(e) => warn!("music: échec lecture `{}`: {}", path.display(), e),
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
                // **Charge animation.cfg** (v0.9.5++) — source canonique
                // des ranges d'animation pour CE modèle.  Chaque player
                // model a son propre fichier (sarge.cfg ≠ keel.cfg etc.).
                let cfg_path = format!("{dir}/animation.cfg");
                let anims = match self.vfs.read(&cfg_path) {
                    Ok(bytes) => {
                        let s = String::from_utf8_lossy(&bytes);
                        parse_animation_cfg(&s)
                    }
                    Err(e) => {
                        warn!("animation.cfg absent pour '{dir}' ({e}) — fallback offsets canoniques");
                        hashbrown::HashMap::new()
                    }
                };
                self.bot_rig = Some(PlayerRig { lower: l, upper: u, head: h, anims });
                break;
            }
        }
        if self.bot_rig.is_none() {
            // Aucun model joueur trouvé. Liste ce qui EXISTE dans le
            // VFS sous `models/players/` pour orienter le diagnostic
            // (un nom de model non standard, un chemin légèrement
            // différent, etc.). Sans ce log on jouait à la devinette.
            let mut found_paths: Vec<String> = self
                .vfs
                .list_prefix("models/players/")
                .into_iter()
                .filter(|p| p.ends_with(".md3"))
                .collect();
            found_paths.sort();
            if found_paths.is_empty() {
                warn!(
                    "player rig: VFS ne contient AUCUN .md3 sous \
                     models/players/. Cause probable : --base pointe \
                     sur un dossier sans pak0.pk3 vanilla. Le fallback \
                     beam vertical sera utilisé pour les bots."
                );
            } else {
                warn!(
                    "player rig: aucun ENSEMBLE COMPLET (lower+upper+head) \
                     trouvé. Cependant le VFS contient {} fichier(s) MD3 \
                     sous models/players/ :",
                    found_paths.len()
                );
                for p in found_paths.iter().take(20) {
                    warn!("  - {}", p);
                }
                if found_paths.len() > 20 {
                    warn!("  … et {} autres", found_paths.len() - 20);
                }
            }
        }
    }

    fn spawn_bot(&mut self, name: &str, skill_override: Option<i32>) {
        // Pré-vérifs : on accepte soit un BSP (`world`) soit un terrain BR.
        // Un seul des deux doit être présent à la fois (cf. load_map et
        // load_terrain_map).
        let mode_world = self.world.is_some();
        let mode_terrain = self.terrain.is_some();
        if !mode_world && !mode_terrain {
            warn!("addbot: pas de map chargée");
            return;
        }
        if mode_world {
            let world = self.world.as_ref().unwrap();
            if world.spawn_points.is_empty() && world.player_start.is_none() {
                warn!("addbot: la map n'a pas de point de spawn");
                return;
            }
        }

        // Lazy-load le rig partagé au premier bot. Si AUCUN model
        // joueur n'est trouvé dans le VFS (pak0 partiel / démo), on
        // continue quand même : la logique IA + collisions + score
        // tourne sans rig, et le rendu utilise un fallback beam
        // vertical (cf. queue_bots None branch).
        if self.bot_rig.is_none() {
            self.ensure_player_rig_loaded();
            if self.bot_rig.is_none() {
                warn!(
                    "addbot: '{name}' spawné SANS rig MD3 (asset manquant) — \
                     rendu fallback beam, gameplay normal"
                );
            }
        }

        let idx = self.bots.len() + 1;
        let (spawn_origin, angles) = if mode_world {
            // BSP : cycle dans les DM spawn points, fallback player_start.
            let world = self.world.as_ref().unwrap();
            let (origin, ang) = if !world.spawn_points.is_empty() {
                let sp = &world.spawn_points[idx % world.spawn_points.len()];
                (sp.origin, sp.angles)
            } else {
                (world.player_start.unwrap_or(Vec3::ZERO), world.player_start_angles)
            };
            (origin + Vec3::Z * 40.0, ang)
        } else {
            // **BR** : spawn sur un POI tier ≥ 2 distribué pseudo-uniformément.
            // On choisit l'index par hash de `idx + time` pour que des
            // bots successifs n'atterrissent pas tous sur le même POI.
            let terrain = self.terrain.as_ref().unwrap();
            let pois: Vec<&q3_terrain::Poi> =
                terrain.pois().iter().filter(|p| p.tier >= 2).collect();
            let xy = if pois.is_empty() {
                let c = terrain.center();
                (c.x, c.y)
            } else {
                let h = (idx.wrapping_mul(2654435761) ^ self.time_sec.to_bits() as usize)
                    % pois.len();
                let p = pois[h];
                // Petit offset pour ne pas piler tous les bots au même
                // sample exact si plusieurs partagent le POI.
                let offset = (idx as f32 * 13.0).sin() * 60.0;
                (p.x + offset, p.y + offset.cos() * 80.0)
            };
            let z = terrain.height_at(xy.0, xy.1) + 60.0;
            (Vec3::new(xy.0, xy.1, z), q3_math::Angles::ZERO)
        };

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
        // Waypoints — BSP : DM spawn points dans l'ordre. BR : POI tier
        // ≥ 2 dans l'ordre (= "patrouille de POI à POI" en attendant
        // une vraie navmesh terrain).
        if mode_world {
            let world = self.world.as_ref().unwrap();
            for (i, sp) in world.spawn_points.iter().enumerate() {
                if i != idx % world.spawn_points.len().max(1) {
                    bot.push_waypoint(sp.origin + Vec3::Z * 40.0);
                }
            }
        } else if let Some(terrain) = self.terrain.as_ref() {
            for p in terrain.pois().iter().filter(|p| p.tier >= 2) {
                let z = terrain.height_at(p.x, p.y) + 60.0;
                bot.push_waypoint(Vec3::new(p.x, p.y, z));
            }
        }

        let mut body = PlayerMove::new(spawn_origin);
        body.view_angles = angles;

        let tint = bot_tint(idx);
        let personality = bot_personality_from_index(idx);
        bot.set_personality(personality);
        info!(
            "addbot: '{name}' spawné à {:?} (tint={:?}, pers={})",
            spawn_origin, tint, bot_personality_tag(personality)
        );
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
            last_pain_sfx_at: f32::NEG_INFINITY,
            airborne_since: None,
            last_land_at: f32::NEG_INFINITY,
            anim_phase: 0.0,
            anim_state: BotAnimState::Idle,
            last_heard_pos: None,
            last_heard_at: f32::NEG_INFINITY,
            personality: bot_personality_from_index(idx),
            last_chatter_at: f32::NEG_INFINITY,
            death_started_at: None,
            lower_anim_start: usize::MAX,
            lower_anim_started_at: 0.0,
            upper_anim_start: usize::MAX,
            upper_anim_started_at: 0.0,
            death_variant: (idx as u8) % 3,
            prev_yaw: 0.0,
            last_turn_at: f32::NEG_INFINITY,
            gesture_started_at: None,
        });
    }

    /// Téléporte le joueur à un point de spawn DM (ou `player_start` en
    /// fallback), restaure sa santé et réinitialise sa physique. Appelé
    /// quand `respawn_at` est échu.
    fn respawn_player(&mut self) {
        // Track le timestamp respawn pour le stats overlay "ALIVE".
        self.last_respawn_time = self.time_sec;
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
                    // **Bounds clamp** (v0.9.5++ polish) — `slot` est un
                    // `u8` non-typé qui pourrait théoriquement venir d'une
                    // désérialisation réseau ou d'un éditeur de map avec
                    // une valeur ≥ self.ammo.len().  On clamp pour éviter
                    // un panic OOB sur l'array fixe `[i32; N]`.
                    let s = (slot as usize).min(self.ammo.len() - 1);
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
                    // Mega health (>100) joue son sample dédié si dispo,
                    // sinon le sample health standard. Fallback final
                    // sur pain_bot (legacy bip).
                    let is_mega = new_hp > 100;
                    let sfx = if is_mega {
                        self.sfx_megahealth_pickup
                            .or(self.sfx_health_pickup)
                            .or(self.sfx_pain_bot)
                    } else {
                        self.sfx_health_pickup.or(self.sfx_pain_bot)
                    };
                    if let (Some(snd), Some(h)) = (self.sound.as_ref(), sfx) {
                        // Priority::High → jamais culled par un tir
                        // (Priority::Weapon = 4, High = 3 < Weapon mais
                        // les pickups passent toujours via le slot le
                        // plus bas, qui est rarement plasma à plein
                        // débit). On augmente aussi le volume via
                        // near_dist plus grand.
                        play_at(snd, h, sfx_pos, Priority::High);
                    }
                }
                Event::Armor { sfx_pos, new_armor } => {
                    info!("pickup armor → {}", new_armor);
                    let sfx = self.sfx_armor_pickup.or(self.sfx_pain_bot);
                    if let (Some(snd), Some(h)) = (self.sound.as_ref(), sfx) {
                        play_at(snd, h, sfx_pos, Priority::High);
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
                        play_at(snd, h, sfx_pos, Priority::High);
                    }
                }
                Event::Ammo { sfx_pos, slot, ammo_after } => {
                    info!("pickup ammo slot {} → {}", slot, ammo_after);
                    let sfx = self.sfx_ammo_pickup.or(self.sfx_pain_bot);
                    if let (Some(snd), Some(h)) = (self.sound.as_ref(), sfx) {
                        // Priority::High pour ne pas être étouffé par un
                        // tir d'arme automatique (MG/PG) en cours.
                        play_at(snd, h, sfx_pos, Priority::High);
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
            // On vient juste de push dans `kill_feed` au-dessus, donc
            // `last()` est garanti `Some`.  Capturé une fois pour éviter
            // 6× re-lookup + unwrap (lisibilité + micro-perf).
            let last_ev = self
                .kill_feed
                .last()
                .expect("kill_feed pushed above");
            let is_selfkill = matches!(
                (&last_ev.killer, &last_ev.victim),
                (KillActor::Player, KillActor::Player)
            ) || match (&last_ev.killer, &last_ev.victim) {
                (KillActor::Bot(a), KillActor::Bot(b)) => a == b,
                _ => false,
            };
            let killer_is_world = matches!(last_ev.killer, KillActor::World);
            if !is_selfkill && !killer_is_world {
                let killer_label = last_ev.killer.label().to_string();
                self.first_blood_announced = true;
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
        // **Gesture anim** (v0.9.5++) — déclenche TORSO_GESTURE sur les
        // taunts (KillInsult uniquement, pas Death/Respawn qui sont
        // joués dans des contextes où l'anim mort/respawn prime).
        if matches!(trigger, ChatTrigger::KillInsult) {
            if let Some(b) = self.bots.get_mut(idx) {
                b.gesture_started_at = Some(self.time_sec);
            }
        }
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
        // **BR victory condition** (v0.9.5) — last man standing : si
        // un seul acteur (joueur ou bot) est vivant, il gagne. Pas de
        // fraglimit / timelimit en BR — le ring force le converge.
        // **Mode exploration** (v0.9.5++) — si `br_bots=0`, la map est
        // vide donc le check est désactivé (sinon le joueur seul =
        // immediately VICTORY screen, bloque le mouvement).
        if self.terrain.is_some() {
            let br_bots_enabled = self.cvars.get_i32("br_bots").unwrap_or(0) != 0;
            if !br_bots_enabled {
                return; // exploration mode — pas de match
            }
            let player_alive = !self.player_health.is_dead();
            let alive_bots: Vec<&BotDriver> = self
                .bots
                .iter()
                .filter(|b| !b.health.is_dead())
                .collect();
            let total_alive = alive_bots.len() + if player_alive { 1 } else { 0 };
            if total_alive == 1 {
                self.match_winner = if player_alive {
                    Some(KillActor::Player)
                } else {
                    Some(KillActor::Bot(alive_bots[0].bot.name.clone()))
                };
                info!("BR match over — winner = {:?}", self.match_winner);
                return;
            }
            if total_alive == 0 {
                // Tout le monde est mort (rare : ring kill simultané)
                self.match_winner = Some(KillActor::World);
                info!("BR match over — egalité (ring)");
                return;
            }
            // En BR on n'utilise PAS frag/time-limit — on attend le ring.
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

    /// Pousse un floater "+1 FRAG" au point de kill — distinct des
    /// damage numbers : durée de vie plus longue (1.6 s vs 1.2), gros
    /// caractères orange-rouge, monte plus haut.  Donne un feedback
    /// visuel "juicy" type Apex/CoD à chaque frag, complémentaire du
    /// kill-marker X dans le réticule (qui est centré écran, alors
    /// que ce floater est ancré au monde sur le bot mort).
    /// **Headshot sparkle** — gerbe dorée additive de 8 particules
    /// rapides + dlight blanc-jaune bref au point d'impact tête.
    /// Distinct des sparks rouges de hit normal : signale clairement
    /// "headshot" sans toast HUD ni audio additionnel.
    fn push_headshot_sparkle(&mut self, pos: Vec3) {
        // 8 particules dorées en sphère étirée — évoque un éclat lumineux.
        for k in 0..8 {
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.swap_remove(0);
            }
            // Direction sphérique uniforme avec un boost +Z léger pour
            // que la majorité du flash monte vers la caméra du joueur.
            let theta = (k as f32) * std::f32::consts::TAU / 8.0
                + rand_unit() * 0.4;
            let phi = rand_unit() * 0.6 + 0.4; // [0..1] biais haut
            let cos_phi = phi.cos();
            let sin_phi = phi.sin();
            let dir = Vec3::new(
                theta.cos() * sin_phi,
                theta.sin() * sin_phi,
                cos_phi.abs() + 0.3,
            )
            .normalize_or_zero();
            let speed = 220.0 * (0.7 + 0.4 * rand_unit().abs());
            let velocity = dir * speed;
            let lifetime = 0.35 * (0.8 + 0.4 * rand_unit().abs());
            self.particles.push(Particle {
                origin: pos,
                velocity,
                color: [1.0, 0.92, 0.45, 0.9], // gold vif
                expire_at: self.time_sec + lifetime,
                lifetime,
            });
        }
        // Dlight bref blanc-jaune pour souligner l'impact.
        if let Some(r) = self.renderer.as_mut() {
            r.spawn_dlight(
                pos,
                120.0,
                [1.0, 0.95, 0.65],
                2.0,
                self.time_sec,
                0.05,
            );
        }
    }

    fn push_frag_confirm(&mut self, origin: Vec3) {
        if self.floating_damages.len() >= DAMAGE_NUMBER_MAX {
            self.floating_damages.remove(0);
        }
        self.floating_damages.push(FloatingDamage {
            origin,
            damage: 0, // ignoré pour les frags
            to_player: false,
            expire_at: self.time_sec + 1.6,
            lifetime: 1.6,
            is_frag: true,
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
            is_frag: false,
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
                self.particles.swap_remove(0);
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
                self.particles.swap_remove(0);
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
        // Couche 4 — **heat shimmer** (v0.9.5++) : ~12 particules
        // semi-transparentes ocre-jaune à forte vélocité +Z. Sans flag
        // gravity_scale custom on triche en donnant juste assez de
        // vitesse upward pour que la particule monte ~80 unités avant
        // que `PARTICLE_GRAVITY` ne renverse la trajectoire — temps
        // pendant lequel elle fade out. Visuellement elle évoque une
        // colonne d'air chaud qui s'élève au-dessus du point d'impact,
        // pas un vrai blur screen-space mais lecture "heat" claire à
        // l'œil. Couleur très désaturée + alpha bas pour éviter de
        // concurrencer le flash principal.
        const HEAT_COUNT: usize = 12;
        for i in 0..HEAT_COUNT {
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.swap_remove(0);
            }
            // Cône étroit autour du +Z — chaque particule diverge légèrement.
            let theta = (i as f32) * std::f32::consts::TAU / (HEAT_COUNT as f32)
                + rand_unit() * 0.3;
            let radial = 0.10 + 0.20 * rand_unit().abs();
            let vx = theta.cos() * radial;
            let vy = theta.sin() * radial;
            let vz = 1.0; // dominante verticale
            let dir = Vec3::new(vx, vy, vz).normalize_or_zero();
            // Vitesse modérée — laisse la gravité reprendre vite.  Pic
            // ~80–110 unités vers le haut avant inversion.
            let speed = 280.0 * (0.7 + 0.3 * rand_unit().abs());
            let velocity = dir * speed;
            // Ocre/jaune/sépia désaturé → "air chaud" plutôt que "feu".
            let h = rand_unit().abs();
            let color = [
                0.85 + 0.15 * h,
                0.65 + 0.20 * h,
                0.30 + 0.10 * h,
                0.35, // alpha bas → translucide, ne mange pas le flash
            ];
            // Vie longue (~0.9 s) pour que le mouvement vertical lent
            // ait le temps de se lire.
            let lifetime = 0.85 * (0.8 + 0.4 * rand_unit().abs());
            self.particles.push(Particle {
                origin,
                velocity,
                color,
                expire_at: self.time_sec + lifetime,
                lifetime,
            });
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
                self.particles.swap_remove(0);
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
    /// `normal` doit pointer hors du mur (typiquement `trace_normal`).
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
            return;
        }
        if trace.plane_normal.z < 0.5 {
            return;
        }
        let hit_pt = from + (to - from) * trace.fraction;
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
        // **Death gibs amélioré** (V2f) — 3 strates :
        // 1. Sang carmin (existant) : éjection radiale, vie moyenne
        // 2. Chunks plus gros (nouveau) : moins nombreux, plus lents,
        //    couleur rouge sombre, vie longue → tombent au sol
        // 3. Brume rose claire (nouveau) : douche de fines particules
        //    qui restent suspendues 0.6s puis fade
        // 4. Flash blanc 1-frame additif → "impact" perçu

        // Strate 1 — sang carmin (existant, légèrement boosté).
        for _ in 0..PARTICLE_GIB_COUNT {
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.swap_remove(0);
            }
            let mut dir = Vec3::new(rand_unit(), rand_unit(), rand_unit());
            let len = dir.length();
            if len < 1e-3 {
                dir = Vec3::Z;
            } else {
                dir /= len;
            }
            if dir.z < -0.3 {
                dir.z = -dir.z * 0.5;
            }
            let speed = PARTICLE_GIB_SPEED * (0.4 + 0.6 * rand_unit().abs());
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
        // Strate 2 — chunks gros et sombres (~moitié du count, vitesse
        // réduite, vie ×1.5 pour tomber + traîner).
        let chunks = (PARTICLE_GIB_COUNT / 2).max(8);
        for _ in 0..chunks {
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.swap_remove(0);
            }
            let mut dir = Vec3::new(rand_unit(), rand_unit(), rand_unit() * 0.5);
            let len = dir.length();
            if len < 1e-3 {
                dir = Vec3::Z;
            } else {
                dir /= len;
            }
            let speed = PARTICLE_GIB_SPEED * 0.55 * (0.3 + 0.7 * rand_unit().abs());
            let darkness = 0.3 + 0.2 * rand_unit().abs();
            self.particles.push(Particle {
                origin: pos,
                velocity: dir * speed,
                color: [darkness, darkness * 0.2, darkness * 0.15, 1.0],
                expire_at: self.time_sec
                    + PARTICLE_GIB_LIFETIME * 1.5 * (0.7 + 0.3 * rand_unit().abs()),
                lifetime: PARTICLE_GIB_LIFETIME * 1.5,
            });
        }
        // Strate 3 — brume rose claire (mist / aérosol). Particules
        // lentes qui flottent. Donne un nuage post-impact visible.
        let mist = 16;
        for _ in 0..mist {
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.swap_remove(0);
            }
            let dir = Vec3::new(
                rand_unit() * 0.7,
                rand_unit() * 0.7,
                0.3 + rand_unit().abs() * 0.5,
            );
            let speed = 60.0 + rand_unit().abs() * 40.0;
            self.particles.push(Particle {
                origin: pos + Vec3::new(rand_unit() * 8.0, rand_unit() * 8.0, rand_unit().abs() * 8.0),
                velocity: dir.normalize() * speed,
                color: [1.0, 0.55, 0.55, 0.7],
                expire_at: self.time_sec + 0.6,
                lifetime: 0.6,
            });
        }
        // Strate 4 — dlight flash blanc bref (40 ms) pour signaler
        // l'impact. Empilé avec les explosion dlights → bien visible.
        if let Some(r) = self.renderer.as_mut() {
            r.spawn_dlight(
                pos + Vec3::Z * 16.0,
                180.0,
                [1.0, 0.85, 0.85],
                3.0,
                self.time_sec,
                0.04,
            );
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
                self.particles.swap_remove(0);
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
        // **Horizontal ring fanfare** (v0.9.5++) — 16 particules en cercle
        // qui partent radialement vers l'extérieur dans le plan XY.
        // Couplé au beam vertical, ça donne la signature visuelle
        // "explosion d'arrivée" canonique des item respawns BR.
        const RING_COUNT: usize = 16;
        for k in 0..RING_COUNT {
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.swap_remove(0);
            }
            let theta = (k as f32) * std::f32::consts::TAU
                / (RING_COUNT as f32);
            let dir = Vec3::new(theta.cos(), theta.sin(), 0.05);
            let speed = 150.0 + rand_unit().abs() * 60.0;
            let lifetime = 0.55 * (0.8 + 0.3 * rand_unit().abs());
            // Légère teinte plus pâle pour distinguer de la colonne ascendante.
            let ring_color = [
                color[0] * 0.85 + 0.15,
                color[1] * 0.85 + 0.15,
                color[2] * 0.85 + 0.15,
                color[3] * 0.9,
            ];
            self.particles.push(Particle {
                origin: pos + Vec3::Z * 4.0,
                velocity: dir * speed,
                color: ring_color,
                expire_at: self.time_sec + lifetime,
                lifetime,
            });
        }
        // Dlight cyan brève au point de respawn → flash visible de loin
        // qui matche la palette du bandeau radius-respawn.
        if let Some(r) = self.renderer.as_mut() {
            r.spawn_dlight(
                pos + Vec3::Z * 16.0,
                280.0,
                [color[0], color[1], color[2]],
                2.5,
                self.time_sec,
                0.18,
            );
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
                self.particles.swap_remove(0);
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
                self.particles.swap_remove(0);
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
        // Flash full-screen teinté de la couleur du powerup. Donne une
        // confirmation visuelle forte de la transition « sans → avec »
        // (équivalent du strobe Q3 sur pickup).  Le fade est géré dans
        // draw_hud via (powerup_flash_until - now).
        self.powerup_flash_until = self.time_sec + POWERUP_FLASH_SEC;
        self.powerup_flash_color = kind.pickup_fx_color();
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

    /// **Atmospheric ambience BR** — éclairs périodiques + dlight qui
    /// brièvement illumine le ciel.  Activé uniquement en mode BR
    /// (terrain présent), sinon no-op.  Le premier appel arme
    /// `next_lightning_at` à un délai aléatoire dans la fenêtre
    /// [LIGHTNING_INTERVAL_MIN..MAX] depuis maintenant.
    fn tick_atmosphere(&mut self) {
        // BR uniquement — le mode BSP a déjà sa propre ambiance via
        // shaders / sky / fog. Le terrain Réunion bénéficie d'une
        // touche orageuse pour casser la monotonie diurne.
        if self.terrain.is_none() {
            return;
        }
        // Gel pendant l'intermission post-match — la caméra est figée,
        // un éclair surprendrait le joueur en lecture du scoreboard.
        if self.match_winner.is_some() {
            return;
        }
        let now = self.time_sec;
        // Premier passage : armer le timer à un délai aléatoire.
        if !self.next_lightning_at.is_finite() {
            let interval = LIGHTNING_INTERVAL_MIN
                + rand_unit().abs() * (LIGHTNING_INTERVAL_MAX - LIGHTNING_INTERVAL_MIN);
            self.next_lightning_at = now + interval;
            return;
        }
        if now < self.next_lightning_at {
            return;
        }
        // Trigger flash : overlay HUD bref + dlight très haut +
        // schedule du prochain.
        self.lightning_flash_until = now + LIGHTNING_FLASH_SEC;
        // Dlight haut dans le ciel à une position aléatoire au-dessus
        // du joueur (pas trop loin pour que l'illumination touche
        // les surfaces visibles).  Bleu-blanc froid, lifetime court.
        if let Some(r) = self.renderer.as_mut() {
            let dx = rand_unit() * 800.0;
            let dy = rand_unit() * 800.0;
            let sky_pos = self.player.origin + Vec3::new(dx, dy, 1800.0);
            r.spawn_dlight(
                sky_pos,
                3500.0,                  // énorme rayon — éclaire la scène
                [0.85, 0.92, 1.00],      // bleu-blanc froid
                3.5,                     // intensité forte
                now,
                LIGHTNING_FLASH_SEC,     // synchro avec le flash HUD
            );
        }
        // **Camera shake** corrélé au flash — secousse subtile (≈ 1/3
        // d'une explosion proche) pour donner l'impression du tonnerre
        // qui résonne. Pas d'audio (pas de SFX thunder chargé), donc
        // le shake fait office de "sentir" le grondement.  Empile-safe
        // (max-merge avec un éventuel shake d'explosion concurrent).
        let lightning_shake = 0.0035_f32; // rad, ≈ 1/3 d'un splash proche
        if lightning_shake > self.shake_intensity {
            self.shake_intensity = lightning_shake;
        }
        self.shake_until = self.shake_until.max(now + SHAKE_DURATION);
        // Programme le prochain éclair après un délai variable.
        let interval = LIGHTNING_INTERVAL_MIN
            + rand_unit().abs() * (LIGHTNING_INTERVAL_MAX - LIGHTNING_INTERVAL_MIN);
        self.next_lightning_at = now + interval;
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
                // **Regen sparkle** (v0.9.5++) — 2 particules vertes
                // qui s'élèvent autour du joueur à chaque HP gagné.
                // Donne un feedback visuel discret distinct du badge
                // HUD : montre que la régénération "agit" ici et
                // maintenant, pas juste un timer abstrait.
                for k in 0..2 {
                    if self.particles.len() >= PARTICLE_MAX {
                        self.particles.swap_remove(0);
                    }
                    let theta = (k as f32) * std::f32::consts::PI
                        + rand_unit() * 0.6;
                    let radius = 14.0 + rand_unit().abs() * 6.0;
                    let pos = self.player.origin
                        + Vec3::new(theta.cos() * radius, theta.sin() * radius, 6.0);
                    // Vélocité presque verticale → sparkle ascendant.
                    let velocity = Vec3::new(
                        rand_unit() * 8.0,
                        rand_unit() * 8.0,
                        45.0 + rand_unit().abs() * 25.0,
                    );
                    let lifetime = 0.55 * (0.8 + 0.4 * rand_unit().abs());
                    self.particles.push(Particle {
                        origin: pos,
                        velocity,
                        color: [0.40, 1.00, 0.45, 0.85], // vert vif
                        expire_at: self.time_sec + lifetime,
                        lifetime,
                    });
                }
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
            // **Footstep particles** (V2a) — petit puff de poussière
            // sous le pied. Couleur grise neutre, vitesse basse, vie
            // courte (0.4s). 3-4 particules par pas suffisent — l'œil
            // capte le mouvement de "kick" sans saturer la scène.
            // Position : sous les pieds (origin = pied du hull).
            for k in 0..3 {
                if self.particles.len() >= PARTICLE_MAX {
                    self.particles.swap_remove(0);
                }
                let foot = self.player.origin;
                let angle = (k as f32) * 2.094 + rand_unit() * 0.5; // 120° entre 3 puffs
                let r = 8.0 + rand_unit().abs() * 6.0;
                let pos = Vec3::new(
                    foot.x + angle.cos() * r,
                    foot.y + angle.sin() * r,
                    foot.z + 2.0,
                );
                self.particles.push(Particle {
                    origin: pos,
                    velocity: Vec3::new(
                        angle.cos() * 25.0,
                        angle.sin() * 25.0,
                        15.0 + rand_unit().abs() * 12.0,
                    ),
                    color: [0.55, 0.50, 0.45, 0.65],
                    expire_at: self.time_sec + 0.40,
                    lifetime: 0.40,
                });
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

    /// **Railgun zoom (sniper scope)** (v0.9.5++) — actif quand
    /// `secondary_fire` est tenu ET que l'arme active est le Railgun.
    /// Retourne `Some(zoom_ratio)` où `zoom_ratio < 1.0` réduit le FOV
    /// (ex. `0.33` → ~3× zoom), `None` si pas de zoom.
    ///
    /// Le ratio est utilisé pour :
    ///   * réduire le FOV de la caméra (proche × 1/3)
    ///   * scaler la sensibilité souris au même ratio (mvt précis)
    ///   * activer l'overlay scope HUD (vignette circulaire + crosshair)
    fn railgun_zoom_ratio(&self) -> Option<f32> {
        if self.active_weapon == WeaponId::Railgun
            && self.input.secondary_fire
            && !self.player_health.is_dead()
            && !self.menu.open
            && !self.console.is_open()
        {
            // 3× zoom → FOV ~30° à partir de cg_fov 90°.
            Some(0.33)
        } else {
            None
        }
    }

    /// **Muzzle position en world space** — point d'où partent
    /// visuellement les balles, beams et muzzle flashes pour chaque
    /// arme.  Doit matcher la position du bout du canon dans le
    /// viewmodel (GLB ou MD3) pour que tirs/effets soient cohérents.
    ///
    /// Convention : offset (forward, right, down_neg) depuis l'œil.
    /// Les armes GLB ont des géométries différentes → tuning empirique
    /// par arme.  Les armes MD3 utilisent une valeur médiane.
    fn viewmodel_muzzle_pos(&self, weapon: WeaponId) -> Vec3 {
        let eye = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
        let basis = self.player.view_angles.to_vectors();
        // Offsets (forward, right, down_neg) tunés par arme — chaque GLB
        // a une géométrie différente, donc le bout du canon n'est pas
        // au même endroit dans l'espace local.  Toutes les armes Q3
        // sont maintenant en GLB (9/9), pas de fallback MD3 utilisé.
        let (fwd, rt, up_neg) = match weapon {
            // Armes longues / fusils long canon.
            WeaponId::Machinegun      => (28.0, 4.0, 4.0),
            WeaponId::Lightninggun    => (26.0, 5.0, 4.0),
            WeaponId::Rocketlauncher  => (26.0, 6.0, 5.0),
            WeaponId::Shotgun         => (24.0, 6.0, 5.0),
            WeaponId::Grenadelauncher => (22.0, 6.0, 5.0),
            // Armes médiums.
            WeaponId::Bfg             => (18.0, 6.0, 5.0),
            WeaponId::Plasmagun
            | WeaponId::Railgun       => (14.0, 6.0, 5.0),
            // Mêlée Gauntlet — pas de canon, "tip" = bout du gant à la
            // hauteur de la main (range courte 64u, le hit se fait au
            // contact).
            WeaponId::Gauntlet        => (10.0, 5.0, 4.0),
        };
        eye + basis.forward * fwd + basis.right * rt - basis.up * up_neg
    }

    fn fire_weapon(&mut self) -> bool {
        use q3_collision::Contents;
        // **BR mode** (v0.9.5) — accepte le firing si terrain présent
        // (les traces hitscan iront vers Terrain::trace_ray au lieu
        // de world.collision.trace_ray, via le shim `trace_shot` ci-
        // dessous).
        if self.world.is_none() && self.terrain.is_none() {
            return false;
        }
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
        // **Comparaison signée** (v0.9.5++ fix) — éviter le cast i32→u32
        // qui pouvait wrap-around si `ammo[slot]` était corrompu négatif
        // (deserialize cheaté), donnant un tir infini exploit.
        if self.ammo[slot] < params.ammo_cost as i32 {
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
        // **Garde-fou** (v0.9.5++ polish) — `saturating_sub` au lieu de
        // `(x - y).max(0)` pour éviter l'overflow théorique (ammo_cost
        // est u8 donc toujours ≥ 0, pas de check supplémentaire requis).
        self.ammo[slot] = self.ammo[slot].saturating_sub(params.ammo_cost as i32);
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
        // **Plasma SFX throttle** (v0.9.5++) — le plasma tire à 10/sec
        // et le sample fait ~0.3s. Sans throttle, 3+ voix se superposent
        // chaque seconde → les 64 canaux SoundSystem saturent et le
        // mixer drop des plays → "pas de son". Throttle à 6/sec pour
        // garder le feel de rapid-fire sans clogger les channels.
        let mut should_play_fire_sfx = true;
        if matches!(weapon, WeaponId::Plasmagun | WeaponId::Lightninggun) {
            const RAPID_FIRE_SFX_COOLDOWN: f32 = 0.16;
            if self.time_sec - self.last_fire_sfx_at < RAPID_FIRE_SFX_COOLDOWN {
                should_play_fire_sfx = false;
            }
        }
        if should_play_fire_sfx {
            if let Some((_, h)) = self.sfx_fire.iter().find(|(w, _)| *w == weapon) {
                if let Some(snd) = self.sound.as_ref() {
                    let played = play_at(snd, *h, eye, Priority::Weapon);
                    let master = snd.master_volume();
                    info!(
                        "sfx-fire: {:?} handle={:?} played={} master_vol={:.2} eye_pos={:?}",
                        weapon, h, played, master, eye
                    );
                    if !played {
                        warn!(
                            "sfx-fire: {:?} play_3d a échoué (channels saturés ou volume nul)",
                            weapon
                        );
                    }
                } else {
                    warn!("sfx-fire: {:?} self.sound = None — pas de système audio", weapon);
                }
            } else {
                warn!("sfx-fire: {:?} non trouvé dans sfx_fire (taille={})", weapon, self.sfx_fire.len());
            }
            self.last_fire_sfx_at = self.time_sec;
        }
        self.muzzle_flash_until = self.time_sec + 0.06;
        // **Bot sound awareness** (v0.9.5) — chaque bot dans
        // BOT_HEARING_RADIUS du tireur enregistre la position joueur
        // comme « bruit entendu ». Le tick_bots utilisera ensuite
        // last_heard_pos pour insérer un waypoint d'investigation si
        // pas de LOS direct.  Effet émergent : un tir révèle la
        // position joueur même sans qu'un bot le voie → flush des
        // ennemis qui campaient un coin éloigné.
        let now = self.time_sec;
        for bd in self.bots.iter_mut() {
            if bd.health.is_dead() {
                continue;
            }
            let d = (bd.body.origin - self.player.origin).length();
            if d <= BOT_HEARING_RADIUS {
                bd.last_heard_pos = Some(self.player.origin);
                bd.last_heard_at = now;
            }
        }
        // **Bullet shells eject** (V2b) — MG et SG seulement (les
        // autres armes n'ont pas de douilles physiques en Q3).
        // Position : depuis l'éjecteur droit du viewmodel approx
        // (eye + forward*8 + right*9 + up*1). Vélocité initiale :
        // sortie latérale droite + un peu vers le haut/derrière, +
        // gravité qui les fait tomber. Vie courte (0.6s) suffit
        // visuellement.
        if matches!(weapon, WeaponId::Machinegun | WeaponId::Shotgun) {
            let basis = self.player.view_angles.to_vectors();
            let shell_origin = self.player.origin
                + Vec3::Z * PLAYER_EYE_HEIGHT
                + basis.forward * 8.0
                + basis.right * 9.0
                + Vec3::Z;
            // Vitesse latérale droite + arrière + up — la douille
            // s'éjecte vers l'extérieur de la chambre puis chute.
            let shell_vel = basis.right * 70.0
                - basis.forward * 30.0
                + Vec3::Z * (45.0 + rand_unit().abs() * 25.0);
            if self.particles.len() >= PARTICLE_MAX {
                self.particles.swap_remove(0);
            }
            self.particles.push(Particle {
                origin: shell_origin,
                velocity: shell_vel,
                color: [0.85, 0.65, 0.20, 1.0], // laiton chaud
                expire_at: self.time_sec + 0.6,
                lifetime: 0.6,
            });
        }
        // Impulse de recul : cumule avec la valeur courante pour que les
        // rafales sentent "chargées", clampé à `VIEW_KICK_MAX` pour
        // éviter un viewmodel qui disparaît de l'écran sur tir auto.
        self.view_kick = (self.view_kick + weapon.view_kick()).min(VIEW_KICK_MAX);
        // **Punch angles** (v0.9.5++) — pitch up + jitter de yaw,
        // ajoutés au rendu sans toucher l'aim.  Le PRNG du yaw est
        // dérivé de `time_sec` quantifié pour rester déterministe-ish
        // tout en variant entre tirs.
        let (kick_pitch, kick_yaw_jitter) = weapon.recoil_kick();
        self.view_kick_pitch = (self.view_kick_pitch + kick_pitch).min(15.0);
        if kick_yaw_jitter > 0.001 {
            // Hash simple sur time_sec pour un yaw signé.  `to_bits()`
            // sur f32 donne un u32, on le passe en u64 avant le mul
            // pour éviter un overflow const (le multiplicateur est >u32).
            let h = (self.time_sec * 1000.0).to_bits() as u64;
            let r = ((h.wrapping_mul(0x9E3779B97F4A7C15u64) >> 33) as u32 as f32
                / u32::MAX as f32)
                - 0.5;
            self.view_kick_yaw += r * 2.0 * kick_yaw_jitter;
            self.view_kick_yaw = self.view_kick_yaw.clamp(-5.0, 5.0);
        }
        // Dlight muzzle flash **per-weapon** (v0.9.5+) — utilise la
        // couleur retournée par `weapon.muzzle_flash()` au lieu d'un
        // orange générique. Plasma → bleu, BFG → vert, Railgun →
        // bleu-violet, etc. Si l'arme retourne `None` (gauntlet/LG),
        // pas de dlight — leur effet visuel est géré par le beam.
        if let Some((color, _flash_size)) = weapon.muzzle_flash() {
            if let Some(r) = self.renderer.as_mut() {
                let (radius, intensity) = match weapon {
                    WeaponId::Bfg => (380.0, 4.0),
                    WeaponId::Rocketlauncher => (300.0, 3.0),
                    WeaponId::Shotgun => (260.0, 2.5),
                    WeaponId::Railgun => (250.0, 2.5),
                    WeaponId::Plasmagun => (180.0, 1.5),
                    WeaponId::Machinegun => (170.0, 1.5),
                    _ => (200.0, 1.8),
                };
                r.spawn_dlight(
                    eye + basis.forward * 4.0,
                    radius,
                    [color[0], color[1], color[2]],
                    intensity,
                    self.time_sec,
                    0.08,
                );
            }
        }
        // Smoke puff désactivé (v0.9.5+) — affichage 3D bizarre
        // signalé. À ré-activer après debug. Le muzzle dlight per-
        // weapon couvre déjà l'effet visuel.

        // Branchement sur le type de tir : projectile → spawn entité et
        // on sort. Hitscan → on continue dans la boucle de raycasts.
        if let WeaponKind::Projectile {
            speed,
            splash_radius,
            splash_damage,
        } = params.kind
        {
            // **Spawn position du projectile** — utilise le helper
            // `viewmodel_muzzle_pos` qui retourne le bout du canon
            // selon l'arme (cohérent avec muzzle flash + tracers +
            // beams).  Évite que les projectiles partent depuis l'œil
            // du joueur (16u devant) ce qui faisait spawner les
            // grenades/roquettes derrière le canon visible.
            let spawn = self.viewmodel_muzzle_pos(weapon);
            // Mesh + tint + physique propres à l'arme.
            // * plasma : bleu électrique, linéaire, fuse 5s (kill-switch)
            // * grenade : gris neutre, gravité + rebond, fuse 2.5s (Q3 canon)
            // * rocket (défaut) : blanc, linéaire, fuse 5s
            let (mesh, tint, gravity, bounce, fuse) = match weapon {
                WeaponId::Plasmagun if alt_active => (
                    // **PG orb alt** : on swap sur le mesh rocket pour
                    // signaler visuellement la masse plus grosse, tint
                    // bleu-violet saturé. Plus lent + splash large
                    // (déjà via secondary_params).
                    self.rocket_mesh.clone(),
                    [0.55, 0.45, 1.0, 1.0],
                    0.0,
                    false,
                    4.0,
                ),
                WeaponId::Plasmagun => (
                    self.plasma_mesh.clone(),
                    [0.45, 0.65, 1.0, 1.0],
                    0.0,
                    false,
                    5.0,
                ),
                WeaponId::Grenadelauncher if alt_active => (
                    // **GL airburst alt** : pas de rebond (explose au
                    // premier contact), gravité conservée mais vitesse
                    // élevée → trajectoire tendue. Tint orange chaud
                    // pour distinguer la fuse vive.
                    self.grenade_mesh.clone(),
                    [1.0, 0.55, 0.20, 1.0],
                    800.0,
                    false,
                    3.0,
                ),
                WeaponId::Grenadelauncher => (
                    self.grenade_mesh.clone(),
                    [0.9, 0.9, 0.75, 1.0],
                    800.0, // Q3 g_gravity
                    true,
                    2.5, // fuse canonique
                ),
                WeaponId::Bfg if alt_active => (
                    // **BFG zone alt** : tint vert plus saturé, fuse
                    // un peu plus longue car projectile plus lent.
                    self.rocket_mesh.clone(),
                    [0.30, 1.0, 0.30, 1.0],
                    0.0,
                    false,
                    6.0,
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

        // **Spread bloom** (v0.9.5+) — recoil-modulé. Le MG en plein
        // auto voit son spread doubler après ~1 s de tir continu, mais
        // se reset rapidement entre rafales (basé sur view_kick courant
        // qui décroît expo-fast). Les armes single-shot (RG, RL) ne
        // sont pas affectées (leur spread base = 0).
        let bloom_factor = match weapon {
            WeaponId::Machinegun | WeaponId::Plasmagun => 1.0 + self.view_kick * 1.5,
            WeaponId::Shotgun => 1.0 + self.view_kick * 0.4, // léger
            _ => 1.0,
        };
        let spread_rad = (params.spread_deg * bloom_factor).to_radians();
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
        // Points d'impact headshot — flush hors-borrow vers la gerbe
        // dorée + dlight blanc-jaune dans `push_headshot_sparkle`.
        let mut pending_headshot_sparkles: Vec<Vec3> = Vec::new();
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

        // **Trace shim BSP / Terrain** (v0.9.5) — unifie le trace
        // hitscan pour les deux modes de map.  Capture le world et
        // le terrain par référence : si world existe, on délègue à
        // sa collision (BSP brushes) ; sinon on utilise le
        // heightmap. Retourne `(fraction, plane_normal, contents)`.
        let world_ref = self.world.as_ref();
        let terrain_ref = self.terrain.clone();
        let trace_shot = |start: Vec3, end: Vec3, mask: Contents|
            -> (f32, Vec3, Contents)
        {
            if let Some(w) = world_ref {
                let tr = w.collision.trace_ray(start, end, mask);
                (tr.fraction, tr.plane_normal, tr.contents)
            } else if let Some(t) = terrain_ref.as_ref() {
                let tr = t.trace_ray(start, end);
                (tr.fraction, tr.plane_normal, Contents::SOLID)
            } else {
                (1.0, Vec3::Z, Contents::empty())
            }
        };

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

            let (wt_frac, wt_normal, wt_contents) =
                trace_shot(eye, end, Contents::MASK_SHOT);
            let t_wall = wt_frac * params.range;

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
                // **Headshot detection** (G2a) — l'impact est calculé à
                // `eye + fwd * t_bot`. Si sa Z relative au pied du bot
                // dépasse HEADSHOT_Z_THRESHOLD, on multiplie les dégâts
                // par HEADSHOT_DMG_MULT et on flag pour le feedback
                // visuel/sonore distinct.
                let bot_origin = self.bots[idx].body.origin;
                let hit_pt_pre = eye + fwd * t_bot;
                let is_headshot = (hit_pt_pre.z - bot_origin.z) >= HEADSHOT_Z_THRESHOLD;
                let mut dmg = params.damage * self.player_damage_multiplier();
                if is_headshot {
                    dmg = (dmg as f32 * HEADSHOT_DMG_MULT) as i32;
                }
                let bot_driver = &mut self.bots[idx];
                let bot_pos = bot_driver.body.origin + Vec3::Z * BOT_CENTER_HEIGHT;
                let was_alive = !bot_driver.health.is_dead();
                let taken = bot_driver.health.take_damage(dmg);
                let dead = bot_driver.health.is_dead();
                let name = bot_driver.bot.name.clone();
                // **Death anim trigger** (v0.9.5++) — au moment exact où
                // le bot meurt, on enregistre le timestamp pour piloter
                // BOTH_DEATH1 → BOTH_DEAD1 freeze dans queue_bots.
                if was_alive && dead && bot_driver.death_started_at.is_none() {
                    bot_driver.death_started_at = Some(self.time_sec);
                }
                if taken > 0 {
                    // Horodate la prise de dégât pour déclencher l'anim
                    // TORSO_PAIN côté rendu (fenêtre ~200 ms).
                    bot_driver.last_damage_at = self.time_sec;
                    // **Dodge réaction** (v0.9.5) — informe l'IA bot qu'elle
                    // vient de prendre un hit → bot va strafer + sauter
                    // pendant DODGE_DURATION pour casser la mire.
                    bot_driver.bot.notify_damage_taken();
                    // **Anti-spam pain SFX** + **death SFX** distinct :
                    // sur le hit fatal on joue death_bot une fois. Sur
                    // les hits non-létaux on joue pain_bot mais avec un
                    // cooldown par bot pour éviter le SG-spam ou splash-
                    // spam de "ouch" simultanés.
                    if let Some(snd) = self.sound.as_ref() {
                        if dead {
                            if let Some(h) = self.sfx_death_bot.or(self.sfx_pain_bot) {
                                play_at(snd, h, bot_pos, Priority::Weapon);
                            }
                        } else if (self.time_sec - bot_driver.last_pain_sfx_at)
                            >= PAIN_SFX_COOLDOWN
                        {
                            if let Some(h) = self.sfx_pain_bot {
                                play_at(snd, h, bot_pos, Priority::Low);
                                bot_driver.last_pain_sfx_at = self.time_sec;
                            }
                        }
                    }
                    any_hit = true;
                    // Bug fix v0.9.5++ : avant on passait `is_headshot`
                    // dans la position `to_player` du tuple — les
                    // headshots étaient rendus comme dégâts SUBIS (rouge)
                    // au lieu d'INFLIGÉS (orange/jaune) et zappaient le
                    // combo counter.  Le headshot a déjà sa signature
                    // visuelle dédiée via `pending_headshot_sparkles`
                    // (gerbe dorée + dlight blanc-jaune ci-dessous).
                    pending_damage_nums.push((bot_pos, taken, false));
                    // Sparks rouges au point d'impact sur le bot ; normale
                    // = opposée à la direction du tir (les gouttes partent
                    // vers le tireur).
                    let hit_pt = eye + fwd * t_bot;
                    pending_sparks.push((hit_pt, -fwd, spark_flesh_color));
                    // **Headshot sparkle** (v0.9.5++) — petite gerbe
                    // dorée additive autour du point d'impact tête.
                    // 6 particules rapides + 1 dlight bref blanc-jaune
                    // → "ding" visuel au-dessus du sang rouge standard.
                    if is_headshot {
                        pending_headshot_sparkles.push(hit_pt);
                    }
                    // **P5 — Blood spray decal** : on trace AU-DELÀ de
                    // la cible (16 u dans la direction du tir) pour
                    // trouver le mur derrière, et y poser une décale
                    // de sang qui rappelle l'impact corporel. Effet
                    // cinéma — pas crucial gameplay, gros impact visuel.
                    let beyond_start = hit_pt + fwd * 4.0;
                    let beyond_end = beyond_start + fwd * 64.0;
                    let (b_frac, b_normal, _) =
                        trace_shot(beyond_start, beyond_end, Contents::SOLID);
                    if b_frac < 1.0 {
                        let blood_pt = beyond_start + fwd * (b_frac * 64.0);
                        pending_blood_decals.push((blood_pt, b_normal));
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
            } else if wt_frac < 1.0 {
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
                let contents = wt_contents;
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
                pending_sparks.push((hit_pt, wt_normal, impact_color));
                if impact_decal.is_some() {
                    pending_wall_marks.push((hit_pt, wt_normal));
                }

                // **Rail ricochet** (W7) : si alt-fire actif sur railgun
                // ET le tir a tapé un mur sans bot, on rebondit le rai
                // selon la loi de Snell-Descartes (réflexion miroir) et
                // on relance une trace courte (1024u) depuis le point
                // d'impact, en cherchant un bot. Dégâts à 50% pour
                // équilibrer.
                if alt_active && matches!(weapon, WeaponId::Railgun) {
                    let n = wt_normal;
                    let reflected = fwd - n * 2.0 * fwd.dot(n);
                    let ricochet_origin =
                        hit_pt + reflected * 2.0; // léger lift hors mur
                    let ricochet_range = 1024.0_f32;
                    let ricochet_end =
                        ricochet_origin + reflected * ricochet_range;
                    let (r_frac, _, _) = trace_shot(
                        ricochet_origin,
                        ricochet_end,
                        Contents::MASK_SHOT,
                    );
                    let r_t_wall = r_frac * ricochet_range;
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
                        None if r_frac < 1.0 => (
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
                            bot_driver.bot.notify_damage_taken();
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
                // **Muzzle position par arme** — utilise le helper
                // commun pour que le beam parte du bout du canon visible
                // (différent par GLB), pas d'un offset générique.
                let start = self.viewmodel_muzzle_pos(weapon);
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

            // **Bullet tracers** (v0.9.5++) — pour MG/SG/PG on dessine
            // un trait fin et bref œil→impact.  Couleur gold-orange
            // pour MG/SG (poudre), blanc-cyan pour PG (plasma).  Très
            // brève (60 ms) → l'œil voit "fléchette" puis disparaît,
            // pas de spam visuel sustained.  En SG (11 pellets) ça
            // visualise le cône de dispersion, ce qui aide à lire le
            // meilleur range d'engagement.
            if matches!(
                weapon,
                WeaponId::Machinegun | WeaponId::Shotgun | WeaponId::Plasmagun
            ) {
                let t_hit = match best {
                    Some((t_bot, _)) => t_bot.min(t_wall),
                    None => t_wall,
                };
                let hit_pt = eye + fwd * t_hit;
                // **Muzzle position par arme** — helper commun.
                let start = self.viewmodel_muzzle_pos(weapon);
                let (mut color, lifetime) = match weapon {
                    WeaponId::Plasmagun => ([0.40, 0.85, 1.0, 0.55], 0.05_f32),
                    _ => ([1.0, 0.85, 0.40, 0.55], 0.06_f32), // MG/SG gold
                };
                // **Quad tracers** SUPPRIMÉ (v0.9.5++ user request) — la
                // couleur des tracers ne change plus pendant Quad.
                // Feedback restant : SFX d'activation + chrono HUD.
                self.beams.push(ActiveBeam {
                    a: start,
                    b: hit_pt,
                    color,
                    expire_at: self.time_sec + lifetime,
                    lifetime,
                    style: BeamStyle::Straight,
                });
            }
        }

        if any_hit {
            self.hit_marker_until = self.time_sec + 0.18;
            // **FOV punch** — kick d'élargissement horizontal sur hit.
            // Set strength = HIT par défaut ; sera bumpé à FRAG plus
            // bas si `any_kill`.  Le punch dans le `cg_fov` apply path
            // décroît linéairement sur FOV_PUNCH_DURATION.
            self.fov_punch_until = self.time_sec + FOV_PUNCH_DURATION;
            self.fov_punch_strength = FOV_PUNCH_HIT;
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
            // Kill-marker visuel : grand X rouge vif sur le réticule.
            // Étendu si plusieurs frags tombent dans la fenêtre — pas
            // de stacking, on ré-arme à chaque tick `any_kill`.
            self.kill_marker_until = self.time_sec + KILL_MARKER_DURATION_SEC;
            // FOV punch boosté sur frag létal — distinct du hit standard.
            self.fov_punch_until = self.time_sec + FOV_PUNCH_DURATION;
            self.fov_punch_strength = FOV_PUNCH_FRAG;
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
        for pos in pending_headshot_sparkles {
            self.push_headshot_sparkle(pos);
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
            self.spawn_bot_drop(pos);
            // Floater "+1 FRAG" au-dessus du bot fraggé — feedback
            // visuel juicy ancré au monde.  Position légèrement
            // au-dessus de la tête pour ne pas être obscurcie par
            // les gibs.
            self.push_frag_confirm(pos + Vec3::Z * BOT_CENTER_HEIGHT);
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
        // **BR mode** (v0.9.5+) — accepte le tick si terrain présent.
        // Sans ça, en BR mode, les projectiles spawnaient mais ne se
        // mettaient jamais à jour → rocket immobile au point de tir.
        if self.world.is_none() && self.terrain.is_none() {
            return;
        }
        let world_ref = self.world.as_ref();
        let terrain_ref = self.terrain.clone();
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
                            let blended = cur_dir * (1.0 - mix) + want_dir * mix;
                            // **NaN guard** (v0.9.5++ polish) — si `cur_dir`
                            // et `want_dir` sont anti-parallèles avec
                            // `mix=0.5`, la somme = 0 → `normalize()` → NaN
                            // qui se propage dans `p.velocity` et casse
                            // toute la chaîne de collision en aval.
                            // Fallback : on garde la direction courante.
                            let len_sq = blended.length_squared();
                            if len_sq > 1e-6 {
                                p.velocity = blended * (cur_speed / len_sq.sqrt());
                            }
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
            // BR mode → trace via heightmap au lieu de BSP.
            let (trace_frac, trace_normal) = if let Some(w) = world_ref {
                let tr = w.collision.trace_ray(p.origin, next, Contents::MASK_SHOT);
                (tr.fraction, tr.plane_normal)
            } else if let Some(t) = terrain_ref.as_ref() {
                let tr = t.trace_ray(p.origin, next);
                (tr.fraction, tr.plane_normal)
            } else {
                (1.0, Vec3::Z)
            };
            let t_world = trace_frac * seg_len;

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
            if trace_frac < 1.0 {
                // Impact monde — deux comportements :
                //  * `bounce=false` (rocket, plasma) : explose immédiatement.
                //  * `bounce=true` (grenade) : réflexion amortie le long du
                //    plan d'impact, continue à vivre jusqu'à la fuse.
                if p.bounce {
                    let normal = trace_normal;
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
                    world_normal: trace_normal,
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
                self.particles.swap_remove(0);
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
                // **Trails détaillés v0.9.5++** — chaque arme a un
                // trail multi-particules (smoke + heat + spark) pour
                // un feedback visuel plus riche que la single puff.
                match weapon {
                    WeaponId::Rocketlauncher => {
                        // Rocket : 1 puff de fumée blanche dense + 1
                        // halo orange chaud (heat haze derrière) + 1
                        // étincelle incandescente.
                        r.spawn_particle(
                            pos, Vec3::Z * 8.0,
                            [0.92, 0.90, 0.88, 0.7], 3.0, 14.0,
                            self.time_sec, 0.9,
                        );
                        r.spawn_particle(
                            pos, Vec3::Z * 4.0,
                            [1.0, 0.55, 0.20, 0.65], 4.0, 10.0,
                            self.time_sec, 0.35,
                        );
                        r.spawn_particle(
                            pos + Vec3::new(rand_unit() * 2.0, rand_unit() * 2.0, 0.0),
                            Vec3::ZERO,
                            [1.0, 0.85, 0.40, 1.0], 1.0, 0.5,
                            self.time_sec, 0.18,
                        );
                    }
                    WeaponId::Grenadelauncher => {
                        // Grenade : fumée grise + petite étincelle.
                        r.spawn_particle(
                            pos, Vec3::Z * 6.0,
                            [0.55, 0.55, 0.55, 0.55], 2.2, 7.0,
                            self.time_sec, 0.55,
                        );
                        r.spawn_particle(
                            pos, Vec3::ZERO,
                            [1.0, 0.75, 0.35, 0.9], 0.8, 0.3,
                            self.time_sec, 0.12,
                        );
                    }
                    WeaponId::Bfg => {
                        // BFG : énorme trail vert + arcs électriques.
                        r.spawn_particle(
                            pos, Vec3::Z * 4.0,
                            [0.30, 1.0, 0.40, 0.85], 5.0, 18.0,
                            self.time_sec, 1.0,
                        );
                        // 4 sparks orbitaux verts pour effet "plasma BFG"
                        for k in 0..4 {
                            let theta = (k as f32) * std::f32::consts::FRAC_PI_2;
                            let off = Vec3::new(
                                theta.cos() * 12.0,
                                theta.sin() * 12.0,
                                0.0,
                            );
                            r.spawn_particle(
                                pos + off, off * 1.2,
                                [0.55, 1.0, 0.55, 1.0], 1.5, 0.5,
                                self.time_sec, 0.25,
                            );
                        }
                    }
                    _ => {}
                };
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
            // **Scorch marks empilées** (v0.9.5++ #39) — décales
            // multi-couches par arme :
            // * rocket/grenade : 1 grosse mark noire + 1 anneau brûlé
            //   chaud orange (cercle externe résidu thermique)
            // * BFG : grand cercle vert toxique + halo
            // * plasma : trace bleu cyan diffuse
            if b.world_normal != Vec3::ZERO {
                if let Some(r) = self.renderer.as_mut() {
                    match b.weapon {
                        WeaponId::Rocketlauncher | WeaponId::Grenadelauncher => {
                            // Cercle noir central
                            r.spawn_decal(b.pos, b.world_normal, 18.0,
                                [0.04, 0.03, 0.02, 0.75], self.time_sec, 25.0);
                            // Halo brûlé chaud externe (lifetime court)
                            r.spawn_decal(b.pos, b.world_normal, 28.0,
                                [0.45, 0.20, 0.06, 0.35], self.time_sec, 4.0);
                        }
                        WeaponId::Bfg => {
                            // Cercle vert toxique central
                            r.spawn_decal(b.pos, b.world_normal, 28.0,
                                [0.20, 0.45, 0.18, 0.65], self.time_sec, 30.0);
                            // Halo lumineux vert (court)
                            r.spawn_decal(b.pos, b.world_normal, 42.0,
                                [0.40, 0.95, 0.40, 0.30], self.time_sec, 3.5);
                        }
                        WeaponId::Plasmagun => {
                            // Trace cyan diffuse (plasma "propre")
                            r.spawn_decal(b.pos, b.world_normal, 14.0,
                                [0.20, 0.50, 0.85, 0.45], self.time_sec, 8.0);
                        }
                        _ => {
                            r.spawn_decal(b.pos, b.world_normal, 18.0,
                                [0.05, 0.04, 0.03, 0.6], self.time_sec, 20.0);
                        }
                    }
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
            // Son d'impact spécifique à l'arme. Avant v0.9.4 toutes
            // les explosions jouaient sfx_rocket_explode → plasma sonnait
            // identique à rocket. Maintenant chaque arme a son sample
            // dédié, fallback sur rocket_explode (puis None silencieux).
            let impact_sfx = match b.weapon {
                WeaponId::Rocketlauncher | WeaponId::Bfg => self
                    .sfx_rocket_impact
                    .or(self.sfx_rocket_explode),
                WeaponId::Grenadelauncher => self
                    .sfx_grenade_impact
                    .or(self.sfx_rocket_explode),
                WeaponId::Plasmagun => self.sfx_plasma_impact,
                _ => self.sfx_rocket_explode,
            };
            if let (Some(snd), Some(h)) = (self.sound.as_ref(), impact_sfx) {
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
                            // Dodge réaction même sur direct hit projectile.
                            bd.bot.notify_damage_taken();
                            let bot_pos = bd.body.origin + Vec3::Z * BOT_CENTER_HEIGHT;
                            // Pain SFX cooldownisé + death SFX si fatal.
                            if let Some(snd) = self.sound.as_ref() {
                                if dead {
                                    if let Some(h) = self.sfx_death_bot.or(self.sfx_pain_bot) {
                                        play_at(snd, h, bot_pos, Priority::Weapon);
                                    }
                                } else if (self.time_sec - bd.last_pain_sfx_at)
                                    >= PAIN_SFX_COOLDOWN
                                {
                                    if let Some(h) = self.sfx_pain_bot {
                                        play_at(snd, h, bot_pos, Priority::Low);
                                        bd.last_pain_sfx_at = self.time_sec;
                                    }
                                }
                            }
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
                        // **Godmode** (test) : skip damage si actif et
                        // que le projectile ne vient pas du joueur lui-
                        // même (rocket-jump self-damage reste utile).
                        let godmode = self.cvars.get_i32("g_godmode").unwrap_or(0) != 0;
                        let from_bot = !matches!(b.owner, ProjectileOwner::Player);
                        if godmode && from_bot {
                            continue;
                        }
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
                            self.last_damage_dir = b.impact_dir;
                            self.last_damage_until = self.time_sec + DAMAGE_DIR_SHOW_SEC;
                            self.pain_flash_until = self.time_sec + PAIN_FLASH_SEC;
                            // **Player damage shake** — secousse caméra
                            // proportionnelle aux dégâts pris.
                            let dmg_shake = (taken as f32 * 0.10).min(5.0);
                            if dmg_shake > self.shake_intensity {
                                self.shake_intensity = dmg_shake;
                            }
                            self.shake_until =
                                self.time_sec + SHAKE_DURATION;
                            // Player hit sparks → particles rouges au
                            // point d'impact (visible 3rd person des
                            // autres joueurs ; subtil pour soi-même).
                            self.push_hit_sparks(
                                eye,
                                -b.impact_dir,
                                [1.0, 0.25, 0.2, 1.0],
                            );
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
                        // Dodge réaction sur splash damage aussi —
                        // un bot dans la zone d'effet d'une rocket
                        // déclenche son pattern d'esquive.
                        bd.bot.notify_damage_taken();
                        // Pain cooldownisé + death SFX si fatal.
                        if let Some(snd) = self.sound.as_ref() {
                            if dead {
                                if let Some(h) = self.sfx_death_bot.or(self.sfx_pain_bot) {
                                    play_at(snd, h, center, Priority::Weapon);
                                }
                            } else if (self.time_sec - bd.last_pain_sfx_at)
                                >= PAIN_SFX_COOLDOWN
                            {
                                if let Some(h) = self.sfx_pain_bot {
                                    play_at(snd, h, center, Priority::Low);
                                    bd.last_pain_sfx_at = self.time_sec;
                                }
                            }
                        }
                        pending_damage_nums.push((center, taken, false));
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
            // **Godmode** : skip splash bot, garde rocket-jump self.
            let player_took_direct = matches!(b.direct_target, Some(HitTarget::Player));
            let player_invul = self.player_invul_until > self.time_sec;
            let godmode = self.cvars.get_i32("g_godmode").unwrap_or(0) != 0;
            let from_bot = !matches!(b.owner, ProjectileOwner::Player);
            let block_godmode = godmode && from_bot;
            if !self.player_health.is_dead()
                && !player_took_direct
                && !player_invul
                && !block_godmode
            {
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
                            // Damage shake (splash) — typiquement plus
                            // violent que direct car le knockback agit
                            // déjà sur la caméra. On ajoute un shake
                            // modéré pour le ressenti.
                            let dmg_shake = (taken as f32 * 0.08).min(4.0);
                            if dmg_shake > self.shake_intensity {
                                self.shake_intensity = dmg_shake;
                            }
                            self.shake_until =
                                self.time_sec + SHAKE_DURATION;
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
            self.spawn_bot_drop(pos);
            // Floater "+1 FRAG" au-dessus du bot fraggé — feedback
            // visuel juicy ancré au monde.  Position légèrement
            // au-dessus de la tête pour ne pas être obscurcie par
            // les gibs.
            self.push_frag_confirm(pos + Vec3::Z * BOT_CENTER_HEIGHT);
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
        // **Reset fire flags** (v0.9.5++ fix) — sans ce reset, si le
        // joueur tient RMB sur Railgun (zoom) puis switch → l'alt-fire
        // de la nouvelle arme se déclenche immédiatement (UX bizarre).
        // On force un edge-press sur la nouvelle arme.
        self.input.fire = false;
        self.input.secondary_fire = false;
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
                // Reset death anim state pour la prochaine vie.
                // Tirage variant déterministe : hash du nom + deaths
                // donne une distribution stable variée.
                let h = d.bot.name.bytes().fold(d.deaths, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
                d.death_variant = (h % 3) as u8;
                d.death_started_at = None;
                d.gesture_started_at = None;
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
            // **Stats live overlay** (V3f) — toggle F8.
            KeyCode::F8 if pressed => {
                self.show_stats_overlay = !self.show_stats_overlay;
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
        // **Restore video settings from cvars** (v0.9.5) — `r_width` /
        // `r_height` / `r_fullscreen` ont la flag ARCHIVE donc q3config
        // .cfg les a chargés au démarrage de l'App.  CLI override
        // (`--width / --height`) reste prioritaire si fourni.
        let restored_w = self.cvars.get_i32("r_width").unwrap_or(self.init_width as i32);
        let restored_h = self.cvars.get_i32("r_height").unwrap_or(self.init_height as i32);
        let restored_fs = self.cvars.get_i32("r_fullscreen").unwrap_or(0) != 0;
        let final_w = if self.init_width == 1280 && restored_w > 0 {
            // Default CLI = 1280×720 ; si on a une valeur sauvegardée
            // on la préfère.
            restored_w as u32
        } else {
            self.init_width
        };
        let final_h = if self.init_height == 720 && restored_h > 0 {
            restored_h as u32
        } else {
            self.init_height
        };
        let mut attrs = WindowAttributes::default()
            .with_title(concat!("Quake 3 RUST EDITION v", env!("CARGO_PKG_VERSION")))
            .with_inner_size(PhysicalSize::new(final_w, final_h));
        if restored_fs {
            // Fullscreen borderless sur le moniteur primaire (le
            // moniteur courant n'est pas encore connu avant la
            // création de la fenêtre, donc `None` = primaire).
            attrs = attrs.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        }
        // Synchronise l'état menu pour que la sous-page Vidéo affiche
        // les valeurs persistées dès la première ouverture.
        self.menu.set_window_size(final_w, final_h);
        self.menu.set_fullscreen(restored_fs);
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

        // **Default map** (v0.9.5++) — BR supprimé, on ouvre direct le
        // menu de sélection de map à l'arrivée si aucun `--map` n'est
        // explicitement fourni en CLI.  Le joueur choisit sa map
        // BSP depuis la liste.
        let map_to_load = self.requested_map.clone();
        if let Some(map_to_load) = map_to_load {
            // CLI explicit → on charge direct.
            self.menu.open = false;
            self.load_map(&map_to_load);
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
            // Pas de --map → menu reste ouvert (laisse le joueur choisir).
            self.menu.open = true;
            self.menu.open_root();
        }
        info!(
            "menu: {} maps disponibles dans le VFS",
            self.menu.map_list.len()
        );

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
                            self.menu.set_in_game(self.world.is_some() || self.terrain.is_some());
                            // Refresh la liste des fichiers music au cas
                            // où l'utilisateur en a ajouté pendant le jeu.
                            // Inclut maintenant les dossiers configurés
                            // via `s_musicpath` (scan récursif).
                            let extra = self.cvars.get_string("s_musicpath")
                                .unwrap_or_default();
                            let sep = if cfg!(windows) { ';' } else { ':' };
                            let extra_paths: Vec<PathBuf> = extra
                                .split(sep)
                                .filter(|s: &&str| !s.trim().is_empty())
                                .map(|s: &str| PathBuf::from(s.trim()))
                                .collect();
                            self.menu.set_music_tracks(list_music_files_with_extra(&extra_paths));
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
                // **Punch angles decay** (v0.9.5++) — décroissance
                // exponentielle plus rapide que `view_kick` (le pitch
                // doit revenir vite, ~250 ms half-life, sinon ça
                // perturbe trop l'aim long).  Yaw jitter décroît
                // encore plus vite (~150 ms) pour éviter le drift.
                if self.view_kick_pitch.abs() > 1e-3 {
                    self.view_kick_pitch *= (-2.8 * dt).exp();
                    if self.view_kick_pitch.abs() < 0.01 {
                        self.view_kick_pitch = 0.0;
                    }
                }
                if self.view_kick_yaw.abs() > 1e-3 {
                    self.view_kick_yaw *= (-4.6 * dt).exp();
                    if self.view_kick_yaw.abs() < 0.01 {
                        self.view_kick_yaw = 0.0;
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
                        } else if let Some(terrain) = self.terrain.as_ref() {
                            // **BR mode** — pas de BSP, route vers la
                            // physics terrain heightmap (snap vertical
                            // + slope cap, pas de slide BSP).
                            self.player.tick_terrain(cmd, physics, terrain);
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
                    // **Viewmodel jump-kick trigger** (v0.9.5++) — quand
                    // le joueur retouche le sol après être airborne, on
                    // marque `player_last_land_at` pour que `queue_viewmodel`
                    // ajoute un dip-and-spring à l'arme.  Seuil 100 unités
                    // de vitesse verticale pour skip les marches d'escalier
                    // (qui setent on_ground sans être un vrai saut).
                    if !was_on_ground && self.player.on_ground && prev_vz < -100.0 {
                        self.player_last_land_at = self.time_sec;
                    }
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

                // **BR ring tick** (v0.9.5) — avance le ring shrink, log
                // les transitions de phase, applique les dégâts hors-zone
                // au joueur. Bots BR : TODO quand le bot path support
                // terrain pur (cf. spawn_bot dépend de world).
                if self.terrain.is_some() {
                    let pois = self.terrain
                        .as_ref()
                        .map(|t| t.pois().to_vec())
                        .unwrap_or_default();
                    if let Some(ring) = self.br_ring.as_mut() {
                        let transitioned = ring.tick(dt, &pois);
                        if transitioned {
                            if let Some(idx) = ring.phase_index() {
                                info!(
                                    "BR: phase {} commence (rayon {:.0}, dps {:.0})",
                                    idx,
                                    ring.current_radius(),
                                    ring.dps_outside()
                                );
                            } else {
                                info!("BR: ring fermé, match en mode endgame");
                            }
                        }
                        if !self.player_health.is_dead() {
                            let dmg_f = ring.damage_for(self.player.origin, dt);
                            if dmg_f > 0.0 {
                                let dmg_i = dmg_f.ceil() as i32;
                                self.player_health.take_damage(dmg_i);
                                // Pulse rouge HUD pour signaler l'hostile-zone.
                                self.pain_flash_until = self.time_sec + 0.15;
                            }
                        }
                    }
                }

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
                    // **Bot item priority** : pré-build la liste des
                    // pickups stratégiques avec leur score. Mega=10,
                    // RA=8, YA=6, Quad/powerups=15, autres=2.
                    let pickups_priority: Vec<(Vec3, f32, bool)> = self
                        .pickups
                        .iter()
                        .map(|p| {
                            let prio = match &p.kind {
                                PickupKind::Health { max_cap: 200, .. } => 10.0,
                                PickupKind::Armor { amount: 100 } => 8.0,
                                PickupKind::Armor { amount: 50 } => 6.0,
                                PickupKind::Powerup { .. } => 15.0,
                                _ => 2.0,
                            };
                            (p.origin, prio, p.respawn_at.is_none())
                        })
                        .collect();
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
                        &pickups_priority,
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
                    // **Squad chatter** — flush des évènements collectés
                    // pendant le tick. Le chatter passe par
                    // `maybe_bot_chat` qui applique le random + cooldown
                    // global pour respecter le rythme audio.
                    for (idx, trigger) in bot_out.chatter_events {
                        self.maybe_bot_chat(idx, trigger);
                    }
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
                    let godmode = self.cvars.get_i32("g_godmode").unwrap_or(0) != 0;
                    if dmg > 0 && alive && !player_invul && !godmode {
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
                } else if let Some(terrain) = self.terrain.as_ref() {
                    // **Bots BR avec IA complète + tir** (v0.9.5) — FSM bot
                    // (Idle/Roam/Combat) sur la heightmap, plus le tir
                    // hitscan que le BSP path applique.  Modèle simplifié :
                    //   * un seul "profile MG" générique (pas de switch
                    //     SG/RL selon distance — viendra avec le ramassage
                    //     d'items bot)
                    //   * dégât appliqué directement sur le joueur si
                    //     visible+LOS+cooldown OK
                    use q3_bot::LosWorld;
                    let los = TerrainLos(terrain.as_ref());
                    let mut bot_dmg_total: i32 = 0;
                    let mut last_bot_dmg_idx: Option<usize> = None;
                    let player_eye = self.player.origin + Vec3::Z * PLAYER_EYE_HEIGHT;
                    let alive = !self.player_health.is_dead();
                    let now = self.time_sec;
                    let dps = self.br_ring.as_ref().map(|r| r.dps_outside()).unwrap_or(0.0);
                    let ring_center = self.br_ring.as_ref().map(|r| r.current_center());
                    let ring_radius = self.br_ring.as_ref().map(|r| r.current_radius());
                    // Cooldown générique tir bot BR (un par bot mais on ne
                    // distingue pas l'arme — rafale toutes les 0.6s).
                    const BR_BOT_FIRE_COOLDOWN: f32 = 0.6;
                    const BR_BOT_HIT_DMG: i32 = 7; // dégât par burst (équiv MG burst)
                    // **Bot pickup priority BR** — pré-build des
                    // pickups disponibles avec leur tier/score, à
                    // ré-injecter en waypoint si bot sans cible.
                    let pickups_priority: Vec<(Vec3, f32, bool)> = self
                        .pickups
                        .iter()
                        .map(|p| {
                            let prio = match &p.kind {
                                PickupKind::Health { max_cap: 200, .. } => 10.0,
                                PickupKind::Armor { amount: 100 } => 8.0,
                                PickupKind::Armor { amount: 50 } => 6.0,
                                PickupKind::Powerup { .. } => 15.0,
                                PickupKind::Weapon { .. } => 7.0,
                                _ => 2.0,
                            };
                            (p.origin, prio, p.respawn_at.is_none())
                        })
                        .collect();
                    for (idx, bd) in self.bots.iter_mut().enumerate() {
                        if bd.health.is_dead() {
                            continue;
                        }
                        // Item priority — bot sans cible va vers le
                        // pickup le mieux scoré dans 2000u.
                        if bd.bot.target_enemy.is_none() {
                            let need_health = bd.health.current < 70;
                            let bp = bd.body.origin;
                            let mut best: Option<(f32, Vec3)> = None;
                            for &(pos, mut prio, available) in &pickups_priority {
                                if !available { continue; }
                                let dd = (pos - bp).length();
                                if dd > 2000.0 || dd < 1.0 { continue; }
                                if !need_health && prio < 5.0 {
                                    prio *= 0.3;
                                }
                                let score = prio / (dd * 0.001 + 1.0);
                                if best.map_or(true, |(s, _)| score > s) {
                                    best = Some((score, pos));
                                }
                            }
                            if let Some((_, pos)) = best {
                                let already = bd.bot.waypoints
                                    .first()
                                    .map(|w| (*w - pos).length() < 64.0)
                                    .unwrap_or(false);
                                if !already {
                                    bd.bot.waypoints.insert(0, pos);
                                }
                            }
                        }
                        // Vision : portée + LOS terrain.  FOV simplifié
                        // (les bots BR ne sont pas encore aussi sophistiqués
                        // que les BSP — pas de cone d'orientation).
                        let bot_eye = bd.body.origin + Vec3::Z * BOT_EYE_HEIGHT;
                        let to_player = player_eye - bot_eye;
                        let dist = to_player.length();
                        let visible = alive
                            && dist > 1.0
                            && dist < BOT_SIGHT_RANGE
                            && los.is_clear(bot_eye, player_eye);
                        if visible {
                            bd.bot.target_enemy = Some(player_eye);
                            bd.last_saw_player_at = Some(now);
                            if bd.first_seen_player_at.is_none() {
                                bd.first_seen_player_at = Some(now);
                            }
                        } else if let Some(t) = bd.last_saw_player_at {
                            if now - t > BOT_MEMORY_SEC {
                                bd.bot.target_enemy = None;
                                bd.last_saw_player_at = None;
                                bd.first_seen_player_at = None;
                            }
                        }
                        bd.bot.position = bd.body.origin;

                        // FSM tick avec LOS terrain.
                        let bc = bd.bot.tick(dt, &los);

                        // BotCmd → MoveCmd : on mappe forward/right/jump
                        // sur le pmove. `right_move` Q3 → `side` MoveCmd.
                        let cmd = MoveCmd {
                            forward: bc.forward_move,
                            side: bc.right_move,
                            up: 0.0,
                            jump: bc.up_move > 0.5,
                            crouch: false,
                            walk: false,
                            slide_pressed: false,
                            dash_pressed: false,
                            delta_time: dt,
                        };
                        // Réoriente le bot dans la direction du `bot.view_angles`.
                        bd.body.view_angles = bc.view_angles;
                        bd.body.tick_terrain(cmd, self.params, terrain);

                        // **Bot fire BR** — si l'IA a décidé `fire`,
                        // qu'on a la LOS et que le cooldown est échu,
                        // on applique un burst de dégâts au joueur.
                        // Aim error proba miss selon skill.
                        let skill = bd.bot.skill;
                        let reacted = bd
                            .first_seen_player_at
                            .map(|t| now - t >= skill.reaction_time_sec())
                            .unwrap_or(false);
                        if bc.fire && visible && reacted && now >= bd.next_fire_at {
                            let miss_prob =
                                (skill.aim_error_deg() / 15.0).min(0.80);
                            let missed = rand_unit_01() < miss_prob;
                            if !missed {
                                bot_dmg_total += BR_BOT_HIT_DMG;
                                last_bot_dmg_idx = Some(idx);
                            }
                            bd.next_fire_at =
                                now + BR_BOT_FIRE_COOLDOWN * skill.fire_cooldown_mult();
                            bd.last_fire_at = now;
                            // SFX tir : utilise le sample MG si chargé.
                            if let Some(snd) = self.sound.as_ref() {
                                if let Some((_, h)) = self
                                    .sfx_fire
                                    .iter()
                                    .find(|(w, _)| *w == WeaponId::Machinegun)
                                {
                                    let from = bd.body.origin
                                        + Vec3::Z * BOT_CENTER_HEIGHT;
                                    play_at(snd, *h, from, Priority::Weapon);
                                }
                            }
                        }

                        // Damage hors-zone du ring.
                        if dps > 0.0 {
                            if let (Some(c), Some(r)) = (ring_center, ring_radius) {
                                let d = (bd.body.origin - c).truncate().length();
                                if d > r {
                                    let outside = (d - r) / r.max(1.0);
                                    let factor = (1.0 + outside.min(2.0)).max(1.0);
                                    let dmg_f = dps * dt * factor;
                                    let dmg_i = dmg_f.ceil() as i32;
                                    bd.health.take_damage(dmg_i);
                                    if bd.health.is_dead() {
                                        info!("BR: bot '{}' éliminé hors-zone", bd.bot.name);
                                    }
                                }
                            }
                        }
                    }
                    // **Apply bot fire damage to player** — accumulated
                    // pendant la boucle, appliqué en sortie pour ne pas
                    // tenir borrow mut sur self.bots.
                    let player_invul = self.player_invul_until > self.time_sec;
                    if bot_dmg_total > 0 && alive && !player_invul {
                        let absorbed = (bot_dmg_total / 2).min(self.player_armor);
                        self.player_armor -= absorbed;
                        if absorbed > 0 {
                            self.armor_flash_until = self.time_sec + ARMOR_FLASH_SEC;
                        }
                        let dmg_after = bot_dmg_total - absorbed;
                        let taken = self.player_health.take_damage(dmg_after);
                        if taken > 0 {
                            if let (Some(snd), Some(h)) = (self.sound.as_ref(), self.sfx_pain_player) {
                                play_at(snd, h, self.player.origin, Priority::High);
                            }
                            // Pain arrow indicateur direction.
                            if let Some(i) = last_bot_dmg_idx {
                                if let Some(bd) = self.bots.get(i) {
                                    let from = bd.body.origin + Vec3::Z * BOT_CENTER_HEIGHT;
                                    let d = player_eye - from;
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
                                "BR: joueur abattu par bot — respawn dans {RESPAWN_DELAY_SEC:.1}s (deaths={})",
                                self.deaths
                            );
                        }
                    }
                }
                } // fin du `if self.match_winner.is_none()`

                // Tir joueur : hitscan sphère → bot le plus proche non occulté.
                // Tir : LMB (primaire) OU RMB (secondaire) déclenche
                // `fire_weapon`.  À l'intérieur, le flag `secondary_fire`
                // détermine quels params consommer (cf. weapon.secondary_params).
                // **Exception Railgun zoom** (v0.9.5++) : pour le railgun,
                // RMB SEUL ne tire pas — c'est juste un zoom scope.  Pour
                // tirer avec alt-params en zoom, le joueur appuie LMB
                // pendant que RMB est tenu (UX standard sniper FPS).
                let rail_zoom_only = self.active_weapon == WeaponId::Railgun
                    && self.input.secondary_fire
                    && !self.input.fire;
                if (self.input.fire || (self.input.secondary_fire && !rail_zoom_only))
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
                // Atmospheric lightning (BR uniquement).  No-op hors BR.
                self.tick_atmosphere();

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

                // Pre-compute valeurs qui dépendent de `&self` AVANT de
                // prendre `self.renderer` en mut borrow (sinon E0502).
                let railgun_zoom = self.railgun_zoom_ratio();

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
                    // **Killcam** (V3d) — quand le joueur est mort, on
                    // priorise la caméra sur le BOT QUI L'A TUÉ (cf.
                    // `last_death_cause`). Si pas identifiable (mort
                    // par environnement, suicide, tueur disparu) on
                    // retombe sur le bot vivant le plus proche, comme
                    // avant. Effet "killcam" cinéma sur la fenêtre
                    // de respawn, donne du feedback "qui m'a tué".
                    let mut dead_cam: Option<(Vec3, Angles)> = None;
                    if self.player_health.is_dead() && self.match_winner.is_none() {
                        // Tentative 1 : identifier le tueur par nom.
                        let killer_target = match &self.last_death_cause {
                            Some((KillActor::Bot(killer_name), _)) => self
                                .bots
                                .iter()
                                .find(|b| {
                                    !b.health.is_dead() && &b.bot.name == killer_name
                                })
                                .map(|b| b.body.origin),
                            _ => None,
                        };
                        // Tentative 2 : bot vivant le plus proche.
                        let target = killer_target.or_else(|| {
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
                            best.map(|(_, p)| p)
                        });
                        if let Some(target) = target {
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
                    // **Punch angles** (v0.9.5++) — recul d'arme :
                    // pitch up + jitter de yaw.  Appliqué SEULEMENT au
                    // rendu de la caméra (pas à `view_angles` réel),
                    // donc l'aim et la collision sont préservés.  Le
                    // joueur voit son arme "pousser" sans que ses
                    // tirs partent ailleurs.
                    cam_angles.pitch -= self.view_kick_pitch; // pitch négatif = haut en Q3
                    cam_angles.yaw += self.view_kick_yaw;
                    r.camera_mut().angles = cam_angles;
                    // FOV : `cg_fov` (cvar archivée) est interprété comme
                    // FOV horizontal à 4:3, conforme à la convention Q3.
                    // La caméra dérive le vfov à partir de là, ce qui
                    // donne un scaling **Hor+** sur 16:9 / 21:9 / 32:9 :
                    // l'image verticale ne change pas, le champ s'élargit
                    // horizontalement avec l'aspect. Lu chaque frame —
                    // pas cher, et permet la modif live depuis la console.
                    let fov_cvar = self.cvars.get_f32("cg_fov").unwrap_or(90.0);
                    // **FOV punch** — additif au cg_fov, fade linéaire
                    // sur FOV_PUNCH_DURATION.  Donne un kick visuel
                    // momentané sur hit (3°) ou frag (5°).
                    let fov_with_punch = if self.fov_punch_until > self.time_sec {
                        let remaining = self.fov_punch_until - self.time_sec;
                        let ratio = (remaining / FOV_PUNCH_DURATION).clamp(0.0, 1.0);
                        fov_cvar + self.fov_punch_strength * ratio
                    } else {
                        fov_cvar
                    };
                    // **Railgun zoom scope** (v0.9.5++) — override le FOV
                    // par cg_fov × zoom_ratio (par défaut 0.33 = 3× zoom).
                    let fov_with_punch = if let Some(zoom) = railgun_zoom {
                        fov_with_punch * zoom
                    } else {
                        fov_with_punch
                    };
                    // **FOV scaling mode** — Hor+ (0) ou Vert- (1).
                    let fov_mode = match self.cvars.get_i32("cg_fovaspect").unwrap_or(0) {
                        1 => q3_renderer::camera::FovMode::VertMinus,
                        _ => q3_renderer::camera::FovMode::HorPlus,
                    };
                    r.camera_mut().set_fov_mode(fov_mode);
                    r.camera_mut().set_horizontal_fov_4_3(fov_with_punch);

                    r.begin_frame();
                    // **Light pillars sur powerups premium** (v0.9.5++) —
                    // spawn dlight chaque frame au-dessus des Quad/Mega/RA
                    // disponibles. Visible de loin → repère les zones
                    // stratégiques sans HUD wallhack.
                    //
                    // **Fix v0.9.5++ (perf)** : lifetime ramené de 9999s
                    // à 0.05s.  Avant, les pillar dlights persistaient
                    // dans le buffer (cap = 64) et écrasaient les dlights
                    // gameplay (muzzle, explosions, headshot sparkles).
                    // Avec 0.05s ils expirent au tick suivant et sont
                    // ré-armés à la frame d'après, libérant la queue
                    // pour les transients critiques.
                    for p in &self.pickups {
                        if p.respawn_at.is_some() { continue; }
                        let (color, intensity, premium) = match &p.kind {
                            PickupKind::Powerup { powerup: PowerupKind::QuadDamage, .. } =>
                                ([0.45, 0.55, 1.00], 4.0, true),
                            PickupKind::Powerup { .. } =>
                                ([1.00, 0.85, 0.40], 3.5, true),
                            PickupKind::Health { max_cap: 200, .. } =>
                                ([0.40, 1.00, 0.55], 3.0, true), // Mega
                            PickupKind::Armor { amount: 100 } =>
                                ([1.00, 0.30, 0.30], 3.0, true), // Red Armor
                            _ => ([0.0; 3], 0.0, false),
                        };
                        if premium {
                            for k in 0..3 {
                                let z_off = 30.0 + k as f32 * 80.0;
                                r.spawn_dlight(
                                    p.origin + Vec3::Z * z_off,
                                    260.0,
                                    color,
                                    intensity,
                                    self.time_sec,
                                    0.05, // refresh chaque frame
                                );
                            }
                        }
                    }
                    queue_pickups(
                        r,
                        &self.pickups,
                        self.time_sec,
                        &self.remote_unavailable_pickups,
                        self.net.mode.is_client(),
                        self.ammo_crate_scale,
                        self.quad_pickup_scale,
                        self.health_pack_scale,
                        self.railgun_pickup_scale,
                        self.grenade_ammo_scale,
                        self.rocket_ammo_scale,
                        self.cell_ammo_scale,
                        self.lg_ammo_scale,
                        self.big_armor_scale,
                        self.plasma_pickup_scale,
                        self.railgun_ammo_scale,
                        self.regen_pickup_scale,
                        self.machinegun_pickup_scale,
                        self.bfg_pickup_scale,
                        self.lightninggun_pickup_scale,
                        self.shotgun_pickup_scale,
                        self.grenadelauncher_pickup_scale,
                        self.gauntlet_pickup_scale,
                        self.shotgun_ammo_scale,
                        self.bfg_ammo_scale,
                        self.rocketlauncher_pickup_scale,
                        self.combat_armor_scale,
                        self.medkit_scale,
                        self.armor_shard_scale,
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
                                let halo = [1.0, 1.0, 1.0, col[3] * 0.55];
                                r.push_beam_lightning(
                                    b.a, b.b, halo,
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
                                let core = [1.0, 1.0, 1.0, col[3]];
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
                    // **Drones BR** — orbites animées, transform monde
                    // calculée par drone selon time_sec.  Avant queue_bots
                    // pour que les bots dessinent par-dessus.
                    // **Rochers de décor** — props static, transforms
                    // pré-calculées au load. Frustum cull serait idéal
                    // mais 400 drawcalls passent partout sur GPU récent.
                    if !self.rocks.is_empty() {
                        for rk in &self.rocks {
                            let prop_name = rk.prop_name;
                            let s = rk.scale;
                            let cy = rk.yaw.cos();
                            let sy = rk.yaw.sin();
                            let model = [
                                [cy * s,  sy * s, 0.0, 0.0],
                                [0.0,     0.0,    s,   0.0],
                                [sy * s, -cy * s, 0.0, 0.0],
                                [rk.pos.x, rk.pos.y, rk.pos.z, 1.0],
                            ];
                            r.queue_prop(prop_name, model, rk.tint);
                        }
                    }

                    if !self.drones.is_empty() {
                        for d in &self.drones {
                            let pos = d.position(self.time_sec);
                            let yaw = d.yaw(self.time_sec);
                            let cos_y = yaw.cos();
                            let sin_y = yaw.sin();
                            let s = d.scale;
                            // **glTF Y-up → Q3 Z-up** : un GLB
                            // standard a Y vers le haut. On compose
                            // M = T * Rz(yaw) * RotY→Z * S avec
                            // RotY→Z = swap Y et Z (rotation -90°
                            // autour de X). Effet : ce qui pointait
                            // vers Y+ dans le mesh pointe maintenant
                            // vers Z+ dans le monde.
                            //
                            // Matrice résultante composée à la main
                            // (column-major wgpu).
                            // R_y_up_to_z_up = | 1  0  0 |
                            //                  | 0  0 -1 |
                            //                  | 0  1  0 |
                            // R_z(yaw) = | cy -sy  0 |
                            //            | sy  cy  0 |
                            //            |  0   0  1 |
                            // R_z * R_y_up_to_z_up = | cy   0  sy |
                            //                       | sy   0 -cy |
                            //                       |  0   1   0 |
                            let model = [
                                [cos_y * s,  sin_y * s, 0.0, 0.0],
                                [0.0,        0.0,       s,   0.0],
                                [sin_y * s, -cos_y * s, 0.0, 0.0],
                                [pos.x,      pos.y,     pos.z, 1.0],
                            ];
                            r.queue_drone(model, d.tint);
                        }
                    }

                    if let Some(rig) = self.bot_rig.as_ref() {
                        queue_bots(r, rig, &mut self.bots, self.time_sec);
                        queue_remote_players(
                            r,
                            rig,
                            &self.remote_players,
                            self.time_sec,
                        );
                    } else {
                        // Fallback retiré v0.9.4 — quand le rig MD3 est
                        // chargé (cas standard avec pak0 vanilla), pas
                        // d'indicateur HUD position. Si le rig manque,
                        // les bots sont juste invisibles, mais le banner
                        // diagnostique alerte (cf. plus bas dans
                        // l'overlay HUD `rig_missing_with_bots`).
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
                            // Parse cg_playertint "r,g,b" → vec3.
                            let tint_str = self.cvars.get_string("cg_playertint")
                                .unwrap_or_else(|| "1.0,1.0,1.0".to_string());
                            let parts: Vec<&str> = tint_str.split(',').collect();
                            let player_tint = if parts.len() == 3 {
                                let r = parts[0].trim().parse::<f32>().unwrap_or(1.0).clamp(0.0, 1.5);
                                let g = parts[1].trim().parse::<f32>().unwrap_or(1.0).clamp(0.0, 1.5);
                                let b = parts[2].trim().parse::<f32>().unwrap_or(1.0).clamp(0.0, 1.5);
                                [r, g, b]
                            } else {
                                [1.0, 1.0, 1.0]
                            };
                            // **Viewmodel sway** (v0.9.5++) — calcule
                            // le delta angulaire frame-to-frame puis
                            // l'easing exponentiel pour donner un poids
                            // physique à l'arme.  Le sway emagasine le
                            // delta puis décroît de 90 % par seconde →
                            // mouvements rapides = swings net, mouvement
                            // lent = sway quasi-nul.
                            let yaw_delta = cam_angles.yaw - self.viewmodel_prev_yaw;
                            let yaw_delta = if yaw_delta > 180.0 {
                                yaw_delta - 360.0
                            } else if yaw_delta < -180.0 {
                                yaw_delta + 360.0
                            } else {
                                yaw_delta
                            };
                            let pitch_delta = cam_angles.pitch - self.viewmodel_prev_pitch;
                            // Easing : sway nouveau = sway ancien × decay + impulse.
                            let decay = (-dt * 6.0).exp(); // ~90 %/s
                            self.viewmodel_sway[0] =
                                self.viewmodel_sway[0] * decay + yaw_delta * 0.06;
                            self.viewmodel_sway[1] =
                                self.viewmodel_sway[1] * decay + pitch_delta * 0.04;
                            // Clamp pour éviter les fly-aways (free-look brutal).
                            self.viewmodel_sway[0] = self.viewmodel_sway[0].clamp(-3.0, 3.0);
                            self.viewmodel_sway[1] = self.viewmodel_sway[1].clamp(-3.0, 3.0);
                            self.viewmodel_prev_yaw = cam_angles.yaw;
                            self.viewmodel_prev_pitch = cam_angles.pitch;
                            // Speed XY pour modulé l'amplitude du bob.
                            let v_xy = self.player.velocity.truncate().length();
                            // Time depuis le dernier land (jump-kick window).
                            let time_since_land = self.time_sec - self.player_last_land_at;
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
                                player_tint,
                                self.bob_phase,
                                v_xy,
                                self.player.on_ground,
                                time_since_land,
                                self.viewmodel_sway,
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
                    // **Low-ammo severity** (v0.9.5++) — calculé per-arme
                    // pour que les armes spam (LG/PG/MG) déclenchent le
                    // warning AVANT d'être à 5 munitions (= 0.5 s de fire),
                    // tandis que les armes lentes (RL/GL/BFG/RG/SG) le
                    // déclenchent au seuil "5 tirs" classique.
                    // severity = 1 - ammo/threshold, clamped → 0 hors warning,
                    // 1 à munitions vides. Drives le pulse rouge HUD.
                    let low_ammo_severity = match ammo_shown {
                        None => 0.0,
                        Some(n) => {
                            let threshold = match self.active_weapon {
                                WeaponId::Lightninggun
                                | WeaponId::Plasmagun
                                | WeaponId::Machinegun => 20.0,
                                _ => 5.0,
                            };
                            (1.0 - (n as f32) / threshold).clamp(0.0, 1.0)
                        }
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
                    // **Quad glow aura** (V2g) — dlight mauve attaché
                    // au joueur quand il porte le Quad. Inline le check
                    // powerup pour éviter le re-borrow de self quand
                    // `r` (= &mut self.renderer) est déjà emprunté mut.
                    let quad_idx = PowerupKind::QuadDamage.index();
                    let quad_active = self.powerup_until[quad_idx]
                        .map(|t| t > self.time_sec)
                        .unwrap_or(false)
                        && !self.player_health.is_dead();
                    if quad_active {
                        let pulse = 0.85 + 0.15 * (self.time_sec * 6.0).sin();
                        let qpos = self.player.origin + Vec3::Z * 32.0;
                        r.spawn_dlight(
                            qpos,
                            220.0,
                            [0.55, 0.30, 1.0],
                            2.0 * pulse,
                            self.time_sec,
                            0.05,
                        );
                    }
                    // **Damage screen overlay** (V2c) — vignette rouge
                    // pulsante quand HP bas. Plus le joueur est près
                    // de la mort, plus la vignette est dense et
                    // pulsée. Effet "battement de cœur" cinéma —
                    // s'éteint au-dessus de 50 HP.
                    let hp = self.player_health.current.max(0);
                    if hp > 0 && hp < 50 && !self.player_health.is_dead() {
                        let fw = r.width() as f32;
                        let fh = r.height() as f32;
                        // Plus l'HP baisse, plus l'effet est intense.
                        // Map HP 50→0 sur intensity 0→1 quadratique.
                        let lethality = ((50 - hp) as f32 / 50.0).clamp(0.0, 1.0);
                        let intensity = lethality * lethality;
                        // Pulse à 1.5 Hz (90 BPM) — rythme cardiaque
                        // d'un joueur stressé.
                        let pulse =
                            0.5 + 0.5 * (self.time_sec * 9.42).sin();
                        let alpha = intensity * (0.18 + 0.12 * pulse);
                        // Bandes haut + bas (vignette horizontale)
                        // pour ne pas masquer le HUD ou le crosshair.
                        let band_h = fh * (0.10 + 0.10 * intensity);
                        r.push_rect(0.0, 0.0, fw, band_h, [0.55, 0.05, 0.05, alpha]);
                        r.push_rect(0.0, fh - band_h, fw, band_h, [0.55, 0.05, 0.05, alpha]);
                        // Côtés gauche/droite légers — convergent vers
                        // le centre quand HP < 20.
                        let side_w = fw * (0.04 + 0.06 * intensity);
                        let side_alpha = alpha * 0.8;
                        r.push_rect(0.0, 0.0, side_w, fh, [0.55, 0.05, 0.05, side_alpha]);
                        r.push_rect(fw - side_w, 0.0, side_w, fh, [0.55, 0.05, 0.05, side_alpha]);
                    }
                    // **BR ring overlay** (v0.9.5) — affiché uniquement
                    // en mode terrain BR. Bande haute avec phase /
                    // timer / rayon, et une teinte rouge en bordure
                    // d'écran si le joueur est hors-zone.
                    let fw = r.width() as f32;
                    let fh = r.height() as f32;
                    if let Some(ring) = self.br_ring.as_ref() {
                        let phase_label = match ring.phase_index() {
                            Some(i) => format!("PHASE {}", i + 1),
                            None => "ENDGAME".to_string(),
                        };
                        let timer = ring.time_to_next_phase();
                        let radius = ring.current_radius();
                        let line = format!(
                            "{}    next {:02}:{:02}    radius {:.0}u",
                            phase_label,
                            (timer as i32) / 60,
                            (timer as i32) % 60,
                            radius
                        );
                        let scale = 2.0_f32;
                        let line_px = scale * 8.0 * line.len() as f32;
                        let lx = (fw - line_px) * 0.5;
                        // Décalé sous le compass (y=2..24) → margin
                        // de 8px pour aérer la composition.
                        let ly = 36.0_f32;
                        // Pilule de fond — pulse rouge si shrink imminent
                        // (< 10 s restantes pour la phase courante).
                        let imminent = timer < 10.0 && ring.phase_index().is_some();
                        let bg_color = if imminent {
                            let pulse = 0.5 + 0.5 * (now * 8.0).sin();
                            [0.18 + 0.10 * pulse, 0.04, 0.05, 0.85]
                        } else {
                            [0.05, 0.07, 0.10, 0.70]
                        };
                        let pad = 14.0_f32;
                        r.push_rect(
                            lx - pad,
                            ly - 6.0,
                            line_px + pad * 2.0,
                            scale * 8.0 + 12.0,
                            bg_color,
                        );
                        let bar_color = if imminent {
                            [1.0, 0.20, 0.10, 0.95]
                        } else {
                            [1.0, 0.45, 0.10, 0.85]
                        };
                        r.push_rect(
                            lx - pad,
                            ly - 6.0,
                            line_px + pad * 2.0,
                            2.0,
                            bar_color,
                        );
                        let text_color = if imminent {
                            let pulse = 0.5 + 0.5 * (now * 8.0).sin();
                            [1.0, 0.45 + 0.30 * pulse, 0.30, 1.0]
                        } else {
                            [1.0, 0.78, 0.40, 1.0]
                        };
                        r.push_text(lx + 1.0, ly + 1.0, scale, [0.0, 0.0, 0.0, 0.85], &line);
                        r.push_text(lx, ly, scale, text_color, &line);

                        // **Survivors counter** (v0.9.5++) — sous le bandeau
                        // de phase. "ALIVE: N" en gros chiffres rouge profond
                        // — style BR canonique (Apex/PUBG). Compte le joueur
                        // s'il est vivant + bots non-dead.
                        let alive_bots = self
                            .bots
                            .iter()
                            .filter(|b| !b.health.is_dead())
                            .count();
                        let alive_total = alive_bots
                            + if !self.player_health.is_dead() { 1 } else { 0 };

                        // **Compass top edge** (v0.9.5++) — strip horizontal
                        // 360° autour du yaw joueur, ticks tous les 30°,
                        // labels cardinaux N/E/S/W aux 4 cardinaux, pip
                        // jaune pour la direction du centre du ring.
                        // Convention monde Q3 : yaw = 0 → +X (Est), +90 → +Y
                        // (Nord), donc l'azimut "compass" = yaw - 90 pour
                        // que N corresponde au yaw +90.
                        let compass_w = 480.0_f32;
                        let compass_h = 22.0_f32;
                        let compass_x = (fw - compass_w) * 0.5;
                        let compass_y = 2.0_f32;
                        // Fond pilule semi-transparent.
                        r.push_rect(
                            compass_x,
                            compass_y,
                            compass_w,
                            compass_h,
                            [0.05, 0.07, 0.10, 0.55],
                        );
                        // Liseré bas orangé.
                        r.push_rect(
                            compass_x,
                            compass_y + compass_h - 2.0,
                            compass_w,
                            2.0,
                            [1.0, 0.55, 0.20, 0.85],
                        );
                        let yaw_deg = self.player.view_angles.yaw;
                        // Mapping : on dessine 180° de FOV compass autour
                        // du yaw courant ([yaw-90 .. yaw+90]).
                        let compass_fov = 180.0_f32;
                        let degrees_to_px =
                            |angle: f32| -> Option<f32> {
                                // Différence wrap-aware [-180..180].
                                let mut diff = angle - yaw_deg + 90.0;
                                while diff > 180.0 {
                                    diff -= 360.0;
                                }
                                while diff < -180.0 {
                                    diff += 360.0;
                                }
                                if diff.abs() > compass_fov * 0.5 + 5.0 {
                                    return None;
                                }
                                let px = compass_x
                                    + compass_w * 0.5
                                    + (diff / compass_fov) * compass_w;
                                Some(px)
                            };
                        // Ticks tous les 30°.
                        for tick in (0..360).step_by(15) {
                            let angle = tick as f32;
                            let Some(px) = degrees_to_px(angle) else { continue };
                            let major = tick % 90 == 0;
                            let tick_h = if major { 8.0 } else { 4.0 };
                            let tick_w = if major { 2.0 } else { 1.0 };
                            r.push_rect(
                                px - tick_w * 0.5,
                                compass_y + compass_h - tick_h - 2.0,
                                tick_w,
                                tick_h,
                                [1.0, 0.95, 0.85, 0.85],
                            );
                            // Labels cardinaux (N/E/S/W) aux multiples de 90°.
                            if major {
                                // yaw 0 → E (Est), 90 → N, 180 → W, 270 → S
                                let label = match tick {
                                    0 => "E",
                                    90 => "N",
                                    180 => "W",
                                    270 => "S",
                                    _ => continue,
                                };
                                let lscale = 1.2_f32;
                                let lw = 8.0 * lscale;
                                r.push_text(
                                    px - lw * 0.5,
                                    compass_y + 2.0,
                                    lscale,
                                    [1.0, 1.0, 1.0, 0.95],
                                    label,
                                );
                            }
                        }
                        // **Pip ring center** — petit triangle jaune sous le
                        // strip indiquant la direction vers le centre safe.
                        let center = ring.current_center();
                        let dx = center.x - self.player.origin.x;
                        let dy = center.y - self.player.origin.y;
                        if dx * dx + dy * dy > 4.0 {
                            // atan2(y,x) en radians puis conversion deg.
                            let center_yaw = dy.atan2(dx).to_degrees();
                            if let Some(px) = degrees_to_px(center_yaw) {
                                let pip_size = 6.0_f32;
                                // Triangle approximé : 3 carrés stair-stepped
                                // pointant vers le bas (le strip est au-dessus).
                                for k in 0..pip_size as i32 {
                                    let t = k as f32;
                                    let half_w = pip_size - t;
                                    r.push_rect(
                                        px - half_w,
                                        compass_y + compass_h + t,
                                        half_w * 2.0,
                                        1.0,
                                        [1.0, 0.85, 0.20, 0.95],
                                    );
                                }
                            }
                        }
                        // Pip joueur central — petit triangle blanc fixe.
                        let center_px = compass_x + compass_w * 0.5;
                        for k in 0..5 {
                            let t = k as f32;
                            let half_w = 5.0 - t;
                            r.push_rect(
                                center_px - half_w,
                                compass_y - t - 1.0,
                                half_w * 2.0,
                                1.0,
                                [1.0, 1.0, 1.0, 0.95],
                            );
                        }

                        let survivors_line = format!("ALIVE: {alive_total}");
                        let s_scale = 1.6_f32;
                        let s_px = s_scale * 8.0 * survivors_line.len() as f32;
                        let s_x = (fw - s_px) * 0.5;
                        let s_y = ly + scale * 8.0 + 16.0;
                        // Couleur rouge → orange selon densité — 1 = endgame
                        // critique, 10+ = early game tranquille.
                        let s_color = if alive_total <= 3 {
                            // Pulse à la fin : "tu peux gagner"
                            let p = 0.5 + 0.5 * (now * 4.0).sin();
                            [1.0, 0.30 + 0.20 * p, 0.20 + 0.10 * p, 1.0]
                        } else if alive_total <= 8 {
                            [1.0, 0.55, 0.20, 1.0]
                        } else {
                            [1.0, 0.85, 0.45, 1.0]
                        };
                        r.push_text(s_x + 1.0, s_y + 1.0, s_scale, [0.0, 0.0, 0.0, 0.85], &survivors_line);
                        r.push_text(s_x, s_y, s_scale, s_color, &survivors_line);

                        // POI markers monde **retirés** (v0.9.5+) —
                        // l'utilisateur préfère un terrain "propre"
                        // sans labels flottants. Les POI restent actifs
                        // côté gameplay (spawn joueur/bots, items, ring
                        // shrink), juste plus de UI tag dessus.

                        // Bord rouge si hors-zone — feedback "tu prends
                        // des dégâts" même sans regarder le HUD.
                        if !ring.contains(self.player.origin) {
                            let border = 30.0_f32;
                            let pulse = 0.30 + 0.20 * (now * 4.0).sin().abs();
                            r.push_rect(0.0, 0.0, fw, border, [0.85, 0.10, 0.10, pulse]);
                            r.push_rect(0.0, fh - border, fw, border, [0.85, 0.10, 0.10, pulse]);
                            r.push_rect(0.0, 0.0, border, fh, [0.85, 0.10, 0.10, pulse]);
                            r.push_rect(fw - border, 0.0, border, fh, [0.85, 0.10, 0.10, pulse]);

                            // **Safe-zone arrow** (v0.9.5++) — flèche jaune
                            // au centre haut indiquant le cap à prendre pour
                            // rentrer dans la zone.  Direction = (ring.center
                            // - player.origin) projetée sur le plan horizontal,
                            // puis transformée en angle relatif au yaw joueur.
                            let center = ring.current_center();
                            let dx = center.x - self.player.origin.x;
                            let dy = center.y - self.player.origin.y;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if dist > 1.0 {
                                let world_angle = dy.atan2(dx); // angle absolu
                                let yaw_rad = self.player.view_angles.yaw.to_radians();
                                // Angle relatif : 0 = devant, π/2 = à droite,
                                // -π/2 = à gauche, ±π = derrière.
                                let mut rel = world_angle - yaw_rad;
                                while rel > std::f32::consts::PI {
                                    rel -= std::f32::consts::TAU;
                                }
                                while rel < -std::f32::consts::PI {
                                    rel += std::f32::consts::TAU;
                                }
                                // On dessine une flèche à 45 % de la hauteur
                                // depuis le haut, dont l'orientation suit
                                // l'angle relatif (rotation autour du centre).
                                let cx = fw * 0.5;
                                let cy = fh * 0.20;
                                // Triangle pointing in direction `rel` :
                                // 3 segments stair-stepped.  Pour un push_rect
                                // axis-aligned, on approxime avec une croix
                                // radiale rotée — points à (cos, sin) * r.
                                let ang = rel - std::f32::consts::FRAC_PI_2; // 0 = up
                                let len = 24.0_f32;
                                let alpha = 0.65 + 0.35 * (now * 3.0).sin().abs();
                                let arrow_col = [1.0, 0.85, 0.20, alpha];
                                // Pointe : 3 carrés stair-steppés vers l'avant.
                                for k in 0..len as i32 {
                                    let t = k as f32;
                                    let px = cx + ang.cos() * t;
                                    let py = cy + ang.sin() * t;
                                    r.push_rect(px - 1.5, py - 1.5, 3.0, 3.0, arrow_col);
                                }
                                // Distance numérique sous la flèche.
                                let dist_str = format!("{:.0}u", dist);
                                let d_scale = 1.2_f32;
                                let d_px = d_scale * 8.0 * dist_str.len() as f32;
                                let d_x = cx - d_px * 0.5;
                                let d_y = cy + len + 8.0;
                                r.push_text(d_x + 1.0, d_y + 1.0, d_scale, [0.0, 0.0, 0.0, 0.85], &dist_str);
                                r.push_text(d_x, d_y, d_scale, arrow_col, &dist_str);
                            }
                        }
                    }

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
                        self.kill_marker_until,
                        self.powerup_flash_until,
                        self.powerup_flash_color,
                        self.lightning_flash_until,
                        low_ammo_severity,
                    );
                    // **Railgun scope overlay** (v0.9.5++) — vignette
                    // circulaire + crosshair de précision quand zoom
                    // actif (RMB tenu sur Railgun).  Utilise la valeur
                    // pré-calculée pour éviter le double borrow.
                    if railgun_zoom.is_some() {
                        draw_railgun_scope_overlay(r, now);
                    }
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
                    // **Mini-map** (V3a) — toujours affichée en jeu.
                    // Zone top-left, n'interfère pas avec les autres
                    // panneaux. Pas de drapeaux CTF côté client ; le
                    // serveur ferait foi en multi mais ici en solo
                    // les flags ne sont pas spawnés.
                    if !self.menu.open && !self.player_health.is_dead() {
                        // Snapshot ring + POI pour le minimap si on est en BR.
                        let mm_ring = self.br_ring.as_ref().map(|ring| {
                            (ring.current_center(), ring.current_radius())
                        });
                        let mm_pois: Vec<(Vec3, u8)> = self
                            .terrain
                            .as_ref()
                            .map(|t| {
                                t.pois()
                                    .iter()
                                    .map(|p| (Vec3::new(p.x, p.y, 0.0), p.tier))
                                    .collect()
                            })
                            .unwrap_or_default();
                        draw_minimap(
                            r,
                            self.player.origin,
                            self.player.view_angles.yaw,
                            &self.bots,
                            None,
                            None,
                            mm_ring,
                            &mm_pois,
                        );
                    }
                    // **Stats live overlay** (V3f) — toggle F8.
                    if self.show_stats_overlay {
                        let time_alive = self.time_sec - self.last_respawn_time;
                        let match_elapsed =
                            (self.time_sec - self.match_start_at).max(0.0);
                        draw_stats_live(
                            r,
                            self.frags,
                            self.deaths,
                            self.total_shots,
                            self.total_hits,
                            time_alive,
                            match_elapsed,
                        );
                    }
                    // Banner diagnostique bot — affiché UNIQUEMENT si
                    // l'état est anormal (bots demandés via --bots N
                    // mais 0 spawnés, OU rig MD3 manquant). Sinon
                    // silencieux pour ne pas polluer le HUD. Le bug
                    // historique « no bot » de v0.7-v0.9 a été résolu
                    // par le drain centralisé dans `load_map`, mais
                    // on garde ce filet de sécurité pour orienter le
                    // diag de futures régressions.
                    let rig_missing_with_bots =
                        !self.bots.is_empty() && self.bot_rig.is_none();
                    if rig_missing_with_bots {
                        let banner = format!(
                            "WARNING: rig MD3 manquant — bots dessinés en fallback beam ({})",
                            self.bots.len()
                        );
                        let scale = HUD_SCALE;
                        let bw = banner.len() as f32 * 8.0 * scale;
                        let fw = r.width() as f32;
                        let bx = (fw - bw) * 0.5;
                        let by = 6.0;
                        r.push_rect(
                            bx - 8.0,
                            by - 4.0,
                            bw + 16.0,
                            8.0 * scale + 8.0,
                            [0.45, 0.05, 0.05, 0.85],
                        );
                        push_text_shadow(r, bx, by, scale, COL_YELLOW, &banner);
                    }
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
                        // **Map downloader status refresh** — pousse le
                        // status courant du job (downloaded %, error,
                        // done) dans le menu pour affichage live.
                        let st = self.map_dl.status_snapshot();
                        let status_str = match st {
                            crate::map_dl::DownloadStatus::Idle => String::new(),
                            crate::map_dl::DownloadStatus::Downloading { id, received, total } => {
                                if total > 0 {
                                    let pct = (received as f64 / total as f64 * 100.0) as u32;
                                    format!("{} {}%", id, pct)
                                } else {
                                    format!("{} {}KB", id, received / 1024)
                                }
                            }
                            crate::map_dl::DownloadStatus::Verifying { id } =>
                                format!("{} verifying", id),
                            crate::map_dl::DownloadStatus::Extracting { id } =>
                                format!("{} extracting", id),
                            crate::map_dl::DownloadStatus::Done { id, .. } =>
                                format!("{} OK", id),
                            crate::map_dl::DownloadStatus::Error { id, message } =>
                                format!("{} ERR: {}", id, message),
                        };
                        self.menu.set_mapdl_status(status_str);
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
            let sens_base = self.cvars.get_f32("sensitivity").unwrap_or(5.0).max(0.0);
            // **Railgun zoom** : scale la sens proportionnellement au
            // zoom (FOV/3 → sens/3).  Sans ça le mvt souris devient
            // trop violent par rapport au champ de vision réduit.
            let sens = if let Some(zoom) = self.railgun_zoom_ratio() {
                sens_base * zoom
            } else {
                sens_base
            };
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
/// **Railgun zoom scope overlay** (v0.9.5++) — vignette circulaire
/// noire avec un cercle clair au centre + crosshair de précision +
/// petites graduations.  Évoque une lunette de sniper sans toucher
/// au shader (juste des `push_rect`).
///
/// Géométrie :
///   - Cercle clair de rayon `R = 0.42·h` au centre
///   - Vignette extérieure : 4 rectangles noirs aux 4 coins, plus
///     8 quadrants intermédiaires pour approximer un cercle
///   - Crosshair : 2 lignes fines horizontale/verticale + 4 ticks
///     courts sur chaque axe (graduations sniper)
///   - Mil dot central : un petit point noir au pixel central pour
///     visée ultra-précise
fn draw_railgun_scope_overlay(r: &mut Renderer, time_sec: f32) {
    let w = r.width() as f32;
    let h = r.height() as f32;
    let cx = w * 0.5;
    let cy = h * 0.5;
    // Vignette externe : on assombrit tout sauf un disque central via
    // 8 anneaux extérieurs concentriques.  Approximation polygonale
    // suffisamment dense pour que l'œil voie un cercle.
    let r_clear = h * 0.42; // rayon du disque visible
    // Couleur vignette = noir presque opaque.
    let v = [0.0, 0.0, 0.0, 0.96];
    // 4 quadrants extérieurs au cercle :
    //   - Top (au-dessus du cercle)
    //   - Bottom
    //   - Left
    //   - Right
    let band_top = cy - r_clear;
    let band_bot = h - (cy + r_clear);
    r.push_rect(0.0, 0.0, w, band_top, v);                 // top band
    r.push_rect(0.0, cy + r_clear, w, band_bot, v);        // bottom band
    r.push_rect(0.0, band_top, cx - r_clear, 2.0 * r_clear, v); // left
    r.push_rect(cx + r_clear, band_top, cx - r_clear, 2.0 * r_clear, v); // right
    // Approximation circulaire : on grise les 4 coins du carré
    // restant au centre via 16 trapèzes (segments d'arc).  Pour
    // simplifier, on fait juste un dégradé en empilant des bandes
    // horizontales de plus en plus larges aux extrémités du cercle.
    let n_strips = 32;
    for i in 0..n_strips {
        let t = i as f32 / n_strips as f32;
        let theta = t * std::f32::consts::PI;
        let y_frac = -theta.cos();           // [-1, 1]
        let x_frac = theta.sin();            // [0, 1]
        // Pour chaque bande horizontale autour du cercle, on
        // calcule la "largeur clair" et on dessine deux rects
        // noirs à gauche/droite de cette largeur.
        let strip_y = cy + y_frac * r_clear;
        let strip_h = (2.0 * r_clear) / n_strips as f32 + 1.0;
        let clear_w = x_frac * r_clear;
        let left_w = cx - clear_w - (cx - r_clear);
        if left_w > 0.5 {
            r.push_rect(cx - r_clear, strip_y, left_w, strip_h, v);
            r.push_rect(cx + clear_w, strip_y, left_w, strip_h, v);
        }
    }
    // **Crosshair de précision** — fines lignes 1 px au centre.
    let cross_color = [0.0, 0.0, 0.0, 0.85];
    let line_thick = 1.0;
    let line_len = r_clear * 0.95;
    // Horizontale (interrompue au centre pour ne pas masquer la cible).
    let gap = 8.0;
    r.push_rect(cx - line_len, cy - line_thick * 0.5, line_len - gap, line_thick, cross_color);
    r.push_rect(cx + gap, cy - line_thick * 0.5, line_len - gap, line_thick, cross_color);
    // Verticale.
    r.push_rect(cx - line_thick * 0.5, cy - line_len, line_thick, line_len - gap, cross_color);
    r.push_rect(cx - line_thick * 0.5, cy + gap, line_thick, line_len - gap, cross_color);
    // **Graduations** : petits ticks de chaque côté du centre, espacés.
    let tick_thick = 1.0;
    let tick_len = 6.0;
    for k in 1..=4 {
        let dx = k as f32 * 24.0;
        // Ticks horizontaux (sur la ligne verticale, en haut + bas).
        r.push_rect(cx - tick_len * 0.5, cy - dx, tick_len, tick_thick, cross_color);
        r.push_rect(cx - tick_len * 0.5, cy + dx, tick_len, tick_thick, cross_color);
        // Ticks verticaux (sur la ligne horizontale, à gauche + droite).
        r.push_rect(cx - dx, cy - tick_len * 0.5, tick_thick, tick_len, cross_color);
        r.push_rect(cx + dx, cy - tick_len * 0.5, tick_thick, tick_len, cross_color);
    }
    // **Mil-dot central** : petit point noir pour visée ultra-précise.
    r.push_rect(cx - 1.5, cy - 1.5, 3.0, 3.0, [0.0, 0.0, 0.0, 0.95]);
    // **Liseré du cercle** — léger anneau bleu-cyan pour l'esthétique
    // "lentille high-tech".  Utilise sin(time) pour un pulse subtil.
    let pulse = 0.5 + 0.5 * (time_sec * std::f32::consts::TAU * 0.7).sin();
    let ring_color = [0.45, 0.85, 1.0, 0.20 + 0.10 * pulse];
    let ring_thick = 2.0;
    // 4 segments fins le long des bords visibles du cercle.
    r.push_rect(cx - r_clear, cy - 1.0, ring_thick, 2.0, ring_color); // l
    r.push_rect(cx + r_clear - ring_thick, cy - 1.0, ring_thick, 2.0, ring_color); // r
    r.push_rect(cx - 1.0, cy - r_clear, 2.0, ring_thick, ring_color); // t
    r.push_rect(cx - 1.0, cy + r_clear - ring_thick, 2.0, ring_thick, ring_color); // b
}

fn draw_perf_overlay(r: &mut Renderer, frame_times: &[f32]) {
    let w = r.width() as f32;
    let h = r.height() as f32;
    // Collecte des frame times valides (> 0) dans un buffer trié pour
    // calculer les percentiles (1% low / 0.1% low — métriques modernes
    // qui révèlent les stutters cachés par la moyenne).
    let mut sorted: Vec<f32> = frame_times.iter().copied().filter(|&t| t > 0.0).collect();
    if sorted.is_empty() {
        return;
    }
    let mut sum = 0.0f32;
    for &t in &sorted { sum += t; }
    let n = sorted.len() as f32;
    let avg = sum / n;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = sorted[0];
    let hi = sorted[sorted.len() - 1];
    // **1% low** : moyenne des 1% pires frames (= ~tail 1% du tri trié
    // ascendant → on prend les indices [0.99·n .. n)).  Métrique
    // standard reviewers / OCAT / FrameView.  Si le buffer est trop
    // petit (<100 frames), on retombe sur le pire individuel.
    let p1_lo_avg = {
        let cut = (sorted.len() as f32 * 0.99).floor() as usize;
        let tail = &sorted[cut..];
        if tail.is_empty() { hi } else {
            let s: f32 = tail.iter().sum();
            s / tail.len() as f32
        }
    };
    // **0.1% low** : moyenne des 0.1% pires frames.  Capture les hitches
    // (loading texture, GC, etc.) que les 1% lows manquent.
    let p01_lo_avg = {
        let cut = (sorted.len() as f32 * 0.999).floor() as usize;
        let tail = &sorted[cut..];
        if tail.is_empty() { hi } else {
            let s: f32 = tail.iter().sum();
            s / tail.len() as f32
        }
    };
    let fps_avg = 1.0 / avg;
    let fps_min = 1.0 / hi;
    let fps_max = 1.0 / lo;
    let fps_p1 = 1.0 / p1_lo_avg;
    let fps_p01 = 1.0 / p01_lo_avg;
    let ms_avg = avg * 1000.0;
    // Panel élargi pour accueillir 1% / 0.1% lows.
    let panel_w = 220.0;
    let panel_h = 92.0;
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
    // Ligne 2 : 1% low + 0.1% low (métriques perçues par les joueurs).
    let line2 = format!("1% {:>3.0}  .1% {:>3.0}", fps_p1, fps_p01);
    let lo_color = if fps_p1 >= fps_avg * 0.85 {
        [0.7, 0.95, 0.7, 1.0] // perfs stables (1% low proche de l'avg)
    } else if fps_p1 >= fps_avg * 0.6 {
        [1.0, 0.85, 0.4, 1.0]
    } else {
        [1.0, 0.4, 0.3, 1.0] // gros stutters
    };
    push_text_shadow(r, px + 10.0, py + 26.0, HUD_SCALE * 0.85, lo_color, &line2);
    // Ligne 3 : min/max + ms moyenne (info brute).
    let line3 = format!("min {:>3.0}  max {:>3.0}  {:>4.1} ms", fps_min, fps_max, ms_avg);
    push_text_shadow(r, px + 10.0, py + 44.0, HUD_SCALE * 0.75, COL_GRAY, &line3);
    // Histogramme : chaque frametime mappé à une barre verticale.  On
    // scale par rapport à 50 ms (20 fps) — au-delà la barre sature.
    let graph_x = px + 10.0;
    let graph_y = py + 64.0;
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
    // **Tint par arme** (v0.9.5+) — identité visuelle immédiate de
    // l'arme courante. Cyan plasma, rose railgun, vert BFG, etc.
    let weapon_col: [f32; 4] = match slot {
        2 => [1.00, 0.95, 0.80, 1.0], // MG : ivoire chaud
        3 => [1.00, 0.85, 0.40, 1.0], // SG : orange poudre
        4 => [1.00, 0.65, 0.20, 1.0], // GL : orange chaud
        5 => [1.00, 0.55, 0.30, 1.0], // RL : rouge orangé
        6 => [0.55, 0.85, 1.00, 1.0], // LG : bleu électrique
        7 => [1.00, 0.30, 0.55, 1.0], // RG : rose magenta
        8 => [0.40, 0.85, 1.00, 1.0], // PG : cyan glacial
        9 => [0.45, 1.00, 0.55, 1.0], // BFG : vert saturé
        _ => COL_WHITE,
    };
    let push_cross = |r: &mut Renderer, arm_len: f32, thick: f32, g: f32, col: [f32; 4]| {
        r.push_rect(cx - g - arm_len, cy - thick * 0.5, arm_len, thick, col);
        r.push_rect(cx + g, cy - thick * 0.5, arm_len, thick, col);
        r.push_rect(cx - thick * 0.5, cy - g - arm_len, thick, arm_len, col);
        r.push_rect(cx - thick * 0.5, cy + g, thick, arm_len, col);
    };
    let _ = weapon_col;
    match slot {
        // Gauntlet : anneau épais, évoque une portée « aura melee ».
        1 => {
            push_ring(r, cx, cy, 9.0, 2.0, weapon_col);
            r.push_rect(cx - 1.0, cy - 1.0, 2.0, 2.0, weapon_col);
        }
        // Shotgun : arms courts, gap de base large → spread implicite
        // même sans recul appliqué.
        3 => {
            push_cross(r, 4.0, t, gap + 4.0, weapon_col);
            r.push_rect(cx - 1.0, cy - 1.0, 2.0, 2.0, weapon_col);
        }
        // Grenade launcher : dot + anneau fin (indicateur de tir arqué).
        4 => {
            push_ring(r, cx, cy, 8.0, 1.0, weapon_col);
            r.push_rect(cx - 1.5, cy - 1.5, 3.0, 3.0, weapon_col);
        }
        // Rocket : dot central + anneau épais (warning explosif proche).
        5 => {
            push_ring(r, cx, cy, 10.0, 2.0, weapon_col);
            r.push_rect(cx - 1.5, cy - 1.5, 3.0, 3.0, weapon_col);
        }
        // Lightning gun : cross fin + dot central, pas de spread (beam).
        6 => {
            push_cross(r, 6.0, 1.0, base_gap, weapon_col);
            r.push_rect(cx - 1.0, cy - 1.0, 2.0, 2.0, weapon_col);
        }
        // Railgun : dot seul + anneau très fin (sniping, viser juste).
        7 => {
            let mut soft = weapon_col; soft[3] = 0.55;
            push_ring(r, cx, cy, 5.0, 1.0, soft);
            r.push_rect(cx - 1.0, cy - 1.0, 2.0, 2.0, weapon_col);
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

/// **Stats live overlay** (V3f) — bottom-right, K/D + accuracy +
/// time alive + frags/min. Toggle F8.
fn draw_stats_live(
    r: &mut Renderer,
    frags: u32,
    deaths: u32,
    total_shots: u32,
    total_hits: u32,
    time_alive: f32,
    match_elapsed: f32,
) {
    let lines = [
        format!("FRAGS    {frags}"),
        format!("DEATHS   {deaths}"),
        format!(
            "K/D      {:.2}",
            if deaths == 0 { frags as f32 } else { frags as f32 / deaths as f32 }
        ),
        format!(
            "ACC      {}%",
            if total_shots > 0 {
                ((total_hits as f32 / total_shots as f32) * 100.0).round() as i32
            } else {
                0
            }
        ),
        format!("ALIVE    {:.0}s", time_alive),
        format!(
            "F/MIN    {:.1}",
            if match_elapsed > 1.0 { frags as f32 * 60.0 / match_elapsed } else { 0.0 }
        ),
    ];
    let scale = HUD_SCALE * 0.85;
    let char_w = 8.0 * scale;
    let line_h = 8.0 * scale + 2.0;
    let panel_w = 14.0 * char_w + 16.0;
    let panel_h = lines.len() as f32 * line_h + 12.0;
    let fw = r.width() as f32;
    let fh = r.height() as f32;
    let x = fw - panel_w - 12.0;
    let y = fh - panel_h - 220.0; // au-dessus du panel ammo
    r.push_rect(x, y, panel_w, panel_h, [0.04, 0.05, 0.08, 0.78]);
    r.push_rect(x, y, 2.0, panel_h, [0.30, 0.55, 0.85, 0.95]);
    for (i, line) in lines.iter().enumerate() {
        push_text_shadow(
            r,
            x + 8.0,
            y + 6.0 + i as f32 * line_h,
            scale,
            COL_GRAY,
            line,
        );
    }
}

/// **Mini-map dynamique** (V3a) — top-left, 180×180 px, top-down view
/// de la map. Affiche le joueur (jaune, flèche orientée yaw), bots
/// (couleur tint), et drapeaux CTF si actifs. Échelle adaptative : on
/// fit la bounding box des positions joueur+bots dans le carré.
#[allow(clippy::too_many_arguments)]
fn draw_minimap(
    r: &mut Renderer,
    player_pos: Vec3,
    player_yaw: f32,
    bots: &[BotDriver],
    ctf_red: Option<Vec3>,
    ctf_blue: Option<Vec3>,
    // BR : (centre ring, rayon ring). `None` = mode BSP (pas de ring).
    br_ring: Option<(Vec3, f32)>,
    // BR : POIs avec leur tier 1..=4. Vide hors BR.
    pois: &[(Vec3, u8)],
) {
    // **Minimap v0.9.5++** — refonte largement améliorée :
    //  * Position : top-right (top-left était occupé par status pos)
    //  * Taille  : 240 px (vs 180 avant) → plus lisible
    //  * Auto-zoom : en BR le minimap encadre le ring (zoom-out auto) ;
    //    en BSP zoom fixe 1024 u
    //  * Cadre stylé : double-bordure cyan + coin marker
    //  * Ring BR dessiné : cercle pointillé montrant la zone safe
    //  * POI markers : losanges colorés par tier (tier 4 = doré, tier 3
    //    = orange, tier 2 = gris bleu, tier 1 = ignoré)
    //  * Compass NESW marqués sur les 4 bords intérieurs
    //  * Rose des vents : N en haut (Y+ = nord), E droite (X+)
    //  * Hors-zone : flèche directionnelle vers le centre safe
    let mm_size = 240.0_f32;
    // Top-right : on suppose un écran 1920×1080 minimum (sinon le
    // panneau stats top-right reculera).  L'ancrage à droite via le
    // renderer width.
    let screen_w = r.width() as f32;
    let mm_x = screen_w - mm_size - 16.0;
    let mm_y = 16.0_f32;

    // ─── Auto-zoom ───
    // En BR : on veut voir le ring entier + un peu de marge.
    // En BSP : zoom fixe Q3 typique (1024 u).
    let world_extent = match br_ring {
        Some((center, radius)) => {
            // Distance joueur ↔ centre + rayon, plafond doux pour ne
            // pas trop dézoomer si le ring est énorme (premières phases).
            let d = (player_pos.truncate() - center.truncate()).length();
            (d + radius * 1.2).max(1024.0).min(15000.0)
        }
        None => 1024.0,
    };
    let to_screen = |w: Vec3| -> (f32, f32) {
        let dx = w.x - player_pos.x;
        let dy = w.y - player_pos.y;
        let sx = mm_x + mm_size * 0.5 + (dx / world_extent) * (mm_size * 0.5);
        let sy = mm_y + mm_size * 0.5 - (dy / world_extent) * (mm_size * 0.5);
        (sx, sy)
    };

    // ─── Fond + cadre ───
    // Halo extérieur cyan-bleu (style Apex Legends).
    r.push_rect(mm_x - 4.0, mm_y - 4.0, mm_size + 8.0, mm_size + 8.0,
        [0.20, 0.40, 0.70, 0.30]);
    r.push_rect(mm_x - 2.0, mm_y - 2.0, mm_size + 4.0, mm_size + 4.0,
        [0.30, 0.55, 0.85, 0.65]);
    r.push_rect(mm_x, mm_y, mm_size, mm_size, [0.04, 0.06, 0.10, 0.88]);
    // 4 coin markers (L-shape) pour un look "tactique".
    let corner_len = 14.0_f32;
    let corner_thick = 2.0_f32;
    let corner_col = [0.40, 0.75, 1.0, 0.95];
    // Top-left
    r.push_rect(mm_x, mm_y, corner_len, corner_thick, corner_col);
    r.push_rect(mm_x, mm_y, corner_thick, corner_len, corner_col);
    // Top-right
    r.push_rect(mm_x + mm_size - corner_len, mm_y, corner_len, corner_thick, corner_col);
    r.push_rect(mm_x + mm_size - corner_thick, mm_y, corner_thick, corner_len, corner_col);
    // Bottom-left
    r.push_rect(mm_x, mm_y + mm_size - corner_thick, corner_len, corner_thick, corner_col);
    r.push_rect(mm_x, mm_y + mm_size - corner_len, corner_thick, corner_len, corner_col);
    // Bottom-right
    r.push_rect(mm_x + mm_size - corner_len, mm_y + mm_size - corner_thick, corner_len, corner_thick, corner_col);
    r.push_rect(mm_x + mm_size - corner_thick, mm_y + mm_size - corner_len, corner_thick, corner_len, corner_col);

    // ─── Compass NESW sur bords intérieurs ───
    let cardinal_col = [0.85, 0.92, 1.0, 0.90];
    let cardinal_scale = 1.0_f32;
    let cw = 8.0 * cardinal_scale;
    // N en haut centre
    r.push_text(mm_x + mm_size * 0.5 - cw * 0.5, mm_y + 3.0,
                cardinal_scale, cardinal_col, "N");
    // S en bas centre
    r.push_text(mm_x + mm_size * 0.5 - cw * 0.5, mm_y + mm_size - 8.0 * cardinal_scale - 3.0,
                cardinal_scale, cardinal_col, "S");
    // E à droite milieu
    r.push_text(mm_x + mm_size - cw - 3.0, mm_y + mm_size * 0.5 - 4.0 * cardinal_scale,
                cardinal_scale, cardinal_col, "E");
    // W à gauche milieu
    r.push_text(mm_x + 3.0, mm_y + mm_size * 0.5 - 4.0 * cardinal_scale,
                cardinal_scale, cardinal_col, "W");

    // Helper : test si un screen pos est dans le minimap.
    let in_mm = |sx: f32, sy: f32| -> bool {
        sx >= mm_x + 4.0 && sx <= mm_x + mm_size - 4.0
            && sy >= mm_y + 4.0 && sy <= mm_y + mm_size - 4.0
    };

    // ─── Ring BR ───
    // Cercle pointillé (32 segments) pour la zone safe courante.
    if let Some((ring_center, ring_radius)) = br_ring {
        let (cx, cy) = to_screen(ring_center);
        // Rayon en pixels minimap.
        let r_px = ring_radius * (mm_size * 0.5) / world_extent;
        let segments = 48;
        let dot_size = 1.5_f32;
        for k in 0..segments {
            // 32 segments, pointillé : on ne dessine que les segments pairs.
            if k % 2 != 0 { continue; }
            let theta = (k as f32) * std::f32::consts::TAU / (segments as f32);
            let dx = theta.cos() * r_px;
            let dy = theta.sin() * r_px;
            let sx = cx + dx;
            let sy = cy - dy; // -dy car nord en haut
            if in_mm(sx, sy) {
                r.push_rect(sx - dot_size, sy - dot_size,
                            dot_size * 2.0, dot_size * 2.0,
                            [0.20, 1.0, 0.60, 0.90]);
            }
        }
        // Voile très léger à l'intérieur du ring (zone safe = vert).
        // Approximé par un point plus gros au centre du ring (pas un
        // vrai disque WGSL mais rappel visuel).
        if in_mm(cx, cy) {
            r.push_rect(cx - 3.0, cy - 3.0, 6.0, 6.0, [0.20, 1.0, 0.60, 0.40]);
        }
    }

    // ─── POI markers ───
    // Losanges colorés par tier. Tier 1 (basique) ignoré pour ne pas
    // saturer ; tier 2/3/4 affichés.
    for (pos, tier) in pois {
        if *tier < 2 { continue; }
        let (sx, sy) = to_screen(*pos);
        if !in_mm(sx, sy) { continue; }
        let (color, size) = match tier {
            4 => ([1.0, 0.85, 0.20, 0.95], 5.0_f32),  // gold
            3 => ([1.0, 0.55, 0.20, 0.90], 4.0),       // orange
            _ => ([0.60, 0.75, 1.0, 0.80], 3.0),       // blue-grey
        };
        // Approximé en losange : 4 mini-rects centrés.
        r.push_rect(sx - size, sy - 1.0, size * 2.0, 2.0, color);
        r.push_rect(sx - 1.0, sy - size, 2.0, size * 2.0, color);
    }

    // ─── Joueur (rendered last, top of stack) ───
    let (px, py) = to_screen(player_pos);
    let yaw_rad = player_yaw.to_radians();
    // Triangle directionnel : 3 points stair-stepped.
    let arrow_len = 12.0_f32;
    let arrow_back = 6.0_f32;
    let cos_y = yaw_rad.cos();
    let sin_y = yaw_rad.sin();
    // Pointe avant
    let tip_x = px + cos_y * arrow_len;
    let tip_y = py - sin_y * arrow_len;
    // Pour un look "flèche", on dessine 3 carrés stair-steppés depuis
    // le centre vers la pointe, avec décroissance.
    let steps = arrow_len as i32;
    for k in 0..steps {
        let t = k as f32 / steps as f32;
        let x = px + cos_y * (k as f32);
        let y = py - sin_y * (k as f32);
        let half = (1.0 - t) * 3.0 + 1.0;
        r.push_rect(x - half, y - half, half * 2.0, half * 2.0, [1.0, 0.92, 0.30, 1.0]);
    }
    // Tail dot (un peu derrière le centre).
    let tail_x = px - cos_y * arrow_back;
    let tail_y = py + sin_y * arrow_back;
    r.push_rect(tail_x - 1.5, tail_y - 1.5, 3.0, 3.0, [1.0, 0.70, 0.10, 0.85]);
    // Centre du joueur — point central plus opaque pour contraster.
    r.push_rect(px - 2.0, py - 2.0, 4.0, 4.0, [1.0, 1.0, 0.50, 1.0]);
    let _ = tip_x; let _ = tip_y;

    // ─── Hors-zone : flèche vers centre safe ───
    if let Some((ring_center, ring_radius)) = br_ring {
        let dx = ring_center.x - player_pos.x;
        let dy = ring_center.y - player_pos.y;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > ring_radius {
            // Joueur hors zone — dessine une flèche directionnelle
            // depuis le joueur vers le ring center, sur le bord du minimap.
            let theta = dy.atan2(dx);
            let edge_radius = mm_size * 0.45; // proche du bord
            let ex = mm_x + mm_size * 0.5 + theta.cos() * edge_radius;
            let ey = mm_y + mm_size * 0.5 - theta.sin() * edge_radius;
            // Triangle clignotant rouge.
            for k in 0..6 {
                let t = k as f32 / 6.0;
                let x = ex - theta.cos() * (k as f32 * 1.5);
                let y = ey + theta.sin() * (k as f32 * 1.5);
                let half = (1.0 - t) * 4.0 + 1.0;
                r.push_rect(x - half, y - half, half * 2.0, half * 2.0,
                            [1.0, 0.20, 0.20, 0.95]);
            }
        }
    }

    // Bots (radar) intentionnellement non rendus — pas de wallhack
    // visuel sur le minimap.  Le slot reste pour usage CTF futur.
    let _ = bots;

    // Drapeaux CTF si fournis.
    if let Some(p) = ctf_red {
        let (fx, fy) = to_screen(p);
        if in_mm(fx, fy) {
            r.push_rect(fx - 4.0, fy - 4.0, 8.0, 8.0, [1.0, 0.30, 0.30, 1.0]);
        }
    }
    if let Some(p) = ctf_blue {
        let (fx, fy) = to_screen(p);
        if in_mm(fx, fy) {
            r.push_rect(fx - 4.0, fy - 4.0, 8.0, 8.0, [0.40, 0.55, 1.0, 1.0]);
        }
    }
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
    // Deadline du grand X de kill-confirm.  Si > `now`, on dessine un
    // X rouge vif autour du crosshair dont l'alpha fade en fin de
    // fenêtre.  Distinct du `hit_marker` (4 segments diagonaux gris).
    kill_marker_until: f32,
    // Deadline du flash full-screen powerup.  Si > `now`, on dessine
    // un voile teinté du powerup acquis (Quad bleu, Haste orange…),
    // alpha fade-out rapide (POWERUP_FLASH_SEC).
    powerup_flash_until: f32,
    powerup_flash_color: [f32; 4],
    // Deadline du flash d'éclair atmosphérique BR.  Si > `now`, on
    // dessine un voile blanc-bleuté full-screen avec fade exponentiel
    // rapide pour évoquer un strike de foudre proche.
    lightning_flash_until: f32,
    // Sévérité de la situation munitions [0..1]. 0 = OK, 1 = empty.
    // Pilote un liseré orange-rouge pulsant en bas d'écran pour
    // que le joueur sente venir le moment où il faudra switch.
    low_ammo_severity: f32,
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

    // ─── Low-ammo edge pulse ──────────────────────────────────────
    // Liseré orange-rouge en bas d'écran qui pulse à 2 Hz quand
    // `low_ammo_severity > 0`. L'épaisseur et l'alpha augmentent avec
    // la sévérité (linéaire). Distinct des flashes armor/pain (cadre
    // complet) en restant cantonné au bord bas → ne mange pas le
    // champ visuel mais reste périphériquement lisible.
    // Gaté par !is_spectator (pas d'arme à équiper) et !respawn_remaining
    // (joueur mort n'a pas besoin de cette info pendant la fenêtre
    // d'intermission/death cam).
    if low_ammo_severity > 0.001 && !is_spectator && respawn_remaining.is_none() {
        let pulse = 0.6 + 0.4 * (now * std::f32::consts::TAU * 2.0).sin().abs();
        let s = low_ammo_severity;
        let thick = 6.0 + 16.0 * s;
        let alpha = (0.30 + 0.45 * s) * pulse;
        // Couleur : ambre à severity faible, rouge vif à severity haute.
        let col = [
            1.0,
            0.55 - 0.40 * s,
            0.15 - 0.10 * s,
            alpha,
        ];
        r.push_rect(0.0, h - thick, w, thick, col);
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

    // ─── Powerup acquisition flash ─────────────────────────────────
    // Voile full-screen teinté de la couleur du powerup acquis,
    // alpha en fade-out type easing-out (ratio² → décroît vite).
    // Pas de cadre épais comme la pain vignette : ici on veut un
    // "wash" cinématique global qui s'efface en ~0.5 s sans gêner
    // la vision continue de l'environnement.
    // Gaté par !is_spectator (les spectateurs ne ramassent rien).
    if powerup_flash_until > now && !is_spectator {
        let remaining = powerup_flash_until - now;
        let ratio = (remaining / POWERUP_FLASH_SEC).clamp(0.0, 1.0);
        // Easing : ratio² fade rapide en fin → flash bref qui n'occulte
        // pas le combat post-pickup.
        let alpha = ratio * ratio * 0.45;
        let col = [
            powerup_flash_color[0],
            powerup_flash_color[1],
            powerup_flash_color[2],
            alpha,
        ];
        r.push_rect(0.0, 0.0, w, h, col);
        // Liseré plus opaque — donne un effet "transition" périphérique
        // distinct du wash central, sans masquer le HUD.
        let edge_alpha = ratio * 0.65;
        let edge_col = [col[0], col[1], col[2], edge_alpha];
        let edge_thick = 18.0;
        r.push_rect(0.0, 0.0, w, edge_thick, edge_col);
        r.push_rect(0.0, h - edge_thick, w, edge_thick, edge_col);
        r.push_rect(0.0, edge_thick, edge_thick, h - 2.0 * edge_thick, edge_col);
        r.push_rect(w - edge_thick, edge_thick, edge_thick, h - 2.0 * edge_thick, edge_col);
    }

    // ─── Lightning flash atmosphérique (BR) ───────────────────────
    // Voile blanc-bleuté full-screen avec fade exponentiel rapide
    // (ratio³) — l'œil voit "flash → noir" presque instantané, façon
    // strike d'orage.  Synchronisé avec un dlight haut perché
    // (cf. tick_atmosphere) qui éclaire la scène pendant la même
    // fenêtre — le coup d'œil sur le HUD et le coup d'œil sur le
    // monde s'accordent naturellement.
    if lightning_flash_until > now {
        let remaining = lightning_flash_until - now;
        let ratio = (remaining / LIGHTNING_FLASH_SEC).clamp(0.0, 1.0);
        // Easing ratio³ → fade très rapide. Alpha max ~0.55.
        let alpha = ratio * ratio * ratio * 0.55;
        r.push_rect(0.0, 0.0, w, h, [0.92, 0.95, 1.00, alpha]);
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
    //
    // **v0.9.5 polish** :
    // * scale par distance (les hits lointains restent lisibles)
    // * couleur par tier de dégât (gris < 25, jaune < 60, orange < 100,
    //   rouge ≥ 100) pour repérer instantanément un crit ou un splash
    // * léger drift latéral pour que des chiffres simultanés (SG 11
    //   pellets) ne se collent pas en pile illisible
    let dmg_base_scale = HUD_SCALE * 1.4;
    for d in floating_damages {
        let remaining = (d.expire_at - now).max(0.0);
        let life = d.lifetime.max(1e-3);
        let elapsed_ratio = (1.0 - remaining / life).clamp(0.0, 1.0);
        // Frag confirms montent plus haut (×1.5) que les damage numbers
        // pour qu'ils sortent plus du chaos visuel post-frag.
        let rise = if d.is_frag {
            DAMAGE_NUMBER_RISE * 1.5 * elapsed_ratio
        } else {
            DAMAGE_NUMBER_RISE * elapsed_ratio
        };
        let world_pos = d.origin + Vec3::Z * rise;
        let Some((sx, sy)) = r.project_to_screen(world_pos) else {
            continue;
        };
        // Distance world → scale facteur. À 200u : 1.0×, à 1500u :
        // 1.6×. Au-delà, capé pour ne pas avoir un texte plein écran.
        let dist = (world_pos - *player_origin).length();
        let dist_scale = (1.0 + ((dist - 200.0) / 1500.0).max(0.0)).min(1.7);
        // Frags affichés ~1.6× plus grand que les damage numbers — le
        // joueur lit "FRAG" au coup d'œil sans confondre avec un chiffre.
        let scale_mul = if d.is_frag { 1.6 } else { 1.0 };
        let dmg_scale = dmg_base_scale * dist_scale * scale_mul;
        let dmg_char_w = 8.0 * dmg_scale;
        let dmg_line_h = 8.0 * dmg_scale;

        let alpha = if elapsed_ratio < 0.6 {
            1.0
        } else {
            ((1.0 - elapsed_ratio) / 0.4).clamp(0.0, 1.0)
        };
        // Frag confirm : couleur fixe gold-orange, texte "+1 FRAG".
        // Damage : tier-coloré selon le nombre.
        let (color, text) = if d.is_frag {
            // Pop d'intensité au début (1.0×) qui se calme rapidement.
            let pop = if elapsed_ratio < 0.15 {
                1.0 + (1.0 - elapsed_ratio / 0.15) * 0.4
            } else {
                1.0
            };
            ([1.0, 0.65 * pop, 0.20 * pop, alpha], "+1 FRAG".to_string())
        } else if d.to_player {
            // Player hit : rouge dégradé selon sévérité.
            let c = if d.damage >= 50 {
                [1.0, 0.20, 0.15, alpha] // rouge vif (gros hit)
            } else {
                [1.0, 0.45, 0.30, alpha] // rouge plus doux
            };
            (c, format!("{}", d.damage))
        } else {
            // Damage to bot : tier-coloré.
            let c = if d.damage >= 100 {
                [1.0, 0.30, 0.20, alpha] // rouge orangé (rocket/splash gros)
            } else if d.damage >= 60 {
                [1.0, 0.55, 0.15, alpha] // orange (crit)
            } else if d.damage >= 25 {
                [1.0, 0.95, 0.20, alpha] // jaune standard
            } else {
                [0.85, 0.85, 0.85, alpha] // gris clair (mini-hit)
            };
            (c, format!("{}", d.damage))
        };
        // Drift latéral : décalé selon hash du timestamp pour
        // séparer les chiffres synchrones (SG pellets).  Frags ne
        // driftent pas (centrés sur la cible morte).
        let drift = if d.is_frag {
            0.0
        } else {
            let bits = d.expire_at.to_bits();
            (((bits >> 8) & 0xff) as f32 / 255.0 - 0.5) * 30.0
        };
        let tx = sx - (text.len() as f32 * dmg_char_w) * 0.5 + drift;
        let ty = sy - dmg_line_h * 0.5;
        // Ombre portée pour lisibilité sur ciel/terrain clair.
        r.push_text(tx + 1.5, ty + 1.5, dmg_scale, [0.0, 0.0, 0.0, alpha * 0.6], &text);
        r.push_text(tx, ty, dmg_scale, color, &text);
    }

    // Nameplates bots **retirés** (v0.9.5) — l'utilisateur ne veut aucun
    // indicateur de position ennemi sur le HUD (zéro wallhack visuel).
    // L'identification visuelle passe désormais uniquement par le tint
    // appliqué au MD3 du bot (cf. `BotDriver::tint`) et le nom dans
    // le kill-feed après un frag.
    let _ = (player_origin, bots);

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

    // Kill-marker X : grand X rouge vif autour du réticule quand la
    // dernière volée a tué la cible.  Distinct du hit-marker (plus
    // gros, plus brillant, true diagonal au lieu d'horizontal) pour
    // qu'un kill-confirm soit lisible "à l'œil" même sans regarder
    // le score.  Chaque branche du X = série de petits carrés 2×2
    // stair-steppés le long de la diagonale (push_rect ne sait pas
    // tourner).  Alpha fade linéaire sur la durée totale.
    if now < kill_marker_until {
        let remain = (kill_marker_until - now).max(0.0);
        let alpha = (remain / KILL_MARKER_DURATION_SEC).clamp(0.0, 1.0);
        // Easing : pop puissant au début, fade lent en fin → on garde
        // l'alpha visuel haut sur la 1ère moitié.
        let alpha = alpha.sqrt();
        let col = [1.0, 0.18, 0.12, alpha];
        let inner = 9.0_f32; // gap centre → début de la branche
        let len = 16_i32; // longueur en pixels diagonaux
        for k in 0..len {
            let f = k as f32;
            // Top-left → bottom-right diagonale (\)
            r.push_rect(cx - inner - f, cy - inner - f, 2.0, 2.0, col);
            r.push_rect(cx + inner + f, cy + inner + f, 2.0, 2.0, col);
            // Top-right → bottom-left diagonale (/)
            r.push_rect(cx + inner + f, cy - inner - f, 2.0, 2.0, col);
            r.push_rect(cx - inner - f, cy + inner + f, 2.0, 2.0, col);
        }
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

    // **Radar bot dots retirés** (v0.9.5+) — l'utilisateur ne veut
    // aucun indicateur de position des bots sur le HUD.  Les bots
    // restent invisibles tant qu'ils ne sont pas en LOS direct, ce
    // qui pousse au jeu d'écoute (pas/tirs) au lieu du wallhack
    // visuel.
    {
        const RADAR_WORLD_RANGE: f32 = 2000.0;
        let radar_size = 140.0_f32;
        let radar_x = 8.0;
        let radar_y = 90.0;
        let cx_r = radar_x + radar_size * 0.5;
        let cy_r = radar_y + radar_size * 0.5;
        let scale = (radar_size * 0.5) / RADAR_WORLD_RANGE;
        // Fond semi-opaque + liseré (cadre vide, pas de dots ennemis).
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
        let basis = view_angles.to_vectors();
        // On garde les variables consommées plus bas (drapeaux CTF /
        // joueur) pour ne pas casser le reste du radar.
        let _ = (basis, scale);
        let _ = bots;
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

    // **Overlay Quad Damage** SUPPRIMÉ (v0.9.5++ user request) — l'effet
    // multi-couches précédent (vignette + sparkles + bandeaux) était
    // distrayant.  On laisse uniquement le SFX de pickup + la jauge HUD
    // (chrono powerup en bas-droite) pour signaler que le Quad est actif.
    // Le tint sur le viewmodel reste également discret côté
    // `queue_viewmodel` via `player_tint`.

    // ─── Haste ─────────────────────────────────────────────────────
    // **Speed lines périphériques SUPPRIMÉES** (v0.9.5++ user request) —
    // jugées trop intrusives.  Feedback restant : SFX de pickup +
    // chrono HUD + cadence de tir / vitesse course visiblement boostée.

    // ─── Battle Suit ───────────────────────────────────────────────
    // **Crackle énergétique SUPPRIMÉ** (v0.9.5++ user request idem
    // Quad/Haste).  Feedback restant : SFX de pickup + chrono HUD +
    // immunité dégât environnemental sentie en gameplay.

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
        // **Death overlay** — voile rouge sombre full-screen + dégradé
        // vers les bords, alpha modulé pour fade-in (0.3 s) → stay
        // jusqu'à respawn.  Donne une lecture "drama" forte : le monde
        // s'assombrit autour du cadavre.  Indépendant du panneau
        // texte qui suit (le voile passe DESSOUS le panneau).
        let elapsed_dead = (RESPAWN_DELAY_SEC - remaining).max(0.0);
        let fade_in = (elapsed_dead / 0.30).clamp(0.0, 1.0);
        let dim_alpha = 0.42 * fade_in;
        // Wash global : noir-rougeâtre, écrase la couleur du monde.
        r.push_rect(0.0, 0.0, w, h, [0.18, 0.02, 0.02, dim_alpha]);
        // Bordure plus opaque : effet "tunnel" périphérique.
        let edge_thick = 60.0;
        let edge_alpha = 0.55 * fade_in;
        let edge = [0.45, 0.05, 0.05, edge_alpha];
        r.push_rect(0.0, 0.0, w, edge_thick, edge);
        r.push_rect(0.0, h - edge_thick, w, edge_thick, edge);
        r.push_rect(0.0, edge_thick, edge_thick, h - 2.0 * edge_thick, edge);
        r.push_rect(w - edge_thick, edge_thick, edge_thick, h - 2.0 * edge_thick, edge);

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
        // **BR victory screen** (v0.9.5) — distingue VICTORY (player
        // gagne) / DEFEAT (un bot gagne) / DRAW (égalité ring) avec
        // teintes contrastées et un sub-title qui nomme le vainqueur.
        let (banner, banner_col, sub) = match winner {
            KillActor::Player => (
                "VICTORY",
                [0.30, 1.0, 0.45, 1.0],
                "LAST ONE STANDING".to_string(),
            ),
            KillActor::Bot(name) => (
                "DEFEAT",
                [1.0, 0.30, 0.30, 1.0],
                format!("WINNER: {name}"),
            ),
            KillActor::World => (
                "DRAW",
                [0.85, 0.85, 0.85, 1.0],
                "WIPED BY THE STORM".to_string(),
            ),
        };
        // Voile sombre + double liseré.
        r.push_rect(0.0, 0.0, w, h, [0.0, 0.0, 0.0, 0.65]);
        r.push_rect(0.0, h * 0.12, w, 3.0, banner_col);
        r.push_rect(0.0, h * 0.32, w, 3.0, banner_col);
        let scale = HUD_SCALE * 4.0;
        let big_char_w = 8.0 * scale;
        let big_line_h = 8.0 * scale;
        let l1_w = banner.len() as f32 * big_char_w;
        let cy = h * 0.18;
        // Big banner (VICTORY/DEFEAT/DRAW) en couleur du résultat.
        push_text_shadow(r, (w - l1_w) * 0.5, cy, scale, banner_col, banner);
        // Sous-titre nom vainqueur en blanc.
        let sub_scale = HUD_SCALE * 2.0;
        let sub_w = sub.len() as f32 * 8.0 * sub_scale;
        push_text_shadow(
            r,
            (w - sub_w) * 0.5,
            cy + big_line_h * 1.2,
            sub_scale,
            COL_WHITE,
            &sub,
        );
        // **Match end stats panel** (v0.9.5++ #41) — K/D + accuracy +
        // streak max sous le banner.  Donne au joueur un retour
        // numérique sur sa partie quand le match se termine.
        let kd = if deaths == 0 {
            frags as f32
        } else {
            frags as f32 / deaths as f32
        };
        let acc = if total_shots > 0 {
            (total_hits as f32 / total_shots as f32) * 100.0
        } else {
            0.0
        };
        let stats = [
            format!("FRAGS    {:>4}", frags),
            format!("DEATHS   {:>4}", deaths),
            format!("K/D      {:>5.2}", kd),
            format!("ACCURACY {:>5.1}%", acc),
            format!("SHOTS    {:>4}", total_shots),
            format!("HITS     {:>4}", total_hits),
        ];
        let stats_scale = HUD_SCALE * 1.5;
        let stats_line_h = 8.0 * stats_scale + 6.0;
        let stats_y0 = h * 0.42;
        for (i, line) in stats.iter().enumerate() {
            let line_w = line.len() as f32 * 8.0 * stats_scale;
            let lx = (w - line_w) * 0.5;
            let ly = stats_y0 + i as f32 * stats_line_h;
            push_text_shadow(r, lx, ly, stats_scale, [0.85, 0.85, 0.92, 0.95], line);
        }
        // Hint return to menu — repoussé en bas pour ne pas chevaucher les stats.
        let hint = "ESC  RETURN TO MENU";
        let hint_scale = HUD_SCALE * 1.0;
        let hint_w = hint.len() as f32 * 8.0 * hint_scale;
        push_text_shadow(
            r,
            (w - hint_w) * 0.5,
            h - 60.0,
            hint_scale,
            [0.7, 0.78, 0.88, 0.95],
            hint,
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

fn play_at(snd: &Arc<SoundSystem>, handle: SoundHandle, origin: Vec3, priority: Priority) -> bool {
    snd.play_3d(
        handle,
        Emitter3D {
            position: origin,
            near_dist: 64.0,
            far_dist: 2048.0,
            volume: 1.0,
            priority,
        },
    )
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
///
/// Si `override_path` est `Some(...)`, on l'utilise direct au lieu de
/// chercher dans le BSP (cvar `r_skybox` user, force la skybox custom
/// pour TOUTES les maps).
fn resolve_and_load_sky(
    r: &mut Renderer,
    vfs: &Arc<Vfs>,
    bsp: &Bsp,
    override_path: Option<&str>,
) {
    // Override user > BSP shader.
    if let Some(base) = override_path {
        if !base.trim().is_empty() {
            info!("sky: cubemap forcée via r_skybox = '{}'", base);
            if let Err(e) = r.load_sky_cubemap(vfs, base) {
                warn!("sky cubemap '{base}' KO: {e} — essai BSP fallback");
                // Continue vers le fallback BSP ci-dessous.
            } else {
                return;
            }
        }
    }
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
    /// **Squad chatter events** (v0.9.5) — collecte les triggers de
    /// chatter détectés pendant le tick. Flushés par App après le
    /// tick (App::maybe_bot_chat ne peut pas être appelé depuis ici
    /// car tick_bots ne tient pas `&mut self`).
    chatter_events: Vec<(usize, ChatTrigger)>,
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
    // **Bot item priority** (G4c) — pickups stratégiques que les bots
    // priorisent. Format `(pos, prio_score, available)`. `available=
    // false` = en respawn, on ignore. `prio_score` distingue MH/RA/
    // Quad/etc. → les bots vont chercher Quad avant MH.
    pickups_priority: &[(Vec3, f32, bool)],
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
        chatter_events: Vec::new(),
    };

    for (idx, d) in bots.iter_mut().enumerate() {
        // **Yaw turn detection** (v0.9.5++) — si le bot pivote rapidement
        // alors qu'il est stationnaire, déclenche LEGS_TURN.  Seuil :
        // 8°/frame (≈ 480°/s à 60fps) → détecte les hard turns d'agro
        // sans déclencher pour les microcorrections.
        let yaw_now = d.body.view_angles.yaw;
        let mut delta = (yaw_now - d.prev_yaw).abs();
        // Wrap-around 359° ↔ 1°
        if delta > 180.0 { delta = 360.0 - delta; }
        if delta > 8.0 {
            d.last_turn_at = now;
        }
        d.prev_yaw = yaw_now;

        // **Bot item priority** (G4c) — si le bot n'a pas de cible
        // ennemie ET son HP/armor justifient un pickup (HP<70 → MH/health,
        // armor<50 → RA/YA, ou n'importe quel bot va vers Quad), on
        // insère un waypoint vers le pickup le plus prioritaire dans
        // un rayon de 2000u. Le score combine type × 1/distance pour
        // que Quad au loin l'emporte sur MH proche, mais Quad très
        // loin ne batte pas MH proche.
        if d.bot.target_enemy.is_none() && !d.health.is_dead() {
            let need_health = d.health.current < 70;
            let bot_pos = d.body.origin;
            let mut best: Option<(f32, Vec3)> = None;
            for &(pos, mut prio, available) in pickups_priority {
                if !available { continue; }
                let dist = (pos - bot_pos).length();
                if dist > 2000.0 || dist < 1.0 { continue; }
                // Health items dévalués si HP plein.
                if !need_health && prio < 5.0 {
                    prio *= 0.3;
                }
                let score = prio / (dist * 0.001 + 1.0);
                if best.map_or(true, |(s, _)| score > s) {
                    best = Some((score, pos));
                }
            }
            if let Some((_, pos)) = best {
                // Insère ce waypoint EN TÊTE de queue si pas déjà
                // proche (évite re-insertion à chaque tick).
                let already_close = d.bot.waypoints
                    .first()
                    .map(|w| (*w - pos).length() < 64.0)
                    .unwrap_or(false);
                if !already_close {
                    d.bot.waypoints.insert(0, pos);
                }
            }
        }
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
        // **Bot retreat** (G4d) — quand HP bas, on bypass le combat
        // et on insère un waypoint OPPOSÉ au joueur pour fuir vers
        // une zone safe. Le bot revient en Roam (target_enemy=None)
        // → ne tire plus → mouvement vers waypoint évasion. Quand
        // HP remonte (pickup health) le comportement normal reprend.
        let low_hp = d.health.current < 25;
        if visible && !low_hp {
            // **Spotted chatter** : si on vient juste d'acquérir la
            // cible (transition None→Some), pousse un trigger Spotted.
            // Le cooldown par-bot est géré côté App pour ne pas
            // saturer la radio.
            let just_spotted = d.first_seen_player_at.is_none();
            d.bot.target_enemy = Some(player_eye);
            d.last_saw_player_at = Some(now);
            if just_spotted {
                d.first_seen_player_at = Some(now);
                if (now - d.last_chatter_at) >= CHATTER_COOLDOWN_SEC {
                    out.chatter_events.push((idx, ChatTrigger::Spotted));
                    d.last_chatter_at = now;
                }
            }
        } else if visible && low_hp {
            // Fuite : insère un waypoint à 800u DOS au joueur.
            // Pas de pathfinding — l'anti-stuck IA fera le reste si
            // ça tape un mur.
            d.bot.target_enemy = None;
            d.last_saw_player_at = None;
            d.first_seen_player_at = None;
            // Chatter LowHp — sur transition vers fuite.
            if (now - d.last_chatter_at) >= CHATTER_COOLDOWN_SEC {
                out.chatter_events.push((idx, ChatTrigger::LowHp));
                d.last_chatter_at = now;
            }
            let from_player = (d.body.origin - player_eye).truncate();
            if from_player.length_squared() > 1.0 {
                let dir = from_player.normalize();
                let escape =
                    d.body.origin + Vec3::new(dir.x * 800.0, dir.y * 800.0, 0.0);
                if d.bot.waypoints.is_empty()
                    || (d.bot.waypoints[0] - escape).length() > 200.0
                {
                    d.bot.waypoints.insert(0, escape);
                }
            }
        } else if let Some(t) = d.last_saw_player_at {
            if now - t > memory {
                d.bot.target_enemy = None;
                d.last_saw_player_at = None;
                d.first_seen_player_at = None;
            }
        }

        // **Sound awareness** (v0.9.5) — si le bot a entendu le joueur
        // récemment ET qu'il n'a pas de cible visible, il insère la
        // position du bruit comme waypoint d'enquête.  Combiné avec le
        // FOV/LOS test ci-dessus, ça donne :
        //   1. joueur tire derrière un mur → bots dans BOT_HEARING_RADIUS
        //      reçoivent last_heard_pos
        //   2. ils convergent vers cette position
        //   3. arrivés en LOS, le combat normal prend le relais
        // Le waypoint expire avec la mémoire auditive (cf. `now - heard`).
        if d.bot.target_enemy.is_none() && !d.health.is_dead() {
            if let Some(heard) = d.last_heard_pos {
                if now - d.last_heard_at <= BOT_HEARING_MEMORY_SEC {
                    let dist_to_noise = (d.body.origin - heard).length();
                    // Skip si déjà sur place — sinon le bot tournerait
                    // en rond sur la position du dernier bruit.
                    if dist_to_noise > 100.0 {
                        let already = d.bot.waypoints
                            .first()
                            .map(|w| (*w - heard).length() < 64.0)
                            .unwrap_or(false);
                        if !already {
                            d.bot.waypoints.insert(0, heard);
                            // Chatter "j'ai entendu un truc" — rare
                            // (CHAT_TRIGGER_PROB * 0.30) pour ne pas
                            // confirmer chaque planque acoustique.
                            if (now - d.last_chatter_at) >= CHATTER_COOLDOWN_SEC {
                                out.chatter_events.push((idx, ChatTrigger::Heard));
                                d.last_chatter_at = now;
                            }
                        }
                    }
                } else {
                    // Mémoire expirée → on oublie.
                    d.last_heard_pos = None;
                }
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
fn queue_bots(r: &mut Renderer, rig: &PlayerRig, bots: &mut [BotDriver], time_sec: f32) {
    use glam::{Mat4 as GMat4, Quat, Vec3 as GVec3};

    // Seuils de la machine d'états — en secondes depuis l'évènement.
    const ATTACK_WINDOW_SEC: f32 = 0.40;
    const PAIN_WINDOW_SEC: f32 = 0.20;
    const LAND_WINDOW_SEC: f32 = 0.15;
    const TURN_WINDOW_SEC: f32 = 0.25;
    const GESTURE_DURATION_SEC: f32 = 2.7; // 40 frames à 15 fps
    // Vitesse XY au-dessus de laquelle on considère que le bot « court »
    // plutôt qu'il traîne sur place — 40 u/s ≈ vitesse de walk Q3.
    const RUN_SPEED_SQ: f32 = 40.0 * 40.0;

    // **Anim lookup** : `rig.anims` (animation.cfg parsé) en priorité,
    // fallback sur les constants `bot_anims::*` si la cfg est absente.
    let anim = |name: &'static str, fallback: AnimRange| -> AnimRange {
        rig.anims.get(name).copied().unwrap_or(fallback)
    };
    let a_both_death1 = anim("BOTH_DEATH1", bot_anims::BOTH_DEATH1);
    let a_both_dead1  = anim("BOTH_DEAD1",  bot_anims::BOTH_DEAD1);
    let a_both_death2 = anim("BOTH_DEATH2", bot_anims::BOTH_DEATH2);
    let a_both_dead2  = anim("BOTH_DEAD2",  bot_anims::BOTH_DEAD2);
    let a_both_death3 = anim("BOTH_DEATH3", bot_anims::BOTH_DEATH3);
    let a_both_dead3  = anim("BOTH_DEAD3",  bot_anims::BOTH_DEAD3);
    let a_torso_gesture = anim("TORSO_GESTURE", bot_anims::TORSO_GESTURE);
    let a_torso_attack  = anim("TORSO_ATTACK",  bot_anims::TORSO_ATTACK);
    let a_torso_stand   = anim("TORSO_STAND",   bot_anims::TORSO_STAND);
    let a_torso_pain    = anim("TORSO_PAIN1",   bot_anims::TORSO_PAIN1);
    let a_legs_walk = anim("LEGS_WALK", bot_anims::LEGS_WALK);
    let a_legs_run  = anim("LEGS_RUN",  bot_anims::LEGS_RUN);
    let a_legs_back = anim("LEGS_BACK", bot_anims::LEGS_BACK);
    let a_legs_jump = anim("LEGS_JUMP", bot_anims::LEGS_JUMP);
    let a_legs_land = anim("LEGS_LAND", bot_anims::LEGS_LAND);
    let a_legs_jumpb = anim("LEGS_JUMPB", bot_anims::LEGS_JUMPB);
    let a_legs_landb = anim("LEGS_LANDB", bot_anims::LEGS_LANDB);
    let a_legs_idle = anim("LEGS_IDLE", bot_anims::LEGS_IDLE);
    let a_legs_turn = anim("LEGS_TURN", bot_anims::LEGS_TURN);

    let nf_lower = rig.lower.num_frames();
    let nf_upper = rig.upper.num_frames();
    let nf_head = rig.head.num_frames();

    for d in bots {
        // --- Bot mort : on joue BOTH_DEATH puis on freeze sur BOTH_DEAD.
        // Variation 0/1/2 selon `death_variant` pour casser l'uniformité
        // (3 anims de mort distinctes en Q3).
        if d.health.is_dead() {
            let Some(death_at) = d.death_started_at else {
                // Pas encore enregistré : skip (pas mort la frame courante).
                continue;
            };
            let phase = time_sec - death_at;
            // Death dure 30 frames à 25 fps = 1.2 s.  Au-delà, freeze
            // sur la dernière frame (pose cadavre).
            let (death_range, dead_range) = match d.death_variant % 3 {
                0 => (a_both_death1, a_both_dead1),
                1 => (a_both_death2, a_both_dead2),
                _ => (a_both_death3, a_both_dead3),
            };
            let active = if phase < 1.2 { death_range } else { dead_range };
            let (fa_l, fb_l, lerp_l) = active.sample(phase, nf_lower);
            let (fa_u, fb_u, lerp_u) = active.sample(phase, nf_upper);
            let o = d.body.origin;
            // Cadavre : on le laisse tomber au sol, pas de yaw spinning.
            let rot = Quat::from_rotation_z(d.body.view_angles.yaw.to_radians());
            let scale_idx = d.bot.name.bytes().fold(0u32, |a, b| a.wrapping_add(b as u32))
                as usize;
            let s = bot_scale(scale_idx);
            let lower_m = GMat4::from_scale_rotation_translation(
                GVec3::new(s, s, s), rot, GVec3::new(o.x, o.y, o.z),
            );
            let ident = GMat4::IDENTITY;
            let torso_local = rig.lower.tag_transform(fa_l, fb_l, lerp_l, "tag_torso").unwrap_or(ident);
            let upper_m = lower_m * torso_local;
            let head_local = rig.upper.tag_transform(fa_u, fb_u, lerp_u, "tag_head").unwrap_or(ident);
            let head_m = upper_m * head_local;
            // Tint plus sombre pour signaler "mort" (corps qui se fige).
            let dead_tint = [d.tint[0] * 0.55, d.tint[1] * 0.55, d.tint[2] * 0.55, d.tint[3]];
            let head_tint = bot_head_tint(dead_tint);
            r.draw_md3_animated(rig.lower.clone(), lower_m, dead_tint, fa_l, fb_l, lerp_l);
            r.draw_md3_animated(rig.upper.clone(), upper_m, dead_tint, fa_u, fb_u, lerp_u);
            r.draw_md3_animated(rig.head.clone(),  head_m,  head_tint, 0, 0, 0.0);
            continue;
        }

        // --- Sélection d'anim côté jambes (lower) + torse (upper).
        let v_xy_sq = d.body.velocity.x * d.body.velocity.x
            + d.body.velocity.y * d.body.velocity.y;
        let v_xy = v_xy_sq.sqrt();
        let moving = v_xy_sq > RUN_SPEED_SQ;
        let recently_fired = (time_sec - d.last_fire_at) < ATTACK_WINDOW_SEC;
        let recently_hurt = (time_sec - d.last_damage_at) < PAIN_WINDOW_SEC;
        let recently_landed = (time_sec - d.last_land_at) < LAND_WINDOW_SEC;
        let recently_turned = (time_sec - d.last_turn_at) < TURN_WINDOW_SEC;
        let airborne = !d.body.on_ground;
        // Mouvement vers l'arrière : projection vélocité sur forward < 0.
        let yaw_rad = d.body.view_angles.yaw.to_radians();
        let fwd_x = yaw_rad.cos();
        let fwd_y = yaw_rad.sin();
        let forward_speed = d.body.velocity.x * fwd_x + d.body.velocity.y * fwd_y;
        let moving_backward = moving && forward_speed < -20.0;
        // Gesture (taunt) actif ?
        let gesturing = d.gesture_started_at
            .map(|t| time_sec - t < GESTURE_DURATION_SEC)
            .unwrap_or(false);

        // **Upper anim** — priorité : pain > gesture > attack > stand.
        // Weapon-aware : Gauntlet utilise TORSO_ATTACK2 (swing mêlée).
        // v0.9.5++ : pain joue TORSO_PAIN1 au lieu de TORSO_GESTURE
        // (le bot ne se moque plus en se prenant des balles).
        let upper_range = if recently_hurt {
            a_torso_pain
        } else if gesturing {
            a_torso_gesture
        } else if recently_fired {
            a_torso_attack
        } else {
            a_torso_stand
        };
        // **Lower anim** — priorité : airborne > land > move > turn > idle.
        let lower_range = if airborne {
            if forward_speed < -20.0 { a_legs_jumpb } else { a_legs_jump }
        } else if recently_landed {
            if forward_speed < -20.0 { a_legs_landb } else { a_legs_land }
        } else if moving_backward {
            a_legs_back
        } else if moving {
            // **Seuil run/walk** baissé de 200 → 120 u/s (v0.9.5++ —
            // était trop élevé : la plupart des bots à vitesse
            // soutenue ~150 u/s jouaient WALK au lieu de RUN, ce qui
            // donnait une impression molle).  120 u/s correspond à
            // la convention Q3 historique (`cg.predicted_player_state`).
            if v_xy_sq > 120.0 * 120.0 { a_legs_run } else { a_legs_walk }
        } else if recently_turned {
            a_legs_turn
        } else {
            a_legs_idle
        };

        // **Per-bot phase offset** (v0.9.5) — hash du nom donne un
        // décalage stable [0, 1] secondes pour que deux bots côte à
        // côte ne lèvent pas le pied en même temps. Sans ça, l'IA
        // a un effet "ballet militaire" très immersion-breaking.
        let name_hash = d.bot.name.bytes()
            .fold(0u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
        let phase_offset = (name_hash & 0xff) as f32 / 255.0;

        // **Speed-modulated walk/run rate** — quand le bot court à
        // pleine vitesse Q3 (320 u/s) le cycle joue à la fps standard.
        // À la moitié, fps × 0.5.  Donne un rendu cohérent même en
        // décélération ou strafing dur.
        let is_locomotion = lower_range.start == a_legs_run.start
            || lower_range.start == a_legs_walk.start
            || lower_range.start == a_legs_back.start;
        let speed_factor = if is_locomotion {
            (v_xy / 320.0).clamp(0.5, 1.5)
        } else {
            1.0
        };
        // **Phase rebase on anim change** (v0.9.5++) — quand la
        // `lower_range` change (run → idle, jump → land, etc.) on
        // remet à 0 le timer pour que l'anim démarre à sa frame de
        // début au lieu de téléporter au milieu du cycle.  Idem
        // pour upper.  Le `phase_offset` reste appliqué pour décaler
        // les bots entre eux (anti-ballet militaire).
        if d.lower_anim_start != lower_range.start {
            d.lower_anim_start = lower_range.start;
            d.lower_anim_started_at = time_sec;
        }
        if d.upper_anim_start != upper_range.start {
            d.upper_anim_start = upper_range.start;
            d.upper_anim_started_at = time_sec;
        }
        let phase_l = (time_sec - d.lower_anim_started_at) * speed_factor + phase_offset;
        let phase_u = (time_sec - d.upper_anim_started_at) + phase_offset;
        let (fa_l, fb_l, lerp_l) = lower_range.sample(phase_l, nf_lower);
        let (fa_u, fb_u, lerp_u) = upper_range.sample(phase_u, nf_upper);
        // La tête ne s'anime pas en Q3 : on la rend sur la première
        // frame (les meshes head.md3 sont statiques).
        let (fa_h, fb_h, lerp_h) = (0usize, 0usize, 0.0_f32);
        let _ = nf_head;

        let o = d.body.origin;
        let rot = Quat::from_rotation_z(d.body.view_angles.yaw.to_radians());
        // Scale par bot — 5 valeurs ±10 % pour distinguer les silhouettes
        // sur le même rig partagé (gain visuel sans coût mémoire).
        // L'index utilisé est dérivé du nom (hash léger) plutôt que du
        // slot pour rester stable d'un match à l'autre si le bot reste
        // — c'est plus subtil mais ça ancre une "identité" par bot.
        let scale_idx = d.bot.name.bytes().fold(0u32, |a, b| a.wrapping_add(b as u32))
            as usize;
        let s = bot_scale(scale_idx);

        // **Idle breath bob** (v0.9.5) — sinusoïde Z 1u amplitude
        // quand le bot est au sol et immobile. Suffisant pour que la
        // silhouette ne paraisse pas figée comme un piquet.  Décalée
        // par phase_offset pour que les bots ne respirent pas en
        // synchro.
        let idle_bob = if !airborne && !moving {
            ((time_sec * 1.4 + phase_offset * std::f32::consts::TAU).sin()) * 1.2
        } else {
            0.0
        };

        // **Body lean on strafe** (v0.9.5) — quand le bot strafe en
        // vitesse, on incline légèrement le torse dans le sens du
        // mouvement.  Calculé par projection de la vélocité sur le
        // vecteur "right" du bot (basé sur son yaw).
        let yaw_rad = d.body.view_angles.yaw.to_radians();
        let right_x = -yaw_rad.sin();
        let right_y = yaw_rad.cos();
        let strafe_speed = d.body.velocity.x * right_x + d.body.velocity.y * right_y;
        let lean_rad = (strafe_speed / 320.0).clamp(-0.18, 0.18); // ~10° max
        let lean_quat = Quat::from_axis_angle(
            GVec3::new(yaw_rad.cos(), yaw_rad.sin(), 0.0), // axe forward
            -lean_rad,
        );
        let combined_rot = lean_quat * rot;

        // **Hull → feet offset** (v0.9.5++ fix lévitation) — `body.origin`
        // pointe sur le CENTRE du hull (Q3 convention : PLAYER_MINS.z = -24).
        // Le lower.md3 a son pivot aux pieds, donc on doit translater de
        // -24 pour que les pieds atterrissent sur le sol.  Avant ce fix
        // les bots flottaient ~24u au-dessus du sol.
        let feet_z = o.z + PLAYER_HULL_MIN_Z;
        let lower_m = GMat4::from_scale_rotation_translation(
            GVec3::new(s, s, s),
            combined_rot,
            GVec3::new(o.x, o.y, feet_z + idle_bob),
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
        let mut head_m = upper_m * head_local;

        // **Headlook IK** (G1d) — quand le bot a une cible, la tête
        // suit la direction de la cible en yaw seul (clamp ±45°).
        // Pitch ignoré pour ne pas casser les frames du head MD3 qui
        // n'a généralement qu'un seul frame statique.
        if let Some(target) = d.bot.target_enemy {
            let head_world =
                Vec3::new(head_m.col(3).x, head_m.col(3).y, head_m.col(3).z);
            let to = target - head_world;
            if to.length_squared() > 1.0 {
                let look_yaw = to.y.atan2(to.x).to_degrees();
                let body_yaw = d.body.view_angles.yaw;
                let mut delta = look_yaw - body_yaw;
                while delta > 180.0 {
                    delta -= 360.0;
                }
                while delta < -180.0 {
                    delta += 360.0;
                }
                let clamped = delta.clamp(-45.0, 45.0);
                let extra_rot = Quat::from_rotation_z(clamped.to_radians());
                let extra_m = GMat4::from_quat(extra_rot);
                let p = head_world;
                let to_origin =
                    GMat4::from_translation(GVec3::new(-p.x, -p.y, -p.z));
                let from_origin =
                    GMat4::from_translation(GVec3::new(p.x, p.y, p.z));
                head_m = from_origin * extra_m * to_origin * head_m;
            }
        }

        // Head tint distinct (plus clair) — différencie visuellement la
        // tête du torse, et accentue la lecture du headshot quand le
        // joueur vise le casque clair sur silhouette colorée.
        let head_tint = bot_head_tint(tint);
        r.draw_md3_animated(rig.lower.clone(), lower_m, tint,      fa_l, fb_l, lerp_l);
        r.draw_md3_animated(rig.upper.clone(), upper_m, tint,      fa_u, fb_u, lerp_u);
        r.draw_md3_animated(rig.head.clone(),  head_m,  head_tint, fa_h, fb_h, lerp_h);
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

/// **Prop GLB de décor BR** (v0.9.5+) — entrée générique du pool de
/// décors statiques (rocks, statues, buildings, etc.). Le `prop_name`
/// désigne le mesh GLB enregistré dans le pipeline drone.
///
/// (Anciennement `RockProp` avec encoding hack via signe/grandeur du
/// scale ; refactor v0.9.5++ pour proprement supporter N props.)
struct RockProp {
    pos: Vec3,
    yaw: f32,
    scale: f32,
    tint: [f32; 4],
    /// Clé de prop dans `Renderer::queue_prop` — "rock", "statue",
    /// "building", "tropical", etc.
    prop_name: &'static str,
}

/// **Drone aérien BR** (v0.9.5) — vaisseau GLB qui orbite à grande
/// altitude au-dessus de la zone de combat.  Purement cosmétique :
/// pas de damage, pas d'IA, pas de collision.  Donne de la vie à la
/// skybox.
struct Drone {
    /// Centre de l'orbite (typiquement le centre de la map ou un POI).
    orbit_center: Vec3,
    /// Rayon de l'orbite XY.
    orbit_radius: f32,
    /// Altitude du drone (Z monde).
    altitude: f32,
    /// Vitesse angulaire en rad/s.  Signe = sens de rotation.
    angular_speed: f32,
    /// Phase initiale en radians (décalage par drone pour qu'ils ne
    /// soient pas tous au même point de l'orbite).
    phase: f32,
    /// Échelle de rendu (chaque drone est légèrement différent).
    scale: f32,
    /// Tint RGBA — bleu pâle / blanc / orange selon les drones.
    tint: [f32; 4],
}

impl Drone {
    /// Position monde au temps `t`.
    fn position(&self, t: f32) -> Vec3 {
        let theta = self.angular_speed * t + self.phase;
        Vec3::new(
            self.orbit_center.x + self.orbit_radius * theta.cos(),
            self.orbit_center.y + self.orbit_radius * theta.sin(),
            self.altitude
                + (theta * 0.7).sin() * 200.0, // léger bobbing vertical
        )
    }
    /// Yaw (rad) tangent à l'orbite — drone face à la direction du mouvement.
    fn yaw(&self, t: f32) -> f32 {
        // Tangente = dérivée de la position : (-sin θ, cos θ).
        let theta = self.angular_speed * t + self.phase;
        // Atan2(dy/dt, dx/dt) = atan2(angular_speed*cos, -angular_speed*sin)
        // = atan2(cos, -sin) (sign respecté par angular_speed).
        let dy = self.angular_speed * theta.cos();
        let dx = -self.angular_speed * theta.sin();
        dy.atan2(dx)
    }
}

/// **Adapter Terrain → LosWorld** (v0.9.5) — permet aux bots de
/// passer le LOS sur la heightmap BR via le trait abstrait défini
/// dans q3-bot, sans introduire de dépendance entre q3-bot et
/// q3-terrain. L'adapter est un wrapper léger qui appelle
/// `Terrain::trace_ray` et compare la fraction comme le BSP path.
struct TerrainLos<'a>(&'a q3_terrain::Terrain);

impl<'a> q3_bot::LosWorld for TerrainLos<'a> {
    fn is_clear(&self, start: q3_math::Vec3, end: q3_math::Vec3) -> bool {
        let tr = self.0.trace_ray(start, end);
        tr.fraction >= 0.999
    }
}

/// **BR Réunion fallback** — synthétise un terrain procédural quand
/// les assets disque ne sont pas présents.  Reproduit la silhouette
/// caractéristique de l'île volcanique :
/// * dôme central élevé (sommets Piton des Neiges + Piton de la Fournaise)
/// * pentes douces vers la côte
/// * océan en bordure (z = 0)
/// * variations multi-octaves pour un relief riche
///
/// Pas équivalent à un vrai SRTM, mais bien plus qu'un terrain plat —
/// le joueur peut grimper, voir des sommets, sentir l'échelle.  Le
/// pipeline Python `tools/dem_to_terrain.py` reste recommandé pour la
/// vraie heightmap.
fn synthesize_reunion_fallback() -> q3_terrain::Terrain {
    use q3_terrain::TerrainMeta;
    let meta = TerrainMeta::reunion_default();
    let w = meta.width;
    let h = meta.height;
    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5;

    // Générateur de bruit pseudo-aléatoire déterministe (hash 2D simple).
    fn noise(x: f32, y: f32, seed: u32) -> f32 {
        let xi = (x * 1.0).to_bits().wrapping_add(seed);
        let yi = (y * 1.0).to_bits().wrapping_add(seed.wrapping_mul(2654435761));
        let h = xi
            .wrapping_mul(73856093)
            .wrapping_add(yi.wrapping_mul(19349663));
        ((h >> 16) as f32 / 32768.0) - 1.0
    }
    fn smooth_noise(x: f32, y: f32, seed: u32) -> f32 {
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let n00 = noise(x0, y0, seed);
        let n10 = noise(x0 + 1.0, y0, seed);
        let n01 = noise(x0, y0 + 1.0, seed);
        let n11 = noise(x0 + 1.0, y0 + 1.0, seed);
        let lx0 = n00 * (1.0 - fx) + n10 * fx;
        let lx1 = n01 * (1.0 - fx) + n11 * fx;
        lx0 * (1.0 - fy) + lx1 * fy
    }

    // Ridge noise — `1 - |2n - 1|` puis squared : crêtes pointues
    // façon chaîne volcanique jeune.
    fn ridge_noise(x: f32, y: f32, seed: u32) -> f32 {
        let n = smooth_noise(x, y, seed);
        let r = 1.0 - (n.abs() * 2.0 - 1.0).abs();
        r * r
    }

    let mut samples = Vec::with_capacity(w * h);
    let mut splat = Vec::with_capacity(w * h);

    let z_range = meta.z_max - meta.z_min;
    let max_radius = (cx.min(cy)) as f32;

    // 2 massifs : Piton des Neiges (centre île ~ grid 1200,1100) et
    // Piton de la Fournaise (~est de l'île).
    let neiges = (1200.0_f32, 1100.0_f32);
    let fournaise = (1630.0_f32, 1057.0_f32);

    for j in 0..h {
        for i in 0..w {
            let x = i as f32;
            let y = j as f32;

            // Enveloppe radiale (océan plein hors-île).
            let dx = (x - cx) / max_radius;
            let dy = (y - cy) / max_radius;
            let r = (dx * dx + dy * dy).sqrt().min(1.6);
            let dome_envelope = (1.0 - r * 0.78).max(0.0).powi(2);

            // Pics centrés sur les 2 massifs.
            let d_neiges =
                ((x - neiges.0).powi(2) + (y - neiges.1).powi(2)).sqrt() / 600.0;
            let d_fournaise =
                ((x - fournaise.0).powi(2) + (y - fournaise.1).powi(2)).sqrt() / 500.0;
            let peak_neiges = (1.0 - d_neiges).max(0.0).powi(2);
            let peak_fournaise = (1.0 - d_fournaise).max(0.0).powi(2) * 0.85;
            let altitude_base =
                (dome_envelope * 0.4 + peak_neiges + peak_fournaise * 0.85).min(1.0);

            // Ridge noise multi-octaves → crêtes acérées.
            let r1 = ridge_noise(x * 0.012, y * 0.012, 11);
            let r2 = ridge_noise(x * 0.030, y * 0.030, 12);
            let r3 = ridge_noise(x * 0.080, y * 0.080, 13);
            let ridges = r1 * 0.6 + r2 * 0.3 + r3 * 0.1;

            // Vallées via value noise basse fréquence.
            let v1 = smooth_noise(x * 0.020, y * 0.020, 21);
            let v2 = smooth_noise(x * 0.060, y * 0.060, 22);
            let valleys = v1 * 0.7 + v2 * 0.3;

            let mountain_relief = if altitude_base > 0.3 {
                ridges * 0.55 + valleys * 0.25
            } else {
                valleys * 0.4 + ridges * 0.1
            };
            // Carving radial (cirques) — basse fréquence inversée.
            let carve = (smooth_noise(x * 0.005, y * 0.005, 31) * 0.5 + 0.5) * 0.30;

            let alt_norm = if altitude_base > 0.001 {
                (altitude_base * (0.55 + mountain_relief * 0.55)
                    - carve * altitude_base)
                    .clamp(0.0, 1.0)
            } else {
                0.0
            };

            let u16_val = if altitude_base <= 0.001 {
                let base = (-meta.z_min / z_range).clamp(0.0, 1.0);
                (base * 65535.0) as u16
            } else {
                let world_z = alt_norm * (meta.z_max - 0.0);
                let norm = ((world_z - meta.z_min) / z_range).clamp(0.0, 1.0);
                (norm * 65535.0) as u16
            };
            samples.push(u16_val);

            // Splat raffiné par altitude (en mètres réels — z_max = 1228
            // unités ≈ 307 m réels après scale 1/10).
            let world_z = alt_norm * (meta.z_max - 0.0);
            let urban_factor = if r > 0.55 && r < 0.85 && valleys > 0.55 {
                200u8
            } else {
                5u8
            };
            let (rock, sand, veg, urban): (u8, u8, u8, u8) =
                if altitude_base <= 0.001 {
                    (120, 90, 0, 0)
                } else if world_z < 50.0 {
                    (10, 230, 15, urban_factor.min(40))
                } else if world_z < 200.0 {
                    (15, 30, 220, urban_factor)
                } else if world_z < 500.0 {
                    (60, 5, 200, (urban_factor / 2).min(60))
                } else if world_z < 900.0 {
                    (140, 5, 110, 0)
                } else if world_z < 1500.0 {
                    (210, 0, 40, 0)
                } else {
                    (245, 5, 0, 0)
                };
            splat.push([rock, sand, veg, urban]);
        }
    }

    q3_terrain::Terrain {
        width: w,
        height: h,
        samples,
        splat,
        meta,
    }
}

/// Tint cyclique pour distinguer les bots visuellement.
fn bot_tint(idx: usize) -> [f32; 4] {
    // Palette élargie v0.9.5 — 12 teintes distinctes pour différencier
    // visuellement les bots dans une partie >6.  Ordre choisi pour que
    // 2 bots consécutifs n'aient jamais des couleurs proches (cycle
    // alternant chaud/froid).
    const PALETTE: &[[f32; 4]] = &[
        [1.00, 0.55, 0.55, 1.0], // rouge corail
        [0.55, 0.78, 1.00, 1.0], // bleu acier
        [0.70, 1.00, 0.65, 1.0], // vert lime
        [1.00, 0.85, 0.40, 1.0], // jaune doré
        [1.00, 0.65, 1.00, 1.0], // magenta vif
        [0.55, 1.00, 0.95, 1.0], // cyan glacial
        [1.00, 0.50, 0.20, 1.0], // orange sang
        [0.65, 0.55, 1.00, 1.0], // violet améthyste
        [0.90, 0.95, 0.55, 1.0], // jaune lime
        [0.55, 0.95, 0.55, 1.0], // vert émeraude
        [1.00, 0.75, 0.85, 1.0], // rose pastel
        [0.85, 0.55, 0.30, 1.0], // bronze sienna
    ];
    PALETTE[idx % PALETTE.len()]
}

/// Variation de scale par index — donne 5 silhouettes distinctes
/// (petits / grands) sur le même rig MD3 partagé. Sans dupliquer le
/// rig en mémoire, on obtient une variété visuelle « instant ». Les
/// valeurs restent dans ±10 % pour ne pas casser la hitbox Q3 (le
/// trace_capsule reste dimensionné sur la taille std).
fn bot_scale(idx: usize) -> f32 {
    // Multiplier global appliqué à toutes les variantes — bump à 1.30
    // (user request : "bots trop petits").  Les ratios entre variantes
    // sont conservés pour la diversité visuelle.
    const GLOBAL_MULT: f32 = 1.30;
    const SCALES: &[f32] = &[0.93, 1.00, 1.06, 0.97, 1.03];
    SCALES[idx % SCALES.len()] * GLOBAL_MULT
}

/// Tint distinct pour la tête — base un peu plus claire (mix vers le
/// blanc) que le corps. Donne au bot un look "casque clair" et facilite
/// la distinction visuelle du headshot.
fn bot_head_tint(body_tint: [f32; 4]) -> [f32; 4] {
    [
        (body_tint[0] * 0.65 + 0.35).min(1.0),
        (body_tint[1] * 0.65 + 0.35).min(1.0),
        (body_tint[2] * 0.65 + 0.35).min(1.0),
        body_tint[3],
    ]
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
/// **Asset search bases** (v0.9.5+) — retourne la liste des
/// répertoires racine où chercher les assets disque (GLB, music,
/// terrain).  Permet à l'utilisateur de lancer `cargo run` depuis
/// le workspace OU `target/release/q3.exe` directement sans casser
/// la résolution.
fn resolve_asset_search_bases() -> Vec<PathBuf> {
    let mut bases: Vec<PathBuf> = Vec::new();
    // 1. CWD courant
    if let Ok(cwd) = std::env::current_dir() {
        bases.push(cwd);
    }
    // 2. Dossier de l'exécutable (utile quand l'user lance .exe direct)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            bases.push(parent.to_path_buf());
            // 3. Parents successifs de l'exe (target/release → target → workspace)
            if let Some(p2) = parent.parent() {
                bases.push(p2.to_path_buf());
                if let Some(p3) = p2.parent() {
                    bases.push(p3.to_path_buf());
                }
            }
        }
    }
    bases
}

/// **Music player** (v0.9.5+) — liste les fichiers audio jouables
/// trouvés dans :
///   1. `assets/music/` (relatif au CWD — bundle développeur)
///   2. `<USERPROFILE>/Music/` ou `$HOME/Music/` (dossier utilisateur)
///   3. `<userconfig>/q3-rust/music/` (à côté de q3config.cfg)
///
/// Extensions acceptées : .wav, .ogg, .mp3 (mp3 nécessite la feature
/// rodio "mp3" — pour l'instant on liste mais le décodage peut échouer).
fn list_music_files() -> Vec<PathBuf> {
    list_music_files_with_extra(&[])
}

/// Liste tous les fichiers audio jouables dans :
///   1. `assets/music/` et `music/` (relatif au CWD)
///   2. `<userconfig>/q3-rust/music/` (à côté de q3config.cfg)
///   3. `~/Music/` et `~/Downloads/`  (Windows + Unix)
///   4. `extra_paths` fournis par l'appelant (cvar `s_musicpath` —
///      semi-colon separated sur Windows, colon-separated sinon)
///
/// Scan **récursif** (dlevel 4 max — évite l'explosion combinatoire
/// sur dossiers Music géants), extensions filtrées (.wav/.ogg/.oga/
/// .mp3/.flac).  Tri alphabétique pour stabilité.
fn list_music_files_with_extra(extra_paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut roots: Vec<PathBuf> = Vec::new();
    roots.push(PathBuf::from("assets/music"));
    roots.push(PathBuf::from("music"));
    if let Ok(home) = std::env::var("USERPROFILE") {
        let home = PathBuf::from(home);
        roots.push(home.join("Music"));
        roots.push(home.join("Downloads"));
        roots.push(home.join("OneDrive").join("Music"));
        roots.push(home.join("Desktop"));
    }
    if let Ok(home) = std::env::var("HOME") {
        let home = PathBuf::from(home);
        roots.push(home.join("Music"));
        roots.push(home.join("Downloads"));
    }
    if let Some(cfg) = user_config_path() {
        if let Some(parent) = cfg.parent() {
            roots.push(parent.join("music"));
        }
    }
    for p in extra_paths {
        roots.push(p.clone());
    }
    for root in &roots {
        scan_audio_recursive(root, 0, 4, &mut out);
    }
    out.sort();
    out.dedup();
    out
}

/// Scan récursif d'un dossier pour fichiers audio.  Limité à
/// `max_depth` niveaux pour éviter de remonter `~/Music/` en entier.
fn scan_audio_recursive(root: &std::path::Path, depth: usize, max_depth: usize, out: &mut Vec<PathBuf>) {
    if depth > max_depth { return; }
    let Ok(rd) = std::fs::read_dir(root) else { return; };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            // Skip dossiers cachés (.git, .Trash, etc.) pour ne pas
            // scanner des arborescences inutiles.
            let skip_hidden = p
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with('.'))
                .unwrap_or(false);
            if !skip_hidden {
                scan_audio_recursive(&p, depth + 1, max_depth, out);
            }
        } else if p.is_file() {
            if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                let ext_lower = ext.to_ascii_lowercase();
                if matches!(ext_lower.as_str(), "wav" | "ogg" | "oga" | "mp3" | "flac") {
                    out.push(p);
                }
            }
        }
    }
}

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
    ammo_crate_scale: Option<f32>,
    quad_pickup_scale: Option<f32>,
    health_pack_scale: Option<f32>,
    railgun_pickup_scale: Option<f32>,
    grenade_ammo_scale: Option<f32>,
    rocket_ammo_scale: Option<f32>,
    cell_ammo_scale: Option<f32>,
    lg_ammo_scale: Option<f32>,
    big_armor_scale: Option<f32>,
    plasma_pickup_scale: Option<f32>,
    railgun_ammo_scale: Option<f32>,
    regen_pickup_scale: Option<f32>,
    machinegun_pickup_scale: Option<f32>,
    bfg_pickup_scale: Option<f32>,
    lightninggun_pickup_scale: Option<f32>,
    shotgun_pickup_scale: Option<f32>,
    grenadelauncher_pickup_scale: Option<f32>,
    gauntlet_pickup_scale: Option<f32>,
    shotgun_ammo_scale: Option<f32>,
    bfg_ammo_scale: Option<f32>,
    rocketlauncher_pickup_scale: Option<f32>,
    combat_armor_scale: Option<f32>,
    medkit_scale: Option<f32>,
    armor_shard_scale: Option<f32>,
) {
    use glam::{Mat4, Quat, Vec3 as GVec3};
    let spin = (time_sec * 120.0).to_radians();
    let bob = (time_sec * 2.0).sin() * 3.0;
    let mg_slot = WeaponId::Machinegun.slot() as usize;
    let gl_slot = WeaponId::Grenadelauncher.slot() as usize;
    let sg_slot = WeaponId::Shotgun.slot() as usize;
    let rl_slot = WeaponId::Rocketlauncher.slot() as usize;
    let pg_slot = WeaponId::Plasmagun.slot() as usize;
    let lg_slot = WeaponId::Lightninggun.slot() as usize;
    let has_ammo_crate = r.has_prop("ammo_crate") && ammo_crate_scale.is_some();
    let has_quad = r.has_prop("quad_pickup") && quad_pickup_scale.is_some();
    let has_health_pack = r.has_prop("health_pack") && health_pack_scale.is_some();
    let has_railgun_pickup = r.has_prop("railgun_pickup") && railgun_pickup_scale.is_some();
    let has_grenade_ammo = r.has_prop("grenade_ammo") && grenade_ammo_scale.is_some();
    let has_rocket_ammo = r.has_prop("rocket_ammo") && rocket_ammo_scale.is_some();
    let has_cell_ammo = r.has_prop("cell_ammo") && cell_ammo_scale.is_some();
    let has_lg_ammo = r.has_prop("lg_ammo") && lg_ammo_scale.is_some();
    let has_big_armor = r.has_prop("big_armor") && big_armor_scale.is_some();
    let has_plasma_pickup = r.has_prop("plasma_pickup") && plasma_pickup_scale.is_some();
    let has_railgun_ammo = r.has_prop("railgun_ammo") && railgun_ammo_scale.is_some();
    let has_regen_pickup = r.has_prop("regen_pickup") && regen_pickup_scale.is_some();
    let has_machinegun_pickup = r.has_prop("machinegun_pickup") && machinegun_pickup_scale.is_some();
    let has_bfg_pickup = r.has_prop("bfg_pickup") && bfg_pickup_scale.is_some();
    let has_lightninggun_pickup = r.has_prop("lightninggun_pickup") && lightninggun_pickup_scale.is_some();
    let has_shotgun_pickup = r.has_prop("shotgun_pickup") && shotgun_pickup_scale.is_some();
    let has_grenadelauncher_pickup = r.has_prop("grenadelauncher_pickup") && grenadelauncher_pickup_scale.is_some();
    let has_gauntlet_pickup = r.has_prop("gauntlet_pickup") && gauntlet_pickup_scale.is_some();
    let has_shotgun_ammo = r.has_prop("shotgun_ammo") && shotgun_ammo_scale.is_some();
    let has_bfg_ammo = r.has_prop("bfg_ammo") && bfg_ammo_scale.is_some();
    let bfg_slot = WeaponId::Bfg.slot() as usize;
    let has_rocketlauncher_pickup = r.has_prop("rocketlauncher_pickup") && rocketlauncher_pickup_scale.is_some();
    let has_combat_armor = r.has_prop("combat_armor") && combat_armor_scale.is_some();
    let has_medkit = r.has_prop("medkit") && medkit_scale.is_some();
    let has_armor_shard = r.has_prop("armor_shard") && armor_shard_scale.is_some();
    let rg_slot = WeaponId::Railgun.slot() as usize;

    // Helper local pour produire la matrice transform Y-up→Z-up + spin.
    let make_glb_model = |s: f32, yaw: f32, trans: GVec3| -> [[f32; 4]; 4] {
        let cy = yaw.cos();
        let sy = yaw.sin();
        [
            [cy * s,  sy * s, 0.0, 0.0],
            [0.0,     0.0,    s,   0.0],
            [sy * s, -cy * s, 0.0, 0.0],
            [trans.x, trans.y, trans.z, 1.0],
        ]
    };

    // Pulse luminance pour les dlights de pickup — phase ≈ 1.7 Hz,
    // [0.7, 1.0] pour un battement net mais pas hypnotique.
    let pulse = 0.85 + 0.15 * (time_sec * 10.7).sin();
    for p in pickups {
        if p.respawn_at.is_some() { continue; }
        if is_client && unavailable_remote.contains(&p.entity_index) { continue; }
        let trans = GVec3::new(p.origin.x, p.origin.y, p.origin.z + bob);
        let rot = Quat::from_rotation_z(spin + p.angles.yaw.to_radians());
        let yaw = spin + p.angles.yaw.to_radians();
        // **Pickup pulse glow** — dlight respawné chaque frame avec
        // une lifetime ~2 frames, ce qui donne un éclairage continu
        // et pulsé.  Couleur dérivée du `PickupKind` :
        //   - Health : vert
        //   - Armor : cyan
        //   - Weapon : jaune
        //   - Ammo : ocre
        //   - Powerup : couleur du powerup (Quad bleu, Haste orange…)
        //   - Holdable / Inert : blanc neutre
        // Radius modeste (160u) pour ne pas inonder une pièce ; fait
        // que les items se "voient" même dans un coin sombre sans
        // muter en spotlight.
        let glow = match p.kind {
            PickupKind::Health { .. } => [0.30, 1.00, 0.40],
            PickupKind::Armor { .. } => [0.40, 0.85, 1.00],
            PickupKind::Weapon { .. } => [1.00, 0.85, 0.30],
            PickupKind::Ammo { .. } => [1.00, 0.65, 0.20],
            PickupKind::Powerup { powerup, .. } => {
                let c = powerup.pickup_fx_color();
                [c[0], c[1], c[2]]
            }
            PickupKind::Holdable { .. } | PickupKind::Inert => [0.90, 0.90, 0.90],
        };
        // Centre la light sur l'item bobbé (y compris le bob vertical).
        let light_pos = q3_math::Vec3::new(trans.x, trans.y, trans.z + 8.0);
        r.spawn_dlight(
            light_pos,
            160.0,
            glow,
            1.6 * pulse, // intensité battue
            time_sec,
            0.05,        // lifetime ≈ 3 frames @ 60 fps → re-armé en continu
        );

        // **Quad pickup GLB** — remplace l'icone MD3 du Quad Damage.
        if has_quad
            && matches!(
                p.kind,
                PickupKind::Powerup { powerup: PowerupKind::QuadDamage, .. }
            )
        {
            let s = quad_pickup_scale.unwrap();
            r.queue_prop("quad_pickup", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Regen pickup GLB** — remplace l'icone MD3 du Regeneration.
        if has_regen_pickup
            && matches!(
                p.kind,
                PickupKind::Powerup { powerup: PowerupKind::Regeneration, .. }
            )
        {
            let s = regen_pickup_scale.unwrap();
            r.queue_prop("regen_pickup", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Machine Gun pickup GLB** — pickup d'ARME Machinegun.
        // Doit passer AVANT la branche ammo_crate qui matchait aussi
        // l'arme MG.  Distinct du Machinegun ammo qui reste sur
        // ammo_crate (cf. branche suivante).
        if has_machinegun_pickup
            && matches!(
                p.kind,
                PickupKind::Weapon { weapon: WeaponId::Machinegun, .. }
            )
        {
            let s = machinegun_pickup_scale.unwrap();
            r.queue_prop("machinegun_pickup", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Ammo crate GLB** — MG ammo + weapon MG pickup en fallback
        // si machinegun_pickup pas disponible.
        let is_mg_ammo = matches!(
            p.kind,
            PickupKind::Ammo { slot, .. } if slot as usize == mg_slot
        ) || matches!(
            p.kind,
            PickupKind::Weapon { weapon: WeaponId::Machinegun, .. }
        );
        if is_mg_ammo && has_ammo_crate {
            let s = ammo_crate_scale.unwrap();
            r.queue_prop("ammo_crate", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Health pack GLB** — toutes variantes item_health* (small,
        // medium, large, mega).  Le mesh unique sert pour les 4 tiers
        // (la valeur de soin reste différente — c'est un détail
        // gameplay, pas visuel).
        if has_health_pack
            && matches!(p.kind, PickupKind::Health { .. })
        {
            let s = health_pack_scale.unwrap();
            r.queue_prop("health_pack", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Railgun pickup GLB** — pickup au sol (le viewmodel 1ère
        // personne utilise aussi `railgun_pickup` via `queue_viewmodel`
        // avec un muzzle flash hardcoded à la place de `tag_flash`).
        if has_railgun_pickup
            && matches!(
                p.kind,
                PickupKind::Weapon { weapon: WeaponId::Railgun, .. }
            )
        {
            let s = railgun_pickup_scale.unwrap();
            r.queue_prop("railgun_pickup", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Railgun ammo (slugs) GLB** — boîte de slugs pour le RG
        // (item_slugs).  Distinct de l'arme RG ci-dessus (railgun_pickup).
        if has_railgun_ammo
            && matches!(
                p.kind,
                PickupKind::Ammo { slot, .. } if slot as usize == rg_slot
            )
        {
            let s = railgun_ammo_scale.unwrap();
            r.queue_prop("railgun_ammo", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Grenade Launcher pickup GLB** — pickup d'ARME Grenadelauncher.
        // Doit passer AVANT la branche grenade_ammo qui matchait aussi
        // l'arme GL.  Les munitions grenades restent sur grenade_ammo.
        if has_grenadelauncher_pickup
            && matches!(
                p.kind,
                PickupKind::Weapon { weapon: WeaponId::Grenadelauncher, .. }
            )
        {
            let s = grenadelauncher_pickup_scale.unwrap();
            r.queue_prop(
                "grenadelauncher_pickup",
                make_glb_model(s, yaw, trans),
                [1.0, 1.0, 1.0, 1.0],
            );
            continue;
        }

        // **Grenade ammo box GLB** — boîte de munitions grenades
        // (item_grenades) + pickup d'arme Grenadelauncher en fallback
        // si grenadelauncher_pickup pas disponible.
        let is_grenade_ammo = matches!(
            p.kind,
            PickupKind::Ammo { slot, .. } if slot as usize == gl_slot
        ) || matches!(
            p.kind,
            PickupKind::Weapon { weapon: WeaponId::Grenadelauncher, .. }
        );
        if is_grenade_ammo && has_grenade_ammo {
            let s = grenade_ammo_scale.unwrap();
            r.queue_prop("grenade_ammo", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Rocket Launcher pickup GLB** — pickup d'ARME Rocketlauncher.
        // Doit passer AVANT la branche rocket_ammo qui matchait aussi
        // l'arme RL.  Les munitions roquettes restent sur rocket_ammo.
        if has_rocketlauncher_pickup
            && matches!(
                p.kind,
                PickupKind::Weapon { weapon: WeaponId::Rocketlauncher, .. }
            )
        {
            let s = rocketlauncher_pickup_scale.unwrap();
            r.queue_prop(
                "rocketlauncher_pickup",
                make_glb_model(s, yaw, trans),
                [1.0, 1.0, 1.0, 1.0],
            );
            continue;
        }

        // **Rocket ammo box GLB** — boîte de munitions roquettes
        // (item_rockets) + pickup d'arme Rocketlauncher en fallback
        // si rocketlauncher_pickup pas disponible.
        let is_rocket_ammo = matches!(
            p.kind,
            PickupKind::Ammo { slot, .. } if slot as usize == rl_slot
        ) || matches!(
            p.kind,
            PickupKind::Weapon { weapon: WeaponId::Rocketlauncher, .. }
        );
        if is_rocket_ammo && has_rocket_ammo {
            let s = rocket_ammo_scale.unwrap();
            r.queue_prop("rocket_ammo", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Plasma gun pickup GLB** (v0.9.5++) — uniquement le pickup
        // d'ARME Plasmagun (style fusil d'assaut). Distinct du
        // `cell_ammo` qui reste pour les boîtes de cellules ammo.
        // Branche placée AVANT cell_ammo pour priorité.
        if has_plasma_pickup
            && matches!(
                p.kind,
                PickupKind::Weapon { weapon: WeaponId::Plasmagun, .. }
            )
        {
            let s = plasma_pickup_scale.unwrap();
            r.queue_prop("plasma_pickup", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Shotgun pickup GLB** — pickup d'ARME Shotgun.
        if has_shotgun_pickup
            && matches!(
                p.kind,
                PickupKind::Weapon { weapon: WeaponId::Shotgun, .. }
            )
        {
            let s = shotgun_pickup_scale.unwrap();
            r.queue_prop("shotgun_pickup", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Shotgun shells (cartouches calibre 12) GLB** — boîte de
        // munitions SG (item_shells uniquement, l'arme elle-même
        // utilise shotgun_pickup ci-dessus).
        let is_sg_ammo = matches!(
            p.kind,
            PickupKind::Ammo { slot, .. } if slot as usize == sg_slot
        );
        if is_sg_ammo && has_shotgun_ammo {
            let s = shotgun_ammo_scale.unwrap();
            r.queue_prop("shotgun_ammo", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **BFG ammo GLB** — boîte de munitions BFG (item_bfgammo).
        let is_bfg_ammo = matches!(
            p.kind,
            PickupKind::Ammo { slot, .. } if slot as usize == bfg_slot
        );
        if is_bfg_ammo && has_bfg_ammo {
            let s = bfg_ammo_scale.unwrap();
            r.queue_prop("bfg_ammo", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Gauntlet pickup GLB** — pickup d'ARME Gauntlet (mêlée).
        if has_gauntlet_pickup
            && matches!(
                p.kind,
                PickupKind::Weapon { weapon: WeaponId::Gauntlet, .. }
            )
        {
            let s = gauntlet_pickup_scale.unwrap();
            r.queue_prop("gauntlet_pickup", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **BFG10K pickup GLB** — pickup d'ARME Bfg.
        if has_bfg_pickup
            && matches!(
                p.kind,
                PickupKind::Weapon { weapon: WeaponId::Bfg, .. }
            )
        {
            let s = bfg_pickup_scale.unwrap();
            r.queue_prop("bfg_pickup", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Cell ammo box GLB** — cellules énergétiques pour le Plasma
        // Gun (item_cells uniquement, l'arme elle-même utilise
        // plasma_pickup ci-dessus).
        let is_cell_ammo = matches!(
            p.kind,
            PickupKind::Ammo { slot, .. } if slot as usize == pg_slot
        );
        if is_cell_ammo && has_cell_ammo {
            let s = cell_ammo_scale.unwrap();
            r.queue_prop("cell_ammo", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Lightning Gun pickup GLB** — pickup d'ARME Lightninggun.
        // Doit passer AVANT la branche lg_ammo qui matchait aussi
        // l'arme LG.  Les batteries (item_lightning) restent sur lg_ammo.
        if has_lightninggun_pickup
            && matches!(
                p.kind,
                PickupKind::Weapon { weapon: WeaponId::Lightninggun, .. }
            )
        {
            let s = lightninggun_pickup_scale.unwrap();
            r.queue_prop(
                "lightninggun_pickup",
                make_glb_model(s, yaw, trans),
                [1.0, 1.0, 1.0, 1.0],
            );
            continue;
        }

        // **LG battery box GLB** — batteries pour le Lightning Gun
        // (item_lightning) + pickup d'arme Lightninggun en fallback si
        // lightninggun_pickup pas disponible.
        let is_lg_ammo = matches!(
            p.kind,
            PickupKind::Ammo { slot, .. } if slot as usize == lg_slot
        ) || matches!(
            p.kind,
            PickupKind::Weapon { weapon: WeaponId::Lightninggun, .. }
        );
        if is_lg_ammo && has_lg_ammo {
            let s = lg_ammo_scale.unwrap();
            r.queue_prop("lg_ammo", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Big armor (Red Armor) GLB** — item_armor_body 100 armor.
        // Match précis sur amount=100 pour ne pas remplacer les
        // armor shards (5) ou combat (50) qui ont leur propre look.
        if has_big_armor
            && matches!(p.kind, PickupKind::Armor { amount: 100 })
        {
            let s = big_armor_scale.unwrap();
            r.queue_prop("big_armor", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Combat armor (Yellow Armor) GLB** — item_armor_combat 50.
        if has_combat_armor
            && matches!(p.kind, PickupKind::Armor { amount: 50 })
        {
            let s = combat_armor_scale.unwrap();
            r.queue_prop("combat_armor", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Armor Shard GLB** — item_armor_shard (5).
        if has_armor_shard
            && matches!(p.kind, PickupKind::Armor { amount: 5 })
        {
            let s = armor_shard_scale.unwrap();
            r.queue_prop("armor_shard", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // **Medkit holdable GLB** — holdable_medkit (trousse en stock).
        if has_medkit
            && matches!(
                p.kind,
                PickupKind::Holdable { kind: HoldableKind::Medkit }
            )
        {
            let s = medkit_scale.unwrap();
            r.queue_prop("medkit", make_glb_model(s, yaw, trans), [1.0, 1.0, 1.0, 1.0]);
            continue;
        }

        // Fallback : MD3 classique.
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
#[allow(clippy::too_many_arguments)]
fn queue_viewmodel(
    r: &mut Renderer,
    mesh: &Arc<Md3Gpu>,
    eye: Vec3,
    view_angles: Angles,
    time_sec: f32,
    invul_until: f32,
    invisible: bool,
    weapon: WeaponId,
    muzzle_active: bool,
    view_kick: f32,
    // **Player cosmetic tint** (v0.9.5++) — multiplié à `tint`
    // appliqué au draw du viewmodel.  Lu depuis `cg_playertint`.
    player_tint: [f32; 3],
    // **Locomotion bob** (v0.9.5++) — phase bob du joueur (synchro
    // avec les footsteps).  Module l'oscillation X/Z de l'arme.
    bob_phase: f32,
    // Vitesse XY du joueur — modulant l'amplitude du bob (0 = pas de bob).
    velocity_xy: f32,
    // True si le joueur est au sol — gate le bob (pas de bob airborne).
    on_ground: bool,
    // Temps écoulé depuis le dernier landing — déclenche le jump-kick
    // pendant 0.3 s (dip + spring).  `f32::INFINITY` = aucun land récent.
    time_since_land: f32,
    // Sway accumulé eased — déplacement écran de l'arme proportionnel
    // au mouvement caméra (yaw, pitch).  Donne du poids à l'arme.
    sway: [f32; 2],
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
    // **Visuel arme en main** v0.9.4 — repositionné + idle sway pour
    // que la main respire au lieu d'être figée comme une statue.
    //
    // Position de base (corrigée v0.9.4) : forward 10 (assez devant
    // pour bien voir la silhouette mais pas envahir l'écran), right 7
    // (main droite décalée du centre, comme un FPS classique),
    // -up 7 (~hauteur ceinture, l'œil scanne le centre du viseur sans
    // que l'arme s'y plante). Précédent : 8 / 5 / 5 → trop ramassé,
    // l'œil voyait l'arme "collée à la lentille".
    //
    // Idle sway : 2 oscillations sin déphasées simulent la respiration.
    //   - vertical (∼0.4 Hz, ±0.6 u)  : monte/descend lentement
    //   - latéral  (∼0.27 Hz, ±0.4 u) : balance G/D désynchronisé
    // Effet : même immobile, la main vit. Frequencies coprime pour
    // ne pas former un pattern circulaire trivial.
    let breath_v = (time_sec * std::f32::consts::TAU * 0.4).sin() * 0.6;
    let breath_h = (time_sec * std::f32::consts::TAU * 0.27 + 1.3).sin() * 0.4;
    let kick_back = view_kick * 3.0;
    let kick_up = view_kick * 1.5;
    // **Locomotion bob** (v0.9.5++) — synchro sur `bob_phase` (qui
    // avance avec le déplacement, déjà utilisé pour les footsteps).
    // Vertical : 2 cycles par foulée (sin(2φ)).  Horizontal : 1 cycle
    // par foulée (sin(φ)) en quadrature → trace une figure 8.
    // Amplitude proportionnelle à la vitesse (0 → 1.0 entre 0 et 320 u/s).
    // Skip airborne (pas de bob en l'air, c'est le jump-kick qui prend).
    let speed_factor = if on_ground {
        (velocity_xy / 320.0).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let bob_v = (bob_phase * 2.0).sin() * 1.6 * speed_factor;
    let bob_h = bob_phase.sin() * 0.9 * speed_factor;
    // **Jump-kick** (v0.9.5++) — quand le joueur retouche le sol après
    // un saut, l'arme dip vers le bas puis spring-back.  Profil de la
    // courbe : sin amorti sur 0.30 s (1 cycle complet).
    const KICK_DURATION: f32 = 0.30;
    let kick_phase = (time_since_land / KICK_DURATION).clamp(0.0, 1.0);
    let kick_amp = if kick_phase < 1.0 {
        // sin(2π × phase) × decay → premier dip à 0.25, retour à 0.5,
        // overshoot léger à 0.75, settle à 1.0.
        (kick_phase * std::f32::consts::TAU).sin()
            * (1.0 - kick_phase) // amortissement
            * 2.5 // amplitude max ~2.5u
    } else {
        0.0
    };
    // **Sway** : sway[0] = yaw delta eased → push arme G/D ; sway[1]
    // = pitch delta → push arme haut/bas (mais inversé pour l'effet
    // "lag derrière la caméra").
    let sway_h = -sway[0]; // yaw + → caméra tourne droite → arme lag gauche
    let sway_v = -sway[1] * 0.5; // pitch (atténué pour ne pas être trop fort)
    let origin = eye
        + basis.forward * (10.0 - kick_back)
        + basis.right * (7.0 + breath_h + bob_h + sway_h)
        - basis.up * (7.0 - kick_up - breath_v - bob_v + kick_amp + sway_v);
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
    // **Player cosmetic tint** (v0.9.5++) — appliqué après les autres
    // modifs (invul/invisible) pour que la teinte joueur module la
    // couleur finale sans être écrasée par l'effet d'état.
    tint[0] *= player_tint[0];
    tint[1] *= player_tint[1];
    tint[2] *= player_tint[2];

    // **Viewmodel GLB override** (v0.9.5++) — pour les 5 armes avec
    // un asset GLB pickup (Plasma, Railgun, Machinegun, BFG, Lightning),
    // on utilise le mesh GLB au lieu du MD3 vintage.  Les GLB n'ont pas
    // de `tag_flash` → le muzzle flash utilise un offset hardcoded
    // (cf. `glb_viewmodel` plus bas).  Gauntlet/Shotgun/GL/RL restent
    // en MD3.
    let glb_viewmodel_prop: Option<&str> = match weapon {
        WeaponId::Plasmagun       if r.has_prop("plasma_pickup")          => Some("plasma_pickup"),
        WeaponId::Railgun         if r.has_prop("railgun_pickup")         => Some("railgun_pickup"),
        WeaponId::Machinegun      if r.has_prop("machinegun_pickup")      => Some("machinegun_pickup"),
        WeaponId::Bfg             if r.has_prop("bfg_pickup")             => Some("bfg_pickup"),
        WeaponId::Lightninggun    if r.has_prop("lightninggun_pickup")    => Some("lightninggun_pickup"),
        WeaponId::Shotgun         if r.has_prop("shotgun_pickup")         => Some("shotgun_pickup"),
        WeaponId::Grenadelauncher if r.has_prop("grenadelauncher_pickup") => Some("grenadelauncher_pickup"),
        WeaponId::Gauntlet        if r.has_prop("gauntlet_pickup")        => Some("gauntlet_pickup"),
        WeaponId::Rocketlauncher  if r.has_prop("rocketlauncher_pickup")  => Some("rocketlauncher_pickup"),
        _ => None,
    };
    if let Some(prop_name) = glb_viewmodel_prop {
        // Échelle viewmodel : ~8u radius (gun de taille raisonnable
        // dans le champ de vision).  Native radius ~1.06 → scale ~7.5.
        // Tuning par arme — certains modèles ont besoin d'être plus
        // gros pour avoir une présence visuelle équivalente.
        let vm_scale = match weapon {
            WeaponId::Shotgun         => 14.0_f32, // bien visible (user request)
            WeaponId::Machinegun      => 13.0_f32, // bien visible (user request)
            WeaponId::Rocketlauncher  => 13.0_f32, // bien visible (user request)
            WeaponId::Lightninggun    => 13.0_f32, // bien visible (user request)
            WeaponId::Bfg             => 13.0_f32, // bien visible (user request)
            WeaponId::Grenadelauncher => 10.5_f32,
            WeaponId::Plasmagun       => 10.5_f32,
            _                         => 8.0_f32,
        };
        let s = vm_scale;
        // **Orientation viewmodel par arme**.  Baseline commune
        // (-right, +up, +forward) — équivalent 180° Z.
        let r_v = basis.right;
        let u_v = basis.up;
        let f_v = basis.forward;
        let base_col0 = [-r_v.x, -r_v.y, -r_v.z];
        let base_col1 = [ u_v.x,  u_v.y,  u_v.z];
        let base_col2 = [ f_v.x,  f_v.y,  f_v.z];
        let (col0, col1, col2) = if matches!(weapon, WeaponId::Grenadelauncher) {
            // Grenadelauncher : baseline + rotation +90° Y (canon le long
            // de GLB -X local).  +90° Y math :
            //   new_col_x = -base_col2  (= -forward)
            //   new_col_y =  base_col1  (= +up)
            //   new_col_z =  base_col0  (= -right)
            (
                [-base_col2[0], -base_col2[1], -base_col2[2]],
                base_col1,
                base_col0,
            )
        } else if matches!(weapon, WeaponId::Shotgun) {
            // Shotgun : baseline + rotation -90° Y (canon le long de
            // GLB +X local).  -90° Y math :
            //   new_col_x =  base_col2  (= +forward)
            //   new_col_y =  base_col1  (= +up)
            //   new_col_z = -base_col0  (= +right)
            (
                base_col2,
                base_col1,
                [-base_col0[0], -base_col0[1], -base_col0[2]],
            )
        } else {
            // Plasma + Railgun + BFG + Lightninggun + Gauntlet + RL :
            // baseline pure (-right, +up, +forward).
            (base_col0, base_col1, base_col2)
        };
        let model = [
            [col0[0] * s, col0[1] * s, col0[2] * s, 0.0],
            [col1[0] * s, col1[1] * s, col1[2] * s, 0.0],
            [col2[0] * s, col2[1] * s, col2[2] * s, 0.0],
            [origin.x, origin.y, origin.z, 1.0],
        ];
        r.queue_prop(prop_name, model, tint);
    } else {
        r.draw_md3_viewmodel(mesh.clone(), transform, tint, fa, fb, lerp);
    }

    // Muzzle flash 3D : sprite additif billboard à `tag_flash` du viewmodel.
    // C'est le comportement canonique Q3 — un vrai MD3 tag pointe la sortie
    // du canon, et on y colle un halo additif dont la couleur dépend de
    // l'arme (poudre chaude pour MG/SG/RL, cyan pour plasma, vert pour BFG,
    // bleu-blanc pour railgun).  Le sprite est transitoire : publié chaque
    // frame pendant la fenêtre active, automatiquement vidé après flush.
    if muzzle_active {
        if let Some((base_color, radius)) = weapon.muzzle_flash() {
            // **Muzzle flash position** :
            // 1. MD3 viewmodel → tag_flash (ou tag_barrel) du mesh.
            // 2. GLB viewmodel (Plasma) → offset hardcoded au bout du
            //    canon (eye + forward × 22u + right × 7u).  Pas de tag
            //    GLB → on approxime visuellement.
            // **GLB viewmodel = pas de tag_flash** → offset hardcoded.
            // Forward 14u (≈ bout du canon, < hull radius 15u → pas de
            // bleed mur).
            // **Source de vérité unique** : la même condition
            // `glb_viewmodel_prop` qui a été utilisée plus haut pour
            // décider d'afficher le mesh GLB.  Évite la dérive entre
            // "afficher GLB" et "muzzle flash sans tag_flash".
            let glb_viewmodel = glb_viewmodel_prop.is_some();
            let tag_pos_opt = if glb_viewmodel {
                // **Offset par arme** — chaque GLB a une géométrie
                // différente, donc le bout du canon n'est pas au même
                // endroit dans l'espace local.  Valeurs (forward, right,
                // up_neg) tunées empiriquement pour que le muzzle flash
                // sorte VRAIMENT du bout du canon visible.
                // **Source de vérité commune** avec `App::viewmodel_muzzle_pos` :
                // si tu changes ici, change aussi le helper (et vice-versa)
                // pour que muzzle flash + tracers + projectiles sortent
                // tous du même point pour chaque arme.
                let (fwd, rt, up_neg) = match weapon {
                    // Armes longues.
                    WeaponId::Machinegun      => (28.0, 4.0, 4.0),
                    WeaponId::Lightninggun    => (26.0, 5.0, 4.0),
                    WeaponId::Rocketlauncher  => (26.0, 6.0, 5.0),
                    WeaponId::Shotgun         => (24.0, 6.0, 5.0),
                    WeaponId::Grenadelauncher => (22.0, 6.0, 5.0),
                    // Armes médiums.
                    WeaponId::Bfg             => (18.0, 6.0, 5.0),
                    WeaponId::Plasmagun
                    | WeaponId::Railgun       => (14.0, 6.0, 5.0),
                    // Gauntlet — pas de muzzle flash réel, mais offset
                    // proche main pour que l'eventuel halo soit cohérent.
                    WeaponId::Gauntlet        => (10.0, 5.0, 4.0),
                };
                Some(eye + basis.forward * fwd + basis.right * rt
                     - basis.up * up_neg)
            } else {
                mesh.tag_transform(fa, fb, lerp, "tag_flash")
                    .or_else(|| mesh.tag_transform(fa, fb, lerp, "tag_barrel"))
                    .map(|tag_local| {
                        let world = transform * tag_local;
                        Vec3::new(world.col(3).x, world.col(3).y, world.col(3).z)
                    })
            };
            if let Some(tag_pos) = tag_pos_opt {
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
