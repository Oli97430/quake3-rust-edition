//! Boucle client — étape 4 du netcode.
//!
//! Pilote la machine d'état [`q3_net::ClientHandshake`] jusqu'à
//! `Connected`, puis envoie périodiquement des `UserCmd` (à 60 Hz par
//! défaut) au serveur et reçoit ses snapshots.
//!
//! # Cycle de vie
//!
//! ```text
//! Disconnected ── start() ──► Challenging
//!                                 │ recv challengeResponse
//!                                 ▼
//!                              Connecting
//!                                 │ recv connectResponse
//!                                 ▼
//!                              Connected (NetChannel actif)
//!                                 │ ── envoie UserCmd ──►
//!                                 │ ◄── reçoit Snapshot
//!                                 ▼
//!                              (boucle stable)
//! ```
//!
//! # Snapshot exposé à App
//!
//! Le dernier snapshot reçu est mémorisé (`latest_snapshot`) et `App`
//! peut le retirer à chaque frame via [`ClientState::take_latest_snapshot`].
//! Si plusieurs snapshots arrivent entre deux frames, on garde le plus
//! récent et on jette les autres — l'interpolation de mouvement perçue
//! par le joueur est plus fine que 20 Hz, mais on n'a pas besoin du
//! détail historique pour appliquer l'état au monde local.
//!
//! # Ce qui n'est PAS fait en v1
//! - **Pas de prédiction client** : l'origine du joueur est écrasée par
//!   le snapshot serveur. Cela donne un input lag = RTT/2 + 50 ms (jitter
//!   buffer 20 Hz) — visible sur Internet, acceptable en LAN. La
//!   prédiction sera l'étape 5.
//! - **Pas d'interpolation** entre snapshots : le mouvement des autres
//!   joueurs téléporte au tick 20 Hz. Visible mais correct.
//! - **Pas de retransmission active** des UserCmd perdus : on en envoie
//!   plusieurs par paquet (`MAX_USERCMDS_PER_PACKET`), ce qui couvre
//!   la perte simple, mais une rafale de pertes provoque un trou.

use super::{Datagram, NetIo};
use q3_net::{
    ClientHandshake, ClientState as HandshakeState, ClientStep, NetChannel, Snapshot,
    SnapshotDelta, UserCmd,
};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::net::SocketAddr;
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Magic 4-byte identifiant pour le format demo `.q3rdm` (Quake 3 Rust DeMo).
/// Volontairement différent de `.dm_68` Q3 pour ne pas prétendre à une
/// compatibilité binaire qu'on n'a pas — le wire format est différent.
pub const DEMO_MAGIC: &[u8; 4] = b"Q3RD";
/// Version du format demo. Bump à chaque changement de structure record.
pub const DEMO_VERSION: u32 = 1;

/// Cadence d'envoi des `UserCmd` au serveur, alignée sur `cl_maxpackets`
/// historique de Q3 (60 par défaut). Plus haut → plus de bande passante
/// pour rien (le serveur ne tourne qu'à `sv_fps 20` autoritatif).
pub const USERCMD_HZ: f32 = 60.0;
const USERCMD_PERIOD: f32 = 1.0 / USERCMD_HZ;

/// Délai de retry pour le `getchallenge` initial. Si le serveur ne répond
/// pas (timeout réseau, paquet perdu) on relance toutes les 2 s — Q3
/// historique fait pareil dans `cl_main.c`.
const HANDSHAKE_RETRY_SEC: f32 = 2.0;

/// Snapshot d'input local fournit par l'engine à chaque tick. Subset de
/// la struct `Input` interne d'`App`, traduit en types réseau.
///
/// L'engine remplit ça depuis l'état clavier/souris du frame courant ;
/// `tick_client` quantifie + le packe dans une `UserCmd`.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalInput {
    /// Axe avant (+1) / arrière (-1).
    pub forward: f32,
    /// Axe strafe droit (+1) / gauche (-1).
    pub side: f32,
    /// Axe vertical (jump pad / debug fly). Toujours 0 en gameplay normal.
    pub up: f32,
    pub jump: bool,
    pub crouch: bool,
    pub walk: bool,
    pub fire: bool,
    pub use_holdable: bool,
    pub view_pitch: f32,
    pub view_yaw: f32,
    pub view_roll: f32,
    /// Slot d'arme actif (0..9). 2 = Machinegun par défaut.
    pub weapon: u8,
}

impl LocalInput {
    fn to_buttons(&self) -> u16 {
        let mut b = 0u16;
        if self.fire {
            b |= q3_net::buttons::FIRE;
        }
        if self.jump {
            b |= q3_net::buttons::JUMP;
        }
        if self.crouch {
            b |= q3_net::buttons::CROUCH;
        }
        if self.walk {
            b |= q3_net::buttons::WALK;
        }
        if self.use_holdable {
            b |= q3_net::buttons::USE_HOLDABLE;
        }
        b
    }
}

/// Sortie de [`ClientState::take_latest_snapshot`] : tout ce qu'App a
/// besoin pour réconcilier sa prédiction locale avec l'autorité serveur.
#[derive(Debug, Clone)]
pub struct PredictionInputs {
    pub snapshot: Snapshot,
    /// `UserCmd` envoyées au serveur mais pas encore acquittées par
    /// `snapshot.ack_cmd`. Doivent être rejouées localement après le
    /// rewind à l'état autoritatif.
    pub cmds_to_replay: Vec<UserCmd>,
}

/// Étape de connexion — agrège la machine d'état du handshake + les
/// transitions internes au client (déconnecté, en attente d'un snapshot).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientStage {
    /// Pas encore démarré le handshake — on émet `getchallenge` au prochain
    /// tick. L'engine peut être instancié avant qu'une connexion soit
    /// désirée, donc cet état n'est PAS une erreur.
    Idle,
    /// Handshake en cours.
    Handshaking,
    /// Handshake fini, on a un slot serveur. On envoie des `UserCmd` à
    /// 60 Hz et on reçoit des snapshots.
    Connected,
    /// Le serveur a refusé / fermé la connexion. On reste dans cet état
    /// jusqu'à un éventuel `start()` explicite (futur : commande console
    /// `reconnect`).
    Failed(String),
}

pub struct ClientState {
    pub server_addr: SocketAddr,
    pub io: Option<NetIo>,
    pub stage: ClientStage,
    /// Userinfo envoyé dans `connect` — contient au minimum `\name\Foo`.
    pub userinfo: String,
    handshake: ClientHandshake,
    channel: NetChannel,
    /// Numéro logique de la prochaine `UserCmd` à émettre. Démarre à 1
    /// (le serveur ignore `cmd_number == 0`, considéré « jamais reçu »).
    next_cmd_number: u32,
    /// Cmds émises mais pas encore acquittées (`cmd_number > server.ack_cmd`).
    /// On les garde pour permettre la prédiction (étape 5) et éviter qu'un
    /// paquet perdu n'introduise un trou : on retransmet jusqu'à
    /// `MAX_USERCMDS_PER_PACKET` cmds dans chaque envoi.
    pending_cmds: VecDeque<UserCmd>,
    /// Historique `(cmd_number, sent_at)` des dernières UserCmd envoyées
    /// — sert à mesurer le RTT quand un snapshot avec `ack_cmd >= n`
    /// arrive : `RTT = now - sent_at`. On garde 32 entrées pour couvrir
    /// la plupart des cas de réordonnancement / retransmission.
    send_times: VecDeque<(u32, Instant)>,
    /// Dernier RTT mesuré en millisecondes. `None` jusqu'au 1er échange
    /// complet. Lu par App pour le HUD ping.
    pub last_rtt_ms: Option<u32>,
    /// Dernier `server_time` reçu, ré-envoyé dans `ClientPacket::server_time_ack`.
    last_server_time: u32,
    /// Dernier ack du serveur — sert à purger `pending_cmds` et calculer
    /// le RTT pour le HUD.
    last_ack_cmd: u32,
    /// Slot du joueur dans `Snapshot::players` — fixé à la première
    /// snapshot reçue.
    pub my_slot: Option<u8>,
    /// Dernier snapshot non encore consommé par App. `None` après
    /// `take_latest_snapshot`.
    latest_snapshot: Option<Snapshot>,
    /// Dernière **baseline full** reçue. Sert à reconstruire les deltas
    /// envoyés ensuite par le serveur. Différent de `latest_snapshot` :
    /// ce dernier peut être un delta-reconstruit (post-apply), tandis
    /// que `last_full_baseline` est intouché. Le serveur publie une
    /// nouvelle baseline toutes les `FULL_SNAPSHOT_INTERVAL` ticks
    /// (cf. server.rs) — on l'écrase à chaque arrivée.
    last_full_baseline: Option<Snapshot>,
    /// Scheduler 60 Hz pour l'envoi des UserCmd.
    send_accum: f32,
    /// Time référence pour calculer `delta_ms` de chaque UserCmd. Passé
    /// à `Some(now)` après l'envoi de la première commande, restauré
    /// à `None` après une déconnexion.
    last_cmd_at: Option<Instant>,
    /// Time du dernier `getchallenge` émis — pour la stratégie de retry.
    last_handshake_send: Option<Instant>,
    pub packets_in: u64,
    pub packets_out: u64,
    /// Fichier de demo en cours d'écriture, si l'utilisateur a passé
    /// `--record <path>`. À chaque snapshot reçu (plein ou delta), on
    /// écrit un record `(server_time u32, payload_len u32, payload)`.
    /// Format trivial — un futur parser peut rejouer les snapshots
    /// pour debug / kill-cam.
    record_writer: Option<BufWriter<File>>,
    /// `Instant` de la première trame écrite — sert à calculer un
    /// timestamp relatif pour chaque record (`elapsed_ms`).
    record_start: Option<Instant>,
}

impl ClientState {
    pub fn new(server_addr: SocketAddr, io: NetIo, userinfo: String) -> Self {
        Self {
            server_addr,
            io: Some(io),
            stage: ClientStage::Idle,
            userinfo: userinfo.clone(),
            handshake: ClientHandshake::new(userinfo),
            channel: NetChannel::new(),
            next_cmd_number: 1,
            pending_cmds: VecDeque::new(),
            send_times: VecDeque::new(),
            last_rtt_ms: None,
            last_server_time: 0,
            last_ack_cmd: 0,
            my_slot: None,
            latest_snapshot: None,
            last_full_baseline: None,
            send_accum: 0.0,
            last_cmd_at: None,
            last_handshake_send: None,
            packets_in: 0,
            packets_out: 0,
            record_writer: None,
            record_start: None,
        }
    }

    /// Active l'enregistrement de démo dans le fichier `path`. Header
    /// écrit immédiatement (magic + version). Erreurs I/O loggées —
    /// pas de panic pour ne pas tuer le client à cause d'un disque plein.
    pub fn start_recording(&mut self, path: &Path) {
        let file = match File::create(path) {
            Ok(f) => f,
            Err(e) => {
                warn!(
                    "net/client: impossible d'ouvrir demo `{}` : {e}",
                    path.display()
                );
                return;
            }
        };
        let mut w = BufWriter::new(file);
        // Header : magic (4) + version (u32 LE) = 8 octets.
        if let Err(e) = w.write_all(DEMO_MAGIC).and_then(|_| {
            w.write_all(&DEMO_VERSION.to_le_bytes())
        }) {
            warn!("net/client: écriture header demo : {e}");
            return;
        }
        info!("net/client: enregistrement demo → {}", path.display());
        self.record_writer = Some(w);
        self.record_start = Some(Instant::now());
    }

    /// Écrit un record dans le fichier demo si l'enregistrement est actif.
    /// Format : `[u32 elapsed_ms][u32 payload_len][bytes]`. Échec d'I/O
    /// désactive l'enregistrement (logué une fois).
    fn write_demo_record(&mut self, payload: &[u8]) {
        if self.record_writer.is_none() {
            return;
        }
        let elapsed = self
            .record_start
            .map(|t| t.elapsed().as_millis().min(u32::MAX as u128) as u32)
            .unwrap_or(0);
        let len = payload.len() as u32;
        let writer = self.record_writer.as_mut().unwrap();
        let res = writer
            .write_all(&elapsed.to_le_bytes())
            .and_then(|_| writer.write_all(&len.to_le_bytes()))
            .and_then(|_| writer.write_all(payload));
        if let Err(e) = res {
            warn!("net/client: I/O demo, arrêt de l'enregistrement : {e}");
            self.record_writer = None;
        }
    }

    /// Envoie un message de chat via OOB `say "<msg>"` au serveur.
    /// No-op si pas connecté (le serveur drop les say sans slot).
    /// Le message est tronqué à 96 octets (limite ServerEvent::Chat).
    pub fn send_chat(&mut self, message: &str) {
        let msg = message.chars().take(96).collect::<String>();
        let oob = q3_net::OobMessage {
            command: "say".into(),
            // Quotes pour que le whitespace soit conservé.
            payload: format!("\"{msg}\"").into_bytes(),
        };
        if let Some(io) = self.io.as_ref() {
            io.send(self.server_addr, oob.to_bytes());
            self.packets_out += 1;
        }
    }

    /// Démarre le handshake — équivalent de `connect` en console Q3.
    /// Idempotent : si déjà en cours / connecté, no-op.
    pub fn start(&mut self) {
        if !matches!(self.stage, ClientStage::Idle | ClientStage::Failed(_)) {
            return;
        }
        info!("net/client: handshake → {}", self.server_addr);
        self.stage = ClientStage::Handshaking;
        let step = self.handshake.start();
        self.send_handshake_step(step);
    }

    /// Retire la dernière snapshot **et** la liste des `UserCmd` non
    /// acquittées pour qu'App puisse faire sa réconciliation
    /// rewind + replay :
    ///
    /// 1. Restaure l'état du joueur (origin, velocity, on_ground, crouching)
    ///    aux valeurs serveur dans le snapshot.
    /// 2. Rejoue chaque `UserCmd` de `cmds_to_replay` via `tick_collide`
    ///    pour avancer l'état local jusqu'à « où le serveur sera après
    ///    avoir consommé toutes nos cmds en vol ».
    ///
    /// Sans étape 2, le joueur snap-back à chaque snapshot (jitter visible).
    /// Sans étape 1, on a une dérive à long terme qui s'accumule.
    pub fn take_latest_snapshot(&mut self) -> Option<PredictionInputs> {
        let snapshot = self.latest_snapshot.take()?;
        // Les cmds acquittées par CE snapshot ont déjà été purgées
        // dans `on_snapshot` ; il reste donc uniquement celles qui
        // doivent être rejouées localement.
        let cmds_to_replay: Vec<UserCmd> = self.pending_cmds.iter().copied().collect();
        Some(PredictionInputs {
            snapshot,
            cmds_to_replay,
        })
    }

    /// Une frame client. `input` peut être `None` quand on n'est pas
    /// encore connecté ou que l'app n'a pas (encore) d'état d'input
    /// (menu ouvert, etc.) — on continue le handshake et on draine,
    /// mais on n'envoie pas d'UserCmd.
    pub fn tick(&mut self, dt_sec: f32, input: Option<&LocalInput>) {
        // 1. Drain des datagrammes entrants.
        let inbox: Vec<Datagram> = self
            .io
            .as_mut()
            .map(|io| io.drain_inbox())
            .unwrap_or_default();
        for dg in inbox {
            self.packets_in += 1;
            // Filtre paranoïaque : on ne croit que ce qui vient du serveur
            // ciblé. Sur Internet ça évite qu'un attaquant qui spoofe une
            // adresse aléatoire injecte des fausses snapshots.
            if dg.addr != self.server_addr {
                debug!(
                    "net/client: paquet d'un addr inattendu {} (attendu {}), drop",
                    dg.addr, self.server_addr
                );
                continue;
            }
            self.handle_inbound(&dg.bytes);
        }

        // 2. Auto-démarrage du handshake : la première frame avec un input
        //    valide bascule de Idle → Handshaking. Permet un usage simple
        //    (`NetRuntime::new + tick`) sans appel explicite à `start()`.
        if matches!(self.stage, ClientStage::Idle) {
            self.start();
        }

        // 3. Retry du handshake si pas de réponse au bout de
        //    HANDSHAKE_RETRY_SEC — paquet OOB perdu en chemin, par
        //    exemple. On ne retente que `Challenging`/`Connecting`,
        //    pas après succès ni après `Failed`.
        if matches!(self.stage, ClientStage::Handshaking) {
            let should_retry = match self.last_handshake_send {
                Some(t) => t.elapsed().as_secs_f32() >= HANDSHAKE_RETRY_SEC,
                None => false,
            };
            if should_retry {
                debug!(
                    "net/client: retry handshake (état {:?})",
                    self.handshake.state()
                );
                let step = match self.handshake.state() {
                    HandshakeState::Disconnected | HandshakeState::Challenging => {
                        // Reset propre + restart pour ré-émettre `getchallenge`.
                        self.handshake = ClientHandshake::new(self.userinfo.clone());
                        self.handshake.start()
                    }
                    HandshakeState::Connecting => {
                        // On dispose du challenge mais le serveur n'a pas
                        // confirmé — on relance le `connect`. Pour rester
                        // simple on redémarre depuis zéro (le serveur
                        // tolère plusieurs `getchallenge` du même addr).
                        self.handshake = ClientHandshake::new(self.userinfo.clone());
                        self.handshake.start()
                    }
                    HandshakeState::Connected => ClientStep::Idle,
                };
                self.send_handshake_step(step);
            }
        }

        // 4. Envoi de UserCmd quand on est connecté et qu'on a un input.
        if matches!(self.stage, ClientStage::Connected) {
            self.send_accum += dt_sec;
            // Sécurité : si l'engine hitch, on ne veut pas spammer 30 cmds
            // d'un coup. Cap à un max raisonnable.
            const MAX_BURST: u32 = 4;
            let mut bursts = 0;
            while self.send_accum >= USERCMD_PERIOD && bursts < MAX_BURST {
                self.send_accum -= USERCMD_PERIOD;
                bursts += 1;
                if let Some(input) = input {
                    self.send_one_usercmd(input);
                }
            }
            // Si on n'a PAS d'input (menu ouvert, console…) on ne reset
            // pas l'accumulateur — le client se met en pause d'envoi mais
            // dès que l'input revient on rattrape (sans dépasser MAX_BURST).
        }
    }

    // ---- Réception ----

    fn handle_inbound(&mut self, bytes: &[u8]) {
        if bytes.len() >= 4 && bytes[..4] == q3_net::OOB_MAGIC {
            // OOB : forward au handshake.
            let step = self.handshake.handle(bytes);
            self.advance_handshake(step);
            return;
        }
        // Connected packet.
        let payload = match self.channel.process_incoming(bytes) {
            Ok(Some(p)) => p,
            Ok(None) => return,
            Err(e) => {
                warn!("net/client: NetChannel: {e}");
                return;
            }
        };
        if payload.is_empty() {
            return;
        }
        // Enregistre le payload brut (incl. tag) pour le replay demo.
        // Indépendant du décodage : si on n'a pas le bon format on
        // n'écrit rien, mais snapshots OK et deltas OK passent.
        if matches!(payload[0], q3_net::TAG_SNAPSHOT | q3_net::TAG_SNAPSHOT_DELTA) {
            let payload_clone = payload.clone();
            self.write_demo_record(&payload_clone);
        }
        match payload[0] {
            q3_net::TAG_SNAPSHOT => match Snapshot::decode(&payload) {
                Ok(snap) => {
                    // Tout snapshot full devient la nouvelle baseline.
                    self.last_full_baseline = Some(snap.clone());
                    self.on_snapshot(snap);
                }
                Err(e) => warn!("net/client: snapshot malformé: {e}"),
            },
            q3_net::TAG_SNAPSHOT_DELTA => match SnapshotDelta::decode(&payload) {
                Ok(delta) => {
                    let Some(baseline) = self.last_full_baseline.as_ref() else {
                        // Pas de baseline → on jette le delta. Le serveur
                        // renvoie un full toutes les ~1 s donc on n'attend
                        // pas longtemps. Logue en debug pour diagnostiquer
                        // un éventuel souci de séquence d'apparition.
                        debug!(
                            "net/client: delta reçu (server_time={}) mais pas de baseline, drop",
                            delta.server_time
                        );
                        return;
                    };
                    if delta.baseline_server_time != baseline.server_time {
                        // Désynchro : le serveur croit qu'on a une autre
                        // baseline. On attend le prochain full (cycle 1 s).
                        debug!(
                            "net/client: delta baseline_time={} ≠ notre full {}, drop",
                            delta.baseline_server_time, baseline.server_time
                        );
                        return;
                    }
                    let snap = delta.apply_to_baseline(baseline);
                    self.on_snapshot(snap);
                }
                Err(e) => warn!("net/client: SnapshotDelta malformé: {e}"),
            },
            tag => debug!("net/client: tag {tag} inconnu (drop)"),
        }
    }

    fn on_snapshot(&mut self, snap: Snapshot) {
        // Première snapshot : on enregistre notre slot pour le rendu.
        if self.my_slot.is_none() {
            self.my_slot = Some(snap.client_slot);
            info!(
                "net/client: snapshot initial reçu (slot={}, server_time={} ms)",
                snap.client_slot, snap.server_time
            );
        }
        self.last_server_time = snap.server_time;
        self.last_ack_cmd = snap.ack_cmd;
        // Mesure du RTT : on cherche dans `send_times` la cmd correspondant
        // au plus récent `cmd_number <= ack_cmd`. RTT = now - sent_at.
        // On itère depuis la fin pour trouver l'entrée la plus récente
        // ackée — donne une mesure RTT pertinente même sous perte de
        // paquets (ack peut sauter plusieurs cmds).
        if let Some(idx) = self
            .send_times
            .iter()
            .rposition(|(n, _)| *n <= snap.ack_cmd)
        {
            let (_, sent_at) = self.send_times[idx];
            self.last_rtt_ms = Some(sent_at.elapsed().as_millis().min(u32::MAX as u128) as u32);
            // Purge les anciennes entrées — pas la peine de garder l'histo
            // de cmds déjà acquittées au-delà de la dernière en attente.
            for _ in 0..=idx {
                self.send_times.pop_front();
            }
        }
        // Purge des cmds acquittées par le serveur. On garde celles
        // > ack_cmd pour la rétention en cas de paquet suivant perdu et
        // pour la prédiction.
        while let Some(front) = self.pending_cmds.front() {
            if front.cmd_number <= snap.ack_cmd {
                self.pending_cmds.pop_front();
            } else {
                break;
            }
        }
        self.latest_snapshot = Some(snap);
    }

    // ---- Émission ----

    fn advance_handshake(&mut self, step: ClientStep) {
        match step {
            ClientStep::Idle => {}
            ClientStep::Send(bytes) => {
                self.send_handshake_step(ClientStep::Send(bytes));
            }
            ClientStep::Done { challenge } => {
                info!(
                    "net/client: connecté à {} (challenge={})",
                    self.server_addr, challenge
                );
                self.stage = ClientStage::Connected;
                self.last_handshake_send = None;
                // Reset le NetChannel : nouvelles séquences depuis la
                // connexion, pas l'historique du précédent (s'il y en avait).
                self.channel = NetChannel::new();
            }
            ClientStep::Failed(reason) => {
                warn!("net/client: handshake échoué : {reason}");
                self.stage = ClientStage::Failed(reason);
            }
        }
    }

    fn send_handshake_step(&mut self, step: ClientStep) {
        match step {
            ClientStep::Send(bytes) => {
                if let Some(io) = self.io.as_ref() {
                    io.send(self.server_addr, bytes);
                    self.packets_out += 1;
                    self.last_handshake_send = Some(Instant::now());
                }
            }
            ClientStep::Done { challenge } => {
                info!(
                    "net/client: connecté à {} (challenge={})",
                    self.server_addr, challenge
                );
                self.stage = ClientStage::Connected;
            }
            ClientStep::Failed(reason) => {
                self.stage = ClientStage::Failed(reason);
            }
            ClientStep::Idle => {}
        }
    }

    fn send_one_usercmd(&mut self, input: &LocalInput) {
        let now = Instant::now();
        let delta_ms = match self.last_cmd_at {
            Some(t) => {
                let ms = (now - t).as_millis();
                ms.min(255) as u8
            }
            None => 0, // première commande — le serveur fallback à 8 ms
        };
        self.last_cmd_at = Some(now);

        let cmd = UserCmd {
            cmd_number: self.next_cmd_number,
            forward: UserCmd::quantize_axis(input.forward),
            side: UserCmd::quantize_axis(input.side),
            up: UserCmd::quantize_axis(input.up),
            buttons: input.to_buttons(),
            view_pitch: UserCmd::quantize_angle(input.view_pitch),
            view_yaw: UserCmd::quantize_angle(input.view_yaw),
            view_roll: UserCmd::quantize_angle(input.view_roll),
            delta_ms,
            weapon: input.weapon,
        };
        self.next_cmd_number = self.next_cmd_number.wrapping_add(1);
        self.pending_cmds.push_back(cmd);
        // Track send time pour mesure RTT au prochain ack.
        self.send_times.push_back((cmd.cmd_number, now));
        while self.send_times.len() > 32 {
            self.send_times.pop_front();
        }

        // Cap la file : on n'envoie de toute façon que les
        // MAX_USERCMDS_PER_PACKET dernières — au-delà c'est de la mémoire
        // gaspillée et il y a peu de chance qu'une cmd vraiment vieille
        // soit utile (le serveur a déjà avancé sans).
        while self.pending_cmds.len() > q3_net::MAX_USERCMDS_PER_PACKET {
            self.pending_cmds.pop_front();
        }

        // Pack + send. On envoie le batch entier (jusqu'à 16 cmds), pas
        // seulement la dernière : ça absorbe une perte simple sans hitch
        // côté serveur.
        let pkt = q3_net::ClientPacket {
            server_time_ack: self.last_server_time,
            cmds: self.pending_cmds.iter().copied().collect(),
        };
        let bytes = match pkt.encode() {
            Ok(b) => b,
            Err(e) => {
                warn!("net/client: encode ClientPacket: {e}");
                return;
            }
        };
        let frags = self.channel.prepare_outgoing(&bytes);
        for frag in frags {
            if let Some(io) = self.io.as_ref() {
                io.send(self.server_addr, frag);
                self.packets_out += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::NetIo;

    #[test]
    fn local_input_to_buttons_combines_flags() {
        let mut i = LocalInput::default();
        assert_eq!(i.to_buttons(), 0);
        i.fire = true;
        i.jump = true;
        let b = i.to_buttons();
        assert_eq!(b & q3_net::buttons::FIRE, q3_net::buttons::FIRE);
        assert_eq!(b & q3_net::buttons::JUMP, q3_net::buttons::JUMP);
        assert_eq!(b & q3_net::buttons::CROUCH, 0);
    }

    /// Vérifie que `take_latest_snapshot` ne renvoie que les `UserCmd`
    /// non acquittées par le snapshot — la base de la prédiction.
    /// On simule manuellement la file `pending_cmds` + un snapshot avec
    /// un `ack_cmd` choisi, puis on vérifie le filtrage.
    #[test]
    fn prediction_inputs_filter_acked_cmds() {
        // ClientState a besoin d'un NetIo, mais on peut bind éphémère
        // sur localhost — seul l'état logique nous intéresse.
        let io = NetIo::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let server_addr = io.local_addr();
        let mut state = ClientState::new(server_addr, io, "\\name\\Test".into());

        // 5 cmds en vol (numéros 1..=5).
        for i in 1..=5u32 {
            state.pending_cmds.push_back(UserCmd {
                cmd_number: i,
                ..Default::default()
            });
        }

        // Snapshot acquittant les cmds 1..=3 (2 et 4-5 restent en vol).
        let snap = q3_net::Snapshot {
            server_time: 1,
            ack_cmd: 3,
            client_slot: 0,
            players: vec![],
            entities: vec![],
            pickups: vec![],
            events: vec![],
            players_info: vec![],
        };
        state.on_snapshot(snap);

        let p = state.take_latest_snapshot().expect("snapshot pris");
        assert_eq!(p.snapshot.ack_cmd, 3);
        // Doit rester [4, 5] — purge inclusive de cmds <= ack_cmd.
        let nums: Vec<u32> = p.cmds_to_replay.iter().map(|c| c.cmd_number).collect();
        assert_eq!(nums, vec![4, 5]);
    }

    /// `last_rtt_ms` est calculé sur l'ack d'une cmd dont on a tracké
    /// le `sent_at`. On simule en pushant manuellement send_times +
    /// déclenchant un on_snapshot avec un ack >= des entrées trackées.
    #[test]
    fn rtt_measured_on_ack() {
        let io = NetIo::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let server_addr = io.local_addr();
        let mut state = ClientState::new(server_addr, io, "\\name\\T".into());

        // Simule l'envoi de 3 cmds ~50 ms dans le passé.
        let past = Instant::now() - std::time::Duration::from_millis(50);
        state.send_times.push_back((1, past));
        state.send_times.push_back((2, past));
        state.send_times.push_back((3, past));

        state.on_snapshot(q3_net::Snapshot {
            server_time: 1,
            ack_cmd: 3, // server a appliqué jusqu'à cmd 3
            client_slot: 0,
            players: vec![],
            entities: vec![],
            pickups: vec![],
            events: vec![],
            players_info: vec![],
        });

        let rtt = state.last_rtt_ms.expect("RTT mesuré");
        assert!(
            (40..=200).contains(&rtt),
            "RTT attendu ~50ms, obtenu {rtt}ms"
        );
        // send_times purgées (toutes ackées).
        assert!(state.send_times.is_empty());
    }

    /// L'enregistrement démo écrit le header magique au démarrage,
    /// puis un record (timestamp + len + payload) à chaque snapshot
    /// reçu via `handle_inbound`. On vérifie le header et la présence
    /// d'au moins un record après injection d'un faux snapshot.
    #[test]
    fn demo_record_writes_header_and_record() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "q3rust-demo-test-{}.q3rdm",
            std::process::id()
        ));

        // Préparer un client avec recording actif.
        let io = NetIo::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let server_addr = io.local_addr();
        let mut state = ClientState::new(server_addr, io, "\\name\\T".into());
        state.start_recording(&path);

        // Construit un payload snapshot minimal (encode + transmis).
        let snap = q3_net::Snapshot {
            server_time: 1234,
            ack_cmd: 0,
            client_slot: 0,
            players: vec![],
            entities: vec![],
            pickups: vec![],
            events: vec![],
            players_info: vec![],
        };
        let payload = snap.encode().unwrap();
        // Simule un paquet entrant via NetChannel pour passer dans le
        // pipeline normal (le NetChannel ajoute son header séquence).
        let mut tx_chan = q3_net::NetChannel::new();
        let raw_packets = tx_chan.prepare_outgoing(&payload);
        for p in &raw_packets {
            state.handle_inbound(p);
        }

        // Force le flush du writer (Drop sur BufWriter le fera mais
        // on veut lire tout de suite).
        state.record_writer = None;

        let bytes = std::fs::read(&path).expect("demo file lisible");
        assert_eq!(&bytes[..4], DEMO_MAGIC, "header magic");
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, DEMO_VERSION);
        // Au moins un record (8 octets header + 4 + 4 + payload).
        assert!(bytes.len() > 8 + 8);
        // Cleanup.
        let _ = std::fs::remove_file(&path);
    }

    /// Un second snapshot avec un ack plus avancé doit purger encore
    /// plus — vérifie que la file est cumulativement réduite.
    #[test]
    fn prediction_inputs_cumulative_acks() {
        let io = NetIo::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let server_addr = io.local_addr();
        let mut state = ClientState::new(server_addr, io, "\\name\\Test".into());

        for i in 1..=10u32 {
            state.pending_cmds.push_back(UserCmd {
                cmd_number: i,
                ..Default::default()
            });
        }

        // 1er snapshot ack=4 → purge 1..=4, replay [5..=10]
        state.on_snapshot(q3_net::Snapshot {
            server_time: 1,
            ack_cmd: 4,
            client_slot: 0,
            players: vec![],
            entities: vec![],
            pickups: vec![],
            events: vec![],
            players_info: vec![],
        });
        let p = state.take_latest_snapshot().unwrap();
        let nums: Vec<u32> = p.cmds_to_replay.iter().map(|c| c.cmd_number).collect();
        assert_eq!(nums, vec![5, 6, 7, 8, 9, 10]);

        // 2e snapshot ack=8 → purge 5..=8, replay [9, 10]
        state.on_snapshot(q3_net::Snapshot {
            server_time: 2,
            ack_cmd: 8,
            client_slot: 0,
            players: vec![],
            entities: vec![],
            pickups: vec![],
            events: vec![],
            players_info: vec![],
        });
        let p = state.take_latest_snapshot().unwrap();
        let nums: Vec<u32> = p.cmds_to_replay.iter().map(|c| c.cmd_number).collect();
        assert_eq!(nums, vec![9, 10]);
    }
}
