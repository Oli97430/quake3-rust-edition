//! Intégration réseau côté engine — runtime tokio + boucle I/O UDP.
//!
//! Le transport bas-niveau (UDP socket, parsing OOB, NetChannel, handshake,
//! format wire) vit dans la crate `q3-net`. Ce module-ci :
//!
//! 1. Choisit le mode réseau depuis la CLI (`--host` / `--connect`).
//! 2. Possède un runtime tokio + une tâche I/O qui lit / écrit sur le
//!    socket de façon asynchrone, et la relie au thread principal de
//!    l'engine via deux `mpsc::UnboundedChannel` (inbox / outbox).
//! 3. Expose `tick_server` / `tick_client` appelés une fois par frame
//!    par `App` — c'est là que sera branchée la logique applicative
//!    (handshake, snapshots, prédiction) dans les étapes suivantes.
//!
//! # Architecture
//!
//! ```text
//!  Engine main thread (sync, winit)        Tokio worker thread (async)
//!  ─────────────────────────────────       ───────────────────────────
//!     App::tick_*(dt)                      io_task(select!) :
//!       │                                    ├─ recv_from socket
//!       │  ┌──── inbox.try_recv ────────────►│       └► inbox.send
//!       │  │                                  │
//!       │  └──── outbox.send ─────────────────► outbox.recv
//!       │                                              └► socket.send_to
//!       └──────► drain_events / draw HUD
//! ```
//!
//! Le runtime est un `multi_thread` à 1 worker : pas de surcoût de
//! synchronisation entre workers, mais il tourne de façon autonome (pas
//! besoin que l'engine `block_on` sur une racine future). Le `Drop` envoie
//! le signal de shutdown à la tâche I/O et coupe proprement.

pub mod client;
pub mod demo;
pub mod server;

pub use client::{ClientStage, ClientState, LocalInput, PredictionInputs};
pub use server::{GameType, ServerState};

use q3_game::World;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::UdpSocket;
use tokio::runtime::Runtime;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

/// Mode de l'application réseau. Sélectionné au démarrage par les
/// flags CLI ou la commande console `connect <addr>` / `map <mapname>`.
#[derive(Debug, Clone)]
pub enum NetMode {
    /// Mode solo — aucun socket, aucun tick réseau. Les bots locaux
    /// remplissent l'arène. C'est le mode historique de ce port.
    SinglePlayer,

    /// Serveur autoritatif écoutant sur `bind_addr`. Reçoit les
    /// `usercmd` des clients, fait tourner la simulation à 20 Hz, et
    /// broadcast les snapshots.
    Server {
        bind_addr: SocketAddr,
        max_clients: u8,
        gametype: GameType,
        friendly_fire: bool,
    },

    /// Client connecté à `server_addr`. Envoie ses `usercmd`, reçoit
    /// les snapshots, fait la prédiction côté local pour masquer la
    /// latence.
    Client { server_addr: SocketAddr },

    /// Lecture d'une démo `.q3rdm` enregistrée précédemment. Aucun
    /// socket : les snapshots sont injectés depuis le fichier au
    /// rythme original (timestamps `elapsed_ms`).
    DemoPlayback { path: std::path::PathBuf },
}

impl NetMode {
    /// Construit le mode depuis les flags CLI. Priorité : `--connect`
    /// > `--host` > solo. Une erreur de parsing revient en solo avec
    /// un warning plutôt que planter — UX : « si ton adresse est
    /// cassée, tu peux au moins jouer en solo ».
    pub fn from_cli(host: Option<&str>, connect: Option<&str>, max_clients: u8) -> Self {
        Self::from_cli_full(host, connect, max_clients, GameType::FreeForAll, true)
    }

    /// Variante complète : permet aussi de choisir le gametype et l'état
    /// de friendly-fire pour un `--host`. Côté `--connect` ces valeurs
    /// sont ignorées (c'est le serveur distant qui décide).
    pub fn from_cli_full(
        host: Option<&str>,
        connect: Option<&str>,
        max_clients: u8,
        gametype: GameType,
        friendly_fire: bool,
    ) -> Self {
        if let Some(addr) = connect {
            match addr.parse::<SocketAddr>() {
                Ok(server_addr) => {
                    info!("net: mode CLIENT → {server_addr}");
                    return NetMode::Client { server_addr };
                }
                Err(e) => warn!("net: --connect '{addr}' invalide ({e}), fallback solo"),
            }
        }
        if let Some(bind) = host {
            match bind.parse::<SocketAddr>() {
                Ok(bind_addr) => {
                    info!(
                        "net: mode SERVER @ {bind_addr} (max {max_clients} clients, {:?}, ff={})",
                        gametype, friendly_fire
                    );
                    return NetMode::Server {
                        bind_addr,
                        max_clients,
                        gametype,
                        friendly_fire,
                    };
                }
                Err(e) => warn!("net: --host '{bind}' invalide ({e}), fallback solo"),
            }
        }
        NetMode::SinglePlayer
    }

    pub fn is_networked(&self) -> bool {
        !matches!(self, NetMode::SinglePlayer)
    }

    pub fn is_server(&self) -> bool {
        matches!(self, NetMode::Server { .. })
    }

    pub fn is_client(&self) -> bool {
        matches!(self, NetMode::Client { .. } | NetMode::DemoPlayback { .. })
    }

    pub fn is_demo(&self) -> bool {
        matches!(self, NetMode::DemoPlayback { .. })
    }
}

impl Default for NetMode {
    fn default() -> Self {
        NetMode::SinglePlayer
    }
}

// ---------------------------------------------------------------------------
// NetIo — pont sync/async pour le socket UDP
// ---------------------------------------------------------------------------

/// Datagramme entrant ou sortant, avec son adresse de pair.
#[derive(Debug)]
pub struct Datagram {
    pub addr: SocketAddr,
    pub bytes: Vec<u8>,
}

/// Pont I/O entre le thread engine (sync) et le worker tokio (async).
///
/// Chaque `NetIo` possède son propre runtime tokio à 1 worker — c'est
/// largement suffisant pour 1 socket UDP, et ça évite d'avoir à partager
/// un runtime global avec le reste de l'app.
///
/// La tâche I/O est `select!`-bouclée et survit jusqu'au `Drop` de ce
/// `NetIo` (signal de shutdown via `oneshot`).
pub struct NetIo {
    /// On garde le runtime vivant tant que ce `NetIo` existe — sa
    /// destruction côté `Drop` arrête proprement la tâche I/O.
    _runtime: Runtime,
    inbox: mpsc::UnboundedReceiver<Datagram>,
    outbox: mpsc::UnboundedSender<Datagram>,
    local_addr: SocketAddr,
    /// `Some` jusqu'au `Drop`, qui consomme le sender pour signaler
    /// l'arrêt à la tâche I/O.
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl NetIo {
    /// Bind un socket UDP sur `addr` (`0.0.0.0:0` pour client éphémère).
    /// Démarre la tâche I/O et retourne l'adresse locale effective.
    pub fn bind(addr: SocketAddr) -> Result<Self, q3_common::Error> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .thread_name("q3-net-io")
            .build()
            .map_err(|e| q3_common::Error::Network(format!("tokio runtime: {e}")))?;

        let (inbox_tx, inbox_rx) = mpsc::unbounded_channel::<Datagram>();
        let (outbox_tx, outbox_rx) = mpsc::unbounded_channel::<Datagram>();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        // On bind dans le runtime puis on `spawn` la tâche I/O. Le
        // `block_on` ne dure que le bind (rapide, < 1 ms) — ensuite
        // tout est asynchrone sur le worker.
        let local_addr = runtime.block_on(async move {
            let socket = UdpSocket::bind(addr).await.map_err(|e| {
                q3_common::Error::Network(format!("UDP bind {addr}: {e}"))
            })?;
            let local = socket
                .local_addr()
                .map_err(|e| q3_common::Error::Network(format!("local_addr: {e}")))?;
            let socket = Arc::new(socket);
            tokio::spawn(io_task(socket, inbox_tx, outbox_rx, shutdown_rx));
            Ok::<_, q3_common::Error>(local)
        })?;

        info!("net: socket UDP bind {} OK", local_addr);

        Ok(Self {
            _runtime: runtime,
            inbox: inbox_rx,
            outbox: outbox_tx,
            local_addr,
            shutdown_tx: Some(shutdown_tx),
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Drain non-blocant — récupère tous les datagrammes accumulés
    /// depuis le dernier appel. Utilisable depuis le thread engine sync.
    pub fn drain_inbox(&mut self) -> Vec<Datagram> {
        let mut out = Vec::new();
        // try_recv() est défini sur `tokio::sync::mpsc::UnboundedReceiver`
        // et n'est pas bloquant — sûr depuis du code sync.
        while let Ok(d) = self.inbox.try_recv() {
            out.push(d);
        }
        out
    }

    /// Pousse un datagramme vers la tâche I/O. Ne bloque pas — la queue
    /// est unbounded. En cas d'erreur (tâche I/O morte), on logue et on
    /// drop : pas de panic pour ne pas tuer l'app sur un blip réseau.
    pub fn send(&self, addr: SocketAddr, bytes: Vec<u8>) {
        if let Err(e) = self.outbox.send(Datagram { addr, bytes }) {
            warn!("net: outbox.send failed (tâche I/O morte ?): {e}");
        }
    }
}

impl Drop for NetIo {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            // Erreur ignorée : la tâche peut déjà être morte (panic), c'est OK.
            let _ = tx.send(());
        }
    }
}

/// Tâche I/O asynchrone — boucle `select!` qui multiplexe :
///   1. shutdown : sortie propre quand le `NetIo` est `Drop`pé
///   2. recv_from : tout datagramme arrivé est forward au inbox
///   3. outbox.recv : tout datagramme sortant est envoyé sur le socket
///
/// La priorité (`biased`) place shutdown en tête pour garantir une sortie
/// rapide même si le socket est saturé. recv vient avant send — un client
/// qui spam ne peut pas affamer la lecture.
async fn io_task(
    socket: Arc<UdpSocket>,
    inbox: mpsc::UnboundedSender<Datagram>,
    mut outbox: mpsc::UnboundedReceiver<Datagram>,
    mut shutdown: oneshot::Receiver<()>,
) {
    // Tampon de réception réutilisé. 2048 = au-dessus de la MTU Ethernet
    // standard (1500), couvre confortablement un fragment Q3 (1400 max
    // côté NetChannel). Pas de raison d'aller plus haut tant qu'on ne
    // reçoit pas des paquets jumbo en LAN.
    let mut buf = vec![0u8; 2048];
    loop {
        tokio::select! {
            biased;
            _ = &mut shutdown => {
                debug!("net: tâche I/O reçoit shutdown, arrêt");
                break;
            }
            res = socket.recv_from(&mut buf) => {
                match res {
                    Ok((n, addr)) => {
                        let dg = Datagram { addr, bytes: buf[..n].to_vec() };
                        if inbox.send(dg).is_err() {
                            // Receveur drop, l'engine est en train de mourir.
                            break;
                        }
                    }
                    Err(e) => {
                        // ConnectionReset (Windows) arrive quand un peer
                        // disparaît — on ne veut pas tuer le serveur pour ça.
                        warn!("net: recv_from: {e}");
                    }
                }
            }
            Some(dg) = outbox.recv() => {
                if let Err(e) = socket.send_to(&dg.bytes, dg.addr).await {
                    warn!("net: send_to {}: {e}", dg.addr);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NetRuntime — état réseau rattaché à App
// ---------------------------------------------------------------------------

pub struct NetRuntime {
    pub mode: NetMode,
    inner: Inner,
    /// Évènements à drainer côté HUD / console.
    events: Vec<NetEvent>,
}

enum Inner {
    None,
    Server(ServerState),
    Client(ClientState),
    Demo(demo::DemoPlayer, std::collections::VecDeque<client::PredictionInputs>),
}

impl NetRuntime {
    /// Initialise le runtime selon le mode. Le bind du socket a lieu
    /// **immédiatement** (synchrone, < 1 ms typiquement) — si ça échoue
    /// on dégrade en `SinglePlayer` plutôt que de planter le binaire.
    pub fn new(mode: NetMode) -> Self {
        Self::new_with_userinfo(mode, "\\name\\Player\\rate\\25000".into())
    }

    /// Variante avec un userinfo custom — utile pour passer
    /// `\spectator\1` ou un nom de joueur côté client.
    pub fn new_with_userinfo(mode: NetMode, userinfo: String) -> Self {
        let (mode, inner) = match mode {
            NetMode::SinglePlayer => (NetMode::SinglePlayer, Inner::None),
            NetMode::Server { bind_addr, max_clients, gametype, friendly_fire } => match NetIo::bind(bind_addr) {
                Ok(io) => {
                    info!(
                        "net: serveur prêt, accepte jusqu'à {max_clients} clients sur {} ({:?}, ff={})",
                        io.local_addr(), gametype, friendly_fire
                    );
                    (
                        NetMode::Server { bind_addr, max_clients, gametype, friendly_fire },
                        Inner::Server(ServerState::new_with_config(
                            bind_addr, max_clients, io, gametype, friendly_fire,
                        )),
                    )
                }
                Err(e) => {
                    warn!("net: bind serveur {bind_addr} a échoué ({e}), fallback solo");
                    (NetMode::SinglePlayer, Inner::None)
                }
            },
            NetMode::DemoPlayback { ref path } => {
                match demo::DemoPlayer::open(path) {
                    Ok(player) => {
                        info!(
                            "net: lecture démo `{}` ({} records)",
                            path.display(),
                            player.record_count()
                        );
                        (
                            NetMode::DemoPlayback { path: path.clone() },
                            Inner::Demo(player, std::collections::VecDeque::new()),
                        )
                    }
                    Err(e) => {
                        warn!("net: ouverture démo échouée ({e}), fallback solo");
                        (NetMode::SinglePlayer, Inner::None)
                    }
                }
            }
            NetMode::Client { server_addr } => {
                // Bind éphémère côté client : l'OS choisit un port libre.
                // On utilise une adresse de bind compatible avec la famille
                // (v4 ou v6) du serveur ciblé.
                let bind: SocketAddr = if server_addr.is_ipv6() {
                    "[::]:0".parse().unwrap()
                } else {
                    "0.0.0.0:0".parse().unwrap()
                };
                match NetIo::bind(bind) {
                    Ok(io) => {
                        info!(
                            "net: client prêt sur {} → cible {}",
                            io.local_addr(),
                            server_addr
                        );
                        (
                            NetMode::Client { server_addr },
                            Inner::Client(ClientState::new(server_addr, io, userinfo.clone())),
                        )
                    }
                    Err(e) => {
                        warn!("net: bind client a échoué ({e}), fallback solo");
                        (NetMode::SinglePlayer, Inner::None)
                    }
                }
            }
        };
        Self {
            mode,
            inner,
            events: Vec::new(),
        }
    }

    /// Statistiques compteurs (in, out) — pour overlay debug. `None` en solo.
    pub fn packet_counters(&self) -> Option<(u64, u64)> {
        match &self.inner {
            Inner::None => None,
            Inner::Server(s) => Some((s.packets_in, s.packets_out)),
            Inner::Client(c) => Some((c.packets_in, c.packets_out)),
            Inner::Demo(..) => None,
        }
    }

    /// Une frame serveur — délègue à [`ServerState::tick`]. `world` est
    /// `None` tant qu'aucune map n'est chargée : dans ce cas on draine
    /// quand même la queue tokio (sinon elle saturerait sous spam) mais
    /// on n'avance pas la simulation.
    pub fn tick_server(&mut self, dt_sec: f32, world: Option<&World>) {
        let Inner::Server(state) = &mut self.inner else {
            return;
        };
        state.tick(dt_sec, world);
    }

    /// Une frame client. `input` peut être `None` (menu / console / pas
    /// encore d'input traité) — dans ce cas on continue le handshake et
    /// on draine, mais on n'envoie pas d'UserCmd.
    pub fn tick_client(&mut self, dt_sec: f32, input: Option<&LocalInput>) {
        match &mut self.inner {
            Inner::Client(state) => state.tick(dt_sec, input),
            Inner::Demo(player, queue) => {
                let mut produced = Vec::new();
                player.tick(&mut produced);
                for p in produced {
                    queue.push_back(p);
                }
                let _ = dt_sec;
                let _ = input;
            }
            _ => {}
        }
    }

    /// Retire le dernier snapshot reçu côté client — l'engine l'applique
    /// à `App::player` / aux remote players. `None` si aucune nouvelle
    /// snapshot n'est arrivée depuis le dernier appel. La struct retournée
    /// inclut aussi les `UserCmd` à rejouer pour la prédiction.
    pub fn take_client_snapshot(&mut self) -> Option<PredictionInputs> {
        match &mut self.inner {
            Inner::Client(c) => c.take_latest_snapshot(),
            Inner::Demo(_, queue) => queue.pop_front(),
            _ => None,
        }
    }

    /// `true` si la démo en cours de lecture a tout joué (utile pour
    /// déclencher un message de fin côté HUD).
    pub fn demo_finished(&self) -> bool {
        matches!(&self.inner, Inner::Demo(p, q) if p.is_finished() && q.is_empty())
    }

    /// Étape de la connexion client — utile pour le HUD (« Connecting… »).
    pub fn client_stage(&self) -> Option<&ClientStage> {
        match &self.inner {
            Inner::Client(c) => Some(&c.stage),
            _ => None,
        }
    }

    /// RTT mesuré en mode client (ms). `None` en solo / serveur, ou
    /// avant la première mesure (1er round-trip cmd → snap ack).
    pub fn client_rtt_ms(&self) -> Option<u32> {
        match &self.inner {
            Inner::Client(c) => c.last_rtt_ms,
            _ => None,
        }
    }

    /// Ajoute un bot serveur. No-op si pas en mode `Server` (`None`
    /// retourné). Le bot apparaît dans les snapshots à la prochaine
    /// frame avec le flag `BOT` set.
    pub fn add_server_bot(
        &mut self,
        name: String,
        skill: q3_bot::BotSkill,
    ) -> Option<u8> {
        match &mut self.inner {
            Inner::Server(state) => {
                // World pas encore chargé ici (App::new) — le bot
                // démarre à Vec3::ZERO, sera repositionné au 1er
                // respawn / frame avec world dispo.
                state.add_bot(name, skill, None)
            }
            _ => None,
        }
    }

    /// Kick un slot (humain ou bot) côté serveur. Retourne `true` si
    /// trouvé/retiré, `false` sinon. No-op si pas en mode Server.
    pub fn kick_slot(&mut self, slot_id: u8) -> bool {
        match &mut self.inner {
            Inner::Server(state) => state.kick_slot(slot_id),
            _ => false,
        }
    }

    /// Force le restart du match côté serveur — émet `MatchStarted`,
    /// reset frags/deaths/health/pickups/powerups de tous les slots.
    /// Retourne `true` si effectivement appliqué, `false` hors mode Server.
    pub fn restart_server_match(&mut self) -> bool {
        match &mut self.inner {
            Inner::Server(state) => {
                state.force_restart_match();
                true
            }
            _ => false,
        }
    }

    /// Envoie un message de chat au serveur (mode client uniquement).
    /// Le serveur le diffusera à tous les autres clients via un
    /// `ServerEvent::Chat` dans le snapshot suivant.
    pub fn send_chat(&mut self, message: &str) -> bool {
        match &mut self.inner {
            Inner::Client(c) => {
                c.send_chat(message);
                true
            }
            _ => false,
        }
    }

    /// Active l'enregistrement de démo dans `path` (mode client). Les
    /// snapshots reçus seront sérialisés en succession dans le fichier.
    /// No-op hors mode Client — la démo est pertinente uniquement
    /// depuis le POV d'un client connecté.
    pub fn start_recording_demo(&mut self, path: &std::path::Path) -> bool {
        match &mut self.inner {
            Inner::Client(c) => {
                c.start_recording(path);
                true
            }
            _ => false,
        }
    }

    /// Évènements à afficher / logger — vidé à chaque appel.
    pub fn drain_events(&mut self) -> Vec<NetEvent> {
        std::mem::take(&mut self.events)
    }
}

/// Évènement réseau visible côté HUD / console.
#[derive(Debug, Clone)]
pub enum NetEvent {
    Connecting,
    Connected,
    Disconnected { reason: String },
    PlayerJoined { name: String },
    PlayerLeft { name: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_fallback_to_singleplayer_on_bad_addr() {
        let m = NetMode::from_cli(Some("not-an-addr"), None, 4);
        assert!(matches!(m, NetMode::SinglePlayer));
    }

    #[test]
    fn cli_connect_takes_precedence_over_host() {
        let m = NetMode::from_cli(
            Some("0.0.0.0:27960"),
            Some("127.0.0.1:27960"),
            4,
        );
        assert!(matches!(m, NetMode::Client { .. }));
    }

    #[test]
    fn singleplayer_has_no_io() {
        let r = NetRuntime::new(NetMode::SinglePlayer);
        assert!(matches!(r.inner, Inner::None));
        assert!(!r.mode.is_networked());
        assert!(r.packet_counters().is_none());
    }

    /// Intégration handshake bout-en-bout via deux NetRuntime (server +
    /// client) qui communiquent via socket UDP réel. Valide que :
    ///   * Le client passe `Idle → Handshaking → Connected`
    ///   * Le serveur alloue un slot pour le client
    ///   * Le client reçoit son slot ID dans la 1re snapshot
    ///   * `take_client_snapshot` rend le snapshot le plus récent
    #[test]
    fn full_handshake_and_first_snapshot() {
        // Serveur
        let server_bind: SocketAddr = "127.0.0.1:0".parse().unwrap();
        // On ne peut pas connaître le port avant le bind, donc on bind
        // d'abord un NetIo séparément pour récupérer le port effectif,
        // puis on construit un NetRuntime sur le même port. Ce détour est
        // spécifique au test — en prod l'utilisateur passe un port fixe.
        // Astuce simple : bind un NetIo, drop, réutiliser le port (TIME_WAIT
        // n'affecte pas UDP). Mais SO_REUSEADDR n'est pas garanti — on
        // utilise plutôt le port 0 et on bypass le NetMode pour construire
        // ServerState directement.
        let server_io = NetIo::bind(server_bind).expect("bind serveur");
        let server_addr = server_io.local_addr();
        let mut server = NetRuntime {
            mode: NetMode::Server {
                bind_addr: server_addr,
                max_clients: 4,
                gametype: GameType::FreeForAll,
                friendly_fire: true,
            },
            inner: Inner::Server(ServerState::new(server_addr, 4, server_io)),
            events: Vec::new(),
        };

        // Client
        let mut client = NetRuntime::new(NetMode::Client { server_addr });
        assert!(matches!(client.inner, Inner::Client(_)));

        // Boucle de tick : on alterne client / server avec un petit dt et
        // un sleep pour laisser tokio processer. 100 itérations × 10 ms
        // = 1 s — largement suffisant pour un handshake LAN (~3 RTT).
        let dt = 0.010_f32;
        let mut got_snapshot = false;
        let mut server_has_slot = false;
        for i in 0..100 {
            client.tick_client(dt, None);
            server.tick_server(dt, None);
            std::thread::sleep(std::time::Duration::from_millis(10));

            if !server_has_slot {
                if let Inner::Server(s) = &server.inner {
                    if !s.slots.is_empty() {
                        server_has_slot = true;
                    }
                }
            }
            if let Some(p) = client.take_client_snapshot() {
                // Le snapshot doit indiquer notre slot et nous y trouver
                // dans `players[]`.
                let snap = &p.snapshot;
                assert!(
                    snap.players.iter().any(|p| p.slot == snap.client_slot),
                    "snapshot[{i}] sans notre joueur"
                );
                got_snapshot = true;
                break;
            }
        }
        assert!(server_has_slot, "le serveur n'a pas alloué de slot");
        assert!(got_snapshot, "le client n'a pas reçu de snapshot");
        assert_eq!(
            client.client_stage().cloned(),
            Some(ClientStage::Connected),
            "client devrait être Connected"
        );
    }

    /// Vérifie que la pipeline delta-compression fonctionne bout-en-bout.
    /// On laisse le serveur émettre plusieurs snapshots à un client ; le
    /// premier est forcément un full (baseline absente), les suivants
    /// sont des deltas. Le client doit en reconstruire au moins un et
    /// afficher des données cohérentes.
    #[test]
    fn delta_compression_path_reconstructs_correctly() {
        let server_io = NetIo::bind("127.0.0.1:0".parse().unwrap()).expect("bind serveur");
        let server_addr = server_io.local_addr();
        let mut server = NetRuntime {
            mode: NetMode::Server {
                bind_addr: server_addr,
                max_clients: 4,
                gametype: GameType::FreeForAll,
                friendly_fire: true,
            },
            inner: Inner::Server(ServerState::new(server_addr, 4, server_io)),
            events: Vec::new(),
        };
        let mut client = NetRuntime::new(NetMode::Client { server_addr });

        // Boucle 200 itérations × 10 ms = 2 s. À 20 Hz snapshot, ça donne
        // ~40 snapshots, dont la majorité doivent être des deltas
        // (FULL_SNAPSHOT_INTERVAL = 20 → 1 full sur 20 envoyés).
        let dt = 0.010_f32;
        let mut snapshots_received = 0;
        let mut last_slot: Option<u8> = None;
        for _ in 0..200 {
            client.tick_client(dt, None);
            server.tick_server(dt, None);
            std::thread::sleep(std::time::Duration::from_millis(10));
            while let Some(p) = client.take_client_snapshot() {
                snapshots_received += 1;
                let snap = &p.snapshot;
                // Cohérence : notre slot doit toujours être présent dans
                // players[], que ce soit un full ou un delta-reconstruit.
                assert!(
                    snap.players.iter().any(|p| p.slot == snap.client_slot),
                    "snapshot incohérent (joueur absent de players[])"
                );
                let s = snap.client_slot;
                if let Some(prev) = last_slot {
                    assert_eq!(prev, s, "client_slot ne devrait pas changer");
                }
                last_slot = Some(s);
            }
        }
        // Avec 40 ticks, on s'attend à recevoir au moins une vingtaine
        // de snapshots — laxiste pour CI lente.
        assert!(
            snapshots_received >= 10,
            "trop peu de snapshots reçus : {snapshots_received}"
        );
    }

    /// Test bout-en-bout : on bind un serveur + un client, on échange un
    /// datagramme via les NetIo réels. Valide que la tâche tokio
    /// fonctionne et que le drain non-blocant remonte les paquets.
    #[test]
    fn netio_roundtrip_via_real_socket() {
        // Le serveur bind une adresse éphémère pour ne pas dépendre
        // d'un port libre fixe sur la CI.
        let server_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let mut server_io = NetIo::bind(server_addr).expect("bind serveur");
        let real_server_addr = server_io.local_addr();

        let mut client_io = NetIo::bind("127.0.0.1:0".parse().unwrap()).expect("bind client");

        client_io.send(real_server_addr, b"hello-server".to_vec());

        // Attente courte du datagramme côté serveur. On ne peut pas
        // fiabiliser sans timeout — UDP local est pratiquement instantané
        // mais l'ordonnancement OS peut retarder. 500 ms est très large.
        let mut received: Option<Vec<u8>> = None;
        for _ in 0..50 {
            let mut bag = server_io.drain_inbox();
            if let Some(dg) = bag.pop() {
                received = Some(dg.bytes);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(received.as_deref(), Some(&b"hello-server"[..]));
    }
}
