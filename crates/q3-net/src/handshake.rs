//! Handshake **client ↔ serveur** à la Q3, transport-agnostique.
//!
//! Ce module implémente uniquement les **machines à états** du handshake.
//! Le vrai envoi/réception UDP est laissé à l'appelant (via
//! [`crate::UdpEndpoint`] ou un transport de test). Les tests plus bas
//! démontrent une convergence complète via une paire de files en mémoire.
//!
//! Flux Q3 (voir `cl_main.c` / `sv_client.c`) :
//!
//! ```text
//! client                                server
//! ------                                ------
//! Disconnected
//!   → OobMessage { command: "getchallenge" }
//! Challenging                           Idle
//!                                          ← génère un `challenge` aléatoire
//!                                            et remet l'état du slot à "Auth"
//!   ← OobMessage { command: "challengeResponse", payload: <challenge> }
//! Challenged
//!   → OobMessage { command: "connect",
//!                  payload: "<userinfo> \"<challenge>\"" }
//! Connecting                            Auth
//!                                          ← valide le challenge, crée le
//!                                            slot, switch sur "Primed"
//!   ← OobMessage { command: "connectResponse" }
//! Connected                             Primed
//!                                          ← attend que le client envoie son
//!                                            premier MSG via NetChannel
//! ```
//!
//! On n'implémente pas `getstatus` / `getinfo` (poll LAN), ni `rcon` ici —
//! ils sont triviaux à rajouter plus tard vu que l'infrastructure OOB y est.
//!
//! # Userinfo
//! Le contenu du champ `userinfo` est passé opaque. À terme il sera parsé
//! dans `q3_common::userinfo`, mais pour le handshake pur on le passe tel
//! quel : c'est à la couche applicative d'en faire quelque chose.

use crate::oob::OobMessage;
use q3_common::Error;
use std::collections::HashMap;
use std::net::SocketAddr;
use tracing::{debug, warn};

/// États du handshake côté client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientState {
    /// Aucun paquet envoyé — état initial / après déconnexion.
    Disconnected,
    /// `getchallenge` envoyé, en attente de `challengeResponse`.
    Challenging,
    /// Challenge reçu, `connect` envoyé, en attente de `connectResponse`.
    Connecting,
    /// Handshake terminé — on est prêt à passer en mode `NetChannel`.
    Connected,
}

/// Résultat du pas de handshake côté client : la machine génère au plus
/// un paquet OOB par appel.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientStep {
    /// Rien à faire.
    Idle,
    /// À envoyer au serveur (bytes = paquet UDP complet, préfixe OOB inclus).
    Send(Vec<u8>),
    /// Le handshake vient de se terminer avec succès ; le challenge final
    /// est retourné pour que la couche supérieure puisse l'utiliser comme
    /// seed du `NetChannel` si besoin.
    Done { challenge: u32 },
    /// Échec (ex: magic OOB invalide, commande inattendue).
    Failed(String),
}

/// Handshake côté client.
#[derive(Debug, Clone)]
pub struct ClientHandshake {
    state: ClientState,
    /// Userinfo envoyé avec le paquet `connect`. Contenu opaque.
    userinfo: String,
    /// Challenge reçu du serveur (rempli en `Challenging → Connecting`).
    challenge: Option<u32>,
}

impl ClientHandshake {
    pub fn new(userinfo: impl Into<String>) -> Self {
        Self {
            state: ClientState::Disconnected,
            userinfo: userinfo.into(),
            challenge: None,
        }
    }

    pub fn state(&self) -> ClientState {
        self.state
    }

    pub fn challenge(&self) -> Option<u32> {
        self.challenge
    }

    /// Démarre le handshake : émet `getchallenge`.
    pub fn start(&mut self) -> ClientStep {
        if self.state != ClientState::Disconnected {
            return ClientStep::Idle;
        }
        self.state = ClientState::Challenging;
        let msg = OobMessage {
            command: "getchallenge".into(),
            payload: Vec::new(),
        };
        ClientStep::Send(msg.to_bytes())
    }

    /// Consomme un paquet OOB du serveur et avance l'état.
    pub fn handle(&mut self, bytes: &[u8]) -> ClientStep {
        let msg = match OobMessage::parse(bytes) {
            Ok(m) => m,
            Err(e) => return ClientStep::Failed(format!("parse OOB: {e}")),
        };
        match (self.state, msg.command.as_str()) {
            (ClientState::Challenging, "challengeResponse") => {
                let Some(challenge) = parse_u32_payload(&msg.payload) else {
                    return ClientStep::Failed("challengeResponse: payload invalide".into());
                };
                self.challenge = Some(challenge);
                self.state = ClientState::Connecting;
                let reply = OobMessage {
                    command: "connect".into(),
                    payload: format!("{} \"{}\"", challenge, self.userinfo).into_bytes(),
                };
                ClientStep::Send(reply.to_bytes())
            }
            (ClientState::Connecting, "connectResponse") => {
                self.state = ClientState::Connected;
                let challenge = self.challenge.unwrap_or(0);
                debug!("handshake: client connecté (challenge={challenge})");
                ClientStep::Done { challenge }
            }
            (_, cmd) => ClientStep::Failed(format!(
                "client: commande `{cmd}` inattendue en état {:?}",
                self.state
            )),
        }
    }
}

/// États d'un slot client côté serveur.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// Le challenge vient d'être émis, en attente du `connect` signé.
    Authorizing,
    /// Handshake complété.
    Connected,
}

#[derive(Debug, Clone)]
struct ClientSlot {
    state: SlotState,
    challenge: u32,
}

/// Handshake côté serveur : maintient un slot par `SocketAddr`.
pub struct ServerHandshake {
    slots: HashMap<SocketAddr, ClientSlot>,
    /// Générateur pseudo-aléatoire interne. Seedé au [`Self::new`] — les
    /// tests utilisent [`Self::new_with_seed`] pour la reproductibilité.
    rng_state: u64,
}

impl ServerHandshake {
    pub fn new() -> Self {
        // Seed rapide basé sur l'horloge monotonic (pas crypto-secure mais
        // suffisant pour un challenge de handshake).
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xDEADBEEF_CAFEBABE);
        Self::new_with_seed(seed)
    }

    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            slots: HashMap::new(),
            rng_state: seed | 1, // éviter 0 (LCG dégénère)
        }
    }

    /// Nombre de clients actuellement en cours de handshake ou connectés.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Vérifie si un slot est dans l'état `Connected`.
    pub fn is_connected(&self, addr: &SocketAddr) -> bool {
        matches!(
            self.slots.get(addr),
            Some(s) if s.state == SlotState::Connected
        )
    }

    /// Retire un slot (ex : client qui quitte, timeout).
    pub fn drop_client(&mut self, addr: &SocketAddr) -> bool {
        self.slots.remove(addr).is_some()
    }

    /// Consomme un paquet OOB d'un client. Retourne :
    /// * `Ok(Some(bytes))` : paquet à renvoyer à `addr`
    /// * `Ok(None)`        : pas de réponse à émettre (commande ignorée)
    /// * `Err(_)`          : paquet invalide — à logger
    pub fn handle(&mut self, addr: SocketAddr, bytes: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        let msg = OobMessage::parse(bytes)?;
        match msg.command.as_str() {
            "getchallenge" => {
                let challenge = self.next_challenge();
                self.slots.insert(
                    addr,
                    ClientSlot {
                        state: SlotState::Authorizing,
                        challenge,
                    },
                );
                debug!("handshake: challenge {challenge} émis pour {addr}");
                let reply = OobMessage {
                    command: "challengeResponse".into(),
                    payload: challenge.to_string().into_bytes(),
                };
                Ok(Some(reply.to_bytes()))
            }
            "connect" => {
                let Some(slot) = self.slots.get_mut(&addr) else {
                    warn!("connect sans challenge préalable de {addr}");
                    return Ok(Some(OobMessage {
                        command: "print".into(),
                        payload: b"No challenge for this address.".to_vec(),
                    }
                    .to_bytes()));
                };
                // Payload : `<challenge> "<userinfo>"`
                let (got_challenge, _userinfo) = match parse_connect_payload(&msg.payload) {
                    Some(x) => x,
                    None => {
                        return Ok(Some(OobMessage {
                            command: "print".into(),
                            payload: b"Malformed connect packet.".to_vec(),
                        }
                        .to_bytes()));
                    }
                };
                if got_challenge != slot.challenge {
                    warn!(
                        "connect refusé de {addr} : challenge mismatch ({got_challenge} vs {})",
                        slot.challenge
                    );
                    return Ok(Some(OobMessage {
                        command: "print".into(),
                        payload: b"Bad challenge.".to_vec(),
                    }
                    .to_bytes()));
                }
                slot.state = SlotState::Connected;
                debug!("handshake: {addr} connecté (challenge {got_challenge})");
                let reply = OobMessage {
                    command: "connectResponse".into(),
                    payload: Vec::new(),
                };
                Ok(Some(reply.to_bytes()))
            }
            "getstatus" | "getinfo" => {
                // Placeholder : LAN-poll. On renvoie une map plate.
                let info = b"\\sv_hostname\\q3-rust\\mapname\\unknown\\sv_maxclients\\8";
                let cmd = if msg.command == "getstatus" {
                    "statusResponse"
                } else {
                    "infoResponse"
                };
                let reply = OobMessage {
                    command: cmd.into(),
                    payload: info.to_vec(),
                };
                Ok(Some(reply.to_bytes()))
            }
            other => {
                warn!("OOB inconnu `{other}` reçu de {addr}");
                Ok(None)
            }
        }
    }

    /// LCG 64-bit → u32. Assez pour un challenge handshake ; pas
    /// cryptographique.
    fn next_challenge(&mut self) -> u32 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.rng_state >> 32) as u32
    }
}

impl Default for ServerHandshake {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_u32_payload(payload: &[u8]) -> Option<u32> {
    std::str::from_utf8(payload).ok()?.trim().parse::<u32>().ok()
}

/// Parse `<challenge> "<userinfo>"`. La validation est minimaliste : on
/// split au premier espace, on enlève les guillemets éventuels du second
/// morceau, le reste est considéré userinfo brut.
fn parse_connect_payload(payload: &[u8]) -> Option<(u32, String)> {
    let s = std::str::from_utf8(payload).ok()?;
    let (a, b) = s.split_once(' ')?;
    let challenge: u32 = a.trim().parse().ok()?;
    let userinfo = b.trim().trim_matches('"').to_string();
    Some((challenge, userinfo))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_ADDR: &str = "127.0.0.1:27960";

    fn addr() -> SocketAddr {
        TEST_ADDR.parse().unwrap()
    }

    #[test]
    fn full_handshake_converges() {
        let mut client = ClientHandshake::new("\\name\\tester\\rate\\25000");
        let mut server = ServerHandshake::new_with_seed(42);

        let ClientStep::Send(c1) = client.start() else {
            panic!("client.start() doit envoyer");
        };
        assert_eq!(client.state(), ClientState::Challenging);

        let s1 = server.handle(addr(), &c1).unwrap().expect("réponse server");
        assert_eq!(server.slot_count(), 1);

        let ClientStep::Send(c2) = client.handle(&s1) else {
            panic!("client doit répondre connect");
        };
        assert_eq!(client.state(), ClientState::Connecting);
        assert!(client.challenge().is_some());

        let s2 = server.handle(addr(), &c2).unwrap().expect("connectResponse");
        assert!(server.is_connected(&addr()));

        let ClientStep::Done { challenge } = client.handle(&s2) else {
            panic!("client doit terminer");
        };
        assert_eq!(challenge, client.challenge().unwrap());
        assert_eq!(client.state(), ClientState::Connected);
    }

    #[test]
    fn bad_challenge_is_rejected() {
        let mut server = ServerHandshake::new_with_seed(1);
        // Envoie getchallenge pour provoquer l'allocation du slot.
        let c1 = OobMessage {
            command: "getchallenge".into(),
            payload: vec![],
        }
        .to_bytes();
        let _ = server.handle(addr(), &c1).unwrap();

        // Répond avec le mauvais challenge.
        let bad = OobMessage {
            command: "connect".into(),
            payload: b"9999 \"\\name\\fake\"".to_vec(),
        }
        .to_bytes();
        let reply = server.handle(addr(), &bad).unwrap().unwrap();
        let parsed = OobMessage::parse(&reply).unwrap();
        assert_eq!(parsed.command, "print");
        assert!(!server.is_connected(&addr()));
    }

    #[test]
    fn connect_without_challenge_rejected() {
        let mut server = ServerHandshake::new_with_seed(1);
        let pkt = OobMessage {
            command: "connect".into(),
            payload: b"123 \"x\"".to_vec(),
        }
        .to_bytes();
        let reply = server.handle(addr(), &pkt).unwrap().unwrap();
        let parsed = OobMessage::parse(&reply).unwrap();
        assert_eq!(parsed.command, "print");
    }

    #[test]
    fn getstatus_returns_info_response() {
        let mut server = ServerHandshake::new_with_seed(1);
        let pkt = OobMessage {
            command: "getstatus".into(),
            payload: vec![],
        }
        .to_bytes();
        let reply = server.handle(addr(), &pkt).unwrap().unwrap();
        let parsed = OobMessage::parse(&reply).unwrap();
        assert_eq!(parsed.command, "statusResponse");
        assert!(parsed.payload.windows(11).any(|w| w == b"sv_hostname"));
    }

    #[test]
    fn drop_client_removes_slot() {
        let mut server = ServerHandshake::new_with_seed(1);
        let c1 = OobMessage {
            command: "getchallenge".into(),
            payload: vec![],
        }
        .to_bytes();
        let _ = server.handle(addr(), &c1).unwrap();
        assert_eq!(server.slot_count(), 1);
        assert!(server.drop_client(&addr()));
        assert_eq!(server.slot_count(), 0);
    }

    #[test]
    fn client_start_is_idempotent() {
        let mut client = ClientHandshake::new("");
        let _ = client.start();
        assert_eq!(client.start(), ClientStep::Idle);
    }
}
