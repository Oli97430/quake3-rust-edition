//! Netcode Quake 3 — transport UDP asynchrone basé sur **tokio**.
//!
//! Reproduit le format de paquet Q3 (voir `net_chan.c`) :
//!
//! * **Connectionless (OOB)** : `int32 magic = -1` suivi d'une chaîne ASCII.
//!   Exemples : `getchallenge`, `getstatus`, `connect "<userinfo>"`, `rcon`.
//! * **Connected** : `int32 sequence` (bit 31 = fragment) + payload.
//!
//! # Améliorations vs C original
//!
//! * **Tokio async** : un unique thread peut servir 64 clients sans bloquer.
//! * **No unsafe** : le parsing utilise `bytes::Buf` avec validation longueur.
//! * **Fragmentation** : géré explicitement dans `NetChannel` plutôt qu'avec
//!   des goto et des `static` partagés.
//! * **TLS-like ID de challenge** : rand 32-bit via `tokio::time` — on
//!   pourrait passer à une vraie PRNG si nécessaire.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

pub mod handshake;
pub mod messages;
pub mod netchan;
pub mod oob;

pub use handshake::{
    ClientHandshake, ClientState, ClientStep, ServerHandshake, SlotState,
};
pub use messages::{
    buttons, player_flags, powerup_flags, sound_id, team, ClientPacket, EntityKindWire,
    EntityState, ExplosionKind, PickupState, PlayerInfo, PlayerState, ServerEvent, Snapshot,
    SnapshotDelta, UserCmd, MAX_ENTITIES_PER_SNAPSHOT, MAX_EVENTS_PER_SNAPSHOT,
    MAX_PLAYERS_PER_SNAPSHOT, MAX_USERCMDS_PER_PACKET, PROTOCOL_VERSION, TAG_CLIENT_PACKET,
    TAG_SNAPSHOT, TAG_SNAPSHOT_DELTA,
};
pub use netchan::{Fragment, NetChannel, MAX_PACKET, MAX_PAYLOAD};
pub use oob::{OobMessage, OOB_MAGIC};

use q3_common::Error;
use std::{io, net::SocketAddr};
use tokio::net::UdpSocket;
use tracing::{debug, warn};

/// Taille max d'un datagramme UDP en pratique pour Q3 (MTU safe).
pub const UDP_MTU: usize = 1400;

/// Socket UDP partagé, utilisé à la fois côté client et côté serveur.
///
/// Les deux extrémités sont symétriques — on envoie des datagrammes et on
/// les reçoit avec leur source, la logique `OOB vs channel` est dans
/// `NetChannel` et `OobMessage`.
pub struct UdpEndpoint {
    socket: UdpSocket,
}

impl UdpEndpoint {
    pub async fn bind(addr: impl tokio::net::ToSocketAddrs) -> Result<Self, Error> {
        let socket = UdpSocket::bind(addr)
            .await
            .map_err(|e| Error::Network(format!("UDP bind: {e}")))?;
        debug!("UDP bound on {:?}", socket.local_addr().ok());
        Ok(Self { socket })
    }

    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    /// Envoie un datagramme à `addr`.
    pub async fn send_to(&self, bytes: &[u8], addr: SocketAddr) -> Result<(), Error> {
        if bytes.len() > UDP_MTU {
            warn!(
                "net: paquet de {} octets > MTU ({}), risque de fragmentation IP",
                bytes.len(),
                UDP_MTU
            );
        }
        self.socket
            .send_to(bytes, addr)
            .await
            .map(|_| ())
            .map_err(|e| Error::Network(format!("UDP send_to: {e}")))
    }

    /// Reçoit le prochain datagramme. Bloque jusqu'à arrivée.
    pub async fn recv_from(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr), Error> {
        self.socket
            .recv_from(buf)
            .await
            .map_err(|e| Error::Network(format!("UDP recv_from: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn endpoint_binds_on_ephemeral() {
        let ep = UdpEndpoint::bind("127.0.0.1:0").await.unwrap();
        let addr = ep.local_addr().unwrap();
        assert!(addr.port() != 0);
    }

    #[tokio::test]
    async fn send_and_receive_roundtrip() {
        let a = UdpEndpoint::bind("127.0.0.1:0").await.unwrap();
        let b = UdpEndpoint::bind("127.0.0.1:0").await.unwrap();
        let a_addr = a.local_addr().unwrap();
        let b_addr = b.local_addr().unwrap();
        a.send_to(b"hello", b_addr).await.unwrap();
        let mut buf = [0u8; 64];
        let (n, from) = b.recv_from(&mut buf).await.unwrap();
        assert_eq!(&buf[..n], b"hello");
        assert_eq!(from, a_addr);
    }
}
