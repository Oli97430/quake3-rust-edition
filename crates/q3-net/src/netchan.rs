//! **NetChannel** — channel fiable/ordonné au-dessus d'UDP. Gère les
//! séquences, la fragmentation et le réassemblage.
//!
//! Format du paquet (après un `u32 sequence` normal — bit 31 = fragment) :
//!
//! ```text
//! u32  sequence   (bit 31 set si fragmenté)
//! [u16 fragment_start  si fragmenté]
//! [u16 fragment_length si fragmenté]
//! ...payload...
//! ```
//!
//! Pour le qport (client id), Q3 insère aussi un `u16` juste après la seq
//! côté client ; on l'expose séparément pour laisser le choix d'utiliser ou
//! pas.

use bytes::{Buf, BufMut, BytesMut};
use q3_common::Error;

/// Taille maximale d'un datagramme Q3. Au-delà, on fragmente.
pub const MAX_PACKET: usize = 1400;
/// Taille maximale d'un message après réassemblage de fragments.  Q3
/// original : `MAX_MSGLEN = 16384`.  Protège contre un peer qui
/// annoncerait `start + length` arbitrairement grand pour allouer 128
/// KiB en RAM par message — on refuse net au-delà de ce seuil.
pub const MAX_REASSEMBLY: usize = 16 * 1024;
/// Payload max après les en-têtes (4 octets seq + 4 octets fragment).
pub const MAX_PAYLOAD: usize = MAX_PACKET - 8;

const FRAGMENT_BIT: u32 = 1 << 31;

/// État d'un channel entre deux pairs. À maintenir côté serveur par client,
/// et côté client unique.
#[derive(Debug, Default)]
pub struct NetChannel {
    /// Prochaine séquence à envoyer.
    pub out_sequence: u32,
    /// Dernière séquence reçue du distant.
    pub in_sequence: u32,
    /// Buffer de réassemblage pour le message fragmenté en cours.
    reassembly: BytesMut,
    /// Séquence du message en cours de réassemblage (0 si aucun).
    reassembly_seq: u32,
}

impl NetChannel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sérialise un message. Retourne un ou plusieurs paquets (fragmentés si
    /// > `MAX_PAYLOAD`). La séquence est auto-incrémentée.
    pub fn prepare_outgoing(&mut self, payload: &[u8]) -> Vec<Vec<u8>> {
        self.out_sequence = self.out_sequence.wrapping_add(1);
        let seq = self.out_sequence;
        if payload.len() <= MAX_PAYLOAD {
            let mut buf = BytesMut::with_capacity(4 + payload.len());
            buf.put_u32_le(seq);
            buf.put_slice(payload);
            return vec![buf.to_vec()];
        }
        // Fragmentation.
        let mut packets = Vec::new();
        let chunk_size = MAX_PAYLOAD - 4; // -4 pour start+length u16 u16
        let mut offset = 0usize;
        while offset < payload.len() {
            let remaining = payload.len() - offset;
            let take = remaining.min(chunk_size);
            let mut buf = BytesMut::with_capacity(8 + take);
            buf.put_u32_le(seq | FRAGMENT_BIT);
            buf.put_u16_le(offset as u16);
            buf.put_u16_le(take as u16);
            buf.put_slice(&payload[offset..offset + take]);
            packets.push(buf.to_vec());
            offset += take;
        }
        packets
    }

    /// Consomme un paquet entrant. Retourne `Some(payload)` si un message
    /// complet (non fragmenté ou dernier fragment) est prêt, sinon `None`.
    pub fn process_incoming(&mut self, bytes: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        let mut cursor = bytes;
        if cursor.len() < 4 {
            return Err(Error::Network("paquet trop court".into()));
        }
        let seq_field = cursor.get_u32_le();
        let fragmented = (seq_field & FRAGMENT_BIT) != 0;
        let seq = seq_field & !FRAGMENT_BIT;

        if seq < self.in_sequence {
            // out-of-order, dropper.
            return Ok(None);
        }

        if !fragmented {
            self.in_sequence = seq;
            // reset éventuel buffer de réassemblage
            if self.reassembly_seq != 0 && self.reassembly_seq != seq {
                self.reassembly.clear();
                self.reassembly_seq = 0;
            }
            return Ok(Some(cursor.to_vec()));
        }

        if cursor.len() < 4 {
            return Err(Error::Network("fragment header tronqué".into()));
        }
        let start = cursor.get_u16_le() as usize;
        let length = cursor.get_u16_le() as usize;
        if cursor.len() < length {
            return Err(Error::Network("fragment body tronqué".into()));
        }

        if self.reassembly_seq != seq {
            self.reassembly.clear();
            self.reassembly_seq = seq;
        }

        // Cap anti-DoS : un peer hostile pourrait annoncer `start` proche
        // de `u16::MAX` pour forcer l'allocation de ~128 KiB par flux.
        // Q3 original tranche à `MAX_MSGLEN = 16 KiB` — on fait pareil et
        // on drop le paquet entier si ça déborde.
        let Some(end) = start.checked_add(length) else {
            return Err(Error::Network("fragment offset overflow".into()));
        };
        if end > MAX_REASSEMBLY {
            return Err(Error::Network(
                format!("fragment trop grand : {end} > {MAX_REASSEMBLY}").into(),
            ));
        }
        if self.reassembly.len() < end {
            self.reassembly.resize(end, 0);
        }
        self.reassembly[start..end].copy_from_slice(&cursor[..length]);

        // Heuristique Q3 : un fragment dont la longueur est < MAX_PAYLOAD - 4
        // est le dernier (cf. `Netchan_Process`).
        let last_fragment = length < (MAX_PAYLOAD - 4);
        if last_fragment {
            let out = self.reassembly.split().to_vec();
            self.reassembly_seq = 0;
            self.in_sequence = seq;
            Ok(Some(out))
        } else {
            Ok(None)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Fragment {
    pub start: u16,
    pub data: Vec<u8>,
    pub is_last: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_message_roundtrips_unfragmented() {
        let mut tx = NetChannel::new();
        let mut rx = NetChannel::new();
        let packets = tx.prepare_outgoing(b"hello");
        assert_eq!(packets.len(), 1);
        let out = rx.process_incoming(&packets[0]).unwrap().unwrap();
        assert_eq!(out, b"hello");
        assert_eq!(rx.in_sequence, 1);
    }

    #[test]
    fn large_message_fragments_and_reassembles() {
        let mut tx = NetChannel::new();
        let mut rx = NetChannel::new();
        let big: Vec<u8> = (0..3500u32).map(|i| (i & 0xFF) as u8).collect();
        let packets = tx.prepare_outgoing(&big);
        assert!(packets.len() > 1);
        let mut last = None;
        for p in &packets {
            last = rx.process_incoming(p).unwrap();
        }
        let received = last.expect("dernier fragment doit finaliser");
        assert_eq!(received, big);
    }

    #[test]
    fn out_of_order_older_packet_is_dropped() {
        let mut tx = NetChannel::new();
        let mut rx = NetChannel::new();
        let p1 = tx.prepare_outgoing(b"first").into_iter().next().unwrap();
        let p2 = tx.prepare_outgoing(b"second").into_iter().next().unwrap();
        // reçoit 2 avant 1
        rx.process_incoming(&p2).unwrap();
        let older = rx.process_incoming(&p1).unwrap();
        assert!(older.is_none(), "paquet plus ancien doit être ignoré");
    }
}
