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
    /// Offset attendu du prochain fragment (front contigu de réassemblage).
    /// Les fragments sont émis en ordre croissant contigu ; un `start` qui
    /// ne colle pas à ce front trahit un fragment manquant/réordonné.
    reassembly_expected: usize,
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
        // **Terminator Q3** — le récepteur reconnaît le dernier fragment à
        // sa longueur `< chunk_size`. Si `payload.len()` est un multiple
        // exact de `chunk_size`, le dernier fragment de données fait
        // pile `chunk_size` et ne serait JAMAIS reconnu comme final → le
        // message n'est jamais livré et empoisonne le réassemblage. On
        // émet alors un fragment terminal de longueur 0 (comme Q3).
        if !payload.is_empty() && payload.len() % chunk_size == 0 {
            let mut buf = BytesMut::with_capacity(8);
            buf.put_u32_le(seq | FRAGMENT_BIT);
            buf.put_u16_le(payload.len() as u16); // start = fin des données
            buf.put_u16_le(0); // length = 0 → marqueur de fin
            packets.push(buf.to_vec());
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

        // Comparaison de séquence robuste au wrap-around `u32` : on regarde
        // la distance SIGNÉE. `<= 0` → paquet ancien OU dupliqué → drop.
        // L'ancien `seq < in_sequence` (non-wrapping) cassait au wrap (après
        // 2³² paquets le canal restait sourd) ET re-livrait les doublons
        // (`seq == in_sequence` passait).  Le calcul par différence gère
        // correctement le retour à zéro : `1u32.wrapping_sub(u32::MAX)` = 2.
        if (seq.wrapping_sub(self.in_sequence) as i32) <= 0 {
            return Ok(None);
        }

        if !fragmented {
            self.in_sequence = seq;
            // reset éventuel buffer de réassemblage
            if self.reassembly_seq != 0 && self.reassembly_seq != seq {
                self.reassembly.clear();
                self.reassembly_seq = 0;
                self.reassembly_expected = 0;
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
            self.reassembly_expected = 0;
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
                format!("fragment trop grand : {end} > {MAX_REASSEMBLY}"),
            ));
        }

        // **Anti-trou, tolérant aux doublons/réordos** — les fragments sont
        // émis en ordre croissant contigu (cf. `prepare_outgoing`). Trois
        // cas selon la position du fragment vs le front de réassemblage :
        //   * `start > expected` → un fragment MANQUE devant celui-ci. Sans
        //     garde, le trou resté à zéro (via `resize`) serait livré comme
        //     un message « complet » (des zéros se décodent en PlayerState
        //     valides → corruption silencieuse). On drop tout ; le prochain
        //     snapshot complet resynchronise.
        //   * `start < expected` ET entièrement dans le préfixe déjà rempli
        //     → simple duplication/réordonnancement UDP d'un fragment déjà
        //     reçu → on l'ignore sans casser le réassemblage en cours.
        //   * chevauchement partiel incohérent → suspect → drop.
        if start > self.reassembly_expected {
            self.reassembly.clear();
            self.reassembly_seq = 0;
            self.reassembly_expected = 0;
            return Ok(None);
        }
        if start < self.reassembly_expected {
            if end <= self.reassembly_expected {
                // Redite bénigne d'un fragment déjà intégré → no-op.
                return Ok(None);
            }
            // Recouvrement partiel qui ne colle pas au flux ordonné → drop.
            self.reassembly.clear();
            self.reassembly_seq = 0;
            self.reassembly_expected = 0;
            return Ok(None);
        }

        if self.reassembly.len() < end {
            self.reassembly.resize(end, 0);
        }
        self.reassembly[start..end].copy_from_slice(&cursor[..length]);
        // Avance le front contigu (contrôle anti-trou du prochain fragment).
        self.reassembly_expected = end;

        // Heuristique Q3 : un fragment dont la longueur est < MAX_PAYLOAD - 4
        // est le dernier (cf. `Netchan_Process`).
        let last_fragment = length < (MAX_PAYLOAD - 4);
        if last_fragment {
            let out = self.reassembly.split().to_vec();
            self.reassembly_seq = 0;
            self.reassembly_expected = 0;
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

    #[test]
    fn exact_multiple_of_chunk_size_delivers() {
        // Bug historique : un payload dont la taille est un multiple exact
        // de `chunk_size` n'émettait pas de terminator → dernier fragment
        // jamais reconnu → message perdu. On vérifie qu'il est livré.
        let chunk = MAX_PAYLOAD - 4;
        let mut tx = NetChannel::new();
        let mut rx = NetChannel::new();
        let big: Vec<u8> = (0..(chunk * 2)).map(|i| (i & 0xFF) as u8).collect();
        let packets = tx.prepare_outgoing(&big);
        // 2 fragments de données + 1 terminator zéro-longueur.
        assert_eq!(packets.len(), 3);
        let mut last = None;
        for p in &packets {
            last = rx.process_incoming(p).unwrap();
        }
        assert_eq!(last.expect("message doit être livré"), big);
    }

    #[test]
    fn missing_fragment_yields_no_corrupt_message() {
        // Bug historique : un fragment intermédiaire perdu laissait un trou
        // de zéros livré comme message « complet ». On saute le 1er fragment.
        let mut tx = NetChannel::new();
        let mut rx = NetChannel::new();
        let big: Vec<u8> = (0..3500u32).map(|i| (i & 0xFF) as u8).collect();
        let packets = tx.prepare_outgoing(&big);
        assert!(packets.len() >= 3);
        // On ne délivre PAS packets[0] (fragment perdu), puis les suivants.
        let mut delivered = None;
        for p in &packets[1..] {
            if let Some(m) = rx.process_incoming(p).unwrap() {
                delivered = Some(m);
            }
        }
        assert!(
            delivered.is_none(),
            "un message avec un fragment manquant ne doit jamais être livré"
        );
    }

    #[test]
    fn duplicated_interior_fragment_is_tolerated() {
        // UDP peut dupliquer un fragment interne. Une redite ne doit PAS
        // détruire le message en cours de réassemblage : A,A,B,C se
        // réassemble comme A,B,C.
        let mut tx = NetChannel::new();
        let mut rx = NetChannel::new();
        let big: Vec<u8> = (0..3500u32).map(|i| (i & 0xFF) as u8).collect();
        let packets = tx.prepare_outgoing(&big);
        assert!(packets.len() >= 3);
        // Rejoue le 1er fragment une 2e fois avant les suivants.
        let order = [0usize, 0, 1, 2];
        let mut delivered = None;
        for &i in &order {
            if let Some(m) = rx.process_incoming(&packets[i]).unwrap() {
                delivered = Some(m);
            }
        }
        assert_eq!(
            delivered.expect("message doit être livré malgré le doublon"),
            big
        );
    }

    #[test]
    fn duplicate_packet_is_dropped() {
        // `seq == in_sequence` (doublon) ne doit pas être re-livré.
        let mut tx = NetChannel::new();
        let mut rx = NetChannel::new();
        let p = tx.prepare_outgoing(b"once").into_iter().next().unwrap();
        assert_eq!(rx.process_incoming(&p).unwrap().unwrap(), b"once");
        assert!(
            rx.process_incoming(&p).unwrap().is_none(),
            "un doublon doit être ignoré"
        );
    }
}
