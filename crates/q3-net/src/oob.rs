//! Messages **out-of-band** — paquets sans connexion (poll, challenge,
//! connect, rcon…). Préfixés par `0xFFFFFFFF` puis une chaîne ASCII.

use q3_common::Error;

/// Les 4 octets magiques `FF FF FF FF` en tête.
pub const OOB_MAGIC: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];

/// Message OOB décodé.
#[derive(Debug, Clone, PartialEq)]
pub struct OobMessage {
    /// Commande (premier token), par ex. `"getchallenge"`, `"connect"`.
    pub command: String,
    /// Reste de la payload après le premier espace. Peut contenir du binaire
    /// encodé (ex: paquet `connect` avec userinfo quoté).
    pub payload: Vec<u8>,
}

impl OobMessage {
    /// Sérialise un message OOB en paquet UDP.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.command.len() + 1 + self.payload.len());
        out.extend_from_slice(&OOB_MAGIC);
        out.extend_from_slice(self.command.as_bytes());
        if !self.payload.is_empty() {
            out.push(b' ');
            out.extend_from_slice(&self.payload);
        }
        out
    }

    /// Parse un datagramme s'il commence par `0xFFFFFFFF`.
    pub fn parse(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() < 4 || bytes[..4] != OOB_MAGIC {
            return Err(Error::Network("OOB: magic manquant".into()));
        }
        let rest = &bytes[4..];
        // On split sur le premier espace, le reste est la payload.
        let sep = rest.iter().position(|&b| b == b' ');
        let (cmd_bytes, payload) = match sep {
            Some(i) => (&rest[..i], rest[i + 1..].to_vec()),
            None => (rest, Vec::new()),
        };
        let command = std::str::from_utf8(cmd_bytes)
            .map_err(|_| Error::Network("OOB: commande non-ASCII".into()))?
            .to_string();
        Ok(Self { command, payload })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_no_payload() {
        let m = OobMessage {
            command: "getchallenge".into(),
            payload: vec![],
        };
        let bytes = m.to_bytes();
        assert_eq!(&bytes[..4], &OOB_MAGIC);
        assert_eq!(&bytes[4..], b"getchallenge");
        let parsed = OobMessage::parse(&bytes).unwrap();
        assert_eq!(parsed, m);
    }

    #[test]
    fn roundtrip_with_payload() {
        let m = OobMessage {
            command: "connect".into(),
            payload: b"\"\\name\\player\\rate\\25000\"".to_vec(),
        };
        let bytes = m.to_bytes();
        let parsed = OobMessage::parse(&bytes).unwrap();
        assert_eq!(parsed, m);
    }

    #[test]
    fn parse_rejects_non_oob() {
        // Commence par une séquence valide (≠ -1).
        let bytes = [0u8, 0, 0, 42, b'h', b'i'];
        assert!(OobMessage::parse(&bytes).is_err());
    }
}
