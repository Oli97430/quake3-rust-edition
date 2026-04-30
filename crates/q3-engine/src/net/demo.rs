//! Lecteur de démos `.q3rdm` — réinjecte des snapshots déjà enregistrés
//! dans la pipeline client comme s'ils arrivaient du wire.
//!
//! # Format
//!
//! Cohérent avec `client.rs::write_demo_record` :
//!
//! ```text
//! [magic = b"Q3RD"][version = u32 LE]
//!   record* :
//!     [elapsed_ms u32 LE][len u32 LE][bytes len]
//! ```
//!
//! Les bytes du payload ré-utilisent le tag `TAG_SNAPSHOT` ou
//! `TAG_SNAPSHOT_DELTA` du wire — la décompression delta passe par la
//! même logique que côté `ClientState::on_packet`, ce qui garantit qu'un
//! delta enregistré reste rejouable tant que sa baseline est dans la
//! démo (le serveur émet un full toutes les ~1 s, donc OK).
//!
//! # Cadence
//!
//! Le player respecte les `elapsed_ms` originaux pour ne pas accélérer /
//! ralentir l'animation à la lecture. `tick(now)` renvoie *tous* les
//! snapshots dont l'horodatage est `<=` au temps écoulé depuis le start
//! de lecture — utile pour rattraper le retard en cas de frame longue.

use crate::net::client::{PredictionInputs, DEMO_MAGIC, DEMO_VERSION};
use q3_net::{Snapshot, SnapshotDelta, TAG_SNAPSHOT, TAG_SNAPSHOT_DELTA};
use std::io::Read;
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Un record déjà décodé du fichier — payload brut + horodatage.
#[derive(Debug, Clone)]
struct DemoRecord {
    elapsed_ms: u32,
    payload: Vec<u8>,
}

/// État de lecture d'une démo. Vit dans `NetRuntime::Inner::Demo`.
#[derive(Debug)]
pub struct DemoPlayer {
    records: Vec<DemoRecord>,
    /// Index du prochain record à émettre. Avance monotone — pas de
    /// rewind support pour l'instant (UX `.q3rdm` minimal).
    next_index: usize,
    /// Instant de démarrage de la lecture, fixé au 1er `tick()`. Tous
    /// les `elapsed_ms` sont relatifs à cet instant.
    start: Option<Instant>,
    /// Dernier snapshot full décodé — sert de baseline pour
    /// reconstituer les deltas suivants. Identique à
    /// `ClientState::last_full_baseline`.
    last_full_baseline: Option<Snapshot>,
    /// Booléen interne pour log de fin-de-démo une seule fois.
    finished_logged: bool,
}

impl DemoPlayer {
    /// Charge `path` en mémoire et parse tous les records. Retourne une
    /// erreur si le header est invalide ou la version inconnue. La taille
    /// d'une démo Q3 typique reste sous quelques MB — pas de raison de
    /// streamer paresseusement pour l'instant.
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| format!("open demo `{}`: {e}", path.display()))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| format!("read demo `{}`: {e}", path.display()))?;
        if bytes.len() < 8 {
            return Err(format!(
                "demo `{}` trop courte ({} octets)",
                path.display(),
                bytes.len()
            ));
        }
        if &bytes[..4] != DEMO_MAGIC {
            return Err(format!("demo `{}`: magic invalide", path.display()));
        }
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != DEMO_VERSION {
            return Err(format!(
                "demo `{}`: version {} non supportée (attendue {})",
                path.display(),
                version,
                DEMO_VERSION
            ));
        }
        let mut records = Vec::new();
        let mut i = 8;
        while i + 8 <= bytes.len() {
            let elapsed_ms = u32::from_le_bytes([
                bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3],
            ]);
            let len = u32::from_le_bytes([
                bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7],
            ]) as usize;
            i += 8;
            if i + len > bytes.len() {
                warn!(
                    "demo `{}`: record tronqué à offset {i} (len={len}, restant={})",
                    path.display(),
                    bytes.len() - i
                );
                break;
            }
            records.push(DemoRecord {
                elapsed_ms,
                payload: bytes[i..i + len].to_vec(),
            });
            i += len;
        }
        info!(
            "demo `{}`: {} records, durée ≈ {} ms",
            path.display(),
            records.len(),
            records.last().map(|r| r.elapsed_ms).unwrap_or(0)
        );
        Ok(Self {
            records,
            next_index: 0,
            start: None,
            last_full_baseline: None,
            finished_logged: false,
        })
    }

    /// Avance le player selon le temps écoulé depuis le 1er `tick`.
    /// Décode chaque record dont `elapsed_ms <= now_ms` et le push dans
    /// `out`. Retourne le nombre de snapshots reconstruits.
    pub fn tick(&mut self, out: &mut Vec<PredictionInputs>) -> usize {
        if self.next_index >= self.records.len() {
            if !self.finished_logged {
                info!("demo: lecture terminée");
                self.finished_logged = true;
            }
            return 0;
        }
        let start = *self.start.get_or_insert_with(Instant::now);
        let now_ms = start.elapsed().as_millis().min(u32::MAX as u128) as u32;
        let mut produced = 0;
        while self.next_index < self.records.len() {
            let rec = &self.records[self.next_index];
            if rec.elapsed_ms > now_ms {
                break;
            }
            // Clone payload pour pouvoir muter self.last_full_baseline.
            let payload = rec.payload.clone();
            self.next_index += 1;
            if payload.is_empty() {
                continue;
            }
            match payload[0] {
                TAG_SNAPSHOT => match Snapshot::decode(&payload) {
                    Ok(snap) => {
                        self.last_full_baseline = Some(snap.clone());
                        out.push(PredictionInputs {
                            snapshot: snap,
                            cmds_to_replay: Vec::new(),
                        });
                        produced += 1;
                    }
                    Err(e) => warn!("demo: snapshot malformé: {e}"),
                },
                TAG_SNAPSHOT_DELTA => match SnapshotDelta::decode(&payload) {
                    Ok(delta) => {
                        let Some(baseline) = self.last_full_baseline.as_ref() else {
                            debug!(
                                "demo: delta sans baseline (server_time={}), drop",
                                delta.server_time
                            );
                            continue;
                        };
                        let snap = delta.apply_to_baseline(baseline);
                        out.push(PredictionInputs {
                            snapshot: snap,
                            cmds_to_replay: Vec::new(),
                        });
                        produced += 1;
                    }
                    Err(e) => warn!("demo: delta malformé: {e}"),
                },
                tag => debug!("demo: tag {tag:#x} ignoré (non-snapshot)"),
            }
        }
        produced
    }

    /// Nombre total de records dans la démo (utile pour tests / HUD).
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// `true` quand tous les records ont été émis. L'engine peut alors
    /// proposer de quitter ou de boucler.
    pub fn is_finished(&self) -> bool {
        self.next_index >= self.records.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn unique_tmp_path(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("q3rdm-test-{tag}-{nanos}.q3rdm"));
        p
    }

    /// Build une démo synthétique : header + 2 records (1 full + 1 delta).
    fn make_demo_file(tag: &str) -> PathBuf {
        let mut full = Snapshot::default();
        full.server_time = 100;
        let full_payload = full.encode().expect("encode full");

        let mut later = full.clone();
        later.server_time = 150;
        let delta = SnapshotDelta::compute_diff(&full, &later);
        let delta_payload = delta.encode().expect("encode delta");

        let path = unique_tmp_path(tag);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(DEMO_MAGIC).unwrap();
        f.write_all(&DEMO_VERSION.to_le_bytes()).unwrap();
        // Record 1 : full @ t=0
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&(full_payload.len() as u32).to_le_bytes()).unwrap();
        f.write_all(&full_payload).unwrap();
        // Record 2 : delta @ t=50
        f.write_all(&50u32.to_le_bytes()).unwrap();
        f.write_all(&(delta_payload.len() as u32).to_le_bytes()).unwrap();
        f.write_all(&delta_payload).unwrap();
        f.flush().unwrap();
        path
    }

    #[test]
    fn demo_player_loads_header_and_counts_records() {
        let path = make_demo_file("count");
        let player = DemoPlayer::open(&path).expect("open");
        assert_eq!(player.record_count(), 2);
        assert!(!player.is_finished());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn demo_player_rejects_bad_magic() {
        let path = unique_tmp_path("badmagic");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"NOPE").unwrap();
        f.write_all(&DEMO_VERSION.to_le_bytes()).unwrap();
        f.flush().unwrap();
        let err = DemoPlayer::open(&path).unwrap_err();
        assert!(err.contains("magic"), "msg: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn demo_player_emits_full_then_delta_in_order() {
        let path = make_demo_file("inorder");
        let mut player = DemoPlayer::open(&path).expect("open");
        // 1er tick à t=0 : doit émettre le full (elapsed_ms=0).
        let mut out = Vec::new();
        player.tick(&mut out);
        assert_eq!(out.len(), 1, "premier tick devrait sortir le full");
        assert_eq!(out[0].snapshot.server_time, 100);
        // 2e tick après 80 ms réels : doit émettre le delta (elapsed_ms=50).
        std::thread::sleep(std::time::Duration::from_millis(80));
        out.clear();
        player.tick(&mut out);
        assert_eq!(out.len(), 1, "second tick devrait sortir le delta");
        assert_eq!(out[0].snapshot.server_time, 150);
        assert!(player.is_finished());
        let _ = std::fs::remove_file(&path);
    }
}
