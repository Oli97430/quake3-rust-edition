//! **Map download manager** (v0.9.5++) — catalogue de maps community
//! recommandées + téléchargement HTTP en arrière-plan + vérification
//! d'intégrité SHA256 + extraction PK3 vers `baseq3/`.
//!
//! Architecture :
//!
//! ```text
//!   Catalog (statique, en code) ─┐
//!                                ├─→ MapDownloader::start_job(id)
//!   User cmd "mapdl get <id>"  ──┘            │
//!                                              │ thread spawn
//!                                              ▼
//!                                        DownloadJob
//!                                          ureq::get() → bytes
//!                                          sha2 verify
//!                                          zip extract → baseq3/
//!                                              │
//!                                              ▼
//!                                        progress: Mutex<Status>
//!                                              │
//!                                              ▼
//!                                       Lecture par App tick
//!                                       (HUD progress bar)
//! ```
//!
//! Catalog volontairement minimal — le but est de fournir un
//! framework, pas de hardcoder 100 URLs.  Le user étend en éditant
//! [`Catalog::default_entries`] ou en chargeant un JSON externe via
//! `mapdl loadcatalog <file>` (futur).
//!
//! ⚠️ **Légalité** : seules des maps community libres de
//! redistribution sont à inclure dans le catalogue.  PAS les pak0.pk3
//! id Software (copyright protégé).

use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use tracing::{info, warn};

/// Une entrée du catalogue de maps téléchargeables.  Tous les champs
/// sont optionnels sauf `id`, `name`, `url` — les autres servent à
/// l'UI (preview, validation d'intégrité).
#[derive(Debug, Clone)]
pub struct CatalogEntry {
    /// Identifiant court — utilisé comme nom de fichier de
    /// destination et clé de la commande console.  Doit être
    /// alphanumeric/underscore, pas d'espace.
    pub id: &'static str,
    /// Nom affiché dans l'UI.
    pub name: &'static str,
    /// URL HTTP(S) directe vers le `.pk3`.
    pub url: &'static str,
    /// SHA256 du `.pk3` attendu (en hex lowercase).  `None` =
    /// vérification skip (déconseillé en prod, OK en dev).
    pub sha256: Option<&'static str>,
    /// Taille attendue en bytes.  Sert à valider et afficher le
    /// download progress en %.
    pub size_bytes: Option<u64>,
    /// Auteur, pour les credits.
    pub author: Option<&'static str>,
    /// 1-line description.
    pub description: Option<&'static str>,
}

/// État courant d'un job de téléchargement.  Lu par l'App tick pour
/// afficher le HUD de progression.
#[derive(Debug, Clone)]
pub enum DownloadStatus {
    /// Pas de job actif.
    Idle,
    /// Téléchargement en cours.  `received` / `total` en bytes.
    Downloading { id: String, received: u64, total: u64 },
    /// Vérification SHA256 (post-DL, pré-extract).
    Verifying { id: String },
    /// Extraction du PK3 (zip) vers `baseq3/`.
    Extracting { id: String },
    /// Terminé avec succès.  `path` = `.pk3` final sur disque.
    Done { id: String, path: PathBuf },
    /// Erreur — message + id pour identifier la map qui a échoué.
    Error { id: String, message: String },
}

impl DownloadStatus {
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Downloading { .. } | Self::Verifying { .. } | Self::Extracting { .. })
    }
}

/// Dispatcher des téléchargements de maps.  Un seul job en parallèle
/// (queue série) pour éviter de saturer la bande passante du joueur
/// + simplifier l'UI.  Si l'utilisateur en lance un 2e pendant un
/// 1er, le 2e attend dans `pending`.
pub struct MapDownloader {
    /// Statut du job courant (lu par l'App pour le HUD).
    pub status: Arc<Mutex<DownloadStatus>>,
    /// Dossier où placer les `.pk3` téléchargés.
    pub destination: PathBuf,
    /// Catalog résolu au boot.
    pub catalog: Vec<CatalogEntry>,
}

impl MapDownloader {
    pub fn new(destination: PathBuf) -> Self {
        Self {
            status: Arc::new(Mutex::new(DownloadStatus::Idle)),
            destination,
            catalog: Self::default_catalog(),
        }
    }

    /// Catalogue **par défaut** (en code) — entrées minimales pour
    /// démarrer.  Le projet peut étendre via JSON externe (TODO).
    ///
    /// **Note** : URLs sont des placeholders communautaires — vérifier
    /// la disponibilité avant de release.  En cas d'URL morte, le
    /// download retournera une erreur claire dans le HUD.
    pub fn default_catalog() -> Vec<CatalogEntry> {
        vec![
            CatalogEntry {
                id: "aerowalk",
                name: "Aerowalk (hub3aerowalk)",
                url: "https://ws.q3df.org/maps/downloads/hub3aerowalk.pk3",
                sha256: None,
                size_bytes: None,
                author: Some("Hoony, port hub"),
                description: Some("Iconic 1v1 duel — 3 levels of vertical play"),
            },
            CatalogEntry {
                id: "cure",
                name: "Cure (CPMA)",
                url: "https://ws.q3df.org/maps/downloads/cure.pk3",
                sha256: None,
                size_bytes: None,
                author: Some("preacher"),
                description: Some("Tournament 1v1 standard, asymmetric flow"),
            },
            CatalogEntry {
                id: "ztn3dm2",
                name: "ZTN3DM2 (Toxicity)",
                url: "https://ws.q3df.org/maps/downloads/ztn3dm2.pk3",
                sha256: None,
                size_bytes: None,
                author: Some("ztn"),
                description: Some("FFA / Tourney, gothic toxic mood"),
            },
            CatalogEntry {
                id: "pukka3tourney2",
                name: "Pukka3Tourney2",
                url: "https://ws.q3df.org/maps/downloads/pukka3tourney2.pk3",
                sha256: None,
                size_bytes: None,
                author: Some("Pukka"),
                description: Some("Visually striking tourney map"),
            },
            CatalogEntry {
                id: "lostworld",
                name: "Lost World",
                url: "https://ws.q3df.org/maps/downloads/lostworld.pk3",
                sha256: None,
                size_bytes: None,
                author: Some("charon"),
                description: Some("Open outdoor with epic scale"),
            },
        ]
    }

    /// Lance un téléchargement en thread séparé.  Pas de queue —
    /// si un job tourne déjà, log un warn et ignore le request.
    pub fn start(&self, entry_id: &str) -> bool {
        // **Atomic check-and-set** (v0.9.5++ fix) — on prend le lock,
        // teste idle, ET pose le statut "Downloading" AVANT de relâcher
        // le lock.  Sans ça, deux appels concurrents pouvaient passer
        // la garde puis spawn deux threads (le 2e perdait son slot).
        {
            let mut st = self.status.lock();
            if st.is_active() {
                warn!("mapdl: job déjà actif, attente terminée requise");
                return false;
            }
            // Pré-pose un état "Downloading" placeholder pour bloquer
            // les appels concurrents avant le spawn du thread.
            *st = DownloadStatus::Downloading {
                id: entry_id.to_string(),
                received: 0,
                total: 0,
            };
        }
        // Cherche l'entrée dans le catalogue.
        let Some(entry) = self.catalog.iter().find(|e| e.id == entry_id).cloned() else {
            // Réinitialise le statut puisqu'on ne lance rien.
            *self.status.lock() = DownloadStatus::Idle;
            warn!("mapdl: id inconnu `{}` (utilise `mapdl list`)", entry_id);
            return false;
        };
        let status = self.status.clone();
        let dest = self.destination.clone();
        let entry_id_log = entry.id; // copie pour le log post-spawn
        // **Spawn safe** — pas d'`expect`, on log et reset en cas
        // d'échec OS (table threads pleine, etc.).
        let spawn_result = thread::Builder::new()
            .name(format!("mapdl-{}", entry.id))
            .spawn(move || {
                if let Err(e) = run_download(&entry, &dest, &status) {
                    let mut st = status.lock();
                    *st = DownloadStatus::Error {
                        id: entry.id.to_string(),
                        message: e,
                    };
                    warn!("mapdl: échec `{}`: {}", entry.id, st_summary(&st));
                }
            });
        if let Err(e) = spawn_result {
            *self.status.lock() = DownloadStatus::Error {
                id: entry_id_log.to_string(),
                message: format!("thread spawn: {}", e),
            };
            warn!("mapdl: spawn thread KO `{}`: {}", entry_id_log, e);
            return false;
        }
        info!("mapdl: download `{}` lancé", entry_id_log);
        true
    }

    /// Status snapshot — utilisé par l'App tick pour le HUD.
    pub fn status_snapshot(&self) -> DownloadStatus {
        self.status.lock().clone()
    }

    /// Liste compacte du catalogue pour la commande `mapdl list`.
    pub fn list_for_console(&self) -> Vec<String> {
        self.catalog.iter()
            .map(|e| format!("  {} — {} ({})",
                e.id, e.name, e.author.unwrap_or("?")))
            .collect()
    }
}

/// Helper court pour log d'erreur.
fn st_summary(st: &DownloadStatus) -> String {
    match st {
        DownloadStatus::Error { message, .. } => message.clone(),
        _ => format!("{:?}", st),
    }
}

/// Plafond de taille d'un PK3 téléchargé — empêche un serveur
/// malveillant de servir un payload géant qui ferait OOM le process.
/// 100 MB est largement au-dessus des plus gros PK3 community
/// (les Q3 maps les plus chargées font ~30 MB).
const MAX_PK3_BYTES: u64 = 100 * 1024 * 1024;

/// Exécute un téléchargement complet : fetch HTTP → vérif SHA256 →
/// extract ZIP → place dans `dest/<id>.pk3`.  Met à jour le status
/// arc à chaque étape.
fn run_download(
    entry: &CatalogEntry,
    dest: &std::path::Path,
    status: &Arc<Mutex<DownloadStatus>>,
) -> Result<(), String> {
    use std::io::Read;

    // ─── Étape 1 : fetch HTTP ─────────────────────────────────────
    {
        let mut st = status.lock();
        *st = DownloadStatus::Downloading {
            id: entry.id.to_string(),
            received: 0,
            total: entry.size_bytes.unwrap_or(0),
        };
    }

    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(60))
        .build();
    let resp = agent.get(entry.url).call()
        .map_err(|e| format!("HTTP error: {}", e))?;

    if resp.status() != 200 {
        return Err(format!("HTTP status {}", resp.status()));
    }

    // Read total size depuis Content-Length (override entry.size_bytes
    // si celui-ci était None).
    let total = resp.header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok())
        .or(entry.size_bytes)
        .unwrap_or(0);

    // **Anti-DoS** — refuse les payloads géants annoncés.
    if total > MAX_PK3_BYTES {
        return Err(format!(
            "Content-Length {} dépasse le plafond {} bytes",
            total, MAX_PK3_BYTES
        ));
    }

    let mut reader = resp.into_reader();
    // Cap la pré-allocation à 16 MB max — évite qu'un serveur ment
    // dans Content-Length pour pré-allouer 4 GB côté client.
    let prealloc = (total.min(16 * 1024 * 1024)) as usize;
    let mut bytes = Vec::with_capacity(prealloc);
    let mut buf = [0u8; 65536]; // 64 KB chunks
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("read: {}", e))?;
        if n == 0 { break; }
        bytes.extend_from_slice(&buf[..n]);
        // **Anti-DoS streaming** — tronque dès qu'on dépasse le cap,
        // protège contre les serveurs qui mentent dans Content-Length.
        if bytes.len() as u64 > MAX_PK3_BYTES {
            return Err(format!(
                "payload dépasse cap {} bytes (serveur mensonge ?)",
                MAX_PK3_BYTES
            ));
        }
        let mut st = status.lock();
        *st = DownloadStatus::Downloading {
            id: entry.id.to_string(),
            received: bytes.len() as u64,
            total,
        };
    }

    // ─── Étape 2 : vérif SHA256 (si fourni) ───────────────────────
    {
        let mut st = status.lock();
        *st = DownloadStatus::Verifying { id: entry.id.to_string() };
    }
    if let Some(expected_hex) = entry.sha256 {
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let got = hasher.finalize();
        let got_hex = hex_encode(&got);
        if !got_hex.eq_ignore_ascii_case(expected_hex) {
            return Err(format!(
                "SHA256 mismatch (got {}, expected {})", got_hex, expected_hex
            ));
        }
    }

    // ─── Étape 3 : validation magic ZIP (PK3 = ZIP) ──────────────
    if bytes.len() < 4 || &bytes[..4] != b"PK\x03\x04" {
        return Err(format!(
            "Pas un fichier ZIP/PK3 valide (magic = {:?})",
            &bytes[..4.min(bytes.len())]
        ));
    }

    // ─── Étape 4 : "Extract" — on copie tel quel comme `.pk3` ────
    // Le moteur lit déjà les PK3 via `q3-filesystem`, pas besoin de
    // décompresser le contenu.  Le `Extracting` status est
    // conservé pour homogénéité UI mais c'est juste une copie.
    {
        let mut st = status.lock();
        *st = DownloadStatus::Extracting { id: entry.id.to_string() };
    }
    std::fs::create_dir_all(dest)
        .map_err(|e| format!("create_dir_all: {}", e))?;
    let final_path = dest.join(format!("{}.pk3", entry.id));
    std::fs::write(&final_path, &bytes)
        .map_err(|e| format!("write: {}", e))?;

    // ─── Étape 5 : sanity check ZIP + BSP magic ──────────────────
    {
        let cursor = std::io::Cursor::new(&bytes);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| format!("zip parse: {}", e))?;
        // **BSP magic check** (v0.9.5++ fix) — vérifie qu'au moins un
        // `.bsp` à l'intérieur a la magic IBSP correcte.  Évite de
        // garder un PK3 valide-zip mais avec BSP corrompu (qui crashera
        // à `map <id>` plus tard).
        let mut found_bsp = false;
        for i in 0..archive.len() {
            let mut entry = match archive.by_index(i) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let name = entry.name().to_ascii_lowercase();
            if !name.ends_with(".bsp") { continue; }
            use std::io::Read;
            let mut header = [0u8; 4];
            if entry.read_exact(&mut header).is_ok() && &header == b"IBSP" {
                found_bsp = true;
                break;
            }
        }
        if !found_bsp {
            return Err("PK3 valide mais aucun BSP IBSP trouvé".into());
        }
    }

    info!("mapdl: `{}` téléchargé OK ({} bytes) → {}",
          entry.id, bytes.len(), final_path.display());
    {
        let mut st = status.lock();
        *st = DownloadStatus::Done {
            id: entry.id.to_string(),
            path: final_path,
        };
    }
    Ok(())
}

/// Helper hex encode (sans dépendre de la crate `hex`).
fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
