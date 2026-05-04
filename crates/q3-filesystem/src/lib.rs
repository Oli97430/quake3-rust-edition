//! Virtual File System à la Quake 3.
//!
//! # Concept
//!
//! Le moteur Q3 lit ses assets à partir d'un répertoire de base (ex.
//! `~/.q3a/baseq3/`) qui contient :
//!
//! * des fichiers « loose » (maps non packagées, configs)
//! * des archives `.pk3` — en fait des fichiers **ZIP** renommés, contenant
//!   `maps/*.bsp`, `textures/`, `sound/`, `scripts/*.shader`, etc.
//!
//! La recherche se fait dans cet ordre (du plus récent au plus ancien) :
//! 1. fichiers loose du répertoire mod, puis de baseq3
//! 2. archives `.pk3` triées par ordre alphabétique inverse
//!
//! Les chemins sont normalisés avec `/` (pas `\`) et en lowercase.
//!
//! # Différences vs C original
//!
//! * Pas de `FILE*` — on renvoie des `Vec<u8>` ou un reader
//! * Thread-safe (plusieurs threads peuvent lire simultanément)
//! * Zéro allocation inutile : le décompresseur zip écrit dans un buffer
//!   préalloué à la taille non-compressée

use hashbrown::HashMap;
use parking_lot::Mutex;
use q3_common::{Error, Result};
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Entrée dans l'index : référence vers une archive + nom interne, ou un
/// fichier loose.
#[derive(Debug, Clone)]
enum Entry {
    Loose(PathBuf),
    Pk3 { archive: usize, name: String, size: u64 },
}

struct Pk3Archive {
    path: PathBuf,
    data: Vec<u8>,
}

/// Virtual file system.
#[derive(Clone)]
pub struct Vfs {
    inner: Arc<VfsInner>,
}

struct VfsInner {
    /// Liste d'archives chargées en mémoire.
    archives: Vec<Pk3Archive>,
    /// Index : chemin normalisé → entrée la plus prioritaire.
    index: HashMap<String, Entry>,
    /// Sérialise l'accès aux readers `ZipArchive` (non-Sync).
    archive_lock: Mutex<()>,
}

impl Vfs {
    /// Construit un VFS en scannant `base_dir/baseq3/` plus chaque dossier mod.
    ///
    /// * `base_dir` : racine du jeu (ex. `~/.q3a/` ou dossier d'install)
    /// * `mods` : noms des mods à monter par-dessus baseq3 (ex. `["osp"]`)
    pub fn mount(base_dir: impl AsRef<Path>, mods: &[&str]) -> Result<Self> {
        let base_dir = base_dir.as_ref();
        let mut archives: Vec<Pk3Archive> = Vec::new();
        let mut index: HashMap<String, Entry> = HashMap::new();

        // Ordre : baseq3 d'abord (priorité basse), puis chaque mod (écrase).
        let mut roots: Vec<PathBuf> = vec![base_dir.join("baseq3")];
        for m in mods {
            roots.push(base_dir.join(m));
        }

        for root in &roots {
            if !root.is_dir() {
                debug!("vfs: skipping missing dir {}", root.display());
                continue;
            }
            scan_root(root, &mut archives, &mut index)?;
        }

        // **Project assets/** (v0.9.5++) — racine additionnelle relative
        // au CWD pour permettre l'override des assets baseq3 sans toucher
        // aux pak0.pk3 originaux.  Indexé en DERNIER → priorité maximum
        // (les loose files écrasent les pk3 baseq3 + mods).  Permet par
        // ex. d'avoir une `assets/textures/env/skybox_custom_*.tga` qui
        // override le sky shader d'une map.
        let project_assets = PathBuf::from("assets");
        if project_assets.is_dir() {
            info!("vfs: ajout root projet `assets/` (override loose files)");
            scan_root(&project_assets, &mut archives, &mut index)?;
        }

        info!(
            "vfs: mounted {} archives, {} indexed files",
            archives.len(),
            index.len()
        );

        Ok(Self {
            inner: Arc::new(VfsInner {
                archives,
                index,
                archive_lock: Mutex::new(()),
            }),
        })
    }

    /// Crée un VFS vide — utile pour les tests.
    pub fn empty() -> Self {
        Self {
            inner: Arc::new(VfsInner {
                archives: Vec::new(),
                index: HashMap::new(),
                archive_lock: Mutex::new(()),
            }),
        }
    }

    /// Lit un fichier par son chemin logique (ex. `"maps/q3dm1.bsp"`).
    /// Case-insensitive, accepte `/` ou `\`.
    pub fn read(&self, path: &str) -> Result<Vec<u8>> {
        let key = normalize_path(path);
        let entry = self.inner.index.get(&key).ok_or_else(|| {
            Error::fs(format!("file not found: {path}"))
        })?;

        match entry {
            Entry::Loose(p) => {
                fs::read(p).map_err(Error::from)
            }
            Entry::Pk3 { archive, name, size } => {
                let pk3 = &self.inner.archives[*archive];
                let _guard = self.inner.archive_lock.lock();
                let cursor = Cursor::new(&pk3.data);
                let mut zip = zip::ZipArchive::new(cursor).map_err(|e| {
                    Error::archive(format!("{}: {e}", pk3.path.display()))
                })?;
                let mut f = zip.by_name(name).map_err(|e| {
                    Error::archive(format!("{}/{name}: {e}", pk3.path.display()))
                })?;
                let mut out = Vec::with_capacity(*size as usize);
                f.read_to_end(&mut out)?;
                Ok(out)
            }
        }
    }

    /// `true` si `path` existe dans le VFS.
    pub fn exists(&self, path: &str) -> bool {
        self.inner.index.contains_key(&normalize_path(path))
    }

    /// Liste tous les fichiers dont le chemin commence par `prefix`.
    pub fn list_prefix(&self, prefix: &str) -> Vec<String> {
        let p = normalize_path(prefix);
        let mut out: Vec<_> = self
            .inner
            .index
            .keys()
            .filter(|k| k.starts_with(&p))
            .cloned()
            .collect();
        out.sort();
        out
    }

    /// Liste tous les fichiers avec un suffixe donné (ex. `".bsp"`).
    pub fn list_suffix(&self, suffix: &str) -> Vec<String> {
        let s = suffix.to_lowercase();
        let mut out: Vec<_> = self
            .inner
            .index
            .keys()
            .filter(|k| k.ends_with(&s))
            .cloned()
            .collect();
        out.sort();
        out
    }

    pub fn file_count(&self) -> usize {
        self.inner.index.len()
    }

    pub fn archive_count(&self) -> usize {
        self.inner.archives.len()
    }
}

fn normalize_path(p: &str) -> String {
    p.replace('\\', "/").trim_start_matches('/').to_lowercase()
}

fn scan_root(
    root: &Path,
    archives: &mut Vec<Pk3Archive>,
    index: &mut HashMap<String, Entry>,
) -> Result<()> {
    info!("vfs: scanning {}", root.display());

    // 1. indexer les fichiers loose
    scan_loose_dir(root, root, index)?;

    // 2. trier les pk3 par nom DÉCROISSANT (pakN > pak0)
    let mut pk3s: Vec<_> = fs::read_dir(root)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("pk3")))
        .collect();
        pk3s.sort_by(|a, b| b.cmp(a));

    for path in pk3s {
        let data = match fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                warn!("vfs: cannot read {}: {e}", path.display());
                continue;
            }
        };
        let archive_idx = archives.len();
        let zip = zip::ZipArchive::new(Cursor::new(&data)).map_err(|e| {
            Error::archive(format!("{}: {e}", path.display()))
        })?;
        let mut file_count = 0usize;
        for i in 0..zip.len() {
            // On récupère juste les métadonnées ici — pas de lecture.
            let mut zip_copy = zip::ZipArchive::new(Cursor::new(&data)).map_err(|e| {
                Error::archive(format!("{}: {e}", path.display()))
            })?;
            let f = match zip_copy.by_index(i) {
                Ok(f) => f,
                Err(_) => continue,
            };
            if f.is_dir() {
                continue;
            }
            let name = f.name().to_string();
            let key = normalize_path(&name);
            if key.is_empty() {
                continue;
            }
            // N'écrase pas une loose déjà indexée.
            index.entry(key).or_insert(Entry::Pk3 {
                archive: archive_idx,
                name,
                size: f.size(),
            });
            file_count += 1;
        }
        debug!("vfs: loaded {} ({} entries)", path.display(), file_count);
        archives.push(Pk3Archive { path, data });
    }

    Ok(())
}

fn scan_loose_dir(
    root: &Path,
    dir: &Path,
    index: &mut HashMap<String, Entry>,
) -> Result<()> {
    let mut visited: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    scan_loose_dir_inner(root, dir, index, &mut visited, 0)
}

/// Implementation interne avec **détection de cycle** (junctions
/// Windows / symlinks Unix qui pointent en boucle).  Sans cette garde,
/// un junction `assets/foo → assets/` provoquait un stack overflow.
fn scan_loose_dir_inner(
    root: &Path,
    dir: &Path,
    index: &mut HashMap<String, Entry>,
    visited: &mut std::collections::HashSet<PathBuf>,
    depth: usize,
) -> Result<()> {
    // Garde de profondeur (16 niveaux suffisent pour tout layout sain).
    if depth > 16 {
        warn!("vfs: scan profondeur >16 à {} — skip (cycle suspecté)", dir.display());
        return Ok(());
    }
    // Canonicalise pour résoudre symlinks/junctions et détecter cycle.
    let canon = match dir.canonicalize() {
        Ok(c) => c,
        Err(_) => dir.to_path_buf(), // fallback si impossible
    };
    if !visited.insert(canon) {
        return Ok(()); // déjà vu → cycle, on skip
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            scan_loose_dir_inner(root, &path, index, visited, depth + 1)?;
        } else if ft.is_file() {
            if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("pk3")) {
                continue; // géré séparément
            }
            if let Ok(rel) = path.strip_prefix(root) {
                let key = normalize_path(&rel.to_string_lossy());
                // loose ÉCRASE pk3 (priorité haute)
                index.insert(key, Entry::Loose(path));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize() {
        assert_eq!(normalize_path("Maps\\Q3DM1.BSP"), "maps/q3dm1.bsp");
        assert_eq!(normalize_path("/maps/test"), "maps/test");
    }

    #[test]
    fn empty_vfs() {
        let v = Vfs::empty();
        assert!(!v.exists("maps/q3dm1.bsp"));
        assert_eq!(v.file_count(), 0);
    }
}
