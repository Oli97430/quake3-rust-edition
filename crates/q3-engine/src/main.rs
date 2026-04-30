//! Entrée principale : parse CLI, monte le VFS, ouvre une fenêtre, charge
//! une map si fournie, lance la boucle d'événements.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

mod app;
mod hud_helpers;
mod logo;
mod menu;
mod net;
mod vr;

use anyhow::{Context, Result};
use clap::Parser;
use q3_common::{cvar::CvarRegistry, cmd::CmdRegistry};
use q3_filesystem::Vfs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};
use winit::event_loop::EventLoop;

#[derive(Debug, Parser)]
#[command(
    name = "q3rust",
    version,
    about = "Quake 3 RUST EDITION — port Rust de Quake III Arena"
)]
struct Cli {
    /// Répertoire de base contenant `baseq3/` + pk3. Si absent, on tente
    /// la variable d'environnement `Q3_BASE`, puis on auto-détecte une
    /// install Steam / retail classique.
    #[arg(long)]
    base: Option<PathBuf>,

    /// Mods à monter par-dessus baseq3 (ex: `--mod osp --mod cpma`).
    #[arg(long = "mod")]
    mods: Vec<String>,

    /// Map à charger au démarrage (chemin logique dans le VFS, ex `maps/q3dm1.bsp`).
    #[arg(long)]
    map: Option<String>,

    /// Désactive la fenêtre + rendu (mode dedicated / headless).
    #[arg(long)]
    dedicated: bool,

    /// Largeur initiale de la fenêtre.
    #[arg(long, default_value_t = 1280)]
    width: u32,

    /// Hauteur initiale de la fenêtre.
    #[arg(long, default_value_t = 720)]
    height: u32,

    /// Héberge un serveur multijoueur autoritatif sur `addr:port`
    /// (ex: `--host 0.0.0.0:27960`). Mutuellement exclusif avec
    /// `--connect` — `--connect` l'emporte.
    #[arg(long)]
    host: Option<String>,

    /// Se connecte à un serveur distant (ex: `--connect 127.0.0.1:27960`).
    /// Si l'adresse est invalide, on retombe en solo avec un warning.
    #[arg(long)]
    connect: Option<String>,

    /// Nombre max de clients côté serveur. Ignoré hors mode `--host`.
    #[arg(long, default_value_t = 8)]
    max_clients: u8,

    /// Nombre de bots à spawn côté serveur au démarrage. Utiles pour
    /// remplir un --host de partie de test sans avoir 4 instances
    /// client connectées. Ignoré hors mode `--host`.
    #[arg(long, default_value_t = 0)]
    bots: u8,

    /// Active le chemin VR (OpenXR). Scaffold actuellement — en absence
    /// de runtime XR détecté, retombe en rendu mono sans rien casser.
    #[arg(long)]
    vr: bool,

    /// Connect en spectateur — pas de body dans le monde, immunité aux
    /// dégâts/pickups, juste pour regarder un match en cours. Ignoré
    /// hors mode `--connect`.
    #[arg(long)]
    spectate: bool,

    /// Équipe à rejoindre en mode TDM : `red`, `blue`, `free` (FFA).
    /// Ignoré hors mode `--connect`. Default = `free`.
    #[arg(long)]
    team: Option<String>,

    /// Enregistre une démo (snapshots reçus) dans le fichier passé.
    /// Format simple `Q3RD` v1 : header + records `(elapsed_ms u32,
    /// payload_len u32, bytes)`. Mode client uniquement.
    #[arg(long)]
    record: Option<PathBuf>,

    /// Mode de jeu côté serveur : `ffa` (default) ou `tdm`. Ignoré
    /// hors mode `--host` — c'est le serveur distant qui décide.
    #[arg(long, default_value = "ffa")]
    gametype: String,

    /// Désactive le friendly-fire en TDM. Sans effet en FFA (où il
    /// n'y a pas d'équipe). Ignoré hors mode `--host`.
    #[arg(long)]
    no_friendly_fire: bool,

    /// Lit une démo `.q3rdm` enregistrée auparavant (cf. `--record`).
    /// Mutuellement exclusif avec `--connect` / `--host` ; si présent,
    /// l'emporte sur le solo et le multi.
    #[arg(long)]
    play: Option<PathBuf>,
}

fn main() -> Result<()> {
    q3_common::log::init();

    let cli = Cli::parse();

    info!(
        "{} v{} — Rust {}",
        q3_common::ENGINE_NAME,
        q3_common::ENGINE_VERSION,
        env!("CARGO_PKG_RUST_VERSION")
    );

    // VFS
    let mods: Vec<&str> = cli.mods.iter().map(String::as_str).collect();
    let base = cli.base.clone().unwrap_or_else(resolve_base_dir);
    info!("vfs: base = {}", base.display());
    let vfs = match Vfs::mount(&base, &mods) {
        Ok(v) => v,
        Err(e) => {
            warn!("vfs: impossible de monter {} : {e}", base.display());
            warn!("vfs: fallback sur un VFS vide");
            Vfs::empty()
        }
    };
    info!(
        "vfs: {} fichiers indexés ({} archives)",
        vfs.file_count(),
        vfs.archive_count()
    );

    // Cvars / Cmds (stubs pour l'instant)
    let _cvars = CvarRegistry::new();
    let _cmds = CmdRegistry::new();

    if cli.dedicated {
        info!("mode dedicated — pas de fenêtre");
        return Ok(());
    }

    // Résolution du mode réseau : `--connect` > `--host` > solo. Si
    // l'une des deux adresses est invalide, `from_cli` warn et retombe
    // silencieusement en solo — mieux que de planter sur une typo CLI.
    let gametype = net::GameType::from_cli(&cli.gametype);
    let friendly_fire = !cli.no_friendly_fire;
    let net_mode = if let Some(path) = cli.play.clone() {
        info!("net: mode DEMO PLAYBACK ← {}", path.display());
        net::NetMode::DemoPlayback { path }
    } else {
        net::NetMode::from_cli_full(
            cli.host.as_deref(),
            cli.connect.as_deref(),
            cli.max_clients,
            gametype,
            friendly_fire,
        )
    };

    let event_loop = EventLoop::new().context("create event loop")?;
    let mut app = app::App::new(
        Arc::new(vfs),
        cli.width,
        cli.height,
        cli.map,
        net_mode,
        cli.vr,
        cli.bots,
        cli.spectate,
        cli.team,
        cli.record,
    );
    event_loop
        .run_app(&mut app)
        .context("event loop")?;

    Ok(())
}

/// Résout un répertoire contenant `baseq3/pak0.pk3` quand l'utilisateur
/// ne passe pas `--base`.
///
/// Priorité :
/// 1. Variable d'environnement `Q3_BASE`.
/// 2. Répertoire courant (`.`) si `./baseq3/pak0.pk3` existe.
/// 3. Une liste d'emplacements Steam / retail classiques.
/// 4. `.` en dernier recours — laisse `Vfs::mount` loguer un VFS vide.
fn resolve_base_dir() -> PathBuf {
    if let Ok(env) = std::env::var("Q3_BASE") {
        let p = PathBuf::from(env);
        if has_baseq3(&p) {
            info!("vfs: auto-detect via $Q3_BASE → {}", p.display());
            return p;
        } else {
            warn!("vfs: $Q3_BASE={} ne contient pas baseq3/pak0.pk3", p.display());
        }
    }
    let cwd = PathBuf::from(".");
    if has_baseq3(&cwd) {
        return cwd;
    }
    let candidates = [
        r"C:\SteamLibrary\steamapps\common\Quake 3 Arena",
        r"D:\SteamLibrary\steamapps\common\Quake 3 Arena",
        r"E:\SteamLibrary\steamapps\common\Quake 3 Arena",
        r"F:\SteamLibrary\steamapps\common\Quake 3 Arena",
        r"C:\Program Files (x86)\Steam\steamapps\common\Quake 3 Arena",
        r"C:\Program Files\Steam\steamapps\common\Quake 3 Arena",
        r"C:\Program Files (x86)\Quake III Arena",
        r"C:\Program Files\Quake III Arena",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if has_baseq3(&p) {
            info!("vfs: auto-detect install Q3 → {}", p.display());
            return p;
        }
    }
    warn!("vfs: aucun baseq3/ trouvé — rendu dégradé, utilise --base ou $Q3_BASE");
    PathBuf::from(".")
}

fn has_baseq3(p: &Path) -> bool {
    p.join("baseq3").join("pak0.pk3").is_file()
}
