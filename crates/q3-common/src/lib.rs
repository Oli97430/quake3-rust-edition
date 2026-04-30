//! Types et services partagés par le moteur : erreurs, cvars, commandes console,
//! logger.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

pub mod cmd;
pub mod console;
pub mod cvar;
pub mod error;
pub mod log;

pub use error::{Error, Result};

/// Version du moteur, affichée dans la console au démarrage.
pub const ENGINE_NAME: &str = "Quake 3 RUST EDITION";
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Protocol version réseau. On garde la valeur historique de Q3 (68) pour
/// rester compatible avec les serveurs `ioquake3` existants si on implémente
/// un jour le netcode legacy. Un protocole "rust" (100) sera ajouté en
/// parallèle pour les améliorations non-wire-compatibles.
pub const PROTOCOL_VERSION_LEGACY: u32 = 68;
pub const PROTOCOL_VERSION_RUST: u32 = 100;
