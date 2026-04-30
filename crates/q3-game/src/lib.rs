//! Logique de jeu Quake 3 — état monde, entités, règles, physique joueur.
//!
//! Pour l'instant : structures de base + un spawner d'entités lisant le
//! lump `LUMP_ENTITIES` de la BSP. La physique / le combat seront ajoutés
//! ensuite.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

pub mod entity;
pub mod health;
pub mod movement;
pub mod world;

pub use entity::{Entity, EntityId, EntityKind};
pub use health::Health;
pub use world::World;
