//! Entités de jeu. Modèle « type enum + données » plutôt que l'héritage OO
//! du C++ pour rester idiomatique Rust.

use q3_math::{Aabb, Angles, Vec3};

/// Identifiant stable d'une entité dans le monde.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId(pub u32);

/// Classe d'entité — correspond aux `classname` de l'éditeur Radiant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityKind {
    Worldspawn,
    InfoPlayerStart,
    InfoPlayerDeathmatch,
    InfoPlayerIntermission,
    Light,
    TargetPosition,
    TriggerMultiple,
    /// Jump pad : envoie le joueur vers son `target` avec une parabole.
    TriggerPush,
    /// Téléporteur : téléporte le joueur à la position/angle de son `target`.
    TriggerTeleport,
    /// Cible de téléportation (aussi utilisée par `trigger_push`).
    MiscTeleporterDest,
    /// Zone qui inflige des dégâts tant que le joueur la touche — lave,
    /// slime, void, arc électrique… Les caractéristiques (`dmg`, `spawnflags`)
    /// sont lues à la charge depuis les KV de l'entité.
    TriggerHurt,
    FuncDoor,
    FuncButton,
    FuncPlat,
    ItemWeapon(String),
    ItemHealth(String),
    ItemArmor(String),
    ItemAmmo(String),
    /// Powerups temporaires (Quad Damage, Haste, Regeneration…). Le
    /// classname original est conservé — l'engine map ça vers un effet
    /// précis et une durée de 30s façon Q3 standard.
    ItemPowerup(String),
    /// Holdables (Medkit, Personal Teleporter). Pris en pickup, stockés
    /// dans un slot d'inventaire et activés à la demande via la touche
    /// `use`. Un seul holdable à la fois, nouveau pickup remplace l'ancien.
    ItemHoldable(String),
    Misc(String),
}

impl EntityKind {
    pub fn from_classname(name: &str) -> Self {
        match name {
            "worldspawn" => Self::Worldspawn,
            "info_player_start" => Self::InfoPlayerStart,
            "info_player_deathmatch" => Self::InfoPlayerDeathmatch,
            "info_player_intermission" => Self::InfoPlayerIntermission,
            "light" => Self::Light,
            "target_position" => Self::TargetPosition,
            "trigger_multiple" => Self::TriggerMultiple,
            "trigger_push" => Self::TriggerPush,
            "trigger_teleport" => Self::TriggerTeleport,
            "misc_teleporter_dest" => Self::MiscTeleporterDest,
            "trigger_hurt" => Self::TriggerHurt,
            "func_door" => Self::FuncDoor,
            "func_button" => Self::FuncButton,
            "func_plat" => Self::FuncPlat,
            n if n.starts_with("weapon_") => Self::ItemWeapon(n.to_string()),
            n if n.starts_with("item_health") => Self::ItemHealth(n.to_string()),
            n if n.starts_with("item_armor") => Self::ItemArmor(n.to_string()),
            n if n.starts_with("ammo_") => Self::ItemAmmo(n.to_string()),
            // Classnames des powerups Q3 connus — on les isole pour les
            // distinguer des items "Misc" banals (func_group, etc).
            "item_quad" | "item_haste" | "item_regen" | "item_invis"
            | "item_flight" | "item_enviro" => Self::ItemPowerup(name.to_string()),
            // Holdables — items consommés sur action du joueur plutôt
            // qu'au ramassage. Q3A en liste deux en base (medkit,
            // teleporter) ; on autorise tout préfixe `holdable_` pour
            // s'adapter aux mods.
            n if n.starts_with("holdable_") => Self::ItemHoldable(n.to_string()),
            other => Self::Misc(other.to_string()),
        }
    }

    /// Pour les items pickupables, retourne le chemin VFS conventionnel du
    /// MD3 correspondant — cf. `g_items.c` dans Q3A source.
    pub fn pickup_model_path(&self) -> Option<&'static str> {
        match self {
            Self::ItemWeapon(n) => weapon_model(n),
            Self::ItemHealth(n) => health_model(n),
            Self::ItemArmor(n) => armor_model(n),
            Self::ItemAmmo(n) => ammo_model(n),
            Self::ItemPowerup(n) => powerup_model(n),
            Self::ItemHoldable(n) => holdable_model(n),
            _ => None,
        }
    }
}

fn weapon_model(name: &str) -> Option<&'static str> {
    Some(match name {
        "weapon_gauntlet" => "models/weapons2/gauntlet/gauntlet.md3",
        "weapon_machinegun" => "models/weapons2/machinegun/machinegun.md3",
        "weapon_shotgun" => "models/weapons2/shotgun/shotgun.md3",
        "weapon_grenadelauncher" => "models/weapons2/grenadel/grenadel.md3",
        "weapon_rocketlauncher" => "models/weapons2/rocketl/rocketl.md3",
        "weapon_lightning" => "models/weapons2/lightning/lightning.md3",
        "weapon_railgun" => "models/weapons2/railgun/railgun.md3",
        "weapon_plasmagun" => "models/weapons2/plasma/plasma.md3",
        "weapon_bfg" => "models/weapons2/bfg/bfg.md3",
        _ => return None,
    })
}

fn health_model(name: &str) -> Option<&'static str> {
    Some(match name {
        "item_health_small" => "models/powerups/health/small_cross.md3",
        "item_health" => "models/powerups/health/medium_cross.md3",
        "item_health_large" => "models/powerups/health/large_cross.md3",
        "item_health_mega" => "models/powerups/health/mega_cross.md3",
        _ => return None,
    })
}

fn armor_model(name: &str) -> Option<&'static str> {
    Some(match name {
        "item_armor_shard" => "models/powerups/armor/shard.md3",
        "item_armor_combat" => "models/powerups/armor/armor_yel.md3",
        "item_armor_body" => "models/powerups/armor/armor_red.md3",
        _ => return None,
    })
}

fn powerup_model(name: &str) -> Option<&'static str> {
    Some(match name {
        "item_quad" => "models/powerups/instant/quad.md3",
        "item_haste" => "models/powerups/instant/haste.md3",
        "item_regen" => "models/powerups/instant/regen.md3",
        "item_invis" => "models/powerups/instant/invis.md3",
        "item_flight" => "models/powerups/instant/flight.md3",
        "item_enviro" => "models/powerups/instant/enviro.md3",
        _ => return None,
    })
}

fn holdable_model(name: &str) -> Option<&'static str> {
    Some(match name {
        // Les holdables Q3 partagent le dossier `models/holdable/`.
        "holdable_medkit" => "models/holdable/medkit.md3",
        "holdable_teleporter" => "models/holdable/teleporter.md3",
        _ => return None,
    })
}

fn ammo_model(name: &str) -> Option<&'static str> {
    Some(match name {
        "ammo_bullets" => "models/powerups/ammo/machinegunam.md3",
        "ammo_shells" => "models/powerups/ammo/shotgunam.md3",
        "ammo_rockets" => "models/powerups/ammo/rocketam.md3",
        "ammo_cells" => "models/powerups/ammo/plasmaam.md3",
        "ammo_slugs" => "models/powerups/ammo/railgunam.md3",
        "ammo_bfg" => "models/powerups/ammo/bfgam.md3",
        "ammo_grenades" => "models/powerups/ammo/grenadeam.md3",
        "ammo_lightning" => "models/powerups/ammo/lightningam.md3",
        _ => return None,
    })
}

/// Données communes à toutes les entités.
#[derive(Debug, Clone)]
pub struct Entity {
    pub id: EntityId,
    pub kind: EntityKind,
    pub origin: Vec3,
    pub angles: Angles,
    pub bounds: Aabb,
    pub targetname: Option<String>,
    pub target: Option<String>,
    pub model: Option<String>,
    /// Paires clé/valeur additionnelles — on garde tout pour ne rien perdre
    /// des entités inconnues.
    pub extra: Vec<(String, String)>,
}

impl Entity {
    pub fn new(id: EntityId, kind: EntityKind) -> Self {
        Self {
            id,
            kind,
            origin: Vec3::ZERO,
            angles: Angles::ZERO,
            bounds: Aabb::EMPTY,
            targetname: None,
            target: None,
            model: None,
            extra: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classnames_map_trigger_push_and_teleport() {
        assert!(matches!(
            EntityKind::from_classname("trigger_push"),
            EntityKind::TriggerPush
        ));
        assert!(matches!(
            EntityKind::from_classname("trigger_teleport"),
            EntityKind::TriggerTeleport
        ));
        assert!(matches!(
            EntityKind::from_classname("misc_teleporter_dest"),
            EntityKind::MiscTeleporterDest
        ));
        assert!(matches!(
            EntityKind::from_classname("trigger_hurt"),
            EntityKind::TriggerHurt
        ));
        // Fallback — classname inconnu conserve le string brut.
        match EntityKind::from_classname("gibberish_xyz") {
            EntityKind::Misc(s) => assert_eq!(s, "gibberish_xyz"),
            other => panic!("attendu Misc, obtenu {other:?}"),
        }
    }

    #[test]
    fn holdables_are_recognized() {
        for name in ["holdable_medkit", "holdable_teleporter"] {
            match EntityKind::from_classname(name) {
                EntityKind::ItemHoldable(n) => assert_eq!(n, name),
                other => panic!("attendu ItemHoldable pour {name}, obtenu {other:?}"),
            }
        }
        // Les holdables exposent un chemin MD3 non-None.
        assert!(EntityKind::ItemHoldable("holdable_medkit".into())
            .pickup_model_path()
            .is_some());
    }
}
