//! État monde : BSP chargé + collision + collection d'entités typées.
//!
//! Construit depuis la lump `ENTITIES` de la BSP, complété par les infos
//! géométriques : bounds des entités `model "*N"` résolus depuis
//! `bsp.models[N]`, liste de spawn points déterministes pour le DM, etc.

use crate::entity::{Entity, EntityId, EntityKind};
use hashbrown::HashMap;
use q3_bsp::Bsp;
use q3_collision::CollisionWorld;
use q3_math::{Aabb, Angles, Vec3};
use tracing::{debug, info};

/// Spawn point candidat pour le matchmaker DM.
#[derive(Debug, Clone, Copy)]
pub struct SpawnPoint {
    pub origin: Vec3,
    pub angles: Angles,
}

pub struct World {
    pub collision: CollisionWorld,
    pub entities: Vec<Entity>,
    /// Premier `info_player_start` rencontré (spawn SP).
    pub player_start: Option<Vec3>,
    pub player_start_angles: Angles,
    /// Tous les `info_player_deathmatch` — pour le mode DM.
    pub spawn_points: Vec<SpawnPoint>,
}

impl World {
    #[inline]
    pub fn bsp(&self) -> &Bsp {
        &self.collision.bsp
    }

    /// Construit un monde à partir d'une map BSP déjà chargée en mémoire.
    pub fn from_bsp(bsp: Bsp) -> Self {
        let raw_entities = bsp.parse_entities();
        info!("world: {} entities dans la BSP", raw_entities.len());

        let mut entities = Vec::with_capacity(raw_entities.len());
        let mut player_start: Option<Vec3> = None;
        let mut player_start_angles = Angles::ZERO;
        let mut spawn_points = Vec::new();
        let mut per_kind: HashMap<String, usize> = HashMap::new();

        for (i, ent) in raw_entities.iter().enumerate() {
            let classname = ent.classname();
            *per_kind.entry(classname.to_string()).or_insert(0) += 1;

            let kind = EntityKind::from_classname(classname);
            let mut e = Entity::new(EntityId(i as u32), kind.clone());

            if let Some(origin) = ent.vec3("origin") {
                e.origin = Vec3::from_array(origin);
            }
            if let Some(angle) = ent.f32("angle") {
                e.angles = Angles::new(0.0, angle, 0.0);
            }
            e.targetname = ent.get("targetname").map(String::from);
            e.target = ent.get("target").map(String::from);
            e.model = ent.get("model").map(String::from);

            // Résolution du `model "*N"` → bounds du sous-modèle BSP
            // correspondant. C'est utile pour les triggers / portes.
            if let Some(model) = e.model.as_deref() {
                if let Some(bounds) = inline_model_bounds(&bsp, model) {
                    e.bounds = bounds;
                }
            }

            // Spawn points.
            match kind {
                EntityKind::InfoPlayerStart => {
                    if player_start.is_none() {
                        player_start = Some(e.origin);
                        player_start_angles = e.angles;
                    }
                }
                EntityKind::InfoPlayerDeathmatch => {
                    spawn_points.push(SpawnPoint {
                        origin: e.origin,
                        angles: e.angles,
                    });
                    if player_start.is_none() {
                        player_start = Some(e.origin);
                        player_start_angles = e.angles;
                    }
                }
                _ => {}
            }

            // Copie des KV restants dans `extra`.
            for kv in &ent.kvs {
                match kv.key.as_str() {
                    "classname" | "origin" | "angle" | "targetname" | "target" | "model" => {}
                    _ => e.extra.push((kv.key.clone(), kv.value.clone())),
                }
            }

            entities.push(e);
        }

        // Log des types les plus fréquents pour le debug.
        let mut kinds: Vec<(&String, &usize)> = per_kind.iter().collect();
        kinds.sort_by(|a, b| b.1.cmp(a.1));
        for (name, count) in kinds.iter().take(8) {
            debug!("world entity: {:<28} ×{}", name, count);
        }
        info!(
            "world: {} spawn points DM, player_start = {:?}",
            spawn_points.len(),
            player_start.unwrap_or(Vec3::ZERO)
        );

        Self {
            collision: CollisionWorld::new(bsp),
            entities,
            player_start,
            player_start_angles,
            spawn_points,
        }
    }

    pub fn find_by_targetname<'a>(
        &'a self,
        name: &'a str,
    ) -> impl Iterator<Item = &'a Entity> + 'a {
        self.entities
            .iter()
            .filter(move |e| e.targetname.as_deref() == Some(name))
    }

    /// Filtre les entités par `EntityKind`.
    pub fn entities_of_kind<'a>(
        &'a self,
        kind: &'a EntityKind,
    ) -> impl Iterator<Item = &'a Entity> + 'a {
        self.entities.iter().filter(move |e| &e.kind == kind)
    }

    /// Spawn DM pseudo-aléatoire, déterministe pour un `seed` donné.
    ///
    /// Retourne `None` si aucun spawn point DM n'est défini (on fallback sur
    /// `player_start`).
    pub fn pick_spawn(&self, seed: u64) -> Option<SpawnPoint> {
        if self.spawn_points.is_empty() {
            return self.player_start.map(|origin| SpawnPoint {
                origin,
                angles: self.player_start_angles,
            });
        }
        let idx = (seed as usize) % self.spawn_points.len();
        Some(self.spawn_points[idx])
    }

    /// Nombre d'entités d'un kind donné, pour stats/debug.
    pub fn count_of(&self, kind: &EntityKind) -> usize {
        self.entities_of_kind(kind).count()
    }
}

/// Pour un champ `model` de la forme `"*N"`, retourne les bounds du
/// sous-modèle BSP correspondant. Utilisé pour les entités-brush (portes,
/// triggers, ascenseurs).
fn inline_model_bounds(bsp: &Bsp, model: &str) -> Option<Aabb> {
    let rest = model.strip_prefix('*')?;
    let idx: usize = rest.parse().ok()?;
    let m = bsp.models.get(idx)?;
    Some(Aabb::new(Vec3::from_array(m.mins), Vec3::from_array(m.maxs)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use q3_bsp::raw::{
        DBrush, DBrushSide, DLeaf, DModel, DNode, DPlane, DShader, DSurface, DrawVert,
    };
    use q3_bsp::Visibility;

    fn cube_bsp_with_entities(text: &str) -> Bsp {
        Bsp {
            entities: text.to_string(),
            shaders: vec![DShader {
                shader: [0; 64],
                surface_flags: 0,
                content_flags: 1,
            }],
            planes: vec![
                DPlane { normal: [1.0, 0.0, 0.0], dist: 16.0 },
                DPlane { normal: [-1.0, 0.0, 0.0], dist: 16.0 },
                DPlane { normal: [0.0, 1.0, 0.0], dist: 16.0 },
                DPlane { normal: [0.0, -1.0, 0.0], dist: 16.0 },
                DPlane { normal: [0.0, 0.0, 1.0], dist: 16.0 },
                DPlane { normal: [0.0, 0.0, -1.0], dist: 16.0 },
            ],
            nodes: vec![DNode {
                plane_num: 0,
                children: [-1, -1],
                mins: [-16, -16, -16],
                maxs: [16, 16, 16],
            }],
            leafs: vec![DLeaf {
                cluster: 0,
                area: 0,
                mins: [-16, -16, -16],
                maxs: [16, 16, 16],
                first_leaf_surface: 0,
                num_leaf_surfaces: 0,
                first_leaf_brush: 0,
                num_leaf_brushes: 1,
            }],
            leaf_surfaces: vec![],
            leaf_brushes: vec![0],
            models: vec![DModel {
                mins: [-16.0; 3],
                maxs: [16.0; 3],
                first_surface: 0,
                num_surfaces: 0,
                first_brush: 0,
                num_brushes: 1,
            }],
            brushes: vec![DBrush {
                first_side: 0,
                num_sides: 6,
                shader_num: 0,
            }],
            brush_sides: (0..6)
                .map(|i| DBrushSide {
                    plane_num: i,
                    shader_num: 0,
                })
                .collect(),
            draw_verts: Vec::<DrawVert>::new(),
            draw_indexes: vec![],
            fogs: vec![],
            surfaces: Vec::<DSurface>::new(),
            lightmap_bytes: vec![],
            lightgrid_bytes: vec![],
            visibility: Visibility::default(),
        }
    }

    #[test]
    fn world_gathers_dm_spawn_points() {
        let ents = r#"
            { "classname" "worldspawn" }
            { "classname" "info_player_deathmatch" "origin" "100 200 32" "angle" "90" }
            { "classname" "info_player_deathmatch" "origin" "-50 0 64" "angle" "0" }
            { "classname" "weapon_rocketlauncher" "origin" "0 0 0" }
        "#;
        let bsp = cube_bsp_with_entities(ents);
        let world = World::from_bsp(bsp);
        assert_eq!(world.spawn_points.len(), 2);
        assert_eq!(world.spawn_points[0].origin, Vec3::new(100.0, 200.0, 32.0));
        // pick_spawn doit retourner l'un des deux.
        let sp = world.pick_spawn(1).unwrap();
        assert!(sp.origin == world.spawn_points[0].origin || sp.origin == world.spawn_points[1].origin);
    }

    #[test]
    fn inline_model_resolves_bounds() {
        let ents = r#"
            { "classname" "worldspawn" }
            { "classname" "func_door" "model" "*0" "origin" "0 0 0" }
        "#;
        let bsp = cube_bsp_with_entities(ents);
        let world = World::from_bsp(bsp);
        let door = world
            .entities
            .iter()
            .find(|e| matches!(e.kind, EntityKind::FuncDoor))
            .unwrap();
        assert_eq!(door.bounds.mins, Vec3::splat(-16.0));
        assert_eq!(door.bounds.maxs, Vec3::splat(16.0));
    }

    #[test]
    fn trigger_push_keeps_target_and_bounds() {
        let ents = r#"
            { "classname" "worldspawn" }
            { "classname" "target_position" "targetname" "t1" "origin" "0 0 512" }
            { "classname" "trigger_push" "model" "*0" "target" "t1" }
            { "classname" "trigger_teleport" "model" "*0" "target" "t1" }
        "#;
        let bsp = cube_bsp_with_entities(ents);
        let world = World::from_bsp(bsp);
        let push = world
            .entities
            .iter()
            .find(|e| matches!(e.kind, EntityKind::TriggerPush))
            .expect("trigger_push manquant");
        assert_eq!(push.target.as_deref(), Some("t1"));
        // Bounds résolues depuis le sous-modèle BSP (cube ±16).
        assert_eq!(push.bounds.mins, Vec3::splat(-16.0));
        assert_eq!(push.bounds.maxs, Vec3::splat(16.0));

        let tp = world
            .entities
            .iter()
            .find(|e| matches!(e.kind, EntityKind::TriggerTeleport))
            .expect("trigger_teleport manquant");
        assert_eq!(tp.target.as_deref(), Some("t1"));

        // Et la cible est trouvable par find_by_targetname.
        let dst = world.find_by_targetname("t1").next().unwrap();
        assert_eq!(dst.origin, Vec3::new(0.0, 0.0, 512.0));
    }
}
