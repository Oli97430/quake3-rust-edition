//! **CMod** — collision model Quake 3.
//!
//! Reproduit fidèlement l'algorithme de `cm_trace.c` du jeu original :
//!
//! 1. **Bounds check** : AABB du trace vs AABB du monde.
//! 2. **BSP traversal** : on descend l'arbre en subdivisant le trace quand
//!    il chevauche un plan.
//! 3. **Leaf test** : pour chaque brush dans la feuille, test enter/leave
//!    plan par plan.
//! 4. **Brush test** : un brush est convexe, donc le trace y entre à
//!    `t = max(enter_t)` et en sort à `t = min(leave_t)`. Si enter < leave
//!    et enter < `trace.fraction`, mise à jour de la trace.
//!
//! # Améliorations vs C original
//!
//! * **Pas d'unsafe** : l'arbre est parcouru par index, avec `.get(i)` qui
//!   retourne `None` au lieu d'un crash silencieux.
//! * **Pas de globals** : le `CollisionWorld` est auto-contenu.
//! * **Patches en attente** : `TODO` explicite dans le code plutôt qu'un
//!   silence. La collision contre les patches sera ajoutée ensuite.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

pub mod content;
pub mod patch_collide;
pub mod trace;

pub use content::{Contents, SurfaceFlags};
pub use patch_collide::{PatchCollide, PatchFacet};
pub use trace::{Trace, TraceBox};

use q3_bsp::{raw::SurfaceType, Bsp};
use q3_math::{Aabb, Plane, Vec3};
use tracing::debug;

/// Monde pour le collision tracer. Construit à partir d'un `Bsp`.
pub struct CollisionWorld {
    pub bsp: Bsp,
    /// Contenu par brush, pré-calculé depuis les shaders.
    brush_contents: Vec<Contents>,
    /// Bounds pré-calculées par brush.
    brush_bounds: Vec<Aabb>,
    /// Patches collidables. Un patch est ajouté pour chaque surface BSP
    /// de type `Patch` dont le shader déclare un contenu qui intersecte
    /// `MASK_PLAYERSOLID` (typiquement `SOLID`). Les patches sans
    /// contenu (decorations, shader `nonsolid`) ne sont **pas** ajoutés.
    patches: Vec<PatchCollide>,
    /// Map d'index : surface BSP → index dans `patches`, ou `None` si
    /// cette surface n'est pas un patch collidable. Permet à `trace_leaf`
    /// de parcourir `leaf_surfaces` (indexe les surfaces, pas les patches)
    /// et de sauter les surfaces non-patches sans relancer de lookup.
    surface_to_patch: Vec<Option<u32>>,
}

impl CollisionWorld {
    pub fn new(bsp: Bsp) -> Self {
        let brush_contents = bsp
            .brushes
            .iter()
            .map(|b| {
                let shader = bsp.shaders.get(b.shader_num as usize);
                let raw = shader.map(|s| s.content_flags).unwrap_or(0);
                Contents::from_bits_truncate(raw as u32)
            })
            .collect();
        let brush_bounds = bsp
            .brushes
            .iter()
            .map(|b| compute_brush_bounds(&bsp, b.first_side, b.num_sides))
            .collect();

        // Construction des PatchCollide. On itère les surfaces ; pour
        // chaque surface de type Patch dont le shader déclare un contenu
        // solide, on génère le micro-brush triangulaire via
        // `build_patch_collide`. Les patches purement décoratifs
        // (shader sans content_flags SOLID/PLAYERCLIP) sont ignorés pour
        // éviter de tester des flags muraux sur des effets visuels.
        let mut patches: Vec<PatchCollide> = Vec::new();
        let mut surface_to_patch: Vec<Option<u32>> = vec![None; bsp.surfaces.len()];
        let collision_mask = Contents::SOLID
            | Contents::PLAYERCLIP
            | Contents::BODY
            | Contents::MONSTERCLIP;
        for (surf_idx, surf) in bsp.surfaces.iter().enumerate() {
            if surf.kind() != SurfaceType::Patch {
                continue;
            }
            let shader = bsp.shaders.get(surf.shader_num as usize);
            let raw = shader.map(|s| s.content_flags).unwrap_or(0);
            let contents = Contents::from_bits_truncate(raw as u32);
            // Fallback historique : certains patches id (railings, arches)
            // shippent sans content_flags explicites mais doivent quand
            // même collider. On force `SOLID` si la surface n'a pas de
            // flag non-solid explicite. C'est conforme à la logique Q3
            // `cm_patch.c` qui n'inspecte pas `contents` pour décider
            // de construire le patch.
            let effective = if contents.intersects(collision_mask) {
                contents
            } else if !contents.contains(Contents::TRANSLUCENT) {
                Contents::SOLID
            } else {
                continue;
            };
            let start = surf.first_vert as usize;
            let end = start.saturating_add(surf.num_verts as usize);
            let Some(cps) = bsp.draw_verts.get(start..end) else {
                continue;
            };
            let Some(pc) = patch_collide::build_patch_collide(
                cps,
                surf.patch_width,
                surf.patch_height,
                effective,
            ) else {
                continue;
            };
            surface_to_patch[surf_idx] = Some(patches.len() as u32);
            patches.push(pc);
        }

        debug!(
            "collision: {} brushes, {} planes, {} nodes, {} patches ({} facets total)",
            bsp.brushes.len(),
            bsp.planes.len(),
            bsp.nodes.len(),
            patches.len(),
            patches.iter().map(|p| p.facets.len()).sum::<usize>(),
        );
        Self {
            bsp,
            brush_contents,
            brush_bounds,
            patches,
            surface_to_patch,
        }
    }

    /// Nombre total de patches collidables (pour tests et stats).
    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }

    pub fn brush_count(&self) -> usize {
        self.bsp.brushes.len()
    }

    /// Trace une box de `mins..maxs` de `start` à `end` contre le monde,
    /// en ne considérant que les brushes intersectant `mask` dans leur
    /// contenu.
    ///
    /// `Trace::fraction` = 1.0 → pas de collision sur le chemin.
    /// `Trace::fraction` < 1.0 → impact à `lerp(start, end, fraction)`.
    pub fn trace_box(&self, start: Vec3, end: Vec3, box_: TraceBox, mask: Contents) -> Trace {
        let mut work = trace::TraceWork {
            start,
            end,
            bounds: box_,
            mask,
            // `Trace::miss()` initialise déjà `start_solid = false` et
            // `all_solid = false`, ce qu'on veut : ces flags ne doivent
            // passer à `true` que si `trace_brush` détecte explicitement
            // que le point de départ est **à l'intérieur** d'un brush.
            //
            // ⚠️ Ici, on avait un bug historique : on les forçait à `true`
            // en entrée puis on ne les remettait à `false` que si
            // `fraction == 1.0` (pas d'impact du tout). Conséquence :
            // **toute trace qui touchait quelque chose renvoyait `all_solid`,
            // même avec un start parfaitement libre**, ce qui faisait
            // croire au `PlayerMove::update_ground` qu'il était en
            // permanence embedded → push-out en boucle → le joueur
            // grimpait de 4u au spawn, passait en mode "air", et la
            // gravité divergeait sans être clippée par le slide (qui lui
            // aussi voyait `all_solid` et tuait juste velocity.z). Résultat :
            // WASD inerte, joueur flottant.
            trace: Trace::miss(end),
        };

        // Walk BSP depuis le root (node 0).
        if !self.bsp.nodes.is_empty() {
            self.recurse_tree(0, 0.0, 1.0, start, end, &mut work);
        }

        let frac = work.trace.fraction;
        work.trace.end_pos = start + (end - start) * frac;
        work.trace
    }

    pub fn trace_ray(&self, start: Vec3, end: Vec3, mask: Contents) -> Trace {
        self.trace_box(start, end, TraceBox::POINT, mask)
    }

    /// Retourne le `Contents` agrégé de tous les brushs contenant `pos`.
    ///
    /// Utilisé pour détecter si la caméra (ou un point arbitraire) est
    /// **à l'intérieur** d'un volume non-solide : eau, lave, slime, fog,
    /// zone de teleport.  Pour les brushs `SOLID` un point est soit à
    /// l'extérieur soit dans l'épaisseur du mur ; la fonction retourne
    /// l'union pour supporter les volumes superposés (eau dans une
    /// lave, etc.).
    ///
    /// Complexité : O(log N) pour trouver le leaf + O(K) pour tester les
    /// brushs du leaf (K typiquement 1–3).  Appelé 1× par frame côté
    /// engine, coût négligeable.
    pub fn point_contents(&self, pos: Vec3) -> Contents {
        if self.bsp.nodes.is_empty() {
            return Contents::empty();
        }
        // Descente BSP : à chaque node on choisit le fils selon le signe
        // de la distance au plan.  Sur l'égalité on prend arbitrairement
        // le positif — un point pile sur un plan est cas limite.
        let mut node_num: i32 = 0;
        while node_num >= 0 {
            let Some(node) = self.bsp.nodes.get(node_num as usize) else {
                return Contents::empty();
            };
            let Some(plane) = self.bsp.planes.get(node.plane_num as usize) else {
                return Contents::empty();
            };
            let p = Plane {
                normal: Vec3::from_array(plane.normal),
                dist: plane.dist,
            };
            let d = p.distance(pos);
            node_num = if d >= 0.0 {
                node.children[0]
            } else {
                node.children[1]
            };
        }
        // `node_num < 0` → feuille, son index est `-node_num - 1`.  On
        // passe par `checked_neg` + `checked_sub` pour couvrir le cas
        // pathologique `i32::MIN` (dont la négation overflow — bug
        // classique des BSP corrompus qu'on a déjà vu in-the-wild).
        let leaf_idx = match node_num
            .checked_neg()
            .and_then(|n| n.checked_sub(1))
            .and_then(|n| usize::try_from(n).ok())
        {
            Some(i) => i,
            None => return Contents::empty(),
        };
        let Some(leaf) = self.bsp.leafs.get(leaf_idx) else {
            return Contents::empty();
        };
        let start = leaf.first_leaf_brush as usize;
        let end = start.saturating_add(leaf.num_leaf_brushes as usize);
        let Some(brush_ids) = self.bsp.leaf_brushes.get(start..end) else {
            return Contents::empty();
        };
        let mut acc = Contents::empty();
        for &brush_id in brush_ids {
            let bi = brush_id as usize;
            let Some(brush) = self.bsp.brushes.get(bi) else {
                continue;
            };
            // Convex brush : le point est dedans ssi il est du côté
            // négatif (ou sur) de chaque plan de ses sides.
            let sides_start = brush.first_side as usize;
            let sides_end = sides_start.saturating_add(brush.num_sides as usize);
            let Some(sides) = self.bsp.brush_sides.get(sides_start..sides_end) else {
                continue;
            };
            let inside = sides.iter().all(|s| {
                let Some(pl) = self.bsp.planes.get(s.plane_num as usize) else {
                    return false;
                };
                let p = Plane {
                    normal: Vec3::from_array(pl.normal),
                    dist: pl.dist,
                };
                p.distance(pos) <= 0.0
            });
            if inside {
                if let Some(c) = self.brush_contents.get(bi) {
                    acc |= *c;
                }
            }
        }
        acc
    }

    /// Récursif : descend l'arbre BSP. `p1f`, `p2f` = fraction du trace sur
    /// l'entrée et la sortie du node courant (utilisé pour clipper).
    fn recurse_tree(
        &self,
        node_num: i32,
        p1f: f32,
        p2f: f32,
        p1: Vec3,
        p2: Vec3,
        work: &mut trace::TraceWork,
    ) {
        if work.trace.fraction <= p1f {
            // Déjà un impact plus proche, inutile d'aller plus loin.
            return;
        }

        // Index négatif = leaf.  Même garde que `contents_at_point` :
        // `i32::MIN` overflow à la négation, on sort proprement.
        if node_num < 0 {
            let Some(leaf_idx) = node_num
                .checked_neg()
                .and_then(|n| n.checked_sub(1))
                .and_then(|n| usize::try_from(n).ok())
            else {
                return;
            };
            self.trace_leaf(leaf_idx, work);
            return;
        }

        let Some(node) = self.bsp.nodes.get(node_num as usize) else {
            return;
        };
        let Some(plane) = self.bsp.planes.get(node.plane_num as usize) else {
            return;
        };
        let plane = Plane {
            normal: Vec3::from_array(plane.normal),
            dist: plane.dist,
        };

        // Distance signée des deux extrémités du trace au plan, ajustée par
        // la demi-boîte (extension du rayon par l'enveloppe englobante).
        let t1 = plane.distance(p1);
        let t2 = plane.distance(p2);
        let offset = work.bounds.offset_for_plane(plane.normal);

        if t1 >= offset && t2 >= offset {
            // totalement du côté + du plan
            self.recurse_tree(node.children[0], p1f, p2f, p1, p2, work);
            return;
        }
        if t1 < -offset && t2 < -offset {
            self.recurse_tree(node.children[1], p1f, p2f, p1, p2, work);
            return;
        }

        // Le trace traverse le plan : on split à la fraction correspondante.
        let (side_near, frac_enter, frac_leave) = if t1 < t2 {
            let inv = 1.0 / (t1 - t2);
            let fe = (t1 - offset + SPLIT_EPSILON) * inv;
            let fl = (t1 + offset + SPLIT_EPSILON) * inv;
            (1, fe, fl)
        } else if t1 > t2 {
            let inv = 1.0 / (t1 - t2);
            let fe = (t1 + offset - SPLIT_EPSILON) * inv;
            let fl = (t1 - offset - SPLIT_EPSILON) * inv;
            (0, fe, fl)
        } else {
            // t1 == t2 : choisit arbitrairement le 1er côté
            (0, 1.0, 0.0)
        };

        let frac_enter = frac_enter.clamp(0.0, 1.0);
        let frac_leave = frac_leave.clamp(0.0, 1.0);

        // Visite 1st half : du côté `side_near` à `1 - side_near`.
        let mid_enter = p1 + (p2 - p1) * frac_enter;
        let mid_leave = p1 + (p2 - p1) * frac_leave;
        let pf_enter = p1f + (p2f - p1f) * frac_enter;
        let pf_leave = p1f + (p2f - p1f) * frac_leave;

        let child_near = node.children[side_near];
        let child_far = node.children[1 - side_near];

        self.recurse_tree(child_near, p1f, pf_enter, p1, mid_enter, work);
        self.recurse_tree(child_far, pf_leave, p2f, mid_leave, p2, work);
    }

    fn trace_leaf(&self, leaf_idx: usize, work: &mut trace::TraceWork) {
        let Some(leaf) = self.bsp.leafs.get(leaf_idx) else {
            return;
        };
        let start_brush = leaf.first_leaf_brush as usize;
        let end_brush = start_brush + leaf.num_leaf_brushes as usize;
        if let Some(lbrushes) = self.bsp.leaf_brushes.get(start_brush..end_brush) {
            for &brush_ref in lbrushes {
                let brush_idx = brush_ref as usize;
                if brush_idx >= self.bsp.brushes.len() {
                    continue;
                }
                self.trace_brush(brush_idx, work);
            }
        }

        // Patches : `leaf_surfaces` indexe `surfaces`, donc on traduit via
        // `surface_to_patch` (un simple Option<u32> pour sauter les
        // surfaces non-patches en O(1)). On accepte la redondance si un
        // patch traverse plusieurs feuilles : `trace_facet` ne met à jour
        // `work.trace` que si le nouveau `enter_frac` est strictement
        // inférieur à l'actuel, donc un retest ne peut pas dégrader le
        // résultat (juste coûter du CPU). Pour q3dm1-like c'est ~quelques
        // % de facets retestés, accepté pour une v1.
        if self.patches.is_empty() {
            return;
        }
        let start_surf = leaf.first_leaf_surface as usize;
        let end_surf = start_surf + leaf.num_leaf_surfaces as usize;
        if let Some(lsurfs) = self.bsp.leaf_surfaces.get(start_surf..end_surf) {
            for &surf_ref in lsurfs {
                let Some(patch_slot) = self.surface_to_patch.get(surf_ref as usize) else {
                    continue;
                };
                let Some(patch_idx) = patch_slot else {
                    continue;
                };
                self.trace_patch(*patch_idx as usize, work);
            }
        }
    }

    /// Teste une trace contre un `PatchCollide`. Stratégie : early-out sur
    /// bounds globales, puis itère chaque facet.
    fn trace_patch(&self, patch_idx: usize, work: &mut trace::TraceWork) {
        let Some(patch) = self.patches.get(patch_idx) else {
            return;
        };
        if !patch.contents.intersects(work.mask) {
            return;
        }
        if !work.bounds_box_overlaps(&patch.bounds) {
            return;
        }
        for facet in &patch.facets {
            self.trace_facet(facet, patch.contents, work);
        }
    }

    /// Teste une trace contre un facet (prisme triangulaire à 5 plans).
    /// Même logique que `trace_brush` mais itère un tableau de plans
    /// synthétiques plutôt que `bsp.brush_sides`.
    fn trace_facet(
        &self,
        facet: &PatchFacet,
        contents: Contents,
        work: &mut trace::TraceWork,
    ) {
        if !work.bounds_box_overlaps(&facet.bounds) {
            return;
        }

        let planes: [Plane; 5] = [
            facet.face,
            facet.back,
            facet.edges[0],
            facet.edges[1],
            facet.edges[2],
        ];

        // Même algo plan-par-plan que `trace_brush` : on cherche
        // `enter_frac = max(entry_fractions)` et `leave_frac = min(exit_fractions)`,
        // impact valide si `enter < leave` et `enter < work.trace.fraction`.
        let mut enter_frac: f32 = -1.0;
        let mut leave_frac: f32 = 1.0;
        let mut get_out = false;
        let mut start_out = false;
        let mut clip_plane: Option<Plane> = None;

        for plane in &planes {
            let offset = work.bounds.offset_for_plane(plane.normal);
            let dist = plane.dist + offset;
            let d1 = plane.normal.dot(work.start) - dist;
            let d2 = plane.normal.dot(work.end) - dist;

            if d2 > 0.0 {
                get_out = true;
            }
            if d1 > 0.0 {
                start_out = true;
            }

            if d1 > 0.0 && d2 >= d1 {
                // Entièrement à l'extérieur de ce plan → pas d'intersection
                // possible avec le prisme.
                return;
            }
            if d1 <= 0.0 && d2 <= 0.0 {
                continue;
            }

            if d1 > d2 {
                let f = ((d1 - SURFACE_CLIP_EPSILON) / (d1 - d2)).max(0.0);
                if f > enter_frac {
                    enter_frac = f;
                    clip_plane = Some(*plane);
                }
            } else {
                let f = ((d1 + SURFACE_CLIP_EPSILON) / (d1 - d2)).min(1.0);
                if f < leave_frac {
                    leave_frac = f;
                }
            }
        }

        if !start_out {
            // Départ à l'intérieur du slab : on marque `start_solid`, mais
            // on ne clippe pas à fraction=0 comme pour un brush massif. En
            // effet, un patch est un slab mince ; être "dedans" signifie
            // probablement qu'on est à 1 unité près de la surface (bord
            // du joueur qui frôle). Q3 fait pareil dans `CM_TraceThroughPatchCollide`.
            work.trace.start_solid = true;
            if !get_out {
                work.trace.all_solid = true;
                work.trace.fraction = 0.0;
                work.trace.contents = contents;
            }
            return;
        }

        if enter_frac < leave_frac && enter_frac > -1.0 && enter_frac < work.trace.fraction {
            let f = enter_frac.max(0.0);
            work.trace.fraction = f;
            if let Some(p) = clip_plane {
                work.trace.plane_normal = p.normal;
                work.trace.plane_dist = p.dist;
            }
            work.trace.contents = contents;
            // brush_index = None pour un facet (pas un vrai brush BSP).
            work.trace.brush_index = None;
        }
    }

    fn trace_brush(&self, brush_idx: usize, work: &mut trace::TraceWork) {
        let Some(brush) = self.bsp.brushes.get(brush_idx) else {
            return;
        };
        if brush.num_sides <= 0 {
            return;
        }
        let contents = self.brush_contents[brush_idx];
        if !contents.intersects(work.mask) {
            return;
        }
        let brush_bounds = self.brush_bounds[brush_idx];
        // Early out si la trace ne touche pas du tout la boîte du brush.
        if !work.bounds_box_overlaps(&brush_bounds) {
            return;
        }

        // Plan-by-plan : calcul de enter_t (max) et leave_t (min).
        let mut enter_frac: f32 = -1.0;
        let mut leave_frac: f32 = 1.0;
        let mut get_out = false;
        let mut start_out = false;
        let mut clip_plane: Option<Plane> = None;

        let first = brush.first_side as usize;
        let last = first + brush.num_sides as usize;
        for side_i in first..last {
            let Some(side) = self.bsp.brush_sides.get(side_i) else {
                continue;
            };
            let Some(raw_plane) = self.bsp.planes.get(side.plane_num as usize) else {
                continue;
            };
            let plane = Plane {
                normal: Vec3::from_array(raw_plane.normal),
                dist: raw_plane.dist,
            };
            // Étend le plan par la demi-boîte vers l'extérieur du brush,
            // pour pouvoir tester la trace comme un point.
            let offset = work.bounds.offset_for_plane(plane.normal);
            let dist = plane.dist + offset;

            let d1 = plane.normal.dot(work.start) - dist;
            let d2 = plane.normal.dot(work.end) - dist;

            if d2 > 0.0 {
                get_out = true;
            }
            if d1 > 0.0 {
                start_out = true;
            }

            // start et end tous deux à l'extérieur → pas d'impact
            if d1 > 0.0 && d2 >= d1 {
                return;
            }
            // start et end tous deux à l'intérieur → pas d'info utile sur ce plan
            if d1 <= 0.0 && d2 <= 0.0 {
                continue;
            }

            if d1 > d2 {
                // entre par ce plan
                let f = ((d1 - SURFACE_CLIP_EPSILON) / (d1 - d2)).max(0.0);
                if f > enter_frac {
                    enter_frac = f;
                    clip_plane = Some(plane);
                }
            } else {
                // sort par ce plan
                let f = ((d1 + SURFACE_CLIP_EPSILON) / (d1 - d2)).min(1.0);
                if f < leave_frac {
                    leave_frac = f;
                }
            }
        }

        if !start_out {
            // le point de départ est dans le brush
            work.trace.start_solid = true;
            if !get_out {
                work.trace.all_solid = true;
                work.trace.fraction = 0.0;
                work.trace.contents = contents;
            }
            return;
        }

        if enter_frac < leave_frac && enter_frac > -1.0 && enter_frac < work.trace.fraction {
            let f = enter_frac.max(0.0);
            work.trace.fraction = f;
            if let Some(p) = clip_plane {
                work.trace.plane_normal = p.normal;
                work.trace.plane_dist = p.dist;
            }
            work.trace.contents = contents;
            work.trace.brush_index = Some(brush_idx as u32);
        }
    }
}

fn compute_brush_bounds(bsp: &Bsp, first_side: i32, num_sides: i32) -> Aabb {
    // On n'a pas les vertex par brush — on reconstruit l'AABB à partir des
    // plans axis-aligned. Pour chaque axe, on cherche la face `+axis` et la
    // face `-axis`. Si un axe n'a pas les deux, on fallback sur infini.
    let first = first_side as usize;
    let last = first + num_sides as usize;
    let mut maxs = [f32::INFINITY; 3];
    let mut mins = [f32::NEG_INFINITY; 3];
    let mut have = [(false, false); 3];
    for side_i in first..last {
        let Some(side) = bsp.brush_sides.get(side_i) else {
            continue;
        };
        let Some(p) = bsp.planes.get(side.plane_num as usize) else {
            continue;
        };
        let n = p.normal;
        for axis in 0..3 {
            if (n[axis] - 1.0).abs() < 1e-4 {
                maxs[axis] = p.dist;
                have[axis].0 = true;
            } else if (n[axis] + 1.0).abs() < 1e-4 {
                mins[axis] = -p.dist;
                have[axis].1 = true;
            }
        }
    }
    if have.iter().all(|(h1, h2)| *h1 && *h2) {
        Aabb::new(Vec3::from_array(mins), Vec3::from_array(maxs))
    } else {
        Aabb::new(Vec3::splat(f32::NEG_INFINITY), Vec3::splat(f32::INFINITY))
    }
}

const SPLIT_EPSILON: f32 = 0.125;
const SURFACE_CLIP_EPSILON: f32 = 0.125;

#[cfg(test)]
mod tests {
    use super::*;
    use q3_bsp::raw::{DBrush, DBrushSide, DPlane};

    fn single_cube_bsp() -> Bsp {
        cube_bsp_with_contents(Contents::SOLID)
    }

    fn cube_bsp_with_contents(contents: Contents) -> Bsp {
        // Construit manuellement une BSP minimale avec 1 brush cube
        // de -16..16 en chaque axe. C'est plus robuste que d'exiger un fichier.
        use q3_bsp::raw::{DLeaf, DModel, DNode, DShader};
        Bsp {
            entities: String::new(),
            shaders: vec![DShader {
                shader: [0; 64],
                surface_flags: 0,
                content_flags: contents.bits() as i32,
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
                children: [-1, -1], // les deux enfants vers le leaf 0 (-(0)-1 = -1)
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
            draw_verts: vec![],
            draw_indexes: vec![],
            fogs: vec![],
            surfaces: vec![],
            lightmap_bytes: vec![],
            lightgrid_bytes: vec![],
            visibility: q3_bsp::Visibility::default(),
        }
    }

    #[test]
    fn ray_misses_empty_space() {
        let world = CollisionWorld::new(single_cube_bsp());
        let t = world.trace_ray(
            Vec3::new(-100.0, 0.0, 0.0),
            Vec3::new(-50.0, 0.0, 0.0),
            Contents::SOLID,
        );
        assert_eq!(t.fraction, 1.0);
    }

    #[test]
    fn ray_hits_solid_cube() {
        let world = CollisionWorld::new(single_cube_bsp());
        // De x=-100 vers x=+100, on doit toucher le cube à x=-16, donc
        // fraction = 84/200 = 0.42
        let t = world.trace_ray(
            Vec3::new(-100.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 0.0),
            Contents::SOLID,
        );
        assert!(t.fraction < 1.0, "fraction = {}", t.fraction);
        assert!((t.fraction - 0.42).abs() < 0.01, "fraction = {}", t.fraction);
    }

    #[test]
    fn ray_in_other_axis_misses() {
        let world = CollisionWorld::new(single_cube_bsp());
        let t = world.trace_ray(
            Vec3::new(-100.0, 100.0, 0.0),
            Vec3::new(100.0, 100.0, 0.0),
            Contents::SOLID,
        );
        assert_eq!(t.fraction, 1.0);
    }

    /// Régression du bug WASD : avant le fix, `trace_box` initialisait
    /// `start_solid=true`/`all_solid=true` en entrée et ne les remettait à
    /// `false` que lorsque `fraction==1.0`. Donc tout trace qui TOUCHAIT
    /// quoi que ce soit (y compris le contact-sol normal, qui touche mais
    /// ne démarre PAS en solide) renvoyait `all_solid=true`, cascadant sur
    /// `update_ground` → push-out permanent → air mode → gravité divergente.
    ///
    /// Invariants testés ici :
    ///
    /// 1. **Miss complet** (rien sur le chemin) → `all_solid=false`,
    ///    `start_solid=false`, `fraction=1.0`.
    /// 2. **Hit partiel depuis l'extérieur** (on tape le cube au milieu
    ///    du trace) → `all_solid=false`, `start_solid=false`,
    ///    `0 < fraction < 1`. C'est le cas quotidien d'un joueur qui
    ///    marche vers un mur : on doit pouvoir distinguer « trace touche
    ///    un obstacle » de « joueur embedded dans le BSP ».
    /// 3. **Start embedded** (origine dans le cube) → `start_solid=true`,
    ///    `all_solid=true` (sortie = toujours dans le cube).
    #[test]
    fn trace_box_start_solid_flag_is_trustworthy() {
        let world = CollisionWorld::new(single_cube_bsp());
        let hull = TraceBox {
            mins: Vec3::splat(-1.0),
            maxs: Vec3::splat(1.0),
        };

        // (1) miss complet
        let t_miss = world.trace_box(
            Vec3::new(-100.0, 100.0, 100.0),
            Vec3::new(100.0, 100.0, 100.0),
            hull,
            Contents::SOLID,
        );
        assert_eq!(t_miss.fraction, 1.0);
        assert!(!t_miss.all_solid, "miss ne doit jamais être all_solid");
        assert!(!t_miss.start_solid, "miss ne doit jamais être start_solid");

        // (2) hit partiel depuis l'extérieur — trace traverse le cube [-16..16]
        // en x ; la boîte joueur (rayon 1) tape à x ≈ -17 (= -16 - 1).
        let t_hit = world.trace_box(
            Vec3::new(-100.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 0.0),
            hull,
            Contents::SOLID,
        );
        assert!(
            t_hit.fraction < 1.0,
            "doit toucher le cube — fraction = {}",
            t_hit.fraction
        );
        assert!(
            !t_hit.all_solid,
            "hit partiel ne doit PAS être all_solid (bug régression !) — \
             fraction={}, all_solid={}",
            t_hit.fraction, t_hit.all_solid
        );
        assert!(
            !t_hit.start_solid,
            "start hors du cube ne doit PAS être start_solid — \
             start_solid={}",
            t_hit.start_solid
        );

        // (3) start embedded : boîte rayon 1 centrée à l'origine (0,0,0)
        // = entièrement dans le cube [-16..16]. On trace vers l'extérieur.
        let t_embed = world.trace_box(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 0.0),
            hull,
            Contents::SOLID,
        );
        assert!(
            t_embed.start_solid,
            "start dans le solide doit flagger start_solid"
        );
        // Le hull sort du cube vers x=+16+1=17 après avoir parcouru 17/100
        // unités. Donc `all_solid=false` (on émerge) mais `start_solid=true`.
        assert!(
            !t_embed.all_solid,
            "on sort du cube → all_solid doit être false (fraction={})",
            t_embed.fraction
        );
    }

    /// point_contents : hors du cube solide → aucun contents.
    #[test]
    fn point_contents_outside_cube_is_empty() {
        let world = CollisionWorld::new(single_cube_bsp());
        // (100, 0, 0) est bien hors du cube [-16..16]
        let c = world.point_contents(Vec3::new(100.0, 0.0, 0.0));
        assert!(c.is_empty(), "hors cube doit renvoyer empty, got {:?}", c);
    }

    /// point_contents : au centre d'un cube SOLID → Contents::SOLID.
    #[test]
    fn point_contents_inside_solid_cube() {
        let world = CollisionWorld::new(single_cube_bsp());
        let c = world.point_contents(Vec3::new(0.0, 0.0, 0.0));
        assert!(
            c.contains(Contents::SOLID),
            "centre cube SOLID doit contenir SOLID, got {:?}",
            c
        );
    }

    /// point_contents : immergé dans un volume WATER → Contents::WATER.
    /// C'est le cas d'usage direct de l'effet underwater.
    #[test]
    fn point_contents_inside_water_brush() {
        let world = CollisionWorld::new(cube_bsp_with_contents(Contents::WATER));
        let c = world.point_contents(Vec3::new(0.0, 0.0, 0.0));
        assert!(
            c.contains(Contents::WATER),
            "centre cube WATER doit contenir WATER, got {:?}",
            c
        );
        // Un brush pur-liquide ne doit PAS rapporter SOLID.
        assert!(
            !c.contains(Contents::SOLID),
            "water brush ne doit pas être SOLID"
        );
    }

    /// point_contents : au coin même du cube (pile sur les plans) — le point
    /// est tangent aux 3 faces. Avec la convention `distance <= 0 = inside`,
    /// le coin est inclus.
    #[test]
    fn point_contents_on_cube_corner_is_inside() {
        let world = CollisionWorld::new(single_cube_bsp());
        let c = world.point_contents(Vec3::new(16.0, 16.0, 16.0));
        assert!(
            c.contains(Contents::SOLID),
            "coin du cube doit être classé inside, got {:?}",
            c
        );
    }

    /// point_contents : un BSP sans nodes doit retourner empty sans panic.
    #[test]
    fn point_contents_empty_bsp() {
        let bsp = Bsp {
            entities: String::new(),
            shaders: vec![],
            planes: vec![],
            nodes: vec![],
            leafs: vec![],
            leaf_surfaces: vec![],
            leaf_brushes: vec![],
            models: vec![],
            brushes: vec![],
            brush_sides: vec![],
            draw_verts: vec![],
            draw_indexes: vec![],
            fogs: vec![],
            surfaces: vec![],
            lightmap_bytes: vec![],
            lightgrid_bytes: vec![],
            visibility: q3_bsp::Visibility::default(),
        };
        let world = CollisionWorld::new(bsp);
        let c = world.point_contents(Vec3::new(0.0, 0.0, 0.0));
        assert!(c.is_empty(), "empty BSP doit renvoyer empty");
    }
}
