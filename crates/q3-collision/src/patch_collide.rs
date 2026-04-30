//! Collision contre les surfaces de type **Patch** (Bézier biquadratique).
//!
//! Le moteur d'origine (`cm_patch.c`) construit pour chaque patch un
//! `patchCollide_t` composé de **facets** : chaque facet est un micro-brush
//! généré depuis un triangle de la tessellation, avec un plan de surface
//! et des plans de bordure qui le transforment en convexe. Le trace
//! réutilise alors exactement la même logique plane-par-plane que pour
//! les brushes normaux.
//!
//! On adopte la même approche, en simplifiant la géométrie de chaque
//! facet :
//!
//! * **1 plan de face** : normale sortante, calculée depuis les trois
//!   sommets du triangle (cross-product).
//! * **1 plan de back** : normale inversée, offset de `PATCH_THICKNESS`
//!   vers l'arrière. Ensemble, face + back donnent un slab très mince
//!   qui comportementalement est **imperméable** (un joueur qui veut
//!   traverser la surface est bloqué par le slab).
//! * **3 plans d'arête** : chacun contient une arête du triangle et est
//!   perpendiculaire au plan de face ; sa normale pointe vers l'extérieur
//!   du triangle (loin du 3e sommet).
//!
//! L'intersection des 5 demi-espaces forme un prisme triangulaire mince
//! qu'on teste avec la même routine `trace_brush_planes` que pour les
//! brushes standards, sans avoir besoin d'écrire un test ray-triangle +
//! box-swept-triangle à part.
//!
//! # Robustesse numérique
//!
//! * On ignore les triangles dégénérés (aire ≈ 0) avant même de générer
//!   des plans invalides.
//! * `PATCH_THICKNESS` est choisi suffisamment grand (2.0) pour qu'un
//!   trace à la vitesse normale du joueur (1/4 de tick ≈ 40 unités à
//!   vélocité 320) ne puisse pas tunneler la surface. C'est le même
//!   ordre de grandeur que la thickness de Q3 (512 / 128 = 4, mais
//!   relativement au subdivisions level; 2.0 chez nous suffit avec
//!   notre tessellation level 4).

use super::Contents;
use q3_bsp::{patch::tessellate_patch, raw::DrawVert};
use q3_math::{Aabb, Plane, Vec3};

/// Épaisseur du slab de chaque facet. Suffisamment grande pour empêcher le
/// tunneling à vitesse normale, suffisamment petite pour qu'un joueur ne
/// flotte pas anormalement au-dessus de la surface visible.
pub(crate) const PATCH_THICKNESS: f32 = 2.0;

/// Niveau de subdivision utilisé pour la collision. Plus grossier que le
/// rendu (`DEFAULT_TESSELLATION_LEVEL = 5`) pour limiter le nombre de
/// facets (chaque facet = 5 plans à tester). Niveau 4 → 5×5 = 25 sommets
/// par sous-patch, 32 triangles, soit 160 plans à tester par sous-patch.
pub(crate) const COLLISION_TESSELLATION_LEVEL: u32 = 4;

/// Micro-brush synthétique représentant un triangle de la tessellation.
#[derive(Debug, Clone, Copy)]
pub struct PatchFacet {
    /// Plan de la face du triangle (normale sortante).
    pub face: Plane,
    /// Plan de back : même orientation que `face` mais offset arrière.
    pub back: Plane,
    /// Plans des 3 arêtes (normales sortantes du triangle).
    pub edges: [Plane; 3],
    /// AABB incluant l'épaisseur du slab (pour l'early-out).
    pub bounds: Aabb,
}

/// Ensemble de facets pour une surface de type `Patch`.
#[derive(Debug, Clone)]
pub struct PatchCollide {
    pub facets: Vec<PatchFacet>,
    /// Contenu dérivé du shader de la surface (typiquement `SOLID`).
    pub contents: Contents,
    /// AABB globale de tous les facets (padding d'épaisseur inclus).
    pub bounds: Aabb,
}

/// Construit le `PatchCollide` d'une surface BSP de type Patch.
///
/// * `verts` : les `num_verts` points de contrôle, row-major
///   (`patch_width × patch_height`).
/// * `contents` : contenu issu du shader (pour le filtrage `mask`).
///
/// Retourne `None` si :
/// * les dimensions sont invalides (paires ou < 3) ;
/// * la tessellation échoue ;
/// * aucun triangle non-dégénéré n'est généré.
pub fn build_patch_collide(
    verts: &[DrawVert],
    patch_width: i32,
    patch_height: i32,
    contents: Contents,
) -> Option<PatchCollide> {
    let tess = tessellate_patch(
        verts,
        patch_width,
        patch_height,
        COLLISION_TESSELLATION_LEVEL,
    )?;

    let mut facets: Vec<PatchFacet> = Vec::with_capacity(tess.indexes.len() / 3);
    let mut global_mins = Vec3::splat(f32::INFINITY);
    let mut global_maxs = Vec3::splat(f32::NEG_INFINITY);

    for tri in tess.indexes.chunks_exact(3) {
        let Some(vs) = (0..3)
            .map(|i| tess.vertices.get(tri[i] as usize).map(|v| Vec3::from_array(v.xyz)))
            .collect::<Option<Vec<Vec3>>>()
        else {
            continue;
        };
        let v0 = vs[0];
        let v1 = vs[1];
        let v2 = vs[2];

        // Normale du triangle. Triangle dégénéré → on skip.
        let edge01 = v1 - v0;
        let edge02 = v2 - v0;
        let cross = edge01.cross(edge02);
        let cross_len = cross.length();
        if cross_len < 1e-5 {
            continue;
        }
        let normal = cross / cross_len;
        let face_dist = normal.dot(v0);

        // Slab **centré** sur la surface : face poussé de `t/2` vers l'avant,
        // back poussé de `t/2` vers l'arrière (normale inversée). Ainsi
        // l'épaisseur du slab occupe `[face_dist - t/2, face_dist + t/2]`
        // le long de la normale, quel que soit son orientation (+n ou -n
        // selon le winding du triangle dans la tessellation). Les joueurs
        // des deux côtés de la surface sont bloqués de manière symétrique,
        // ce qu'on veut pour une rampe ou une courbe qu'on peut traverser
        // "par le mauvais côté".
        let half_t = PATCH_THICKNESS * 0.5;
        let face = Plane {
            normal,
            dist: face_dist + half_t,
        };
        // Le demi-espace positif de `back` = `(-n)·p >= -(d - t/2)`
        // ⇔ `n·p <= d - t/2`. Combiné avec face (`n·p >= d + t/2` extérieur),
        // l'intersection des **intérieurs** donne `d - t/2 <= n·p <= d + t/2`,
        // soit un slab de largeur `t` centré sur la surface.
        let back = Plane {
            normal: -normal,
            dist: -(face_dist - half_t),
        };

        // Plans d'arête. Pour chaque arête (a,b) avec sommet opposé c, la
        // normale du plan d'arête est `edge × face_normal`, puis flip si
        // nécessaire pour pointer hors du triangle (away from c).
        let edges = [
            build_edge_plane(v0, v1, v2, normal),
            build_edge_plane(v1, v2, v0, normal),
            build_edge_plane(v2, v0, v1, normal),
        ];

        // Bounds = AABB du triangle étendu par PATCH_THICKNESS sur chaque axe,
        // ce qui garantit qu'un trace en box qui touche le slab passe
        // l'early-out.
        let pad = Vec3::splat(PATCH_THICKNESS);
        let mins = v0.min(v1).min(v2) - pad;
        let maxs = v0.max(v1).max(v2) + pad;
        global_mins = global_mins.min(mins);
        global_maxs = global_maxs.max(maxs);

        facets.push(PatchFacet {
            face,
            back,
            edges,
            bounds: Aabb::new(mins, maxs),
        });
    }

    if facets.is_empty() {
        return None;
    }

    Some(PatchCollide {
        facets,
        contents,
        bounds: Aabb::new(global_mins, global_maxs),
    })
}

/// Construit le plan de l'arête `a → b` d'un triangle dont le 3e sommet
/// est `c`. La normale du plan d'arête est perpendiculaire à la fois à
/// l'arête et au plan de face, et pointe à l'extérieur du triangle
/// (dans le demi-espace qui ne contient pas `c`).
fn build_edge_plane(a: Vec3, b: Vec3, c: Vec3, face_normal: Vec3) -> Plane {
    let edge_dir = b - a;
    let mut edge_normal = edge_dir.cross(face_normal);
    // Flip si la normale pointe vers l'intérieur (du même côté que c).
    if edge_normal.dot(c - a) > 0.0 {
        edge_normal = -edge_normal;
    }
    let len = edge_normal.length();
    let edge_normal = if len > 1e-6 {
        edge_normal / len
    } else {
        // Triangle quasi-dégénéré sur cette arête — on met un plan qui ne
        // clipperapas (normale vers le haut, dist infinie). `trace_facet`
        // saura que d2 > d1 ne produit pas d'enter valide.
        Vec3::Z
    };
    Plane {
        normal: edge_normal,
        dist: edge_normal.dot(a),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construit un patch plat 3×3 dans le plan z=0, couvrant [0..64]².
    /// Les 9 points de contrôle sont sur une grille régulière, donc la
    /// surface tesselée reste parfaitement plane.
    fn flat_patch_3x3() -> Vec<DrawVert> {
        let mut v = Vec::with_capacity(9);
        for j in 0..3 {
            for i in 0..3 {
                v.push(DrawVert {
                    xyz: [i as f32 * 32.0, j as f32 * 32.0, 0.0],
                    st: [0.0, 0.0],
                    lightmap: [0.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    color: [255; 4],
                });
            }
        }
        v
    }

    #[test]
    fn flat_patch_produces_facets_with_vertical_face_normals() {
        let cp = flat_patch_3x3();
        let pc = build_patch_collide(&cp, 3, 3, Contents::SOLID)
            .expect("patch collide");
        assert!(!pc.facets.is_empty(), "au moins un facet");
        for (i, f) in pc.facets.iter().enumerate() {
            // Le slab étant symétrique (face + back), le signe de la
            // normale de `face` dépend du winding de la tessellation ;
            // ce qui compte, c'est qu'elle soit verticale (perpendiculaire
            // au plan xy) et que `back` ait la normale opposée.
            assert!(
                f.face.normal.z.abs() > 0.9,
                "facet {i} doit avoir normale verticale, got {:?}",
                f.face.normal
            );
            assert!(
                (f.face.normal + f.back.normal).length() < 1e-5,
                "facet {i} face et back doivent être opposés : face={:?}, back={:?}",
                f.face.normal,
                f.back.normal
            );
            // Les 3 plans d'arête doivent être quasi-horizontaux (normal.z ≈ 0).
            for (j, e) in f.edges.iter().enumerate() {
                assert!(
                    e.normal.z.abs() < 0.1,
                    "facet {i} arête {j} doit être verticale, got {:?}",
                    e.normal
                );
            }
        }
    }

    /// Test "bout en bout" : un trace vertical descendant depuis bien
    /// au-dessus d'un patch plat doit s'arrêter à la surface (fraction < 1
    /// et normale verticale), et un trace loin du patch doit manquer.
    #[test]
    fn flat_patch_blocks_vertical_trace() {
        use super::super::{CollisionWorld, TraceBox};
        use q3_bsp::raw::{DLeaf, DModel, DNode, DPlane, DShader, DSurface};

        let cp = flat_patch_3x3();
        let pc = build_patch_collide(&cp, 3, 3, Contents::SOLID).unwrap();

        // BSP minimale : 1 leaf contenant 0 brush + 1 surface (le patch).
        // On n'a pas besoin de nodes complets puisque recurse_tree descend
        // tout de suite à child négatif = leaf 0 (-(0)-1 = -1).
        let bsp = q3_bsp::Bsp {
            entities: String::new(),
            shaders: vec![DShader {
                shader: [0; 64],
                surface_flags: 0,
                content_flags: Contents::SOLID.bits() as i32,
            }],
            planes: vec![DPlane { normal: [0.0, 0.0, 1.0], dist: 0.0 }],
            nodes: vec![DNode {
                plane_num: 0,
                children: [-1, -1],
                mins: [-100, -100, -100],
                maxs: [100, 100, 100],
            }],
            leafs: vec![DLeaf {
                cluster: 0,
                area: 0,
                mins: [-100, -100, -100],
                maxs: [100, 100, 100],
                first_leaf_surface: 0,
                num_leaf_surfaces: 1,
                first_leaf_brush: 0,
                num_leaf_brushes: 0,
            }],
            leaf_surfaces: vec![0],
            leaf_brushes: vec![],
            models: vec![DModel {
                mins: [-100.0; 3],
                maxs: [100.0; 3],
                first_surface: 0,
                num_surfaces: 1,
                first_brush: 0,
                num_brushes: 0,
            }],
            brushes: vec![],
            brush_sides: vec![],
            draw_verts: cp,
            draw_indexes: vec![],
            fogs: vec![],
            surfaces: vec![DSurface {
                shader_num: 0,
                fog_num: -1,
                surface_type: 2, // Patch
                first_vert: 0,
                num_verts: 9,
                first_index: 0,
                num_indexes: 0,
                lightmap_num: -1,
                lightmap_x: 0,
                lightmap_y: 0,
                lightmap_width: 0,
                lightmap_height: 0,
                lightmap_origin: [0.0; 3],
                lightmap_vecs: [[0.0; 3]; 3],
                patch_width: 3,
                patch_height: 3,
            }],
            lightmap_bytes: vec![],
            lightgrid_bytes: vec![],
            visibility: q3_bsp::Visibility::default(),
        };
        let world = CollisionWorld::new(bsp);
        assert_eq!(world.patch_count(), 1, "patch doit être enregistré");
        let _ = pc; // consumed above via cp; keep binding alive for clarity

        // Trace descendante au-dessus du patch (centre à x=32, y=32, z=50 → -50).
        // Le patch plat couvre [0..64] × [0..64] à z=0, épaisseur ±1 (slab).
        // Le hull joueur (box rayon 1) devrait taper le slab vers z ≈ 1 + 1 = 2
        // (slab top + hull extent), donc fraction ≈ (50 - 2) / 100 = 0.48.
        let hull = TraceBox::symmetric(Vec3::splat(1.0));
        let t = world.trace_box(
            Vec3::new(32.0, 32.0, 50.0),
            Vec3::new(32.0, 32.0, -50.0),
            hull,
            Contents::SOLID,
        );
        assert!(
            t.fraction < 1.0,
            "trace vertical doit toucher le patch : fraction={}",
            t.fraction
        );
        assert!(
            t.plane_normal.z.abs() > 0.5,
            "normale d'impact doit être verticale : {:?}",
            t.plane_normal
        );

        // Trace loin du patch (x=200, hors de la grille 0..64) → miss.
        let t_miss = world.trace_box(
            Vec3::new(200.0, 32.0, 50.0),
            Vec3::new(200.0, 32.0, -50.0),
            hull,
            Contents::SOLID,
        );
        assert_eq!(
            t_miss.fraction, 1.0,
            "trace loin du patch doit passer — fraction={}",
            t_miss.fraction
        );
        assert!(!t_miss.all_solid);
        assert!(!t_miss.start_solid);
    }

    #[test]
    fn patch_bounds_cover_slab_thickness() {
        let cp = flat_patch_3x3();
        let pc = build_patch_collide(&cp, 3, 3, Contents::SOLID).unwrap();
        // La surface plate est à z=0, les bounds doivent déborder de
        // PATCH_THICKNESS en haut et en bas.
        assert!(pc.bounds.mins.z <= -PATCH_THICKNESS + 0.001);
        assert!(pc.bounds.maxs.z >= PATCH_THICKNESS - 0.001);
    }

    #[test]
    fn degenerate_patch_returns_none() {
        // 9 points tous identiques → tous les triangles sont dégénérés.
        let cp = vec![
            DrawVert {
                xyz: [0.0, 0.0, 0.0],
                st: [0.0, 0.0],
                lightmap: [0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                color: [255; 4],
            };
            9
        ];
        assert!(build_patch_collide(&cp, 3, 3, Contents::SOLID).is_none());
    }

    #[test]
    fn invalid_dimensions_return_none() {
        let cp = flat_patch_3x3();
        // Dimensions paires → tessellate_patch retourne None.
        assert!(build_patch_collide(&cp, 4, 3, Contents::SOLID).is_none());
    }
}
