//! Tessellation des **patches de Bézier bicubiques** (en fait biquadratiques)
//! du format BSP de Quake 3.
//!
//! # Format
//!
//! Un patch est un grille de `patch_width × patch_height` points de contrôle
//! (les deux dimensions sont **impaires**, typiquement 3×3, 5×3, 9×5, etc.).
//! Elle est interprétée comme l'union de **sous-patches biquadratiques 3×3**,
//! qui se touchent par leurs bords : un patch 5×3 = deux sous-patches 3×3
//! partageant leur colonne centrale.
//!
//! # Tessellation
//!
//! Chaque sous-patch `P[3][3]` est évalué à `(level+1)²` points (u, v) avec
//! la formule biquadratique :
//!
//! ```text
//! S(u,v) = Σ_{i=0..2} Σ_{j=0..2} B_i(u) · B_j(v) · P[i][j]
//! ```
//!
//! où `B_0(t) = (1-t)²`, `B_1(t) = 2t(1-t)`, `B_2(t) = t²` sont les polynômes
//! de Bernstein de degré 2.
//!
//! Toutes les composantes du `DrawVert` (position, normal, UVs, couleur) sont
//! interpolées avec la même formule. Les normales sont renormalisées à la fin.
//!
//! # Déduplication des coutures
//!
//! Les sous-patches adjacents partagent leurs colonnes/lignes de bord par
//! construction : on génère donc un tableau global de
//! `(n_sub_u · level + 1) × (n_sub_v · level + 1)` vertices, sans duplication,
//! puis on produit les triangles en bandes (row-major).

use crate::raw::DrawVert;

/// Niveau de tessellation par défaut. Donne 6×6 = 36 verts par sous-patch
/// — équivalent à `r_subdivisions = 4` du moteur original.
pub const DEFAULT_TESSELLATION_LEVEL: u32 = 5;

/// Résultat de la tessellation d'un patch : vertices + index triangles.
#[derive(Debug, Clone)]
pub struct TessellatedPatch {
    pub vertices: Vec<DrawVert>,
    /// Indices dans `vertices` — liste de triangles (3 par entrée).
    pub indexes: Vec<u32>,
}

/// Tessellate un patch complet.
///
/// * `control_points` : tableau `patch_width × patch_height`, row-major
///   (même layout que dans la BSP).
/// * `patch_width` / `patch_height` : dimensions impaires ≥ 3.
/// * `level` : nombre de subdivisions entre deux points de contrôle
///   adjacents. 5 donne un résultat propre pour la plupart des maps.
///
/// Retourne `None` si les dimensions sont invalides (paires, < 3,
/// ou incohérentes avec la taille du slice).
pub fn tessellate_patch(
    control_points: &[DrawVert],
    patch_width: i32,
    patch_height: i32,
    level: u32,
) -> Option<TessellatedPatch> {
    if patch_width < 3 || patch_height < 3 {
        return None;
    }
    if patch_width % 2 == 0 || patch_height % 2 == 0 {
        return None;
    }
    let (pw, ph) = (patch_width as usize, patch_height as usize);
    if control_points.len() != pw * ph {
        return None;
    }
    let level = level.max(1) as usize;

    let n_sub_u = (pw - 1) / 2;
    let n_sub_v = (ph - 1) / 2;

    let grid_w = n_sub_u * level + 1;
    let grid_h = n_sub_v * level + 1;

    let mut vertices: Vec<DrawVert> = Vec::with_capacity(grid_w * grid_h);

    for gy in 0..grid_h {
        // Déterminer dans quelle sous-bande verticale on est.
        let (sv, local_v) = sub_coords(gy, n_sub_v, level);
        for gx in 0..grid_w {
            let (su, local_u) = sub_coords(gx, n_sub_u, level);

            // Les 9 points de contrôle de la sous-patch (su, sv).
            let p = |i: usize, j: usize| -> &DrawVert {
                let x = 2 * su + i;
                let y = 2 * sv + j;
                &control_points[y * pw + x]
            };

            // Évaluation biquadratique. On réutilise l'intermédiaire colonne.
            let u = local_u as f32 / level as f32;
            let v = local_v as f32 / level as f32;
            let (bu0, bu1, bu2) = bernstein2(u);
            let (bv0, bv1, bv2) = bernstein2(v);

            // Accumule toutes les composantes en f32 puis reconvertit couleur → u8.
            let mut pos = [0.0f32; 3];
            let mut st = [0.0f32; 2];
            let mut lm = [0.0f32; 2];
            let mut nrm = [0.0f32; 3];
            let mut col = [0.0f32; 4];

            for (j, bj) in [bv0, bv1, bv2].into_iter().enumerate() {
                let c0 = p(0, j);
                let c1 = p(1, j);
                let c2 = p(2, j);
                let w0 = bu0 * bj;
                let w1 = bu1 * bj;
                let w2 = bu2 * bj;
                for k in 0..3 {
                    pos[k] += w0 * c0.xyz[k] + w1 * c1.xyz[k] + w2 * c2.xyz[k];
                    nrm[k] += w0 * c0.normal[k] + w1 * c1.normal[k] + w2 * c2.normal[k];
                }
                for k in 0..2 {
                    st[k] += w0 * c0.st[k] + w1 * c1.st[k] + w2 * c2.st[k];
                    lm[k] += w0 * c0.lightmap[k] + w1 * c1.lightmap[k] + w2 * c2.lightmap[k];
                }
                for k in 0..4 {
                    col[k] +=
                        w0 * c0.color[k] as f32 + w1 * c1.color[k] as f32 + w2 * c2.color[k] as f32;
                }
            }

            // Renormalise la normale.
            let nlen = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
            if nlen > 1e-6 {
                nrm[0] /= nlen;
                nrm[1] /= nlen;
                nrm[2] /= nlen;
            }

            vertices.push(DrawVert {
                xyz: pos,
                st,
                lightmap: lm,
                normal: nrm,
                color: [
                    col[0].clamp(0.0, 255.0) as u8,
                    col[1].clamp(0.0, 255.0) as u8,
                    col[2].clamp(0.0, 255.0) as u8,
                    col[3].clamp(0.0, 255.0) as u8,
                ],
            });
        }
    }

    // Triangulation row-major : pour chaque cellule (gx, gy) → 2 triangles.
    let mut indexes: Vec<u32> = Vec::with_capacity((grid_w - 1) * (grid_h - 1) * 6);
    for gy in 0..grid_h - 1 {
        for gx in 0..grid_w - 1 {
            let i00 = (gy * grid_w + gx) as u32;
            let i10 = i00 + 1;
            let i01 = i00 + grid_w as u32;
            let i11 = i01 + 1;
            // triangle 1
            indexes.push(i00);
            indexes.push(i01);
            indexes.push(i11);
            // triangle 2
            indexes.push(i00);
            indexes.push(i11);
            indexes.push(i10);
        }
    }

    Some(TessellatedPatch { vertices, indexes })
}

/// Retourne `(sub_index, local_index)` pour la coordonnée `g` dans la grille
/// globale, sachant qu'il y a `n_sub` sous-patches et `level` subdivisions
/// par sous-patch.
#[inline]
fn sub_coords(g: usize, n_sub: usize, level: usize) -> (usize, usize) {
    if g == n_sub * level {
        // Dernier point exact : il appartient à la dernière sous-patch,
        // position locale = level.
        (n_sub - 1, level)
    } else {
        let sub = g / level;
        let local = g % level;
        (sub.min(n_sub - 1), local)
    }
}

/// Polynômes de Bernstein quadratiques.
#[inline]
fn bernstein2(t: f32) -> (f32, f32, f32) {
    let omt = 1.0 - t;
    (omt * omt, 2.0 * t * omt, t * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cp(x: f32, y: f32, z: f32) -> DrawVert {
        DrawVert {
            xyz: [x, y, z],
            st: [0.0; 2],
            lightmap: [0.0; 2],
            normal: [0.0, 0.0, 1.0],
            color: [255, 255, 255, 255],
        }
    }

    #[test]
    fn tessellate_flat_3x3() {
        // 3×3 contrôles dans le plan z=0, quadrant [0..4]×[0..4].
        let cps: Vec<DrawVert> = (0..9)
            .map(|i| {
                let x = (i % 3) as f32 * 2.0;
                let y = (i / 3) as f32 * 2.0;
                cp(x, y, 0.0)
            })
            .collect();
        let t = tessellate_patch(&cps, 3, 3, 4).unwrap();
        assert_eq!(t.vertices.len(), 25); // 5×5
        assert_eq!(t.indexes.len(), 4 * 4 * 6);
        // Coin (0,0)
        assert_eq!(t.vertices[0].xyz, [0.0, 0.0, 0.0]);
        // Coin opposé (4,4)
        let last = t.vertices.last().unwrap();
        assert!((last.xyz[0] - 4.0).abs() < 1e-4);
        assert!((last.xyz[1] - 4.0).abs() < 1e-4);
    }

    #[test]
    fn tessellate_5x3_two_subpatches() {
        let cps: Vec<DrawVert> = (0..15)
            .map(|i| {
                let x = (i % 5) as f32;
                let y = (i / 5) as f32;
                cp(x, y, 0.0)
            })
            .collect();
        let t = tessellate_patch(&cps, 5, 3, 4).unwrap();
        // 2 sous-patches en u × 1 en v : grille = (8+1) × (4+1) = 9 × 5 = 45
        assert_eq!(t.vertices.len(), 45);
    }

    #[test]
    fn rejects_even_dimensions() {
        let cps = vec![cp(0.0, 0.0, 0.0); 12];
        assert!(tessellate_patch(&cps, 4, 3, 4).is_none());
    }

    #[test]
    fn center_of_3x3_flat_matches_middle_control() {
        // Si tous les points de contrôle sont dans le plan z=0 et que la
        // grille est régulière, le centre tesselé doit être (milieu, 0).
        let cps: Vec<DrawVert> = (0..9)
            .map(|i| {
                let x = (i % 3) as f32 * 2.0;
                let y = (i / 3) as f32 * 2.0;
                cp(x, y, 0.0)
            })
            .collect();
        let t = tessellate_patch(&cps, 3, 3, 4).unwrap();
        // Vertex central dans 5×5 = index 12.
        let v = &t.vertices[12];
        assert!((v.xyz[0] - 2.0).abs() < 1e-3);
        assert!((v.xyz[1] - 2.0).abs() < 1e-3);
        assert!(v.xyz[2].abs() < 1e-3);
    }
}
