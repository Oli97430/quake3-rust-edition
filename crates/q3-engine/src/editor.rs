//! Mode éditeur visuel — placement manuel de props GLB.
//!
//! Quand `editor_enabled` est vrai dans l'App :
//!   * la souris est libre (curseur visible, pas de capture FPS)
//!   * un panneau UI à droite de l'écran liste les props chargés et
//!     les boutons d'action (Import / Save / Load / Delete / etc.)
//!   * cliquer en dehors du panneau spawn le prop actif au point visé
//!     (raycast écran → terrain/world)
//!   * cliquer sur un prop déjà placé le sélectionne pour modif
//!
//! La logique métier reste dans `app.rs` (les `RockProp` vivent dans
//! `App::rocks`).  Ce module ne fait que :
//!   1. Afficher le panneau (rectangles + texte via le renderer)
//!   2. Tester si un clic souris est sur un bouton et émettre un
//!      `EditorClick` que l'App consomme.

#![allow(dead_code)]

use glam::Vec2;

/// Action remontée par un clic dans l'UI editor — l'App applique
/// le changement d'état correspondant.
#[derive(Debug, Clone)]
pub enum EditorClick {
    /// Aucune action (clic dans le vide ou drag).
    None,
    /// Ouvrir le file picker pour importer un GLB local.
    ImportGlb,
    /// Sélectionner un prop chargé comme prop actif à spawn.
    /// `String` = nom du prop dans `editor_prop_scales`.
    SelectActiveProp(String),
    /// Sélectionner un prop placé par index dans `App::rocks`.
    SelectPlaced(usize),
    /// Déclenche un spawn au point monde déjà calculé (clic 3D).
    /// L'App fait le raycast elle-même puisqu'elle a la caméra.
    SpawnAtAim,
    /// Multiplier le scale du sélectionné.
    ScaleMul(f32),
    /// Ajouter au yaw du sélectionné (degrés).
    YawAdd(f32),
    /// Toggle alignement terrain.
    ToggleAlign,
    /// Supprimer le sélectionné.
    Delete,
    /// Supprimer un prop placé par index (bouton [X] sur la ligne).
    DeletePlaced(usize),
    /// Sauver les placements (`reunion_edits.txt`).
    Save,
    /// Recharger les placements depuis le disque.
    Load,
    /// Fermer le mode éditeur.
    Close,
}

/// Géométrie d'un bouton — coin haut-gauche en pixels écran + taille.
#[derive(Debug, Clone, Copy)]
pub struct ButtonRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl ButtonRect {
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.w && py >= self.y && py <= self.y + self.h
    }
}

/// État runtime du panneau — recalculé à chaque hit-test/draw (les
/// boutons sont positionnés dynamiquement selon le nombre d'entrées).
pub struct EditorPanel {
    /// Largeur du panneau en pixels.
    pub width: f32,
    /// Marge depuis le bord droit de l'écran.
    pub margin_right: f32,
    /// Hauteur d'une ligne de bouton.
    pub line_h: f32,
    /// Nom du prop actif sélectionné dans la liste "props chargés".
    /// Quand l'utilisateur clique en monde, c'est ce prop qui est spawné.
    pub active_prop: Option<String>,
    /// Scroll offset pour la liste des placements (si > N visibles).
    pub placed_scroll: i32,
    /// Position curseur en pixels pour le hover feedback.
    pub hover_pos: Option<(f32, f32)>,
}

impl Default for EditorPanel {
    fn default() -> Self {
        Self {
            width: 380.0,
            margin_right: 12.0,
            line_h: 28.0,
            active_prop: None,
            placed_scroll: 0,
            hover_pos: None,
        }
    }
}

/// Snapshot des données affichées par le panneau pour un frame.
/// Construit par l'App depuis ses propres données.
pub struct PanelView<'a> {
    /// Liste des props chargés (nom, scale par défaut).
    pub loaded_props: &'a [(String, f32)],
    /// Liste des props placés (idx, prop_name, position).
    pub placed_props: &'a [(usize, &'a str, [f32; 3])],
    /// Index du prop placé sélectionné (pour highlight).
    pub selected_idx: Option<usize>,
}

/// Calcule les bounds du panneau pour un écran de taille (sw, sh).
pub fn panel_bounds(panel: &EditorPanel, sw: f32, _sh: f32) -> (f32, f32, f32, f32) {
    let x = sw - panel.width - panel.margin_right;
    let y = 16.0;
    (x, y, panel.width, _sh - 32.0)
}

/// Hit-test : retourne l'action correspondant au point cliqué.
/// Renvoie `EditorClick::None` si le clic n'est pas sur un bouton du
/// panneau (l'App interprètera alors comme un clic 3D pour spawn).
pub fn hit_test(
    panel: &mut EditorPanel,
    view: &PanelView<'_>,
    px: f32,
    py: f32,
    sw: f32,
    sh: f32,
) -> EditorClick {
    let (panel_x, panel_y, pw, ph) = panel_bounds(panel, sw, sh);
    // Si le clic n'est pas dans la bande verticale du panneau, c'est
    // un clic 3D → SpawnAtAim (l'App décide si elle fait quelque chose).
    if px < panel_x {
        return EditorClick::SpawnAtAim;
    }

    let mut y = panel_y + 14.0;
    let lh = panel.line_h;
    let pad = 8.0;

    // ── Ligne 1 : titre (non cliquable) ──
    y += lh;

    // ── Ligne 2 : bouton "IMPORT GLB" ──
    if hit_line(px, py, panel_x, y, pw, lh) {
        return EditorClick::ImportGlb;
    }
    y += lh + pad;

    // ── Section "Props chargés" — header ──
    y += lh;

    // ── Liste des props chargés ──
    for (name, _) in view.loaded_props {
        if hit_line(px, py, panel_x, y, pw, lh) {
            return EditorClick::SelectActiveProp(name.clone());
        }
        y += lh;
    }
    y += pad;

    // ── Boutons d'action sur le sélectionné ──
    let action_buttons: &[(&str, EditorClick)] = &[
        ("SCALE +10%",     EditorClick::ScaleMul(1.10)),
        ("SCALE -10%",     EditorClick::ScaleMul(1.0 / 1.10)),
        ("YAW +15\u{00b0}",  EditorClick::YawAdd(15.0)),
        ("YAW -15\u{00b0}",  EditorClick::YawAdd(-15.0)),
        ("TOGGLE ALIGN",   EditorClick::ToggleAlign),
        ("DELETE",         EditorClick::Delete),
    ];
    for (_, action) in action_buttons {
        if hit_line(px, py, panel_x, y, pw, lh) {
            return action.clone();
        }
        y += lh;
    }
    y += pad;

    // ── Boutons Save/Load/Close ──
    let footer: &[(&str, EditorClick)] = &[
        ("SAVE", EditorClick::Save),
        ("LOAD", EditorClick::Load),
        ("CLOSE EDITOR", EditorClick::Close),
    ];
    for (_, action) in footer {
        if hit_line(px, py, panel_x, y, pw, lh) {
            return action.clone();
        }
        y += lh;
    }
    y += pad;

    // ── Section "Props placés" — header ──
    y += lh;

    // ── Liste des props placés ──
    // Scroll : on saute `placed_scroll` entrées.
    // Clic droit de la ligne (30px) = [X] delete, le reste = select.
    let max_lines = ((panel_y + ph - y - 8.0) / lh).floor().max(0.0) as usize;
    let scroll = (panel.placed_scroll.max(0) as usize).min(
        view.placed_props.len().saturating_sub(max_lines),
    );
    let visible = &view.placed_props[scroll..];
    let del_zone_w = 30.0; // largeur du bouton [X] à droite
    for (idx, _name, _pos) in visible.iter().take(max_lines) {
        if hit_line(px, py, panel_x, y, pw, lh) {
            // Zone [X] à l'extrême droite ?
            if px >= panel_x + pw - del_zone_w - 8.0 {
                return EditorClick::DeletePlaced(*idx);
            }
            return EditorClick::SelectPlaced(*idx);
        }
        y += lh;
    }

    EditorClick::None
}

/// Retourne l'index de la ligne survolée (hover) pour le feedback
/// visuel. -1 = rien. L'index est interne au layout — le draw
/// l'utilise pour savoir quoi highlighter.
pub fn hover_line_index(
    panel: &EditorPanel,
    px: f32,
    py: f32,
    sw: f32,
    sh: f32,
    total_loaded: usize,
    total_placed: usize,
) -> i32 {
    let (panel_x, panel_y, _, _) = panel_bounds(panel, sw, sh);
    if px < panel_x { return -1; }

    let lh = panel.line_h;
    let pad = 8.0;
    let mut y = panel_y + 14.0;
    let mut idx: i32 = 0;

    // Titre
    y += lh; idx += 1;
    // Import
    if py >= y && py < y + lh { return idx; }
    y += lh + pad; idx += 1;
    // Header props chargés
    y += lh; idx += 1;
    // Props chargés
    for _ in 0..total_loaded {
        if py >= y && py < y + lh { return idx; }
        y += lh; idx += 1;
    }
    y += pad;
    // 6 action buttons
    for _ in 0..6 {
        if py >= y && py < y + lh { return idx; }
        y += lh; idx += 1;
    }
    y += pad;
    // 3 footer buttons
    for _ in 0..3 {
        if py >= y && py < y + lh { return idx; }
        y += lh; idx += 1;
    }
    y += pad;
    // Header placés
    y += lh; idx += 1;
    // Props placés
    for _ in 0..total_placed {
        if py >= y && py < y + lh { return idx; }
        y += lh; idx += 1;
    }

    -1
}

fn hit_line(px: f32, py: f32, x: f32, y: f32, w: f32, h: f32) -> bool {
    px >= x && px <= x + w && py >= y && py <= y + h
}

/// Convertit une position curseur écran (px) + size écran (sw, sh) en
/// rayon monde (origine, direction normalisée) à partir de la matrice
/// view×proj inverse de la caméra.  Utilise glam exclusivement.
pub fn screen_to_ray(
    cursor: Vec2,
    screen: Vec2,
    inv_view_proj: glam::Mat4,
    cam_pos: glam::Vec3,
) -> (glam::Vec3, glam::Vec3) {
    // NDC : x ∈ [-1, 1] de gauche à droite, y ∈ [-1, 1] de bas à haut.
    let ndc_x = (cursor.x / screen.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (cursor.y / screen.y) * 2.0;
    // Point sur le plan far en NDC.
    let far_ndc = glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let world = inv_view_proj * far_ndc;
    let world = if world.w.abs() > 1e-6 {
        glam::Vec3::new(world.x / world.w, world.y / world.w, world.z / world.w)
    } else {
        glam::Vec3::new(world.x, world.y, world.z)
    };
    let dir = (world - cam_pos).normalize_or_zero();
    (cam_pos, dir)
}
