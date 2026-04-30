//! Helpers HUD bas niveau extraits de `app.rs` (lui-même très large).
//!
//! Tout ce qui est ici est **pur** : palette de couleurs, constantes
//! d'échelle, fonctions stateless qui dessinent dans un `Renderer`. Pas
//! de référence à `App`, pas de cvar, pas de matchstate. Le but de
//! cette extraction n'est pas de cacher de la logique, c'est de
//! distinguer la **plomberie de rendu HUD** de la logique de match :
//! quand on touche au HUD on ouvre ce fichier, quand on touche à la
//! logique on reste dans `app.rs`.
//!
//! Les futurs extractions (`hud_panels.rs`, `hud_scoreboard.rs`…) suivront
//! ce même pattern — les helpers de la couche basse vivent ici, les
//! widgets composites ailleurs.

use q3_renderer::Renderer;

// ─── Palette ────────────────────────────────────────────────────────────

pub const COL_WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
pub const COL_GRAY: [f32; 4] = [0.75, 0.75, 0.75, 1.0];
pub const COL_YELLOW: [f32; 4] = [1.0, 0.86, 0.2, 1.0];
pub const COL_RED: [f32; 4] = [1.0, 0.25, 0.2, 1.0];
pub const COL_CONSOLE_BG: [f32; 4] = [0.02, 0.02, 0.05, 0.85];
pub const COL_CONSOLE_BORDER: [f32; 4] = [0.4, 0.4, 0.55, 0.9];

/// Palette « moderne » HUD 2026 — panneaux semi-opaques anthracite avec
/// liseré accent cyan électrique. Contraste calibré pour rester lisible
/// sur un fond Q3 typique (textures brunes, lumière chaude) sans trahir
/// le look rétro-FPS.
pub const COL_PANEL_BG: [f32; 4] = [0.04, 0.05, 0.08, 0.72];
pub const COL_PANEL_EDGE: [f32; 4] = [0.15, 0.60, 0.85, 0.95];
pub const COL_PANEL_EDGE_DIM: [f32; 4] = [0.15, 0.60, 0.85, 0.35];
pub const COL_TEXT_SHADOW: [f32; 4] = [0.0, 0.0, 0.0, 0.85];

// ─── Échelle ────────────────────────────────────────────────────────────

pub const HUD_SCALE: f32 = 2.0;
pub const LINE_H: f32 = 8.0 * HUD_SCALE + 2.0;

// ─── Safe-area ultra-wide ───────────────────────────────────────────────

/// Aspect maximum auquel les éléments HUD ancrés aux coins (HP, ammo,
/// kill-feed, mini-map…) restent collés. Au-delà — typiquement 21:9
/// (≈ 2.37) et 32:9 (≈ 3.56) — on les ramène vers une zone safe 16:9
/// centrée. Les éléments **plein écran** (vignette, console) continuent
/// d'utiliser `w` et `h` bruts.
pub const HUD_SAFE_MAX_ASPECT: f32 = 16.0 / 9.0;

/// Calcule la zone HUD "safe" en coordonnées écran : un rect 16:9 max
/// centré horizontalement. En 4:3 ou 16:9 → toute la largeur. En
/// 21:9 / 32:9 → bandes ignorées par les ancrages de coin. Retourne
/// `(x, sw)`.
pub fn hud_safe_rect_x(w: f32, h: f32) -> (f32, f32) {
    let max_w = h * HUD_SAFE_MAX_ASPECT;
    if w <= max_w {
        (0.0, w)
    } else {
        let sw = max_w;
        let x = (w - sw) * 0.5;
        (x, sw)
    }
}

// ─── Primitives de dessin HUD ───────────────────────────────────────────

/// Texte avec ombre portée 2 px bas-droite — rend le HUD lisible sur
/// n'importe quel fond. Pas de blur, pas de stroke : on dessine deux fois.
pub fn push_text_shadow(
    r: &mut Renderer,
    x: f32,
    y: f32,
    scale: f32,
    color: [f32; 4],
    text: &str,
) {
    r.push_text(x + 2.0, y + 2.0, scale, COL_TEXT_SHADOW, text);
    r.push_text(x, y, scale, color, text);
}

/// Dessine un panneau HUD moderne : fond anthracite translucide + accents
/// cyan top épais + bottom fin atténué. L'œil lit le contour comme
/// arrondi quand on encadre seulement 2 côtés sur 4.
pub fn push_panel(r: &mut Renderer, x: f32, y: f32, w: f32, h: f32) {
    r.push_rect(x, y, w, h, COL_PANEL_BG);
    r.push_rect(x, y, w, 2.0, COL_PANEL_EDGE);
    r.push_rect(x, y + h - 1.0, w, 1.0, COL_PANEL_EDGE_DIM);
}

/// Barre horizontale avec dégradé linéaire (8 quartiers) entre `low_color`
/// (ratio=0) et `high_color` (ratio=1). Pas un vrai dégradé GPU mais un
/// escalier qui passe à l'œil sur 80-200 px de barre.
pub fn push_bar_gradient(
    r: &mut Renderer,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    ratio: f32,
    low_color: [f32; 3],
    high_color: [f32; 3],
) {
    r.push_rect(x - 1.0, y - 1.0, w + 2.0, h + 2.0, [0.0, 0.0, 0.0, 0.65]);
    r.push_rect(x, y, w, h, [0.10, 0.12, 0.16, 0.85]);
    let fill_w = (w * ratio.clamp(0.0, 1.0)).max(0.0);
    if fill_w < 0.5 {
        return;
    }
    let r_c = low_color[0] + (high_color[0] - low_color[0]) * ratio;
    let g_c = low_color[1] + (high_color[1] - low_color[1]) * ratio;
    let b_c = low_color[2] + (high_color[2] - low_color[2]) * ratio;
    r.push_rect(x, y, fill_w, h, [r_c, g_c, b_c, 0.95]);
    r.push_rect(x, y, fill_w, 1.0, [1.0, 1.0, 1.0, 0.35]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_rect_passthrough_below_or_equal_16_9() {
        // 16:9 et plus étroit (4:3, 16:10) → pas de letterbox HUD.
        let (x, w) = hud_safe_rect_x(1280.0, 720.0); // 16:9
        assert_eq!((x, w), (0.0, 1280.0));
        let (x, w) = hud_safe_rect_x(1024.0, 768.0); // 4:3
        assert_eq!((x, w), (0.0, 1024.0));
    }

    #[test]
    fn safe_rect_clamps_on_ultrawide() {
        // 21:9 (2520×1080) → safe rect = 1920 centré, soit x=300.
        let (x, w) = hud_safe_rect_x(2520.0, 1080.0);
        assert!((w - 1920.0).abs() < 0.5, "21:9 safe_w {w}");
        assert!((x - 300.0).abs() < 0.5, "21:9 safe_x {x}");
        // 32:9 (3840×1080) → safe rect = 1920 centré, x=960.
        let (x, w) = hud_safe_rect_x(3840.0, 1080.0);
        assert!((w - 1920.0).abs() < 0.5, "32:9 safe_w {w}");
        assert!((x - 960.0).abs() < 0.5, "32:9 safe_x {x}");
    }

    #[test]
    fn line_h_matches_hud_scale_plus_padding() {
        // HUD_SCALE=2 → glyph 8×8 mis à l'échelle = 16, +2 padding = 18.
        assert!((LINE_H - 18.0).abs() < 0.001);
    }
}
