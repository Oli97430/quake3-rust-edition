//! Logo « Q3 RUST » — dessiné entièrement à partir des primitives HUD du
//! renderer (rects + texte SDF), pas d'asset à distribuer.
//!
//! L'idée : produire un badge reconnaissable qu'on peut afficher à
//! n'importe quelle taille, n'importe où (menu principal, HUD in-game,
//! splash d'intermission), sans dépendance à une image embarquée.
//!
//! Composition :
//! 1. Rectangle de fond en dégradé (bandeau sombre)
//! 2. Accent orange — barre horizontale sous le nom
//! 3. Ombre portée du texte (offset +2 px) en noir 80 %
//! 4. Texte "Q3 RUST" en couleur chaude claire (crème)
//! 5. Baseline bleu nuit en dessous + tagline "ARENA" plus discret
//!
//! Tout est paramétré par un `scale` unique : 1.0 → badge menu (~680×160
//! à 1080p), 0.35 → watermark HUD (~240×56), etc.

use q3_renderer::Renderer;

/// Palette officielle du logo — une fois définie ici, plus de magic numbers
/// dispersés dans l'UI.  Tout est sRGB linéaire (clamp 0..1).
pub mod palette {
    /// Crème chaude — corps du texte principal.
    pub const CREAM: [f32; 4] = [1.00, 0.92, 0.74, 1.0];
    /// Orange vif — accent / barres / highlights selected.
    pub const ORANGE: [f32; 4] = [1.00, 0.55, 0.15, 1.0];
    /// Bleu nuit — baseline / tagline, fond de badge.
    pub const MIDNIGHT: [f32; 4] = [0.08, 0.12, 0.22, 1.0];
    /// Blanc cassé pour la tagline.
    pub const STEEL: [f32; 4] = [0.75, 0.82, 0.92, 1.0];
}

/// Dessine le logo centré horizontalement sur `cx`, avec sa baseline
/// texte à `cy`.  `scale=1.0` rend le nom en `px_per_glyph = 64` (c.-à-d.
/// `text_scale = 8.0`) — amplement lisible en plein écran menu.
pub fn draw(r: &mut Renderer, cx: f32, cy: f32, scale: f32) {
    // Police SDF : un glyphe = 8 texels logiques × text_scale pixels.
    let text_scale = 8.0 * scale;
    let char_w = 8.0 * text_scale;
    // 7 caractères → garde la largeur visuelle proche de l'ancien
    // "BUKELE" (6 chars) sans casser le layout des padding/centrage.
    let name = "Q3 RUST";
    let name_px = name.len() as f32 * char_w;
    let name_x = cx - name_px * 0.5;
    let name_y = cy;

    // ----------- 1. Bandeau de fond dégradé (deux rects superposés) -----
    // Le "dégradé" est simulé par deux rectangles de couleurs distinctes
    // dont l'empilement donne une transition franche mais stylisée — on
    // n'a pas de gradient primitive, donc on monte / on descend.
    let pad_x = char_w * 0.6;
    let pad_y_top = text_scale * 2.0;
    let pad_y_bot = text_scale * 4.0;
    let bg_w = name_px + pad_x * 2.0;
    let bg_h = text_scale * 8.0 + pad_y_top + pad_y_bot;
    let bg_x = name_x - pad_x;
    let bg_y = name_y - pad_y_top;
    // Moitié haute : bleu nuit presque opaque.
    r.push_rect(
        bg_x,
        bg_y,
        bg_w,
        bg_h * 0.55,
        [
            palette::MIDNIGHT[0],
            palette::MIDNIGHT[1],
            palette::MIDNIGHT[2],
            0.82,
        ],
    );
    // Moitié basse : plus sombre, alpha plus faible → transition douce.
    r.push_rect(
        bg_x,
        bg_y + bg_h * 0.55,
        bg_w,
        bg_h * 0.45,
        [0.04, 0.06, 0.12, 0.70],
    );

    // ----------- 2. Accent orange (barre horizontale) ------------------
    // Juste sous la ligne de texte principal.  L'épaisseur scale avec
    // le logo.
    let bar_y = name_y + text_scale * 8.0 + text_scale * 1.2;
    let bar_h = (text_scale * 0.6).max(2.0);
    r.push_rect(
        name_x - char_w * 0.15,
        bar_y,
        name_px + char_w * 0.30,
        bar_h,
        palette::ORANGE,
    );

    // ----------- 3. Ombre portée du texte principal --------------------
    // Décalage proportionnel au scale pour rester cohérent.
    let shadow_off = (text_scale * 0.18).max(1.5);
    r.push_text(
        name_x + shadow_off,
        name_y + shadow_off,
        text_scale,
        [0.0, 0.0, 0.0, 0.85],
        name,
    );

    // ----------- 4. Texte principal "Q3 RUST" ---------------------------
    r.push_text(name_x, name_y, text_scale, palette::CREAM, name);

    // ----------- 5. Tagline sous la barre orange -----------------------
    let tag = "ARENA";
    let tag_scale = text_scale * 0.45;
    let tag_char_w = 8.0 * tag_scale;
    // Espacement artificiel de +100% entre lettres — "letter-spacing"
    // donne un look "badge gaming" sans rendre une police custom.
    let tag_spacing = tag_char_w * 2.0;
    let tag_px = tag.len() as f32 * tag_spacing;
    let tag_x = cx - tag_px * 0.5 + tag_char_w * 0.5;
    let tag_y = bar_y + bar_h + text_scale * 0.6;
    // On émet chaque lettre séparément pour contrôler le spacing.
    let mut x = tag_x;
    for ch in tag.chars() {
        let s: String = ch.to_string();
        // Petite ombre à chaque lettre pour tenir sur un fond varié.
        r.push_text(
            x + tag_scale * 0.15,
            tag_y + tag_scale * 0.15,
            tag_scale,
            [0.0, 0.0, 0.0, 0.75],
            &s,
        );
        r.push_text(x, tag_y, tag_scale, palette::STEEL, &s);
        x += tag_spacing;
    }
}

/// Variante "watermark" : badge compact dans un coin d'écran, sans
/// tagline ni fond — juste le nom + sa barre de soulignement.  Utile
/// en HUD in-game pour brander l'appli sans obstruer le gameplay.
pub fn draw_watermark(r: &mut Renderer, x: f32, y: f32, scale: f32) {
    let text_scale = 8.0 * scale;
    // 7 caractères → garde la largeur visuelle proche de l'ancien
    // "BUKELE" (6 chars) sans casser le layout des padding/centrage.
    let name = "Q3 RUST";
    let char_w = 8.0 * text_scale;
    let name_px = name.len() as f32 * char_w;
    let shadow_off = (text_scale * 0.15).max(1.0);
    r.push_text(
        x + shadow_off,
        y + shadow_off,
        text_scale,
        [0.0, 0.0, 0.0, 0.65],
        name,
    );
    r.push_text(
        x,
        y,
        text_scale,
        [
            palette::CREAM[0],
            palette::CREAM[1],
            palette::CREAM[2],
            0.85,
        ],
        name,
    );
    // Petite barre orange sous le watermark.
    let bar_h = (text_scale * 0.25).max(1.0);
    r.push_rect(
        x,
        y + text_scale * 8.0 + text_scale * 0.3,
        name_px,
        bar_h,
        [
            palette::ORANGE[0],
            palette::ORANGE[1],
            palette::ORANGE[2],
            0.75,
        ],
    );
}
