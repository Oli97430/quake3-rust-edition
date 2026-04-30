//! Menu principal — écrans Root / Play / Options.
//!
//! Le menu est *piloté* par l'App : celle-ci route les inputs (clavier /
//! souris) vers `Menu::on_*` et récupère une `MenuAction` qui dicte la
//! transition (charger une map, quitter, résumer le jeu).
//!
//! Le dessin réutilise la pipeline HUD du renderer (`push_rect` + `push_text`)
//! — pas de shader dédié, pas de texture supplémentaire : un bandeau plein
//! opaque + le bitmap font 8×8 suffisent pour un look cohérent avec la
//! console et le HUD.
//!
//! # Persistance
//!
//! Les options modifiées s'écrivent directement dans la `CvarRegistry` via
//! la méthode `set`. Comme ces cvars sont enregistrées avec le flag
//! `ARCHIVE`, elles sont automatiquement sauvegardées dans `q3config.cfg`
//! au prochain `exiting` (cf. `app::user_config_path`).

use q3_common::cvar::CvarRegistry;
use q3_renderer::Renderer;
use winit::keyboard::KeyCode;

/// Pages du menu. `Root` est l'entrée, `Play` liste les maps, `Options`
/// expose les cvars ajustables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MenuPage {
    Root,
    Play,
    Options,
}

/// Action retournée par le menu à l'App. `None` = aucun effet ; l'App
/// continue à router les inputs au menu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MenuAction {
    None,
    /// Le joueur ferme le menu — l'App doit rendre la main au gameplay.
    /// Émis sur `RESUME` et sur `Escape` quand le menu est sur Root et
    /// qu'une partie est en cours.
    Resume,
    /// Charger la map dont le chemin VFS est donné (ex `maps/q3dm1.bsp`).
    LoadMap(String),
    /// Demande de quitter l'application.
    Quit,
}

/// Rectangle cliquable en pixels écran — utilisé à la fois pour le dessin
/// et la détection de clic/hover.
#[derive(Debug, Clone, Copy)]
struct ItemRect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl ItemRect {
    fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.w && py >= self.y && py <= self.y + self.h
    }
}

/// État du menu. Mutable — stocké dans `App`.
pub struct Menu {
    /// Le menu est-il affiché et reçoit-il les inputs ?
    pub open: bool,
    /// Page courante.
    pub page: MenuPage,
    /// Index de l'item sélectionné au clavier (flèches haut/bas + Entrée).
    pub selected: usize,
    /// Dernière position souris (coin haut-gauche = (0,0), en pixels).
    /// Utilisée pour colorer l'item survolé et pour décider du clic.
    pub mouse: (f32, f32),
    /// Liste des maps trouvées dans le VFS (cachée au boot pour ne pas
    /// re-scanner à chaque frame).
    pub map_list: Vec<String>,
    /// Offset de défilement pour `Play` quand la liste dépasse l'écran.
    pub play_scroll: usize,
    /// Une partie est-elle en cours ? Active l'item `RESUME` sur Root.
    pub in_game: bool,
}

impl Menu {
    /// Nombre d'items max visibles en même temps sur la page `Play`.
    const PLAY_VISIBLE: usize = 14;
    /// Hauteur d'une ligne de menu, en pixels.
    const LINE_H: f32 = 36.0;
    /// Scale du texte des items (font 8×8 * scale).
    const ITEM_SCALE: f32 = 3.0;
    /// Scale du titre.
    const TITLE_SCALE: f32 = 6.0;

    pub fn new(map_list: Vec<String>, in_game: bool) -> Self {
        Self {
            open: false,
            page: MenuPage::Root,
            selected: 0,
            mouse: (0.0, 0.0),
            map_list,
            play_scroll: 0,
            in_game,
        }
    }

    /// Met à jour la liste des maps (ex. après `fs_restart` ou un mod monté
    /// à chaud). Non-utilisé dans le MVP mais prévu pour plus tard.
    #[allow(dead_code)]
    pub fn set_map_list(&mut self, list: Vec<String>) {
        self.map_list = list;
        if self.page == MenuPage::Play {
            self.selected = 0;
            self.play_scroll = 0;
        }
    }

    /// À appeler quand le joueur entre ou sort du gameplay. Affecte la
    /// présence de l'item `RESUME`.
    pub fn set_in_game(&mut self, in_game: bool) {
        self.in_game = in_game;
        // Si on était sur `Root` et l'item RESUME apparaît/disparaît,
        // l'index sélectionné peut déborder — on le reclampe.
        self.clamp_selected();
    }

    /// Ouvre le menu sur `Root`. Reset le scroll et la sélection.
    pub fn open_root(&mut self) {
        self.open = true;
        self.page = MenuPage::Root;
        self.selected = 0;
        self.play_scroll = 0;
    }

    /// Ferme le menu sans action side-effect — juste l'état local.
    pub fn close(&mut self) {
        self.open = false;
    }

    /// Change de page et reset la sélection/scroll.
    fn go(&mut self, page: MenuPage) {
        self.page = page;
        self.selected = 0;
        self.play_scroll = 0;
    }

    fn clamp_selected(&mut self) {
        let count = self.item_count();
        if count == 0 {
            self.selected = 0;
        } else if self.selected >= count {
            self.selected = count - 1;
        }
    }

    /// Nombre d'items interactifs de la page courante.
    fn item_count(&self) -> usize {
        match self.page {
            MenuPage::Root => {
                // PLAY, OPTIONS, QUIT, (+ RESUME si in_game).
                3 + if self.in_game { 1 } else { 0 }
            }
            MenuPage::Play => {
                // 1 item "BACK" + N maps.
                1 + self.map_list.len()
            }
            MenuPage::Options => {
                // BACK + 4 réglages.
                5
            }
        }
    }

    // ============================================================
    //   Input
    // ============================================================

    /// Gère une touche clavier. Retourne l'action résultante.
    pub fn on_key(&mut self, key: KeyCode, cvars: &CvarRegistry) -> MenuAction {
        match key {
            KeyCode::Escape => {
                if self.page == MenuPage::Root {
                    if self.in_game {
                        self.open = false;
                        return MenuAction::Resume;
                    } else {
                        return MenuAction::Quit;
                    }
                } else {
                    self.go(MenuPage::Root);
                }
            }
            KeyCode::ArrowUp | KeyCode::KeyW => {
                if self.selected == 0 {
                    self.selected = self.item_count().saturating_sub(1);
                } else {
                    self.selected -= 1;
                }
                self.ensure_scroll_in_view();
            }
            KeyCode::ArrowDown | KeyCode::KeyS => {
                let n = self.item_count();
                if n > 0 {
                    self.selected = (self.selected + 1) % n;
                }
                self.ensure_scroll_in_view();
            }
            KeyCode::Enter | KeyCode::NumpadEnter | KeyCode::Space => {
                return self.activate_selected(cvars);
            }
            KeyCode::ArrowLeft => {
                if self.page == MenuPage::Options {
                    self.adjust_option(self.selected, -1, cvars);
                }
            }
            KeyCode::ArrowRight => {
                if self.page == MenuPage::Options {
                    self.adjust_option(self.selected, 1, cvars);
                }
            }
            _ => {}
        }
        MenuAction::None
    }

    /// Met à jour la position souris. Le clamp de `selected` suit l'hover
    /// pour que le clavier enchaîne depuis l'item sous le curseur.
    pub fn on_mouse_move(&mut self, x: f32, y: f32, fb_w: f32, fb_h: f32) {
        self.mouse = (x, y);
        // Recalcule les rects et cherche celui qui contient la souris.
        let rects = self.layout(fb_w, fb_h);
        for (i, r) in rects.iter().enumerate() {
            if r.contains(x, y) {
                self.selected = i;
                break;
            }
        }
    }

    /// Clic gauche — active l'item sous le curseur s'il y en a un.
    pub fn on_mouse_click(
        &mut self,
        x: f32,
        y: f32,
        fb_w: f32,
        fb_h: f32,
        cvars: &CvarRegistry,
    ) -> MenuAction {
        let rects = self.layout(fb_w, fb_h);
        for (i, r) in rects.iter().enumerate() {
            if r.contains(x, y) {
                self.selected = i;
                return self.activate_selected(cvars);
            }
        }
        MenuAction::None
    }

    /// Molette — sur la page `Play`, défile la liste.
    pub fn on_scroll(&mut self, lines: f32) {
        if self.page != MenuPage::Play {
            return;
        }
        let delta = lines.round() as isize;
        if delta > 0 {
            self.play_scroll = self.play_scroll.saturating_sub(delta as usize);
        } else if delta < 0 {
            let max = self
                .map_list
                .len()
                .saturating_sub(Self::PLAY_VISIBLE);
            self.play_scroll = (self.play_scroll + (-delta) as usize).min(max);
        }
    }

    fn ensure_scroll_in_view(&mut self) {
        if self.page != MenuPage::Play {
            return;
        }
        // Index dans la liste maps (selected 0 = BACK, 1..N+1 = maps).
        if self.selected == 0 {
            self.play_scroll = 0;
            return;
        }
        let map_idx = self.selected - 1;
        if map_idx < self.play_scroll {
            self.play_scroll = map_idx;
        } else if map_idx >= self.play_scroll + Self::PLAY_VISIBLE {
            self.play_scroll = map_idx + 1 - Self::PLAY_VISIBLE;
        }
    }

    fn activate_selected(&mut self, cvars: &CvarRegistry) -> MenuAction {
        match self.page {
            MenuPage::Root => {
                // Ordre logique fixe : 0=RESUME, 1=PLAY, 2=OPTIONS, 3=QUIT.
                // Sans partie en cours, l'item RESUME est caché et tous les
                // autres glissent d'un cran vers le haut, d'où l'offset +1.
                let offset: usize = if self.in_game { 0 } else { 1 };
                let logical = self.selected + offset;
                match logical {
                    0 => {
                        return MenuAction::Resume;
                    }
                    1 => self.go(MenuPage::Play),
                    2 => self.go(MenuPage::Options),
                    3 => return MenuAction::Quit,
                    _ => {}
                }
            }
            MenuPage::Play => {
                if self.selected == 0 {
                    self.go(MenuPage::Root);
                } else {
                    let idx = self.selected - 1;
                    if let Some(map) = self.map_list.get(idx).cloned() {
                        // Après avoir déclenché le load, on referme le menu
                        // pour redonner le contrôle au jeu dès la première
                        // frame post-load.
                        self.open = false;
                        return MenuAction::LoadMap(map);
                    }
                }
            }
            MenuPage::Options => {
                if self.selected == 0 {
                    self.go(MenuPage::Root);
                } else {
                    // Un clic sur un réglage numérique = pas d'effet
                    // direct ; on documente en bas de page qu'on utilise
                    // les flèches. Sur le toggle (invert pitch), on inverse.
                    if self.selected == 4 {
                        let cur = cvars.get_f32("m_pitch").unwrap_or(0.022);
                        let _ = cvars.set("m_pitch", &format!("{:.3}", -cur));
                    }
                }
            }
        }
        MenuAction::None
    }

    fn adjust_option(&mut self, index: usize, dir: i32, cvars: &CvarRegistry) {
        let d = dir as f32;
        match index {
            1 => {
                // sensitivity : pas 0.5, clampé [0.5, 30]
                let cur = cvars.get_f32("sensitivity").unwrap_or(5.0);
                let new = (cur + 0.5 * d).clamp(0.5, 30.0);
                let _ = cvars.set("sensitivity", &format!("{new:.2}"));
            }
            2 => {
                // s_volume : pas 0.05, clampé [0, 1]
                let cur = cvars.get_f32("s_volume").unwrap_or(0.8);
                let new = (cur + 0.05 * d).clamp(0.0, 1.0);
                let _ = cvars.set("s_volume", &format!("{new:.2}"));
            }
            3 => {
                // cg_fov : pas 1, clampé [60, 130]. Enregistrée à la
                // première modif si absente (cas où on part d'un install
                // sans config.cfg).
                let cur = cvars.get_f32("cg_fov").unwrap_or(90.0);
                let new = (cur + 1.0 * d).clamp(60.0, 130.0);
                let _ = cvars.set("cg_fov", &format!("{new:.0}"));
            }
            4 => {
                // toggle invert pitch — flip le signe de m_pitch
                let cur = cvars.get_f32("m_pitch").unwrap_or(0.022);
                let _ = cvars.set("m_pitch", &format!("{:.3}", -cur));
            }
            _ => {}
        }
    }

    // ============================================================
    //   Layout + Draw
    // ============================================================

    /// Calcule les rectangles cliquables des items de la page courante.
    /// L'ordre du `Vec` correspond à `selected` — `rects[self.selected]`
    /// est toujours l'item actif clavier.
    fn layout(&self, fb_w: f32, fb_h: f32) -> Vec<ItemRect> {
        let mut out = Vec::new();
        let item_w = (fb_w * 0.5).max(320.0);
        let x = (fb_w - item_w) * 0.5;
        let mut y = fb_h * 0.25 + Self::TITLE_SCALE * 8.0 + 40.0;
        let line_h = Self::LINE_H;

        match self.page {
            MenuPage::Root => {
                let mut items: Vec<&str> = Vec::with_capacity(4);
                if self.in_game {
                    items.push("RESUME");
                }
                items.push("PLAY");
                items.push("OPTIONS");
                items.push("QUIT");
                for _ in items {
                    out.push(ItemRect { x, y, w: item_w, h: line_h });
                    y += line_h + 8.0;
                }
            }
            MenuPage::Play => {
                out.push(ItemRect { x, y, w: item_w, h: line_h });
                y += line_h + 16.0;
                let visible_end = (self.play_scroll + Self::PLAY_VISIBLE)
                    .min(self.map_list.len());
                for _ in self.play_scroll..visible_end {
                    out.push(ItemRect { x, y, w: item_w, h: line_h });
                    y += line_h + 2.0;
                }
                // Les maps hors de la fenêtre scrollée ont un rect
                // "fantôme" (hors écran) pour garder l'indexation stable.
                // Sans ça, `selected` pointerait vers le mauvais item.
                let total_maps = self.map_list.len();
                for _ in visible_end..(self.play_scroll + total_maps - visible_end + self.play_scroll) {
                    out.push(ItemRect { x, y: -1000.0, w: 0.0, h: 0.0 });
                }
                // Simplification : on émet autant de rects que d'items
                // (BACK + toutes les maps) ; ceux hors-écran sont marqués
                // (-1000, 0, 0). On corrige ici pour rester simple :
                out.clear();
                out.push(ItemRect {
                    x,
                    y: fb_h * 0.25 + Self::TITLE_SCALE * 8.0 + 40.0,
                    w: item_w,
                    h: line_h,
                });
                let mut yy = fb_h * 0.25 + Self::TITLE_SCALE * 8.0 + 40.0 + line_h + 16.0;
                for i in 0..self.map_list.len() {
                    let visible = i >= self.play_scroll
                        && i < self.play_scroll + Self::PLAY_VISIBLE;
                    if visible {
                        out.push(ItemRect { x, y: yy, w: item_w, h: line_h });
                        yy += line_h + 2.0;
                    } else {
                        out.push(ItemRect { x: -1000.0, y: -1000.0, w: 0.0, h: 0.0 });
                    }
                }
            }
            MenuPage::Options => {
                // BACK + 4 réglages
                for _ in 0..5 {
                    out.push(ItemRect { x, y, w: item_w, h: line_h });
                    y += line_h + 12.0;
                }
            }
        }
        out
    }

    /// Dessine le menu. L'App l'appelle *après* le reste du HUD pour
    /// qu'il soit au premier plan. Un bandeau opaque dim l'arrière-plan.
    pub fn draw(&self, r: &mut Renderer, cvars: &CvarRegistry, fb_w: f32, fb_h: f32) {
        if !self.open {
            return;
        }

        // Arrière-plan assombri (le jeu reste vaguement visible derrière).
        let bg_alpha = if self.in_game { 0.78 } else { 1.0 };
        r.push_rect(0.0, 0.0, fb_w, fb_h, [0.04, 0.04, 0.06, bg_alpha]);

        // Sur la page d'accueil on dessine le vrai logo Q3 RUST (badge
        // stylisé, pas juste du texte).  Sur les sous-pages on garde le
        // titre de section en typographie classique pour ne pas écraser
        // le contenu avec 2 barres décoratives.
        match self.page {
            MenuPage::Root => {
                // Logo centré dans le tiers supérieur.
                let cx = fb_w * 0.5;
                let cy = fb_h * 0.12;
                // Scale adapté à la largeur de la fenêtre — on vise ~40 %
                // de la largeur d'écran pour le nom, ce qui donne une
                // présence forte sans dépasser.
                let target_name_frac = 0.40;
                let text_scale = (fb_w * target_name_frac) / (8.0 * 8.0 * "Q3 RUST".len() as f32);
                let scale = (text_scale / 8.0).clamp(2.0, 10.0);
                crate::logo::draw(r, cx, cy, scale);
            }
            _ => {
                let title = match self.page {
                    MenuPage::Play => "SELECT MAP",
                    MenuPage::Options => "OPTIONS",
                    MenuPage::Root => unreachable!(),
                };
                let title_px = Self::TITLE_SCALE * 8.0 * title.len() as f32;
                let title_x = (fb_w - title_px) * 0.5;
                let title_y = fb_h * 0.12;
                r.push_text(
                    title_x + 2.0,
                    title_y + 2.0,
                    Self::TITLE_SCALE,
                    [0.0, 0.0, 0.0, 1.0],
                    title,
                );
                r.push_text(
                    title_x,
                    title_y,
                    Self::TITLE_SCALE,
                    crate::logo::palette::ORANGE,
                    title,
                );
            }
        }

        // Items
        let rects = self.layout(fb_w, fb_h);
        let labels = self.labels(cvars);
        for (i, (rect, label)) in rects.iter().zip(labels.iter()).enumerate() {
            if rect.w <= 0.0 {
                continue; // item hors-écran (scroll)
            }
            let is_selected = i == self.selected;
            let is_hover = rect.contains(self.mouse.0, self.mouse.1);
            // Fond de l'item
            let bg = if is_selected {
                [1.0, 0.5, 0.15, 0.25]
            } else if is_hover {
                [1.0, 1.0, 1.0, 0.08]
            } else {
                [1.0, 1.0, 1.0, 0.02]
            };
            r.push_rect(rect.x, rect.y, rect.w, rect.h, bg);
            // Barre gauche indicateur sélection (orange plein)
            if is_selected {
                r.push_rect(rect.x, rect.y, 6.0, rect.h, [1.0, 0.5, 0.15, 1.0]);
            }
            // Label
            let color = if is_selected {
                [1.0, 0.8, 0.4, 1.0]
            } else {
                [0.85, 0.85, 0.85, 1.0]
            };
            let tx = rect.x + 18.0;
            let ty = rect.y + (rect.h - Self::ITEM_SCALE * 8.0) * 0.5;
            r.push_text(tx, ty, Self::ITEM_SCALE, color, label);
        }

        // Pied de page : aide contextuelle
        let hint = match self.page {
            MenuPage::Root => "UP/DOWN: select     ENTER: confirm     ESC: back/quit",
            MenuPage::Play => {
                "UP/DOWN: select     ENTER: load     WHEEL: scroll     ESC: back"
            }
            MenuPage::Options => "LEFT/RIGHT: adjust     ENTER: toggle     ESC: back",
        };
        let hint_scale = 2.0;
        let hint_px = hint_scale * 8.0 * hint.len() as f32;
        let hint_x = (fb_w - hint_px) * 0.5;
        let hint_y = fb_h - 40.0;
        r.push_text(hint_x, hint_y, hint_scale, [0.6, 0.6, 0.6, 1.0], hint);
    }

    /// Retourne les labels affichés pour la page courante, dans le même
    /// ordre que `layout`.
    fn labels(&self, cvars: &CvarRegistry) -> Vec<String> {
        match self.page {
            MenuPage::Root => {
                let mut v: Vec<String> = Vec::with_capacity(4);
                if self.in_game {
                    v.push("RESUME".into());
                }
                v.push("PLAY".into());
                v.push("OPTIONS".into());
                v.push("QUIT".into());
                v
            }
            MenuPage::Play => {
                let mut v: Vec<String> = Vec::with_capacity(self.map_list.len() + 1);
                v.push("< BACK".into());
                for m in &self.map_list {
                    // Affiche le basename sans `maps/` ni `.bsp` pour un
                    // visuel plus propre. On garde le path complet dans
                    // `map_list` pour le load.
                    let short = m
                        .strip_prefix("maps/")
                        .unwrap_or(m)
                        .strip_suffix(".bsp")
                        .unwrap_or(m);
                    v.push(short.to_string());
                }
                v
            }
            MenuPage::Options => {
                let sens = cvars.get_f32("sensitivity").unwrap_or(5.0);
                let vol = cvars.get_f32("s_volume").unwrap_or(0.8);
                let fov = cvars.get_f32("cg_fov").unwrap_or(90.0);
                let inv = cvars.get_f32("m_pitch").unwrap_or(0.022) < 0.0;
                vec![
                    "< BACK".into(),
                    format!("SENSITIVITY        {:>5.2}", sens),
                    format!("VOLUME             {:>5.2}", vol),
                    format!("FOV                {:>5.0}", fov),
                    format!("INVERT PITCH       {}", if inv { "ON" } else { "OFF" }),
                ]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(in_game: bool, maps: &[&str]) -> (Menu, CvarRegistry) {
        let cvars = CvarRegistry::new();
        use q3_common::cvar::CvarFlags;
        cvars.register("sensitivity", "5.0", CvarFlags::ARCHIVE);
        cvars.register("s_volume", "0.8", CvarFlags::ARCHIVE);
        cvars.register("m_pitch", "0.022", CvarFlags::ARCHIVE);
        cvars.register("cg_fov", "90", CvarFlags::ARCHIVE);
        let menu = Menu::new(maps.iter().map(|s| s.to_string()).collect(), in_game);
        (menu, cvars)
    }

    #[test]
    fn root_without_game_has_play_options_quit() {
        let (menu, _) = fixture(false, &[]);
        assert_eq!(menu.item_count(), 3);
    }

    #[test]
    fn root_with_game_includes_resume() {
        let (menu, _) = fixture(true, &[]);
        assert_eq!(menu.item_count(), 4);
    }

    #[test]
    fn escape_on_root_without_game_quits() {
        let (mut menu, cv) = fixture(false, &[]);
        menu.open_root();
        assert_eq!(menu.on_key(KeyCode::Escape, &cv), MenuAction::Quit);
    }

    #[test]
    fn escape_on_root_with_game_resumes() {
        let (mut menu, cv) = fixture(true, &[]);
        menu.open_root();
        assert_eq!(menu.on_key(KeyCode::Escape, &cv), MenuAction::Resume);
        assert!(!menu.open);
    }

    #[test]
    fn play_navigation_loads_map() {
        let (mut menu, cv) = fixture(false, &["maps/q3dm1.bsp", "maps/q3dm6.bsp"]);
        menu.open_root();
        // Navigate to PLAY (first item without RESUME)
        assert_eq!(menu.on_key(KeyCode::Enter, &cv), MenuAction::None);
        assert_eq!(menu.page, MenuPage::Play);
        // ↓ passe BACK et tombe sur q3dm1
        menu.on_key(KeyCode::ArrowDown, &cv);
        assert_eq!(menu.selected, 1);
        let act = menu.on_key(KeyCode::Enter, &cv);
        assert_eq!(act, MenuAction::LoadMap("maps/q3dm1.bsp".into()));
        assert!(!menu.open, "charger une map referme le menu");
    }

    #[test]
    fn options_left_right_adjust_sensitivity() {
        let (mut menu, cv) = fixture(false, &[]);
        menu.open_root();
        // Root → OPTIONS (index 1 sans RESUME : PLAY=0, OPTIONS=1)
        menu.selected = 1;
        menu.on_key(KeyCode::Enter, &cv);
        assert_eq!(menu.page, MenuPage::Options);
        // index 1 = sensitivity ; ArrowRight augmente de 0.5
        menu.selected = 1;
        menu.on_key(KeyCode::ArrowRight, &cv);
        assert_eq!(cv.get_f32("sensitivity"), Some(5.5));
        menu.on_key(KeyCode::ArrowLeft, &cv);
        menu.on_key(KeyCode::ArrowLeft, &cv);
        assert_eq!(cv.get_f32("sensitivity"), Some(4.5));
    }

    #[test]
    fn options_sensitivity_clamps() {
        let (mut menu, cv) = fixture(false, &[]);
        menu.page = MenuPage::Options;
        menu.selected = 1;
        for _ in 0..200 {
            menu.on_key(KeyCode::ArrowRight, &cv);
        }
        assert_eq!(cv.get_f32("sensitivity"), Some(30.0));
        for _ in 0..200 {
            menu.on_key(KeyCode::ArrowLeft, &cv);
        }
        assert_eq!(cv.get_f32("sensitivity"), Some(0.5));
    }

    #[test]
    fn options_invert_pitch_toggles_sign() {
        let (mut menu, cv) = fixture(false, &[]);
        menu.page = MenuPage::Options;
        menu.selected = 4;
        menu.on_key(KeyCode::Enter, &cv);
        assert!(cv.get_f32("m_pitch").unwrap() < 0.0);
        menu.on_key(KeyCode::Enter, &cv);
        assert!(cv.get_f32("m_pitch").unwrap() > 0.0);
    }

    #[test]
    fn escape_from_subpage_goes_to_root() {
        let (mut menu, cv) = fixture(false, &[]);
        menu.open_root();
        menu.page = MenuPage::Options;
        assert_eq!(menu.on_key(KeyCode::Escape, &cv), MenuAction::None);
        assert_eq!(menu.page, MenuPage::Root);
    }

    #[test]
    fn mouse_click_loads_map() {
        let (mut menu, cv) = fixture(false, &["maps/dm1.bsp", "maps/dm2.bsp"]);
        menu.open_root();
        menu.page = MenuPage::Play;
        let rects = menu.layout(1280.0, 720.0);
        // rects[1] = première map après BACK
        let r = rects[1];
        let action = menu.on_mouse_click(r.x + r.w * 0.5, r.y + r.h * 0.5, 1280.0, 720.0, &cv);
        assert_eq!(action, MenuAction::LoadMap("maps/dm1.bsp".into()));
    }

    #[test]
    fn scroll_keeps_selection_in_view() {
        let mut maps: Vec<String> = (0..30).map(|i| format!("maps/m{i}.bsp")).collect();
        let cv = CvarRegistry::new();
        let mut menu = Menu::new(std::mem::take(&mut maps), false);
        menu.page = MenuPage::Play;
        // Sélectionne le dernier item (index = 30)
        menu.selected = 30;
        menu.ensure_scroll_in_view();
        // map_idx = 29 ; window 14 ; scroll = 29 + 1 - 14 = 16
        assert_eq!(menu.play_scroll, 16);
        // Navigation vers le haut → scroll remonte
        menu.selected = 1;
        menu.ensure_scroll_in_view();
        assert_eq!(menu.play_scroll, 0);
        let _ = cv;
    }
}
