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
    /// Sous-page « réglages gameplay » — sensibilité, FOV, invert pitch.
    Gameplay,
    /// Sous-page « affichage » — résolution, fullscreen, vsync, bloom.
    Video,
    /// Sous-page « audio » — master, musique, SFX.
    Audio,
    /// Sous-page « lecteur audio » — liste des fichiers music du
    /// dossier utilisateur + bouton stop.
    Music,
    /// **Map downloader** (v0.9.5++) — liste des maps community du
    /// catalogue + statut de téléchargement.  Click → lance DL.
    MapDownloader,
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
    /// Applique la résolution `(width, height)` à la fenêtre via winit.
    /// L'App appelle `Window::request_inner_size`. Sur fullscreen
    /// borderless, la taille est ignorée par l'OS mais on garde la
    /// trace pour les retours en mode fenêtré.
    ApplyResolution { width: u32, height: u32 },
    /// Bascule entre fenêtré et fullscreen borderless.
    ToggleFullscreen,
    /// Active / désactive vsync. Demande recréation de la swapchain.
    ToggleVsync,
    /// **Music player** (v0.9.5+) — joue le fichier audio donné en
    /// loop comme musique de fond.  Path absolu.
    PlayMusicFile(std::path::PathBuf),
    /// Stoppe la musique de fond.
    StopMusic,
    /// **Map downloader** (v0.9.5++) — lance le DL d'une map du
    /// catalogue par son `id`.  Le moteur utilise `MapDownloader::start`.
    DownloadMap(String),
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

/// Résolutions proposées dans la sous-page Vidéo. Cycle via flèches
/// gauche/droite.  Couvre les ratios 16:9 / 16:10 standard, le 21:9
/// ultra-wide (UW), le 32:9 super-ultra-wide (SUW) et la 4K.
///
/// Le moteur applique automatiquement un scaling **Hor+** sur l'aspect
/// (cf. `Camera::set_horizontal_fov_4_3`) — passer en 32:9 élargit le
/// champ visuel horizontalement sans rogner verticalement, ce qui est
/// le comportement attendu par les joueurs sur écrans ultra-larges.
pub const RESOLUTIONS: &[(u32, u32, &str)] = &[
    // ─── 16:9 / 16:10 standard ───
    (1280,  720, "1280×720 HD 16:9"),
    (1600,  900, "1600×900 16:9"),
    (1920, 1080, "1920×1080 FHD 16:9"),
    (1920, 1200, "1920×1200 16:10"),
    (2560, 1440, "2560×1440 QHD 16:9"),
    (3840, 2160, "3840×2160 4K 16:9"),
    // ─── 21:9 ultra-wide ───
    (2560, 1080, "2560×1080 UW 21:9"),
    (3440, 1440, "3440×1440 UWQHD 21:9"),
    (3840, 1600, "3840×1600 UW+ 21:9"),
    (5120, 2160, "5120×2160 5K2K 21:9"),
    // ─── 32:9 super-ultra-wide ───
    (3840, 1080, "3840×1080 SUW 32:9"),
    (5120, 1440, "5120×1440 DQHD 32:9"),
    (7680, 2160, "7680×2160 Dual4K 32:9"),
];

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
    /// Index courant dans `RESOLUTIONS` — affiché dans la page Vidéo,
    /// avancé via flèches G/D, appliqué via Entrée.
    pub resolution_idx: usize,
    /// État courant fullscreen (affiché ON/OFF dans la page Vidéo).
    pub fullscreen: bool,
    /// Vsync activée (affichée ON/OFF, toggleable).
    pub vsync: bool,
    /// **Music player** (v0.9.5+) — liste des fichiers audio détectés
    /// (rafraîchie à l'entrée de la sous-page Music).  Chaque entrée
    /// est un path absolu cliquable pour lancer la lecture.
    pub music_tracks: Vec<std::path::PathBuf>,
    /// Path de la track en cours de lecture, `None` = silence.
    /// Affiché en haut de la page Music pour feedback visuel.
    pub music_now_playing: Option<std::path::PathBuf>,
    /// Offset de scroll pour la page Music quand >N tracks.
    pub music_scroll: usize,
    /// **Map downloader** (v0.9.5++) — catalogue résolu au boot,
    /// `(id, label)` chacun cliquable pour lancer le DL.
    pub mapdl_catalog: Vec<(String, String)>,
    /// Status courant du job de DL (texte court à afficher en haut
    /// de la page MapDownloader).  Vide si idle.
    pub mapdl_status: String,
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
            resolution_idx: 2, // 1920×1080 par défaut
            fullscreen: false,
            vsync: true,
            music_tracks: Vec::new(),
            music_now_playing: None,
            music_scroll: 0,
            mapdl_catalog: Vec::new(),
            mapdl_status: String::new(),
        }
    }

    /// Charge le catalogue map downloader dans le menu.  Appelé par
    /// l'App au boot (et après un éventuel reload futur).
    pub fn set_mapdl_catalog(&mut self, entries: Vec<(String, String)>) {
        self.mapdl_catalog = entries;
    }

    /// Met à jour la ligne de statut du downloader.
    pub fn set_mapdl_status(&mut self, status: String) {
        self.mapdl_status = status;
    }

    /// Recharge la liste des fichiers music affichés dans la sous-page.
    pub fn set_music_tracks(&mut self, tracks: Vec<std::path::PathBuf>) {
        self.music_tracks = tracks;
        self.music_scroll = 0;
        // clamp selected si on en avait sélectionné un.
        let n = self.item_count();
        if self.selected >= n {
            self.selected = n.saturating_sub(1);
        }
    }

    pub fn set_music_now_playing(&mut self, p: Option<std::path::PathBuf>) {
        self.music_now_playing = p;
    }

    /// Synchronise l'état "résolution courante" du menu avec la taille
    /// fenêtre réelle (utile au boot ou après un resize externe).  Si
    /// la taille ne matche aucun preset, on prend le plus proche.
    pub fn set_window_size(&mut self, width: u32, height: u32) {
        let mut best = (0usize, u32::MAX);
        for (i, &(w, h, _)) in RESOLUTIONS.iter().enumerate() {
            let dw = w.abs_diff(width);
            let dh = h.abs_diff(height);
            let score = dw + dh;
            if score < best.1 {
                best = (i, score);
            }
        }
        self.resolution_idx = best.0;
    }

    /// Synchronise l'état fullscreen du menu (appelé après que l'App
    /// applique le toggle pour que l'UI affiche l'état réel).
    pub fn set_fullscreen(&mut self, fs: bool) {
        self.fullscreen = fs;
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
            MenuPage::Play => 1 + self.map_list.len(),
            MenuPage::Options => {
                // BACK + 4 sous-pages (Gameplay/Video/Audio/MapDownloader).
                5
            }
            MenuPage::Gameplay => {
                // BACK + sensitivity + FOV + invert pitch + mouse smoothing.
                5
            }
            MenuPage::Video => {
                // BACK + résolution + fullscreen + vsync + bloom + apply.
                6
            }
            MenuPage::Audio => {
                // BACK + master + SFX + music + button MUSIC PLAYER.
                5
            }
            MenuPage::Music => {
                // BACK + STOP + N tracks visibles (cap 12). Si la
                // liste est vide, on garde juste 2 items interactifs
                // (les 2 lignes "no music" sont des labels indicatifs
                // — on les ajoute aux labels mais elles ne comptent
                // pas dans item_count pour ne pas pouvoir les sélectionner).
                if self.music_tracks.is_empty() {
                    2
                } else {
                    2 + self.music_tracks.len().min(12)
                }
            }
            MenuPage::MapDownloader => {
                // BACK + N entrées catalogue (cap 12 visibles).
                1 + self.mapdl_catalog.len().min(12)
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
            KeyCode::ArrowLeft => match self.page {
                MenuPage::Gameplay | MenuPage::Video | MenuPage::Audio => {
                    self.adjust_option(self.selected, -1, cvars);
                }
                _ => {}
            },
            KeyCode::ArrowRight => match self.page {
                MenuPage::Gameplay | MenuPage::Video | MenuPage::Audio => {
                    self.adjust_option(self.selected, 1, cvars);
                }
                _ => {}
            },
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
                let offset: usize = if self.in_game { 0 } else { 1 };
                let logical = self.selected + offset;
                match logical {
                    0 => return MenuAction::Resume,
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
                        self.open = false;
                        return MenuAction::LoadMap(map);
                    }
                }
            }
            MenuPage::Options => match self.selected {
                0 => self.go(MenuPage::Root),
                1 => self.go(MenuPage::Gameplay),
                2 => self.go(MenuPage::Video),
                3 => self.go(MenuPage::Audio),
                4 => self.go(MenuPage::MapDownloader),
                _ => {}
            },
            MenuPage::Gameplay => {
                if self.selected == 0 {
                    self.go(MenuPage::Options);
                } else if self.selected == 3 {
                    // Toggle invert pitch — flip signe m_pitch.
                    let cur = cvars.get_f32("m_pitch").unwrap_or(0.022);
                    let _ = cvars.set("m_pitch", &format!("{:.3}", -cur));
                }
            }
            MenuPage::Video => match self.selected {
                0 => self.go(MenuPage::Options),
                2 => return MenuAction::ToggleFullscreen,
                3 => return MenuAction::ToggleVsync,
                4 => {
                    // Toggle bloom via cvar.
                    let cur = cvars.get_i32("r_bloom").unwrap_or(1);
                    let _ = cvars.set("r_bloom", &format!("{}", 1 - cur));
                }
                5 => {
                    // APPLY → applique la résolution courante.
                    let (w, h, _) = RESOLUTIONS[self.resolution_idx];
                    return MenuAction::ApplyResolution { width: w, height: h };
                }
                _ => {}
            },
            MenuPage::Audio => match self.selected {
                0 => self.go(MenuPage::Options),
                4 => self.go(MenuPage::Music),
                _ => {}
            },
            MenuPage::Music => {
                if self.selected == 0 {
                    self.go(MenuPage::Audio);
                } else if self.selected == 1 {
                    return MenuAction::StopMusic;
                } else {
                    let idx = self.selected - 2 + self.music_scroll;
                    if let Some(path) = self.music_tracks.get(idx).cloned() {
                        return MenuAction::PlayMusicFile(path);
                    }
                }
            }
            MenuPage::MapDownloader => {
                if self.selected == 0 {
                    self.go(MenuPage::Options);
                } else {
                    let idx = self.selected - 1;
                    if let Some((id, _label)) = self.mapdl_catalog.get(idx).cloned() {
                        return MenuAction::DownloadMap(id);
                    }
                }
            }
        }
        MenuAction::None
    }

    fn adjust_option(&mut self, index: usize, dir: i32, cvars: &CvarRegistry) {
        let d = dir as f32;
        match self.page {
            MenuPage::Gameplay => match index {
                1 => {
                    let cur = cvars.get_f32("sensitivity").unwrap_or(5.0);
                    let new = (cur + 0.5 * d).clamp(0.5, 30.0);
                    let _ = cvars.set("sensitivity", &format!("{new:.2}"));
                }
                2 => {
                    let cur = cvars.get_f32("cg_fov").unwrap_or(90.0);
                    let new = (cur + 1.0 * d).clamp(60.0, 130.0);
                    let _ = cvars.set("cg_fov", &format!("{new:.0}"));
                }
                3 => {
                    let cur = cvars.get_f32("m_pitch").unwrap_or(0.022);
                    let _ = cvars.set("m_pitch", &format!("{:.3}", -cur));
                }
                4 => {
                    // mouse smoothing : 0..1, pas 0.05. 0 = pas de smoothing.
                    let cur = cvars.get_f32("m_smoothing").unwrap_or(0.0);
                    let new = (cur + 0.05 * d).clamp(0.0, 0.9);
                    let _ = cvars.set("m_smoothing", &format!("{new:.2}"));
                }
                _ => {}
            },
            MenuPage::Video => match index {
                1 => {
                    // Cycle résolution.
                    let n = RESOLUTIONS.len();
                    let cur = self.resolution_idx as i32;
                    let next = (cur + dir).rem_euclid(n as i32) as usize;
                    self.resolution_idx = next;
                }
                _ => {} // les toggles/apply passent par activate_selected
            },
            MenuPage::Audio => match index {
                1 => {
                    let cur = cvars.get_f32("s_volume").unwrap_or(0.8);
                    let new = (cur + 0.05 * d).clamp(0.0, 1.0);
                    let _ = cvars.set("s_volume", &format!("{new:.2}"));
                }
                2 => {
                    let cur = cvars.get_f32("s_sfxvolume").unwrap_or(1.0);
                    let new = (cur + 0.05 * d).clamp(0.0, 1.0);
                    let _ = cvars.set("s_sfxvolume", &format!("{new:.2}"));
                }
                3 => {
                    let cur = cvars.get_f32("s_musicvolume").unwrap_or(0.25);
                    let new = (cur + 0.05 * d).clamp(0.0, 1.0);
                    let _ = cvars.set("s_musicvolume", &format!("{new:.2}"));
                }
                _ => {}
            },
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
                for _ in 0..self.item_count() {
                    out.push(ItemRect { x, y, w: item_w, h: line_h });
                    y += line_h + 12.0;
                }
            }
            MenuPage::Gameplay | MenuPage::Video | MenuPage::Audio => {
                for _ in 0..self.item_count() {
                    out.push(ItemRect { x, y, w: item_w, h: line_h });
                    y += line_h + 10.0;
                }
            }
            MenuPage::Music => {
                for _ in 0..self.item_count() {
                    out.push(ItemRect { x, y, w: item_w, h: line_h });
                    y += line_h + 4.0; // tracks compactées
                }
            }
            MenuPage::MapDownloader => {
                for _ in 0..self.item_count() {
                    out.push(ItemRect { x, y, w: item_w, h: line_h });
                    y += line_h + 4.0;
                }
            }
        }
        out
    }

    /// Dessine le menu. L'App l'appelle *après* le reste du HUD pour
    /// qu'il soit au premier plan. Un bandeau opaque dim l'arrière-plan.
    ///
    /// **Refonte v0.9.5 — look moderne** :
    /// * fond gradient (anthracite → bleu nuit) au lieu d'un aplat
    /// * vignette sombre sur les bords pour cadrer le regard
    /// * panel central glassy (rect avec bordure orange + double layer)
    /// * items en cards (fond + bordure + ombre portée)
    /// * indicateur de sélection : barre orange + chevron `▶` à gauche
    /// * footer pillé sur fond translucide pour rester lisible
    pub fn draw(&self, r: &mut Renderer, cvars: &CvarRegistry, fb_w: f32, fb_h: f32) {
        if !self.open {
            return;
        }

        // ─────────── Arrière-plan moderne (v0.9.5++) ───────────
        // 1) base très sombre quasi-noir avec très léger bleu (clean)
        let bg_alpha = if self.in_game { 0.88 } else { 1.0 };
        r.push_rect(0.0, 0.0, fb_w, fb_h, [0.02, 0.03, 0.05, bg_alpha]);
        // 2) **Radial vignette** approximé en bandes empilées concentriques
        //    autour du centre — donne un focus "cinéma" doux.
        let vignette_steps = 12;
        for i in 0..vignette_steps {
            let t = i as f32 / vignette_steps as f32;
            // Bandes horizontales du haut (assombrissement progressif).
            let band_h = fb_h * 0.05;
            let y_top = i as f32 * band_h;
            let y_bot = fb_h - (i as f32 + 1.0) * band_h;
            let a = 0.04 * (1.0 - t);
            r.push_rect(0.0, y_top, fb_w, band_h, [0.0, 0.0, 0.0, a * bg_alpha]);
            r.push_rect(0.0, y_bot, fb_w, band_h, [0.0, 0.0, 0.0, a * bg_alpha]);
        }
        // 3) **Diagonal stripes pattern** — un grain visuel subtil qui
        //    casse l'aplat noir. 12 bandes diagonales, alpha très bas.
        let stripe_count = 14;
        let stripe_gap = fb_w / stripe_count as f32;
        for i in 0..stripe_count {
            let x0 = (i as f32) * stripe_gap;
            // Bande verticale fine légèrement orangée — accent map maker.
            r.push_rect(x0, 0.0, 1.0, fb_h, [1.0, 0.55, 0.20, 0.025 * bg_alpha]);
        }
        // 4) **Side accent bars** — 6px de chaque côté orange dégradé,
        //    rappel de l'identité visuelle Q3 sans bouffer le centre.
        let accent_w = 6.0_f32;
        let accent_segments = 16;
        let seg_h = fb_h / accent_segments as f32;
        for i in 0..accent_segments {
            let t = (i as f32 / accent_segments as f32 - 0.5).abs() * 2.0;
            let alpha = (1.0 - t) * 0.85;
            let y = i as f32 * seg_h;
            r.push_rect(0.0, y, accent_w, seg_h + 1.0, [1.0, 0.50, 0.12, alpha * bg_alpha]);
            r.push_rect(fb_w - accent_w, y, accent_w, seg_h + 1.0, [1.0, 0.50, 0.12, alpha * bg_alpha]);
        }
        // 5) **Top / bottom accent lines** — fines lignes horizontales
        //    qui ancrent le titre et le footer.
        r.push_rect(accent_w, fb_h * 0.075, fb_w - accent_w * 2.0, 2.0,
                    [1.0, 0.55, 0.15, 0.75]);
        r.push_rect(accent_w, fb_h * 0.075 + 5.0, fb_w - accent_w * 2.0, 1.0,
                    [1.0, 0.85, 0.30, 0.30]);
        r.push_rect(accent_w, fb_h - 76.0, fb_w - accent_w * 2.0, 2.0,
                    [1.0, 0.55, 0.15, 0.75]);
        r.push_rect(accent_w, fb_h - 81.0, fb_w - accent_w * 2.0, 1.0,
                    [1.0, 0.85, 0.30, 0.30]);
        // 6) **Coins cyan-blanc 4 corners** — petits L marqueurs façon
        //    UI gaming moderne (Apex / Cyberpunk).
        let corner_len = 24.0_f32;
        let corner_thick = 2.0_f32;
        let cornsr_col = [0.50, 0.85, 1.00, 0.85];
        // Top-left
        r.push_rect(accent_w + 4.0, accent_w + 4.0, corner_len, corner_thick, cornsr_col);
        r.push_rect(accent_w + 4.0, accent_w + 4.0, corner_thick, corner_len, cornsr_col);
        // Top-right
        r.push_rect(fb_w - accent_w - 4.0 - corner_len, accent_w + 4.0, corner_len, corner_thick, cornsr_col);
        r.push_rect(fb_w - accent_w - 4.0 - corner_thick, accent_w + 4.0, corner_thick, corner_len, cornsr_col);
        // Bottom-left
        r.push_rect(accent_w + 4.0, fb_h - accent_w - 4.0 - corner_thick, corner_len, corner_thick, cornsr_col);
        r.push_rect(accent_w + 4.0, fb_h - accent_w - 4.0 - corner_len, corner_thick, corner_len, cornsr_col);
        // Bottom-right
        r.push_rect(fb_w - accent_w - 4.0 - corner_len, fb_h - accent_w - 4.0 - corner_thick, corner_len, corner_thick, cornsr_col);
        r.push_rect(fb_w - accent_w - 4.0 - corner_thick, fb_h - accent_w - 4.0 - corner_len, corner_thick, corner_len, cornsr_col);
        let _ = vignette_steps;

        // ─────────── Titre / logo ───────────
        match self.page {
            MenuPage::Root => {
                // Logo centré dans le tiers supérieur.
                let cx = fb_w * 0.5;
                let cy = fb_h * 0.14;
                let target_name_frac = 0.40;
                let text_scale = (fb_w * target_name_frac) / (8.0 * 8.0 * "Q3 RUST".len() as f32);
                let scale = (text_scale / 8.0).clamp(2.0, 10.0);
                crate::logo::draw(r, cx, cy, scale);
                // Sous-titre tagline sous le logo.
                let tag = "ARENA · RUST EDITION";
                let tag_scale = 2.0;
                let tag_px = tag_scale * 8.0 * tag.len() as f32;
                let tag_x = (fb_w - tag_px) * 0.5;
                let tag_y = cy + scale * 8.0 + 28.0;
                r.push_text(tag_x, tag_y, tag_scale, [0.55, 0.62, 0.75, 0.85], tag);
            }
            _ => {
                let title = match self.page {
                    MenuPage::Play => "SELECT MAP",
                    MenuPage::Options => "OPTIONS",
                    MenuPage::Gameplay => "GAMEPLAY",
                    MenuPage::Video => "VIDEO",
                    MenuPage::Audio => "AUDIO",
                    MenuPage::Music => "MUSIC PLAYER",
                    MenuPage::MapDownloader => "MAP DOWNLOADER",
                    MenuPage::Root => unreachable!(),
                };
                let title_px = Self::TITLE_SCALE * 8.0 * title.len() as f32;
                let title_x = (fb_w - title_px) * 0.5;
                let title_y = fb_h * 0.13;
                // **Drop shadow** profond pour la profondeur (offset 4px)
                r.push_text(
                    title_x + 4.0,
                    title_y + 4.0,
                    Self::TITLE_SCALE,
                    [0.0, 0.0, 0.0, 0.85],
                    title,
                );
                // **Chromatic ghost** — RGB split moderne (CRT / sci-fi).
                // Cyan offset gauche-haut + rouge offset droite-bas → la
                // couleur principale orange est lisible au centre, les
                // ghosts donnent un edge flashy "neon".
                r.push_text(
                    title_x - 2.0,
                    title_y - 1.0,
                    Self::TITLE_SCALE,
                    [0.30, 0.85, 1.0, 0.55],
                    title,
                );
                r.push_text(
                    title_x + 2.0,
                    title_y + 1.0,
                    Self::TITLE_SCALE,
                    [1.0, 0.20, 0.30, 0.55],
                    title,
                );
                // **Texte principal** orange par-dessus
                r.push_text(
                    title_x,
                    title_y,
                    Self::TITLE_SCALE,
                    crate::logo::palette::ORANGE,
                    title,
                );
                // **Underline composite** — barre épaisse + halo + 2 dots latéraux
                let ul_w = title_px * 0.65;
                let ul_x = (fb_w - ul_w) * 0.5;
                let ul_y = title_y + Self::TITLE_SCALE * 8.0 + 14.0;
                // Halo (alpha bas, hauteur double)
                r.push_rect(ul_x - 4.0, ul_y - 1.0, ul_w + 8.0, 6.0, [1.0, 0.40, 0.10, 0.25]);
                // Barre principale
                r.push_rect(ul_x, ul_y, ul_w, 3.0, [1.0, 0.55, 0.15, 0.95]);
                // Highlight interne plus clair
                r.push_rect(ul_x, ul_y, ul_w, 1.0, [1.0, 0.85, 0.40, 0.85]);
                // Dots latéraux (terminaisons décoratives)
                r.push_rect(ul_x - 12.0, ul_y, 4.0, 3.0, [1.0, 0.70, 0.20, 0.85]);
                r.push_rect(ul_x + ul_w + 8.0, ul_y, 4.0, 3.0, [1.0, 0.70, 0.20, 0.85]);
            }
        }

        // ─────────── Panel central derrière les items ───────────
        // Card glassy (anthracite + bordure orange fine en haut/bas)
        // qui sert de fond cohérent aux options.
        let panel_w = (fb_w * 0.55).max(360.0);
        let panel_x = (fb_w - panel_w) * 0.5;
        // Calé verticalement entre le titre et le footer.
        let panel_y = fb_h * 0.28;
        let panel_h = fb_h - panel_y - 90.0;
        // 1) ombre portée (rect noir 50% offset 6px)
        r.push_rect(
            panel_x + 6.0,
            panel_y + 6.0,
            panel_w,
            panel_h,
            [0.0, 0.0, 0.0, 0.45],
        );
        // 2) corps anthracite semi-opaque (effet « glass »)
        r.push_rect(panel_x, panel_y, panel_w, panel_h, [0.08, 0.10, 0.14, 0.78]);
        // 3) liseré orange en haut + bas pour le « cadre »
        r.push_rect(panel_x, panel_y, panel_w, 2.0, [1.0, 0.50, 0.12, 0.85]);
        r.push_rect(
            panel_x,
            panel_y + panel_h - 2.0,
            panel_w,
            2.0,
            [1.0, 0.50, 0.12, 0.85],
        );
        // 4) liseré clair latéral discret (profondeur)
        r.push_rect(panel_x, panel_y, 1.0, panel_h, [1.0, 1.0, 1.0, 0.08]);
        r.push_rect(
            panel_x + panel_w - 1.0,
            panel_y,
            1.0,
            panel_h,
            [1.0, 1.0, 1.0, 0.04],
        );

        // ─────────── Items ───────────
        let rects = self.layout(fb_w, fb_h);
        let labels = self.labels(cvars);
        for (i, (rect, label)) in rects.iter().zip(labels.iter()).enumerate() {
            if rect.w <= 0.0 {
                continue; // item hors-écran (scroll)
            }
            let is_selected = i == self.selected;
            let is_hover = rect.contains(self.mouse.0, self.mouse.1);

            // Card de l'item — 3 layers pour un look « lifted » :
            //   * ombre portée (offset 2px) pour les non-sélectionnés
            //   * fond gradient haut-bas
            //   * bordure 1px claire en haut, sombre en bas
            // Sélectionné : layer orange saturé qui prend toute la card.
            if is_selected {
                // **SÉLECTIONNÉ — look "card lifted" moderne** :
                // 1. Glow externe large + diffus
                r.push_rect(rect.x - 6.0, rect.y - 4.0, rect.w + 12.0, rect.h + 8.0,
                            [1.0, 0.45, 0.10, 0.10]);
                r.push_rect(rect.x - 3.0, rect.y - 2.0, rect.w + 6.0, rect.h + 4.0,
                            [1.0, 0.55, 0.15, 0.20]);
                // 2. Corps gradient horizontal : plus chaud à gauche → fade à droite
                let gradient_steps = 8;
                for k in 0..gradient_steps {
                    let t = k as f32 / gradient_steps as f32;
                    let x0 = rect.x + rect.w * t;
                    let w_step = rect.w / gradient_steps as f32 + 1.0;
                    let alpha = 0.45 - t * 0.30; // 0.45 → 0.15
                    r.push_rect(x0, rect.y, w_step, rect.h, [1.0, 0.40, 0.10, alpha]);
                }
                // 3. Barre latérale gauche épaisse + highlight interne
                r.push_rect(rect.x, rect.y, 8.0, rect.h, [1.0, 0.65, 0.20, 1.0]);
                r.push_rect(rect.x, rect.y, 8.0, 2.0, [1.0, 1.0, 0.70, 0.95]); // top hl
                r.push_rect(rect.x + 6.0, rect.y, 2.0, rect.h, [1.0, 0.85, 0.40, 0.95]); // edge hl
                // 4. Bordures haut + bas pour le bevel
                r.push_rect(rect.x, rect.y, rect.w, 1.0, [1.0, 0.90, 0.55, 0.85]);
                r.push_rect(rect.x, rect.y + rect.h - 1.0, rect.w, 1.0, [0.40, 0.15, 0.03, 0.85]);
                // 5. Bordure droite cyan accent (contraste froid)
                r.push_rect(rect.x + rect.w - 2.0, rect.y, 2.0, rect.h, [0.30, 0.80, 1.0, 0.55]);
            } else if is_hover {
                // **HOVER — preview subtil** : panel gris clair + bordure cyan
                r.push_rect(rect.x, rect.y, rect.w, rect.h, [0.20, 0.30, 0.45, 0.35]);
                r.push_rect(rect.x, rect.y, 4.0, rect.h, [0.40, 0.85, 1.0, 0.85]);
                r.push_rect(rect.x, rect.y, rect.w, 1.0, [0.50, 0.85, 1.0, 0.40]);
            } else {
                // **IDLE — minimaliste** : juste un voile + 1 barre fine bleu-gris
                r.push_rect(rect.x, rect.y, rect.w, rect.h, [0.10, 0.13, 0.18, 0.45]);
                r.push_rect(rect.x, rect.y, 2.0, rect.h, [0.50, 0.60, 0.75, 0.45]);
            }

            // Chevron à gauche pour l'item sélectionné (point-virgule
            // visuel : on remplace le `▶` Unicode — non rendu par le
            // bitmap font 8×8 — par une flèche tracée à la main avec
            // 3 push_rect qui forment un triangle à droite).
            if is_selected {
                let cx = rect.x + 16.0;
                let cy = rect.y + rect.h * 0.5;
                let s = 4.0;
                // 3 lignes horizontales décalées pour un triangle pointant >
                r.push_rect(cx, cy - s, 2.0, 2.0, [1.0, 1.0, 1.0, 0.95]);
                r.push_rect(cx + 2.0, cy - 2.0, 2.0, 4.0, [1.0, 1.0, 1.0, 0.95]);
                r.push_rect(cx + 4.0, cy - 1.0, 2.0, 2.0, [1.0, 1.0, 1.0, 0.95]);
                let _ = s;
            }

            // Label — ombre portée + texte coloré
            let color = if is_selected {
                [1.0, 0.95, 0.65, 1.0]
            } else if is_hover {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                [0.78, 0.82, 0.88, 1.0]
            };
            let tx = rect.x + if is_selected { 30.0 } else { 22.0 };
            let ty = rect.y + (rect.h - Self::ITEM_SCALE * 8.0) * 0.5;
            r.push_text(tx + 2.0, ty + 2.0, Self::ITEM_SCALE, [0.0, 0.0, 0.0, 0.55], label);
            r.push_text(tx, ty, Self::ITEM_SCALE, color, label);
        }

        // ─────────── Footer hint ───────────
        let hint = match self.page {
            MenuPage::Root => "UP/DOWN  SELECT      ENTER  CONFIRM      ESC  BACK/QUIT",
            MenuPage::Play => {
                "UP/DOWN  SELECT      ENTER  LOAD      WHEEL  SCROLL      ESC  BACK"
            }
            MenuPage::Options => "UP/DOWN  SELECT      ENTER  OPEN      ESC  BACK",
            MenuPage::Gameplay | MenuPage::Audio => {
                "LEFT/RIGHT  ADJUST      ENTER  TOGGLE      ESC  BACK"
            }
            MenuPage::Music => {
                "UP/DOWN  SELECT TRACK      ENTER  PLAY      ESC  BACK"
            }
            MenuPage::Video => {
                "LEFT/RIGHT  CHANGE      ENTER  APPLY/TOGGLE      ESC  BACK"
            }
            MenuPage::MapDownloader => {
                "UP/DOWN  SELECT MAP      ENTER  DOWNLOAD      ESC  BACK"
            }
        };
        let hint_scale = 1.5;
        let hint_px = hint_scale * 8.0 * hint.len() as f32;
        let hint_x = (fb_w - hint_px) * 0.5;
        let hint_y = fb_h - 50.0;
        // Pilule de fond
        let pill_pad = 14.0;
        let pill_x = hint_x - pill_pad;
        let pill_y = hint_y - 6.0;
        let pill_w = hint_px + pill_pad * 2.0;
        let pill_h = hint_scale * 8.0 + 12.0;
        r.push_rect(pill_x, pill_y, pill_w, pill_h, [0.05, 0.06, 0.10, 0.65]);
        r.push_rect(pill_x, pill_y, pill_w, 1.0, [1.0, 0.50, 0.12, 0.4]);
        r.push_text(hint_x, hint_y, hint_scale, [0.7, 0.78, 0.88, 0.95], hint);

        // Version stamp en bas-droite (discret, mais utile pour debug)
        let version = "Q3 RUST  v0.9";
        let v_scale = 1.0;
        let v_px = v_scale * 8.0 * version.len() as f32;
        let v_x = fb_w - v_px - 16.0;
        let v_y = fb_h - 18.0;
        r.push_text(v_x, v_y, v_scale, [0.4, 0.45, 0.55, 0.7], version);
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
                vec![
                    "< BACK".into(),
                    "GAMEPLAY        >".into(),
                    "VIDEO           >".into(),
                    "AUDIO           >".into(),
                    "MAP DOWNLOADER  >".into(),
                ]
            }
            MenuPage::Gameplay => {
                let sens = cvars.get_f32("sensitivity").unwrap_or(5.0);
                let fov = cvars.get_f32("cg_fov").unwrap_or(90.0);
                let inv = cvars.get_f32("m_pitch").unwrap_or(0.022) < 0.0;
                let smooth = cvars.get_f32("m_smoothing").unwrap_or(0.0);
                vec![
                    "< BACK".into(),
                    format!("SENSITIVITY      < {:>5.2} >", sens),
                    format!("FIELD OF VIEW    < {:>5.0} >", fov),
                    format!("INVERT PITCH       {}", if inv { "ON" } else { "OFF" }),
                    format!("MOUSE SMOOTHING  < {:>4.2} >", smooth),
                ]
            }
            MenuPage::Video => {
                let res = RESOLUTIONS[self.resolution_idx];
                let bloom = cvars.get_i32("r_bloom").unwrap_or(1) != 0;
                vec![
                    "< BACK".into(),
                    format!("RESOLUTION       < {} >", res.2),
                    format!("FULLSCREEN         {}", if self.fullscreen { "ON" } else { "OFF" }),
                    format!("VSYNC              {}", if self.vsync { "ON" } else { "OFF" }),
                    format!("BLOOM              {}", if bloom { "ON" } else { "OFF" }),
                    "[ APPLY RESOLUTION ]".into(),
                ]
            }
            MenuPage::Audio => {
                let master = cvars.get_f32("s_volume").unwrap_or(0.8);
                let sfx = cvars.get_f32("s_sfxvolume").unwrap_or(1.0);
                let music = cvars.get_f32("s_musicvolume").unwrap_or(0.25);
                vec![
                    "< BACK".into(),
                    format!("MASTER VOLUME    < {:>5.2} >", master),
                    format!("SFX VOLUME       < {:>5.2} >", sfx),
                    format!("MUSIC VOLUME     < {:>5.2} >", music),
                    "MUSIC PLAYER       >".into(),
                ]
            }
            MenuPage::Music => {
                let mut v: Vec<String> = Vec::with_capacity(2 + self.music_tracks.len().min(12));
                v.push("< BACK".into());
                let stop_label = if self.music_now_playing.is_some() {
                    "[ STOP MUSIC ]"
                } else {
                    "[ STOP MUSIC ]   (silence)"
                };
                v.push(stop_label.into());
                let cap = self.music_tracks.len().min(12);
                let scroll_end = (self.music_scroll + cap).min(self.music_tracks.len());
                for i in self.music_scroll..scroll_end {
                    let p = &self.music_tracks[i];
                    let short = p
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("?");
                    let truncated = if short.len() > 40 {
                        format!("{}...", &short[..37])
                    } else {
                        short.to_string()
                    };
                    let now_playing = self
                        .music_now_playing
                        .as_ref()
                        .map(|np| np == p)
                        .unwrap_or(false);
                    if now_playing {
                        v.push(format!("> {} <", truncated));
                    } else {
                        v.push(format!("  {}", truncated));
                    }
                }
                if self.music_tracks.is_empty() {
                    v.push("(no music files found)".into());
                    v.push("place .ogg/.wav in ~/Music or assets/music".into());
                }
                v
            }
            MenuPage::MapDownloader => {
                let mut v: Vec<String> = Vec::with_capacity(2 + self.mapdl_catalog.len().min(12));
                v.push("< BACK".into());
                let cap = self.mapdl_catalog.len().min(12);
                for i in 0..cap {
                    let (_, label) = &self.mapdl_catalog[i];
                    let truncated = if label.len() > 50 {
                        format!("{}...", &label[..47])
                    } else {
                        label.clone()
                    };
                    v.push(format!("  {}", truncated));
                }
                if self.mapdl_catalog.is_empty() {
                    v.push("(catalog empty — see map_dl::default_catalog)".into());
                }
                if !self.mapdl_status.is_empty() {
                    // Note : pas un item interactif, juste affichage en bas.
                    // Le label sera dessiné séparément ; ici on le remet en
                    // dernier élément pour qu'il apparaisse dans la liste.
                    v.push(format!("  [{}]", self.mapdl_status));
                }
                v
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
    fn gameplay_left_right_adjust_sensitivity() {
        let (mut menu, cv) = fixture(false, &[]);
        // Sub-page Gameplay : index 1 = sensitivity.
        menu.page = MenuPage::Gameplay;
        menu.selected = 1;
        menu.on_key(KeyCode::ArrowRight, &cv);
        assert_eq!(cv.get_f32("sensitivity"), Some(5.5));
        menu.on_key(KeyCode::ArrowLeft, &cv);
        menu.on_key(KeyCode::ArrowLeft, &cv);
        assert_eq!(cv.get_f32("sensitivity"), Some(4.5));
    }

    #[test]
    fn gameplay_sensitivity_clamps() {
        let (mut menu, cv) = fixture(false, &[]);
        menu.page = MenuPage::Gameplay;
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
    fn gameplay_invert_pitch_toggles_sign() {
        let (mut menu, cv) = fixture(false, &[]);
        // Sub-page Gameplay : index 3 = INVERT PITCH (BACK=0, sens=1, FOV=2, invert=3)
        menu.page = MenuPage::Gameplay;
        menu.selected = 3;
        menu.on_key(KeyCode::Enter, &cv);
        assert!(cv.get_f32("m_pitch").unwrap() < 0.0);
        menu.on_key(KeyCode::Enter, &cv);
        assert!(cv.get_f32("m_pitch").unwrap() > 0.0);
    }

    #[test]
    fn resolutions_include_21_9_and_32_9() {
        // Vérifie qu'on couvre bien les ratios ultra-wide.
        let has_21_9 = RESOLUTIONS
            .iter()
            .any(|&(w, h, _)| ((w as f32 / h as f32) - 21.0 / 9.0).abs() < 0.05);
        let has_32_9 = RESOLUTIONS
            .iter()
            .any(|&(w, h, _)| ((w as f32 / h as f32) - 32.0 / 9.0).abs() < 0.05);
        assert!(has_21_9, "RESOLUTIONS doit contenir au moins un 21:9");
        assert!(has_32_9, "RESOLUTIONS doit contenir au moins un 32:9");
    }

    #[test]
    fn ultrawide_specific_modes_present() {
        // Couverture explicite des résolutions UW les plus communes.
        let has = |w: u32, h: u32| RESOLUTIONS.iter().any(|&(rw, rh, _)| rw == w && rh == h);
        assert!(has(3440, 1440), "manque le 3440×1440 (UWQHD 21:9)");
        assert!(has(2560, 1080), "manque le 2560×1080 (UW 21:9)");
        assert!(has(5120, 1440), "manque le 5120×1440 (DQHD 32:9)");
    }

    #[test]
    fn set_window_size_matches_uw_preset() {
        let (mut menu, _) = fixture(false, &[]);
        // Si la fenêtre boot en 3440×1440, on doit retomber sur l'UWQHD.
        menu.set_window_size(3440, 1440);
        let (w, h, _) = RESOLUTIONS[menu.resolution_idx];
        assert_eq!((w, h), (3440, 1440));
        // Idem sur 5120×1440 (32:9).
        menu.set_window_size(5120, 1440);
        let (w, h, _) = RESOLUTIONS[menu.resolution_idx];
        assert_eq!((w, h), (5120, 1440));
    }

    #[test]
    fn video_resolution_cycles() {
        let (mut menu, cv) = fixture(false, &[]);
        menu.page = MenuPage::Video;
        menu.selected = 1;
        let initial = menu.resolution_idx;
        menu.on_key(KeyCode::ArrowRight, &cv);
        assert_eq!(menu.resolution_idx, (initial + 1) % RESOLUTIONS.len());
        menu.on_key(KeyCode::ArrowLeft, &cv);
        menu.on_key(KeyCode::ArrowLeft, &cv);
        assert_eq!(
            menu.resolution_idx,
            (initial + RESOLUTIONS.len() - 1) % RESOLUTIONS.len()
        );
    }

    #[test]
    fn video_apply_emits_action() {
        let (mut menu, cv) = fixture(false, &[]);
        menu.page = MenuPage::Video;
        menu.resolution_idx = 2; // 1920×1080
        // index 5 = APPLY (BACK=0, res=1, fs=2, vsync=3, bloom=4, apply=5)
        menu.selected = 5;
        let act = menu.on_key(KeyCode::Enter, &cv);
        assert_eq!(act, MenuAction::ApplyResolution { width: 1920, height: 1080 });
    }

    #[test]
    fn video_fullscreen_toggle_emits() {
        let (mut menu, cv) = fixture(false, &[]);
        menu.page = MenuPage::Video;
        menu.selected = 2;
        let act = menu.on_key(KeyCode::Enter, &cv);
        assert_eq!(act, MenuAction::ToggleFullscreen);
    }

    #[test]
    fn audio_master_volume_adjusts() {
        let (mut menu, cv) = fixture(false, &[]);
        menu.page = MenuPage::Audio;
        menu.selected = 1;
        menu.on_key(KeyCode::ArrowRight, &cv);
        let v = cv.get_f32("s_volume").unwrap();
        assert!((v - 0.85).abs() < 1e-3);
    }

    #[test]
    fn options_navigates_to_subpages() {
        let (mut menu, cv) = fixture(false, &[]);
        menu.page = MenuPage::Options;
        menu.selected = 1; // GAMEPLAY
        menu.on_key(KeyCode::Enter, &cv);
        assert_eq!(menu.page, MenuPage::Gameplay);
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
