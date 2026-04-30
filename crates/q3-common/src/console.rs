//! Console moteur — pont entre [`CvarRegistry`] et [`CmdRegistry`].
//!
//! Rôle : accepter une ligne texte, la tokeniser et :
//!
//! * si le premier token est une **commande** enregistrée → l'exécuter ;
//! * sinon, si le premier token est une **cvar** existante :
//!   * sans argument → afficher sa valeur courante ;
//!   * avec arguments → `set cvar value` ;
//! * sinon → message "unknown command".
//!
//! La console garde aussi un buffer de lignes affichées (utilisé plus tard
//! par l'overlay texte) et un historique des commandes tapées
//! (flèche haut/bas dans le vrai Q3).
//!
//! # Améliorations vs C original
//!
//! * Pas de globals : on passe `Console` par ref.
//! * Builtins réutilisables : `register_builtins(engine_hooks)` fournit les
//!   commandes `quit`, `echo`, `set`, `seta`, `reset`, `cvarlist`, `cmdlist`.

use crate::cmd::{tokenize, Args, CmdRegistry};
use crate::cvar::{CvarFlags, CvarRegistry};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use tracing::{info, warn};

/// Crochets fournis par l'hôte pour les commandes système (ex : `quit`).
///
/// Clonable, partagé avec les closures des commandes. Les hooks sont des
/// `Arc<Mutex<Option<Box<FnMut>>>>` pour permettre une prise et une pose
/// dynamique par l'hôte.
#[derive(Clone, Default)]
pub struct EngineHooks {
    quit: Arc<Mutex<Option<Box<dyn FnMut() + Send + 'static>>>>,
    map: Arc<Mutex<Option<Box<dyn FnMut(&str) + Send + 'static>>>>,
}

impl EngineHooks {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_quit(&self, f: impl FnMut() + Send + 'static) {
        *self.quit.lock() = Some(Box::new(f));
    }

    pub fn set_map(&self, f: impl FnMut(&str) + Send + 'static) {
        *self.map.lock() = Some(Box::new(f));
    }

    fn trigger_quit(&self) {
        if let Some(ref mut f) = *self.quit.lock() {
            f();
        } else {
            warn!("console: 'quit' non connecté au moteur");
        }
    }

    fn trigger_map(&self, name: &str) {
        if let Some(ref mut f) = *self.map.lock() {
            f(name);
        } else {
            warn!("console: 'map' non connecté au moteur");
        }
    }
}

/// Buffer de sortie + état de saisie + registres.
pub struct Console {
    cvars: CvarRegistry,
    cmds: CmdRegistry,
    lines: VecDeque<String>,
    max_lines: usize,
    input: String,
    history: VecDeque<String>,
    history_index: Option<usize>,
    max_history: usize,
    open: bool,
}

impl Console {
    pub fn new(cvars: CvarRegistry, cmds: CmdRegistry) -> Self {
        Self {
            cvars,
            cmds,
            lines: VecDeque::with_capacity(512),
            max_lines: 512,
            input: String::new(),
            history: VecDeque::with_capacity(64),
            history_index: None,
            max_history: 64,
            open: false,
        }
    }

    pub fn cvars(&self) -> &CvarRegistry {
        &self.cvars
    }

    pub fn cmds(&self) -> &CmdRegistry {
        &self.cmds
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    pub fn toggle(&mut self) {
        self.open = !self.open;
    }

    pub fn set_open(&mut self, open: bool) {
        self.open = open;
    }

    pub fn print(&mut self, line: impl Into<String>) {
        let line = line.into();
        info!("{}", line);
        if self.lines.len() >= self.max_lines {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }

    pub fn lines(&self) -> impl Iterator<Item = &str> {
        self.lines.iter().map(String::as_str)
    }

    pub fn input(&self) -> &str {
        &self.input
    }

    pub fn input_mut(&mut self) -> &mut String {
        &mut self.input
    }

    pub fn push_char(&mut self, c: char) {
        if !c.is_control() {
            self.input.push(c);
        }
    }

    pub fn backspace(&mut self) {
        self.input.pop();
    }

    /// Tab-completion façon Q3 : complète le **premier token** de la ligne
    /// courante (le nom de commande / cvar) en cherchant les préfixes qui
    /// matchent parmi les noms enregistrés. Cas :
    ///
    /// * **0 match** : rien (l'UI peut émettre un beep séparément si elle
    ///   veut signaler l'absence).
    /// * **1 match** : remplace par le nom + ` ` (prêt à taper les args).
    /// * **N matches** : remplace par le plus long préfixe commun (pour
    ///   avancer sans deviner), puis imprime la liste triée dans le buffer
    ///   console, max 24 noms par appel (au-delà, `...` résume la suite).
    ///
    /// Si le dernier caractère de `input` est un espace (le joueur a déjà
    /// validé son token initial), ou s'il y a plusieurs tokens, on ne
    /// complète pas : Q3 ne propose pas de complétion sur les valeurs de
    /// cvar ou sur les arguments libres (map names sont gérés par le menu).
    pub fn tab_complete(&mut self) {
        // Pas d'input → rien à compléter.
        if self.input.is_empty() {
            return;
        }
        // Complétion uniquement sur le premier token : si l'utilisateur a
        // déjà un espace, on est dans les arguments.
        if self.input.chars().any(char::is_whitespace) {
            return;
        }
        let prefix = self.input.to_ascii_lowercase();

        // Union cvars + cmds, dédupliquée, triée, filtrée par préfixe.
        let mut candidates: Vec<String> = Vec::new();
        for n in self.cvars.names() {
            if n.to_ascii_lowercase().starts_with(&prefix) {
                candidates.push(n);
            }
        }
        for n in self.cmds.names() {
            if n.to_ascii_lowercase().starts_with(&prefix) {
                candidates.push(n);
            }
        }
        candidates.sort();
        candidates.dedup();

        match candidates.len() {
            0 => {}
            1 => {
                // Remplace + espace pour enchaîner sur les args.
                self.input = format!("{} ", candidates[0]);
            }
            _ => {
                // Plus long préfixe commun — on avance ce qu'on peut sans
                // choisir pour l'utilisateur.
                let lcp = longest_common_prefix(&candidates);
                if lcp.len() > self.input.len() {
                    self.input = lcp;
                }
                // Affichage en liste pour guider le joueur.
                self.print(format!("{} matches:", candidates.len()));
                const LIST_LIMIT: usize = 24;
                for n in candidates.iter().take(LIST_LIMIT) {
                    self.print(format!("  {n}"));
                }
                if candidates.len() > LIST_LIMIT {
                    self.print(format!("  ... ({} more)", candidates.len() - LIST_LIMIT));
                }
            }
        }
    }

    pub fn submit(&mut self) {
        let line = std::mem::take(&mut self.input);
        if line.trim().is_empty() {
            return;
        }
        // historique (pas de doublons consécutifs)
        if self.history.back().map(String::as_str) != Some(&line) {
            if self.history.len() >= self.max_history {
                self.history.pop_front();
            }
            self.history.push_back(line.clone());
        }
        self.history_index = None;
        self.print(format!("] {line}"));
        self.execute(&line);
    }

    /// Exécute une ligne sans passer par l'input (utile pour les exec).
    pub fn execute(&mut self, line: &str) {
        let tokens = tokenize(line);
        if tokens.is_empty() {
            return;
        }
        let name = tokens[0].to_ascii_lowercase();

        if self.cmds.exists(&name) {
            let tv: Vec<String> = tokens.iter().cloned().collect();
            let args = Args::new(&tv);
            // `exists` puis dispatch — ici on re-dispatche via execute
            // qui re-tokenise. Plus simple : on appelle directement.
            if !self.cmds.execute(line) {
                warn!("console: cmd `{}` disparue en cours de dispatch", name);
            }
            let _ = args; // silencer unused
            return;
        }

        if self.cvars.get(&name).is_some() {
            if tokens.len() == 1 {
                let v = self.cvars.get_string(&name).unwrap_or_default();
                self.print(format!("\"{name}\" is \"{v}\""));
            } else {
                let value = tokens[1..].join(" ");
                match self.cvars.set(&name, &value) {
                    Ok(()) => self.print(format!("{name} = {value}")),
                    Err(e) => self.print(format!("erreur: {e}")),
                }
            }
            return;
        }

        self.print(format!("unknown command: {name}"));
    }

    /// Récupère la commande précédente dans l'historique (flèche haut).
    pub fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let idx = match self.history_index {
            None => self.history.len() - 1,
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.history_index = Some(idx);
        self.input = self.history[idx].clone();
    }

    pub fn history_next(&mut self) {
        let Some(i) = self.history_index else {
            return;
        };
        if i + 1 >= self.history.len() {
            self.history_index = None;
            self.input.clear();
        } else {
            self.history_index = Some(i + 1);
            self.input = self.history[i + 1].clone();
        }
    }
}

/// Plus long préfixe commun à toutes les chaînes de `names`. Tronque byte-
/// à-byte — les noms de cvars / cmds sont ASCII en pratique (convention Q3),
/// donc pas de souci UTF-8. Renvoie `""` sur entrée vide.
fn longest_common_prefix(names: &[String]) -> String {
    let mut iter = names.iter();
    let Some(first) = iter.next() else {
        return String::new();
    };
    let mut end = first.len();
    for s in iter {
        end = end.min(s.len());
        while end > 0 && !first.as_bytes()[..end].eq_ignore_ascii_case(&s.as_bytes()[..end]) {
            end -= 1;
        }
        if end == 0 {
            break;
        }
    }
    // On rend le préfixe dans la casse du premier match (les registres Q3
    // stockent en minuscules, donc ça suit de toute façon).
    first[..end].to_string()
}

/// Enregistre les commandes builtins dans `cmds`, adossées à `cvars` et
/// `hooks`. Idempotent.
pub fn register_builtins(cmds: &CmdRegistry, cvars: &CvarRegistry, hooks: &EngineHooks) {
    // quit
    {
        let h = hooks.clone();
        cmds.add("quit", move |_args: &Args| {
            h.trigger_quit();
        });
    }
    // map
    {
        let h = hooks.clone();
        cmds.add("map", move |args: &Args| {
            if args.count() < 2 {
                warn!("usage: map <mapname>");
                return;
            }
            h.trigger_map(args.argv(1));
        });
    }
    // echo
    cmds.add("echo", |args: &Args| {
        info!("{}", args.args_from(1));
    });
    // set <name> <value>
    {
        let c = cvars.clone();
        cmds.add("set", move |args: &Args| {
            if args.count() < 3 {
                warn!("usage: set <cvar> <value>");
                return;
            }
            let _ = c.set(args.argv(1), &args.args_from(2));
        });
    }
    // seta <name> <value> (persistée)
    {
        let c = cvars.clone();
        cmds.add("seta", move |args: &Args| {
            if args.count() < 3 {
                warn!("usage: seta <cvar> <value>");
                return;
            }
            // register avec ARCHIVE, puis set
            c.register(args.argv(1), &args.args_from(2), CvarFlags::ARCHIVE);
            let _ = c.set(args.argv(1), &args.args_from(2));
        });
    }
    // reset <name>
    {
        let c = cvars.clone();
        cmds.add("reset", move |args: &Args| {
            if args.count() < 2 {
                warn!("usage: reset <cvar>");
                return;
            }
            let _ = c.reset(args.argv(1));
        });
    }
    // cvarlist
    {
        let c = cvars.clone();
        cmds.add("cvarlist", move |_args: &Args| {
            for name in c.names() {
                if let Some(v) = c.get_string(&name) {
                    info!("{name:<24} \"{v}\"");
                }
            }
        });
    }
    // cmdlist
    {
        let reg = cmds.clone();
        cmds.add("cmdlist", move |_args: &Args| {
            for name in reg.names() {
                info!("{name}");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn setup() -> (Console, EngineHooks) {
        let cvars = CvarRegistry::new();
        let cmds = CmdRegistry::new();
        let hooks = EngineHooks::new();
        register_builtins(&cmds, &cvars, &hooks);
        (Console::new(cvars, cmds), hooks)
    }

    #[test]
    fn unknown_command_is_reported() {
        let (mut con, _) = setup();
        con.execute("flibberty");
        assert!(con.lines().any(|l| l.contains("unknown command")));
    }

    #[test]
    fn set_then_read_cvar() {
        let (mut con, _) = setup();
        con.cvars().register("sensitivity", "3", CvarFlags::ARCHIVE);
        con.execute("sensitivity 7");
        assert_eq!(con.cvars().get_f32("sensitivity"), Some(7.0));
        con.execute("sensitivity");
        assert!(con.lines().any(|l| l.contains("\"sensitivity\" is \"7\"")));
    }

    #[test]
    fn quit_calls_hook() {
        let (mut con, hooks) = setup();
        let count = Arc::new(AtomicUsize::new(0));
        let c2 = count.clone();
        hooks.set_quit(move || {
            c2.fetch_add(1, Ordering::SeqCst);
        });
        con.execute("quit");
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn submit_pushes_prompt_and_executes() {
        let (mut con, hooks) = setup();
        let count = Arc::new(AtomicUsize::new(0));
        let c2 = count.clone();
        hooks.set_map(move |_| {
            c2.fetch_add(1, Ordering::SeqCst);
        });
        con.input_mut().push_str("map q3dm1");
        con.submit();
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(con.lines().any(|l| l.starts_with("] map q3dm1")));
    }

    #[test]
    fn tab_complete_single_match_appends_space() {
        let (mut con, _) = setup();
        con.cvars().register("sensitivity", "5", CvarFlags::ARCHIVE);
        con.input_mut().push_str("sens");
        con.tab_complete();
        // Unique cvar préfixé `sens` → remplacement total + espace.
        assert_eq!(con.input(), "sensitivity ");
    }

    #[test]
    fn tab_complete_multiple_matches_expands_common_prefix() {
        let (mut con, _) = setup();
        con.cvars().register("cg_fov", "90", CvarFlags::ARCHIVE);
        con.cvars().register("cg_crosshair", "2", CvarFlags::ARCHIVE);
        con.cvars().register("cg_draw2d", "1", CvarFlags::ARCHIVE);
        // `cg` ciblera uniquement les trois cvars `cg_*` (les builtins
        // `cmdlist`, `cvarlist` commencent par `c` mais pas `cg`).
        con.input_mut().push_str("cg");
        con.tab_complete();
        // LCP des trois cvars = "cg_" — on doit l'avoir gagné.
        assert_eq!(con.input(), "cg_");
        assert!(con.lines().any(|l| l.contains("3 matches:")));
        // Les trois noms apparaissent dans la liste imprimée.
        assert!(con.lines().any(|l| l.contains("cg_fov")));
        assert!(con.lines().any(|l| l.contains("cg_crosshair")));
        assert!(con.lines().any(|l| l.contains("cg_draw2d")));
    }

    #[test]
    fn tab_complete_no_match_leaves_input_untouched() {
        let (mut con, _) = setup();
        con.input_mut().push_str("zzzzz");
        con.tab_complete();
        assert_eq!(con.input(), "zzzzz");
    }

    #[test]
    fn tab_complete_skips_when_past_first_token() {
        let (mut con, _) = setup();
        con.cvars().register("sensitivity", "5", CvarFlags::ARCHIVE);
        con.input_mut().push_str("set sens");
        con.tab_complete();
        // Second token — on ne touche à rien (pas de complétion d'args).
        assert_eq!(con.input(), "set sens");
    }

    #[test]
    fn tab_complete_handles_cmds_and_cvars_union() {
        let (mut con, _) = setup();
        con.cvars().register("quit_delay", "0", CvarFlags::empty());
        // Le builtin `quit` est une cmd, et on ajoute une cvar `quit_delay`.
        con.input_mut().push_str("qui");
        con.tab_complete();
        // LCP = "quit" (les deux commencent par `quit`).
        assert!(con.input().starts_with("quit"));
        // Au moins un des deux est bien repéré.
        assert!(con.lines().any(|l| l.contains("matches:")));
    }

    #[test]
    fn tab_complete_empty_input_is_noop() {
        let (mut con, _) = setup();
        con.tab_complete();
        assert!(con.input().is_empty());
        // Aucune ligne imprimée (pas de listing spammé sur Tab vide).
        assert_eq!(con.lines().count(), 0);
    }

    #[test]
    fn history_prev_recalls_last_command() {
        let (mut con, _) = setup();
        con.cvars().register("r_mode", "1024", CvarFlags::empty());
        con.input_mut().push_str("r_mode 1920");
        con.submit();
        assert!(con.input().is_empty());
        con.history_prev();
        assert_eq!(con.input(), "r_mode 1920");
    }
}
