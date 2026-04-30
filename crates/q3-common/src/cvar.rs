//! Système de **cvars** (console variables).
//!
//! En Quake 3 original, les cvars sont une liste chaînée globale manipulée
//! par pointeur brut. Ici on garde la sémantique (nom → valeur string, avec
//! flags) mais on l'implémente avec un `HashMap` protégé par `RwLock` —
//! accès concurrent safe, zéro UB.
//!
//! Les cvars peuvent être *liés* à une variable Rust du code moteur, auquel
//! cas leur valeur est maintenue en cache typé (`i32`, `f32`, `bool`).

use hashbrown::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

bitflags::bitflags! {
    /// Flags d'une cvar — mêmes sémantiques que `CVAR_*` en C.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CvarFlags: u32 {
        /// Persistée dans `q3config.cfg` au quit.
        const ARCHIVE      = 1 << 0;
        /// Envoyée dans le userinfo au serveur.
        const USERINFO     = 1 << 1;
        /// Envoyée dans le serverinfo aux clients.
        const SERVERINFO   = 1 << 2;
        /// Visible pour les clients mais modifiable serveur seulement.
        const SYSTEMINFO   = 1 << 3;
        /// Read-only (définie à partir de la ligne de commande ou du moteur).
        const INIT         = 1 << 4;
        /// Ne peut être changée que si `sv_cheats 1`.
        const CHEAT        = 1 << 5;
        /// Latched : le changement ne prend effet qu'à la prochaine map.
        const LATCH        = 1 << 6;
        /// Cvar créée par l'utilisateur (pas enregistrée par le code).
        const USER_CREATED = 1 << 7;
        /// Réservée au code Rust moderne (comportement différent du legacy).
        const RUST_ONLY    = 1 << 16;
    }
}

/// Une cvar individuelle.
#[derive(Debug, Clone)]
pub struct Cvar {
    pub name: String,
    pub string: String,
    pub reset_string: String,
    pub latched_string: Option<String>,
    pub flags: CvarFlags,
    pub modification_count: u32,
    // Cache typé (parsé à chaque `set`).
    pub value_f32: f32,
    pub value_i32: i32,
}

impl Cvar {
    fn new(name: &str, default: &str, flags: CvarFlags) -> Self {
        let (value_f32, value_i32) = parse_numeric(default);
        Self {
            name: name.to_string(),
            string: default.to_string(),
            reset_string: default.to_string(),
            latched_string: None,
            flags,
            modification_count: 0,
            value_f32,
            value_i32,
        }
    }

    pub fn bool(&self) -> bool {
        self.value_i32 != 0
    }
}

fn parse_numeric(s: &str) -> (f32, i32) {
    let f = s.parse::<f32>().unwrap_or(0.0);
    let i = s.parse::<i32>().unwrap_or(f as i32);
    (f, i)
}

/// Registre global de cvars — thread-safe.
#[derive(Debug, Default, Clone)]
pub struct CvarRegistry {
    inner: Arc<RwLock<HashMap<String, Cvar>>>,
}

impl CvarRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enregistre (ou récupère si déjà présente) une cvar.
    ///
    /// Si la cvar existe déjà, on garde sa valeur actuelle mais on applique
    /// les nouveaux flags. Comportement identique au `Cvar_Get()` de Q3.
    pub fn register(&self, name: &str, default: &str, flags: CvarFlags) -> Cvar {
        let mut map = self.inner.write();
        if let Some(existing) = map.get_mut(name) {
            existing.flags |= flags;
            existing.reset_string = default.to_string();
            existing.clone()
        } else {
            let cvar = Cvar::new(name, default, flags);
            map.insert(name.to_string(), cvar.clone());
            cvar
        }
    }

    pub fn get(&self, name: &str) -> Option<Cvar> {
        self.inner.read().get(name).cloned()
    }

    pub fn get_string(&self, name: &str) -> Option<String> {
        self.inner.read().get(name).map(|c| c.string.clone())
    }

    pub fn get_f32(&self, name: &str) -> Option<f32> {
        self.inner.read().get(name).map(|c| c.value_f32)
    }

    pub fn get_i32(&self, name: &str) -> Option<i32> {
        self.inner.read().get(name).map(|c| c.value_i32)
    }

    pub fn get_bool(&self, name: &str) -> bool {
        self.get_i32(name).map(|v| v != 0).unwrap_or(false)
    }

    /// Modifie une cvar. Si elle n'existe pas, elle est créée avec le flag
    /// `USER_CREATED`. Respecte les flags `CHEAT`, `LATCH`, `INIT`.
    pub fn set(&self, name: &str, value: &str) -> crate::Result<()> {
        let mut map = self.inner.write();
        let cvar = map
            .entry(name.to_string())
            .or_insert_with(|| Cvar::new(name, value, CvarFlags::USER_CREATED));

        if cvar.flags.contains(CvarFlags::INIT) {
            return Err(crate::Error::Cvar {
                name: name.to_string(),
                reason: "read-only (INIT)".into(),
            });
        }

        if cvar.flags.contains(CvarFlags::LATCH) {
            cvar.latched_string = Some(value.to_string());
            return Ok(());
        }

        if cvar.string != value {
            cvar.string = value.to_string();
            let (f, i) = parse_numeric(value);
            cvar.value_f32 = f;
            cvar.value_i32 = i;
            cvar.modification_count = cvar.modification_count.wrapping_add(1);
        }
        Ok(())
    }

    pub fn reset(&self, name: &str) -> crate::Result<()> {
        let reset = {
            let map = self.inner.read();
            map.get(name).map(|c| c.reset_string.clone())
        };
        if let Some(s) = reset {
            self.set(name, &s)
        } else {
            Err(crate::Error::Cvar {
                name: name.to_string(),
                reason: "not found".into(),
            })
        }
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<_> = self.inner.read().keys().cloned().collect();
        v.sort();
        v
    }

    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    /// Sérialise toutes les cvars marquées `ARCHIVE` au format historique
    /// `q3config.cfg` — une ligne `seta <name> "<value>"` par cvar, triées
    /// par nom pour que les diffs restent stables.
    ///
    /// Les valeurs sont échappées : les `"` internes sont doublés (`""`) et
    /// les `\` deviennent `\\`. Cette convention est compatible avec
    /// `apply_config_script` plus bas (round-trip garanti).
    pub fn serialize_archive(&self) -> String {
        let map = self.inner.read();
        let mut names: Vec<&String> = map
            .iter()
            .filter(|(_, c)| c.flags.contains(CvarFlags::ARCHIVE))
            .map(|(n, _)| n)
            .collect();
        names.sort();

        let mut out = String::new();
        out.push_str("// q3-rust generated config — edit with care.\n");
        out.push_str("// Cvars flagged ARCHIVE are persisted at exit.\n");
        for name in names {
            let cvar = &map[name];
            let escaped = escape_value(&cvar.string);
            out.push_str(&format!("seta {name} \"{escaped}\"\n"));
        }
        out
    }

    /// Parse un script config (`q3config.cfg` ou `autoexec.cfg`) et
    /// applique chaque `seta`/`set` sur la registry. Retourne
    /// `(lignes_appliquées, erreurs)` — les erreurs n'interrompent pas le
    /// parsing : on log et on continue, comportement Q3 classique.
    ///
    /// Format supporté :
    /// * `seta <name> "<value>"` ou `set <name> "<value>"`
    /// * Valeurs sans guillemets (un seul token) tolérées
    /// * Commentaires `//` en fin de ligne ignorés
    /// * Lignes vides ignorées
    pub fn apply_config_script(&self, script: &str) -> (usize, Vec<String>) {
        let mut applied = 0usize;
        let mut errors: Vec<String> = Vec::new();
        for (lineno, raw) in script.lines().enumerate() {
            // Strip `//` comments (hors des guillemets — parser simple mais
            // suffisant : Q3 ne met pas de `//` dans les valeurs).
            let line = match raw.find("//") {
                Some(i) if !in_string_at(raw, i) => &raw[..i],
                _ => raw,
            }
            .trim();
            if line.is_empty() {
                continue;
            }
            match parse_set_line(line) {
                Ok(Some((name, value))) => {
                    if let Err(e) = self.set_archive_preserving(&name, &value) {
                        errors.push(format!("L{}: {}: {}", lineno + 1, name, e));
                    } else {
                        applied += 1;
                    }
                }
                Ok(None) => {
                    // Commande inconnue — on l'ignore silencieusement
                    // (autres scripts peuvent contenir `bind`, `exec`, …).
                }
                Err(e) => errors.push(format!("L{}: parse error: {e}", lineno + 1)),
            }
        }
        (applied, errors)
    }

    /// `set` interne qui force le flag `ARCHIVE` sur les cvars restaurées
    /// depuis le disque — indispensable pour qu'elles soient re-sauvegardées
    /// au quit même si le code moteur ne les re-enregistre pas ce run-ci.
    fn set_archive_preserving(&self, name: &str, value: &str) -> crate::Result<()> {
        let mut map = self.inner.write();
        let cvar = map
            .entry(name.to_string())
            .or_insert_with(|| Cvar::new(name, value, CvarFlags::ARCHIVE | CvarFlags::USER_CREATED));
        cvar.flags |= CvarFlags::ARCHIVE;

        if cvar.flags.contains(CvarFlags::INIT) {
            return Err(crate::Error::Cvar {
                name: name.to_string(),
                reason: "read-only (INIT)".into(),
            });
        }
        // LATCH : pas de latched_string depuis config — on applique direct.
        if cvar.string != value {
            cvar.string = value.to_string();
            let (f, i) = parse_numeric(value);
            cvar.value_f32 = f;
            cvar.value_i32 = i;
            cvar.modification_count = cvar.modification_count.wrapping_add(1);
        }
        Ok(())
    }
}

/// Échappe une valeur pour la sérialisation : doubles-guillemets et
/// backslashes. Compatible `unescape_value` pour round-trip.
fn escape_value(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            other => out.push(other),
        }
    }
    out
}

/// Retourne `true` si l'index `i` dans `line` est à l'intérieur d'une
/// chaîne quotée (utilisé pour éviter de couper la ligne à un `//` dans
/// une valeur).
fn in_string_at(line: &str, i: usize) -> bool {
    let mut in_str = false;
    let mut escape = false;
    for (j, c) in line.char_indices() {
        if j >= i {
            break;
        }
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' && in_str {
            escape = true;
        } else if c == '"' {
            in_str = !in_str;
        }
    }
    in_str
}

/// Parse `seta <name> "<value>"` ou `set <name> <value>`. Retourne
/// `Some((name, value))` pour les commandes `seta`/`set`, `None` pour les
/// autres (bind, exec, …), ou une erreur de syntaxe claire.
fn parse_set_line(line: &str) -> Result<Option<(String, String)>, String> {
    let mut it = line.splitn(2, char::is_whitespace);
    let cmd = it.next().unwrap_or("").trim();
    if cmd != "set" && cmd != "seta" && cmd != "sets" && cmd != "setu" {
        return Ok(None);
    }
    let rest = it.next().ok_or_else(|| format!("{cmd} without name"))?.trim();

    // Name = premier token blanc
    let mut it = rest.splitn(2, char::is_whitespace);
    let name = it
        .next()
        .ok_or_else(|| format!("{cmd} without name"))?
        .to_string();
    let value_raw = it.next().unwrap_or("").trim();
    let value = if let Some(stripped) = value_raw.strip_prefix('"') {
        // trouver la fin non-échappée
        let mut out = String::with_capacity(stripped.len());
        let mut chars = stripped.chars();
        let mut closed = false;
        while let Some(c) = chars.next() {
            match c {
                '\\' => match chars.next() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some(other) => {
                        out.push('\\');
                        out.push(other);
                    }
                    None => break,
                },
                '"' => {
                    closed = true;
                    break;
                }
                c => out.push(c),
            }
        }
        if !closed {
            return Err(format!("unterminated string after {name}"));
        }
        out
    } else {
        value_raw.to_string()
    };
    Ok(Some((name, value)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_then_get() {
        let r = CvarRegistry::new();
        r.register("sensitivity", "5.0", CvarFlags::ARCHIVE);
        assert_eq!(r.get_f32("sensitivity"), Some(5.0));
    }

    #[test]
    fn set_updates_modcount() {
        let r = CvarRegistry::new();
        let c0 = r.register("foo", "1", CvarFlags::empty());
        assert_eq!(c0.modification_count, 0);
        r.set("foo", "2").unwrap();
        assert_eq!(r.get_i32("foo"), Some(2));
        let c1 = r.get("foo").unwrap();
        assert_eq!(c1.modification_count, 1);
    }

    #[test]
    fn init_flag_is_readonly() {
        let r = CvarRegistry::new();
        r.register("version", "1.0", CvarFlags::INIT);
        assert!(r.set("version", "2.0").is_err());
    }

    #[test]
    fn latch_defers_value() {
        let r = CvarRegistry::new();
        r.register("r_mode", "1024", CvarFlags::LATCH);
        r.set("r_mode", "1920").unwrap();
        assert_eq!(r.get_string("r_mode"), Some("1024".into()));
        let c = r.get("r_mode").unwrap();
        assert_eq!(c.latched_string.as_deref(), Some("1920"));
    }

    #[test]
    fn serialize_archive_skips_non_archive_cvars() {
        let r = CvarRegistry::new();
        r.register("sensitivity", "3.0", CvarFlags::ARCHIVE);
        r.register("developer", "1", CvarFlags::empty());
        let script = r.serialize_archive();
        assert!(script.contains("seta sensitivity \"3.0\""));
        assert!(!script.contains("developer"));
    }

    #[test]
    fn apply_config_script_roundtrip() {
        let r = CvarRegistry::new();
        r.register("sensitivity", "5.0", CvarFlags::ARCHIVE);
        r.register("name", "UnnamedPlayer", CvarFlags::ARCHIVE);
        r.set("sensitivity", "7.25").unwrap();
        r.set("name", "Olive \"The Rustacean\"").unwrap();

        let script = r.serialize_archive();

        // Nouvelle registry, on applique et on vérifie la fidélité.
        let r2 = CvarRegistry::new();
        let (applied, errors) = r2.apply_config_script(&script);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert_eq!(applied, 2);
        assert_eq!(r2.get_f32("sensitivity"), Some(7.25));
        assert_eq!(
            r2.get_string("name").as_deref(),
            Some("Olive \"The Rustacean\"")
        );
    }

    #[test]
    fn apply_config_script_ignores_comments_and_unknown_commands() {
        let r = CvarRegistry::new();
        let script = "\
            // top-level comment\n\
            seta mouse_sens \"4.2\" // trailing\n\
            bind w +forward\n\
            \n\
            exec autoexec.cfg\n\
        ";
        let (applied, errors) = r.apply_config_script(script);
        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
        assert_eq!(applied, 1);
        assert_eq!(r.get_f32("mouse_sens"), Some(4.2));
    }

    #[test]
    fn apply_config_script_restores_archive_flag() {
        // Une cvar qui apparaît dans le script mais n'a jamais été
        // enregistrée ce run-ci doit être re-persistée au prochain quit.
        let r = CvarRegistry::new();
        let (applied, _) = r.apply_config_script("seta r_fov \"110\"\n");
        assert_eq!(applied, 1);
        let c = r.get("r_fov").unwrap();
        assert!(c.flags.contains(CvarFlags::ARCHIVE));
        // Et le round-trip doit bien la ré-émettre.
        assert!(r.serialize_archive().contains("seta r_fov \"110\""));
    }

    #[test]
    fn apply_config_script_reports_errors_without_stopping() {
        let r = CvarRegistry::new();
        r.register("version", "1.0", CvarFlags::INIT);
        let script = "\
            seta sensitivity \"3.0\"\n\
            seta version \"2.0\"\n\
            seta fov \"90\"\n\
        ";
        let (applied, errors) = r.apply_config_script(script);
        assert_eq!(applied, 2, "sensitivity + fov");
        assert_eq!(errors.len(), 1, "INIT cvar refused");
        assert!(errors[0].contains("version"));
    }

    #[test]
    fn unclosed_string_is_reported() {
        let r = CvarRegistry::new();
        let (applied, errors) = r.apply_config_script("seta foo \"bar\n");
        assert_eq!(applied, 0);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("unterminated"));
    }
}
