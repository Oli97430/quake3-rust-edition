//! Système de commandes console.
//!
//! Une commande est identifiée par un nom (ex: `map`, `quit`, `bind`) et
//! exécute une closure quand elle est invoquée. Le tokenizer respecte les
//! guillemets et les commentaires `//`, comme dans `cmd.c` original.

use hashbrown::HashMap;
use parking_lot::RwLock;
use smallvec::SmallVec;
use std::sync::Arc;

/// Arguments d'une commande — déjà tokenisés.
///
/// Inspiré de `Cmd_Argv() / Cmd_Argc()` en C, mais avec un slice borrowed
/// au lieu d'une table globale mutable.
pub struct Args<'a> {
    tokens: &'a [String],
}

impl<'a> Args<'a> {
    pub fn new(tokens: &'a [String]) -> Self {
        Self { tokens }
    }

    pub fn count(&self) -> usize {
        self.tokens.len()
    }

    pub fn argv(&self, i: usize) -> &str {
        self.tokens.get(i).map(String::as_str).unwrap_or("")
    }

    /// Concatène tous les arguments à partir de `start` (inclus) avec espaces.
    pub fn args_from(&self, start: usize) -> String {
        self.tokens
            .get(start..)
            .map(|s| s.join(" "))
            .unwrap_or_default()
    }
}

pub type CmdFn = Arc<dyn Fn(&Args) + Send + Sync>;

/// Registre global de commandes — thread-safe.
#[derive(Clone, Default)]
pub struct CmdRegistry {
    inner: Arc<RwLock<HashMap<String, CmdFn>>>,
}

impl CmdRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add<F>(&self, name: &str, f: F)
    where
        F: Fn(&Args) + Send + Sync + 'static,
    {
        self.inner
            .write()
            .insert(name.to_lowercase(), Arc::new(f));
    }

    pub fn remove(&self, name: &str) {
        self.inner.write().remove(&name.to_lowercase());
    }

    pub fn exists(&self, name: &str) -> bool {
        self.inner.read().contains_key(&name.to_lowercase())
    }

    /// Exécute une ligne de commande (parse + dispatch).
    ///
    /// Retourne `true` si une commande a été trouvée et exécutée, `false`
    /// sinon (à charge de l'appelant de traiter le cas "cvar set" ou
    /// "commande inconnue").
    pub fn execute(&self, line: &str) -> bool {
        let tokens = tokenize(line);
        if tokens.is_empty() {
            return false;
        }
        let name = tokens[0].to_lowercase();
        let handler = self.inner.read().get(&name).cloned();
        if let Some(f) = handler {
            let args = Args::new(&tokens);
            f(&args);
            true
        } else {
            false
        }
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<_> = self.inner.read().keys().cloned().collect();
        v.sort();
        v
    }
}

/// Tokenisation d'une ligne de commande.
///
/// Règles (reproduites de `Cmd_TokenizeString2` en C) :
/// * les espaces séparent les tokens
/// * les `"..."` englobent un token qui peut contenir des espaces
/// * `//` démarre un commentaire jusqu'à la fin de ligne
/// * les sauts de ligne terminent le parsing (une ligne = une commande)
pub fn tokenize(input: &str) -> SmallVec<[String; 8]> {
    let mut tokens = SmallVec::<[String; 8]>::new();
    let mut chars = input.chars().peekable();

    'outer: while let Some(&c) = chars.peek() {
        // skip whitespace
        if c.is_whitespace() {
            if c == '\n' {
                break;
            }
            chars.next();
            continue;
        }

        // comment
        if c == '/' {
            let mut it = chars.clone();
            it.next();
            if it.peek() == Some(&'/') {
                break;
            }
        }

        // quoted token
        if c == '"' {
            chars.next();
            let mut s = String::new();
            for ch in chars.by_ref() {
                if ch == '"' {
                    tokens.push(s);
                    continue 'outer;
                }
                s.push(ch);
            }
            // unterminated quote — push what we have
            tokens.push(s);
            break;
        }

        // plain token
        let mut s = String::new();
        while let Some(&ch) = chars.peek() {
            if ch.is_whitespace() {
                break;
            }
            if ch == '"' {
                break;
            }
            if ch == '/' {
                let mut it = chars.clone();
                it.next();
                if it.peek() == Some(&'/') {
                    break;
                }
            }
            s.push(ch);
            chars.next();
        }
        if !s.is_empty() {
            tokens.push(s);
        }
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_simple() {
        let t = tokenize("map q3dm1");
        assert_eq!(t.as_slice(), &["map".to_string(), "q3dm1".to_string()]);
    }

    #[test]
    fn tokenize_quoted() {
        let t = tokenize(r#"bind a "say hello world""#);
        assert_eq!(
            t.as_slice(),
            &["bind".to_string(), "a".to_string(), "say hello world".to_string()]
        );
    }

    #[test]
    fn tokenize_comment() {
        let t = tokenize("set r_mode 3 // high quality");
        assert_eq!(
            t.as_slice(),
            &["set".to_string(), "r_mode".to_string(), "3".to_string()]
        );
    }

    #[test]
    fn execute_dispatches() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let r = CmdRegistry::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let c2 = counter.clone();
        r.add("test", move |args| {
            c2.fetch_add(args.count(), Ordering::SeqCst);
        });
        assert!(r.execute("test one two three"));
        assert_eq!(counter.load(Ordering::SeqCst), 4);
    }
}
