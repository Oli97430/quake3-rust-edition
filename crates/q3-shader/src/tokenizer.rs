//! Tokenizer pour les scripts shader (et plus généralement pour les formats
//! texte id Tech 3 : `.shader`, `.arena`, `.cfg`, etc.).
//!
//! Règles :
//! * whitespace sépare les tokens
//! * `//` démarre un commentaire jusqu'à EOL
//! * `/*` … `*/` commentaire de bloc
//! * `"..."` englobe un token quoté
//! * `{` et `}` sont des tokens individuels (pas besoin d'espace autour)

pub struct Tokenizer<'a> {
    src: &'a str,
    pos: usize,
    peeked: Option<String>,
}

impl<'a> Tokenizer<'a> {
    pub fn new(src: &'a str) -> Self {
        Self { src, pos: 0, peeked: None }
    }

    pub fn next(&mut self) -> Option<String> {
        if let Some(p) = self.peeked.take() {
            return Some(p);
        }
        self.read_token()
    }

    pub fn peek(&mut self) -> Option<String> {
        if self.peeked.is_none() {
            self.peeked = self.read_token();
        }
        self.peeked.clone()
    }

    /// Consomme jusqu'à trouver un `}` au niveau courant (utile pour skipper
    /// un shader malformé). Retourne une fois `}` consommé.
    pub fn skip_block(&mut self) {
        let mut depth = 1;
        while let Some(tok) = self.next() {
            match tok.as_str() {
                "{" => depth += 1,
                "}" => {
                    depth -= 1;
                    if depth == 0 {
                        return;
                    }
                }
                _ => {}
            }
        }
    }

    /// Consomme le reste de la ligne courante. Utilisé pour skipper une
    /// directive inconnue sans avaler le token `}` qui pourrait la suivre.
    pub fn skip_line(&mut self) {
        let bytes = self.src.as_bytes();
        while self.pos < bytes.len() {
            let b = bytes[self.pos];
            self.pos += 1;
            if b == b'\n' {
                return;
            }
        }
    }

    fn read_token(&mut self) -> Option<String> {
        let bytes = self.src.as_bytes();
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= bytes.len() {
                return None;
            }
            let b = bytes[self.pos];
            if b == b'{' || b == b'}' {
                self.pos += 1;
                return Some((b as char).to_string());
            }
            if b == b'"' {
                self.pos += 1;
                let start = self.pos;
                while self.pos < bytes.len() && bytes[self.pos] != b'"' {
                    self.pos += 1;
                }
                let end = self.pos;
                if self.pos < bytes.len() {
                    self.pos += 1; // skip closing "
                }
                return Some(
                    std::str::from_utf8(&bytes[start..end])
                        .unwrap_or("")
                        .to_string(),
                );
            }
            // token "plein"
            let start = self.pos;
            while self.pos < bytes.len() {
                let c = bytes[self.pos];
                if c.is_ascii_whitespace() || c == b'{' || c == b'}' || c == b'"' {
                    break;
                }
                // Stop sur "//" ou "/*"
                if c == b'/' && self.pos + 1 < bytes.len() {
                    let next = bytes[self.pos + 1];
                    if next == b'/' || next == b'*' {
                        break;
                    }
                }
                self.pos += 1;
            }
            let end = self.pos;
            if start == end {
                return None;
            }
            return Some(
                std::str::from_utf8(&bytes[start..end])
                    .unwrap_or("")
                    .to_string(),
            );
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        let bytes = self.src.as_bytes();
        loop {
            // whitespace
            while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.pos + 1 >= bytes.len() {
                return;
            }
            // line comment
            if bytes[self.pos] == b'/' && bytes[self.pos + 1] == b'/' {
                self.pos += 2;
                while self.pos < bytes.len() && bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }
            // block comment
            if bytes[self.pos] == b'/' && bytes[self.pos + 1] == b'*' {
                self.pos += 2;
                while self.pos + 1 < bytes.len()
                    && !(bytes[self.pos] == b'*' && bytes[self.pos + 1] == b'/')
                {
                    self.pos += 1;
                }
                self.pos = (self.pos + 2).min(bytes.len());
                continue;
            }
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_tokens() {
        let mut tk = Tokenizer::new("foo bar\n{ baz }");
        assert_eq!(tk.next().as_deref(), Some("foo"));
        assert_eq!(tk.next().as_deref(), Some("bar"));
        assert_eq!(tk.next().as_deref(), Some("{"));
        assert_eq!(tk.next().as_deref(), Some("baz"));
        assert_eq!(tk.next().as_deref(), Some("}"));
        assert_eq!(tk.next(), None);
    }

    #[test]
    fn quoted_strings() {
        let mut tk = Tokenizer::new(r#"  "hello world"  next"#);
        assert_eq!(tk.next().as_deref(), Some("hello world"));
        assert_eq!(tk.next().as_deref(), Some("next"));
    }

    #[test]
    fn strips_comments() {
        let mut tk = Tokenizer::new("a // line comment\n b /* block */ c");
        assert_eq!(tk.next().as_deref(), Some("a"));
        assert_eq!(tk.next().as_deref(), Some("b"));
        assert_eq!(tk.next().as_deref(), Some("c"));
    }

    #[test]
    fn peek_is_idempotent() {
        let mut tk = Tokenizer::new("x y");
        assert_eq!(tk.peek().as_deref(), Some("x"));
        assert_eq!(tk.peek().as_deref(), Some("x"));
        assert_eq!(tk.next().as_deref(), Some("x"));
        assert_eq!(tk.next().as_deref(), Some("y"));
    }
}
