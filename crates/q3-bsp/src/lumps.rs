//! Helpers de haut niveau autour des lumps — itération sur les entités,
//! collection des triangles d'une surface, etc.

use crate::raw::{DSurface, DrawVert, SurfaceType};
use crate::Bsp;

/// Un couple clé/valeur parsé d'une entity.
#[derive(Debug, Clone)]
pub struct EntityKv {
    pub key: String,
    pub value: String,
}

/// Une entité logique du monde (info_player_start, func_door, light, etc.).
#[derive(Debug, Clone, Default)]
pub struct Entity {
    pub kvs: Vec<EntityKv>,
}

impl Entity {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.kvs
            .iter()
            .find(|kv| kv.key == key)
            .map(|kv| kv.value.as_str())
    }

    pub fn classname(&self) -> &str {
        self.get("classname").unwrap_or("")
    }

    /// Parse la chaîne "x y z" en trois floats. Valeur par défaut si manquant.
    pub fn vec3(&self, key: &str) -> Option<[f32; 3]> {
        let s = self.get(key)?;
        let mut it = s.split_ascii_whitespace().map(|t| t.parse::<f32>().ok());
        let x = it.next()??;
        let y = it.next()??;
        let z = it.next()??;
        Some([x, y, z])
    }

    pub fn f32(&self, key: &str) -> Option<f32> {
        self.get(key)?.parse().ok()
    }

    pub fn i32(&self, key: &str) -> Option<i32> {
        self.get(key)?.parse().ok()
    }
}

/// Parse le lump entities (texte à la syntaxe `{ "key" "value" ... }`).
///
/// Format exactement identique au `G_SpawnStrings` du code original :
/// chaque entité est une paire d'accolades, chaque ligne une paire de
/// chaînes quotées.
pub fn parse_entities(text: &str) -> Vec<Entity> {
    let mut out = Vec::new();
    let mut chars = text.chars().peekable();

    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        if c != '{' {
            // skip garbage
            chars.next();
            continue;
        }
        chars.next(); // consume {

        let mut ent = Entity::default();
        loop {
            // skip ws
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() {
                    chars.next();
                } else {
                    break;
                }
            }
            match chars.peek() {
                Some('}') => {
                    chars.next();
                    break;
                }
                Some('"') => {
                    chars.next();
                    let key = consume_quoted(&mut chars);
                    // whitespace
                    while let Some(&c) = chars.peek() {
                        if c.is_whitespace() {
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        let value = consume_quoted(&mut chars);
                        ent.kvs.push(EntityKv { key, value });
                    }
                }
                None => break,
                Some(_) => {
                    chars.next();
                }
            }
        }
        out.push(ent);
    }
    out
}

fn consume_quoted(chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
    let mut s = String::new();
    for c in chars.by_ref() {
        if c == '"' {
            break;
        }
        s.push(c);
    }
    s
}

/// Slice typé pointant vers les verts/indexes d'une surface.
pub struct SurfaceGeometry<'a> {
    pub kind: SurfaceType,
    pub verts: &'a [DrawVert],
    pub indexes: &'a [i32],
    pub shader_num: i32,
    pub lightmap_num: i32,
    pub patch_dim: (i32, i32),
}

impl Bsp {
    /// Parse le lump entities et renvoie la liste des entités logiques.
    pub fn parse_entities(&self) -> Vec<Entity> {
        parse_entities(&self.entities)
    }

    /// Donne un accès slice aux verts/indexes de la surface `i`.
    pub fn surface_geometry(&self, i: usize) -> Option<SurfaceGeometry<'_>> {
        let s: &DSurface = self.surfaces.get(i)?;
        let v_start = s.first_vert as usize;
        let v_end = v_start.checked_add(s.num_verts as usize)?;
        let i_start = s.first_index as usize;
        let i_end = i_start.checked_add(s.num_indexes as usize)?;
        Some(SurfaceGeometry {
            kind: s.kind(),
            verts: self.draw_verts.get(v_start..v_end)?,
            indexes: self.draw_indexes.get(i_start..i_end)?,
            shader_num: s.shader_num,
            lightmap_num: s.lightmap_num,
            patch_dim: (s.patch_width, s.patch_height),
        })
    }

    /// Retourne le nom du shader référencé par la surface `i`, s'il est valide.
    pub fn surface_shader_name(&self, i: usize) -> Option<&str> {
        let s = self.surfaces.get(i)?;
        self.shaders
            .get(s.shader_num as usize)
            .map(|sh| sh.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_entity() {
        let text = r#"
            {
                "classname" "worldspawn"
                "message" "Test Map"
            }
        "#;
        let ents = parse_entities(text);
        assert_eq!(ents.len(), 1);
        assert_eq!(ents[0].classname(), "worldspawn");
        assert_eq!(ents[0].get("message"), Some("Test Map"));
    }

    #[test]
    fn parse_multi_entity() {
        let text = r#"
            { "classname" "worldspawn" }
            { "classname" "info_player_start" "origin" "10 20 30" }
        "#;
        let ents = parse_entities(text);
        assert_eq!(ents.len(), 2);
        assert_eq!(ents[1].classname(), "info_player_start");
        assert_eq!(ents[1].vec3("origin"), Some([10.0, 20.0, 30.0]));
    }
}
