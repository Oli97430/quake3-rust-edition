//! Parseur de scripts `.shader` Quake 3.
//!
//! # Format (résumé)
//!
//! Un fichier `.shader` contient une liste de shaders, chacun ayant la forme :
//!
//! ```text
//! textures/base_wall/metal
//! {
//!     surfaceparm metalsteps
//!     cull back
//!     {
//!         map textures/base_wall/metal.tga
//!         rgbGen identity
//!     }
//!     {
//!         map $lightmap
//!         blendFunc filter
//!         rgbGen identity
//!     }
//! }
//! ```
//!
//! On parse le **sous-ensemble principal** utilisé par les maps de base du
//! jeu original. Les directives inconnues sont loggées puis ignorées — le
//! parsing continue sans erreur (comportement identique à `R_ParseShader`).

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

pub mod tokenizer;
pub mod value;

pub use tokenizer::Tokenizer;
pub use value::*;

use hashbrown::HashMap;
use smallvec::SmallVec;
use std::sync::Arc;
use tracing::{debug, warn};

/// Un shader parsé.
#[derive(Debug, Clone, Default)]
pub struct Shader {
    pub name: String,
    pub stages: Vec<Stage>,
    pub cull: CullMode,
    pub sort: Option<f32>,
    pub surface_params: SmallVec<[String; 4]>,
    pub deform_vertexes: Vec<DeformVertexes>,
    pub sky_parms: Option<SkyParms>,
    pub fog_parms: Option<FogParms>,
    pub polygon_offset: bool,
    pub no_mipmaps: bool,
    pub no_picmip: bool,
    pub portal: bool,
    pub is_sky: bool,
    pub q3map_lightimage: Option<String>,
    pub qer_editorimage: Option<String>,
}

/// Une passe de rendu.
#[derive(Debug, Clone, Default)]
pub struct Stage {
    pub map: MapSource,
    pub blend: Option<BlendFunc>,
    pub rgb_gen: Option<RgbGen>,
    pub alpha_gen: Option<AlphaGen>,
    pub tc_gen: Option<TcGen>,
    pub tc_mods: SmallVec<[TcMod; 2]>,
    pub alpha_func: Option<AlphaFunc>,
    pub depth_func: DepthFunc,
    pub depth_write: Option<bool>,
    pub detail: bool,
    pub clamp: bool,
}

/// Registre : nom de shader (lowercase) → `Shader`.
#[derive(Debug, Clone, Default)]
pub struct ShaderRegistry {
    shaders: HashMap<String, Arc<Shader>>,
}

impl ShaderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse un fichier `.shader` complet et ajoute tous ses shaders au registre.
    ///
    /// Le `source_name` est seulement utilisé pour les messages de log.
    pub fn parse_file(&mut self, source: &str, source_name: &str) -> usize {
        let mut tk = Tokenizer::new(source);
        let mut count = 0usize;

        while let Some(name) = tk.next() {
            if name == "{" {
                warn!(
                    "shader parser ({source_name}): accolade sans nom — skip"
                );
                tk.skip_block();
                continue;
            }
            // Attend l'accolade d'ouverture.
            let Some(next) = tk.next() else { break };
            if next != "{" {
                warn!(
                    "shader parser ({source_name}): shader `{name}` : attendu '{{', trouvé `{next}`"
                );
                continue;
            }
            let mut shader = Shader {
                name: name.to_lowercase(),
                ..Default::default()
            };
            parse_shader_body(&mut tk, &mut shader, source_name);
            self.shaders.insert(shader.name.clone(), Arc::new(shader));
            count += 1;
        }

        debug!("shader parser ({source_name}) : {count} shaders parsés");
        count
    }

    pub fn get(&self, name: &str) -> Option<&Shader> {
        self.shaders.get(&name.to_lowercase()).map(|a| a.as_ref())
    }

    pub fn get_arc(&self, name: &str) -> Option<Arc<Shader>> {
        self.shaders.get(&name.to_lowercase()).cloned()
    }

    pub fn len(&self) -> usize {
        self.shaders.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shaders.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.shaders.keys().map(String::as_str)
    }
}

fn parse_shader_body(tk: &mut Tokenizer, shader: &mut Shader, source_name: &str) {
    while let Some(tok) = tk.next() {
        if tok == "}" {
            return;
        }
        if tok == "{" {
            let mut stage = Stage::default();
            parse_stage_body(tk, &mut stage, source_name, &shader.name);
            shader.stages.push(stage);
            continue;
        }
        let key = tok.to_lowercase();
        match key.as_str() {
            "cull" => {
                if let Some(v) = tk.next() {
                    shader.cull = match v.to_lowercase().as_str() {
                        "none" | "twosided" | "disable" => CullMode::None,
                        "front" => CullMode::Front,
                        "back" => CullMode::Back,
                        _ => CullMode::Front,
                    };
                }
            }
            "sort" => {
                if let Some(v) = tk.next() {
                    shader.sort = parse_sort(&v);
                }
            }
            "surfaceparm" => {
                if let Some(v) = tk.next() {
                    shader.surface_params.push(v.to_lowercase());
                }
            }
            "polygonoffset" => shader.polygon_offset = true,
            "nomipmaps" => shader.no_mipmaps = true,
            "nopicmip" => shader.no_picmip = true,
            "portal" => shader.portal = true,
            "skyparms" => {
                let far = tk.next();
                let cloud = tk.next();
                let near = tk.next();
                shader.sky_parms = Some(SkyParms {
                    far_box: or_none(far),
                    cloud_height: cloud
                        .as_deref()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(512.0),
                    near_box: or_none(near),
                });
                shader.is_sky = true;
            }
            "fogparms" => {
                // Grammaire canonique Q3 : `fogparms ( r g b ) distance`
                // Mais plusieurs shaders id (ex: `scripts/eerie.shader` dans
                // pak0.pk3) shippent sans parenthèses : `fogparms r g b d1 d2`
                // (le 4e nombre = `depthForOpaque`, le 5e = distance de fog).
                // On accepte les deux formes pour ne pas avaler `{` sur la
                // stage suivante, ce qui casserait tout le reste du fichier.
                let has_paren = tk.peek().as_deref() == Some("(");
                if has_paren {
                    tk.next(); // consume "("
                }
                let r = tk.next().and_then(|s| s.parse().ok()).unwrap_or(0.5);
                let g = tk.next().and_then(|s| s.parse().ok()).unwrap_or(0.5);
                let b = tk.next().and_then(|s| s.parse().ok()).unwrap_or(0.5);
                if has_paren {
                    // Consume ")" — si c'est pas ça, on skip au passage plutôt
                    // que de laisser flotter (le parseur Q3 strict abandonne
                    // le shader ici ; on est plus lenient).
                    let _ = tk.next();
                }
                // Dans la variante sans parenthèse, il y a deux nombres après
                // (depthForOpaque puis distance). On prend le dernier comme
                // distance effective. Variante canonique : un seul.
                let first = tk.next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(1024.0);
                let distance = if !has_paren {
                    tk.next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(first)
                } else {
                    first
                };
                shader.fog_parms = Some(FogParms {
                    color: [r, g, b],
                    distance,
                });
            }
            "deformvertexes" => {
                if let Some(d) = parse_deform(tk) {
                    shader.deform_vertexes.push(d);
                }
            }
            "q3map_lightimage" => {
                shader.q3map_lightimage = tk.next();
            }
            "qer_editorimage" => {
                shader.qer_editorimage = tk.next();
            }
            _ => {
                // ligne inconnue → on la consomme jusqu'à EOL implicite.
                // La plupart des directives de shader tiennent sur une ligne ;
                // on utilise un heuristique : consommer les tokens jusqu'au
                // prochain '{' ou '}' ou nouveau mot-clé.
                tk.skip_line();
            }
        }
    }
}

fn parse_stage_body(tk: &mut Tokenizer, stage: &mut Stage, source: &str, shader_name: &str) {
    while let Some(tok) = tk.next() {
        if tok == "}" {
            return;
        }
        match tok.to_lowercase().as_str() {
            "map" => {
                if let Some(v) = tk.next() {
                    stage.map = match v.as_str() {
                        "$lightmap" => MapSource::Lightmap,
                        "$whiteimage" => MapSource::White,
                        other => MapSource::Texture(other.to_string()),
                    };
                }
            }
            "clampmap" => {
                if let Some(v) = tk.next() {
                    stage.map = MapSource::Texture(v);
                    stage.clamp = true;
                }
            }
            "animmap" => {
                let freq = tk.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);
                let mut frames: SmallVec<[String; 8]> = SmallVec::new();
                while let Some(n) = tk.peek() {
                    if is_keyword(&n) || n == "}" {
                        break;
                    }
                    frames.push(tk.next().unwrap());
                }
                stage.map = MapSource::Animated { freq, frames };
            }
            "blendfunc" => {
                stage.blend = parse_blend_func(tk);
            }
            "rgbgen" => {
                stage.rgb_gen = parse_rgb_gen(tk);
            }
            "alphagen" => {
                stage.alpha_gen = parse_alpha_gen(tk);
            }
            "tcgen" | "texgen" => {
                stage.tc_gen = parse_tc_gen(tk);
            }
            "tcmod" => {
                if let Some(m) = parse_tc_mod(tk) {
                    stage.tc_mods.push(m);
                }
            }
            "alphafunc" => {
                if let Some(v) = tk.next() {
                    stage.alpha_func = match v.to_uppercase().as_str() {
                        "GT0" => Some(AlphaFunc::Gt0),
                        "LT128" => Some(AlphaFunc::Lt128),
                        "GE128" => Some(AlphaFunc::Ge128),
                        _ => None,
                    };
                }
            }
            "depthfunc" => {
                if let Some(v) = tk.next() {
                    stage.depth_func = match v.to_lowercase().as_str() {
                        "equal" => DepthFunc::Equal,
                        _ => DepthFunc::LessEqual,
                    };
                }
            }
            "depthwrite" => {
                stage.depth_write = Some(true);
            }
            "detail" => {
                stage.detail = true;
            }
            _other => {
                tk.skip_line();
                let _ = source;
                let _ = shader_name;
            }
        }
    }
}

fn or_none(s: Option<String>) -> Option<String> {
    match s {
        Some(s) if s != "-" => Some(s),
        _ => None,
    }
}

fn is_keyword(s: &str) -> bool {
    matches!(
        s.to_lowercase().as_str(),
        "map"
            | "clampmap"
            | "animmap"
            | "blendfunc"
            | "rgbgen"
            | "alphagen"
            | "tcgen"
            | "tcmod"
            | "alphafunc"
            | "depthfunc"
            | "depthwrite"
            | "detail"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_two_stage_shader() {
        let src = r#"
        textures/base_wall/metal
        {
            surfaceparm metalsteps
            cull back
            {
                map textures/base_wall/metal.tga
                rgbGen identity
            }
            {
                map $lightmap
                blendFunc filter
                rgbGen identity
            }
        }
        "#;
        let mut r = ShaderRegistry::new();
        assert_eq!(r.parse_file(src, "test"), 1);
        let sh = r.get("textures/base_wall/metal").unwrap();
        assert_eq!(sh.stages.len(), 2);
        assert_eq!(sh.cull, CullMode::Back);
        assert_eq!(sh.surface_params.as_slice(), &["metalsteps".to_string()]);
        assert!(matches!(sh.stages[0].map, MapSource::Texture(_)));
        assert!(matches!(sh.stages[1].map, MapSource::Lightmap));
    }

    #[test]
    fn skips_unknown_directives() {
        let src = r#"
        foo
        {
            plopblarg 1 2 3
            {
                map x.tga
                glorb 42
            }
        }
        "#;
        let mut r = ShaderRegistry::new();
        assert_eq!(r.parse_file(src, "test"), 1);
        assert!(r.get("foo").is_some());
    }

    #[test]
    fn parses_blendfunc_keywords() {
        let src = r#"
        foo
        {
            { map x.tga blendFunc add }
            { map y.tga blendFunc filter }
            { map z.tga blendFunc blend }
            { map w.tga blendFunc GL_ONE GL_ZERO }
        }
        "#;
        let mut r = ShaderRegistry::new();
        r.parse_file(src, "test");
        let s = r.get("foo").unwrap();
        assert_eq!(
            s.stages[0].blend,
            Some(BlendFunc::Custom(BlendFactor::One, BlendFactor::One))
        );
        assert_eq!(
            s.stages[1].blend,
            Some(BlendFunc::Custom(BlendFactor::DstColor, BlendFactor::Zero))
        );
    }

    #[test]
    fn comments_are_stripped() {
        let src = "// header comment\nfoo\n{\n  // inside\n  cull none\n}\n";
        let mut r = ShaderRegistry::new();
        assert_eq!(r.parse_file(src, "test"), 1);
        assert_eq!(r.get("foo").unwrap().cull, CullMode::None);
    }

    /// Régression : shader id officiel `eerie.shader` (pak0.pk3) utilise
    /// `fogparms r g b depthForOpaque distance` SANS parenthèses, alors que
    /// la grammaire canonique attend `fogparms ( r g b ) distance`. Avant
    /// le fix, notre parseur avalait aveuglément 6 tokens après `fogparms`,
    /// consommant le `{` de la stage suivante et corrompant tout le fichier
    /// (warn cascade `shader '}' : attendu '{', trouvé <nom du shader suivant>`).
    #[test]
    fn fogparms_without_parentheses_still_parses_following_stage() {
        let src = r#"
        textures/eerie/lavahell
        {
            fogparms 0.8519142 0.309723 0.0 128 128
            {
                map textures/eerie/lavahell.tga
            }
        }
        textures/eerie/lavahell2
        {
            {
                map textures/eerie/lavahell.tga
            }
        }
        "#;
        let mut r = ShaderRegistry::new();
        // Les DEUX shaders doivent être reconnus : sans le fix, le second
        // nom était lu comme un token orphelin après la cascade.
        assert_eq!(r.parse_file(src, "test_eerie"), 2);
        let lavahell = r.get("textures/eerie/lavahell").unwrap();
        assert_eq!(lavahell.stages.len(), 1);
        let fog = lavahell.fog_parms.as_ref().expect("fog_parms");
        assert!((fog.color[0] - 0.8519142).abs() < 1e-5);
        assert!((fog.color[1] - 0.309723).abs() < 1e-5);
        assert_eq!(fog.color[2], 0.0);
        // Le dernier nombre de la variante sans parens = distance effective.
        assert_eq!(fog.distance, 128.0);
        assert!(r.get("textures/eerie/lavahell2").is_some());
    }

    /// Forme canonique Q3 : `fogparms ( r g b ) distance` — ne doit pas
    /// régresser avec le fix « tolerant » ci-dessus.
    #[test]
    fn fogparms_with_parentheses_canonical_form() {
        let src = r#"
        textures/canon/fog
        {
            fogparms ( 0.1 0.2 0.3 ) 512
            {
                map $lightmap
            }
        }
        "#;
        let mut r = ShaderRegistry::new();
        assert_eq!(r.parse_file(src, "test_canon"), 1);
        let sh = r.get("textures/canon/fog").unwrap();
        let fog = sh.fog_parms.as_ref().expect("fog_parms");
        assert!((fog.color[0] - 0.1).abs() < 1e-5);
        assert!((fog.color[1] - 0.2).abs() < 1e-5);
        assert!((fog.color[2] - 0.3).abs() < 1e-5);
        assert_eq!(fog.distance, 512.0);
        assert_eq!(sh.stages.len(), 1);
    }
}
