//! Types de valeurs pour les directives de shader (rgbGen, tcMod, blendFunc…).
//!
//! Chaque variante correspond à une forme acceptée par le parser de Q3.

use crate::Tokenizer;
use smallvec::SmallVec;

// ---- cull -----------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CullMode {
    #[default]
    Front,
    Back,
    None,
}

// ---- map source -----------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Default)]
pub enum MapSource {
    #[default]
    None,
    /// Fichier texture (TGA/JPG).
    Texture(String),
    /// `$lightmap` — texture de lightmap courante.
    Lightmap,
    /// `$whiteimage` — texture blanche 1×1.
    White,
    /// `animMap freq f1 f2 f3 …`
    Animated { freq: f32, frames: SmallVec<[String; 8]> },
}

// ---- blend ----------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendFunc {
    /// Raccourcis du format shader. `add = ONE ONE`, `filter = DST_COLOR ZERO`,
    /// `blend = SRC_ALPHA ONE_MINUS_SRC_ALPHA`.
    Custom(BlendFactor, BlendFactor),
}

fn parse_blend_factor(s: &str) -> Option<BlendFactor> {
    Some(match s.to_uppercase().as_str() {
        "GL_ZERO" => BlendFactor::Zero,
        "GL_ONE" => BlendFactor::One,
        "GL_SRC_COLOR" => BlendFactor::SrcColor,
        "GL_ONE_MINUS_SRC_COLOR" => BlendFactor::OneMinusSrcColor,
        "GL_DST_COLOR" => BlendFactor::DstColor,
        "GL_ONE_MINUS_DST_COLOR" => BlendFactor::OneMinusDstColor,
        "GL_SRC_ALPHA" => BlendFactor::SrcAlpha,
        "GL_ONE_MINUS_SRC_ALPHA" => BlendFactor::OneMinusSrcAlpha,
        "GL_DST_ALPHA" => BlendFactor::DstAlpha,
        "GL_ONE_MINUS_DST_ALPHA" => BlendFactor::OneMinusDstAlpha,
        "GL_SRC_ALPHA_SATURATE" => BlendFactor::SrcAlphaSaturate,
        _ => return None,
    })
}

pub(crate) fn parse_blend_func(tk: &mut Tokenizer) -> Option<BlendFunc> {
    let first = tk.next()?;
    // Forme courte : add / filter / blend
    match first.to_lowercase().as_str() {
        "add" => return Some(BlendFunc::Custom(BlendFactor::One, BlendFactor::One)),
        "filter" => {
            return Some(BlendFunc::Custom(BlendFactor::DstColor, BlendFactor::Zero))
        }
        "blend" => {
            return Some(BlendFunc::Custom(
                BlendFactor::SrcAlpha,
                BlendFactor::OneMinusSrcAlpha,
            ))
        }
        _ => {}
    }
    // Forme longue : src dst
    let second = tk.next()?;
    let sf = parse_blend_factor(&first)?;
    let df = parse_blend_factor(&second)?;
    Some(BlendFunc::Custom(sf, df))
}

// ---- rgbGen / alphaGen ----------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum RgbGen {
    Identity,
    IdentityLighting,
    Entity,
    OneMinusEntity,
    Vertex,
    ExactVertex,
    LightingDiffuse,
    Wave(Waveform),
    Const([f32; 3]),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlphaGen {
    Identity,
    Const(f32),
    Entity,
    OneMinusEntity,
    Vertex,
    LightingSpecular,
    Wave(Waveform),
    Portal(f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveFunc {
    Sin,
    Triangle,
    Square,
    Sawtooth,
    InverseSawtooth,
    Noise,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Waveform {
    pub func: WaveFunc,
    pub base: f32,
    pub amp: f32,
    pub phase: f32,
    pub freq: f32,
}

fn parse_wave_func(s: &str) -> Option<WaveFunc> {
    Some(match s.to_lowercase().as_str() {
        "sin" => WaveFunc::Sin,
        "triangle" => WaveFunc::Triangle,
        "square" => WaveFunc::Square,
        "sawtooth" => WaveFunc::Sawtooth,
        "inversesawtooth" => WaveFunc::InverseSawtooth,
        "noise" => WaveFunc::Noise,
        _ => return None,
    })
}

fn parse_wave(tk: &mut Tokenizer) -> Option<Waveform> {
    let func = parse_wave_func(&tk.next()?)?;
    let base = tk.next()?.parse().ok()?;
    let amp = tk.next()?.parse().ok()?;
    let phase = tk.next()?.parse().ok()?;
    let freq = tk.next()?.parse().ok()?;
    Some(Waveform { func, base, amp, phase, freq })
}

pub(crate) fn parse_rgb_gen(tk: &mut Tokenizer) -> Option<RgbGen> {
    let v = tk.next()?;
    Some(match v.to_lowercase().as_str() {
        "identity" => RgbGen::Identity,
        "identitylighting" => RgbGen::IdentityLighting,
        "entity" => RgbGen::Entity,
        "oneminusentity" => RgbGen::OneMinusEntity,
        "vertex" => RgbGen::Vertex,
        "exactvertex" => RgbGen::ExactVertex,
        "lightingdiffuse" => RgbGen::LightingDiffuse,
        "wave" => RgbGen::Wave(parse_wave(tk)?),
        "const" => {
            // ( r g b )
            let _ = tk.next();
            let r = tk.next()?.parse().ok()?;
            let g = tk.next()?.parse().ok()?;
            let b = tk.next()?.parse().ok()?;
            let _ = tk.next();
            RgbGen::Const([r, g, b])
        }
        _ => return None,
    })
}

pub(crate) fn parse_alpha_gen(tk: &mut Tokenizer) -> Option<AlphaGen> {
    let v = tk.next()?;
    Some(match v.to_lowercase().as_str() {
        "identity" => AlphaGen::Identity,
        "entity" => AlphaGen::Entity,
        "oneminusentity" => AlphaGen::OneMinusEntity,
        "vertex" => AlphaGen::Vertex,
        "lightingspecular" => AlphaGen::LightingSpecular,
        "wave" => AlphaGen::Wave(parse_wave(tk)?),
        "const" => AlphaGen::Const(tk.next()?.parse().ok()?),
        "portal" => AlphaGen::Portal(tk.next()?.parse().ok()?),
        _ => return None,
    })
}

// ---- tcGen / tcMod --------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum TcGen {
    Base,
    Lightmap,
    Environment,
    Vector { s: [f32; 3], t: [f32; 3] },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TcMod {
    Rotate(f32),
    Scroll(f32, f32),
    Scale(f32, f32),
    Stretch(Waveform),
    Turb { base: f32, amp: f32, phase: f32, freq: f32 },
    Transform([f32; 6]),
}

pub(crate) fn parse_tc_gen(tk: &mut Tokenizer) -> Option<TcGen> {
    let v = tk.next()?;
    Some(match v.to_lowercase().as_str() {
        "base" | "texture" => TcGen::Base,
        "lightmap" => TcGen::Lightmap,
        "environment" => TcGen::Environment,
        "vector" => {
            let s = read_vec3_paren(tk)?;
            let t = read_vec3_paren(tk)?;
            TcGen::Vector { s, t }
        }
        _ => return None,
    })
}

fn read_vec3_paren(tk: &mut Tokenizer) -> Option<[f32; 3]> {
    let _ = tk.next(); // (
    let a = tk.next()?.parse().ok()?;
    let b = tk.next()?.parse().ok()?;
    let c = tk.next()?.parse().ok()?;
    let _ = tk.next(); // )
    Some([a, b, c])
}

pub(crate) fn parse_tc_mod(tk: &mut Tokenizer) -> Option<TcMod> {
    let v = tk.next()?;
    Some(match v.to_lowercase().as_str() {
        "rotate" => TcMod::Rotate(tk.next()?.parse().ok()?),
        "scroll" => {
            let a = tk.next()?.parse().ok()?;
            let b = tk.next()?.parse().ok()?;
            TcMod::Scroll(a, b)
        }
        "scale" => {
            let a = tk.next()?.parse().ok()?;
            let b = tk.next()?.parse().ok()?;
            TcMod::Scale(a, b)
        }
        "stretch" => TcMod::Stretch(parse_wave(tk)?),
        "turb" => {
            let base = tk.next()?.parse().ok()?;
            let amp = tk.next()?.parse().ok()?;
            let phase = tk.next()?.parse().ok()?;
            let freq = tk.next()?.parse().ok()?;
            TcMod::Turb { base, amp, phase, freq }
        }
        "transform" => {
            let mut m = [0.0f32; 6];
            for slot in m.iter_mut() {
                *slot = tk.next()?.parse().ok()?;
            }
            TcMod::Transform(m)
        }
        _ => return None,
    })
}

// ---- alpha / depth --------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaFunc {
    Gt0,
    Lt128,
    Ge128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DepthFunc {
    #[default]
    LessEqual,
    Equal,
}

// ---- deformVertexes -------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum DeformVertexes {
    Wave {
        spread: f32,
        wave: Waveform,
    },
    Normal {
        amplitude: f32,
        frequency: f32,
    },
    Bulge {
        width: f32,
        height: f32,
        speed: f32,
    },
    Move {
        dir: [f32; 3],
        wave: Waveform,
    },
    AutoSprite,
    AutoSprite2,
}

pub(crate) fn parse_deform(tk: &mut Tokenizer) -> Option<DeformVertexes> {
    let v = tk.next()?;
    Some(match v.to_lowercase().as_str() {
        "wave" => DeformVertexes::Wave {
            spread: tk.next()?.parse().ok()?,
            wave: parse_wave(tk)?,
        },
        "normal" => DeformVertexes::Normal {
            amplitude: tk.next()?.parse().ok()?,
            frequency: tk.next()?.parse().ok()?,
        },
        "bulge" => DeformVertexes::Bulge {
            width: tk.next()?.parse().ok()?,
            height: tk.next()?.parse().ok()?,
            speed: tk.next()?.parse().ok()?,
        },
        "move" => DeformVertexes::Move {
            dir: [
                tk.next()?.parse().ok()?,
                tk.next()?.parse().ok()?,
                tk.next()?.parse().ok()?,
            ],
            wave: parse_wave(tk)?,
        },
        "autosprite" => DeformVertexes::AutoSprite,
        "autosprite2" => DeformVertexes::AutoSprite2,
        _ => return None,
    })
}

// ---- sky / fog ------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct SkyParms {
    pub far_box: Option<String>,
    pub cloud_height: f32,
    pub near_box: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FogParms {
    pub color: [f32; 3],
    pub distance: f32,
}

// ---- sort -----------------------------------------------------------------

pub(crate) fn parse_sort(s: &str) -> Option<f32> {
    // Valeurs nommées du jeu original.
    Some(match s.to_lowercase().as_str() {
        "portal" => 1.0,
        "sky" => 2.0,
        "opaque" => 3.0,
        "decal" => 4.0,
        "seethrough" => 5.0,
        "banner" => 6.0,
        "additive" => 9.0,
        "nearest" => 16.0,
        "underwater" => 8.0,
        num => num.parse().ok()?,
    })
}
