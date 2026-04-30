//! Type d'erreur centralisé. Chaque sous-système a sa variante pour permettre
//! un matching précis en amont.

use thiserror::Error;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("filesystem: {0}")]
    Fs(String),

    #[error("archive (pk3): {0}")]
    Archive(String),

    #[error("bsp: {0}")]
    Bsp(String),

    #[error("shader: {0}")]
    Shader(String),

    #[error("renderer: {0}")]
    Renderer(String),

    #[error("cvar `{name}`: {reason}")]
    Cvar { name: String, reason: String },

    #[error("command `{name}`: {reason}")]
    Cmd { name: String, reason: String },

    #[error("parse: {0}")]
    Parse(String),

    #[error("network: {0}")]
    Network(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Utf8(#[from] std::str::Utf8Error),

    #[error(transparent)]
    FromUtf8(#[from] std::string::FromUtf8Error),
}

impl Error {
    pub fn bsp(msg: impl Into<String>) -> Self {
        Self::Bsp(msg.into())
    }
    pub fn fs(msg: impl Into<String>) -> Self {
        Self::Fs(msg.into())
    }
    pub fn archive(msg: impl Into<String>) -> Self {
        Self::Archive(msg.into())
    }
    pub fn parse(msg: impl Into<String>) -> Self {
        Self::Parse(msg.into())
    }
    pub fn renderer(msg: impl Into<String>) -> Self {
        Self::Renderer(msg.into())
    }
}
