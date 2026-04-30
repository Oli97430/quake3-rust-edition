//! Chargeur d'images pour Q3 : **TGA**, **JPG**, **PNG**.
//!
//! Q3 cherche typiquement une texture par nom sans extension (ex. `textures/
//! base_floor/concrete`) et teste d'abord `.tga` puis `.jpg`. Ce crate
//! reproduit cette logique et maintient un cache.
//!
//! # Améliorations vs C original
//!
//! * **PNG** en plus (absent du Q3 d'origine).
//! * **sRGB correct** : on décode en `Rgba8` pixel-parfait et on confie la
//!   conversion au sampler GPU (pas d'overbright CPU).
//! * **Cache thread-safe** via `parking_lot::RwLock`.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

use hashbrown::HashMap;
use image::{DynamicImage, ImageFormat};
use parking_lot::RwLock;
use q3_common::{Error, Result};
use q3_filesystem::Vfs;
use std::sync::Arc;
use tracing::{debug, warn};

/// Image RGBA 8 bits prête à être uploadée au GPU.
#[derive(Debug, Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` octets en ordre RGBA, top-left origin.
    pub pixels: Arc<[u8]>,
    /// `true` si l'image contient des pixels vraiment transparents (utile
    /// pour le shader system : choisit blend mode par défaut).
    pub has_alpha: bool,
}

impl Image {
    /// Décode depuis un buffer. Le format est deviné par la signature
    /// magique, sinon par `hint` si fournie (ex. `"tga"`).
    pub fn decode(bytes: &[u8], hint: Option<&str>) -> Result<Self> {
        let format = guess_format(bytes, hint);
        let dynamic = match format {
            Some(fmt) => image::load_from_memory_with_format(bytes, fmt),
            None => image::load_from_memory(bytes),
        }
        .map_err(|e| Error::Parse(format!("image decode failed: {e}")))?;
        Ok(Self::from_dynamic(dynamic))
    }

    fn from_dynamic(img: DynamicImage) -> Self {
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let pixels: Arc<[u8]> = rgba.into_raw().into();
        // détection alpha : si au moins un pixel est < 255 → vraie alpha.
        let has_alpha = pixels.chunks_exact(4).any(|p| p[3] < 255);
        Self {
            width: w,
            height: h,
            pixels,
            has_alpha,
        }
    }

    /// Crée un pixel uni de la couleur donnée (utile pour `$whiteimage` etc.).
    pub fn solid(color: [u8; 4], size: u32) -> Self {
        let n = (size * size) as usize;
        let mut pixels = Vec::with_capacity(n * 4);
        for _ in 0..n {
            pixels.extend_from_slice(&color);
        }
        Self {
            width: size,
            height: size,
            pixels: pixels.into(),
            has_alpha: color[3] < 255,
        }
    }
}

fn guess_format(bytes: &[u8], hint: Option<&str>) -> Option<ImageFormat> {
    if bytes.len() >= 8 && &bytes[..8] == b"\x89PNG\r\n\x1a\n" {
        return Some(ImageFormat::Png);
    }
    if bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
        return Some(ImageFormat::Jpeg);
    }
    match hint.map(str::to_ascii_lowercase).as_deref() {
        Some("tga") => Some(ImageFormat::Tga),
        Some("png") => Some(ImageFormat::Png),
        Some("jpg") | Some("jpeg") => Some(ImageFormat::Jpeg),
        _ => None,
    }
}

/// Cache d'images adossé à un VFS.
#[derive(Clone)]
pub struct ImageCache {
    vfs: Vfs,
    inner: Arc<RwLock<HashMap<String, Arc<Image>>>>,
}

impl ImageCache {
    pub fn new(vfs: Vfs) -> Self {
        Self {
            vfs,
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Résout un nom logique (ex. `"textures/base_floor/concrete"`) en
    /// essayant `.tga`, `.jpg`, `.png` dans cet ordre Quake-like.
    pub fn load(&self, name: &str) -> Result<Arc<Image>> {
        let key = name.trim_end_matches('/').to_ascii_lowercase();
        if let Some(hit) = self.inner.read().get(&key) {
            return Ok(hit.clone());
        }
        let (bytes, hint) = self.read_with_extensions(&key)?;
        let img = Image::decode(&bytes, Some(hint))?;
        let arc = Arc::new(img);
        self.inner.write().insert(key, arc.clone());
        Ok(arc)
    }

    fn read_with_extensions(&self, base: &str) -> Result<(Vec<u8>, &'static str)> {
        // Si le nom a déjà une extension, on honore.
        if let Some(ext) = base.rsplit('.').next().filter(|e| e.len() <= 4) {
            let ext_lc = ext.to_ascii_lowercase();
            if matches!(ext_lc.as_str(), "tga" | "jpg" | "jpeg" | "png") {
                let bytes = self.vfs.read(base)?;
                let hint = match ext_lc.as_str() {
                    "tga" => "tga",
                    "png" => "png",
                    _ => "jpg",
                };
                return Ok((bytes, hint));
            }
        }
        for (ext, hint) in [("tga", "tga"), ("jpg", "jpg"), ("png", "png")] {
            let candidate = format!("{base}.{ext}");
            match self.vfs.read(&candidate) {
                Ok(b) => {
                    debug!("image: '{}' résolu via .{}", base, ext);
                    return Ok((b, hint));
                }
                Err(_) => continue,
            }
        }
        warn!("image: '{}' introuvable (tga/jpg/png)", base);
        Err(Error::Fs(format!("image not found: {base}")))
    }

    /// Retourne le nombre d'images en cache.
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    pub fn clear(&self) {
        self.inner.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_image_has_expected_pixels() {
        let img = Image::solid([255, 0, 0, 255], 2);
        assert_eq!(img.width, 2);
        assert_eq!(img.height, 2);
        assert_eq!(img.pixels.len(), 16);
        assert!(!img.has_alpha);
        assert_eq!(&img.pixels[..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn solid_image_with_alpha_flagged() {
        let img = Image::solid([0, 0, 0, 128], 1);
        assert!(img.has_alpha);
    }

    #[test]
    fn png_is_detected_by_magic() {
        let mut fake = vec![0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];
        fake.extend_from_slice(b"garbage");
        assert_eq!(guess_format(&fake, None), Some(ImageFormat::Png));
    }

    #[test]
    fn jpeg_magic() {
        let fake = [0xFF, 0xD8, 0xFF, 0xE0, 0, 0];
        assert_eq!(guess_format(&fake, None), Some(ImageFormat::Jpeg));
    }

    #[test]
    fn tga_needs_hint() {
        let fake = [0u8; 18];
        assert!(guess_format(&fake, None).is_none());
        assert_eq!(guess_format(&fake, Some("tga")), Some(ImageFormat::Tga));
    }
}
