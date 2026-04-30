//! Chargeur de **textures cubiques** pour les skyboxes Q3.
//!
//! Q3 référence les skyboxes par un basename (ex. `env/killsky`) et suffixe
//! chaque face :
//!
//! | suffixe | direction Q3    | layer wgpu cube |
//! |---------|-----------------|-----------------|
//! | `_rt`   | +Y (gauche inv) | 0 (+X)          |
//! | `_lf`   | -Y              | 1 (-X)          |
//! | `_up`   | +Z              | 2 (+Y)          |
//! | `_dn`   | -Z              | 3 (-Y)          |
//! | `_ft`   | +X (avant)      | 4 (+Z)          |
//! | `_bk`   | -X              | 5 (-Z)          |
//!
//! Le shader de ciel convertit la direction de vue Q3 (X=fwd, Y=left, Z=up)
//! en direction wgpu (X=right, Y=up, Z=fwd) avant de sampler.
//!
//! Les 6 faces doivent avoir la **même** résolution. Format GPU : RGBA8 sRGB.

use q3_common::{Error, Result};
use q3_filesystem::Vfs;
use q3_image::ImageCache;
use std::sync::Arc;
use tracing::debug;

/// Ordre des suffixes Q3 — aligné avec l'ordre des layers wgpu cube
/// (`+X, -X, +Y, -Y, +Z, -Z`).
pub const FACE_SUFFIXES: [&str; 6] = ["_rt", "_lf", "_up", "_dn", "_ft", "_bk"];

/// Texture cubique + sampler prêts à binder.
pub struct Cubemap {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub size: u32,
    /// Basename logique (ex. `env/killsky`).
    pub base_path: String,
}

impl Cubemap {
    /// Tente de charger les 6 faces. Les chemins essayés sont
    /// `<base_path><suffixe>.tga` (puis `.jpg` / `.png` via le fallback de
    /// `ImageCache::load`).
    ///
    /// Renvoie `Err` si une face manque ou si les 6 faces n'ont pas la
    /// même résolution.
    pub fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _vfs: &Vfs,
        images: &ImageCache,
        base_path: &str,
    ) -> Result<Self> {
        let mut faces = Vec::with_capacity(6);
        let mut size: Option<u32> = None;
        for suffix in FACE_SUFFIXES.iter() {
            let name = format!("{base_path}{suffix}");
            let img = images
                .load(&name)
                .map_err(|e| Error::renderer(format!("cubemap: face '{name}' KO: {e}")))?;
            if img.width != img.height {
                return Err(Error::renderer(format!(
                    "cubemap: face '{name}' non carrée ({}x{})",
                    img.width, img.height
                )));
            }
            match size {
                None => size = Some(img.width),
                Some(s) if s != img.width => {
                    return Err(Error::renderer(format!(
                        "cubemap: face '{name}' fait {}, attendu {s}",
                        img.width
                    )));
                }
                _ => {}
            }
            faces.push(img);
        }
        let size = size.ok_or_else(|| Error::renderer("cubemap: aucune face"))?;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sky-cubemap"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (layer, img) in faces.iter().enumerate() {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                img.pixels.as_ref(),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(img.width * 4),
                    rows_per_image: Some(img.height),
                },
                wgpu::Extent3d {
                    width: img.width,
                    height: img.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("sky-cubemap-view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            array_layer_count: Some(6),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky-cubemap-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        debug!("cubemap: '{base_path}' chargée ({size}×{size} × 6 faces)");
        Ok(Self {
            texture,
            view,
            sampler,
            size,
            base_path: base_path.to_string(),
        })
    }

    /// Bind group layout standard pour une texture cubique + son sampler.
    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky-cubemap-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    /// Crée un bind group lié à `layout`.
    pub fn bind_group(
        self: &Arc<Self>,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky-cubemap-bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }
}
