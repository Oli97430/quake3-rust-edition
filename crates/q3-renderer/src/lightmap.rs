//! Upload des lightmaps BSP en `texture_2d_array`.
//!
//! Chaque map Q3 embarque 0..N lightmaps 128×128 RGB. On les concatène dans
//! une texture array wgpu. Une couche "blanche" est ajoutée à la fin, utilisée
//! quand une surface a `lightmap_num < 0` (pas de lightmap).

use q3_bsp::{Bsp, LIGHTMAP_SIZE};
use std::num::NonZeroU32;

pub struct LightmapArray {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub layer_count: u32,
    /// Index de la couche blanche de fallback (= layer_count - 1).
    pub white_layer: u32,
}

impl LightmapArray {
    /// Construit une array d'au moins 1 couche (la blanche) même si la BSP
    /// n'a pas de lightmaps.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, bsp: &Bsp) -> Self {
        let n_map = bsp.num_lightmaps() as u32;
        let total = n_map + 1; // +1 pour la couche blanche de fallback

        let size = wgpu::Extent3d {
            width: LIGHTMAP_SIZE as u32,
            height: LIGHTMAP_SIZE as u32,
            depth_or_array_layers: total,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lightmap-array"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Les lightmaps Q3 sont déjà gamma-corrigées ; on les stocke en
            // RGBA8Unorm (pas sRGB) pour préserver la courbe originale.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload chaque couche. La BSP stocke en RGB ; on étend à RGBA avec α=255.
        let mut rgba = vec![255u8; (LIGHTMAP_SIZE * LIGHTMAP_SIZE * 4) as usize];
        for layer in 0..n_map {
            let Some(rgb) = bsp.lightmap(layer as usize) else {
                continue;
            };
            for (i, chunk) in rgb.chunks_exact(3).enumerate() {
                rgba[i * 4] = overbright(chunk[0]);
                rgba[i * 4 + 1] = overbright(chunk[1]);
                rgba[i * 4 + 2] = overbright(chunk[2]);
                rgba[i * 4 + 3] = 255;
            }
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(LIGHTMAP_SIZE as u32 * 4).map(|n| n.get()),
                    rows_per_image: NonZeroU32::new(LIGHTMAP_SIZE as u32).map(|n| n.get()),
                },
                wgpu::Extent3d {
                    width: LIGHTMAP_SIZE as u32,
                    height: LIGHTMAP_SIZE as u32,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Couche blanche de fallback.
        rgba.fill(255);
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: n_map },
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(LIGHTMAP_SIZE as u32 * 4).map(|n| n.get()),
                rows_per_image: NonZeroU32::new(LIGHTMAP_SIZE as u32).map(|n| n.get()),
            },
            wgpu::Extent3d {
                width: LIGHTMAP_SIZE as u32,
                height: LIGHTMAP_SIZE as u32,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("lightmap-array-view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lightmap-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            layer_count: total,
            white_layer: total - 1,
        }
    }
}

/// Approche "overbright" de Q3 : les lightmaps sont encodées à 0.5× leur
/// valeur réelle, pour laisser de la marge aux shaders additifs. On les
/// doublé au décodage (clamp 255).
#[inline]
fn overbright(v: u8) -> u8 {
    v.saturating_mul(2)
}
