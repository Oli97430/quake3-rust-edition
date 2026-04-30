//! Dynamic lights : halos radiaux qui illuminent les surfaces proches.
//!
//! Modèle physique simplifié : chaque dlight = point + rayon + couleur +
//! intensité.  La contribution ajoutée au fragment est
//! `color * intensity * max(0, 1 - dist/radius)^2` — tombée quadratique
//! classique, pas le `1 - d/r` linéaire de Q3 original (trop doux), mais
//! plus proche des `r_dynamiclight 1` modernes.
//!
//! Bind group layout côté wgpu :
//!
//! ```wgsl
//! struct GpuDlight {
//!     pos_radius: vec4<f32>,       // xyz=centre monde, w=rayon
//!     color_intensity: vec4<f32>,  // rgb=couleur, a=intensité (multiplicateur)
//! };
//! struct DlightBuffer {
//!     count: u32,
//!     _pad: vec3<u32>,
//!     lights: array<GpuDlight, MAX_DLIGHTS>,
//! };
//! ```
//!
//! Le buffer est mis à jour chaque frame juste avant le rendu du monde.

use bytemuck::{Pod, Zeroable};
use q3_math::Vec3;
use std::collections::VecDeque;
use std::sync::Arc;
use tracing::debug;

/// Max de dlights simultanées dans le buffer.  16 suffit largement pour
/// un combat à 8 bots : chacun porte au plus 1 muzzle flash + 1 projectile
/// en vol à un instant donné.  Les spawns au-delà évincent la plus
/// ancienne (FIFO).
pub const MAX_DLIGHTS: usize = 16;

#[derive(Debug, Clone, Copy)]
pub struct Dlight {
    pub center: Vec3,
    pub radius: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub spawn_time: f32,
    pub lifetime: f32,
}

impl Dlight {
    /// Intensité résiduelle à `now` (fade linéaire sur les derniers 50 %
    /// de la vie).  Renvoie `None` si expirée.
    pub fn intensity_at(&self, now: f32) -> Option<f32> {
        let age = now - self.spawn_time;
        if age < 0.0 || age >= self.lifetime {
            return None;
        }
        let half = self.lifetime * 0.5;
        let k = if age < half {
            1.0
        } else {
            1.0 - (age - half) / half
        };
        Some(self.intensity * k.clamp(0.0, 1.0))
    }
}

/// Entrée GPU d'un dlight — doit matcher le layout de `material.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuDlight {
    /// xyz = centre monde, w = rayon.
    pub pos_radius: [f32; 4],
    /// rgb = couleur, a = intensité.
    pub color_intensity: [f32; 4],
}

/// Buffer complet passé en uniform.  `count` + padding pour l'alignement
/// std140 (uniform buffers sur wgpu demandent 16 bytes d'alignement pour
/// les `array` et `vec3` suivants).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuDlightBuffer {
    pub count: u32,
    pub _pad: [u32; 3],
    pub lights: [GpuDlight; MAX_DLIGHTS],
}

impl Default for GpuDlightBuffer {
    fn default() -> Self {
        Self {
            count: 0,
            _pad: [0; 3],
            lights: [GpuDlight {
                pos_radius: [0.0; 4],
                color_intensity: [0.0; 4],
            }; MAX_DLIGHTS],
        }
    }
}

pub struct DlightSet {
    queue: Arc<wgpu::Queue>,
    pub bind_group_layout: Arc<wgpu::BindGroupLayout>,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    dlights: VecDeque<Dlight>,
}

impl DlightSet {
    pub fn new(device: &wgpu::Device, queue: Arc<wgpu::Queue>) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dlight-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                // FRAGMENT only — les sommets ne sont pas éclairés
                // (pas de diffuse per-vertex dans le pipeline actuel).
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dlight-ubuf"),
            size: std::mem::size_of::<GpuDlightBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Initialise à zéro — pas de dlight active tant qu'on n'a pas
        // appelé `flush`.
        queue.write_buffer(
            &buffer,
            0,
            bytemuck::bytes_of(&GpuDlightBuffer::default()),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dlight-bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            queue,
            bind_group_layout: Arc::new(bgl),
            buffer,
            bind_group,
            dlights: VecDeque::with_capacity(32),
        }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn spawn(&mut self, dlight: Dlight) {
        if self.dlights.len() >= MAX_DLIGHTS * 4 {
            // Évince — hard cap au cas où l'appelant oublie de prune.
            self.dlights.pop_front();
        }
        self.dlights.push_back(dlight);
    }

    pub fn prune(&mut self, now: f32) {
        self.dlights
            .retain(|d| (now - d.spawn_time) < d.lifetime);
    }

    pub fn clear(&mut self) {
        self.dlights.clear();
    }

    pub fn len(&self) -> usize {
        self.dlights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dlights.is_empty()
    }

    /// Écrit les dlights vivantes dans l'uniform buffer pour la frame
    /// courante.  Si plus de `MAX_DLIGHTS` sont actives, on garde les
    /// `MAX_DLIGHTS` plus récentes (ordre d'insertion).
    pub fn flush(&mut self, now: f32) {
        let mut gpu = GpuDlightBuffer::default();
        let mut i = 0;
        // Itère à l'envers pour préserver les plus récentes.
        for d in self.dlights.iter().rev() {
            if i >= MAX_DLIGHTS {
                break;
            }
            let Some(intensity) = d.intensity_at(now) else {
                continue;
            };
            gpu.lights[i] = GpuDlight {
                pos_radius: [d.center.x, d.center.y, d.center.z, d.radius],
                color_intensity: [d.color[0], d.color[1], d.color[2], intensity],
            };
            i += 1;
        }
        gpu.count = i as u32;
        self.queue
            .write_buffer(&self.buffer, 0, bytemuck::bytes_of(&gpu));
        if i > 0 {
            debug!("dlight: flush {i} actives");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intensity_fades_to_zero_at_end_of_lifetime() {
        let d = Dlight {
            center: Vec3::ZERO,
            radius: 200.0,
            color: [1.0, 0.8, 0.4],
            intensity: 2.0,
            spawn_time: 10.0,
            lifetime: 1.0,
        };
        // 0 % : pleine intensité
        assert_eq!(d.intensity_at(10.0), Some(2.0));
        // 50 % : toujours pleine (début du fade)
        assert!((d.intensity_at(10.5).unwrap() - 2.0).abs() < 1e-5);
        // 75 % : mi-fade, intensité = 1.0
        assert!((d.intensity_at(10.75).unwrap() - 1.0).abs() < 1e-5);
        // 100 % : expirée
        assert_eq!(d.intensity_at(11.0), None);
        // pré-spawn : None
        assert_eq!(d.intensity_at(9.0), None);
    }

    #[test]
    fn gpu_dlight_buffer_size_is_correct() {
        // Le layout doit être stable : 16 bytes d'en-tête (count + 3x u32
        // padding) + MAX_DLIGHTS * 32 bytes (2 × vec4).
        let expected = 16 + MAX_DLIGHTS * 32;
        assert_eq!(std::mem::size_of::<GpuDlightBuffer>(), expected);
    }

    #[test]
    fn pod_check_compiles() {
        // Si `Pod` est implémenté correctement, cast_slice doit marcher
        // sans UB (la vérification est statique via bytemuck).
        let buf = GpuDlightBuffer::default();
        let bytes: &[u8] = bytemuck::bytes_of(&buf);
        assert_eq!(bytes.len(), std::mem::size_of::<GpuDlightBuffer>());
    }
}
