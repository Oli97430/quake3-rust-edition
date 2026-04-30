//! Particules billboard — petits quads caméra-facing pour les puffs de
//! fumée d'explosion, éclats de gunspark, etc.
//!
//! Chaque particule est avancée CPU-side à chaque flush :
//! `pos(t) = pos0 + velocity * age`.  Elle grossit de `size_start` à
//! `size_end` linéairement sur sa durée de vie, et son alpha fondue de
//! `color.a` à 0 sur les 50 % finaux.
//!
//! **Différences avec le système Q3 original :**
//! * Pas d'accélération (gravité / friction) — simple translation constante.
//!   Suffisant pour des puffs courte durée ; les effets à gravité viendront
//!   quand le système se généralisera aux « gibs » et douilles.
//! * Billboard parfait écran-aligné — pas d'« oriented billboards »
//!   (rectangle orienté le long d'une direction comme pour les trails).
//!
//! Le caller construit des [`Particle`] et appelle [`ParticleRenderer::spawn`].
//! Le renderer purge les particules expirées chaque frame (via `prune`) et
//! reconstruit le vertex buffer dans `flush`.

use bytemuck::{Pod, Zeroable};
use q3_math::Vec3;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::DEPTH_FORMAT;

/// Capacité max — 6 verts par particule, 2048 particules → 12288 verts.
/// La pruning est agressive donc on ne s'en approche jamais en pratique.
const MAX_PARTICLES: usize = 2048;
const VERTS_PER_PARTICLE: usize = 6;
const MAX_VERTICES: u32 = (MAX_PARTICLES * VERTS_PER_PARTICLE) as u32;

#[derive(Debug, Clone, Copy)]
pub struct Particle {
    /// Position monde au spawn.
    pub pos: Vec3,
    /// Vitesse monde (unités/s) — reste constante sur la vie de la particule.
    pub velocity: Vec3,
    /// Couleur RGBA de base. `a` est l'alpha pic (fade multiplie dessus).
    pub color: [f32; 4],
    /// Taille initiale du quad (demi-côté, unités Q3).
    pub size_start: f32,
    /// Taille finale (en fin de vie). Typiquement > `size_start` pour
    /// simuler l'expansion d'un nuage de fumée.
    pub size_end: f32,
    /// Timestamp de spawn (secondes, horloge applicative).
    pub spawn_time: f32,
    /// Durée de vie (s). La particule disparaît à `spawn_time + lifetime`.
    pub lifetime: f32,
}

impl Particle {
    /// État interpolé à `now` : `(pos, size, alpha)`.  `None` si expirée.
    pub fn sample(&self, now: f32) -> Option<(Vec3, f32, f32)> {
        let age = now - self.spawn_time;
        if age < 0.0 || age >= self.lifetime {
            return None;
        }
        let t = (age / self.lifetime).clamp(0.0, 1.0);
        let pos = self.pos + self.velocity * age;
        let size = self.size_start + (self.size_end - self.size_start) * t;
        // Fade linéaire sur la seconde moitié de la vie (1.0 jusqu'à
        // t=0.5, puis décroit linéairement vers 0).
        let k = if t < 0.5 { 1.0 } else { 1.0 - (t - 0.5) * 2.0 };
        let alpha = self.color[3] * k.clamp(0.0, 1.0);
        Some((pos, size, alpha))
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ParticleVertex {
    position: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

impl ParticleVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2,
        2 => Float32x4,
    ];
    const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct ParticleRenderer {
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,

    particles: VecDeque<Particle>,
    cpu: Vec<ParticleVertex>,
}

impl ParticleRenderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        camera_bgl: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/particle.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle-pipeline-layout"),
            bind_group_layouts: &[camera_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[ParticleVertex::layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Pas de cull — billboards bifaciales.
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                // Depth-test ON (occlusion par murs) mais pas de depth-write
                // (les particules ne s'occludent pas entre elles — on accepte
                // le léger « popping » en échange d'un alpha-blending propre).
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle-vbuf"),
            size: (MAX_VERTICES as u64) * (std::mem::size_of::<ParticleVertex>() as u64),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            queue,
            pipeline,
            vertex_buffer,
            particles: VecDeque::with_capacity(512),
            cpu: Vec::with_capacity(512 * VERTS_PER_PARTICLE),
        }
    }

    pub fn spawn(&mut self, p: Particle) {
        if self.particles.len() >= MAX_PARTICLES {
            self.particles.pop_front();
        }
        self.particles.push_back(p);
    }

    pub fn prune(&mut self, now: f32) {
        self.particles
            .retain(|p| (now - p.spawn_time) < p.lifetime);
    }

    pub fn clear(&mut self) {
        self.particles.clear();
    }

    pub fn len(&self) -> usize {
        self.particles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Flush : émet les quads billboard vers la passe courante. Le caller
    /// doit avoir bindé `camera_bind_group` au slot 0.
    /// `cam_right` et `cam_up` sont les axes caméra monde (unit) utilisés
    /// pour orienter les billboards — typiquement `Angles::to_vectors()`.
    pub fn flush<'a>(
        &'a mut self,
        pass: &mut wgpu::RenderPass<'a>,
        now: f32,
        cam_right: Vec3,
        cam_up: Vec3,
    ) {
        self.cpu.clear();
        for p in self.particles.iter() {
            let Some((pos, size, alpha)) = p.sample(now) else {
                continue;
            };
            let color = [p.color[0], p.color[1], p.color[2], alpha];
            append_billboard(&mut self.cpu, pos, size, cam_right, cam_up, color);
        }
        if self.cpu.is_empty() {
            return;
        }
        let max = MAX_VERTICES as usize;
        if self.cpu.len() > max {
            self.cpu.truncate(max);
        }
        self.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.cpu));
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..self.cpu.len() as u32, 0..1);
    }
}

/// Construit les 6 vertices d'un quad caméra-aligned centré sur `center`,
/// de demi-côté `size`, orienté selon (`right`, `up`).
fn append_billboard(
    out: &mut Vec<ParticleVertex>,
    center: Vec3,
    size: f32,
    right: Vec3,
    up: Vec3,
    color: [f32; 4],
) {
    let r = right * size;
    let u = up * size;
    let p00 = center - r - u; // (0, 0)
    let p10 = center + r - u; // (1, 0)
    let p11 = center + r + u; // (1, 1)
    let p01 = center - r + u; // (0, 1)
    out.push(v(p00, [0.0, 0.0], color));
    out.push(v(p10, [1.0, 0.0], color));
    out.push(v(p11, [1.0, 1.0], color));
    out.push(v(p00, [0.0, 0.0], color));
    out.push(v(p11, [1.0, 1.0], color));
    out.push(v(p01, [0.0, 1.0], color));
}

fn v(pos: Vec3, uv: [f32; 2], color: [f32; 4]) -> ParticleVertex {
    ParticleVertex {
        position: pos.to_array(),
        uv,
        color,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy(pos: Vec3, vel: Vec3, spawn: f32, life: f32) -> Particle {
        Particle {
            pos,
            velocity: vel,
            color: [1.0, 1.0, 1.0, 1.0],
            size_start: 1.0,
            size_end: 4.0,
            spawn_time: spawn,
            lifetime: life,
        }
    }

    #[test]
    fn sample_returns_none_outside_lifetime() {
        let p = dummy(Vec3::ZERO, Vec3::ZERO, 2.0, 1.0);
        assert!(p.sample(1.0).is_none());
        assert!(p.sample(3.0).is_none());
        assert!(p.sample(10.0).is_none());
    }

    #[test]
    fn sample_interpolates_position_and_size() {
        let p = dummy(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0), 0.0, 2.0);
        // À t=1 (mi-vie) : position = (10, 0, 0), size = 2.5, alpha = 1.0.
        let (pos, size, alpha) = p.sample(1.0).unwrap();
        assert!((pos.x - 10.0).abs() < 1e-5);
        assert!((size - 2.5).abs() < 1e-5);
        assert!((alpha - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sample_alpha_fades_on_second_half() {
        let p = dummy(Vec3::ZERO, Vec3::ZERO, 0.0, 4.0);
        // t = 0 → 1.0, t = 0.5 → 1.0, t = 0.75 → 0.5, t = 1.0 → expiré.
        assert!((p.sample(0.0).unwrap().2 - 1.0).abs() < 1e-5);
        assert!((p.sample(2.0).unwrap().2 - 1.0).abs() < 1e-5);
        assert!((p.sample(3.0).unwrap().2 - 0.5).abs() < 1e-5);
        assert!(p.sample(4.0).is_none());
    }

    #[test]
    fn append_billboard_produces_six_verts_in_plane() {
        let mut out = Vec::new();
        let center = Vec3::new(5.0, 6.0, 7.0);
        let right = Vec3::new(1.0, 0.0, 0.0);
        let up = Vec3::new(0.0, 0.0, 1.0);
        append_billboard(&mut out, center, 2.0, right, up, [1.0; 4]);
        assert_eq!(out.len(), 6);
        // Tous les points restent sur le plan y = 6 (car right/up n'ont pas
        // de composante y).
        for v in &out {
            assert!((v.position[1] - 6.0).abs() < 1e-5);
        }
    }

    #[test]
    fn prune_removes_expired() {
        let mut v = vec![
            dummy(Vec3::ZERO, Vec3::ZERO, 0.0, 1.0),
            dummy(Vec3::ZERO, Vec3::ZERO, 5.0, 2.0),
        ];
        let now = 6.0;
        v.retain(|p| (now - p.spawn_time) < p.lifetime);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].spawn_time, 5.0);
    }
}
