//! Décales : marques de tir / explosion sur les surfaces du monde.
//!
//! Chaque décale = un quad orienté sur la surface impactée, dessiné en
//! alpha-blend avec masque circulaire procédural (pas de texture pour
//! l'instant — un vrai port des `gfx/damage/*` viendra plus tard).
//!
//! **Simplifications par rapport au Q3 original :**
//! * Pas de clipping polygonal contre la géométrie : on dessine un simple
//!   quad dans le plan tangent à la surface.  Les murs incurvés et les
//!   coins montrent une découpe rectangulaire plutôt qu'une vraie
//!   fragmentation surface-aligned.
//! * Pas de `CM_MarkFragments` : on fait confiance à la normale d'impact
//!   fournie par l'appelant (typiquement `trace.plane_normal`).
//! * Offset constant de 1u le long de la normale pour éviter le z-fight.

use bytemuck::{Pod, Zeroable};
use q3_math::Vec3;
use std::collections::VecDeque;
use std::sync::Arc;

use crate::DEPTH_FORMAT;

/// Capacité max du vbuf dynamique — 6 verts par décale, 1024 décales
/// simultanées → 6144 vertices.  En pratique on purge les décales
/// expirées chaque frame donc on ne s'en approche jamais.
const MAX_DECALS: usize = 1024;
const VERTS_PER_DECAL: usize = 6;
const MAX_VERTICES: u32 = (MAX_DECALS * VERTS_PER_DECAL) as u32;

/// Décale de référence : stocké tant qu'elle n'a pas expiré.
#[derive(Debug, Clone, Copy)]
pub struct Decal {
    /// Centre dans le monde.
    pub center: Vec3,
    /// Normale de la surface impactée (unit). Oriente le quad.
    pub normal: Vec3,
    /// Rayon du quad (la décale touche un carré `2*radius` de côté).
    pub radius: f32,
    /// Couleur RGBA de base — `a` se voit atténué par le fade temporel.
    pub color: [f32; 4],
    /// Timestamp de spawn (même horloge que l'appelant — secondes).
    pub spawn_time: f32,
    /// Durée de vie totale (s). La décale disparaît à `spawn_time + lifetime`.
    pub lifetime: f32,
}

impl Decal {
    /// Alpha résiduel à `now`.  Fade linéaire, dernière 25 % de la vie.
    /// Renvoie `None` si la décale a expiré.
    pub fn alpha_at(&self, now: f32) -> Option<f32> {
        let age = now - self.spawn_time;
        if age < 0.0 || age >= self.lifetime {
            return None;
        }
        let fade_start = self.lifetime * 0.75;
        let k = if age < fade_start {
            1.0
        } else {
            1.0 - (age - fade_start) / (self.lifetime - fade_start)
        };
        Some(self.color[3] * k.clamp(0.0, 1.0))
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DecalVertex {
    position: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

impl DecalVertex {
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

pub struct DecalRenderer {
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,

    /// Décales actives.  Bornée par `MAX_DECALS` — un spawn sur Vec full
    /// évince la plus ancienne (FIFO).  VecDeque : éviction O(1) en tête
    /// quand on est au cap (vs. O(n) avec Vec::remove(0)).
    decals: VecDeque<Decal>,

    /// Buffer CPU temporaire reconstruit chaque frame depuis `decals`.
    cpu: Vec<DecalVertex>,
}

impl DecalRenderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        camera_bgl: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("decal-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/decal.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("decal-pipeline-layout"),
            bind_group_layouts: &[camera_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("decal-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[DecalVertex::layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Pas de cull : on oriente les verts selon la normale
                // d'impact, mais on n'a pas besoin de vérifier la face.
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            // Depth test ON (une décale derrière un mur ne doit pas
            // apparaître), pas de depth write (une décale ne cache pas
            // une autre, et ne corromp pas le z-buffer pour les beams).
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
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
            label: Some("decal-vbuf"),
            size: (MAX_VERTICES as u64) * (std::mem::size_of::<DecalVertex>() as u64),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            queue,
            pipeline,
            vertex_buffer,
            decals: VecDeque::with_capacity(256),
            cpu: Vec::with_capacity(256 * VERTS_PER_DECAL),
        }
    }

    /// Spawne une nouvelle décale. Si la cap est atteinte, la plus
    /// ancienne est évincée.
    pub fn spawn(&mut self, decal: Decal) {
        if self.decals.len() >= MAX_DECALS {
            self.decals.pop_front();
        }
        self.decals.push_back(decal);
    }

    /// Purge les décales expirées.  À appeler après chaque tick (`now` en
    /// secondes, même horloge que `Decal::spawn_time`).
    pub fn prune(&mut self, now: f32) {
        self.decals
            .retain(|d| (now - d.spawn_time) < d.lifetime);
    }

    /// Vide toutes les décales — appelé lors d'un changement de map.
    pub fn clear(&mut self) {
        self.decals.clear();
    }

    pub fn len(&self) -> usize {
        self.decals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.decals.is_empty()
    }

    /// Flush : émet les quads vers la passe courante.  Le caller doit
    /// avoir bindé `camera_bind_group` au slot 0.
    pub fn flush<'a>(&'a mut self, pass: &mut wgpu::RenderPass<'a>, now: f32) {
        self.cpu.clear();
        for d in self.decals.iter() {
            let Some(alpha) = d.alpha_at(now) else {
                continue;
            };
            let color = [d.color[0], d.color[1], d.color[2], alpha];
            append_quad(&mut self.cpu, d.center, d.normal, d.radius, color);
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

/// Construit les 6 vertices (2 triangles CCW) d'un quad centré sur
/// `center`, dans le plan orthogonal à `normal`, de demi-côté `radius`.
/// Décalé de 1u le long de la normale pour éviter le z-fight.
fn append_quad(
    out: &mut Vec<DecalVertex>,
    center: Vec3,
    normal: Vec3,
    radius: f32,
    color: [f32; 4],
) {
    let n = normalize_or_up(normal);
    let (t, b) = tangent_frame(n);
    // 1u de push le long de la normale : décale devant la surface,
    // inaudible visuellement, suffisant pour éviter le z-fight sur du
    // Depth32Float standard.
    let c = center + n * 1.0;
    let t = t * radius;
    let b = b * radius;
    let p00 = c - t - b; // (0, 0)
    let p10 = c + t - b; // (1, 0)
    let p11 = c + t + b; // (1, 1)
    let p01 = c - t + b; // (0, 1)
    // Triangle 1 : p00, p10, p11
    out.push(v(p00, [0.0, 0.0], color));
    out.push(v(p10, [1.0, 0.0], color));
    out.push(v(p11, [1.0, 1.0], color));
    // Triangle 2 : p00, p11, p01
    out.push(v(p00, [0.0, 0.0], color));
    out.push(v(p11, [1.0, 1.0], color));
    out.push(v(p01, [0.0, 1.0], color));
}

fn v(pos: Vec3, uv: [f32; 2], color: [f32; 4]) -> DecalVertex {
    DecalVertex {
        position: pos.to_array(),
        uv,
        color,
    }
}

/// Renvoie la normale unit ou `(0,0,1)` si elle est dégénérée.
fn normalize_or_up(n: Vec3) -> Vec3 {
    let len_sq = n.dot(n);
    if len_sq < 1e-8 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        n / len_sq.sqrt()
    }
}

/// Construit deux vecteurs orthonormés `(t, b)` perpendiculaires à
/// `n` (unit).  Standard : on pick l'axe monde le moins aligné avec `n`,
/// on fait deux cross pour dériver `t` et `b`.
fn tangent_frame(n: Vec3) -> (Vec3, Vec3) {
    let axis = if n.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let t = axis.cross(n).normalize();
    let b = n.cross(t);
    (t, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_fades_to_zero_at_end() {
        let d = Decal {
            center: Vec3::ZERO,
            normal: Vec3::new(0.0, 0.0, 1.0),
            radius: 8.0,
            color: [1.0, 0.5, 0.0, 1.0],
            spawn_time: 0.0,
            lifetime: 4.0,
        };
        assert_eq!(d.alpha_at(0.0), Some(1.0));
        assert_eq!(d.alpha_at(2.0), Some(1.0));
        // À t=3 (fin de la fenêtre de plein alpha à 75 %), alpha=1.0 encore.
        assert!((d.alpha_at(3.0).unwrap() - 1.0).abs() < 1e-5);
        // À t=3.5, mi-fade : alpha = 0.5.
        assert!((d.alpha_at(3.5).unwrap() - 0.5).abs() < 1e-5);
        // À t=4, expiré.
        assert_eq!(d.alpha_at(4.0), None);
        assert_eq!(d.alpha_at(10.0), None);
    }

    #[test]
    fn tangent_frame_is_orthonormal_for_cardinal_normals() {
        for &n in &[
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        ] {
            let (t, b) = tangent_frame(n);
            assert!((t.dot(n)).abs() < 1e-5, "t ⊥ n pour n={n:?}, t={t:?}");
            assert!((b.dot(n)).abs() < 1e-5, "b ⊥ n pour n={n:?}, b={b:?}");
            assert!((t.dot(b)).abs() < 1e-5, "t ⊥ b pour n={n:?}");
            assert!((t.length() - 1.0).abs() < 1e-5);
            assert!((b.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn append_quad_produces_six_coplanar_verts() {
        let mut out = Vec::new();
        let center = Vec3::new(10.0, 20.0, 30.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        append_quad(&mut out, center, normal, 8.0, [1.0; 4]);
        assert_eq!(out.len(), 6);
        // Tous offsets le long de la normale : z = 30 + 1 = 31.
        for v in &out {
            assert!((v.position[2] - 31.0).abs() < 1e-4);
        }
    }

    #[test]
    fn prune_removes_expired_decals() {
        // On ne peut pas construire un DecalRenderer sans GPU device.
        // Test la logique en isolation sur un Vec<Decal>.
        let mut decals = vec![
            Decal {
                center: Vec3::ZERO,
                normal: Vec3::new(0.0, 0.0, 1.0),
                radius: 4.0,
                color: [1.0; 4],
                spawn_time: 0.0,
                lifetime: 1.0,
            },
            Decal {
                center: Vec3::ZERO,
                normal: Vec3::new(0.0, 0.0, 1.0),
                radius: 4.0,
                color: [1.0; 4],
                spawn_time: 5.0,
                lifetime: 2.0,
            },
        ];
        let now = 6.0;
        decals.retain(|d| (now - d.spawn_time) < d.lifetime);
        assert_eq!(decals.len(), 1);
        assert_eq!(decals[0].spawn_time, 5.0);
    }
}
