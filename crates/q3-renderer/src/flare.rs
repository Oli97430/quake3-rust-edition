//! Flares / coronas BSP — billboards additifs sur les sources lumineuses
//! embarquées dans la map.
//!
//! Q3 encode les flares comme des surfaces spéciales (`SurfaceType::Flare`)
//! qui ne contiennent ni vertices ni indices mais utilisent trois champs
//! du header de surface de façon détournée :
//!
//! | Champ              | Usage flare                      |
//! |--------------------|----------------------------------|
//! | `lightmap_origin`  | position monde du flare          |
//! | `lightmap_vecs[0]` | couleur RGB                      |
//! | `lightmap_vecs[2]` | normale (direction face caméra)  |
//!
//! On extrait la liste une fois au chargement de la map et on l'uploade
//! dans un vbuf statique.  À chaque frame, la CPU reconstruit les 6
//! vertices par flare en fonction de l'orientation caméra (même technique
//! que le renderer de particules).  Blending additif, depth-test mais
//! pas de depth-write — un flare masqué par un mur est bien occludé mais
//! il ne bloque pas ce qui est devant lui.

use bytemuck::{Pod, Zeroable};
use q3_bsp::{raw::SurfaceType, Bsp};
use q3_math::Vec3;
use std::sync::Arc;
use tracing::debug;

use crate::DEPTH_FORMAT;

/// Capacité max — on cap à 512 flares par map (les plus grosses Q3 en
/// ont ~50, le cap laisse beaucoup de marge).  6 verts par flare.
const MAX_FLARES: usize = 512;
const VERTS_PER_FLARE: usize = 6;
const MAX_VERTICES: u32 = (MAX_FLARES * VERTS_PER_FLARE) as u32;

/// Flare statique extrait d'une surface BSP.
#[derive(Debug, Clone, Copy)]
pub struct Flare {
    /// Centre monde du halo.
    pub origin: Vec3,
    /// Direction de visibilité — le flare fade quand la caméra s'en
    /// écarte latéralement.  (`Vec3::ZERO` = omnidirectionnel.)
    pub normal: Vec3,
    /// Couleur RGB (linéaire).
    pub color: [f32; 3],
}

impl Flare {
    /// Extrait tous les flares d'un BSP.  Gère les valeurs aberrantes
    /// (couleur nulle → blanc par défaut, normale nulle = omnidirectionnel).
    pub fn extract_from(bsp: &Bsp) -> Vec<Self> {
        let mut out = Vec::new();
        for surf in bsp.surfaces.iter() {
            if surf.kind() != SurfaceType::Flare {
                continue;
            }
            let origin = Vec3::from(surf.lightmap_origin);
            let mut color = surf.lightmap_vecs[0];
            // Certaines maps encodent (0,0,0) comme "pas de couleur spécifiée"
            // → on retombe sur un blanc chaud plutôt qu'un flare noir.
            if color[0] + color[1] + color[2] < 1e-3 {
                color = [1.0, 0.9, 0.7];
            }
            let normal = Vec3::from(surf.lightmap_vecs[2]);
            out.push(Flare {
                origin,
                normal,
                color,
            });
            if out.len() >= MAX_FLARES {
                break;
            }
        }
        out
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FlareVertex {
    position: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

impl FlareVertex {
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

/// Sprite transitoire ajouté par l'engine pendant la frame (muzzle flash,
/// explosion pop, etc.).  Partage le pipeline additif des flares mais
/// vit une seule frame et porte son propre rayon + couleur + alpha.
#[derive(Debug, Clone, Copy)]
struct DynamicSprite {
    origin: Vec3,
    color: [f32; 4],
    radius: f32,
}

pub struct FlareRenderer {
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,

    flares: Vec<Flare>,
    dynamic: Vec<DynamicSprite>,
    cpu: Vec<FlareVertex>,
    /// Rayon monde du quad (demi-côté) — commun à tous les flares BSP.
    radius: f32,
}

impl FlareRenderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        camera_bgl: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flare-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/flare.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("flare-pipeline-layout"),
            bind_group_layouts: &[camera_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("flare-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[FlareVertex::layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
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
                    // Blending additif : un corona éclaire toujours sans
                    // obscurcir la source de lumière sous-jacente.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flare-vbuf"),
            size: (MAX_VERTICES as u64) * (std::mem::size_of::<FlareVertex>() as u64),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            queue,
            pipeline,
            vertex_buffer,
            flares: Vec::new(),
            dynamic: Vec::new(),
            cpu: Vec::new(),
            // 16u ≈ 0.4m monde Q3 : corona visible mais pas envahissante.
            radius: 16.0,
        }
    }

    /// Ajoute un billboard additif transitoire — typique muzzle flash, pop
    /// d'explosion, éclat plasma.  Les entrées sont consommées et vidées
    /// à chaque `flush` : l'engine doit republier chaque frame tant que
    /// l'effet doit rester visible.  `color` est multiplié par l'intensité
    /// radiale du shader (cœur + halo) ; un color=(1.0, 0.85, 0.3) donne
    /// une lueur chaude type poudre.
    pub fn push_dynamic(&mut self, origin: Vec3, color: [f32; 4], radius: f32) {
        if radius <= 0.0 {
            return;
        }
        if self.dynamic.len() + self.flares.len() >= MAX_FLARES {
            // Cap global — le pipeline ne peut pas dépasser la vbuf taille.
            // Les sprites transitoires cèdent la place aux coronas BSP, pas
            // l'inverse : on ignore silencieusement au-delà.
            return;
        }
        self.dynamic.push(DynamicSprite { origin, color, radius });
    }

    /// Vidange manuelle du buffer transitoire — normalement pas nécessaire
    /// puisque `flush` vide automatiquement, mais utile si on skip la passe
    /// flare pour une raison (ex: pause du rendu monde).
    pub fn clear_dynamic(&mut self) {
        self.dynamic.clear();
    }

    /// Remplace la liste active de flares (typiquement après `upload_bsp`).
    pub fn set_flares(&mut self, flares: Vec<Flare>) {
        debug!("flare: {} coronas actifs", flares.len());
        self.flares = flares;
    }

    pub fn clear(&mut self) {
        self.flares.clear();
    }

    pub fn len(&self) -> usize {
        self.flares.len()
    }

    pub fn is_empty(&self) -> bool {
        self.flares.is_empty() && self.dynamic.is_empty()
    }

    /// Flush additif.  `cam_right` et `cam_up` sont les axes caméra monde
    /// utilisés pour orienter les billboards.  `eye` sert au facteur de
    /// visibilité (cos entre l'axe eye→flare et la normale du flare).
    pub fn flush<'a>(
        &'a mut self,
        pass: &mut wgpu::RenderPass<'a>,
        eye: Vec3,
        cam_right: Vec3,
        cam_up: Vec3,
    ) {
        self.cpu.clear();
        for f in self.flares.iter() {
            let vis = visibility_factor(eye, f.origin, f.normal);
            if vis <= 0.01 {
                continue;
            }
            let color = [f.color[0], f.color[1], f.color[2], vis];
            append_billboard(&mut self.cpu, f.origin, self.radius, cam_right, cam_up, color);
        }
        // Sprites transitoires : full-bright (pas de `visibility_factor`,
        // ils sont par nature omnidirectionnels — une muzzle flash se voit
        // sous n'importe quel angle).  L'alpha vient directement de la
        // couleur fournie par l'engine (qui peut donc baisser l'alpha en
        // fin de vie pour un fade-out manuel).
        for s in self.dynamic.iter() {
            append_billboard(
                &mut self.cpu,
                s.origin,
                s.radius,
                cam_right,
                cam_up,
                s.color,
            );
        }
        // Vidange les transitoires maintenant qu'ils sont en GPU — la
        // prochaine frame devra les republier s'ils sont toujours actifs.
        self.dynamic.clear();
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

/// Facteur `[0,1]` de visibilité : 1 quand la caméra est parfaitement
/// dans l'axe du flare, 0 quand elle est perpendiculaire ou derrière.
/// Si la normale est nulle, le flare est omnidirectionnel (toujours 1).
fn visibility_factor(eye: Vec3, origin: Vec3, normal: Vec3) -> f32 {
    let len_sq = normal.dot(normal);
    if len_sq < 1e-4 {
        return 1.0;
    }
    let to_eye = eye - origin;
    let d = to_eye.length();
    if d < 1e-4 {
        return 1.0;
    }
    let n = normal / len_sq.sqrt();
    let cos = n.dot(to_eye / d);
    // Fade doux via smoothstep(0, 1) de `cos` — pas de cut-off dur.
    cos.clamp(0.0, 1.0)
}

fn append_billboard(
    out: &mut Vec<FlareVertex>,
    center: Vec3,
    size: f32,
    right: Vec3,
    up: Vec3,
    color: [f32; 4],
) {
    let r = right * size;
    let u = up * size;
    let p00 = center - r - u;
    let p10 = center + r - u;
    let p11 = center + r + u;
    let p01 = center - r + u;
    out.push(v(p00, [0.0, 0.0], color));
    out.push(v(p10, [1.0, 0.0], color));
    out.push(v(p11, [1.0, 1.0], color));
    out.push(v(p00, [0.0, 0.0], color));
    out.push(v(p11, [1.0, 1.0], color));
    out.push(v(p01, [0.0, 1.0], color));
}

fn v(pos: Vec3, uv: [f32; 2], color: [f32; 4]) -> FlareVertex {
    FlareVertex {
        position: pos.to_array(),
        uv,
        color,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visibility_is_one_for_omnidirectional() {
        let eye = Vec3::new(100.0, 0.0, 0.0);
        let v = visibility_factor(eye, Vec3::ZERO, Vec3::ZERO);
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn visibility_peaks_on_axis_and_zeros_behind() {
        let origin = Vec3::ZERO;
        let normal = Vec3::new(0.0, 0.0, 1.0);
        // Eye pile au-dessus (dans l'axe de la normale) → 1.
        let on = visibility_factor(Vec3::new(0.0, 0.0, 100.0), origin, normal);
        assert!((on - 1.0).abs() < 1e-5);
        // Eye pile en-dessous (opposé à la normale) → 0.
        let behind = visibility_factor(Vec3::new(0.0, 0.0, -100.0), origin, normal);
        assert_eq!(behind, 0.0);
        // Eye sur le côté (perpendiculaire) → 0.
        let side = visibility_factor(Vec3::new(100.0, 0.0, 0.0), origin, normal);
        assert!(side.abs() < 1e-5);
    }

    #[test]
    fn append_billboard_emits_six_verts() {
        let mut out = Vec::new();
        append_billboard(
            &mut out,
            Vec3::new(1.0, 2.0, 3.0),
            4.0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            [1.0; 4],
        );
        assert_eq!(out.len(), 6);
        // Tous les points ont z = 3 (right/up ne touchent pas z).
        for v in &out {
            assert!((v.position[2] - 3.0).abs() < 1e-5);
        }
    }
}
