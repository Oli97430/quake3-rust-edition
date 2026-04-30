//! Rendu de modèles **MD3** — animés (interpolation inter-frames).
//!
//! * Une passe dédiée après le monde opaque (queue *world*).
//! * Une passe *overlay* optionnelle (viewmodel) avec depth rechargée à 1.0
//!   pour éviter que le viewmodel soit clippé par la géométrie du niveau.
//! * Un uniform par instance : model matrix + tint + lerp factor (x), le reste
//!   du `vec4` sert de padding pour l'alignement 16 octets.
//! * Le matériau diffuse vient du `MaterialCache` existant : on réutilise
//!   les shaders scripts Q3 (ex : `models/weapons2/rocketl/rocketl.jpg`).
//!
//! Tous les frames d'une surface sont concaténés dans un **unique** vertex
//! buffer. On passe le même buffer deux fois au pipeline (slots 0 et 1) avec
//! des *offsets* différents — slot 0 = frame_a, slot 1 = frame_b.

use crate::{material::MaterialCache, DEPTH_FORMAT};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use hashbrown::HashMap;
use q3_common::{Error, Result};
use q3_filesystem::Vfs;
use q3_math::Vec3;
use q3_model::{Md3, Surface, MD3_XYZ_SCALE};
use std::sync::Arc;
use tracing::debug;
use wgpu::util::DeviceExt;

/// Position + normale d'un vertex pour une frame donnée.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Md3FrameVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl Md3FrameVertex {
    const ATTRIBS_A: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3, // pos_a
        1 => Float32x3, // normal_a
    ];
    const ATTRIBS_B: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        2 => Float32x3, // pos_b
        3 => Float32x3, // normal_b
    ];

    pub const fn layout_a() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS_A,
        }
    }
    pub const fn layout_b() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS_B,
        }
    }
}

/// UV constant (indépendant de la frame).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Md3Uv {
    pub uv: [f32; 2],
}

impl Md3Uv {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![
        4 => Float32x2,
    ];
    pub const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ModelUniform {
    model: [[f32; 4]; 4],
    color: [f32; 4],
    /// x = lerp factor [0..1]
    /// y = viewmodel flag (0 = monde/bot, 1 = viewmodel → shading boosté)
    /// zw = padding (alignement 16 octets).
    lerp_pad: [f32; 4],
}

/// Sous-mesh d'un MD3 (une surface = un shader).
pub struct Md3SurfaceGpu {
    /// Toutes les frames concaténées : `[frame0 verts | frame1 verts | …]`.
    pub frames_buffer: wgpu::Buffer,
    /// `num_verts * size_of::<Md3FrameVertex>()` — stride d'une frame.
    pub frame_stride: u64,
    pub uv_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub num_frames: usize,
    pub num_verts: usize,
    pub shader_name: String,
}

/// Un MD3 complet sur GPU + sa copie CPU (pour query les tags).
pub struct Md3Gpu {
    pub surfaces: Vec<Md3SurfaceGpu>,
    /// Copie CPU du modèle parsé — contient frames / tags / num_tags.
    pub model: Arc<Md3>,
    /// Nom logique (chemin VFS) — debug / cache key.
    pub path: String,
}

impl Md3Gpu {
    /// Nombre de frames d'animation disponibles pour ce modèle.
    pub fn num_frames(&self) -> usize {
        self.model.frames.len()
    }

    /// Calcule la transformation locale d'un tag, interpolée entre deux
    /// frames. Délègue à `Md3::tag_transform`.
    pub fn tag_transform(
        &self,
        frame_a: usize,
        frame_b: usize,
        lerp: f32,
        name: &str,
    ) -> Option<Mat4> {
        self.model.tag_transform(frame_a, frame_b, lerp, name)
    }
}

/// Une instance à dessiner cette frame.
struct Md3Instance {
    mesh: Arc<Md3Gpu>,
    transform: Mat4,
    color: [f32; 4],
    frame_a: usize,
    frame_b: usize,
    lerp: f32,
}

/// Cache + pipeline + deux queues (world + overlay/viewmodel).
pub struct Md3Renderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    /// Bind group layout pour l'uniform "model + color + lerp" (group 1).
    model_bgl: wgpu::BindGroupLayout,
    by_path: HashMap<String, Arc<Md3Gpu>>,

    /// Instances "monde" (pickups, persos).
    world_instances: Vec<Md3Instance>,
    /// Instances "overlay" (viewmodel) — rendues après un depth clear.
    overlay_instances: Vec<Md3Instance>,

    /// Pool d'uniforms partagé entre les deux queues : indices
    /// `0..world_instances.len()` = world, puis overlay.
    uniform_buffers: Vec<wgpu::Buffer>,
    uniform_bgs: Vec<wgpu::BindGroup>,
}

impl Md3Renderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        camera_bgl: &wgpu::BindGroupLayout,
        material_bgl: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
    ) -> Self {
        let model_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("md3-model-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("md3-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/model.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("md3-pipeline-layout"),
            bind_group_layouts: &[camera_bgl, &model_bgl, material_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("md3-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[
                    Md3FrameVertex::layout_a(),
                    Md3FrameVertex::layout_b(),
                    Md3Uv::layout(),
                ],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                // MD3 triangles sont CCW une fois les vertices lus selon la
                // convention du format. Historiquement on déclarait `Cw` parce
                // que notre ancienne `view_matrix` avait det=-1 (miroir
                // horizontal), ce qui inversait le sens de parcours en écran.
                // Depuis la correction de `camera.rs` (col. LEFT = -basis.right),
                // la view est orthogonale propre (det=+1) : on s'aligne donc
                // sur la convention CCW partagée avec toutes les autres
                // pipelines (monde, décals, particules, sky, beam, text…).
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            model_bgl,
            by_path: HashMap::new(),
            world_instances: Vec::new(),
            overlay_instances: Vec::new(),
            uniform_buffers: Vec::new(),
            uniform_bgs: Vec::new(),
        }
    }

    /// Charge un MD3 depuis le VFS (cached). Le chemin sert de clé.
    ///
    /// Si un fichier `_default.skin` existe à côté (convention Q3 pour les
    /// modèles de joueur : `lower.md3` → `lower_default.skin`), ses mappings
    /// `<surface>,<texture>` sont appliqués par dessus les shaders embarqués
    /// dans le MD3. C'est indispensable pour les player models dont les
    /// `shader_name` internes sont **vides** — sans ça, les bots tombent sur
    /// la texture fallback (damier rose/noir).
    pub fn load(&mut self, vfs: &Vfs, path: &str) -> Result<Arc<Md3Gpu>> {
        let key = path.to_ascii_lowercase();
        if let Some(m) = self.by_path.get(&key) {
            return Ok(m.clone());
        }
        let bytes = vfs
            .read(path)
            .map_err(|e| Error::renderer(format!("md3 read {path}: {e}")))?;
        let md3 = Arc::new(Md3::parse(&bytes)?);
        debug!(
            "md3: {} surfaces / {} frames / {} tags pour `{}`",
            md3.surfaces.len(),
            md3.frames.len(),
            md3.num_tags,
            path
        );

        // Lookup optionnel du .skin companion.
        let skin_map = load_default_skin(vfs, path);

        let mut surfaces = Vec::with_capacity(md3.surfaces.len());
        for surf in &md3.surfaces {
            let mut gpu_surf = self.upload_surface(surf);
            // Override via le .skin si une entrée matche ce surface name.
            if let Some(map) = skin_map.as_ref() {
                if let Some(tex) = map.get(&surf.name.to_ascii_lowercase()) {
                    debug!(
                        "md3 skin override: `{}` surf `{}` → `{}`",
                        path, surf.name, tex
                    );
                    gpu_surf.shader_name = tex.clone();
                }
            }
            surfaces.push(gpu_surf);
        }
        let gpu = Arc::new(Md3Gpu {
            surfaces,
            model: md3,
            path: path.to_string(),
        });
        self.by_path.insert(key, gpu.clone());
        Ok(gpu)
    }

    fn upload_surface(&self, surf: &Surface) -> Md3SurfaceGpu {
        // Toutes les frames concaténées dans un unique Vec.
        let total_verts = surf.num_frames * surf.num_verts;
        let mut verts: Vec<Md3FrameVertex> = Vec::with_capacity(total_verts);
        for f in 0..surf.num_frames {
            for v in 0..surf.num_verts {
                let raw = &surf.xyz_normals[f * surf.num_verts + v];
                let pos = [
                    raw.xyz[0] as f32 * MD3_XYZ_SCALE,
                    raw.xyz[1] as f32 * MD3_XYZ_SCALE,
                    raw.xyz[2] as f32 * MD3_XYZ_SCALE,
                ];
                let n = decode_normal(raw.normal);
                verts.push(Md3FrameVertex {
                    position: pos,
                    normal: [n.x, n.y, n.z],
                });
            }
        }
        let frame_stride =
            (surf.num_verts * std::mem::size_of::<Md3FrameVertex>()) as u64;

        let uvs: Vec<Md3Uv> = surf.uvs.iter().map(|uv| Md3Uv { uv: *uv }).collect();

        let mut indices: Vec<u32> = Vec::with_capacity(surf.triangles.len() * 3);
        for tri in &surf.triangles {
            indices.extend_from_slice(tri);
        }

        let frames_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("md3-frames"),
                contents: bytemuck::cast_slice(&verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let uv_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("md3-uvs"),
                contents: bytemuck::cast_slice(&uvs),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("md3-ibuf"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let shader_name = surf.shaders.first().cloned().unwrap_or_default();
        Md3SurfaceGpu {
            frames_buffer,
            frame_stride,
            uv_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            num_frames: surf.num_frames,
            num_verts: surf.num_verts,
            shader_name,
        }
    }

    /// Queue une instance "monde" (pickup, perso, décor).
    pub fn queue_world(
        &mut self,
        mesh: Arc<Md3Gpu>,
        transform: Mat4,
        color: [f32; 4],
        frame_a: usize,
        frame_b: usize,
        lerp: f32,
    ) {
        self.world_instances.push(Md3Instance {
            mesh,
            transform,
            color,
            frame_a,
            frame_b,
            lerp,
        });
    }

    /// Queue une instance "overlay" (viewmodel). Rendu dans un second pass
    /// avec la depth rechargée à 1.0 pour ne pas être clippé par le niveau.
    pub fn queue_overlay(
        &mut self,
        mesh: Arc<Md3Gpu>,
        transform: Mat4,
        color: [f32; 4],
        frame_a: usize,
        frame_b: usize,
        lerp: f32,
    ) {
        self.overlay_instances.push(Md3Instance {
            mesh,
            transform,
            color,
            frame_a,
            frame_b,
            lerp,
        });
    }

    pub fn begin_frame(&mut self) {
        self.world_instances.clear();
        self.overlay_instances.clear();
    }

    pub fn world_count(&self) -> usize {
        self.world_instances.len()
    }

    pub fn overlay_count(&self) -> usize {
        self.overlay_instances.len()
    }

    /// Écrit les uniforms et dessine les instances *world*.
    pub fn flush_world<'a>(
        &'a mut self,
        pass: &mut wgpu::RenderPass<'a>,
        mats: &mut MaterialCache,
    ) {
        if self.world_instances.is_empty() {
            return;
        }
        self.ensure_uniform_capacity(self.world_instances.len() + self.overlay_instances.len());
        Self::write_uniforms(&self.queue, &self.uniform_buffers, &self.world_instances, 0);
        Self::flush_range(
            pass,
            mats,
            &self.pipeline,
            &self.world_instances,
            &self.uniform_bgs,
            0,
        );
    }

    /// Écrit les uniforms et dessine les instances *overlay* (viewmodel).
    /// L'appelant doit fournir un pass avec depth rechargée à 1.0.
    pub fn flush_overlay<'a>(
        &'a mut self,
        pass: &mut wgpu::RenderPass<'a>,
        mats: &mut MaterialCache,
    ) {
        if self.overlay_instances.is_empty() {
            return;
        }
        self.ensure_uniform_capacity(self.world_instances.len() + self.overlay_instances.len());
        let base = self.world_instances.len();
        // Flag viewmodel = 1.0 → le shader applique rim + specular + gain
        // supplémentaire pour faire ressortir l'arme tenue en main.
        Self::write_uniforms_flag(
            &self.queue,
            &self.uniform_buffers,
            &self.overlay_instances,
            base,
            1.0,
        );
        Self::flush_range(
            pass,
            mats,
            &self.pipeline,
            &self.overlay_instances,
            &self.uniform_bgs,
            base,
        );
    }

    fn write_uniforms(
        queue: &wgpu::Queue,
        uniform_buffers: &[wgpu::Buffer],
        instances: &[Md3Instance],
        base: usize,
    ) {
        Self::write_uniforms_flag(queue, uniform_buffers, instances, base, 0.0);
    }

    fn write_uniforms_flag(
        queue: &wgpu::Queue,
        uniform_buffers: &[wgpu::Buffer],
        instances: &[Md3Instance],
        base: usize,
        viewmodel_flag: f32,
    ) {
        for (i, inst) in instances.iter().enumerate() {
            let u = ModelUniform {
                model: inst.transform.to_cols_array_2d(),
                color: inst.color,
                lerp_pad: [inst.lerp.clamp(0.0, 1.0), viewmodel_flag, 0.0, 0.0],
            };
            queue.write_buffer(&uniform_buffers[base + i], 0, bytemuck::bytes_of(&u));
        }
    }

    fn flush_range<'a>(
        pass: &mut wgpu::RenderPass<'a>,
        mats: &mut MaterialCache,
        pipeline: &'a wgpu::RenderPipeline,
        instances: &'a [Md3Instance],
        uniform_bgs: &'a [wgpu::BindGroup],
        base: usize,
    ) {
        pass.set_pipeline(pipeline);
        for (i, inst) in instances.iter().enumerate() {
            pass.set_bind_group(1, &uniform_bgs[base + i], &[]);
            for surf in &inst.mesh.surfaces {
                let mat = mats.resolve(&surf.shader_name);
                pass.set_bind_group(2, &mat.bind_group, &[]);
                let fa = inst.frame_a.min(surf.num_frames.saturating_sub(1));
                let fb = inst.frame_b.min(surf.num_frames.saturating_sub(1));
                let off_a = (fa as u64) * surf.frame_stride;
                let off_b = (fb as u64) * surf.frame_stride;
                pass.set_vertex_buffer(
                    0,
                    surf.frames_buffer.slice(off_a..off_a + surf.frame_stride),
                );
                pass.set_vertex_buffer(
                    1,
                    surf.frames_buffer.slice(off_b..off_b + surf.frame_stride),
                );
                pass.set_vertex_buffer(2, surf.uv_buffer.slice(..));
                pass.set_index_buffer(surf.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..surf.index_count, 0, 0..1);
            }
        }
    }

    fn ensure_uniform_capacity(&mut self, n: usize) {
        while self.uniform_buffers.len() < n {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("md3-uniform"),
                size: std::mem::size_of::<ModelUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("md3-uniform-bg"),
                layout: &self.model_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            self.uniform_buffers.push(buf);
            self.uniform_bgs.push(bg);
        }
    }

    pub fn cached_count(&self) -> usize {
        self.by_path.len()
    }

    pub fn instance_count(&self) -> usize {
        self.world_instances.len() + self.overlay_instances.len()
    }
}

/// Charge le fichier `.skin` companion d'un MD3 si présent et retourne la
/// table `surface_name_lowercase → texture_path`.
///
/// Convention Q3 : `models/players/sarge/lower.md3` → cherche
/// `models/players/sarge/lower_default.skin`. Format texte, une entrée par
/// ligne :
///
/// ```text
/// l_legs,models/players/sarge/band.tga
/// tag_torso,
/// ```
///
/// Les lignes avec un texture path vide (tags, commentaires) sont ignorées.
fn load_default_skin(vfs: &Vfs, md3_path: &str) -> Option<HashMap<String, String>> {
    // Dérive "<dir>/<basename>_default.skin".
    let (stem, dot) = md3_path.rsplit_once('.')?;
    if !dot.eq_ignore_ascii_case("md3") {
        // Pas un .md3 "normal" ; pas de skin implicite à tenter.
        let _ = stem;
    }
    let skin_path = format!("{stem}_default.skin");
    let bytes = vfs.read(&skin_path).ok()?;
    let text = std::str::from_utf8(&bytes).ok()?;

    let mut map: HashMap<String, String> = HashMap::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        let Some((surf, tex)) = line.split_once(',') else {
            continue;
        };
        let surf = surf.trim();
        let tex = tex.trim();
        if surf.is_empty() || tex.is_empty() {
            // tag_* entries ou lignes vides côté texture.
            continue;
        }
        map.insert(surf.to_ascii_lowercase(), tex.to_string());
    }

    if map.is_empty() {
        None
    } else {
        debug!(
            "md3 skin: `{}` → {} entrée(s)",
            skin_path,
            map.len()
        );
        Some(map)
    }
}

fn decode_normal(packed: u16) -> Vec3 {
    let lat = (packed & 0xFF) as f32 * (core::f32::consts::TAU / 255.0);
    let lng = ((packed >> 8) & 0xFF) as f32 * (core::f32::consts::TAU / 255.0);
    let (sl, cl) = lat.sin_cos();
    let (sg, cg) = lng.sin_cos();
    Vec3::new(cl * sg, sl * sg, cg)
}

