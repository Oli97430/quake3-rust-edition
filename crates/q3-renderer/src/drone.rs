//! Pipeline rendu drones / props GLB — **instanced rendering**.
//!
//! v0.9.5+ refactor : on utilise un buffer instance per-frame avec
//! `step_mode: Instance` au lieu d'un uniform partagé. Avant ça les
//! draws successifs partageaient le même slot → toutes les transforms
//! collapsaient sur la dernière écriture → artefacts géométriques
//! (triangles s'étirant entre 2 transforms drones, blancs car le
//! shader émettait du tint additionné).
//!
//! Architecture :
//! * Un `pipeline` partagé pour tous les meshes (vertex layout unique).
//! * Un `mesh: Option<DroneMeshGpu>` pour le drone principal.
//! * Une `props: HashMap<String, DroneMeshGpu>` pour les autres GLBs.
//! * Un `instance_buffer` (vertex, step_mode Instance) écrit chaque
//!   frame avec toutes les instances de la frame, organisé par
//!   groupes (drones puis chaque prop name).  Chaque draw_indexed
//!   utilise `instance_range` pour sélectionner ses instances.

use bytemuck::{cast_slice, Pod, Zeroable};
use hashbrown::HashMap;
use q3_model::glb::{GlbMesh, GlbVertex};
use std::sync::Arc;
use tracing::info;
use wgpu::util::DeviceExt;

use crate::{DEPTH_FORMAT, SCENE_HDR_FORMAT};

/// Données per-instance — mat4 (col-major) + tint vec4. 80 octets.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DroneInstance {
    pub model: [[f32; 4]; 4],
    pub tint: [f32; 4],
}

impl DroneInstance {
    pub const STRIDE_BYTES: usize = std::mem::size_of::<Self>();
}

fn vertex_layouts() -> [wgpu::VertexBufferLayout<'static>; 2] {
    [
        // Buffer 0 : mesh vertices, 48 octets (pos + normal + uv).
        wgpu::VertexBufferLayout {
            array_stride: GlbVertex::STRIDE_BYTES as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        },
        // Buffer 1 : instances, 80 octets (mat4 + vec4 tint).
        wgpu::VertexBufferLayout {
            array_stride: DroneInstance::STRIDE_BYTES as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 64,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        },
    ]
}

pub struct DroneMeshGpu {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    /// Bind group PBR : baseColor + sampler + normal + metallicRoughness
    /// + material factors uniform.  Si une texture est absente, on
    /// bind le fallback approprié (blanc pour albedo, flat (0.5, 0.5,
    /// 1) pour normalmap, (0, 1, 0) pour MR = roughness 1 metallic 0).
    pub material_bg: wgpu::BindGroup,
    /// Buffer factors (live) — 32 octets : base_color (vec4) + mr (vec4).
    /// Conservé ici pour garder le bind group valide, pas modifié à
    /// runtime (les factors sont fixés au load).
    _factors_buf: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialFactors {
    base_color: [f32; 4],
    /// .x = metallicFactor, .y = roughnessFactor, .z/w = unused
    mr: [f32; 4],
}

impl DroneMeshGpu {
    pub fn from_mesh(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &GlbMesh,
        material_bgl: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        white_view: &wgpu::TextureView,
        flat_normal_view: &wgpu::TextureView,
        default_mr_view: &wgpu::TextureView,
    ) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("drone-glb-vb"),
            contents: cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("drone-glb-ib"),
            contents: cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Helper local pour upload une GlbTexture en wgpu Texture +
        // créer sa View. Pour normalmap on utilise Rgba8Unorm (linéaire,
        // sans sRGB). Pour baseColor on utilise sRGB.
        let upload = |label: &str, tex: &q3_model::glb::GlbTexture, srgb: bool| {
            let size = wgpu::Extent3d {
                width: tex.width,
                height: tex.height,
                depth_or_array_layers: 1,
            };
            let format = if srgb {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            };
            let tex_obj = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &tex_obj,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &tex.data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * tex.width),
                    rows_per_image: Some(tex.height),
                },
                size,
            );
            tex_obj.create_view(&wgpu::TextureViewDescriptor::default())
        };

        let bc_view: Option<wgpu::TextureView> = mesh
            .base_color_texture
            .as_ref()
            .map(|t| upload("drone-glb-baseColor", t, true));
        let nm_view: Option<wgpu::TextureView> = mesh
            .normal_texture
            .as_ref()
            .map(|t| upload("drone-glb-normalMap", t, false));
        let mr_view: Option<wgpu::TextureView> = mesh
            .metallic_roughness_texture
            .as_ref()
            .map(|t| upload("drone-glb-metallicRoughness", t, false));

        let bc_ref = bc_view.as_ref().unwrap_or(white_view);
        let nm_ref = nm_view.as_ref().unwrap_or(flat_normal_view);
        let mr_ref = mr_view.as_ref().unwrap_or(default_mr_view);

        let factors = MaterialFactors {
            base_color: mesh.base_color_factor,
            mr: [mesh.metallic_factor, mesh.roughness_factor, 0.0, 0.0],
        };
        let factors_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("drone-glb-factors"),
            contents: bytemuck::bytes_of(&factors),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let material_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("drone-glb-mat-bg"),
            layout: material_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(bc_ref) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(nm_ref) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(mr_ref) },
                wgpu::BindGroupEntry { binding: 4, resource: factors_buf.as_entire_binding() },
            ],
        });
        Self {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
            material_bg,
            _factors_buf: factors_buf,
        }
    }
}

/// Capacité max d'instances dans le buffer per-frame.  ~1000 = drones
/// (8) + rocks (400) + tropical (250) + marge.  Recreated grand si
/// dépassé via `ensure_capacity`.
const INITIAL_INSTANCE_CAPACITY: usize = 1024;

pub struct DroneRenderer {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    mesh: Option<DroneMeshGpu>,
    pub mesh_radius: f32,
    pub mesh_center: [f32; 3],
    props: HashMap<String, DroneMeshGpu>,
    prop_radii: HashMap<String, f32>,
    /// Scratch buffer pour concaténer les instances avant write_buffer.
    /// Réutilisé entre frames via `clear()` → pas d'alloc heap par frame
    /// (avant on faisait `Vec::with_capacity(total)` × N flush/sec).
    scratch_instances: Vec<DroneInstance>,
    /// BGL pour le matériau (texture + sampler) — partagée par tous
    /// les meshes. Group 1 du pipeline.
    material_bgl: wgpu::BindGroupLayout,
    /// Sampler bilinéaire wrap — partagé par tous les meshes.
    sampler: wgpu::Sampler,
    /// Fallbacks 1×1 partagés par tous les meshes.
    white_view: wgpu::TextureView,
    flat_normal_view: wgpu::TextureView,
    default_mr_view: wgpu::TextureView,
    /// Owners pour garder les textures vivantes.
    _white_texture: wgpu::Texture,
    _flat_normal_tex: wgpu::Texture,
    _default_mr_tex: wgpu::Texture,
}

impl DroneRenderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        camera_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("drone-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/drone.wgsl").into()),
        });

        // **Material BGL** — PBR : 3 textures + sampler + factors uniform.
        let tex_entry = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("drone-material-bgl"),
            entries: &[
                tex_entry(0), // baseColor
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex_entry(2), // normal
                tex_entry(3), // metallicRoughness
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Filtrage trilinear + anisotropic 16× pour les props PBR.
        // Visible sur les rocks/statues vues en perspective rasante :
        // texture diffuse + normal map restent nettes à grande distance
        // au lieu de virer en bouillie floue.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("drone-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });

        // **Fallback textures 1×1** — uploadées une fois, partagées
        // par tous les meshes qui n'ont pas la map correspondante.
        let make_1x1 = |label: &str, srgb: bool, rgba: [u8; 4]| {
            let format = if srgb {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            };
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba,
                wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        // Blanc sRGB pour baseColor manquant.
        let (white_texture, white_view) = make_1x1("drone-white-fallback", true, [255, 255, 255, 255]);
        // Normalmap "flat" — (0.5, 0.5, 1.0) en linéaire = normale Z+
        // pure, pas de perturbation. Linear (pas sRGB).
        let (_flat_normal_tex, flat_normal_view) =
            make_1x1("drone-flat-normal-fallback", false, [128, 128, 255, 255]);
        // MR par défaut : G=255 (roughness 1.0 max) B=0 (non-metallic).
        let (_default_mr_tex, default_mr_view) =
            make_1x1("drone-default-mr-fallback", false, [0, 255, 0, 255]);

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("drone-pipeline-layout"),
            bind_group_layouts: &[camera_bgl, &material_bgl],
            push_constant_ranges: &[],
        });

        let layouts = vertex_layouts();
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("drone-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &layouts,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
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
                    format: SCENE_HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let instance_capacity = INITIAL_INSTANCE_CAPACITY;
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("drone-instance-vb"),
            size: (instance_capacity * DroneInstance::STRIDE_BYTES) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        info!("drone-gpu: pipeline créé (instanced, capacity {})", instance_capacity);

        Self {
            device,
            queue,
            pipeline,
            instance_buffer,
            instance_capacity,
            mesh: None,
            mesh_radius: 0.0,
            mesh_center: [0.0; 3],
            props: HashMap::new(),
            prop_radii: HashMap::new(),
            material_bgl,
            sampler,
            white_view,
            flat_normal_view,
            default_mr_view,
            _white_texture: white_texture,
            _flat_normal_tex,
            _default_mr_tex,
            scratch_instances: Vec::with_capacity(64),
        }
    }

    pub fn upload_mesh(&mut self, mesh: &GlbMesh) {
        let gpu = DroneMeshGpu::from_mesh(
            &self.device,
            &self.queue,
            mesh,
            &self.material_bgl,
            &self.sampler,
            &self.white_view,
            &self.flat_normal_view,
            &self.default_mr_view,
        );
        self.mesh = Some(gpu);
        self.mesh_radius = mesh.radius();
        self.mesh_center = mesh.center();
        info!("drone-gpu: mesh upload — {} verts", mesh.vertices.len());
    }

    pub fn has_mesh(&self) -> bool {
        self.mesh.is_some()
    }

    pub fn upload_prop(&mut self, name: &str, mesh: &GlbMesh) {
        let gpu = DroneMeshGpu::from_mesh(
            &self.device,
            &self.queue,
            mesh,
            &self.material_bgl,
            &self.sampler,
            &self.white_view,
            &self.flat_normal_view,
            &self.default_mr_view,
        );
        info!(
            "prop '{}': upload {} verts / {} idx, radius {:.1}, baseColor texture {}",
            name,
            mesh.vertices.len(),
            mesh.indices.len(),
            mesh.radius(),
            if mesh.base_color_texture.is_some() { "PRESENT" } else { "absent" },
        );
        self.props.insert(name.to_string(), gpu);
        self.prop_radii.insert(name.to_string(), mesh.radius());
    }

    pub fn has_prop(&self, name: &str) -> bool {
        self.props.contains_key(name)
    }

    /// Radius natif du prop (en unités du mesh, avant scale). Utile
    /// pour calculer un scale auto qui matche une taille cible monde.
    pub fn prop_radius(&self, name: &str) -> Option<f32> {
        self.prop_radii.get(name).copied()
    }

    /// Reduit ou expand le buffer instance si la capacité n'est pas suffisante.
    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.instance_capacity {
            return;
        }
        let new_cap = n.next_power_of_two().max(INITIAL_INSTANCE_CAPACITY);
        self.instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("drone-instance-vb"),
            size: (new_cap * DroneInstance::STRIDE_BYTES) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.instance_capacity = new_cap;
        info!("drone-gpu: instance buffer grown to {}", new_cap);
    }

    /// Upload les instances + dessine en groupes (drone, puis chaque prop).
    /// `drones` et `props_grouped` doivent être pré-organisés par mesh.
    pub fn flush_and_draw<'a>(
        &'a mut self,
        pass: &mut wgpu::RenderPass<'a>,
        drones: &[DroneInstance],
        props_grouped: &[(&'a str, &[DroneInstance])],
    ) {
        // Calcule la layout offsets dans le buffer.
        let total: usize = drones.len()
            + props_grouped.iter().map(|(_, v)| v.len()).sum::<usize>();
        if total == 0 {
            return;
        }
        self.ensure_capacity(total);

        // **Scratch buffer réutilisé** (v0.9.5++ perf) — au lieu de
        // `Vec::with_capacity(total)` qui allouait par frame, on
        // réutilise `self.scratch_instances` via `clear() + extend()`.
        // Économie : 1 alloc heap/frame × ~60 fps = ~60 allocs/s
        // évitées sur scènes avec drones+props.
        self.scratch_instances.clear();
        self.scratch_instances.reserve(total);
        self.scratch_instances.extend_from_slice(drones);
        for (_, v) in props_grouped {
            self.scratch_instances.extend_from_slice(v);
        }
        self.queue.write_buffer(
            &self.instance_buffer,
            0,
            cast_slice(&self.scratch_instances),
        );

        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

        let mut cursor: u32 = 0;
        if !drones.is_empty() {
            if let Some(m) = self.mesh.as_ref() {
                pass.set_bind_group(1, &m.material_bg, &[]);
                pass.set_vertex_buffer(0, m.vertex_buffer.slice(..));
                pass.set_index_buffer(m.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                let count = drones.len() as u32;
                pass.draw_indexed(0..m.index_count, 0, cursor..(cursor + count));
                cursor += count;
            } else {
                cursor += drones.len() as u32;
            }
        }
        for (name, v) in props_grouped {
            if v.is_empty() {
                continue;
            }
            if let Some(m) = self.props.get(*name) {
                pass.set_bind_group(1, &m.material_bg, &[]);
                pass.set_vertex_buffer(0, m.vertex_buffer.slice(..));
                pass.set_index_buffer(m.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                let count = v.len() as u32;
                pass.draw_indexed(0..m.index_count, 0, cursor..(cursor + count));
                cursor += count;
            } else {
                cursor += v.len() as u32;
            }
        }
    }
}
