//! HDR10 output — detection display HDR et sortie PQ/ST.2084.
//!
//! wgpu expose les capacites HDR du display via les formats de surface.
//! Si `Rgba16Float` ou `Rgb10a2Unorm` est disponible comme format de
//! surface, on peut activer la sortie HDR10 et utiliser le shader PQ
//! au lieu de l'ACES tonemap SDR.
//!
//! # Pourquoi un module separe ?
//!
//! Le pipeline SDR (post.rs) convertit HDR → sRGB via ACES.  Le pipeline
//! HDR10 convertit HDR → PQ via ST.2084.  Les deux lisent la meme scene
//! `Rgba16Float` mais produisent des signaux fondamentalement differents.
//! Les separer evite de spaghettifier `PostFx` avec des branches HDR
//! partout et permet de desactiver le module proprement si le display
//! n'est pas HDR.

use std::sync::Arc;
use wgpu::{Device, Queue, TextureFormat, TextureView};

const HDR_OUTPUT_WGSL: &str = include_str!("shaders/hdr_output.wgsl");

/// Formats de surface HDR supportes par ordre de preference.
const HDR_SURFACE_FORMATS: &[TextureFormat] = &[
    TextureFormat::Rgba16Float,
    TextureFormat::Rgb10a2Unorm,
];

/// Uniforme HDR10 — doit matcher `HdrParams` dans hdr_output.wgsl.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HdrParams {
    exposure: f32,
    peak_nits: f32,
    paper_white: f32,
    bloom_intensity: f32,
}

pub struct HdrOutput {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// True si le display supporte un format HDR et que l'utilisateur
    /// a active la sortie HDR (cvar `r_hdr 1`).
    pub enabled: bool,
    /// Luminance pic du display en nits (typiquement 1000 pour un
    /// moniteur HDR entry-level, 1600+ pour un OLED haut de gamme).
    pub peak_nits: f32,
    /// Luminance SDR reference ("paper white") en nits.  Les surfaces
    /// a luminance 1.0 en scene lineaire seront affichees a cette
    /// luminosite.  200 nits est un bon defaut (ecran desktop typique).
    pub paper_white_nits: f32,
    /// Format de surface HDR selectionne.
    pub surface_format: TextureFormat,
}

impl HdrOutput {
    /// Cree le module HDR10.  `surface_format` doit etre un des formats
    /// HDR-capable (Rgba16Float ou Rgb10a2Unorm).  Si le format n'est
    /// pas HDR, le module est cree mais `enabled` sera false.
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        surface_format: TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hdr-output-shader"),
            source: wgpu::ShaderSource::Wgsl(HDR_OUTPUT_WGSL.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hdr-output-bgl"),
                entries: &[
                    // binding 0 : hdr_input texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: true,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1 : bloom texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: true,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 2 : sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                    // binding 3 : params uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hdr-output-layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hdr-output-pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_fullscreen",
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_hdr10",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });

        let uniform_buffer =
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hdr-output-uniform"),
                size: std::mem::size_of::<HdrParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hdr-output-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let is_hdr = HDR_SURFACE_FORMATS.contains(&surface_format);

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            uniform_buffer,
            sampler,
            enabled: is_hdr,
            peak_nits: 1000.0,
            paper_white_nits: 200.0,
            surface_format,
        }
    }

    /// Detecte si les capacites de surface incluent un format HDR.
    /// Retourne le meilleur format HDR disponible, ou None.
    pub fn detect_hdr(caps: &wgpu::SurfaceCapabilities) -> Option<TextureFormat> {
        HDR_SURFACE_FORMATS.iter().find(|&&fmt| caps.formats.contains(&fmt)).copied()
    }

    /// Applique la conversion HDR10 (PQ curve) sur la scene HDR → surface.
    pub fn apply(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hdr_view: &TextureView,
        bloom_view: &TextureView,
        output_view: &TextureView,
        exposure: f32,
        bloom_intensity: f32,
    ) {
        if !self.enabled {
            return;
        }

        // Upload params
        let params = HdrParams {
            exposure,
            peak_nits: self.peak_nits,
            paper_white: self.paper_white_nits,
            bloom_intensity,
        };
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        // Bind group ephemere (les views changent a chaque frame)
        let bind_group =
            self.device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hdr-output-bg"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                hdr_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                bloom_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(
                                &self.sampler,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                    ],
                });

        let mut pass =
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr10-output-pass"),
                color_attachments: &[Some(
                    wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    },
                )],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Met a jour les parametres HDR (peak nits, paper white).
    pub fn set_params(&mut self, peak_nits: f32, paper_white: f32) {
        self.peak_nits = peak_nits.max(100.0);
        self.paper_white_nits = paper_white.clamp(80.0, 500.0);
    }
}
