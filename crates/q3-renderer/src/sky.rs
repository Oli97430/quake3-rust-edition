//! Ciel — deux pipelines au choix :
//!
//! * **procédural** — gradient horizon → zenith (fallback si pas de cubemap)
//! * **cubemap**    — échantillonne une `texture_cube` chargée depuis les
//!   6 faces `env/<base>_{rt,lf,up,dn,ft,bk}.tga`
//!
//! Les deux pipelines partagent la même géométrie (un triangle plein écran)
//! et la même convention de depth (LessEqual @ z=1.0). Le choix se fait
//! dynamiquement dans `Renderer::render()` en fonction de la présence d'une
//! cubemap active, ce qui permet de basculer d'une map à l'autre sans
//! reconstruire le pipeline.

use crate::cubemap::Cubemap;
use crate::DEPTH_FORMAT;
use std::sync::Arc;

pub struct SkyRenderer {
    device: Arc<wgpu::Device>,
    procedural_pipeline: wgpu::RenderPipeline,
    cubemap_pipeline: wgpu::RenderPipeline,
    cubemap_bgl: wgpu::BindGroupLayout,
    /// Cubemap active (None = fallback procédural).
    cubemap: Option<Arc<Cubemap>>,
    cubemap_bind_group: Option<wgpu::BindGroup>,
}

impl SkyRenderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        camera_bgl: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
    ) -> Self {
        // Pipeline procédural.
        let proc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky-procedural-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sky.wgsl").into()),
        });
        let proc_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky-procedural-layout"),
            bind_group_layouts: &[camera_bgl],
            push_constant_ranges: &[],
        });
        let procedural_pipeline = Self::build_pipeline(
            &device,
            &proc_shader,
            &proc_layout,
            format,
            "sky-procedural-pipeline",
        );

        // Pipeline cubemap.
        let cubemap_bgl = Cubemap::bind_group_layout(&device);
        let cube_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky-cubemap-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sky_cubemap.wgsl").into()),
        });
        let cube_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky-cubemap-layout"),
            bind_group_layouts: &[camera_bgl, &cubemap_bgl],
            push_constant_ranges: &[],
        });
        let cubemap_pipeline = Self::build_pipeline(
            &device,
            &cube_shader,
            &cube_layout,
            format,
            "sky-cubemap-pipeline",
        );

        Self {
            device,
            procedural_pipeline,
            cubemap_pipeline,
            cubemap_bgl,
            cubemap: None,
            cubemap_bind_group: None,
        }
    }

    fn build_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[],
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
                module: shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        })
    }

    /// Active (ou retire si `None`) une cubemap. Le bind group est reconstruit.
    pub fn set_cubemap(&mut self, cubemap: Option<Arc<Cubemap>>) {
        self.cubemap_bind_group = cubemap
            .as_ref()
            .map(|c| c.bind_group(&self.device, &self.cubemap_bgl));
        self.cubemap = cubemap;
    }

    /// Pipeline procédural (gradient fallback).
    pub fn procedural_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.procedural_pipeline
    }

    /// Pipeline cubemap (si `set_cubemap` a été appelée).
    pub fn cubemap_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.cubemap_pipeline
    }

    /// Bind group de la cubemap active, ou `None` → utiliser procédural.
    pub fn cubemap_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.cubemap_bind_group.as_ref()
    }

    /// Retourne le basename de la cubemap active (debug / logs).
    pub fn cubemap_base(&self) -> Option<&str> {
        self.cubemap.as_deref().map(|c| c.base_path.as_str())
    }

    /// Compat : le renderer historique appelait `.pipeline()`. On renvoie le
    /// pipeline à utiliser pour la frame en cours.
    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        if self.cubemap_bind_group.is_some() {
            &self.cubemap_pipeline
        } else {
            &self.procedural_pipeline
        }
    }
}
