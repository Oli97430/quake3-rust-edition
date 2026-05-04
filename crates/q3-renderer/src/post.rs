//! Post-process : tonemap ACES + bloom additif sur scene HDR.
//!
//! # Architecture HDR (v0.9+)
//!
//! La scene est rendue dans une texture offscreen `hdr_input` au format
//! `Rgba16Float`. Cette stack lit directement cette texture (pas de
//! `copy_texture_to_texture` sur la swapchain comme avant) et exécute :
//!
//! 1. **Bright extract** : lit `hdr_input`, écrit dans `bloom_a` les
//!    pixels dont la luminance dépasse [`BLOOM_THRESHOLD`]. Sur HDR le
//!    seuil peut être > 1.0 (vrai bloom physique : seuls les vraiment
//!    sur-exposés brillent).
//! 2. **Blur séparable** : H sur `bloom_a` → `bloom_b`, V sur `bloom_b`
//!    → `bloom_a`. Gauss 9-tap.
//! 3. **Composite tonemap** : lit `hdr_input` + `bloom_a`, applique
//!    ACES Filmic, écrit le résultat sRGB sur la swapchain (remplacement
//!    total — la passe écrit chaque pixel).
//!
//! Le HUD est dessiné après cette passe sur la swapchain directement
//! (LDR sRGB), donc il n'hérite pas du tonemap ni du bloom.

use std::sync::Arc;
use wgpu::{Device, Queue, Texture, TextureFormat, TextureView};

const BLOOM_DOWNSAMPLE: u32 = 4;
/// Seuil HDR : > 1.0 ne garde que les pixels vraiment sur-exposés
/// (muzzle flash, soleil), pas les surfaces lambda. C'est le seuil
/// "physiquement correct" qui n'était pas exploitable en LDR.
const BLOOM_THRESHOLD: f32 = 1.05;
const BLOOM_INTENSITY: f32 = 0.65;
/// Multiplicateur scène HDR avant tonemap.  Bump à 1.20 (v0.9.5++)
/// pour compenser l'effet cumulé edge-AO + vignette + color grading
/// qui empilaient ~25 % d'atténuation perceptive sur les midtones.
const ACES_EXPOSURE: f32 = 1.20;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ComposeUniform {
    intensity: f32,
    exposure: f32,
    /// `time_sec` du moteur — anime le film grain (offset hash par frame
    /// pour éviter la statique fixe) et tout futur effet temporel.
    time: f32,
    /// 1.0 si le soleil est dans le frustum visible, 0.0 sinon.  Module
    /// l'intensité des god rays (off-screen → pas de raymarch).
    sun_visibility: f32,
    /// Position UV du soleil sur l'écran [0..1].  Dérivé chaque frame
    /// depuis `camera.view_proj × sun_dir × far_distance`.  Hors
    /// frustum → sun_visibility = 0 et ces UV sont ignorées.
    sun_uv_x: f32,
    sun_uv_y: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ExtractUniform {
    threshold: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurUniform {
    texel: [f32; 2],
    direction: [f32; 2],
}

pub struct PostFx {
    device: Arc<Device>,
    queue: Arc<Queue>,
    #[allow(dead_code)]
    surface_format: TextureFormat,

    // **Multi-mip bloom chain** (v0.9.5++ #11) — 2 niveaux indépendants.
    // Mip 1 (`_a` / `_b`) au 1/4 résolution écran, mip 2 (`_a2` / `_b2`)
    // au 1/16.  Le mip 2 est downsamplé depuis mip 1 (post-blur), puis
    // re-blurré.  Le composite final additionne les deux avec des poids
    // donnant un halo serré (mip1) + un halo large (mip2) = bloom
    // organique multi-échelle.
    bloom_a: Texture,
    bloom_a_view: TextureView,
    bloom_b: Texture,
    bloom_b_view: TextureView,
    bloom_a2: Texture,
    bloom_a2_view: TextureView,
    bloom_b2: Texture,
    bloom_b2_view: TextureView,

    sampler: wgpu::Sampler,

    bgl_single: wgpu::BindGroupLayout,
    bgl_compose: wgpu::BindGroupLayout,

    extract_pipeline: wgpu::RenderPipeline,
    extract_uniform: wgpu::Buffer,
    extract_bg: Option<wgpu::BindGroup>,

    blur_pipeline: wgpu::RenderPipeline,
    blur_uniform_h: wgpu::Buffer,
    blur_uniform_v: wgpu::Buffer,
    blur_bg_h: wgpu::BindGroup,
    blur_bg_v: wgpu::BindGroup,
    // Mip 2 : blur uniforms (texel size adapté au /16) + bind groups.
    blur_uniform_h2: wgpu::Buffer,
    blur_uniform_v2: wgpu::Buffer,
    blur_bg_h2: wgpu::BindGroup,
    blur_bg_v2: wgpu::BindGroup,
    // Pipeline downsample mip1 → mip2 via bilinear filter implicite.
    downsample_pipeline: wgpu::RenderPipeline,
    downsample_uniform: wgpu::Buffer,
    downsample_bg: wgpu::BindGroup,

    compose_pipeline: wgpu::RenderPipeline,
    compose_uniform: wgpu::Buffer,
    /// Compose utilise 2 bind groups (group 0 single + group 1 extra).
    compose_bg: Option<(wgpu::BindGroup, wgpu::BindGroup)>,

    width: u32,
    height: u32,
}

impl PostFx {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        surface_format: TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("postfx-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bw = (width / BLOOM_DOWNSAMPLE).max(1);
        let bh = (height / BLOOM_DOWNSAMPLE).max(1);
        // Mip 2 au 1/16 résolution écran (= 1/4 du mip 1).  Le facteur
        // 4× est choisi pour produire une vraie séparation visuelle
        // entre les deux mips dans le composite (mip1 = halo serré
        // ~32 px à 1080p, mip2 = halo large ~128 px).
        let bw2 = (width / (BLOOM_DOWNSAMPLE * 4)).max(1);
        let bh2 = (height / (BLOOM_DOWNSAMPLE * 4)).max(1);
        // Format bloom = HDR aussi pour préserver les valeurs > 1.0
        // pendant le blur. Sinon le blur clipperait à 1.0 et le composite
        // perdrait le côté "sur-exposé brûlant" du highlight.
        let bloom_format = crate::SCENE_HDR_FORMAT;
        let (bloom_a, bloom_a_view) = create_bloom_texture(&device, bloom_format, bw, bh, "bloom-a");
        let (bloom_b, bloom_b_view) = create_bloom_texture(&device, bloom_format, bw, bh, "bloom-b");
        let (bloom_a2, bloom_a2_view) = create_bloom_texture(&device, bloom_format, bw2, bh2, "bloom-a2");
        let (bloom_b2, bloom_b2_view) = create_bloom_texture(&device, bloom_format, bw2, bh2, "bloom-b2");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("postfx-shader"),
            source: wgpu::ShaderSource::Wgsl(POSTFX_WGSL.into()),
        });

        // BGL "single" : sampler + 1 texture + uniform — pour extract / blur.
        let bgl_single = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("postfx-bgl-single"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
        let layout_single = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("postfx-layout-single"),
            bind_group_layouts: &[&bgl_single],
            push_constant_ranges: &[],
        });

        // BGL "compose-extra" — bind group additionnel sur GROUP(1)
        // pour le compose : bloom_tex (mip1) + bloom_tex2 (mip2) +
        // compose_uniform + depth_tex (#8 SSAO).  Le compose pipeline
        // a layout = [bgl_single, bgl_compose_extra].
        let bgl_compose = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("postfx-bgl-compose-extra"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // **Depth texture** (v0.9.5++ #8 SSAO) — Depth32Float
                // bound as filterable=false ; lu via `textureLoad` côté
                // WGSL (pas besoin de sampler additionnel).
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let layout_compose = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("postfx-layout-compose"),
            bind_group_layouts: &[&bgl_single, &bgl_compose],
            push_constant_ranges: &[],
        });

        // ─── Extract pipeline (HDR → bloom_a) ───
        let extract_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("postfx-extract-uniform"),
            size: std::mem::size_of::<ExtractUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &extract_uniform,
            0,
            bytemuck::bytes_of(&ExtractUniform {
                threshold: BLOOM_THRESHOLD,
                _pad: [0.0; 3],
            }),
        );
        let extract_pipeline = make_pipeline(
            &device, &shader, &layout_single, "fs_extract",
            bloom_format, wgpu::BlendState::REPLACE, "postfx-extract",
        );

        // ─── Blur pipeline (réutilisé H et V) ───
        let blur_uniform_h = make_blur_uniform(&device, &queue, bw, bh, [1.0, 0.0], "blur-h-u");
        let blur_uniform_v = make_blur_uniform(&device, &queue, bw, bh, [0.0, 1.0], "blur-v-u");
        let blur_pipeline = make_pipeline(
            &device, &shader, &layout_single, "fs_blur",
            bloom_format, wgpu::BlendState::REPLACE, "postfx-blur",
        );
        let blur_bg_h = make_bg_single(
            &device, &bgl_single, &sampler, &bloom_a_view, &blur_uniform_h, "blur-h-bg",
        );
        let blur_bg_v = make_bg_single(
            &device, &bgl_single, &sampler, &bloom_b_view, &blur_uniform_v, "blur-v-bg",
        );

        // ─── Mip 2 blur uniforms + bind groups (texel size /16) ───
        let blur_uniform_h2 = make_blur_uniform(&device, &queue, bw2, bh2, [1.0, 0.0], "blur-h2-u");
        let blur_uniform_v2 = make_blur_uniform(&device, &queue, bw2, bh2, [0.0, 1.0], "blur-v2-u");
        let blur_bg_h2 = make_bg_single(
            &device, &bgl_single, &sampler, &bloom_a2_view, &blur_uniform_h2, "blur-h2-bg",
        );
        let blur_bg_v2 = make_bg_single(
            &device, &bgl_single, &sampler, &bloom_b2_view, &blur_uniform_v2, "blur-v2-bg",
        );

        // ─── Downsample pipeline (mip1 → mip2) ───
        // Lit `bloom_a` (post-blur, /4 res), écrit dans `bloom_a2` (/16 res).
        // Le passage à plus petite résolution + sampler linear donne un
        // box-filter 2×2 implicite par pixel destination → équivalent à
        // un downsample 4× avec moyenne quadratique.  L'uniform contient
        // juste le texel size (réutilise BlurUniform pour économie).
        let downsample_uniform = make_blur_uniform(&device, &queue, bw, bh, [0.0, 0.0], "downsample-u");
        let downsample_pipeline = make_pipeline(
            &device, &shader, &layout_single, "fs_downsample",
            bloom_format, wgpu::BlendState::REPLACE, "postfx-downsample",
        );
        let downsample_bg = make_bg_single(
            &device, &bgl_single, &sampler, &bloom_a_view, &downsample_uniform, "downsample-bg",
        );

        // ─── Compose tonemap (HDR + bloom → surface sRGB) ───
        let compose_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("postfx-compose-uniform"),
            size: std::mem::size_of::<ComposeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &compose_uniform,
            0,
            bytemuck::bytes_of(&ComposeUniform {
                intensity: BLOOM_INTENSITY,
                exposure: ACES_EXPOSURE,
                time: 0.0,
                sun_visibility: 0.0,
                sun_uv_x: 0.5,
                sun_uv_y: 0.5,
                _pad: [0.0; 2],
            }),
        );
        let compose_pipeline = make_pipeline(
            &device, &shader, &layout_compose, "fs_compose",
            surface_format, wgpu::BlendState::REPLACE, "postfx-compose",
        );

        Self {
            device,
            queue,
            surface_format,
            bloom_a,
            bloom_a_view,
            bloom_b,
            bloom_b_view,
            bloom_a2,
            bloom_a2_view,
            bloom_b2,
            bloom_b2_view,
            sampler,
            bgl_single,
            bgl_compose,
            extract_pipeline,
            extract_uniform,
            extract_bg: None,
            blur_pipeline,
            blur_uniform_h,
            blur_uniform_v,
            blur_bg_h,
            blur_bg_v,
            blur_uniform_h2,
            blur_uniform_v2,
            blur_bg_h2,
            blur_bg_v2,
            downsample_pipeline,
            downsample_uniform,
            downsample_bg,
            compose_pipeline,
            compose_uniform,
            compose_bg: None,
            width,
            height,
        }
    }

    /// Lie le view HDR scene en entrée. Doit être appelé une fois après
    /// la création de PostFx, et à chaque resize. Sépare la création
    /// statique des bind groups (qui dépendent du view, recréé sur
    /// resize) de la stack des pipelines (qui ne change jamais).
    ///
    /// Pour le compose on crée DEUX bind groups :
    /// * group(0) — single layout réutilisé : sampler + hdr_tex +
    ///   uniform "fake" (juste pour valider le BGL ; pas lu par le
    ///   compose shader).
    /// * group(1) — compose-extra : bloom_tex + bloom_tex2 +
    ///   compose_uniform + depth_tex (#8 SSAO).
    pub fn set_hdr_input(&mut self, hdr_view: &TextureView, depth_view: &TextureView) {
        self.extract_bg = Some(make_bg_single(
            &self.device,
            &self.bgl_single,
            &self.sampler,
            hdr_view,
            &self.extract_uniform,
            "postfx-extract-bg",
        ));
        // Compose : binding `tex` = HDR view (group 0), binding `bloom_tex`
        // = bloom_a (group 1). On réutilise extract_uniform comme stub
        // sur le single BGL — non lu par fs_compose mais nécessaire
        // pour que le BGL soit complet.
        let compose_g0 = make_bg_single(
            &self.device,
            &self.bgl_single,
            &self.sampler,
            hdr_view,
            &self.extract_uniform,
            "postfx-compose-bg-g0",
        );
        let compose_g1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("postfx-compose-bg-g1"),
            layout: &self.bgl_compose,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_a_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.bloom_a2_view) },
                wgpu::BindGroupEntry { binding: 2, resource: self.compose_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
            ],
        });
        self.compose_bg = Some((compose_g0, compose_g1));
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        let bw = (width / BLOOM_DOWNSAMPLE).max(1);
        let bh = (height / BLOOM_DOWNSAMPLE).max(1);
        let bw2 = (width / (BLOOM_DOWNSAMPLE * 4)).max(1);
        let bh2 = (height / (BLOOM_DOWNSAMPLE * 4)).max(1);
        let bloom_format = crate::SCENE_HDR_FORMAT;
        let (a, av) = create_bloom_texture(&self.device, bloom_format, bw, bh, "bloom-a");
        let (b, bv) = create_bloom_texture(&self.device, bloom_format, bw, bh, "bloom-b");
        let (a2, a2v) = create_bloom_texture(&self.device, bloom_format, bw2, bh2, "bloom-a2");
        let (b2, b2v) = create_bloom_texture(&self.device, bloom_format, bw2, bh2, "bloom-b2");
        self.bloom_a = a;
        self.bloom_a_view = av;
        self.bloom_b = b;
        self.bloom_b_view = bv;
        self.bloom_a2 = a2;
        self.bloom_a2_view = a2v;
        self.bloom_b2 = b2;
        self.bloom_b2_view = b2v;
        self.queue.write_buffer(
            &self.blur_uniform_h, 0,
            bytemuck::bytes_of(&BlurUniform {
                texel: [1.0 / bw as f32, 1.0 / bh as f32],
                direction: [1.0, 0.0],
            }),
        );
        self.queue.write_buffer(
            &self.blur_uniform_v, 0,
            bytemuck::bytes_of(&BlurUniform {
                texel: [1.0 / bw as f32, 1.0 / bh as f32],
                direction: [0.0, 1.0],
            }),
        );
        self.queue.write_buffer(
            &self.blur_uniform_h2, 0,
            bytemuck::bytes_of(&BlurUniform {
                texel: [1.0 / bw2 as f32, 1.0 / bh2 as f32],
                direction: [1.0, 0.0],
            }),
        );
        self.queue.write_buffer(
            &self.blur_uniform_v2, 0,
            bytemuck::bytes_of(&BlurUniform {
                texel: [1.0 / bw2 as f32, 1.0 / bh2 as f32],
                direction: [0.0, 1.0],
            }),
        );
        self.blur_bg_h = make_bg_single(
            &self.device, &self.bgl_single, &self.sampler,
            &self.bloom_a_view, &self.blur_uniform_h, "blur-h-bg",
        );
        self.blur_bg_v = make_bg_single(
            &self.device, &self.bgl_single, &self.sampler,
            &self.bloom_b_view, &self.blur_uniform_v, "blur-v-bg",
        );
        self.blur_bg_h2 = make_bg_single(
            &self.device, &self.bgl_single, &self.sampler,
            &self.bloom_a2_view, &self.blur_uniform_h2, "blur-h2-bg",
        );
        self.blur_bg_v2 = make_bg_single(
            &self.device, &self.bgl_single, &self.sampler,
            &self.bloom_b2_view, &self.blur_uniform_v2, "blur-v2-bg",
        );
        self.downsample_bg = make_bg_single(
            &self.device, &self.bgl_single, &self.sampler,
            &self.bloom_a_view, &self.downsample_uniform, "downsample-bg",
        );
        // extract_bg + compose_bg dépendent du hdr_view externe → ils
        // sont re-binder via `set_hdr_input` par le Renderer.
        self.extract_bg = None;
        self.compose_bg = None;
        self.width = width;
        self.height = height;
    }

    /// Applique la passe finale : extract bright HDR, blur, compose
    /// tonemap → surface. À appeler entre la fin du rendu scene et le
    /// début du HUD. `surface_view` doit pointer sur la swapchain
    /// courante (LoadOp::Clear → on remplit chaque pixel).
    /// Met à jour les paramètres dynamiques du compose shader :
    /// * `time` — anime le film grain (sinon pattern statique)
    /// * `sun_uv` — position UV du soleil pour le raymarch god rays
    /// * `sun_visibility` — 0.0 hors frustum, 1.0 visible
    /// Doit être appelé chaque frame avant `apply`.
    pub fn set_compose_params(
        &self,
        time: f32,
        sun_uv: [f32; 2],
        sun_visibility: f32,
    ) {
        self.queue.write_buffer(
            &self.compose_uniform,
            0,
            bytemuck::bytes_of(&ComposeUniform {
                intensity: BLOOM_INTENSITY,
                exposure: ACES_EXPOSURE,
                time,
                sun_visibility,
                sun_uv_x: sun_uv[0],
                sun_uv_y: sun_uv[1],
                _pad: [0.0; 2],
            }),
        );
    }

    /// Backwards-compat shim — appelle set_compose_params avec sun off.
    /// Conservé pour ne pas casser des callers externes éventuels.
    pub fn set_time(&self, time: f32) {
        self.set_compose_params(time, [0.5, 0.5], 0.0);
    }

    pub fn apply(&self, encoder: &mut wgpu::CommandEncoder, surface_view: &TextureView) {
        let (Some(extract_bg), Some((compose_g0, compose_g1))) =
            (&self.extract_bg, &self.compose_bg)
        else {
            return;
        };
        // Mip 1 : extract bright + Gaussian blur séparable.
        run_pass(encoder, &self.bloom_a_view, &self.extract_pipeline, extract_bg, "postfx-extract-pass");
        run_pass(encoder, &self.bloom_b_view, &self.blur_pipeline, &self.blur_bg_h, "postfx-blur-h-pass");
        run_pass(encoder, &self.bloom_a_view, &self.blur_pipeline, &self.blur_bg_v, "postfx-blur-v-pass");
        // Mip 2 : downsample bloom_a (post-blur, 1/4 res) → bloom_a2
        // (1/16 res), puis blur séparable au niveau /16 → halo plus
        // large que le mip 1, pour un bloom multi-échelle organique.
        run_pass(encoder, &self.bloom_a2_view, &self.downsample_pipeline, &self.downsample_bg, "postfx-downsample-pass");
        run_pass(encoder, &self.bloom_b2_view, &self.blur_pipeline, &self.blur_bg_h2, "postfx-blur-h2-pass");
        run_pass(encoder, &self.bloom_a2_view, &self.blur_pipeline, &self.blur_bg_v2, "postfx-blur-v2-pass");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("postfx-compose-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.compose_pipeline);
        pass.set_bind_group(0, compose_g0, &[]);
        pass.set_bind_group(1, compose_g1, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn create_bloom_texture(
    device: &Device, format: TextureFormat, w: u32, h: u32, label: &str,
) -> (Texture, TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w.max(1), height: h.max(1), depth_or_array_layers: 1,
        },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn make_blur_uniform(
    device: &Device, queue: &Queue, w: u32, h: u32, direction: [f32; 2], label: &str,
) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<BlurUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &buf, 0,
        bytemuck::bytes_of(&BlurUniform {
            texel: [1.0 / w.max(1) as f32, 1.0 / h.max(1) as f32],
            direction,
        }),
    );
    buf
}

fn make_pipeline(
    device: &Device, shader: &wgpu::ShaderModule, layout: &wgpu::PipelineLayout,
    fs_entry: &str, format: TextureFormat, blend: wgpu::BlendState, label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader, entry_point: "vs_fullscreen",
            compilation_options: Default::default(), buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader, entry_point: fs_entry,
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format, blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn make_bg_single(
    device: &Device, layout: &wgpu::BindGroupLayout, sampler: &wgpu::Sampler,
    view: &TextureView, uniform: &wgpu::Buffer, label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Sampler(sampler) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view) },
            wgpu::BindGroupEntry { binding: 2, resource: uniform.as_entire_binding() },
        ],
    })
}

fn run_pass(
    encoder: &mut wgpu::CommandEncoder, view: &TextureView,
    pipeline: &wgpu::RenderPipeline, bg: &wgpu::BindGroup, label: &str,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view, resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bg, &[]);
    pass.draw(0..3, 0..1);
}

const POSTFX_WGSL: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    var x: f32 = -1.0 + f32((vid & 1u) << 2u);
    var y: f32 = -1.0 + f32((vid & 2u) << 1u);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

// ─── Group 0 — common (sampler + main tex + single uniform) ───
// Utilisé tel quel par extract/blur. Pour le compose, `tex` porte
// le HDR scene (binding 1) ; le `u_param` (binding 2) est présent
// mais ignoré par fs_compose (qui lit u_compose dans group(1)).
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
struct GenericU { a: f32, b: f32, c: f32, d: f32 };
@group(0) @binding(2) var<uniform> u_param: GenericU;

// ─── Group 1 — compose-extra (bloom + compose uniform) ───
// Présent UNIQUEMENT pour le compose pipeline. Les pipelines extract
// et blur ne lisent pas ces variables (validation OK : binding inutilisé
// sur un BGL absent du pipeline_layout = warning, pas une erreur).
@group(1) @binding(0) var bloom_tex: texture_2d<f32>;
@group(1) @binding(1) var bloom_tex2: texture_2d<f32>;
struct ComposeUParam {
    intensity: f32,
    exposure: f32,
    time: f32,
    sun_visibility: f32,
    sun_uv: vec2<f32>,
    p2: vec2<f32>,
};
@group(1) @binding(2) var<uniform> u_compose: ComposeUParam;
@group(1) @binding(3) var depth_tex: texture_depth_2d;

// --- ACES Filmic tonemap (Hill / Narkowicz fitted) ---
// On garde la fit Narkowicz : courbe rationnelle 5-paramètres bien plus
// lumineuse en midtones que la fit Hill 2017 (cette dernière compresse
// trop les midtones → maps perçues sombres).  La perte de précision
// chroma dans les highlights vaut largement la lisibilité des scènes
// non brûlées.  Test : input 1.0 → output 0.804 (vs ~0.65 chez Hill).
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

// --- Bright extract (HDR → bloom_a) ---
@fragment
fn fs_extract(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(tex, samp, in.uv);
    // Sur HDR le seuil peut être > 1.0 — vraies sur-expositions.
    let lum = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let knee = 0.25;
    let soft = clamp((lum - u_param.a + knee) / (2.0 * knee), 0.0, 1.0);
    let mult = soft * soft;
    return vec4<f32>(c.rgb * mult, 1.0);
}

// --- Gaussian blur 9-tap ---
@fragment
fn fs_blur(in: VsOut) -> @location(0) vec4<f32> {
    let texel_x = u_param.a;
    let texel_y = u_param.b;
    let dir = vec2<f32>(u_param.c, u_param.d);
    let off = vec2<f32>(texel_x, texel_y) * dir;
    let w0 = 0.227027;
    let w1 = 0.1945946;
    let w2 = 0.1216216;
    let w3 = 0.054054;
    let w4 = 0.016216;
    var col = textureSample(tex, samp, in.uv).rgb * w0;
    col = col + textureSample(tex, samp, in.uv + off * 1.0).rgb * w1;
    col = col + textureSample(tex, samp, in.uv - off * 1.0).rgb * w1;
    col = col + textureSample(tex, samp, in.uv + off * 2.0).rgb * w2;
    col = col + textureSample(tex, samp, in.uv - off * 2.0).rgb * w2;
    col = col + textureSample(tex, samp, in.uv + off * 3.0).rgb * w3;
    col = col + textureSample(tex, samp, in.uv - off * 3.0).rgb * w3;
    col = col + textureSample(tex, samp, in.uv + off * 4.0).rgb * w4;
    col = col + textureSample(tex, samp, in.uv - off * 4.0).rgb * w4;
    return vec4<f32>(col, 1.0);
}

// --- Downsample : sample direct via linear filter ---
// Le rasterizer rend dans une texture plus petite ; chaque fragment
// destination échantillonne le source à son UV avec sampler Linear,
// qui interpole 2×2 source texels = box filter implicite.  Suffisant
// pour la chaîne mip de bloom (qualité hi-fi pas requise puisqu'on
// va re-blurrer derrière).  Coût : 1 sample par pixel destination.
@fragment
fn fs_downsample(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}

// --- Compose : HDR scene + bloom (multi-mip) → tonemap + vignette + edge-AO
// + lens flare ghosts → surface sRGB.
//
// **v0.9.5++** :
// * vignette cinématique (assombrit les coins → focus optique)
// * "edge-AO" : détecte les discontinuités luminance entre pixels
//   voisins et accentue légèrement le côté sombre — donne l'impression
//   d'occlusion ambiante sans buffer de profondeur ni multi-sample.
//   Cheap (8 samples) mais visible sur les coins de mur, joints
//   d'objets, plis de terrain.
// * **Lens flare ghosts** : technique screen-space classique. Pour
//   chaque pixel on échantillonne `bloom_tex` à des positions
//   "miroir" symétriques autour du centre — si une source HDR très
//   brillante (soleil, explosion) existe ailleurs sur l'écran, elle
//   contribue ici sous forme d'un fantôme atténué + teinté. Tinte
//   légèrement chromatique (rouge/cyan) pour évoquer une aberration
//   chromatique d'optique. Halo radial doux pour le bloom global.
//   N'utilise QUE le bloom_tex déjà calculé → coût quasi nul.
// PRNG hash pour le film grain — fonction sin/dot classique, retourne
// un float dans [0,1) pseudo-aléatoire stable par UV.  L'instabilité
// temporelle vient du décalage de l'UV par `u_compose.time` côté
// appelant — sinon le pattern serait figé.
fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// Color grading lift-gamma-gain "blockbuster" — version douce
// (v0.9.5++) : tints quasi-neutres pour ne pas assombrir les ombres
// (le shadow_tint < 1.0 sur le canal R faisait perdre de la
// luminance perçue).  Split-tone garde l'esprit cinéma sans peser
// sur la lisibilité de la map.
fn color_grade(rgb: vec3<f32>) -> vec3<f32> {
    let lum = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    // Shadow tint : très léger pull bleu (R 0.95 → garde luminance).
    let shadow_tint = vec3<f32>(0.95, 1.00, 1.06);
    // Highlight tint : warm doux qui réchauffe sans saturer.
    let highlight_tint = vec3<f32>(1.06, 1.02, 0.92);
    let mix_t = smoothstep(0.0, 0.7, lum);
    let tint = mix(shadow_tint, highlight_tint, mix_t);
    let graded = rgb * tint;
    // Saturation +5 % — boost couleurs sans cartoonifier.
    let lum2 = dot(graded, vec3<f32>(0.2126, 0.7152, 0.0722));
    let satured = mix(vec3<f32>(lum2), graded, 1.05);
    // Mix final : 25 % graded + 75 % original — grading très subtil pour
    // ne pas teinter visiblement l'image (user complainait d'un "filtre
    // moche").  Le tonemap ACES suivant donne déjà du caractère.
    return mix(rgb, satured, 0.25);
}

@fragment
fn fs_compose(in: VsOut) -> @location(0) vec4<f32> {
    // **Chromatic aberration** — DÉSACTIVÉ (v0.9.5++ polish) — le user
    // trouvait l'effet trop visible ("filtre moche").  On garde `v`
    // calculé pour le vignette plus bas, mais on sample tex en pleine
    // résolution sans split RGB.
    let v = in.uv - vec2<f32>(0.5, 0.5);
    let scene_rgb = textureSample(tex, samp, in.uv).rgb;
    var scene = scene_rgb * u_compose.exposure;
    // **Multi-mip bloom** (v0.9.5++ #11) — combine 2 niveaux indépendants :
    //   * bloom_tex (mip 1, 1/4 res, blur Gaussien) = halo serré ~32px
    //   * bloom_tex2 (mip 2, 1/16 res, blur séparé) = halo large ~128px
    // Pondération : mip 1 conserve la signature locale d'une source vive,
    // mip 2 ajoute un voile diffus très loin autour.  Total ≈ intensity
    // pour rester dans la même enveloppe visuelle qu'avant.
    let bloom_mip1 = textureSample(bloom_tex, samp, in.uv).rgb * 0.65;
    let bloom_mip2 = textureSample(bloom_tex2, samp, in.uv).rgb * 0.55;
    let bloom = (bloom_mip1 + bloom_mip2) * u_compose.intensity;

    // **God rays** (v0.9.5++ #12) — raymarch radial depuis le pixel
    // courant vers la position UV du soleil.  À chaque step on
    // échantillonne `bloom_tex` (qui contient déjà les bright pixels
    // extraits, dont le sun disk).  L'accumulation pondérée crée des
    // "shafts" de lumière typiques d'une atmosphère poussiéreuse.
    //
    // Algorithme (Lewis & Sousa 2008 simplifié) :
    //   1. direction = sun_uv - pixel_uv
    //   2. step = direction / N_STEPS
    //   3. attenuation par step (exp décroissant)
    //   4. somme = Σ bloom_sample(uv + i*step) * weight(i) * decay^i
    //
    // 24 steps suffisent pour un visuel dense sans coût excessif.
    // Skip total si sun off-screen (sun_visibility = 0).
    var god_rays = vec3<f32>(0.0);
    if (u_compose.sun_visibility > 0.5) {
        let to_sun = u_compose.sun_uv - in.uv;
        // **Early-out perf** (v0.9.5++ polish) — `proximity` calculé
        // d'abord ; si trop loin du soleil, le résultat final = 0
        // (multiplié par proximity), inutile de payer 24 textureSample.
        let dist_to_sun = length(to_sun);
        let proximity = clamp(1.0 - dist_to_sun * 1.5, 0.0, 1.0);
        if (proximity > 0.001) {
            let n_steps = 24;
            let step = to_sun / f32(n_steps);
            var sample_uv = in.uv;
            var attenuation = 1.0;
            let decay = 0.93; // chaque step perd 7 % d'intensité
            let weight = 0.18; // poids initial — règle l'intensité globale
            for (var i = 0; i < 24; i = i + 1) {
                sample_uv = sample_uv + step;
                // Sample bloom_tex (mip1, full Gaussian blur) — contient
                // les sources vives = sun disk + tout autre bright pixel.
                let s = textureSample(bloom_tex, samp, sample_uv).rgb;
                god_rays = god_rays + s * weight * attenuation;
                attenuation = attenuation * decay;
            }
            // Tint chaud (jaune-orangé) pour évoquer l'aube/crépuscule.
            god_rays = god_rays * vec3<f32>(1.0, 0.92, 0.75) * proximity * 1.4;
        }
    }

    // **SSAO depth-based** (v0.9.5++ #8) — vraie occlusion ambiante
    // calculée depuis le depth buffer.  Algorithme Crytek-simplifié :
    //   1. Sample center depth z0
    //   2. Sample 8 voisins en disque (4-pixel radius)
    //   3. Pour chaque voisin : si son depth est PLUS PROCHE de la
    //      caméra que z0 dans une plage [near, far], il occulte le
    //      pixel courant (il bouche la lumière)
    //   4. ao = 1 - moyenne_occlusion × intensity
    // Coût : 9 textureLoad par pixel, négligeable sur GPU moderne.
    // Skip sur le sky (z >= 0.999) — pas d'AO sur l'horizon.
    let depth_dims = textureDimensions(depth_tex);
    let center_pixel = vec2<i32>(
        i32(in.uv.x * f32(depth_dims.x)),
        i32(in.uv.y * f32(depth_dims.y)),
    );
    let center_depth = textureLoad(depth_tex, center_pixel, 0);
    var ao_factor = 1.0;
    if (center_depth < 0.999) {
        // **Kernel SSAO précomputé** (v0.9.5++ perf) — `var` array
        // (pas `let`) pour permettre l'indexation runtime en WGSL.
        // Économise 16 trig ops/fragment vs le cos/sin original.
        var kernel: array<vec2<i32>, 8>;
        kernel[0] = vec2<i32>( 4,  0);
        kernel[1] = vec2<i32>( 3,  3);
        kernel[2] = vec2<i32>( 0,  4);
        kernel[3] = vec2<i32>(-3,  3);
        kernel[4] = vec2<i32>(-4,  0);
        kernel[5] = vec2<i32>(-3, -3);
        kernel[6] = vec2<i32>( 0, -4);
        kernel[7] = vec2<i32>( 3, -3);
        var occlusion = 0.0;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let off = kernel[i];
            let sp = center_pixel + off;
            let sp_clamped = vec2<i32>(
                clamp(sp.x, 0, i32(depth_dims.x) - 1),
                clamp(sp.y, 0, i32(depth_dims.y) - 1),
            );
            let sample_depth = textureLoad(depth_tex, sp_clamped, 0);
            let delta = center_depth - sample_depth;
            if (delta > 0.0001 && delta < 0.005) {
                occlusion = occlusion + (1.0 - smoothstep(0.0, 0.005, delta));
            }
        }
        ao_factor = 1.0 - (occlusion / 8.0) * 0.55;
    }

    // **Vignette** — dégradé radial doux qui assombrit les coins.
    // Magnitude RÉDUITE 0.20 → 0.10 (v0.9.5++ polish) — le user trouvait
    // les coins trop sombres.  Reste perceptible pour donner du focus
    // optique sans étouffer les grandes maps.
    let vignette = 1.0 - smoothstep(0.35, 0.85, length(v) * 1.4) * 0.10;

    // **Lens flare ghosts** — 4 fantômes mirroir sur la ligne pixel→centre.
    // Chaque ghost = sample bloom_tex à une position décalée, avec
    // atténuation par luminance et fade vers les bords.
    let to_center = vec2<f32>(0.5, 0.5) - in.uv;
    var ghosts = vec3<f32>(0.0);
    // Ghost 1 : symétrique exact (offset = 1.0 → de l'autre côté du centre)
    let g1_uv = in.uv + to_center * 2.0;
    let g1_w = clamp(1.0 - length(g1_uv - vec2<f32>(0.5, 0.5)) * 1.6, 0.0, 1.0);
    ghosts = ghosts + textureSample(bloom_tex, samp, g1_uv).rgb * g1_w * 0.45
                    * vec3<f32>(1.0, 0.92, 0.82); // teinte chaude
    // Ghost 2 : 0.6× distance — proche du centre
    let g2_uv = in.uv + to_center * 1.6;
    let g2_w = clamp(1.0 - length(g2_uv - vec2<f32>(0.5, 0.5)) * 1.4, 0.0, 1.0);
    ghosts = ghosts + textureSample(bloom_tex, samp, g2_uv).rgb * g2_w * 0.30
                    * vec3<f32>(0.85, 1.0, 1.05); // teinte cyan
    // Ghost 3 : 0.3× — petit halo près du centre
    let g3_uv = in.uv + to_center * 1.3;
    let g3_w = clamp(1.0 - length(g3_uv - vec2<f32>(0.5, 0.5)) * 1.2, 0.0, 1.0);
    ghosts = ghosts + textureSample(bloom_tex, samp, g3_uv).rgb * g3_w * 0.20;
    // Ghost 4 : décalage opposé — fantôme externe
    let g4_uv = in.uv - to_center * 0.5;
    let g4_w = clamp(1.0 - length(g4_uv - vec2<f32>(0.5, 0.5)) * 1.8, 0.0, 1.0);
    ghosts = ghosts + textureSample(bloom_tex, samp, g4_uv).rgb * g4_w * 0.25
                    * vec3<f32>(1.05, 0.95, 0.85);
    // Halo radial : brouillard léger autour du centre quand bloom fort
    let center_bloom = textureSample(bloom_tex, samp, vec2<f32>(0.5, 0.5)).rgb;
    let halo = center_bloom * (1.0 - smoothstep(0.0, 0.6, length(v))) * 0.12;
    ghosts = ghosts + halo;
    // Atténuation globale : ghosts beaucoup plus subtils (× 0.30 vs 0.85
    // avant) pour ne pas saturer l'image — visibles seulement quand le
    // soleil est dans le FOV avec sources vives derrière.
    ghosts = ghosts * 0.30;

    // **Color grading** appliqué AVANT tonemap : on agit sur l'HDR
    // linéaire pour que le tonemap ACES écrase ensuite proprement les
    // hautes lumières teintées sans clipping fluo.
    let hdr_combined = (scene + bloom + ghosts + god_rays) * ao_factor * vignette;
    let graded = color_grade(hdr_combined);
    let mapped = aces_tonemap(graded);

    // **Film grain** appliqué APRÈS tonemap (en LDR sRGB).  Magnitude
    // RÉDUITE 0.025 → 0.008 (v0.9.5++ polish) — le user trouvait le
    // bruit trop présent.  Reste assez visible pour briser les bandes
    // de gradient sans se voir comme du noise.
    let grain_uv = in.uv * vec2<f32>(1920.0, 1080.0) + vec2<f32>(u_compose.time * 91.7, u_compose.time * 53.3);
    let grain = (hash21(grain_uv) - 0.5) * 0.008;
    let final_color = mapped + vec3<f32>(grain);

    return vec4<f32>(final_color, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn extract_uniform_size_aligned_16() { assert_eq!(std::mem::size_of::<ExtractUniform>(), 16); }
    #[test] fn blur_uniform_size_aligned_16() { assert_eq!(std::mem::size_of::<BlurUniform>(), 16); }
    #[test] fn compose_uniform_size_aligned_32() {
        // v0.9.5++ #12 : étendu à 32 bytes pour ajouter sun_uv (vec2)
        // + sun_visibility (f32).  Toujours aligné 16 bytes côté std140.
        assert_eq!(std::mem::size_of::<ComposeUniform>(), 32);
        assert_eq!(std::mem::size_of::<ComposeUniform>() % 16, 0);
    }
    #[test]
    fn bloom_constants_in_reasonable_ranges() {
        // En HDR le seuil PEUT dépasser 1.0 (vrai bloom physique).
        assert!(BLOOM_THRESHOLD > 0.5 && BLOOM_THRESHOLD < 5.0);
        assert!(BLOOM_INTENSITY > 0.0 && BLOOM_INTENSITY < 2.0);
        assert!((2..=8).contains(&BLOOM_DOWNSAMPLE));
    }
}
