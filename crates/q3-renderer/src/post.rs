//! Post-process : bloom additif sur la surface existante.
//!
//! # Approche
//!
//! Pas de pipeline HDR (refactoring trop large pour ce passage). On
//! reste sur la surface sRGB de la swapchain et on fait un *bloom LDR*
//! :
//!
//! 1. **Capture** : `copy_texture_to_texture` swapchain → `scene_capture`.
//! 2. **Bright extract** : passe fullscreen, lit `scene_capture`, écrit
//!    dans `bloom_a` (résolution /4) la part lumineuse > seuil.
//! 3. **Blur séparable** : H sur `bloom_a` → `bloom_b`, puis V sur
//!    `bloom_b` → `bloom_a`. Gauss 9-tap.
//! 4. **Composite** : passe fullscreen avec `BlendState::ADD`, lit
//!    `bloom_a` et empile additivement sur la surface.
//!
//! Le coût total est ~4 passes plein-écran à 1/16 du nombre de pixels
//! (parties blur), négligeable face au worldrender.
//!
//! # Limites connues
//!
//! * Pas de réelle dynamique HDR : les pixels saturent à 1.0 avant le
//!   bright-extract, donc le bloom est plus subtil qu'avec un vrai
//!   pipeline `Rgba16Float`. C'est l'effet "UE3-era" — agréable
//!   visuellement et fidèle à l'époque Q3-mod (q3map, OSP).
//! * Doit être inséré **entre la scène et le HUD** sinon le HUD
//!   "glow" — appel orchestré par `Renderer::render`.

use std::sync::Arc;
use wgpu::{Device, Queue, Texture, TextureFormat, TextureView};

/// Facteur de downsampling de la chaîne de blur — `/4` est un bon
/// compromis qualité/perf pour 1080p+ (240×135 sample buffers).
const BLOOM_DOWNSAMPLE: u32 = 4;

/// Seuil de luminance au-delà duquel un pixel contribue au bloom.
/// `0.85` : pas trop sensible (sinon tout glow), assez bas pour que
/// les ciels ensoleillés et les muzzle flashes sortent.
const BLOOM_THRESHOLD: f32 = 0.85;

/// Multiplicateur d'intensité du bloom au composite. `0.6` reste
/// discret — l'œil tolère mal un bloom dominant ; on rajoute, on
/// n'écrase pas la scène.
const BLOOM_INTENSITY: f32 = 0.6;

/// Uniforme passé au shader composite (juste l'intensité pour l'instant ;
/// élargir en cas d'ajout de tonemap).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ComposeUniform {
    intensity: f32,
    _pad: [f32; 3],
}

/// Uniforme du bright-extract : juste le seuil. Aligné std140.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ExtractUniform {
    threshold: f32,
    _pad: [f32; 3],
}

/// Uniforme du blur séparable : direction (1,0) ou (0,1) en pixels
/// inverses (offsets stockés directement pour le shader).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurUniform {
    /// `texel_size_x, texel_size_y` — UV step pour 1 pixel.
    texel: [f32; 2],
    /// `1, 0` ou `0, 1` selon l'axe de blur.
    direction: [f32; 2],
}

/// Stack de ressources post-process. Recréé sur resize via `resize`.
pub struct PostFx {
    device: Arc<Device>,
    queue: Arc<Queue>,
    surface_format: TextureFormat,

    // Textures.
    scene_capture: Texture,
    scene_capture_view: TextureView,
    bloom_a: Texture,
    bloom_a_view: TextureView,
    bloom_b: Texture,
    bloom_b_view: TextureView,

    // Échantillonneur partagé (linéaire/clamp).
    sampler: wgpu::Sampler,

    // Pipelines + bind groups + uniform buffers.
    extract_pipeline: wgpu::RenderPipeline,
    extract_uniform: wgpu::Buffer,
    extract_bg: wgpu::BindGroup,

    blur_pipeline: wgpu::RenderPipeline,
    blur_uniform_h: wgpu::Buffer,
    blur_uniform_v: wgpu::Buffer,
    blur_bg_h: wgpu::BindGroup, // lit bloom_a, blur H, écrit bloom_b
    blur_bg_v: wgpu::BindGroup, // lit bloom_b, blur V, écrit bloom_a

    compose_pipeline: wgpu::RenderPipeline,
    compose_uniform: wgpu::Buffer,
    compose_bg: wgpu::BindGroup,

    width: u32,
    height: u32,
}

impl PostFx {
    /// Crée la stack pour une surface `(width, height)` au format `surface_format`.
    /// Tous les pipelines sont compilés une fois ici — les passes
    /// runtime sont juste set_pipeline + set_bind_group + draw(3).
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

        let (scene_capture, scene_capture_view) =
            create_capture_texture(&device, surface_format, width, height);
        let bw = (width / BLOOM_DOWNSAMPLE).max(1);
        let bh = (height / BLOOM_DOWNSAMPLE).max(1);
        let (bloom_a, bloom_a_view) = create_bloom_texture(&device, surface_format, bw, bh, "bloom-a");
        let (bloom_b, bloom_b_view) = create_bloom_texture(&device, surface_format, bw, bh, "bloom-b");

        // Shader unique multi-entrypoint pour les 3 passes (extract / blur / composite).
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("postfx-shader"),
            source: wgpu::ShaderSource::Wgsl(POSTFX_WGSL.into()),
        });

        // Bind group layout commun : sampler + texture + uniform.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("postfx-bgl"),
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("postfx-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        // ---- Bright extract pipeline (écrit dans bloom_a) ----
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
            &device,
            &shader,
            &pipeline_layout,
            "fs_extract",
            surface_format,
            wgpu::BlendState::REPLACE,
            "postfx-extract",
        );
        let extract_bg = make_bind_group(
            &device,
            &bgl,
            &sampler,
            &scene_capture_view,
            &extract_uniform,
            "postfx-extract-bg",
        );

        // ---- Blur (réutilisé H et V via 2 uniformes différents) ----
        let blur_uniform_h = make_blur_uniform(&device, &queue, bw, bh, [1.0, 0.0], "blur-h-u");
        let blur_uniform_v = make_blur_uniform(&device, &queue, bw, bh, [0.0, 1.0], "blur-v-u");
        let blur_pipeline = make_pipeline(
            &device,
            &shader,
            &pipeline_layout,
            "fs_blur",
            surface_format,
            wgpu::BlendState::REPLACE,
            "postfx-blur",
        );
        // H : lit bloom_a (sortie de extract), écrit bloom_b
        let blur_bg_h = make_bind_group(
            &device,
            &bgl,
            &sampler,
            &bloom_a_view,
            &blur_uniform_h,
            "postfx-blur-h-bg",
        );
        // V : lit bloom_b (sortie de blur H), écrit bloom_a
        let blur_bg_v = make_bind_group(
            &device,
            &bgl,
            &sampler,
            &bloom_b_view,
            &blur_uniform_v,
            "postfx-blur-v-bg",
        );

        // ---- Composite pipeline (additive sur surface) ----
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
                _pad: [0.0; 3],
            }),
        );
        let compose_pipeline = make_pipeline(
            &device,
            &shader,
            &pipeline_layout,
            "fs_compose",
            surface_format,
            // ADD : finalColor = src.rgb + dst.rgb (alpha = dst.a). Mélange
            // additif standard pour bloom — empile la lumière sans
            // remplacer ce qui est déjà dessiné par la scène.
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent::REPLACE,
            },
            "postfx-compose",
        );
        let compose_bg = make_bind_group(
            &device,
            &bgl,
            &sampler,
            &bloom_a_view,
            &compose_uniform,
            "postfx-compose-bg",
        );

        Self {
            device,
            queue,
            surface_format,
            scene_capture,
            scene_capture_view,
            bloom_a,
            bloom_a_view,
            bloom_b,
            bloom_b_view,
            sampler,
            extract_pipeline,
            extract_uniform,
            extract_bg,
            blur_pipeline,
            blur_uniform_h,
            blur_uniform_v,
            blur_bg_h,
            blur_bg_v,
            compose_pipeline,
            compose_uniform,
            compose_bg,
            width,
            height,
        }
    }

    /// Recrée les textures et bind groups pour une nouvelle taille de
    /// surface. À appeler depuis `Renderer::resize` avant la 1re passe.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        let (sc, scv) = create_capture_texture(&self.device, self.surface_format, width, height);
        self.scene_capture = sc;
        self.scene_capture_view = scv;
        let bw = (width / BLOOM_DOWNSAMPLE).max(1);
        let bh = (height / BLOOM_DOWNSAMPLE).max(1);
        let (a, av) = create_bloom_texture(&self.device, self.surface_format, bw, bh, "bloom-a");
        let (b, bv) = create_bloom_texture(&self.device, self.surface_format, bw, bh, "bloom-b");
        self.bloom_a = a;
        self.bloom_a_view = av;
        self.bloom_b = b;
        self.bloom_b_view = bv;
        // Met à jour les blur uniforms (texel_size dépend de bw/bh).
        self.queue.write_buffer(
            &self.blur_uniform_h,
            0,
            bytemuck::bytes_of(&BlurUniform {
                texel: [1.0 / bw as f32, 1.0 / bh as f32],
                direction: [1.0, 0.0],
            }),
        );
        self.queue.write_buffer(
            &self.blur_uniform_v,
            0,
            bytemuck::bytes_of(&BlurUniform {
                texel: [1.0 / bw as f32, 1.0 / bh as f32],
                direction: [0.0, 1.0],
            }),
        );

        // Bind groups dépendent des views, à recréer.
        // Layout : on récupère via les pipelines.
        let bgl = self.extract_pipeline.get_bind_group_layout(0);
        self.extract_bg = make_bind_group(
            &self.device,
            &bgl,
            &self.sampler,
            &self.scene_capture_view,
            &self.extract_uniform,
            "postfx-extract-bg",
        );
        self.blur_bg_h = make_bind_group(
            &self.device,
            &bgl,
            &self.sampler,
            &self.bloom_a_view,
            &self.blur_uniform_h,
            "postfx-blur-h-bg",
        );
        self.blur_bg_v = make_bind_group(
            &self.device,
            &bgl,
            &self.sampler,
            &self.bloom_b_view,
            &self.blur_uniform_v,
            "postfx-blur-v-bg",
        );
        self.compose_bg = make_bind_group(
            &self.device,
            &bgl,
            &self.sampler,
            &self.bloom_a_view,
            &self.compose_uniform,
            "postfx-compose-bg",
        );

        self.width = width;
        self.height = height;
    }

    /// Applique bloom sur la surface : copy + extract + blur×2 + composite.
    /// Appelé entre la fin du rendu scène et le début du HUD.
    ///
    /// `surface_tex` doit avoir l'usage `COPY_SRC` (déjà configuré pour
    /// les screenshots — réutilisé tel quel ici).
    pub fn apply(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        surface_tex: &Texture,
        surface_view: &TextureView,
    ) {
        // 1. Copy surface → scene_capture (full size, format égal).
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: surface_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.scene_capture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        // 2. Bright-extract : scene_capture → bloom_a.
        run_fullscreen_pass(
            encoder,
            &self.bloom_a_view,
            &self.extract_pipeline,
            &self.extract_bg,
            "postfx-extract-pass",
        );
        // 3a. Blur H : bloom_a → bloom_b.
        run_fullscreen_pass(
            encoder,
            &self.bloom_b_view,
            &self.blur_pipeline,
            &self.blur_bg_h,
            "postfx-blur-h-pass",
        );
        // 3b. Blur V : bloom_b → bloom_a.
        run_fullscreen_pass(
            encoder,
            &self.bloom_a_view,
            &self.blur_pipeline,
            &self.blur_bg_v,
            "postfx-blur-v-pass",
        );

        // 4. Composite additif : bloom_a → surface.
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("postfx-compose-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.compose_pipeline);
        pass.set_bind_group(0, &self.compose_bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn create_capture_texture(
    device: &Device,
    format: TextureFormat,
    w: u32,
    h: u32,
) -> (Texture, TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("postfx-scene-capture"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn create_bloom_texture(
    device: &Device,
    format: TextureFormat,
    w: u32,
    h: u32,
    label: &str,
) -> (Texture, TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn make_blur_uniform(
    device: &Device,
    queue: &Queue,
    w: u32,
    h: u32,
    direction: [f32; 2],
    label: &str,
) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<BlurUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &buf,
        0,
        bytemuck::bytes_of(&BlurUniform {
            texel: [1.0 / w.max(1) as f32, 1.0 / h.max(1) as f32],
            direction,
        }),
    );
    buf
}

fn make_pipeline(
    device: &Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    fs_entry: &str,
    format: TextureFormat,
    blend: wgpu::BlendState,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: "vs_fullscreen",
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: fs_entry,
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn make_bind_group(
    device: &Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    view: &TextureView,
    uniform: &wgpu::Buffer,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform.as_entire_binding(),
            },
        ],
    })
}

fn run_fullscreen_pass(
    encoder: &mut wgpu::CommandEncoder,
    view: &TextureView,
    pipeline: &wgpu::RenderPipeline,
    bg: &wgpu::BindGroup,
    label: &str,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
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
// Fullscreen triangle (couvre [-1, 1]² avec un seul triangle géant).
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    // Trois sommets : (-1,-1), (3,-1), (-1,3) — couvrent l'écran avec
    // un seul triangle, économise un sommet vs un quad de 6.
    var x: f32 = -1.0 + f32((vid & 1u) << 2u);
    var y: f32 = -1.0 + f32((vid & 2u) << 1u);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;

struct ExtractU { threshold: f32, _pad: vec3<f32> };
struct BlurU { texel: vec2<f32>, direction: vec2<f32> };
struct ComposeU { intensity: f32, _pad: vec3<f32> };

// On utilise le même binding 2 pour les 3 shaders mais le type d'uniform
// diffère selon le pipeline. WGSL ne tolère pas un type ad-hoc, on
// déclare le triplet au plus large (4 floats) et on alias.
struct GenericU {
    a: f32,  // threshold | intensity
    b: f32,  // unused | direction.x
    c: f32,  // unused | direction.y
    d: f32,  // unused
};
@group(0) @binding(2) var<uniform> u_param: GenericU;

// --- Bright extract -----------------------------------------------------
@fragment
fn fs_extract(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(tex, samp, in.uv);
    // Luminance pondérée Rec.709.
    let lum = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    // Soft knee : autour du seuil, on fade en douceur plutôt qu'un cut
    // dur — sinon les arêtes "tournent au glow" de façon binaire et ça
    // se voit. Largeur 0.1.
    let knee = 0.1;
    let soft = clamp((lum - u_param.a + knee) / (2.0 * knee), 0.0, 1.0);
    let mult = soft * soft;
    return vec4<f32>(c.rgb * mult, 1.0);
}

// --- Gaussian blur séparable (9-tap symétrique) ------------------------
@fragment
fn fs_blur(in: VsOut) -> @location(0) vec4<f32> {
    // Direction est passée en `_pad` du même struct → reinterprété ici.
    // On lit `b, c` comme `direction.x, direction.y` et `a` comme
    // texel_size.x. Pour la propreté, on fait 2 uniforms distincts en Rust.
    // En WGSL on accepte le packing 'a' + paire (b,c).
    let texel_x = u_param.a;
    let texel_y = u_param.b;
    let dir = vec2<f32>(u_param.c, u_param.d);
    let off = vec2<f32>(texel_x, texel_y) * dir;
    // Coefficients gaussiens 9-tap (sigma ~ 2.0). Symétriques.
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

// --- ACES tonemap (Hill / Narkowicz fitted) -----------------------------
// Approximation analytique d'ACES Filmic — donne un look "cinéma"
// (rolloff naturel des highlights, saturation préservée). Appliqué
// sur la couleur avant écriture surface, ça compresse le bloom additif
// dans une plage SDR plus douce qu'un simple clamp à 1.0.
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

// --- Compose : additif sur la surface ----------------------------------
@fragment
fn fs_compose(in: VsOut) -> @location(0) vec4<f32> {
    let bloom = textureSample(tex, samp, in.uv).rgb * u_param.a;
    // Tonemap ACES sur le bloom seul. La surface en dessous a déjà
    // été tonemap-perçue par l'œil (sRGB 0..1) ; on évite le double-
    // mapping en appliquant ACES uniquement sur la contribution add
    // qu'on empile dessus, pour que les highlights ne saturent pas
    // brutalement à blanc.
    return vec4<f32>(aces_tonemap(bloom), 1.0);
}
"#;

#[cfg(test)]
mod tests {
    //! Pas de GPU ici — juste vérifier que les structs uniformes ont les
    //! bonnes tailles et que les constantes restent dans la fenêtre
    //! d'usage prévue. Le pipeline runtime est testé par smoke côté app.
    use super::*;

    #[test]
    fn extract_uniform_size_aligned_16() {
        // Std140 attend des alignements 16 octets pour un uniform buffer.
        assert_eq!(std::mem::size_of::<ExtractUniform>(), 16);
    }

    #[test]
    fn blur_uniform_size_aligned_16() {
        assert_eq!(std::mem::size_of::<BlurUniform>(), 16);
    }

    #[test]
    fn compose_uniform_size_aligned_16() {
        assert_eq!(std::mem::size_of::<ComposeUniform>(), 16);
    }

    #[test]
    fn bloom_constants_in_reasonable_ranges() {
        // Threshold > 0.5 (sinon glow partout) et < 1.0 (sinon rien
        // n'extrait jamais).
        assert!(BLOOM_THRESHOLD > 0.5 && BLOOM_THRESHOLD < 1.0);
        // Intensity : > 0 (sinon rien), < 2 (sinon screen blanchi).
        assert!(BLOOM_INTENSITY > 0.0 && BLOOM_INTENSITY < 2.0);
        // Downsample : 2..=8 raisonnable pour qualité/perf.
        assert!((2..=8).contains(&BLOOM_DOWNSAMPLE));
    }
}
