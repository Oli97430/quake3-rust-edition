//! Renderer **wgpu** + fenêtre **winit**.
//!
//! Remplace complètement le backend OpenGL 1.x du Q3 original par une API
//! moderne (Vulkan sur Linux/Windows, DirectX 12 sur Windows, Metal sur macOS).
//! Bénéfices concrets :
//!
//! * fini le fixed-function pipeline, tout passe par des shaders WGSL
//! * multithread safe (device + queue sont `Send + Sync`)
//! * meilleure utilisation du GPU (barriers explicites, moins de syncs CPU)
//! * cross-platform sans `#ifdef`

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(clippy::all)]

pub mod beam;
pub mod bsp_mesh;
pub mod camera;
pub mod cubemap;
pub mod decal;
pub mod dlight;
pub mod flare;
pub mod fog;
pub mod lightmap;
pub mod material;
pub mod md3;
pub mod particle;
pub mod post;
pub mod drone;
pub mod sky;
pub mod terrain;
pub mod text;

use bytemuck::{Pod, Zeroable};
use q3_bsp::Bsp;
use q3_common::{Error, Result};
use q3_math::{Angles, Vec3};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use wgpu::util::DeviceExt;
use winit::window::Window;

use self::beam::BeamRenderer;
use self::bsp_mesh::BspMesh;
use self::camera::Camera;
use self::decal::{Decal, DecalRenderer};
use self::dlight::{Dlight, DlightSet};
use self::flare::{Flare, FlareRenderer};
use self::fog::{FogSet, FogUniform, FogVolume};
use self::lightmap::LightmapArray;
use self::material::{MaterialCache, PipelineCache};
use self::md3::{Md3Gpu, Md3Renderer};
use self::particle::{Particle, ParticleRenderer};
use self::sky::SkyRenderer;
use self::text::TextRenderer;
use q3_filesystem::Vfs;
use q3_image::ImageCache;
use q3_shader::ShaderRegistry;

/// Format de surface demandé. Les drivers le fallback si besoin.
const PREFERRED_SURFACE_FORMATS: &[wgpu::TextureFormat] = &[
    wgpu::TextureFormat::Bgra8UnormSrgb,
    wgpu::TextureFormat::Rgba8UnormSrgb,
];

/// Format du depth buffer.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// **Cascaded Shadow Maps** (v0.9.5++ #7) — résolution de la shadow map
/// solaire.  2048² donne ~1 pixel par 0.4 unités Q3 dans une frustum
/// orthographique de ~800u, suffisant pour des contours nets sans
/// shimmering.  Format Depth32Float pour la précision dans les zones
/// proches (acne control).  Future itération : passer à 1024² ×
/// 2 cascades pour mieux gérer les distances.
pub const SHADOW_MAP_SIZE: u32 = 2048;

/// **Format HDR** de la passe scene. Tous les pipelines world/3D
/// (worldspawn, MD3, sky, particles, decals, beams, flares) ciblent
/// ce format au lieu de la surface sRGB. Permet de stocker des
/// luminances > 1.0 (muzzle flashes, particles bouillonnants, glow
/// du soleil) sans les écrêter avant le tonemap final.
///
/// `Rgba16Float` est universellement supporté par wgpu sur Vulkan /
/// DX12 / Metal. La surface finale reste en sRGB 8 bits — la passe
/// `tonemap_to_surface` convertit HDR → SDR via ACES.
pub const SCENE_HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Vertex GPU : reconstruit à partir d'un `DrawVert` BSP.
///
/// `lightmap_layer` : index de la couche dans le `texture_2d_array` de
/// lightmaps. Propagé par-vertex plutôt que par-drawcall pour permettre un
/// unique drawcall du monde.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_uv: [f32; 2],
    pub lightmap_uv: [f32; 2],
    pub color: [f32; 4],
    pub lightmap_layer: u32,
    pub _pad: u32,
}

impl GpuVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        0 => Float32x3, // position
        1 => Float32x3, // normal
        2 => Float32x2, // tex_uv
        3 => Float32x2, // lightmap_uv
        4 => Float32x4, // color
        5 => Uint32,    // lightmap_layer
    ];

    pub const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Contexte GPU principal.
/// État interne d'une capture d'écran en vol — créée dans `begin_capture`,
/// consommée par `finish_capture` dans la même frame. `buffer` contient la
/// copie GPU de la swapchain en attente de map côté CPU.
struct PendingCapture {
    buffer: wgpu::Buffer,
    path: PathBuf,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    format: wgpu::TextureFormat,
}

pub struct Renderer {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    depth_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,

    /// Texture HDR offscreen (Rgba16Float, taille surface). Toutes les
    /// passes scene/3D y dessinent. Au final de la frame, `tonemap.apply`
    /// lit cette texture + le bloom et écrit en sRGB sur la swapchain.
    /// Recréée sur resize. `view` cloné pour le bind dans PostFx.
    hdr_color_tex: wgpu::Texture,
    hdr_color_view: wgpu::TextureView,

    /// **Previous frame HDR** (v0.9.5++ #10) — copie de `hdr_color_tex`
    /// après le post-process de la frame précédente.  Lue par les
    /// shaders water_shade pour faire du Screen-Space Reflection.
    /// Le 1-frame lag est invisible sur l'eau (qui n'a pas de motion
    /// vectors disponibles côté joueur).  Recréée sur resize.
    prev_hdr_color_tex: wgpu::Texture,
    prev_hdr_color_view: wgpu::TextureView,
    /// Mirror du depth buffer, copié à la fin de chaque frame.  Lu par
    /// SSR water — évite le conflit usage RT vs RESOURCE qui se
    /// produirait si on bindait le `depth_view` courant pendant la
    /// passe qui écrit dedans.
    prev_depth_tex: wgpu::Texture,
    prev_depth_view: wgpu::TextureView,
    ssr_bgl: wgpu::BindGroupLayout,
    ssr_sampler: wgpu::Sampler,
    ssr_bind_group: wgpu::BindGroup,

    /// **Cascaded Shadow Maps** (v0.9.5++ #7) — texture depth-only
    /// rendue depuis le POV du soleil.  Lue par les shaders material/
    /// terrain/drone pour modulé la lumière directe par l'occlusion.
    /// 2048² Depth32Float = ~16 MB.  Re-rendue à chaque frame.
    /// (Tex+sampler retenus pour rester propriétaire de la ressource GPU
    /// même si seul `shadow_view` est utilisé directement par les passes.)
    #[allow(dead_code)]
    shadow_tex: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    /// Uniform contenant la matrice view_proj du soleil — utilisée par
    /// les shaders material pour transformer world_pos en shadow UV.
    /// Mise à jour chaque frame depuis la position joueur (frustum
    /// orthographique centré sur le joueur, taille fixe 1500u).
    shadow_uniform: wgpu::Buffer,
    shadow_bgl: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    shadow_sampler: wgpu::Sampler,
    /// Bind group complet (uniform + tex + sampler) — utilisé par les
    /// pipelines material pour SAMPLE la shadow map.
    shadow_bind_group: wgpu::BindGroup,
    /// Bind group minimal (uniform seul) — utilisé par le shadow pipeline
    /// pendant la passe d'écriture de la shadow map.  Doit être séparé
    /// pour ne pas inclure `shadow_view` qui est render target dans
    /// cette passe (conflit usage texture sample vs DEPTH_STENCIL_WRITE).
    #[allow(dead_code)]
    shadow_pass_bgl: wgpu::BindGroupLayout,
    shadow_pass_bind_group: wgpu::BindGroup,
    /// Pipeline depth-only pour la passe shadow.  Réutilise le vertex
    /// layout du BSP (`WorldVertex`) mais avec un fragment shader vide
    /// (juste depth-write).  Pas de fog/dlight/bloom/etc.
    shadow_pipeline: wgpu::RenderPipeline,

    /// **TAA — Temporal Anti-Aliasing** (v0.9.5++ #9).
    /// `taa_history_tex` : copie du résolu de la frame N-1 (sample-able).
    /// `taa_resolved_tex` : cible RT de la passe TAA resolve (frame N).
    /// Chaque frame : on rend la scène jittered dans `hdr_color`, puis
    /// la TAA resolve pass blend `hdr_color` (current) + `taa_history`
    /// (previous resolved) avec neighborhood clamping → écrit dans
    /// `taa_resolved`.  À la fin, copy `taa_resolved → taa_history` pour
    /// la frame suivante.  PostFx lit `taa_resolved` au lieu de `hdr_color`.
    taa_history_tex: wgpu::Texture,
    taa_history_view: wgpu::TextureView,
    taa_resolved_tex: wgpu::Texture,
    taa_resolved_view: wgpu::TextureView,
    taa_bgl: wgpu::BindGroupLayout,
    taa_bind_group: wgpu::BindGroup,
    taa_pipeline: wgpu::RenderPipeline,
    taa_sampler: wgpu::Sampler,
    /// Compteur de frame pour la séquence Halton(2,3).  Incrementé
    /// chaque render, modulo 8 (8 samples Halton donnent un pattern
    /// quasi-uniforme suffisant pour le supersampling temporel).
    taa_frame_index: u32,

    pipeline: wgpu::RenderPipeline,
    camera: Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_bgl: wgpu::BindGroupLayout,

    lightmap_bgl: wgpu::BindGroupLayout,
    lightmap_array: Option<LightmapArray>,
    lightmap_bind_group: Option<wgpu::BindGroup>,

    bsp_mesh: Option<BspMesh>,
    /// Pipeline terrain (BR maps).  `Some` quand un terrain a été
    /// uploadé via `upload_terrain`.  Mutuellement exclusif avec `bsp_mesh`
    /// côté usage : on ne dessine pas les deux dans la même frame.
    terrain_gpu: Option<terrain::TerrainGpu>,
    /// Pipeline drones GLB (BR aerial).  Mesh partagé entre tous les
    /// drones, instance uniform ré-écrit par drone.
    drone_gpu: Option<drone::DroneRenderer>,
    /// Transforms drones queues pour la frame courante.  Vidé après
    /// chaque flush dans la passe principale.  Le rendu drone se fait
    /// avec un seul instance uniform → on doit dessiner séquentiellement.
    pending_drones: Vec<drone::DroneInstance>,
    /// Props static GLB (rochers, plantes, palmiers...) — clé = nom du
    /// prop, +/= instance data.  Sera groupé par nom pour minimiser
    /// les changements de mesh à la frame.
    pending_props: Vec<(String, drone::DroneInstance)>,
    fog_set: FogSet,
    fog_uniform: FogUniform,

    material_cache: Option<MaterialCache>,
    pipeline_cache: Option<PipelineCache>,

    md3: Option<Md3Renderer>,
    sky: SkyRenderer,

    beam: BeamRenderer,
    decal: DecalRenderer,
    dlight: DlightSet,
    particle: ParticleRenderer,
    flare: FlareRenderer,

    text: TextRenderer,

    window: Arc<Window>,
    clear_color: wgpu::Color,

    /// Chemin de la prochaine capture d'écran à déclencher dans `render()`.
    /// Posé par `queue_screenshot`, consommé à la fin du prochain `render`.
    /// `None` = rien à faire, coût zéro sur le chemin chaud.
    pending_screenshot: Option<PathBuf>,

    /// Stack post-process (bloom additif). Inséré entre la fin du rendu
    /// scène et le HUD. Activable via `bloom_enabled` — si désactivé,
    /// l'objet reste alloué mais `apply` n'est pas appelée → coût zéro.
    post: post::PostFx,
    /// Toggle runtime du bloom (cvar `r_bloom`). Par défaut `true`.
    pub bloom_enabled: bool,
}

impl Renderer {
    /// Initialise wgpu avec la fenêtre donnée.
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();
        let size = (size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        // SAFETY : la Window est dans un Arc qui vivra aussi longtemps que la
        // Surface (Renderer détient les deux).
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| Error::renderer(format!("create_surface: {e}")))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| Error::renderer("aucun GPU adapter compatible"))?;

        info!(
            "gpu: {} ({:?}) backend={:?}",
            adapter.get_info().name,
            adapter.get_info().device_type,
            adapter.get_info().backend,
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("q3-device"),
                    required_features: wgpu::Features::empty(),
                    // `downlevel_defaults` plafonne à `max_bind_groups=4`.
                    // v0.9.5++ : on a 7 bind groups dans le pipeline material —
                    // camera (0), lightmap (1), material (2), dlights (3),
                    // fog (4), SSR (5), shadow CSM (6).  On bump à 8 (cap
                    // standard Vulkan/D3D12 desktop, supporté universellement).
                    required_limits: wgpu::Limits {
                        max_bind_groups: 8,
                        ..wgpu::Limits::downlevel_defaults()
                    }
                    .using_resolution(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| Error::renderer(format!("request_device: {e}")))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = PREFERRED_SURFACE_FORMATS
            .iter()
            .copied()
            .find(|f| caps.formats.contains(f))
            .unwrap_or(caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            // `COPY_SRC` permet de lire la texture de la swapchain pour les
            // captures d'écran (`queue_screenshot`) via
            // `copy_texture_to_buffer`. L'overhead est nul quand aucun
            // screenshot n'est pending : on ne copie rien.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            format,
            width: size.0,
            height: size.1,
            present_mode: caps
                .present_modes
                .iter()
                .copied()
                .find(|m| *m == wgpu::PresentMode::Mailbox)
                .unwrap_or(wgpu::PresentMode::Fifo),
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let (depth_tex, depth_view) = create_depth_view(&device, size.0, size.1);
        let (hdr_color_tex, hdr_color_view) =
            create_hdr_color_view(&device, size.0, size.1);
        let (prev_hdr_color_tex, prev_hdr_color_view) =
            create_prev_hdr_view(&device, size.0, size.1);
        let (prev_depth_tex, prev_depth_view) =
            create_prev_depth_view(&device, size.0, size.1);
        // Sampler SSR : Linear filtrant pour le HDR (interp.) +
        // ClampToEdge (UV en sortie de frustum → bord).
        let ssr_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssr-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // BGL SSR : prev_hdr (filterable) + depth (Depth) + sampler.
        let ssr_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ssr-bgl"),
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
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let ssr_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssr-bg"),
            layout: &ssr_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&prev_hdr_color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&prev_depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&ssr_sampler) },
            ],
        });

        // ─── **Cascaded Shadow Maps** (v0.9.5++ #7) ───
        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow-map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&wgpu::TextureViewDescriptor::default());
        // Uniform : 4×4 matrice du soleil (view × proj orthographique).
        let shadow_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow-uniform"),
            size: 64, // 4×4×f32
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Sampler comparison pour PCF — comparaison hardware avec bias.
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow-sampler-cmp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow-bgl"),
            entries: &[
                // Uniform sun_view_proj — accessible vertex (shadow pass)
                // ET fragment (sample côté material).
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Shadow texture (lu uniquement par les fragment shaders
                // côté material, la shadow pass écrit dedans en
                // RENDER_ATTACHMENT).
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow-bg"),
            layout: &shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: shadow_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&shadow_sampler) },
            ],
        });
        // BGL et bind group MINIMAUX pour la passe shadow elle-même.
        // Inclut UNIQUEMENT l'uniform (le shadow_view est déjà render
        // target → ne peut pas être bind comme texture en même temps).
        let shadow_pass_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow-pass-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let shadow_pass_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow-pass-bg"),
            layout: &shadow_pass_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: shadow_uniform.as_entire_binding() }],
        });
        // ─── **TAA Resolve pipeline** (v0.9.5++ #9) ───
        let (taa_history_tex, taa_history_view) =
            create_taa_history_view(&device, size.0, size.1);
        let (taa_resolved_tex, taa_resolved_view) =
            create_taa_resolved_view(&device, size.0, size.1);
        let taa_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("taa-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let taa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taa-bgl"),
            entries: &[
                // 0 : current frame HDR (jittered)
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
                // 1 : history HDR (frame N-1 résolu)
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
                // 2 : sampler partagé
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let taa_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taa-bg"),
            layout: &taa_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&taa_history_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&taa_sampler) },
            ],
        });
        // WGSL inline — TAA resolve avec neighborhood clamping (variance
        // clipping de Karis 2014).  Algorithme :
        //   1. Sample current(uv)
        //   2. Compute box AABB de 9 voisins du current (3×3)
        //   3. Sample history(uv) puis clip dans la box
        //   4. Blend 0.92 * clamped_history + 0.08 * current
        // Le clamp suppresse le ghosting des objets en mouvement (sans
        // motion vectors — limitation acceptée).  Sur scène statique
        // l'accumulation 8-frames donne un effet supersampling 8x.
        let taa_shader_src = r#"
@group(0) @binding(0) var current_tex: texture_2d<f32>;
@group(0) @binding(1) var history_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    let x = -1.0 + f32((vid & 1u) << 2u);
    let y = -1.0 + f32((vid & 2u) << 1u);
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

@fragment
fn fs_taa(in: VsOut) -> @location(0) vec4<f32> {
    let dims = textureDimensions(current_tex);
    let texel = vec2<f32>(1.0 / f32(dims.x), 1.0 / f32(dims.y));
    let curr = textureSample(current_tex, samp, in.uv).rgb;
    // Sample history.
    let hist = textureSample(history_tex, samp, in.uv).rgb;
    // Neighborhood AABB 3×3 sur le current — limites min/max chrominance.
    var c_min = curr;
    var c_max = curr;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) { continue; }
            let off = vec2<f32>(f32(dx), f32(dy)) * texel;
            let s = textureSample(current_tex, samp, in.uv + off).rgb;
            c_min = min(c_min, s);
            c_max = max(c_max, s);
        }
    }
    // Clamp history dans la box du voisinage current → suppresse le
    // ghosting (samples qui dérivent trop sont reset).
    let hist_clamped = clamp(hist, c_min, c_max);
    // Blend exponentiel : history dominante (accumule), current
    // contribute juste assez pour rafraîchir + introduire sub-pixel.
    // Alpha 0.08 = ~12 frames d'accumulation effective.
    let blended = mix(hist_clamped, curr, 0.08);
    return vec4<f32>(blended, 1.0);
}
"#;
        let taa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("taa-shader"),
            source: wgpu::ShaderSource::Wgsl(taa_shader_src.into()),
        });
        let taa_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taa-pipeline-layout"),
            bind_group_layouts: &[&taa_bgl],
            push_constant_ranges: &[],
        });
        let taa_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("taa-pipeline"),
            layout: Some(&taa_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &taa_shader,
                entry_point: "vs_fullscreen",
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &taa_shader,
                entry_point: "fs_taa",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SCENE_HDR_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Pipeline shadow : vertex BSP only (depth write), fragment empty.
        let shadow_shader_src = r#"
struct SunU { view_proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> sun: SunU;
struct VsIn {
    @location(0) position: vec3<f32>,
};
@vertex
fn vs_main(in: VsIn) -> @builtin(position) vec4<f32> {
    return sun.view_proj * vec4<f32>(in.position, 1.0);
}
"#;
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow-pipeline-shader"),
            source: wgpu::ShaderSource::Wgsl(shadow_shader_src.into()),
        });
        let shadow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow-pipeline-layout"),
            bind_group_layouts: &[&shadow_pass_bgl],
            push_constant_ranges: &[],
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow-pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                // Use the same vertex layout as BSP rendering — only
                // position (location 0) is consumed, others ignored.
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
            },
            fragment: None, // depth-only
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Cull front faces — réduit le shadow acne classique
                // (le self-shadowing de la face avant). Convention
                // Q3 BSP : winding CCW pour les triangles "outside".
                cull_mode: Some(wgpu::Face::Front),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,        // bias en unités depth
                    slope_scale: 2.0,    // bias proportionnel à la pente
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Camera
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 64.0),
            Angles::ZERO,
            size.0 as f32 / size.1 as f32,
        );
        let camera_uniform = camera.uniform(0.0);
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera-uniform"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                // FRAGMENT inclus pour que le shader sky procédural puisse
                // lire `inv_view_proj_rot` et reconstruire une direction
                // monde par pixel. Les pipelines VERTEX-only (monde, beams,
                // md3) ne sont pas gênés — un bind visible à plus de stages
                // est toujours valide.
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/world.wgsl").into()),
        });

        let lightmap_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lightmap-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world-pipeline-layout"),
            bind_group_layouts: &[&camera_bgl, &lightmap_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("world-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[GpuVertex::layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                // Q3 BSP utilise un winding horaire (glFrontFace(GL_CW)).
                // Les faces "intérieures" des brushes sont stockées CW quand
                // on les regarde depuis l'intérieur de la room. Si on laisse
                // Ccw, tous les sols sont cullés comme back-faces et on ne
                // voit que les murs (faux positifs par hasard d'orientation).
                front_face: wgpu::FrontFace::Cw,
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
                    // Pipeline world : cible HDR pour préserver les
                    // luminances > 1 jusqu'au tonemap final.
                    format: SCENE_HDR_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        // Sub-renderers world/3D ciblent SCENE_HDR_FORMAT — les valeurs
        // HDR sont préservées jusqu'au tonemap. Le TextRenderer reste
        // sur le format surface (LDR sRGB) car le HUD est dessiné
        // APRÈS le tonemap, directement sur la swapchain.
        let text = TextRenderer::new(device.clone(), queue.clone(), format);
        let sky = SkyRenderer::new(device.clone(), &camera_bgl, SCENE_HDR_FORMAT);
        let beam = BeamRenderer::new(device.clone(), queue.clone(), &camera_bgl, SCENE_HDR_FORMAT);
        let decal = DecalRenderer::new(device.clone(), queue.clone(), &camera_bgl, SCENE_HDR_FORMAT);
        let dlight = DlightSet::new(&device, queue.clone());
        let fog_uniform = FogUniform::new(&device, queue.clone());
        let particle = ParticleRenderer::new(device.clone(), queue.clone(), &camera_bgl, SCENE_HDR_FORMAT);
        let flare = FlareRenderer::new(device.clone(), queue.clone(), &camera_bgl, SCENE_HDR_FORMAT);

        // Post-process : créé avec des clones pour pouvoir conserver
        // device/queue dans Renderer (qui les expose en `pub`). Le
        // bind group input HDR sera fait juste après, quand on aura
        // le `hdr_color_view`.
        let mut post = post::PostFx::new(
            device.clone(),
            queue.clone(),
            format,
            size.0,
            size.1,
        );
        // Post lit depuis taa_resolved (sortie TAA), pas hdr_color brut.
        post.set_hdr_input(&taa_resolved_view, &depth_view);

        Ok(Self {
            device,
            queue,
            surface,
            surface_config,
            depth_tex,
            depth_view,
            hdr_color_tex,
            hdr_color_view,
            prev_hdr_color_tex,
            prev_hdr_color_view,
            prev_depth_tex,
            prev_depth_view,
            ssr_bgl,
            ssr_sampler,
            ssr_bind_group,
            shadow_tex,
            shadow_view,
            shadow_uniform,
            shadow_bgl,
            shadow_sampler,
            shadow_bind_group,
            shadow_pass_bgl,
            shadow_pass_bind_group,
            shadow_pipeline,
            taa_history_tex,
            taa_history_view,
            taa_resolved_tex,
            taa_resolved_view,
            taa_bgl,
            taa_bind_group,
            taa_pipeline,
            taa_sampler,
            taa_frame_index: 0,
            pipeline,
            camera,
            camera_buffer,
            camera_bind_group,
            camera_bgl,
            lightmap_bgl,
            lightmap_array: None,
            lightmap_bind_group: None,
            bsp_mesh: None,
            terrain_gpu: None,
            drone_gpu: None,
            pending_drones: Vec::new(),
            pending_props: Vec::new(),
            fog_set: FogSet::default(),
            fog_uniform,
            material_cache: None,
            pipeline_cache: None,
            md3: None,
            sky,
            beam,
            decal,
            dlight,
            particle,
            flare,
            text,
            window,
            clear_color: wgpu::Color {
                r: 0.05,
                g: 0.06,
                b: 0.09,
                a: 1.0,
            },
            pending_screenshot: None,
            post,
            bloom_enabled: true,
        })
    }

    /// Construction synchrone (bloquante) — utile pour les tests / le main.
    pub fn new_blocking(window: Arc<Window>) -> Result<Self> {
        pollster::block_on(Self::new(window))
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    pub fn set_clear_color(&mut self, r: f64, g: f64, b: f64) {
        self.clear_color = wgpu::Color { r, g, b, a: 1.0 };
    }

    /// Charge une map BSP sur le GPU (remplace celle déjà chargée s'il y en a).
    pub fn upload_bsp(&mut self, bsp: &Bsp) -> Result<()> {
        let lightmap = LightmapArray::new(&self.device, &self.queue, bsp);
        let white_layer = lightmap.white_layer;

        let mesh = BspMesh::build(&self.device, bsp, white_layer)?;
        debug!(
            "renderer: upload BSP ({} verts, {} tris, {} draws, {} lightmap layers)",
            mesh.vertex_count, mesh.triangle_count, mesh.draw_count, lightmap.layer_count
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lightmap-bg"),
            layout: &self.lightmap_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&lightmap.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&lightmap.sampler),
                },
            ],
        });

        self.lightmap_array = Some(lightmap);
        self.lightmap_bind_group = Some(bind_group);
        self.bsp_mesh = Some(mesh);

        // Construit les volumes de brouillard. Si les matériaux sont déjà
        // attachés, on résout `fogparms` à la volée ; sinon le FogSet reste
        // avec des volumes sans parms et sera résolu dans `attach_materials`.
        let registry = self.material_cache.as_ref().map(|m| m.shader_registry());
        self.fog_set = FogSet::build(bsp, registry);

        // Les décales et dlights d'une ancienne map ne vivent pas dans la suivante.
        self.decal.clear();
        self.dlight.clear();
        self.particle.clear();

        // Flares embarqués dans la map (coronas sur lampes, textures
        // émissives…) — extraits une fois et conservés toute la partie.
        let flares = Flare::extract_from(bsp);
        self.flare.set_flares(flares);

        Ok(())
    }

    /// Volumes de brouillard de la map courante.
    pub fn fog_set(&self) -> &FogSet {
        &self.fog_set
    }

    /// Si `eye` est à l'intérieur d'un volume de brouillard, retourne-le.
    pub fn active_fog_at(&self, eye: Vec3) -> Option<&FogVolume> {
        self.fog_set.active_at(eye)
    }

    /// Active le système de matériaux (shader scripts + textures).
    ///
    /// Sans appel à cette méthode, le renderer utilise le pipeline "world"
    /// monolithique (une seule passe lightmap-only). Une fois les matériaux
    /// attachés, [`render`](Self::render) groupe les drawcalls par
    /// `BlendClass` et résout une texture diffuse par shader.
    pub fn attach_materials(&mut self, registry: ShaderRegistry, images: ImageCache) {
        let mat_cache = MaterialCache::new(
            self.device.clone(),
            self.queue.clone(),
            registry,
            images,
        );
        let pipe_cache = PipelineCache::new(
            self.device.clone(),
            include_str!("shaders/material.wgsl"),
            &self.camera_bgl,
            &self.lightmap_bgl,
            mat_cache.bind_group_layout(),
            &self.dlight.bind_group_layout,
            &self.fog_uniform.bind_group_layout,
            &self.ssr_bgl,
            &self.shadow_bgl,
            SCENE_HDR_FORMAT,
        );
        let md3 = Md3Renderer::new(
            self.device.clone(),
            self.queue.clone(),
            &self.camera_bgl,
            mat_cache.bind_group_layout(),
            SCENE_HDR_FORMAT,
        );
        debug!("renderer: matériaux attachés");
        self.material_cache = Some(mat_cache);
        self.pipeline_cache = Some(pipe_cache);
        self.md3 = Some(md3);
    }

    /// **Drone GLB** — upload du mesh partagé pour les drones BR.
    /// Crée le pipeline si absent puis charge le mesh.  Appelé une
    /// seule fois par session (le mesh est partagé).
    pub fn upload_drone_mesh(&mut self, mesh: &q3_model::glb::GlbMesh) {
        let dr = self.drone_gpu.get_or_insert_with(|| {
            drone::DroneRenderer::new(self.device.clone(), self.queue.clone(), &self.camera_bgl)
        });
        dr.upload_mesh(mesh);
    }

    /// Dessine un drone — appelé pendant la passe HDR scene.  Le
    /// caller fournit la transform monde + le tint.  No-op si le
    /// pipeline n'est pas créé / le mesh pas chargé.
    pub fn queue_drone(&mut self, model: [[f32; 4]; 4], tint: [f32; 4]) {
        if let Some(dr) = self.drone_gpu.as_ref() {
            if !dr.has_mesh() { return; }
            self.pending_drones.push(drone::DroneInstance { model, tint });
        }
    }

    /// Vérifie si le pipeline drone a un mesh chargé.
    pub fn drone_has_mesh(&self) -> bool {
        self.drone_gpu.as_ref().map_or(false, |d| d.has_mesh())
    }

    /// Radius natif (unités mesh) du drone GLB. `0.0` si non chargé.
    pub fn drone_radius(&self) -> f32 {
        self.drone_gpu.as_ref().map_or(0.0, |d| d.mesh_radius)
    }

    /// **Prop GLB nommé** — upload d'un mesh de décor (rocher, plante,
    /// etc.) sous une clé string. Réutilise le pipeline drone.
    pub fn upload_prop(&mut self, name: &str, mesh: &q3_model::glb::GlbMesh) {
        let dr = self.drone_gpu.get_or_insert_with(|| {
            drone::DroneRenderer::new(self.device.clone(), self.queue.clone(), &self.camera_bgl)
        });
        dr.upload_prop(name, mesh);
    }

    pub fn has_prop(&self, name: &str) -> bool {
        self.drone_gpu.as_ref().map_or(false, |d| d.has_prop(name))
    }

    /// Radius natif (unités mesh) du prop GLB identifié par `name`.
    /// `None` si le prop n'est pas chargé. Utile pour calculer un
    /// scale auto au call site (target_size_world / radius).
    pub fn prop_radius(&self, name: &str) -> Option<f32> {
        self.drone_gpu.as_ref().and_then(|d| d.prop_radius(name))
    }

    /// Queue une instance de prop nommé pour rendu cette frame.
    pub fn queue_prop(&mut self, name: &str, model: [[f32; 4]; 4], tint: [f32; 4]) {
        if let Some(dr) = self.drone_gpu.as_ref() {
            if !dr.has_prop(name) { return; }
            self.pending_props.push((name.to_string(), drone::DroneInstance { model, tint }));
        }
    }

    /// **Battle Royale** — upload d'un `Terrain` heightmap pour
    /// rendu hors-BSP.  Crée le pipeline terrain dédié et garde un
    /// `Arc<Terrain>` qui sera tickée chaque frame par la sélection
    /// LOD selon la position caméra.
    pub fn upload_terrain(&mut self, terrain: std::sync::Arc<q3_terrain::Terrain>) {
        let gpu = terrain::TerrainGpu::new(
            self.device.clone(),
            self.queue.clone(),
            &self.camera_bgl,
            terrain,
        );
        self.terrain_gpu = Some(gpu);
        // Le BSP devient inactif quand on bascule en mode terrain
        // (mutuellement exclusifs côté gameplay).
        self.bsp_mesh = None;

        // Si des fogs ont déjà été chargés par un précédent `upload_bsp`,
        // résout leurs fogparms maintenant qu'un registry shader est dispo.
        if !self.fog_set.is_empty() {
            if let Some(mats) = self.material_cache.as_ref() {
                self.fog_set.resolve_parms(mats.shader_registry());
            }
        }
    }

    /// Charge une cubemap de ciel depuis `env/<base_path>_{rt,lf,up,dn,ft,bk}.tga`
    /// et l'installe dans le `SkyRenderer`. Si une face manque, renvoie `Err`
    /// sans toucher à la cubemap actuelle.
    pub fn load_sky_cubemap(&mut self, vfs: &Vfs, base_path: &str) -> Result<()> {
        let Some(mats) = self.material_cache.as_ref() else {
            return Err(Error::renderer(
                "sky cubemap: matériaux non attachés (appelle attach_materials d'abord)",
            ));
        };
        let cube = cubemap::Cubemap::load(
            &self.device,
            &self.queue,
            vfs,
            mats.image_cache(),
            base_path,
        )?;
        self.sky.set_cubemap(Some(Arc::new(cube)));
        info!("sky cubemap: '{base_path}' activée");
        Ok(())
    }

    /// Retire la cubemap active (retombe sur le gradient procédural).
    pub fn clear_sky_cubemap(&mut self) {
        self.sky.set_cubemap(None);
    }

    /// Basename de la cubemap active, ou `None` si on est en mode procédural.
    pub fn active_sky_cubemap(&self) -> Option<&str> {
        self.sky.cubemap_base()
    }

    /// Accès au shader registry attaché (si `attach_materials` a été appelée).
    /// Utile pour que l'application résolve les shaders de ciel depuis le BSP.
    pub fn shader_registry(&self) -> Option<&ShaderRegistry> {
        self.material_cache.as_ref().map(|m| m.shader_registry())
    }

    /// Charge (ou récupère du cache) un MD3 depuis le VFS.
    pub fn load_md3(&mut self, vfs: &Vfs, path: &str) -> Result<Arc<Md3Gpu>> {
        let Some(md3) = self.md3.as_mut() else {
            return Err(Error::renderer("md3: matériaux non attachés"));
        };
        md3.load(vfs, path)
    }

    /// Queue une instance MD3 pour la frame en cours (frame 0 statique).
    pub fn draw_md3(&mut self, mesh: Arc<Md3Gpu>, transform: glam::Mat4, color: [f32; 4]) {
        if let Some(md3) = self.md3.as_mut() {
            md3.queue_world(mesh, transform, color, 0, 0, 0.0);
        }
    }

    /// Queue une instance MD3 *animée* : interpole entre `frame_a` et
    /// `frame_b` avec un facteur `lerp` dans `[0, 1]`. Les indices de frame
    /// hors-bornes sont clampés automatiquement au dessin.
    pub fn draw_md3_animated(
        &mut self,
        mesh: Arc<Md3Gpu>,
        transform: glam::Mat4,
        color: [f32; 4],
        frame_a: usize,
        frame_b: usize,
        lerp: f32,
    ) {
        if let Some(md3) = self.md3.as_mut() {
            md3.queue_world(mesh, transform, color, frame_a, frame_b, lerp);
        }
    }

    /// Queue un viewmodel : rendu dans une passe overlay dédiée avec depth
    /// rechargée à 1.0 pour qu'il ne soit pas clippé par la géométrie du
    /// niveau.
    pub fn draw_md3_viewmodel(
        &mut self,
        mesh: Arc<Md3Gpu>,
        transform: glam::Mat4,
        color: [f32; 4],
        frame_a: usize,
        frame_b: usize,
        lerp: f32,
    ) {
        if let Some(md3) = self.md3.as_mut() {
            md3.queue_overlay(mesh, transform, color, frame_a, frame_b, lerp);
        }
    }

    /// Calcule la transformation locale d'un tag MD3 (ex. `"tag_weapon"`),
    /// interpolée entre deux frames. Retourne `None` si le tag est absent.
    pub fn md3_tag_transform(
        &self,
        mesh: &Md3Gpu,
        frame_a: usize,
        frame_b: usize,
        lerp: f32,
        name: &str,
    ) -> Option<glam::Mat4> {
        mesh.tag_transform(frame_a, frame_b, lerp, name)
    }

    /// Prépare une nouvelle frame — remet à zéro les queues HUD / MD3. À
    /// appeler une fois par frame avant d'émettre du texte ou des MD3.
    pub fn begin_frame(&mut self) {
        let (w, h) = (self.surface_config.width, self.surface_config.height);
        self.text.begin_frame(w, h);
        if let Some(md3) = self.md3.as_mut() {
            md3.begin_frame();
        }
        self.beam.begin_frame();
    }

    /// Queue un faisceau monde `a → b` de couleur constante `color`.
    /// Rendu additif, depth-test mais pas de depth-write.
    pub fn push_beam(&mut self, a: Vec3, b: Vec3, color: [f32; 4]) {
        self.beam.push(a.to_array(), b.to_array(), color);
    }

    /// Spawne une décale (marque de tir / impact) sur la surface touchée.
    /// `normal` pointe hors de la surface — typiquement `trace.plane_normal`.
    /// `lifetime` en secondes ; `now` est l'horloge applicative (elle doit
    /// être la même que celle passée à [`prune_decals`](Self::prune_decals)
    /// et [`render_with_time`](Self::render_with_time)).
    pub fn spawn_decal(
        &mut self,
        center: Vec3,
        normal: Vec3,
        radius: f32,
        color: [f32; 4],
        now: f32,
        lifetime: f32,
    ) {
        self.decal.spawn(Decal {
            center,
            normal,
            radius,
            color,
            spawn_time: now,
            lifetime,
        });
    }

    /// Retire toutes les décales expirées (`age >= lifetime`).
    pub fn prune_decals(&mut self, now: f32) {
        self.decal.prune(now);
    }

    /// Supprime toutes les décales — à appeler au changement de map.
    pub fn clear_decals(&mut self) {
        self.decal.clear();
    }

    /// Nombre de décales actives (utile pour les tests / debug HUD).
    pub fn decal_count(&self) -> usize {
        self.decal.len()
    }

    /// Spawne une lumière dynamique (halo radial rocket / muzzle / explo).
    /// `radius` en unités Q3 (100–300 typique), `color` RGB normalisé,
    /// `intensity` multiplicateur (~1.0 muzzle, ~2.0 rocket, ~4.0 explo),
    /// `lifetime` en secondes.
    pub fn spawn_dlight(
        &mut self,
        center: Vec3,
        radius: f32,
        color: [f32; 3],
        intensity: f32,
        now: f32,
        lifetime: f32,
    ) {
        self.dlight.spawn(Dlight {
            center,
            radius,
            color,
            intensity,
            spawn_time: now,
            lifetime,
        });
    }

    /// Supprime toutes les dlights — à appeler au changement de map.
    pub fn clear_dlights(&mut self) {
        self.dlight.clear();
    }

    /// Nombre de dlights actives cette frame (utile pour le HUD debug).
    pub fn dlight_count(&self) -> usize {
        self.dlight.len()
    }

    /// Spawne une particule billboard (smoke puff, gunspark, etc.).
    pub fn spawn_particle(
        &mut self,
        pos: Vec3,
        velocity: Vec3,
        color: [f32; 4],
        size_start: f32,
        size_end: f32,
        now: f32,
        lifetime: f32,
    ) {
        self.particle.spawn(Particle {
            pos,
            velocity,
            color,
            size_start,
            size_end,
            spawn_time: now,
            lifetime,
        });
    }

    /// Supprime toutes les particules — à appeler au changement de map.
    pub fn clear_particles(&mut self) {
        self.particle.clear();
    }

    /// Nombre de particules actives (utile HUD debug).
    pub fn particle_count(&self) -> usize {
        self.particle.len()
    }

    /// Nombre de flares BSP actifs dans la map courante.
    pub fn flare_count(&self) -> usize {
        self.flare.len()
    }

    /// Émet un billboard additif transitoire (muzzle flash, pop d'explosion).
    /// Vit une seule frame — l'appelant doit republier à chaque tick tant
    /// que l'effet doit rester visible.  `color` est RGBA linéaire ; le
    /// shader `flare.wgsl` module le cœur/halo radialement, donc l'alpha
    /// sert de gain global du sprite.  `radius` est en unités monde Q3
    /// (16u ≈ 0.4 m — calibrer à l'œil selon l'effet).
    pub fn push_muzzle_flash(&mut self, origin: Vec3, color: [f32; 4], radius: f32) {
        self.flare.push_dynamic(origin, color, radius);
    }

    /// Variante avec couleur différente aux deux extrémités (fade).
    pub fn push_beam_gradient(
        &mut self,
        a: Vec3,
        color_a: [f32; 4],
        b: Vec3,
        color_b: [f32; 4],
    ) {
        self.beam
            .push_gradient(a.to_array(), color_a, b.to_array(), color_b);
    }

    /// Queue un beam « lightning » zigzaguant entre `a` et `b`.  Les
    /// extrémités sont respectées ; les sommets intermédiaires sont
    /// dispersés dans le plan transverse d'une amplitude `jitter` (unités
    /// Q3).  `seed` pilote le motif — à faire évoluer chaque frame pour
    /// que l'arc crépite.
    pub fn push_beam_lightning(
        &mut self,
        a: Vec3,
        b: Vec3,
        color: [f32; 4],
        jitter: f32,
        segments: u32,
        seed: u64,
    ) {
        self.beam
            .push_lightning(a.to_array(), b.to_array(), color, jitter, segments, seed);
    }

    /// Queue un beam en spirale (hélice) autour de l'axe `a→b` — utilisé
    /// pour le trail Railgun.  Le rayon fade à 0 aux deux extrémités pour
    /// que la spirale naisse et meure proprement sur l'axe.
    pub fn push_beam_spiral(
        &mut self,
        a: Vec3,
        color_a: [f32; 4],
        b: Vec3,
        color_b: [f32; 4],
        radius: f32,
        turns: f32,
        segments: u32,
    ) {
        self.beam.push_spiral(
            a.to_array(),
            b.to_array(),
            color_a,
            color_b,
            radius,
            turns,
            segments,
        );
    }

    /// Alias historique — conservé pour ne pas casser l'appelant engine.
    pub fn begin_hud(&mut self) {
        self.begin_frame();
    }

    /// Émet un bloc de texte à la position pixel (coin haut-gauche).
    /// `scale=1.0` = 8×8 natif ; `scale=2.0` = 16×16 px, etc.
    pub fn push_text(&mut self, x: f32, y: f32, scale: f32, color: [f32; 4], text: &str) {
        self.text.push_text(x, y, scale, color, text);
    }

    /// Émet un rectangle plein de couleur `color` (RGBA).
    pub fn push_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        self.text.push_rect(x, y, w, h, color);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
        let (dtex, dview) = create_depth_view(&self.device, w, h);
        self.depth_tex = dtex;
        self.depth_view = dview;
        // Re-alloue le HDR target — il vit en parallèle de la swapchain.
        let (tex, view) = create_hdr_color_view(&self.device, w, h);
        self.hdr_color_tex = tex;
        self.hdr_color_view = view;
        // **Prev HDR + Prev Depth + SSR rebind** (v0.9.5++ #10).
        let (ptex, pview) = create_prev_hdr_view(&self.device, w, h);
        self.prev_hdr_color_tex = ptex;
        self.prev_hdr_color_view = pview;
        let (pdtex, pdview) = create_prev_depth_view(&self.device, w, h);
        self.prev_depth_tex = pdtex;
        self.prev_depth_view = pdview;
        self.ssr_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ssr-bg"),
            layout: &self.ssr_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.prev_hdr_color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.prev_depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.ssr_sampler) },
            ],
        });
        // **TAA rebind** (v0.9.5++ #9) — history + resolved nouveaux.
        let (htex, hview) = create_taa_history_view(&self.device, w, h);
        self.taa_history_tex = htex;
        self.taa_history_view = hview;
        let (rtex, rview) = create_taa_resolved_view(&self.device, w, h);
        self.taa_resolved_tex = rtex;
        self.taa_resolved_view = rview;
        self.taa_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taa-bg"),
            layout: &self.taa_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.hdr_color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.taa_history_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.taa_sampler) },
            ],
        });
        self.camera.set_aspect(w as f32 / h as f32);
        // Post-process : recrée scene_capture + bloom mip chain à la
        // nouvelle résolution. Fait une fois par resize, pas chaud.
        self.post.resize(w, h);
        // PostFx doit re-binder son input HDR sur le nouveau view.
        self.post.set_hdr_input(&self.taa_resolved_view, &self.depth_view);
    }

    /// Rend une frame. `now` est l'horloge applicative (secondes) utilisée
    /// par les effets temporels — fade des décales, fade d'intensité des
    /// dlights, etc.
    pub fn render(&mut self, now: f32) -> Result<()> {
        // Purge les effets expirés avant de construire les vertices /
        // l'uniform buffer dlight.
        self.decal.prune(now);
        self.dlight.prune(now);
        self.dlight.flush(now);
        self.particle.prune(now);

        // Met à jour le fog uniform : si l'œil est dans un volume, le shader
        // appliquera la formule `mix(fog_color, lit, exp(-d/distance))` ;
        // sinon `active=0` et c'est un no-op dans le fragment shader.
        let active_fog = self.fog_set.active_at(self.camera.position).cloned();
        self.fog_uniform.write(active_fog.as_ref());

        // **TAA jitter** (v0.9.5++ #9) — Halton(2,3) en NDC space.
        // Increment frame_index modulo 8 (8 samples Halton bien
        // distribués couvrent un pixel 8× → MSAA 8x effective).
        self.taa_frame_index = (self.taa_frame_index + 1) % 8;
        let h_idx = self.taa_frame_index + 1; // halton(0) = 0 → skip
        let jitter_x_unit = halton(h_idx, 2) - 0.5; // [-0.5, 0.5]
        let jitter_y_unit = halton(h_idx, 3) - 0.5;
        let w_f = self.surface_config.width.max(1) as f32;
        let h_f = self.surface_config.height.max(1) as f32;
        // Convert pixel → NDC : multiplier par 2/dim (NDC range = 2).
        self.camera.taa_jitter = [
            jitter_x_unit * 2.0 / w_f,
            jitter_y_unit * 2.0 / h_f,
        ];

        // update camera uniform (avec le jitter appliqué)
        let uniform = self.camera.uniform(now);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));

        // **BR terrain** — sélection LOD + génération chunks pour la
        // frame courante. Doit se faire AVANT le render pass car ça
        // crée des buffers (alloc GPU) hors-pass.
        if let Some(tg) = self.terrain_gpu.as_mut() {
            tg.update_chunks(self.camera.position);
        }

        // Pré-résolution des matériaux hors du render pass — ça termine le
        // borrow mutable sur `material_cache` / `pipeline_cache` avant
        // qu'on démarre le pass (qui a besoin de `self.camera_bind_group`
        // etc.).
        use self::material::BlendClass;
        let mut buckets: Option<[Vec<(Arc<material::Material>, u32, u32)>; 4]> = None;
        let mut pipelines: Option<[Arc<wgpu::RenderPipeline>; 4]> = None;
        if let (Some(mesh), Some(mats), Some(pipes)) = (
            self.bsp_mesh.as_ref(),
            self.material_cache.as_mut(),
            self.pipeline_cache.as_mut(),
        ) {
            let mut bucs: [Vec<(Arc<material::Material>, u32, u32)>; 4] =
                [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            for dr in mesh.draws.iter() {
                let mat = mats.resolve(&dr.shader_name);
                let slot = match mat.blend_class {
                    BlendClass::Opaque => 0,
                    BlendClass::AlphaTest => 1,
                    BlendClass::AlphaBlend => 2,
                    BlendClass::Additive => 3,
                };
                bucs[slot].push((mat, dr.first_index, dr.index_count));
            }
            pipelines = Some([
                pipes.get(BlendClass::Opaque),
                pipes.get(BlendClass::AlphaTest),
                pipes.get(BlendClass::AlphaBlend),
                pipes.get(BlendClass::Additive),
            ]);
            buckets = Some(bucs);
        }

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                warn!("surface lost/outdated — reconfigure");
                self.surface.configure(&self.device, &self.surface_config);
                return Ok(());
            }
            Err(e) => return Err(Error::renderer(format!("get_current_texture: {e}"))),
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // **HDR pipeline** : la scene est rendue dans `hdr_view`
        // (Rgba16Float, taille surface). Le tonemap final écrit en sRGB
        // sur `view` (swapchain). Le HUD est ensuite dessiné par-dessus
        // sur `view` directement (LDR, pas de bloom).
        let hdr_view = &self.hdr_color_view;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame-encoder"),
            });

        // ─── **Shadow pass** (v0.9.5++ #7 CSM) ───
        // Calcule la matrice sun_view_proj orthographique centrée sur
        // le joueur, puis rend les surfaces opaques BSP dans la
        // shadow map (depth-only).  Single cascade pour cette version.
        {
            // Direction du soleil (matche sky.wgsl + lib.rs god rays).
            let sun_dir = glam::Vec3::new(-0.4, -0.3, 0.86).normalize();
            // Centre du frustum shadow = position joueur (légèrement
            // décalée vers le sol pour mieux capter le terrain).
            let target = self.camera.position - glam::Vec3::Z * 32.0;
            // Position virtuelle du soleil très loin dans la direction
            // opposée à `sun_dir` (le soleil "envoie" la lumière dans
            // la direction -sun_dir vers le sol).
            let sun_pos = target + sun_dir * 4000.0;
            let sun_view = glam::Mat4::look_at_rh(sun_pos, target, glam::Vec3::Z);
            // Frustum orthographique : ~1500u de côté centré sur le
            // joueur, profondeur 8000u (couvre le terrain visible).
            let half_extent = 1500.0_f32;
            let near = 100.0_f32;
            let far = 8000.0_f32;
            let sun_proj = glam::Mat4::orthographic_rh(
                -half_extent, half_extent,
                -half_extent, half_extent,
                near, far,
            );
            let sun_view_proj = sun_proj * sun_view;
            self.queue.write_buffer(
                &self.shadow_uniform,
                0,
                bytemuck::cast_slice(&sun_view_proj.to_cols_array()),
            );
            // Render pass shadow.
            if let Some(mesh) = self.bsp_mesh.as_ref() {
                let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow-pass"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.shadow_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                shadow_pass.set_pipeline(&self.shadow_pipeline);
                shadow_pass.set_bind_group(0, &self.shadow_pass_bind_group, &[]);
                shadow_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                shadow_pass.set_index_buffer(
                    mesh.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                // Draw uniquement les surfaces OPAQUE (bucs[0]) — skip
                // transparent/additif qui ne devraient pas projeter
                // d'ombre franche.  Si pas encore de buckets (1ère
                // frame), draw tout le mesh par défaut.
                if let Some(bucs) = buckets.as_ref() {
                    for (_, first_idx, idx_count) in &bucs[0] {
                        shadow_pass.draw_indexed(
                            *first_idx..*first_idx + *idx_count,
                            0,
                            0..1,
                        );
                    }
                }
            }
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            if let (Some(mesh), Some(lm_bg)) =
                (self.bsp_mesh.as_ref(), self.lightmap_bind_group.as_ref())
            {
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_bind_group(1, lm_bg, &[]);
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                match (buckets.as_ref(), pipelines.as_ref()) {
                    (Some(bucs), Some(pipes)) => {
                        // Bind group dlight (slot 3) partagé par toutes les
                        // passes matériau : le buffer a déjà été flushé en
                        // début de `render` avec les lumières actives de
                        // cette frame.
                        pass.set_bind_group(3, self.dlight.bind_group(), &[]);
                        // Bind group fog (slot 4) : buffer mis à jour en
                        // début de frame selon la position de la caméra.
                        pass.set_bind_group(4, self.fog_uniform.bind_group(), &[]);
                        // Bind group SSR (slot 5) : prev_hdr + depth + sampler.
                        // Lu uniquement par water_shade — autres branches l'ignorent.
                        pass.set_bind_group(5, &self.ssr_bind_group, &[]);
                        // Bind group Shadow (slot 6) : sun_view_proj +
                        // shadow_tex + comparison sampler. Lu par toutes
                        // les branches material pour modulé l'éclairage
                        // direct par l'occlusion solaire.
                        pass.set_bind_group(6, &self.shadow_bind_group, &[]);
                        // Ordre Opaque → AlphaTest → AlphaBlend → Additive
                        // (approximation du `sort` Q3).
                        for (slot, pipeline) in pipes.iter().enumerate() {
                            if bucs[slot].is_empty() {
                                continue;
                            }
                            pass.set_pipeline(pipeline);
                            for (mat, first, count) in bucs[slot].iter() {
                                pass.set_bind_group(2, &mat.bind_group, &[]);
                                pass.draw_indexed(*first..*first + *count, 0, 0..1);
                            }
                        }
                    }
                    _ => {
                        // Fallback : pipeline monolithique lightmap-only
                        // (pas de dlights dans ce chemin — debug uniquement).
                        pass.set_pipeline(&self.pipeline);
                        pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }
            }

            // **Battle Royale terrain** — chunks LOD-adaptés, drawn
            // dans la même passe HDR que le BSP (mais en mode BR il
            // n'y a pas de BSP, donc on remplit l'écran). update_chunks
            // a déjà préparé les buffers via la position caméra du
            // début de frame (cf. début de `render`).
            if let Some(tg) = self.terrain_gpu.as_ref() {
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                tg.draw(&mut pass);
            }

            // **Drones GLB** — dessinés séquentiellement avec un seul
            // instance uniform ré-écrit entre chaque draw.  Pas
            // optimal sur 100+ instances mais largement suffisant
            // pour 4-8 drones planants.
            // **Drones / props GLB instanced** (v0.9.5+ refactor) —
            // Tout passe par un vertex buffer per-instance (step
            // Instance) avec un seul write par frame. Plus de bug
            // de uniform partagé. Les props sont groupés par nom
            // pour qu'on fasse 1 draw_indexed par mesh distinct.
            if !self.pending_drones.is_empty() || !self.pending_props.is_empty() {
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                // Group props par nom.
                let mut groups: hashbrown::HashMap<String, Vec<drone::DroneInstance>> =
                    hashbrown::HashMap::new();
                for (name, inst) in &self.pending_props {
                    groups.entry(name.clone()).or_default().push(*inst);
                }
                let groups_ref: Vec<(&str, &[drone::DroneInstance])> = groups
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_slice()))
                    .collect();
                if let Some(dr) = self.drone_gpu.as_mut() {
                    dr.flush_and_draw(&mut pass, &self.pending_drones, &groups_ref);
                }
            }
        }
        self.pending_drones.clear();
        self.pending_props.clear();

        // --- passe SKY : remplit les pixels restés à depth=1.0 (pas de
        // géométrie derrière). LessEqual + z=1.0 → ne touche QUE les trous.
        // Choix dynamique : cubemap si chargée, sinon gradient procédural.
        {
            let mut sky_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sky-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            sky_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            if let Some(cube_bg) = self.sky.cubemap_bind_group() {
                sky_pass.set_pipeline(self.sky.cubemap_pipeline());
                sky_pass.set_bind_group(1, cube_bg, &[]);
            } else {
                sky_pass.set_pipeline(self.sky.procedural_pipeline());
            }
            sky_pass.draw(0..3, 0..1);
        }

        // --- passe MD3 world : pickups, persos — depth load, alpha blend.
        if let (Some(md3), Some(mats)) = (self.md3.as_mut(), self.material_cache.as_mut()) {
            if md3.world_count() > 0 {
                let mut md3_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("md3-world-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                md3_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                md3.flush_world(&mut md3_pass, mats);
            }
        }

        // --- passe DECALS : marques de tir / explosion sur les surfaces.
        // Dessinée avant les beams pour que les faisceaux additifs
        // restent visibles par-dessus les décales alpha-blend.
        if !self.decal.is_empty() {
            let mut decal_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("decal-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            decal_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            self.decal.flush(&mut decal_pass, now);
        }

        // --- passe PARTICLES : puffs de fumée d'explosion, billboards
        // caméra-facing.  Dessinée après les décales (la fumée monte
        // au-dessus de la marque de brûlure) mais avant les beams
        // (les trails additifs restent lisibles par-dessus).
        if !self.particle.is_empty() {
            let basis = self.camera.angles.to_vectors();
            let mut part_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("particle-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            part_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            self.particle
                .flush(&mut part_pass, now, basis.right, basis.up);
        }

        // --- passe FLARES : coronas additifs sur les sources lumineuses
        // embarquées dans le BSP. Camera-facing billboards, blending
        // additif, depth-test sans depth-write.
        if !self.flare.is_empty() {
            let basis = self.camera.angles.to_vectors();
            let mut flare_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("flare-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            flare_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            self.flare
                .flush(&mut flare_pass, self.camera.position, basis.right, basis.up);
        }

        // --- passe BEAMS : faisceaux LG / trails RG. Depth test mais
        // pas de depth write, blending additif. Invisibles derrière les
        // murs, pas de z-fight avec d'autres beams.
        if !self.beam.is_empty() {
            let mut beam_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("beam-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            beam_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            self.beam.flush(&mut beam_pass);
        }

        // --- passe MD3 overlay : viewmodel — depth CLEAR à 1.0 pour
        // ne pas être clippé par le monde. Couleur conservée (load).
        if let (Some(md3), Some(mats)) = (self.md3.as_mut(), self.material_cache.as_mut()) {
            if md3.overlay_count() > 0 {
                let mut overlay_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("md3-overlay-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                overlay_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                md3.flush_overlay(&mut overlay_pass, mats);
            }
        }

        // --- passe POST-PROCESS : bloom additif sur la scène -------------
        // Insérée entre la fin du rendu 3D et le HUD pour que le HUD
        // n'hérite pas du glow (le texte HUD ne doit pas pulser comme
        // un néon — sauf demande explicite). En pipeline HDR, `apply`
        // fait le tonemap final → écrit la scène complète sur la
        // surface (remplacement total, pas additif). Donc OBLIGATOIRE
        // sinon la swapchain reste à zéro et le HUD apparaît seul.
        // **God rays setup** (v0.9.5++ #12) — projette la direction du
        // soleil en UV écran pour le raymarch radial dans fs_compose.
        // Direction monde du soleil : matche celle de sky.wgsl
        // (-0.4, -0.3, 0.86) pour cohérence visuelle.
        let sun_dir_world = glam::Vec3::new(-0.4, -0.3, 0.86).normalize();
        // On positionne un point soleil très loin dans cette direction
        // depuis la caméra ; clip-space division donne la projection.
        let sun_world =
            self.camera.position + sun_dir_world * 100_000.0;
        let view_proj = self.camera.view_proj();
        let sun_clip = view_proj * glam::Vec4::new(
            sun_world.x, sun_world.y, sun_world.z, 1.0,
        );
        let (sun_uv, sun_vis) = if sun_clip.w > 1e-3 {
            let ndc_x = sun_clip.x / sun_clip.w;
            let ndc_y = sun_clip.y / sun_clip.w;
            // NDC [-1, 1] → UV [0, 1], Y flip pour conv. texture.
            let u = ndc_x * 0.5 + 0.5;
            let v = -ndc_y * 0.5 + 0.5;
            // Visible : dans le frustum + pas trop loin au-delà des bords
            // (laisse une marge de 0.3 pour amorcer le raymarch même si
            // le soleil est légèrement off-screen).
            let in_screen = u > -0.3 && u < 1.3 && v > -0.3 && v < 1.3;
            (
                [u, v],
                if in_screen { 1.0 } else { 0.0 },
            )
        } else {
            // Soleil derrière la caméra → off.
            ([0.5, 0.5], 0.0)
        };
        // ─── **TAA Resolve pass** (v0.9.5++ #9) ───
        // Lit hdr_color (jittered) + taa_history → blend avec
        // neighborhood clamping → écrit dans taa_resolved.  Le post
        // pipeline lit ensuite taa_resolved au lieu de hdr_color
        // (rebind dynamique via set_hdr_input).
        {
            let mut taa_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("taa-resolve-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.taa_resolved_view,
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
            taa_pass.set_pipeline(&self.taa_pipeline);
            taa_pass.set_bind_group(0, &self.taa_bind_group, &[]);
            taa_pass.draw(0..3, 0..1);
        }

        // Push tous les paramètres dynamiques au compose shader.
        self.post.set_compose_params(now, sun_uv, sun_vis);
        self.post.apply(&mut encoder, &view);

        // --- passe HUD : load color, pas de depth, alpha-blending --------
        {
            let mut hud_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hud-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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
            self.text.flush(&mut hud_pass);
        }

        // Capture d'écran : si un chemin est en attente, on injecte une copie
        // de la swapchain vers un staging buffer *avant* de submit, puis on
        // mappe le buffer côté CPU et on écrit un TGA à la fin du `render`.
        // Le coût est nul quand rien n'est pending (le `Option::take` ne
        // déclenche aucune allocation).
        let capture = self
            .pending_screenshot
            .take()
            .map(|path| self.begin_capture(&mut encoder, &frame.texture, path));

        // **Prev-HDR copy** (v0.9.5++ #10) — sauvegarde le HDR de cette
        // frame pour que le water_shade de la frame suivante puisse y
        // raymarch les SSR.  Coût : 1 copy GPU plein écran (~0.1ms).
        let copy_size = wgpu::Extent3d {
            width: self.surface_config.width.max(1),
            height: self.surface_config.height.max(1),
            depth_or_array_layers: 1,
        };
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.hdr_color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.prev_hdr_color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            copy_size,
        );
        // Copy depth aussi pour SSR water hit-test next frame.
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.depth_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            wgpu::ImageCopyTexture {
                texture: &self.prev_depth_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::DepthOnly,
            },
            copy_size,
        );
        // **TAA history copy** (v0.9.5++ #9) — resolved → history pour
        // alimenter le blend de la frame suivante.
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.taa_resolved_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.taa_history_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            copy_size,
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        // La capture doit attendre que le GPU finisse la copie avant de
        // mapper. On poll synchrone (`Maintain::Wait`) — bloque ~1 frame au
        // pire, acceptable pour un shortcut F11 manuel. Toute erreur est
        // logguée sans faire échouer le rendu : un screenshot raté ne doit
        // jamais tuer la session de jeu en cours.
        if let Some(cap) = capture {
            if let Err(e) = self.finish_capture(cap) {
                error!("screenshot: {e}");
            }
        }

        Ok(())
    }

    /// Demande une capture d'écran au prochain frame — écrite en TGA 32-bit
    /// BGRA non compressé à `path`. Un nouvel appel *avant* le prochain
    /// `render` écrase la demande précédente (comportement "last-wins").
    pub fn queue_screenshot(&mut self, path: PathBuf) {
        self.pending_screenshot = Some(path);
    }

    /// Prépare la copy-to-buffer côté encoder + alloue le staging buffer.
    /// Appelée depuis `render()` quand une capture est pending.
    fn begin_capture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        path: PathBuf,
    ) -> PendingCapture {
        let w = self.surface_config.width;
        let h = self.surface_config.height;
        // wgpu exige que `bytes_per_row` soit aligné sur 256 octets. On
        // stocke la largeur "paddée" pour pouvoir déstripper la pad au
        // moment du write (copie par ligne).
        let bpp = 4u32;
        let unpadded = w * bpp;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let buffer_size = (padded as u64) * (h as u64);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot-staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        PendingCapture {
            buffer,
            path,
            width: w,
            height: h,
            padded_bytes_per_row: padded,
            format: self.surface_config.format,
        }
    }

    /// Mappe le staging buffer, écrit le TGA, détruit le buffer. L'appelant
    /// doit avoir submit l'encoder (sinon la copie n'a pas eu lieu et le
    /// `map_async` restera pendant indéfiniment).
    fn finish_capture(&self, cap: PendingCapture) -> Result<()> {
        let PendingCapture {
            buffer,
            path,
            width,
            height,
            padded_bytes_per_row,
            format,
        } = cap;
        // map_async → poll Wait → recv : on veut une API bloquante ici,
        // le shortcut F11 est intrinsèquement synchrone pour l'utilisateur.
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| Error::renderer(format!("screenshot map channel: {e}")))?
            .map_err(|e| Error::renderer(format!("screenshot map: {e}")))?;

        let view = slice.get_mapped_range();
        // Déstripage : on copie ligne par ligne en enlevant le padding, et
        // on convertit au passage vers BGRA peu importe l'ordre source.
        // TGA type 2 32-bit stocke en BGRA (B,G,R,A par pixel), avec un
        // bit 0x20 dans `image_descriptor` = origine haut-gauche (sinon
        // le viewer affichera l'image à l'envers).
        let row_bytes = (width * 4) as usize;
        let mut pixels = Vec::with_capacity(row_bytes * height as usize);
        let swap_rb = matches!(
            format,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
        );
        for row in 0..height as usize {
            let start = row * padded_bytes_per_row as usize;
            let end = start + row_bytes;
            let src = &view[start..end];
            if swap_rb {
                // source RGBA → dest BGRA
                for px in src.chunks_exact(4) {
                    pixels.push(px[2]);
                    pixels.push(px[1]);
                    pixels.push(px[0]);
                    pixels.push(px[3]);
                }
            } else {
                // source déjà BGRA — copie brute
                pixels.extend_from_slice(src);
            }
        }
        drop(view);
        buffer.unmap();

        // Construction header TGA (18 octets) + data.
        let mut out = Vec::with_capacity(18 + pixels.len());
        out.extend_from_slice(&[
            0, // id length
            0, // color map type
            2, // image type = uncompressed true-color
            0, 0, 0, 0, 0, // color map spec
            0, 0, // x origin
            0, 0, // y origin
        ]);
        out.extend_from_slice(&(width as u16).to_le_bytes());
        out.extend_from_slice(&(height as u16).to_le_bytes());
        out.push(32); // bits per pixel
        out.push(0x28); // bit 0x20 = top-left origin, 0x08 = 8 alpha bits
        out.extend_from_slice(&pixels);

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::renderer(format!("screenshot mkdir {}: {e}", parent.display())))?;
        }
        std::fs::write(&path, &out)
            .map_err(|e| Error::renderer(format!("screenshot write {}: {e}", path.display())))?;
        info!("screenshot: {} ({}×{})", path.display(), width, height);
        Ok(())
    }

    pub fn width(&self) -> u32 {
        self.surface_config.width
    }
    pub fn height(&self) -> u32 {
        self.surface_config.height
    }

    /// Projette un point monde Q3 vers des coordonnées écran (en pixels,
    /// origine haut-gauche, Y vers le bas comme pour le HUD).
    ///
    /// Retourne `None` si le point est derrière la caméra ou hors frustum
    /// horizontal/vertical. Sert aux overlays HUD qui suivent une entité
    /// 3D (chiffres de dégâts flottants, noms de joueurs, markers…).
    pub fn project_to_screen(&self, world: Vec3) -> Option<(f32, f32)> {
        let vp = self.camera.view_proj();
        // Homogène : w = clip_pos.w. On rejette tout ce qui est derrière le
        // near plane (w <= 0) — sinon on se retrouve à afficher des points
        // "miroir" derrière la caméra avec une position fausse.
        let clip = vp * world.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        // Clipping grossier en NDC : au-delà de ±1.2 on est très off-screen,
        // inutile de dessiner (on laisse un léger debord pour que le texte
        // qui sort du bord ne clignote pas).
        if !(-1.2..=1.2).contains(&ndc_x) || !(-1.2..=1.2).contains(&ndc_y) {
            return None;
        }
        let w = self.surface_config.width as f32;
        let h = self.surface_config.height as f32;
        // NDC Y va de -1 (bas) à +1 (haut) ; pixels vont de 0 (haut) à h (bas).
        let sx = (ndc_x * 0.5 + 0.5) * w;
        let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * h;
        Some((sx, sy))
    }
}

fn create_depth_view(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        // **TEXTURE_BINDING + COPY_SRC** (v0.9.5++) — TEXTURE_BINDING
        // pour SSAO post-process (lecture après main pass dans une
        // PASSE SÉPARÉE — OK).  COPY_SRC pour copier le depth dans
        // `prev_depth_tex` à la fin de chaque frame, qui sera lu par
        // les material shaders de la frame suivante (SSR water).
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// **Halton sequence** (base prime) — utilisé pour générer les jitters
/// sub-pixel TAA.  Halton(2,3) donne 8 samples bien distribués dans
/// [0,1]² qui couvrent uniformément la zone d'un pixel.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0_f32;
    let mut r = 0.0_f32;
    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// **TAA history texture** — Rgba16Float HDR pour préserver les
/// highlights pendant le blend.  COPY_DST + TEXTURE_BINDING.
fn create_taa_history_view(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("taa-history"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SCENE_HDR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// **TAA resolved texture** — RT de la passe TAA + COPY_SRC pour la
/// copie vers history à la fin.  Doit aussi être lue par PostFx.
fn create_taa_resolved_view(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("taa-resolved"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SCENE_HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// **Previous-frame depth** (v0.9.5++ #10) — destinée à recevoir la
/// copie du `depth` à la fin de chaque frame.  Lue par water_shade
/// SSR pour le hit-test ; évite le conflit RT-vs-RESOURCE qui se
/// produirait si on bindait le `depth_view` courant pendant qu'il
/// est utilisé comme depth attachment dans la même passe.
fn create_prev_depth_view(device: &wgpu::Device, w: u32, h: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("prev-depth"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Crée la texture HDR offscreen pour la passe scene. Usages :
/// `RENDER_ATTACHMENT` (write par les passes scene), `TEXTURE_BINDING`
/// (read par PostFx pour bright extract + tonemap composite).
fn create_hdr_color_view(
    device: &wgpu::Device,
    w: u32,
    h: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hdr-color"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SCENE_HDR_FORMAT,
        // **COPY_SRC ajouté** (v0.9.5++ #10 SSR) — la texture HDR est
        // copiée vers `prev_hdr_color` à la fin de chaque frame pour
        // alimenter le SSR de l'eau au prochain frame.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// **Previous-frame HDR** (v0.9.5++ #10) — texture destinée à recevoir
/// la copie du `hdr_color_tex` à la fin de chaque frame.  Lue par les
/// shaders water_shade pour SSR.  Doit avoir COPY_DST + TEXTURE_BINDING.
fn create_prev_hdr_view(
    device: &wgpu::Device,
    w: u32,
    h: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("prev-hdr-color"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SCENE_HDR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}
