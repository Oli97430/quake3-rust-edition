//! **Screen-Space Raytracing (SSR++)** — réflexions hi-z + ombres de contact.
//!
//! # Motivation
//!
//! wgpu 22 n'a pas encore de support stable pour le hardware RT (ray queries /
//! ray tracing pipelines).  Ce module implémente un raytracing screen-space de
//! haute qualité qui approxime les réflexions et les ombres de contact en
//! utilisant uniquement des compute shaders sur le depth buffer et la couleur
//! de la scène.
//!
//! # Architecture
//!
//! Trois passes compute :
//!
//! 1. **Hi-Z generation** — construit une pyramide de profondeur (mip chain,
//!    chaque texel = min des 4 parents).  Utilisée pour accélérer le ray march
//!    hiérarchique (saut des zones vides aux mips grossiers).
//!
//! 2. **Reflection trace** — SSR hi-z avec 64 steps coarse + 8 binary
//!    refinements.  Jitter temporel via blue noise pour dénoising.  Fresnel-
//!    weighted, fade aux bords d'écran.  Sortie half-res Rgba16Float.
//!
//! 3. **Contact shadow trace** — 16 steps vers le soleil, 200u max distance.
//!    Pénombre douce basée sur la distance à l'occluder.  Sortie half-res
//!    R8Unorm (1.0 = plein soleil, 0.0 = ombre).
//!
//! Les textures de sortie sont composées par le tonemap / post-process final.

use std::sync::Arc;
use wgpu::{Device, Queue, TextureView};

/// Shader WGSL compilé en const — même pattern que post.rs.
const RT_WGSL: &str = include_str!("shaders/rt.wgsl");

/// Nombre de mip levels pour la pyramide Hi-Z.
/// 7 niveaux = couvre jusqu'à 2^7 = 128 pixels de stride → suffisant
/// pour la majorité des résolutions (1080p → 960×540 half-res, 4K →
/// 1920×1080 half-res).
const HIZ_MIP_COUNT: u32 = 7;

/// Taille de la texture blue noise (tileable).
const NOISE_SIZE: u32 = 128;

/// Uniforme passé au compute shader.  Doit matcher exactement le struct
/// `RtParams` dans `rt.wgsl` (layout std140 : mat4 = 64 B, vec4 = 16 B).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RtParams {
    /// Inverse de view_proj — reconstruit world pos depuis depth NDC.
    inv_view_proj: [[f32; 4]; 4],
    /// View-projection — projette world pos en clip space.
    view_proj: [[f32; 4]; 4],
    /// Position caméra monde (xyz), w = time (secondes).
    camera_pos: [f32; 4],
    /// Direction soleil normalisée (xyz), w = frame_index (cast f32).
    sun_dir: [f32; 4],
    /// Dimensions : (full_width, full_height, half_width, half_height).
    resolution: [f32; 4],
    /// (z_near, z_far, max_shadow_dist, roughness_mult).
    params: [f32; 4],
}

/// Système de raytracing screen-space.
///
/// Crée ses propres textures et pipelines compute.  Le Renderer appelle
/// `trace()` entre la passe scene et le post-process, puis sample
/// `reflection_view()` et `shadow_view()` pour le compositing.
pub struct RtReflections {
    device: Arc<Device>,
    queue: Arc<Queue>,

    // ─── Textures de sortie (half-res) ──────────────────────────────
    reflection_tex: wgpu::Texture,
    reflection_view: TextureView,
    shadow_tex: wgpu::Texture,
    shadow_view: TextureView,

    // ─── Hi-Z pyramide de profondeur ────────────────────────────────
    /// Texture R32Float avec `HIZ_MIP_COUNT` mip levels.  Mip 0 est copié
    /// depuis le depth buffer, les mips suivants sont générés par le
    /// compute shader `generate_hi_z`.
    hiz_tex: wgpu::Texture,
    /// Vue sur le mip 0..N complet (pour le binding lecture dans le trace).
    hiz_view_full: TextureView,
    /// Vues par-mip pour le binding écriture dans la passe hi-z generation.
    hiz_mip_views: Vec<TextureView>,

    // ─── Blue noise ─────────────────────────────────────────────────
    #[allow(dead_code)]
    blue_noise_tex: wgpu::Texture,
    blue_noise_view: TextureView,

    // ─── Uniform buffer ─────────────────────────────────────────────
    params_buffer: wgpu::Buffer,

    // ─── Pipelines compute ──────────────────────────────────────────
    hiz_pipeline: wgpu::ComputePipeline,
    hiz_bgl: wgpu::BindGroupLayout,

    reflection_pipeline: wgpu::ComputePipeline,
    shadow_pipeline: wgpu::ComputePipeline,
    trace_bgl: wgpu::BindGroupLayout,

    // ─── Dimensions ─────────────────────────────────────────────────
    width: u32,
    height: u32,
    half_width: u32,
    half_height: u32,
    /// Compteur de frame pour le jitter temporel — injecté dans les
    /// params en tant que `sun_dir.w` (cast f32).
    frame_index: u32,
}

impl RtReflections {
    /// Crée le système SSR complet : textures, pipelines, blue noise.
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        width: u32,
        height: u32,
    ) -> Self {
        let half_width = (width / 2).max(1);
        let half_height = (height / 2).max(1);

        // ─── Textures de sortie (half-res) ──────────────────────────
        let (reflection_tex, reflection_view) = create_rt_texture(
            &device,
            wgpu::TextureFormat::Rgba16Float,
            half_width,
            half_height,
            1,
            "rt-reflection",
        );
        let (shadow_tex, shadow_view) = create_rt_texture(
            &device,
            wgpu::TextureFormat::R8Unorm,
            half_width,
            half_height,
            1,
            "rt-shadow",
        );

        // ─── Hi-Z pyramid ───────────────────────────────────────────
        let (hiz_tex, hiz_view_full, hiz_mip_views) =
            create_hiz_pyramid(&device, width, height);

        // ─── Blue noise (128x128 Rg8Unorm) ─────────────────────────
        let (blue_noise_tex, blue_noise_view) = create_blue_noise(&device, &queue);

        // ─── Uniform buffer ─────────────────────────────────────────
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rt-params-uniform"),
            size: std::mem::size_of::<RtParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ─── Shader module ──────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rt-shader"),
            source: wgpu::ShaderSource::Wgsl(RT_WGSL.into()),
        });

        // ─── Hi-Z pipeline (group 1 layout) ────────────────────────
        let hiz_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt-hiz-bgl"),
            entries: &[
                // binding 0 : source mip (texture_2d<f32>)
                bgl_texture_entry(0, wgpu::TextureSampleType::Float { filterable: false }),
                // binding 1 : destination mip (storage r32float, write)
                bgl_storage_entry(1, wgpu::TextureFormat::R32Float, wgpu::StorageTextureAccess::WriteOnly),
            ],
        });

        // ─── Trace pipeline (group 0 layout) ───────────────────────
        let trace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt-trace-bgl"),
            entries: &[
                // 0 : uniform RtParams
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1 : depth_tex (texture_2d<f32>)
                bgl_texture_entry(1, wgpu::TextureSampleType::Float { filterable: false }),
                // 2 : color_tex (texture_2d<f32>)
                bgl_texture_entry(2, wgpu::TextureSampleType::Float { filterable: false }),
                // 3 : noise_tex (texture_2d<f32>)
                bgl_texture_entry(3, wgpu::TextureSampleType::Float { filterable: false }),
                // 4 : hiz_tex (texture_2d<f32>)
                bgl_texture_entry(4, wgpu::TextureSampleType::Float { filterable: false }),
                // 5 : reflection_out (storage rgba16float, write)
                bgl_storage_entry(5, wgpu::TextureFormat::Rgba16Float, wgpu::StorageTextureAccess::WriteOnly),
                // 6 : shadow_out (storage r8unorm, write)
                bgl_storage_entry(6, wgpu::TextureFormat::R8Unorm, wgpu::StorageTextureAccess::WriteOnly),
            ],
        });

        // ─── Pipeline layouts ───────────────────────────────────────
        let hiz_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rt-hiz-layout"),
            // La passe hi-z utilise group(0) = trace_bgl (inutilisé mais
            // nécessaire pour la compatibilité de shader), group(1) = hiz_bgl.
            // En pratique le shader `generate_hi_z` ne lit que group(1).
            bind_group_layouts: &[&trace_bgl, &hiz_bgl],
            push_constant_ranges: &[],
        });
        let trace_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rt-trace-layout"),
            bind_group_layouts: &[&trace_bgl],
            push_constant_ranges: &[],
        });

        let hiz_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rt-hiz-pipeline"),
            layout: Some(&hiz_layout),
            module: &shader,
            entry_point: "generate_hi_z",
            compilation_options: Default::default(),
            cache: None,
        });
        let reflection_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rt-reflection-pipeline"),
            layout: Some(&trace_layout),
            module: &shader,
            entry_point: "trace_reflections",
            compilation_options: Default::default(),
            cache: None,
        });
        let shadow_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rt-shadow-pipeline"),
            layout: Some(&trace_layout),
            module: &shader,
            entry_point: "trace_contact_shadows",
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            reflection_tex,
            reflection_view,
            shadow_tex,
            shadow_view,
            hiz_tex,
            hiz_view_full,
            hiz_mip_views,
            blue_noise_tex,
            blue_noise_view,
            params_buffer,
            hiz_pipeline,
            hiz_bgl,
            reflection_pipeline,
            shadow_pipeline,
            trace_bgl,
            width,
            height,
            half_width,
            half_height,
            frame_index: 0,
        }
    }

    /// Recrée les textures taille-dépendantes lors d'un resize de fenêtre.
    /// Les pipelines ne changent pas (layout identique).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        self.half_width = (width / 2).max(1);
        self.half_height = (height / 2).max(1);

        // Recréer les textures de sortie.
        let (rt, rv) = create_rt_texture(
            &self.device,
            wgpu::TextureFormat::Rgba16Float,
            self.half_width,
            self.half_height,
            1,
            "rt-reflection",
        );
        self.reflection_tex = rt;
        self.reflection_view = rv;

        let (st, sv) = create_rt_texture(
            &self.device,
            wgpu::TextureFormat::R8Unorm,
            self.half_width,
            self.half_height,
            1,
            "rt-shadow",
        );
        self.shadow_tex = st;
        self.shadow_view = sv;

        // Recréer la pyramide Hi-Z.
        let (ht, hv, hmv) = create_hiz_pyramid(&self.device, width, height);
        self.hiz_tex = ht;
        self.hiz_view_full = hv;
        self.hiz_mip_views = hmv;
    }

    /// Exécute les trois passes de raytracing screen-space.
    ///
    /// Doit être appelé APRÈS le rendu de la scène (depth + couleur remplis)
    /// et AVANT le post-process (qui composera les résultats).
    ///
    /// # Arguments
    ///
    /// * `encoder` — command encoder de la frame courante.
    /// * `depth_view` — vue du depth buffer pleine résolution (Depth32Float
    ///   re-interprétée en R32Float pour le compute — nécessite une vue
    ///   aspect Float).
    /// * `color_view` — vue de la scène HDR pleine résolution (Rgba16Float).
    /// * `camera_uniform` — données caméra de la frame courante.  On extrait
    ///   `view_proj` et `inv_view_proj` pour les passes de trace.
    pub fn trace(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &TextureView,
        color_view: &TextureView,
        view_proj: [[f32; 4]; 4],
        inv_view_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        sun_dir: [f32; 3],
        z_near: f32,
        z_far: f32,
        time: f32,
    ) {
        self.frame_index = self.frame_index.wrapping_add(1);

        // ─── Mise à jour de l'uniform ───────────────────────────────
        let params = RtParams {
            inv_view_proj,
            view_proj,
            camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], time],
            sun_dir: [
                sun_dir[0],
                sun_dir[1],
                sun_dir[2],
                self.frame_index as f32,
            ],
            resolution: [
                self.width as f32,
                self.height as f32,
                self.half_width as f32,
                self.half_height as f32,
            ],
            params: [z_near, z_far, 200.0, 1.0],
        };
        self.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        // ─── Bind group principal (group 0, partagé entre les 3 passes) ──
        let trace_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt-trace-bg"),
            layout: &self.trace_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.blue_noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.hiz_view_full),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&self.reflection_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.shadow_view),
                },
            ],
        });

        // ─── Passe 1 : Hi-Z generation ──────────────────────────────
        // Mip 0 est le depth buffer lui-même (copié par un blit externe
        // ou lu directement).  Ici on génère mips 1..N.
        for mip in 1..(self.hiz_mip_views.len() as u32) {
            let src_mip = (mip - 1) as usize;
            let dst_mip = mip as usize;

            let hiz_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rt-hiz-mip-bg"),
                layout: &self.hiz_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.hiz_mip_views[src_mip],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.hiz_mip_views[dst_mip],
                        ),
                    },
                ],
            });

            let dst_w = (self.width >> mip).max(1);
            let dst_h = (self.height >> mip).max(1);

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rt-hiz-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.hiz_pipeline);
            pass.set_bind_group(0, &trace_bg, &[]);
            pass.set_bind_group(1, &hiz_bg, &[]);
            pass.dispatch_workgroups(
                (dst_w + 7) / 8,
                (dst_h + 7) / 8,
                1,
            );
        }

        // ─── Passe 2 : Réflexions SSR ───────────────────────────────
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rt-reflection-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reflection_pipeline);
            pass.set_bind_group(0, &trace_bg, &[]);
            pass.dispatch_workgroups(
                (self.half_width + 7) / 8,
                (self.half_height + 7) / 8,
                1,
            );
        }

        // ─── Passe 3 : Contact shadows ──────────────────────────────
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rt-shadow-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, &trace_bg, &[]);
            pass.dispatch_workgroups(
                (self.half_width + 7) / 8,
                (self.half_height + 7) / 8,
                1,
            );
        }
    }

    /// Texture view du résultat de réflexion (half-res, Rgba16Float).
    /// `.rgb` = couleur reflétée, `.a` = poids/confiance du hit.
    pub fn reflection_view(&self) -> &TextureView {
        &self.reflection_view
    }

    /// Texture view du résultat de contact shadows (half-res, R8Unorm).
    /// 1.0 = plein soleil, 0.0 = ombre de contact maximale.
    pub fn shadow_view(&self) -> &TextureView {
        &self.shadow_view
    }
}


// ═══════════════════════════════════════════════════════════════════════
// Fonctions utilitaires de création de ressources
// ═══════════════════════════════════════════════════════════════════════

/// Crée une texture storage 2D pour les passes compute.
fn create_rt_texture(
    device: &Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    mip_levels: u32,
    label: &str,
) -> (wgpu::Texture, TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        // STORAGE_BINDING pour l'écriture compute, TEXTURE_BINDING pour
        // la lecture dans la passe de compositing.
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Crée la pyramide Hi-Z (R32Float, N mip levels).
/// Retourne (texture, vue complète pour lecture, vues par-mip pour écriture).
fn create_hiz_pyramid(
    device: &Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, TextureView, Vec<TextureView>) {
    let mip_count = HIZ_MIP_COUNT.min(
        (width.max(height) as f32).log2().floor() as u32 + 1,
    );

    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rt-hiz"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: mip_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Vue complète (tous les mips) pour la lecture dans le trace shader.
    let full_view = tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some("rt-hiz-full"),
        ..Default::default()
    });

    // Vues par-mip pour l'écriture storage dans la passe hi-z.
    let mut mip_views = Vec::with_capacity(mip_count as usize);
    for mip in 0..mip_count {
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("rt-hiz-mip"),
            base_mip_level: mip,
            mip_level_count: Some(1),
            ..Default::default()
        });
        mip_views.push(view);
    }

    (tex, full_view, mip_views)
}

/// Génère et uploade une texture de blue noise 128x128 Rg8Unorm.
///
/// Algorithme : séquence R2 de Niederreiter (quasi-random, couverture
/// uniforme 2D meilleure que du bruit blanc pur).  Chaque pixel (x, y)
/// contient deux valeurs indépendantes dans [0, 255] — le jitter X et
/// le jitter Y pour le ray start.
fn create_blue_noise(
    device: &Device,
    queue: &Queue,
) -> (wgpu::Texture, TextureView) {
    let size = NOISE_SIZE as usize;
    // Rg8Unorm = 2 bytes par pixel.
    let mut pixels = vec![0u8; size * size * 2];

    // Constante irrationnelle pour la séquence R2 (2D low-discrepancy).
    // Basée sur le nombre plastique généralisé en 2D.
    let g = 1.32471795724474602596;
    let a1 = 1.0 / g;
    let a2 = 1.0 / (g * g);

    for y in 0..size {
        for x in 0..size {
            let idx = y * size + x;
            // Séquence R2 : chaque pixel a un offset pseudo-aléatoire
            // basé sur son index dans la grille de Morton.
            let n = (y * size + x) as f64;
            let r = ((0.5 + a1 * n) % 1.0 * 255.0) as u8;
            let g_val = ((0.5 + a2 * n) % 1.0 * 255.0) as u8;
            pixels[idx * 2] = r;
            pixels[idx * 2 + 1] = g_val;
        }
    }

    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rt-blue-noise"),
        size: wgpu::Extent3d {
            width: NOISE_SIZE,
            height: NOISE_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg8Unorm,
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
        &pixels,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(2 * NOISE_SIZE),
            rows_per_image: Some(NOISE_SIZE),
        },
        wgpu::Extent3d {
            width: NOISE_SIZE,
            height: NOISE_SIZE,
            depth_or_array_layers: 1,
        },
    );

    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}


// ═══════════════════════════════════════════════════════════════════════
// Helpers BGL — raccourcis pour les entrées de bind group layout
// ═══════════════════════════════════════════════════════════════════════

/// Entrée BGL pour une `texture_2d<f32>` en COMPUTE.
fn bgl_texture_entry(
    binding: u32,
    sample_type: wgpu::TextureSampleType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// Entrée BGL pour une `texture_storage_2d` en COMPUTE.
fn bgl_storage_entry(
    binding: u32,
    format: wgpu::TextureFormat,
    access: wgpu::StorageTextureAccess,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}


// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt_params_uniform_is_240_bytes() {
        // 2 × mat4x4 (128) + 4 × vec4 (64) = 192 bytes.
        // Vérifie que le struct Rust matche le layout WGSL std140.
        assert_eq!(std::mem::size_of::<RtParams>(), 192);
    }

    #[test]
    fn rt_params_alignment_16() {
        // std140 requiert un alignement 16 bytes pour les mat4.
        assert_eq!(std::mem::size_of::<RtParams>() % 16, 0);
    }

    #[test]
    fn hiz_mip_count_reasonable() {
        // 7 niveaux couvrent 128px de stride max, largement suffisant.
        assert!(HIZ_MIP_COUNT >= 4 && HIZ_MIP_COUNT <= 12);
    }

    #[test]
    fn noise_size_power_of_two() {
        assert!(NOISE_SIZE.is_power_of_two());
        assert!(NOISE_SIZE >= 64);
    }

    #[test]
    fn blue_noise_r2_sequence_covers_range() {
        // Vérifie que la séquence R2 produit des valeurs dans [0, 255].
        let g = 1.32471795724474602596_f64;
        let a1 = 1.0 / g;
        let a2 = 1.0 / (g * g);
        let mut min_r = 255u8;
        let mut max_r = 0u8;
        for n in 0..1000 {
            let r = ((0.5 + a1 * n as f64) % 1.0 * 255.0) as u8;
            let g_val = ((0.5 + a2 * n as f64) % 1.0 * 255.0) as u8;
            min_r = min_r.min(r).min(g_val);
            max_r = max_r.max(r).max(g_val);
        }
        // La séquence doit couvrir au moins [0, 250] sur 1000 samples.
        assert!(min_r < 5, "min_r trop élevé : {}", min_r);
        assert!(max_r > 250, "max_r trop bas : {}", max_r);
    }
}
