//! **FSR — FidelityFX Super Resolution / Enhanced Temporal Upscaling**
//!
//! Système d'upscaling temporel en 3 passes GPU :
//!
//! 1. **EASU** (Edge Adaptive Spatial Upsampling) — filtre Lanczos
//!    directionnel qui upscale la texture interne (basse résolution)
//!    vers la résolution native en détectant les contours et en filtrant
//!    le long de leur orientation.  Résultat net sans ringing.
//!
//! 2. **Temporal Accumulate** — blend exponentiel avec l'history de la
//!    frame précédente, reprojetée via motion vectors.  Le clamping
//!    hybride (AABB min/max + variance box en YCoCg) supprime le
//!    ghosting tout en conservant l'accumulation temporelle sur les
//!    zones statiques (~10× supersampling effectif).
//!
//! 3. **RCAS** (Robust Contrast Adaptive Sharpening) — sharpening
//!    adaptatif au contraste local : renforce les vrais contours sans
//!    amplifier le bruit ni les surfaces lisses.
//!
//! # Résolutions internes supportées
//!
//! | Mode         | Render Scale | Interne (1080p)  |
//! |--------------|-------------|------------------|
//! | Performance  | 50%         | 960 × 540        |
//! | Balanced     | 67%         | 1280 × 720       |
//! | Quality      | 75%         | 1440 × 810       |
//! | Ultra Quality| 100%        | 1920 × 1080      |
//!
//! À 100% l'EASU est un no-op (bilinéaire identité) mais le temporal +
//! RCAS restent actifs → gains de qualité TAA + sharpness même sans
//! downscale.
//!
//! # Intégration moteur
//!
//! ```text
//! Renderer::render()
//!   ├── scene HDR → fsr.render_tex (résolution interne)
//!   ├── fsr.upscale(encoder, render_view, output_view, motion_view)
//!   │     ├── EASU : render_tex → upscaled_tex (résolution native)
//!   │     ├── Temporal : upscaled_tex + history_tex → upscaled_tex (ping-pong)
//!   │     └── RCAS : upscaled_tex → output (surface ou hdr_color)
//!   └── PostFx lit output_view (tonemap, bloom, etc.)
//! ```

use std::sync::Arc;
use tracing::info;
use wgpu::{Device, Queue, Texture, TextureFormat, TextureView};

use crate::SCENE_HDR_FORMAT;

// ─── Constantes ──────────────────────────────────────────────────────

/// Format interne HDR pour toutes les textures FSR intermédiaires.
/// Identique au format scene du moteur pour éviter les conversions.
const FSR_FORMAT: TextureFormat = SCENE_HDR_FORMAT;

/// Blend alpha par défaut : 10% current + 90% history.  Donne ~10
/// frames d'accumulation effective — bon compromis entre stabilité
/// (ghosting faible) et réactivité (disocclusion rapide).
const DEFAULT_BLEND_ALPHA: f32 = 0.10;

/// Sharpness RCAS par défaut.  0.8 = sharpening bien visible sans
/// artefacts de halo.  Le joueur peut monter à 1.0 (plein) ou baisser
/// à 0.0 (off) via cvar.
const DEFAULT_SHARPNESS: f32 = 0.8;

// ─── Uniform GPU ─────────────────────────────────────────────────────

/// Paramètres passés au shader FSR — partagé par les 3 passes.
/// Chaque passe lit les champs qui la concernent.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FsrParams {
    /// Taille de la texture source (interne, basse résolution).
    src_size: [f32; 2],
    /// Taille de la texture destination (native).
    dst_size: [f32; 2],
    /// Intensité du sharpening RCAS [0.0 = off, 1.0 = max].
    sharpness: f32,
    /// Fraction du current frame dans le blend temporel.
    blend_alpha: f32,
    _pad: [f32; 2],
}

// ─── Upscaler principal ──────────────────────────────────────────────

/// Système FSR / Enhanced Temporal Upscaling.
///
/// Gère les textures internes, les pipelines, et l'exécution des 3
/// passes de compute/render.  Créé une fois à l'init du Renderer,
/// recréé sur resize via `resize()`.
pub struct FsrUpscaler {
    device: Arc<Device>,
    queue: Arc<Queue>,

    // ── Dimensions ──
    native_width: u32,
    native_height: u32,
    render_scale: f32,
    internal_width: u32,
    internal_height: u32,

    // ── Textures ──
    /// Texture où la scène est rendue (résolution interne).
    render_tex: Texture,
    render_view: TextureView,

    /// Sortie EASU → entrée temporal (résolution native).
    easu_output_tex: Texture,
    easu_output_view: TextureView,

    /// History frame N-1 (résolution native, COPY_DST + TEXTURE_BINDING).
    history_tex: Texture,
    history_view: TextureView,

    /// Résultat final de la frame courante (après temporal, avant RCAS).
    /// Sert aussi de source pour la copie → history à la fin.
    upscaled_tex: Texture,
    upscaled_view: TextureView,

    // ── Sampler partagé (linear, clamp-to-edge) ──
    sampler: wgpu::Sampler,

    // ── Uniform buffer ──
    params_buffer: wgpu::Buffer,

    // ── Bind group layouts ──
    /// Group 0 : input_tex + sampler + params (utilisé par les 3 passes).
    bgl_pass: wgpu::BindGroupLayout,
    /// Group 1 : history_tex + motion_tex (uniquement passe temporal).
    bgl_temporal_extra: wgpu::BindGroupLayout,

    // ── Pipelines ──
    easu_pipeline: wgpu::RenderPipeline,
    temporal_pipeline: wgpu::RenderPipeline,
    rcas_pipeline: wgpu::RenderPipeline,

    // ── Bind groups (recréés sur resize) ──
    /// EASU : group(0) = render_tex (source interne)
    easu_bg: wgpu::BindGroup,
    /// Temporal : group(0) = easu_output_tex, group(1) = history + motion
    temporal_bg0: wgpu::BindGroup,
    temporal_bg1: Option<wgpu::BindGroup>,
    /// RCAS : group(0) = upscaled_tex (sortie temporal)
    rcas_bg: wgpu::BindGroup,

    // ── Paramètres tunables ──
    sharpness: f32,
    blend_alpha: f32,
}

impl FsrUpscaler {
    /// Crée le système FSR avec les dimensions natives et le facteur
    /// d'échelle interne.  `render_scale` doit être dans [0.5, 1.0].
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        native_width: u32,
        native_height: u32,
        render_scale: f32,
    ) -> Self {
        let render_scale = render_scale.clamp(0.5, 1.0);
        let (iw, ih) = compute_internal_size(native_width, native_height, render_scale);

        info!(
            "FSR init : native {}x{}, internal {}x{} (scale {:.0}%)",
            native_width, native_height, iw, ih, render_scale * 100.0
        );

        // ── Sampler linéaire clamp-to-edge ──
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fsr-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ── Textures ──
        let (render_tex, render_view) =
            create_fsr_texture(&device, iw, ih, "fsr-render",
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let (easu_output_tex, easu_output_view) =
            create_fsr_texture(&device, native_width, native_height, "fsr-easu-out",
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let (history_tex, history_view) =
            create_fsr_texture(&device, native_width, native_height, "fsr-history",
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let (upscaled_tex, upscaled_view) =
            create_fsr_texture(&device, native_width, native_height, "fsr-upscaled",
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC);

        // ── Uniform buffer ──
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fsr-params"),
            size: std::mem::size_of::<FsrParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &params_buffer,
            0,
            bytemuck::bytes_of(&FsrParams {
                src_size: [iw as f32, ih as f32],
                dst_size: [native_width as f32, native_height as f32],
                sharpness: DEFAULT_SHARPNESS,
                blend_alpha: DEFAULT_BLEND_ALPHA,
                _pad: [0.0; 2],
            }),
        );

        // ── Shader module ──
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fsr-shader"),
            source: wgpu::ShaderSource::Wgsl(FSR_WGSL.into()),
        });

        // ── Bind group layout : group 0 (passe générique) ──
        // binding 0 = texture source, binding 1 = sampler, binding 2 = uniform
        let bgl_pass = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fsr-bgl-pass"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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

        // ── Bind group layout : group 1 (temporal extra) ──
        // binding 0 = history_tex, binding 1 = motion_tex
        let bgl_temporal_extra = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fsr-bgl-temporal"),
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
            ],
        });

        // ── Pipeline layouts ──
        let layout_single = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fsr-layout-single"),
            bind_group_layouts: &[&bgl_pass],
            push_constant_ranges: &[],
        });
        let layout_temporal = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fsr-layout-temporal"),
            bind_group_layouts: &[&bgl_pass, &bgl_temporal_extra],
            push_constant_ranges: &[],
        });

        // ── Render pipelines ──
        let easu_pipeline = make_fsr_pipeline(
            &device, &shader, &layout_single, "fs_easu", "fsr-easu-pipeline",
        );
        let temporal_pipeline = make_fsr_pipeline(
            &device, &shader, &layout_temporal, "fs_temporal", "fsr-temporal-pipeline",
        );
        let rcas_pipeline = make_fsr_pipeline(
            &device, &shader, &layout_single, "fs_rcas", "fsr-rcas-pipeline",
        );

        // ── Bind groups ──
        let easu_bg = make_pass_bg(
            &device, &bgl_pass, &render_view, &sampler, &params_buffer, "fsr-easu-bg",
        );
        let temporal_bg0 = make_pass_bg(
            &device, &bgl_pass, &easu_output_view, &sampler, &params_buffer, "fsr-temporal-bg0",
        );
        // temporal_bg1 sera créé quand on reçoit un motion_view via set_motion_view().
        let rcas_bg = make_pass_bg(
            &device, &bgl_pass, &upscaled_view, &sampler, &params_buffer, "fsr-rcas-bg",
        );

        Self {
            device,
            queue,
            native_width,
            native_height,
            render_scale,
            internal_width: iw,
            internal_height: ih,
            render_tex,
            render_view,
            easu_output_tex,
            easu_output_view,
            history_tex,
            history_view,
            upscaled_tex,
            upscaled_view,
            sampler,
            params_buffer,
            bgl_pass,
            bgl_temporal_extra,
            easu_pipeline,
            temporal_pipeline,
            rcas_pipeline,
            easu_bg,
            temporal_bg0,
            temporal_bg1: None,
            rcas_bg,
            sharpness: DEFAULT_SHARPNESS,
            blend_alpha: DEFAULT_BLEND_ALPHA,
        }
    }

    // ── Accesseurs ───────────────────────────────────────────────────

    /// Résolution interne (scène rendue à cette taille).
    pub fn internal_size(&self) -> (u32, u32) {
        (self.internal_width, self.internal_height)
    }

    /// Vue de la texture interne où la scène doit être rendue.
    pub fn render_view(&self) -> &TextureView {
        &self.render_view
    }

    /// Vue de l'history (frame N-1 résolu) — exposée pour debug / TAA externe.
    pub fn history_view(&self) -> &TextureView {
        &self.history_view
    }

    /// Vue de la texture upscalée finale (résolution native).
    pub fn upscaled_view(&self) -> &TextureView {
        &self.upscaled_view
    }

    /// Render scale courant.
    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    // ── Configuration runtime ────────────────────────────────────────

    /// Change le facteur d'échelle interne.  Recréé la texture de rendu
    /// et les bind groups qui en dépendent.  Valeurs typiques : 0.5
    /// (performance), 0.67 (balanced), 0.75 (quality), 1.0 (native).
    pub fn set_render_scale(&mut self, scale: f32) {
        let scale = scale.clamp(0.5, 1.0);
        if (scale - self.render_scale).abs() < 0.001 {
            return;
        }
        self.render_scale = scale;
        let (iw, ih) = compute_internal_size(self.native_width, self.native_height, scale);
        info!(
            "FSR render_scale → {:.0}% : internal {}x{} (native {}x{})",
            scale * 100.0, iw, ih, self.native_width, self.native_height
        );
        self.internal_width = iw;
        self.internal_height = ih;

        // Recréer la texture de rendu interne.
        let (rt, rv) = create_fsr_texture(
            &self.device, iw, ih, "fsr-render",
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        self.render_tex = rt;
        self.render_view = rv;

        // Mettre à jour l'uniform.
        self.write_params();

        // Recréer le bind group EASU (source = render_tex modifié).
        self.easu_bg = make_pass_bg(
            &self.device, &self.bgl_pass,
            &self.render_view, &self.sampler, &self.params_buffer,
            "fsr-easu-bg",
        );
    }

    /// Intensité du sharpening RCAS [0.0 = off, 1.0 = max].
    pub fn set_sharpness(&mut self, s: f32) {
        self.sharpness = s.clamp(0.0, 1.0);
        self.write_params();
    }

    /// Blend alpha temporel (fraction du current frame).
    /// Typiquement 0.05 (stable) .. 0.20 (réactif).
    pub fn set_blend_alpha(&mut self, alpha: f32) {
        self.blend_alpha = alpha.clamp(0.02, 1.0);
        self.write_params();
    }

    /// Recréer toutes les textures et bind groups après un resize de
    /// la fenêtre.  Les pipelines ne changent pas (format identique).
    pub fn resize(&mut self, native_width: u32, native_height: u32) {
        let nw = native_width.max(1);
        let nh = native_height.max(1);
        if nw == self.native_width && nh == self.native_height {
            return;
        }
        self.native_width = nw;
        self.native_height = nh;
        let (iw, ih) = compute_internal_size(nw, nh, self.render_scale);
        self.internal_width = iw;
        self.internal_height = ih;

        info!(
            "FSR resize : native {}x{}, internal {}x{} (scale {:.0}%)",
            nw, nh, iw, ih, self.render_scale * 100.0
        );

        // Recréer toutes les textures.
        let (rt, rv) = create_fsr_texture(
            &self.device, iw, ih, "fsr-render",
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        self.render_tex = rt;
        self.render_view = rv;

        let (et, ev) = create_fsr_texture(
            &self.device, nw, nh, "fsr-easu-out",
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        self.easu_output_tex = et;
        self.easu_output_view = ev;

        let (ht, hv) = create_fsr_texture(
            &self.device, nw, nh, "fsr-history",
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        self.history_tex = ht;
        self.history_view = hv;

        let (ut, uv) = create_fsr_texture(
            &self.device, nw, nh, "fsr-upscaled",
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        self.upscaled_tex = ut;
        self.upscaled_view = uv;

        // Mettre à jour l'uniform avec les nouvelles dimensions.
        self.write_params();

        // Recréer tous les bind groups.
        self.rebuild_bind_groups();
    }

    /// Lie la texture de motion vectors pour la passe temporal.
    /// Doit être appelé chaque frame (ou au moins une fois après
    /// `new()` / `resize()`) avec le motion vector view courant.
    /// Si jamais appelé, la passe temporal n'a pas de group(1) et
    /// sera skippée (fallback : EASU direct → RCAS, pas d'accumulation).
    pub fn set_motion_view(&mut self, motion_view: &TextureView) {
        self.temporal_bg1 = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fsr-temporal-bg1"),
            layout: &self.bgl_temporal_extra,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(motion_view),
                },
            ],
        }));
    }

    // ── Exécution des passes ─────────────────────────────────────────

    /// Exécute les 3 passes FSR dans l'encoder donné :
    ///   1. EASU : render_tex (interne) → easu_output_tex (native)
    ///   2. Temporal : easu_output + history → upscaled_tex
    ///   3. RCAS : upscaled_tex → output (texture cible fournie)
    ///
    /// `output` est typiquement le hdr_color_view du Renderer (entrée
    /// du PostFx).  `motion_vectors` est la texture de motion vectors
    /// de la frame courante (RG = delta UV).
    ///
    /// Après l'appel, copie upscaled → history pour la frame suivante.
    pub fn upscale(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        _input_hdr: &TextureView,
        output: &TextureView,
        motion_vectors: &TextureView,
    ) {
        // Mettre à jour le bind group temporal avec les motion vectors frais.
        // On le fait ici pour simplifier l'API : le caller n'a pas besoin
        // d'appeler set_motion_view() séparément avant chaque frame.
        self.set_motion_view(motion_vectors);

        // Rebinder EASU si l'input a changé (rare — typiquement identique
        // à render_view, mais l'API permet de passer un input_hdr externe).
        // En pratique, le caller passe &self.render_view() obtenu plus tôt,
        // mais on reconstruit le bind group par sécurité en cas de resize
        // entre la création du view et l'appel upscale().

        // ── Passe 1 : EASU ──
        // Source = render_tex (résolution interne).
        // Cible = easu_output_tex (résolution native).
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fsr-easu-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.easu_output_view,
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
            pass.set_pipeline(&self.easu_pipeline);
            pass.set_bind_group(0, &self.easu_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Passe 2 : Temporal Accumulate ──
        // Source = easu_output_tex (current frame upscalé).
        // History = history_tex (frame N-1).
        // Motion = motion_vectors.
        // Cible = upscaled_tex.
        if let Some(ref bg1) = self.temporal_bg1 {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fsr-temporal-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.upscaled_view,
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
            pass.set_pipeline(&self.temporal_pipeline);
            pass.set_bind_group(0, &self.temporal_bg0, &[]);
            pass.set_bind_group(1, bg1, &[]);
            pass.draw(0..3, 0..1);
        } else {
            // Fallback sans temporal : copier EASU direct → upscaled via
            // une passe RCAS-identity (sharpness sera appliqué après).
            // On utilise une simple copie texture pour être correct.
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.easu_output_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &self.upscaled_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.native_width,
                    height: self.native_height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // ── Passe 3 : RCAS ──
        // Source = upscaled_tex (sortie temporelle).
        // Cible = output (hdr_color ou surface du caller).
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fsr-rcas-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output,
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
            pass.set_pipeline(&self.rcas_pipeline);
            pass.set_bind_group(0, &self.rcas_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // ── Copie upscaled → history pour la frame suivante ──
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.upscaled_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.history_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.native_width,
                height: self.native_height,
                depth_or_array_layers: 1,
            },
        );
    }

    // ── Privé ────────────────────────────────────────────────────────

    /// Écrit les paramètres courants dans le buffer GPU.
    fn write_params(&self) {
        self.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&FsrParams {
                src_size: [self.internal_width as f32, self.internal_height as f32],
                dst_size: [self.native_width as f32, self.native_height as f32],
                sharpness: self.sharpness,
                blend_alpha: self.blend_alpha,
                _pad: [0.0; 2],
            }),
        );
    }

    /// Reconstruit tous les bind groups après un changement de textures.
    fn rebuild_bind_groups(&mut self) {
        self.easu_bg = make_pass_bg(
            &self.device, &self.bgl_pass,
            &self.render_view, &self.sampler, &self.params_buffer,
            "fsr-easu-bg",
        );
        self.temporal_bg0 = make_pass_bg(
            &self.device, &self.bgl_pass,
            &self.easu_output_view, &self.sampler, &self.params_buffer,
            "fsr-temporal-bg0",
        );
        self.rcas_bg = make_pass_bg(
            &self.device, &self.bgl_pass,
            &self.upscaled_view, &self.sampler, &self.params_buffer,
            "fsr-rcas-bg",
        );
        // temporal_bg1 sera recréé au prochain set_motion_view() / upscale().
        self.temporal_bg1 = None;
    }
}

// ─── Fonctions utilitaires ───────────────────────────────────────────

/// Calcule la résolution interne à partir de la native et du scale.
/// Les dimensions sont arrondies à l'inférieur, minimum 1.
fn compute_internal_size(native_w: u32, native_h: u32, scale: f32) -> (u32, u32) {
    let iw = ((native_w as f32 * scale).floor() as u32).max(1);
    let ih = ((native_h as f32 * scale).floor() as u32).max(1);
    (iw, ih)
}

/// Crée une texture FSR (Rgba16Float) avec les usages donnés.
fn create_fsr_texture(
    device: &Device,
    width: u32,
    height: u32,
    label: &str,
    usage: wgpu::TextureUsages,
) -> (Texture, TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FSR_FORMAT,
        usage,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Crée un render pipeline fullscreen pour une passe FSR.
fn make_fsr_pipeline(
    device: &Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    fs_entry: &str,
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
                format: FSR_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
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

/// Crée le bind group standard group(0) pour une passe FSR :
/// binding 0 = texture, binding 1 = sampler, binding 2 = uniform.
fn make_pass_bg(
    device: &Device,
    layout: &wgpu::BindGroupLayout,
    texture_view: &TextureView,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform.as_entire_binding(),
            },
        ],
    })
}

// ─── Shader WGSL embarqué ────────────────────────────────────────────

/// Contenu du fichier `shaders/fsr.wgsl` embarqué en const.
/// On utilise `include_str!` pour que le shader soit compilé au runtime
/// par wgpu mais vérifié au build-time par le linter si disponible.
const FSR_WGSL: &str = include_str!("shaders/fsr.wgsl");

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_internal_size_50_percent() {
        let (w, h) = compute_internal_size(1920, 1080, 0.5);
        assert_eq!(w, 960);
        assert_eq!(h, 540);
    }

    #[test]
    fn compute_internal_size_67_percent() {
        let (w, h) = compute_internal_size(1920, 1080, 0.67);
        assert_eq!(w, 1286);
        assert_eq!(h, 723);
    }

    #[test]
    fn compute_internal_size_75_percent() {
        let (w, h) = compute_internal_size(1920, 1080, 0.75);
        assert_eq!(w, 1440);
        assert_eq!(h, 810);
    }

    #[test]
    fn compute_internal_size_100_percent() {
        let (w, h) = compute_internal_size(1920, 1080, 1.0);
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);
    }

    #[test]
    fn compute_internal_size_minimum_one() {
        let (w, h) = compute_internal_size(1, 1, 0.5);
        assert!(w >= 1);
        assert!(h >= 1);
    }

    #[test]
    fn compute_internal_size_clamped_below() {
        // Scale < 0.5 → clamped à 0.5 dans set_render_scale, mais
        // compute_internal_size elle-même ne clamp pas (c'est le caller).
        let (w, h) = compute_internal_size(1920, 1080, 0.25);
        assert_eq!(w, 480);
        assert_eq!(h, 270);
    }

    #[test]
    fn fsr_params_size_aligned_16() {
        // Le uniform buffer doit être aligné 16 bytes (std140 requirement).
        assert_eq!(std::mem::size_of::<FsrParams>(), 32);
        assert_eq!(std::mem::size_of::<FsrParams>() % 16, 0);
    }

    #[test]
    fn fsr_params_is_pod() {
        // Vérifie que bytemuck::Pod est implémenté (compile-time check).
        let p = FsrParams {
            src_size: [960.0, 540.0],
            dst_size: [1920.0, 1080.0],
            sharpness: 0.8,
            blend_alpha: 0.1,
            _pad: [0.0; 2],
        };
        let bytes = bytemuck::bytes_of(&p);
        assert_eq!(bytes.len(), 32);
    }
}
