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
const ACES_EXPOSURE: f32 = 1.0;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ComposeUniform {
    intensity: f32,
    exposure: f32,
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
    surface_format: TextureFormat,

    // Bloom mip chain — `_a` capture le bright extract puis le blur V,
    // `_b` est l'intermédiaire blur H. Ping-pong.
    bloom_a: Texture,
    bloom_a_view: TextureView,
    bloom_b: Texture,
    bloom_b_view: TextureView,

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
        // Format bloom = HDR aussi pour préserver les valeurs > 1.0
        // pendant le blur. Sinon le blur clipperait à 1.0 et le composite
        // perdrait le côté "sur-exposé brûlant" du highlight.
        let bloom_format = crate::SCENE_HDR_FORMAT;
        let (bloom_a, bloom_a_view) = create_bloom_texture(&device, bloom_format, bw, bh, "bloom-a");
        let (bloom_b, bloom_b_view) = create_bloom_texture(&device, bloom_format, bw, bh, "bloom-b");

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
        // pour le compose : bloom_tex + compose_uniform. Le compose
        // pipeline a layout = [bgl_single, bgl_compose_extra]. Sépare
        // les bindings du chemin single (extract/blur) qui restent
        // sur group(0). Évite le conflit `@group(0) @binding(2)` qui
        // existait quand on avait UN seul bind group multi-rôles.
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
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
    /// * group(1) — compose-extra : bloom_tex + compose_uniform.
    pub fn set_hdr_input(&mut self, hdr_view: &TextureView) {
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
                wgpu::BindGroupEntry { binding: 1, resource: self.compose_uniform.as_entire_binding() },
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
        let bloom_format = crate::SCENE_HDR_FORMAT;
        let (a, av) = create_bloom_texture(&self.device, bloom_format, bw, bh, "bloom-a");
        let (b, bv) = create_bloom_texture(&self.device, bloom_format, bw, bh, "bloom-b");
        self.bloom_a = a;
        self.bloom_a_view = av;
        self.bloom_b = b;
        self.bloom_b_view = bv;
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
        self.blur_bg_h = make_bg_single(
            &self.device, &self.bgl_single, &self.sampler,
            &self.bloom_a_view, &self.blur_uniform_h, "blur-h-bg",
        );
        self.blur_bg_v = make_bg_single(
            &self.device, &self.bgl_single, &self.sampler,
            &self.bloom_b_view, &self.blur_uniform_v, "blur-v-bg",
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
    pub fn apply(&self, encoder: &mut wgpu::CommandEncoder, surface_view: &TextureView) {
        let (Some(extract_bg), Some((compose_g0, compose_g1))) =
            (&self.extract_bg, &self.compose_bg)
        else {
            return;
        };
        run_pass(encoder, &self.bloom_a_view, &self.extract_pipeline, extract_bg, "postfx-extract-pass");
        run_pass(encoder, &self.bloom_b_view, &self.blur_pipeline, &self.blur_bg_h, "postfx-blur-h-pass");
        run_pass(encoder, &self.bloom_a_view, &self.blur_pipeline, &self.blur_bg_v, "postfx-blur-v-pass");
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
struct ComposeUParam { intensity: f32, exposure: f32, p1: f32, p2: f32 };
@group(1) @binding(1) var<uniform> u_compose: ComposeUParam;

// --- ACES Filmic tonemap (Hill / Narkowicz fitted) ---
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

// --- Compose : HDR scene + bloom → tonemap → surface sRGB ---
//
// La surface est sRGB → wgpu fait le linear→sRGB encoding lors du write.
// On retourne donc des valeurs LINÉAIRES tonemappées.
//
// HDR = group(0) binding(1) (`tex`)
// bloom = group(1) binding(0) (`bloom_tex`)
// compose param = group(1) binding(1) (`u_compose`)
// — déclarés en haut du fichier, partagés avec les autres entrypoints
// du même module shader.
@fragment
fn fs_compose(in: VsOut) -> @location(0) vec4<f32> {
    let scene = textureSample(tex, samp, in.uv).rgb * u_compose.exposure;
    let bloom = textureSample(bloom_tex, samp, in.uv).rgb * u_compose.intensity;
    let mapped = aces_tonemap(scene + bloom);
    return vec4<f32>(mapped, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn extract_uniform_size_aligned_16() { assert_eq!(std::mem::size_of::<ExtractUniform>(), 16); }
    #[test] fn blur_uniform_size_aligned_16() { assert_eq!(std::mem::size_of::<BlurUniform>(), 16); }
    #[test] fn compose_uniform_size_aligned_16() { assert_eq!(std::mem::size_of::<ComposeUniform>(), 16); }
    #[test]
    fn bloom_constants_in_reasonable_ranges() {
        // En HDR le seuil PEUT dépasser 1.0 (vrai bloom physique).
        assert!(BLOOM_THRESHOLD > 0.5 && BLOOM_THRESHOLD < 5.0);
        assert!(BLOOM_INTENSITY > 0.0 && BLOOM_INTENSITY < 2.0);
        assert!((2..=8).contains(&BLOOM_DOWNSAMPLE));
    }
}
