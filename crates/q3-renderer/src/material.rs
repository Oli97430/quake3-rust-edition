//! **Materials** — pont entre les `.shader` Q3 et les pipelines wgpu.
//!
//! Stratégie :
//!
//! 1. On récupère un [`q3_shader::Shader`] depuis le `ShaderRegistry`
//!    (fallback = shader "générique lightmap-only").
//! 2. On choisit la première `Stage` qui référence une texture réelle
//!    (i.e. pas `$lightmap`) → c'est la texture diffuse.
//! 3. On détermine la **blend class** à partir du `BlendFunc` de la stage
//!    (opaque / alphatest / blend / add). Chaque classe a son propre
//!    `RenderPipeline`, créé à la volée et mis en cache.
//! 4. Le bind group du matériau contient **la texture diffuse** ; la
//!    lightmap reste partagée avec tout le monde via le bind group 1.
//!
//! # Limitations actuelles
//!
//! * Multi-pass (plusieurs stages superposés) pas supporté — on ne rend que
//!   la première stage significative. Pour Q3 base, ça couvre ~80% des
//!   matériaux (les autres étant des sky / fog / effets).
//! * `tcmod`, `rgbgen wave`, `deformVertexes` pas encore animés.

use crate::{GpuVertex, DEPTH_FORMAT};
use bytemuck::{Pod, Zeroable};
use hashbrown::HashMap;
use q3_common::Result;
use q3_image::{Image, ImageCache};
use q3_shader::{
    BlendFactor, BlendFunc, MapSource, ShaderRegistry,
    value::{DeformVertexes, RgbGen, TcMod, WaveFunc},
};
use std::sync::Arc;
use tracing::{debug, warn};

/// Mode `deformVertexes` encodé pour le shader.
pub mod deform_mode {
    /// Pas de déformation — position passée telle quelle.
    pub const NONE: f32 = 0.0;
    /// `deformVertexes wave <spread> <func> <base> <amp> <phase> <freq>` :
    /// déplace chaque vertex le long de sa normale par une wave spatialement
    /// déphasée (`phase += spread * (x+y+z)`).  Typique des flammes / plasma.
    pub const WAVE: f32 = 1.0;
    /// `deformVertexes move <dx> <dy> <dz> <func> ...` : déplace tous les
    /// vertices uniformément selon `dir` par la valeur de la wave.  Typique
    /// des bannières qui flottent au vent.
    pub const MOVE: f32 = 2.0;
}

/// Mode `rgbgen` encodé pour le shader (petite enum côté GPU).
pub mod rgb_mode {
    /// Aucun rgbgen — le shader garde sa logique par défaut (vertex color
    /// baked × lightmap × diffuse).
    pub const DEFAULT: f32 = 0.0;
    /// `rgbgen wave <func> <base> <amp> <phase> <freq>` — glow animé.
    pub const WAVE: f32 = 1.0;
    /// `rgbgen const ( r g b )` — couleur fixe qui remplace la vertex color.
    pub const CONST: f32 = 2.0;
}

/// WaveFunc encodé en u32 pour le shader.
pub mod wave_kind {
    pub const SIN: u32 = 0;
    pub const TRIANGLE: u32 = 1;
    pub const SQUARE: u32 = 2;
    pub const SAWTOOTH: u32 = 3;
    pub const INVERSE_SAWTOOTH: u32 = 4;
    pub const NOISE: u32 = 5;

    pub fn from(func: super::WaveFunc) -> u32 {
        use super::WaveFunc::*;
        match func {
            Sin => SIN,
            Triangle => TRIANGLE,
            Square => SQUARE,
            Sawtooth => SAWTOOTH,
            InverseSawtooth => INVERSE_SAWTOOTH,
            Noise => NOISE,
        }
    }
}

/// Params d'animation d'une texture + couleur + déformation de vertex,
/// passés au shader via un UBO 144 B par matériau.
///
/// * `anim` agrège les `tcmod scroll` (somme) et `tcmod rotate` (somme).
///   Les autres (`Scale`, `Stretch`, `Turb`, `Transform`) sont ignorés
///   pour l'instant — ça couvre les cas mono-tcmod (majorité).
/// * `rgb_info.x` est un mode énuméré (`rgb_mode::*`) qui indique au
///   fragment shader comment interpréter `rgb_wave` / `rgb_const` /
///   `wave_kind`.
/// * `deform_info.x` est un mode énuméré (`deform_mode::*`) qui indique
///   au vertex shader s'il faut déplacer la position.  `deform_info.y` =
///   spread (pour mode Wave).  `deform_dir` = direction (pour mode Move).
///   `deform_wave` + `deform_kind` = la waveform à évaluer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct MaterialParams {
    /// `.xy` = vitesses de scroll en UV/s ; `.z` = rotation rad/s ; `.w` pad.
    pub anim: [f32; 4],
    /// `.x` = mode rgbgen (voir `rgb_mode`) ; reste réservé.
    pub rgb_info: [f32; 4],
    /// Waveform pour `rgbgen wave` : `(base, amp, phase, freq)`.
    pub rgb_wave: [f32; 4],
    /// Couleur pour `rgbgen const` : `(r, g, b, _)`.
    pub rgb_const: [f32; 4],
    /// Discriminant WaveFunc pour `rgbgen wave` (voir `wave_kind`).
    pub wave_kind: [u32; 4],
    /// `.x` = mode deform (voir `deform_mode`) ; `.y` = spread (Wave).
    pub deform_info: [f32; 4],
    /// Direction pour `deformVertexes move` : `(dx, dy, dz, _)`.
    pub deform_dir: [f32; 4],
    /// Waveform pour deform : `(base, amp, phase, freq)`.
    pub deform_wave: [f32; 4],
    /// Discriminant WaveFunc pour la deform.
    pub deform_kind: [u32; 4],
}

impl MaterialParams {
    pub const ZERO: Self = Self {
        anim: [0.0; 4],
        rgb_info: [0.0; 4],
        rgb_wave: [0.0; 4],
        rgb_const: [0.0; 4],
        wave_kind: [0; 4],
        deform_info: [0.0; 4],
        deform_dir: [0.0; 4],
        deform_wave: [0.0; 4],
        deform_kind: [0; 4],
    };

    /// Agrège une liste de `TcMod` en params linéaires.
    pub fn from_tc_mods(tc_mods: &[TcMod]) -> Self {
        let mut me = Self::ZERO;
        me.set_tc_mods(tc_mods);
        me
    }

    fn set_tc_mods(&mut self, tc_mods: &[TcMod]) {
        let mut sx = 0.0f32;
        let mut sy = 0.0f32;
        let mut rot = 0.0f32;
        for m in tc_mods {
            match m {
                TcMod::Scroll(x, y) => {
                    sx += *x;
                    sy += *y;
                }
                TcMod::Rotate(deg_per_sec) => {
                    rot += deg_per_sec.to_radians();
                }
                _ => {}
            }
        }
        self.anim = [sx, sy, rot, 0.0];
    }

    /// Sélectionne le premier `DeformVertexes::Wave` ou `DeformVertexes::Move`
    /// de la liste et l'encode dans les champs deform.  Les autres variantes
    /// (`Normal`, `Bulge`, `AutoSprite*`) ne sont pas supportées pour l'instant —
    /// elles demandent des VS plus spécifiques ou un traitement CPU.
    fn set_deform_vertexes(&mut self, deforms: &[DeformVertexes]) {
        for d in deforms {
            match d {
                DeformVertexes::Wave { spread, wave } => {
                    self.deform_info[0] = deform_mode::WAVE;
                    self.deform_info[1] = *spread;
                    self.deform_wave = [wave.base, wave.amp, wave.phase, wave.freq];
                    self.deform_kind[0] = wave_kind::from(wave.func);
                    return;
                }
                DeformVertexes::Move { dir, wave } => {
                    self.deform_info[0] = deform_mode::MOVE;
                    self.deform_dir = [dir[0], dir[1], dir[2], 0.0];
                    self.deform_wave = [wave.base, wave.amp, wave.phase, wave.freq];
                    self.deform_kind[0] = wave_kind::from(wave.func);
                    return;
                }
                _ => continue,
            }
        }
    }

    fn set_rgb_gen(&mut self, rgb_gen: Option<&RgbGen>) {
        match rgb_gen {
            Some(RgbGen::Wave(w)) => {
                self.rgb_info[0] = rgb_mode::WAVE;
                self.rgb_wave = [w.base, w.amp, w.phase, w.freq];
                self.wave_kind[0] = wave_kind::from(w.func);
            }
            Some(RgbGen::Const(rgb)) => {
                self.rgb_info[0] = rgb_mode::CONST;
                self.rgb_const = [rgb[0], rgb[1], rgb[2], 0.0];
            }
            // Les autres variantes (`Identity`, `Vertex`, `Entity`,
            // `LightingDiffuse`…) sont déjà couvertes par la couleur de
            // vertex baked du mesh — on laisse le mode à `DEFAULT`.
            _ => {}
        }
    }

    /// Construit depuis une stage shader complète (tc_mods + rgb_gen) en
    /// y agrégeant aussi les `deformVertexes` du shader parent (commun à
    /// toutes les stages — c'est pour ça qu'il est passé à part).
    pub fn from_stage(
        tc_mods: &[TcMod],
        rgb_gen: Option<&RgbGen>,
        deforms: &[DeformVertexes],
    ) -> Self {
        let mut me = Self::ZERO;
        me.set_tc_mods(tc_mods);
        me.set_rgb_gen(rgb_gen);
        me.set_deform_vertexes(deforms);
        me
    }
}

/// Classification grossière d'une stage pour choisir un pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendClass {
    /// Opaque, pas de blending (cull/depth normal).
    Opaque,
    /// `alphaFunc GT0` — on discard les pixels alpha=0.
    AlphaTest,
    /// `blendFunc blend` ou équivalent `SRC_ALPHA/ONE_MINUS_SRC_ALPHA`.
    AlphaBlend,
    /// `blendFunc add` ou `GL_ONE / GL_ONE`.
    Additive,
}

impl BlendClass {
    fn from_stage(blend: Option<BlendFunc>, alpha_test: bool) -> Self {
        if alpha_test {
            return Self::AlphaTest;
        }
        let Some(BlendFunc::Custom(src, dst)) = blend else {
            return Self::Opaque;
        };
        use BlendFactor::*;
        match (src, dst) {
            (One, One) => Self::Additive,
            (SrcAlpha, OneMinusSrcAlpha) => Self::AlphaBlend,
            (One, Zero) => Self::Opaque,
            // Filter (= DstColor / Zero) : multiplication opaque, traitée
            // comme opaque pour l'instant.
            (DstColor, Zero) | (Zero, SrcColor) => Self::Opaque,
            _ => Self::AlphaBlend,
        }
    }
}

/// Un matériau résolu, prêt à être utilisé pour un drawcall.
pub struct Material {
    pub diffuse_view: wgpu::TextureView,
    pub diffuse_sampler: wgpu::Sampler,
    /// Petit UBO 16 B contenant les params d'animation (`scroll`, `rotate`).
    /// Gardé sur le Material pour que le bind group reste lié à un buffer
    /// vivant tant que le Material existe.
    pub params_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub blend_class: BlendClass,
    pub shader_name: String,
    pub anim_params: MaterialParams,
}

/// Cache de matériaux — crée et mémorise les Materials par nom de shader.
pub struct MaterialCache {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    registry: ShaderRegistry,
    images: ImageCache,
    material_bgl: wgpu::BindGroupLayout,
    by_name: HashMap<String, Arc<Material>>,
    fallback: Arc<Material>,
}

impl MaterialCache {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        registry: ShaderRegistry,
        images: ImageCache,
    ) -> Self {
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material-bgl"),
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
                // Params d'animation — lus en VERTEX (tcmod/deform) ET en
                // FRAGMENT (rgbgen wave, qui module la couleur finale).
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let fallback = make_fallback_material(&device, &queue, &material_bgl);

        Self {
            device,
            queue,
            registry,
            images,
            material_bgl,
            by_name: HashMap::new(),
            fallback: Arc::new(fallback),
        }
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.material_bgl
    }

    /// Accès au cache d'images (pour partager le parsing TGA/JPEG avec
    /// d'autres sous-systèmes, ex. la cubemap de ciel).
    pub fn image_cache(&self) -> &ImageCache {
        &self.images
    }

    /// Accès au registre de shaders .shader (utile à l'app pour retrouver
    /// le nom de skybox de la map courante).
    pub fn shader_registry(&self) -> &ShaderRegistry {
        &self.registry
    }

    /// Résout un shader par nom. Crée le matériau si absent. Toujours
    /// `Some(_)` car on a un fallback.
    pub fn resolve(&mut self, name: &str) -> Arc<Material> {
        let key = name.to_ascii_lowercase();
        if let Some(m) = self.by_name.get(&key) {
            return m.clone();
        }
        let mat = self.build(&key).unwrap_or_else(|| self.fallback.clone());
        self.by_name.insert(key, mat.clone());
        mat
    }

    fn build(&self, key: &str) -> Option<Arc<Material>> {
        // Deux chemins de résolution, dans l'ordre Q3 :
        //
        // 1. **Shader explicite** — le nom est déclaré dans un
        //    `scripts/*.shader` : on utilise la première stage texturée,
        //    avec ses tc_mods / rgb_gen / blend.
        //
        // 2. **Shader implicite** (le cas MAJORITAIRE en pratique) — le
        //    nom n'a aucun script associé et est simplement un chemin de
        //    texture que le BSP référence directement (ex.
        //    `textures/base_floor/diamond2c`).  Convention id : on charge
        //    l'image comme texture opaque, sans animation, sans clamp.
        //    Sans ce fallback, toute map ship vanilla rend 100 % en
        //    damier rose "texture manquante".
        if let Some(shader) = self.registry.get(key) {
            if let Some(stage) = shader
                .stages
                .iter()
                .find(|s| matches!(s.map, MapSource::Texture(_) | MapSource::Animated { .. }))
            {
                let tex_name: &str = match &stage.map {
                    MapSource::Texture(t) => t,
                    MapSource::Animated { frames, .. } => frames.first()?,
                    _ => return None,
                };
                return self.build_material(
                    key,
                    tex_name,
                    stage.clamp,
                    &stage.tc_mods,
                    stage.rgb_gen.as_ref(),
                    &shader.deform_vertexes,
                    stage.alpha_func.is_some(),
                    stage.blend,
                    shader.name.clone(),
                );
            }
        }
        // Pas de shader explicite utilisable → implicite.  On utilise le
        // key lui-même comme nom de texture et on assume les defaults Q3
        // (opaque, pas de clamp, pas de tcmod, pas de rgb_gen, pas de
        // deform, pas d'alphatest, blend = None).
        self.build_material(
            key,
            key,
            false,
            &[],
            None,
            &[],
            false,
            None,
            key.to_string(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_material(
        &self,
        key: &str,
        tex_name: &str,
        clamp: bool,
        tc_mods: &[TcMod],
        rgb_gen: Option<&RgbGen>,
        deforms: &[DeformVertexes],
        alpha_test: bool,
        blend: Option<BlendFunc>,
        shader_name: String,
    ) -> Option<Arc<Material>> {
        let base = strip_ext(tex_name);
        let img = match self.images.load(base) {
            Ok(i) => i,
            Err(e) => {
                warn!("material '{}' : image '{}' KO ({})", key, tex_name, e);
                return None;
            }
        };
        let (view, sampler) = upload_texture(&self.device, &self.queue, &img, clamp);
        let anim_params = MaterialParams::from_stage(tc_mods, rgb_gen, deforms);
        let params_buffer = create_params_buffer(&self.device, &self.queue, anim_params);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material-bg"),
            layout: &self.material_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let class = BlendClass::from_stage(blend, alpha_test);
        if anim_params != MaterialParams::ZERO {
            debug!(
                "material '{}' → {:?}, tex='{}', anim={:?}",
                key, class, tex_name, anim_params.anim
            );
        } else {
            debug!("material '{}' → {:?}, tex='{}'", key, class, tex_name);
        }
        Some(Arc::new(Material {
            diffuse_view: view,
            diffuse_sampler: sampler,
            params_buffer,
            bind_group,
            blend_class: class,
            shader_name,
            anim_params,
        }))
    }

    pub fn fallback(&self) -> Arc<Material> {
        self.fallback.clone()
    }

    pub fn len(&self) -> usize {
        self.by_name.len()
    }
}

/// Cache de `RenderPipeline` par `BlendClass` + format de surface.
pub struct PipelineCache {
    device: Arc<wgpu::Device>,
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    pipelines: HashMap<BlendClass, Arc<wgpu::RenderPipeline>>,
}

impl PipelineCache {
    pub fn new(
        device: Arc<wgpu::Device>,
        shader_wgsl: &str,
        camera_bgl: &wgpu::BindGroupLayout,
        lightmap_bgl: &wgpu::BindGroupLayout,
        material_bgl: &wgpu::BindGroupLayout,
        dlight_bgl: &wgpu::BindGroupLayout,
        fog_bgl: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world-material-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_wgsl.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world-material-pipeline-layout"),
            bind_group_layouts: &[camera_bgl, lightmap_bgl, material_bgl, dlight_bgl, fog_bgl],
            push_constant_ranges: &[],
        });
        Self {
            device,
            shader,
            pipeline_layout,
            format,
            pipelines: HashMap::new(),
        }
    }

    pub fn get(&mut self, class: BlendClass) -> Arc<wgpu::RenderPipeline> {
        if let Some(p) = self.pipelines.get(&class) {
            return p.clone();
        }
        let pipeline = self.build(class);
        self.pipelines.insert(class, pipeline.clone());
        pipeline
    }

    fn build(&self, class: BlendClass) -> Arc<wgpu::RenderPipeline> {
        let (blend, depth_write) = match class {
            BlendClass::Opaque | BlendClass::AlphaTest => (None, true),
            BlendClass::AlphaBlend => (
                Some(wgpu::BlendState::ALPHA_BLENDING),
                false,
            ),
            BlendClass::Additive => (
                Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent::OVER,
                }),
                false,
            ),
        };
        // Entry point dépend de la classe (alphatest a son propre fs).
        let fs_entry = match class {
            BlendClass::AlphaTest => "fs_main_alphatest",
            _ => "fs_main",
        };

        let pipeline =
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("world-material-pipeline"),
                    layout: Some(&self.pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &self.shader,
                        entry_point: "vs_main",
                        compilation_options: Default::default(),
                        buffers: &[GpuVertex::layout()],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Cw,
                        cull_mode: Some(wgpu::Face::Back),
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: depth_write,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &self.shader,
                        entry_point: fs_entry,
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: self.format,
                            blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview: None,
                    cache: None,
                });
        Arc::new(pipeline)
    }
}

fn strip_ext(path: &str) -> &str {
    match path.rfind('.') {
        Some(i) if path[i..].len() <= 5 => &path[..i],
        _ => path,
    }
}

fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    img: &Image,
    clamp: bool,
) -> (wgpu::TextureView, wgpu::Sampler) {
    let extent = wgpu::Extent3d {
        width: img.width,
        height: img.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("material-diffuse"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &img.pixels,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * img.width),
            rows_per_image: Some(img.height),
        },
        extent,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mode = if clamp {
        wgpu::AddressMode::ClampToEdge
    } else {
        wgpu::AddressMode::Repeat
    };
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("material-sampler"),
        address_mode_u: mode,
        address_mode_v: mode,
        address_mode_w: mode,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    (view, sampler)
}

fn make_fallback_material(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bgl: &wgpu::BindGroupLayout,
) -> Material {
    // Damier rose/noir 16×16 — highlight visuel évident des textures manquantes.
    let mut pixels = Vec::with_capacity(16 * 16 * 4);
    for y in 0..16 {
        for x in 0..16 {
            let checker = ((x / 4) + (y / 4)) % 2 == 0;
            let rgba = if checker { [255, 64, 255, 255] } else { [32, 16, 32, 255] };
            pixels.extend_from_slice(&rgba);
        }
    }
    let img = Image {
        width: 16,
        height: 16,
        pixels: pixels.into(),
        has_alpha: false,
    };
    let (view, sampler) = upload_texture(device, queue, &img, false);
    let params_buffer = create_params_buffer(device, queue, MaterialParams::ZERO);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fallback-material-bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    Material {
        diffuse_view: view,
        diffuse_sampler: sampler,
        params_buffer,
        bind_group,
        blend_class: BlendClass::Opaque,
        shader_name: "__fallback".into(),
        anim_params: MaterialParams::ZERO,
    }
}

/// Crée un UBO 16 B (un `vec4<f32>`) et y écrit `params.anim`.
fn create_params_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    params: MaterialParams,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("material-params-ubuf"),
        size: std::mem::size_of::<MaterialParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&params));
    buffer
}

/// Utilitaire : parcourt `scripts/*.shader` dans le VFS et remplit un
/// [`ShaderRegistry`]. Les erreurs unitaires (fichier non-UTF8, etc.) sont
/// loggées et n'interrompent pas le chargement.
pub fn load_shader_registry(vfs: &q3_filesystem::Vfs) -> Result<ShaderRegistry> {
    let mut registry = ShaderRegistry::new();
    let mut count = 0usize;
    let files = vfs.list_suffix(".shader");
    for path in &files {
        match vfs.read(path) {
            Ok(bytes) => match std::str::from_utf8(&bytes) {
                Ok(s) => {
                    count += registry.parse_file(s, path);
                }
                Err(e) => warn!("shader '{}' non-UTF8: {}", path, e),
            },
            Err(e) => warn!("shader '{}' KO: {}", path, e),
        }
    }
    debug!(
        "shader registry : {} shaders chargés depuis {} fichiers",
        count,
        files.len()
    );
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_from_empty_tc_mods_is_zero() {
        let p = MaterialParams::from_tc_mods(&[]);
        assert_eq!(p, MaterialParams::ZERO);
    }

    #[test]
    fn params_from_scroll() {
        let p = MaterialParams::from_tc_mods(&[TcMod::Scroll(0.5, -0.25)]);
        assert_eq!(p.anim, [0.5, -0.25, 0.0, 0.0]);
    }

    #[test]
    fn params_sum_multiple_scrolls() {
        let p = MaterialParams::from_tc_mods(&[
            TcMod::Scroll(0.5, 0.0),
            TcMod::Scroll(0.0, 0.25),
        ]);
        assert_eq!(p.anim, [0.5, 0.25, 0.0, 0.0]);
    }

    #[test]
    fn params_rotate_converts_deg_to_rad() {
        // 180°/s → π rad/s.
        let p = MaterialParams::from_tc_mods(&[TcMod::Rotate(180.0)]);
        assert!((p.anim[2] - std::f32::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn params_ignores_unsupported_tcmods() {
        let p = MaterialParams::from_tc_mods(&[
            TcMod::Scale(2.0, 2.0),
            TcMod::Transform([1.0; 6]),
            TcMod::Scroll(0.1, 0.0),
        ]);
        // Seul le scroll est retenu.
        assert_eq!(p.anim, [0.1, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn material_params_is_144_bytes() {
        // 9 × vec4 (16) = 144 bytes, std140-aligned.
        assert_eq!(std::mem::size_of::<MaterialParams>(), 144);
    }

    #[test]
    fn from_stage_default_leaves_rgb_mode_default() {
        let p = MaterialParams::from_stage(&[], None, &[]);
        assert_eq!(p.rgb_info[0], rgb_mode::DEFAULT);
    }

    #[test]
    fn from_stage_with_rgbgen_wave_sin() {
        use q3_shader::value::Waveform;
        let w = Waveform {
            func: WaveFunc::Sin,
            base: 0.5,
            amp: 0.5,
            phase: 0.0,
            freq: 2.0,
        };
        let p = MaterialParams::from_stage(&[], Some(&RgbGen::Wave(w)), &[]);
        assert_eq!(p.rgb_info[0], rgb_mode::WAVE);
        assert_eq!(p.rgb_wave, [0.5, 0.5, 0.0, 2.0]);
        assert_eq!(p.wave_kind[0], wave_kind::SIN);
    }

    #[test]
    fn from_stage_with_rgbgen_wave_triangle() {
        use q3_shader::value::Waveform;
        let w = Waveform {
            func: WaveFunc::Triangle,
            base: 0.1,
            amp: 0.9,
            phase: 0.25,
            freq: 0.5,
        };
        let p = MaterialParams::from_stage(&[], Some(&RgbGen::Wave(w)), &[]);
        assert_eq!(p.wave_kind[0], wave_kind::TRIANGLE);
        assert_eq!(p.rgb_wave[2], 0.25);
    }

    #[test]
    fn from_stage_with_rgbgen_const() {
        let p = MaterialParams::from_stage(&[], Some(&RgbGen::Const([1.0, 0.5, 0.25])), &[]);
        assert_eq!(p.rgb_info[0], rgb_mode::CONST);
        assert_eq!(p.rgb_const, [1.0, 0.5, 0.25, 0.0]);
    }

    #[test]
    fn from_stage_with_rgbgen_identity_stays_default() {
        let p = MaterialParams::from_stage(&[], Some(&RgbGen::Identity), &[]);
        assert_eq!(p.rgb_info[0], rgb_mode::DEFAULT);
    }

    #[test]
    fn from_stage_combines_tc_mods_and_rgbgen() {
        let p = MaterialParams::from_stage(
            &[TcMod::Scroll(0.5, 0.0)],
            Some(&RgbGen::Const([1.0, 0.0, 0.0])),
            &[],
        );
        assert_eq!(p.anim[0], 0.5);
        assert_eq!(p.rgb_info[0], rgb_mode::CONST);
        assert_eq!(p.rgb_const[0], 1.0);
    }

    #[test]
    fn from_stage_deform_wave_encoded() {
        use q3_shader::value::Waveform;
        let w = Waveform {
            func: WaveFunc::Sin,
            base: 0.0,
            amp: 2.0,
            phase: 0.0,
            freq: 0.5,
        };
        let p = MaterialParams::from_stage(
            &[],
            None,
            &[DeformVertexes::Wave {
                spread: 100.0,
                wave: w,
            }],
        );
        assert_eq!(p.deform_info[0], deform_mode::WAVE);
        assert_eq!(p.deform_info[1], 100.0);
        assert_eq!(p.deform_wave, [0.0, 2.0, 0.0, 0.5]);
        assert_eq!(p.deform_kind[0], wave_kind::SIN);
    }

    #[test]
    fn from_stage_deform_move_encoded() {
        use q3_shader::value::Waveform;
        let w = Waveform {
            func: WaveFunc::Triangle,
            base: 0.0,
            amp: 4.0,
            phase: 0.25,
            freq: 1.0,
        };
        let p = MaterialParams::from_stage(
            &[],
            None,
            &[DeformVertexes::Move {
                dir: [0.0, 0.0, 1.0],
                wave: w,
            }],
        );
        assert_eq!(p.deform_info[0], deform_mode::MOVE);
        assert_eq!(p.deform_dir, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(p.deform_kind[0], wave_kind::TRIANGLE);
    }

    #[test]
    fn from_stage_deform_normal_ignored() {
        // `Normal` (= ripple tangentiel) n'est pas supporté pour l'instant.
        let p = MaterialParams::from_stage(
            &[],
            None,
            &[DeformVertexes::Normal {
                amplitude: 1.0,
                frequency: 1.0,
            }],
        );
        assert_eq!(p.deform_info[0], deform_mode::NONE);
    }

    #[test]
    fn from_stage_deform_picks_first_supported() {
        // Skip AutoSprite (ignoré), prend le Wave qui suit.
        use q3_shader::value::Waveform;
        let w = Waveform {
            func: WaveFunc::Sin,
            base: 0.0,
            amp: 1.0,
            phase: 0.0,
            freq: 1.0,
        };
        let p = MaterialParams::from_stage(
            &[],
            None,
            &[
                DeformVertexes::AutoSprite,
                DeformVertexes::Wave {
                    spread: 50.0,
                    wave: w,
                },
            ],
        );
        assert_eq!(p.deform_info[0], deform_mode::WAVE);
        assert_eq!(p.deform_info[1], 50.0);
    }
}
