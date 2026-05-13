//! Systeme de particules **GPU compute** — remplacement haute performance
//! du CPU billboard de `particle.rs` pour des counts 10-30x superieurs.
//!
//! # Pourquoi un systeme GPU ?
//!
//! Le systeme CPU (`particle.rs`) reconstruit 6 vertices par particule
//! chaque frame cote CPU puis upload le buffer.  Ca tient a 2048 particules,
//! mais les effets visuels modernes (trails rocket, pluie de debris,
//! explosions massives) demandent 20-60k particules simultanées.  A ce
//! volume, le CPU bottleneck (upload + transform) depasse le budget frame.
//!
//! # Architecture
//!
//! * **Compute pipeline** (`cs_update`) : met a jour position, velocite,
//!   vie, applique gravite + drag + turbulence.  Dispatch 1D : ceil(N/256)
//!   workgroups de 256 threads chacun.
//! * **Render pipeline** (`vs_main` / `fs_main`) : dessine chaque particule
//!   vivante comme un quad billboard camera-facing (4 vertices/particule
//!   via `vertex_index`, instance par particule).  Triangle strip.
//! * **Ping-pong** : deux buffers storage (`buf_a`, `buf_b`).  Chaque
//!   frame, le compute lit l'un et ecrit l'autre, puis le render lit le
//!   buffer ecrit.  Au frame suivant on inverse.  Cela evite les data
//!   hazards sans barriere explicite supplementaire.
//! * **Compteur atomique** dans le compute pour compter les particules
//!   vivantes — alimente le `vertex_count` d'un indirect draw buffer.
//!
//! # Types de particules
//!
//! `ParticleType` encode le comportement physique et visuel :
//! Blood, Spark, Smoke, Debris, Fire, PlasmaTrail, RocketTrail.
//! Chaque type a sa gravite, son drag, sa turbulence et sa palette
//! definis cote shader (constants dans `gpu_particle.wgsl`).

use bytemuck::{Pod, Zeroable};
use q3_math::Vec3;
use std::sync::Arc;
use tracing::warn;
use wgpu::util::DeviceExt;

use crate::{DEPTH_FORMAT, SCENE_HDR_FORMAT};

/// Capacite maximale du systeme. Doit matcher `sim.max_particles` passe
/// au compute shader.  65 536 = 256 workgroups de 256 threads — nombre
/// rond, tient facilement dans le budget VRAM meme sur un GPU integre
/// (~5 MB pour les 2 buffers ping-pong).
const MAX_PARTICLES: u32 = 65_536;

/// Workgroup size — doit correspondre a `@workgroup_size(256)` dans le
/// shader WGSL.  Changer l'un sans l'autre = bug silencieux.
const WORKGROUP_SIZE: u32 = 256;

/// Taille d'une particule GPU en octets.  Doit matcher exactement le
/// `struct Particle` cote WGSL (std430 layout).
///
/// Layout :
///   position : vec3<f32>  (12)  + life : f32 (4)   = 16
///   velocity : vec3<f32>  (12)  + max_life : f32(4) = 16
///   color    : vec4<f32>  (16)                      = 16
///   size     : f32 (4)  + ptype : u32 (4) + pad×2   = 16
///                                               total = 64
const PARTICLE_STRIDE: u64 = 64;

// ────────────────────────────────────────────────────────────────────
//  Types publics
// ────────────────────────────────────────────────────────────────────

/// Type de particule — determine la physique et le rendu cote shader.
/// Les valeurs numeriques matchent les `switch` dans `gpu_particle.wgsl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ParticleType {
    Blood       = 0,
    Spark       = 1,
    Smoke       = 2,
    Debris      = 3,
    Fire        = 4,
    PlasmaTrail = 5,
    RocketTrail = 6,
}

/// Mode d'emission : one-shot (burst unique) ou continu (N par seconde).
#[derive(Debug, Clone, Copy)]
pub enum EmissionMode {
    /// Emettre `count` particules immediatement, une seule fois.
    OneShot,
    /// Emettre `count` particules par seconde, tant que l'emitter vit.
    Continuous,
}

/// Emitter : description d'une source de particules a soumettre au
/// systeme via `emit()`.  Le systeme convertit ca en ecritures brutes
/// dans le buffer GPU.
#[derive(Debug, Clone)]
pub struct ParticleEmitter {
    /// Position monde de l'emitter (coordonnees Q3).
    pub position: Vec3,
    /// Direction principale d'emission (normalisee).
    pub direction: Vec3,
    /// Angle de dispersion du cone autour de `direction` (radians).
    /// 0 = tir droit, PI = hemisphere complete.
    pub spread_angle: f32,
    /// Nombre de particules par burst (OneShot) ou par seconde (Continuous).
    pub count: u32,
    /// Type de particule.
    pub ptype: ParticleType,
    /// Mode d'emission.
    pub mode: EmissionMode,
    /// Vitesse initiale min/max (unites Q3/s).  La vitesse effective
    /// est interpolee aleatoirement (pseudo-random cote CPU au spawn).
    pub speed_min: f32,
    pub speed_max: f32,
    /// Taille initiale des quads (demi-cote, unites Q3).
    pub size: f32,
    /// Duree de vie des particules (secondes).
    pub lifetime: f32,
    /// Couleur RGBA de modulation.  Le shader a deja des couleurs par
    /// type ; cette valeur les multiplie (tint / alpha override).
    pub color: [f32; 4],
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            direction: Vec3::new(0.0, 0.0, 1.0),
            spread_angle: std::f32::consts::FRAC_PI_4,
            count: 16,
            ptype: ParticleType::Smoke,
            mode: EmissionMode::OneShot,
            speed_min: 50.0,
            speed_max: 200.0,
            size: 4.0,
            lifetime: 1.5,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

// ────────────────────────────────────────────────────────────────────
//  GPU-side structs (bytemuck) — doivent matcher le WGSL exactement
// ────────────────────────────────────────────────────────────────────

/// Particule GPU (64 octets, std430).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuParticle {
    position: [f32; 3],
    life: f32,
    velocity: [f32; 3],
    max_life: f32,
    color: [f32; 4],
    size: f32,
    ptype: u32,
    _pad0: f32,
    _pad1: f32,
}

/// Parametres de simulation (16 octets, uniform).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SimParams {
    dt: f32,
    time: f32,
    max_particles: u32,
    _pad: u32,
}

/// Arguments de draw indirect (16 octets).  Le compute ecrit
/// `vertex_count` via atomicAdd ; le reste est fixe.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct IndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

// ────────────────────────────────────────────────────────────────────
//  Systeme principal
// ────────────────────────────────────────────────────────────────────

pub struct GpuParticleSystem {
    _device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // ── Compute ──
    compute_pipeline: wgpu::ComputePipeline,
    _compute_bgl: wgpu::BindGroupLayout,
    /// Bind groups compute ping-pong.  `[0]` lit buf_a, ecrit buf_b ;
    /// `[1]` lit buf_b, ecrit buf_a.
    compute_bind_groups: [wgpu::BindGroup; 2],

    // ── Render ──
    render_pipeline: wgpu::RenderPipeline,
    _render_bgl: wgpu::BindGroupLayout,
    /// Bind groups render : `[0]` lit buf_b (destination du compute[0]),
    /// `[1]` lit buf_a.
    render_bind_groups: [wgpu::BindGroup; 2],

    // ── Buffers ──
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    sim_uniform: wgpu::Buffer,
    indirect_buffer: wgpu::Buffer,
    /// Buffer CPU staging pour reset le compteur atomique a zero avant
    /// chaque dispatch.  4 octets de zeros.
    indirect_reset: wgpu::Buffer,

    /// Index ping-pong : 0 ou 1.  Determine quel compute_bind_group
    /// et quel render_bind_group utiliser cette frame.
    ping: usize,

    /// Particules en attente de spawn — accumulees entre les appels a
    /// `emit()` et flushees dans `update()` juste avant le dispatch.
    pending_spawns: Vec<GpuParticle>,

    /// Curseur circulaire dans le buffer pour placer les nouvelles
    /// particules.  On ecrase les plus anciennes quand on wrappe.
    write_cursor: u32,

    /// Compteur simple pour le PRNG cote CPU (seed des velocites au spawn).
    rng_state: u32,
}

impl GpuParticleSystem {
    /// Cree le systeme.  `camera_bgl` est le bind group layout camera
    /// partage par tous les pipelines du renderer (group 0, binding 0,
    /// uniform CameraUniform).
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        camera_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu-particle-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gpu_particle.wgsl").into()),
        });

        // ── Buffers particules (ping-pong) ──
        let buf_size = PARTICLE_STRIDE * (MAX_PARTICLES as u64);
        let particle_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST;

        let buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-particle-buf-a"),
            size: buf_size,
            usage: particle_usage,
            mapped_at_creation: false,
        });
        let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-particle-buf-b"),
            size: buf_size,
            usage: particle_usage,
            mapped_at_creation: false,
        });

        // ── Sim uniform ──
        let sim_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu-particle-sim-uniform"),
            contents: bytemuck::bytes_of(&SimParams {
                dt: 0.016,
                time: 0.0,
                max_particles: MAX_PARTICLES,
                _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ── Indirect draw buffer ──
        let indirect_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu-particle-indirect"),
            contents: bytemuck::bytes_of(&IndirectArgs {
                vertex_count: 0,
                instance_count: MAX_PARTICLES,
                first_vertex: 0,
                first_instance: 0,
            }),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
        });

        // Buffer de 16 octets de zeros pour reset le compteur atomique
        // avant chaque dispatch. On ecrit 0 dans vertex_count et on
        // conserve instance_count = MAX_PARTICLES.
        let indirect_reset = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu-particle-indirect-reset"),
            contents: bytemuck::bytes_of(&IndirectArgs {
                vertex_count: 0,
                instance_count: MAX_PARTICLES,
                first_vertex: 0,
                first_instance: 0,
            }),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // ────────────────────────────────────────────────────────────
        //  Compute pipeline
        // ────────────────────────────────────────────────────────────

        let compute_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu-particle-compute-bgl"),
                entries: &[
                    // binding 0 : particles_src (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1 : particles_dst (read-write storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 2 : sim params (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 3 : indirect args (read-write storage for atomic)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu-particle-compute-layout"),
                // Le compute n'utilise pas le camera bind group (group 0) —
                // on ne le bind pas dans le layout. Le compute a sa propre
                // group(1) qui en pratique est bind en group(0) du dispatch.
                //
                // CORRECTION : dans le shader, camera est group(0) et les
                // storage sont group(1). Mais le compute n'a PAS besoin de
                // camera — il n'utilise que group(1). Pour simplifier, on
                // fait un layout compute separe avec seulement la BGL compute
                // en position 0 (pas de camera). Le shader compute ne declare
                // pas de group(0) camera, seulement group(1).
                //
                // En realite le shader WGSL declare group(1) pour le compute.
                // wgpu mappe les bind groups par index dans le layout. Donc
                // on doit fournir un slot vide pour group(0) OU reorganiser.
                // La solution propre : utiliser un pipeline layout compute
                // avec [camera_bgl, compute_bgl] meme si le compute n'utilise
                // pas camera. Ca permet au runtime de valider group(1).
                bind_group_layouts: &[camera_bgl, &compute_bgl],
                push_constant_ranges: &[],
            });

        let compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpu-particle-compute"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "cs_update",
                compilation_options: Default::default(),
                cache: None,
            });

        // ── Compute bind groups (ping-pong) ──
        // [0] : src=A, dst=B   |   [1] : src=B, dst=A
        let make_compute_bg =
            |label: &str, src: &wgpu::Buffer, dst: &wgpu::Buffer| -> wgpu::BindGroup {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &compute_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: src.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: dst.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: sim_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: indirect_buffer.as_entire_binding(),
                        },
                    ],
                })
            };

        let compute_bg_0 = make_compute_bg("gpu-particle-compute-bg-0", &buf_a, &buf_b);
        let compute_bg_1 = make_compute_bg("gpu-particle-compute-bg-1", &buf_b, &buf_a);

        // ────────────────────────────────────────────────────────────
        //  Render pipeline
        // ────────────────────────────────────────────────────────────

        let render_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu-particle-render-bgl"),
                entries: &[
                    // binding 0 : particle buffer (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu-particle-render-layout"),
                bind_group_layouts: &[camera_bgl, &render_bgl],
                push_constant_ranges: &[],
            });

        let render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gpu-particle-render"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    compilation_options: Default::default(),
                    // Pas de vertex buffer — les vertices sont generes
                    // proceduralement via vertex_index + instance_index.
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    // Pas de cull — billboards bifaciales.
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    // Depth-test ON (occlusion par geometrie monde) mais pas
                    // de depth-write — les particules sont translucides et ne
                    // doivent pas s'occluder entre elles.
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
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
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            });

        // ── Render bind groups ──
        // Apres compute[0] (A→B), le render lit B.
        // Apres compute[1] (B→A), le render lit A.
        let make_render_bg = |label: &str, buf: &wgpu::Buffer| -> wgpu::BindGroup {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &render_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            })
        };

        let render_bg_0 = make_render_bg("gpu-particle-render-bg-0", &buf_b);
        let render_bg_1 = make_render_bg("gpu-particle-render-bg-1", &buf_a);

        Self {
            _device: device,
            queue,
            compute_pipeline,
            _compute_bgl: compute_bgl,
            compute_bind_groups: [compute_bg_0, compute_bg_1],
            render_pipeline,
            _render_bgl: render_bgl,
            render_bind_groups: [render_bg_0, render_bg_1],
            buf_a,
            buf_b,
            sim_uniform,
            indirect_buffer,
            indirect_reset,
            ping: 0,
            pending_spawns: Vec::with_capacity(256),
            write_cursor: 0,
            rng_state: 42,
        }
    }

    // ────────────────────────────────────────────────────────────────
    //  Emission
    // ────────────────────────────────────────────────────────────────

    /// Soumet un emitter.  Les particules sont accumulees en RAM et
    /// flushees vers le GPU au prochain `update()`.
    ///
    /// En mode `Continuous`, le caller doit rappeler `emit()` chaque
    /// frame (ou a intervalle fixe) — le systeme ne retient pas
    /// l'emitter en interne pour rester stateless.
    pub fn emit(&mut self, emitter: &ParticleEmitter) {
        let count = match emitter.mode {
            EmissionMode::OneShot => emitter.count,
            // En continu, le count represente le rate/s ; on le scale
            // par dt=1/60 comme approximation. Le caller passera le
            // vrai dt au prochain `update` mais on n'a pas cette info
            // ici — l'approximation est suffisante pour un look correct.
            EmissionMode::Continuous => (emitter.count as f32 / 60.0).ceil() as u32,
        };

        for _ in 0..count {
            // Limite hard pour eviter de deborder le buffer.
            if self.pending_spawns.len() >= MAX_PARTICLES as usize {
                warn!("gpu_particle: capacite spawn atteinte, particules ignorees");
                return;
            }

            let (vx, vy, vz) = self.random_cone_direction(
                emitter.direction,
                emitter.spread_angle,
            );
            let speed = self.rand_range(emitter.speed_min, emitter.speed_max);

            self.pending_spawns.push(GpuParticle {
                position: emitter.position.to_array(),
                life: emitter.lifetime,
                velocity: [vx * speed, vy * speed, vz * speed],
                max_life: emitter.lifetime,
                color: emitter.color,
                size: emitter.size,
                ptype: emitter.ptype as u32,
                _pad0: 0.0,
                _pad1: 0.0,
            });
        }
    }

    /// Vide toutes les particules (utile au changement de map).
    pub fn clear(&mut self) {
        self.pending_spawns.clear();
        self.write_cursor = 0;
        // Ecrit des zeros dans les deux buffers pour tuer toutes les
        // particules vivantes.
        let zeros = vec![0u8; (PARTICLE_STRIDE * MAX_PARTICLES as u64) as usize];
        self.queue.write_buffer(&self.buf_a, 0, &zeros);
        self.queue.write_buffer(&self.buf_b, 0, &zeros);
    }

    // ────────────────────────────────────────────────────────────────
    //  Update (compute dispatch)
    // ────────────────────────────────────────────────────────────────

    /// Flush les spawns en attente vers le GPU puis dispatche le compute
    /// shader de simulation.  `dt` en secondes, `time` = horloge moteur.
    ///
    /// Le caller doit fournir `camera_bind_group` (group 0) — meme s'il
    /// n'est pas lu par le compute, le pipeline layout l'exige pour la
    /// validation wgpu.
    pub fn update(
        &mut self,
        dt: f32,
        time: f32,
        encoder: &mut wgpu::CommandEncoder,
        camera_bind_group: &wgpu::BindGroup,
    ) {
        // ── Flush pending spawns dans le buffer SOURCE du ping courant ──
        if !self.pending_spawns.is_empty() {
            let src_buf = if self.ping == 0 { &self.buf_a } else { &self.buf_b };
            for particle in self.pending_spawns.drain(..) {
                let offset = (self.write_cursor as u64) * PARTICLE_STRIDE;
                self.queue
                    .write_buffer(src_buf, offset, bytemuck::bytes_of(&particle));
                self.write_cursor = (self.write_cursor + 1) % MAX_PARTICLES;
            }
        }

        // ── Mise a jour du uniform de simulation ──
        self.queue.write_buffer(
            &self.sim_uniform,
            0,
            bytemuck::bytes_of(&SimParams {
                dt,
                time,
                max_particles: MAX_PARTICLES,
                _pad: 0,
            }),
        );

        // ── Reset du compteur atomique (vertex_count = 0) ──
        encoder.copy_buffer_to_buffer(
            &self.indirect_reset,
            0,
            &self.indirect_buffer,
            0,
            std::mem::size_of::<IndirectArgs>() as u64,
        );

        // ── Dispatch compute ──
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu-particle-compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            // group(0) = camera (requis par le layout meme si le compute
            // ne l'utilise pas — le shader declare group(1) pour ses storage).
            pass.set_bind_group(0, camera_bind_group, &[]);
            pass.set_bind_group(1, &self.compute_bind_groups[self.ping], &[]);
            let workgroups = MAX_PARTICLES.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // ── Flip ping-pong pour la prochaine frame ──
        self.ping = 1 - self.ping;
    }

    // ────────────────────────────────────────────────────────────────
    //  Render
    // ────────────────────────────────────────────────────────────────

    /// Dessine les particules dans la render pass courante.  Le caller
    /// doit avoir bind `camera_bind_group` au slot 0 avant d'appeler.
    ///
    /// Utilise un draw indirect alimente par le compteur atomique du
    /// compute — seules les particules vivantes generent des vertices.
    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.render_pipeline);
        // Le render bind group correspondant lit le buffer DESTINATION
        // du compute qui vient de s'executer (ping a deja ete flippe
        // dans `update`, donc le render_bind_group[ping_actuel] lit le
        // bon buffer).
        pass.set_bind_group(1, &self.render_bind_groups[self.ping], &[]);
        // Draw indirect : 4 vertices par instance (triangle strip quad),
        // instance_count = nombre de particules vivantes (ecrit par compute).
        pass.draw_indirect(&self.indirect_buffer, 0);
    }

    /// Callback sur resize fenetre.  Le systeme de particules n'a pas
    /// de ressources dependantes de la resolution (pas de render target
    /// propre, pas de screen-space effects), donc c'est un no-op.
    /// On garde la methode pour la coherence d'interface avec les
    /// autres sous-systemes du renderer.
    #[allow(unused_variables)]
    pub fn resize(&mut self, width: u32, height: u32) {
        // Pas de ressource dependant de la resolution pour l'instant.
        // Reserve pour de futurs effets screen-space (distortion heat,
        // refraction trails, etc.).
    }

    /// Nombre maximal de particules supportees.
    pub const fn max_particles() -> u32 {
        MAX_PARTICLES
    }

    // ────────────────────────────────────────────────────────────────
    //  PRNG minimaliste (xorshift32)
    // ────────────────────────────────────────────────────────────────

    /// xorshift32 rapide — pas crypto-secure, mais suffisant pour
    /// disperser visuellement des particules.
    fn next_u32(&mut self) -> u32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng_state = x;
        x
    }

    /// Float dans [0, 1).
    fn rand_f32(&mut self) -> f32 {
        (self.next_u32() & 0x00FF_FFFF) as f32 / 16_777_216.0
    }

    /// Float dans [min, max].
    fn rand_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.rand_f32() * (max - min)
    }

    /// Genere une direction aleatoire dans un cone autour de `dir`
    /// avec un demi-angle `spread` (radians).  Retourne (x, y, z)
    /// normalise.
    fn random_cone_direction(&mut self, dir: Vec3, spread: f32) -> (f32, f32, f32) {
        if spread <= 0.0001 {
            return (dir.x, dir.y, dir.z);
        }

        // Angle aleatoire dans le cone
        let cos_spread = spread.cos();
        let cos_theta = self.rand_range(cos_spread, 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi = self.rand_range(0.0, std::f32::consts::TAU);

        // Vecteur local (dans le repere ou dir = +Z)
        let lx = sin_theta * phi.cos();
        let ly = sin_theta * phi.sin();
        let lz = cos_theta;

        // Construire une base orthonormee autour de `dir`.
        // On prend un vecteur "up" qui n'est pas colineaire a dir.
        let up_ref = if dir.z.abs() < 0.99 {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let right = dir.cross(up_ref).normalize();
        let up = right.cross(dir).normalize();

        // Rotation du vecteur local dans le repere monde
        let wx = right.x * lx + up.x * ly + dir.x * lz;
        let wy = right.y * lx + up.y * ly + dir.y * lz;
        let wz = right.z * lx + up.z * ly + dir.z * lz;

        let len = (wx * wx + wy * wy + wz * wz).sqrt().max(0.0001);
        (wx / len, wy / len, wz / len)
    }
}

// ────────────────────────────────────────────────────────────────────
//  Presets d'emitters — raccourcis pour les effets courants du jeu
// ────────────────────────────────────────────────────────────────────

impl ParticleEmitter {
    /// Giclure de sang (impact balle sur joueur).
    pub fn blood(position: Vec3, direction: Vec3) -> Self {
        Self {
            position,
            direction,
            spread_angle: 0.8,
            count: 24,
            ptype: ParticleType::Blood,
            mode: EmissionMode::OneShot,
            speed_min: 80.0,
            speed_max: 300.0,
            size: 3.0,
            lifetime: 0.8,
            color: [1.0, 1.0, 1.0, 0.9],
        }
    }

    /// Etincelles (impact balle sur metal/mur).
    pub fn sparks(position: Vec3, normal: Vec3) -> Self {
        Self {
            position,
            direction: normal,
            spread_angle: 1.0,
            count: 32,
            ptype: ParticleType::Spark,
            mode: EmissionMode::OneShot,
            speed_min: 150.0,
            speed_max: 500.0,
            size: 1.5,
            lifetime: 0.5,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }

    /// Fumee d'explosion.
    pub fn explosion_smoke(position: Vec3) -> Self {
        Self {
            position,
            direction: Vec3::new(0.0, 0.0, 1.0),
            spread_angle: std::f32::consts::PI,
            count: 48,
            ptype: ParticleType::Smoke,
            mode: EmissionMode::OneShot,
            speed_min: 20.0,
            speed_max: 120.0,
            size: 8.0,
            lifetime: 2.5,
            color: [1.0, 1.0, 1.0, 0.6],
        }
    }

    /// Debris (explosion, destruction de props).
    pub fn debris(position: Vec3) -> Self {
        Self {
            position,
            direction: Vec3::new(0.0, 0.0, 1.0),
            spread_angle: std::f32::consts::PI * 0.8,
            count: 20,
            ptype: ParticleType::Debris,
            mode: EmissionMode::OneShot,
            speed_min: 100.0,
            speed_max: 400.0,
            size: 2.0,
            lifetime: 1.2,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }

    /// Flammes (incendie, torche).
    pub fn fire(position: Vec3) -> Self {
        Self {
            position,
            direction: Vec3::new(0.0, 0.0, 1.0),
            spread_angle: 0.4,
            count: 60,
            ptype: ParticleType::Fire,
            mode: EmissionMode::Continuous,
            speed_min: 30.0,
            speed_max: 80.0,
            size: 5.0,
            lifetime: 1.0,
            color: [1.0, 1.0, 1.0, 0.8],
        }
    }

    /// Trail plasma (projectile plasma gun).
    pub fn plasma_trail(position: Vec3, direction: Vec3) -> Self {
        Self {
            position,
            direction,
            spread_angle: 0.2,
            count: 8,
            ptype: ParticleType::PlasmaTrail,
            mode: EmissionMode::Continuous,
            speed_min: 10.0,
            speed_max: 40.0,
            size: 3.5,
            lifetime: 0.6,
            color: [1.0, 1.0, 1.0, 0.7],
        }
    }

    /// Trail rocket (projectile RL).
    pub fn rocket_trail(position: Vec3, direction: Vec3) -> Self {
        Self {
            position,
            direction: -direction, // La trainee part vers l'arriere
            spread_angle: 0.5,
            count: 40,
            ptype: ParticleType::RocketTrail,
            mode: EmissionMode::Continuous,
            speed_min: 15.0,
            speed_max: 60.0,
            size: 6.0,
            lifetime: 1.8,
            color: [1.0, 1.0, 1.0, 0.5],
        }
    }
}

// ────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_particle_struct_size_matches_stride() {
        // Verifie que le layout Rust correspond au stride attendu cote
        // GPU (64 octets std430).  Un desalignement ici cause un
        // decalage silencieux des champs dans le shader.
        assert_eq!(
            std::mem::size_of::<GpuParticle>(),
            PARTICLE_STRIDE as usize,
            "GpuParticle doit faire exactement {} octets (std430 alignment)",
            PARTICLE_STRIDE,
        );
    }

    #[test]
    fn sim_params_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<SimParams>(), 16);
    }

    #[test]
    fn indirect_args_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<IndirectArgs>(), 16);
    }

    #[test]
    fn particle_type_values_match_shader_switch() {
        // Les valeurs de l'enum Rust doivent correspondre aux `case`
        // du shader WGSL — sinon un Blood serait simule comme un Spark.
        assert_eq!(ParticleType::Blood as u32, 0);
        assert_eq!(ParticleType::Spark as u32, 1);
        assert_eq!(ParticleType::Smoke as u32, 2);
        assert_eq!(ParticleType::Debris as u32, 3);
        assert_eq!(ParticleType::Fire as u32, 4);
        assert_eq!(ParticleType::PlasmaTrail as u32, 5);
        assert_eq!(ParticleType::RocketTrail as u32, 6);
    }

    #[test]
    fn xorshift_produces_nonzero_sequence() {
        // Le PRNG ne doit pas degenerer vers zero (piege classique du
        // xorshift avec seed=0, mais on seed a 42).
        let mut sys_rng_state: u32 = 42;
        for _ in 0..100 {
            let mut x = sys_rng_state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            sys_rng_state = x;
            assert_ne!(x, 0, "xorshift ne doit jamais produire 0 avec seed non-nul");
        }
    }

    #[test]
    fn rand_f32_stays_in_unit_range() {
        let mut state: u32 = 42;
        for _ in 0..1000 {
            let mut x = state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            state = x;
            let f = (x & 0x00FF_FFFF) as f32 / 16_777_216.0;
            assert!(f >= 0.0 && f < 1.0, "rand_f32 hors [0,1) : {f}");
        }
    }

    #[test]
    fn default_emitter_is_valid() {
        let e = ParticleEmitter::default();
        assert!(e.lifetime > 0.0);
        assert!(e.count > 0);
        assert!(e.speed_max >= e.speed_min);
        assert!(e.size > 0.0);
    }

    #[test]
    fn preset_blood_has_correct_type() {
        let e = ParticleEmitter::blood(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(e.ptype, ParticleType::Blood);
        assert!(matches!(e.mode, EmissionMode::OneShot));
    }

    #[test]
    fn preset_rocket_trail_is_continuous() {
        let e = ParticleEmitter::rocket_trail(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
        );
        assert_eq!(e.ptype, ParticleType::RocketTrail);
        assert!(matches!(e.mode, EmissionMode::Continuous));
    }

    #[test]
    fn workgroup_dispatch_covers_all_particles() {
        let workgroups = (MAX_PARTICLES + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        assert!(
            workgroups * WORKGROUP_SIZE >= MAX_PARTICLES,
            "le dispatch doit couvrir toutes les particules"
        );
    }
}
