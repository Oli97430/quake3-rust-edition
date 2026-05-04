//! Pipeline rendu terrain — consomme un `q3_terrain::Terrain` et le
//! dessine en chunks LOD-adaptatifs.
//!
//! # Stratégie
//!
//! * **À l'upload** (`TerrainGpu::new`) on stocke juste le `Arc<Terrain>` ;
//!   les buffers GPU sont générés à la volée par chunk.
//! * **Chaque frame** (`update_chunks`) on parcourt les chunks visibles
//!   selon la position caméra, on calcule le LOD optimal pour chacun,
//!   et on (re)génère le `ChunkGpu` correspondant si le LOD a changé
//!   depuis la frame précédente.
//! * **Render** (`draw`) on issue un drawcall par chunk visible.
//!
//! # Cache
//!
//! Le cache `chunks: HashMap<(coord, lod), ChunkGpu>` garde les buffers
//! par paire (coord, LOD).  Quand un chunk change de LOD, l'ancien
//! buffer reste en cache (il sera ré-utilisable si le joueur revient
//! au même LOD), mais on cap le total à un budget mémoire pour éviter
//! la croissance indéfinie sur une carte 1330 chunks × 4 LOD.
//!
//! # Limitations connues (v0.9.5 — premier pass jouable)
//!
//! * **Pas de frustum culling** : tous les chunks sont mis à jour, le
//!   GPU les rejette via depth.  Sur 1330 chunks LOD3 (16k tris) ça
//!   tient encore largement, on pourra optimiser plus tard.
//! * **Pas de stitching** : entre deux chunks de LOD différent on a
//!   des fissures sub-pixel.  Acceptable au visuel — on les masquera
//!   par des skirts verticaux dans une release ultérieure.
//! * **Splat colors uniforms** : le shader applique 4 couleurs fixes
//!   (cf. `terrain.wgsl`) au lieu de 4 textures PBR.  Solution
//!   contemporaine, mais beaucoup plus simple à shipper que
//!   l'authoring textures.

use bytemuck::{cast_slice, Pod, Zeroable};
use hashbrown::HashMap;
use q3_terrain::mesh::{ChunkCoord, ChunkMesh, LodLevel, TerrainVertex};
use q3_terrain::Terrain;
use std::sync::Arc;
use tracing::{debug, info};
use wgpu::util::DeviceExt;

use crate::{DEPTH_FORMAT, SCENE_HDR_FORMAT};

/// Uniforme passé au pipeline terrain (group 1) — paramètres d'éclairage
/// + fog. La caméra reste sur group 0 (partagée avec le pipeline world).
///
/// std140-aligned : 3 × vec4 = 48 octets.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TerrainParams {
    pub sun_dir: [f32; 4],    // .xyz = direction (vers soleil), .w = intensité
    pub fog_color: [f32; 4],  // .rgb couleur fog, .a unused
    pub fog_params: [f32; 4], // .x near, .y far, .z density, .w water_level
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            // Soleil tropical haut sud-ouest (Réunion).  Z+ = haut, X+ = est.
            sun_dir: [-0.4, -0.3, 0.86, 1.0],
            // Fog bleu pâle océan-ciel pour fondre l'horizon en mer.
            fog_color: [0.55, 0.68, 0.78, 1.0],
            fog_params: [3_000.0, 18_000.0, 0.85, 0.0],
        }
    }
}

/// Buffers GPU d'un chunk au LOD donné.
struct ChunkGpu {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl ChunkGpu {
    fn from_mesh(device: &wgpu::Device, mesh: &ChunkMesh) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-chunk-vb"),
            contents: cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-chunk-ib"),
            contents: cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
        }
    }
}

/// Layout vertex pour `TerrainVertex`.
fn terrain_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: TerrainVertex::STRIDE_BYTES as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            // pos
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            // normal (offset 16 = 12 + 4 padding)
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            // splat (offset 32 = 16 + 12 + 4)
            wgpu::VertexAttribute {
                offset: 32,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x4,
            },
        ],
    }
}

/// Renderer terrain — détient le pipeline + le cache chunks + le terrain.
pub struct TerrainGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    /// Bind group layout pour les paramètres terrain (group 1).
    #[allow(dead_code)]
    params_bgl: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    params_bind_group: wgpu::BindGroup,
    params: TerrainParams,
    /// Données CPU — référence partagée avec l'engine pour les traces
    /// de collision (pas dupliqué côté GPU).
    terrain: Arc<Terrain>,
    /// Cache chunks générés.  La clé combine coord + LOD pour pouvoir
    /// garder plusieurs LODs en mémoire (le joueur peut revenir
    /// rapidement à un LOD précédent).  Capé via `MAX_CACHE_CHUNKS`.
    chunks: HashMap<(ChunkCoord, LodLevel), ChunkGpu>,
    /// LOD actuellement sélectionné par chunk.  Le `update_chunks`
    /// pose ici le LOD à utiliser pour la frame ; `draw` lit cette
    /// table pour issuer ses drawcalls.
    current_lod: HashMap<ChunkCoord, LodLevel>,
}

/// Cap dur sur le nombre d'entrées du cache chunks.  Au-delà, on évince
/// les plus anciens via une politique simple (clear total).  Suffisant
/// au MVP — quand on aura des stats d'usage on passera sur LRU.
const MAX_CACHE_CHUNKS: usize = 4096;

impl TerrainGpu {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        camera_bgl: &wgpu::BindGroupLayout,
        terrain: Arc<Terrain>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/terrain.wgsl").into()),
        });

        // group 1 = TerrainParams uniform
        let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain-params-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let params = TerrainParams::default();
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-params-ubo"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain-params-bg"),
            layout: &params_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-pipeline-layout"),
            bind_group_layouts: &[camera_bgl, &params_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[terrain_vertex_layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                // Triangulation chunk : `tl → bl → br, tl → br → tr` avec
                // tl=(i=0,j=0), bl=(i=0,j=1), br=(i=1,j=1), tr=(i=1,j=0)
                // → winding CW vu d'au-dessus (Z+).  On veut que la face
                // UP soit la face avant, donc `FrontFace::Cw` + cull Back
                // pour que les surfaces vues d'en haut s'affichent et
                // celles vues d'en dessous (sous-sol) soient cullées.
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
                    format: SCENE_HDR_FORMAT,
                    blend: None, // terrain est opaque
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        info!(
            "terrain-gpu: pipeline créé, terrain {}×{} = {} chunks max",
            terrain.width,
            terrain.height,
            terrain.n_chunks_x() * terrain.n_chunks_y()
        );

        Self {
            device,
            queue,
            pipeline,
            params_bgl,
            params_buffer,
            params_bind_group,
            params,
            terrain,
            chunks: HashMap::new(),
            current_lod: HashMap::new(),
        }
    }

    /// Met à jour les `TerrainParams` (sun, fog, water_level).  La copie
    /// CPU→GPU se fait via `queue.write_buffer` — légère.
    pub fn set_params(&mut self, params: TerrainParams) {
        self.params = params;
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    /// Renvoie le terrain CPU partagé (pour les traces de collision
    /// côté gameplay sans devoir cloner le `Arc` ailleurs).
    pub fn terrain(&self) -> &Arc<Terrain> {
        &self.terrain
    }

    /// Distance max au-delà de laquelle un chunk est culled (pas
    /// rendu).  Calé sur le `fog_params.y` par défaut + marge pour
    /// éviter le pop visible quand un chunk apparaît juste à la
    /// limite du fog.
    pub const RENDER_DISTANCE: f32 = 22_000.0;

    /// Sélectionne le LOD optimal pour chaque chunk visible selon la
    /// caméra, et génère/met en cache les buffers GPU nécessaires.
    ///
    /// **Culling** : on rejette les chunks plus loin que
    /// `RENDER_DISTANCE` du joueur — sur 1330 chunks Réunion, à 1500u
    /// du centre on en garde typiquement 200-400, économise ~70 % des
    /// drawcalls par rapport au "tout draw".
    ///
    /// À appeler une fois par frame avant `draw`.
    pub fn update_chunks(&mut self, camera_pos: q3_math::Vec3) {
        if self.chunks.len() > MAX_CACHE_CHUNKS {
            debug!(
                "terrain-gpu: cache plein ({}), reset",
                self.chunks.len()
            );
            self.chunks.clear();
        }

        self.current_lod.clear();
        let cd_sq = Self::RENDER_DISTANCE * Self::RENDER_DISTANCE;
        for coord in self.terrain.iter_chunks() {
            let center = self.terrain.chunk_center(coord);
            let dx = center.x - camera_pos.x;
            let dy = center.y - camera_pos.y;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq > cd_sq {
                continue; // distance culling
            }
            let lod = self.terrain.select_lod(coord, camera_pos);
            self.current_lod.insert(coord, lod);
            if !self.chunks.contains_key(&(coord, lod)) {
                let mesh = self.terrain.build_chunk_mesh(coord, lod);
                let gpu = ChunkGpu::from_mesh(&self.device, &mesh);
                self.chunks.insert((coord, lod), gpu);
            }
        }
    }

    /// Dessine tous les chunks de la carte (selon leur LOD courant) dans
    /// le `RenderPass` fourni.  Le caller doit avoir bind le `camera_bg`
    /// sur group 0.  On bind ici le `params_bg` sur group 1.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(1, &self.params_bind_group, &[]);
        for (coord, lod) in &self.current_lod {
            if let Some(g) = self.chunks.get(&(*coord, *lod)) {
                pass.set_vertex_buffer(0, g.vertex_buffer.slice(..));
                pass.set_index_buffer(g.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..g.index_count, 0, 0..1);
            }
        }
    }

    /// Stats simples pour debug HUD : chunks affichés à chaque LOD.
    pub fn stats(&self) -> TerrainStats {
        let mut by_lod = [0u32; 4];
        for lod in self.current_lod.values() {
            by_lod[*lod as usize] += 1;
        }
        TerrainStats {
            visible_chunks: self.current_lod.len() as u32,
            cached_chunks: self.chunks.len() as u32,
            chunks_by_lod: by_lod,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TerrainStats {
    pub visible_chunks: u32,
    pub cached_chunks: u32,
    pub chunks_by_lod: [u32; 4],
}
