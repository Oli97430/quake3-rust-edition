//! Conversion `Bsp` → buffers GPU.
//!
//! On agglomère les surfaces `Planar`, `TriangleSoup` et `Patch` (tessellées
//! via `tessellate_patch`) en un seul vertex/index buffer partagé ; un
//! `Vec<DrawRange>` indexe les tranches par `(shader_num, lightmap_num)`
//! pour regrouper les drawcalls.  Les `Flare` (billboards) restent hors
//! scope — elles sont traitées par un pass séparé.

use crate::GpuVertex;
use bytemuck::cast_slice;
use q3_bsp::{
    patch::{tessellate_patch, DEFAULT_TESSELLATION_LEVEL},
    raw::{DrawVert, SurfaceType},
    Bsp,
};
use q3_common::Result;
use tracing::debug;
use wgpu::util::DeviceExt;

/// Métadonnées d'un drawcall (une surface ≈ un shader + lightmap).
#[derive(Debug, Clone)]
pub struct DrawRange {
    pub first_index: u32,
    pub index_count: u32,
    pub shader_num: i32,
    pub lightmap_num: i32,
    /// Nom du shader résolu depuis `bsp.shaders[shader_num]` — lowercase,
    /// prêt à passer au `MaterialCache`.
    pub shader_name: String,
}

pub struct BspMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub vertex_count: u32,
    pub index_count: u32,
    pub triangle_count: u32,
    pub draw_count: u32,
    pub draws: Vec<DrawRange>,
}

impl BspMesh {
    /// Construit le mesh GPU. `white_layer` est l'index de la couche blanche
    /// dans le `LightmapArray` — utilisée comme fallback quand une surface
    /// n'a pas de lightmap (`lightmap_num < 0`).
    pub fn build(device: &wgpu::Device, bsp: &Bsp, white_layer: u32) -> Result<Self> {
        // Réserve approximativement — on ajoutera les patches plus tard.
        let approx_verts = bsp.draw_verts.len();
        let approx_indexes = bsp.draw_indexes.len();

        let mut vertices: Vec<GpuVertex> = Vec::with_capacity(approx_verts);
        let mut indexes: Vec<u32> = Vec::with_capacity(approx_indexes);
        let mut draws: Vec<DrawRange> = Vec::with_capacity(bsp.surfaces.len());

        // Pour associer un lightmap_layer à chaque vertex, on doit savoir
        // quelle surface l'a référencé en premier. On construit donc un
        // tableau `vert_lightmap_layer[i]` = layer de la 1ère surface
        // qui a utilisé le vertex `i`. Les surfaces Planar/TriSoup partagent
        // souvent leurs verts avec la même lightmap dans la pratique.
        let base_vertex_count = bsp.draw_verts.len();
        let mut vert_layer = vec![u32::MAX; base_vertex_count];
        for surf in bsp.surfaces.iter() {
            if !matches!(
                surf.kind(),
                SurfaceType::Planar | SurfaceType::TriangleSoup
            ) {
                continue;
            }
            let layer = resolve_layer(surf.lightmap_num, white_layer);
            let start = surf.first_vert as usize;
            let end = start + surf.num_verts as usize;
            for slot in vert_layer.get_mut(start..end).into_iter().flatten() {
                if *slot == u32::MAX {
                    *slot = layer;
                }
            }
        }
        for v in vert_layer.iter_mut() {
            if *v == u32::MAX {
                *v = white_layer;
            }
        }
        for (dv, &layer) in bsp.draw_verts.iter().zip(vert_layer.iter()) {
            vertices.push(to_gpu_vert(dv, layer));
        }

        let mut patch_count = 0usize;

        for surf in bsp.surfaces.iter() {
            match surf.kind() {
                SurfaceType::Planar | SurfaceType::TriangleSoup => {
                    let first_index = indexes.len() as u32;
                    let base_vert = surf.first_vert;
                    let start = surf.first_index as usize;
                    let end = start + surf.num_indexes as usize;
                    if let Some(slice) = bsp.draw_indexes.get(start..end) {
                        for &idx in slice {
                            indexes.push((base_vert + idx) as u32);
                        }
                    }
                    let count = indexes.len() as u32 - first_index;
                    if count > 0 {
                        draws.push(DrawRange {
                            first_index,
                            index_count: count,
                            shader_num: surf.shader_num,
                            lightmap_num: surf.lightmap_num,
                            shader_name: resolve_shader_name(bsp, surf.shader_num),
                        });
                    }
                }
                SurfaceType::Patch => {
                    let start = surf.first_vert as usize;
                    let end = start + surf.num_verts as usize;
                    let Some(cps) = bsp.draw_verts.get(start..end) else {
                        continue;
                    };
                    let Some(tess) = tessellate_patch(
                        cps,
                        surf.patch_width,
                        surf.patch_height,
                        DEFAULT_TESSELLATION_LEVEL,
                    ) else {
                        continue;
                    };
                    let first_index = indexes.len() as u32;
                    let vert_base = vertices.len() as u32;
                    let layer = resolve_layer(surf.lightmap_num, white_layer);
                    vertices.extend(tess.vertices.iter().map(|v| to_gpu_vert(v, layer)));
                    for idx in tess.indexes {
                        indexes.push(vert_base + idx);
                    }
                    let count = indexes.len() as u32 - first_index;
                    draws.push(DrawRange {
                        first_index,
                        index_count: count,
                        shader_num: surf.shader_num,
                        lightmap_num: surf.lightmap_num,
                        shader_name: resolve_shader_name(bsp, surf.shader_num),
                    });
                    patch_count += 1;
                }
                SurfaceType::Flare | SurfaceType::Bad => {
                    // flares = billboards sprites, gérés par un pass séparé plus tard
                }
            }
        }

        debug!(
            "bsp_mesh: {} base verts + {} patches tessellés",
            base_vertex_count, patch_count
        );

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bsp-vertex-buffer"),
            contents: cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bsp-index-buffer"),
            contents: cast_slice(&indexes),
            usage: wgpu::BufferUsages::INDEX,
        });

        Ok(Self {
            vertex_buffer,
            index_buffer,
            vertex_count: vertices.len() as u32,
            index_count: indexes.len() as u32,
            triangle_count: (indexes.len() / 3) as u32,
            draw_count: draws.len() as u32,
            draws,
        })
    }
}

fn to_gpu_vert(dv: &DrawVert, lightmap_layer: u32) -> GpuVertex {
    GpuVertex {
        position: dv.xyz,
        normal: dv.normal,
        tex_uv: dv.st,
        lightmap_uv: dv.lightmap,
        color: [
            dv.color[0] as f32 / 255.0,
            dv.color[1] as f32 / 255.0,
            dv.color[2] as f32 / 255.0,
            dv.color[3] as f32 / 255.0,
        ],
        lightmap_layer,
        _pad: 0,
    }
}

fn resolve_layer(lm_num: i32, white_layer: u32) -> u32 {
    if lm_num < 0 {
        white_layer
    } else {
        lm_num as u32
    }
}

fn resolve_shader_name(bsp: &Bsp, shader_num: i32) -> String {
    if shader_num < 0 {
        return String::new();
    }
    bsp.shaders
        .get(shader_num as usize)
        .map(|s| s.name().to_ascii_lowercase())
        .unwrap_or_default()
}
