//! Volumes de brouillard BSP.
//!
//! Chaque entrée du lump `fogs` définit :
//! * un shader (dont le bloc `fogparms` donne `color` + `distance`)
//! * un `brush_num` dont l'AABB axis-aligned est le volume de brouillard
//!
//! Sémantique Q3 (simplifiée) : quand la caméra est *à l'intérieur* du brush,
//! l'écran entier est brumisé ; quand la caméra regarde une surface qui
//! traverse le brush, le fragment est brumisé selon sa distance à la face
//! visible.  Pour l'instant on n'implémente que le premier cas (caméra dans
//! un volume) — c'est la situation visuellement la plus saillante (nager
//! dans de l'eau ou du lava) et celle qui ne nécessite aucun shader
//! par-surface.

use bytemuck::{Pod, Zeroable};
use q3_bsp::Bsp;
use q3_math::{Aabb, Vec3};
use q3_shader::{ShaderRegistry, value::FogParms};
use std::sync::Arc;
use tracing::debug;

/// Un volume de brouillard résolu.
#[derive(Debug, Clone)]
pub struct FogVolume {
    /// AABB axis-aligned du brush associé.
    pub aabb: Aabb,
    /// Nom du shader, lowercase. Conservé pour diagnostic / lookup tardif.
    pub shader_name: String,
    /// `fogparms` extrait du shader (None si le shader n'en définit pas,
    /// ou si le registry n'était pas disponible à la construction).
    pub fog_parms: Option<FogParms>,
}

/// Liste de tous les volumes de brouillard d'une map.
#[derive(Debug, Clone, Default)]
pub struct FogSet {
    pub volumes: Vec<FogVolume>,
}

impl FogSet {
    /// Construit le set à partir des fogs du BSP. Si `registry` est `Some`,
    /// on résout en même temps les `fogparms` de chaque shader.
    pub fn build(bsp: &Bsp, registry: Option<&ShaderRegistry>) -> Self {
        let mut volumes = Vec::with_capacity(bsp.fogs.len());
        for fog in bsp.fogs.iter() {
            let Some(aabb) = bsp.brush_aabb(fog.brush_num as usize) else {
                continue;
            };
            let shader_name = fog.name().to_ascii_lowercase();
            let fog_parms = registry
                .and_then(|r| r.get(&shader_name))
                .and_then(|s| s.fog_parms.clone());
            volumes.push(FogVolume {
                aabb,
                shader_name,
                fog_parms,
            });
        }
        debug!(
            "fog: {} volumes construits ({} avec fogparms)",
            volumes.len(),
            volumes.iter().filter(|v| v.fog_parms.is_some()).count()
        );
        Self { volumes }
    }

    /// Re-résout les `fog_parms` de chaque volume depuis un registry (utile
    /// quand le BSP est uploadé *avant* `attach_materials`). Les volumes
    /// eux-mêmes ne sont pas recréés.
    pub fn resolve_parms(&mut self, registry: &ShaderRegistry) {
        for v in self.volumes.iter_mut() {
            v.fog_parms = registry
                .get(&v.shader_name)
                .and_then(|s| s.fog_parms.clone());
        }
        debug!(
            "fog: re-résolution, {} / {} volumes ont maintenant un fogparms",
            self.volumes.iter().filter(|v| v.fog_parms.is_some()).count(),
            self.volumes.len()
        );
    }

    /// Retourne le premier volume qui contient `eye`, ou `None`. Les volumes
    /// Q3 ne devraient jamais se recouvrir dans une map bien compilée, donc
    /// le premier match suffit.
    pub fn active_at(&self, eye: Vec3) -> Option<&FogVolume> {
        self.volumes.iter().find(|v| v.aabb.contains(eye))
    }

    pub fn is_empty(&self) -> bool {
        self.volumes.is_empty()
    }

    pub fn len(&self) -> usize {
        self.volumes.len()
    }
}

/// État GPU envoyé au frag shader chaque frame.
///
/// Layout WGSL correspondant :
/// ```wgsl
/// struct FogState {
///     color_distance: vec4<f32>,  // rgb = couleur, a = distance (u)
///     active: u32,                // 0 ou 1 — caméra dans un volume ?
///     _pad: vec3<u32>,
/// };
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuFog {
    pub color_distance: [f32; 4],
    pub active: u32,
    pub _pad: [u32; 3],
}

impl Default for GpuFog {
    fn default() -> Self {
        Self {
            color_distance: [0.5, 0.5, 0.5, 1024.0],
            active: 0,
            _pad: [0; 3],
        }
    }
}

/// Uniform buffer « fog actif » — mis à jour chaque frame par le renderer.
pub struct FogUniform {
    queue: Arc<wgpu::Queue>,
    pub bind_group_layout: Arc<wgpu::BindGroupLayout>,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl FogUniform {
    pub fn new(device: &wgpu::Device, queue: Arc<wgpu::Queue>) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fog-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fog-ubuf"),
            size: std::mem::size_of::<GpuFog>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&GpuFog::default()));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fog-bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            queue,
            bind_group_layout: Arc::new(bgl),
            buffer,
            bind_group,
        }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Écrit dans le buffer GPU : volume actif si `Some`, sinon état
    /// inactif (le shader fera un no-op sur les fragments).
    pub fn write(&self, active: Option<&FogVolume>) {
        let gpu = match active.and_then(|v| v.fog_parms.as_ref()) {
            Some(parms) => GpuFog {
                color_distance: [parms.color[0], parms.color[1], parms.color[2], parms.distance],
                active: 1,
                _pad: [0; 3],
            },
            None => GpuFog {
                active: 0,
                ..GpuFog::default()
            },
        };
        self.queue
            .write_buffer(&self.buffer, 0, bytemuck::bytes_of(&gpu));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use q3_bsp::raw::{
        DBrush, DBrushSide, DFog, DLeaf, DModel, DNode, DPlane, DShader, DSurface, DrawVert,
    };
    use q3_bsp::Visibility;

    fn axis_planes() -> Vec<DPlane> {
        vec![
            DPlane { normal: [1.0, 0.0, 0.0], dist: 32.0 },
            DPlane { normal: [-1.0, 0.0, 0.0], dist: 32.0 },
            DPlane { normal: [0.0, 1.0, 0.0], dist: 16.0 },
            DPlane { normal: [0.0, -1.0, 0.0], dist: 16.0 },
            DPlane { normal: [0.0, 0.0, 1.0], dist: 8.0 },
            DPlane { normal: [0.0, 0.0, -1.0], dist: 8.0 },
        ]
    }

    fn bsp_with_one_fog_brush(shader_name: &str) -> Bsp {
        let mut shader_bytes = [0u8; 64];
        for (i, b) in shader_name.as_bytes().iter().enumerate().take(63) {
            shader_bytes[i] = *b;
        }
        Bsp {
            entities: String::new(),
            shaders: vec![DShader {
                shader: [0; 64],
                surface_flags: 0,
                content_flags: 0,
            }],
            planes: axis_planes(),
            nodes: vec![DNode {
                plane_num: 0,
                children: [-1, -1],
                mins: [-64; 3],
                maxs: [64; 3],
            }],
            leafs: vec![DLeaf {
                cluster: 0,
                area: 0,
                mins: [-64; 3],
                maxs: [64; 3],
                first_leaf_surface: 0,
                num_leaf_surfaces: 0,
                first_leaf_brush: 0,
                num_leaf_brushes: 0,
            }],
            leaf_surfaces: vec![],
            leaf_brushes: vec![],
            models: vec![DModel {
                mins: [-64.0; 3],
                maxs: [64.0; 3],
                first_surface: 0,
                num_surfaces: 0,
                first_brush: 0,
                num_brushes: 1,
            }],
            brushes: vec![DBrush {
                first_side: 0,
                num_sides: 6,
                shader_num: 0,
            }],
            brush_sides: (0..6)
                .map(|i| DBrushSide {
                    plane_num: i,
                    shader_num: 0,
                })
                .collect(),
            draw_verts: Vec::<DrawVert>::new(),
            draw_indexes: vec![],
            fogs: vec![DFog {
                shader: shader_bytes,
                brush_num: 0,
                visible_side: -1,
            }],
            surfaces: Vec::<DSurface>::new(),
            lightmap_bytes: vec![],
            lightgrid_bytes: vec![],
            visibility: Visibility::default(),
        }
    }

    #[test]
    fn build_without_registry_has_aabb_no_parms() {
        let bsp = bsp_with_one_fog_brush("textures/liquids/slime");
        let fogs = FogSet::build(&bsp, None);
        assert_eq!(fogs.volumes.len(), 1);
        let v = &fogs.volumes[0];
        assert_eq!(v.shader_name, "textures/liquids/slime");
        assert!(v.fog_parms.is_none());
        assert_eq!(v.aabb.mins, Vec3::new(-32.0, -16.0, -8.0));
        assert_eq!(v.aabb.maxs, Vec3::new(32.0, 16.0, 8.0));
    }

    #[test]
    fn build_with_registry_resolves_fogparms() {
        let bsp = bsp_with_one_fog_brush("textures/liquids/slime");
        let src = r#"
            textures/liquids/slime
            {
                fogparms ( 0.1 0.5 0.2 ) 256
            }
        "#;
        let mut reg = ShaderRegistry::new();
        reg.parse_file(src, "test.shader");
        let fogs = FogSet::build(&bsp, Some(&reg));
        let v = &fogs.volumes[0];
        let fp = v.fog_parms.as_ref().expect("fogparms");
        assert_eq!(fp.color, [0.1, 0.5, 0.2]);
        assert_eq!(fp.distance, 256.0);
    }

    #[test]
    fn active_at_detects_containment() {
        let bsp = bsp_with_one_fog_brush("anyshader");
        let fogs = FogSet::build(&bsp, None);
        assert!(fogs.active_at(Vec3::ZERO).is_some());
        assert!(fogs.active_at(Vec3::new(100.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn gpu_fog_size_is_32_bytes() {
        // vec4 (16) + u32 (4) + 3×u32 pad (12) = 32 bytes, std140-ready.
        assert_eq!(std::mem::size_of::<GpuFog>(), 32);
    }

    #[test]
    fn gpu_fog_default_is_inactive() {
        let g = GpuFog::default();
        assert_eq!(g.active, 0);
    }

    #[test]
    fn resolve_parms_fills_after_build() {
        let bsp = bsp_with_one_fog_brush("textures/liquids/water");
        let mut fogs = FogSet::build(&bsp, None);
        assert!(fogs.volumes[0].fog_parms.is_none());

        let src = "textures/liquids/water { fogparms ( 0.0 0.3 0.7 ) 512 }";
        let mut reg = ShaderRegistry::new();
        reg.parse_file(src, "t.shader");
        fogs.resolve_parms(&reg);

        let fp = fogs.volumes[0].fog_parms.as_ref().expect("fogparms");
        assert_eq!(fp.distance, 512.0);
        assert_eq!(fp.color, [0.0, 0.3, 0.7]);
    }
}
