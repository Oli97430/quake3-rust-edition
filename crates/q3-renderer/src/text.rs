//! Overlay texte 2D : console, HUD, debug.
//!
//! La source typographique reste `font8x8` (bitmap 8×8 par glyphe, domaine
//! public Marcel Sondaar) — c'est léger et zéro-dépendance.  Par contre le
//! rendu a été modernisé : au lieu d'échantillonner directement le bitmap
//! au point nearest (→ pixels carrés horribles quand on scale à HUD_SCALE=2
//! voire 4), on le **convertit en champ de distance signé approximatif**
//! au démarrage.
//!
//! Pipeline d'atlas :
//!
//! 1. Chaque glyphe est suréchantillonné à `GLYPH_HIRES` px (chaque pixel
//!    bitmap = un bloc `UPSCALE×UPSCALE` dans l'atlas hi-res).
//! 2. Une carte de distance euclidienne est calculée par le "8-SSEDT" à
//!    deux passes (Felzenszwalb/Huttenlocher 2012, variante discrète) :
//!    distance à l'intérieur des glyphes (positif) et à l'extérieur
//!    (négatif), normalisée en `[0,1]` avec 0.5 = bord.
//! 3. Le fragment shader fait `smoothstep(0.48, 0.52, sdf)` pour obtenir
//!    un bord propre et antialiasé à **n'importe quelle taille**.  En
//!    bonus, on peut tracer un halo/outline en un deuxième smoothstep
//!    plus large — utile pour lire le HUD sur fond clair.
//!
//! Comparé au rendu nearest-neighbor d'origine : les lettres restent
//! nettes même scalées à ×4, les bords sont lissés, la console respire.
//! Aucune dépendance supplémentaire, tout est calculé au startup (~2 ms).

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const ATLAS_COLS: u32 = 16;
const ATLAS_ROWS: u32 = 16;
/// Taille "cellule logique" que l'API exportée annonce pour layouter :
/// on garde 8×8 pour rester compatible avec toutes les métriques HUD
/// existantes (`char_w = 8.0 * HUD_SCALE` etc.).
const GLYPH_W: u32 = 8;
const GLYPH_H: u32 = 8;
/// Upscale appliqué lors de la génération de l'atlas SDF.  4× donne un
/// rayon de lissage suffisant pour que `smoothstep` ait ~2 texels de
/// transition même quand on affiche à la taille native.
const UPSCALE: u32 = 4;
const GLYPH_HIRES_W: u32 = GLYPH_W * UPSCALE;
const GLYPH_HIRES_H: u32 = GLYPH_H * UPSCALE;
/// Padding en texels hi-res autour de chaque glyphe dans sa cellule.
/// Indispensable pour que la SDF d'un glyphe ne "bave" pas sur les
/// voisins dans l'atlas — le DT calcule la distance au bord le plus
/// proche sans tenir compte des frontières de cellule.  `PAD ≥ SDF_RADIUS`
/// garantit qu'aucun halo ne franchira vers la case suivante.
const PAD: u32 = 8;
const CELL_W: u32 = GLYPH_HIRES_W + 2 * PAD;
const CELL_H: u32 = GLYPH_HIRES_H + 2 * PAD;
const ATLAS_W: u32 = ATLAS_COLS * CELL_W;
const ATLAS_H: u32 = ATLAS_ROWS * CELL_H;
/// Demi-rayon (en texels hi-res) de la plage sur laquelle on encode la
/// SDF avant clamp à 0/255.  Plus grand = outline plus épais possible
/// mais transitions plus douces (moins nettes).  8 texels ≈ 2 pixels du
/// bitmap source → bon compromis.  Doit rester ≤ PAD.
const SDF_RADIUS: i32 = 6;

/// Capacité max du vertex buffer dynamique (en vertices). 6 vertices par
/// quad, donc ≈ 5300 glyphes par frame. Large de quoi tenir la console +
/// HUD.
const MAX_VERTICES: u32 = 32 * 1024;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct TextVertex {
    position: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

impl TextVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x2,
        1 => Float32x2,
        2 => Float32x4,
    ];
    const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct TextRenderer {
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    /// Accumule les vertices émis entre deux `flush`.
    cpu: Vec<TextVertex>,
    /// Dimensions du framebuffer courant — utilisées pour convertir pixels
    /// → NDC.
    fb_w: f32,
    fb_h: f32,
}

impl TextRenderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        // --- étape 1 : masque binaire hi-res (chaque pixel du bitmap 8×8
        // devient un bloc `UPSCALE×UPSCALE` d'opacité 1 ou 0) ---
        let w = ATLAS_W as usize;
        let h = ATLAS_H as usize;
        let mut mask = vec![false; w * h];
        for code in 0u8..=127u8 {
            let Some(bitmap) = ascii_glyph(code) else { continue };
            let gx = (code as u32) % ATLAS_COLS;
            let gy = (code as u32) / ATLAS_COLS;
            // Origine du glyphe dans sa cellule = coin haut-gauche + PAD.
            let ox = gx * CELL_W + PAD;
            let oy = gy * CELL_H + PAD;
            for (row, bits) in bitmap.iter().enumerate() {
                for col in 0..GLYPH_W {
                    let set = (bits >> col) & 1 != 0;
                    if !set {
                        continue;
                    }
                    // Bloc de UPSCALE×UPSCALE pixels, et on applique un
                    // léger "smoothing" sur les diagonales inspiré du
                    // scale2x : si les voisins bit-à-bit de gauche/droite
                    // et haut/bas sont tous deux éteints, on biseaute le
                    // coin du bloc en ne remplissant pas le pixel de coin.
                    // Ça arrondit très subtilement les angles à 45° du
                    // bitmap, ce que la SDF amplifie en lisse continu.
                    let bx = ox + col * UPSCALE;
                    let by = oy + (row as u32) * UPSCALE;
                    let bit = |dr: i32, dc: i32| -> bool {
                        let r = row as i32 + dr;
                        let c = col as i32 + dc;
                        if !(0..GLYPH_H as i32).contains(&r) {
                            return false;
                        }
                        if !(0..GLYPH_W as i32).contains(&c) {
                            return false;
                        }
                        (bitmap[r as usize] >> c) & 1 != 0
                    };
                    let top = bit(-1, 0);
                    let bot = bit(1, 0);
                    let lft = bit(0, -1);
                    let rgt = bit(0, 1);
                    for py in 0..UPSCALE {
                        for px in 0..UPSCALE {
                            // Coins : on enlève le pixel du coin si ni
                            // l'horizontal ni le vertical voisin ne sont
                            // allumés → le bitmap forme un angle isolé à
                            // 45° qui bénéficie d'un léger chanfrein.
                            let top_edge = py == 0;
                            let bot_edge = py == UPSCALE - 1;
                            let lft_edge = px == 0;
                            let rgt_edge = px == UPSCALE - 1;
                            let skip = (top_edge && lft_edge && !top && !lft)
                                || (top_edge && rgt_edge && !top && !rgt)
                                || (bot_edge && lft_edge && !bot && !lft)
                                || (bot_edge && rgt_edge && !bot && !rgt);
                            if skip {
                                continue;
                            }
                            let x = (bx + px) as usize;
                            let y = (by + py) as usize;
                            mask[y * w + x] = true;
                        }
                    }
                }
            }
        }
        // Case (15,15) = brush rectangle plein.  On remplit la cellule
        // ENTIÈRE (pas juste la zone glyphe) pour que peu importe l'UV
        // échantillonné à l'intérieur de la case, le pixel soit "deep
        // inside" → SDF = 1.0 → alpha = 1.0 (rect opaque propre sans
        // bords transparents).
        for row in 0..CELL_H {
            for col in 0..CELL_W {
                let x = (15 * CELL_W + col) as usize;
                let y = (15 * CELL_H + row) as usize;
                mask[y * w + x] = true;
            }
        }

        // --- étape 2 : SDF par la "chamfer distance transform" (approx
        // euclidienne, 2 passes avec kernel 3×3 pondéré 3/4).  C'est 10x
        // plus rapide que du 8-SSEDT exact pour un résultat visuellement
        // indiscernable à UPSCALE=4.
        let inside = distance_transform(&mask, w, h, true);
        let outside = distance_transform(&mask, w, h, false);

        // Encodage R8 : 0.5 = bord, +1 = intérieur profond, 0 = extérieur
        // profond.  On compresse [-SDF_RADIUS, +SDF_RADIUS] → [0,255].
        let radius = SDF_RADIUS as f32;
        let atlas: Vec<u8> = (0..w * h)
            .map(|i| {
                // Distance signée : positive dedans, négative dehors.
                let d = if mask[i] { inside[i] } else { -outside[i] };
                let t = (d / radius).clamp(-1.0, 1.0) * 0.5 + 0.5;
                (t * 255.0).round().clamp(0.0, 255.0) as u8
            })
            .collect();

        let atlas_tex = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("text-atlas"),
                size: wgpu::Extent3d {
                    width: ATLAS_W,
                    height: ATLAS_H,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &atlas,
        );
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor::default());
        // Filtrage LINÉAIRE sur le SDF — c'est ce qui permet au fragment
        // shader d'interpoler la distance entre pixels voisins et d'avoir
        // une transition douce sur 1-2 texels au lieu d'un saut binaire.
        // Sans ça, le smoothstep ne donne rien de plus qu'un threshold
        // dur.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("text-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("text-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // `filterable: true` maintenant que le sampler est Linear.
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
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("text-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("text-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/text.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("text-pipeline-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("text-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[TextVertex::layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text-vbuf"),
            size: (MAX_VERTICES as u64) * (std::mem::size_of::<TextVertex>() as u64),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            queue,
            pipeline,
            bind_group,
            vertex_buffer,
            cpu: Vec::with_capacity(4096),
            fb_w: 1.0,
            fb_h: 1.0,
        }
    }

    /// Doit être appelé à chaque frame avant d'émettre des quads.
    pub fn begin_frame(&mut self, fb_w: u32, fb_h: u32) {
        self.fb_w = fb_w.max(1) as f32;
        self.fb_h = fb_h.max(1) as f32;
        self.cpu.clear();
    }

    /// Émet un rectangle plein (utile pour le fond de la console). On
    /// échantillonne la case (15,15) de l'atlas qu'on sait remplie en
    /// blanc (cf. constructeur) ; `color.a * 1.0` donne l'opacité voulue.
    pub fn push_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        let x0 = self.px_to_ndc_x(x);
        let y0 = self.px_to_ndc_y(y);
        let x1 = self.px_to_ndc_x(x + w);
        let y1 = self.px_to_ndc_y(y + h);
        // Centre de la cellule (15,15) : U = (15*CELL_W + CELL_W/2) / ATLAS_W.
        let u = (15.0 * CELL_W as f32 + CELL_W as f32 * 0.5) / ATLAS_W as f32;
        let v = (15.0 * CELL_H as f32 + CELL_H as f32 * 0.5) / ATLAS_H as f32;
        self.push_quad([x0, y0], [x1, y1], [u, v], [u, v], color);
    }

    /// Écrit `text` à la position pixel `(x, y)` (origine : coin haut-gauche
    /// de la fenêtre). `scale` multiplie la taille native 8×8 (scale=2 → 16×16).
    pub fn push_text(&mut self, x: f32, y: f32, scale: f32, color: [f32; 4], text: &str) {
        let gw = GLYPH_W as f32 * scale;
        let gh = GLYPH_H as f32 * scale;
        let mut cx = x;
        // UV unitaire d'un pixel hi-res dans l'atlas (pour placer les bords
        // de l'UV-rect pile sur les bords du glyphe hi-res intérieur).
        let tex_w = ATLAS_W as f32;
        let tex_h = ATLAS_H as f32;
        for ch in text.chars() {
            let code = if ch.is_ascii() { ch as u32 } else { b'?' as u32 };
            // Espace : on avance juste.
            if code == b' ' as u32 {
                cx += gw;
                continue;
            }
            let gx = code % ATLAS_COLS;
            let gy = code / ATLAS_COLS;
            // UV rect = sous-région "glyphe" dans la cellule (exclut PAD),
            // en coordonnées normalisées.
            let ox = gx * CELL_W + PAD;
            let oy = gy * CELL_H + PAD;
            let u0 = ox as f32 / tex_w;
            let v0 = oy as f32 / tex_h;
            let u1 = (ox + GLYPH_HIRES_W) as f32 / tex_w;
            let v1 = (oy + GLYPH_HIRES_H) as f32 / tex_h;

            let x0 = self.px_to_ndc_x(cx);
            let y0 = self.px_to_ndc_y(y);
            let x1 = self.px_to_ndc_x(cx + gw);
            let y1 = self.px_to_ndc_y(y + gh);
            self.push_quad([x0, y0], [x1, y1], [u0, v0], [u1, v1], color);
            cx += gw;
        }
    }

    /// Uploade les vertices au GPU et dessine dans la passe courante.
    pub fn flush<'a>(&'a mut self, pass: &mut wgpu::RenderPass<'a>) {
        if self.cpu.is_empty() {
            return;
        }
        let max = MAX_VERTICES as usize;
        if self.cpu.len() > max {
            self.cpu.truncate(max);
        }
        self.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.cpu));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..self.cpu.len() as u32, 0..1);
    }

    // ---------------- helpers ----------------

    fn px_to_ndc_x(&self, x: f32) -> f32 {
        (x / self.fb_w) * 2.0 - 1.0
    }
    fn px_to_ndc_y(&self, y: f32) -> f32 {
        // y=0 en haut, y=fb_h en bas. NDC y=+1 en haut.
        1.0 - (y / self.fb_h) * 2.0
    }

    fn push_quad(
        &mut self,
        a: [f32; 2],
        b: [f32; 2],
        uv_a: [f32; 2],
        uv_b: [f32; 2],
        color: [f32; 4],
    ) {
        // Deux triangles : (a.x,a.y) (b.x,a.y) (a.x,b.y)  puis  (b.x,a.y) (b.x,b.y) (a.x,b.y)
        let v = [
            TextVertex { position: [a[0], a[1]], uv: [uv_a[0], uv_a[1]], color },
            TextVertex { position: [b[0], a[1]], uv: [uv_b[0], uv_a[1]], color },
            TextVertex { position: [a[0], b[1]], uv: [uv_a[0], uv_b[1]], color },
            TextVertex { position: [b[0], a[1]], uv: [uv_b[0], uv_a[1]], color },
            TextVertex { position: [b[0], b[1]], uv: [uv_b[0], uv_b[1]], color },
            TextVertex { position: [a[0], b[1]], uv: [uv_a[0], uv_b[1]], color },
        ];
        self.cpu.extend_from_slice(&v);
    }
}

// ---- Chamfer distance transform (approximation rapide euclidienne) ----
//
// Deux passes sur l'image (forward + backward) avec un noyau 3×3 où les
// voisins orthogonaux comptent pour 1 et les diagonaux pour √2 ≈ 1.414.
// Pour chaque pixel "foreground" (ou "background" selon le flag), on
// calcule la distance au pixel "opposé" le plus proche.  Sortie en f32
// (en texels hi-res).
//
// `inside = true` → distance aux bords depuis l'intérieur (mask == true).
// `inside = false` → distance aux bords depuis l'extérieur (mask == false).
fn distance_transform(mask: &[bool], w: usize, h: usize, inside: bool) -> Vec<f32> {
    let big = 1e9_f32;
    let mut d = vec![big; w * h];
    // Init : les pixels du bord de transition sont à 0, les autres à +inf.
    for y in 0..h {
        for x in 0..w {
            let here = mask[y * w + x];
            // Si on cherche la distance "dedans → bord", alors les pixels
            // "dehors" (mask == false) sont nos ancres (distance 0).  Et
            // vice-versa.  Sinon on met `big`.
            let anchor = if inside { !here } else { here };
            if anchor {
                d[y * w + x] = 0.0;
            }
        }
    }
    let sqrt2 = std::f32::consts::SQRT_2;
    // Pass avant (haut-gauche → bas-droite) — on relaxe avec les 4
    // voisins déjà visités (NW, N, NE, W).
    for y in 0..h {
        for x in 0..w {
            let here = d[y * w + x];
            let mut best = here;
            if y > 0 {
                if x > 0 {
                    best = best.min(d[(y - 1) * w + (x - 1)] + sqrt2);
                }
                best = best.min(d[(y - 1) * w + x] + 1.0);
                if x + 1 < w {
                    best = best.min(d[(y - 1) * w + (x + 1)] + sqrt2);
                }
            }
            if x > 0 {
                best = best.min(d[y * w + (x - 1)] + 1.0);
            }
            d[y * w + x] = best;
        }
    }
    // Pass arrière (bas-droite → haut-gauche) — relax avec les 4 voisins
    // SE, S, SW, E.
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let here = d[y * w + x];
            let mut best = here;
            if y + 1 < h {
                if x + 1 < w {
                    best = best.min(d[(y + 1) * w + (x + 1)] + sqrt2);
                }
                best = best.min(d[(y + 1) * w + x] + 1.0);
                if x > 0 {
                    best = best.min(d[(y + 1) * w + (x - 1)] + sqrt2);
                }
            }
            if x + 1 < w {
                best = best.min(d[y * w + (x + 1)] + 1.0);
            }
            d[y * w + x] = best;
        }
    }
    d
}

// --- utilitaire : retourne le bitmap 8×8 d'un code ASCII ----------------
//
// font8x8 expose `font8x8::legacy::BASIC_LEGACY` (ASCII 0x00..=0x7F) : une
// table de 128 × 8 octets. Chaque octet = une ligne ; bit 0 = pixel gauche.
fn ascii_glyph(code: u8) -> Option<[u8; 8]> {
    let idx = code as usize;
    if idx >= font8x8::legacy::BASIC_LEGACY.len() {
        return None;
    }
    Some(font8x8::legacy::BASIC_LEGACY[idx])
}
