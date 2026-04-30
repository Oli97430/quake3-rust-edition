//! `xtask` — petits outils de build du workspace.
//!
//! Pour l'instant : `gen-icon` qui produit `assets/icon.ico` à partir
//! de `assets/branding/logo.svg`. Idiomatique cargo : on lance via
//! `cargo run -p xtask -- gen-icon`. Pas de dépendance système (pas
//! d'`inkscape` / `magick` requis) : tout en Rust pur via `resvg`.

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    match cmd.as_str() {
        "gen-icon" => gen_icon(),
        other => {
            eprintln!("Usage: cargo run -p xtask -- <gen-icon>");
            eprintln!("Commande inconnue : `{other}`");
            std::process::exit(2);
        }
    }
}

/// Génère `assets/icon.ico` (multi-résolutions 16/32/48/64/128/256) à
/// partir de `assets/branding/logo.svg`. L'icone est embarqué au
/// build par `winres` (cf. `crates/q3-engine/build.rs`).
fn gen_icon() -> anyhow::Result<()> {
    let workspace = workspace_root();
    let svg_path = workspace.join("assets/branding/logo.svg");
    let out_path = workspace.join("assets/icon.ico");

    println!("rendu  : {}", svg_path.display());
    println!("sortie : {}", out_path.display());

    let svg_data = std::fs::read(&svg_path)
        .map_err(|e| anyhow::anyhow!("lecture {}: {e}", svg_path.display()))?;
    let opt = usvg::Options::default();
    let tree = usvg::Tree::from_data(&svg_data, &opt)
        .map_err(|e| anyhow::anyhow!("parse SVG : {e}"))?;

    // Tailles standard Windows : 16 (small), 32 (icon list), 48 (large),
    // 64 (ext explorer), 128 (preview), 256 (jumbo Win10/11). Couvre
    // tous les contextes d'affichage moderne.
    let sizes = [16u32, 32, 48, 64, 128, 256];
    let mut icon_dir = ico::IconDir::new(ico::ResourceType::Icon);

    for &px in &sizes {
        let mut pixmap = tiny_skia::Pixmap::new(px, px)
            .ok_or_else(|| anyhow::anyhow!("alloc pixmap {px}×{px}"))?;
        let view = tree.size().to_int_size().to_size();
        // Scale uniforme pour rendre le SVG (carré → pas de letterbox).
        let sx = px as f32 / view.width();
        let sy = px as f32 / view.height();
        let transform = tiny_skia::Transform::from_scale(sx, sy);
        resvg::render(&tree, transform, &mut pixmap.as_mut());

        // tiny-skia est en BGRA premultipliée. ico::IconImage attend
        // RGBA non prémultipliée. On convertit pixel par pixel.
        let mut rgba = Vec::with_capacity((px * px * 4) as usize);
        for chunk in pixmap.data().chunks_exact(4) {
            // tiny-skia stocke en RGBA premultiplié (cf. doc), pas BGRA.
            let (r, g, b, a) = (chunk[0], chunk[1], chunk[2], chunk[3]);
            // Démultiplie : c_out = c_in / a (clampé alpha=0 → 0).
            let (r, g, b) = if a == 0 {
                (0, 0, 0)
            } else {
                let a_f = a as f32 / 255.0;
                (
                    (r as f32 / a_f).round().min(255.0) as u8,
                    (g as f32 / a_f).round().min(255.0) as u8,
                    (b as f32 / a_f).round().min(255.0) as u8,
                )
            };
            rgba.extend_from_slice(&[r, g, b, a]);
        }

        let image = ico::IconImage::from_rgba_data(px, px, rgba);
        icon_dir.add_entry(ico::IconDirEntry::encode(&image)?);
        println!("  + {px}×{px} encodé");
    }

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(&out_path)?;
    icon_dir.write(file)?;
    println!("✓ icone {} taille(s) écrite dans {}", sizes.len(), out_path.display());
    Ok(())
}

/// Résout la racine du workspace en remontant depuis `CARGO_MANIFEST_DIR`
/// (qui pointe sur `xtask/`). Plus robuste que de hardcoder un `..`.
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .map(PathBuf::from)
        .unwrap_or(manifest)
}
