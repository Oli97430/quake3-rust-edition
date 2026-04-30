//! Build script — embarque l'icone Windows et les métadonnées de version
//! dans l'exécutable. No-op sur Linux / macOS.
//!
//! L'icone source est `<workspace>/assets/icon.ico`. Si absent, on
//! n'embarque rien et le build continue (l'exe aura l'icone par défaut
//! Windows). Évite de bloquer un build dev sur l'absence d'un asset
//! optionnel.

#[cfg(windows)]
fn main() {
    // Le build.rs tourne dans le dossier du crate (`crates/q3-engine`),
    // l'asset est 2 niveaux au-dessus, à la racine du workspace.
    let icon_path = "../../assets/icon.ico";

    println!("cargo:rerun-if-changed={}", icon_path);
    println!("cargo:rerun-if-changed=build.rs");

    if !std::path::Path::new(icon_path).is_file() {
        println!(
            "cargo:warning=icon `{}` absent — build sans icone embarquée. \
             Génère-le avec `inkscape + magick` (cf. assets/branding/README.md).",
            icon_path
        );
        return;
    }

    let mut res = winres::WindowsResource::new();
    res.set_icon(icon_path);
    res.set("ProductName", "Quake 3 RUST EDITION");
    res.set("FileDescription", "Quake 3 RUST EDITION — id Tech 3 reimagined in Rust");
    res.set("LegalCopyright", "GPL-2.0-or-later");
    res.set("CompanyName", "Quake 3 RUST EDITION contributors");
    if let Err(e) = res.compile() {
        // Pas d'unwrap — un Windows SDK pas tout à fait standard ou un
        // rc.exe en panne ne doit pas tuer le build, juste l'avertir.
        println!("cargo:warning=winres compile failed: {e}");
    }
}

#[cfg(not(windows))]
fn main() {
    // No-op hors Windows — Linux / macOS ne stockent pas l'icone dans
    // le binaire. L'icone OS provient du .desktop ou du .app bundle.
}
