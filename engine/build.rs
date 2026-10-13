//! With the `embed` feature, bake the model, the game manifest, the ROMs and their credits into
//! the `chipformer` binary so it runs with no other files.

use std::fmt::Write;
use std::path::Path;

fn main() {
    if std::env::var_os("CARGO_FEATURE_EMBED").is_none() {
        return;
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let model = root.join("model").join("model.chipformer");
    let games = root.join("games");
    for p in [&model, &games.join("manifest.json"), &games.join("ATTRIBUTION.md")] {
        println!("cargo:rerun-if-changed={}", p.display());
        if !p.exists() {
            panic!(
                "the embed feature needs {}; run `uv run python -m chipformer.compile --out model` \
                 in the repository root first",
                p.display()
            );
        }
    }
    let roms_dir = games.join("roms");
    println!("cargo:rerun-if-changed={}", roms_dir.display());
    let mut roms: Vec<_> = std::fs::read_dir(&roms_dir)
        .expect("read games/roms")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "ch8"))
        .collect();
    roms.sort();

    let abs = |p: &Path| p.display().to_string(); // absolute: CARGO_MANIFEST_DIR is
    let mut src = String::new();
    writeln!(src, "pub static MODEL: &[u8] = include_bytes!(r#\"{}\"#);", abs(&model)).unwrap();
    writeln!(src, "pub static MANIFEST: &str = include_str!(r#\"{}\"#);", abs(&games.join("manifest.json"))).unwrap();
    writeln!(src, "pub static ATTRIBUTION: &str = include_str!(r#\"{}\"#);", abs(&games.join("ATTRIBUTION.md")))
        .unwrap();
    writeln!(src, "pub static ROMS: &[(&str, &[u8])] = &[").unwrap();
    for p in &roms {
        let name = p.file_name().unwrap().to_string_lossy();
        writeln!(src, "    (\"roms/{name}\", include_bytes!(r#\"{}\"#)),", abs(p)).unwrap();
    }
    writeln!(src, "];").unwrap();
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("embedded.rs");
    std::fs::write(out, src).unwrap();
}
