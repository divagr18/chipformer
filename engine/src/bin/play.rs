//! Play a CHIP-8 ROM on the hand-built transformer.
//!
//!     play <model_dir> --game <id> [--games games/manifest.json] [options]
//!     play <model_dir> <rom> [--keys pong|classic] [--quirks Q] [options]
//!     options: [--ipf N] [--scale 12] [--persist 20] [--no-fade]
//!
//! A manifest game brings its ROM, speed, quirk profile and controls: one-player games take
//! the arrow keys or WASD plus Space/E; two-player games put player 1 on A/D (or W/S) and
//! player 2 on Left/Right (or Up/Down). Keys a game doesn't use keep the classic layout
//! (1 2 3 C / 4 5 6 D / 7 8 9 E / A 0 B F on 1234 / QWER / ASDF / ZXCV).
//! For a ROM path, `--keys pong` (default) maps classic Pong's paddles to A/D and Left/Right,
//! and `--keys classic` uses the classic layout only. Esc quits.
//!
//! The `chipformer` binary is the same player with the model and games built in.

use chipformer_engine::desktop::{self, Options};
use chipformer_engine::games::parse_manifest;
use chipformer_engine::model::Model;
use std::path::Path;

fn opt(args: &[String], name: &str, default: &str) -> String {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or(default.into())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: play <model_dir> (<rom> | --game <id>) [--games games/manifest.json] [--ipf N] \
                   [--scale 12] [--keys pong|classic] [--quirks Q] [--persist 20] [--no-fade]");
        std::process::exit(2);
    }

    // a manifest game brings its ROM, speed, quirks and controls; flags override them
    let games;
    let (rom, mut o) = if args.iter().any(|a| a == "--game") {
        let path = opt(&args, "--games", "games/manifest.json");
        games = parse_manifest(&std::fs::read_to_string(&path).expect("read manifest")).unwrap();
        let id = opt(&args, "--game", "");
        let g = games.iter().find(|g| g.id == id).unwrap_or_else(|| panic!("no game {id:?} in {path}"));
        let rom = std::fs::read(Path::new(&path).parent().unwrap().join(&g.file)).expect("read rom");
        eprintln!("{}: {}", g.title, g.controls);
        (rom, Options::for_game(g))
    } else {
        let pong = match opt(&args, "--keys", "pong").as_str() {
            "classic" => false,
            "pong" => true,
            other => panic!("--keys must be pong or classic, not {other}"),
        };
        (std::fs::read(&args[2]).expect("read rom"), Options::for_rom(&args[2], pong))
    };
    o.ipf = opt(&args, "--ipf", &o.ipf.to_string()).parse().expect("--ipf");
    o.quirks = opt(&args, "--quirks", &o.quirks);
    o.scale = opt(&args, "--scale", "12").parse().expect("--scale");
    o.persist = opt(&args, "--persist", "20").parse().expect("--persist");
    o.fade = !args.iter().any(|a| a == "--no-fade");

    let model = Model::load(Path::new(&args[1]));
    desktop::run(&model, &rom, &o).unwrap_or_else(|e| panic!("{e}"));
}
