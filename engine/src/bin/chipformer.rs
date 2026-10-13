//! chipformer: CHIP-8 games running on a transformer whose weights were set by hand.
//!
//! A single file: the model, the games and their credits are built into the executable
//! (the `embed` feature; see build.rs).
//!
//!     chipformer                       banner, featured games, usage
//!     chipformer list [--all]          featured (or all verified) games
//!     chipformer play <game>           play a built-in game
//!     chipformer play <rom.ch8>        play your own ROM [--quirks Q] [--keys pong|classic]
//!     chipformer credits               game authors, sources and licenses
//!
//! `play` options: --ipf N, --scale N, --persist N, --no-fade.
//! Hidden: `play <game> --headless [--frames N]` runs without a window and prints throughput
//! and the final screen (used as a smoke test in CI).

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/embedded.rs"));
}

use chipformer_engine::desktop::{self, Options, ROLLOVER_TOKENS};
use chipformer_engine::games::{Game, parse_manifest};
use chipformer_engine::host::Session;
use chipformer_engine::model::Model;
use std::io::{IsTerminal, Write};
use std::process::exit;
use std::time::Instant;

const VERSION: &str = env!("CARGO_PKG_VERSION");

const USAGE: &str = "\
usage:
  chipformer list [--all]          the featured games (--all: every verified game)
  chipformer play <game>           play one, e.g. `chipformer play tetris`
  chipformer play <rom.ch8>        play your own ROM [--quirks octo|chip48|cosmac|flags] [--keys pong|classic]
  chipformer credits               game authors, sources and licenses

play options: --ipf N (instructions per frame), --scale N (window scale, default 12),
              --persist N (anti-flicker, default 20), --no-fade
in game: the controls are printed when it starts; Esc quits";

fn opt(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn fail(msg: &str) -> ! {
    eprintln!("chipformer: {msg}");
    exit(2);
}

fn games() -> Vec<Game> {
    let mut g: Vec<Game> = parse_manifest(embedded::MANIFEST)
        .unwrap_or_else(|e| fail(&e))
        .into_iter()
        .filter(|g| g.status == "ok")
        .collect();
    g.sort_by_key(|g| (g.featured.unwrap_or(usize::MAX), g.title.to_lowercase()));
    g
}

fn rom_of(g: &Game) -> &'static [u8] {
    embedded::ROMS
        .iter()
        .find(|(name, _)| *name == g.file)
        .map(|(_, bytes)| *bytes)
        .unwrap_or_else(|| fail(&format!("{} is not built in", g.file)))
}

fn load_model() -> Model {
    Model::from_sparse_bytes(embedded::MODEL).unwrap_or_else(|e| fail(&e))
}

fn banner(m: Option<&Model>) {
    println!("chipformer {VERSION}: CHIP-8 games running on a transformer.");
    println!("Every pixel is the model predicting its next token. The weights were set by hand;");
    println!("it never saw a training example.");
    if let Some(m) = m {
        println!("{} layers, {} attention heads, {} nonzero weights.", m.n_layers(), m.n_heads(), m.nnz());
    }
}

fn list(all: bool) {
    let games = games();
    let shown: Vec<&Game> = games.iter().filter(|g| all || g.featured.is_some()).collect();
    let w = shown.iter().map(|g| g.id.len()).max().unwrap_or(0);
    for g in &shown {
        let by = if g.authors.is_empty() { String::new() } else { format!(" by {}", g.authors.join(", ")) };
        println!("  {:w$}  {}{by}", g.id, g.title);
        if !g.controls.is_empty() {
            println!("  {:w$}  {}", "", g.controls);
        }
    }
    if !all {
        println!("\n{} more verified games: chipformer list --all", games.len() - shown.len());
    }
}

/// Run `frames` frames without a window and print throughput plus the final screen.
fn headless(model: &Model, rom: &[u8], o: &Options, frames: usize) {
    let mut s = Session::new(model, rom, &o.quirks, false).unwrap_or_else(|e| fail(&e));
    let t = Instant::now();
    for _ in 0..frames {
        s.frame(0, o.ipf, ROLLOVER_TOKENS).unwrap_or_else(|e| fail(&e));
    }
    let secs = t.elapsed().as_secs_f64();
    for y in 0..s.spec.height {
        let row: String = (0..s.spec.width).map(|x| if s.visible(x, y, o.persist) { '#' } else { '.' }).collect();
        println!("{row}");
    }
    println!(
        "{}: {frames} frames, {} instructions, {} tokens in {secs:.2}s = {:.0} tok/s, {:.1}x real time",
        o.title,
        s.instructions,
        s.fed,
        s.fed as f64 / secs,
        frames as f64 / 60.0 / secs
    );
}

fn play(args: &[String]) {
    let target = args.get(2).filter(|a| !a.starts_with("--")).unwrap_or_else(|| fail(USAGE));
    let games = games();
    let (rom, mut o): (Vec<u8>, Options) = if let Some(g) = games.iter().find(|g| &g.id == target) {
        println!("{}: {}", g.title, g.controls);
        (rom_of(g).to_vec(), Options::for_game(g))
    } else if std::path::Path::new(target).is_file() {
        let pong = match opt(args, "--keys").as_deref() {
            None | Some("classic") => false,
            Some("pong") => true,
            Some(other) => fail(&format!("--keys must be pong or classic, not {other}")),
        };
        let rom = std::fs::read(target).unwrap_or_else(|e| fail(&format!("{target}: {e}")));
        println!("{target}: classic CHIP-8 keypad on 1234 / QWER / ASDF / ZXCV");
        (rom, Options::for_rom(target, pong))
    } else {
        let close: Vec<&str> =
            games.iter().filter(|g| g.id.contains(target.as_str())).map(|g| g.id.as_str()).collect();
        let hint = if close.is_empty() { String::new() } else { format!(" (did you mean: {})", close.join(", ")) };
        fail(&format!("no game or file {target:?}{hint}; see `chipformer list`"));
    };
    let num = |name: &str, v: &mut usize| {
        if let Some(x) = opt(args, name) {
            *v = x.parse().unwrap_or_else(|_| fail(&format!("{name} needs a number")));
        }
    };
    num("--ipf", &mut o.ipf);
    num("--scale", &mut o.scale);
    let mut persist = o.persist as usize;
    num("--persist", &mut persist);
    o.persist = persist as u64;
    if let Some(q) = opt(args, "--quirks") {
        o.quirks = q;
    }
    o.fade = !args.iter().any(|a| a == "--no-fade");

    let model = load_model();
    if args.iter().any(|a| a == "--headless") {
        let mut frames = 600;
        num("--frames", &mut frames);
        headless(&model, &rom, &o, frames);
    } else {
        desktop::run(&model, &rom, &o).unwrap_or_else(|e| fail(&e));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        None => {
            banner(Some(&load_model()));
            println!("\nFeatured games:");
            list(false);
            println!("\n{USAGE}");
            // started by double-clicking: ask instead of closing the console straight away
            if std::io::stdin().is_terminal() {
                print!("\nGame to play (Enter for tetris, q to quit): ");
                std::io::stdout().flush().ok();
                let mut line = String::new();
                std::io::stdin().read_line(&mut line).ok();
                let id = match line.trim() {
                    "" => "tetris",
                    "q" | "quit" => return,
                    id => id,
                };
                play(&["chipformer".into(), "play".into(), id.into()]);
            }
        }
        Some("list") => list(args.iter().any(|a| a == "--all")),
        Some("play") => play(&args),
        Some("credits") => print!("{}", embedded::ATTRIBUTION),
        Some("--version" | "-V") => println!("chipformer {VERSION}"),
        Some("--help" | "-h" | "help") => {
            banner(None);
            println!("\n{USAGE}");
        }
        Some(other) => fail(&format!("unknown command {other:?}\n\n{USAGE}")),
    }
}
