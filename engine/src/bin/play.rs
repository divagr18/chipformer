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
//! Every frame: key changes are injected as writes, one TICK is injected, and the model
//! generates `ipf` instructions. A pixel stays on screen for `persist` instructions after it
//! goes dark (CHIP-8 games erase and redraw sprites with XOR, so sampling a single instant
//! makes them blink), followed by a short phosphor fade. If the model is slower than real time the
//! game slows down rather than skipping instructions.

use chipformer_engine::games::{TAP_FRAMES, parse_manifest};
use chipformer_engine::host::Session;
use chipformer_engine::model::Model;
use minifb::{Key, Window, WindowOptions};
use std::path::Path;
use std::time::{Duration, Instant};

const CLASSIC: [(Key, usize); 16] = [
    (Key::Key1, 0x1), (Key::Key2, 0x2), (Key::Key3, 0x3), (Key::Key4, 0xC),
    (Key::Q, 0x4), (Key::W, 0x5), (Key::E, 0x6), (Key::R, 0xD),
    (Key::A, 0x7), (Key::S, 0x8), (Key::D, 0x9), (Key::F, 0xE),
    (Key::Z, 0xA), (Key::X, 0x0), (Key::C, 0xB), (Key::V, 0xF),
];

/// `--keys pong` for a ROM given by path: classic Pong reads keypad 1/4 (left paddle) and C/D.
const PONG: [(Key, usize); 4] = [(Key::A, 0x1), (Key::D, 0x4), (Key::Left, 0xC), (Key::Right, 0xD)];

const ROLLOVER_TOKENS: usize = 250_000; // restart the context from a state dump past this
const FG: [f32; 3] = [230.0, 230.0, 210.0];
const BG: [f32; 3] = [16.0, 16.0, 24.0];
const FADE: f32 = 0.45; // brightness kept per frame by a pixel that went dark

fn opt(args: &[String], name: &str, default: &str) -> String {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or(default.into())
}

fn key_named(name: &str) -> Key {
    match name {
        "W" => Key::W, "A" => Key::A, "S" => Key::S, "D" => Key::D, "E" => Key::E, "Q" => Key::Q,
        "Up" => Key::Up, "Down" => Key::Down, "Left" => Key::Left, "Right" => Key::Right,
        "Space" => Key::Space, "Enter" => Key::Enter,
        other => panic!("no key named {other}"),
    }
}

/// `primary` bindings, plus the classic layout for every keyboard key and CHIP-8 key they leave free.
fn with_classic(primary: Vec<(Key, usize)>) -> Vec<(Key, usize)> {
    let mut m: Vec<(Key, usize)> = CLASSIC
        .iter()
        .filter(|(k, c)| !primary.iter().any(|p| p.0 == *k || p.1 == *c))
        .copied()
        .collect();
    m.extend(primary);
    m
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: play <model_dir> (<rom> | --game <id>) [--games games/manifest.json] [--ipf N] \
                   [--scale 12] [--keys pong|classic] [--quirks Q] [--persist 20] [--no-fade]");
        std::process::exit(2);
    }
    let scale: usize = opt(&args, "--scale", "12").parse().expect("--scale");
    let fade = !args.iter().any(|a| a == "--no-fade");
    let persist: u64 = opt(&args, "--persist", "20").parse().expect("--persist");

    // a manifest game brings its ROM, speed, quirks and controls; flags override them
    let (rom, mut ipf, mut quirks, keymap, title, tap) = if args.iter().any(|a| a == "--game") {
        let path = opt(&args, "--games", "games/manifest.json");
        let games = parse_manifest(&std::fs::read_to_string(&path).expect("read manifest")).unwrap();
        let id = opt(&args, "--game", "");
        let g = games.iter().find(|g| g.id == id).unwrap_or_else(|| panic!("no game {id:?} in {path}"));
        let rom = std::fs::read(Path::new(&path).parent().unwrap().join(&g.file)).expect("read rom");
        let keys = g.bindings().into_iter().map(|(k, c)| (key_named(k), c)).collect();
        eprintln!("{}: {}", g.title, g.controls);
        (rom, g.ipf, g.quirks.clone(), with_classic(keys), g.title.clone(), g.tap_chips())
    } else {
        let keymap = match opt(&args, "--keys", "pong").as_str() {
            "classic" => CLASSIC.to_vec(),
            "pong" => with_classic(PONG.to_vec()),
            other => panic!("--keys must be pong or classic, not {other}"),
        };
        (std::fs::read(&args[2]).expect("read rom"), 10, String::new(), keymap, args[2].clone(), vec![])
    };
    ipf = opt(&args, "--ipf", &ipf.to_string()).parse().expect("--ipf");
    quirks = opt(&args, "--quirks", &quirks);

    let model = Model::load(Path::new(&args[1]));
    let mut s = Session::new(&model, &rom, &quirks, false).unwrap_or_else(|e| panic!("{e}"));
    let (w, h) = (s.spec.width, s.spec.height);

    let mut window = Window::new("chipformer", w * scale, h * scale, WindowOptions::default())
        .expect("open window");
    let mut buf = vec![0u32; w * h * scale * scale];
    let mut glow = vec![0f32; w * h];
    let frame = Duration::from_micros(16_667);
    let (mut frames, mut tokens_at, mut clock) = (0u32, s.state.pos, Instant::now());
    let mut rollovers = 0;
    let (mut was_down, mut pulse) = ([false; 16], [0u32; 16]);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let start = Instant::now();
        let mut down = [false; 16];
        for &(k, chip) in &keymap {
            down[chip] |= window.is_key_down(k);
        }
        // tap keys: one short pulse per press, however long the key is held
        for &chip in &tap {
            if down[chip] && !was_down[chip] {
                pulse[chip] = TAP_FRAMES;
            }
            was_down[chip] = down[chip];
            down[chip] = pulse[chip] > 0;
            pulse[chip] = pulse[chip].saturating_sub(1);
        }
        for (chip, &d) in down.iter().enumerate() {
            s.set_key(chip, d);
        }
        s.tick().unwrap_or_else(|e| panic!("{e}"));
        for _ in 0..ipf {
            s.step().unwrap_or_else(|e| panic!("{e}"));
        }
        if s.state.pos > ROLLOVER_TOKENS {
            s.rollover();
            rollovers += 1;
            tokens_at = s.state.pos;
        }

        for y in 0..h {
            for x in 0..w {
                let g = &mut glow[y * w + x];
                *g = if s.visible(x, y, persist) { 1.0 } else if fade { *g * FADE } else { 0.0 };
                let c = |i: usize| (BG[i] + (FG[i] - BG[i]) * *g) as u32;
                let rgb = (c(0) << 16) | (c(1) << 8) | c(2);
                for dy in 0..scale {
                    let row = (y * scale + dy) * w * scale + x * scale;
                    buf[row..row + scale].fill(rgb);
                }
            }
        }
        window.update_with_buffer(&buf, w * scale, h * scale).expect("draw");

        frames += 1;
        if clock.elapsed() >= Duration::from_secs(1) {
            let secs = clock.elapsed().as_secs_f64();
            let tps = (s.state.pos.saturating_sub(tokens_at)) as f64 / secs;
            window.set_title(&format!(
                "chipformer: {title} - {:.0} fps (60 = real time), {:.0} tok/s, context {} tokens, {} rollovers",
                frames as f64 / secs, tps, s.state.pos, rollovers
            ));
            (frames, tokens_at, clock) = (0, s.state.pos, Instant::now());
        }
        if let Some(rest) = frame.checked_sub(start.elapsed()) {
            std::thread::sleep(rest);
        }
    }
}
