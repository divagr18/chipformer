//! The desktop window shared by the `play` and `chipformer` binaries.
//!
//! Every frame: key changes are injected as writes, one TICK is injected, and the model
//! generates `ipf` instructions. A pixel stays on screen for `persist` instructions after it
//! goes dark (CHIP-8 games erase and redraw sprites with XOR, so sampling a single instant
//! makes them blink), followed by a short phosphor fade. If the model is slower than real time the
//! game slows down rather than skipping instructions.

use crate::games::{Game, GameOver, TAP_FRAMES};
use crate::host::Session;
use crate::model::Model;
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use std::time::{Duration, Instant};

/// The classic keypad: 1 2 3 C / 4 5 6 D / 7 8 9 E / A 0 B F on 1234 / QWER / ASDF / ZXCV.
pub const CLASSIC: [(Key, usize); 16] = [
    (Key::Key1, 0x1), (Key::Key2, 0x2), (Key::Key3, 0x3), (Key::Key4, 0xC),
    (Key::Q, 0x4), (Key::W, 0x5), (Key::E, 0x6), (Key::R, 0xD),
    (Key::A, 0x7), (Key::S, 0x8), (Key::D, 0x9), (Key::F, 0xE),
    (Key::Z, 0xA), (Key::X, 0x0), (Key::C, 0xB), (Key::V, 0xF),
];

/// Classic Pong reads keypad 1/4 (left paddle) and C/D: A/D and Left/Right.
pub const PONG: [(Key, usize); 4] = [(Key::A, 0x1), (Key::D, 0x4), (Key::Left, 0xC), (Key::Right, 0xD)];

pub const ROLLOVER_TOKENS: usize = 250_000; // restart the context from a state dump past this
const FG: [f32; 3] = [221.0, 245.0, 158.0]; // the site's phosphor
const BG: [f32; 3] = [11.0, 16.0, 13.0];
const FADE: f32 = 0.45; // brightness kept per frame by a pixel that went dark

pub fn key_named(name: &str) -> Key {
    match name {
        "W" => Key::W, "A" => Key::A, "S" => Key::S, "D" => Key::D, "E" => Key::E, "Q" => Key::Q,
        "Up" => Key::Up, "Down" => Key::Down, "Left" => Key::Left, "Right" => Key::Right,
        "Space" => Key::Space, "Enter" => Key::Enter,
        other => panic!("no key named {other}"),
    }
}

/// `primary` bindings, plus the classic layout for every keyboard key and CHIP-8 key they leave free.
pub fn with_classic(primary: Vec<(Key, usize)>) -> Vec<(Key, usize)> {
    let mut m: Vec<(Key, usize)> = CLASSIC
        .iter()
        .filter(|(k, c)| !primary.iter().any(|p| p.0 == *k || p.1 == *c))
        .copied()
        .collect();
    m.extend(primary);
    m
}

pub struct Options<'g> {
    pub title: String,
    pub ipf: usize,
    pub quirks: String,
    pub keymap: Vec<(Key, usize)>,
    /// CHIP-8 keys delivered as one short pulse per press
    pub tap: Vec<usize>,
    pub game_over: Option<&'g GameOver>,
    pub scale: usize,
    pub persist: u64,
    pub fade: bool,
}

impl<'g> Options<'g> {
    /// A manifest game's speed, quirks, controls and rules.
    pub fn for_game(g: &'g Game) -> Self {
        let keys = g.bindings().into_iter().map(|(k, c)| (key_named(k), c)).collect();
        Options {
            title: g.title.clone(),
            ipf: g.ipf,
            quirks: g.quirks.clone(),
            keymap: with_classic(keys),
            tap: g.tap_chips(),
            game_over: g.game_over.as_ref(),
            scale: 12,
            persist: 20,
            fade: true,
        }
    }

    /// A bare ROM: classic keypad (or Pong's paddles), 10 instructions per frame.
    pub fn for_rom(title: &str, pong_keys: bool) -> Self {
        Options {
            title: title.to_string(),
            ipf: 10,
            quirks: String::new(),
            keymap: if pong_keys { with_classic(PONG.to_vec()) } else { CLASSIC.to_vec() },
            tap: vec![],
            game_over: None,
            scale: 12,
            persist: 20,
            fade: true,
        }
    }
}

/// Open a window and play `rom` on the transformer until the window closes or Esc.
pub fn run(model: &Model, rom: &[u8], o: &Options) -> Result<(), String> {
    let mut s = Session::new(model, rom, &o.quirks, false)?;
    let (w, h, scale) = (s.spec.width, s.spec.height, o.scale);
    let base_title = format!("chipformer - {} (running on a transformer)", o.title);
    let mut window = Window::new(&base_title, w * scale, h * scale, WindowOptions::default())
        .map_err(|e| format!("open window: {e}"))?;
    let mut buf = vec![0u32; w * h * scale * scale];
    let mut glow = vec![0f32; w * h];
    let frame = Duration::from_micros(16_667);
    let (mut frames, mut tokens_at, mut clock) = (0u32, s.state.pos, Instant::now());
    let (mut was_down, mut pulse) = ([false; 16], [0u32; 16]);
    let mut full_since: Option<Instant> = None;
    let mut over = false;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let start = Instant::now();
        if over {
            if window.is_key_pressed(Key::Enter, KeyRepeat::No) {
                s = Session::new(model, rom, &o.quirks, false)?;
                (over, full_since, tokens_at) = (false, None, s.state.pos);
                glow.fill(0.0);
            }
        } else {
            let mut down = [false; 16];
            for &(k, chip) in &o.keymap {
                down[chip] |= window.is_key_down(k);
            }
            // tap keys: one short pulse per press, however long the key is held
            for &chip in &o.tap {
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
            s.tick()?;
            for _ in 0..o.ipf {
                s.step()?;
            }
            if s.state.pos > ROLLOVER_TOKENS {
                s.rollover();
                tokens_at = s.state.pos;
            }
            if let Some(rule) = o.game_over {
                if !rule.region_lit(|x, y| s.pixel(x, y)) {
                    full_since = None;
                } else if full_since.get_or_insert(start).elapsed().as_secs_f64() > rule.seconds {
                    over = true;
                    window.set_title(&format!(
                        "chipformer - {}: GAME OVER ({}). Enter plays again, Esc quits",
                        o.title, rule.message
                    ));
                }
            }
        }

        for y in 0..h {
            for x in 0..w {
                let g = &mut glow[y * w + x];
                *g = if s.visible(x, y, o.persist) { 1.0 } else if o.fade { *g * FADE } else { 0.0 };
                let dim = if over { 0.35 } else { 1.0 };
                let c = |i: usize| ((BG[i] + (FG[i] - BG[i]) * *g) * dim) as u32;
                let rgb = (c(0) << 16) | (c(1) << 8) | c(2);
                for dy in 0..scale {
                    let row = (y * scale + dy) * w * scale + x * scale;
                    buf[row..row + scale].fill(rgb);
                }
            }
        }
        window.update_with_buffer(&buf, w * scale, h * scale).map_err(|e| format!("draw: {e}"))?;

        frames += 1;
        if !over && clock.elapsed() >= Duration::from_secs(1) {
            let secs = clock.elapsed().as_secs_f64();
            let tps = (s.state.pos.saturating_sub(tokens_at)) as f64 / secs;
            window.set_title(&format!(
                "{base_title} - {:.0} fps (60 = real time), {:.0} tok/s, context {} tokens",
                frames as f64 / secs, tps, s.state.pos
            ));
            (frames, tokens_at, clock) = (0, s.state.pos, Instant::now());
        }
        if let Some(rest) = frame.checked_sub(start.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    Ok(())
}
