//! Host loop: boots the model with a ROM and talks to it in the trace format
//! (docs/trace_format.md). Mirrors chipformer/session.py.
//!
//! The host feeds the boot prompt, injects TICK tokens and key writes right
//! after a STEP, and keeps a mirror of the cells built from the model's own
//! writes (for the screen). It never computes machine semantics.

use crate::model::{Model, State};
use std::collections::VecDeque;
use serde_json::Value;

const MAX_RECORD: usize = 2 * 40 + 2;

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

#[derive(Clone)]
pub struct Spec {
    pub val_base: u32,
    pub addr_base: u32,
    pub step: u32,
    pub tick: u32,
    pub boot: u32,
    pub num_cells: usize,
    pub null: usize,
    pub cls: usize,
    pub pc: usize,
    pub rng: usize,
    pub keys: usize,
    pub display: usize,
    pub width: usize,
    pub height: usize,
    pub program_start: usize,
    pub font_base: usize,
    pub rng_seed: u32,
    pub font: Vec<u32>,
    pub quirks: usize,
    pub quirk_names: Vec<String>,
    pub profiles: Vec<(String, Vec<String>)>,
}

impl Spec {
    pub fn from_config(config: &Value) -> Spec {
        let m = &config["machine"];
        let u = |k: &str| m[k].as_u64().unwrap_or_else(|| panic!("config.machine.{k}"));
        Spec {
            val_base: u("VAL_BASE") as u32,
            addr_base: u("ADDR_BASE") as u32,
            step: u("STEP") as u32,
            tick: u("TICK") as u32,
            boot: u("BOOT") as u32,
            num_cells: u("NUM_CELLS") as usize,
            null: u("NULL") as usize,
            cls: u("CLS") as usize,
            pc: u("PC") as usize,
            rng: u("RNG") as usize,
            keys: u("KEYS") as usize,
            display: u("DISPLAY") as usize,
            width: u("DISPLAY_W") as usize,
            height: u("DISPLAY_H") as usize,
            program_start: u("PROGRAM_START") as usize,
            font_base: u("FONT_BASE") as usize,
            rng_seed: u("RNG_SEED") as u32,
            font: m["FONT"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect(),
            quirks: u("QUIRKS") as usize,
            quirk_names: strings(&m["QUIRK_NAMES"]),
            profiles: m["PROFILES"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), strings(v)))
                .collect(),
        }
    }

    /// Quirk cell values for a profile name ("chip48") or comma-separated flags ("shift,clip").
    pub fn quirk_cells(&self, quirks: &str) -> Result<Vec<(usize, u32)>, String> {
        let on: Vec<String> = match self.profiles.iter().find(|(name, _)| name == quirks) {
            Some((_, flags)) => flags.clone(),
            None => quirks.split(',').filter(|q| !q.is_empty()).map(str::to_string).collect(),
        };
        if let Some(bad) = on.iter().find(|q| !self.quirk_names.contains(q)) {
            return Err(format!("unknown quirk {bad:?}; known: {:?} or profiles {:?}", self.quirk_names,
                self.profiles.iter().map(|p| &p.0).collect::<Vec<_>>()));
        }
        Ok(self.quirk_names.iter().enumerate().map(|(i, q)| (self.quirks + i, on.contains(q) as u32)).collect())
    }

    pub fn is_addr(&self, t: u32) -> bool {
        t >= self.addr_base && t < self.step
    }

    pub fn is_val(&self, t: u32) -> bool {
        t >= self.val_base && t < self.addr_base
    }

    /// Initial cells for a ROM: font, program, PC and RNG seed.
    pub fn boot_cells(&self, rom: &[u8], quirks: &[(usize, u32)]) -> Vec<u32> {
        let mut cells = vec![0u32; self.num_cells];
        for (i, &b) in self.font.iter().enumerate() {
            cells[self.font_base + i] = b;
        }
        for (i, &b) in rom.iter().enumerate() {
            cells[self.program_start + i] = b as u32;
        }
        cells[self.pc] = self.program_start as u32;
        cells[self.rng] = self.rng_seed;
        for &(a, v) in quirks {
            cells[a] = v;
        }
        cells
    }

    /// BOOT, CLS write, every other nonzero cell in address order, STEP.
    pub fn boot_tokens(&self, cells: &[u32]) -> Vec<u32> {
        let mut out = vec![self.boot, self.addr_base + self.cls as u32, self.val_base + cells[self.cls]];
        for (a, &v) in cells.iter().enumerate() {
            if a != self.cls && a != self.null && v != 0 {
                out.push(self.addr_base + a as u32);
                out.push(self.val_base + v);
            }
        }
        out.push(self.step);
        out
    }

    pub fn apply(&self, cells: &mut [u32], addr: usize, val: u32) {
        if addr == self.cls {
            let n = self.width * self.height / 8;
            cells[self.display..self.display + n].iter_mut().for_each(|c| *c = 0);
        }
        cells[addr] = val;
    }
}

pub struct Session<'m> {
    pub spec: Spec,
    pub state: State<'m>,
    pub cells: Vec<u32>,
    pub tokens: Vec<u32>,
    pub keep_tokens: bool,
    pred: u32,
    /// instructions executed so far, and when each pixel last went dark (in instructions):
    /// CHIP-8 games erase and redraw sprites with XOR, so a screen sampled at one instant
    /// shows half-drawn frames; keeping a pixel visible for a few instructions after it goes
    /// dark hides that (see visible())
    pub instructions: u64,
    dark_since: Vec<u64>,
    lit_since: Vec<u64>,
    /// how long each pixel had been lit when it last went dark
    lit_for: Vec<u64>,
    /// tokens fed since the session started (across context rollovers)
    pub fed: u64,
    pub rollovers: u32,
    /// the most recent tokens (for showing the token stream)
    pub recent: VecDeque<u32>,
}

const RECENT_TOKENS: usize = 2048;

impl<'m> Session<'m> {
    /// `quirks`: a profile name ("octo", "chip48", "cosmac") or comma-separated quirk flags.
    pub fn new(model: &'m Model, rom: &[u8], quirks: &str, keep_tokens: bool) -> Result<Self, String> {
        let spec = Spec::from_config(&model.config);
        let cells = spec.boot_cells(rom, &spec.quirk_cells(quirks)?);
        let dark_since = vec![0; spec.width * spec.height];
        let mut s = Session {
            spec, state: State::new(model), cells, tokens: vec![], keep_tokens, pred: 0,
            instructions: 0, lit_since: dark_since.clone(), lit_for: dark_since.clone(), dark_since, fed: 0, rollovers: 0,
            recent: VecDeque::with_capacity(RECENT_TOKENS),
        };
        let boot = s.spec.boot_tokens(&s.cells);
        s.inject(&boot);
        Ok(s)
    }

    fn push(&mut self, t: u32) {
        if self.keep_tokens {
            self.tokens.push(t);
        }
        if self.recent.len() == RECENT_TOKENS {
            self.recent.pop_front();
        }
        self.recent.push_back(t);
        self.fed += 1;
        self.pred = self.state.feed(t);
    }

    fn inject(&mut self, toks: &[u32]) {
        for &t in toks {
            self.push(t);
        }
    }

    /// Let the model write until it emits STEP; returns the number of writes.
    fn generate_record(&mut self) -> Result<usize, String> {
        let mut writes = vec![];
        for _ in 0..MAX_RECORD {
            let tok = self.pred;
            if tok == self.spec.step {
                self.push(tok);
                for &(a, v) in &writes {
                    self.note_dark_pixels(a, v);
                    self.spec.apply(&mut self.cells, a, v);
                }
                return Ok(writes.len());
            }
            if !self.spec.is_addr(tok) {
                return Err(format!("model emitted token {tok} where ADDR/STEP was expected at position {}", self.state.pos));
            }
            self.push(tok);
            let val = self.pred;
            if !self.spec.is_val(val) {
                return Err(format!("model emitted token {val} after an ADDR at position {}", self.state.pos));
            }
            self.push(val);
            writes.push(((tok - self.spec.addr_base) as usize, val - self.spec.val_base));
        }
        Err("record never ended".into())
    }

    pub fn step(&mut self) -> Result<usize, String> {
        self.instructions += 1;
        self.generate_record()
    }

    /// Record when each pixel a pending write (or CLS) turns on or off.
    fn note_dark_pixels(&mut self, addr: usize, val: u32) {
        let bytes = self.spec.width * self.spec.height / 8;
        let d = self.spec.display;
        let changes: Vec<(usize, u32, u32)> = if addr == self.spec.cls {
            (0..bytes).map(|i| (i, self.cells[d + i], 0)).collect()
        } else if (d..d + bytes).contains(&addr) {
            vec![(addr - d, self.cells[addr] & !val, val & !self.cells[addr])]
        } else {
            return;
        };
        for (byte, gone, came) in changes {
            for bit in 0..8 {
                let i = byte * 8 + bit;
                if gone >> (7 - bit) & 1 == 1 {
                    self.dark_since[i] = self.instructions;
                    self.lit_for[i] = self.instructions - self.lit_since[i];
                }
                if came >> (7 - bit) & 1 == 1 {
                    self.lit_since[i] = self.instructions;
                }
            }
        }
    }

    pub fn tick(&mut self) -> Result<usize, String> {
        let t = self.spec.tick;
        self.inject(&[t]);
        self.generate_record()
    }

    pub fn set_key(&mut self, key: usize, down: bool) {
        let a = self.spec.keys + key;
        if (self.cells[a] != 0) == down {
            return;
        }
        let toks = [self.spec.addr_base + a as u32, self.spec.val_base + down as u32, self.spec.step];
        self.inject(&toks);
        self.cells[a] = down as u32;
    }

    /// Restart the context from a state dump of the mirror (keeps positions bounded).
    pub fn rollover(&mut self) {
        self.state.reset();
        let boot = self.spec.boot_tokens(&self.cells);
        self.inject(&boot);
        self.rollovers += 1;
    }

    /// One 60 Hz frame: inject key changes (bit k of `keys` = CHIP-8 key k held), one TICK,
    /// then `ipf` instructions; roll the context over once it passes `rollover_at` tokens.
    pub fn frame(&mut self, keys: u16, ipf: usize, rollover_at: usize) -> Result<(), String> {
        for k in 0..16 {
            self.set_key(k, keys >> k & 1 == 1);
        }
        self.tick()?;
        for _ in 0..ipf {
            self.step()?;
        }
        if self.state.pos > rollover_at {
            self.rollover();
        }
        Ok(())
    }

    pub fn pixel(&self, x: usize, y: usize) -> bool {
        let byte = self.cells[self.spec.display + y * (self.spec.width / 8) + x / 8];
        (byte >> (7 - x % 8)) & 1 == 1
    }

    /// Lit now, or a stable pixel that went dark less than `persist` instructions ago.
    ///
    /// Games erase and redraw sprites with XOR: Pong erases a paddle and redraws it ~17
    /// instructions later every loop, so persist >= 16 removes that blinking (measured: 47% of
    /// frames show a missing paddle with persist 0, none with 16). Only pixels that had been
    /// lit for at least STABLE instructions are held: games also draw short-lived probe
    /// sprites (Tetris tests collisions and full lines when a piece lands) and holding those
    /// would make them flash (measured on Tetris: 170 flashing frames in 30 s with no stability
    /// rule, 0 with STABLE = 8, and no change to paddle visibility).
    pub fn visible(&self, x: usize, y: usize, persist: u64) -> bool {
        const STABLE: u64 = 8;
        let i = y * self.spec.width + x;
        self.pixel(x, y) || (self.instructions - self.dark_since[i] < persist && self.lit_for[i] >= STABLE)
    }
}
