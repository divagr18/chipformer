//! engine predict <model_dir> <tokens.u32> <out.u32>
//!     Teacher-forced: feed every token, write the predicted next token per position.
//! engine run <model_dir> <rom> [--instr N] [--ipf K] [--seed S] [--quirks Q] [--rollover T] [--dump trace.u32]
//!     Free-running CHIP-8 with a timer tick every K instructions and random key presses;
//!     reports throughput and optionally dumps the token trace for verification.

use chipformer_engine::host::Session;
use chipformer_engine::model::{Model, State};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn read_u32s(p: &Path) -> Vec<u32> {
    std::fs::read(p)
        .expect("read tokens")
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn write_u32s(p: &Path, v: &[u32]) {
    std::fs::write(p, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).expect("write");
}

fn load(dir: &str) -> Model {
    let t = Instant::now();
    let m = Model::load(Path::new(dir));
    eprintln!(
        "loaded {dir}: {} layers, d_model {}, {} heads ({} key caches, {} value stores), {} nonzero weights ({:.2}s)",
        m.n_layers(),
        m.d,
        m.n_heads(),
        m.cache_counts().0,
        m.cache_counts().1,
        m.nnz(),
        t.elapsed().as_secs_f64()
    );
    m
}

fn opt<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|v| v.parse().ok().unwrap_or_else(|| panic!("bad value for {name}")))
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("predict") if args.len() >= 5 => {
            let model = load(&args[2]);
            let tokens = read_u32s(Path::new(&args[3]));
            let mut st = State::new(&model);
            let t = Instant::now();
            let preds: Vec<u32> = tokens.iter().map(|&tok| st.feed(tok)).collect();
            let dt = t.elapsed().as_secs_f64();
            eprintln!("{} positions in {dt:.2}s ({:.0} tok/s)", tokens.len(), tokens.len() as f64 / dt);
            write_u32s(Path::new(&args[4]), &preds);
        }
        Some("run") if args.len() >= 4 => {
            let model = load(&args[2]);
            let rom = std::fs::read(&args[3]).expect("read rom");
            let instr: usize = opt(&args, "--instr", 10_000);
            let ipf: usize = opt(&args, "--ipf", 10);
            let rollover: usize = opt(&args, "--rollover", 250_000);
            let mut seed: u64 = opt(&args, "--seed", 1);
            let dump: Option<PathBuf> = args.iter().position(|a| a == "--dump").map(|i| PathBuf::from(&args[i + 1]));
            let t = Instant::now();
            let quirks: String = opt(&args, "--quirks", String::new());
            let mut s = Session::new(&model, &rom, &quirks, dump.is_some()).unwrap_or_else(|e| panic!("{e}"));
            let boot_pos = s.state.pos;
            eprintln!("boot prompt: {boot_pos} tokens in {:.2}s", t.elapsed().as_secs_f64());
            let t = Instant::now();
            let (mut generated, mut rollovers) = (0usize, 0usize);
            for i in 0..instr {
                if i % ipf == 0 {
                    s.tick().unwrap_or_else(|e| panic!("{e}"));
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    if seed % 10 < 3 {
                        s.set_key(((seed >> 8) % 16) as usize, (seed >> 16) % 2 == 0);
                    }
                }
                s.step().unwrap_or_else(|e| panic!("{e}"));
                if s.state.pos > rollover {
                    generated += s.state.pos;
                    s.rollover();
                    generated -= s.state.pos;
                    rollovers += 1;
                }
            }
            let dt = t.elapsed().as_secs_f64();
            let n = generated + s.state.pos - boot_pos;
            eprintln!("{rollovers} context rollovers");
            println!(
                "{instr} instructions, {n} tokens in {dt:.2}s: {:.0} tok/s, {:.0} instr/s",
                n as f64 / dt,
                instr as f64 / dt
            );
            if let Some(p) = s.state.prof {
                let total: u128 = p.iter().sum();
                eprintln!("time split: attention {:.0}%, ffn {:.0}%, unembed {:.0}%",
                    100.0 * p[0] as f64 / total as f64, 100.0 * p[1] as f64 / total as f64, 100.0 * p[2] as f64 / total as f64);
            }
            if let Some(p) = dump {
                write_u32s(&p, &s.tokens);
            }
        }
        _ => {
            eprintln!("usage:\n  engine predict <model_dir> <tokens.u32> <out.u32>\n  engine run <model_dir> <rom> [--instr N] [--ipf K] [--seed S] [--dump trace.u32]");
            std::process::exit(2);
        }
    }
}
