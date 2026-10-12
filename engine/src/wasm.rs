//! WebAssembly interface: plain C-ABI exports, no bindings generator needed.
//!
//! JS usage (see web/loader.mjs): copy bytes in with `cf_alloc`, call `cf_load_model` once,
//! `cf_start` per game, then `cf_frame` every animation frame and read the screen and token
//! stream back out of linear memory. Calls return 0 on success, -1 on error (`cf_error`).

use crate::host::Session;
use crate::model::Model;
use std::cell::RefCell;

thread_local! {
    static MODEL: RefCell<Option<&'static Model>> = const { RefCell::new(None) };
    static SESSION: RefCell<Option<Session<'static>>> = const { RefCell::new(None) };
    static ERROR: RefCell<String> = const { RefCell::new(String::new()) };
    static SCREEN: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static TOKENS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

const ROLLOVER_TOKENS: usize = 250_000;

fn fail(e: impl Into<String>) -> i32 {
    ERROR.with(|err| *err.borrow_mut() = e.into());
    -1
}

fn with_session<T>(f: impl FnOnce(&mut Session<'static>) -> Result<T, String>) -> Result<T, String> {
    SESSION.with(|s| f(s.borrow_mut().as_mut().ok_or("no game started")?))
}

fn status(r: Result<(), String>) -> i32 {
    r.map_or_else(fail, |_| 0)
}

unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(ptr, len) } }
}

#[unsafe(no_mangle)]
pub extern "C" fn cf_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len.max(1));
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p
}

/// # Safety
/// `ptr` must come from `cf_alloc(len)` and not be freed twice.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cf_free(ptr: *mut u8, len: usize) {
    drop(unsafe { Vec::from_raw_parts(ptr, 0, len.max(1)) });
}

/// Load a model.chipformer file. The model lives for the rest of the program.
/// # Safety
/// `ptr..ptr+len` must be readable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cf_load_model(ptr: *const u8, len: usize) -> i32 {
    match Model::from_sparse_bytes(unsafe { bytes(ptr, len) }) {
        Ok(m) => {
            let m: &'static Model = Box::leak(Box::new(m));
            MODEL.with(|slot| *slot.borrow_mut() = Some(m));
            0
        }
        Err(e) => fail(e),
    }
}

/// Boot a ROM. `quirks` is a profile name or comma-separated flags (UTF-8). `keep` = 1
/// keeps the whole token trace (for verification); otherwise only recent tokens are kept.
/// # Safety
/// Both buffers must be readable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cf_start(rom: *const u8, rom_len: usize, quirks: *const u8, quirks_len: usize,
                                  keep: u32) -> i32 {
    let Some(model) = MODEL.with(|m| *m.borrow()) else { return fail("no model loaded") };
    let rom = unsafe { bytes(rom, rom_len) };
    let quirks = match std::str::from_utf8(unsafe { bytes(quirks, quirks_len) }) {
        Ok(q) => q,
        Err(_) => return fail("quirks must be UTF-8"),
    };
    match Session::new(model, rom, quirks, keep != 0) {
        Ok(s) => {
            SESSION.with(|slot| *slot.borrow_mut() = Some(s));
            0
        }
        Err(e) => fail(e),
    }
}

/// One 60 Hz frame: keys (bit k = CHIP-8 key k held), a TICK, then `ipf` instructions.
#[unsafe(no_mangle)]
pub extern "C" fn cf_frame(keys: u32, ipf: u32) -> i32 {
    status(with_session(|s| s.frame(keys as u16, ipf as usize, ROLLOVER_TOKENS)))
}

/// Execute exactly one instruction (single-step mode).
#[unsafe(no_mangle)]
pub extern "C" fn cf_step() -> i32 {
    status(with_session(|s| s.step().map(|_| ())))
}

#[unsafe(no_mangle)]
pub extern "C" fn cf_tick() -> i32 {
    status(with_session(|s| s.tick().map(|_| ())))
}

#[unsafe(no_mangle)]
pub extern "C" fn cf_set_key(key: u32, down: u32) -> i32 {
    status(with_session(|s| {
        s.set_key(key as usize & 15, down != 0);
        Ok(())
    }))
}

/// Visible pixels (64x32 bytes, 1 = lit): lit now or dark for fewer than `persist` instructions.
#[unsafe(no_mangle)]
pub extern "C" fn cf_screen(persist: u32) -> *const u8 {
    SCREEN.with(|buf| {
        let mut buf = buf.borrow_mut();
        buf.clear();
        let _ = with_session(|s| {
            for y in 0..s.spec.height {
                for x in 0..s.spec.width {
                    buf.push(s.visible(x, y, persist as u64) as u8);
                }
            }
            Ok(())
        });
        buf.as_ptr()
    })
}

/// Copy out tokens: `all` = 1 for the whole kept trace, else the most recent ones.
/// Returns the count; read them with `cf_tokens_ptr`.
#[unsafe(no_mangle)]
pub extern "C" fn cf_tokens(all: u32) -> u32 {
    TOKENS.with(|buf| {
        let mut buf = buf.borrow_mut();
        buf.clear();
        let _ = with_session(|s| {
            if all != 0 {
                buf.extend_from_slice(&s.tokens);
            } else {
                buf.extend(s.recent.iter());
            }
            Ok(())
        });
        buf.len() as u32
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn cf_tokens_ptr() -> *const u32 {
    TOKENS.with(|buf| buf.borrow().as_ptr())
}

/// Attention telemetry: `on` = 1 starts recording, for every token fed, each head's attention
/// distance (how many positions back it read). Read with `cf_telemetry_ptr/len`; call
/// `cf_telemetry_clear` before each frame to keep only the latest frame.
#[unsafe(no_mangle)]
pub extern "C" fn cf_telemetry(on: u32) -> i32 {
    status(with_session(|s| {
        s.state.telemetry = (on != 0).then(Vec::new);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn cf_telemetry_clear() {
    let _ = with_session(|s| {
        if let Some(t) = s.state.telemetry.as_mut() {
            t.clear();
        }
        Ok(())
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn cf_telemetry_len() -> u32 {
    with_session(|s| Ok(s.state.telemetry.as_ref().map_or(0, Vec::len) as u32)).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn cf_telemetry_ptr() -> *const u32 {
    with_session(|s| Ok(s.state.telemetry.as_ref().map_or(std::ptr::null(), |t| t.as_ptr()))).unwrap_or(std::ptr::null())
}

/// The host's mirror of a machine cell (built from the model's own writes), e.g. PC or memory.
#[unsafe(no_mangle)]
pub extern "C" fn cf_peek(addr: u32) -> u32 {
    with_session(|s| Ok(s.cells.get(addr as usize).copied().unwrap_or(0))).unwrap_or(0)
}

/// 0: tokens fed, 1: instructions executed, 2: context length, 3: context rollovers.
#[unsafe(no_mangle)]
pub extern "C" fn cf_stat(which: u32) -> f64 {
    with_session(|s| {
        Ok(match which {
            0 => s.fed as f64,
            1 => s.instructions as f64,
            2 => s.state.pos as f64,
            3 => s.rollovers as f64,
            _ => f64::NAN,
        })
    })
    .unwrap_or(f64::NAN)
}

#[unsafe(no_mangle)]
pub extern "C" fn cf_error_ptr() -> *const u8 {
    ERROR.with(|e| e.borrow().as_ptr())
}

#[unsafe(no_mangle)]
pub extern "C" fn cf_error_len() -> usize {
    ERROR.with(|e| e.borrow().len())
}
