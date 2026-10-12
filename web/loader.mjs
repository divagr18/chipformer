// Load the chipformer WebAssembly engine (engine/src/wasm.rs). Works in browsers and Node.
//
//   const engine = await createEngine(wasmBytes, modelBytes);
//   engine.start(romBytes, "shift,clip");
//   engine.frame(keysMask, ipf);          // once per 60 Hz frame
//   engine.screen(20);                     // 64*32 bytes, 1 = lit
//   engine.tokens();                       // most recent tokens (Uint32Array)

export async function createEngine(wasmBytes, modelBytes) {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const x = instance.exports;
  const utf8 = new TextEncoder();

  const put = (bytes) => {
    const ptr = x.cf_alloc(bytes.length);
    new Uint8Array(x.memory.buffer, ptr, bytes.length).set(bytes);
    return [ptr, bytes.length];
  };
  const error = () =>
    new TextDecoder().decode(new Uint8Array(x.memory.buffer, x.cf_error_ptr(), x.cf_error_len()));
  const check = (rc) => {
    if (rc !== 0) throw new Error(`chipformer: ${error()}`);
  };

  const [mp, ml] = put(new Uint8Array(modelBytes));
  check(x.cf_load_model(mp, ml));
  x.cf_free(mp, ml);

  return {
    start(romBytes, quirks = "", keepTrace = false) {
      const [rp, rl] = put(new Uint8Array(romBytes));
      const [qp, ql] = put(utf8.encode(quirks));
      try {
        check(x.cf_start(rp, rl, qp, ql, keepTrace ? 1 : 0));
      } finally {
        x.cf_free(rp, rl);
        x.cf_free(qp, ql);
      }
    },
    frame: (keys, ipf) => check(x.cf_frame(keys >>> 0, ipf)),
    step: () => check(x.cf_step()),
    tick: () => check(x.cf_tick()),
    setKey: (key, down) => check(x.cf_set_key(key, down ? 1 : 0)),
    screen: (persist = 20) => new Uint8Array(x.memory.buffer, x.cf_screen(persist), 64 * 32).slice(),
    peek: (addr) => x.cf_peek(addr),
    tokens(all = false) {
      const n = x.cf_tokens(all ? 1 : 0);
      return new Uint32Array(x.memory.buffer, x.cf_tokens_ptr(), n).slice();
    },
    stats: () => ({
      tokens: x.cf_stat(0),
      instructions: x.cf_stat(1),
      context: x.cf_stat(2),
      rollovers: x.cf_stat(3),
    }),
  };
}

// Human-readable token, using the `machine` constants from the model config.
export function tokenName(t, m) {
  if (t < m.ADDR_BASE) return `=0x${(t - m.VAL_BASE).toString(16).toUpperCase()}`;
  if (t < m.STEP) return `@${cellName(t - m.ADDR_BASE, m)}`;
  return { [m.STEP]: "STEP", [m.TICK]: "TICK", [m.BOOT]: "BOOT" }[t] ?? `?${t}`;
}

export function cellName(a, m) {
  const hex = (v, n) => v.toString(16).toUpperCase().padStart(n, "0");
  if (a < 0x1000) return `M[${hex(a, 3)}]`;
  if (a >= 0x1000 && a < 0x1010) return `V${hex(a - 0x1000, 1)}`;
  const named = { 0x1010: "I", [m.PC]: "PC", 0x1012: "SP", 0x1013: "DT", 0x1014: "ST", [m.RNG]: "RNG", [m.CLS]: "CLS", [m.NULL]: "NULL" };
  if (named[a]) return named[a];
  if (a >= m.QUIRKS && a < m.QUIRKS + 6) return `Q.${m.QUIRK_NAMES[a - m.QUIRKS]}`;
  if (a >= 0x1020 && a < 0x1030) return `STACK${hex(a - 0x1020, 1)}`;
  if (a >= m.KEYS && a < m.KEYS + 16) return `KEY${hex(a - m.KEYS, 1)}`;
  if (a >= m.DISPLAY && a < m.NULL) {
    const i = a - m.DISPLAY;
    return `SCR${Math.floor(i / 8)}.${i % 8}`;
  }
  return `C${hex(a, 4)}`;
}
