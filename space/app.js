// chipformer Space: a CHIP-8 player whose CPU is a transformer.
// The WebAssembly engine (engine/src/wasm.rs) runs the model; this file only moves bytes in and
// out, paces frames, reads keys and draws. Ported from the site's engine.ts and useArcade.ts.

import { ASSET_BASE, ASSET_VERSION, GAMES, LAYERS, LIBRARY, MACHINE, MODEL_FACTS } from "./data.js";

const FRAME_MS = 1000 / 60;
const PERSIST = 20; // instructions a pixel stays lit after it goes dark (hides XOR redraw flicker)
const FADE = 0.42; // phosphor decay per frame
const TAP_FRAMES = 3;
const DEMO_SECONDS = 28;
const PHOSPHOR = [221, 245, 158];
const GLASS = [8, 12, 10];
const HEADS = LAYERS.reduce((n, l) => n + l.heads.length, 0);
const MAX_UNITS = Math.max(...LAYERS.map((l) => l.units));
const DEMOS = ["space-invaders", "tetris", "brix", "blinky", "br8kout", "ufo", "super-pong", "cave-explorer"]
  .map((id) => GAMES.find((g) => g.id === id)).filter(Boolean);

const $ = (id) => document.getElementById(id);
const fmt = (n) => Math.round(n).toLocaleString("en-US");
const hex = (v, n) => v.toString(16).toUpperCase().padStart(n, "0");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// ---------------------------------------------------------------- engine

class Engine {
  constructor(x) { this.x = x; }

  static async create(wasm, model) {
    const { instance } = await WebAssembly.instantiate(wasm, {});
    const e = new Engine(instance.exports);
    const [ptr, len] = e.put(model);
    e.check(e.x.cf_load_model(ptr, len));
    e.x.cf_free(ptr, len);
    return e;
  }

  put(bytes) {
    const ptr = this.x.cf_alloc(bytes.length);
    new Uint8Array(this.x.memory.buffer, ptr, bytes.length).set(bytes);
    return [ptr, bytes.length];
  }

  check(rc) {
    if (rc !== 0) {
      const b = new Uint8Array(this.x.memory.buffer, this.x.cf_error_ptr(), this.x.cf_error_len());
      throw new Error(new TextDecoder().decode(b));
    }
  }

  start(rom, quirks) {
    const [rp, rl] = this.put(rom);
    const [qp, ql] = this.put(new TextEncoder().encode(quirks.join(",")));
    try { this.check(this.x.cf_start(rp, rl, qp, ql, 0)); } finally {
      this.x.cf_free(rp, rl);
      this.x.cf_free(qp, ql);
    }
  }

  frame(keys, ipf) { this.check(this.x.cf_frame(keys >>> 0, ipf)); }
  step() { this.check(this.x.cf_step()); }
  /** 64*32 bytes, 1 = lit (or dark for fewer than `persist` instructions). */
  screen(persist) { return new Uint8Array(this.x.memory.buffer, this.x.cf_screen(persist), 64 * 32); }
  recentTokens() {
    const n = this.x.cf_tokens(0);
    return new Uint32Array(this.x.memory.buffer, this.x.cf_tokens_ptr(), n).slice();
  }
  peek(addr) { return this.x.cf_peek(addr); }
  setTelemetry(on) { this.check(this.x.cf_telemetry(on ? 1 : 0)); }
  clearTelemetry() { this.x.cf_telemetry_clear(); }
  /** Attention distances since the last clear: one per head per token, heads in model order. */
  telemetry() {
    const n = this.x.cf_telemetry_len();
    return n ? new Uint32Array(this.x.memory.buffer, this.x.cf_telemetry_ptr(), n) : new Uint32Array();
  }
  stats() {
    return { tokens: this.x.cf_stat(0), instructions: this.x.cf_stat(1), context: this.x.cf_stat(2) };
  }
  nextInstruction() {
    const pc = this.peek(MACHINE.PC);
    return { pc, op: (this.peek(pc) << 8) | this.peek((pc + 1) & 0xfff) };
  }
}

let enginePromise = null;
function loadEngine(progress) {
  if (!enginePromise) {
    enginePromise = (async () => {
      const v = `?v=${ASSET_VERSION}`;
      const [wasm, model] = await Promise.all([
        fetch(`${ASSET_BASE}/engine.wasm${v}`).then((r) => {
          if (!r.ok) throw new Error(`engine.wasm: HTTP ${r.status}`);
          return r.arrayBuffer();
        }),
        fetchWithProgress(`${ASSET_BASE}/model.chipformer${v}`, progress),
      ]);
      return Engine.create(wasm, model);
    })();
    enginePromise.catch(() => (enginePromise = null));
  }
  return enginePromise;
}

async function fetchWithProgress(url, progress) {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) progress(Math.min(1, got / total));
  }
  progress(1);
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out;
}

const romCache = new Map();
function loadRom(g) {
  if (!romCache.has(g.id)) {
    romCache.set(g.id, fetch(`${ASSET_BASE}/roms/${g.id}.ch8?v=${ASSET_VERSION}`).then(async (r) => {
      if (!r.ok) throw new Error(`${g.id}.ch8: HTTP ${r.status}`);
      return new Uint8Array(await r.arrayBuffer());
    }));
  }
  return romCache.get(g.id);
}

// ---------------------------------------------------------------- keys

/** Keyboard keys (KeyboardEvent.code) for each role; mirrors engine/src/games.rs. */
function roleKeys(role, two) {
  switch (role) {
    case "up": return two ? ["KeyW", "KeyA"] : ["ArrowUp", "KeyW"];
    case "down": return two ? ["KeyS", "KeyD"] : ["ArrowDown", "KeyS"];
    case "player2Up": return ["ArrowUp", "ArrowLeft"];
    case "player2Down": return ["ArrowDown", "ArrowRight"];
    case "left": return ["ArrowLeft", "KeyA"];
    case "right": return ["ArrowRight", "KeyD"];
    case "a": return ["Space", "KeyE"];
    case "b": return ["Enter", "KeyQ"];
  }
  return [];
}

/** The classic CHIP-8 keypad on a QWERTY keyboard: 123C/456D/789E/A0BF on 1234/QWER/ASDF/ZXCV. */
const CLASSIC = [
  ["Digit1", 0x1], ["Digit2", 0x2], ["Digit3", 0x3], ["Digit4", 0xc],
  ["KeyQ", 0x4], ["KeyW", 0x5], ["KeyE", 0x6], ["KeyR", 0xd],
  ["KeyA", 0x7], ["KeyS", 0x8], ["KeyD", 0x9], ["KeyF", 0xe],
  ["KeyZ", 0xa], ["KeyX", 0x0], ["KeyC", 0xb], ["KeyV", 0xf],
];
const isTwoPlayer = (g) => Object.keys(g.keys).some((r) => r.startsWith("player2"));

function keyBindings(g) {
  const two = isTwoPlayer(g);
  const map = new Map();
  const used = new Set();
  const add = (code, chip) => map.set(code, [...(map.get(code) ?? []), chip]);
  for (const [role, chip] of Object.entries(g.keys)) {
    for (const code of roleKeys(role, two)) add(code, chip);
    used.add(chip);
  }
  for (const [code, chip] of CLASSIC) if (!map.has(code) && !used.has(chip)) add(code, chip);
  return map;
}

const KEY_LABELS = { ArrowUp: "↑", ArrowDown: "↓", ArrowLeft: "←", ArrowRight: "→", Space: "Space", Enter: "Enter" };
const keyLabel = (code) => KEY_LABELS[code] ?? code.replace(/^Key|^Digit/, "");
const ROLE_LABELS = {
  up: "Up", down: "Down", left: "Left", right: "Right", a: "Action", b: "Second action",
  player2Up: "Player 2 up", player2Down: "Player 2 down",
};

// ---------------------------------------------------------------- tokens

function cellName(a) {
  if (a < 0x1000) return `M[${hex(a, 3)}]`;
  if (a < 0x1010) return `V${hex(a - 0x1000, 1)}`;
  const named = { 0x1010: "I", [MACHINE.PC]: "PC", 0x1012: "SP", 0x1013: "DT", 0x1014: "ST",
    [MACHINE.RNG]: "RNG", [MACHINE.CLS]: "CLS", [MACHINE.NULL]: "NULL" };
  if (named[a]) return named[a];
  if (a >= MACHINE.QUIRKS && a < MACHINE.QUIRKS + 6) return `Q.${MACHINE.QUIRK_NAMES[a - MACHINE.QUIRKS]}`;
  if (a >= 0x1020 && a < 0x1030) return `STACK${hex(a - 0x1020, 1)}`;
  if (a >= MACHINE.KEYS && a < MACHINE.KEYS + 16) return `KEY${hex(a - MACHINE.KEYS, 1)}`;
  if (a >= MACHINE.DISPLAY && a < MACHINE.NULL) {
    const i = a - MACHINE.DISPLAY;
    return `SCREEN[${Math.floor(i / 8)}:${i % 8}]`;
  }
  return `CELL ${hex(a, 4)}`;
}

function tokenText(t) {
  if (t < MACHINE.ADDR_BASE) return `=${hex(t - MACHINE.VAL_BASE, 2)}`;
  if (t < MACHINE.STEP) return `@${cellName(t - MACHINE.ADDR_BASE)}`;
  return t === MACHINE.STEP ? "STEP" : t === MACHINE.TICK ? "TICK" : t === MACHINE.BOOT ? "BOOT" : `#${t}`;
}

/** The most recent write (an ADDR token followed by its VAL) in a token window. */
function lastWrite(tokens) {
  for (let i = tokens.length - 2; i >= 0; i--) {
    const t = tokens[i];
    if (t >= MACHINE.ADDR_BASE && t < MACHINE.STEP && tokens[i + 1] < MACHINE.ADDR_BASE)
      return { addr: t - MACHINE.ADDR_BASE, value: tokens[i + 1] - MACHINE.VAL_BASE };
  }
  return null;
}

/** Mnemonic and a plain-English description of a CHIP-8 opcode. */
function describe(op) {
  const x = (op >> 8) & 0xf, y = (op >> 4) & 0xf, n = op & 0xf, nn = op & 0xff, nnn = op & 0xfff;
  const vx = `V${hex(x, 1)}`, vy = `V${hex(y, 1)}`;
  const d = (text) => ({ code: hex(op, 4), text });
  switch (op >> 12) {
    case 0x0: return d(op === 0x00e0 ? "Clear the screen" : op === 0x00ee ? "Return from a subroutine" : "Machine-code call (ignored)");
    case 0x1: return d(`Jump to ${hex(nnn, 3)}`);
    case 0x2: return d(`Call the subroutine at ${hex(nnn, 3)}`);
    case 0x3: return d(`Skip the next instruction if ${vx} = ${hex(nn, 2)}`);
    case 0x4: return d(`Skip the next instruction if ${vx} ≠ ${hex(nn, 2)}`);
    case 0x5: return d(`Skip the next instruction if ${vx} = ${vy}`);
    case 0x6: return d(`Set ${vx} to ${hex(nn, 2)}`);
    case 0x7: return d(`Add ${hex(nn, 2)} to ${vx}`);
    case 0x8: return d({
      0x0: `Copy ${vy} into ${vx}`, 0x1: `${vx} |= ${vy}`, 0x2: `${vx} &= ${vy}`, 0x3: `${vx} ^= ${vy}`,
      0x4: `${vx} += ${vy}, VF = carry`, 0x5: `${vx} -= ${vy}, VF = no borrow`,
      0x6: "Shift right, VF = the bit shifted out", 0x7: `${vx} = ${vy} − ${vx}, VF = no borrow`,
      0xe: "Shift left, VF = the bit shifted out",
    }[n] ?? "Unknown arithmetic (no effect)");
    case 0x9: return d(`Skip the next instruction if ${vx} ≠ ${vy}`);
    case 0xa: return d(`Point I at ${hex(nnn, 3)}`);
    case 0xb: return d(`Jump to ${hex(nnn, 3)} plus a register`);
    case 0xc: return d(`${vx} = random byte & ${hex(nn, 2)}`);
    case 0xd: return d(`Draw an 8×${n} sprite at (${vx}, ${vy}), VF = collision`);
    case 0xe: return d(nn === 0x9e ? `Skip if the key in ${vx} is held` : nn === 0xa1 ? `Skip if the key in ${vx} is not held` : "Unknown key instruction (no effect)");
    case 0xf: return d({
      0x07: `${vx} = delay timer`, 0x0a: `Wait for a key, store it in ${vx}`, 0x15: `Delay timer = ${vx}`,
      0x18: `Sound timer = ${vx}`, 0x1e: `I += ${vx}`, 0x29: `Point I at the font glyph for ${vx}`,
      0x33: `Write the decimal digits of ${vx} to memory`, 0x55: `Store V0–${vx} in memory`,
      0x65: `Load V0–${vx} from memory`,
    }[nn] ?? "Unknown instruction (no effect)");
  }
  return d("Unknown instruction");
}

// ---------------------------------------------------------------- state

const S = {
  mode: "boot", // boot | attract | howto | play | paused | operator | gameover | error
  game: GAMES.find((g) => g.id === location.hash.slice(1)) ?? DEMOS[0] ?? GAMES[0],
  engine: null,
  progress: 0,
  error: "",
  bindings: new Map(),
  tapChips: new Set(),
  pressed: new Map(), // keyboard code -> CHIP-8 keys
  touch: new Set(), // CHIP-8 keys held on the touch controls
  pulses: new Map(), // tap keys: frames left
  glow: new Float32Array(64 * 32),
  ema: new Float32Array(HEADS),
  sound: true,
  audio: null,
  loadSeq: 0,
};

function setMode(m) {
  S.mode = m;
  renderOverlay();
  renderTools();
  renderInspector();
}

function setGame(g) {
  S.game = g;
  S.bindings = keyBindings(g);
  S.tapChips = new Set(g.tap.map((r) => g.keys[r]).filter((c) => c !== undefined));
  renderChips();
  renderHints();
  renderTouch();
  renderCredit();
}

async function load(g) {
  const seq = ++S.loadSeq;
  const [e, rom] = await Promise.all([loadEngine((p) => { S.progress = p; if (S.mode === "boot") renderOverlay(); }), loadRom(g)]);
  if (seq !== S.loadSeq) return false; // a newer choice won
  e.start(rom, g.quirks);
  e.setTelemetry(true);
  S.engine = e;
  S.glow.fill(0);
  S.pressed.clear();
  S.touch.clear();
  S.pulses.clear();
  setGame(g);
  drawScreen();
  return true;
}

function fail(err) {
  S.error = err instanceof Error ? err.message : String(err);
  beep(false);
  setMode("error");
}

async function startDemo(g) {
  try { if (await load(g)) setMode("attract"); } catch (err) { fail(err); }
}

async function choose(g, then = "howto") {
  try {
    if (await load(g)) {
      setMode(then);
      history.replaceState(null, "", `#${g.id}`);
    }
  } catch (err) { fail(err); }
}

// ---------------------------------------------------------------- drawing

const canvas = $("screen");
const ctx = canvas.getContext("2d");
const small = document.createElement("canvas");
small.width = 64;
small.height = 32;
const sctx = small.getContext("2d");
const img = sctx.createImageData(64, 32);

function drawScreen() {
  const e = S.engine;
  if (!e) return;
  const vis = e.screen(PERSIST), g = S.glow;
  for (let i = 0; i < 2048; i++) {
    g[i] = vis[i] ? 1 : g[i] * FADE;
    for (let c = 0; c < 3; c++) img.data[i * 4 + c] = GLASS[c] + (PHOSPHOR[c] - GLASS[c]) * g[i];
    img.data[i * 4 + 3] = 255;
  }
  sctx.putImageData(img, 0, 0);
  ctx.globalCompositeOperation = "source-over";
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
  if ("filter" in ctx) {
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = 0.38;
    ctx.imageSmoothingEnabled = true;
    ctx.filter = "blur(16px)";
    ctx.drawImage(small, 0, 0, canvas.width, canvas.height);
    ctx.filter = "none";
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
  }
  // LCD pixel grid, drawn into the canvas
  ctx.fillStyle = "rgba(0, 0, 0, 0.42)";
  const cw = canvas.width / 64, ch = canvas.height / 32;
  for (let x = 1; x < 64; x++) ctx.fillRect(Math.round(x * cw) - 1, 0, 2, canvas.height);
  for (let y = 1; y < 32; y++) ctx.fillRect(0, Math.round(y * ch) - 1, canvas.width, 2);
}

// ---------------------------------------------------------------- the live model panel

const headCells = [];
function buildLayers() {
  const ol = $("layers");
  LAYERS.forEach((l, i) => {
    const li = document.createElement("li");
    li.className = "layer";
    li.title = l.heads.length ? `layer ${i} heads read: ${l.heads.join(", ")}` : `layer ${i}: arithmetic only`;
    li.innerHTML = `<b>L${String(i).padStart(2, "0")}</b><span class="heads"></span>` +
      `<span class="relu" style="width:${Math.max(8, Math.sqrt(l.units / MAX_UNITS) * 100)}%"></span>`;
    const heads = li.querySelector(".heads");
    for (const h of l.heads) {
      const s = document.createElement("span");
      s.className = "head";
      s.title = h;
      heads.appendChild(s);
      headCells.push(s);
    }
    ol.appendChild(li);
  });
  $("embed-note").textContent = `${fmt(MODEL_FACTS.vocab)} → ${MODEL_FACTS.dModel} dims`;
}

let lastPanel = 0;
function updatePanel(e, force = false) {
  const d = e.telemetry();
  const n = Math.floor(d.length / HEADS);
  for (let h = 0; h < HEADS && n; h++) {
    let sum = 0;
    for (let t = 0; t < n; t++) sum += Math.log2(d[t * HEADS + h] + 1);
    S.ema[h] = S.ema[h] * 0.6 + (sum / n / 20) * 0.4;
  }
  const now = performance.now();
  if (!force && now - lastPanel < 50) return;
  lastPanel = now;
  for (let h = 0; h < HEADS; h++) headCells[h].style.opacity = String(0.3 + 0.7 * Math.min(1, S.ema[h]));
  const toks = e.recentTokens();
  $("tokens-in").textContent = Array.from(toks.slice(-7), tokenText).join("  ");
  const w = lastWrite(toks.slice(-40));
  if (w) {
    $("next-cell").textContent = `@${cellName(w.addr)}`;
    $("next-val").textContent = `=${hex(w.value, w.value > 0xff ? 3 : 2)}`;
  }
  $("next-count").textContent = `#${fmt(e.stats().tokens)}`;
}

function renderInspector() {
  const box = $("inspector");
  const show = (S.mode === "paused" || S.mode === "operator") && S.engine;
  box.hidden = !show;
  if (!show) return;
  const next = S.engine.nextInstruction();
  const d = describe(next.op);
  $("insp-pc").textContent = `Next instruction · ${hex(next.pc, 3)}`;
  $("insp-code").textContent = d.code;
  $("insp-text").textContent = d.text;
}

// ---------------------------------------------------------------- sound

function beep(on) {
  if (on && S.sound && !S.audio) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    const ac = new Ctx();
    const osc = ac.createOscillator(), gain = ac.createGain();
    osc.type = "square";
    osc.frequency.value = 440;
    gain.gain.value = 0;
    osc.connect(gain).connect(ac.destination);
    osc.start();
    S.audio = { ac, gain };
  }
  const a = S.audio;
  if (a) a.gain.gain.setTargetAtTime(on && S.sound ? 0.03 : 0, a.ac.currentTime, 0.008);
}

// ---------------------------------------------------------------- main loop

let acc = 0, prevT = performance.now();
let demoKeys = 0, demoUntil = 0, demoSince = performance.now();
let last = { t: performance.now(), tokens: 0, instructions: 0 };
let fullSince = 0; // game-over rule: when the watched screen region became lit

function loop(now) {
  requestAnimationFrame(loop);
  const e = S.engine, m = S.mode, g = S.game;
  if (!e || !(m === "attract" || m === "play")) {
    prevT = now;
    acc = 0;
    return;
  }
  // The game advances at exactly 60 frames a second whatever the display's refresh rate.
  // Catch up at most 3 frames after a stall.
  acc += Math.min(100, now - prevT);
  prevT = now;
  if (acc < FRAME_MS) return;
  if (m === "attract") {
    if (now > demoUntil) {
      const roles = Object.values(g.keys);
      demoKeys = Math.random() < 0.6 && roles.length ? 1 << roles[Math.floor(Math.random() * roles.length)] : 0;
      demoUntil = now + 100 + Math.random() * 450;
    }
    if (now - demoSince > DEMO_SECONDS * 1000) {
      demoSince = now;
      acc = 0;
      void startDemo(DEMOS[(DEMOS.indexOf(g) + 1) % DEMOS.length]);
      return;
    }
  }
  e.clearTelemetry();
  for (let n = 0; acc >= FRAME_MS; n++) {
    if (n === 3) { acc = 0; break; }
    acc -= FRAME_MS;
    let keys = 0;
    if (m === "play") {
      for (const chips of S.pressed.values()) for (const c of chips) if (!S.tapChips.has(c)) keys |= 1 << c;
      for (const c of S.touch) if (!S.tapChips.has(c)) keys |= 1 << c;
      for (const [c, left] of S.pulses) {
        if (left > 0) {
          keys |= 1 << c;
          S.pulses.set(c, left - 1);
        }
      }
    } else keys = demoKeys;
    try { e.frame(keys, g.ipf); } catch (err) { fail(err); return; }
  }
  drawScreen();
  updatePanel(e);
  const rule = g.gameOver;
  if (m === "play" && rule) {
    const px = e.screen(0);
    let lit = false;
    for (const y of rule.rows) for (let x = rule.cols[0]; x <= rule.cols[1]; x++) lit ||= px[y * 64 + x] === 1;
    if (!lit) fullSince = 0;
    else if (!fullSince) fullSince = now;
    else if (now - fullSince > rule.seconds * 1000) {
      fullSince = 0;
      beep(false);
      setMode("gameover");
      return;
    }
  }
  beep(m === "play" && e.peek(0x1014) > 0);
  if (now - last.t > 500) {
    const s = e.stats(), dt = (now - last.t) / 1000;
    const tps = Math.max(0, (s.tokens - last.tokens) / dt), ips = Math.max(0, (s.instructions - last.instructions) / dt);
    $("readout").textContent = `${fmt(tps)} tokens/s · ${fmt(ips)} instructions/s · ${(ips / (g.ipf * 60)).toFixed(1)}× the game's normal speed · running in your browser`;
    last = { t: now, tokens: s.tokens, instructions: s.instructions };
  }
}

// ---------------------------------------------------------------- actions

function step() {
  const e = S.engine;
  if (!e) return;
  try {
    e.clearTelemetry();
    e.step();
  } catch (err) { fail(err); return; }
  drawScreen();
  updatePanel(e, true);
  renderInspector();
}

function pause() {
  if (S.mode !== "play") return;
  beep(false);
  setMode("paused");
}

function resume() { setMode("play"); }

function stepAction() {
  if (S.mode !== "operator") { beep(false); setMode("operator"); }
  step();
}

function restart() { void choose(S.game, "play"); }

/** Enter / Space / Esc outside play mode. */
function nav(action) {
  switch (S.mode) {
    case "attract": if (action === "start") void choose(S.game); break;
    case "howto": if (action === "start") setMode("play"); break;
    case "play": if (action === "back") pause(); break;
    case "paused": if (action === "start") resume(); break;
    case "operator": if (action === "start") step(); else if (action === "back") setMode("paused"); break;
    case "gameover": if (action === "start") restart(); break;
    case "error": if (action === "start") void choose(S.game); break;
  }
}

// ---------------------------------------------------------------- rendering

function renderChips() {
  const box = $("chips");
  if (!box.childElementCount) {
    for (const g of GAMES) {
      const b = document.createElement("button");
      b.className = "chip";
      b.textContent = g.title;
      b.dataset.id = g.id;
      b.onclick = () => void choose(g);
      box.appendChild(b);
    }
  }
  for (const b of box.children) b.setAttribute("aria-pressed", String(b.dataset.id === S.game.id && S.mode !== "attract"));
}

function keysHtml(g, max, p1 = "Player 1") {
  const two = isTwoPlayer(g);
  return Object.keys(g.keys).slice(0, max).map((r) =>
    `<span>${roleKeys(r, two).slice(0, 2).map((k) => `<kbd class="key">${esc(keyLabel(k))}</kbd>`).join("")}` +
    `<span>${two && (r === "up" || r === "down") ? `${p1} ${r}` : ROLE_LABELS[r]}</span></span>`).join("");
}

function renderHints() {
  $("hints").innerHTML = keysHtml(S.game, 5) + `<span><kbd class="key">Esc</kbd><span>Pause</span></span>`;
}

function renderOverlay() {
  const o = $("overlay"), g = S.game;
  o.className = "overlay";
  const dim = (html) => { o.className = "overlay center dim"; o.innerHTML = `<div class="stack">${html}</div>`; };
  switch (S.mode) {
    case "boot":
      dim(`<p>Loading the transformer</p><div class="bar"><i style="width:${Math.round(S.progress * 100)}%"></i></div>` +
        `<p class="label" style="color:rgb(255 255 255/.6)">${fmt(MODEL_FACTS.nonzero)} weights · ${(MODEL_FACTS.modelBytes / 1e6).toFixed(1)} MB</p>`);
      break;
    case "attract":
      o.className = "overlay corner";
      o.innerHTML = `<span class="label tag">Demo · ${esc(g.title)} · random input</span>` +
        `<button class="btn btn-acid" data-act="choose" style="margin-left:auto">Play ${esc(g.title)} <span aria-hidden="true">↗</span></button>`;
      break;
    case "howto":
      dim(`<p class="title">${esc(g.title)}</p><p class="sub">${esc(g.controls)}.</p>` +
        `<div class="keys fine">${keysHtml(g, 6, "P1")}</div>` +
        `<button class="btn btn-acid" data-act="start">Start</button><p class="label fine" style="color:rgb(255 255 255/.5)">or press any game key</p>`);
      break;
    case "paused":
      dim(`<p style="font-size:22px;letter-spacing:-.03em">Paused</p><div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center">` +
        `<button class="btn btn-acid" data-act="resume">Resume</button><button class="btn btn-ghost" data-act="step">Step one instruction</button></div>`);
      break;
    case "operator":
      o.className = "overlay center";
      o.style.alignItems = "flex-start";
      o.innerHTML = `<span style="margin-top:14px;background:rgb(0 0 0/.8);color:#fff;padding:6px 12px;font-size:12px">Stepping · Space runs one instruction · Esc back</span>`;
      return;
    case "gameover":
      dim(`<p style="font-size:26px;letter-spacing:-.03em">Game over</p>${g.gameOver ? `<p class="sub">${esc(g.gameOver.message)}.</p>` : ""}` +
        `<button class="btn btn-acid" data-act="restart">Play again</button>`);
      break;
    case "error":
      dim(`<p style="font-size:17px">Something went wrong</p><p class="mono" style="font-size:12px;color:rgb(255 255 255/.6)">${esc(S.error)}</p>` +
        `<button class="btn btn-acid" data-act="choose">Try again</button>`);
      break;
    default:
      o.innerHTML = "";
  }
  o.style.alignItems = "";
}

$("overlay").addEventListener("click", (ev) => {
  const act = ev.target.closest("[data-act]")?.dataset.act;
  if (act === "choose") void choose(S.game);
  else if (act === "start") setMode("play");
  else if (act === "resume") resume();
  else if (act === "step") stepAction();
  else if (act === "restart") restart();
});

function renderTools() {
  const play = $("btn-play");
  play.textContent = S.mode === "play" ? "Pause" : "Play";
  $("btn-step").disabled = S.mode === "boot" || S.mode === "error";
  renderChips();
}

$("btn-play").onclick = () => {
  const m = S.mode;
  if (m === "play") pause();
  else if (m === "paused" || m === "operator" || m === "howto") resume();
  else if (m === "gameover") restart();
  else void choose(S.game, "play");
};
$("btn-step").onclick = stepAction;
$("btn-restart").onclick = restart;
$("btn-sound").onclick = (ev) => {
  S.sound = !S.sound;
  ev.currentTarget.setAttribute("aria-pressed", String(S.sound));
  if (!S.sound) beep(false);
};
$("insp-step").onclick = stepAction;

function renderTouch() {
  const box = $("touch"), g = S.game;
  const pad = (role, label, w, h, round = false) => {
    const missing = g.keys[role] === undefined;
    return `<button type="button" class="pad${round ? " round" : ""}" data-role="${role}" aria-label="${ROLE_LABELS[role]}"` +
      `${missing ? " disabled" : ""} style="width:${w}rem;height:${h}rem">${label}</button>`;
  };
  if (isTwoPlayer(g)) {
    box.innerHTML = `<div class="pcol">${pad("up", "▲", 4, 3.5)}${pad("down", "▼", 4, 3.5)}<span class="label">Player 1</span></div>` +
      `<div class="pcol">${pad("player2Up", "▲", 4, 3.5)}${pad("player2Down", "▼", 4, 3.5)}<span class="label">Player 2</span></div>`;
  } else {
    box.innerHTML = `<div class="dpad"><span></span>${pad("up", "▲", 3, 3)}<span></span>${pad("left", "◀", 3, 3)}<span></span>` +
      `${pad("right", "▶", 3, 3)}<span></span>${pad("down", "▼", 3, 3)}<span></span></div>` +
      `<div class="ab">${pad("b", "B", 3.5, 3.5, true)}${pad("a", "A", 4, 4, true)}</div>`;
  }
  for (const b of box.querySelectorAll(".pad")) {
    const set = (down) => {
      b.dataset.held = String(down);
      setRole(b.dataset.role, down);
    };
    b.onpointerdown = (ev) => { ev.preventDefault(); b.setPointerCapture(ev.pointerId); set(true); navigator.vibrate?.(8); };
    b.onpointerup = () => set(false);
    b.onpointercancel = () => set(false);
    b.onlostpointercapture = () => b.dataset.held === "true" && set(false);
    b.oncontextmenu = (ev) => ev.preventDefault();
  }
}

function setRole(role, down) {
  const chip = S.game.keys[role];
  if (chip === undefined) return;
  if (down && (S.mode === "howto" || S.mode === "attract")) {
    if (S.mode === "attract") { void choose(S.game, "play"); return; }
    setMode("play");
  }
  if (S.tapChips.has(chip) && down && !S.touch.has(chip)) S.pulses.set(chip, TAP_FRAMES);
  if (down) S.touch.add(chip);
  else S.touch.delete(chip);
}

function renderCredit() {
  const g = S.game;
  const desc = g.description ? `<p class="desc">${esc(g.description)}</p>` : "";
  $("credit").innerHTML = `<span class="label">Now playing · ${esc(g.year ?? "CHIP-8")} · ${esc(g.license)}</span>` +
    `<h3>${esc(g.title)}</h3><div class="who">${esc(g.authors.join(", ") || "Unknown author")}</div>${desc}` +
    `<a class="src mono" href="${esc(g.source)}" target="_blank" rel="noopener">Source ↗</a>`;
}

function renderFacts() {
  const rows = [
    ["Architecture", "Decoder-only transformer: embeddings, causal softmax attention, ReGLU feed-forward"],
    ["Layers · heads", `${MODEL_FACTS.layers} layers · ${MODEL_FACTS.heads} heads · width ${MODEL_FACTS.dModel}`],
    ["Weights", `${fmt(MODEL_FACTS.params)} parameters, ${fmt(MODEL_FACTS.nonzero)} nonzero, all set by hand`],
    ["Training", "None. Zero gradient steps."],
    ["Correctness", `${LIBRARY.verified} of ${LIBRARY.total} games match a reference emulator token for token`],
  ];
  $("facts").innerHTML = `<span class="label">The model</span><dl>${rows.map(([k, v]) =>
    `<div class="row"><dt class="label">${k}</dt><dd>${v}</dd></div>`).join("")}</dl>`;
  $("fact-weights").textContent = fmt(MODEL_FACTS.nonzero);
}

// ---------------------------------------------------------------- keyboard

const editable = (t) => t instanceof HTMLElement && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName));

window.addEventListener("keydown", (ev) => {
  if (editable(ev.target) || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const m = S.mode;
  if (m === "play") {
    if (ev.code === "Escape") return pause();
    const chips = S.bindings.get(ev.code);
    if (!chips) return;
    ev.preventDefault();
    if (!ev.repeat) for (const c of chips) if (S.tapChips.has(c)) S.pulses.set(c, TAP_FRAMES);
    S.pressed.set(ev.code, chips);
    return;
  }
  let action = ev.code === "Enter" || ev.code === "Space" ? "start" : ev.code === "Escape" ? "back" : null;
  if (m === "howto" && S.bindings.has(ev.code)) action = "start";
  if (!action || (ev.repeat && m !== "operator")) return;
  ev.preventDefault();
  nav(action);
});
window.addEventListener("keyup", (ev) => S.pressed.delete(ev.code));
window.addEventListener("blur", () => S.pressed.clear());
document.addEventListener("visibilitychange", () => { if (document.hidden) pause(); });
window.addEventListener("hashchange", () => {
  const g = GAMES.find((x) => x.id === location.hash.slice(1));
  if (g && g.id !== S.game.id) void choose(g);
});

// ---------------------------------------------------------------- boot

buildLayers();
renderFacts();
setGame(S.game);
setMode("boot");
requestAnimationFrame(loop);
if (location.hash.length > 1 && GAMES.some((g) => g.id === location.hash.slice(1))) void choose(S.game);
else void startDemo(S.game);
