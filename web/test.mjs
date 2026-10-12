// Run a ROM on the WebAssembly engine with exactly the schedule of the native
// `engine run` (a TICK every ipf instructions, xorshift key presses) and dump the trace.
//
//   node web/test.mjs <wasm> <model.chipformer> <rom> <quirks> <instr> <ipf> <seed> <out.u32>

import { readFileSync, writeFileSync } from "node:fs";
import { createEngine } from "./loader.mjs";

const [wasm, model, rom, quirks, instrS, ipfS, seedS, out] = process.argv.slice(2);
const instr = Number(instrS), ipf = Number(ipfS);
const engine = await createEngine(readFileSync(wasm), readFileSync(model));
engine.start(readFileSync(rom), quirks === "-" ? "" : quirks, true);

const mask = (1n << 64n) - 1n;
let seed = BigInt(seedS);
const t0 = performance.now();
for (let i = 0; i < instr; i++) {
  if (i % ipf === 0) {
    engine.tick();
    seed ^= (seed << 13n) & mask;
    seed ^= seed >> 7n;
    seed ^= (seed << 17n) & mask;
    if (seed % 10n < 3n) engine.setKey(Number((seed >> 8n) % 16n), (seed >> 16n) % 2n === 0n);
  }
  engine.step();
}
const secs = (performance.now() - t0) / 1000;
const s = engine.stats();
writeFileSync(out, Buffer.from(engine.tokens(true).buffer));
console.log(`${instr} instructions, ${s.tokens} tokens in ${secs.toFixed(2)}s: ` +
  `${Math.round(s.tokens / secs)} tok/s (wasm in Node)`);
