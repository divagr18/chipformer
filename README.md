# chipformer

**Tetris, running entirely inside a transformer.**

chipformer is a transformer that *is* a CHIP-8 computer. The game goes in as the prompt, and
every token the model predicts is one write to the machine's memory: a register, the program
counter, a row of pixels. Paint those writes onto a 64×32 screen and you are playing Tetris. Your
keypresses and the 60 Hz clock go into the context window; every frame, collision and score
comes out of the model. There is no emulator behind it, just the loop every language model runs:
predict the next token, append it, repeat.

One model plays Tetris, Space Invaders, Brix, Pong and 100+ other games in real time. Switching
games means switching the prompt.

- **Play it in your browser:** [divagr.com/chipformer](https://divagr.com/chipformer)
- **Model on Hugging Face:** [divagr1925/chipformer](https://huggingface.co/divagr1925/chipformer)
  (loads with `transformers`)
- **Hugging Face Space:** [spaces/divagr1925/chipformer](https://huggingface.co/spaces/divagr1925/chipformer)
- **Desktop app:** a single file with the model and games inside, on the
  [Releases page](https://github.com/divagr18/chipformer/releases)

The ROM is the prompt. Every instruction becomes a short run of tokens (`ADDR a, VAL v` writes,
then `STEP`). Keypresses and 60 Hz timer ticks are tokens the host inserts into the stream.
Nothing about CHIP-8 is in the inference code; it all lives in the weights.

## Play on your desktop

Download `chipformer` for Windows, macOS or Linux from
[Releases](https://github.com/divagr18/chipformer/releases). It is one ~3 MB file with the
model and every game inside.

```
chipformer                     # pick a game
chipformer list                # featured games and controls (--all for all 119)
chipformer play tetris
chipformer play my_game.ch8 --quirks octo
chipformer credits
```

The binaries are unsigned. On macOS, right-click → Open the first time (or
`xattr -d com.apple.quarantine chipformer`). On Windows, SmartScreen may ask you to confirm.

## Load it with `transformers`

It is an ordinary checkpoint (fp64 safetensors plus a ~150-line PyTorch module), so the standard
loader works:

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "divagr1925/chipformer", trust_remote_code=True, dtype=torch.float64)
game = model.play(open("tetris.ch8", "rb").read(), quirks="shift,loadstore,clip,jump", ipf=15)
for _ in range(60):
    game.frame([])          # held CHIP-8 keys for this frame
print(game.ascii())         # the 64x32 screen
```

PyTorch decodes one token at a time at ~150–250 tok/s on a CPU, a few frames per second. It
produces exactly the same tokens as the reference emulator. For real time, use the Rust engine,
the desktop binary or the browser demo.

## The model

| | |
|---|---|
| Layers | 18 |
| d_model | 382 |
| Attention heads | 42, each with **2-dimensional** queries and keys |
| FFN | ReGLU, 22,276 hidden units in total |
| Parameters | 32.3M dense, 139,229 nonzero |
| Vocabulary | 8,708 tokens (4,096 values, 4,609 cell addresses, STEP/TICK/BOOT) |
| Speed | ~100,000 tok/s on one CPU core natively; real time in the browser via WebAssembly |

The architecture is plain: token and position embeddings, residual blocks of causal softmax
attention plus a ReGLU feed-forward layer, and a linear unembedding. `chipformer/dense.py` is a
~40-line NumPy forward pass with no CHIP-8 knowledge. On real traces its residual stream is
bit-identical to the circuit specification. Everything runs in fp64.

## How it works

- **Memory is attention.** Reading cell `A` means attending to the latest write to `A`. The
  key is `(2TA, -TA^2 + s)` and the query is `(q, 1)`, so the score is `T(q^2 - (A-q)^2) + s`:
  an exact address match wins, and recency `s` breaks ties. Cells never written read as 0.
- **Hardmax.** Scores are integers, so softmax at temperature 1/1000 is exactly one-hot in fp64.
- **Exact arithmetic from ReGLU.** For integer x, `step(x) = ReLU(x+1) - ReLU(x)`, and
  `a*ReLU(c) = a*c` for c in {0,1}. Adds, compares, bitwise ops, muxes and opcode decoding are
  all built from these (`chipformer/dsl/lib.py`).
- **Records read the old state.** Each instruction is computed once at its `STEP` position.
  Later positions copy that context with one attention head, so writes earlier in the same
  record never corrupt later ones.
- **Quirks are data.** CHIP-8 interpreters disagree on a few behaviours (shifts, load/store,
  sprite wrapping...). The quirk profile is six memory cells in the boot prompt, so one network
  runs games written for any of them.
- **Log-time KV cache.** Every head is "max over lines `kx*q + ky` at `q`", so the Rust engine
  keeps a Li Chao tree per key cache: exact O(log n) attention instead of O(n).
- **Exact engine shortcuts** (`engine/src/fast.rs`, disable with `CHIPFORMER_NO_FAST=1`). The
  plain sparse engine runs at ~7k tok/s; these exact rewrites take it to ~100k:
  - fold constant dims into biases
  - compute each distinct FFN input sum once per layer
  - evaluate groups of `c*ReLU(s+b)` units on one input with a binary search over prefix sums
  - argmax the rank-1 unembedding as a max of lines
  - share one KV cache between heads with identical keys

The weights are compiled rather than learned, which is what makes every instruction exact. The circuit is written in a small DSL (`chipformer/chip8.py`). Each node is automatically placed
in the earliest legal sublayer, and `chipformer/compile.py` turns the graph into weight
matrices in about 0.3 s.

## Display

CHIP-8 games draw with XOR: to move a sprite they erase it and then redraw it. A screen
sampled at one instant often catches a sprite half-erased (in Pong, 47% of frames would show a
missing paddle). The players show a pixel only once it has been stable for a few instructions,
and keep it lit briefly after it goes dark.

## Building from source

```
uv sync
uv run python -m chipformer.compile --out model          # build the weights (0.3 s)
cargo build --release --manifest-path engine/Cargo.toml
engine/target/release/play model --game tetris
```

## Layout

| Path | What |
|---|---|
| `chipformer/spec.py`, `docs/trace_format.md` | cell map, vocabulary, trace grammar |
| `chipformer/ref/` | reference CHIP-8 emulator |
| `chipformer/dsl/` | circuit DSL, exact evaluators (incremental and batched) |
| `chipformer/chip8.py` | CHIP-8 as a circuit |
| `chipformer/compile.py`, `chipformer/dense.py` | graph to weights, generic NumPy transformer |
| `chipformer/check.py`, `chipformer/fuzz.py` | differential testing, random-program fuzzer |
| `chipformer/games.py`, `games/` | game library: manifest, ROMs, credits |
| `chipformer/web.py`, `web/` | WebAssembly build and Node test |
| `engine/` | Rust engine: `engine predict/run`, `play`, `chipformer`, wasm exports |

## Verification

- `uv run pytest` runs the following:
  - exhaustive checks of the primitives
  - circuit vs reference emulator, token by token, on the Timendus test suite and fuzzed
    programs (all 35 opcodes, self-modifying code, every quirk combination)
  - dense transformer vs circuit (bit-identical)
  - free-running generation, and every featured game through the Rust engine
- `engine run model roms/pong.ch8 --instr 30000 --dump t.u32` dumps a free-running trace.
  `chipformer.check.verify_trace` replays it against the reference emulator.
- `node web/test.mjs` checks that the WebAssembly build produces the same trace as native.

Not supported: SUPER-CHIP / XO-CHIP opcodes and the display-wait (vBlank) quirk.

## Prior art

Programs compiled into transformer weights: Tracr, ALTA, looped transformers (Giannou et al.,
2023), and Percepta's transformer-vm. chipformer's twist is real-time play: one fixed network,
many games, with interpreter quirks as data.

## License

Code and weights: MIT. The CHIP-8 ROMs in `games/roms` belong to their authors; CHIP-8 Archive
games are CC0 and the classics come from freely distributable packs. See
[`games/ATTRIBUTION.md`](games/ATTRIBUTION.md).
