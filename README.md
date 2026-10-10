# chipformer

A decoder-only transformer whose weights are **constructed, not trained**, that executes CHIP-8
programs. It plays Pong in real time on one CPU core.

The ROM is the prompt. Generating text is executing the program: every instruction becomes a
short run of tokens (`ADDR a, VAL v` writes, then `STEP`). Keypresses and 60 Hz timer ticks are
tokens the host inserts into the stream. Nothing about CHIP-8 is in the inference code; it all
lives in the weights.

```
uv sync --extra play
uv run python -m chipformer.compile --out model          # build the weights (0.3 s)
cd engine && cargo build --release && cd ..
engine/target/release/play model roms/pong.ch8           # A/D left paddle, Left/Right right paddle
```

`play` options:

| Option | Effect |
|---|---|
| `--keys classic` | standard CHIP-8 keypad layout (1234 / QWER / ASDF / ZXCV) |
| `--ipf N` | instructions per frame, i.e. game speed (default 10) |
| `--persist N` | instructions a pixel stays lit after going dark, for anti-flicker (default 20) |
| `--no-fade` | turn off the phosphor fade |
| `--scale N` | window scale |

## The model

| | |
|---|---|
| Layers | 18 |
| d_model | 332 |
| Attention heads | 36, each with **2-dimensional** queries and keys |
| FFN | ReGLU, 22,188 hidden units in total |
| Parameters | 28.0M dense, 138,698 nonzero |
| Vocabulary | 8,708 tokens (4,096 values, 4,609 cell addresses, STEP/TICK/BOOT) |
| Speed | ~99,000 tok/s on one CPU core (≈14,800 CHIP-8 instructions/s; Pong needs ~600) |

The architecture is plain: token and position embeddings, residual blocks of causal softmax
attention plus a ReGLU feed-forward layer, and a linear unembedding. `chipformer/dense.py` is a
~40-line NumPy forward pass with no CHIP-8 knowledge. On real traces its residual stream is
bit-identical to the circuit specification.

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
- **Log-time KV cache.** Every head is "max over lines `kx*q + ky` at `q`", so the Rust engine
  keeps a Li Chao tree per key cache: exact O(log n) attention instead of O(n).
- **Exact engine shortcuts** (`engine/src/fast.rs`, disable with `CHIPFORMER_NO_FAST=1`). The
  plain sparse engine runs at ~7k tok/s; these exact rewrites take it to ~99k:
  - fold constant dims into biases
  - compute each distinct FFN input sum once per layer
  - evaluate groups of `c*ReLU(s+b)` units on one input with a binary search over prefix sums
  - argmax the rank-1 unembedding as a max of lines
  - share one KV cache between heads with identical keys (36 heads use 5 key caches)

The circuit is written in a small DSL (`chipformer/chip8.py`). Each node is automatically placed
in the earliest legal sublayer, and `chipformer/compile.py` turns the graph into weight
matrices.

## Display

CHIP-8 games draw with XOR: to move a sprite they erase it and then redraw it. A screen
sampled at one instant often catches a sprite half-erased. In Pong, 47% of frames would show a
missing paddle, because the game erases both paddles and redraws them about 17 instructions
later on every loop. The player keeps each pixel visible for `--persist` instructions after it
goes dark (16 is enough for Pong), plus a short phosphor fade.

## Layout

| Path | What |
|---|---|
| `chipformer/spec.py`, `docs/trace_format.md` | cell map, vocabulary, trace grammar |
| `chipformer/ref/` | reference CHIP-8 emulator (`play.py` for a pygame version) |
| `chipformer/dsl/` | circuit DSL, exact evaluators (incremental and batched) |
| `chipformer/chip8.py` | CHIP-8 as a circuit |
| `chipformer/compile.py`, `chipformer/dense.py` | graph to weights, generic NumPy transformer |
| `chipformer/check.py`, `chipformer/fuzz.py` | differential testing, random-program fuzzer |
| `engine/` | Rust engine: `engine predict/run`, `play` |

## Verification

- `uv run pytest` runs the following:
  - exhaustive checks of the primitives
  - circuit vs reference emulator, token by token, on Pong, the Timendus test suite and fuzzed
    programs (all 35 opcodes, self-modifying code)
  - dense transformer vs circuit (bit-identical)
  - free-running generation
  - the Rust engine, including context rollover
- `engine run model roms/pong.ch8 --instr 30000 --dump t.u32` dumps a free-running trace.
  `chipformer.check.verify_trace` replays it against the reference emulator.

Quirks follow the CHIP-48 profile (`docs/quirks.md`). ROMs are not included.
