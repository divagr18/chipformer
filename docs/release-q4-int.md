# q4-int: Qwen3.5-4B with a computer in its weights

`q4-int` is stock Qwen3.5-4B with a small exact computer compiled into extra layers. The computer
is a CHIP-8 machine plus a bignum arithmetic BIOS. Only config numbers change: hidden size, layer
count and vocabulary. It loads as `Qwen3_5ForConditionalGeneration` (or, in the text-only variant,
`Qwen3_5ForCausalLM`) in stock transformers. The circuit is bf16-exact, and Qwen's own weights are
unchanged (zero-padded).

This note describes branch `q35-int`. The checkpoint path, commit and measured numbers are at the
end.

## What is in it

| | |
|---|---|
| Base | Qwen3.5-4B (32 layers, hidden 2560) |
| Computer | 12 chip layers *before* Qwen's 32 (layers 0-11), then Qwen's layers |
| Hidden size | 3200 (Qwen's 2560 dims + ballast + the circuit's text-visible dims). Circuit dims that only chip tokens need are overlaid on Qwen's dims (`overlay.py`) |
| Size | ~14 GB bf16 (text-only variant: ~13 GB, no vision tower) |
| Vocabulary | Qwen's, plus 8708 chip tokens (`<\|chip…\|>`) at `chip_token_offset` |
| Circuit | 12 layers, 3220 dims, 84 heads, 28,253 MLP units (`chipformer.json` → `circuit`) |

**What the computer does (all exact):**
- **Inline.** When Qwen writes `A op B =`, the circuit takes over at the `=`, runs the BIOS
  routine and prints the result as ordinary text.
  - Forms: `\times`, `\cdot`, `×`, `*`, `+`, `-`, `/`, `÷`, `\div`, `mod`, `\bmod`, `%` and
    `gcd(A, B) =`.
  - Signed operands for mul, div and gcd (`-5`, `(-5)`).
  - Thresholds: mul/div/mod/gcd need ≥ 4 digits in A and B together, add/sub ≥ 7.
- **Whole expressions (v18).** For example `(3 + 4) \times 12 / 7 =` or `2^{10} - 5 =`.
  - The BIOS evaluates them with exact fractions: + − × / ^, brackets, unary minus, decimals.
  - The single-op path wins when both match.
- **Tools.** Qwen's native tool calls `add`, `sub`, `mul`, `div`, `mod`, `gcd` and `calc(expression)`.
  - The circuit answers at `</tool_call>` with `\n<tool_response>\nR\n</tool_response>\n`, then
    an empty thinking block `<think>\n\n</think>\n\n` (two `\n` tokens each), as Qwen's next
    assistant turn starts.
  - Negative results print `-R`.
- **Output format:**
  - Integers as digits.
  - A quotient that ends within 10 decimals prints as that decimal (`0.625`).
  - Otherwise the reduced fraction `p/q` (`1785/13201915 = 357/2640383`).
  - Division by 0, or a result the BIOS can't represent (more than 24 digits, a malformed
    expression), prints `...`.
  - Inline results end with a closer: `.`, or `$` / `$$` inside math.

## How to run it

1. **The BIOS boot record.** Every prompt starts with the BIOS's boot record, as chip tokens
   (about 5,000 tokens, prefill only):
   ```python
   from chipformer.q35.lib import bcd
   from chipformer.ref.emu import Chip8
   from chipformer.trace import boot_tokens
   meta = json.load(open(f"{path}/chipformer.json"))
   bios = [t + meta["chip_token_offset"] for t in boot_tokens(Chip8(bcd.ROM, bcd.QUIRKS).cells)]
   ids = bios + tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True,
                                        enable_thinking=False)["input_ids"]
   ```
   - The entry points are compiled into the weights, so the BIOS must be the one this checkpoint
     was built with. Check `chipformer.json` → `bios.sha1`, `ops`.
   - `bcd.ROM.segments` carries code below 0x200; `Chip8` loads it.
2. **The chat template.** Qwen's own template (`chat_template.jinja`), unchanged.
   - Tools: `chipformer.q35.demo.TOOLS` (add, sub, mul, div, mod, gcd), plus `demo.CALC_TOOL`.
   - With tools, Qwen calls them in its native XML format.
   - Without tools, inline mode works on Qwen's plain text.
3. **System prompts that help** (`chipformer/q35/evals.py`):
   - Tool mode: `NUDGE` ("You have tools for exact arithmetic (...). Use them for every
     arithmetic step they cover instead of calculating by hand.").
   - Inline mode: `NUDGE3` (single operations, including division).
   - Whole expressions: `NUDGE4`.
   - Products and quotients inline, sums through tools: `NUDGE_MIX3`.
4. **Generation config.** `generation_config.json` bans `<|im_start|>` and `<|endoftext|>`
   (`bad_words_ids`).
   - Qwen ends its turns with `<|im_end|>`, which stays the EOS.
   - Without the bans, Qwen sometimes opens a fake user turn, or ends with `<|endoftext|>` right
     after a tool response, before answering.
   - Greedy decoding (`do_sample=False`).
5. **Decoders:**
   - **Stock `generate()`** (`cache_implementation="static"`) runs the computer exactly. The
     circuit needs nothing from the host: one `generate()` call computes and prints.
     `chipformer/q35/demo.py` is a complete example.
   - **The numbers below use `chipformer.q35.batch.BatchDecoder`** with two decoder-side options.
     Both change only position ids and the KV layout, not weights or computer outputs.
     - `freeze_chip_positions` (fp): Qwen's RoPE positions don't advance on chip tokens.
     - `transparent_chip` (tc): Qwen's layers don't see chip tokens. Its text computation then
       equals the same text with the trace deleted.
     - Inline mode is markedly better with fp + tc. Tool mode works with both decoders.
   - **`--fast` (`skip_qwen`)** runs only the chip layers for rows on chip tokens. It is
     token-identical and about 2× faster at batch 16.
   - **flash-linear-attention / causal-conv1d** are not needed. The reference kernels are exact.
6. **Cache length.** A trace costs hundreds to thousands of chip tokens: a multiplication about
   100-400, a division with a fraction 1-9k, an expression 4-15k. Give the cache room, e.g.
   `--max-len 16384` (32768 for whole expressions); with tc, Qwen doesn't see them.

## chipformer.json

| Field | Meaning |
|---|---|
| `layout` | `v2`: chip layers first |
| `chip_token_offset`, `chip_vocab` | Where the 8708 chip tokens start in the vocabulary |
| `chip_layers`, `qwen_layers_from` | 12: Qwen's layers start at 12 |
| `hidden`, `chip_dims_offset`, `ballast_dims` | The widened residual: Qwen's dims `[0, 2560)`, then ballast and circuit slots |
| `circuit` | Layers, dims, heads, MLP units of the compiled circuit |
| `slots`, `overlay_slots`, `overlay` | Circuit dims of their own, dims overlaid on Qwen's, and the overlay plan's summary |
| `muted` | Qwen's layers are muted on chip tokens (no decay, no writes) |
| `text_ids` | The text classes the circuit reads and prints, by Qwen token id |
| `bios` | The BIOS the weights expect: sha1, size, segments, entry points |

## Measured (q35-inline, A10G 24 GB, 2026-10-19)

**The checkpoint.** `q35-inline:~/weights/dist/q4-int`.
- Size: 14.65 GB in 7 shards.
- Contents: config, tokenizer, `chat_template.jinja`, `generation_config.json`, `chipformer.json`,
  LICENSE, preprocessor configs.
- Text-only variant: `~/weights/dist/q4-int-text`, 13.98 GB.
- Built from `q35-int` (see the commit below) with `python -m chipformer.q35.merge <Qwen3.5-4B>
  --out q4-int` (overlay on), then `python -m chipformer.q35.textonly`.

**Shape.**
- Hidden 3200, ballast 50.
- 12 chip layers + 32 Qwen layers.
- Circuit: 3222 dims, 84 heads, 28,259 MLP units.
- 516 slots of its own, 1129 overlaid on Qwen's dims.
- BIOS sha1 `5e885286e8` (3072 B at 0x200, plus 174 B at 0x0E0).

**Selftest: 17 tool calls, 0 wrong.** Calls cover add, sub, mul and signed mul, div (decimal,
fraction, signed), mod, gcd and calc, including a negative calc.

**Results** (fp + tc, BatchDecoder `--fast`, greedy):

| Run | Correct | Computer results exact |
|---|---|---|
| Tool mode (`merged_nb_ne_fp_tc`), synthetic arith n=20 | 20/20 | 33/33 tool calls, 7/7 inline fires |
| Inline (`merged_inline_nudge_fp_tc`), arith n=20 | 15/20 (4 stopped at the cache length) | 72/72 fires |
| Tool mode, DM mul_big n=10 | 7/10 (4 cache stops) | 6/6 calls, 22/22 fires |
| Inline + `nudge3`, DM mul_div_multiple_longer n=10 | 3/10 (7 cache stops at 16k) | see the IR replay |
| Whole expressions (`nudge4`), DM mul_div_multiple_longer n=10 | 4/10 (2 cache stops at 24k) | 45/45 scored fires |

**IR replay** (the circuit's IR == the reference BIOS, on every computer-owned position of the
DM records above): 30 records, 177,816 positions, 0 mismatches, 0 stray chip tokens.

**Notes:**
- The DM division sets are short: these runs are 10 problems each.
- Most losses there are cache-length stops. A non-terminating division now runs gcd to print
  `p/q`, which costs 1-9k trace tokens (median 3.6k) instead of about 600.
  - With tc, Qwen doesn't see those tokens, but the static cache still holds them.
  - Use `--max-len 32768` or more on these sets, on a GPU with room: an A10G ran out of memory
    at 32k with the `--fast` CUDA graphs, and `--chip-per-full 0` turns those off.
- On a few multi-operation windows (markdown lists, `1.` list numbers), the scorer's
  attribution of an inline fire to an expression is heuristic. The IR replay is the exactness
  check.
- Not run for this release: text KL vs base, GSM8K and MMLU.

**Tests** (CPU, one file at a time):
- The q35 suite passes: ir, ir_vheads, mute, numerics, macros 19, expr 47, v2, chip8, inline 113.
- `test_q35_overlay` passes 5/5 (noisy-IR leak checks). The merged-checkpoint overlay test was
  run on the box and passes.
- `test_q35_store`: 10/11. One store-only failure, `test_circuit_with_arithmetic`: with
  `--store`, the store's tool response lacks the tool-response thinking tail (owner bits at its
  end). The store is optional and not in this checkpoint.

**Not in this release:**
- v17 (decimal operands for the single-op path). Its BIOS (≈550 B) doesn't fit the 4 KB next to
  v18's evaluator. Decimals work through the expression evaluator instead.
- The maths library (pow, sqrt, logs, …).
