# Making the merged checkpoint smaller

The merged Qwen3.5-4B + CHIP-8 checkpoint (v15s, the overlay build) is 15.15 GB against the base
model's 9.34 GB, although the circuit itself is only about 0.7M nonzero weights. This note
measures where the extra bytes go, ranks the ways to remove them, and reports the ones built.

The rules every option must keep:
- The result loads in stock transformers: config-only changes, no `trust_remote_code`, no custom
  kernels.
- Qwen's own text behaviour stays within the v15 noise floor.
- The circuit stays exact in bf16.

## Where the bytes go (v15s, measured from the safetensors)

| Part | Params | GB (bf16) | Nonzero |
|---|---|---|---|
| 16 chip layers (width 3072) | 2.168B | 4.34 | 0.7M |
| Qwen's 32 layers (width 2560 → 3072) | 4.284B | 8.57 | 3.570B (the rest is zero padding) |
| Embedding (257,088 × 3072, tied) | 0.790B | 1.58 | 0.640B |
| Vision tower | 0.336B | 0.67 | 0.334B |
| **Total** | **7.577B** | **15.15** | |

The stock architecture forces one hidden size and one intermediate size on every layer. So both
levers are the same two numbers:
- **The chip layer count.** One chip layer at width 3072 is 135.5M params (0.27 GB).
- **The hidden size.** Every dim costs about 2.2M params (4.4 MB) across all 44-48 layers and the
  embedding.

The overlay already put the circuit's chip-token-only dims inside Qwen's 2560. The width above
2560 holds only the dims the circuit needs at text positions (the "text-visible" slots: 499 in
v15s) and the ballast.

## Ranked options

GB is the saving at 4B. "Stock" means the result loads in stock transformers.

| # | Option | Saves | Stock | Risk | Status |
|---|---|---|---|---|---|
| 1 | Fewer chip layers: paired RAM value heads, list scheduling, fused exit (16 → 12) | 4 × 0.26 GB ≈ 1.0 GB | yes | low (IR and GPU exact) | **built** |
| 2 | Narrower: text-visible slots 499 → 318, so hidden 3072 → 2944 | 128 dims ≈ 0.56 GB | yes | low | **built** |
| 3 | Text-only export (`Qwen3_5ForCausalLM`, no vision tower) | 0.67 GB | yes, but no image input | none for text | **built** (`textonly.py`) |
| 4 | Download size: Hub chunk dedup plus compression | (see below) | n/a | none | measured |
| 5 | More text-visible slot cuts (call capture, j thermometer, ctx, div) → 2880 / 2816 | 0.28 GB per 64 dims | yes | medium (circuit rework) | not done |
| 6 | Shorter critical path (the BMUL digit chain sets the 12 layers now) | 0.26 GB per layer | yes | high (macro redesign) | not done |
| 7 | FP8 Qwen layers (`quantization_config`) | ~4 GB | yes (fp8 loader) | changes Qwen's outputs; no official Qwen3.5-4B FP8 | rejected |
| 8 | Interleave the circuit into Qwen's own layers | negative (see below) | — | — | rejected |

## What was built

### Chip layers 16 → 12 (4B)
1. **Paired RAM value heads.**
   - Qwen3.5-4B has two value heads per key head, and the circuit used only the first.
   - Now each RAM key head holds both 2 KB halves of CHIP-8's memory, one per value head.
     - The write gate (beta) picks the half by address bit 11, which a "prev" head column
       delivers from the ADDR token.
     - On a read, the output gate passes only the half asked for.
   - Both RAM copies drop from 32 key heads in two attention sublayers to 16 in one.
   - The value-head machinery is the store fork's (`ir.VHead`, per-value-head beta and outputs).
2. **RAM reads summed into shared dims.** Every RAM head adds its 8 read bits into the same 8
   dims (head `into`): a query matches at most one head and the rest read exactly 0. Before, each
   head had 8 dims of its own (256 per copy).
3. **List scheduling** (`ir.Graph.reschedule`).
   - Build-time placement is greedy in build order, so a node off the critical path could take
     the room a critical one needed. The ramA fetch heads lost their sublayer to `trim.R`,
     `stack` and `digit.count`.
   - The scheduler re-places every node sublayer by sublayer, longest remaining path first.
   - It keeps the build-order meaning of dims several nodes write: a reader sees only the writes
     built before it.
4. **Fused exit.** The exit computes the output code straight from the output cases
   (`2κ·AND(case, bit, owner) − κ·owner`), so the separate `next.*` select sublayer before it is
   gone. The IR check now decodes the predicted token from the exit code itself (`c.next_bits`),
   which is what the lm_head reads.

Result: 4B 12 layers (the critical path, uncapped, is also 12). The tiny / 0.8B targets
(16 value heads, no pairing) go 16 → 13. The 0.8B's overlay pool is full, though, and the packed
schedule needs more chip-region slots there. So `merge.py` builds both the scheduled and the
build-order circuit and keeps the smaller checkpoint.

### Text-visible slots 499 → 318 (hidden 3072 → 2944)
The slot audit (`dist/size/audit.py`) finds the dims live at the peak of the chip region. The big
items were representations that need many dims where a few would do:

| Item | Before | After | How |
|---|---|---|---|
| ADDR token's address | 49 one-hot dims | 13 bits | Write keys read them through a conv lag tap, so they can't overlay. Each key is now an AND of 13 bit literals (`bit_lit`: 0 on non-ADDR tokens, like the one-hots). |
| Vector-read address (`vread_addr`) | 48 one-hots | 12 bits | `prims.add12_bits`. Each RAM row's query is an AND of bit literals. It is needed on text tokens because a print reads its digits there. |
| Read bits (sum into shared dims) | 128 | 8 | |
| `I+j` | | 12 bits | |
| Display target row/column | 40 one-hots | 8 bits | |

With 318 chip slots (the table above plus the display target gated to 0 on text, and the
scheduler's headroom below), `pick_hidden` gives 2944 (m = 46 ballast dims;
2560 + 46 + 1 + 318 ≤ 2944, 19 to spare).

### Text-only export
`python -m chipformer.q35.textonly MERGED OUT` writes the language model as a stock
`Qwen3_5ForCausalLM`: keys `model.language_model.*` → `model.*`, `text_config` as the config, and
no vision tower. `evals.load_model` (used by evals and arith_bench) loads either kind.

### Two fixes the shrink needed
- **Counter slots are never handed over** (`emit.allocate`, "sticky").
  - The counter heads' outputs reach 2^-114.
  - Once normed by the ballast, such a value underflows, so the a-form unit that hands a slot
    over (or the exit's clear) cannot subtract it exactly.
  - Before, a residue always happened to be absorbed by a larger value sharing the slot.
  - Below the head's declared `tiny` (2^-90) the residue is left in an unread chip dim, and
    `test_q35_v2` allows exactly that.
- **The scheduler leaves 1/16 of every MLP free.** A full sublayer leaves the allocator no unit
  for a handover, which costs slots: 339 at a 9216 cap, 318 at 8640, same depth.

## Results (q35-batch, L4)

**Size** (bytes on disk; base Qwen3.5-4B is 9.34 GB):

| Checkpoint | Hidden | Chip layers | Params | GB | vs v15s |
|---|---|---|---|---|---|
| v15s (start) | 3072 | 16 | 7.577B | 15.15 | |
| **v15x** (this branch) | **2944** | **12** | **6.756B** | **13.53** | **−1.62 GB (−11%)** |
| v15x text-only | 2944 | 12 | 6.421B | 12.86 | −2.29 GB (−15%) |

v15x breakdown:

| Part | Params | Nonzero |
|---|---|---|
| Chip layers | 1.558B | 0.8M |
| Qwen layers | 4.105B | 3.570B |
| Embedding | 0.757B | |
| Vision | 0.335B | |

So the extra over base is now 4.2 GB, against 5.8 GB before:
- 3.1 GB of chip-layer zeros
- 1.1 GB of Qwen padding to 2944

The tiny standalone checkpoint goes from 16 layers at 2048 to 13 layers at 1536, about 38% fewer
bytes per token.

**Exactness and behaviour:**
- **Circuit (IR):** the test ROMs on two quirk profiles and the fuzz pass on the 4B and tiny
  targets with 0 bad predictions.
- **Tests:** the full q35 suite passes, one file at a time (147 passed):
  - v2 12, chip8 16 (including the HF tiny checkpoint, bf16 and fp32)
  - inline 74, macros 18, overlay 6 (including the noisy-IR leak checks and the synthetic merged
    checkpoint)
  - store 10, mute 2, numerics 5, ir 3, vheads 1
- **Real 4B v15x on GPU:**
  - The evals selftest: 7 tool calls, 0 wrong.
  - Inline `tc` arith: 14/20, with all 61 computer results exact. v15s got 16 on the same
    problems: 0 wins, 2 losses, both Qwen's own text, not significant. v15z, the same Qwen
    weights at 13 chip layers, got 15. The spread is batch numerics: with `tc`, Qwen's text
    can't depend on the chip layers.
  - DM mul_big (8 problems): 7/8, with 10/10 results exact.
  - IR replay of those records: 1,440 owned positions, 0 mismatches, 0 stray chip tokens.
- **Text vs base** (`text_eval`, same items as v15s):

| Model | MMLU choice KL | MMLU acc diff | ARC choice KL | ARC acc_norm diff |
|---|---|---|---|---|
| v15s | 6.9e-4 | −0.5 | 8.8e-4 | +0.2 |
| v15x | 7.2e-4 | −0.1 (p=1.0) | 8.3e-4 | +0.2 (p=0.73) |

  That is unchanged.
- **Text-only export:** loads as `Qwen3_5ForCausalLM`, gives logits identical to the full model
  (max |diff| 0.0), and its evals selftest is 7/7 exact.

## Download size

Measured by hashing 64 KB blocks (chunk dedup, as the Hub's Xet storage does) and compressing
the unique blocks (zlib level 1 with bf16 byte grouping, close to Xet's LZ4 plus byte grouping):

| Checkpoint | On disk | All-zero blocks | Unique blocks | Unique, compressed |
|---|---|---|---|---|
| base Qwen3.5-4B | 9.32 GB | 0.00 | 9.32 | 6.71 |
| v15s | 15.15 GB | 4.17 | 10.84 | 6.78 |
| v15x | 13.51 GB | 2.90 | 10.47 | **6.75** |
| v15x text-only | 12.84 GB | 2.90 | 9.81 | **6.27** |

So a deduplicating, compressing transfer moves about as much for the merged model as for the
base (6.75 vs 6.71 GB). The zero padding sits in short runs inside rows, so chunk dedup misses
it, but compression removes it. What remains bigger is GPU memory and decode bandwidth: those
follow the 13.5 GB on disk.

No stock-loadable format stores zeros cheaply:
- compressed-tensors dropped sparse compression, and transformers' loader only handles
  quantization formats.
- safetensors has no shared or zero tensors.

So the dedup and compression of the transfer is what a download gets.

## FP8 (option 7)

- **No official FP8 4B.** Qwen publishes FP8 checkpoints only for Qwen3.5-27B and larger. An FP8
  Qwen3.5-4B would be our own quantization, so it is no longer "Qwen as shipped".
- **The chip layers can't follow.** Of the 2,193 chip-weight 128×128 blocks that have nonzeros,
  1,684 are exactly e4m3 under some power-of-2 block scale, but 509 aren't: those weights need 5
  to 8 significant bits. The chip layers would have to stay bf16 (`modules_to_not_convert`).
- **What it would save:** Qwen's 32 layers at 1 byte per weight, about 4.1 GB at 2944.
- Rejected under the "Qwen as shipped" rule. It is worth revisiting if an official FP8 4B
  appears; widening it would need re-blocking the padded columns.

## Interleaving the circuit into Qwen's layers (option 8)

The idea: raise `intermediate_size` and the GDN head counts a little for all 32 layers, and run
the circuit inside Qwen's own layers instead of 12 extra ones. It doesn't pay, for three
measured reasons.

1. **Chip tokens.** At chip positions Qwen's sublayers still run. In stock generate (no `tc`),
   Qwen's conv windows and full-attention KV then take in each chip token's projection.
   - Today that projection is exactly 0, apart from the mute marker, because the chip dims are
     cleared before Qwen's layers.
   - With the circuit inside Qwen's layers, chip values would sit in the residual there. With
     the overlay they would sit in Qwen's own dims, and Qwen's in_proj/k_proj read every one of
     them, so text after a trace would change.
   - So interleaving rules out the overlay: every chip slot needs its own dim, and hidden goes
     back to about 2560 + 1700 = 4250 (the v15 layout). That alone costs more than the 12 chip
     layers (+1.7 B params of padding on Qwen's 32 layers versus 1.56 B for the chip layers).
2. **Text tokens.** The circuit computes at text positions too: capture, the print, the owner
   bit.
   - That needs the ballast, which pins every RMSNorm on a known power of 2, but Qwen's
     sublayers share those norms and need Qwen's own RMS.
   - So the text-position work can't move into Qwen's layers, and a front block of chip layers
     stays anyway.
3. **Capacity.** The full-attention layers (8 of 32) have no DeltaNet heads, and the per-layer
   head and unit increase applies to all 32 layers. The circuit's peaks (16 RAM heads in one
   sublayer, about 9k units in one MLP) would need X ≈ 2-3k extra units and about 6 extra key
   heads per layer, roughly 1.1B params by themselves.

Only a decoder-only variant (always `tc`, Qwen never sees chip tokens) avoids reason 1. It breaks
the computer in stock `generate()`, so it is rejected too.

## What's left (by payoff)

1. **More text-visible slot cuts.** At the peak the largest groups are:
   - the tool-call capture (`call.*`, 72)
   - the write-index thermometer `j>` (65)
   - ctx copies (43)
   - division (40)

   Getting to ≤ 274 slots gives hidden 2880, and ≤ ~210 gives 2816, about 0.28 GB each.
2. **The critical path** is the BMUL digit chain: ctx → digit reads → a·b → s//10 → wrap → carry
   → digit one-hot → exit, 12 layers. Each layer saved is about 0.26 GB, and it needs a macro
   redesign.
3. **Non-uniform ballast**, so hidden needn't be m·4^k. Then hidden = 2560 + slots + a few
   ballast dims, rounded to 64. The power-of-4 rule needs only Σ b_i² = hidden·4^r with b_i
   powers of 2.
