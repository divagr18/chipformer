# VX: the vector unit in the circuit (design B, early layers)

This is the plan the `vx` branch builds. `chipformer/rv/vx.py` is the reference: its token costs
and semantics are the contract, and the circuit has to match it token for token.

## Why the early layers

The core's datapaths sit in layers 5–9 and are nearly full there (MLP free: L5 0.6k, L6 1.7k, L8
1.2k, L9 0.7k; heads: L3–L9 full). The free room is:

- heads: L1 11, L2 16, L10 16, L11 16
- MLP units: L1 about 8.9k, L2 7.8k, L3 8.8k, L4 4.9k, L7 5.1k, L10 7.9k, L11 8.1k
- live dims: low in L1–L4 (about 120–240) against a peak of about 1,090 at L8–L9

Exact arithmetic costs one MLP stage, and so one layer, per carry or threshold level. A multiply
takes about 5 stages and a 64-lane dot product about 6. A unit that reads its operands in layers
2–3 has 8–9 MLP stages before the exit (L11). A unit on top of the core would add one layer per
stage.

## The rule the design follows

1. **Memories:** the unit's memories are its own heads in L1–L3, so its addresses never need the
   register file (L5) or RAM (L7–8).
   - L1 attention queries can use only token features.
   - L2 queries can use L1 outputs, the `j` thermometers included, but only in linear (andpre)
     forms.
   - L3 queries can use L2 MLP outputs, adders included.
2. **Writes:** every write takes its value from token payloads (lags 0..3: up to 64 bits), its key
   from L1/L2 dims at t-1 (the record's config state and `j`), and its beta from the token.
3. **Configuration:** the unit can't see the registers or the decoded instruction. Each vector op's
   record starts with VC tokens (16-bit payloads) that the exit fills from the instruction's
   registers. L1 state heads keep them for the rest of the record.
4. **Depth:** a token does at most one dot product or one multiply chain that fits the stages
   before the exit. An op that needs more takes more tokens per element, and the intermediate goes
   through the token stream.
5. **Trackers:** a stream token is VB instead of VE when its element is the new max. The tracker's
   value is then written from that token's payloads.

## Memories

| Memory | Contents | Rows | Heads, layer |
|---|---|---|---|
| VM | 16 vector registers × 64 int32; bank A = v0–v7, bank B = v8–v15, one read port each | element pairs (64 bits), written every 4th stream token | 2, L2 |
| VA | 256 int8 activations, 4 chunks of 64 | byte pairs in pair-specific columns, read a chunk at a time (multi-hot) | 1–2, L2 |
| KC | K cache: rows (l, g, t) of 8 int8, 1024 rows | 8 bytes written from 4 tokens' payloads | 3, L2 |
| VC | V cache: rows (l, g, t) of 8 int8, laid out so a read of column j over t < 32 is one query | as KC | 3, L2 |
| WROM | int4 weights, one row-chunk (8 words, 64 cells) plus the row scale S_r per value-head row, 4,152 row-chunks | read-only (snapshot) | 17, L3 (+1 L2) |
| state | op config, trackers, RoPE cos/sin | single rows | L1 |

To free L3, the ROM shrinks to 128 KB (16 heads, L4 only). The ROM fetch timing doesn't change.

## Ops (what `vx.py` v3 models)

- **Record:** VC tokens (config), then the stream (VE/VB), then NT.
- **Stream limit:** 128 tokens (JMAX goes from 66 to 130 or more).

| Op | Does | Tokens |
|---|---|---|
| VMAT | out = acc × S_r into a VM register (k = 1: 64 lanes); argmax mode | 2 a row (k = 1), 6 a row (k = 3) |
| VMATR | scores: VA q · K rows (signed max tracked); p·V: VA p · V column j, × r0 >> 12 | 2 an output |
| VQ | a VM register's pairs → int8 pairs into VA or a cache row | 1 per 2 elements |
| VEW | ADD (window × factor, then add); ROPE (pairs, cos/sin from state); GLU (two multiplies chained) | ADD, ROPE: 2 an element; GLU: 4 an element |
| VEXP | exp-table softmax numerators, VM scores → VA | 1 an element |
| VRED | Σ VA² or Σ VA over a chunk | 1–2 |
| VGET | rd = tracker | value record |
| VPUT | VM[v][i] = register | 1 VC + 2 + NT |
| VSETR | RoPE cos/sin into state | VC + payload |

SiLU and exp are fixed-point function units in the MLP (generic activation units, not model data).
The model's weights, scales and caches stay data.

## Build order

1. `vx.py` v3 and `stories_vx.c` on it: tokens and the story.
2. Infrastructure: VE/VB/VC token features and output codes; JMAX; the ROM move; L1 state heads;
   VU decode in the exit; snapshot of the unit's state.
3. Ops one at a time, each teacher-forced against the reference on random programs (the
   `run_check` harness), starting with VPUT/VGET, then VMAT, VQ, VEW, VMATR/VEXP/VRED.
4. Whole-program checks (the IR on prefixes, the executor free-running) and the layer, hidden-size
   and checkpoint measurements.
