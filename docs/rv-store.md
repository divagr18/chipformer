# In-weights object store v2 on the RV32 Qwenputer (status)

Branch `rv-store` (from `rv-p3b` 8488330). Plan: `lets-make-our-own-effervescent-chipmunk.md`.

Code: `chipformer/rv/store.py` (reference, protocol), `chipformer/rv/store_rv.py` (circuit),
`chipformer/rv/core.py` (`build(text=T, store=Config())`), tests `tests/test_rv_store.py`.

## Status

| Phase | State |
|---|---|
| F0 port + F1 hashed directory (IR) | done, gate passed (below) |
| F2 capacity and fit (merge, executor, batch) | not started (waits for go-ahead: GPU) |
| F3 persistence (`save_store` / `load_store`) | IR round trip tested; chat / rvbatch not started |
| F4 merge and GPU checks | not started |
| F5 benchmarks, F6 write-up | not started |

## What it is

Five tools next to `calc`: `put(name, text)`, `append(name, text)`, `get(name)`, `list()`,
`delete(name)`. The chip layers capture the name and text while Qwen writes the call, keep them
in Gated DeltaNet heads, and answer at `</tool_call>` with a few chip records and a
`<tool_response>` they print themselves. A `calc` call (any other function) still starts the
RV32 machine.

The answer ends like calc's: `\n<tool_response>\n` body `</tool_response>\n<think>\n\n</think>\n\n`
(each `\n\n` printed as two `\n`); Qwen predicts again after the last `\n`.

## Design as built

**Dispatch.** The op is the token after `function` `=` and before `>` (`=get` / `=list` are
single tokens), so `get_weather` is not `get`. A store call's `</tool_call>` turns `S` on, which
gates the core's GO case and owner term; `calc` calls are unchanged.

**Directory (content-addressed).** A name's key is its first 6 tokens (18 id bits each) and its
length (3 bits). The hash is a fixed random GF(2) map of those 111 bits to 10 bits (parity units,
one MLP sublayer). Collisions probe the next slot (linear probing). A delete leaves a tombstone
(used, not alive) that probes walk past; a re-put of the same name revives it. `list` walks an
order table (slot per name, in order of first creation).

**Records** (one chip token each, `spec.STORE_FAMILIES`, 10-bit payloads):

| Record | Meaning | Writes |
|---|---|---|
| `SP(s)` | probe slot s (the hash, then s + 1, ...) | (reads dir[s]) |
| `SN(g)` | put / append that creates or replaces | dir[s] = (name, alive, head g, tail g), len[g], NEXTG = g + 1 |
| `SA(g)` | append to a live file | next[tail] = g, tail = g, len[g], NEXTG = g + 1 |
| `SO(n)` | the slot is new | order[n] = s, NORD = n + 1 |
| `SD` | delete | dir[s] becomes a tombstone |
| `SG(c)` | get: chunk c is printed next | GCUR = c |
| `SL(i)` | list: order entry i | LCUR = i |

The slot s of SN / SA / SD / SO is the last SP's: those heads take it through a conv tap (lag 1
or 2) of the SP payload's group features, so no register is read for it.

**Tables** (one-hot keys, exact overwrites, no decay; at the 4B size):

| Table | Rows | Key heads |
|---|---|---|
| content | 512 chunks x 8 rows of 14 tokens (2 value heads x 7) | 32 |
| dir | 1024 slots: name, length, alive, used, head \| tail | 8 |
| next | 512 chunks: next chunk, has next | 4 |
| order | 512 entries (value head = index bit 0) | 2 |
| len | 512 chunks (value head = chunk bit 0) | 2 |
| regs | NEXTG, NORD, GCUR, LCUR | 1 |

Capacity: 512 chunks x 104 tokens = **53,248 tokens**, 512 names (ever created), 1024 slots
(at most 50% load). A put / append is one chunk; a file is a chain of chunks.

**Content rows.** A staging head holds the current 7-token value-head row so far (one row per
slot, the slot-0 write carries the ballast and clears the rest); every text token rewrites its
content row with it, so the row's last write holds all 7 tokens. A get's print reads chunk GCUR,
row H // 14, and the output gates pick value head (H // 7) % 2 and token H % 7 into 18 dims.

**Any token in, any token out.** Every Qwen row carries its 18 id bits (input) and an id code
`128 * (2 bit - 1)` (output). To print token X the exit leaves the class code at 0 and
`KAPPA * (2 bit - 1)` of X's id in the id dims (as v1). Special tokens (`<|endoftext|>` and up)
are stored as token 0.

## Changes from the plan

- **Records are single RV-style tokens** (7 store families after the controls) instead of v1's
  ADDR / VAL pairs: one token per record.
- **Hash:** a seeded random GF(2) linear map instead of a plain XOR fold (same parity units, better
  spread for names that differ in few bits). Tests use `hash_bits` 0 to 3 to force collisions.
- **Tombstones** are revived only by the same name; a different name never takes a dead slot. So
  at most 512 names can ever be created (deleted ones count), and `list` keeps a re-created name
  in its first place.
- **Content packing: 14 tokens per key row (7 per value head) instead of v1's 8**, through a
  staging head. That takes 32 content heads instead of 52 for 512 chunks. With v1's packing the
  build had 15 chip layers; with this one, 12.
- **len and next are separate tables** (v1 had one meta table), so an append writes its chunk's
  length and the old tail's link at one token without reading the old length first.
- **`full`** is answered when the chunks or the names run out (v1 answered `ok` and dropped the
  text).
- **Name dispatch** uses token detectors on the row features (the id bits are there anyway), not
  byte features.
- The any-print id code is computed in the exit sublayer itself (no extra sublayers).
- `ir.kernel_check` exempts values below a head's `tiny` bound, as `evaluate` already did (the
  core's own `cap` counter underflows there on long calls).

## Costs (IR, Qwen3.5-4B target)

| | core + bridge | + store |
|---|---|---|
| chip layers | 11 | 12 |
| key heads | 103 | 164 |
| IR dims | 2918 | 3753 |
| residual slots | 1265 | 1806 |
| FFN units | 42,114 | 47,532 |
| hidden (`pick_hidden`) | 3968 | 4608 |
| checkpoint (estimate) | 18.7 GB | about 22 GB |

The store's live dims at the core's peak (sublayers 12 to 18) are 454; the name buffer (108) and
the staging head (126) are most of it. Hidden 4352 needs at most 1775 slots. Placing the
late-read FFNs just before their readers was tried: the peak fell to 1530 live dims, but the
slots went up (1822 to 2258, and 13 layers when they were forced into sublayer 18, whose MLP is
nearly full, so the allocator had no room for handover units). Reverted.

Chip tokens per op (reference, 4B size, random names; mean at 256 / 448 names in 1024 slots):

| op | chip tokens |
|---|---|
| put, new name (SP... SN SO) | 3.15 / 3.36 |
| put, existing (SP... SN) | 2.15 / 2.5 |
| append, existing (SP... SA) | 2.15 / 2.2 |
| get, found, 1-2 chunks (SP... SG...) | 2.15 / 2.25 |
| get, not found | 1.38 / 1.73 |
| delete, found (SP... SD) | 2.05 / 2.8 |
| list | 1 per name ever created |

v1 needed 2 per directory entry up to the match for every op.

## Tests (`tests/test_rv_store.py`, IR, every bf16 rounding checked)

- Token ids match the tokenizer; `call_tokens` spells calls as the chat template does.
- Reference vs a plain dict: 1000 random ops at the 4B size, plus 400 with a 2-bit hash, 200 with
  a store that fills up, and 300 with every name colliding.
- Circuit vs reference, token for token and owner at every position:
  - all ops, empty texts, a revived tombstone;
  - calc calls next to store calls;
  - random calls (tiny hashes, overwrites, append chains, deletes and re-puts, names equal in
    their first 6 tokens, empty and over-long texts, any token, full stores, calc in between);
  - the full 4B size with the HF kernels' head reads replayed (`kernel=True`);
  - thousands of text tokens between calls;
  - a fresh session whose store heads start from session 1's state (the snapshot).

- Extended fuzz (not in the suite): 2 x 150 random ops with every name colliding (`hash_bits` 0,
  16 slots) and a 3-bit hash (64 slots), calc calls in between: 39k tokens, 26,249 owned
  positions, 0 mismatches.
- Existing suites on this branch: test_rv_bridge 9, test_rv_ref 130, test_rv_fuzz 4,
  test_rv_core 94, test_q35_ir 3, test_q35_ir_vheads 1, all pass. test_rv_hf crashes with a
  Windows access violation in torch 2.14.1+cpu here, in the untouched rv-p3b worktree too; it
  is run on the GPU box in F2.

## Found along the way

- `calc-bridge.elf` (rv-p3b) answered wrongly when calls followed each other (`12*34` then `7*6`
  gave `2`): digits left in the bignum registers. Fixed on rv-p3b (456c947) and cherry-picked
  here (05ad59b); test_circuit_with_calc now checks all three answers.
- `fastexec.Generator` iterated `OUT_NAMES` with `S.tok("Q", 0)` (a KeyError since P3 added
  `Q`); it now skips `Q`.
