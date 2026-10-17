# An object store inside the model (v1)

`chipformer/q35/store.py` (reference and protocol), `store_q35.py` (circuit), `store_check.py`
(differential checks), `store_snapshot.py` (durability), `store_eval.py` (GPU evals).

Qwen gets five native tools: `put(name, text)`, `append(name, text)`, `get(name)`, `list()`,
`delete(name)`. The chip layers capture the name and the text as Qwen writes the tool call,
keep them in Gated DeltaNet state, and at `</tool_call>` answer with a few chip records and a
`<tool_response>` they print themselves. Nothing runs on the host. The checkpoint stays a stock
`Qwen3_5ForConditionalGeneration` with config-only changes, and it is exact in bf16.

## Any token in, any token out

Every vocabulary row carries two new feature groups in chip dims:

- **id bits** (18 dims, 0/1): the bits of the row's own token id. They are an input feature,
  so the circuit can read which token any text token is. Text classes only told it digits and
  a few tags.
- **id code** (18 dims, `CODE_AMP * (2 bit - 1)`): an output code. To print token X, the
  exit leaves `KAPPA * (2 bit - 1)` of X's id in these dims and 0 in the class code. Row X
  then scores 18 units, every other Qwen row at most 16 and chip rows 0. The final norm gives
  these dims 4x the class code's gain, so the margin is 8 class-code units. That is 4x the
  margin the class prints (digits, tags) already rely on.

The rows of special tokens (`<|endoftext|>` and up) get a zero id code, so the store can never
print `<|im_end|>` or `<|im_start|>`. Both groups sit in chip dims, which the exit clears before
Qwen's layers, so Qwen's own computation is unchanged. That is checked by text KL on GPU.

## Storage

All tables are Gated DeltaNet heads with one-hot keys: an exact overwrite of one row. They
never decay, and a BOOT does not clear them.

| Table | Rows | Row | Heads at 4B |
|---|---|---|---|
| content | chunk x 16 rows | 8 tokens: a key head's 2 value heads hold 4 each | 80 (640 chunks) |
| meta | chunk | length, next chunk, has next | 5 |
| dir | entry (128) | 6 name tokens, name length, alive | 1 |
| dht | entry | first chunk, last chunk (the two value heads) | 1 |
| registers | 6 cells | NEXTG, NFILES, SK, GCUR, TMP, TMPL | 1 |

Capacity at 4B: 640 chunks x 104 tokens = **66,560 tokens**. Each put / append call is one
chunk, and a file is a chain of chunks. The address space is 640 x 128 = 81,920 token slots
(CMAX = 104 of each 128 are used). The 0.8B has one value head per key head, so 4 tokens per
row: 256 chunks x 104 = 26,624 tokens in 64 heads. More heads would buy more (see Limits).

**Writing content as Qwen writes it.** A content row is written at every text token of the call
with the key (chunk, index / 8). Its value columns take the token's id bits in slot index % 4,
summed over the conv taps of lags 0..3. So the row's write at the group's last token carries
all 4 of its tokens. The two value heads write on alternate groups of 4 (write gate beta =
sigmoid(+-128), exactly 1 or 0), so one key row holds 8 tokens. Storing costs no trace tokens.

**Counting without positions.** The index of a text token is a row that halves at every value
token (`x = 2^(64 - n)`), read through thresholds. `2^-(n + 14)` must stay a normal float, so a
call stores at most CMAX = 104 tokens. A longer text keeps its first 104.

**Addresses.** Every register and table address is a set of 2-bit one-hot groups (4 dims per 2
bits). A key or query channel is the sum of its groups minus (n - 1): exactly 1 on its own row,
<= 0 elsewhere. So a 128-row x 80-head table needs no 10,000-way one-hot.

## The protocol

At a store call's `</tool_call>` (`S` turns on, and every other part of the circuit is held off
until the NL after the store's `</tool_response>`):

1. **search**: `ADDR(SK) VAL(k)` for k = 0, 1, ... until entry k is alive with the same name
   (6 tokens and length, compared bit by bit) or k = NFILES. 2 chip tokens per entry.
2. **writes**, depending on the op and the outcome:
   - put: `[NFILES+1]`, `DIRW 1`, `DIRH g`, `DIRT g`, `METAW len`, `NEXTG g+1`.
   - append to an existing file: `TMP <- tail`, `TMPL <- len(tail)`, `METAN g`, `DIRT g`,
     `METAW len`, `NEXTG g+1`.
   - delete: `DIRW 0`.
3. **print**: `\n<tool_response>\n` + body + `</tool_response>\n`. The bodies:
   - `ok`, `not found` or `empty`, each followed by `\n`.
   - get: the chain's chunks, any token per printed token. A `GCUR <- next` record between
     chunks (invisible to Qwen under tc).
   - list: each live name, ending in its newline.

Ports (`DIRW`, `DIRH`, `DIRT`, `METAW`, `METAN`): a VAL written to a port writes a whole table
row, keyed by a register. A value needed from a table (an old tail, a chunk's length) first
goes into a register through a record. A head's result can only reach later tokens through
the token stream.

Names and texts are the spans between the parameter's newline and its `</parameter>`. They keep
that last newline, so a get prints back exactly what Qwen wrote, and an append starts on a new
line.

## Durability

- **Snapshot** (`store_snapshot.snapshot / restore`): the files are the persistent heads' rows.
  A snapshot is those value heads' recurrent states, about 30 MB at 4B however much is
  stored. `restore` builds a fresh stock `DynamicCache` with only those heads filled; a new
  conversation prefilled on it starts with an empty context while the computer keeps its files.
  Needs transformers' GDN prefill with an initial state (5.x has it).
- **Disk image** (`store_snapshot.disk_image / replay`): one put (and appends) per live file.
  Prefilling those calls and the circuit's answers at a session start rebuilds the store.

## Costs

| | 4B (16 key / 32 value heads, MLP 9216) | 0.8B (16 / 16, MLP 3584) |
|---|---|---|
| chip layers | 16 (unchanged) | 16 (unchanged) |
| heads | 104 -> 204 | 104 -> 186 |
| IR dims | 3623 -> 4427 | 3623 -> 4431 |
| residual slots | 1561 -> 2009 | 1889 -> 2698 |
| hidden | 4224 -> 4608 | 3072 -> 3840 |

Without the store (`build(store=None)`) the circuit is unchanged, dim for dim.

## Tests (`tests/test_q35_store.py`)

- **Ids:** the circuit's constants match Qwen3.5's tokenizer, and `StoreMachine.call` spells a
  call exactly as the template does.
- **Reference vs oracle:** random calls on the reference agree with a plain filesystem.
- **Circuit vs reference, exact in the IR** (every bf16 rounding checked):
  - all five ops;
  - random calls with random tokens (overwrites, append chains, deletes, names equal in their
    first 6 tokens, empty and over-long texts);
  - next to arithmetic tool calls, with stored text containing "a × b =";
  - 100k+ tokens of other text between calls;
  - restore from a snapshot of the persistent heads;
  - a replayed disk image.

## Limits (v1)

- **Text per call:** 104 tokens. Longer texts need appends (each a new chunk).
- **Names:** compared on their first 6 tokens and their length (with the trailing newline).
- **Capacity:** 127 directory entries and 640 chunks. Chunk ids are 10 bits and are not
  reused, so after 640 put / append calls new text is no longer stored (the answers still say
  ok). More capacity costs heads (80 per 66K tokens at 4B) or a second key-row layout.
- **Content:** a text that contains `</` followed by `parameter` ends early (that is the
  template's closer).
- **Parameter order:** the name and text are recognised by their parameter names, in either
  order.
- **Search:** linear, 2 chip tokens per directory entry.
