# RV32 P3 status (deferred until P2 lands)

Branch `rv-p3`, from rv-p1 `de634e7`. Code: `chipformer/rv/{bridge,merge,chat,evalarith,replay,textcheck}.py`,
the bridge in `core.build(text=...)`, `env/progs/calc.c`, `tests/test_rv_bridge.py`. Design notes are in
`docs/rv32.md` (P3 section).

## Built and verified
- **The bridge in the circuit:** tool-call text capture into a text buffer read as memory at 0x8000; GO at
  `</tool_call>`; putchar echoed as Qwen tokens (bytes plus `<tool_response>`, `</tool_response>`,
  `<think>`, `</think>`); owner, tau and mu; the HALT echo. Still 11 chip layers, 73 heads.
- **IR tests (`tests/test_rv_bridge.py`, 6 tests):** pass. The circuit equals the reference on owner bits and
  every owned token, including bignum calls, prefilled history and two calls in one stream. The rv P1 suites
  (`test_rv_ref`, `test_rv_core`) still pass.
- **Merged Qwen3.5-4B (layout rv1):** a text-only `Qwen3_5ForCausalLM`, hidden 3328 (no overlay needed), 15 GB.
- **End to end on GPU:** Qwen calls `calc`, the machine inside the weights computes
  123456789 × 987654321 = 121932631112635269, and Qwen answers it. The lockstep check against the reference
  machine was clean, and the stock head's argmax agreed with the circuit at every owned position.
- **Arith n=40** (evals' seeded set, exact scoring), on an L4 box AWS reclaimed afterwards, so its result
  files are gone:

  | Run | Correct | Notes |
  |---|---|---|
  | rv1 | **38/40** | 54 calc calls, all exact; 0 lockstep errors; 0 head mismatches; median about 3.5k chip tokens per call, about 70 s per problem on an L4 |
  | base Qwen + host-executed calc tool (same response format) | 39/40 | |

  Both rv1 misses are Qwen's own: one re-calls in a loop, the other re-subtracts a result the computer
  already returned. The base-without-tool run was not finished.
- **Bug found and fixed:** the select dims outlived the exit and derailed Qwen's norms after a run. The
  merged circuit now keeps only code, tau and mu after the exit.

## Not done / next
- **0.8B does not fit this core:** it has 1 value head per key head and 3584 MLP units, while the core
  needs 2 value heads per key head and up to 6198 units in one sublayer. It would need a core variant;
  I'd defer it to after P2.
- **Not run yet:**
  - the text-fidelity check (`rv/textcheck.py`, KL vs base against the sdpa/eager noise floor)
  - the offline IR replay of eval records (`rv/replay.py`)
  - the base-without-tool control
- Rebase onto rv-p2 (DIV, Zba, 256 KB ROM, custom bignum uops); then run DeepMind sets via the
  `evalarith`-style harness.
- **Speed:** about 50-60 chip tokens/s on the eager HF chip-only path. The compact executor inside
  generation isn't wired in.
- **Capture window:** CAP_MAX = 80 tokens. The thermometer weights are 8·2^m, which overflows bf16 at
  96 after the emitter's 2^36 fold; a wider window needs a two-level counter.

## Box state
- **q35-batch:** nothing of mine is running; GPU idle. My files:
  - `~/wrv3` (code)
  - `~/weights/dist/rv-q4` (the merged checkpoint, 15 GB; the disk is at 97%, delete it if space is needed)
  - `~/rvres` (`chat.log`, plus a partial `arith.rv1.jsonl`)
- **q35-s1 / q35-s2:** terminated by AWS (spot).
