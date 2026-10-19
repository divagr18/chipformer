# RV32 P3 status

Branch `rv-p3b` (on rv-p2). The full write-up is the P3 section of `docs/rv32.md`.

## Where it is
- rv-q4p2 (Qwen3.5-4B + the P2 core + the bridge, 43 layers, 18.7 GB) works end to end. Qwen
  calls `calc`, the machine inside the weights computes exactly, and Qwen answers.
- Lockstep, the stock head and the IR replay are clean on every record checked.
- The chip layers run on the Triton executor in generation: bit-identical to the HF modules,
  13x faster (3,200 chip tokens/s on an L4).
- Batched decoding (`evalarith --batch`): results don't depend on the batch's other rows, but
  aren't bit-identical to batch-1 decoding.
- calc v2: exact rationals. v1's "a fraction only as the final value" made 610 of the 1,040
  mul_div calls fail.

## Code
- `chipformer/rv/`:
  - the bridge: `core.py`, `spec.py`, `ref.py`, `bridge.py`;
  - `merge.py`, `chat.py` (RvChat; ExecRow);
  - `rvbatch.py`, `evalarith.py`;
  - the checks: `replay.py`, `textcheck.py`, `execcheck.py`, `batchcheck.py`;
  - `calc.py`, `env/calc/calc.c`.
- Tests: `tests/test_rv_bridge.py` (9 tests, the circuit vs the reference in the IR).

## Boxes
- **q35-eval** (L4, us-east-1): the gate runs (L4 batch-1 records in `~/rvres`).
- **q35-inline** (A10G, us-east-1): needed room for the checkpoint, so `~/weights/dist/q4-v15`
  was deleted (allowed); `q4-v18` is kept.
- **q35-eval2** (A10G, us-west-2).
- Each holds `~/weights/dist/rv-q4p2`, the three bit-identical (per-tensor sha1). The A10G
  builds were rebuilt there, with a temporary swap file; their 8 anti-sink `k_proj` tensors
  were copied from the L4 build.
- Code in `~/wrv3`.
