"""P0.5 timing: per-token decode cost of the RV32 core's chip layers at Qwen3.5-4B width.

    python -m chipformer.rv.timing [--hidden 3072] [--steps 300]

Builds the chip layers (8 here) at the given hidden size with 4B head shapes, then times batch-1
decode steps over a real trace (the snapshot injected): the dense HF modules, then the same
model after fast.compact (each linear map reads only its nonzero block, exact). Reports ms per
token, tokens/s and instructions/s (at the measured tokens per instruction).
"""

import argparse
import time

import torch

from chipformer.q35 import fast
from chipformer.q35.emit import Layout
from chipformer.rv import core, elf, hf, ref
from chipformer.rv.measure import DATA


def timed_decode(model, c, ids, toks, steps):
    dev = next(model.parameters()).device
    x = torch.tensor([[ids[t] for t in toks]], device=dev)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([[ids[None]]], device=dev), use_cache=True)
        cache = out.past_key_values
        hf.inject(c, cache)
        for t in range(20):  # warm-up
            model(input_ids=x[:, t:t + 1], past_key_values=cache, use_cache=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for t in range(20, 20 + steps):
            model(input_ids=x[:, t:t + 1], past_key_values=cache, use_cache=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t0) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=3072)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--prog", default="c-crc32")
    ap.add_argument("--executor-only", action="store_true")
    args = ap.parse_args()
    c = core.build()
    prog = elf.program(DATA / f"{args.prog}.elf")
    core.snapshot(c, ref.Machine.from_program(prog))
    toks = ref.Machine.from_program(prog).run(10**6)
    n_instr = ref.trace_stats(toks)["instructions"]
    tpi = len(toks) / n_instr
    m = next(m for m in range(3, 16) if args.hidden % m == 0 and _is_pow4(args.hidden // m))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.executor_only:
        return run_executor(c, toks, steps_of(args, toks), tpi, dev)
    import chipformer.rv.hf as H

    orig = H.pick_layout
    H.pick_layout = lambda n: Layout(hidden=args.hidden, m=m)
    try:
        model, _, ids = hf.build_model(c, toks)
    finally:
        H.pick_layout = orig
    model = model.to(dev)
    steps = steps_of(args, toks)
    dense = timed_decode(model, c, ids, toks, steps)
    counts = fast.compact(model, qwen=False, n_chip=c.n_layers())
    comp = timed_decode(model, c, ids, toks, steps)
    for name, s in (("dense", dense), ("compact", comp)):
        print(f"{name:8s} {1e3 * s:7.3f} ms/token  {1 / s:8.0f} tok/s  "
              f"{1 / s / tpi:8.0f} instr/s  (hidden {args.hidden}, {c.n_layers()} layers, "
              f"{tpi:.2f} tok/instr)", flush=True)
    print("compact weight counts", counts)
    del model
    torch.cuda.empty_cache()
    run_executor(c, toks, steps, tpi, dev)


def steps_of(args, toks):
    return min(args.steps, len(toks) - 25)


def run_executor(c, toks, steps, tpi, dev):
    from chipformer.rv import fastexec

    n = fastexec.check(c, toks[:300], dev)
    print(f"executor == IR on {n} positions ({dev})", flush=True)
    for graph in (False, True):
        s = fastexec.timed(c, toks, steps, graph=graph)
        print(f"executor{' (CUDA graph)' if graph else ' (eager)':15s} {1e3 * s:7.3f} ms/token  "
              f"{1 / s:8.0f} tok/s  {1 / s / tpi:8.0f} instr/s", flush=True)


def _is_pow4(x):
    while x > 1 and x % 4 == 0:
        x //= 4
    return x == 1


if __name__ == "__main__":
    main()
