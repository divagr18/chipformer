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
    ap.add_argument("--batches", default="1,8,32", help="executor batch sizes (P1.6)")
    ap.add_argument("--sparse", default="0", help="executors: 0 dense blocks, 1 sparse, "
                                                   "2 the Triton executor v2 (e.g. 0,1,2)")
    ap.add_argument("--hf-compile", action="store_true",
                    help="also time rv-tiny's HF modules: static cache + torch.compile")
    args = ap.parse_args()
    c = core.build()
    prog = elf.program(DATA / f"{args.prog}.elf")
    core.snapshot(c, ref.Machine.from_program(prog))
    toks = ref.Machine.from_program(prog).run(10**6)
    n_instr = ref.trace_stats(toks)["instructions"]
    tpi = len(toks) / n_instr
    m = next(m for m in range(3, 16) if args.hidden % m == 0 and _is_pow4(args.hidden // m))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    batches = [int(b) for b in args.batches.split(",")]
    if args.hf_compile:
        hf_compiled(c, prog, toks, steps_of(args, toks), tpi)
        hf_compiled(c, prog, toks, steps_of(args, toks), tpi, compact=True)
    if args.executor_only:
        for sp in args.sparse.split(","):
            if sp == "2":
                run_executor2(c, toks, steps_of(args, toks), tpi, batches)
            else:
                run_executor(c, toks, steps_of(args, toks), tpi, dev, batches, sp == "1")
        return None
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
    run_executor(c, toks, steps, tpi, dev, batches)


def hf_compiled(c, prog, toks, steps, tpi, compact=False):
    """rv-tiny's stock HF modules decoding free-running: a StaticCache (snapshot injected) and
    torch.compile(mode="reduce-overhead"), i.e. CUDA graphs. Checks the generated tokens."""
    from transformers import StaticCache

    from chipformer.rv import spec as S
    from chipformer.rv import tiny

    model, _ = tiny.build(c)
    model = model.cuda()
    if compact:  # every linear map (and the tied embedding / output head) reads its nonzero block
        fast.compact(model, qwen=True, n_chip=c.n_layers())
    cache = StaticCache(config=model.config, max_cache_len=steps + 64)
    with torch.no_grad():
        model(input_ids=torch.tensor([[S.VOCAB]], device="cuda"), past_key_values=cache,
              use_cache=True, cache_position=torch.tensor([0], device="cuda"))
        core.snapshot(c, ref.Machine.from_program(prog))
        hf.inject(c, cache)
        fwd = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
        tok = torch.tensor([[toks[0]]], device="cuda")
        got, t0 = [], None
        for t in range(steps):
            if t == 20:
                torch.cuda.synchronize()
                t0 = time.time()
            out = fwd(input_ids=tok, past_key_values=cache, use_cache=True,
                      cache_position=torch.tensor([t + 1], device="cuda"))
            tok = out.logits[:, -1:].argmax(-1)
            got.append(tok)
        torch.cuda.synchronize()
        s = (time.time() - t0) / (steps - 20)
    got = torch.cat(got, 1)[0].tolist()
    ok = got == toks[1:steps + 1]
    print(f"HF (rv-tiny, hidden {model.layout.hidden}, StaticCache + torch.compile "
          f"reduce-overhead{', fast.compact' if compact else ''}) {1e3 * s:7.3f} ms/token  {1 / s:8.0f} tok/s  {1 / s / tpi:8.0f} "
          f"instr/s  free-running == reference: {ok}", flush=True)
    del model, cache
    torch.cuda.empty_cache()


def steps_of(args, toks):
    return min(args.steps, len(toks) - 25)


def run_executor(c, toks, steps, tpi, dev, batches=(1,), sparse=False):
    from chipformer.rv import fastexec

    kind = "sparse" if sparse else "dense blocks"
    n = fastexec.check(c, toks[:300], dev, sparse=sparse)
    print(f"executor ({kind}) == IR on {n} positions ({dev})", flush=True)
    for b in batches:
        s = fastexec.timed(c, toks, steps, graph=True, compile_=True, batch=b, sparse=sparse)
        print(f"executor {kind} batch {b:3d} (compiled + CUDA graph) {1e3 * s:7.3f} ms/step  "
              f"{b / s:9.0f} tok/s  {b / s / tpi:9.0f} instr/s", flush=True)


def run_executor2(c, toks, steps, tpi, batches=(1,)):
    from chipformer.rv import fastexec, tritonexec

    n = tritonexec.check(c, toks[:200])
    print(f"executor v2 (triton) == IR on {n} positions", flush=True)
    for b in batches:
        s = fastexec.timed2(c, toks, steps, batch=b)
        print(f"executor v2 (triton) batch {b:3d} (CUDA graph) {1e3 * s:7.3f} ms/step  "
              f"{b / s:9.0f} tok/s  {b / s / tpi:9.0f} instr/s", flush=True)


def _is_pow4(x):
    while x > 1 and x % 4 == 0:
        x //= 4
    return x == 1


if __name__ == "__main__":
    main()
