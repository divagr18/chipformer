"""P0.2 on a GPU: the RV32 core in stock HF modules with the fla kernels (or the torch fallbacks),
bf16 and fp32, chunked prefill and token-by-token decode, against the IR evaluator.

    python -m chipformer.rv.gpucheck [--dtype bf16|fp32] [--progs signext,call,...] [--decode]

Each program's ROM / RAM / registers / PC are injected into the HF cache (hf.inject), so this
also covers the byte-column 2-hot ROM reads, the sibling value heads' write gates, and the lag-1
key commits through prefill-then-decode. Prints one line per (program, mode) and exits 1 on any
mismatch.
"""

import argparse
import sys
import time

import torch

from chipformer.rv import core, elf, hf, ref
from chipformer.rv.measure import DATA
from chipformer.rv.programs import ALL, program


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--progs", default="signext,call,memcpy,rv32ui-lb,rv32ui-sh,rv32ui-jalr,"
                                       "rv32ui-sra,c-crc32,c-strops")
    ap.add_argument("--decode", action="store_true", help="also decode token by token")
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    try:
        import fla  # noqa: F401

        kernels = "fla"
    except ImportError:
        kernels = "torch fallbacks"
    print(f"device {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}, "
          f"{args.dtype}, kernels: {kernels}", flush=True)
    c = core.build()
    bad = 0
    for name in args.progs.split(","):
        prog = program(name) if name in ALL else elf.program(DATA / f"{name}.elf")
        core.snapshot(c, ref.Machine.from_program(prog))
        toks = ref.Machine.from_program(prog).run(10**6)
        model, lay, ids = hf.build_model(c, toks, dtype)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        for mode in ["prefill"] + (["decode"] if args.decode else []):
            t = time.time()
            if args.dtype == "fp32":
                n, nd, wrong = hf.check(c, model, lay, ids, toks, mode, tokens=True)
                bad += wrong > 0
                print(f"{name:14s} {mode:7s} {n} positions: {nd} dims not bit-exact, "
                      f"{wrong} wrong tokens ({time.time() - t:.1f} s)", flush=True)
                continue
            try:
                n = hf.check(c, model, lay, ids, toks, mode)
                print(f"{name:14s} {mode:7s} exact on {n} positions ({time.time() - t:.1f} s)",
                      flush=True)
            except AssertionError as e:
                bad += 1
                print(f"{name:14s} {mode:7s} MISMATCH {str(e)[:300]}", flush=True)
        del model
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
