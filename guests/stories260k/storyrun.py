"""stories260K on the circuit's executor (GPU, batch 1): free-running, checked against the reference.

python storyrun.py ELF [--v1] [--steps N]
--v1: the torch.compile executor (fastexec) instead of the Triton one; --steps: only time the
first N steps (still checked against the reference prefix).
"""
import argparse
import time

import torch

from chipformer.rv import core, elf, fastexec, ref
from chipformer.rv import spec as S

ap = argparse.ArgumentParser()
ap.add_argument("elf")
ap.add_argument("--v1", action="store_true")
ap.add_argument("--steps", type=int, default=0)
args = ap.parse_args()

c = core.build()
prog = elf.program(args.elf)
m = ref.Machine.from_program(prog)
t = time.time()
want = m.run(10**9)
st = ref.trace_stats(want)
print(f"reference: {st['tokens']:,} tokens, {st['instructions']:,} instructions "
      f"({time.time() - t:.0f} s): {m.out.decode(errors='replace')!r}", flush=True)
n = (args.steps or len(want) - 1)
fr = fastexec.FreeRun(c, 1, "cuda", compile_=args.v1, v2=not args.v1)
core.snapshot(c, ref.Machine.from_program(prog))
fr.load(0, S.tok("GO"))
fr.capture()
torch.cuda.synchronize()
t = time.time()
got = fr.run(n)[0]
dt = time.time() - t
bad = next((i for i in range(n) if got[i] != want[i + 1]), None)
print("exact" if bad is None else f"MISMATCH at step {bad}: {got[bad]} vs {S.name(want[bad + 1])}")
out = bytes(S.family(int(x))[1] for x in got[:n] if S.family(int(x))[0] == "BYTE").decode(errors="replace")
print(f"{'v1' if args.v1 else 'v2'}: {n:,} steps in {dt:.1f} s = {n / dt:,.0f} tok/s "
      f"({dt / n * 1e3:.3f} ms/step); text so far {out!r}", flush=True)
