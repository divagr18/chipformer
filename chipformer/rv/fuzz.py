"""Random RV32IM + Zba programs, and free-running checks of the circuit executor against ref.py.

    python -m chipformer.rv.fuzz --programs 128 --batch 32 [--elf c-fib c-qsort ...]

A program is a loop over a random block: every ALU, shift, compare, MUL, DIV and Zba op on random
registers, loads and stores (all widths) to a 1 KB RAM buffer and loads from a 256-byte DROM
table through masked addresses, forward branches, calls to a leaf function, putchar of a
register's low byte, and LUI/AUIPC. The
executor generates every program's trace by itself (fastexec.FreeRun: each predicted token is
the next input), batch rows in parallel, and every row must equal the reference trace up to its
HALT.
"""

import argparse
import random
import time

from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble

REGS = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "t2", "t3", "t4", "s1", "s2", "s3", "s4"]
OPS3 = ["add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu", "mul", "mulh",
        "mulhsu", "mulhu", "div", "divu", "rem", "remu", "sh1add", "sh2add", "sh3add"]
OPSI = ["addi", "andi", "ori", "xori", "slti", "sltiu"]
OPSS = ["slli", "srli", "srai"]
LOADS = [("lw", 0x3FC), ("lh", 0x3FE), ("lhu", 0x3FE), ("lb", 0x3FF), ("lbu", 0x3FF)]
STORES = [("sw", 0x3FC), ("sh", 0x3FE), ("sb", 0x3FF)]


def gen_program(seed: int, block: int = 40, iters: int = 30) -> str:
    """A random fault-free RV32IM + Zba program (s0: loop counter, s11: the RAM buffer, s10: the
    DROM table, t5: addresses)."""
    rng = random.Random(seed)
    val = lambda: rng.choice([rng.randint(-2**31, 2**31 - 1), rng.randint(-40, 40),
                              rng.randint(0, 0x3FF)])
    lines = ["_start: la s11, buf", "la s10, rtab", f"li s0, {iters}"]
    lines += [f"li {r}, {val()}" for r in REGS]
    lines.append("loop:")
    skip = 0
    for _ in range(block):
        k = rng.random()
        d, a, b = rng.choice(REGS), rng.choice(REGS), rng.choice(REGS)
        if k < 0.35:
            lines.append(f"{rng.choice(OPS3)} {d}, {a}, {b}")
        elif k < 0.50:
            lines.append(f"{rng.choice(OPSI)} {d}, {a}, {rng.randint(-2048, 2047)}")
        elif k < 0.56:
            lines.append(f"{rng.choice(OPSS)} {d}, {a}, {rng.randint(0, 31)}")
        elif k < 0.64:
            op, mask = rng.choice(LOADS)
            lines += [f"andi t5, {a}, {mask}", "add t5, t5, s11", f"{op} {d}, 0(t5)"]
        elif k < 0.70:
            op, mask = rng.choice(LOADS)
            lines += [f"andi t5, {a}, {mask & 0xFF}", "add t5, t5, s10", f"{op} {d}, 0(t5)"]
        elif k < 0.82:
            op, mask = rng.choice(STORES)
            lines += [f"andi t5, {a}, {mask}", "add t5, t5, s11", f"{op} {b}, 0(t5)"]
        elif k < 0.90:
            br = rng.choice(["beq", "bne", "blt", "bge", "bltu", "bgeu"])
            lines += [f"{br} {a}, {b}, skip{skip}", f"{rng.choice(OPS3)} {d}, {a}, {b}",
                      f"skip{skip}:"]
            skip += 1
        elif k < 0.94:
            lines += [f"mv a6, {a}", "call leaf", f"mv {d}, a6"]
        elif k < 0.97:
            lines += [f"andi a0, {a}, 127", "li a7, 1", "ecall"]
        else:
            lines.append(f"lui {d}, {rng.randint(0, 2**20 - 1)}" if rng.random() < 0.5
                         else f"auipc {d}, {rng.randint(0, 2**20 - 1)}")
    lines += ["addi s0, s0, -1", "bnez s0, loop", "li a0, 0", "li a7, 93", "ecall",
              "leaf: slli a6, a6, 3", "xori a6, a6, 0x5a5", "add a6, a6, s0", "ret",
              ".data", "buf: .space 1024",
              ".rodata", "rtab: .word " + ", ".join(str(rng.getrandbits(32)) for _ in range(64))]
    return "\n".join(lines)


def free_run(c, progs: list, batch: int, device="cuda", compile_=True, v2=False) -> dict:
    """Free-run every program (batch rows at a time); returns totals, raises on any mismatch.
    A program is an asm / ELF Program or a function returning a fresh ref.Machine."""
    from chipformer.rv import fastexec

    def fresh(prog):
        return prog() if callable(prog) else ref.Machine.from_program(prog)

    fr = fastexec.FreeRun(c, batch, device, compile_, v2=v2)
    total_instr = total_tok = 0
    t0 = time.time()
    for i in range(0, len(progs), batch):
        chunk = progs[i:i + batch]
        refs = []
        for row in range(batch):
            prog = chunk[row % len(chunk)]
            core.snapshot(c, fresh(prog))
            fr.load(row, S.tok("GO"))
            refs.append(fresh(prog).run(10**8))
        n = max(len(r) for r in refs) - 1
        got = fr.run(n)
        for row, r in enumerate(refs[:len(chunk)]):
            want = r[1:]
            bad = [t for t in range(len(want)) if got[row, t] != want[t]]
            if bad:
                t = bad[0]
                raise AssertionError(f"program {i + row}: token {t + 1} is {got[row, t]}, "
                                     f"expected {S.name(want[t])}")
            total_tok += len(r)
            total_instr += ref.trace_stats(r)["instructions"]
        if (i // batch) % 20 == 0:
            print(f"  .. {i + len(chunk)} / {len(progs)} rows, {total_instr} instructions, "
                  f"{time.time() - t0:.0f} s", flush=True)
    return {"programs": len(progs), "instructions": total_instr, "tokens": total_tok,
            "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--programs", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--elf", nargs="*", default=[])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--v2", action="store_true", help="the Triton executor (tritonexec)")
    args = ap.parse_args()
    c = core.build()
    progs = [assemble(gen_program(s, iters=args.iters)) for s in range(args.programs)]
    if progs:
        res = free_run(c, progs, args.batch, args.device, args.device != "cpu", args.v2)
        print(f"fuzz: {res}", flush=True)
    if args.elf:
        from chipformer.rv.measure import DATA

        eprogs = [elf.program(DATA / f"{n}.elf") for n in args.elf]
        res = free_run(c, eprogs, len(eprogs), args.device, args.device != "cpu", args.v2)
        print(f"elf {args.elf}: {res}", flush=True)


if __name__ == "__main__":
    main()
