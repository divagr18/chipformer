"""The RV32 core circuit (chipformer/rv/core.py) against the reference trace, teacher-forced."""

import json
import os
import random
from pathlib import Path

import pytest

from chipformer.q35 import ir
from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble
from chipformer.rv.programs import ALL, program

DATA = Path(__file__).parent / "data" / "rv"
ELFS = sorted(json.loads((DATA / "qemu.json").read_text()))
SLOW = {"c-sieve", "c-sort"}  # 52k / 14k tokens: set RV_SLOW=1 to run them


@pytest.fixture(scope="module")
def circ():
    return core.build()


def run_check(c, prog, kernel=True):
    core.snapshot(c, ref.Machine.from_program(prog))
    toks = ref.Machine.from_program(prog).run()
    mat = ir.evaluate(c, toks, check=True, kernel=kernel)
    pred = core.predictions(c, mat)
    bad = [(i, S.name(toks[i + 1]), pred[i]) for i in range(len(toks) - 1) if pred[i] != toks[i + 1]]
    assert not bad, bad[:5]
    return toks


@pytest.mark.parametrize("name", list(ALL))
def test_programs(circ, name):
    run_check(circ, program(name))


@pytest.mark.parametrize("name", ELFS)
def test_compiled(circ, name):
    """riscv-tests rv32ui and gcc -O2 C programs, teacher-forced over their whole run."""
    if name in SLOW and not os.environ.get("RV_SLOW"):
        pytest.skip("long trace (RV_SLOW=1)")
    toks = run_check(circ, elf.program(DATA / f"{name}.elf"), kernel=len(toks_len(name)) < 1500)
    assert S.family(toks[-1])[0] == "HALT"


def toks_len(name):
    return ref.Machine.from_program(elf.program(DATA / f"{name}.elf")).run(10**7)


def test_depth_and_caps(circ):
    st = circ.stats()
    assert st["layers"] <= 12, st
    assert st["max_heads"] <= circ.target.key_heads
    assert st["max_units"] <= circ.target.intermediate


def random_alu_program(rng: random.Random, n: int) -> str:
    regs = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "t2", "s1"]
    lines = [f"li {r}, {rng.randint(-2**31, 2**31 - 1)}" for r in regs]
    ops3 = ["add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu"]
    opsi = ["addi", "andi", "ori", "xori", "slti", "sltiu"]
    opss = ["slli", "srli", "srai"]
    for _ in range(n):
        k = rng.random()
        d, a, b = rng.choice(regs), rng.choice(regs), rng.choice(regs)
        if k < 0.5:
            lines.append(f"{rng.choice(ops3)} {d}, {a}, {b}")
        elif k < 0.8:
            lines.append(f"{rng.choice(opsi)} {d}, {a}, {rng.randint(-2048, 2047)}")
        elif k < 0.9:
            lines.append(f"{rng.choice(opss)} {d}, {a}, {rng.randint(0, 31)}")
        else:
            lines.append(f"lui {d}, {rng.randint(0, 2**20 - 1)}")
    lines.append("ecall")
    return "_start: " + "\n".join(lines)


@pytest.mark.parametrize("seed", range(4))
def test_alu_fuzz(circ, seed):
    rng = random.Random(seed)
    run_check(circ, assemble(random_alu_program(rng, 120)), kernel=seed == 0)


def test_branch_fuzz(circ):
    rng = random.Random(7)
    lines = ["_start: li s0, 40"]
    for i in range(12):
        a, b = rng.randint(-5, 5), rng.randint(-5, 5)
        br = rng.choice(["beq", "bne", "blt", "bge", "bltu", "bgeu"])
        lines += [f"li a0, {a}", f"li a1, {b}", f"{br} a0, a1, skip{i}", "addi s1, s1, 1",
                  f"skip{i}: addi s0, s0, -1"]
    lines += ["loop: addi s0, s0, -1", "bgtz s0, loop", "ecall"]
    run_check(circ, assemble("\n".join(lines)))


def test_refresh_after_63(circ):
    # 70 straight-line instructions: an R record is due before the 64th
    src = "_start: " + "\n".join(f"addi a0, a0, {i % 7}" for i in range(70)) + "\necall"
    toks = run_check(circ, assemble(src))
    assert any(S.family(t)[0] == "R" for t in toks)
