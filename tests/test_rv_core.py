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
SLOW = {"c-sieve", "c-sort", "c-crc32p", "c-gmpmul"}  # 12k-44k tokens: RV_SLOW=1 runs them
# 0.13M-8M instructions: the executor runs these free-running (test_rv_fuzz.py), not the IR
FREE_RUN = {"c-fib", "c-sudoku", "c-qsort", "c-sieve4", "mpy-s1", "mpy-s5"}
EXIT = "\nli a7, 93\necall"


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
    """riscv-tests rv32ui / rv32um / rv32uzba and gcc -O2 C programs, teacher-forced over their
    run."""
    if name in FREE_RUN:
        pytest.skip("checked free-running by the executor")
    if name in SLOW and not os.environ.get("RV_SLOW"):
        pytest.skip("long trace (RV_SLOW=1)")
    prog = elf.program(DATA / f"{name}.elf")
    n = len(ref.Machine.from_program(prog).run(10**7))
    toks = run_check(circ, prog, kernel=n < 1500)
    assert S.family(toks[-1])[0] == "HALT"


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
    return "_start: " + "\n".join(lines) + EXIT


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
    lines += ["loop: addi s0, s0, -1", "bgtz s0, loop"]
    run_check(circ, assemble("\n".join(lines) + EXIT))


def test_refresh_after_63(circ):
    # 70 straight-line instructions: an R record is due before the 64th
    src = "_start: " + "\n".join(f"addi a0, a0, {i % 7}" for i in range(70)) + EXIT
    toks = run_check(circ, assemble(src))
    assert any(S.family(t)[0] == "R" for t in toks)


@pytest.mark.parametrize("skip", [5, 40, 133])
def test_snapshot_mid_run(circ, skip):
    """A snapshot taken mid-run (n > 0, live registers and RAM) continues exactly."""
    prog = program("call")
    m = ref.Machine.from_program(prog)
    for _ in range(skip):
        m.record()
    assert 0 < m.n < S.N_MAX
    core.snapshot(circ, m)
    toks = m.run()
    mat = ir.evaluate(circ, toks, check=True, kernel=True)
    pred = core.predictions(circ, mat)
    assert all(pred[i] == toks[i + 1] for i in range(len(toks) - 1))


FAULT_PROGRAMS = {
    "illegal": "_start: li a0, 7\n.word 0x0000000b",
    "illegal (csr write)": "_start: li a0, 1\ncsrw mscratch, a0",
    "ebreak": "_start: nop\nebreak",
    "misaligned load": "_start: li a0, 0x102\nlw a1, 0(a0)",
    "misaligned store": "_start: li a0, 0x101\nsh a0, 0(a0)",
    "load out of range": "_start: li a0, 0x8000\nlw a1, 0(a0)",
    "store out of range": "_start: li a0, -4\nsb a0, 0(a0)",
    "bad jump": "_start: li a0, 0x4000\njr a0",
    "bad jump (misaligned)": "_start: la a0, t\naddi a0, a0, 2\njr a0\nt: nop",
    "ecall": "_start: li a7, 5\necall",
}


@pytest.mark.parametrize("name", list(FAULT_PROGRAMS))
def test_faults(circ, name):
    """Each fault halts with FLT(cause), the PC's 8 hex digits and HALT, exactly as the reference."""
    toks = run_check(circ, assemble(FAULT_PROGRAMS[name]))
    cause = name.split(" (")[0]
    fams = [S.family(t) for t in toks]
    assert fams[-10] == ("FLT", S.FAULTS.index(cause)) and fams[-1][0] == "HALT"
    assert all(f[0] == "HEX" for f in fams[-9:-1])


def test_csr_stubs_and_putchar(circ):
    src = """_start: csrr a0, mhartid
        csrr a1, misa
        csrr a2, cycle
        csrr a3, instret
        li a7, 1
        li a0, 72
        ecall
        li a0, 105
        ecall""" + EXIT
    prog = assemble(src)
    toks = run_check(circ, prog)
    m = ref.Machine.from_program(prog)
    m.run()
    assert m.out == b"Hi" and m.x[11] == S.MISA and m.x[12] == m.x[13] == 0
    assert [S.family(t) for t in toks if S.family(t)[0] == "BYTE"] == [("BYTE", 72), ("BYTE", 105)]


@pytest.mark.parametrize("seed", range(3))
def test_mul_fuzz(circ, seed):
    rng = random.Random(100 + seed)
    regs = ["a0", "a1", "a2", "a3", "t0", "t1"]
    lines = [f"li {r}, {rng.choice([rng.randint(-2**31, 2**31 - 1), rng.randint(-9, 9)])}"
             for r in regs]
    for _ in range(60):
        op = rng.choice(["mul", "mulh", "mulhsu", "mulhu", "add", "xor"])
        lines.append(f"{op} {rng.choice(regs)}, {rng.choice(regs)}, {rng.choice(regs)}")
    run_check(circ, assemble("_start: " + "\n".join(lines) + EXIT), kernel=seed == 0)


DIV_EDGE = [0, 1, -1, 2, -2, 7, -7, 15, 16, 255, 256, 1000, 2**31 - 1, -2**31, -2**31 + 1, 0xFFFF,
            -0x10000, 10, 3]


@pytest.mark.parametrize("seed", range(4))
def test_div_fuzz(circ, seed):
    """DIV / DIVU / REM / REMU on edge values and random operands of every size (digits skipped
    or not, signs, division by 0, overflow), rd = x0 and rd == rs1 included."""
    rng = random.Random(200 + seed)
    regs = ["a0", "a1", "a2", "a3", "t0", "t1", "s1"]

    def val():
        return rng.choice([rng.choice(DIV_EDGE), rng.randint(-2**31, 2**31 - 1),
                           rng.randint(-2**(rng.randint(1, 31)), 2**rng.randint(1, 31))])

    lines = [f"li {r}, {val()}" for r in regs]
    for i in range(36):
        if i % 6 == 5:
            lines += [f"li {r}, {val()}" for r in rng.sample(regs, 3)]
        op = rng.choice(["div", "divu", "rem", "remu"])
        d = "zero" if rng.random() < 0.1 else rng.choice(regs)
        lines.append(f"{op} {d}, {rng.choice(regs)}, {rng.choice(regs)}")
    run_check(circ, assemble("_start: " + "\n".join(lines) + EXIT), kernel=seed == 0)


def test_zba_and_rodata(circ):
    """sh1add / sh2add / sh3add, and loads of every width from the DROM (.rodata)."""
    src = """_start: li a0, 0x1234567
        li a1, -77
        sh1add a2, a0, a1
        sh2add a3, a1, a0
        sh3add a4, a0, a0
        la s0, tab
        lw t0, 0(s0)
        lw t1, 4(s0)
        lb t2, 5(s0)
        lbu t3, 7(s0)
        lh t4, 6(s0)
        lhu t5, 2(s0)
        sh2add t6, a0, s0""" + EXIT + """
        .rodata
        tab: .word 0x89abcdef, 0x80ff7f01"""
    prog = assemble(src)
    run_check(circ, prog)
    m = ref.Machine.from_program(prog)
    m.run()
    assert m.x[12] == (0x1234567 * 2 - 77) & 0xFFFFFFFF
    assert (m.x[5], m.x[7], m.x[28], m.x[29], m.x[30]) == (0x89abcdef, 0x7f, 0x80, 0xFFFF80FF,
                                                           0x89ab)


def bn_ops(n=0, off=0, m=0, d=0, s=0, ovw=0):
    """A digit-file row op's operand word (rs2)."""
    return d | s << 3 | ovw << 6 | n << 7 | off << 13 | m << 20


def test_digit_file(circ):
    """BSCAN / BLD / BMAC (add, scaled, overwrite) / BMSB (shifted, with a borrow) / BPRD / BGET."""
    src = f"""_start: la a0, txt
        bscan a1, a0
        add a2, a0, a1
        slli t2, a1, 7
        ori t0, t2, {bn_ops(d=1)}
        bld zero, a2, t0
        li a2, {bn_ops(n=6, m=1, d=2, s=1)}
        bmac a3, zero, a2
        bmac a3, zero, a2
        li t0, {bn_ops(n=5, m=7, d=3, s=1, ovw=1)}
        bmac a4, zero, t0
        li t0, {bn_ops(n=5, off=2, m=3, d=2, s=1)}
        bmsb a6, zero, t0
        bprd 2, 7
        bprd 2, 6
        bprd 2, 5
        bprd 2, 0
        li t1, {bn_ops(off=3, d=3)}
        bget a5, zero, t1
        addi a0, a0, 6
        bscan s1, a0""" + EXIT + """
        .data
        txt: .byte 0x31, 0x32, 0x33, 0x34, 0x39, 0x2b, 0x37, 0x41"""
    prog = assemble(src)
    run_check(circ, prog)
    m = ref.Machine.from_program(prog)
    m.run()
    assert m.bn[1][:5] == [9, 4, 3, 2, 1] and m.out == b"0638"
    assert (m.x[11], m.x[13], m.x[14], m.x[15], m.x[16], m.x[9]) == (5, 0, 0, 6, 1, 1)


@pytest.mark.parametrize("expr", ["123456789*987654321", "-1000000007//37", "gcd(462,1071)",
                                  "(7-19)%5", "3**13", "12345678901234567890-12345678901234567891"])
def test_calc(circ, expr):
    """The calc firmware (env/calc, the digit-file ops), teacher-forced in the IR."""
    from chipformer.rv import calc

    m = calc.machine(expr)
    core.snapshot(circ, m)
    toks = calc.machine(expr).run(10**6)
    mat = ir.evaluate(circ, toks, check=True, kernel=len(toks) < 1500)
    pred = core.predictions(circ, mat)
    assert all(pred[i] == toks[i + 1] for i in range(len(toks) - 1))
    mm = calc.machine(expr)
    mm.run(10**6)
    assert mm.out.decode() == calc.expected(expr)
