"""The proposed macros (lib/macros_next.py): their reference semantics, the limits the circuit
would have to meet, and the maths library running on them (lib/mathlib_next.py) printing
exactly what the BIOS-only library prints."""

import random

import pytest

from chipformer import spec
from chipformer.q35.lib import bcd
from chipformer.q35.lib import macros_next as X
from chipformer.q35.lib import mathlib as M
from chipformer.q35.lib import mathlib_next as N

PC0 = 0x400


def machine(a: str = "", b: str = "", **reg):
    c = [0] * spec.NUM_CELLS
    for i, ch in enumerate(a):
        c[bcd.A_ADDR + i] = int(ch)
    for i, ch in enumerate(b):
        c[bcd.B_ADDR + i] = int(ch)
    c[bcd.LEN], c[bcd.LEN + 1] = len(a), len(b)
    c[spec.PC] = PC0
    for k, v in reg.items():
        c[spec.V + int(k[1:], 16) if k[0] == "v" else {"i": spec.I}[k]] = v
    return c


def apply(c, w):
    for a, v in w:
        c[a] = v


def digits(c, at, n):
    return "".join(str(c[at + j]) for j in range(n))


def test_bmov():
    c = machine("12", "345", v0=5, i=bcd.R_ADDR + 3)
    for j, d in enumerate("90817"):
        c[bcd.R_ADDR + 3 + j] = int(d)
    w = X.next_writes(c, X.encode(X.BMOV, 1))
    assert w[-3:] == [(bcd.LEN + 1, 5), (spec.V + 15, 0), (spec.PC, PC0 + 2)]
    apply(c, w)
    assert digits(c, bcd.B_ADDR, 5) == "90817" and digits(c, bcd.A_ADDR, 2) == "12"
    for v0, i, x in [(0, bcd.R_ADDR, 0), (25, bcd.R_ADDR, 0), (3, 0x5FF, 0), (3, X.PAGE_END - 2, 0),
                     (3, bcd.R_ADDR, 2)]:  # refused: only VF = 1
        c = machine("12", v0=v0, i=i)
        assert X.next_writes(c, X.encode(X.BMOV, x)) == [(spec.V + 15, 1), (spec.PC, PC0 + 2)]


def test_bsdiv():
    rng = random.Random(0)
    for _ in range(500):
        a = str(rng.randrange(1, 10 ** rng.randint(1, bcd.MAX_DIGITS)))
        d = rng.randint(2, 25)
        c = machine(a, vc=d)
        w = X.next_writes(c, X.encode(X.BSDIV, 0xC))
        assert len(w) <= 64
        apply(c, w)
        q, r = divmod(int(a), d)
        assert digits(c, bcd.A_ADDR, c[bcd.LEN]) == str(q) and c[spec.V + 15] == r, (a, d)
    for d in (0, 1, 26, 255):
        assert X.next_writes(machine("9", v3=d), X.encode(X.BSDIV, 3)) == [(spec.V + 15, 255),
                                                                         (spec.PC, PC0 + 2)]


def run_to_end(c, op, limit=400):
    """Apply the macro's records until it moves the PC; the records' write counts."""
    sizes = []
    for _ in range(limit):
        w = X.next_writes(c, op)
        sizes.append(len(w))
        apply(c, w)
        if c[spec.PC] != PC0:
            return sizes
    raise AssertionError("BSHAD did not finish")


def test_bshad_log_and_exp():
    """Y = 0 finds Briggs' counts for y in [1, 10); Y = 1 multiplies them back up: z = y within
    the truncations, and the counts match a Python reference."""
    rng = random.Random(1)
    L = M.PREC
    for _ in range(60):
        y = rng.randrange(10 ** (L - 1), 10 ** L)
        c = machine(str(y), ve=0)
        sizes = run_to_end(c, X.encode(X.BSHAD, 0xE, 0))
        assert max(sizes) <= L + 1 <= 64
        cnt, ref, v = [c[X.CNT + k] for k in range(X.LEVELS)], [], y
        for k in range(X.LEVELS):
            n = 0
            while v + v // 10 ** k < 10 ** L:
                v, n = v + v // 10 ** k, n + 1
            ref.append(n)
        assert cnt == ref and int(digits(c, bcd.A_ADDR, L)) == v, y
        # back up from 1: z = prod (1 + 10^-k)^n_k = 10 / (the y it came from) ... 10^L / y
        c2 = machine("1" + "0" * (L - 1), vd=0)
        for k in range(X.LEVELS):
            c2[X.CNT + k] = cnt[k]
        run_to_end(c2, X.encode(X.BSHAD, 0xD, 1))
        z = int(digits(c2, bcd.A_ADDR, L))
        assert all(c2[X.CNT + k] == 0 for k in range(X.LEVELS))
        assert abs(z * y - 10 ** (2 * L - 1)) <= 10 ** (2 * L - 1) * 2e-14, (y, z)


def test_bshad_repeats_without_a_pc_write():
    c = machine("1" + "0" * 16, ve=0)
    w = X.next_writes(c, X.encode(X.BSHAD, 0xE, 0))  # 1 * 2 < 10: accept, PC stays
    assert all(a != spec.PC for a, _ in w) and (X.CNT, 1) in w
    c = machine("9" * 17, ve=15)  # the last level, rejected: VX = 16 and PC moves on
    assert X.next_writes(c, X.encode(X.BSHAD, 0xE, 0)) == [(spec.V + 0xE, 16), (spec.PC, PC0 + 2)]


@pytest.mark.parametrize("op,a,b,kw", [
    ("pow", "7", "20", {}), ("fact", "25", "0", {}), ("ncr", "30", "11", {}),
    ("gcd", "832040123457", "514229876543", {}), ("lcm", "123456", "7890", {}),
    ("to_base", "987654321098", "2", {}), ("to_base", "987654321098", "7", {}),
    ("to_base", "0", "3", {}), ("from_base", [1, 0, 1, 1], "2", {}),
    ("log10", "2", "0", {}), ("ln", "123456789", "0", {}), ("log2", "1024", "0", {}),
    ("logb", "1000", "7", {}), ("exp", "31416", "0", {"frac": 4}),
    ("exp10", "5", "0", {"frac": 1, "neg": True}), ("exp", "5", "0", {"neg": True})])
def test_library_on_the_macros(op, a, b, kw):
    """mathlib_next prints what mathlib prints, in fewer tokens where a macro applies."""
    g0, n0, _ = M.run(op, a, b, **kw)
    g1, n1, _ = N.run(op, a, b, **kw)
    assert g1 == g0, (op, a, b, g0, g1)
    assert n1 <= n0
    if op in ("log10", "ln", "log2", "logb", "exp", "to_base") and a != "0":
        assert n1 < 0.7 * n0, (op, n0, n1)


def test_free_opcodes():
    """The proposed kinds are free in macros.py's 5XYN space (and 5XY0 stays SE)."""
    from chipformer.q35.lib import macros

    assert not set(X.KINDS) & set(macros.KINDS) and 0 not in X.KINDS
