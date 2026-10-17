"""The BIOS maths library (lib/mathlib.py) on the reference: every routine against Python's
exact arithmetic (integers, fractions, Decimal at 80 digits), the print formats, and a trace
budget per routine."""

import math
import random
from decimal import ROUND_FLOOR, Decimal, getcontext

import pytest

from chipformer import spec
from chipformer.q35.lib import bcd
from chipformer.q35.lib import mathlib as m

getcontext().prec = 80
DIG = "0123456789abcdef"
# trace tokens from the CALL record to the print, per call (the worst of the random runs
# below, plus headroom): what a call costs in the merged model's context
BUDGET = {"pow": 8_000, "fact": 30_000, "npr": 30_000, "ncr": 60_000, "idiv": 2_500,
          "mod": 2_500, "gcd": 50_000, "lcm": 55_000, "spf": 420_000, "to_base": 28_000,
          "from_base": 30_000, "isqrt": 10_000, "sqrt": 20_000, "log10": 40_000, "ln": 42_000,
          "log2": 80_000, "logb": 80_000, "exp10": 60_000, "exp": 60_000}


def trunc(x: Decimal, dec=bcd.DEC) -> str:
    q = (x * 10 ** dec).to_integral_value(rounding=ROUND_FLOOR)
    t = str(int(q)).rjust(dec + 1, "0")
    return t[:-dec] + "." + t[-dec:] + "..."


def sig11(x: Decimal) -> str:
    e = x.adjusted()
    digs = str(int(x.scaleb(10 - e).to_integral_value(rounding=ROUND_FLOOR)))
    if e >= 11:
        return digs[0] + "." + digs[1:] + "e" + str(e)
    if e >= 0:
        return digs[:e + 1] + "." + digs[e + 1:] + "..." if e < 10 else digs + "..."
    return "0." + "0" * (-e - 1) + digs + "..." if e >= -21 else "0.0000000000..."


def spf(n: int) -> str:
    if n < 2:
        return str(n)
    for d in range(2, 1000):
        if d * d > n:
            return str(n)
        if n % d == 0:
            return str(d)
    return str(n) if n < 10 ** 6 else "..."


def base(n: int, k: int) -> str:
    out = ""
    while True:
        out = DIG[n % k] + out
        n //= k
        if not n:
            return out


def want(op, a, b, frac=0, neg=False):
    """The exact print, or None where "..." (too big for the macros) is also fine."""
    if op == "from_base":
        return str(int("".join(DIG[d] for d in a), int(b)))
    x, y = int(a), int(b)
    if op == "pow":
        return str(x ** y)
    if op == "fact":
        return str(math.factorial(x)) if x <= 39 else "..."
    if op in ("npr", "ncr"):
        return str((math.perm if op == "npr" else math.comb)(x, y)) if y <= x else "0"
    if op in ("idiv", "mod"):
        return str(x // y if op == "idiv" else x % y)
    if op in ("gcd", "lcm"):
        return str((math.gcd if op == "gcd" else math.lcm)(x, y))
    if op == "spf":
        return spf(x)
    if op == "to_base":
        return base(x, y)
    if op in ("isqrt", "sqrt"):
        r = math.isqrt(x)
        return str(r) if op == "isqrt" or r * r == x else trunc(Decimal(x).sqrt())
    if op in ("log10", "ln", "log2", "logb"):
        bb = {"log10": 10, "log2": 2, "logb": y}.get(op)
        v = Decimal(x).ln() / (Decimal(bb).ln() if bb else 1)
        n = round(v)
        if bb and bb ** n == x or op == "ln" and x == 1:
            return str(n)
        return trunc(v)
    if op in ("exp10", "exp"):
        t = Decimal(x).scaleb(-frac) * (-1 if neg else 1)
        if op == "exp10" and t == t.to_integral_value():
            c = int(t)
            return "1" + "0" * c if c >= 0 else "0." + "0" * (-c - 1) + "1"
        if t == 0:
            return "1"
        return sig11(Decimal(10) ** t if op == "exp10" else t.exp())


def num(rng, d):
    return str(rng.randrange(10 ** (d - 1) if d > 1 else 0, 10 ** d))


def cases(op, n, seed):
    rng = random.Random(f"{op}-{seed}")
    for _ in range(n):
        kw = {}
        if op == "pow":
            a, b = num(rng, rng.randint(1, 4)), str(rng.randint(0, 30))
        elif op == "fact":
            a, b = str(rng.randint(0, 41)), "0"
        elif op in ("npr", "ncr"):
            k = rng.randint(0, 50)
            a, b = str(k), str(rng.randint(0, k + 2))
        elif op in ("idiv", "mod"):
            a, b = num(rng, rng.randint(1, 24)), num(rng, rng.randint(1, 14))
            b = b if b != "0" else "3"
        elif op in ("gcd", "lcm"):
            g = rng.randrange(1, 10 ** rng.randint(1, 6))
            a, b = (str(g * rng.randrange(1, 10 ** rng.randint(1, 8))) for _ in "ab")
        elif op == "spf":
            a, b = str(rng.choice([rng.randrange(0, 10 ** 4), rng.randrange(0, 10 ** 6),
                                   rng.randrange(10 ** 6, 10 ** 12)])), "0"
        elif op == "to_base":
            a, b = num(rng, rng.randint(1, 18)), str(rng.randint(2, 16))
        elif op == "from_base":
            k = rng.randint(2, 16)
            a = [rng.randrange(1, k)] + [rng.randrange(k) for _ in range(rng.randint(0, 15))]
            b = str(k)
        elif op in ("isqrt", "sqrt"):
            v = int(num(rng, rng.randint(1, 24)))
            if rng.random() < .25:
                v = rng.randrange(1, 10 ** 12) ** 2
            a, b = str(v), "0"
        elif op in ("log10", "ln", "log2"):
            v = int(num(rng, rng.randint(1, 24))) or 1
            if rng.random() < .2:
                v = rng.choice([10 ** rng.randint(0, 23), 2 ** rng.randint(0, 79)])
            a, b = str(v), "0"
        elif op == "logb":
            k = rng.choice([2, 3, 7, 10, 16, rng.randrange(2, 10 ** 5)])
            v = rng.choice([k ** rng.randint(0, 15), int(num(rng, rng.randint(1, 18))) or 1])
            a, b = str(v)[:24], str(k)
        else:  # exp10, exp
            frac = rng.randint(0, 8)
            top = (60 if op == "exp10" else 150) * 10 ** frac
            a = str(rng.choice([rng.randrange(top), rng.randrange(10) * 10 ** frac]))
            b, kw = "0", {"frac": frac, "neg": rng.random() < .4}
        yield a, b, kw


SLOW = {"spf": 6, "log10": 8, "ln": 8, "log2": 6, "logb": 6, "exp10": 12, "exp": 12}


@pytest.mark.parametrize("op", m.OP_NAMES)
def test_routine(op):
    """Random cases (seeded), exact against Python; within the trace budget."""
    for a, b, kw in cases(op, SLOW.get(op, 30), 0):
        got, n, bm = m.run(op, a, b, **kw)
        w = want(op, a, b, **kw)
        if op == "to_base" and int(b) > 10:  # a-f print as "." / "...": check the cells
            c = bm.emu.cells
            got = "".join(DIG[c[(c[spec.I] + j) & 0xFFF]] for j in range(c[spec.V]))
        big = op in ("pow", "npr", "ncr") and len(w) > bcd.MAX_DIGITS
        assert got == w or big and got == "...", (op, a, b, kw, got, w)
        assert n <= BUDGET[op], (op, a, b, n)


EDGES = [("pow", "0", "0", "1"), ("pow", "0", "5", "0"), ("pow", "2", "64", "18446744073709551616"),
         ("pow", "9", "48", str(9 ** 48)), ("pow", "2", "256", "..."),
         ("fact", "0", "0", "1"), ("fact", "39", "0", str(math.factorial(39))), ("fact", "40", "0", "..."),
         ("npr", "5", "6", "0"), ("ncr", "60", "30", str(math.comb(60, 30))), ("ncr", "7", "0", "1"),
         ("idiv", "5", "0", "..."), ("mod", "999999999999999999999999", "1", "0"),
         ("idiv", "7", "123456789", "0"), ("mod", "7", "123456789", "7"),
         ("gcd", "0", "0", "0"), ("gcd", "0", "12", "12"), ("lcm", "0", "5", "0"),
         ("lcm", "999999999999999999999999", "999999999999999999999998",
          str(math.lcm(10 ** 24 - 1, 10 ** 24 - 2))),
         ("spf", "0", "0", "0"), ("spf", "1", "0", "1"), ("spf", "2", "0", "2"),
         ("spf", "999983", "0", "999983"), ("spf", "1000003", "0", "..."),
         ("spf", "998001", "0", "3"), ("spf", "994009", "0", "997"),
         ("to_base", "0", "2", "0"), ("to_base", "255", "16", "ff"), ("to_base", "5", "17", "..."),
         ("isqrt", "0", "0", "0"), ("isqrt", "999999999999999999999999", "0", "999999999999"),
         ("sqrt", "2", "0", "1.4142135623..."), ("sqrt", "144", "0", "12"),
         ("sqrt", "999999999999999999999999", "0", trunc(Decimal(10 ** 24 - 1).sqrt())),
         ("log10", "1", "0", "0"), ("log10", "1000", "0", "3"), ("log10", "0", "0", "..."),
         ("ln", "1", "0", "0"), ("log2", "1024", "0", "10"), ("log2", "1023", "0",
                                                           trunc(Decimal(1023).ln() / Decimal(2).ln())),
         ("logb", "81", "3", "4"), ("logb", "1", "7", "0"), ("logb", "5", "1", "...")]


@pytest.mark.parametrize("op,a,b,w", EDGES)
def test_edges(op, a, b, w):
    got, _, bm = m.run(op, a, b)
    if op == "to_base" and w not in ("...",) and int(b) > 10:
        c = bm.emu.cells
        got = "".join(DIG[c[(c[spec.I] + j) & 0xFFF]] for j in range(c[spec.V]))
    assert got == w, (op, a, b, got, w)


@pytest.mark.parametrize("op,a,kw,w", [
    ("exp10", "3", {}, "1000"), ("exp10", "3", {"neg": True}, "0.001"),
    ("exp10", "0", {"frac": 4}, "1"), ("exp", "0", {}, "1"),
    ("exp", "1", {}, "2.7182818284..."), ("exp10", "5", {"frac": 1}, "3.1622776601..."),
    ("exp", "100", {}, sig11(Decimal(100).exp())), ("exp", "5", {"neg": True}, sig11(Decimal(-5).exp())),
    ("exp", "600", {}, "..."), ("exp", "600", {"neg": True}, "0.0000000000...")])
def test_exp_formats(op, a, kw, w):
    assert m.run(op, a, "0", **kw)[0] == w


def test_inline_print():
    """An inline call (MODE 1) prints " result" and the closer, like the BIOS's routines."""
    assert m.run("sqrt", "2", "0", mode=1)[0] == " 1.4142135623...."
    assert m.run("pow", "2", "10", mode=1)[0] == " 1024."
    assert m.run("gcd", "12", "18", mode=3)[0] == " 6$"


def test_layout():
    """The BIOS keeps its bytes; the jump table is the same in every profile; it all fits."""
    for p, rom in m.ROMS.items():
        assert rom[:len(bcd.ROM)] == bcd.ROM
        assert len(rom) + 0x200 <= m.TOP
        assert all(m.LABELSETS[p][f"jt_{op}"] == m.OPS[op] for op in m.OP_NAMES)
    assert set(m.PROFILE_OF.values()) == set(m.PROFILES)
