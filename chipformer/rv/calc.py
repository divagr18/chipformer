"""The calc firmware (env/calc): exact decimal integer expressions on the digit file.

    python -m chipformer.rv.calc --cases 10000          # reference machine vs Python
    python -m chipformer.rv.calc --expr "12345678*87654321"

``machine(expr)`` is the firmware's snapshot with the expression in calc_in (RAM 0x100);
``expected(expr)`` is Python's answer in the firmware's output format; ``cases(n, seed)`` draws
random problems per operation (add, sub, mul, div, mod, gcd, pow, mixed expressions).
"""

import argparse
import random
import time
from math import gcd
from pathlib import Path

from chipformer.rv import elf, ref
from chipformer.rv import spec as S

ELF = Path(__file__).parents[2] / "tests" / "data" / "rv" / "calc.elf"
CALC_IN, MAXD = 0x100, 62


def machine(expr: str, path=ELF) -> ref.Machine:
    m = ref.Machine.from_program(elf.program(path))
    b = expr.encode() + b"\0"
    assert len(b) <= 256
    for i, x in enumerate(b):
        a = CALC_IN + i
        w = a >> 2
        m.ram[w] = m.ram[w] & ~(0xFF << 8 * (a & 3)) | x << 8 * (a & 3)
    return m


def _floordiv(a, b):
    if b == 0:
        raise ZeroDivisionError
    return a // b


def _mod(a, b):
    if b == 0:
        raise ZeroDivisionError
    return a % b


def expected_decimal(expr: str) -> str:
    """A decimal expression: + - * exact (other operators need integers), printed without
    trailing zeros after the point."""
    import decimal
    import re

    if re.search(r"[/%^]|gcd", expr) or "**" in expr:
        return "error\n"
    src = re.sub(r"\d+\.\d+|\d+", lambda m: f"D('{m[0]}')", expr)
    with decimal.localcontext() as ctx:
        ctx.prec = 400
        try:
            v = eval(src, {"__builtins__": {}}, {"D": decimal.Decimal})  # noqa: S307
        except SyntaxError:
            return "error\n"
        s = format(v, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return "error\n" if len(s.replace("-", "").replace(".", "").lstrip("0")) > MAXD else f"{s}\n"


class _Err(Exception):
    pass


def _value(node):
    """An integer expression's value as a Fraction: / is exact, but a non-integer value may only
    be the final result (or negated); // and % floor, ** needs an exponent 0..999."""
    import ast
    from fractions import Fraction

    def ints(*vs):
        if any(v.denominator != 1 for v in vs):
            raise _Err
        return [int(v) for v in vs]

    if isinstance(node, ast.Constant) and type(node.value) is int:
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _value(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "gcd" and \
            len(node.args) == 2 and not node.keywords:
        return Fraction(gcd(*ints(*map(_value, node.args))))
    if not isinstance(node, ast.BinOp):
        raise _Err
    a, b = ints(_value(node.left), _value(node.right))
    op = type(node.op)
    if op in (ast.Div, ast.FloorDiv, ast.Mod) and b == 0:
        raise _Err
    if op is ast.Pow and not 0 <= b <= 999:
        raise _Err
    f = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
         ast.Div: lambda: Fraction(a, b), ast.FloorDiv: lambda: a // b, ast.Mod: lambda: a % b,
         ast.Pow: lambda: a**b}.get(op)
    if f is None:
        raise _Err
    return Fraction(f())


def expected(expr: str) -> str:
    """Python's value (/ exact: an integer or a reduced fraction p/q; // and % floor, ** / ^
    powers, gcd) or "error", as the firmware prints it."""
    import ast

    if "." in expr:
        return expected_decimal(expr)
    try:
        v = _value(ast.parse(expr.replace("^", "**"), mode="eval").body)
    except (_Err, SyntaxError):
        return "error\n"
    s = str(v)  # Fraction: "p" or "p/q"
    return "error\n" if any(len(x) > MAXD for x in s.lstrip("-").split("/")) else f"{s}\n"


def _n(rng, d):
    return str(rng.randint(10 ** (d - 1) if d > 1 else 0, 10**d - 1))


OPS = ["add", "sub", "mul", "div", "mod", "gcd", "pow", "expr", "frac"]


def case(rng: random.Random, op: str) -> str:
    da, db = rng.randint(1, 24), rng.randint(1, 24)
    a, b = _n(rng, da), _n(rng, db)
    sa = "-" if rng.random() < 0.2 else ""
    sb = "-" if rng.random() < 0.2 else ""
    if op == "add":
        return f"{sa}{a}+{b}"
    if op == "sub":
        return f"{a}-{sb}{b}" if sb else f"{sa}{a}-{b}"
    if op == "mul":
        return f"{sa}{a}*{_n(rng, rng.randint(1, 24))}"
    if op in ("div", "mod"):
        a = _n(rng, rng.randint(1, 30))
        b = _n(rng, rng.randint(1, max(1, min(len(a), 15))))
        return f"{sa}{a}{'//' if op == 'div' else '%'}{'(-' + b + ')' if sb else b}"
    if op == "gcd":
        g = int(_n(rng, rng.randint(1, 6)))
        x, y = int(_n(rng, rng.randint(1, 12))) * g, int(_n(rng, rng.randint(1, 12))) * g
        return f"gcd({x},{y})"
    if op == "pow":
        base = _n(rng, rng.randint(1, 4))
        e = rng.randint(0, max(0, MAXD // len(base) - 1))
        return f"{sa}{base}**{e}" if not sa else f"({sa}{base})**{e}"
    if op == "frac":  # exact division: often a reduced fraction, sometimes an integer
        g = int(_n(rng, rng.randint(1, 8)))
        x, y = int(_n(rng, rng.randint(1, 20))) * g, int(_n(rng, rng.randint(1, 12))) * g
        if rng.random() < 0.3:
            x = y * int(_n(rng, rng.randint(1, 8)))
        return rng.choice([f"{sa}{x}/{y}", f"({x}+{y})/{sb}{y}", f"{x}*{g}/({y}-{g})"])
    # a random expression of 2-4 operands
    k = rng.randint(2, 4)
    nums = [_n(rng, rng.randint(1, 8)) for _ in range(k)]
    s = nums[0]
    for x in nums[1:]:
        o = rng.choice(["+", "-", "*", "//", "%"])
        s = f"({s}){o}{x}" if rng.random() < 0.3 else f"{s}{o}{x}"
    return s


def cases(n: int, seed: int = 0) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    return [(op, case(rng, op)) for i in range(n) for op in [OPS[i % len(OPS)]]]


def run(expr: str) -> tuple[str, list[int]]:
    m = machine(expr)
    toks = m.run(10**7)
    assert m.halted and m.fault is None, (expr, m.fault)
    return m.out.decode(), toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expr", default=None)
    ap.add_argument("--free-run", action="store_true", help="the circuit's executor on the GPU")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    if args.free_run:
        from functools import partial

        from chipformer.rv import core, fuzz

        c = core.build()
        exprs = [e for _, e in cases(args.cases, args.seed)]
        exprs.sort(key=lambda e: len(machine(e).run(10**7)))  # (a batch runs to its longest)
        res = fuzz.free_run(c, [partial(machine, e) for e in exprs], args.batch)
        print(f"calc free-running: {res}", flush=True)
        return
    if args.expr:
        out, toks = run(args.expr)
        st = ref.trace_stats(toks)
        print(repr(out), expected(args.expr) == out, st)
        return
    t = time.time()
    per = {op: [0, 0, 0] for op in OPS}  # cases, instructions, tokens
    bad = 0
    for op, e in cases(args.cases, args.seed):
        out, toks = run(e)
        if out != expected(e):
            bad += 1
            print("MISMATCH", e, repr(out), repr(expected(e)))
        st = ref.trace_stats(toks)
        per[op][0] += 1
        per[op][1] += st["instructions"]
        per[op][2] += st["tokens"]
    print(f"{args.cases} cases, {bad} mismatches ({time.time() - t:.0f} s)")
    for op, (n, i, tk) in per.items():
        if n:
            print(f"  {op:5s} {n:5d} cases  {i / n:8.0f} instructions  {tk / n:8.0f} tokens")
    _ = S


if __name__ == "__main__":
    main()
