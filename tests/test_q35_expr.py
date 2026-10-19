"""Whole expressions (v18): the tape capture, the BIOS evaluator eval, and the circuit."""

import random
import re
from fractions import Fraction

import pytest

from chipformer.q35 import bridge as B
from chipformer.q35 import check as Q
from chipformer.q35 import chip8_q35
from chipformer.q35.lib import bcd
from test_q35_inline import TIMES, classes, printed

INV = {v: k for k, v in B.EXPANSION.items()}


def tape_of(text: str) -> list[int]:
    """A string of symbols as tape codes, using fused codes where they exist."""
    out, i = [], 0
    multi = sorted(B.MULTI, key=len, reverse=True)
    while i < len(text):
        if text[i].isdigit():
            out.append(int(text[i]))
            i += 1
            continue
        m = next((m for m in multi if text.startswith(m, i)), text[i])
        out.append(INV[m])
        i += len(m)
    return out


def python_value(text: str):
    """The expression's exact value by Python (^ is **, a bracket after an operand multiplies)."""
    text = re.sub(r"([\d)])\(", r"\1*(", text).replace("^", "**")
    text = re.sub(r"\d+(\.\d*)?", lambda m: f"Fraction('{m[0]}')", text)
    try:
        return eval(text, {"Fraction": Fraction})  # noqa: S307
    except ZeroDivisionError:
        return None


def run_eval(tape, mode=1):
    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(2)
    n0 = len(bm.tokens)
    bm._ecall(tape, mode)
    steps = 0
    while not bm.halted:
        bm.step()
        steps += 1
        assert steps < 20000, "eval does not halt"
    out = [t for t, g in zip(bm.tokens[n0:], bm.generated[n0:]) if B.is_text(t)]
    shown = {**{h: str(i) for i, h in enumerate(B.HEX)}, B.PERIOD: ".", B.ELLIPSIS: "...",
             B.DIV: "/"}
    # inline: " -" first; a tool response: " -" instead of the newline after <tool_response>
    return B.MINUS in out[:3], "".join(shown.get(t, "") for t in out[1:-1]), len(bm.tokens) - n0


def random_expr(rng, d=0):
    def num():
        n = str(rng.randrange(0, 10 ** rng.randint(1, 4)))
        if rng.random() < 0.15:
            n += "." + str(rng.randrange(0, 1000))
        if rng.random() < 0.25:
            n = rng.choice(["(-" + n + ")", "-" + n])
        return n

    def operand():
        return num() if d > 2 or rng.random() < 0.65 else "(" + random_expr(rng, d + 1) + ")"

    parts = [operand()]
    for _ in range(rng.randint(1, 3)):
        op = rng.choice("+-*/" * 3 + "^")
        rhs = operand()
        if op == "^":
            rhs = str(rng.randrange(0, 5)) if rng.random() < 0.8 else "-" + str(rng.randrange(1, 3))
        parts += [op, rhs]
    return "".join(parts)


FIXED = [("3+4*5", "23"), ("(3+4)*5", "35"), ("2^10-5", "1019"), ("((-8)/(-60))*329", "658/15"),
         ("20*(-8)/(-60)*329*9/((-36)/2)*7*(-6)/49", "376"), ("1/3", "1/3"),
         ("-2^2", "-4"), ("2^-2", "0.25"), ("0.5*4", "2"), ("1.25+2.5", "3.75"), ("7-10", "-3"),
         ("-(3-5)", "2"), ("2(3+4)", "14"), ("10/4", "2.5"), ("1/0", "..."), ("0*(-5)/7", "0"),
         ("(((9/12)/((-2)/232*-6))/((244521/6456)/303))/(((-2)/(-2))/1)", "116"),
         ("3+*4", "..."), ("2^123", "..."), ("99999999999999999999*99999", "...")]


@pytest.mark.parametrize("text,want", FIXED)
def test_eval_fixed(text, want):
    neg, body, _ = run_eval(tape_of(text))
    assert ("-" if neg else "") + body == want
    assert bcd.eval_text(tape_of(text)) == (want.startswith("-"), want.lstrip("-"))


def test_eval_random():
    """eval (the BIOS on the reference machine) == bcd.eval_text == Python's exact value."""
    rng = random.Random(18)
    n = errors = 0
    lengths = []
    for _ in range(400):
        text = random_expr(rng)
        tape = tape_of(text)
        if len(tape) > B.TAPE_MAX:
            continue
        want = bcd.eval_text(tape)
        neg, body, n_tok = run_eval(tape)
        assert (neg, body) == want, (text, want, (neg, body))
        lengths.append(n_tok)
        v = python_value(text)
        if want[1] == "...":
            errors += 1
            continue
        n += 1
        got = Fraction(body.rstrip(".")) * (-1 if neg else 1)
        if body.endswith("..."):  # 10 decimals, truncated
            assert abs(v - got) < Fraction(1, 10**10), (text, v, body)
        else:
            assert v == got, (text, v, body)
    assert n > 250 and errors < n // 4, (n, errors)
    lengths.sort()
    # (a quotient that does not end runs gcd for its reduced fraction)
    assert lengths[len(lengths) // 2] < 12000, lengths[len(lengths) // 2]


def scan(tokens):
    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(3)
    calls = bm.qwen(classes(tokens))
    return bm, calls


BS = chr(92)
SCAN = [
    ([" ", "(", "3", " +", " ", "4", ")", *TIMES, "12", " /", " ", "7", " ="], "(3+4)*12/7"),
    ([" ", "2", "^{", "10", "}", " -", " ", "5", " ="], "2^(10)-5"),
    ([" ", "((-", "8", ")/", "(-", "60", "))*", "329", " ="], "((-8)/(-60))*329"),
    ([" $", " ", "1", ".", "25", " +", " ", "2", ".", "5", " ="], "1.25+2.5"),
    ([" (", "1", ",", "234", " +", " ", "5", ")", " ", "*", " ", "2", " ="], "(1234+5)*2"),
    ([" =", " (", "12", " -", " ", "3", ")", "*", "45", " ="], "(12-3)*45"),  # fresh after "="
    ([" ", "3", " +", " ", "4", " +", " ", "5", " ="], None),  # too few digits
    (["x", " -", " ", "3", " +", " ", "45", " ="], None),  # starts after a letter
    (["f", "(", "3", " +", " ", "45", ")", " ="], None),
    ([" ", "12", " +", " ", "3", " apples", " +", " ", "45", " ="], None),  # broken window
    ([" ", "1234", *TIMES, "5678", " ="], "fast"),  # one operator: the single-op path
    ([" ", "^", "12", " +", " ", "345", " ="], None),  # an operator can't start it
    # a "." that ends a number breaks the window: no 3.11 - 8, and 12+345+6 alone
    ([" ", "3", ".", " $", "11", " -", " ", "8", " ="], None),
    ([" ", "1", ".", " ", "12", "+", "345", "+", "6", " ="], "12+345+6"),
]


@pytest.mark.parametrize("tokens,want", SCAN)
def test_scan(tokens, want):
    _bm, calls = scan(tokens)
    if want is None:
        assert calls == []
    elif want == "fast":
        assert [c[0] for c in calls] == ["mul"]
    else:
        assert len(calls) == 1 and calls[0][0] == "eval"
        assert "".join(B.EXPANSION[c] for c in calls[0][1]) == want


def test_prints():
    bm, _ = scan([" $", " ", "(", "3", " -", " ", "40", ")", " ", "*", " ", "2", " =", "$"])
    assert printed(bm) == " -74$"
    bm, _ = scan([" ", "100", " /", " ", "(", "3", " +", " ", "4", ")", " ="])
    assert printed(bm) == " 100/7."
    bm, _ = scan([" ", "(", "12", " /", " ", "0", ")", " +", " ", "5", " ="])
    assert printed(bm) == " ...."  # can't: "..."


@pytest.fixture(scope="module")
def circuit():
    return chip8_q35.build()


def check(circuit, bm):
    n, bad, bad_owner = Q.check_bridge(circuit, bm.tokens, bm.generated, B.owner_expected(bm))
    assert not bad and not bad_owner, (bad[:5], bad_owner[:5])
    return n


@pytest.mark.parametrize("tokens,want", SCAN[:4] + SCAN[5:7] + SCAN[-2:])
def test_circuit_scan(circuit, tokens, want):
    bm, _ = scan(tokens)
    bm.text(2)
    check(circuit, bm)


def test_circuit_prints(circuit):
    """A negative result and "..." in one session."""
    bm, calls = scan([" $", " ", "(", "3", " -", " ", "40", ")", " ", "*", " ", "2", " =", "$",
                      " ", "and", " ", "(", "12", " /", " ", "0", ")", " +", " ", "5", " ="])
    assert [c[0] for c in calls] == ["eval", "eval"] and printed(bm) == " -74$ ...."
    bm.text(2)
    check(circuit, bm)


def test_circuit_tool(circuit):
    """The tool calc(expression): the tape is the call's argument, the result a tool response."""
    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(3)
    bm.calc_call(classes(["(", "3", "+", "4", ")*", "5"]))
    while not bm.halted:
        bm.step()
    bm.text(2)
    assert printed(bm) == "???35" + "?" * (3 + len(B.TOOL_TAIL))
    check(circuit, bm)


def test_tool_negative():
    """A negative value prints with "-", inline and (since v16) as a tool response."""
    assert run_eval(tape_of("3-10"), mode=0)[:2] == (True, "7")
    assert run_eval(tape_of("3-10"), mode=1)[:2] == (True, "7")
    assert run_eval(tape_of("10-3*2"), mode=0)[:2] == (False, "4")
