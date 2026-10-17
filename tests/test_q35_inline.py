"""Inline arithmetic: the circuit answers "A op B =" in Qwen's plain text."""

import random

import pytest

from chipformer.q35 import bridge as B
from chipformer.q35 import check as Q
from chipformer.q35 import chip8_q35
from chipformer.q35.lib import bcd

BS = chr(92)
WORDS = {" ": B.SPACE, " =": B.EQ, "=": B.EQ, " +": B.PLUS, "+": B.PLUS, " -": B.MINUS,
         "-": B.MINUS, " ×": B.TIMES, "*": B.TIMES, " *": B.TIMES, " " + BS: B.BSL, "times": B.TIMESW,
         ",": B.COMMA, "$": B.DOLLAR, " $": B.DOLLAR, "(": B.LPAREN, "\n": B.NL, "\n\n": B.WS}


def classes(tokens: list[str]) -> list[int]:
    """Qwen tokens (digits split, as its tokenizer does) to text classes."""
    out = []
    for t in tokens:
        if t.isdigit():
            out += [B.HEX[int(ch)] for ch in t]
        else:
            out.append(WORDS.get(t, B.OTHER))
    return out


def run(tokens):
    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(3)
    calls = bm.qwen(classes(tokens))
    bm.text(3)
    return bm, calls


def printed(bm) -> str:
    out = []
    for t, g in zip(bm.tokens, bm.generated):
        if g and B.is_text(t):
            out.append(" " if t == B.SPACE else str(B.HEX.index(t)) if t in B.HEX else "?")
    return "".join(out)


TIMES = [" " + BS, "times", " "]
CASES = [
    (["$", "12", *TIMES, "34", " =", " "], [("mul", "12", "34")]),
    ([" ", "1", ",", "234", ",", "567", *TIMES, "8", " ="], [("mul", "1234567", "8")]),
    ([" ", "37", " ×", " ", "43", " ="], [("mul", "37", "43")]),
    ([" ", "37", "*", "43", "="], [("mul", "37", "43")]),
    ([" ", " $", " ", "300", " +", " ", "4", " ="], [("add", "300", "4")]),
    ([" ", "100", " -", " ", "30", " ="], [("sub", "100", "30")]),
    (["\n\n", "42450", " *", " ", "8376", " ="], [("mul", "42450", "8376")]),
    ([" ", "2", ".", "55", *TIMES, "300", " ="], []),  # decimal
    (["x", "^", "22", " +", " ", "100", " ="], []),  # not a plain first number
    ([" ", "20", " +", " ", "30", " +", " ", "40", " ="], []),  # chains are Qwen's
    ([" ", "20", *TIMES, "30", " +", " ", "40", " ="], []),  # precedence
    (["}", *TIMES, "30", " +", " ", "40", " ="], []),  # operator before the first number
    (["-", "50", *TIMES, "30", " ="], []),  # negative
    ([" ", "50", *TIMES, "-", "30", " ="], []),
    (["(", "50", " +", " ", "30", ")", " ="], []),
    ([" ", "12", *TIMES, "3", " ="], []),  # small: Qwen's own
    ([" ", "5", " +", " ", "7", " ="], []),
    ([" ", "50", " +", " ", "30", " apples", " ="], []),
    ([" ", "30", " ", "40", *TIMES, "50", " ="], []),  # two numbers before the operator
    ([" ", "50", " =", " ", "30", *TIMES, "40", " ="], []),
    (["x", "4245", *TIMES, "60", " ="], []),  # a letter right before the first number
]


@pytest.mark.parametrize("tokens,want", CASES)
def test_scan(tokens, want):
    _bm, calls = run(tokens)
    assert calls == want


def test_scan_prints_and_chains():
    bm, calls = run(["$", "12", *TIMES, "34", " =", "$", " ", "and", " ", "408", *TIMES, "2", " ="])
    assert calls == [("mul", "12", "34"), ("mul", "408", "2")]
    assert printed(bm) == " 408 816"
    # the printed result is the first number of the next expression
    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(3)
    calls = bm.qwen(classes([" ", "12", *TIMES, "34", " ="]))
    calls += bm.qwen(classes([*TIMES, "2", " ="]))
    assert calls == [("mul", "12", "34"), ("mul", "408", "2")] and printed(bm) == " 408 816"


def test_negative_sub_prints_nothing():
    bm, calls = run([" ", "30", " -", " ", "50", " ="])
    assert calls == [("sub", "30", "50")] and printed(bm) == " "


@pytest.fixture(scope="module")
def circuit():
    return chip8_q35.build()


def check(circuit, bm):
    n, bad, bad_owner = Q.check_bridge(circuit, bm.tokens, bm.generated, B.owner_expected(bm))
    assert not bad and not bad_owner, (bad[:5], bad_owner[:5])
    return n


@pytest.mark.parametrize("tokens,want", CASES)
def test_circuit_cases(circuit, tokens, want):
    bm, _ = run(tokens)
    check(circuit, bm)


def test_circuit_with_tool_calls(circuit):
    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(4)
    bm.qwen(classes(["$", "48", *TIMES, "12", " =", "$", " ", "then"]))
    bm.tool_call("add", "576", "99")
    while not bm.halted:
        bm.step()
    bm.qwen(classes([" ", "675", " -", " ", "5", " =", "."]))
    assert printed(bm) == " 576???675??? 670"  # ? = the tool response's tags and newlines
    check(circuit, bm)


def test_circuit_fuzz(circuit):
    """Random streams of the classes the capture cares about: the circuit fires exactly where
    the reference does, and the records and prints that follow are exact."""
    rng = random.Random(5)
    pool = ([B.HEX[d] for d in range(10)] * 4 + [B.SPACE] * 4
            + [B.EQ, B.PLUS, B.MINUS, B.TIMES, B.BSL, B.TIMESW, B.COMMA, B.DOLLAR, B.LPAREN,
               B.NL, B.OTHER, B.OTHER])
    fired = 0
    for _ in range(12):
        bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
        bm.text(2)
        stream = []
        for _ in range(60):
            if rng.random() < 0.15:  # plant an expression
                a, b = str(rng.randrange(1, 10**rng.randint(1, 5))), str(rng.randrange(10**3))
                op = rng.choice([" +", " -", "*", TIMES])
                stream += [B.SPACE, *classes([a]), *classes(op if isinstance(op, list) else [op]),
                           B.SPACE, *classes([b]), B.EQ]
            stream.append(rng.choice(pool))
        fired += len(bm.qwen(stream))
        check(circuit, bm)
    assert fired >= 8
