"""Inline arithmetic: the circuit answers "A op B =" in Qwen's plain text."""

import random

import pytest

from chipformer.q35 import bridge as B
from chipformer.q35 import check as Q
from chipformer.q35 import chip8_q35
from chipformer.q35.lib import bcd

BS = chr(92)
WORDS = {" ": B.SPACE, " =": B.EQ, "=": B.EQ, " +": B.PLUS, "+": B.PLUS, " -": B.MINUS,
         "-": B.MINUS2, "−": B.MINUS2, " ×": B.TIMES, "*": B.TIMES, " *": B.TIMES, " " + BS: B.BSL, "times": B.TIMESW,
         ",": B.COMMA, "$": B.DOLLAR, " $": B.SDOLLAR, "$$": B.DDOLLAR, " $$": B.SDDOLLAR, "(": B.LPAREN, "\n": B.NL, "\n\n": B.WS,
         "/": B.DIV, " /": B.DIV, "÷": B.DIV, "div": B.FN_DIV, ".": B.PERIOD}


def classes(tokens: list[str]) -> list[int]:
    """Qwen tokens (digits split, as its tokenizer does) to text classes."""
    out = []
    for t in tokens:
        if t.isdigit():
            out += [B.HEX[int(ch)] for ch in t]
        else:
            out.append(WORDS.get(t, B.CLASS_OF.get(t, B.OTHER)))
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
            close = {B.SPACE: " ", B.MINUS: " -", B.PERIOD: ".", B.DOLLAR: "$", B.DDOLLAR: "$$",
                     B.ELLIPSIS: "..."}
            out.append(close[t] if t in close else str(B.HEX.index(t)) if t in B.HEX else "?")
    return "".join(out)


TIMES = [" " + BS, "times", " "]
EV = "eval"  # a whole expression (test_q35_expr.py)
CASES = [
    (["$", "12", *TIMES, "34", " =", " "], [("mul", "12", "34")]),
    ([" ", "1", ",", "234", ",", "567", *TIMES, "8", " ="], [("mul", "1234567", "8")]),
    ([" ", "37", " ×", " ", "43", " ="], [("mul", "37", "43")]),
    ([" ", "37", "*", "43", "="], [("mul", "37", "43")]),
    ([" ", " $", " ", "3000", " +", " ", "400", " ="], [("add", "3000", "400")]),
    ([" ", " $", " ", "300", " +", " ", "4", " ="], []),  # small add / sub: Qwen's own
    ([" ", "10000", " -", " ", "30", " ="], [("sub", "10000", "30")]),
    ([" ", "100", " -", " ", "30", " ="], []),
    ([" ", "25", " -", " ", "1", ",", "786", ",", "338", " ="], [("sub", "25", "1786338")]),
    ([" ", "2500", "−", "17860", " ="], [("sub", "2500", "17860")]),
    (["\n\n", "42450", " *", " ", "8376", " ="], [("mul", "42450", "8376")]),
    ([" ", "2", ".", "55", *TIMES, "300", " ="], [EV]),  # decimal: an expression
    (["x", "^", "22", " +", " ", "100", " ="], []),  # not a plain first number
    ([" ", "20", " +", " ", "30", " +", " ", "40", " ="], [EV]),  # chains: expressions
    ([" ", "20", *TIMES, "30", " +", " ", "40", " ="], [EV]),
    (["}", *TIMES, "30", " +", " ", "40", " ="], []),  # operator before the first number
    (["-", "50", *TIMES, "30", " ="], []),  # negative
    ([" ", "50", *TIMES, "-", "30", " ="], [EV]),
    (["(", "50", " +", " ", "30", ")", " ="], []),
    ([" ", "12", *TIMES, "3", " ="], []),  # small: Qwen's own
    ([" ", "5", " +", " ", "7", " ="], []),
    ([" ", "50", " +", " ", "30", " apples", " ="], []),
    ([" ", "30", " ", "40", *TIMES, "50", " ="], []),  # two numbers before the operator
    ([" ", "50", " =", " ", "30", *TIMES, "40", " ="], [("mul", "30", "40")]),  # fresh after =
    (["x", "^", "2", " =", " ", "4818396", *TIMES, "7", " ="], [("mul", "4818396", "7")]),
    ([" ", "50", " =", "30", *TIMES, "40", " ="], []),  # a number right after "=" is not counted
    (["x", "4245", *TIMES, "60", " ="], []),  # a letter right before the first number
    ([" ", "1686", "/", "6", " ="], [("div", "1686", "6")]),
    (["$", "259749", " /", " ", "3162", " ="], [("div", "259749", "3162")]),
    ([" ", "720", " " + BS, "div", " ", "45", " ="], [("div", "720", "45")]),
    ([" ", "100", "÷", "7", " ="], [("div", "100", "7")]),
    ([" ", "1", "/", "2", " ="], []),  # small
    ([" ", "km", "/", "1234", " ="], []),
    ([" ", "1234", " ", "div", " ", "5", " ="], []),  # "div" without a backslash
]


@pytest.mark.parametrize("tokens,want", CASES)
def test_scan(tokens, want):
    _bm, calls = run(tokens)
    assert [EV if c[0] == EV else c for c in calls] == want


def test_scan_prints_and_chains():
    bm, calls = run(["$", "12", *TIMES, "34", " =", "$", " ", "and", " ", "408", *TIMES, "2", " ="])
    assert calls == [("mul", "12", "34"), ("mul", "408", "2")]
    assert printed(bm) == " 408$ 816."  # "$" closes the math it opened in, else "."
    # after a "." closer the single-op path does not go on from the printed result, but a whole
    # expression does: "408." is a number (v18)
    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(3)
    calls = bm.qwen(classes([" ", "12", *TIMES, "34", " ="]))
    calls += bm.qwen(classes([*TIMES, "2", " ="]))
    assert [c[0] for c in calls] == ["mul", "eval"] and printed(bm) == " 408. 816."
    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(3)
    calls = bm.qwen(classes([" $$", " ", "3000", " +", " ", "4000", " ="]))
    assert calls == [("add", "3000", "4000")] and printed(bm) == " 7000$$"


def test_div_prints():
    for a, b, want in [("1686", "6", " 281."), ("1000", "8", " 125."), ("10", "16", " 0.625."),
                       ("100", "7", " 14.2857142857...."), ("1", "3000", " 0.0003333333...."),
                       ("99999999", "9999", " 10001."),
                       ("5", "125000000000000", " 0.0000000000....")]:
        bm, calls = run([" ", a, "/", b, " ="])
        assert calls == [("div", a, b)] and printed(bm) == want, (a, b, printed(bm))
    bm, _ = run(["$", "355", "/", "113", " ="])
    assert printed(bm) == " 3.1415929203...$"
    bm, _ = run([" ", "1234", "/", "0", " ="])  # division by zero prints nothing
    assert printed(bm) == " "


def test_negative_sub_prints_minus():
    bm, calls = run([" ", "3000", " -", " ", "5000", " ="])
    assert calls == [("sub", "3000", "5000")] and printed(bm) == " -2000."
    bm, calls = run(["$", "21", " -", " ", "97000", " ="])
    assert calls == [("sub", "21", "97000")] and printed(bm) == " -96979$"
    bm, calls = run([" ", "4567", " -", " ", "4567", " ="])
    assert printed(bm) == " 0."


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
    bm.qwen(classes([" ", "675", " -", " ", "5000", " =", "."]))
    assert printed(bm) == " 576$???675??? -4325."  # ? = the tool response's tags and newlines
    bm.tool_call("div", "355", "113")
    while not bm.halted:
        bm.step()
    assert printed(bm).endswith("???3.1415929203...???")
    check(circuit, bm)
    bm.tool_call("sub", "5", "675")  # a tool call below 0 prints nothing
    while not bm.halted:
        bm.step()
    assert printed(bm).endswith("..." + "?" * 9)  # div's closing tags, then tags but no digits
    check(circuit, bm)


def test_template_tool_tags_do_not_fire(circuit):
    """Qwen's tool instructions show an empty "<tool_call></tool_call>": after inline operators in
    earlier text it must not start a call (the routine is cleared at <tool_call>)."""
    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(3)
    calls = bm.qwen(classes([" ", "-", " ", "x", " +", " ", "y", " ", "*", " "])
                    + [B.TC_OPEN, B.TC_CLOSE, B.NL] + classes([" ", "1234", " +"]) + [B.OTHER] * 3)
    assert calls == []
    check(circuit, bm)
    bm.tool_call("mul", "37", "43")  # and a real call afterwards still works
    while not bm.halted:
        bm.step()
    bm.text(3)
    check(circuit, bm)


def test_circuit_fuzz(circuit):
    """Random streams of the classes the capture cares about: the circuit fires exactly where
    the reference does, and the records and prints that follow are exact."""
    rng = random.Random(5)
    pool = ([B.HEX[d] for d in range(10)] * 4 + [B.SPACE] * 4
            + [B.EQ, B.PLUS, B.MINUS, B.MINUS2, B.DIV, B.TIMES, B.BSL, B.TIMESW, B.COMMA, B.DOLLAR, B.LPAREN,
               B.NL, B.OTHER, B.OTHER])
    fired = 0
    for _ in range(12):
        bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
        bm.text(2)
        stream = []
        for _ in range(60):
            if rng.random() < 0.15:  # plant an expression
                a, b = str(rng.randrange(1, 10**rng.randint(1, 5))), str(rng.randrange(10**rng.randint(1, 4)))
                op = rng.choice([" +", " -", "-", "*", TIMES, "/", [" " + chr(92), "div", " "]])
                stream += [B.SPACE, *classes([a]), *classes(op if isinstance(op, list) else [op]),
                           B.SPACE, *classes([b]), B.EQ]
            stream.append(rng.choice(pool))
        fired += len(bm.qwen(stream))
        check(circuit, bm)
    assert fired >= 8
