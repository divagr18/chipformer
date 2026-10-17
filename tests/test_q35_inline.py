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
         "/": B.DIV, " /": B.DIV, "÷": B.DIV, "div": B.FN_DIV, ".": B.PERIOD,
         "(-": B.LPMINUS, " (-": B.SLPMINUS, ")": B.RPAREN, ")/": B.DIV, ")=": B.EQ,
         " mod": B.MODOP, " %": B.MODOP, "mod": B.FN_MOD, "b": B.HEX[11], "gcd": B.FN_GCD,
         " gcd": B.FN_GCD}


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
            close = {B.SPACE: " ", B.MINUS: " -", B.PERIOD: ".", B.DOLLAR: "$", B.DDOLLAR: "$$",
                     B.ELLIPSIS: "..."}
            out.append(close[t] if t in close else str(B.HEX.index(t)) if t in B.HEX else "?")
    return "".join(out)


TIMES = [" " + BS, "times", " "]
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
    ([" ", "2", ".", "55", *TIMES, "300", " ="], []),  # decimal
    (["x", "^", "22", " +", " ", "100", " ="], []),  # not a plain first number
    ([" ", "20", " +", " ", "30", " +", " ", "40", " ="], []),  # chains are Qwen's
    ([" ", "20", *TIMES, "30", " +", " ", "40", " ="], []),  # precedence
    (["}", *TIMES, "30", " +", " ", "40", " ="], []),  # operator before the first number
    (["-", "50", *TIMES, "30", " ="], []),  # "-" after a letter: an operator
    ([" ", "50", *TIMES, "-", "30", " ="], [("mul", "50", "30")]),  # signed
    (["\n", "-", "5000", *TIMES, "30", " ="], [("mul", "5000", "30")]),
    (["\n", "-", " ", "50", *TIMES, "30", " ="], [("mul", "50", "30")]),  # a list's "- "
    (["x", " -", " ", "12", *TIMES, "34", " ="], []),  # x - 12 * 34: "-" is an operator
    ([" ", "12", " -", "34", *TIMES, "5", " ="], []),
    (["\n", "-", "1418261", " /", " (-", "408", ")", " ="], [("div", "1418261", "408")]),
    (["$", "(-", "1418261", ")/", "(-", "408", ")", " ="], [("div", "1418261", "408")]),
    (["f", "(-", "3000", ")", *TIMES, "4", " ="], []),  # f(-3000): not a number's sign
    ([" ", "=", " -", "5", " +", " ", "1234567", " ="], []),  # add / sub: unsigned only
    ([" ", "1234567", " -", " -", "5", " ="], []),
    (["\n", "-", "1234", " mod", " ", "56", " ="], []),
    ([" ", "1785", " mod", " ", "13", " ="], [("mod", "1785", "13")]),
    ([" ", "1785", " %", " ", "13", " ="], [("mod", "1785", "13")]),
    ([" ", "1785", " " + BS, "mod", " ", "13", " ="], [("mod", "1785", "13")]),
    ([" ", "1785", " " + BS, "b", "mod", " ", "13", " ="], [("mod", "1785", "13")]),
    ([" ", "1785", " ", "mod", " ", "13", " ="], []),  # "mod" without a backslash
    ([" ", "gcd", "(", "1785", ",", " ", "13201915", ")", " ="], [("gcd", "1785", "13201915")]),
    (["$", BS, "gcd", "(", "1785", ",", " ", "-", "35", ")=", "$"], [("gcd", "1785", "35")]),
    ([" ", "1785", " gcd", "(", "12", ",", " ", "35", ")", " ="], []),  # gcd after a number
    ([" ", "gcd", "(", "1785", ",", "35", ")", " ="], []),  # "1785,35" is one number
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
    assert calls == want


def test_scan_prints_and_chains():
    bm, calls = run(["$", "12", *TIMES, "34", " =", "$", " ", "and", " ", "408", *TIMES, "2", " ="])
    assert calls == [("mul", "12", "34"), ("mul", "408", "2")]
    assert printed(bm) == " 408$ 816."  # "$" closes the math it opened in, else "."
    # after a "." closer the printed result does not start a new expression
    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(3)
    calls = bm.qwen(classes([" ", "12", *TIMES, "34", " ="]))
    calls += bm.qwen(classes([*TIMES, "2", " ="]))  # the "." closer ends the expression
    assert calls == [("mul", "12", "34")] and printed(bm) == " 408."
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


def test_signed_mod_gcd_prints():
    for tokens, want in [
            (["\n", "-", "1418261", " /", " (-", "408", ")", " ="], " 3476.1299019607...."),
            (["$", "-", "2548", " /", " ", "52", " ="], " -49$"),
            ([" ", "12", *TIMES, "-", "34", " ="], " -408."),
            ([" (-", "1234", ")", *TIMES, "(-", "5", ")", " ="], " 6170."),
            (["\n", "-", "1", " /", " ", "3000", " ="], " -0.0003333333...."),
            (["\n", "-", "1234", " /", " ", "0", " ="], " "),  # by zero: nothing, no sign
            ([" ", "1785", " mod", " ", "13", " ="], " 4."),
            ([" ", "1785", " mod", " ", "1785", " ="], " 0."),
            ([" ", "17", " mod", " ", "5000", " ="], " 17."),
            ([" ", "00017", " mod", " ", "5000", " ="], " 17."),
            ([" ", "1785", " mod", " ", "0", " ="], " "),
            ([" ", "gcd", "(", "1785", ",", " ", "13201915", ")", " ="], " 5."),
            ([" ", "gcd", "(", "13201915", ",", " ", "1785", ")", " ="], " 5."),
            ([" ", "gcd", "(", "1234", ",", " ", "0", ")", " ="], " 1234."),
            ([" ", "gcd", "(", "0", ",", " ", "1234", ")", " ="], " 1234."),
            ([" ", "gcd", "(", "123456789012", ",", " ", "987654321098", ")", " ="], " 2."),
            ([" ", "gcd", "(", "2", "0" * 23, ",", " ", "4", "0" * 20, ")", " ="],
             " 4" + "0" * 20 + ".")]:
        bm, calls = run(tokens)
        assert len(calls) == 1 and printed(bm) == want, (tokens, printed(bm))


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
    tail = "?" * (3 + len(B.TOOL_TAIL))  # "\n</tool_response>\n" and the empty thinking block
    assert printed(bm) == f" 576$???675{tail} -4325."  # ? = the tool response's tags and newlines
    bm.tool_call("div", "355", "113")
    while not bm.halted:
        bm.step()
    assert printed(bm).endswith("???3.1415929203..." + tail)
    check(circuit, bm)
    bm.tool_call("sub", "5", "675")  # below 0: " -" instead of the newline
    while not bm.halted:
        bm.step()
    assert printed(bm).endswith("..." + tail + "??" + " -670" + tail)
    for op, x, y, want in [("div", "-1418261", "-408", "3476.1299019607..."),
                           ("mul", "-63075", "8808", " -555564600"), ("div", "-5", "0", "?"),
                           ("mod", "13201915", "1785", "55"), ("gcd", "1785", "13201915", "5"),
                           ("gcd", "-12", "18", "6")]:
        bm.tool_call(op, x, y)
        while not bm.halted:
            bm.step()
        assert printed(bm).endswith(want + tail), (op, x, y, printed(bm)[-40:])
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
               B.NL, B.OTHER, B.OTHER, B.LPMINUS, B.SLPMINUS, B.RPAREN, B.MODOP, B.FN_MOD,
               B.FN_GCD, B.HEX[11]])
    fired = 0
    for _ in range(12):
        bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
        bm.text(2)
        stream = []
        for _ in range(60):
            if rng.random() < 0.15:  # plant an expression
                a, b = str(rng.randrange(1, 10**rng.randint(1, 5))), str(rng.randrange(10**rng.randint(1, 4)))
                op = rng.choice([" +", " -", "-", "*", TIMES, "/", [" " + chr(92), "div", " "],
                                 " mod", [" " + chr(92), "b", "mod", " "], "gcd"])
                sa, sb = rng.choice([[], ["-"], [" (-"]]), rng.choice([[], ["-"], ["(-"]])
                if op == "gcd":
                    stream += classes([" ", "gcd", "(", *sa, a, ",", " ", *sb, b, ")", " ="])
                else:
                    stream += [B.SPACE, *classes([*sa, a]),
                               *classes(op if isinstance(op, list) else [op]),
                               B.SPACE, *classes([*sb, b]), B.EQ]
            stream.append(rng.choice(pool))
        fired += len(bm.qwen(stream))
        check(circuit, bm)
    assert fired >= 8
