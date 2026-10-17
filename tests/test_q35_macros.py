"""The looping bignum macros (BMUL0 / BMUL, BTRIM, BRSUB, BADD / BSUB with jump targets):
reference results, trace lengths, and the circuit against the reference."""

import random

import pytest

from chipformer.q35 import bridge as B
from chipformer.q35 import check as Q
from chipformer.q35 import chip8_q35
from chipformer.q35.lib import bcd
from chipformer.q35.lib.macros import macro_writes

CASES = [("mul", "7", "8"), ("mul", "123456", "7"), ("mul", "7", "123456"), ("mul", "0", "4567"),
         ("mul", "99999999", "99999999"), ("mul", "1000", "1000"), ("add", "0", "0"),
         ("add", "999999", "1"), ("sub", "10000", "9999"), ("sub", "4567", "4567"),
         ("sub", "1234", "98765"), ("sub", "5", "6"), ("sub", "0", "0")]


def run_call(op, a, b, mode):
    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(3)
    n0 = len(bm.tokens)
    bm._call(op, a, b, mode=mode)
    while not bm.halted:
        bm.step()
    return bm, n0


def shown(bm, n0):
    sym = {B.SPACE: " ", B.MINUS: " -", B.PERIOD: "."}
    return "".join(sym.get(t, str(B.HEX.index(t)) if t in B.HEX else "")
                   for t, g in zip(bm.tokens[n0:], bm.generated[n0:]) if g and B.is_text(t))


@pytest.mark.parametrize("op,a,b", CASES)
def test_reference_results(op, a, b):
    v = {"add": int(a) + int(b), "sub": int(a) - int(b), "mul": int(a) * int(b)}[op]
    bm, n0 = run_call(op, a, b, 1)
    assert shown(bm, n0) == (" -" if v < 0 else " ") + str(abs(v)) + "."
    bm, n0 = run_call(op, a, b, 0)  # a tool call: < 0 prints nothing
    assert shown(bm, n0) == ("" if v < 0 else str(v))


def test_trace_lengths():
    """Tokens from the call to the end of the print (the v13 BIOS took 209 / 383 / 84 / 576)."""
    for (op, a, b), most in [(("mul", "123456", "7890"), 140), (("mul", "12345678", "87654321"), 265),
                             (("add", "1234", "567"), 60), (("sub", "1234", "5678"), 85)]:
        bm, n0 = run_call(op, a, b, 1)
        assert len(bm.tokens) - n0 <= most, (op, a, b, len(bm.tokens) - n0)


def test_btrim_only_reads_the_result():
    from chipformer import spec

    cells = [0] * 0x2000
    for i, v0, trims in [(bcd.R_ADDR, 3, True), (bcd.R_ADDR, 1, False), (0x300, 3, False),
                         (bcd.R_ADDR + 2 * bcd.MAX_DIGITS, 3, False)]:
        cells[spec.I], cells[spec.V], cells[spec.PC] = i, v0, 0x210
        w = macro_writes(cells, 0x5008)
        want = [(spec.V, v0 - 1), (spec.I, i + 1)] if trims else [(spec.PC, 0x212)]  # stays: no PC
        assert w == want, (i, v0, w)


@pytest.fixture(scope="module")
def circuit():
    return chip8_q35.build()


OPW = {"add": B.PLUS, "sub": B.MINUS, "mul": B.TIMES}


def inline(bm, op, a, b, opener=()):
    """Qwen writes " a op b =" (the circuit fires if the grammar allows it)."""
    return bm.qwen([*opener, B.SPACE, *(B.HEX[int(ch)] for ch in a), OPW[op], B.SPACE,
                    *(B.HEX[int(ch)] for ch in b), B.EQ])


def test_circuit_macros(circuit):
    """Every case, inline and as a tool call, in one session: records and prints exact."""
    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(3)
    fired = 0
    for op, a, b in CASES:
        fired += len(inline(bm, op, a, b))
        bm.text(2)
        bm.tool_call(op, a, b)
        while not bm.halted:
            bm.step()
        bm.text(2)
    assert fired >= 8
    _n, bad, bad_owner = Q.check_bridge(circuit, bm.tokens, bm.generated, B.owner_expected(bm))
    assert not bad and not bad_owner, (bad[:5], bad_owner[:5])


def test_circuit_random_sessions(circuit):
    rng = random.Random(7)
    fired = 0
    for _ in range(4):
        bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
        bm.text(2)
        for _ in range(3):
            op = rng.choice(["add", "sub", "mul"])
            la, lb = (rng.randint(1, 9), rng.randint(1, 9)) if op == "mul" else (
                rng.randint(1, 20), rng.randint(1, 20))
            a, b = (str(rng.randrange(10 ** n)) for n in (la, lb))
            if rng.random() < 0.5:
                fired += len(inline(bm, op, a, b, rng.choice([(), (B.DOLLAR,), (B.DDOLLAR,)])))
            else:
                bm.tool_call(op, a, b)
                while not bm.halted:
                    bm.step()
            bm.text(3)
        _n, bad, bad_owner = Q.check_bridge(circuit, bm.tokens, bm.generated, B.owner_expected(bm))
        assert not bad and not bad_owner, (bad[:5], bad_owner[:5])
    assert fired >= 2


def test_ops_are_fixed():
    assert bcd.OPS == {"add": 0x202, "sub": 0x206, "mul": 0x20A}
