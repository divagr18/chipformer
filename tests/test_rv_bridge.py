"""The P3 bridge (rv/bridge.py) in the IR: a tool call's text is captured, </tool_call> starts a
run, putchar output is echoed as Qwen tokens; the circuit must predict every token it owns and
own exactly the positions the reference says."""

from pathlib import Path

import numpy as np
import pytest

from chipformer.q35 import ir
from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S

CALC = Path(__file__).parent / "data" / "rv" / "c-calc.elf"


@pytest.fixture(scope="module")
def T():
    try:
        return BR.Text(BR.default_tokenizer())
    except FileNotFoundError:
        pytest.skip("no Qwen3.5 tokenizer in the HF cache")


@pytest.fixture(scope="module")
def circuit(T):
    return core.build(text=T)


def call(T, expr: str) -> list[int]:
    return T.encode(f"<tool_call>\n<function=calc>\n<parameter=expression>\n{expr}\n"
                    "</parameter>\n</function>\n</tool_call>")


def session(T):
    m = ref.Machine.from_program(elf.program(str(CALC)))
    BR.park(m)
    return BR.Session(T, m), m


def check(c, ses, m0, skip: set[int] = frozenset()):
    """IR == reference: owner at every position, the next token wherever the circuit owns it
    (positions in ``skip``: owned, but the stream is prefilled history)."""
    core.snapshot(c, m0)
    toks = ses.toks
    carry, mats = {}, []
    for a in range(0, len(toks), 2048):
        mats.append(ir.evaluate(c, toks[a:a + 2048], carry=carry))
    mat = np.concatenate(mats)
    pred = core.predictions(c, mat)
    own = np.rint(ir.ev(c.owner, mat)).astype(int)
    assert [i for i in range(len(toks)) if own[i] != int(ses.own[i])] == []
    bad = [i for i in range(len(toks) - 1)
           if ses.own[i] and i not in skip and pred[i] != toks[i + 1]]
    assert bad == [], [(i, toks[i], pred[i], toks[i + 1]) for i in bad[:5]]


def fresh_machine():
    m = ref.Machine.from_program(elf.program(str(CALC)))
    BR.park(m)
    return m


@pytest.mark.parametrize("expr,want", [("12*34", 408), ("2*(3+4)-50", -36),
                                       ("99999999999*99999999999", 99999999999**2)])
def test_calc_call(T, circuit, expr, want):
    m0 = fresh_machine()
    ses = BR.Session(T, fresh_machine())
    ses.qwen(T.encode("<|im_start|>assistant\n"))
    ses.qwen(call(T, expr))
    ses.qwen(T.encode("So the answer is"))
    out = T.decode_bytes(ses.runs[-1]).decode()
    assert out == f"\n<tool_response>\n{want}\n</tool_response>\n<think>\n\n</think>\n\n"
    check(circuit, ses, m0)


def test_history_then_call(T, circuit):
    """A prefilled earlier call (its response as text) must not start a run, and the next real
    call still works (the buffer is cleared at <tool_call>)."""
    m0 = fresh_machine()
    ses = BR.Session(T, fresh_machine())
    hist = call(T, "1+1") + T.encode("\n<tool_response>\n2\n</tool_response>\nIt is 2. ")
    k = len(ses.toks)
    ses.history(hist)
    skip = {k + hist.index(T.TCC)}
    ses.qwen(call(T, "7*6"))
    assert T.decode_bytes(ses.runs[-1]).decode().split("\n")[2] == "42"
    check(circuit, ses, m0, skip)


def test_two_calls_and_overflow(T, circuit):
    """Two runs in one stream; a call longer than the buffer is truncated (its later tokens
    all land in the last row), and the circuit still matches the reference."""
    m0 = fresh_machine()
    ses = BR.Session(T, fresh_machine())
    ses.qwen(call(T, "3*3"))
    ses.qwen(T.encode(" and "))
    long = call(T, "+".join(["1"] * 60))
    assert len(long) > S.CAP_MAX + 2
    ses.qwen(long)
    outs = [T.decode_bytes(r).decode().split("\n")[2] for r in ses.runs]
    assert outs[0] == "9" and outs[1] != "60"
    check(circuit, ses, m0)


def test_reference_buffer(T):
    ses, m = session(T)
    ses.qwen(call(T, "5"))
    rows = [r.rstrip(bytes(1)) for r in m.text[1:S.CAP_MAX + 1] if r.strip(bytes(1))]
    assert b"".join(rows).startswith(b"\n<function=calc>")
