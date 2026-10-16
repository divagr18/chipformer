"""v2 circuit: prints (00FD), the owner bit, and the exit that clears the chip dims."""

from pathlib import Path

import numpy as np
import pytest

from chipformer.q35 import bridge as B
from chipformer.q35 import check as Q
from chipformer.q35 import chip8_q35

ROOT = Path(__file__).resolve().parent.parent
TINY = ROOT / "dist" / "q35-tiny"


def print_rom(digits: list[int]) -> bytes:
    return bytes([0xA2, 0x0A, 0x60, len(digits), 0x00, 0xFD, 0x12, 0x06, 0, 0] + digits)


def print_trace(digits, before=5, after=8):
    bm = B.BridgeMachine(print_rom(digits))
    while not bm.halted:
        bm.step()
    bm.text(after)
    toks = [B.OTHER] * before + bm.tokens
    gen = [False] * before + bm.generated
    owner = [0] * before + B.owner_expected(bm)
    return toks, gen, owner


@pytest.fixture(scope="module")
def circuit():
    return chip8_q35.build()


@pytest.mark.parametrize("digits", [[4, 8, 1, 7], [], list(range(16)), [9] * 58, [3] * 60])
def test_print(circuit, digits):
    toks, gen, owner = print_trace(digits)
    n, bad, bad_owner = Q.check_bridge(circuit, toks, gen, owner)
    assert n >= 7 and not bad and not bad_owner, (bad, bad_owner)


def test_print_text(circuit):
    toks, *_ = print_trace([4, 8, 1, 7])
    printed = [t for t in toks if B.is_text(t) and t != B.OTHER]
    assert printed == [B.NL, B.TR_OPEN, B.NL, *(B.HEX[d] for d in (4, 8, 1, 7)),
                       B.NL, B.TR_CLOSE, B.NL]


@pytest.mark.skipif(not TINY.exists(), reason="run python -m chipformer.q35.tiny first")
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_checkpoint_print_and_exit(circuit, dtype):
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    from chipformer.q35.emit import allocate
    from chipformer.q35.tiny import pick_layout

    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=getattr(torch, dtype)).eval()
    toks, gen, _ = print_trace(list(range(12)))
    caught = []
    last = model.model.layers[circuit.n_layers() - 1]
    hook = last.register_forward_hook(lambda m, i, o: caught.append(o[0] if isinstance(o, tuple)
                                                                     else o))
    with torch.no_grad():
        logits = model(torch.tensor([toks])).logits[0].float()
    hook.remove()
    pred = logits[:, :B.TEXT_BASE + B.N_TEXT].argmax(-1).numpy()
    want = np.array(toks[1:] + [-1])
    g = np.array(gen[1:] + [False])
    assert not (g & (pred != want)).any()
    # after the exit only mu, the output code and tau remain in the chip region
    alloc = allocate(circuit, None, set(circuit.keep), 3584)
    lay = pick_layout(alloc.n_slots + 1)
    lay.slots = alloc.slots
    res = caught[0][0].float()
    kept = {lay.dim(d) for d in circuit.keep}
    others = [i for i in range(lay.hidden) if i not in kept]
    assert res[:, others].abs().max() == 0


@pytest.mark.parametrize("calls", [[("mul", "37", "43")], [("add", "48213", "9921"),
                                                          ("mul", "12", "34")]])
def test_tool_call(circuit, calls):
    from chipformer.q35.lib import bcd

    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(6)
    for op, a, b in calls:
        bm.tool_call(op, a, b)
        while not bm.halted:
            bm.step()
        bm.text(5)
    n, bad, bad_owner = Q.check_bridge(circuit, bm.tokens, bm.generated, B.owner_expected(bm))
    assert n > 100 and not bad and not bad_owner, (bad, bad_owner)


def test_bios_arithmetic():
    import random

    from chipformer.q35.lib import bcd
    from chipformer.ref.emu import apply_writes

    rng = random.Random(1)
    for _ in range(100):
        op = rng.choice(["add", "mul"])
        a, b = str(rng.randrange(10**rng.randint(1, 10))), str(rng.randrange(10**rng.randint(1, 10)))
        bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
        apply_writes(bm.emu.cells, bcd.call_writes(op, a, b))
        while not bm.halted:
            bm.step()
        got = "".join("0123456789abcdef"[B.HEX.index(t)] for t in bm.tokens if t in B.HEX)
        assert got == str(int(a) + int(b) if op == "add" else int(a) * int(b))


def test_bignum_macros(circuit):
    """Random multi-call sessions: BADD / BZERO / BMAC records are predicted exactly."""
    import random

    from chipformer.q35.lib import bcd

    rng = random.Random(11)
    for _ in range(6):
        bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
        bm.text(3)
        for _ in range(2):
            op = rng.choice(["add", "mul"])
            a, b = str(rng.randrange(10**rng.randint(1, 12))), str(rng.randrange(10**rng.randint(1, 12)))
            bm.tool_call(op, a, b)
            while not bm.halted:
                bm.step()
            bm.text(4)
        _n, bad, bad_owner = Q.check_bridge(circuit, bm.tokens, bm.generated, B.owner_expected(bm))
        assert not bad and not bad_owner, (bad, bad_owner)
