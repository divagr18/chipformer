"""Differential testing of the q35 circuit against the reference trace, token by token."""

import numpy as np

from chipformer import spec
from chipformer.check import Mismatch
from chipformer.q35 import ir
from chipformer.q35.chip8_q35 import Chip8Q35, decode_code
from chipformer.trace import Machine


def predictions(c: Chip8Q35, tokens: list[int], check: bool = True) -> np.ndarray:
    """Next-token prediction at every position, from the circuit's output bits."""
    m = ir.evaluate(c, tokens, check)
    bits = np.stack([ir.ev(b, m) for b in c.next_bits], 1)
    assert np.isin(bits, [0.0, 1.0]).all(), "output bits must be 0/1"
    return np.array([decode_code(row) for row in bits.astype(int).tolist()]), m


def check(c: Chip8Q35, m: Machine, limit: int = 10, exact: bool = True):
    """(number of checked predictions, first mismatches)."""
    pred, _ = predictions(c, m.tokens, exact)
    want = np.array(m.tokens[1:] + [-1])
    gen = np.array(m.generated[1:] + [False])
    bad = np.nonzero(gen & (pred != want))[0]
    starts = [p for p, _ in getattr(m, "ops", [])]
    out = []
    for p in bad[:limit]:
        i = np.searchsorted(starts, p + 1, side="right") - 1
        op = m.ops[i][1] if i >= 0 and starts else "?"
        out.append(Mismatch(int(p), int(want[p]), float(pred[p]), op))
    return int(gen.sum()), out


def describe(mm: list[Mismatch]) -> str:
    return "\n".join(f"  {x}" for x in mm) or "  none"


__all__ = ["check", "describe", "predictions", "spec"]


def check_bridge(c: Chip8Q35, tokens: list[int], generated: list[bool], owner_want, limit=10):
    """Predictions at generated positions, and the owner bit everywhere it is defined."""
    pred, m = predictions(c, tokens)
    owner = ir.ev(c.owner, m)
    want = np.array(tokens[1:] + [-1])
    gen = np.array(generated[1:] + [False])
    bad = [(int(p), int(want[p]), int(pred[p])) for p in np.nonzero(gen & (pred != want))[0]]
    bad_owner = [(p, float(owner[p]), w) for p, w in enumerate(owner_want)
                 if w is not None and owner[p] != w]
    return int(gen.sum()), bad[:limit], bad_owner[:limit]
