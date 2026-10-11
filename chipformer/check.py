"""Differential testing: circuit predictions vs the reference trace, token by token."""

import random
from dataclasses import dataclass

import numpy as np

from chipformer import spec
from chipformer.dsl.batch import evaluate_batch, output_batch
from chipformer.dsl.graph import Graph
from chipformer.ref.emu import fetch, opcode_class
from chipformer.trace import Machine


def run_machine(rom: bytes, steps: int, ipf: int = 10, seed: int = 0,
                key_rate: float = 0.1, quirks: "str | tuple[str, ...] | None" = None) -> Machine:
    """Reference run with a timer tick every ``ipf`` instructions and random key changes."""
    rng = random.Random(seed)
    m = Machine(rom, quirks)
    m.ops = []  # opcode class of each executed instruction, for diagnostics
    for i in range(steps):
        if i % ipf == 0:
            m.tick()
            if rng.random() < key_rate:
                m.set_key(rng.randrange(16), rng.random() < 0.5)
        m.ops.append((len(m.tokens), opcode_class(fetch(m.emu.cells))))
        m.step()
    return m


@dataclass
class Mismatch:
    pos: int  # position whose prediction was wrong (predicts token pos+1)
    want: int
    got: float
    op: str

    def __str__(self) -> str:
        return (f"position {self.pos} in {self.op} record: predicted {self.got} "
                f"({spec.token_name(int(self.got)) if float(self.got).is_integer() else '?'}), "
                f"want {self.want} ({spec.token_name(self.want)})")


def check(graph: Graph, m: Machine, limit: int = 10) -> tuple[int, list[Mismatch]]:
    """Returns (number of checked predictions, first mismatches)."""
    res = evaluate_batch(graph, m.tokens)
    pred = output_batch(graph, "next", res)
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


def verify_trace(tokens: list[int]) -> tuple[int, str | None]:
    """Check a free-running trace against the reference semantics, record by record.

    Works for any key/tick schedule: host-injected key records are applied as
    given, every instruction and tick record must equal what the reference
    emulator computes from the state so far. Returns (records checked, error).
    """
    from chipformer.ref.emu import apply_writes, instruction_writes, state_writes, tick_writes
    from chipformer.trace import parse

    cells = [0] * spec.NUM_CELLS
    records = parse(tokens, allow_reboot=True)
    for i, r in enumerate(records):
        if r.kind == "boot":
            # first boot: the prompt; later boots (rollover) must dump the current state exactly
            expected = r.writes if i == 0 else state_writes(cells)
            cells = [0] * spec.NUM_CELLS
        elif r.kind == "tick":
            expected = tick_writes(cells)
        elif len(r.writes) == 1 and spec.KEYS <= r.writes[0][0] < spec.KEYS + 16:
            expected = r.writes  # injected key change
        else:
            expected = instruction_writes(cells)
        if r.writes != expected:
            op = opcode_class(fetch(cells))
            return i, f"record {i} ({r.kind}, {op}): got {r.writes[:6]}, want {expected[:6]}"
        apply_writes(cells, r.writes)
    return len(records), None
