"""The CHIP-8 circuit and its compiled weights vs the reference emulator."""

import random
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from chipformer.check import check, run_machine
from chipformer.chip8 import build
from chipformer.compile import compile_graph
from chipformer.dense import forward
from chipformer.dsl.batch import evaluate_batch
from chipformer.dsl.evaluate import Evaluator
from chipformer.fuzz import random_program, random_quirks
from chipformer.session import Session
from chipformer.trace import Machine

ROMS = Path(__file__).resolve().parent.parent / "roms"
PONG = ROMS / "pong.ch8"
TEST_ROMS = sorted((ROMS / "test").glob("*.ch8")) if (ROMS / "test").exists() else []


@cache
def circuit():
    return build()


def assert_exact(m: Machine) -> None:
    n, bad = check(circuit(), m)
    assert n > 0
    assert not bad, "\n".join(map(str, bad))


@pytest.mark.parametrize("seed", range(6))
def test_fuzz_programs(seed):
    quirks = random_quirks(seed)
    assert_exact(run_machine(random_program(seed), 1500, seed=seed, key_rate=0.5, quirks=quirks))


@pytest.mark.skipif(not PONG.exists(), reason="roms/pong.ch8 not provided")
@pytest.mark.parametrize("quirks", ["chip48", "octo", "cosmac"])
def test_pong(quirks):
    assert_exact(run_machine(PONG.read_bytes(), 2000, seed=1, key_rate=0.3, quirks=quirks))


@pytest.mark.parametrize("rom", TEST_ROMS, ids=lambda p: p.name)
def test_rom(rom):
    assert_exact(run_machine(rom.read_bytes(), 1500, seed=2, key_rate=0.3))


def test_dense_transformer_is_bit_identical():
    g = circuit()
    w, config = compile_graph(g)
    m = run_machine(random_program(7), 80, seed=7, key_rate=0.5)
    r, logits = forward(w, config, m.tokens)
    assert np.array_equal(r, evaluate_batch(g, m.tokens))
    gen = np.array(m.generated[1:])
    assert np.array_equal(logits.argmax(1)[:-1][gen], np.array(m.tokens[1:])[gen])


def test_free_running_generation():
    rom, quirks = random_program(11), random_quirks(11)
    e = Evaluator(circuit())
    s = Session(lambda tok: int(e.output("next", e.feed(tok))), rom, quirks)
    m = Machine(rom, quirks)
    rng = random.Random(0)
    for i in range(25):
        if i % 10 == 0:
            s.tick(), m.tick()
            k, down = rng.randrange(16), rng.random() < 0.5
            s.set_key(k, down), m.set_key(k, down)
        s.step(), m.step()
    assert s.tokens == m.tokens
    assert s.cells == m.emu.cells
