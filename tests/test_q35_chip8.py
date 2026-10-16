"""CHIP-8 as a q35 circuit, and as a stock Qwen3.5 checkpoint.

The circuit is checked against the reference trace token by token with the exact IR evaluator
(which also asserts bf16 exactness of every intermediate). The checkpoint test needs
``python -m chipformer.q35.tiny`` (dist/q35-tiny) and loads it with AutoModelForCausalLM.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from chipformer.check import run_machine, verify_trace
from chipformer.fuzz import random_program, random_quirks
from chipformer.q35 import check as Q
from chipformer.q35 import chip8_q35

ROOT = Path(__file__).resolve().parent.parent
TINY = ROOT / "dist" / "q35-tiny"
GAMES = {g["id"]: g for g in json.loads((ROOT / "games" / "manifest.json").read_text(
    encoding="utf-8"))["games"]}


@pytest.fixture(scope="module")
def circuit():
    return chip8_q35.build()


def game(gid, steps, seed=2):
    g = GAMES[gid]
    rom = (ROOT / "games" / g["file"]).read_bytes()
    return run_machine(rom, steps, ipf=g["ipf"], seed=seed, key_rate=0.3, quirks=tuple(g["quirks"]))


@pytest.mark.parametrize("rom", ["3-corax+.ch8", "4-flags.ch8", "5-quirks.ch8"])
@pytest.mark.parametrize("quirks", ["octo", "chip48", "cosmac"])
def test_test_roms(circuit, rom, quirks):
    m = run_machine((ROOT / "roms" / "test" / rom).read_bytes(), 300, ipf=10, seed=1,
                    quirks=quirks)
    n, bad = Q.check(circuit, m)
    assert n > 1000 and not bad, Q.describe(bad)


@pytest.mark.parametrize("seed", range(4))
def test_fuzz(circuit, seed):
    m = run_machine(random_program(seed), 300, ipf=7, seed=seed, key_rate=0.3,
                    quirks=random_quirks(seed))
    _, bad = Q.check(circuit, m)
    assert not bad, Q.describe(bad)


@pytest.mark.skipif(not TINY.exists(), reason="run python -m chipformer.q35.tiny first")
@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_checkpoint_prefill(dtype):
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=getattr(torch, dtype)).eval()
    m = game("brix", 150)
    with torch.no_grad():
        pred = model(torch.tensor([m.tokens])).logits[0, :, :8708].argmax(-1).numpy()
    want = np.array(m.tokens[1:] + [-1])
    gen = np.array(m.generated[1:] + [False])
    assert gen.sum() > 500
    assert not (gen & (pred != want)).any()


@pytest.mark.skipif(not TINY.exists(), reason="run python -m chipformer.q35.tiny first")
def test_checkpoint_free_running():
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM

    from chipformer.q35 import host

    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.bfloat16).eval()
    g = GAMES["tetris"]
    s = host.session(model, (ROOT / "games" / g["file"]).read_bytes(), tuple(g["quirks"]))
    for i in range(24):
        if i % g["ipf"] == 0:
            s.tick()
        if i == 8:
            s.set_key(5, True)
        s.step()
    n, err = verify_trace(s.tokens)
    assert err is None, err
    assert n >= 28
