"""The Hugging Face export: loads through transformers and runs games token-exact."""

from functools import cache

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from chipformer.chip8 import build
from chipformer.compile import compile_graph
from chipformer.dense import forward
from chipformer.hf_export import GAMES, example_game, export
from chipformer.trace import Machine


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    out = tmp_path_factory.mktemp("hf")
    export(out)
    return transformers.AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True,
                                                             dtype=torch.float64)


@cache
def dense_weights():
    return compile_graph(build())


def run_both(model, frames: int, sparse: bool):
    """The same input schedule through the transformer and the reference emulator."""
    g = example_game()
    rom = (GAMES / g["file"]).read_bytes()
    quirks = ",".join(g["quirks"])
    game = model.play(rom, quirks=quirks, ipf=g["ipf"], keep_tokens=True, sparse=sparse)
    ref = Machine(rom, quirks)
    keys = g["keys"]
    for f in range(frames):
        held = [keys["a"]] if f % 30 == 2 else [keys["left"]] if 5 <= f % 30 < 9 else \
            [keys["down"]] if f % 30 >= 20 else []
        game.frame(held)
        for k in range(16):
            ref.set_key(k, k in held)
        ref.tick()
        for _ in range(g["ipf"]):
            ref.step()
    return game, ref


def test_loads_as_causal_lm(model):
    assert type(model).__name__ == "ChipformerForCausalLM"
    assert next(model.parameters()).dtype == torch.float64
    assert sum(int((p != 0).sum()) for p in model.parameters()) == dense_weights()[1]["nonzero_params"]


def test_matches_numpy_dense_forward(model):
    w, config = dense_weights()
    tokens = Machine((GAMES / example_game()["file"]).read_bytes()).tokens[:400]
    _, want = forward(w, config, tokens)
    model.sparse(False)
    with torch.inference_mode():
        got = model(torch.tensor([tokens])).logits[0].numpy()
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-6)
    assert (got.argmax(1) == want.argmax(1)).all()


@pytest.mark.parametrize("sparse", [True, False])
def test_game_trace_is_exact(model, sparse):
    game, ref = run_both(model, 40 if sparse else 8, sparse)
    assert len(game.tokens) > len(ref.tokens[:1]) and game.tokens == ref.tokens
    assert game.cells == ref.emu.cells


def test_rollover_keeps_running(model):
    model.config.rollover_at = 2500
    try:
        game, ref = run_both(model, 60, True)
    finally:
        model.config.rollover_at = 250_000
    assert game.rollovers >= 1
    assert game.cells == ref.emu.cells
