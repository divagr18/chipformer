"""The overlay (overlay.py): circuit dims that only chip tokens need live in Qwen's dims.

IR level: the switched-off circuit predicts exactly what the reference does, and random Qwen
values in the overlaid dims at text positions change nothing else. Weights level: a merge of a
small random Qwen3.5 (the real tokenizer, 0.8B shapes, two layers) keeps Qwen's dims of every
text token bit-identical through the chip layers, clears them on chip tokens, and prints the
reference trace.
"""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from chipformer.q35 import bridge as B
from chipformer.q35 import check as Q
from chipformer.q35 import chip8_q35, overlay
from chipformer.q35.lib import bcd

TOKENIZERS = sorted(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-*/snapshots/*/"
                                     "tokenizer.json"))


def traces() -> list[B.BridgeMachine]:
    from test_q35_inline import CASES, run

    out = [run(toks)[0] for toks, _ in CASES[::3]]
    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(4)
    for op, a, b in [("mul", "37", "43"), ("add", "48213", "9921"), ("div", "355", "113"),
                     ("sub", "5", "675")]:
        bm.tool_call(op, a, b)
        while not bm.halted:
            bm.step()
        bm.text(3)
    out.append(bm)
    rng = random.Random(3)
    pool = [B.HEX[d] for d in range(10)] * 3 + [B.SPACE, B.EQ, B.TIMES, B.DIV, B.PLUS, B.OTHER]
    bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
    bm.text(2)
    for _ in range(6):
        bm.qwen([B.SPACE, B.HEX[4], B.HEX[2], B.HEX[1], B.HEX[7], B.TIMES, B.HEX[6], B.HEX[3],
                 B.EQ] + [rng.choice(pool) for _ in range(12)])
    out.append(bm)
    return out


@pytest.fixture(scope="module")
def overlaid():
    c = chip8_q35.build()
    p = overlay.build(c, set(c.keep), overlay.corpus(docs=12))
    return c, p


def test_plan(overlaid):
    c, p = overlaid
    assert len(p.dims) > len(c.dims) // 2
    assert not p.dims & set(c.keep)
    assert p.q_bound > 100  # far above any Qwen embedding value


def test_circuit_is_unchanged(overlaid):
    c, _ = overlaid
    for bm in traces():
        n, bad, bad_owner = Q.check_bridge(c, bm.tokens, bm.generated, B.owner_expected(bm))
        assert not bad and not bad_owner, (n, bad, bad_owner)


@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_qwen_values_change_nothing(overlaid, scale):
    c, p = overlaid
    rng = np.random.default_rng(int(scale))
    for bm in traces():
        overlay.check_noisy(c, p, bm.tokens, rng, scale)


def test_allocation_uses_qwen_dims(overlaid):
    from chipformer.q35.emit import allocate

    c, p = overlaid
    keep = set(c.keep)
    wide = allocate(chip8_q35.build(), None, keep, c.target.intermediate)
    a = allocate(c, None, keep, c.target.intermediate, overlay=p.dims, overlay_cap=1024)
    assert len(a.final_over) <= 1024 and set(a.over) <= p.dims
    assert a.n_slots < wide.n_slots - 400


def small_qwen(dst: Path, tokenizer: Path) -> None:
    """A random Qwen3.5 with 0.8B shapes (the circuit's target) and two layers."""
    import shutil

    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    cfg = json.loads((tokenizer.parent / "config.json").read_text())
    tc = cfg["text_config"]
    tc.update(hidden_size=1024, intermediate_size=3584, linear_num_value_heads=16,
              num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
              full_attention_interval=2, mtp_num_hidden_layers=0, num_attention_heads=8,
              num_key_value_heads=2, tie_word_embeddings=True)
    cfg["vision_config"].update(depth=1, hidden_size=64, intermediate_size=128, num_heads=2,
                                out_hidden_size=1024)
    cfg["tie_word_embeddings"] = True
    torch.manual_seed(0)
    config = Qwen3_5Config(**{k: v for k, v in cfg.items() if k not in ("architectures",
                                                                       "model_type")})
    model = Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16)
    model.save_pretrained(dst)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (tokenizer.parent / f).exists():
            shutil.copy(tokenizer.parent / f, dst / f)


@pytest.mark.skipif(not TOKENIZERS, reason="needs a Qwen3.5 tokenizer in the HF cache")
def test_merged_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from chipformer.q35 import merge

    src, dst = tmp_path / "src", tmp_path / "merged"
    small_qwen(src, TOKENIZERS[-1])
    c, lay = merge.build_merged(src, dst, mute_qwen=False)
    meta = json.loads((dst / "chipformer.json").read_text())
    assert meta["overlay_slots"] > 0 and lay.hidden < 1024 + meta["slots"] + 2 * lay.m + 400
    cfg, sd = merge.load(dst)
    model = merge.instantiate(cfg, sd, torch.float32)
    lm = model.model.language_model
    v0, n_chip, d0 = meta["chip_token_offset"], meta["chip_layers"], meta["chip_dims_offset"]
    ids = {int(k): v for k, v in meta["text_ids"].items()}
    other = next(i for i in range(1000, 2000) if i not in ids.values())
    for bm in traces()[-2:]:
        toks = [v0 + t if not B.is_text(t) else ids.get(t, other) for t in bm.tokens]
        caught = []
        hook = lm.layers[n_chip - 1].register_forward_hook(
            lambda m, i, o: caught.append(o[0] if isinstance(o, tuple) else o))
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([toks])).logits[0]
        hook.remove()
        res = caught[0][0]
        emb = lm.embed_tokens.weight[torch.tensor(toks)]
        txt = torch.tensor([B.is_text(t) for t in bm.tokens])
        assert torch.equal(res[txt, :d0], emb[txt, :d0])  # Qwen's input, bit for bit
        assert (res[~txt, :d0] == 0).all()  # chip tokens: overlay slots cleared
        gen = np.array(bm.generated[1:] + [False])
        want = np.array(toks[1:] + [-1])
        pred = logits.argmax(-1).numpy()
        assert not (gen & (pred != want)).any(), np.nonzero(gen & (pred != want))[0][:5]
