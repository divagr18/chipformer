"""The mute patch: on chip tokens Qwen's own layers keep their state and write nothing.

A small random Qwen3.5 (text + vision) is widened like a real merge, with ballast dims after
Qwen's. Chip-like tokens carry the ballast; text tokens don't.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from chipformer.q35 import merge
from chipformer.q35.emit import Layout
from chipformer.q35.mute import mute

D0, HIDDEN, M = 128, 256, 1  # 1 * 2^80 / 256 = 4^36
VOCAB = 64


def small_model():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    cfg = Qwen3_5Config(
        text_config={"vocab_size": VOCAB, "hidden_size": D0, "intermediate_size": 256,
                     "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
                     "head_dim": 256, "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                     "linear_key_head_dim": 64, "linear_value_head_dim": 64,
                     "tie_word_embeddings": True},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                       "out_hidden_size": D0},
        tie_word_embeddings=True)
    model = Qwen3_5ForConditionalGeneration(cfg).eval()
    for n, p in model.named_parameters():
        if n.endswith("norm.weight"):
            p.data.normal_(0, 0.1)  # nonzero zero-centred gains, like a trained model
    return model


@pytest.fixture(scope="module")
def models():
    base = small_model()
    cfg = base.config.to_dict()
    sd = {k: v.detach().clone() for k, v in base.state_dict().items() if k != "lm_head.weight"}
    cfg2, sd2 = merge.widen(cfg, sd, HIDDEN)
    plain = merge.instantiate(cfg2, {k: v.clone() for k, v in sd2.items()}, torch.float32)
    lay = Layout(hidden=HIDDEN, off=D0, m=M)
    mute(sd2, cfg2["text_config"], 4, lay.off, lay.ballast * lay.scale)
    muted = merge.instantiate(cfg2, sd2, torch.float32)
    return plain, muted, lay


def embed(model, lay, toks, chip):
    e = model.get_input_embeddings()(torch.tensor([toks]))
    e = e.clone()
    for i, is_chip in enumerate(chip):
        if is_chip:
            e[0, i, :D0] = 0
            e[0, i, lay.off] = lay.ballast
    return e


def run(model, e):
    states = []
    with torch.no_grad():
        out = model(inputs_embeds=e, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    for i, layer in enumerate(cache.layers):
        if hasattr(layer, "recurrent_states") and layer.recurrent_states:
            states.append(layer.recurrent_states[0].clone())
    return out, states


def test_text_is_bit_identical(models):
    plain, muted, lay = models
    toks = list(range(3, 40))
    e = embed(plain, lay, toks, [False] * len(toks))
    a, _ = run(plain, e)
    b, _ = run(muted, e)
    assert torch.equal(a.logits, b.logits)


def test_chip_tokens_keep_qwen_state(models):
    _, muted, lay = models
    text = list(range(3, 23))
    e_text = embed(muted, lay, text, [False] * len(text))
    e_chip = embed(muted, lay, [5] * 30, [True] * 30)
    out, s_text = run(muted, e_text)
    # decode the chip tokens one by one: the recurrent kernel leaves the state bit-identical
    cache = out.past_key_values
    with torch.no_grad():
        for i in range(e_chip.shape[1]):
            o = muted(inputs_embeds=e_chip[:, i : i + 1], past_key_values=cache, use_cache=True,
                      output_hidden_states=True)
            cache = o.past_key_values
            for h in o.hidden_states:  # chip positions add nothing to Qwen's dims
                assert h[0, :, :D0].abs().max() == 0
    s_dec = [layer.recurrent_states[0] for layer in cache.layers
             if hasattr(layer, "recurrent_states") and layer.recurrent_states]
    assert s_text and len(s_dec) == len(s_text)
    for x, y in zip(s_text, s_dec):
        assert torch.equal(x, y), "chip tokens changed a Qwen DeltaNet state"
    # one chunked prefill over text + chip: equal up to the kernel's rounding
    _, s_both = run(muted, torch.cat([e_text, e_chip], 1))
    for x, y in zip(s_text, s_both):
        assert torch.allclose(x, y, rtol=1e-5, atol=1e-7)
