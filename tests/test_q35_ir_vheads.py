"""The IR's newer head features in real HF modules: a key head's second value head with its own
write gate, value conv taps, and outputs added onto shared dims. The residual stream of a stock
Qwen3_5ForCausalLM equals the IR evaluator's, exactly."""

import random

import numpy as np
import torch

from chipformer.dsl.graph import Expr
from chipformer.q35 import ir
from chipformer.q35.emit import embedding_rows, emit_layers
from chipformer.q35.ir import Chan, Target
from chipformer.q35.prims import Circ
from chipformer.q35.prims import thead
from chipformer.q35.tiny import BASE, pick_layout

N_TOK = 28  # tokens 0..15: data (4 bits); 16..19: write slot s; 20..23 / 24..27: read slot s


def graph():
    c = Circ(Target(key_heads=2, value_heads=4, intermediate=256))
    f = c.token_feature
    bit = [f(f"b{i}", lambda t, i=i: int(t < 16 and t >> i & 1)) for i in range(4)]
    wr = [f(f"w{s}", lambda t, s=s: int(t == 16 + s)) for s in range(4)]
    rd = [f(f"r{s}", lambda t, s=s: int(t == 20 + s)) for s in range(4)]
    rd2 = [f(f"r2.{s}", lambda t, s=s: int(t == 24 + s)) for s in range(4)]
    hi = f("hi", lambda t: int(t in (18, 19, 22, 23, 26, 27)))  # slots 2, 3: second value head
    one = c.one
    # rows = slot s; a write token stores the bits of the two data tokens before it (lags 1, 2)
    data = {2 + 4 * j + i: Chan(bit[i], (0, 1, 0, 0) if j == 0 else (0, 0, 1, 0))
            for j in range(2) for i in range(4)}
    keys = {s: wr[s] for s in range(4)}
    qs = {s: rd[s] for s in range(4)}
    wa = c.and_([sum(wr, Expr()), one - hi], "wa")
    wb = c.and_([sum(wr, Expr()), hi], "wb")
    outs = thead(c, "t", keys, qs, [
        {"data": data, "ref": one, "beta": wa, "gate": one - hi},
        {"data": data, "ref": one, "beta": wb, "gate": hi, "into": "main"}])
    read = outs[0]
    # a second head (one value head, all slots) adds onto the same dims, read by its own tokens
    keys2 = {s: wr[s] for s in range(4)}
    qs2 = {s: rd2[s] for s in range(4)}
    thead(c, "t2", keys2, qs2, [{"data": data, "ref": one, "into": read}])
    out = [c.node(c.t_and([r_]), f"out{i}") for i, r_ in enumerate(read)]
    return c, out


def hf(c):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    lay = pick_layout(len(c.dims))
    n = c.n_layers()
    tc = dict(BASE, hidden_size=lay.hidden, vocab_size=64, num_hidden_layers=n + 1,
              layer_types=["linear_attention"] * n + ["full_attention"],
              linear_num_key_heads=2, linear_num_value_heads=4, intermediate_size=256)
    sd = emit_layers(c, lay, tc, 0, "model.")
    emb = torch.zeros(64, lay.hidden)
    emb[:N_TOK] = embedding_rows(c, lay, list(range(N_TOK)), lay.hidden)
    sd["model.embed_tokens.weight"] = emb
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**tc))
    for p in model.parameters():
        p.data.zero_()
    model.load_state_dict(sd, strict=False)
    return model.to(torch.bfloat16).eval(), lay


def test_vheads_in_hf():
    c, out = graph()
    model, lay = hf(c)
    rng = random.Random(0)
    toks = []
    for _ in range(40):
        toks += [rng.randrange(16), rng.randrange(16), 16 + rng.randrange(4)]
        if rng.random() < 0.6:
            toks.append(rng.choice([20, 24]) + rng.randrange(4))
    m = ir.evaluate(c, toks)
    with torch.no_grad():
        hs = model(input_ids=torch.tensor([toks]), output_hidden_states=True).hidden_states
    h = hs[c.n_layers()][0].double().numpy()
    for e in out:
        (d,) = e.terms
        want = m[:, d]
        assert np.isin(want, [0.0, 1.0]).all() and want.any()
        assert (h[:, lay.dim(d)] == want).all(), (d, np.nonzero(h[:, lay.dim(d)] != want)[0][:5])
