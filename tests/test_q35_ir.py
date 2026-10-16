"""The q35 IR, emitter and tiny stock-Qwen model agree on a toy memory machine.

Tokens: VAL v (0..255), ADDR a (0..255), READ a, CLS, NOP. ADDR a, VAL v writes cell a; READ a
predicts VAL(cell a), every other token predicts NOP; CLS clears memory and restarts a counter of
VAL tokens, decoded to [w > m] for m < 40.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from chipformer.q35 import ir, tiny
from chipformer.q35.ir import A_BIT, A_REF, Chan, Graph, K

VAL, ADDR, READ, CLS, NOP = 0, 256, 512, 768, 769
VOCAB = list(range(770))
NM = 40
CODE = 10  # bits of the token id


def build():
    g = Graph()
    one = g.one
    f = g.token_feature
    is_val = f("is_val", lambda t: int(t < 256))
    vb = [f(f"vb{i}", lambda t, i=i: (t >> i) & 1 if t < 256 else 0) for i in range(8)]
    an = [[f(f"an{j}[{x}]", lambda t, j=j, x=x: int(256 <= t < 512 and (t - 256) >> 4 * j & 15 == x))
           for x in range(16)] for j in range(2)]
    rn = [[f(f"rn{j}[{x}]", lambda t, j=j, x=x: int(512 <= t < 768 and (t - 512) >> 4 * j & 15 == x))
           for x in range(16)] for j in range(2)]
    is_cls = f("is_cls", lambda t: int(t == CLS))
    is_read = f("is_read", lambda t: int(512 <= t < 768))
    code = [f(f"code{i}", lambda t, i=i: 128 * (2 * ((t >> i) & 1) - 1)) for i in range(CODE)]

    found, bits = [], [[] for _ in range(8)]
    for h in range(2):
        key = {s: Chan(K * (an[0][s & 15] + an[1][(s >> 4) + 8 * h] - one), (0, 1, 0, 0))
               for s in range(128)}
        query = {s: K * (rn[0][s & 15] + rn[1][(s >> 4) + 8 * h] - one) for s in range(128)}
        val = {0: A_REF * one, 1: A_REF * one, **{2 + i: A_BIT * vb[i] for i in range(8)}}
        outs = [(0, 2.0**-10)] + [(2 + i, 2.0**7) for i in range(8)]
        z = {c: K * one for c, _ in outs}
        res = g.head(f"bank{h}", query, key, val, 238 * is_cls, z, outs)
        found.append(res[0])
        for i in range(8):
            bits[i].append(res[1 + i])

    (x,) = g.head("counter",
                  {0: K * (one - is_cls), 1: K * (one - is_cls)},
                  {0: K * is_cls, 1: K * (one - is_cls)},
                  {0: A_REF * (one - is_cls), 1: A_REF * (one - is_cls), 2: A_BIT * is_cls},
                  110 * is_val + 238 * is_cls, {2: K * one}, [(2, 2.0**7)])
    dec = [g.ffn([(1, one, 7 * one - 8 * 2**m * x), (-1, one, 6 * one - 8 * 2**m * x)],
                 f"w>{m}") for m in range(NM)]
    hit = g.ffn([(1, one, found[0] + found[1] + bits[0][0] + bits[0][1] - one)], "found&b0")

    rb = [b[0] + b[1] for b in bits]
    nxt = []
    for i in range(CODE):
        terms = [(1, one, one - is_read)] if (NOP >> i) & 1 else []
        if i < 8:
            terms.append((1, one, is_read + rb[i] - one))
        nxt.append(g.ffn(terms, f"next{i}") if terms else 0 * one)
    S = 2**12
    for i in range(CODE):
        g.ffn([(2 * S, one, nxt[i]), (-S, one, one), (-1, one, code[i]), (1, one, -1 * code[i])],
              f"code{i}", into=code[i])
    return g, {"found": found, "bits": bits, "x": x, "dec": dec, "hit": hit, "code": code}


def make_tokens(n, seed=0):
    rng = np.random.default_rng(seed)
    toks, cells, w = [CLS], {}, 0
    want = {"x": [0.0], "read": [None], "next": [NOP]}
    while len(toks) < n:
        r = rng.random()
        if r < 0.45:
            a, v = int(rng.integers(256)), int(rng.integers(256))
            toks += [ADDR + a, VAL + v]
            cells[a] = v
            want["read"] += [None, None]
            want["next"] += [NOP, NOP]
            w += 1
            want["x"] += [2.0**-(w - 1), 2.0**-w]
        elif r < 0.9:
            a = int(rng.integers(256))
            toks.append(READ + a)
            want["read"].append(cells.get(a))
            want["next"].append(VAL + cells.get(a, 0))
            want["x"].append(2.0**-w)
        elif r < 0.95:
            toks.append(CLS)
            cells, w = {}, 0
            want["read"].append(None)
            want["next"].append(NOP)
            want["x"].append(0.0)
        else:
            toks.append(-1)  # a text token
            want["read"].append(None)
            want["next"].append(None)
            want["x"].append(None)
    return toks, want


@pytest.fixture(scope="module")
def graph():
    return build()


def test_ir_semantics(graph):
    g, d = graph
    toks, want = make_tokens(3000, seed=1)
    m = ir.evaluate(g, toks)
    for t, tok in enumerate(toks):
        if tok < 0:
            continue
        if want["x"][t] is not None and t and want["x"][t] >= 2.0**-100:
            x = ir.ev(d["x"], m[t : t + 1])[0]
            assert x == want["x"][t], (t, x, want["x"][t])
            w = round(-np.log2(x))
            for mm, e in enumerate(d["dec"]):
                assert ir.ev(e, m[t : t + 1])[0] == float(w > mm), (t, mm, w)
        if want["read"][t] is not None:
            v = sum(int(ir.ev(b[0] + b[1], m[t : t + 1])[0]) << i for i, b in enumerate(d["bits"]))
            assert v == want["read"][t]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_tiny_model_matches_ir(graph, dtype):
    g, d = graph
    model = tiny.build(g, VOCAB, [next(iter(c.terms)) for c in d["code"]], dtype=dtype)
    lay = model.layout
    toks, want = make_tokens(1200, seed=2)
    ids = torch.tensor([[t if t >= 0 else len(VOCAB) for t in toks]])  # padding row = text
    m = ir.evaluate(g, toks)
    caught = []
    n_chip = g.n_layers()
    hook = model.model.layers[n_chip - 1].register_forward_hook(lambda mod, i, o: caught.append(o))
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    hook.remove()
    res = caught[0][0].float().numpy().astype(np.float64)
    cols = [lay.dim(i) for i in range(1, len(g.dims))]
    got, ref = res[:, cols], m[:, 1:]
    chip = np.array(toks) >= 0
    if dtype == torch.bfloat16:
        assert np.array_equal(got[chip], ref[chip])
    else:
        assert np.abs(got[chip] - ref[chip]).max() < 1e-3
    pred = logits[:, : len(VOCAB)].argmax(-1).numpy()
    for t, w in enumerate(want["next"]):
        if w is not None:
            assert pred[t] == w, (t, pred[t], w)
