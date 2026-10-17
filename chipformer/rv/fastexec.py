"""A compact executor for a q35 circuit graph (the P0.5 prototype): the circuit's arithmetic
on its live dims only, one decode step per token, captured as a CUDA graph.

Per sublayer, all of a layer's heads run as batched tensors:

  q, k, v   (lagged inputs) @ W, silu == relu; keys are one-hot, so a write is a row scatter
  decay     the state times 1 / 0.5 / 0 per value head
  read      the (multi-)hot query rows summed: a gather; normalized by the ballast columns
  MLP       relu(x @ Wb) * (x @ Wa) @ Wd over the layer's units

This is the IR's semantics in fp32 (every value the circuit produces is exact there), not the
HF modules; ``check`` compares it with ir.evaluate. It measures how fast the circuit itself can
step when only its nonzero structure is read.
"""

import time

import numpy as np
import torch

from chipformer.dsl.graph import ONE
from chipformer.q35.ir import DECAY, FFN, HD, REF_COLS, Head


def _mat(exprs, n_dims, scale=1.0):
    """[n_dims, len(exprs)] with exprs[i]'s coefficients in column i."""
    w = np.zeros((n_dims, len(exprs)))
    for i, e in enumerate(exprs):
        for d, cf in e.terms.items():
            w[d, i] += float(cf) * scale
    return w


class Executor:
    def __init__(self, c, device="cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False  # fp32 products must stay exact
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.c, self.dev = c, device
        D = len(c.dims)
        self.D = D
        levels = sorted({n.level for n in c.nodes})
        self.steps = []
        f = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        for lv in levels:
            nodes = [n for n in c.nodes if n.level == lv]
            if isinstance(nodes[0], FFN):
                terms = [(cf, a, b, n.out) for n in nodes for cf, a, b in n.terms]
                wa = _mat([a for _, a, _, _ in terms], D)
                wb = _mat([b for _, _, b, _ in terms], D)
                wd = np.zeros((len(terms), D))
                for u, (cf, _, _, out) in enumerate(terms):
                    wd[u, out] = float(cf)
                self.steps.append(("ffn", f(wa), f(wb), f(wd)))
                continue
            heads = [n for n in nodes if isinstance(n, Head)]
            H = len(heads)
            vh = [(h, x, vd) for h, n in enumerate(heads) for x, vd in enumerate(n.vheads())]
            V = len(vh)
            wq = np.zeros((4, D, H * HD))
            wk = np.zeros((4, D, H * HD))
            wv = np.zeros((4, D, V * HD))
            wa_, wbeta, wz = np.zeros((D, V)), np.zeros((D, V)), np.zeros((D, V * HD))
            beta_const = np.zeros(V)
            wo = np.zeros((V * HD, D))
            for h, n in enumerate(heads):
                for ch_, w_ in ((n.q, wq), (n.k, wk)):
                    for s, ch in ch_.items():
                        for lag, tw in enumerate(ch.taps):
                            if tw:
                                for d, cf in ch.expr.terms.items():
                                    w_[lag, d, h * HD + s] += float(cf) * tw
            for j, (h, x, vd) in enumerate(vh):
                for col, ch in vd.v.items():
                    for lag, tw in enumerate(ch.taps):
                        if tw:
                            for d, cf in ch.expr.terms.items():
                                wv[lag, d, j * HD + col] += float(cf) * tw
                for d, cf in vd.a.terms.items():
                    wa_[d, j] += float(cf)
                if vd.beta is None:
                    beta_const[j] = 1.0
                else:
                    for d, cf in vd.beta.terms.items():
                        wbeta[d, j] += float(cf)
                for col, e in vd.z.items():
                    for d, cf in e.terms.items():
                        wz[d, j * HD + col] += float(cf)
                for col, d, w in vd.out:
                    wo[j * HD + col, d] += w
            kh = torch.tensor([h for h, _, _ in vh], device=device)
            init = torch.zeros(V, HD, HD, device=device)
            self.steps.append(["heads", f(wq), f(wk), f(wv), f(wa_), f(wbeta), f(beta_const),
                               f(wz), f(wo), kh, H, V, init, heads])
        self.feat_ids = sorted(c.token_fns)

    def load_init(self):
        """The heads' initial states (the snapshot)."""
        for st in self.steps:
            if st[0] != "heads":
                continue
            S = st[12]
            S.zero_()
            j = 0
            for n in st[13]:
                init = n.init or []
                for x in range(len(n.vheads())):
                    for row, vec in (init[x] if x < len(init) else {}).items():
                        S[j, row] = torch.tensor(vec, dtype=torch.float32, device=self.dev)
                    j += 1

    def embed(self, toks):
        c = self.c
        out = np.zeros((len(toks), self.D))
        out[:, ONE] = 1.0
        for d in self.feat_ids:
            out[:, d] = [float(c.feature_value(d, t)) for t in toks]
        return torch.tensor(out, dtype=torch.float32, device=self.dev)

    def reset(self):
        """Fresh per-sequence buffers: lagged inputs per head sublayer, states from the snapshot."""
        self.load_init()
        self.hist = [torch.zeros(3, self.D, device=self.dev) if st[0] == "heads" else None
                     for st in self.steps]

    def step(self, x):
        """One position: x [D] (the embedding row) -> the residual after every node."""
        for i, st in enumerate(self.steps):
            if st[0] == "ffn":
                _, wa, wb, wd = st
                x = x + (torch.relu(x @ wb) * (x @ wa)) @ wd
                continue
            _, wq, wk, wv, wa, wbeta, bconst, wz, wo, kh, H, V, S, _ = st
            hist = self.hist[i]
            xs = torch.stack([x, hist[0], hist[1], hist[2]])  # lags 0..3
            q = torch.relu(torch.einsum("ld,ldk->k", xs, wq)).view(H, HD)
            k = torch.relu(torch.einsum("ld,ldk->k", xs, wk)).view(H, HD)
            v = torch.relu(torch.einsum("ld,ldk->k", xs, wv)).view(V, HD)
            a = x @ wa
            fac = torch.where(a == 0, 1.0, torch.where(a == 110, 0.5, 0.0))
            beta = ((x @ wbeta + bconst) == 1).float()
            S.mul_(fac[:, None, None])
            kv = k[kh]  # [V, 128] (one-hot or zero)
            w = (kv > 0).float() * beta[:, None]  # rows to overwrite
            S.copy_(S * (1 - w[:, :, None]) + w[:, :, None] * v[:, None, :])
            qv = (q[kh] > 0).float()
            o = torch.einsum("vr,vrc->vc", qv, S)
            ref = (o[:, list(REF_COLS)] ** 2).sum(1)
            norm = torch.where(ref > 0, torch.sqrt(ref / HD), torch.ones_like(ref))
            o = torch.where((ref > 0)[:, None], o / norm[:, None], torch.zeros_like(o))
            z = torch.relu(x @ wz).view(V, HD)
            x = x + (o * z).reshape(-1) @ wo
            hist[2].copy_(hist[1])
            hist[1].copy_(hist[0])
            hist[0].copy_(xs[0])
        return x


def check(c, toks, device="cpu") -> int:
    """Executor == ir.evaluate on every dim at every position (exact)."""
    from chipformer.q35 import ir

    ex = Executor(c, device)
    ex.reset()
    emb = ex.embed(toks)
    want = ir.evaluate(c, toks, check=False)
    for t in range(len(toks)):
        got = ex.step(emb[t]).double().cpu().numpy()
        bad = np.nonzero(got != want[t])[0]
        assert not len(bad), (t, [(c.dims[d].name, got[d], want[t, d]) for d in bad[:5]])
    return len(toks)


def timed(c, toks, steps=500, graph=True) -> float:
    """Seconds per token (CUDA graph of one step)."""
    ex = Executor(c, "cuda")
    ex.reset()
    emb = ex.embed(toks[:steps + 20])
    xin = torch.zeros(ex.D, device="cuda")
    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                xin.copy_(emb[0])
                ex.step(xin)
        torch.cuda.current_stream().wait_stream(s)
        ex.reset()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ex.step(xin)
        ex.reset()
        run = g.replay
    else:
        def run():
            ex.step(xin)
    for t in range(20):
        xin.copy_(emb[t])
        run()
    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(20, 20 + steps):
        xin.copy_(emb[t])
        run()
    torch.cuda.synchronize()
    return (time.time() - t0) / steps
