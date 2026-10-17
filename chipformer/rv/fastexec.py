"""A compact executor for a q35 circuit graph (the P0.5 prototype): the circuit's arithmetic
on its live structure only, one decode step per token, captured as a CUDA graph.

Per sublayer, all of a layer's heads run as batched tensors:

  q, k, v   (lagged inputs) @ W, silu == relu, with every matrix cut to the input dims it reads
            and the output columns it writes
  decay     the state times 1 / 0.5 / 0, only for the value heads that ever decay
  write     keys are one-hot: one row scatter per value head (where beta and a key are set)
  read      queries are at most 2-hot: a top-2 row gather, normalized by the ballast columns
  MLP       relu(x[R] @ Wb) * (x[R] @ Wa) @ Wd over the layer's units, written to its dims

This is the IR's semantics in fp32 (every value the circuit produces is exact there), not the
HF modules; ``check`` compares it with ir.evaluate on every dim. It measures how fast the
circuit itself can step when only its nonzero structure is read.
"""

import time

import numpy as np
import torch

from chipformer.dsl.graph import ONE
from chipformer.q35.ir import FFN, HD, REF_COLS, Head


class Lin:
    """y = x @ W for a W [L, D, N] over lagged inputs x [L, D], reading only W's nonzero block:
    the input dims each lag uses and the output columns any lag writes."""

    def __init__(self, w: np.ndarray, device):
        w = w if w.ndim == 3 else w[None]
        self.n = w.shape[2]
        cols = np.nonzero(np.abs(w).sum((0, 1)))[0]
        rows = [np.nonzero(np.abs(w[lag]).sum(1))[0] for lag in range(w.shape[0])]
        self.lags = [lag for lag in range(w.shape[0]) if len(rows[lag])]
        self.rows = [torch.tensor(rows[lag], device=device) for lag in self.lags]
        blocks = [w[lag][rows[lag]][:, cols] for lag in self.lags]
        self.w = torch.tensor(np.concatenate(blocks) if blocks else np.zeros((0, len(cols))),
                              dtype=torch.float32, device=device)
        self.cols = torch.tensor(cols, device=device)
        self.empty = not len(cols) or not self.lags

    def __call__(self, xs):
        """xs: [L, D] (lag 0 first); returns [N]."""
        y = torch.zeros(self.n, device=xs.device)
        if self.empty:
            return y
        xin = torch.cat([xs[lag][r] for lag, r in zip(self.lags, self.rows)])
        y[self.cols] = xin @ self.w
        return y


def _mat(exprs, n_dims):
    w = np.zeros((n_dims, len(exprs)))
    for i, e in enumerate(exprs):
        for d, cf in e.terms.items():
            w[d, i] += float(cf)
    return w


class Executor:
    def __init__(self, c, device="cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False  # fp32 products must stay exact
        self.c, self.dev = c, device
        D = len(c.dims)
        self.D = D
        self.steps = []
        for lv in sorted({n.level for n in c.nodes}):
            nodes = [n for n in c.nodes if n.level == lv]
            if isinstance(nodes[0], FFN):
                terms = [(cf, a, b, n.out) for n in nodes for cf, a, b in n.terms]
                wd = np.zeros((len(terms), D))
                for u, (cf, _, _, out) in enumerate(terms):
                    wd[u, out] = float(cf)
                self.steps.append(("ffn", Lin(_mat([a for _, a, _, _ in terms], D), device),
                                   Lin(_mat([b for _, _, b, _ in terms], D), device),
                                   Lin(wd, device)))
                continue
            heads = [n for n in nodes if isinstance(n, Head)]
            H = len(heads)
            vh = [(h, vd) for h, n in enumerate(heads) for vd in n.vheads()]
            V = len(vh)
            wq, wk = np.zeros((4, D, H * HD)), np.zeros((4, D, H * HD))
            wv = np.zeros((4, D, V * HD))
            wa_, wbeta, wz = np.zeros((D, V)), np.zeros((D, V)), np.zeros((D, V * HD))
            bconst = np.zeros(V)
            wo = np.zeros((V * HD, D))
            for h, n in enumerate(heads):
                for chans, w_ in ((n.q, wq), (n.k, wk)):
                    for s, ch in chans.items():
                        for lag, tw in enumerate(ch.taps):
                            for d, cf in ch.expr.terms.items():
                                w_[lag, d, h * HD + s] += float(cf) * tw
            for j, (h, vd) in enumerate(vh):
                for col, ch in vd.v.items():
                    for lag, tw in enumerate(ch.taps):
                        for d, cf in ch.expr.terms.items():
                            wv[lag, d, j * HD + col] += float(cf) * tw
                for d, cf in vd.a.terms.items():
                    wa_[d, j] += float(cf)
                if vd.beta is None:
                    bconst[j] = 1.0
                else:
                    for d, cf in vd.beta.terms.items():
                        wbeta[d, j] += float(cf)
                for col, e in vd.z.items():
                    for d, cf in e.terms.items():
                        wz[d, j * HD + col] += float(cf)
                for col, d, w in vd.out:
                    wo[j * HD + col, d] += w
            decays = [j for j in range(V) if np.abs(wa_[:, j]).sum()]
            t = lambda a, **kw: torch.tensor(a, device=device, **kw)  # noqa: E731
            self.steps.append({
                "q": Lin(wq, device), "k": Lin(wk, device), "v": Lin(wv, device),
                "a": Lin(wa_[:, decays], device), "decays": t(decays, dtype=torch.long),
                "beta": Lin(wbeta, device), "bconst": t(bconst, dtype=torch.float32),
                "z": Lin(wz, device), "o": Lin(wo, device),
                "kh": t([h for h, _ in vh], dtype=torch.long),
                "ar": torch.arange(V, device=device), "H": H, "V": V, "heads": heads,
                "S": torch.zeros(V, HD, HD, device=device)})
        self.feat_ids = sorted(c.token_fns)

    def load_init(self):
        """The heads' initial states (the snapshot)."""
        for st in self.steps:
            if isinstance(st, tuple):
                continue
            S = st["S"]
            S.zero_()
            j = 0
            for n in st["heads"]:
                init = n.init or []
                for x in range(len(n.vheads())):
                    for row, vec in (init[x] if x < len(init) else {}).items():
                        S[j, row] = torch.tensor(vec, dtype=torch.float32, device=self.dev)
                    j += 1

    def embed(self, toks):
        out = np.zeros((len(toks), self.D))
        out[:, ONE] = 1.0
        for d in self.feat_ids:
            out[:, d] = [float(self.c.feature_value(d, t)) for t in toks]
        return torch.tensor(out, dtype=torch.float32, device=self.dev)

    def reset(self):
        """Fresh per-sequence buffers: lagged inputs per head sublayer, states from the snapshot."""
        self.load_init()
        if getattr(self, "hist", None) is None:
            self.hist = [None if isinstance(st, tuple) else torch.zeros(3, self.D, device=self.dev)
                         for st in self.steps]
        for h in self.hist:  # in place: a captured graph keeps these buffers
            if h is not None:
                h.zero_()

    def step(self, x):
        """One position: x [D] (the embedding row) -> the residual after every node."""
        for i, st in enumerate(self.steps):
            if isinstance(st, tuple):
                _, wa, wb, wd = st
                x1 = x[None]
                x = x + wd((torch.relu(wb(x1)) * wa(x1))[None])
                continue
            H, V, S, kh, ar = st["H"], st["V"], st["S"], st["kh"], st["ar"]
            hist = self.hist[i]
            xs = torch.cat([x[None], hist])  # lags 0..3
            q = torch.relu(st["q"](xs)).view(H, HD)
            k = torch.relu(st["k"](xs)).view(H, HD)
            v = torch.relu(st["v"](xs)).view(V, HD)
            if len(st["decays"]):
                a = st["a"](x[None])
                fac = torch.where(a == 0, 1.0, torch.where(a == 110, 0.5, 0.0))
                S[st["decays"]] *= fac[:, None, None]
            beta = (st["beta"](x[None]) + st["bconst"]) == 1
            kv = k[kh]
            kmax, idx = kv.max(1)
            write = (kmax > 0) & beta
            old = S[ar, idx]
            S[ar, idx] = torch.where(write[:, None], v, old)
            qv, qi = torch.topk(q[kh], 2, dim=1)  # queries are at most 2-hot
            rows = S[ar[:, None], qi] * (qv > 0).float()[:, :, None]
            o = rows.sum(1)
            ref = (o[:, REF_COLS[0]:REF_COLS[-1] + 1] ** 2).sum(1)
            norm = torch.where(ref > 0, torch.sqrt(ref / HD), torch.ones_like(ref))
            o = torch.where((ref > 0)[:, None], o / norm[:, None], torch.zeros_like(o))
            z = torch.relu(st["z"](x[None])).view(V, HD)
            x = x + st["o"](((o * z).reshape(-1))[None])
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
    """Seconds per token (a CUDA graph of one step; its replays are checked against the IR)."""
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
            out = ex.step(xin)
        from chipformer.q35 import ir

        n = min(300, len(emb))
        want = ir.evaluate(c, toks[:n], check=False)
        ex.reset()
        for t in range(n):
            xin.copy_(emb[t])
            g.replay()
            assert (out.double().cpu().numpy() == want[t]).all(), f"graph replay differs at {t}"
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
