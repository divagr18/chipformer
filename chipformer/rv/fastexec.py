"""A compact executor for a q35 circuit graph: the circuit's arithmetic on its live structure
only, batch rows, one decode step per token, captured as a CUDA graph.

Per sublayer, all of a layer's heads run as batched tensors:

  q, k, v   (lagged inputs) @ W, silu == relu, with every matrix cut to the input dims it reads
            and the output columns it writes; one matmul per sublayer
  decay     the state times 1 / 0.5 / 0, only for the value heads that ever decay
  write     keys are one-hot: one row scatter per value head (where beta and a key are set)
  read      queries are at most 2-hot: a top-2 row gather, normalized by the ballast columns
  MLP       relu(x[R] @ Wb) * (x[R] @ Wa) @ Wd over the layer's units, written to its dims

This is the IR's semantics in fp32 (every value the circuit produces is exact there), not the
HF modules; ``check`` compares it with ir.evaluate on every dim. ``FreeRun`` lets it generate:
each step decodes the output code into the next token and builds that token's embedding on the
GPU, so a whole run is replays of one graph.
"""

import time

import numpy as np
import torch

from chipformer.dsl.graph import ONE
from chipformer.q35.ir import FFN, HD, REF_COLS, Head


class Lin:
    """y = x @ W for a W [L, D, N] over lagged inputs xs [B, L, D], reading only W's nonzero
    block: the input dims each lag uses and the output columns any lag writes."""

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
        """xs: [B, L, D] (lag 0 first); returns [B, N]."""
        y = torch.zeros(xs.shape[0], self.n, device=xs.device)
        if self.empty:
            return y
        xin = torch.cat([xs[:, lag][:, r] for lag, r in zip(self.lags, self.rows)], 1)
        y[:, self.cols] = xin @ self.w
        return y


def _mat(exprs, n_dims):
    w = np.zeros((n_dims, len(exprs)))
    for i, e in enumerate(exprs):
        for d, cf in e.terms.items():
            w[d, i] += float(cf)
    return w


class Executor:
    def __init__(self, c, device="cuda", batch: int = 1):
        torch.backends.cuda.matmul.allow_tf32 = False  # fp32 products must stay exact
        self.c, self.dev, self.B = c, device, batch
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
                wab = np.concatenate([_mat([a for _, a, _, _ in terms], D),
                                      _mat([b for _, _, b, _ in terms], D)], 1)
                self.steps.append(("ffn", Lin(wab, device), len(terms), Lin(wd, device)))
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
            lag0 = np.zeros((4, D, len(decays) + V + V * HD))  # a, beta, z read lag 0 only
            lag0[0] = np.concatenate([wa_[:, decays], wbeta, wz], 1)
            proj = np.concatenate([wq, wk, wv, lag0], 2)  # one matmul for the sublayer
            nq, nv, nd = H * HD, V * HD, len(decays)
            self.steps.append({
                "proj": Lin(proj, device), "split": [nq, nq, nv, nd, V, nv],
                "decays": torch.tensor(decays, device=device, dtype=torch.long),
                "bconst": torch.tensor(bconst, device=device, dtype=torch.float32),
                "o": Lin(wo, device), "kh": torch.tensor([h for h, _ in vh], device=device),
                "H": H, "V": V, "heads": heads,
                "S": torch.zeros(batch, V, HD, HD, device=device)})
        self.feat_ids = sorted(c.token_fns)
        self.bi = torch.arange(batch, device=device)
        self.hist = [None if isinstance(st, tuple) else torch.zeros(batch, 3, D, device=device)
                     for st in self.steps]

    def load_init(self, row: int = 0):
        """The heads' initial states (the snapshot currently in the graph) into batch row ``row``."""
        for st in self.steps:
            if isinstance(st, tuple):
                continue
            S = st["S"][row]
            S.zero_()
            j = 0
            for n in st["heads"]:
                init = n.init or []
                for x in range(len(n.vheads())):
                    for r, vec in (init[x] if x < len(init) else {}).items():
                        S[j, r] = torch.tensor(vec, dtype=torch.float32, device=self.dev)
                    j += 1

    def embed(self, toks):
        out = np.zeros((len(toks), self.D))
        out[:, ONE] = 1.0
        for d in self.feat_ids:
            out[:, d] = [float(self.c.feature_value(d, t)) for t in toks]
        return torch.tensor(out, dtype=torch.float32, device=self.dev)

    def reset(self, rows=None):
        """Fresh lagged inputs for the rows (in place: a captured graph keeps these buffers)."""
        for h in self.hist:
            if h is not None:
                (h.zero_() if rows is None else h[rows].zero_())

    def step(self, x):
        """One position: x [B, D] (the embedding rows) -> the residual after every node."""
        B, bi = x.shape[0], self.bi[:x.shape[0]]
        for i, st in enumerate(self.steps):
            if isinstance(st, tuple):
                _, wab, nu, wd = st
                ab = wab(x[:, None])
                x = x + wd((torch.relu(ab[:, nu:]) * ab[:, :nu])[:, None])
                continue
            V, S, kh = st["V"], st["S"], st["kh"]
            hist = self.hist[i]
            xs = torch.cat([x[:, None], hist], 1)  # lags 0..3
            q, k, v, a, bt, z = torch.split(st["proj"](xs), st["split"], 1)
            q, k, v = (torch.relu(e).view(B, -1, HD) for e in (q, k, v))
            if len(st["decays"]):
                fac = torch.where(a == 0, 1.0, torch.where(a == 110, 0.5, 0.0))
                S[:, st["decays"]] *= fac[:, :, None, None]
            beta = (bt + st["bconst"]) == 1
            kv = k[:, kh]
            kmax, idx = kv.max(2)
            write = (kmax > 0) & beta
            vi = torch.arange(V, device=x.device)[None, :]
            old = S[bi[:, None], vi, idx]
            S[bi[:, None], vi, idx] = torch.where(write[:, :, None], v, old)
            qv, qi = torch.topk(q[:, kh], 2, dim=2)  # queries are at most 2-hot
            rows = S[bi[:, None, None], vi[:, :, None], qi] * (qv > 0).float()[..., None]
            o = rows.sum(2)
            ref = (o[..., REF_COLS[0]:REF_COLS[-1] + 1] ** 2).sum(2)
            norm = torch.where(ref > 0, torch.sqrt(ref / HD), torch.ones_like(ref))
            o = torch.where((ref > 0)[..., None], o / norm[..., None], torch.zeros_like(o))
            z = torch.relu(z).view(B, V, HD)
            x = x + st["o"](((o * z).reshape(B, -1))[:, None])
            hist.copy_(xs[:, :3])
        return x


def check(c, toks, device="cpu") -> int:
    """Executor == ir.evaluate on every dim at every position (exact)."""
    from chipformer.q35 import ir

    ex = Executor(c, device)
    ex.load_init()
    emb = ex.embed(toks)
    want = ir.evaluate(c, toks, check=False)
    for t in range(len(toks)):
        got = ex.step(emb[t:t + 1])[0].double().cpu().numpy()
        bad = np.nonzero(got != want[t])[0]
        assert not len(bad), (t, [(c.dims[d].name, got[d], want[t, d]) for d in bad[:5]])
    return len(toks)


class FreeRun:
    """Generation on the GPU: each step decodes the circuit's output code (core.OUT_NAMES) into
    the next token and writes that token's embedding as the next input, inside one graph."""

    def __init__(self, c, batch: int = 1, device="cuda", compile_=True):
        from chipformer.rv import core as C

        self.c, self.B, self.dev = c, batch, device
        self.ex = Executor(c, device, batch)
        if compile_:
            self.ex.step = torch.compile(self.ex.step, mode="max-autotune-no-cudagraphs",
                                         dynamic=False)
        self.fam_dims = [self._dim(e) for e in c.out_fam]
        self.pay_dims = [self._dim(e) for e in c.out_pay]
        # token embeddings are linear in the payload bits within a family
        nf, nb = len(C.OUT_NAMES), C.NPAY
        base, bits = np.zeros((nf, self.ex.D)), np.zeros((nf, nb, self.ex.D))
        ids = np.zeros(nf, dtype=np.int64)
        for f, nm in enumerate(C.OUT_NAMES):
            from chipformer.rv import spec as S

            t0 = S.tok(nm) if nm in S.CONTROLS else S.tok(nm, 0)
            ids[f] = t0
            base[f] = self.ex.embed([t0])[0].cpu().numpy()
            if nm not in S.CONTROLS:
                for i in range(S.FAMILIES[nm][1].bit_length() - 1):
                    bits[f, i] = self.ex.embed([t0 + (1 << i)])[0].cpu().numpy() - base[f]
        t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
        self.base, self.bits, self.ids = t(base), t(bits), t(ids, torch.long)
        self.x = torch.zeros(batch, self.ex.D, device=device)
        self.tok = torch.zeros(batch, dtype=torch.long, device=device)
        self.graph = None

    @staticmethod
    def _dim(e):
        ((d, w),) = e.terms.items()
        assert w == 1
        return d

    def load(self, row: int, first_token: int):
        """Row ``row`` starts from the snapshot now in the graph's tables, then first_token."""
        self.ex.load_init(row)
        self.ex.reset([row])
        self.x[row] = self.ex.embed([first_token])[0]

    def _step(self):
        out = self.ex.step(self.x)
        fam = (torch.round(out[:, self.fam_dims]) * (2 ** torch.arange(len(self.fam_dims),
                                                                        device=self.dev))).sum(1)
        pay = torch.round(out[:, self.pay_dims])
        fid = fam.long().clamp(0, len(self.ids) - 1)
        self.tok.copy_(self.ids[fid] + (pay * 2 ** torch.arange(pay.shape[1], device=self.dev)
                                        ).sum(1).long())
        self.x.copy_(self.base[fid] + torch.einsum("bi,bid->bd", pay, self.bits[fid]))

    def capture(self):
        x0 = self.x.clone()
        states = [st["S"].clone() for st in self.ex.steps if not isinstance(st, tuple)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step()
        # undo the warm-up steps
        self.x.copy_(x0)
        for st, s0 in zip([st for st in self.ex.steps if not isinstance(st, tuple)], states):
            st["S"].copy_(s0)
        self.ex.reset()
        self.graph = g

    def run(self, n_steps: int) -> np.ndarray:
        """[B, n_steps] generated tokens."""
        cuda = self.dev != "cpu"
        if cuda and self.graph is None:
            self.capture()
        out = torch.zeros(self.B, n_steps, dtype=torch.long, device=self.dev)
        for t in range(n_steps):
            self.graph.replay() if cuda else self._step()
            out[:, t].copy_(self.tok)
        if cuda:
            torch.cuda.synchronize()
        return out.cpu().numpy()


def timed(c, toks, steps=500, graph=True, compile_=False, batch=1) -> float:
    """Seconds per step of B rows (teacher-forced; the graph's replays are checked against the
    IR first)."""
    ex = Executor(c, "cuda", batch)
    for b in range(batch):
        ex.load_init(b)
    if compile_:
        ex.step = torch.compile(ex.step, mode="max-autotune-no-cudagraphs", dynamic=False)
    emb = ex.embed(toks[:steps + 20])
    xin = torch.zeros(batch, ex.D, device="cuda")
    states = [st["S"].clone() for st in ex.steps if not isinstance(st, tuple)]

    def restore():
        for st, s0 in zip([st for st in ex.steps if not isinstance(st, tuple)], states):
            st["S"].copy_(s0)
        ex.reset()

    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                xin.copy_(emb[0].expand(batch, -1))
                ex.step(xin)
        torch.cuda.current_stream().wait_stream(s)
        restore()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = ex.step(xin)
        from chipformer.q35 import ir

        n = min(300, len(emb))
        want = ir.evaluate(c, toks[:n], check=False)
        restore()
        for t in range(n):
            xin.copy_(emb[t].expand(batch, -1))
            g.replay()
            got = out.double().cpu().numpy()
            assert (got == want[t][None]).all(), f"graph replay differs at {t}"
        restore()
        run = g.replay
    else:
        def run():
            ex.step(xin)
    for t in range(20):
        xin.copy_(emb[t].expand(batch, -1))
        run()
    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(20, 20 + steps):
        xin.copy_(emb[t].expand(batch, -1))
        run()
    torch.cuda.synchronize()
    return (time.time() - t0) / steps
