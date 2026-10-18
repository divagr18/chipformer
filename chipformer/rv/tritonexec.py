"""Executor v2: the circuit's step as a few Triton kernels per sublayer, under one CUDA graph.

Same semantics as fastexec.Executor (the IR's arithmetic in fp32, every value exact), but the
weights are kept as sparse rows (CSR) and every sublayer is one fused kernel:

  MLP    one program per block of units: each unit's a and b are sparse dot products with the
         residual, then coef * a * relu(b) is added (atomically) into the output buffer
  heads  one program per value head: its q / k / v / z channels are sparse dot products over the
         lagged residual (the conv taps folded into the weights), then decay, the one-row write
         (key = argmax), the k-hot read normalized by the ballast, the output gate and the output
         columns, added into the output buffer
  apply  residual += output buffer (and the lag history of the next head sublayer moves on)

    Executor2(c, batch).step(x) -> x after the whole circuit
"""

import numpy as np
import torch
import triton
import triton.language as tl

from chipformer.dsl.graph import ONE
from chipformer.q35.ir import FFN, HD, Head


def _csr(rows):
    """CSR arrays (row_ptr, idx, w) and per-block max row length of a list of {col: weight}."""
    rp, ix, w = [0], [], []
    for r in rows:
        for k, v in r.items():
            if v:
                ix.append(k)
                w.append(v)
        rp.append(len(ix))
    if not ix:
        ix, w = [0], [0.0]
    return np.array(rp, np.int32), np.array(ix, np.int32), np.array(w, np.float32)


def _block_max(rp, block):
    n = len(rp) - 1
    lens = np.diff(rp)
    return np.array([lens[i:i + block].max() if i < n else 0 for i in range(0, n, block)] or [0],
                    np.int32)


@triton.jit
def _ffn_kernel(x_ptr, y_ptr, D, U, rpa, ixa, wa, rpb, ixb, wb, coef, outd, kblk,
                BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    bat = tl.program_id(1)
    u = pid * BLOCK + tl.arange(0, BLOCK)
    m = u < U
    xb = x_ptr + bat * D
    a0 = tl.load(rpa + u, mask=m, other=0)
    na = tl.load(rpa + u + 1, mask=m, other=0) - a0
    b0 = tl.load(rpb + u, mask=m, other=0)
    nb = tl.load(rpb + u + 1, mask=m, other=0) - b0
    kk = tl.load(kblk + pid)
    acc_a = tl.zeros([BLOCK], dtype=tl.float32)
    acc_b = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(kk):
        ma = m & (k < na)
        ia = tl.load(ixa + a0 + k, mask=ma, other=0)
        acc_a += tl.load(wa + a0 + k, mask=ma, other=0.0) * tl.load(xb + ia, mask=ma, other=0.0)
        mb = m & (k < nb)
        ib = tl.load(ixb + b0 + k, mask=mb, other=0)
        acc_b += tl.load(wb + b0 + k, mask=mb, other=0.0) * tl.load(xb + ib, mask=mb, other=0.0)
    val = tl.load(coef + u, mask=m, other=0.0) * acc_a * tl.maximum(acc_b, 0.0)
    od = tl.load(outd + u, mask=m, other=0)
    tl.atomic_add(y_ptr + bat * D + od, val, mask=m)


@triton.jit
def _rows(xs, rp, ix, w, base, K, ch):
    """relu of 128 sparse rows (row base + ch) over xs."""
    r0 = tl.load(rp + base + ch)
    n = tl.load(rp + base + ch + 1) - r0
    acc = tl.zeros([128], dtype=tl.float32)
    for k in range(K):
        mk = k < n
        i = tl.load(ix + r0 + k, mask=mk, other=0)
        acc += tl.load(w + r0 + k, mask=mk, other=0.0) * tl.load(xs + i, mask=mk, other=0.0)
    return tl.maximum(acc, 0.0)


@triton.jit
def _head_kernel(xs_ptr, xs_stride, S_ptr, y_ptr, D, V,
                 qrp, qix, qw, qK, krp, kix, kw, kK, vrp, vix, vw, vK, zrp, zix, zw, zK,
                 arp, aix, aw, brp, bix, bw, bconst, odim, ow, khead, decays):
    vh = tl.program_id(0)
    bat = tl.program_id(1)
    ch = tl.arange(0, 128)
    xs = xs_ptr + bat * xs_stride
    h = tl.load(khead + vh)
    k = _rows(xs, krp, kix, kw, h * 128, tl.load(kK + vh), ch)
    v = _rows(xs, vrp, vix, vw, vh * 128, tl.load(vK + vh), ch)
    z = _rows(xs, zrp, zix, zw, vh * 128, tl.load(zK + vh), ch)
    # a and beta: scalar sparse dots (lag 0)
    a0 = tl.load(arp + vh)
    na = tl.load(arp + vh + 1) - a0
    lane = tl.arange(0, 32)
    av = tl.sum(tl.load(aw + a0 + lane, mask=lane < na, other=0.0)
                * tl.load(xs + tl.load(aix + a0 + lane, mask=lane < na, other=0),
                          mask=lane < na, other=0.0), axis=0)
    b0 = tl.load(brp + vh)
    nb = tl.load(brp + vh + 1) - b0
    bv = tl.sum(tl.load(bw + b0 + lane, mask=lane < nb, other=0.0)
                * tl.load(xs + tl.load(bix + b0 + lane, mask=lane < nb, other=0),
                          mask=lane < nb, other=0.0), axis=0) + tl.load(bconst + vh)
    Sb = S_ptr + (bat * V + vh) * 16384
    if tl.load(decays + vh) != 0:
        fac = tl.where(av == 0.0, 1.0, tl.where(av == 110.0, 0.5, 0.0))
        if fac != 1.0:
            for r in range(128):
                row = tl.load(Sb + r * 128 + ch)
                tl.store(Sb + r * 128 + ch, row * fac)
    kmax = tl.max(k, axis=0)
    krow = tl.argmax(k, axis=0)
    if (kmax > 0.0) & (bv == 1.0):
        tl.store(Sb + krow * 128 + ch, v)
    tl.debug_barrier()
    # the read: the sum of the rows whose query is set (multi-hot queries have equal values)
    o = tl.zeros([128], dtype=tl.float32)
    qK_ = tl.load(qK + vh)
    for r0 in range(0, 128, 16):
        rows = r0 + tl.arange(0, 16)
        s0 = tl.load(qrp + h * 128 + rows)
        n = tl.load(qrp + h * 128 + rows + 1) - s0
        acc = tl.zeros([16], dtype=tl.float32)
        for kq in range(qK_):
            mk = kq < n
            i = tl.load(qix + s0 + kq, mask=mk, other=0)
            acc += tl.load(qw + s0 + kq, mask=mk, other=0.0) * tl.load(xs + i, mask=mk, other=0.0)
        hot = acc > 0.0
        tile = tl.load(Sb + rows[:, None] * 128 + ch[None, :])
        o += tl.sum(tl.where(hot[:, None], tile, 0.0), axis=0)
    ref = tl.sum(tl.where(ch < 2, o * o, 0.0), axis=0)
    norm = tl.math.sqrt_rn(ref / 128.0)
    out = tl.where(ref > 0.0, tl.math.div_rn(o, tl.where(ref > 0.0, norm, 1.0)), 0.0)
    wv = tl.load(ow + vh * 128 + ch)
    dd = tl.load(odim + vh * 128 + ch)
    tl.atomic_add(y_ptr + bat * D + dd, out * z * wv, mask=wv != 0.0)


@triton.jit
def _apply_kernel(x_ptr, y_ptr, h_ptr, D, n, HIST: tl.constexpr, BLOCK: tl.constexpr):
    """x += y, y = 0; with HIST, the head sublayer's lag history moves on first (h: [3, D] per
    row, h[0] the newest)."""
    pid = tl.program_id(0)
    bat = tl.program_id(1)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    xo = x_ptr + bat * D + i
    x = tl.load(xo, mask=m, other=0.0)
    if HIST:
        hb = h_ptr + bat * 3 * D + i
        h1 = tl.load(hb + D, mask=m, other=0.0)
        h0 = tl.load(hb, mask=m, other=0.0)
        tl.store(hb + 2 * D, h1, mask=m)
        tl.store(hb + D, h0, mask=m)
        tl.store(hb, x, mask=m)
    yo = y_ptr + bat * D + i
    tl.store(xo, x + tl.load(yo, mask=m, other=0.0), mask=m)
    tl.store(yo, tl.zeros([BLOCK], dtype=tl.float32), mask=m)


class Executor2:
    BLOCK = 64

    def __init__(self, c, batch: int = 1, device="cuda"):
        self.c, self.B, self.dev = c, batch, device
        D = len(c.dims)
        self.D = D
        t = lambda a, dt=torch.int32: torch.tensor(np.asarray(a), dtype=dt, device=device)
        self.steps = []
        for lv in sorted({n.level for n in c.nodes}):
            nodes = [n for n in c.nodes if n.level == lv]
            if isinstance(nodes[0], FFN):
                units = [(float(cf), a, b, n.out) for n in nodes for cf, a, b in n.terms]
                ra = _csr([{d: float(v) for d, v in a.terms.items()} for _, a, _, _ in units])
                rb = _csr([{d: float(v) for d, v in b.terms.items()} for _, _, b, _ in units])
                kb = np.maximum(_block_max(ra[0], self.BLOCK), _block_max(rb[0], self.BLOCK))
                self.steps.append({"kind": "ffn", "U": len(units),
                                   "a": [t(x) if i < 2 else t(x, torch.float32)
                                         for i, x in enumerate(ra)],
                                   "b": [t(x) if i < 2 else t(x, torch.float32)
                                         for i, x in enumerate(rb)],
                                   "coef": t([u[0] for u in units], torch.float32),
                                   "out": t([u[3] for u in units]), "kblk": t(kb)})
                continue
            heads = [n for n in nodes if isinstance(n, Head)]
            vh = [(h, vd) for h, n in enumerate(heads) for vd in n.vheads()]
            V = len(vh)

            def lagged(ch, D=D):
                return {lag * D + d: float(cf) * tw for lag, tw in enumerate(ch.taps) if tw
                        for d, cf in ch.expr.terms.items()}

            qrows, krows = [], []
            for n in heads:
                for s in range(HD):
                    qrows.append(lagged(n.q[s]) if s in n.q else {})
                    krows.append(lagged(n.k[s]) if s in n.k else {})
            vrows, zrows, arows, brows, bconst, odim, ow, khead, decays = ([] for _ in range(9))
            for h, vd in vh:
                for col in range(HD):
                    vrows.append(lagged(vd.v[col]) if col in vd.v else {})
                    zrows.append({d: float(cf) for d, cf in vd.z[col].terms.items()}
                                 if col in vd.z else {})
                arows.append({d: float(cf) for d, cf in vd.a.terms.items()})
                if vd.beta is None:
                    brows.append({})
                    bconst.append(1.0)
                else:
                    brows.append({d: float(cf) for d, cf in vd.beta.terms.items()})
                    bconst.append(0.0)
                od, w_ = np.zeros(HD, np.int32), np.zeros(HD, np.float32)
                for col, d, w in vd.out:
                    od[col], w_[col] = d, w
                odim.append(od)
                ow.append(w_)
                khead.append(h)
                decays.append(int(bool(vd.a.terms)))
            assert max(len(r) for r in arows + brows) <= 32
            q, k, v, z = (_csr(r) for r in (qrows, krows, vrows, zrows))
            a_, b_ = _csr(arows), _csr(brows)

            def kmax(rp, rows):
                """per value head: the longest of its 128 rows (rows: the row block of each)"""
                lens = np.diff(rp).reshape(-1, HD).max(1)
                return np.array([lens[r] for r in rows], np.int32)

            st = {"kind": "head", "V": V, "heads": heads}
            for nm, arr in (("q", q), ("k", k), ("v", v), ("z", z), ("a", a_), ("b", b_)):
                st[nm] = [t(arr[0]), t(arr[1]), t(arr[2], torch.float32)]
            st["qK"] = t(kmax(q[0], khead))
            st["kK"] = t(kmax(k[0], khead))
            st["vK"] = t(kmax(v[0], range(V)))
            st["zK"] = t(kmax(z[0], range(V)))
            st["bconst"] = t(bconst, torch.float32)
            st["odim"], st["ow"] = t(np.concatenate(odim)), t(np.concatenate(ow), torch.float32)
            st["khead"], st["decays"] = t(khead), t(decays)
            st["S"] = torch.zeros(batch, V, HD, HD, device=device)
            st["hist"] = torch.zeros(batch, 4, D, device=device)  # [x_t, x_t-1, x_t-2, x_t-3]
            self.steps.append(st)
        self.y = torch.zeros(batch, D, device=device)
        self.feat_ids = sorted(c.token_fns)

    def load_init(self, row: int = 0):
        for st in self.steps:
            if st["kind"] != "head":
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

    def reset(self, rows=None):
        for st in self.steps:
            if st["kind"] == "head":
                (st["hist"].zero_() if rows is None else st["hist"][rows].zero_())

    def embed(self, toks):
        out = np.zeros((len(toks), self.D))
        out[:, ONE] = 1.0
        for d in self.feat_ids:
            out[:, d] = [float(self.c.feature_value(d, t)) for t in toks]
        return torch.tensor(out, dtype=torch.float32, device=self.dev)

    def step(self, x):
        """x [B, D] (in place: the embedding in, the residual after every node out)."""
        B, D = x.shape[0], self.D
        nblk = triton.cdiv(D, 1024)
        for st in self.steps:
            if st["kind"] == "ffn":
                (rpa, ixa, wa), (rpb, ixb, wb) = st["a"], st["b"]
                _ffn_kernel[(triton.cdiv(st["U"], self.BLOCK), B)](
                    x, self.y, D, st["U"], rpa, ixa, wa, rpb, ixb, wb, st["coef"], st["out"],
                    st["kblk"], BLOCK=self.BLOCK)
                _apply_kernel[(nblk, B)](x, self.y, x, D, D, HIST=False, BLOCK=1024)
                continue
            hist = st["hist"]
            # lag 0 is the current input: hist[:, 0] = x, the older lags already moved on
            hist[:, 1:].copy_(hist[:, :3].clone())
            hist[:, 0].copy_(x)
            args = [hist, 4 * D, st["S"], self.y, D, st["V"]]
            for nm, kn in (("q", "qK"), ("k", "kK"), ("v", "vK"), ("z", "zK")):
                args += st[nm] + [st[kn]]
            args += st["a"] + st["b"] + [st["bconst"], st["odim"], st["ow"], st["khead"],
                                         st["decays"]]
            _head_kernel[(st["V"], B)](*args)
            _apply_kernel[(nblk, B)](x, self.y, x, D, D, HIST=False, BLOCK=1024)
        return x


def check(c, toks, device="cuda") -> int:
    """Executor2 == ir.evaluate on every dim at every position."""
    from chipformer.q35 import ir

    ex = Executor2(c, 1, device)
    ex.load_init()
    emb = ex.embed(toks)
    want = ir.evaluate(c, toks, check=False)
    for t in range(len(toks)):
        x = emb[t:t + 1].clone()
        got = ex.step(x)[0].double().cpu().numpy()
        bad = np.nonzero(got != want[t])[0]
        assert not len(bad), (t, [(c.dims[d].name, got[d], want[t, d]) for d in bad[:5]])
    return len(toks)
