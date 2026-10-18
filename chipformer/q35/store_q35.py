"""The object store (store.py) as part of the q35 circuit.

Built in two parts from ``chip8_q35.build``:

``capture``  token features of Qwen's rows (their id bits), what is inside a store tool call
             (the function, the name and text parameters), the name buffer, the content heads'
             writes (as Qwen writes the text), and ``S``: 1 from a store call's </tool_call> to
             the end of its answer. The rest of the circuit is gated off while S is 1.
``respond``  the answer: chip records (register and device-port writes) and the printed
             <tool_response>. Its cases join the circuit's output cases; prints of arbitrary
             tokens ("any-prints") go out through the id code instead of the class code.

All tables are Gated DeltaNet heads with one-hot keys (exact overwrites):

  content  rows of 8 tokens: key head = 16 rows per chunk, its two value heads (Qwen3.5-4B)
           hold 4 tokens each, written token by token with conv taps (lags 0..3) so a row's
           last write carries its whole group; 4 tokens per row with one value head
  meta     chunk -> (length, next, has next)
  dir      entry -> (name tokens, name length, alive), dht: entry -> (first chunk, last chunk)

Every register and table address is a set of 2-bit one-hot groups: a key or query channel is
the sum of its groups' literals minus (n - 1), exactly 1 on its own row and <= 0 elsewhere.
"""

from fractions import Fraction

from chipformer import spec
from chipformer.dsl.graph import Expr
from chipformer.q35 import bridge as B
from chipformer.q35 import store as S
from chipformer.q35.ir import A_BIT, A_REF, REF_COLS, Chan, K
from chipformer.q35.prims import from_bits, onehot_bits

CODE_AMP = 128  # the id code in a row's output dims: CODE_AMP * (2 * bit - 1)


def sel(c, e):
    """Key / query pre-activation K * (2e - 1) for a 0/1 (or <= 0 when false) form e."""
    if isinstance(e, Chan):
        return Chan(e.expr * (2 * K) - c.one * K, e.taps)
    return Chan(e * (2 * K) - c.one * K)


def chan(e, scale):
    if isinstance(e, Chan):
        return Chan(e.expr * scale, e.taps)
    return Chan(e * scale)


def thead(c, name, keys, queries, vheads, tiny=0.0):
    """A head of 0/1 tables: one dict per value head with data {column: 0/1 form or Chan},
    ref (1 where a written row carries the ballast), optional gate (0/1: its outputs), beta
    (write gate) and into (dims to add the outputs onto). Returns each value head's outputs."""
    specs = []
    for x in vheads:
        v = {col: chan(e, A_BIT) for col, e in x["data"].items()}
        for col in REF_COLS:
            v[col] = chan(x["ref"], A_REF)
        gate = c.one if x.get("gate") is None else x["gate"]
        cols = sorted(x["data"])
        specs.append({"v": v, "a": x.get("a", 0), "z": {col: K * gate for col in cols},
                      "outs": [(col, 2.0**7) for col in cols], "beta": x.get("beta"),
                      "into": x.get("into")})
    first, rest = specs[0], specs[1:]
    outs = c.head(name, {s: sel(c, e) for s, e in queries.items()},
                  {s: sel(c, e) for s, e in keys.items()}, first["v"], first["a"], first["z"],
                  first["outs"], tiny=tiny, beta=first["beta"], into=first["into"], vx=rest)
    res, i = [], 0
    for x in specs:
        n = len(x["outs"])
        res.append(outs[i:i + n])
        i += n
    return res


def groups(c, bits, name):
    """2-bit one-hot groups of an integer's bits: groups[k][v] = [bits 2k..2k+1 == v]."""
    out = []
    for k in range(0, len(bits), 2):
        bb = bits[k:k + 2]
        oh = []
        for v in range(1 << len(bb)):
            lits = [b if v >> i & 1 else c.one - b for i, b in enumerate(bb)]
            oh.append(c.and_(lits, f"{name}.g{k // 2}=={v}"))
        out.append(oh)
    return out


def addr_lits(gr, value):
    """The group literals of one address (groups of 2 bits, the last may have 1)."""
    nbits = 2 * len(gr) - (len(gr[-1]) == 2)
    assert value >> nbits == 0, (value, nbits)
    return [g[value >> 2 * k & (len(g) - 1)] for k, g in enumerate(gr)]


def match_form(lits):
    """1 when all literals are 1, <= 0 otherwise (a key / query channel or AND-form)."""
    return sum(lits, Expr()) - (len(lits) - 1)


class Store:
    """Handles between the two parts."""


def capture(c, vocab: "S.Vocab", in_call, tco, tcc, n_chunks: int | None = None) -> Store:
    one = c.one
    st = Store()
    st.vocab = vocab
    r_v = c.target.value_heads // c.target.key_heads
    st.tpr = 8 if r_v >= 2 else 4  # tokens per content row
    st.n_chunks = n_chunks or (640 if r_v >= 2 else 256)
    f = c.token_feature
    qid = S.qid_of

    # --- features of Qwen's rows: the id bits, and the tokens the capture looks for
    def fq(name, pred):
        return f(name, lambda t: int(qid(t) is not None and pred(qid(t))), full=True)

    st.xb = [f(f"id.b{i}", lambda t, i=i: S.id_code(qid(t))[i], full=True)
             for i in range(S.ID_BITS)]
    st.yc = [f(f"id.code{i}", lambda t, i=i: 0 if qid(t) is None or qid(t) >= S.SPECIAL
               else CODE_AMP * (2 * S.id_code(qid(t))[i] - 1), full=True)
             for i in range(S.ID_BITS)]
    nl = fq("st.nl", vocab.has_nl)
    f_eq, f_fn = fq("st.eq", lambda q: q == S.Q_EQ), fq("st.function", lambda q: q == S.Q_FUNCTION)
    f_par = fq("st.parameter", lambda q: q == S.Q_PARAMETER)
    f_lts = fq("st.lt_slash", lambda q: q == S.Q_LTSLASH)
    op_bare = {op: fq(f"st.op.{op}", lambda q, i=i: q == i) for op, i in S.OP_BARE.items()}
    op_join = {op: fq(f"st.op={op}", lambda q, i=i: q == i) for op, i in S.OP_JOINED.items()}
    p_bare = {p: fq(f"st.p.{p}", lambda q, i=i: q == i) for p, i in S.PARAM_BARE.items()}
    p_join = {p: fq(f"st.p={p}", lambda q, i=i: q == i) for p, i in S.PARAM_JOINED.items()}
    cells = list(S.REGS.values()) + list(S.PORTS.values())
    st.at = {a: f(f"st.at@{a:x}", lambda t, a=a: int(t == spec.addr_token(a))) for a in cells}
    st_addr = sum(st.at.values(), Expr())

    # --- previous tokens (conv taps), one head
    T = c.text
    prev_cols = {2: Chan(f_eq, (0, 1, 0, 0)), 3: Chan(f_fn, (0, 1, 0, 0)),
                 4: Chan(f_fn, (0, 0, 1, 0)), 5: Chan(f_par, (0, 1, 0, 0)),
                 6: Chan(f_par, (0, 0, 1, 0)), 7: Chan(f_lts, (0, 1, 0, 0)),
                 8: Chan(T[B.TR_CLOSE], (0, 1, 0, 0))}
    for i, a in enumerate(cells):
        prev_cols[9 + i] = Chan(match_form(c.addr_lits(a)), (0, 1, 0, 0))
    outs = c.slot_head("st.prev", {0: one}, {0: one}, prev_cols, one)
    prev_eq, prev_fn, prev2_fn, prev_par, prev2_par, prev_lts, prev_trc = outs[:7]
    st.pv = dict(zip(cells, outs[7:]))  # at a VAL token: its write's cell is a

    # --- inside a tool call: the function, the parameters and their values
    op_t = {}
    for op in S.OPS:
        terms = c.t_and([op_bare[op], prev_eq, prev2_fn, in_call])
        if op in op_join:
            terms += c.t_and([op_join[op], prev_fn, in_call])
        op_t[op] = c.node(terms, f"st.op_tok.{op}")
    op_any = sum(op_t.values(), Expr())
    pn, pt = (c.node(c.t_and([p_join[p], prev_par, in_call])
                     + c.t_and([p_bare[p], prev_eq, prev2_par, in_call]), f"st.param.{p}")
              for p in ("name", "text"))
    closer = c.and_([f_par, prev_lts, in_call], "st.closer")
    ops = c.slot_head("st.op", {0: tco + op_any}, {0: one},
                      {2 + i: op_t[op] for i, op in enumerate(S.OPS)}, tco + op_any)
    st.op = dict(zip(S.OPS, ops))
    pk_name, pk_text = c.slot_head("st.pk", {0: tco + pn + pt + closer}, {0: one},
                                   {2: pn, 3: pt}, tco + pn + pt + closer)
    x_nl = counter(c, "st.nlc", tco + pn + pt, nl)
    n1, n2 = (c.node(ge_terms(c, x_nl, tco + pn + pt, m), f"st.nlc>={m}") for m in (1, 2))

    def value(pk, name):
        return c.node(c.t_and([pk, n2, in_call]) + c.t_and([pk, n1, one - n2, one - nl, in_call]),
                      name)

    st.val_name, st.val_text = value(pk_name, "st.val.name"), value(pk_text, "st.val.text")

    # name: up to NMAX tokens in a buffer (each in its own column block, all read at once), and
    # its length (counted with the closer's "</")
    nreset = tco + pn
    x_ni = counter(c, "st.ni", nreset, st.val_name)
    ni = [c.node(ge_terms(c, x_ni, nreset, n + 1) + neg(ge_terms(c, x_ni, nreset, n + 2)),
                 f"st.ni=={n}") for n in range(S.NMAX)]  # [this is name token n]
    st.nlen = count_bits(c, x_ni, nreset, S.NMAX + 2, 3, "st.nlen")
    data = {}
    for n in range(S.NMAX):
        for b in range(S.ID_BITS):
            data[2 + S.ID_BITS * n + b] = match_form([st.xb[b], ni[n]])
    keys = {n: match_form([st.val_name, ni[n]]) for n in range(S.NMAX)}
    keys[127] = nreset
    # rows of name tokens carry no ballast; the reset writes row 127 with it, so a read of all
    # rows has a constant norm. The reset also clears the buffer (decay 0).
    st.nbuf = thead(c, "st.nbuf", keys, {s: one for s in keys},
                    [{"data": data, "ref": Chan(nreset), "a": 238 * nreset}])[0]
    nbuf_tok = [st.nbuf[S.ID_BITS * n:S.ID_BITS * (n + 1)] for n in range(S.NMAX)]
    st.nbuf_tok = nbuf_tok

    # text: index of each value token, its length
    treset = tco + pt
    x_ci = counter(c, "st.ci", treset, st.val_text)
    st.ibits = count_bits(c, x_ci, treset, S.CMAX + 1, 7, "st.i")  # max(count - 1, 0)
    st.tlen = st.ibits  # after the text: its length (the closer's "</" was counted)

    # --- registers (cells between the keys and the display; never written by a boot). The
    # ballast row is rewritten at every text token (and BOOT), so the store needs no BIOS.
    bal = c.is_boot + c.is_text
    keys, rdata, col, blocks = {127: Chan(bal)}, {}, 2, {}
    for slot, (rname, cell) in enumerate(S.REGS.items()):
        keys[slot] = c.write_key(cell)
        lits = c.addr_lits(cell)
        nb = S.REG_BITS[cell]
        blocks[rname] = list(range(col, col + nb))
        for i in range(nb):
            rdata[col] = Chan(c.vb[i] + sum(lits, Expr()) - one * Fraction(len(lits), 2),
                              (1, 1, 0, 0))
            col += 1
    out = c.slot_head("st.regs", keys, {s: one for s in keys}, rdata, Chan(bal))
    by_col = dict(zip(sorted(rdata), out))
    st.reg = {rname: [by_col[k] for k in cols] for rname, cols in blocks.items()}

    # --- S: from a store call's </tool_call> to the end of its answer (the NL after its
    # </tool_response>)
    fin = c.and_([T[B.NL], prev_trc], "st.final_nl")
    op_set = sum(st.op.values(), Expr())
    store_tcc = c.and_([tcc, op_set], "st.call_end")
    (st.S,) = c.slot_head("st.S", {0: tcc + tco + fin}, {0: one}, {2: store_tcc},
                          tcc + tco + fin)

    # --- the content heads: written at each text token, chunk NEXTG, index i < CMAX (the
    # index saturates beyond, so a longer text keeps its first CMAX tokens)
    content = c.node(c.t_and([st.val_text]) + neg(c.t_and([st.val_text], a=c.node(
        ge_terms(c, x_ci, treset, S.CMAX + 1), "st.ci>CMAX"))), "st.content")
    g_bits = st.reg["NEXTG"]
    gw = groups(c, g_bits, "st.nextg")
    tpr = st.tpr
    lo = tpr.bit_length() - 1  # index bits inside a row: 3 (8 per row) or 2
    iw = groups(c, st.ibits[lo:], "st.irow")
    slot_oh = [c.and_([b if s >> i & 1 else one - b for i, b in enumerate(st.ibits[:2])],
                      f"st.islot=={s}") for s in range(4)]
    e = [[c.and_([st.xb[b], slot_oh[s], content], f"st.e{s}.{b}") for b in range(S.ID_BITS)]
         for s in range(4)]
    if tpr == 8:
        beta = [c.and_([content, one - st.ibits[2]], "st.beta.a"),
                c.and_([content, st.ibits[2]], "st.beta.b")]
    st.content_w = dict(content=content, gw=gw, iw=iw, e=e, beta=beta if tpr == 8 else None)
    return st


def neg(terms):
    return [(-coef, a, b) for coef, a, b in terms]


XS = 26  # counters come out as x = 2^(XS - n): large n stays a normal float in the residual and
# the thresholds' gate weights 2^(m + 2 - XS) * K / (norm scale ~2^-36) <= 2^124 stay inside bf16
# (m <= 106), while x^2 <= 2^52 vanishes next to the ballast's square (~2^84) of every RMSNorm


def counter(c, name, reset, count, tiny=2.0**-118):
    """x = 2^(XS - n), n = tokens with count = 1 since the last reset (0 at the reset itself).

    The row written at a reset decays by half at each counted token; a ballast row is
    rewritten at every other token, so a read is normed by a constant. n <= 112 keeps the
    normed 2^-(n + 14) a normal float."""
    one = c.one
    q = {0: one - reset, 1: one - reset}
    k = {0: reset, 1: one - reset}
    v = {2: Chan(reset * A_BIT)}
    for col in REF_COLS:
        v[col] = Chan((one - reset) * A_REF)
    (x,) = c.head(name, {s: sel(c, e) for s, e in q.items()}, {s: sel(c, e) for s, e in k.items()},
                  v, 110 * count + 238 * reset, {2: K * one}, [(2, 2.0**(7 + XS))], tiny=tiny)
    c.big.add(next(iter(x.terms)))  # up to 2^XS: see emit.allocate
    return x


def ge_terms(c, x, reset, m):
    """Unit terms of [n >= m] (m >= 1) from a counter's x = 2^(XS - n) (0 at reset tokens)."""
    one = c.one
    k = 8 * Fraction(2) ** (m - 1 - XS)
    return [(1, one - reset, 7 * one - k * x), (-1, one - reset, 6 * one - k * x)]


def count_bits(c, x, reset, nmax, nbits, name):
    """Bits of max(min(n, nmax) - 1, 0), straight from x (no thermometer dims)."""
    out = []
    for b in range(nbits):
        terms = []
        for v in range(2, nmax + 1):  # [n >= v] adds bit b of (v - 1) minus that of (v - 2)
            d = ((v - 1) >> b & 1) - ((v - 2) >> b & 1)
            if d:
                terms += [(d * co, a_, b_) for co, a_, b_ in ge_terms(c, x, reset, v)]
        out.append(c.node(terms, f"{name}.b{b}"))
    return out


def inc_bits(c, bits, name):
    """Bits of (v + 1) mod 2^n."""
    one = c.one
    out = [one - bits[0]]
    for i in range(1, len(bits)):
        out.append(c.node(c.t_and([bits[i]]) + c.t_and(bits[:i]) + c.t_and([bits[i], *bits[:i]], -2),
                          f"{name}.b{i}"))
    return out


def respond(c, st: Store, case_list, cnt, jeq, tcc):
    """The answer's cases (appended to the circuit's), and the any-print outputs."""
    one = c.one
    T = c.text
    Sx = st.S
    op = st.op
    reg = st.reg
    pv, at = st.pv, st.at
    is_text, is_addr, is_val = c.is_text, c.is_addr, c.is_val
    trc = T[B.TR_CLOSE]

    def case(lits, kind, payload):
        case_list.append(([Sx, *lits], kind, payload))

    def cls_code(cls):
        from chipformer.q35.chip8_q35 import TEXT_CODE
        return TEXT_CODE + cls - B.TEXT_BASE

    # flags since the call's end: a GCUR write (the get's chain started), an SK write in a
    # list (its body started), a DIRW write (delete found its entry)
    def flag(name, at_tok, data=None):
        (x,) = c.slot_head(name, {0: tcc + at_tok}, {0: one}, {2: at_tok if data is None else data},
                           tcc + at_tok)
        return x

    chain = flag("st.chain", pv[S.GCUR])
    lst = flag("st.listing", pv[S.SK], match_form([pv[S.SK], op["list"]]))  # (a search writes SK too)
    wrote = flag("st.wrote", pv[S.DIRW])

    # H: text tokens since the call's end or the last store ADDR token
    st_addr = sum(at.values(), Expr())
    hreset = tcc + st_addr
    x_h = counter(c, "st.H", hreset, is_text - tcc)
    hbits = count_bits_n(c, x_h, hreset, S.CMAX + 2, 7, "st.H")
    hv = from_bits(hbits)

    def heq(v):
        return c.node(c.t_eq(hv, v, 0, 127), f"st.H=={v}")

    hs = {v: heq(v) for v in range(8)}

    # --- tables read during the answer
    sk_g = groups(c, reg["SK"], "st.sk")
    # dir: entry SK -> name, length, alive; written at DIRW's VAL token
    dir_data = {}
    for n in range(S.NMAX):
        for b in range(S.ID_BITS):
            dir_data[2 + S.ID_BITS * n + b] = st.nbuf_tok[n][b]
    for b in range(3):
        dir_data[2 + S.ID_BITS * S.NMAX + b] = st.nlen[b]
    dir_data[2 + S.ID_BITS * S.NMAX + 3] = c.vb[0]
    rows = range(S.NFILES_MAX + 1)
    dkeys = {k: match_form([pv[S.DIRW], *addr_lits(sk_g, k)]) for k in rows}
    dq = {k: match_form([Sx, *addr_lits(sk_g, k)]) for k in rows}
    d_out = thead(c, "st.dir", dkeys, dq, [{"data": dir_data, "ref": one}])[0]
    d_name = [d_out[S.ID_BITS * n:S.ID_BITS * (n + 1)] for n in range(S.NMAX)]
    d_len = d_out[S.ID_BITS * S.NMAX:S.ID_BITS * S.NMAX + 3]
    d_alive = d_out[S.ID_BITS * S.NMAX + 3]
    # dirH / dirT (first and last chunk of the entry)
    vb10 = c.vb[:10]
    if st.tpr == 8:
        hkeys = {k: match_form([pv[S.DIRH] + pv[S.DIRT], *addr_lits(sk_g, k)]) for k in rows}
        dh, dt = thead(c, "st.dht", hkeys, dq,
                       [{"data": {2 + i: b for i, b in enumerate(vb10)}, "ref": one,
                         "beta": pv[S.DIRH]},
                        {"data": {2 + i: b for i, b in enumerate(vb10)}, "ref": one,
                         "beta": pv[S.DIRT]}])
    else:
        dh = thead(c, "st.dirH", {k: match_form([pv[S.DIRH], *addr_lits(sk_g, k)]) for k in rows},
                   dq, [{"data": {2 + i: b for i, b in enumerate(vb10)}, "ref": one}])[0]
        dt = thead(c, "st.dirT", {k: match_form([pv[S.DIRT], *addr_lits(sk_g, k)]) for k in rows},
                   dq, [{"data": {2 + i: b for i, b in enumerate(vb10)}, "ref": one}])[0]

    # meta: chunk -> length, next, has next. Written at METAW (chunk NEXTG: length from the VAL
    # token) and METAN (chunk TMP: length TMPL, next from the VAL token); read at TMPL's ADDR
    # token (chunk TMP) and during a get's print (chunk GCUR)
    wm, wn = pv[S.METAW], pv[S.METAN]
    ws = [c.node(c.t_and([g_, one - wn]) + c.t_and([t_, wn]), f"st.meta.w.b{i}")
          for i, (g_, t_) in enumerate(zip(reg["NEXTG"], reg["TMP"]))]
    at_tmpl = at[S.TMPL]
    qsb = [c.node(c.t_and([g_, one - at_tmpl]) + c.t_and([t_, at_tmpl]), f"st.meta.q.b{i}")
           for i, (g_, t_) in enumerate(zip(reg["GCUR"], reg["TMP"]))]
    get_print = c.and_([Sx, chain, one - is_addr], "st.get_print")
    q_act = c.and_([Sx, chain], "st.in_chain") + at_tmpl  # (the chain's ADDR(GCUR) reads next)
    wsg, qsg = groups(c, ws[:7], "st.meta.w"), groups(c, qsb[:7], "st.meta.q")
    whi, qhi = groups(c, ws[7:], "st.meta.wh"), groups(c, qsb[7:], "st.meta.qh")
    m_len = [c.node(c.t_and([c.vb[i], wm]) + c.t_and([reg["TMPL"][i], wn]), f"st.meta.len.b{i}")
             for i in range(7)]
    m_data = {2 + i: b for i, b in enumerate(m_len)}
    for i in range(10):
        m_data[9 + i] = c.and_([c.vb[i], wn], f"st.meta.next.b{i}")
    m_data[19] = wn
    m_rows = None
    n_meta = -(-st.n_chunks // 128)
    for h in range(n_meta):
        keys = {r: match_form([wm + wn, *addr_lits(wsg, r), *addr_lits(whi, h)]) for r in range(128)}
        qs = {r: match_form([q_act, *addr_lits(qsg, r), *addr_lits(qhi, h)]) for r in range(128)}
        o = thead(c, f"st.meta{h}", keys, qs, [{"data": m_data, "ref": one, "into": m_rows}])[0]
        m_rows = m_rows or o
    mr_len, mr_next, mr_has = m_rows[0:7], m_rows[7:17], m_rows[17]

    # content: written in capture's terms; read at chunk GCUR, index H
    cw = st.content_w
    tpr = st.tpr
    lo = tpr.bit_length() - 1
    rpc = S.CHUNK // tpr  # rows per chunk
    cph = 128 // rpc  # chunks per head
    n_heads = -(-st.n_chunks // cph)
    gr = groups(c, reg["GCUR"], "st.gcur")
    hr = groups(c, hbits[lo:], "st.hrow")
    hslot = [c.and_([b if s_ >> i & 1 else one - b for i, b in enumerate(hbits[:2])],
                    f"st.hslot=={s_}") for s_ in range(4)]
    cdata = {2 + S.ID_BITS * s_ + b: Chan(cw["e"][s_][b], (1, 1, 1, 1))
             for s_ in range(4) for b in range(S.ID_BITS)}
    c_rows = None
    for h in range(n_heads):
        keys, qs = {}, {}
        for r in range(128):
            g = h * cph + r // rpc
            if g >= st.n_chunks:
                continue
            ir = r % rpc
            keys[r] = match_form([cw["content"], *addr_lits(cw["gw"], g), *addr_lits(cw["iw"], ir)])
            qs[r] = match_form([get_print, *addr_lits(gr, g), *addr_lits(hr, ir)])
        if tpr == 8:  # the second value head (tokens 4..7 of a row) adds onto the same dims
            vh = [{"data": cdata, "ref": one, "beta": cw["beta"][0], "gate": one - hbits[2],
                   "into": c_rows},
                  {"data": cdata, "ref": one, "beta": cw["beta"][1], "gate": hbits[2],
                   "into": c_rows or "main"}]
            o = thead(c, f"st.content{h}", keys, qs, vh)
            c_rows = c_rows or o[0]
        else:
            o = thead(c, f"st.content{h}", keys, qs, [{"data": cdata, "ref": one, "into": c_rows}])
            c_rows = c_rows or o[0]
    c_tok = [c_rows[S.ID_BITS * s_:S.ID_BITS * (s_ + 1)] for s_ in range(4)]

    # --- comparisons
    mism = []
    for n in range(S.NMAX):
        for b in range(S.ID_BITS):
            a_, b_ = d_name[n][b], st.nbuf_tok[n][b]
            mism += [(1, one, a_ - b_), (1, one, b_ - a_)]
    for b in range(3):
        mism += [(1, one, d_len[b] - st.nlen[b]), (1, one, st.nlen[b] - d_len[b])]
    n_mism = c.node(mism, "st.mismatch")
    match = c.node(c.t_eq(n_mism, 0, 0, 3 * S.NMAX * S.ID_BITS, lits=[d_alive]), "st.match")
    sk, nf = from_bits(reg["SK"]), from_bits(reg["NFILES"])
    end = c.node(c.t_eq(sk - nf, 0, -127, 127), "st.end")
    nf0 = c.node(c.t_eq(nf, 0, 0, 127), "st.nf==0")
    in_chunk = c.node(c.t_lt(hv - from_bits(mr_len), 0, -127, 127), "st.H<len")
    nlen_d = from_bits(d_len)
    h_lt_n = c.node(c.t_lt(hv, S.NMAX, 0, 127), "st.H<NMAX")
    # a list prints an entry's name: alive, H < its length, H < NMAX
    in_name = c.node(c.t_lt(hv - nlen_d, 0, -127, 127, lits=[d_alive, h_lt_n]), "st.name_tok")
    sk1, nf1, ng1 = (inc_bits(c, reg[r_], f"st.{r_}+1") for r_ in ("SK", "NFILES", "NEXTG"))

    # --- the answer
    p_put, p_app, p_get, p_lst, p_del = (op[o_] for o_ in S.OPS)
    putapp = p_put + p_app
    nonlist = one - p_lst
    NL, TRO = cls_code(B.NL), cls_code(B.TR_OPEN)
    TRC = cls_code(B.TR_CLOSE)
    # the call's end
    case([tcc, nonlist], 1, S.SK)
    case([tcc, p_lst], 2, NL)
    # ADDR(SK): 0 first (right after </tool_call>, or after the list's header), else SK + 1
    first = c.node(c.t_and([jeq(0), nonlist]) + c.t_and([jeq(3), p_lst]), "st.sk.first")
    case([at[S.SK], first], 0, 0)
    case([at[S.SK], one - first], 0, sk1)
    # the search: VAL(SK)
    sv = [pv[S.SK], nonlist]
    case([*sv, match, p_put + p_del], 1, S.DIRW)
    case([*sv, match, p_app], 1, S.TMP)
    case([*sv, match, p_get], 2, NL)
    case([*sv, one - match, end, putapp], 1, S.NFILES)
    case([*sv, one - match, end, p_get + p_del], 2, NL)
    case([*sv, one - match, one - end], 1, S.SK)
    # writes
    case([*c.addr_lits(S.NFILES)], 0, nf1)
    case([pv[S.NFILES]], 1, S.DIRW)
    case([*c.addr_lits(S.DIRW), putapp], 0, 1)
    case([*c.addr_lits(S.DIRW), p_del], 0, 0)
    case([pv[S.DIRW], putapp], 1, S.DIRH)
    case([pv[S.DIRW], p_del], 2, NL)
    case([*c.addr_lits(S.DIRH)], 0, reg["NEXTG"])
    case([pv[S.DIRH]], 1, S.DIRT)
    case([*c.addr_lits(S.DIRT)], 0, reg["NEXTG"])
    case([pv[S.DIRT]], 1, S.METAW)
    case([*c.addr_lits(S.METAW)], 0, st.tlen)
    case([pv[S.METAW]], 1, S.NEXTG)
    case([*c.addr_lits(S.NEXTG)], 0, ng1)
    case([pv[S.NEXTG]], 2, NL)
    case([*c.addr_lits(S.TMP)], 0, dt)
    case([pv[S.TMP]], 1, S.TMPL)
    case([*c.addr_lits(S.TMPL)], 0, mr_len)
    case([pv[S.TMPL]], 1, S.METAN)
    case([*c.addr_lits(S.METAN)], 0, reg["NEXTG"])
    case([pv[S.METAN]], 1, S.DIRT)
    case([*c.addr_lits(S.GCUR), one - chain], 0, dh)
    case([*c.addr_lits(S.GCUR), chain], 0, mr_next)
    # the header, then the body
    hdr = [is_text, one - chain, one - lst, one - trc]
    case([*hdr, hs[1]], 2, TRO)
    case([*hdr, hs[2]], 2, NL)
    m_ok = c.node(c.t_and([putapp]) + c.t_and([p_del, wrote]), "st.msg.ok")
    m_nf = c.node(c.t_and([p_del, one - wrote]) + c.t_and([p_get, one - match]), "st.msg.nf")
    m_em = c.and_([p_lst, nf0], "st.msg.empty")
    case([*hdr, hs[3], p_get, match], 1, S.GCUR)
    case([*hdr, hs[3], p_lst, one - nf0], 1, S.SK)
    anyp_cases = []  # (literals, 18 bits: ints or forms)

    def anyc(lits, bits):
        anyp_cases.append(([Sx, *lits], bits))

    def qbits(q):
        return S.id_code(q)

    msgs = [(m_ok, S.MSG["ok"]), (m_nf, S.MSG["not_found"]), (m_em, S.MSG["empty"])]
    for m_, toks in msgs:
        for i, q in enumerate(toks):
            anyc([*hdr, hs[3 + i], m_], qbits(q))
        case([*hdr, hs[3 + len(toks)], m_], 2, NL)
        case([*hdr, hs[4 + len(toks)], m_], 2, TRC)
    case([trc, is_text], 2, NL)
    # a get's chunks: content[GCUR][H] while H < length, then the next chunk or the end
    gp = [chain, one - is_addr, one - trc]
    for s_ in range(4):
        anyc([*gp, in_chunk, hslot[s_]], c_tok[s_])
    case([*gp, one - in_chunk, mr_has], 1, S.GCUR)
    case([*gp, one - in_chunk, one - mr_has], 2, TRC)
    # a list: after VAL(SK) and each printed name token
    lp = [p_lst, lst, one - is_addr, one - trc]
    case([*lp, end], 2, TRC)
    for n in range(S.NMAX):
        anyc([*lp, one - end, in_name, hs[n] if n < 8 else heq(n)], d_name[n])
    case([*lp, one - end, one - in_name], 1, S.SK)

    # any-prints: their condition and the 18 id bits of the token
    anyp = c.node([t for lits, _ in anyp_cases for t in c.t_and(lits)], "st.anyprint")
    ybits = []
    for b in range(S.ID_BITS):
        terms = []
        for lits, bits in anyp_cases:
            v = bits[b]
            if isinstance(v, int):
                if v:
                    terms += c.t_and(lits)
            else:
                terms += c.t_and([*lits, v])
        ybits.append(c.node(terms, f"st.ybit{b}"))
    owner = c.t_and([is_text, Sx])
    st.debug = dict(match=match, end=end, H=hbits, chain=chain, lst=lst, wrote=wrote,
                    in_chunk=in_chunk, in_name=in_name, mr_len=mr_len, d_alive=d_alive)
    return owner, anyp, ybits


PERSISTENT = ("st.regs", "st.dir", "st.dht", "st.dirH", "st.dirT", "st.meta", "st.content")


def is_persistent(name: str) -> bool:
    """Heads that hold the store's files and registers (a snapshot keeps exactly these)."""
    return any(name == p or (p in ("st.meta", "st.content") and name.startswith(p)
                             and name[len(p):].isdigit()) for p in PERSISTENT)


def persistent_heads(c) -> list[list[int]]:
    """[chip layer, value head] of every value head that holds store state, in the emitter's
    numbering (key head j of a layer owns value heads j * r .. j * r + r - 1)."""
    from chipformer.q35.ir import Head

    r = c.target.value_heads // c.target.key_heads
    out = []
    for layer in range(c.n_layers()):
        heads = [n for n in c.nodes if isinstance(n, Head) and n.level == 2 * layer + 1]
        for j, n in enumerate(heads):
            if is_persistent(n.name):
                out += [[layer, j * r + x] for x in range(len(n.vheads()))]
    return out


def count_bits_n(c, x, reset, nmax, nbits, name):
    """Bits of min(n, nmax) from x = 2^-n."""
    out = []
    for b in range(nbits):
        terms = []
        for v in range(1, nmax + 1):
            d = (v >> b & 1) - ((v - 1) >> b & 1)
            if d:
                terms += [(d * co, a_, b_) for co, a_, b_ in ge_terms(c, x, reset, v)]
        out.append(c.node(terms, f"{name}.b{b}"))
    return out
