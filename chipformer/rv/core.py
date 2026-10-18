"""The RV32IM (minus DIV) core as a q35 circuit (spec.py has the trace format).

Every position computes the instruction that the next token belongs to: the state it reads is
the state after the position's own commit, so at a record's last token the circuit already works
on the next instruction, and inside a record it recomputes the current one (nothing changes
until the record's last token). Writes take their key from t-1 (a conv tap), their write gate
beta from the token at t, and their value from token payloads (lags 0..3) or from dims the
circuit computed at t-1 below the writing layer.

  L1   attn  anchor (PC head), count (2^-n), j (tokens of the current record, 2^-j)
       mlp   n and j thermometers
  L2   mlp   PC word w = anchor + n; link word w + 1
  L3-4 attn  ROM, two sublayers: the pre-decoded fields of word w (byte columns, 2-hot query,
             z word select)
       mlp   field one-hots, immediate bits, gated link
  L5   attn  register file (2 read ports; writes commit from t-1)
       mlp   adder stage 1 (4 operand modes), logic ops, shifter stage 1, store data,
             multiplier stage 1 (nibble products)
  L6   mlp   adder stage 2, shifter stage 2, multiplier stage 2 (product nibbles)
  L7-8 attn  RAM, two sublayers (32 KB of words)
       mlp   loads, SB/SH merge, compares, faults; multiplier stages 3-5 (column carries,
             carry-lookahead)
  L9+  mlp   rd value select and size, branch decision, then the output code
"""

import numpy as np

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import A_BIT, A_REF, HD, REF_COLS, Chan, K, Target
from chipformer.q35.prims import Circ, chan, from_bits, sel, thead
from chipformer.rv import ref as RF
from chipformer.rv import spec as S

T4B = Target(key_heads=16, value_heads=32, intermediate=9216)  # Qwen3.5-4B's layers
# output family ids: the chip families and controls, then Q (P3: a printable Qwen token, the
# echo of a putchar; its payload indexes the bridge's printable tokens)
OUT_NAMES = list(S.FAMILIES) + S.CONTROL_NAMES + ["Q"]
FAM_ID = {n: i for i, n in enumerate(OUT_NAMES)}
NFAM = 5  # family id bits in the output code
NPAY = 16  # payload bits in the output code
OUT_W = 2.0**7  # head output weight (as thead)
JMAX = 10  # longest record: FLT, 8 HEX, HALT
CODE_AMP = 128  # a token's code in its (tied) embedding row: CODE_AMP * (2 * bit - 1)
KAPPA = 2.0**-20  # the predicted code after the exit: KAPPA * (2 * bit - 1)
TAU_TEXT = 2.0**-40  # (P3) tau on text tokens; -1 on chip tokens (see q35.merge)


def code_bits(fam: str, p: int) -> list[int]:
    fid = FAM_ID[fam]
    return [fid >> i & 1 for i in range(NFAM)] + [p >> i & 1 for i in range(NPAY)]


def token_code(t: int) -> list[int]:
    """The 21 output-code bits of a chip token: family id (5) then payload (16)."""
    return code_bits(*S.family(t))


PAY_W = {"V": 15, "F0": 15, "F1": 15, "F2": 2, "J": 16, "R": 16, "BYTE": 9, "FLT": 3, "HEX": 4}


def lits_of(bits, v):
    """0/1 literals that are all 1 exactly when the bits spell v."""
    return [b if v >> i & 1 else 1 - b for i, b in enumerate(bits)]


def andpre(lits) -> Expr:
    """sum(lits) - (n - 1): 1 when all literals are 1, <= 0 otherwise (a key / query / z form)."""
    return sum(lits, Expr()) - (len(lits) - 1)


def famof(t: int) -> tuple:
    """(family, payload) of a chip token; (None, 0) for a text token (P3)."""
    return S.family(t) if t < S.VOCAB else (None, 0)


class RvCore(Circ):
    def __init__(self, target: Target | None = None, text=None) -> None:
        """``text`` (P3, bridge.Text): the Qwen vocabulary the bridge reads and prints; None
        builds the standalone machine."""
        super().__init__(target or T4B)
        f = self.token_feature
        self.text = text

        def fam_is(nm):
            return lambda t: int(famof(t)[0] == nm)

        def pay_bit(nm, i):
            return lambda t: (famof(t)[1] >> i & 1) if famof(t)[0] == nm else 0

        names = list(S.FAMILIES) + S.CONTROL_NAMES
        paid = ("V", "F0", "F1", "F2", "J", "R") + (("BYTE",) if text else ())
        self.fam = {n: f(f"is{n}", fam_is(n), full=True) for n in names}
        self.pay = {n: [f(f"{n}.b{i}", pay_bit(n, i), full=True) for i in range(w)]
                    for n, w in PAY_W.items() if n in paid}
        self.final = sum((self.fam[n] for n in S.FINAL), Expr())
        self.tables = {}  # name -> head node (for snapshots)

        def code_of(t):
            """A token's own output code, or None (a text token the circuit never prints)."""
            if t < S.VOCAB:
                return token_code(t)
            p = text.printable_index(t - S.TEXT_TOKEN) if text else None
            return None if p is None else code_bits("Q", p)

        # the token's own code (read by the tied output head; cleared before the exit writes the
        # predicted one)
        self.code = [f(f"code{i}", lambda t, i=i: 0 if (cb := code_of(t)) is None
                       else CODE_AMP * (2 * cb[i] - 1), full=True)
                     for i in range(NFAM + NPAY)]
        if text:
            tt = S.TEXT_TOKEN
            self.is_text = f("is_text", lambda t: int(t >= tt), full=True)
            self.mu = f("mu", lambda t: int(t < S.VOCAB), full=True)  # Qwen mutes on chip tokens
            self.tau = f("tau", lambda t: TAU_TEXT if t >= tt else -1, full=True)
            self.isTCO = f("isTCO", lambda t: int(t == tt + text.TCO), full=True)
            self.isTCC = f("isTCC", lambda t: int(t == tt + text.TCC), full=True)

            def tbyte(k, i):
                return lambda t: (text.row_bytes(t - tt)[k] >> i & 1) if t >= tt else 0

            # a text token's UTF-8 bytes (the first TEXT_ROW), bit by bit
            self.tb = [[f(f"tb{k}.b{i}", tbyte(k, i), full=True) for i in range(8)]
                       for k in range(S.TEXT_ROW)]

    def zeros(self, n: int, name: str) -> list:
        """Fresh dims that start at 0 (heads add onto them)."""
        return [self.token_feature(f"{name}[{i}]", lambda t: 0, full=True) for i in range(n)]

    def table(self, name, queries, keys, vheads, levels="bit", tiny=0.0):
        """A head with explicit per-column values / gates: vheads are dicts with v {col: Chan},
        z {col: Expr}, outs [(col, weight)], into [Expr] (one per out), beta, a."""
        first, rest = vheads[0], vheads[1:]
        outs = self.head(name, {s: sel(self, e) for s, e in queries.items()},
                         {s: sel(self, e) for s, e in keys.items()}, first["v"], first.get("a", 0),
                         first["z"], first["outs"], tiny=tiny, beta=first.get("beta"),
                         into=first.get("into"),
                         vx=[{"v": x["v"], "a": x.get("a", 0), "z": x["z"], "outs": x["outs"],
                              "beta": x.get("beta"), "into": x.get("into")} for x in rest])
        node = self.nodes[-1]
        node.levels = levels
        self.tables[name] = node
        return outs

    def int_bits(self, z: Expr, hi: int, name: str, n: int) -> list:
        """Bits of an integer 0..hi (thresholds at every flip of each bit)."""
        out = []
        for j in range(n):
            terms = []
            for theta in range(1, hi + 1):
                d = (theta >> j & 1) - ((theta - 1) >> j & 1)
                if d:
                    terms += self.t_ge(z, theta, 0, hi, d)
            out.append(self.node(terms, f"{name}.b{j}"))
        return out

    def counter(self, name, reset, step):
        """A 2^-k counter head: k = the `step` tokens since the latest `reset` token (0 there)."""
        one = self.one
        ((x,),) = thead(self, name, {0: reset, 1: one - reset}, {0: one - reset, 1: one - reset},
                        [{"data": {2: reset}, "ref": one - reset,
                          "a": 110 * step + 238 * reset}], tiny=2.0**-90)
        self.tables[name] = self.nodes[-1]
        self.nodes[-1].levels = "decay"  # a snapshot mid-count holds 2^-k
        return x

    def thermo(self, x, reset, n, name):
        """[k > m] for m < n, from x = 2^-k (0 at reset tokens)."""
        one = self.one
        return [self.node([(1, one - reset, 7 * one - 8 * 2**m * x),
                           (-1, one - reset, 6 * one - 8 * 2**m * x)], f"{name}>{m}")
                for m in range(n)]


def adder_stage2(c, g, p_, d, n_nib, name):
    """Result bits of a carry-lookahead add from per-nibble generate / propagate / digit bits;
    also the carry out of the top nibble."""
    one = c.one

    def carry_sources(k):
        """Mutually exclusive literal sets, one of them true iff a carry enters nibble k."""
        return [[g[j]] + p_[j + 1:k] for j in range(k)]

    r = []
    for k in range(n_nib):
        for b in range(4):
            terms = c.t_and([d[k][b]])
            for src in carry_sources(k):
                low = d[k][:b]
                terms += c.t_and(src + low + [one - d[k][b]])
                terms += c.t_and(src + low + [d[k][b]], -1)
            r.append(c.node(terms, f"{name}.r{4 * k + b}"))
    cout = c.node([t for src in carry_sources(n_nib) for t in c.t_and(src)], f"{name}.cout")
    return r, cout


def stage1_digits(c, sums, name, hi):
    """Generate, propagate and digit bits of nibble sums; sums[k] = [(int Expr, literal)]."""
    g, p_, d = [], [], []
    for k, forms in enumerate(sums):
        gt, pt, dt = [], [], [[] for _ in range(4)]
        for s, lit in forms:
            lits = [lit] if lit is not None else []
            gt += c.t_ge(s, 16, 0, hi, lits=lits)
            pt += c.t_eq(s, 15, 0, hi, lits=lits)
            for b in range(4):
                for theta in range(1, hi + 1):
                    dd = (theta % 16 >> b & 1) - ((theta - 1) % 16 >> b & 1)
                    if dd:
                        dt[b] += c.t_ge(s, theta, 0, hi, dd, lits=lits)
        g.append(c.node(gt, f"{name}.g{k}"))
        p_.append(c.node(pt, f"{name}.p{k}"))
        d.append([c.node(dt[b], f"{name}.d{k}.{b}") for b in range(4)])
    return g, p_, d


def build(target: Target | None = None, text=None) -> RvCore:
    """The circuit; with ``text`` (P3) also the bridge to Qwen's text: a tool call's text goes
    into a buffer the program reads (TEXT_ADDR), ``</tool_call>`` starts a run (GO), putchar
    output is echoed as Qwen tokens, and the circuit owns the positions of a run."""
    c = RvCore(target, text)
    one, fam, pay = c.one, c.fam, c.pay
    isV, isJR = fam["V"], fam["J"] + fam["R"]
    isSS = fam["SS0"] + fam["SS1"]

    # ---------------------------------------------------------------- L1: anchor, count, j
    jr_bits = [pay["J"][i] + pay["R"][i] for i in range(16)]
    (anchor,) = thead(c, "pc", {0: one}, {0: one},
                      [{"data": {2 + i: jr_bits[i] for i in range(16)}, "ref": one, "beta": isJR}])
    c.tables["pc"] = c.nodes[-1]
    xn = c.counter("count", isJR, c.final - isJR)
    rec_start = c.final + fam["GO"]
    xj = c.counter("j", rec_start, one - rec_start - (c.is_text if text else 0))
    cnt = c.thermo(xn, isJR, S.N_MAX, "n")
    jt = c.thermo(xj, rec_start, JMAX, "j")
    n_int = sum(cnt, Expr())
    at_max = cnt[S.N_MAX - 1]  # n == 63: the refresh comes first

    def jeq(v):
        return (one if v == 0 else jt[v - 1]) - (jt[v] if v < JMAX else 0)

    j0, j1, j2, j3 = (jeq(v) for v in range(4))

    if text:  # ------------------------------------------------ P3 bridge state
        isGO, isHALT = fam["GO"], fam["HALT"]
        # running: from GO to HALT; armed: inside a tool call (its text is captured)
        ((running,),) = thead(c, "run", {0: one}, {0: one},
                              [{"data": {2: isGO}, "ref": one, "beta": isGO + isHALT}])
        c.tables["run"] = c.nodes[-1]
        ((armed,),) = thead(c, "arm", {0: one}, {0: one},
                            [{"data": {2: c.isTCO}, "ref": one, "beta": c.isTCO + c.isTCC}])
        c.tables["arm"] = c.nodes[-1]
        capw = c.and_([c.is_text, armed, one - c.isTCO], "cap.w")  # a captured text token
        xcap = c.counter("cap", c.isTCO, capw)  # its index (1, 2, ..) since <tool_call>
        capt = c.thermo(xcap, c.isTCO, S.CAP_MAX, "cap")
        caprow = {i: capt[i - 1] - (capt[i] if i < S.CAP_MAX else 0)
                  for i in range(1, S.CAP_MAX + 1)}

    # ---------------------------------------------------------------- L2: PC word, link
    def add_small(abits, extra, name):
        """Bits 0..14 of (anchor + n + extra) for n <= 63: low 6 bits by thresholds, the rest by
        the carry into bit 6."""
        L = from_bits(abits[:6]) + n_int + extra
        lo, hi = 0, 63 + 63 + extra
        bits = []
        for j in range(6):
            terms = []
            for theta in range(lo + 1, hi + 1):
                d = (theta >> j & 1) - ((theta - 1) >> j & 1)
                if d:
                    terms += c.t_ge(L, theta, lo, hi, d)
            bits.append(c.node(terms, f"{name}.b{j}"))
        for j in range(6, 15):
            below = abits[6:j]
            terms = c.t_and([abits[j]])
            terms += c.t_ge(L, 64, lo, hi, 1, lits=below + [one - abits[j]])
            terms += c.t_ge(L, 64, lo, hi, -1, lits=below + [abits[j]])
            bits.append(c.node(terms, f"{name}.b{j}"))
        return bits

    w = add_small(anchor, 0, "w")
    w1 = add_small(anchor, 1, "w+1")  # the link: PC + 4 = ROM_BASE + 4 * (w + 1)
    assert S.ROM_BASE == 1 << 16 and S.ROM_WORDS == 1 << 15
    # PC = 0x10000 + 4 w (w < 2^15): bits 2..15 = w[0..13], bit 16 = !w[14], bit 17 = w[14]
    pc_bits = [Expr(), Expr()] + w[:14] + [one - w[14], w[14]] + [Expr()] * 14

    # ---------------------------------------------------------------- L3-4: ROM fetch
    # word w: key head w[14:10] (two sublayers of 16), row pair (p, p + 64) with p = w[9:4],
    # half w[3] (row p: columns 2..33, row p + 64: 34..65), word w[2:0], 4 byte columns; value
    # head 0 = (uop, rdw, rs1, rs2), value head 1 = the immediate's 4 bytes
    fdims = c.zeros(4, "rom.f")
    idims = c.zeros(4, "rom.imm")
    for h in range(S.ROM_WORDS >> 10):
        hl = lits_of(w[10:15], h)
        q = {}
        for p in range(64):
            e = andpre(lits_of(w[4:10], p) + hl)
            q[p], q[p + 64] = e, e
        vhs = []
        for dims in (fdims, idims):
            z, outs, into = {}, [], []
            for half in range(2):
                for k in range(8):
                    ze = K * andpre(lits_of([w[3]], half) + lits_of(w[0:3], k))
                    for b in range(4):
                        col = 2 + 32 * half + 4 * k + b
                        z[col] = ze
                        outs.append((col, OUT_W))
                        into.append(dims[b])
            vhs.append({"v": {}, "z": z, "outs": outs, "into": into})
        c.table(f"rom{h}", q, {}, vhs, levels="byte")
    U, RDW, RS1, RS2 = (256 * d for d in fdims)  # integer field values
    IMMB = [256 * d for d in idims]

    uops = {u: c.node(c.t_eq(U, RF.U[u], 0, 255), f"uop={u}") for u in RF.UOPS
            if u not in ("DIV", "DIVU", "REM", "REMU") and not u.startswith("CSR")}

    def any_of(*names):
        return sum((uops[n] for n in names), Expr())

    rdw = {r: c.node(c.t_eq(RDW, r, 0, 31), f"rdw={r}") for r in range(1, 32)}
    rs1 = {r: c.node(c.t_eq(RS1, r, 0, 31), f"rs1={r}") for r in range(1, 32)}
    rs2 = {r: c.node(c.t_eq(RS2, r, 0, 31), f"rs2={r}") for r in range(1, 32)}
    rdw_any = sum(rdw.values(), Expr())
    imm = [b for k in range(4) for b in c.int_bits(IMMB[k], 255, f"imm{k}", 8)]
    # PC + 4 = 0x10000 + 4 (w + 1): bits 2..15 = (w+1)[0..13], bit 16 = !(w+1)[14], bit 17
    link = [Expr()] * 2 + w1[:14] + [one - w1[14], w1[14]] + [Expr()] * 14
    jl = [Expr() if not link[i].terms else
          c.node(c.t_eq(U, RF.U["JAL"], 0, 255, lits=[link[i]])
                 + c.t_eq(U, RF.U["JALR"], 0, 255, lits=[link[i]]), f"link.b{i}")
          for i in range(32)]

    # ---------------------------------------------------------------- L5: register file
    P = pay["V"]
    isF1, isF2 = fam["F1"], fam["F2"]
    vcols = {}  # bit -> list of value Chans (the commit's sources; exactly one is live)
    for i in range(32):
        cs = []
        if i < 15:
            cs.append(Chan(pay["F0"][i]))
            cs.append(Chan(-1 * P[i] + isF1 + isV, (1, -1, 0, 0)))  # V at t-1 when t is F1
            cs.append(Chan(-1 * P[i] + isF2 + isV, (1, 0, -1, 0)))  # V at t-2 when t is F2
        elif i < 30:
            cs.append(Chan(pay["F1"][i - 15] + pay["F0"][14]))
            cs.append(Chan(-1 * P[i - 15] + isF2 + isV, (1, -1, 0, 0)))
        else:
            cs.append(Chan(pay["F2"][i - 30] + pay["F1"][14] + pay["F0"][14]))
        if jl[i].terms:
            cs.append(Chan(jl[i], (0, 1, 0, 0)))  # the link, computed at t-1
        vcols[i] = cs
    reg_beta = fam["F0"] + fam["F1"] + fam["F2"] + fam["J"]
    reg_keys = {r: Chan(rdw[r], (0, 1, 0, 0)) for r in range(1, 32)}
    ports = []
    for pname, rq in (("rs1", rs1), ("rs2", rs2)):
        dims = c.zeros(32, f"x.{pname}")
        v, z, outs, into, col = {}, {}, [], [], 2
        c.reg_cols = {}
        for i in range(32):
            for ch in vcols[i]:
                v[col] = chan(ch, A_BIT)
                z[col] = K * one
                outs.append((col, OUT_W))
                into.append(dims[i])
                c.reg_cols.setdefault(i, []).append(col)
                col += 1
        for rc in REF_COLS:
            v[rc] = chan(one, A_REF)
        c.table(f"regs.{pname}", rq, reg_keys,
                [{"v": v, "z": z, "outs": outs, "into": into, "beta": reg_beta}])
        ports.append(dims)
    A, B = ports

    def nib(bits, k, inv=False):
        x = from_bits(bits[4 * k:4 * k + 4])
        return 15 - x if inv else x

    # ---------------------------------------------------------------- L5-6: adder
    md = {"ADD": any_of("ADD"),
          "ADDI": any_of("ADDI", "LB", "LH", "LW", "LBU", "LHU", "SB", "SH", "SW", "JALR"),
          "SUB": any_of("SUB", "SLT", "SLTU", "BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"),
          "SUBI": any_of("SLTI", "SLTIU")}
    bsrc = {"ADD": (B, False, 0), "ADDI": (imm, False, 0), "SUB": (B, True, 1),
            "SUBI": (imm, True, 1)}
    sums = [[(nib(A, k) + nib(bsrc[m][0], k, bsrc[m][1]) + (bsrc[m][2] if k == 0 else 0), lit)
             for m, lit in md.items()] for k in range(8)]
    g, p_, d = stage1_digits(c, sums, "add", 31)
    r, c32 = adder_stage2(c, g, p_, d, 8, "add")

    # ---------------------------------------------------------------- L5-6: logic, shifter
    logic = []
    for i in range(32):
        terms = []
        for op, bb in (("AND", B), ("ANDI", imm)):
            terms += c.t_and([A[i], bb[i], uops[op]])
        for op, bb in (("OR", B), ("ORI", imm)):
            terms += c.t_and([uops[op]]) + c.t_and([1 - A[i], 1 - bb[i], uops[op]], -1)
        for op, bb in (("XOR", B), ("XORI", imm)):
            terms += c.t_and([A[i], 1 - bb[i], uops[op]]) + c.t_and([1 - A[i], bb[i], uops[op]])
        logic.append(c.node(terms, f"logic.b{i}"))

    sh_ops = {"L": ("SLL", "SLLI"), "R": ("SRL", "SRLI"), "A": ("SRA", "SRAI")}

    def src_bit(dirn, i):
        if 0 <= i < 32:
            return A[i]
        return A[31] if dirn == "A" and i >= 32 else None

    y = {}
    for dirn, (rop, iop) in sh_ops.items():
        y[dirn] = []
        for i in range(32):
            terms = []
            for op, sb in ((rop, B), (iop, imm)):
                for kk in range(8):
                    j = i - 4 * kk if dirn == "L" else i + 4 * kk
                    bit = src_bit(dirn, j)
                    if bit is None:
                        continue
                    terms += c.t_and([uops[op], bit] + lits_of(sb[2:5], kk))
            y[dirn].append(c.node(terms, f"sh{dirn}1.b{i}"))
    shift = []
    for i in range(32):
        terms = []
        for dirn, (rop, iop) in sh_ops.items():
            yy = y[dirn]
            for op, sb in ((rop, B), (iop, imm)):
                for kk in range(4):
                    j = i - kk if dirn == "L" else i + kk
                    bit = yy[j] if 0 <= j < 32 else (yy[31] if dirn == "A" and j >= 32 else None)
                    if bit is None:
                        continue
                    terms += c.t_and([uops[op], bit] + lits_of(sb[0:2], kk))
        shift.append(c.node(terms, f"shift.b{i}"))

    # ---------------------------------------------------------------- L5-9: multiplier
    # 64-bit product of nibbles: p_ij = a_i b_j <= 225 (stage 1), split into nibbles (stage 2);
    # column k sums lo_ij (i + j = k), hi_ij (i + j = k - 1) and, for MULH / MULHSU, the
    # two's-complement corrections -a31 * b and -b31 * a of the high word (as ~x + 1); a column
    # stays <= 249, so it splits into a digit and a carry (stage 3); then a carry-lookahead add
    # of digits and carries (stages 4-5)
    mh_a = any_of("MULH", "MULHSU")  # rs1 signed
    mh_b = uops["MULH"]  # rs2 signed
    prod = {}
    for i in range(8):
        for j in range(8):
            terms = []
            for v in range(1, 16):
                terms += c.t_and(lits_of(A[4 * i:4 * i + 4], v), v, a=nib(B, j))
            prod[i, j] = c.node(terms, f"mul.p{i}{j}")
    corr_a = [c.node(c.t_and([A[31], mh_a], a=nib(B, j, inv=True)), f"mul.ca{j}") for j in range(8)]
    corr_b = [c.node(c.t_and([B[31], mh_b], a=nib(A, j, inv=True)), f"mul.cb{j}") for j in range(8)]
    plus1 = c.node(c.t_and([A[31], mh_a]) + c.t_and([B[31], mh_b]), "mul.plus1")
    lo, hi = {}, {}
    for (i, j), pij in prod.items():
        hi[i, j] = c.node([t for m in range(1, 15) for t in c.t_ge(pij, 16 * m, 0, 225)],
                          f"mul.hi{i}{j}")
        lo[i, j] = pij - 16 * hi[i, j]
    cols = []
    for k in range(16):
        s = sum((lo[i, k - i] for i in range(8) if 0 <= k - i < 8), Expr())
        s += sum((hi[i, k - 1 - i] for i in range(8) if 0 <= k - 1 - i < 8), Expr())
        if k >= 8:
            s += corr_a[k - 8] + corr_b[k - 8]
        if k == 8:
            s += plus1
        cols.append(s)
    ccar, cdig = [], []
    for k, s in enumerate(cols):
        cc = c.node([t for m in range(1, 16) for t in c.t_ge(s, 16 * m, 0, 255)], f"mul.c{k}")
        ccar.append(cc)
        cdig.append(s - 16 * cc)
    tsum = [[(cdig[k] + (ccar[k - 1] if k else 0), None)] for k in range(16)]
    mg, mp, mdg = stage1_digits(c, tsum, "mul", 30)
    mres, _ = adder_stage2(c, mg, mp, mdg, 16, "mul")
    mul_lo, mul_hi = mres[:32], mres[32:]

    # store data for SW (0 on V tokens, so an SB/SH record's last V does not leak it)
    sd = [c.and_([B[i], one - isV], f"sd.b{i}") for i in range(32)]
    is_store = any_of("SB", "SH", "SW")

    # ---------------------------------------------------------------- L7-8: RAM
    # word index wi = EA[14:2] (RAM at 0): key head = (wi[12], wi[3:0]) = (EA[14], EA[5:2]),
    # value head wi[4] = EA[6], row wi[11:5] = EA[13:7]
    EA = r
    mem = c.zeros(32, "mem")
    n_ram_heads = S.RAM_WORDS >> 8
    for h in range(n_ram_heads):
        hl = lits_of(EA[2:6], h & 15) + lits_of(EA[14:14 + (n_ram_heads >> 4).bit_length() - 1],
                                                 h >> 4)
        keys, q = {}, {}
        for row in range(128):
            rl = lits_of(EA[7:14], row)
            keys[row] = Chan(andpre(rl + hl + [is_store]), (0, 1, 0, 0))
            q[row] = andpre(rl + hl + ([1 - EA[15]] if text else []))
        vhs = []
        for x in range(2):
            v, z, outs, into = {}, {}, [], []
            for i in range(32):  # block A: SW data from t-1
                v[2 + i] = chan(Chan(sd[i], (0, 1, 0, 0)), A_BIT)
            for i in range(32):  # block B: the SB/SH word from the record's V tokens
                lag = 3 - i // 15 if i < 30 else 1
                taps = [0, 0, 0, 0]
                taps[0], taps[lag] = 1, -1
                v[34 + i] = chan(Chan(-1 * P[i % 15 if i < 30 else i - 30] + isSS + isV,
                                      tuple(taps)), A_BIT)
            for rc in REF_COLS:
                v[rc] = chan(one, A_REF)
            for col in range(2, 66):
                z[col] = K * (EA[6] if x else 1 - EA[6])
                outs.append((col, OUT_W))
                into.append(mem[(col - 2) % 32])
            vhs.append({"v": v, "z": z, "outs": outs, "into": into,
                        "beta": fam[f"ST{x}"] + fam[f"SS{x}"]})
        c.table(f"ram{h}", q, keys, vhs)
    in_text = []
    if text:  # the text buffer: row i = the i-th captured token's bytes, read as memory
        in_text = [EA[15]] + [1 - EA[b] for b in range(11, 15)]
        q = {i: andpre(lits_of(EA[4:11], i) + in_text) for i in caprow}
        keys = {i: Chan(e) for i, e in caprow.items()}
        vhs = []
        for x in range(2):  # value head x: bytes 8x .. 8x+7 (words 2x, 2x+1 of the row)
            v, z, outs, into = {}, {}, [], []
            for k in range(8):
                for i in range(8):
                    col = 2 + 8 * k + i
                    v[col] = chan(c.tb[8 * x + k][i], A_BIT)
                    z[col] = K * andpre(lits_of([EA[3]], x) + lits_of([EA[2]], k // 4))
                    outs.append((col, OUT_W))
                    into.append(mem[8 * (k % 4) + i])
            for rc in REF_COLS:
                v[rc] = chan(one, A_REF)
            vhs.append({"v": v, "z": z, "outs": outs, "into": into, "beta": capw,
                        "a": 238 * c.isTCO})  # cleared at <tool_call>
        c.table("tbuf", q, keys, vhs)
        in_text = in_text + [1 - b for b in EA[16:32]]

    # ---------------------------------------------------------------- loads, merge, compares
    lane = EA[0:2]
    ld = []
    for i in range(32):
        terms = c.t_and([uops["LW"], mem[i]] + lits_of(lane, 0))
        for k in range(4):
            if i < 8:
                terms += c.t_and([uops["LBU"], mem[8 * k + i]] + lits_of(lane, k))
                terms += c.t_and([uops["LB"], mem[8 * k + i]] + lits_of(lane, k))
            else:
                terms += c.t_and([uops["LB"], mem[8 * k + 7]] + lits_of(lane, k))
        for hh in range(2):
            if i < 16:
                terms += c.t_and([uops["LHU"], mem[16 * hh + i]] + lits_of(lane, 2 * hh))
                terms += c.t_and([uops["LH"], mem[16 * hh + i]] + lits_of(lane, 2 * hh))
            else:
                terms += c.t_and([uops["LH"], mem[16 * hh + 15]] + lits_of(lane, 2 * hh))
        ld.append(c.node(terms, f"load.b{i}"))
    merged = []
    for i in range(32):
        terms = []
        for k in range(4):
            inl = 8 * k <= i < 8 * k + 8
            src = B[i - 8 * k] if inl else mem[i]
            terms += c.t_and([uops["SB"], src] + lits_of(lane, k))
        for hh in range(2):
            inl = 16 * hh <= i < 16 * hh + 16
            src = B[i - 16 * hh] if inl else mem[i]
            terms += c.t_and([uops["SH"], src] + lits_of(lane, 2 * hh))
        merged.append(c.node(terms, f"merge.b{i}"))
    b31 = {"SUB": B[31], "SUBI": imm[31]}
    slt = c.node([t for m in b31 for t in c.t_and([md[m], A[31], 1 - b31[m]])
                  + c.t_and([md[m], A[31], b31[m], r[31]])
                  + c.t_and([md[m], 1 - A[31], 1 - b31[m], r[31]])], "slt")
    sltu = c.node(c.t_and([md["SUB"] + md["SUBI"], 1 - c32]), "sltu")
    eq = c.and_([1 - b for b in r], "eq")

    # ---------------------------------------------------------------- faults, ecalls
    loads = any_of("LB", "LH", "LW", "LBU", "LHU")
    halfs, words = any_of("LH", "LHU"), uops["LW"]
    s_halfs, s_words = uops["SH"], uops["SW"]
    ram_bits = (4 * S.RAM_WORDS - 1).bit_length()
    in_ram = [1 - b for b in EA[ram_bits:32]]  # all 0: the address is in RAM
    a7_is = {v: c.and_(lits_of(A, v), f"a7=={v}") for v in (1, 93)}
    ecall = uops["ECALL"]
    # JALR target (EA & ~1) must be word-aligned and inside ROM: EA[31:18] = 0, EA[17:16] = 01/10
    rom_hi = [1 - b for b in EA[18:32]]
    mis_ld = c.node(c.t_and([halfs, EA[0]]) + c.t_and([words, EA[0]])
                    + c.t_and([words, 1 - EA[0], EA[1]]), "fault.misld")
    mis_st = c.node(c.t_and([s_halfs, EA[0]]) + c.t_and([s_words, EA[0]])
                    + c.t_and([s_words, 1 - EA[0], EA[1]]), "fault.misst")
    oor_ld = c.node(c.t_and([loads, 1 - mis_ld]) + c.t_and([loads, 1 - mis_ld] + in_ram, -1)
                    + (c.t_and([loads, 1 - mis_ld] + in_text, -1) if in_text else []),
                    "fault.oorld")
    oor_st = c.node(c.t_and([is_store, 1 - mis_st])
                    + c.t_and([is_store, 1 - mis_st] + in_ram, -1), "fault.oorst")
    jalr = uops["JALR"]
    # bad JALR targets (exclusive cases): misaligned; beyond 2^18; EA[17:16] = 00 or 11
    fc6_terms = (c.t_and([jalr, EA[1]])
                 + c.t_and([jalr, 1 - EA[1]]) + c.t_and([jalr, 1 - EA[1]] + rom_hi, -1)
                 + c.t_and([jalr, 1 - EA[1], 1 - EA[16], 1 - EA[17]] + rom_hi)
                 + c.t_and([jalr, 1 - EA[1], EA[16], EA[17]] + rom_hi)
                 + c.t_and([uops["JAL"], imm[16]]))
    fc = {0: c.and_([uops["ILLEGAL"]], "fault.0"), 1: c.and_([uops["EBREAK"]], "fault.1"),
          2: mis_ld, 3: mis_st, 4: oor_ld, 5: oor_st,
          7: c.node(c.t_and([ecall]) + c.t_and([ecall, a7_is[1]], -1)
                    + c.t_and([ecall, a7_is[93]], -1), "fault.7")}
    branches = any_of("BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU")

    # ---------------------------------------------------------------- rd value, size, branch
    addish = any_of("ADD", "ADDI", "SUB")
    sltish, sltuish = any_of("SLT", "SLTI"), any_of("SLTU", "SLTIU")
    logicops = any_of("AND", "ANDI", "OR", "ORI", "XOR", "XORI")
    shiftops = any_of("SLL", "SLLI", "SRL", "SRLI", "SRA", "SRAI")
    mulhs = any_of("MULH", "MULHSU", "MULHU")
    srcs = [(addish, r), (uops["CONST"], imm), (logicops, logic), (shiftops, shift), (loads, ld),
            (uops["MUL"], mul_lo), (mulhs, mul_hi)]
    val, k0t, k1t = [], [], []
    for i in range(32):
        terms = []
        for gate, bits in srcs:
            terms += c.t_and([bits[i], gate])
        if i == 0:
            terms += c.t_and([slt, sltish]) + c.t_and([sltu, sltuish])
        val.append(c.node(terms, f"val.b{i}"))
    for gate, bits in srcs:
        k0t += c.t_and([gate] + bits[14:32]) + c.t_and([gate] + [1 - b for b in bits[14:32]])
        k1t += c.t_and([gate] + bits[29:32]) + c.t_and([gate] + [1 - b for b in bits[29:32]])
    k0t += c.t_and([sltish + sltuish])
    k1t += c.t_and([sltish + sltuish])
    gate_any = addish + uops["CONST"] + logicops + shiftops + loads + uops["MUL"] + mulhs
    k0 = c.node(k0t, "fits15")
    k1 = c.node(k1t, "fits30")
    taken = c.node(c.t_and([uops["BEQ"], eq]) + c.t_and([uops["BNE"], 1 - eq])
                   + c.t_and([uops["BLT"], slt]) + c.t_and([uops["BGE"], 1 - slt])
                   + c.t_and([uops["BLTU"], sltu]) + c.t_and([uops["BGEU"], 1 - sltu]),
                   "taken")
    # fault 6: a bad JALR target, a bad JAL target, or a taken branch to a bad target
    fc[6] = c.node(fc6_terms + c.t_and([taken, imm[16]]), "fault.6")
    any_fault = sum(fc.values(), Expr())
    cause = [sum((fc[k] for k in fc if k >> b & 1), Expr()) for b in range(3)]

    # ---------------------------------------------------------------- the next token
    valc = gate_any + sltish + sltuish
    ok = one - any_fault
    nr = [j0, one - at_max, ok]  # the first token of a record (no refresh, no fault)
    cases = []  # (literals, family, payload bits or int)
    # P3: at a BYTE / HALT the next token is the echo (a Qwen token), at </tool_call> it is GO;
    # every other case is off there
    nb = [one - fam["BYTE"] - fam["HALT"] - c.isTCC] if text else []

    def case(lits, fam_name, payload=0):
        cases.append((lits + nb, fam_name, payload))

    case([j0, at_max], "R", w)
    case(nr + [valc, rdw_any, k0], "F0", val[0:15])
    case(nr + [valc, rdw_any, 1 - k0], "V", val[0:15])
    case([j1, ok, valc, rdw_any, k1], "F1", val[15:30])
    case([j1, ok, valc, rdw_any, 1 - k1], "V", val[15:30])
    case([j2, ok, valc, rdw_any], "F2", val[30:32])
    case(nr + [valc, 1 - rdw_any], "NT")
    case(nr + [branches, taken], "J", imm[0:16])
    case(nr + [branches, 1 - taken], "NT")
    case(nr + [uops["JAL"]], "J", imm[0:16])
    case(nr + [uops["JALR"]], "J", r[2:16] + [EA[17]])  # word = EA[17:2] - 0x4000
    case(nr + [uops["SW"], 1 - EA[6]], "ST0")
    case(nr + [uops["SW"], EA[6]], "ST1")
    sbsh = any_of("SB", "SH")
    case(nr + [sbsh], "V", merged[0:15])
    case([j1, ok, sbsh], "V", merged[15:30])
    case([j2, ok, sbsh], "V", merged[30:32])
    case([j3, ok, sbsh, 1 - EA[6]], "SS0")
    case([j3, ok, sbsh, EA[6]], "SS1")
    case(nr + [ecall, a7_is[1]], "BYTE", B[0:9])
    case(nr + [ecall, a7_is[93]], "HALT")
    case(nr + [uops["FENCE"]], "NT")
    # a fault: FLT(cause), the PC's 8 hex digits (most significant first), HALT
    case([j0, one - at_max, any_fault], "FLT", cause)
    for k in range(8):
        case([jeq(1 + k), any_fault], "HEX", pc_bits[4 * (7 - k):4 * (8 - k)])
    case([jeq(9), any_fault], "HALT")
    owner = None
    if text:
        cases.append(([fam["BYTE"]], "Q", c.pay["BYTE"]))
        nl = text.printable_index(text.NL)  # a run ends in a newline
        cases.append(([fam["HALT"]], "Q", [one if nl >> i & 1 else Expr() for i in range(9)]))
        cases.append(([c.isTCC], "GO", 0))
        # who predicts the next token: the circuit from </tool_call> through a run's last echo
        owner = c.node(c.t_and([running]) + c.t_and([fam["HALT"]])
                       + c.t_and([c.isTCC, one - running]), "owner")
    c.owner = owner
    own = [owner] if text else []
    fam_cases = [(l_, FAM_ID[fn]) for l_, fn, _ in cases]
    pay_cases = [(l_, pl) for l_, _, pl in cases if not isinstance(pl, int)]
    if text:
        # merged: no separate select dims (they would outlive the exit and reach Qwen's norms);
        # predictions read the exit's code dims
        c.out_fam = c.out_pay = None
        exit_min = max(n.level for n in c.nodes)
    else:
        fam_bits = c.select_bits(fam_cases, NFAM, "exit.fam")
        pay_bits = c.select_bits(pay_cases, NPAY, "exit.pay")
        c.out_fam, c.out_pay = fam_bits, pay_bits
        exit_min = max(c.level(b) for b in fam_bits) - 1
    # the exit: the input code is cleared, and the same cases write KAPPA * (2 * bit - 1) into
    # the code dims in the output stage itself (no extra layer); a stock output head tied to the
    # embedding then scores every token by how many code bits agree
    for e in c.code:
        c.ffn([(-1, one, e), (1, one, -1 * e)], "code.clear", into=e)
    for i, e in enumerate(c.code):
        cs, bit = (fam_cases, i) if i < NFAM else (pay_cases, i - NFAM)
        terms = c.t_and(own, -KAPPA) if text else c.t_const(-KAPPA)
        for lits, val in cs:
            if isinstance(val, int):
                if val >> bit & 1:
                    terms += c.t_and(lits + own, 2 * KAPPA)
            elif bit < len(val) and val[bit].terms:
                terms += c.t_and([*lits, val[bit]] + own, 2 * KAPPA)
        c.ffn(terms, f"exit.code{i}", into=e, min_level=exit_min)
    c.keep = [next(iter(e.terms)) for e in c.code]
    if text:
        # tau: 0 wherever the circuit predicts (chip tokens: -1 + 1; owned text: TAU - TAU), the
        # text value where Qwen does, which keeps chip tokens out of Qwen's text (q35.merge)
        c.ffn([(1, c.mu, one)] + c.t_and([owner, c.is_text], -TAU_TEXT), "exit.tau",
              into=c.tau, min_level=exit_min)
        c.keep += [next(iter(c.tau.terms)), next(iter(c.mu.terms))]
    c.probe = {"w": w, "anchor": anchor, "cnt": cnt, "jt": jt, "U": U, "A": A, "B": B,
               "imm": imm, "r": r, "val": val, "mem": mem, "ld": ld, "k0": k0, "k1": k1,
               "taken": taken, "logic": logic, "shift": shift, "merged": merged,
               "mul": mres, "fault": fc}
    return c


# --- snapshots (the state the host injects) ---------------------------------------------------


def reg_row(c: RvCore, value: int) -> np.ndarray:
    row = np.zeros(HD)
    row[list(REF_COLS)] = A_REF
    for i in range(32):
        if value >> i & 1:
            row[c.reg_cols[i][0]] = A_BIT
    return row


def counter_rows(k: int) -> dict:
    r0 = np.zeros(HD)
    r0[2] = A_BIT * 2.0**-k
    r1 = np.zeros(HD)
    r1[list(REF_COLS)] = A_REF
    return {0: r0, 1: r1}


def snapshot(c: RvCore, m: "RF.Machine") -> None:
    """Load a machine's state into the circuit's tables (the heads' initial states), as at the
    start of a record (j = 0)."""
    pc = np.zeros(HD)
    pc[list(REF_COLS)] = A_REF
    for i in range(16):
        pc[2 + i] = A_BIT * (m.anchor >> i & 1)
    c.tables["pc"].init = [{0: pc}]
    c.tables["count"].init = [counter_rows(m.n)]
    c.tables["j"].init = [counter_rows(0)]
    regs = {r: reg_row(c, m.x[r]) for r in range(1, 32)}
    for p in ("rs1", "rs2"):
        c.tables[f"regs.{p}"].init = [regs]
    # ROM: every row of every head carries the ballast (so each read has the same norm)
    for h in range(S.ROM_WORDS >> 10):
        rows0 = {rr: np.zeros(HD) for rr in range(128)}
        rows1 = {rr: np.zeros(HD) for rr in range(128)}
        for rows in (rows0, rows1):
            for row in rows.values():
                row[list(REF_COLS)] = A_REF
        for wl in range(1024):
            wi = h << 10 | wl
            word = m.rom[wi]
            fields = RF.rom_fields(word, S.ROM_BASE + 4 * wi) if word else (0, 0, 0, 0, 0)
            uop, rdw, rs1, rs2, imm = fields
            pp, half, k = wl >> 4, wl >> 3 & 1, wl & 7
            rr = pp + 64 * half
            for b, v in enumerate((uop, rdw, rs1, rs2)):
                rows0[rr][2 + 32 * half + 4 * k + b] = v
            for b in range(4):
                rows1[rr][2 + 32 * half + 4 * k + b] = imm >> 8 * b & 0xFF
        c.tables[f"rom{h}"].init = [rows0, rows1]
    for h in range(S.RAM_WORDS >> 8):
        r0, r1 = {}, {}
        for wi in range(S.RAM_WORDS):
            if (wi & 15 | (wi >> 12) << 4) != h or not m.ram[wi]:
                continue
            row = np.zeros(HD)
            row[list(REF_COLS)] = A_REF
            for i in range(32):
                row[2 + i] = A_BIT * (m.ram[wi] >> i & 1)
            (r1 if wi >> 4 & 1 else r0)[wi >> 5 & 127] = row
        c.tables[f"ram{h}"].init = [r0, r1]


def decode_code(fid: int, p: int, text=None) -> int:
    """The token of an output code (family id, payload), -1 if it names none; a Q code is the
    text token of ``text``'s printable token p."""
    if not 0 <= fid < len(OUT_NAMES):
        return -1
    nm = OUT_NAMES[fid]
    if nm == "Q":
        q = text.printable[p] if text and p < len(text.printable) else None
        return -1 if q is None else S.TEXT_TOKEN + q
    if nm in S.CONTROLS:
        return S.tok(nm)
    return S.tok(nm, p) if p < S.FAMILIES[nm][1] else -1


def predictions(c: RvCore, mat: np.ndarray) -> list[int]:
    """The predicted next token at every position (from the output code dims)."""
    def bits(es):
        return [np.rint(sum(mat[:, d] * float(w_) for d, w_ in e.terms.items()))
                if e.terms else np.zeros(mat.shape[0]) for e in es]

    if c.out_fam is None:  # merged: the exit's code dims (KAPPA * (2 bit - 1) where it owns)
        cb = [(sum(mat[:, d] * float(w_) for d, w_ in e.terms.items()) > 0).astype(int)
              for e in c.code]
        fb, pb = cb[:NFAM], cb[NFAM:]
    else:
        fb, pb = bits(c.out_fam), bits(c.out_pay)
    out = []
    for t in range(mat.shape[0]):
        fid = int(sum(int(fb[i][t]) << i for i in range(NFAM)))
        p = int(sum(int(pb[i][t]) << i for i in range(NPAY)))
        out.append(decode_code(fid, p, c.text))
    return out
