"""The RV32I core as a q35 circuit (P0 skeleton; spec.py has the trace format).

Every position computes the instruction that the next token belongs to: the state it reads is
the state after the position's own commit, so at a record's last token the circuit already works
on the next instruction, and inside a record (V tokens) it recomputes the current one (nothing
changes until the record's last token). Writes take their key from t-1 (a conv tap), their write
gate beta from the token at t, and their value from token payloads (lags 0..3) or from dims the
circuit computed at t-1 below the writing layer.

  L1  attn  anchor (PC head), count (2^-n), previous-token kinds
      mlp   n thermometer; j (the record's tokens so far)
  L2  mlp   PC word w = anchor + n; link word w + 1
  L3  attn  ROM: the pre-decoded fields of word w (byte columns, 2-hot query, z word select)
      mlp   field one-hots, immediate bits, gated link
  L4  attn  register file (2 read ports; writes commit from t-1)
      mlp   adder stage 1 (4 operand modes), logic ops, shifter stage 1, store data
  L5  mlp   adder stage 2 (carries + result bits), shifter stage 2
  L6  attn  RAM (1 sublayer: 16 key heads x 2 value heads x 128 rows of one word)
      mlp   load lanes, SB/SH merge, compares
  L7  mlp   rd value select, its size, branch decision
  L8  mlp   output code: the next token
"""


import numpy as np

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import A_BIT, A_REF, HD, REF_COLS, Chan, K, Target
from chipformer.q35.prims import Circ, chan, from_bits, sel, thead
from chipformer.rv import ref as RF
from chipformer.rv import spec as S

T4B = Target(key_heads=16, value_heads=32, intermediate=9216)  # Qwen3.5-4B's layers
FAM_ID = {n: i for i, n in enumerate(["V", "F0", "F1", "F2", "J", "R", "NT", "ST0", "ST1", "SS0",
                                       "SS1", "HALT", "GO"])}
NFAM = 4  # family id bits in the output code
OUT_W = 2.0**7  # head output weight (as thead)


def lits_of(bits, v):
    """0/1 literals that are all 1 exactly when the bits spell v."""
    return [b if v >> i & 1 else 1 - b for i, b in enumerate(bits)]


def andpre(lits) -> Expr:
    """sum(lits) - (n - 1): 1 when all literals are 1, <= 0 otherwise (a key / query / z form)."""
    return sum(lits, Expr()) - (len(lits) - 1)


class RvCore(Circ):
    def __init__(self, target: Target | None = None) -> None:
        super().__init__(target or T4B)
        f = self.token_feature

        def fam_is(nm):
            return lambda t: int(S.family(t)[0] == nm)

        def pay_bit(nm, i):
            return lambda t: (S.family(t)[1] >> i & 1) if S.family(t)[0] == nm else 0

        names = list(S.FAMILIES) + list(S.CONTROLS)
        self.fam = {n: f(f"is{n}", fam_is(n), full=True) for n in names}
        width = {"V": 15, "F0": 15, "F1": 15, "F2": 2, "J": 15, "R": 15}
        self.pay = {n: [f(f"{n}.b{i}", pay_bit(n, i), full=True) for i in range(w)]
                    for n, w in width.items()}
        self.final = sum((self.fam[n] for n in S.FINAL), Expr())
        self.tables = {}  # name -> head node (for snapshots)

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

    def byte_bits(self, z: Expr, name: str, n: int = 8) -> list:
        """Bits of an integer 0..255 (thresholds at every flip of each bit)."""
        out = []
        for j in range(n):
            terms = []
            for theta in range(1, 256):
                d = (theta >> j & 1) - ((theta - 1) >> j & 1)
                if d:
                    terms += self.t_ge(z, theta, 0, 255, d)
            out.append(self.node(terms, f"{name}.b{j}"))
        return out


def build(target: Target | None = None) -> RvCore:
    c = RvCore(target)
    one, fam, pay = c.one, c.fam, c.pay
    isV, isJR = fam["V"], fam["J"] + fam["R"]
    isSS = fam["SS0"] + fam["SS1"]

    # ---------------------------------------------------------------- L1: anchor, count, prev
    jr_bits = [pay["J"][i] + pay["R"][i] for i in range(15)]
    (anchor,) = thead(c, "pc", {0: one}, {0: one},
                      [{"data": {2 + i: jr_bits[i] for i in range(15)}, "ref": one, "beta": isJR}])
    c.tables["pc"] = c.nodes[-1]
    commit = c.final - isJR
    ((xn,),) = thead(c, "count", {0: isJR, 1: one - isJR}, {0: one - isJR, 1: one - isJR},
                     [{"data": {2: isJR}, "ref": one - isJR, "a": 110 * commit + 238 * isJR}],
                     tiny=2.0**-90)
    c.tables["count"] = c.nodes[-1]
    ((pv1, pv2, _pv3),) = thead(c, "prev", {0: one}, {0: one},
                               [{"data": {2: Chan(isV, (0, 1, 0, 0)), 3: Chan(isV, (0, 0, 1, 0)),
                                          4: Chan(isV, (0, 0, 0, 1))}, "ref": one}])
    # n thermometer: cnt[m] = [n > m] (0 at J / R, where n restarts)
    cnt = [c.node([(1, one - isJR, 7 * one - 8 * 2**m * xn),
                   (-1, one - isJR, 6 * one - 8 * 2**m * xn)], f"n>{m}") for m in range(S.N_MAX)]
    n_int = sum(cnt, Expr())
    at_max = cnt[S.N_MAX - 1]  # n == 63: the refresh comes first
    j0 = one - isV
    j1 = c.and_([isV, one - pv1], "j1")
    j2 = c.and_([isV, pv1, one - pv2], "j2")
    j3 = c.and_([isV, pv1, pv2], "j3")

    # ---------------------------------------------------------------- L2: PC word, link
    def add_small(abits, extra, name):
        """Bits of (anchor + n + extra) for n <= 63: low 6 bits by thresholds, the rest by the
        carry into bit 6."""
        L = from_bits(abits[:6]) + n_int + extra
        lo, hi = 0, 63 + 63 + extra  # (from 0, so every flip above 0 has its threshold)
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
    w1 = add_small(anchor, 1, "w+1")  # the link: PC + 4 = 4 * (w + 1)

    # ---------------------------------------------------------------- L3: ROM fetch
    # word w: key head w[13:10], row pair (p, p + 64) with p = w[9:4], half w[3] (row p holds
    # half 0 in columns 2..33, row p + 64 half 1 in 34..65), word w[2:0], 4 byte columns; value
    # head 0 = (uop, rdw, rs1, rs2), value head 1 = the immediate's 4 bytes
    fdims = c.zeros(4, "rom.f")
    idims = c.zeros(4, "rom.imm")
    rom_heads = S.ROM_WORDS >> 10
    for h in range(rom_heads):
        hl = lits_of(w[10:14], h)
        q = {}
        for p in range(64):
            e = andpre(lits_of(w[4:10], p) + hl)
            q[p], q[p + 64] = e, e
        vhs = []
        for x, dims in ((0, fdims), (1, idims)):
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
            if u not in ("DIV", "DIVU", "REM", "REMU", "MUL", "MULH", "MULHSU", "MULHU")}

    def any_of(*names):
        return sum((uops[n] for n in names), Expr())

    rdw = {r: c.node(c.t_eq(RDW, r, 0, 31), f"rdw={r}") for r in range(1, 32)}
    rs1 = {r: c.node(c.t_eq(RS1, r, 0, 31), f"rs1={r}") for r in range(1, 32)}
    rs2 = {r: c.node(c.t_eq(RS2, r, 0, 31), f"rs2={r}") for r in range(1, 32)}
    rdw_any = sum(rdw.values(), Expr())
    imm = [b for k in range(4) for b in c.byte_bits(IMMB[k], f"imm{k}")]
    # PC + 4 = ROM_BASE + 4 * (w + 1): bits 2..15 from w + 1 (a 64 KB ROM), bit 16 = ROM_BASE
    assert S.ROM_BASE == 1 << 16 and S.ROM_WORDS == 1 << 14
    link = [Expr()] * 2 + w1[:14] + [one] + [Expr()] * 15
    jl = [Expr() if not link[i].terms else
          c.node(c.t_eq(U, RF.U["JAL"], 0, 255, lits=[link[i]] if i < 16 else [])
                 + c.t_eq(U, RF.U["JALR"], 0, 255, lits=[link[i]] if i < 16 else []),
                 f"link.b{i}")
          for i in range(32)]

    # ---------------------------------------------------------------- L4: register file
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

    # ---------------------------------------------------------------- L4-L5: ALU
    md = {"ADD": any_of("ADD"),
          "ADDI": any_of("ADDI", "LB", "LH", "LW", "LBU", "LHU", "SB", "SH", "SW", "JALR"),
          "SUB": any_of("SUB", "SLT", "SLTU", "BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"),
          "SUBI": any_of("SLTI", "SLTIU")}
    bsrc = {"ADD": (B, False, 0), "ADDI": (imm, False, 0), "SUB": (B, True, 1),
            "SUBI": (imm, True, 1)}

    def nib(bits, k, inv=False):
        x = from_bits(bits[4 * k:4 * k + 4])
        return 15 - x if inv else x

    g, p_, d = [], [], []  # per nibble: generate, propagate, digit bits (s mod 16)
    for k in range(8):
        gt, pt, dt = [], [], [[] for _ in range(4)]
        for m, lit in md.items():
            bb, inv, cin = bsrc[m]
            s = nib(A, k) + nib(bb, k, inv) + (cin if k == 0 else 0)
            hi = 31
            gt += c.t_ge(s, 16, 0, hi, lits=[lit])
            pt += c.t_eq(s, 15, 0, hi, lits=[lit])
            for b in range(4):
                for theta in range(1, hi + 1):
                    dd = (theta % 16 >> b & 1) - ((theta - 1) % 16 >> b & 1)
                    if dd:
                        dt[b] += c.t_ge(s, theta, 0, hi, dd, lits=[lit])
        g.append(c.node(gt, f"add.g{k}"))
        p_.append(c.node(pt, f"add.p{k}"))
        d.append([c.node(dt[b], f"add.d{k}.{b}") for b in range(4)])

    def carry_sources(k):
        """Mutually exclusive literal sets, one of them true iff a carry enters nibble k."""
        return [[g[j]] + p_[j + 1:k] for j in range(k)]

    r = []
    for k in range(8):
        for b in range(4):
            terms = c.t_and([d[k][b]])
            for src in carry_sources(k):
                low = d[k][:b]
                terms += c.t_and(src + low + [one - d[k][b]])
                terms += c.t_and(src + low + [d[k][b]], -1)
            r.append(c.node(terms, f"add.r{4 * k + b}"))
    c32 = c.node([t for src in carry_sources(8) for t in c.t_and(src)], "add.c32")

    # logic ops (one dim per bit, gated by the op)
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

    # shifter: by 4 * shamt[4:2], then by shamt[1:0]; left, logical right, arithmetic right
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

    # store data for SW (0 on V tokens, so an SB/SH record's last V does not leak it)
    sd = [c.and_([B[i], one - isV], f"sd.b{i}") for i in range(32)]
    is_store = any_of("SB", "SH", "SW")

    # ---------------------------------------------------------------- L6: RAM
    # word index wi = EA[13:2] (RAM at 0): key head wi[3:0] = EA[5:2], value head wi[4] = EA[6],
    # row wi[11:5] = EA[13:7]
    EA = r
    mem = c.zeros(32, "mem")
    for h in range(16):
        hl = lits_of(EA[2:6], h)
        keys, q = {}, {}
        for row in range(128):
            rl = lits_of(EA[7:14], row)
            keys[row] = Chan(andpre(rl + hl + [is_store]), (0, 1, 0, 0))
            q[row] = andpre(rl + hl)
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

    # ---------------------------------------------------------------- L6: loads, merge, compares
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

    # ---------------------------------------------------------------- L7: rd value, size, branch
    addish = any_of("ADD", "ADDI", "SUB")
    sltish, sltuish = any_of("SLT", "SLTI"), any_of("SLTU", "SLTIU")
    logicops = any_of("AND", "ANDI", "OR", "ORI", "XOR", "XORI")
    shiftops = any_of("SLL", "SLLI", "SRL", "SRLI", "SRA", "SRAI")
    loads = any_of("LB", "LH", "LW", "LBU", "LHU")
    srcs = [(addish, r), (uops["CONST"], imm), (logicops, logic), (shiftops, shift), (loads, ld)]
    val, k0t, k1t = [], [], []
    for i in range(32):
        terms = []
        for gate, bits in srcs:
            terms += c.t_and([bits[i], gate])
        if i == 0:
            terms += c.t_and([slt, sltish]) + c.t_and([sltu, sltuish])
        val.append(c.node(terms, f"val.b{i}"))
    # size: the value fits F0 (bits 14..31 equal) / F1 (bits 29..31 equal); per source, so it
    # is ready in the same stage as the value (slt values are 0/1: they always fit F0)
    for gate, bits in srcs:
        k0t += c.t_and([gate] + bits[14:32]) + c.t_and([gate] + [1 - b for b in bits[14:32]])
        k1t += c.t_and([gate] + bits[29:32]) + c.t_and([gate] + [1 - b for b in bits[29:32]])
    k0t += c.t_and([sltish + sltuish])
    k1t += c.t_and([sltish + sltuish])
    gate_any = addish + uops["CONST"] + logicops + shiftops + loads
    k0 = c.node(k0t, "fits15")
    k1 = c.node(k1t, "fits30")
    taken = c.node(c.t_and([uops["BEQ"], eq]) + c.t_and([uops["BNE"], 1 - eq])
                   + c.t_and([uops["BLT"], slt]) + c.t_and([uops["BGE"], 1 - slt])
                   + c.t_and([uops["BLTU"], sltu]) + c.t_and([uops["BGEU"], 1 - sltu]),
                   "taken")

    # ---------------------------------------------------------------- L8: the next token
    valc = gate_any + sltish + sltuish
    nr = [j0, one - at_max]  # the first token of a record (no refresh)
    cases = []  # (literals, family, payload bits or int)

    def case(lits, fam_name, payload=0):
        cases.append((lits, fam_name, payload))

    case([j0, at_max], "R", w)
    case(nr + [valc, rdw_any, k0], "F0", val[0:15])
    case(nr + [valc, rdw_any, 1 - k0], "V", val[0:15])
    case([j1, valc, rdw_any, k1], "F1", val[15:30])
    case([j1, valc, rdw_any, 1 - k1], "V", val[15:30])
    case([j2, valc, rdw_any], "F2", val[30:32])
    case(nr + [valc, 1 - rdw_any], "NT")
    branches = any_of("BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU")
    case(nr + [branches, taken], "J", imm[0:15])
    case(nr + [branches, 1 - taken], "NT")
    case(nr + [uops["JAL"]], "J", imm[0:15])
    case(nr + [uops["JALR"]], "J", r[2:16])  # (the target's word in ROM; bit 16 is ROM_BASE)
    case(nr + [uops["SW"], 1 - EA[6]], "ST0")
    case(nr + [uops["SW"], EA[6]], "ST1")
    sbsh = any_of("SB", "SH")
    case(nr + [sbsh], "V", merged[0:15])
    case([j1, sbsh], "V", merged[15:30])
    case([j2, sbsh], "V", merged[30:32])
    case([j3, sbsh, 1 - EA[6]], "SS0")
    case([j3, sbsh, EA[6]], "SS1")
    case(nr + [any_of("ECALL", "EBREAK", "ILLEGAL")], "HALT")
    case(nr + [uops["FENCE"]], "NT")
    fam_bits = c.select_bits([(l_, FAM_ID[fn]) for l_, fn, _ in cases], NFAM, "out.fam")
    pay_bits = c.select_bits([(l_, pl) for l_, _, pl in cases if not isinstance(pl, int)], 15,
                             "out.pay")
    c.out_fam, c.out_pay = fam_bits, pay_bits
    c.probe = {"w": w, "anchor": anchor, "cnt": cnt, "j": (j1, j2, j3), "U": U, "A": A, "B": B,
               "imm": imm, "r": r, "val": val, "mem": mem, "ld": ld, "k0": k0, "k1": k1,
               "taken": taken, "logic": logic, "shift": shift, "merged": merged}
    return c


# --- snapshots (the state the host injects) ---------------------------------------------------


def reg_row(c: RvCore, value: int) -> np.ndarray:
    row = np.zeros(HD)
    row[list(REF_COLS)] = A_REF
    for i in range(32):
        if value >> i & 1:
            row[c.reg_cols[i][0]] = A_BIT
    return row


def snapshot(c: RvCore, m: "RF.Machine") -> None:
    """Load a machine's state into the circuit's tables (the heads' initial states)."""
    pc = np.zeros(HD)
    pc[list(REF_COLS)] = A_REF
    for i in range(15):
        pc[2 + i] = A_BIT * (m.anchor >> i & 1)
    c.tables["pc"].init = [{0: pc}]
    cnt0 = np.zeros(HD)
    cnt0[2] = A_BIT * 2.0**-m.n
    cnt1 = np.zeros(HD)
    cnt1[list(REF_COLS)] = A_REF
    c.tables["count"].init = [{0: cnt0, 1: cnt1}]
    regs = {r: reg_row(c, m.x[r]) for r in range(1, 32) if True}
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
    for h in range(16):
        r0, r1 = {}, {}
        for wi in range(S.RAM_WORDS):
            if wi & 15 != h or not m.ram[wi]:
                continue
            row = np.zeros(HD)
            row[list(REF_COLS)] = A_REF
            for i in range(32):
                row[2 + i] = A_BIT * (m.ram[wi] >> i & 1)
            (r1 if wi >> 4 & 1 else r0)[wi >> 5 & 127] = row
        c.tables[f"ram{h}"].init = [r0, r1]


def predictions(c: RvCore, mat: np.ndarray) -> list[int]:
    """The predicted next token at every position (from the output code dims)."""
    def bits(es):
        return [np.rint(sum(mat[:, d] * float(w_) for d, w_ in e.terms.items()))
                if e.terms else np.zeros(mat.shape[0]) for e in es]

    fb, pb = bits(c.out_fam), bits(c.out_pay)
    names = {i: n for n, i in FAM_ID.items()}
    out = []
    for t in range(mat.shape[0]):
        fid = int(sum(int(fb[i][t]) << i for i in range(NFAM)))
        p = int(sum(int(pb[i][t]) << i for i in range(15)))
        nm = names.get(fid)
        if nm is None:
            out.append(-1)
        elif nm in S.CONTROLS:
            out.append(S.tok(nm))
        else:
            size = S.FAMILIES[nm][1]
            out.append(S.tok(nm, p) if p < size else -1)
    return out
