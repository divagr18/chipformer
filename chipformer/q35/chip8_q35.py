"""CHIP-8 as a q35 circuit: the same machine as chipformer/chip8.py, built for Qwen3.5 layers.

Same structure as the original: every position computes the instruction context from the state
it reads (meaningful at STEP/BOOT positions), a DeltaNet slot carries that context from the
latest record start to the rest of the record, and a per-position micro-step picks the output.
What changes is how state is read:

  memory     the delta rule with one-hot keys overwrites a state row, so a cell's row always
             holds its latest write; a read is a one-hot query. RAM is two layers of 16 heads
             (head = address bits 3:0, so PC and PC+1 always land in different heads).
  registers  one head, one row per register in its own column block, read all at once.
  positions  none: the write index j comes from a head whose row halves at every VAL token
             and resets at STEP/BOOT (x = 2^-j), and "latest STEP/TICK" style lookups are
             rows overwritten at those tokens.

All arithmetic is on bits, one-hot nibbles and small integers, so it is exact in bf16.
"""

from chipformer import spec
from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Chan, Target
from chipformer.q35.prims import Circ, cat, from_bits, onehot_bits

JMAX = 40  # write indices above this all read as JMAX + 1
NCODE = 15  # output code: [is_addr, is_ctrl, payload bits 0..12]
CODE_AMP = 128
V, VF = spec.V, spec.V + 0xF


def token_code(t: int) -> list[int]:
    """0/1 code of a chip token; the embedding stores CODE_AMP * (2 * bit - 1)."""
    if spec.is_val(t):
        kind, payload = (0, 0), t - spec.VAL_BASE
    elif spec.is_addr(t):
        kind, payload = (1, 0), t - spec.ADDR_BASE
    else:
        kind, payload = (0, 1), t - spec.STEP
    return [*kind] + [payload >> i & 1 for i in range(13)]


def decode_code(bits: list[int]) -> int:
    payload = sum(b << i for i, b in enumerate(bits[2:]))
    if bits[1]:
        return spec.STEP + payload
    return (spec.ADDR_BASE if bits[0] else spec.VAL_BASE) + payload


class Chip8Q35(Circ):
    def __init__(self, target: Target | None = None) -> None:
        super().__init__(target or Target())
        f = self.token_feature
        isv, isa = spec.is_val, spec.is_addr
        self.is_val = f("is_val", lambda t: int(isv(t)))
        self.is_addr = f("is_addr", lambda t: int(isa(t)))
        self.is_step = f("is_step", lambda t: int(t == spec.STEP))
        self.is_tick = f("is_tick", lambda t: int(t == spec.TICK))
        self.is_boot = f("is_boot", lambda t: int(t == spec.BOOT))
        self.vb = [f(f"vb{i}", lambda t, i=i: (t - spec.VAL_BASE) >> i & 1 if isv(t) else 0)
                   for i in range(12)]
        self.an = [[f(f"a{k}[{x}]",
                      lambda t, k=k, x=x: int(isa(t) and (t - spec.ADDR_BASE) >> 4 * k & 15 == x))
                    for x in range(16)] for k in range(3)]
        self.a12 = f("a12", lambda t: int(isa(t) and (t - spec.ADDR_BASE) >> 12 & 1))
        self.is_cls_addr = f("is_cls_addr", lambda t: int(t == spec.addr_token(spec.CLS)))
        self.code = [f(f"code{i}", lambda t, i=i: CODE_AMP * (2 * token_code(t)[i] - 1))
                     for i in range(NCODE)]
        # tau: -1 on chip tokens (cancelled by the last layer). In a merged model the text
        # tokens' rows hold a tiny tau and the final norm amplifies it, pushing the chip tokens'
        # logits far down at text positions.
        self.tau = f("tau", lambda t: -1)
        self.rs = self.is_step + self.is_boot  # record start

    def addr_lits(self, a: int) -> list:
        """0/1 literals that are all 1 exactly at an ADDR a token (one-hot, no constants)."""
        lits = [self.an[k][a >> 4 * k & 15] for k in range(3)]
        return lits + [self.a12] if a >> 12 else lits

    def write_key(self, a: int) -> Chan:
        """Key pre-activation (AND units) for a write to cell a, seen at its VAL token."""
        lits = self.addr_lits(a)
        if not a >> 12:
            lits = lits + [self.one - self.a12]
        return Chan(sum(lits, Expr()) - (len(lits) - 1), (0, 1, 0, 0))

    def bank(self, name: str, cells: dict[int, int], query: dict, nbits: int, a=0) -> list:
        """One head of trace-written cells (cell -> slot); returns the read bits."""
        keys = {s: self.write_key(c) for c, s in cells.items()}
        data = {2 + i: self.vb[i] for i in range(nbits)}
        return self.slot_head(name, keys, query, data, self.one, a)


def build(target: Target | None = None) -> Chip8Q35:
    c = Chip8Q35(target)
    one, rs = c.one, c.rs

    # ------------------------------------------------------------ registers (one head)
    regs = [("PC", spec.PC, 12), ("I", spec.I, 12), ("SP", spec.SP, 4), ("DT", spec.DT, 8),
            ("ST", spec.ST, 8), ("RNG", spec.RNG, 12), ("V0", V, 8)]
    regs += [(f"key{k:X}", spec.KEYS + k, 1) for k in range(16)]
    regs += [(f"q.{q}", spec.QUIRKS + i, 1) for i, q in enumerate(spec.QUIRK_NAMES)]
    keys, data, col, blocks = {127: Chan(c.is_boot)}, {}, 2, {}
    for slot, (name, cell, nb) in enumerate(regs):
        keys[slot] = c.write_key(cell)
        lits = c.addr_lits(cell)
        blocks[name] = list(range(col, col + nb))
        for i in range(nb):
            # the VAL bit (this token) AND the address (previous token), over both conv taps
            data[col] = Chan(c.vb[i] + sum(lits, Expr()) - one * (len(lits) // 2), (1, 1, 0, 0))
            col += 1
    out = c.slot_head("regs", keys, {s: one for s in keys}, data, Chan(c.is_boot))
    by_col = dict(zip(sorted(data), out))
    R = {name: [by_col[k] for k in cols] for name, cols in blocks.items()}
    pcb, ib, spb, dtb, stb, rngb, v0b = (R[k] for k in ("PC", "I", "SP", "DT", "ST", "RNG", "V0"))
    keyb = [R[f"key{k:X}"][0] for k in range(16)]
    q = {name: R[f"q.{name}"][0] for name in spec.QUIRK_NAMES}

    # ------------------------------------------------------------ micro-step counter
    (x,) = c.slot_head("counter", {0: rs, 1: one - rs}, {0: one - rs, 1: one - rs},
                       {2: rs}, one - rs, 110 * c.is_val + 238 * rs, tiny=2.0**-60)
    cnt = [c.node([(1, one - rs, 7 * one - 8 * 2**m * x), (-1, one - rs, 6 * one - 8 * 2**m * x)],
                  f"j>{m}") for m in range(JMAX + 1)]  # cnt[m] = [j > m]
    j = sum(cnt, Expr())

    def jeq(v):  # [j == v], a 0/1 linear form
        return (one if v == 0 else cnt[v - 1]) - (cnt[v] if v <= JMAX else 0)

    (in_tick,) = c.slot_head("in_tick", {0: rs + c.is_tick}, {0: one}, {2: c.is_tick},
                             rs + c.is_tick)

    # ------------------------------------------------------------ display bytes ("old")
    disp_q = {}
    old = [Expr() for _ in range(8)]
    for h in range(2):
        cells = {spec.DISPLAY + (h << 7) + s: s for s in range(128)}
        query = {s: sum(c.addr_lits(cell), Expr()) - 3 for cell, s in cells.items()}
        disp_q[h] = query
        bits = c.bank(f"display{h}", cells, query, 8, 238 * c.is_cls_addr)
        old = [o + b for o, b in zip(old, bits)]

    # ------------------------------------------------------------ PC one-hots, opcode fetch
    oh = [c.onehot_of_bits(pcb[4 * k:4 * k + 4], f"pc.n{k}") for k in range(3)]
    c4 = pcb[0:4]
    c8 = pcb[0:8]

    def nib_lits(bits, v):
        return [b if v >> i & 1 else one - b for i, b in enumerate(bits)]

    def carry_onehot(bits, carry_bits, name):
        out = []
        for v in range(16):
            here, below = nib_lits(bits, v), nib_lits(bits, (v - 1) % 16)
            terms = (c.t_and(here) + c.t_and(here + carry_bits, -1)
                     + c.t_and(below + carry_bits))
            out.append(c.node(terms, f"{name}[{v}]"))
        return out

    oh1_pc1 = carry_onehot(pcb[4:8], c4, "pc1.n1")
    oh2_pc1 = carry_onehot(pcb[8:12], c8, "pc1.n2")
    oh0_pc1 = [oh[0][(v - 1) % 16] for v in range(16)]
    even, odd = [Expr() for _ in range(8)], [Expr() for _ in range(8)]
    for half in range(2):
        for h in range(16):
            n1, n2 = (oh1_pc1, oh2_pc1) if h == 0 else (oh[1], oh[2])
            cells = {(half << 11) | (s >> 4 << 8) | ((s & 15) << 4) | h: s for s in range(128)}
            query = {s: oh[0][h] + oh0_pc1[h] + n1[s & 15] + n2[(s >> 4) + 8 * half] - 2
                     for s in range(128)}
            bits = c.bank(f"ramA{half}.{h}", cells, query, 8)
            acc = even if h % 2 == 0 else odd
            for i in range(8):
                acc[i] = acc[i] + bits[i]
    pc0 = pcb[0]
    b1 = [c.node(c.t_and([pc0, odd[i]]) + c.t_and([one - pc0, even[i]]), f"op_hi.b{i}")
          for i in range(8)]
    b2 = [c.node(c.t_and([pc0, even[i]]) + c.t_and([one - pc0, odd[i]]), f"op_lo.b{i}")
          for i in range(8)]

    # ------------------------------------------------------------ decode
    hib, xb, yb, nb = b1[4:], b1[:4], b2[4:], b2[:4]
    ind = {}
    for h, name in [(1, "1NNN"), (2, "2NNN"), (3, "3XNN"), (4, "4XNN"), (6, "6XNN"),
                    (7, "7XNN"), (0xA, "ANNN"), (0xB, "BNNN"), (0xC, "CXNN"), (0xD, "DXYN")]:
        ind[name] = c.and_(nib_lits(hib, h), name)
    ind["00E0"] = c.and_(nib_lits(b1[:4], 0) + nib_lits(b1[4:], 0) + nib_lits(b2[:4], 0)
                         + nib_lits(b2[4:], 0xE), "00E0")
    ind["00EE"] = c.and_(nib_lits(b1[:4], 0) + nib_lits(b1[4:], 0) + nib_lits(b2[:4], 0xE)
                         + nib_lits(b2[4:], 0xE), "00EE")
    for h, name in [(5, "5XY0"), (9, "9XY0")]:
        ind[name] = c.and_(nib_lits(hib, h) + nib_lits(nb, 0), name)
    for m in [0, 1, 2, 3, 4, 5, 6, 7, 0xE]:
        ind[f"8XY{m:X}"] = c.and_(nib_lits(hib, 8) + nib_lits(nb, m), f"8XY{m:X}")
    for h, code in [(0xE, 0x9E), (0xE, 0xA1), (0xF, 0x07), (0xF, 0x0A), (0xF, 0x15),
                    (0xF, 0x18), (0xF, 0x1E), (0xF, 0x29), (0xF, 0x33), (0xF, 0x55), (0xF, 0x65)]:
        name = f"{h:X}X{code:02X}"
        ind[name] = c.and_(nib_lits(hib, h) + nib_lits(b2[:4], code & 15)
                           + nib_lits(b2[4:], code >> 4), name)
    x_oh = c.onehot_of_bits(xb, "x")
    y_oh = c.onehot_of_bits(yb, "y")
    n_oh = c.onehot_of_bits(nb, "n")

    # ------------------------------------------------------------ V registers, stack
    vcells = {V + r: r for r in range(16)}
    vxb = c.bank("VX", vcells, {r: x_oh[r] for r in range(16)}, 8)
    vyb = c.bank("VY", vcells, {r: y_oh[r] for r in range(16)}, 8)
    sp_oh = c.onehot_of_bits(spb, "sp")
    stack_top = c.bank("stack", {spec.STACK + r: r for r in range(16)},
                       {r: sp_oh[(r + 1) % 16] for r in range(16)}, 12)

    # ------------------------------------------------------------ ALU
    vx, vy, nn = from_bits(vxb), from_bits(vyb), from_bits(b2)
    def lo4(bits):
        return from_bits(bits[:4])


    def nibbles8(s0, s0_lo, s0_hi, z, z_lo, z_hi, name):
        """Bits of z mod 256, given s0 == z (mod 16) over a small range."""
        r0 = []
        for v in range(16):
            terms = []
            for k in range(-2, 3):
                terms += c.t_eq(s0, v + 16 * k, s0_lo, s0_hi)
            r0.append(c.node(terms, f"{name}.n0[{v}]"))
        off = -(z_lo // 256) * 256 if z_lo < 0 else 0
        r1 = c.nib_onehot(z + off, z_lo + off, z_hi + off, f"{name}.n1", div=16)
        return onehot_bits(r0, 4) + onehot_bits(r1, 4)

    add_xy = nibbles8(lo4(vxb) + lo4(vyb), 0, 30, vx + vy, 0, 510, "vx+vy")
    add_xn = nibbles8(lo4(vxb) + lo4(b2), 0, 30, vx + nn, 0, 510, "vx+nn")
    sub_xy = nibbles8(lo4(vxb) - lo4(vyb), -15, 15, vx - vy, -255, 255, "vx-vy")
    sub_yx = nibbles8(lo4(vyb) - lo4(vxb), -15, 15, vy - vx, -255, 255, "vy-vx")
    carry = c.node(c.t_ge(vx + vy, 256, 0, 510), "carry")
    ge_xy = c.node(c.t_ge(vx - vy, 0, -255, 255), "vx>=vy")
    ge_yx = c.node(c.t_ge(vy - vx, 0, -255, 255), "vy>=vx")
    eq_x_nn = c.node(c.t_eq(vx - nn, 0, -255, 255), "vx==nn")
    eq_x_y = c.node(c.t_eq(vx - vy, 0, -255, 255), "vx==vy")
    and_xy = [c.and_([xa, ya], f"vx&vy.b{i}") for i, (xa, ya) in enumerate(zip(vxb, vyb))]
    or_xy = [xa + ya - a for xa, ya, a in zip(vxb, vyb, and_xy)]
    xor_xy = [xa + ya - 2 * a for xa, ya, a in zip(vxb, vyb, and_xy)]
    src = [c.node(c.t_and([q["shift"], xa]) + c.t_and([one - q["shift"], ya]), f"src.b{i}")
           for i, (xa, ya) in enumerate(zip(vxb, vyb))]
    shr, shl = src[1:] + [0], [0] + src[:7]

    # BCD of VX
    hund = c.node(c.t_ge(vx, 100, 0, 255) + c.t_ge(vx, 200, 0, 255), "vx/100")
    tens_all = c.node(cat(c.t_ge(vx, 10 * k, 0, 255) for k in range(1, 26)), "vx/10")
    ones_d = vx - 10 * tens_all
    tens_d = tens_all - 10 * hund

    def small_bits(z, hi, name):
        oh_ = [c.node(c.t_eq(z, v, 0, hi), f"{name}=={v}") for v in range(hi + 1)]
        return onehot_bits(oh_, 4)

    bcd = [small_bits(hund, 2, "hund"), small_bits(tens_d, 9, "tens"),
           small_bits(ones_d, 9, "ones")]

    # keys
    def key_is(k):  # literals for VX & 0xF == k
        return nib_lits(vxb[:4], k)

    key_vx = c.node(cat(c.t_and(key_is(k) + [keyb[k]]) for k in range(16)), "key[vx]")
    none_held = [one - kb for kb in keyb]
    fx0a_val = []
    for i in range(8):
        terms = [t for k in range(16) if k >> i & 1
                 for t in c.t_and([keyb[k]] + none_held[:k])]
        terms += c.t_and(none_held + [vxb[i]])
        fx0a_val.append(c.node(terms, f"fx0a.b{i}"))
    any_key = c.node(c.t_const(1) + c.t_and(none_held, -1), "any_key")

    # CXNN: one Galois LFSR step, then the low byte AND NN
    shifted = rngb[1:] + [0]
    rng_next, rnd = [], []
    for i in range(12):
        if spec.RNG_MASK >> i & 1:
            a_, b_ = shifted[i], rngb[0]
            if isinstance(a_, int):
                bit = b_
            else:
                bit = c.node(c.t_and([a_, one - b_]) + c.t_and([one - a_, b_]), f"lfsr{i}")
            rng_next.append(bit)
            if i < 8:
                rnd.append(c.node(c.t_and([b2[i], a_, one - b_]) + c.t_and([b2[i], one - a_, b_])
                                  if not isinstance(a_, int) else c.t_and([b2[i], b_]),
                                  f"rnd.b{i}"))
        else:
            rng_next.append(shifted[i])
            if i < 8:
                rnd.append(c.and_([b2[i], shifted[i]], f"rnd.b{i}"))

    # FX29: I = FONT + 5 * (VX & 0xF)
    font_i = []
    for i in range(12):
        terms = [t for v in range(16) if (spec.FONT_BASE + 5 * v) >> i & 1
                 for t in c.t_and(key_is(v))]
        font_i.append(c.node(terms, f"font.b{i}"))

    def dec8(bits, name):
        """Bits of max(v - 1, 0)."""
        out = []
        for i in range(len(bits)):
            neg = [one - b for b in bits[:i]]  # x - 1 flips bit i iff all lower bits are 0
            terms = [(1, one, bits[i])] + (c.t_and(neg) if neg else c.t_const(1))
            terms += c.t_and([bits[i], *neg], -2) + c.t_and([one - b for b in bits], -1)
            out.append(c.node(terms, f"{name}.b{i}"))
        return out

    dt_dec, st_dec = dec8(dtb, "dt-1"), dec8(stb, "st-1")
    sp_dec = onehot_bits([sp_oh[(v + 1) % 16] for v in range(16)], 4)
    sp_inc = onehot_bits([sp_oh[(v - 1) % 16] for v in range(16)], 4)

    # 12-bit sums
    ihi = c.onehot_of_bits(ib[8:12], "I.n2")
    i_lo = from_bits(ib[:8])

    def bits12(nibs):
        return [b for n_ in nibs for b in onehot_bits(n_, 4)]

    pc_lo = from_bits(pcb[:8])
    pc2 = bits12(c.add12(pc_lo, oh[2], 2, 2, "pc+2"))
    pc4 = bits12(c.add12(pc_lo, oh[2], 4, 4, "pc+4"))
    i_vx = bits12(c.add12(i_lo, ihi, vx, 255, "I+vx"))
    i_p = [ib, bits12(c.add12(i_lo, ihi, 1, 1, "I+1")), bits12(c.add12(i_lo, ihi, 2, 2, "I+2"))]
    xint = from_bits(xb)
    i_next = bits12(c.add12(i_lo, ihi, xint + 1, 16, "I+x+1"))
    jump_reg = [c.node(c.t_and([q["jump"], xa]) + c.t_and([one - q["jump"], va]), f"jreg.b{i}")
                for i, (xa, va) in enumerate(zip(vxb, v0b))]
    bnnn = bits12(c.add12(nn, x_oh, from_bits(jump_reg), 255, "bnnn"))
    nnn = list(b2) + list(xb)

    # ------------------------------------------------------------ fixed writes
    vxa = list(xb) + [0] * 8 + [1]  # V + x: 0x1000 | x
    stack_sp = list(spb) + [0, 1, 0, 0, 0, 0, 0, 0, 1]  # 0x1020 | sp
    i_a = [list(b) + [0] for b in i_p]
    fixed = [
        ([ind["00E0"]], [(spec.CLS, 0)]),
        ([ind["00EE"]], [(spec.SP, sp_dec)]),
        ([ind["2NNN"]], [(stack_sp, pc2), (spec.SP, sp_inc)]),
        ([ind["6XNN"]], [(vxa, list(b2))]),
        ([ind["7XNN"]], [(vxa, add_xn)]),
        ([ind["8XY0"]], [(vxa, list(vyb))]),
        ([ind["ANNN"]], [(spec.I, nnn)]),
        ([ind["CXNN"]], [(vxa, rnd), (spec.RNG, rng_next)]),
        ([ind["FX07"]], [(vxa, list(dtb))]),
        ([ind["FX0A"]], [(vxa, fx0a_val)]),
        ([ind["FX15"]], [(spec.DT, list(vxb))]),
        ([ind["FX18"]], [(spec.ST, list(vxb))]),
        ([ind["FX1E"]], [(spec.I, i_vx)]),
        ([ind["FX29"]], [(spec.I, font_i)]),
        ([ind["FX33"]], [(i_a[0], bcd[0]), (i_a[1], bcd[1]), (i_a[2], bcd[2])]),
    ]
    for op, result in [("8XY1", or_xy), ("8XY2", and_xy), ("8XY3", xor_xy)]:
        fixed.append(([ind[op], one - q["logic"]], [(vxa, result)]))
        fixed.append(([ind[op], q["logic"]], [(vxa, result), (VF, 0)]))
    for op, result, flag in [("8XY4", add_xy, carry), ("8XY5", sub_xy, ge_xy),
                             ("8XY6", shr, src[0]), ("8XY7", sub_yx, ge_yx),
                             ("8XYE", shl, src[7])]:
        fixed.append(([ind[op], one - q["vforder"]], [(vxa, result), (VF, [flag])]))
        fixed.append(([ind[op], q["vforder"]], [(VF, [flag]), (vxa, result)]))
    fa = [c.select_bits([(lits, w[m][0]) for lits, w in fixed if len(w) > m], 13, f"A{m}")
          for m in range(3)]
    fv = [c.select_bits([(lits, w[m][1]) for lits, w in fixed if len(w) > m], 12, f"W{m}")
          for m in range(3)]
    has = [c.node(cat(c.t_and(lits) for lits, w in fixed if len(w) > m), f"has{m}")
           for m in range(3)]
    is_vec = ind["FX55"] + ind["FX65"] + ind["DXYN"]
    ls_extra = c.node(c.t_and([ind["FX55"], one - q["loadstore"]])
                      + c.t_and([ind["FX65"], one - q["loadstore"]]), "ls_extra")

    nw_oh = []
    NW = 2 * 15 + 2  # writes before PC: up to 2N + 1 for DXYN
    for v in range(1, NW + 1):
        terms = [t for lits, w in fixed if len(w) == v for t in c.t_and(lits)]
        for op in ("FX55", "FX65"):
            if 0 <= v - 1 < 16:
                terms += c.t_and([ind[op], q["loadstore"], *nib_lits(xb, v - 1)])
            if 0 <= v - 2 < 16:
                terms += c.t_and([ind[op], one - q["loadstore"], *nib_lits(xb, v - 2)])
        if v % 2 == 1 and (v - 1) // 2 < 16:
            terms += c.t_and([ind["DXYN"], *nib_lits(nb, (v - 1) // 2)])
        nw_oh.append(c.node(terms, f"nw=={v}"))
    nw_oh = [one - sum(nw_oh, Expr())] + nw_oh

    skip = c.node(c.t_and([ind["3XNN"], eq_x_nn]) + c.t_and([ind["4XNN"], one - eq_x_nn])
                  + c.t_and([ind["5XY0"], eq_x_y]) + c.t_and([ind["9XY0"], one - eq_x_y])
                  + c.t_and([ind["EX9E"], key_vx]) + c.t_and([ind["EXA1"], one - key_vx]),
                  "skip")
    wait = c.node(c.t_and([ind["FX0A"], one - any_key]), "wait")
    jump = ind["1NNN"] + ind["2NNN"]
    default = one - skip - jump - ind["00EE"] - ind["BNNN"] - wait
    next_pc = c.select_bits([([skip], pc4), ([jump], nnn), ([ind["00EE"]], stack_top),
                             ([ind["BNNN"]], bnnn), ([wait], list(pcb)), ([default], pc2)],
                            12, "next_pc")

    # ------------------------------------------------------------ context copy
    ctx_vals = {
        "x": x_oh, "n": n_oh, "I": list(ib), "Ihi": ihi, "nw": nw_oh, "next_pc": next_pc,
        "fixed": [one - is_vec], "has": has, "A0": fa[0], "A1": fa[1], "A2": fa[2],
        "W0": fv[0], "W1": fv[1], "W2": fv[2], "fx55": [ind["FX55"]], "fx65": [ind["FX65"]],
        "dxyn": [ind["DXYN"]], "dx_byte": vxb[3:6], "dx_off": vxb[0:3], "dy": vyb[0:5],
        "dt_dec": dt_dec, "st_dec": st_dec, "ls_extra": [ls_extra], "i_next": i_next,
        "clip": [q["clip"]],
    }
    flat = [(name, i, e) for name, vals in ctx_vals.items() for i, e in enumerate(vals)]
    k = {name: [None] * len(vals) for name, vals in ctx_vals.items()}
    per_head = 126
    for hd in range(0, len(flat), per_head):
        chunk = flat[hd:hd + per_head]
        data = {2 + i: e for i, (_, _, e) in enumerate(chunk)}
        outs = c.slot_head(f"ctx{hd // per_head}", {0: rs}, {0: one}, data, rs)
        for (name, i, _), o in zip(chunk, outs):
            k[name][i] = o

    # ------------------------------------------------------------ micro-step
    at_val, at_addr = c.is_addr, one - c.is_addr
    not_tick = one - in_tick
    is_pc = c.node(cat(c.t_and([k["nw"][v], jeq(v)]) for v in range(NW + 1)), "j==nw")
    is_end = c.node(cat(c.t_and([k["nw"][v], jeq(v + 1)]) for v in range(NW + 1)),
                    "j==nw+1")
    le_x = c.node(cat(c.t_and([k["x"][v], one - cnt[v]]) for v in range(16)), "j<=x")
    lt_2n = c.node(cat(c.t_and([k["n"][v], one - cnt[2 * v - 1]]) for v in range(1, 16)),
                   "j<2n")
    eq_2n = c.node(cat(c.t_and([k["n"][v], jeq(2 * v)]) for v in range(16)), "j==2n")
    is_i_adv = c.node(cat(c.t_and([k["x"][v], jeq(v + 1)]) for v in range(16)), "j==x+1")

    row_i = sum((cnt[2 * r - 1] for r in range(1, JMAX // 2 + 1)), Expr())  # j // 2
    side = sum((jeq(v) for v in range(1, JMAX + 1, 2)), Expr())  # j % 2
    kI_lo = from_bits(k["I"][:8])
    i_j = bits12(c.add12(kI_lo, k["Ihi"], j, JMAX + 1, "I+j"))
    # vread address: I + j (FX65) or I + j // 2 (DXYN)
    d = c.node([(1, k["fx65"][0], j), (1, k["dxyn"][0], row_i)], "vec_off")
    qa = c.add12(kI_lo, k["Ihi"], d, JMAX + 1, "vread_addr")
    vread = [Expr() for _ in range(8)]
    ram_read = k["fx65"][0] + k["dxyn"][0]
    for half in range(2):
        for h in range(16):
            cells = {(half << 11) | (s >> 4 << 8) | ((s & 15) << 4) | h: s for s in range(128)}
            query = {s: qa[0][h] + qa[1][s & 15] + qa[2][(s >> 4) + 8 * half] + ram_read - 3
                     for s in range(128)}
            bits = c.bank(f"ramB{half}.{h}", cells, query, 8)
            vread = [a_ + b_ for a_, b_ in zip(vread, bits)]
    vj = c.bank("V[j]", vcells, {r: jeq(r) + k["fx55"][0] - 1 for r in range(16)}, 8)
    vread = [a_ + b_ for a_, b_ in zip(vread, vj)]

    # DXYN: write j targets display byte (dy + j//2, dx_byte + j%2)
    dy = from_bits(k["dy"])
    dxb = from_bits(k["dx_byte"])
    row = dy + row_i
    col_ = dxb + side
    row_w = [c.node(c.t_eq(row, r, 0, 31 + 20) + c.t_eq(row, r + 32, 0, 51), f"row_w=={r}")
             for r in range(32)]
    col_w = [c.node(c.t_eq(col_, v, 0, 8) + c.t_eq(col_, v + 8, 0, 8), f"col_w=={v}")
             for v in range(8)]
    col8 = list(k["dx_byte"]) + [side]  # col == 8 exactly when dx_byte == 7 and side == 1
    on_screen = c.node(c.t_lt(row, 32, 0, 51) + c.t_lt(row, 32, 0, 51, -1, lits=col8),
                       "on_screen")
    drawn = c.node(c.t_const(1) + c.t_and([k["clip"][0]], -1)
                   + c.t_and([k["clip"][0], on_screen]), "drawn")
    idx_bits = onehot_bits(col_w, 3) + onehot_bits(row_w, 5)
    target = c.select_bits([([drawn], idx_bits + [1, 0, 0, 0, 1]),
                            ([one - drawn], spec.NULL)], 13, "target")
    off_bits = k["dx_off"]
    part = []
    for p in range(8):
        terms = []
        for kk in range(16):
            o, right = kk // 2, kk % 2
            s_ = p - 8 + o if right else p + o
            if 0 <= s_ < 8:
                lits = nib_lits(off_bits, o)[:3] + [side if right else one - side, vread[s_]]
                terms += c.t_and(lits)
        part.append(c.node(terms, f"part{p}"))
    overlap = [c.and_([old[p], part[p]], f"overlap{p}") for p in range(8)]
    new_byte = [old[p] + part[p] - 2 * overlap[p] for p in range(8)]
    draw_row = [c.is_addr, not_tick, k["dxyn"][0], lt_2n]
    collided = c.node(c.t_ge(sum(overlap, Expr()), 1, 0, 8, lits=draw_row), "collided")
    (any_collision,) = c.slot_head("collision", {0: rs + collided}, {0: one}, {2: collided},
                                   rs + collided)

    # ------------------------------------------------------------ output
    A = []  # (literals, kind, payload) with kind 0 = VAL, 1 = ADDR, 2 = STEP

    def case(lits, kind, payload):
        A.append((lits, kind, payload))

    for m in range(3):
        case([at_val, not_tick, k["fixed"][0], jeq(m), k["has"][m]], 0, k[f"W{m}"])
        case([at_addr, not_tick, k["fixed"][0], jeq(m), k["has"][m]], 1, k[f"A{m}"])
    jbits = onehot_bits([jeq(v) for v in range(16)], 4)
    case([at_val, not_tick, k["fx55"][0], le_x], 0, vread)
    case([at_addr, not_tick, k["fx55"][0], le_x], 1, i_j)
    case([at_val, not_tick, k["fx65"][0], le_x], 0, vread)
    case([at_addr, not_tick, k["fx65"][0], le_x], 1, jbits + [0] * 8 + [1])
    case([at_val, not_tick, k["ls_extra"][0], is_i_adv], 0, k["i_next"])
    case([at_addr, not_tick, k["ls_extra"][0], is_i_adv], 1, spec.I)
    case([at_val, not_tick, k["dxyn"][0], lt_2n, drawn], 0, new_byte)
    case([at_addr, not_tick, k["dxyn"][0], lt_2n], 1, target)
    case([at_val, not_tick, k["dxyn"][0], eq_2n], 0, [any_collision])
    case([at_addr, not_tick, k["dxyn"][0], eq_2n], 1, VF)
    case([at_val, not_tick, is_pc], 0, k["next_pc"])
    case([at_addr, not_tick, is_pc], 1, spec.PC)
    case([at_addr, not_tick, is_end], 2, 0)
    case([at_val, in_tick, jeq(0)], 0, k["dt_dec"])
    case([at_addr, in_tick, jeq(0)], 1, spec.DT)
    case([at_val, in_tick, jeq(1)], 0, k["st_dec"])
    case([at_addr, in_tick, jeq(1)], 1, spec.ST)
    case([at_addr, in_tick, jeq(2)], 2, 0)

    nxt = [c.node([t for lits, kind, _ in A if kind == 1 for t in c.t_and(lits)], "next.is_addr"),
           c.node([t for lits, kind, _ in A if kind == 2 for t in c.t_and(lits)], "next.is_ctrl")]
    nxt += c.select_bits([(lits, p) for lits, _, p in A], 13, "next")
    c.next_bits = nxt
    S = 2**12
    for i, e in enumerate(c.code):
        c.ffn([(2 * S, one, nxt[i]), (-S, one, one), (-1, one, e), (1, one, -1 * e)],
              f"logit{i}", into=e)
    c.ffn([(1, one, one)], "tau.cancel", into=c.tau)
    c.out = {"regs": R, "x": x, "cnt": cnt, "in_tick": in_tick, "old": old, "b1": b1, "b2": b2,
             "ind": ind, "vx": vxb, "vy": vyb, "ctx": k, "vread": vread, "next": nxt}
    return c
