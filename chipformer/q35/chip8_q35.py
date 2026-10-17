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

from fractions import Fraction

from chipformer import spec
from chipformer.dsl.graph import Expr
from chipformer.q35 import bridge as B
from chipformer.q35.ir import Chan, Target
from chipformer.q35.lib import bcd as bios
from chipformer.q35.prims import Circ, cat, from_bits, onehot_bits

JMAX = 64  # write indices above this all read as JMAX + 1
NCODE = 15  # output code: [is_addr, is_ctrl, payload bits 0..12]
CODE_AMP = 128
KAPPA = 2**-20  # output code after the exit: tiny, so it never disturbs Qwen's norms
TAU_TEXT = 2.0**-40
V, VF = spec.V, spec.V + 0xF
TEXT_CODE = 16  # text classes are ctrl codes with payload 16 + class (STEP/TICK/BOOT: 0..2)


def token_code(t: int) -> list[int]:
    """0/1 code of a token; the embedding stores CODE_AMP * (2 * bit - 1)."""
    if B.is_text(t):
        kind, payload = (0, 1), TEXT_CODE + t - B.TEXT_BASE
    elif spec.is_val(t):
        kind, payload = (0, 0), t - spec.VAL_BASE
    elif spec.is_addr(t):
        kind, payload = (1, 0), t - spec.ADDR_BASE
    else:
        kind, payload = (0, 1), t - spec.STEP
    return [*kind] + [payload >> i & 1 for i in range(13)]


def decode_code(bits: list[int]) -> int:
    payload = sum(b << i for i, b in enumerate(bits[2:]))
    if bits[1]:
        return B.TEXT_BASE + payload - TEXT_CODE if payload >= TEXT_CODE else spec.STEP + payload
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
        self.is_text = f("is_text", lambda t: int(B.is_text(t)))
        self.mu = f("mu", lambda t: int(not B.is_text(t)))  # chip token: Qwen's layers mute on it
        self.text = {cls: f(f"text{cls - B.TEXT_BASE}", lambda t, cls=cls: int(t == cls))
                     for cls in range(B.TEXT_BASE, B.TEXT_BASE + B.N_TEXT)}
        self.code = [f(f"code{i}", lambda t, i=i: 0 if t == B.OTHER else
                       CODE_AMP * (2 * token_code(t)[i] - 1)) for i in range(NCODE)]
        # tau: -1 on chip tokens (cancelled by the last layer), a tiny value on text tokens.
        # The final norm amplifies it, pushing the chip tokens' logits far down wherever Qwen
        # speaks.
        self.tau = f("tau", lambda t: TAU_TEXT if B.is_text(t) else -1)
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
    regs += [("LA", bios.LEN, 5), ("LB", bios.LEN + 1, 5)]  # bignum argument lengths
    regs += [("MODE", bios.MODE, 4)]  # inline call, inside "$", inside "$$", negative
    keys, data, col, blocks = {127: Chan(c.is_boot)}, {}, 2, {}
    for slot, (name, cell, nb) in enumerate(regs):
        keys[slot] = c.write_key(cell)
        lits = c.addr_lits(cell)
        blocks[name] = list(range(col, col + nb))
        for i in range(nb):
            # the VAL bit (this token) AND the address (previous token), over both conv taps
            data[col] = Chan(c.vb[i] + sum(lits, Expr()) - one * Fraction(len(lits), 2), (1, 1, 0, 0))
            col += 1
    out = c.slot_head("regs", keys, {s: one for s in keys}, data, Chan(c.is_boot))
    by_col = dict(zip(sorted(data), out))
    R = {name: [by_col[k] for k in cols] for name, cols in blocks.items()}
    pcb, ib, spb, dtb, stb, rngb, v0b = (R[k] for k in ("PC", "I", "SP", "DT", "ST", "RNG", "V0"))
    keyb = [R[f"key{k:X}"][0] for k in range(16)]
    lab, lbb = R["LA"], R["LB"]
    mode, mode_d, mode_dd, mode_neg = R["MODE"]
    q = {name: R[f"q.{name}"][0] for name in spec.QUIRK_NAMES}

    # ------------------------------------------------------------ micro-step counter
    # j = VAL tokens (inside a record) or text tokens (inside a print) since the record start;
    # a CALL record starts at the </tool_call> token
    tco, tcc = c.text[B.TC_OPEN], c.text[B.TC_CLOSE]
    rsc = rs + tcc
    (x,) = c.slot_head("counter", {0: rsc, 1: one - rsc}, {0: one - rsc, 1: one - rsc},
                       {2: rsc}, one - rsc, 110 * (c.is_val + c.is_text - tcc) + 238 * rsc,
                       tiny=2.0**-90)
    cnt = [c.node([(1, one - rsc, 7 * one - 8 * 2**m * x),
                   (-1, one - rsc, 6 * one - 8 * 2**m * x)], f"j>{m}")
           for m in range(JMAX + 1)]  # cnt[m] = [j > m]
    j = sum(cnt, Expr())

    def jeq(v):  # [j == v], a 0/1 linear form
        return (one if v == 0 else cnt[v - 1]) - (cnt[v] if v <= JMAX else 0)

    (in_tick,) = c.slot_head("in_tick", {0: rs + c.is_tick}, {0: one}, {2: c.is_tick},
                             rs + c.is_tick)

    # ------------------------------------------------------------ argument capture
    # Tool calls: inside <tool_call> ... </tool_call> the circuit keeps the routine's name and
    # the digits of each argument (runs of digit tokens, in order) for the CALL record.
    # Inline: outside tool calls it keeps the same for "A op B" since the last breaker token, and
    # answers at "=" (bridge.InlineScan is the reference).
    T = c.text
    is_dig = sum((T[B.HEX[v]] for v in range(10)), Expr())
    dvb = [sum((T[B.HEX[v]] for v in range(10) if v >> i & 1), Expr()) for i in range(4)]
    fns = {op: T[cls] for op, cls in B.FUNCS.items()}
    fn_any = sum(fns.values(), Expr())
    (in_call,) = c.slot_head("in_call", {0: tco + tcc + rs}, {0: one}, {2: tco}, tco + tcc + rs)
    dollars = sum((T[d_] for d_ in B.DOLLARS), Expr())
    safe = (T[B.SPACE] + T[B.NL] + T[B.WS] + dollars + T[B.LPAREN]
            + c.mu)  # may precede a first number
    open1, open2 = T[B.DOLLAR] + T[B.SDOLLAR], T[B.DDOLLAR] + T[B.SDDOLLAR]  # "$", "$$"
    prev_cols = {2: Chan(is_dig, (0, 1, 0, 0)), 3: Chan(is_dig, (0, 0, 1, 0)),
                 4: Chan(T[B.COMMA], (0, 1, 0, 0)), 5: Chan(T[B.BSL], (0, 1, 0, 0)),
                 6: Chan(safe, (0, 1, 0, 0)), 7: Chan(c.is_text, (0, 1, 0, 0)),
                 8: Chan(T[B.EQ], (0, 1, 0, 0)), 9: Chan(open1, (0, 1, 0, 0)),
                 10: Chan(open2, (0, 1, 0, 0)), 11: Chan(open1, (0, 0, 1, 0)),
                 12: Chan(open2, (0, 0, 1, 0)), 13: Chan(T[B.SPACE], (0, 1, 0, 0))}
    (prev_dig, prev2_dig, prev_cm, prev_bs, prev_safe, prev_text, prev_eq, prev_o1, prev_o2,
     prev2_o1, prev2_o2, prev_sp) = c.slot_head("prev", {0: one}, {0: one}, prev_cols, one)
    # the math a first number sits in: right after "$" / "$$", or after it and a space
    in_o1 = c.node(c.t_and([prev_o1]) + c.t_and([prev_sp, prev2_o1]), "open.$")
    in_o2 = c.node(c.t_and([prev_o2]) + c.t_and([prev_sp, prev2_o2]), "open.$$")
    cont = c.node(c.t_and([is_dig, prev_dig]) + c.t_and([is_dig, prev_cm, prev2_dig]), "arg.cont")
    run_start = c.node(c.t_and([is_dig, one - prev_dig, one - prev_cm])
                       + c.t_and([is_dig, prev_cm, one - prev2_dig]), "arg.start")
    tw_op = c.and_([T[B.TIMESW], prev_bs], "times.op")  # \\times, \\cdot
    dv_op = c.and_([T[B.FN_DIV], prev_bs], "div.op")  # \\div
    inl = {"add": T[B.PLUS], "sub": T[B.MINUS] + T[B.MINUS2], "mul": T[B.TIMES] + tw_op,
           "div": T[B.DIV] + dv_op}
    op_tok = sum(inl.values(), Expr())
    comma_join = c.and_([T[B.COMMA], prev_dig], "comma.join")
    neutral = (is_dig + T[B.SPACE] + dollars + T[B.LPAREN] + T[B.BSL] + T[B.EQ] + op_tok
               + comma_join)
    brk = one - neutral  # chip tokens included
    # the window also starts afresh right after "=" (so "x^2 = A op B =" still counts)
    after_eq = [prev_eq, one - brk, one - in_call]
    reset = c.node(c.t_and([tco]) + c.t_and([brk - tco, one - in_call]) + c.t_and(after_eq),
                   "arg.reset")
    op_reset = c.node(c.t_and([brk]) + c.t_and(after_eq), "op.reset")
    # a digit or operator right after "=" is reset, not counted (the decay must be exactly 238)
    both_d = c.and_([is_dig, prev_eq, one - in_call], "after_eq.digit")
    both_o = c.and_([op_tok, prev_eq, one - in_call], "after_eq.op")
    (xp,) = c.slot_head("arg.count", {0: reset, 1: one - reset}, {0: one - reset, 1: one - reset},
                        {2: reset}, one - reset, 110 * (run_start - both_d) + 238 * reset,
                        tiny=2.0**-90)
    (xd,) = c.slot_head("digit.count", {0: run_start, 1: one - run_start},
                        {0: one - run_start, 1: one - run_start}, {2: run_start},
                        one - run_start, 110 * cont + 238 * run_start, tiny=2.0**-90)
    (xo,) = c.slot_head("op.count", {0: op_reset, 1: one - op_reset},
                        {0: one - op_reset, 1: one - op_reset}, {2: op_reset}, one - op_reset,
                        110 * (op_tok - both_o) + 238 * op_reset, tiny=2.0**-90)

    def thermo(xv, gate, n, name):  # [count > m] for m < n, from x = 2^-count
        return [c.node([(1, one - gate, 7 * one - 8 * 2**m * xv),
                        (-1, one - gate, 6 * one - 8 * 2**m * xv)], f"{name}>{m}")
                for m in range(n)]

    cp = thermo(xp, reset, 3, "p")  # p = argument number (1, 2)
    cd = thermo(xd, run_start, B.MAX_ARG + 1, "d")  # d = digit index within the argument
    co = thermo(xo, op_reset, 2, "ops")  # operators since the last breaker

    def peq(v):
        return cp[v - 1] - cp[v]

    def deq(w):
        return (one if w == 0 else cd[w - 1]) - cd[w]

    # lengths and the routine: slots 0/1 = length of argument 1/2 (rewritten at every digit),
    # slot 2 = the routine (at its name token, or the inline operator) and whether it came right
    # after the first argument, slot 127 = ballast
    len_bits = [sum((deq(w) for w in range(B.MAX_ARG) if (w + 1) >> i & 1), Expr())
                for i in range(5)]
    data = {}
    for a_ in (1, 2):
        for i in range(5):
            data[2 + 5 * (a_ - 1) + i] = is_dig + peq(a_) + len_bits[i] - 2
    ops = list(B.FUNCS)
    for i, op in enumerate(ops):
        data[12 + i] = fns[op] + inl[op] - (dv_op if op == "div" else 0)  # "\div" counts once
    at_fn = c.node(c.t_and([fn_any, in_call]) + c.t_and([op_tok, one - in_call]), "op.at")
    data[12 + len(ops)] = at_fn + peq(1) - 1
    # <tool_call> clears the routine (a row of zeros), so a tool call needs a name inside it: an
    # inline operator in earlier text must not make the template's "<tool_call></tool_call>" fire
    lk = {0: is_dig + peq(1) - 1, 1: is_dig + peq(2) - 1, 2: at_fn + tco,
          127: brk - fn_any - tco}
    out = c.slot_head("call.args", lk, {0: one, 1: one, 2: one, 127: one}, data,
                      brk - fn_any - tco)
    la_bits, lb_bits, op_oh = out[0:5], out[5:10], out[10:10 + len(ops)]
    op_ok = out[10 + len(ops)]
    a_safe, a_o1, a_o2 = c.slot_head("a.safe", {0: run_start + peq(1) - 1}, {0: one},
                                     {2: prev_safe, 3: in_o1, 4: in_o2}, run_start + peq(1) - 1)
    la_, lb_ = from_bits(la_bits), from_bits(lb_bits)
    trig = c.and_([T[B.EQ], one - in_call, peq(2), co[0] - co[1], op_ok, a_safe,
                   c.node(c.t_ge(la_, 1, 0, 31), "la>0"), c.node(c.t_ge(lb_, 1, 0, 31), "lb>0"),
                   c.node(cat(c.t_ge(la_ + lb_, B.INLINE_MIN_DIGITS[op], 0, 62,
                                     lits=[op_oh[ops.index(op)]]) for op in ops), "inline.big"),
                   sum(op_oh, Expr())], "inline.trig")
    # 1 from an inline trigger until the next text token: the mode of the CALL record
    (inline_mode,) = c.slot_head("inline.mode", {0: c.is_text}, {0: one}, {2: trig}, c.is_text)

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
    ind["00FD"] = c.and_(nib_lits(b1[:4], 0) + nib_lits(b1[4:], 0) + nib_lits(b2[:4], 0xD)
                         + nib_lits(b2[4:], 0xF), "00FD")  # print and halt (bridge.py)
    ind["00EE"] = c.and_(nib_lits(b1[:4], 0) + nib_lits(b1[4:], 0) + nib_lits(b2[:4], 0xE)
                         + nib_lits(b2[4:], 0xE), "00EE")
    for h, name in [(5, "5XY0"), (9, "9XY0")]:
        ind[name] = c.and_(nib_lits(hib, h) + nib_lits(nb, 0), name)
    for m, name in [(1, "BADD"), (2, "BZERO"), (3, "BMUL"), (4, "BSUB"), (5, "BDIV"),
                    (6, "BADB"), (7, "BRSUB"), (8, "BTRIM")]:  # lib/macros.py
        ind[name] = c.and_(nib_lits(hib, 5) + nib_lits(nb, m), name)
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

    # (before the BTRIM nodes: the first FFN sublayer is nearly full and placement is greedy)
    la_i, lb_i = from_bits(lab), from_bits(lbb)
    mx = c.node([(1, one, la_i), (1, one, lb_i - la_i)], "max(LA,LB)")  # la + relu(lb - la)
    # BTRIM: is mem[I] a leading 0 of the result (I inside it, V0 >= 2)? Read at decode from a
    # bank of the result cells queried by I, so it is a plain conditional instruction.
    # (the queries are AND-forms of I's bits, no FFN units)
    r_cells = 2 * bios.MAX_DIGITS
    r_end = bios.R_ADDR + r_cells
    assert bios.R_ADDR >> 8 == (r_end - 1) >> 8
    trim_q = {v: sum(nib_lits(ib[0:4], (bios.R_ADDR + v) & 15)
                     + nib_lits(ib[4:8], (bios.R_ADDR + v) >> 4 & 15)
                     + nib_lits(ib[8:12], (bios.R_ADDR + v) >> 8), Expr()) - 11
              for v in range(r_cells)}
    trim_d = c.bank("trim.R", {bios.R_ADDR + v: v for v in range(r_cells)}, trim_q, 4)
    trim_in = c.node(c.t_in(from_bits(ib[:8]), bios.R_ADDR & 255, (r_end - 1 & 255) + 1, 0, 255,
                            lits=nib_lits(ib[8:12], bios.R_ADDR >> 8)), "trim.in")
    v0_ge2 = c.node(c.t_ge(from_bits(v0b), 2, 0, 255), "v0>=2")
    trim_z = c.and_([one - b_ for b_ in trim_d] + [v0_ge2, trim_in], "trim.z")

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
        ([ind["BTRIM"], trim_z], [(V, dec8(v0b, "v0-1")), (spec.I, i_p[1])]),
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

    # BMUL: VX = rows done (row LB - 1 - VX); 5X13 is the first row (R read as 0: no BZERO),
    # the last row ends the loop
    vx5 = from_bits(vxb[:5])
    mul_first = c.and_(nib_lits(yb, 1), "mul.first")
    mul_last = c.node(c.t_eq(vx5 + 1 - lb_i, 0, -31, 32), "mul.last")
    vx_inc = [one - vxb[0]] + [c.node(c.t_and([vxb[i]]) + c.t_and(vxb[:i])
                                      + c.t_and([vxb[i], *vxb[:i]], -2), f"vx+1.b{i}")
                               for i in range(1, 5)]
    nw_oh = []
    NW = 2 * bios.MAX_DIGITS + 2  # writes before PC: up to LA + LB + 1 for BZERO
    for v in range(1, NW + 1):
        terms = [t for lits, w in fixed if len(w) == v for t in c.t_and(lits)]
        for op in ("FX55", "FX65"):
            if 0 <= v - 1 < 16:
                terms += c.t_and([ind[op], q["loadstore"], *nib_lits(xb, v - 1)])
            if 0 <= v - 2 < 16:
                terms += c.t_and([ind[op], one - q["loadstore"], *nib_lits(xb, v - 2)])
        if v % 2 == 1 and (v - 1) // 2 < 16:
            terms += c.t_and([ind["DXYN"], *nib_lits(nb, (v - 1) // 2)])
        # n = max + 1 digits, then V0 and I (BRSUB: and MODE)
        terms += c.t_eq(mx, v - 3, 0, 31, lits=[ind["BADD"] + ind["BSUB"]])
        terms += c.t_eq(mx, v - 4, 0, 31, lits=[ind["BRSUB"]])
        terms += c.t_eq(la_i + lb_i, v - 1, 0, 62, lits=[ind["BZERO"]])
        # LA digits, the carry, VX (the last row: and V0, I)
        terms += c.t_eq(la_i, v - 2, 0, 31, lits=[ind["BMUL"], one - mul_last])
        terms += c.t_eq(la_i, v - 4, 0, 31, lits=[ind["BMUL"], mul_last])
        # the quotient digit, LB digits and the window's top digit, VX, VF
        terms += c.t_eq(lb_i, v - 4, 0, 31, lits=[ind["BDIV"] + ind["BADB"]])
        nw_oh.append(c.node(terms, f"nw=={v}"))
    nw_oh = [one - sum(nw_oh, Expr())] + nw_oh

    skip = c.node(c.t_and([ind["3XNN"], eq_x_nn]) + c.t_and([ind["4XNN"], one - eq_x_nn])
                  + c.t_and([ind["5XY0"], eq_x_y]) + c.t_and([ind["9XY0"], one - eq_x_y])
                  + c.t_and([ind["EX9E"], key_vx]) + c.t_and([ind["EXA1"], one - key_vx]),
                  "skip")
    wait = c.node(c.t_and([ind["FX0A"], one - any_key]), "wait")
    jump = ind["1NNN"] + ind["2NNN"]
    # macros that repeat (PC stays) and macros that jump to T = 0x200 + 2 * XY (BSUB: unless
    # it borrows, decided at its PC write)
    stay = c.node(c.t_and([ind["BTRIM"], trim_z])
                  + c.t_and([ind["BMUL"], one - mul_first, one - mul_last]), "macro.stay")
    mul_skip = c.and_([ind["BMUL"], mul_first, mul_last], "mul.skip")  # one row: past the loop
    to_t = ind["BADD"] + ind["BSUB"] + ind["BRSUB"]
    tgt = [0] + list(yb) + list(xb) + [1, 0, 0]
    default = one - skip - jump - ind["00EE"] - ind["BNNN"] - wait - stay - to_t - mul_skip
    next_pc = c.select_bits([([skip + mul_skip], pc4), ([jump], nnn), ([ind["00EE"]], stack_top),
                             ([ind["BNNN"]], bnnn), ([wait + stay], list(pcb)), ([to_t], tgt),
                             ([default], pc2)],
                            12, "next_pc")

    # ------------------------------------------------------------ context copy
    ctx_vals = {
        "x": x_oh, "n": n_oh, "I": list(ib), "Ihi": ihi, "nw": nw_oh, "next_pc": next_pc,
        "fixed": [one - is_vec], "has": has, "A0": fa[0], "A1": fa[1], "A2": fa[2],
        "W0": fv[0], "W1": fv[1], "W2": fv[2], "fx55": [ind["FX55"]], "fx65": [ind["FX65"]],
        "dxyn": [ind["DXYN"]], "dx_byte": vxb[3:6], "dx_off": vxb[0:3], "dy": vyb[0:5],
        "dt_dec": dt_dec, "st_dec": st_dec, "ls_extra": [ls_extra], "i_next": i_next,
        "clip": [q["clip"]], "fd": [ind["00FD"]], "v0": v0b,
        "badd": [ind["BADD"]], "bzero": [ind["BZERO"]], "bmul": [ind["BMUL"]],
        "bsub": [ind["BSUB"]], "brsub": [ind["BRSUB"]], "pc2": pc2,
        "la": lab, "lb": lbb, "vx": vxb[:5], "first": [mul_first], "last": [mul_last],
        "vx1": vx_inc, "stay": [stay], "bdiv": [ind["BDIV"]], "badb": [ind["BADB"]],
        "t": vxb[:6], "pc": list(pcb),
    }
    # the macros' digit chain is the deepest path: its entries get a head of their own, so no
    # later entry holds them up
    fast = ("la", "lb", "vx", "badd", "bsub", "brsub", "bmul", "bzero", "first", "bdiv", "badb",
            "t")
    groups = [[(name, i, e) for name in fast for i, e in enumerate(ctx_vals[name])],
              [(name, i, e) for name, vals in ctx_vals.items() if name not in fast
               for i, e in enumerate(vals)]]
    k = {name: [None] * len(vals) for name, vals in ctx_vals.items()}
    per_head = 126
    chunks = [g[hd:hd + per_head] for g in groups for hd in range(0, len(g), per_head)]
    for h, chunk in enumerate(chunks):
        data = {2 + i: e for i, (_, _, e) in enumerate(chunk)}
        outs = c.slot_head(f"ctx{h}", {0: rs}, {0: one}, data, rs)
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
    # vread address: I + j (FX65), I + j // 2 (DXYN) or I + j - 3 (the digits of a print)
    d = c.node([(1, k["fx65"][0], j), (1, k["dxyn"][0], row_i),
                (1, k["fd"][0], j - 3 * one + 2 * mode)], "vec_off")
    qa = c.add12(kI_lo, k["Ihi"], d, JMAX + 1, "vread_addr")
    vread = [Expr() for _ in range(8)]
    ram_read = k["fx65"][0] + k["dxyn"][0] + k["fd"][0]
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

    # ---------------------------------------------------------------- CALL record
    # At </tool_call> (with a known routine) the circuit writes the arguments' digits to RAM,
    # their lengths, then PC = the routine, then STEP; the machine then runs the routine.
    op_valid = sum(op_oh, Expr())
    # a tool call starts its record at </tool_call>; an inline one predicts STEP at "=" and starts
    # at that STEP (so j can stay a plain token-feature counter)
    call_val = c.node(c.t_and([tcc, op_valid]) + c.t_and([rs, inline_mode, prev_text]), "call.v")
    (calling,) = c.slot_head("calling", {0: tcc + rs + trig}, {0: one}, {2: call_val + trig},
                             tcc + rs + trig)
    neq = one - T[B.EQ]
    la, lb = from_bits(la_bits), from_bits(lb_bits)
    za, zz = j - la, j - la - lb
    in_a = c.node(c.t_lt(za, 0, -B.MAX_ARG, JMAX + 1), "call.in_a")
    in_b = c.node(c.t_ge(za, 0, -B.MAX_ARG, JMAX + 1)
                  + c.t_ge(zz, 0, -2 * B.MAX_ARG, JMAX + 1, -1), "call.in_b")
    ez = [c.node(c.t_eq(zz, v, -2 * B.MAX_ARG, JMAX + 1), f"call.e{v}") for v in range(5)]
    q_a = [c.and_([jeq(w), in_a], f"call.qa{w}") for w in range(B.MAX_ARG)]
    q_b = [c.node(c.t_eq(za, w, -B.MAX_ARG, JMAX + 1, lits=[in_b]), f"call.qb{w}")
           for w in range(B.MAX_ARG)]
    keys = {32 * (a_ - 1) + w: is_dig + peq(a_) + deq(w) - 2
            for a_ in (1, 2) for w in range(B.MAX_ARG)}
    query = {**{w: q_a[w] for w in range(B.MAX_ARG)},
             **{32 + w: q_b[w] for w in range(B.MAX_ARG)}}
    digit_bits = c.slot_head("call.digits", keys, query, {2 + i: dvb[i] for i in range(4)}, one)
    pad = [0] * (32 - B.MAX_ARG)
    a_addr = onehot_bits(q_a + pad, 5)
    b_addr = onehot_bits(q_b + pad, 5)
    op_div = op_oh[ops.index("div")]  # div's A goes to W + 1 (bcd.call_writes)
    a_addr = [(bios.A_ADDR >> i & 1) * (one - op_div) + ((bios.W_ADDR + 1) >> i & 1) * op_div
              if i >= 5 else a_addr[i] for i in range(13)]
    b_addr = [bios.B_ADDR >> i & 1 if i >= 5 else b_addr[i] for i in range(13)]
    entry = [sum((op_oh[k_] for k_, op in enumerate(ops) if bios.OPS[op] >> i & 1), Expr())
             for i in range(12)]
    case([trig], 2, 0)
    case([at_addr, calling, neq, in_a], 1, a_addr)
    case([at_val, calling, in_a], 0, digit_bits)
    case([at_addr, calling, neq, in_b], 1, b_addr)
    case([at_val, calling, in_b], 0, digit_bits)
    case([at_addr, calling, neq, ez[0]], 1, bios.LEN)
    case([at_val, calling, ez[0]], 0, la_bits)
    case([at_addr, calling, neq, ez[1]], 1, bios.LEN + 1)
    case([at_val, calling, ez[1]], 0, lb_bits)
    case([at_addr, calling, neq, ez[2]], 1, bios.MODE)
    mode_bits = [inline_mode, c.and_([inline_mode, a_o1 + a_o2], "mode.$"),
                 c.and_([inline_mode, a_o2], "mode.$$")]
    case([at_val, calling, ez[2]], 0, mode_bits)
    case([at_addr, calling, neq, ez[3]], 1, spec.PC)
    case([at_val, calling, ez[3]], 0, entry)
    case([at_addr, calling, neq, ez[4]], 2, 0)

    # ---------------------------------------------------------------- bignum macros
    MD = bios.MAX_DIGITS
    kb_add, kb_zero, kb_mac, kb_sub = k["badd"][0], k["bzero"][0], k["bmul"][0], k["bsub"][0]
    kb_rsub = k["brsub"][0]
    kb_as = kb_add + kb_sub + kb_rsub  # BADD, BSUB and BRSUB share their layout
    kla, klb = from_bits(k["la"]), from_bits(k["lb"])
    krow = klb - 1 - from_bits(k["vx"])  # BMUL's row
    wi = j - c.is_val  # index of the write this token belongs to
    kmx = c.node([(1, one, kla), (1, one, klb - kla)], "k.max")
    ia, ib = kla - 1 - wi, klb - 1 - wi  # digit indices, counted from the right end

    def eqs(z, lo, hi, n, name, lits=()):  # one-hot of an integer z over 0..n-1
        return [c.node(c.t_eq(z, v, lo, hi, lits=list(lits)), f"{name}=={v}") for v in range(n)]

    lo_, hi_ = -72, 72  # bounds of index arithmetic, garbage states included (5-bit fields)
    qa = eqs(ia, lo_, hi_, MD, "mac.ia", [kb_as + kb_mac])
    qb = [c.node(c.t_eq(ib, v, lo_, hi_, lits=[kb_as]) + c.t_eq(krow, v, -32, 31, lits=[kb_mac])
                 + c.t_eq(klb - wi, v, lo_, hi_, lits=[k["bdiv"][0] + k["badb"][0]]),
                 f"mac.ib=={v}") for v in range(MD)]
    kk2 = kla + krow - wi  # BMUL target digit i + row + 1
    qr = eqs(kk2, lo_, hi_, 2 * MD, "mac.k", [kb_mac, one - k["first"][0]])  # first row: 0
    # long division (BDIV / BADB): t = VX, the window W[t-LB .. t], quotient digit at P[pq]
    kb_div, kb_adb = k["bdiv"][0], k["badb"][0]
    kb_dv = kb_div + kb_adb
    kt = from_bits(k["t"])
    NWC = 2 + MD + bios.DEC  # W: a 0, A, the decimals' zeros
    gt_la = c.node(c.t_ge(kt - kla, 1, -31, 63), "div.t>la")
    pq = kt + gt_la  # P index of the quotient digit
    # writes: 0 = the quotient digit P[pq], 1 .. LB+1 = the window (from the right; LB+1 is its
    # top digit), LB+2 = VX, LB+3 = VF
    dv_p = c.and_([kb_dv, jeq(0)], "div.w0")
    in_dv = c.node(c.t_lt(j - klb, 2, lo_, hi_, lits=[kb_dv, cnt[0]]), "div.digit")
    rel = j - klb
    dv_at = [c.node(c.t_eq(rel, r_, lo_, hi_, lits=[kb_dv]), f"div.w{r_}") for r_ in (2, 3)]
    qw = eqs(kt - wi + 1, -64, 65, NWC, "div.kw", [kb_dv])

    def digit_bank(name, base, n, query):
        cells = {base + i: i for i in range(n)}
        return c.bank(name, cells, {i: query[i] for i in range(n)}, 4)

    a_bits = digit_bank("bank.A", bios.A_ADDR, MD, qa)
    b_bits = digit_bank("bank.B", bios.B_ADDR, MD, qb)
    r_bits = digit_bank("bank.R", bios.R_ADDR, 2 * MD, qr)
    w_bits = digit_bank("bank.W", bios.W_ADDR, NWC, qw)
    a_i, b_i, r_i, w_i = from_bits(a_bits), from_bits(b_bits), from_bits(r_bits), from_bits(w_bits)
    p_i = from_bits(digit_bank("bank.P", bios.R_ADDR, 2 * MD,
                               eqs(pq, 0, 64, 2 * MD, "div.pqr", [kb_adb])))  # BADB: P - 1

    # BDIV's quotient digit, at its first ADDR token (nothing is written yet): q = #{k in 1..9 :
    # R3 + delta >= k * B2} (lib/macros.div_estimate). R3 + delta - k*B2 reaches 999, beyond
    # bf16's exact integers, so it is split as 10 * H + L (H = 10 W0 + W1 - k b0,
    # L = W2 + delta - k b1, both small): [10 H + L >= 0] = sum over c of [ceil(-L/10) == c] [H >= c]
    wq = [from_bits(digit_bank(f"bank.Wq{c_}", bios.W_ADDR, NWC,
                               eqs(kt - klb + c_, -64, 64, NWC, f"div.q{c_}", [kb_div])))
          for c_ in range(3)]
    b0 = from_bits(digit_bank("bank.B0", bios.B_ADDR, 1, {0: kb_div}))
    b1 = from_bits(digit_bank("bank.B1", bios.B_ADDR + 1, 1, {0: kb_div}))
    lb1 = c.node(c.t_eq(klb, 1, 0, 31), "div.lb1")
    lb3 = c.node(c.t_ge(klb, 3, 0, 31), "div.lb3")
    th = [one]
    for k_ in range(1, 10):
        h_ = 10 * wq[0] + wq[1] - k_ * b0  # -81 .. 99
        l_ = wq[2] + lb3 - k_ * b1  # -81 .. 10
        ge_h = {cc: c.node(c.t_ge(h_, cc, -81, 99), f"div.H{k_}>={cc}") for cc in range(-1, 10)}
        l_is = {cc: c.node(c.t_in(l_, -10 * cc, -10 * cc + 10, -81, 10), f"div.L{k_}~{cc}")
                for cc in range(-1, 10)}
        th.append(c.node(cat(c.t_and([l_is[cc], ge_h[cc], one - lb1]) for cc in range(-1, 10))
                         + c.t_and([ge_h[0], lb1]), f"div.q>={k_}"))
    th.append(Expr())
    q_new = onehot_bits([th[v] - th[v + 1] for v in range(10)], 4)
    # the record's first write carries q in its VAL token, so the window's writes read it from
    # the token stream (the estimate itself is too deep to feed their carry chain)
    at_q = c.and_([c.is_val, kb_div, jeq(1)], "div.at_q")
    q_bits = c.slot_head("div.q", {0: at_q}, {0: one}, {2 + i: c.vb[i] for i in range(4)}, at_q)

    # Carries. A carry can only reach the next write through the token stream, so each VAL
    # token reconstructs the carry its write produced from the digit it wrote, d:
    #   BADD: carry = [a + b >= d + 9];  BMUL: carry = s // 10 + [d < s % 10], s = R[k] + a*b.
    # R[k] is overwritten at that VAL token, so its ADDR token keeps s // 10 and s % 10 in a
    # register (as one-hots) for the VAL token to read.
    prod = c.node(cat(c.t_and([a_bits[p_], b_bits[q_]], 1 << (p_ + q_))
                      for p_ in range(4) for q_ in range(4)), "a*b")
    s_mac = r_i + prod  # <= 90 (<= 240 in garbage states)
    # BADB is BMAC's carry chain with s = W[k] + b. BDIV subtracts: W[k] - q*b - borrow, kept as
    # g = q*b - W[k] + 9 (0 .. 90) so that borrow' = g // 10 + [d >= 10 - g % 10].
    qprod = c.node(cat(c.t_and([q_bits[p_], b_bits[q_]], 1 << (p_ + q_))
                       for p_ in range(4) for q_ in range(4)), "q*b")
    s_adb = w_i + b_i  # <= 18 (<= 30)
    g_div = qprod - w_i + 9  # 0 .. 90 (-6 .. 234)
    fl_oh = [c.node(c.t_in(s_mac, 10 * v, 10 * v + 10, 0, 240, lits=[kb_mac])
                    + c.t_in(s_adb, 10 * v, 10 * v + 10, 0, 30, lits=[kb_adb])
                    + c.t_in(g_div, 10 * v, 10 * v + 10, -15, 240, lits=[kb_div]),
                    f"s//10=={v}") for v in range(10)]
    sm_oh = [c.node(cat(c.t_eq(s_mac, v + 10 * q_, 0, 240, lits=[kb_mac]) for q_ in range(10))
                    + cat(c.t_eq(s_adb, v + 10 * q_, 0, 30, lits=[kb_adb]) for q_ in range(4))
                    + cat(c.t_eq(g_div, v + 10 * q_, -15, 240, lits=[kb_div])
                          for q_ in range(10)), f"s%10=={v}")
             for v in range(10)]
    at_mac = c.and_([c.is_addr, kb_mac + kb_dv], "mac.at")
    keep = c.slot_head("mac.s", {0: at_mac}, {0: one},
                       {2 + i: e for i, e in enumerate(fl_oh + sm_oh)}, at_mac)
    fl_r, sm_r = keep[:10], keep[10:]
    tok_d = from_bits(c.vb[:4])  # the digit this VAL token wrote
    d_lt = [c.node(c.t_lt(tok_d, v, 0, 15), f"d<{v}") for v in range(10)]
    wrapped = c.node(cat(c.t_and([sm_r[v], d_lt[v]]) for v in range(1, 10)), "mac.wrap")
    cout_mac = sum((v * fl_r[v] for v in range(10)), Expr()) + wrapped  # <= 9
    cout_add = c.node(c.t_ge(a_i + b_i - tok_d, 9, -15, 30), "add.cout")
    cout_sub = c.node(c.t_ge(tok_d - a_i + b_i, 1, -15, 30), "sub.borrow")  # [d > a - b]
    cout_rsub = c.node(c.t_ge(tok_d - b_i + a_i, 1, -15, 30), "rsub.borrow")  # [d > b - a]
    wrap_dv = c.node(cat(c.t_and([sm_r[v], one - d_lt[10 - v]]) for v in range(1, 10)),
                     "div.wrap")  # [d >= 10 - g % 10]
    cout_div = sum((v * fl_r[v] for v in range(10)), Expr()) + wrap_dv  # <= 9
    cm_oh = eqs(cout_mac, 0, 10, 10, "mac.cout")
    cd_oh = eqs(cout_div, 0, 10, 10, "div.cout")
    cout_bits = [c.node(c.t_and([c.is_val, kb_mac + kb_adb, b_])
                        + c.t_and([c.is_val, kb_div, bd_])
                        + (c.t_and([c.is_val, kb_add, cout_add])
                           + c.t_and([c.is_val, kb_sub, cout_sub])
                           + c.t_and([c.is_val, kb_rsub, cout_rsub]) if i == 0 else []),
                        f"cout.b{i}")
                 for i, (b_, bd_) in enumerate(zip(onehot_bits(cm_oh, 4), onehot_bits(cd_oh, 4)))]
    macro = kb_add + kb_mac + kb_sub + kb_rsub + kb_dv
    # only digit writes pass the carry on, so it survives the V0 / I / VX writes after them
    # (BSUB's PC write reads the final borrow)
    # (at a VAL token wi = j - 1; written so, is_val is not in both the value and the literals,
    # whose weights would add up to a non-bf16 257)
    # (BADD / BSUB / BRSUB: not their top digit either, so the carry into it stays: it is R[0])
    dig_w = c.node(c.t_lt(j - 1 - kmx, 0, lo_, hi_, lits=[c.is_val, kb_as])
                   + c.t_lt(j - 1 - kla, 0, lo_, hi_, lits=[c.is_val, kb_mac])
                   # division: the window's writes (wi = 1 .. LB+1), the last one carrying BDIV's
                   # "negative" / BADB's "carried out"
                   + c.t_lt(j - klb, 3, lo_, hi_, lits=[c.is_val, kb_dv, cnt[1]]),
                   "macro.digit_w")
    dv_win = dig_w  # (only the division's terms are live in its records)
    carry = c.slot_head("carry", {0: rs + dig_w + macro - 1}, {0: one},
                        {2 + i: cout_bits[i] for i in range(4)}, rs + dig_w + macro - 1)
    c_in = from_bits(carry)

    # BADD digit: (a + b + carry) mod 10; BMAC digit: (R + a*b + carry) mod 10
    t_add = a_i + b_i + c_in  # <= 19 (<= 45)
    t_mac = s_mac + c_in  # <= 99 (<= 255)
    t_sub = a_i - b_i - c_in  # -10 .. 9 (-30 .. 15)
    t_rsub = b_i - a_i - c_in
    t_adb = s_adb + c_in  # <= 27 (<= 45)
    t_div = 9 - g_div - c_in  # W[k] - q*b - borrow: -99 .. 9 (-246 .. 24)
    dsum = [c.node(c.t_eq(t_add, v, 0, 45, lits=[kb_add]) + c.t_eq(t_add, v + 10, 0, 45,
                                                                   lits=[kb_add])
                   + cat(c.t_eq(t_mac, v + 10 * q_, 0, 255, lits=[kb_mac]) for q_ in range(10))
                   + c.t_eq(t_sub, v, -30, 15, lits=[kb_sub])
                   + c.t_eq(t_sub, v - 10, -30, 15, lits=[kb_sub])
                   + c.t_eq(t_rsub, v, -30, 15, lits=[kb_rsub])
                   + c.t_eq(t_rsub, v - 10, -30, 15, lits=[kb_rsub])
                   + cat(c.t_eq(t_adb, v + 10 * q_, 0, 45, lits=[kb_adb]) for q_ in range(3))
                   + cat(c.t_eq(t_div, v - 10 * q_, -246, 24, lits=[kb_div]) for q_ in range(10)),
                   f"mac.d=={v}") for v in range(10)]
    digit_val = onehot_bits(dsum, 4)

    def addr_bits(oh, base):  # bits of base + index for a one-hot index
        return [sum((e for i, e in enumerate(oh) if (base + i) >> b_ & 1), Expr())
                for b_ in range(13)]

    # what a token predicts is about the next write, index j (reads above use wi)
    kk_add = eqs(kmx - j, lo_, hi_, MD + 1, "add.k", [kb_as])  # n - 1 - w
    kk_mac = eqs(kla + krow - j, lo_, hi_, 2 * MD, "mac.kw", [kb_mac])
    zero_k = eqs(j, 0, JMAX + 1, 2 * MD, "zero.k", [kb_zero])
    row_oh = eqs(krow, -32, 31, MD, "mac.row", [kb_mac])
    # V0 = n - [R[0] == 0] = max + carry into the top digit, I = R_ADDR + 1 - carry
    n_add = eqs(kmx + c_in, 0, 47, MD + 2, "add.n", [kb_as])
    i_add = [one - carry[0]] + [bios.R_ADDR >> b_ & 1 for b_ in range(1, 12)]
    n_zero = eqs(kla + klb, 0, 62, 2 * MD + 1, "zero.n", [kb_zero])
    in_add = c.node(c.t_lt(j - kmx, 1, lo_, hi_, lits=[kb_as]), "add.digit")
    end_add = c.node(c.t_eq(j - kmx, 1, lo_, hi_, lits=[kb_as]), "add.end")  # V0 = n
    end_add2 = c.node(c.t_eq(j - kmx, 2, lo_, hi_, lits=[kb_as]), "add.end2")  # I = R
    end_rsub = c.node(c.t_eq(j - kmx, 3, lo_, hi_, lits=[kb_rsub]), "rsub.end")  # MODE |= 8
    in_zero = c.node(c.t_lt(j - kla - klb, 0, lo_, hi_, lits=[kb_zero]), "zero.digit")
    end_zero = c.node(c.t_eq(j - kla - klb, 0, lo_, hi_, lits=[kb_zero]), "zero.end")
    in_mac = c.node(c.t_lt(j - kla, 0, lo_, hi_, lits=[kb_mac]), "mac.digit")
    end_mac = c.node(c.t_eq(j - kla, 0, lo_, hi_, lits=[kb_mac]), "mac.end")  # R[row]
    mul_vx = c.node(c.t_eq(j - kla, 1, lo_, hi_, lits=[kb_mac]), "mul.vx")  # VX += 1
    mul_v0 = c.node(c.t_eq(j - kla, 2, lo_, hi_, lits=[kb_mac, k["last"][0]]), "mul.v0")
    mul_i = c.node(c.t_eq(j - kla, 3, lo_, hi_, lits=[kb_mac, k["last"][0]]), "mul.i")
    # the last row's R[0] = carry is its own VAL token: keep [it is 0] for V0 and I
    mul_endv = c.node(c.t_eq(j - 1 - kla, 0, lo_, hi_, lits=[c.is_val, kb_mac]), "mul.endv")
    (mul_z,) = c.slot_head("mul.top0", {0: rs + mul_endv}, {0: one},
                           {2: c.and_([one - b_ for b_ in c.vb[:4]], "val==0")}, rs + mul_endv)
    n_mul = eqs(kla + klb - mul_z, -1, 62, 2 * MD + 1, "mul.n", [kb_mac])
    i_mul = [mul_z] + [bios.R_ADDR >> b_ & 1 for b_ in range(1, 12)]
    v7 = spec.V + 7

    # machine records (not inside a print or a CALL record)
    nfd = one - k["fd"][0] - calling
    for m in range(3):
        case([at_val, nfd, not_tick, k["fixed"][0], jeq(m), k["has"][m]], 0, k[f"W{m}"])
        case([at_addr, nfd, not_tick, k["fixed"][0], jeq(m), k["has"][m]], 1, k[f"A{m}"])
    jbits = onehot_bits([jeq(v) for v in range(16)], 4)
    case([at_val, nfd, not_tick, k["fx55"][0], le_x], 0, vread)
    case([at_addr, nfd, not_tick, k["fx55"][0], le_x], 1, i_j)
    case([at_val, nfd, not_tick, k["fx65"][0], le_x], 0, vread)
    case([at_addr, nfd, not_tick, k["fx65"][0], le_x], 1, jbits + [0] * 8 + [1])
    case([at_val, nfd, not_tick, k["ls_extra"][0], is_i_adv], 0, k["i_next"])
    case([at_addr, nfd, not_tick, k["ls_extra"][0], is_i_adv], 1, spec.I)
    case([at_val, nfd, not_tick, k["dxyn"][0], lt_2n, drawn], 0, new_byte)
    case([at_addr, nfd, not_tick, k["dxyn"][0], lt_2n], 1, target)
    case([at_val, nfd, not_tick, k["dxyn"][0], eq_2n], 0, [any_collision])
    case([at_addr, nfd, not_tick, k["dxyn"][0], eq_2n], 1, VF)
    case([at_val, nfd, not_tick, is_pc, one - kb_sub - kb_dv], 0, k["next_pc"])
    case([at_val, nfd, not_tick, is_pc, kb_sub, one - carry[0]], 0, k["next_pc"])  # T
    case([at_val, nfd, not_tick, is_pc, kb_sub, carry[0]], 0, k["pc2"])  # borrowed: on
    case([at_addr, nfd, not_tick, is_pc, one - k["stay"][0]], 1, spec.PC)
    case([at_addr, nfd, not_tick, is_pc, k["stay"][0]], 2, 0)  # PC stays: no PC write
    case([at_addr, nfd, not_tick, is_end], 2, 0)
    case([at_val, nfd, in_tick, jeq(0)], 0, k["dt_dec"])
    case([at_addr, nfd, in_tick, jeq(0)], 1, spec.DT)
    case([at_val, nfd, in_tick, jeq(1)], 0, k["st_dec"])
    case([at_addr, nfd, in_tick, jeq(1)], 1, spec.ST)
    case([at_addr, nfd, in_tick, jeq(2)], 2, 0)
    case([at_addr, nfd, in_add], 1, addr_bits(kk_add, bios.R_ADDR))
    case([at_val, nfd, in_add], 0, digit_val)
    case([at_addr, nfd, end_add], 1, V)
    case([at_val, nfd, end_add, one - kb_rsub], 0, onehot_bits(n_add, 6))
    case([at_val, nfd, end_add, kb_rsub, mode], 0, onehot_bits(n_add, 6))  # tool call: 0
    case([at_addr, nfd, end_add2], 1, spec.I)
    case([at_val, nfd, end_add2], 0, i_add)
    case([at_addr, nfd, end_rsub], 1, bios.MODE)
    case([at_val, nfd, end_rsub], 0, [mode, mode_d, mode_dd, 1])
    case([at_addr, nfd, in_zero], 1, addr_bits(zero_k, bios.R_ADDR))
    case([at_addr, nfd, end_zero], 1, v7)
    case([at_val, nfd, end_zero], 0, onehot_bits(n_zero, 6))
    case([at_addr, nfd, in_mac], 1, addr_bits(kk_mac, bios.R_ADDR))
    case([at_val, nfd, in_mac], 0, digit_val)
    case([at_addr, nfd, end_mac], 1, addr_bits(row_oh, bios.R_ADDR))
    case([at_val, nfd, end_mac], 0, list(carry))
    case([at_addr, nfd, mul_vx], 1, onehot_bits(k["x"], 4) + [0] * 8 + [1])  # V + x
    case([at_val, nfd, mul_vx], 0, k["vx1"])
    case([at_addr, nfd, mul_v0], 1, V)
    case([at_val, nfd, mul_v0], 0, onehot_bits(n_mul, 6))
    case([at_addr, nfd, mul_i], 1, spec.I)
    case([at_val, nfd, mul_i], 0, i_mul)

    # long division records: the window (LB digits from the right, then its top digit), the
    # quotient digit, VX, VF, then PC (back to the same instruction while it goes on)
    kw_oh = eqs(kt - j + 1, -64, 65, NWC, "div.kwj", [kb_dv])
    case([at_addr, nfd, in_dv], 1, addr_bits(kw_oh, bios.W_ADDR))
    case([at_val, nfd, in_dv], 0, digit_val)
    pq_oh = eqs(pq, 0, 64, 2 * MD, "div.pq", [kb_dv])
    case([at_addr, nfd, dv_p], 1, addr_bits(pq_oh, bios.R_ADDR))
    pm1 = [c.node(c.t_eq(p_i, v + 1, 0, 15), f"P-1=={v}") for v in range(15)]
    p_dec = [sum((e for v, e in enumerate(pm1) if v >> b_ & 1), Expr()) + one - sum(pm1, Expr())
             for b_ in range(8)]  # P - 1 (0 - 1 = 255)
    case([at_val, nfd, dv_p, kb_div], 0, q_new)
    case([at_val, nfd, dv_p, kb_adb], 0, p_dec)
    case([at_addr, nfd, dv_at[0]], 1, onehot_bits(k["x"], 4) + [0] * 8 + [1])
    # flags, after the window: its last carry is BDIV's "negative" / BADB's "carried out", and a
    # counter of the window's nonzero digits says whether the remainder is 0
    nz_tok = c.and_([c.is_val, dv_win, one - d_lt[1]], "div.nz_tok")
    (xz,) = c.slot_head("div.nz", {0: rs, 1: one - rs}, {0: one - rs, 1: one - rs}, {2: rs},
                        one - rs, 110 * nz_tok + 238 * rs, tiny=2.0**-90)
    dnz = c.node([(1, one - rs, 7 * one - 8 * xz), (-1, one - rs, 6 * one - 8 * xz)], "div.nz")
    ge_la = c.node(c.t_ge(kt - kla, 0, -31, 63), "div.t>=la")
    at_end = c.node(c.t_eq(kt - kla, bios.DEC, -31, 63), "div.t=end")
    neg = carry[0]  # 0/1 everywhere (c_in is 0/1 only here)
    f1 = c.node(c.t_and([kb_div, neg]) + c.t_and([kb_adb, one - neg]), "div.f1")
    f2 = c.node(c.t_and([kb_div, one - neg, one - dnz, ge_la])
                + c.t_and([kb_adb, neg, one - dnz, ge_la]), "div.f2")
    f3 = c.node(c.t_and([kb_div, one - neg, dnz, at_end])
                + c.t_and([kb_adb, neg, dnz, at_end]), "div.f3")
    f0 = one - f1 - f2 - f3
    t_oh = eqs(kt, 0, 64, 64, "div.t", [kb_dv])
    t1_oh = eqs(kt + 1, 0, 65, 64, "div.t+1", [kb_dv])
    case([at_val, nfd, dv_at[0], f0], 0, onehot_bits(t1_oh, 8))
    case([at_val, nfd, dv_at[0], one - f0], 0, onehot_bits(t_oh, 8))
    case([at_addr, nfd, dv_at[1]], 1, spec.V + 15)
    case([at_val, nfd, dv_at[1]], 0, [f1 + f3, f2 + f3])
    case([at_val, nfd, not_tick, is_pc, kb_div, f0], 0, k["pc"])
    case([at_val, nfd, not_tick, is_pc, kb_div, one - f0], 0, k["next_pc"])
    case([at_val, nfd, not_tick, is_pc, kb_adb, f1], 0, k["pc"])
    case([at_val, nfd, not_tick, is_pc, kb_adb, one - f1], 0, k["next_pc"])


    # a print: STEP, then NL <tool_response> NL digits NL </tool_response> NL as text tokens.
    # Position j (STEP = 0, then each printed token) predicts printed token j; n = min(V0, 58).
    v0 = from_bits(k["v0"])
    z = j - v0
    small = c.node(c.t_lt(v0, B.MAX_PRINT + 1, 0, 255), "print.small")
    big = one - small
    lim = B.MAX_PRINT

    def zj(dz, name):  # [j == n + dz]
        return c.node(c.t_eq(z, dz, -255, JMAX + 1, lits=[small])
                      + c.t_and([big, jeq(lim + dz)]), name)

    nl_end, trc, nl_last = zj(3, "print.nl1"), zj(4, "print.trc"), zj(5, "print.nl2")
    # tool call: [3 <= j < 3 + n] = [j >= 3] - [j >= 3 + n]; inline: [1 <= j < 1 + n]
    tool = one - mode
    digit = c.node(c.t_and([cnt[2], tool]) + c.t_ge(z, 3, -255, JMAX + 1, -1, lits=[small, tool])
                   + c.t_and([big, tool, cnt[lim + 2]], -1)
                   + c.t_and([cnt[0], mode]) + c.t_ge(z, 1, -255, JMAX + 1, -1, lits=[small, mode])
                   + c.t_and([big, mode, cnt[lim]], -1), "print.digit")
    nz = c.node(c.t_ge(v0, 1, 0, 255), "print.nonempty")
    upto = c.node(c.t_lt(z, 6, -255, JMAX + 1, lits=[small, tool])
                  + c.t_and([big, tool, one - cnt[lim + 5]])
                  + c.t_lt(z, 2, -255, JMAX + 1, lits=[small, mode, nz])
                  + c.t_lt(z, 1, -255, JMAX + 1, lits=[small, mode, one - nz])
                  + c.t_and([big, mode, one - cnt[lim + 1]]), "print.upto")
    # inline: a space after the digits, so Qwen does not read on into the number
    trail = c.node(c.t_eq(z, 1, -255, JMAX + 1, lits=[small, mode, nz])
                   + c.t_and([big, mode, jeq(lim + 1)]), "print.trail")
    in_print = [k["fd"][0], c.is_step + c.is_text, one - calling]
    code = {cls: TEXT_CODE + cls - B.TEXT_BASE
            for cls in (B.NL, B.TR_OPEN, B.TR_CLOSE, B.SPACE, B.MINUS, B.PERIOD, B.DOLLAR,
                        B.DDOLLAR)}
    for lits, cls in [([tool, jeq(0)], B.NL), ([tool, jeq(1)], B.TR_OPEN), ([tool, jeq(2)], B.NL),
                      ([tool, nl_end], B.NL), ([tool, trc], B.TR_CLOSE), ([tool, nl_last], B.NL),
                      ([mode, jeq(0), one - mode_neg], B.SPACE),
                      ([mode, jeq(0), mode_neg], B.MINUS), ([trail, one - mode_d], B.PERIOD),
                      ([trail, mode_d, one - mode_dd], B.DOLLAR), ([trail, mode_dd], B.DDOLLAR)]:
        case(in_print + lits, 2, code[cls])
    # TEXT_CODE + low nibble; 10 and 11 (a quotient's "." and "...") print as their classes
    p10, p11 = (c.and_(nib_lits(vread[:4], v), f"print.is{v}") for v in (10, 11))
    case(in_print + [digit, one - p10 - p11], 2, list(vread[:4]) + [1])
    case(in_print + [digit, p10], 2, TEXT_CODE + B.PERIOD - B.TEXT_BASE)
    case(in_print + [digit, p11], 2, TEXT_CODE + B.ELLIPSIS - B.TEXT_BASE)
    # who predicts the next token: the circuit on chip tokens and inside a print, else Qwen
    owner = c.node(c.t_and([c.mu]) + c.t_and([c.is_text, k["fd"][0], cnt[0], upto,
                                              one - calling])
                   + c.t_and([c.is_text, calling]), "owner")

    nxt = [c.node([t for lits, kind, _ in A if kind == 1 for t in c.t_and(lits)], "next.is_addr"),
           c.node([t for lits, kind, _ in A if kind == 2 for t in c.t_and(lits)], "next.is_ctrl")]
    nxt += c.select_bits([(lits, p) for lits, _, p in A], 13, "next")
    c.next_bits = nxt
    c.owner = owner
    c.cases = A  # (literals, kind, payload), for debugging
    # ---------------------------------------------------------- exit (the last chip sublayer)
    # The emitter clears every other chip dim here, so after the chip layers a token holds only
    # its Qwen part, mu, the output code (KAPPA * +-1 where the circuit owns the position, else
    # 0) and tau (0 on chip tokens, the text value on text tokens).
    for i, e in enumerate(c.code):  # the input code goes first, so the exit adds onto 0
        c.ffn([(-1, one, e), (1, one, -1 * e)], f"code.clear{i}", into=e)
    top = max(n.level for n in c.nodes)
    for i, e in enumerate(c.code):
        c.ffn(c.t_and([nxt[i], owner], 2 * KAPPA) + c.t_and([owner], -KAPPA),
              f"exit.code{i}", into=e, min_level=top)
    # tau: 0 wherever the circuit predicts (chip tokens: -1 + 1; owned text: TAU_TEXT - TAU_TEXT),
    # TAU_TEXT where Qwen does, which keeps chip tokens out of Qwen's text
    c.ffn([(1, c.mu, one)] + c.t_and([owner, c.is_text], -TAU_TEXT), "exit.tau", into=c.tau,
          min_level=top)
    c.keep = [next(iter(e.terms)) for e in c.code] + [next(iter(c.tau.terms)),
                                                       next(iter(c.mu.terms))]
    c.out = {"regs": R, "x": x, "cnt": cnt, "in_tick": in_tick, "old": old, "b1": b1, "b2": b2,
             "ind": ind, "vx": vxb, "vy": vyb, "ctx": k, "vread": vread, "next": nxt}
    return c
