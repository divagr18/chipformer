"""CHIP-8 as a circuit graph: the transformer's "program".

Every position runs the same circuit and outputs ``next``, the id of the next
token. The computation has two halves:

1. Context (meaningful at record starts, i.e. STEP/BOOT positions): fetch PC
   and the opcode, decode, read operands, and compute every write of the
   record (addresses, values, count, next PC).
2. Micro-step (every position): copy the context from the latest record start
   with one attention head, work out which write of the record comes next
   (j) and whether we are predicting its ADDR or its VAL, do the per-write
   reads (FX55/FX65/DXYN), and pick the output.

Reading the context at the record start means every value is computed from
the state before the record, as the trace format requires (see
docs/trace_format.md), even after earlier writes in the same record.
"""

from chipformer import spec
from chipformer.dsl.graph import Expr
from chipformer.dsl.lib import MAX_POS, Circuit

JMAX = 40  # write indices are clamped here (largest record: DXYN with 32 writes)
NOT_FOUND = 2 * MAX_POS  # pushes "latest flagged position" checks negative when no flag exists



class Chip8Circuit(Circuit):
    def __init__(self) -> None:
        super().__init__()
        f = self.token_feature
        self.is_val = f("is_val", lambda t: int(spec.is_val(t)))
        self.is_addr = f("is_addr", lambda t: int(spec.is_addr(t)))
        self.is_step = f("is_step", lambda t: int(t == spec.STEP))
        self.is_tick = f("is_tick", lambda t: int(t == spec.TICK))
        self.is_boot = f("is_boot", lambda t: int(t == spec.BOOT))
        self.val = f("val", lambda t: t - spec.VAL_BASE if spec.is_val(t) else 0)
        self.addr = f("addr", lambda t: t - spec.ADDR_BASE if spec.is_addr(t) else 0)
        addr2 = f("addr^2", lambda t: (t - spec.ADDR_BASE) ** 2 if spec.is_addr(t) else 0)
        # at a VAL position: the address being written (from the ADDR token before it)
        self.wa, self.wa2 = self.prev([self.addr, addr2], "write_addr")

    def read(self, q: "Expr | int", name: str) -> Expr:
        """Current value of cell q: the latest write to it, or 0 if never written."""
        q = q if isinstance(q, Expr) else self.one * q
        v, found = self.latest_keyed(q, self.wa, self.wa2, self.is_val, [self.val],
                                     spec.NUM_CELLS, name)
        return self.eqglu(v, found - q, 0, f"{name}.hit")

    def read_display(self, q: Expr, name: str) -> Expr:
        """Like read, but a display byte written before the latest CLS reads as 0."""
        v, pos, found = self.latest_keyed(q, self.wa, self.wa2, self.is_val, [self.val, self.pos],
                                          spec.NUM_CELLS, name)
        hit = self.eqglu(v, found - q, 0, f"{name}.hit")
        (cls_pos,) = self.latest_keyed(self.one * spec.CLS, self.wa, self.wa2, self.is_val,
                                       [self.pos], spec.NUM_CELLS, f"{name}.cls")[:1]
        return self.stepglu(hit, pos - cls_pos - 1, f"{name}.after_cls")

    def flagged_since(self, pos: Expr, flag: Expr, start: Expr, name: str) -> Expr:
        """[latest position with flag=1 is after start]; pos/flag come from a latest() lookup."""
        return self.step(pos - start - 1 - NOT_FOUND * (1 - flag), name)


def build() -> Chip8Circuit:
    c = Chip8Circuit()
    V, VF = spec.V, spec.V + 0xF

    # ---------------------------------------------------------------- record
    (p0,) = c.latest(c.is_step + c.is_boot, [c.pos], "rec_start")
    ptick, tflag = c.latest(c.is_tick, [c.pos, c.is_tick], "last_tick")
    in_tick = c.flagged_since(ptick, tflag, p0, "in_tick")

    # ---------------------------------------------------------------- context
    pc = c.read(spec.PC, "PC")
    ireg = c.read(spec.I, "I")
    sp = c.read(spec.SP, "SP")
    dt = c.read(spec.DT, "DT")
    st = c.read(spec.ST, "ST")
    rng = c.read(spec.RNG, "RNG")
    v0 = c.read(V, "V0")
    keys = [c.read(spec.KEYS + k, f"key{k:X}") for k in range(16)]
    q = {name: c.read(spec.QUIRKS + i, f"q.{name}") for i, name in enumerate(spec.QUIRK_NAMES)}

    pc1 = c.wrap(pc + 1, 0x1000, "pc+1")
    pc2 = c.wrap(pc + 2, 0x1000, "pc+2")
    pc4 = c.wrap(pc + 4, 0x1000, "pc+4")
    sp_dec = c.wrap(sp + 15, 16, "sp-1")
    sp_inc = c.wrap(sp + 1, 16, "sp+1")
    dt_dec = dt - c.step(dt - 1, "dt>0")
    st_dec = st - c.step(st - 1, "st>0")

    b1 = c.read(pc, "op_hi")
    b2 = c.read(pc1, "op_lo")
    stack_top = c.read(spec.STACK + sp_dec, "stack_top")
    hi, x = c.divmod(b1, 16, 15, "op.hi")
    y, n = c.divmod(b2, 16, 15, "op.y")
    nn, nnn = b2, 256 * x + b2
    vx = c.read(V + x, "VX")
    vy = c.read(V + y, "VY")

    ind = decode(c, b1, b2, hi, n)

    xb = c.bits(vx, 8, "vx.b")
    yb = c.bits(vy, 8, "vy.b")
    vx_lo = Circuit.from_bits(xb[:4])
    and_xy = c.band_bits(xb, yb, "vx&vy")
    s = vx + vy
    carry = c.step(s - 256, "carry")
    ge_xy = c.step(vx - vy, "vx>=vy")
    ge_yx = c.step(vy - vx, "vy>=vx")
    eq_x_nn = c.eq(vx, nn, "vx==nn")
    eq_x_y = c.eq(vx, vy, "vx==vy")
    hundreds = c.floor_div(vx, 100, 2, "vx/100")
    tens_all = c.floor_div(vx, 10, 25, "vx/10")
    i_plus = [c.wrap(ireg + m, 0x1000, f"i+{m}") for m in range(3)]

    # keys: EX9E/EXA1 test key[VX & 0xF]; FX0A takes the lowest held key
    key_vx = c.index(keys, vx_lo, "key[vx]")
    prefix = [sum(keys[:k], Expr()) for k in range(16)]
    lowest = c.ffn([t for k in range(16) for t in (
        (k, keys[k], prefix[k] + 1), (-2 * k, keys[k], prefix[k]), (k, keys[k], prefix[k] - 1))],
        "lowest_key")
    any_key = c.step(sum(keys, Expr()) - 1, "any_key")

    # CXNN: one Galois LFSR step, then the low byte masked with NN
    rb = c.bits(rng, 12, "rng.b")
    shifted = rb[1:] + [Expr()]
    sbits = []
    for i in range(12):
        if spec.RNG_MASK >> i & 1:
            sbits.append(shifted[i] + rb[0] - 2 * c.gate(shifted[i], rb[0], f"lfsr{i}"))
        else:
            sbits.append(shifted[i])
    rng_next = Circuit.from_bits(sbits)
    nnb = c.bits(nn, 8, "nn.b")
    rnd = c.ffn([(1 << i, nnb[i], sbits[i]) for i in range(8)], "rnd&nn")

    # 8XY6/8XYE shift VX in place with the shift quirk, otherwise VY
    shr = c.mux(q["shift"], Circuit.from_bits(xb[1:]), Circuit.from_bits(yb[1:]), "shr")
    shl = c.mux(q["shift"], 2 * Circuit.from_bits(xb[:7]), 2 * Circuit.from_bits(yb[:7]), "shl")
    lsb = c.mux(q["shift"], xb[0], yb[0], "lsb")
    msb = c.mux(q["shift"], xb[7], yb[7], "msb")

    # Instructions whose writes (besides PC) are computed at the record start, as
    # (condition, writes) variants: conditions are one-hot, quirks pick the variant.
    fixed: list[tuple[Expr, list]] = [(ind[op], w) for op, w in {
        "00E0": [(spec.CLS, 0)],
        "00EE": [(spec.SP, sp_dec)],
        "2NNN": [(spec.STACK + sp, pc2), (spec.SP, sp_inc)],
        "6XNN": [(V + x, nn)],
        "7XNN": [(V + x, c.wrap(vx + nn, 256, "vx+nn"))],
        "8XY0": [(V + x, vy)],
        "ANNN": [(spec.I, nnn)],
        "CXNN": [(V + x, rnd), (spec.RNG, rng_next)],
        "FX07": [(V + x, dt)],
        "FX0A": [(V + x, c.mux(any_key, lowest, vx, "fx0a"))],
        "FX15": [(spec.DT, vx)],
        "FX18": [(spec.ST, vx)],
        "FX1E": [(spec.I, c.wrap(ireg + vx, 0x1000, "i+vx"))],
        "FX29": [(spec.I, spec.FONT_BASE + 5 * vx_lo)],
        "FX33": [(i_plus[0], hundreds), (i_plus[1], tens_all - 10 * hundreds),
                 (i_plus[2], vx - 10 * tens_all)],
    }.items()]
    for op, result in [("8XY1", vx + vy - and_xy), ("8XY2", and_xy), ("8XY3", vx + vy - 2 * and_xy)]:
        fixed.append((c.gate(ind[op], 1 - q["logic"], f"{op}.plain"), [(V + x, result)]))
        fixed.append((c.gate(ind[op], q["logic"], f"{op}.vfreset"), [(V + x, result), (VF, 0)]))
    for op, result, flag in [("8XY4", s - 256 * carry, carry),
                             ("8XY5", vx - vy + 256 * (1 - ge_xy), ge_xy), ("8XY6", shr, lsb),
                             ("8XY7", vy - vx + 256 * (1 - ge_yx), ge_yx), ("8XYE", shl, msb)]:
        fixed.append((c.gate(ind[op], 1 - q["vforder"], f"{op}.flaglast"),
                      [(V + x, result), (VF, flag)]))
        fixed.append((c.gate(ind[op], q["vforder"], f"{op}.flagfirst"),
                      [(VF, flag), (V + x, result)]))
    fa = [c.select([(cond, w[m][0]) for cond, w in fixed if len(w) > m], f"A{m}") for m in range(3)]
    fv = [c.select([(cond, w[m][1]) for cond, w in fixed if len(w) > m], f"W{m}") for m in range(3)]
    has = [sum((cond for cond, w in fixed if len(w) > m), Expr()) for m in range(3)]
    is_vec = ind["FX55"] + ind["FX65"] + ind["DXYN"]

    # FX55/FX65 advance I by X+1 after the transfer unless the loadstore quirk is set
    ls_extra = c.gate(ind["FX55"] + ind["FX65"], 1 - q["loadstore"], "i_advance")
    i_next = c.wrap(ireg + x + 1, 0x1000, "i+x+1")
    nw = (sum((len(w) * cond for cond, w in fixed), Expr()) + ls_extra
          + c.ffn([(1, x + 1, ind["FX55"]), (1, x + 1, ind["FX65"]), (1, 2 * n + 1, ind["DXYN"])],
                  "vec_count"))

    skip = c.ffn([(1, ind["3XNN"], eq_x_nn), (1, ind["4XNN"], 1 - eq_x_nn),
                  (1, ind["5XY0"], eq_x_y), (1, ind["9XY0"], 1 - eq_x_y),
                  (1, ind["EX9E"], key_vx), (1, ind["EXA1"], 1 - key_vx)], "skip")
    wait = c.gate(ind["FX0A"], 1 - any_key, "wait")
    bnnn = c.wrap(nnn + c.mux(q["jump"], vx, v0, "jump_reg"), 0x1000, "jump_target")
    next_pc = pc2 + c.select([
        (skip, pc4 - pc2), (ind["1NNN"] + ind["2NNN"], nnn - pc2),
        (ind["00EE"], stack_top - pc2), (ind["BNNN"], bnnn - pc2), (wait, pc - pc2)], "next_pc")

    x0b, y0b = xb[:6], yb[:5]  # sprite origin = (VX % 64, VY % 32)
    ctx = {
        "x": x, "n": n, "I": ireg, "nw": nw, "next_pc": next_pc,
        "fixed": 1 - is_vec, "has0": has[0], "has1": has[1], "has2": has[2],
        "A0": fa[0], "A1": fa[1], "A2": fa[2], "W0": fv[0], "W1": fv[1], "W2": fv[2],
        "fx55": ind["FX55"], "fx65": ind["FX65"], "dxyn": ind["DXYN"],
        "dx_byte": Circuit.from_bits(x0b[3:]), "dx_off": Circuit.from_bits(x0b[:3]),
        "dy": Circuit.from_bits(y0b), "dt_dec": dt_dec, "st_dec": st_dec,
        "ls_extra": ls_extra, "i_next": i_next, "clip": q["clip"],
    }
    k = dict(zip(ctx, c.latest(c.is_step + c.is_boot, list(ctx.values()), "ctx")))

    # ---------------------------------------------------------------- micro-step
    j_raw = (c.pos - p0 - in_tick - c.is_addr) / 2
    j = j_raw - c.relu(j_raw - JMAX, "j.clamp")
    j_is = [c.eq(j, m, f"j=={m}") for m in range(3)]
    row_i, side = c.divmod(j, 2, JMAX // 2, "j/2")
    is_pc = c.eq(j - k["nw"], 0, "j==nw")
    is_end = c.eq(j - k["nw"] - 1, 0, "j==nw+1")
    le_x = c.step(k["x"] - j, "j<=x")
    lt_2n = c.step(2 * k["n"] - 1 - j, "j<2n")
    eq_2n = c.eq(j - 2 * k["n"], 0, "j==2n")
    is_i_advance = c.eq(j - k["x"] - 1, 0, "j==x+1")

    i_j = c.wrap(k["I"] + j, 0x1000, "i+j")
    i_row = c.wrap(k["I"] + row_i, 0x1000, "i+row")
    q_vec = c.select([(k["fx55"], V + j), (k["fx65"], i_j), (k["dxyn"], i_row)], "q_vec")
    vread = c.read(q_vec, "vread")
    old = c.read_display(c.addr, "old")  # at an ADDR position: the target byte's current value

    # DXYN rows: write j targets display byte (dy + j//2, dx_byte + j%2). Off-screen bytes
    # go to NULL with the clip quirk, otherwise the row and byte wrap around the screen.
    row = k["dy"] + row_i
    col = k["dx_byte"] + side
    row_w = row - 32 * c.step(row - 32, "row>=32")
    col_w = col - 8 * c.step(col - 8, "col>=8")
    on_screen = c.all_of(c.step(31 - row, "row<32"), c.step(7 - col, "col<8"), name="on_screen")
    drawn = 1 - c.gate(1 - on_screen, k["clip"], "clipped")
    target = spec.NULL + c.gate(spec.DISPLAY + 8 * row_w + col_w - spec.NULL, drawn, "target")
    shift = [c.eq(2 * k["dx_off"] + side, kk, f"shift{kk}") for kk in range(16)]
    sb = c.bits(vread, 8, "sprite.b")
    part_bits = []
    for p in range(8):
        terms = []
        for kk in range(16):
            o, right = kk // 2, kk % 2
            src = p - 8 + o if right else p + o  # (s << (8-o)) & 0xFF  or  s >> o
            if 0 <= src < 8:
                terms.append((1, sb[src], shift[kk]))
        part_bits.append(c.ffn(terms, f"part{p}"))
    part = Circuit.from_bits(part_bits)
    ob = c.bits(old, 8, "old.b")
    overlap = c.ffn([(1 << p, ob[p], part_bits[p]) for p in range(8)], "old&part")
    new_byte = old + part - 2 * overlap
    not_tick = 1 - in_tick
    draw_row = c.all_of(c.is_addr, not_tick, k["dxyn"], lt_2n, name="draw_row")
    collided = c.step(overlap - 1 - 256 * (1 - draw_row), "collided")
    cpos, cflag = c.latest(collided, [c.pos, collided], "last_collision")
    any_collision = c.flagged_since(cpos, cflag, p0, "any_collision")

    # ---------------------------------------------------------------- output
    at_val, at_addr = c.is_addr, 1 - c.is_addr
    A = spec.ADDR_BASE
    cases: list[tuple[Expr, Expr | int]] = []

    def case(conds: list[Expr], value: "Expr | int", name: str) -> None:
        cases.append((c.all_of(*conds, name=name), value))

    for m in range(3):
        case([at_val, not_tick, k["fixed"], j_is[m], k[f"has{m}"]], k[f"W{m}"], f"fix{m}.val")
        case([at_addr, not_tick, k["fixed"], j_is[m], k[f"has{m}"]], A + k[f"A{m}"], f"fix{m}.addr")
    case([at_val, not_tick, k["fx55"], le_x], vread, "fx55.val")
    case([at_addr, not_tick, k["fx55"], le_x], A + i_j, "fx55.addr")
    case([at_val, not_tick, k["fx65"], le_x], vread, "fx65.val")
    case([at_addr, not_tick, k["fx65"], le_x], A + V + j, "fx65.addr")
    case([at_val, not_tick, k["ls_extra"], is_i_advance], k["i_next"], "i_advance.val")
    case([at_addr, not_tick, k["ls_extra"], is_i_advance], A + spec.I, "i_advance.addr")
    case([at_val, not_tick, k["dxyn"], lt_2n, drawn], new_byte, "draw.val")
    case([at_addr, not_tick, k["dxyn"], lt_2n], A + target, "draw.addr")
    case([at_val, not_tick, k["dxyn"], eq_2n], any_collision, "vf.val")
    case([at_addr, not_tick, k["dxyn"], eq_2n], A + VF, "vf.addr")
    case([at_val, not_tick, is_pc], k["next_pc"], "pc.val")
    case([at_addr, not_tick, is_pc], A + spec.PC, "pc.addr")
    case([at_addr, not_tick, is_end], spec.STEP, "end")
    case([at_val, in_tick, j_is[0]], k["dt_dec"], "dt.val")
    case([at_addr, in_tick, j_is[0]], A + spec.DT, "dt.addr")
    case([at_val, in_tick, j_is[1]], k["st_dec"], "st.val")
    case([at_addr, in_tick, j_is[1]], A + spec.ST, "st.addr")
    case([at_addr, in_tick, j_is[2]], spec.STEP, "tick.end")
    c.output("next", c.select(cases, "next"))
    return c


def decode(c: Circuit, b1: Expr, b2: Expr, hi: Expr, n: Expr) -> dict[str, Expr]:
    """One-hot opcode classes (names as in ref.emu.opcode_class), one FFN sublayer."""
    ind = {}
    for h, name in [(1, "1NNN"), (2, "2NNN"), (3, "3XNN"), (4, "4XNN"), (6, "6XNN"),
                    (7, "7XNN"), (0xA, "ANNN"), (0xB, "BNNN"), (0xC, "CXNN"), (0xD, "DXYN")]:
        ind[name] = c.eq(hi, h, name)
    full = 256 * b1 + b2
    ind["00E0"] = c.eq(full, 0x00E0, "00E0")
    ind["00EE"] = c.eq(full, 0x00EE, "00EE")
    hn = 16 * hi + n
    ind["5XY0"] = c.eq(hn, 0x50, "5XY0")
    ind["9XY0"] = c.eq(hn, 0x90, "9XY0")
    for m in [0, 1, 2, 3, 4, 5, 6, 7, 0xE]:
        ind[f"8XY{m:X}"] = c.eq(hn, 0x80 + m, f"8XY{m:X}")
    hb = 256 * hi + b2
    for code in [0x9E, 0xA1]:
        ind[f"EX{code:02X}"] = c.eq(hb, 0xE00 + code, f"EX{code:02X}")
    for code in [0x07, 0x0A, 0x15, 0x18, 0x1E, 0x29, 0x33, 0x55, 0x65]:
        ind[f"FX{code:02X}"] = c.eq(hb, 0xF00 + code, f"FX{code:02X}")
    return ind
