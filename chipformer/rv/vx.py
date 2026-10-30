"""VX: a streaming vector unit for the RV core (reference semantics and trace records; docs/vx.md).

Custom-2 (opcode 0x5B, R-type, funct7 = 0). The unit sits in the circuit's first layers with its
own memories, so it never sees the registers or RAM: an op's record starts with VC tokens that
carry its configuration (16 bits each, packed from the registers rd / rs1 / rs2), then streams its
results as VE tokens (16 bits; VB: the same, and the element is the tracker's new max), then
commits with NT. VGET answers with a value record; VPUT's VC tokens carry the data itself.

Memories (all the unit's own):
  VM    16 vector registers of 64 int32 (bank A v0..v7, bank B v8..v15; an op reads at most one
        register per bank), stored as element pairs: results are written a pair at a time
  VA    256 int8 activations, read 64 at a time by VMAT
  KC    the K cache: 1024 rows of 8 int8          VCC  the V cache, likewise
  WROM  row-chunks of 72 cells (`.wrom`): 64 int4 weights (cell u means u - 8), then the row's
        scale S < 2^16 in 4 cells (nibbles, least significant first), then 4 unused
  RoPE  4 (cos, sin) int16 pairs                  trackers: vmx, vsum, the argmax (vbest, varg)

Sized for the circuit: multiplies of a 16-bit window of a value, clamp16(v >> sh), by a 15- or
16-bit factor; dot products of at most 64 int8 x int4 or 32 int8 x int8 lanes a token; SiLU and
exp as fixed-point function units. `check` raises on operands outside those widths.

shift(x, s) is x << s for s >= 0, x >> -s below; clamp16 clamps to [-32768, 32767]. Every config
field below is packed, in the order listed, into the record's VC tokens, 16 bits each.

  f3 0  VMAT   rd: dst 4, n 7, k 2, mode 1, keep 1, chunk 2; rs1: base 13, doff 6
               rows r < n: acc = sum over c < k of VA[64 (chunk + c):][:64] . row-chunk
               base + r k + c; out = acc * S (the last chunk's) mod 2^32. Mode 0: VM[dst][doff + r]
               = out (n, doff even), max |out| tracked, 2 tokens per chunk; mode 1: argmax of out,
               a token per chunk.
  f3 1  VMATR  rd: dst 4, n 6, mode 1, keep 1, off 8; rs1: base 10, eoff 6; rs2 (mode 1): s 17
               (unsigned)
               mode 0 (scores): out[t] = VA[off:][:8] . KC[base + t], t < n, signed max tracked
               mode 1 (p.V): out[j] = (VA[off:][:n] . VCC[base + t][j] over t) * s >> 12, j < 8,
               max |out| tracked. out goes to VM[dst][eoff + i], 2 tokens each; an odd n's last
               pair gets a 0 (not tracked).
  f3 2  VQ     rd: src 4, eoff 6, n 7, dst 2, sh 5; rs1: off 10; rs2: r 15
               q_i = clamp(round(clamp16(VM[src][eoff + i] >> sh) * r / 2^15), +-127), i < n
               (n even); dst 0: VA[off + i]; 1 / 2: KC / VCC row off + 32 (i / 8), byte i mod 8.
               A token per 2 elements.
  f3 3  VEW    rd: a 4, b 4, n 7, op 2, keep 1, eoff 6; rs1: sh 5, so 6 (signed), shb 5;
               rs2: p 15
               i < n (n even) of VM[a][eoff + i], in place, max |result| tracked:
               op 0 ADD  a += shift(clamp16(b >> sh) * p, so)                       2 tokens each
               op 1 GLU  y = shift(clamp16(a >> sh) * p, so), a = SILU(y) * clamp16(b >> shb) >> 8
                                                                                    4 tokens each
               (b = VM[b][eoff + i])
               op 2 ROPE pairs (a, a'): u = clamp16(a >> sh), w = clamp16(a' >> sh), by RoPE pair
                         (i mod 8) / 2: (shift(u c - w s, sh - 14), shift(u s + w c, sh - 14))
                                                                                    2 tokens each
  f3 4  VEXP   rd: src 4, n 7, off 8, ksh 5; rs2: c 16
               i < n: e = min(vmx - VM[src][i], 65535), d = e * c >> ksh; VA[off + i] = EXP(d)
               (an odd n's last pair gets a 0). A token per 2 elements.
  f3 5  VGET   rs1 = sel: rd = vmx / vsum / varg / bit length of vmx: a value record
  f3 6  VRED   rd: n 7, mode 1, off 8: vsum = sum VA[off + i]^2 (mode 0) or VA[off + i], i < n.
               One token.
  f3 7  VPUT   rd: target 10; rs1, rs2: the data (2 VC tokens each)
               target < 512: VM pair target (register target / 32) = (rs1, rs2);
               512 + k: RoPE pair k = (cos rs1, sin rs2)
SILU(y) (y in Q16): the Q12 table round(x / (1 + e^-x) 4096) at x = (y >> 11) / 32 for |y| < 8.0,
min(y >> 4, 2^23) from 8.0, 0 below -8.0. EXP(d) = round(127 e^(-d / 32)) for d < 256, else 0.
Streams are at most 128 tokens (software splits longer vectors into ops).
"""

import math

from chipformer.rv import spec as S

OPS = ["VMAT", "VMATR", "VQ", "VEW", "VEXP", "VGET", "VRED", "VPUT"]
MAX_STREAM = 128
M32 = 0xFFFFFFFF
CHUNK_CELLS = 72
SILU_T = [round((x := (i - 256) / 32) / (1 + math.exp(-x)) * 4096) for i in range(512)]
EXP_T = [round(127 * math.exp(-i / 32)) for i in range(256)]
FIELDS = {  # op -> [(register, lo bit, width)], packed into VC tokens in this order
    "VMAT": [("rd", 0, 4), ("rd", 4, 7), ("rd", 11, 2), ("rd", 13, 1), ("rd", 14, 1),
             ("rd", 15, 2), ("rs1", 0, 13), ("rs1", 13, 6)],
    "VMATR": [("rd", 0, 4), ("rd", 4, 6), ("rd", 10, 1), ("rd", 11, 1), ("rd", 12, 8),
              ("rs1", 0, 10), ("rs1", 10, 6), ("rs2", 0, 17)],
    "VQ": [("rd", 0, 4), ("rd", 4, 6), ("rd", 10, 7), ("rd", 17, 2), ("rd", 19, 5),
           ("rs1", 0, 10), ("rs2", 0, 15)],
    "VEW": [("rd", 0, 4), ("rd", 4, 4), ("rd", 8, 7), ("rd", 15, 2), ("rd", 17, 1),
            ("rd", 18, 6), ("rs1", 0, 5), ("rs1", 5, 6), ("rs1", 11, 5), ("rs2", 0, 15)],
    "VEXP": [("rd", 0, 4), ("rd", 4, 7), ("rd", 11, 8), ("rd", 19, 5), ("rs2", 0, 16)],
    "VRED": [("rd", 0, 7), ("rd", 7, 1), ("rd", 8, 8)],
    "VPUT": [("rd", 0, 10), ("rs1", 0, 32), ("rs2", 0, 32)],
}


def s32(v):
    v &= M32
    return v - (1 << 32) if v >> 31 else v


def sx(v, n):
    v &= (1 << n) - 1
    return v - (1 << n) if v >> (n - 1) else v


def clamp16(v):
    return max(-32768, min(32767, v))


def shift(x, s):
    return x << s if s >= 0 else x >> -s


def check(ok, what):
    if not ok:
        raise ValueError(f"VX operand out of range: {what}")


def silu(y):
    if y >= 8 << 16:
        return min(y >> 4, 1 << 23)
    if y < -(8 << 16):
        return 0
    return SILU_T[(y >> 11) + 256]


def ve(v, best=False):
    return S.tok("VB" if best else "VE", v & 0xFFFF)


def config(op, regs):
    """The op's config fields (by register name) and its VC tokens."""
    vals, bits, nb = [], 0, 0
    for r, lo, w in FIELDS[op]:
        f = regs[r] >> lo & ((1 << w) - 1)
        vals.append(f)
        bits |= f << nb
        nb += w
    return vals, [S.tok("VC", bits >> 16 * i & 0xFFFF) for i in range(-(-nb // 16))]


class VX:
    def __init__(self):
        self.vm = [[0] * 64 for _ in range(16)]
        self.va = [0] * 256
        self.kc = [[0] * 8 for _ in range(1024)]
        self.vc = [[0] * 8 for _ in range(1024)]
        self.rope = [(0, 0)] * 4
        self.vmx = self.vsum = self.varg = 0
        self.vbest = None  # the argmax value so far
        self.vrow = 0  # rows seen by argmax since the reset


def wrom_chunk(m, c):
    """(64 weights, the scale) of WROM row-chunk c (cells past the data read as 0: w = -8)."""
    o = CHUNK_CELLS * c
    cell = [m.wrom[o + i] & 15 if o + i < len(m.wrom) else 0 for i in range(68)]
    return [u - 8 for u in cell[:64]], sum(cell[64 + i] << 4 * i for i in range(4))


def execute(m, ins) -> list[int]:
    """The op's record (commits it)."""
    v, x, op = m.vx, m.x, ins.op
    m.n += 1
    regs = {"rd": x[ins.rd], "rs1": x[ins.rs1], "rs2": x[ins.rs2]}
    if op == "VGET":
        r = [v.vmx, v.vsum, v.varg, max(v.vmx, 0).bit_length()][regs["rs1"] & 3]
        if ins.rd:
            x[ins.rd] = r & M32
            return S.value_record(r)
        return [S.tok("NT")]
    f, toks = config(op, regs)
    n_vc = len(toks)

    def tracked(val, signed=False):
        """A word result's two tokens; the second is VB when it is the tracker's new max."""
        t = s32(val) if signed else abs(s32(val))
        best = t > v.vmx
        v.vmx = max(v.vmx, t)
        return [ve(val), ve(val >> 16, best)]

    def put(reg, i, val):
        check(0 <= i < 64, f"VM element {i}")
        v.vm[reg][i] = s32(val)

    if op == "VMAT":
        dst, n, k, mode, keep, chunk, base, doff = f
        check(k in (1, 3) and (mode or n % 2 == doff % 2 == 0) and 0 < n <= 64,
              f"VMAT n {n} k {k} doff {doff}")
        if not keep:
            v.vmx, v.vbest, v.vrow = 0, None, 0
        for r in range(n):
            acc, sc = 0, 0
            for c in range(k):
                w, sc = wrom_chunk(m, base + r * k + c)
                acc += sum(v.va[64 * (chunk + c) + i] * w[i] for i in range(64))
            o = s32(acc * sc)
            if mode == 0:
                put(dst, doff + r, o)
                toks += [ve(0)] * (2 * (k - 1)) + tracked(o)
            else:
                best = v.vbest is None or o > v.vbest
                if best:
                    v.vbest, v.varg = o, v.vrow
                v.vrow += 1
                toks += [ve(0)] * (k - 1) + [ve(o, best)]
    elif op == "VMATR":
        dst, n, mode, keep, off, base, eoff, s = f
        check(0 < n <= 32, f"VMATR n {n}")
        if not keep:
            v.vmx = -(1 << 31) if mode == 0 else 0
        if mode == 0:
            outs = [sum(v.va[(off + i) & 255] * v.kc[base + t][i] for i in range(8))
                    for t in range(n)]
        else:
            outs = [(sum(v.va[(off + t) & 255] * v.vc[base + t][j] for t in range(n)) * s) >> 12
                    for j in range(8)]
        for i, o in enumerate(outs):
            put(dst, eoff + i, o)
            toks += tracked(o, signed=mode == 0)
        if len(outs) % 2:
            put(dst, eoff + len(outs), 0)
            toks += [ve(0), ve(0)]
    elif op == "VQ":
        src, eoff, n, dst, sh, off, r = f
        check(n % 2 == 0 and 0 < n <= 64, f"VQ n {n}")
        qs = []
        for i in range(n):
            p = clamp16(v.vm[src][eoff + i] >> sh) * r
            q = (abs(p) + (1 << 14)) >> 15
            q = max(-127, min(127, q if p >= 0 else -q))
            qs.append(q)
            if dst == 0:
                v.va[(off + i) & 255] = q
            else:
                (v.kc if dst == 1 else v.vc)[off + 32 * (i // 8)][i % 8] = q
        toks += [ve((qs[i] & 255) | (qs[i + 1] & 255) << 8) for i in range(0, n, 2)]
    elif op == "VEW":
        a, b, n, sub, keep, eoff, sh, so, shb, p = f
        so = sx(so, 6)
        check(n % 2 == 0 and 0 < n <= 64 and sub < 3 and (sub == 2 or a >> 3 != b >> 3),
              f"VEW a {a} b {b} n {n} op {sub}")
        if not keep:
            v.vmx = 0
        old = v.vm[a][eoff:eoff + n]
        out = []
        for i in range(n):
            ai = old[i]
            if sub == 0:
                o = ai + shift(clamp16(v.vm[b][eoff + i] >> sh) * p, so)
            elif sub == 1:
                o = (silu(shift(clamp16(ai >> sh) * p, so)) * clamp16(v.vm[b][eoff + i] >> shb)) >> 8
            else:
                c, sn = v.rope[(i % 8) // 2]
                u, w = clamp16(old[i & ~1] >> sh), clamp16(old[i | 1] >> sh)
                o = shift(u * c - w * sn, sh - 14) if i % 2 == 0 else shift(u * sn + w * c, sh - 14)
            out.append(s32(o))
        for i, o in enumerate(out):
            put(a, eoff + i, o)
            toks += [ve(0), ve(0)] * (sub == 1) + tracked(o)
    elif op == "VEXP":
        src, n, off, ksh, c = f
        check(0 < n <= 64, f"VEXP n {n}")
        ps = []
        for i in range(n):
            e = min(v.vmx - v.vm[src][i], 65535)
            check(e >= 0, "VEXP above the max")
            d = (e * c) >> ksh
            ps.append(EXP_T[d] if d < 256 else 0)
            v.va[(off + i) & 255] = ps[-1]
        if n % 2:
            ps.append(0)
            v.va[(off + n) & 255] = 0
        toks += [ve(ps[i] | ps[i + 1] << 8) for i in range(0, n, 2)]
    elif op == "VRED":
        n, mode, off = f
        vals = [v.va[(off + i) & 255] for i in range(n)]
        v.vsum = sum(q * q for q in vals) if mode == 0 else sum(vals)
        toks.append(ve(v.vsum))
    elif op == "VPUT":
        target, lo, hi = f
        if target < 512:
            put(target >> 5, 2 * (target & 31), lo)
            put(target >> 5, 2 * (target & 31) + 1, hi)
        else:
            check(target < 516, f"VPUT target {target}")
            v.rope[target - 512] = (sx(lo, 16), sx(hi, 16))
    if len(toks) - n_vc > MAX_STREAM:
        raise ValueError(f"{op}: a {len(toks) - n_vc}-token stream (max {MAX_STREAM})")
    return toks + [S.tok("NT")]
