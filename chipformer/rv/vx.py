"""VX: a streaming vector unit for the RV core (reference semantics and trace records only).

Custom-2 (opcode 0x5B, R-type, funct7 = 0). A vector op is one instruction whose record streams
its results, then commits with NT. rd is read as the op's config word (the register file's RD read
port) and is not written, except by VGET. Every token carries at most 16 bits of result (a VE
token); results that only update the unit's own state (the trackers, VA) are written from the
token payloads, so the trace costs what has to come back down the stack.

State: VA, 256 signed bytes (the activation vector, read whole by VMAT / VMATR); vp, a RAM / DROM
byte pointer and vw, a WROM word pointer (VSET; VMAT advances both); the trackers vmx (max |v| or
max v), vsm (a sum), varg (argmax row, counted across VMAT ops until a reset).

  f3 0  VMAT  rd=cfg rs1=out rs2=F   rows r < n of WROM int4 words at vw (k words a row):
              acc = VA[off:off+8k] . row; mode 0: out[r] = mulh(acc * S[r], F) with S[r] the RAM
              word at vp + 4r, two VE per row; mode 1: argmax of out[r] over rows (no write), one VE
              per row. cfg: n[0:7] k[7:12] off[12:20] mode[20] keep[22]. vw += n k, vp += 4 n
  f3 1  VMATR rd=cfg rs1=out rs2=s   rows r < n of int8 bytes at vp + r stride (len bytes):
              out[r] = (acc * s) >> (12 if sh12 else 0), two VE per row.
              cfg: n[0:7] len[7:13] off[13:21] stride/4[21:29] sh12[29] signed[30] keep[31]
  f3 2  VQ    rd=cfg rs1=src rs2=r   q_i = clamp(round(src[i] * r / 2^24), -127, 127) for i < n
              int32 src words; dst 0: VA[off + i], 1: RAM bytes at vp (word aligned), one VE per 2;
              dst 2: RAM bytes at vp + i stride, a merged word each (VE VE VE, like SB).
              vsm = sum q_i^2. cfg: n[0:8] dst[8:10] off[10:18] stride[18:26] keep[26]
  f3 3  VEW   rd=cfg rs1=a rs2=p     i < n int32 words a[i], two VE each, in place:
              op 0 ADD a[i] += b[i] (b: words at vp); 1 GLU a[i] = SILU(a[i]) * (b[i] >> 8) >> 4
              (b at vp, SILU: int32 table at p, a / 2048 + 256, a >> 4 above 8.0, 0 below -8.0);
              2 ROPE pairs (a[i], a[i+1]) by the int16 (cos, sin) pairs at p, (i mod hs) (hs[12:16]);
              3 EMB a[i] = (int8 at vp + i) * p >> 16. vp += 4 n (ADD / GLU) or n (EMB).
              cfg: n[0:8] op[8:11] keep[11] hs[12:16]
  f3 4  VEXP  rd=cfg rs1=s rs2=c     i < n: d = mulh(vmx - s[i], c) unsigned; VA[off + i] = T[d]
              (uint8 table at vp) if d < 256 else 0; vsm = sum. One VE per 2. cfg: n[0:8] off[8:16]
  f3 5  VGET  rd rs1=sel             rd = vmx / vsm / varg (sel 0 / 1 / 2): a value record
  f3 6  VRED  rd=cfg rs1=a           vmx = max |a[i]| over i < n int32 words, 8 a token.
              cfg: n[0:8] keep[8]
  f3 7  VSET  rs1 rs2                vp = x[rs1], vw = x[rs2] (from the register reads): NT
Streams are at most 64 VE tokens (software splits longer vectors into ops).
"""

from chipformer.rv import spec as S

OPS = ["VMAT", "VMATR", "VQ", "VEW", "VEXP", "VGET", "VRED", "VSET"]
MAX_STREAM = 64
M32 = 0xFFFFFFFF


def s32(v):
    v &= M32
    return v - (1 << 32) if v >> 31 else v


def s8(v):
    v &= 0xFF
    return v - 256 if v >> 7 else v


def mulh(a, b):
    return (s32(a) * s32(b)) >> 32


def ve(v):
    return S.tok("VE", v & 0xFFFF)


def word_toks(v):
    return [ve(v), ve(v >> 16)]


class VX:
    def __init__(self):
        self.va = [0] * 256
        self.vp = self.vw = 0
        self.vmx = self.vsm = self.varg = 0
        self.vbest = None  # (acc, row) of the argmax so far
        self.vrow = 0  # rows seen by argmax since the reset


def rd_word(m, ea):
    i = m.ram_index(ea)
    w = m.ram[i] if i is not None else m.drom_word(ea)
    if w is None:
        raise IndexError(hex(ea))
    return w


def rd_byte(m, ea):
    return rd_word(m, ea & ~3) >> 8 * (ea & 3) & 0xFF


def rd_half(m, ea):
    return rd_word(m, ea & ~3) >> 8 * (ea & 2) & 0xFFFF


def wr_word(m, ea, v):
    i = m.ram_index(ea)
    if i is None or ea & 3:
        raise IndexError(hex(ea))
    m.ram[i] = v & M32


def wr_byte(m, ea, v):
    w = rd_word(m, ea & ~3)
    sh = 8 * (ea & 3)
    wr_word(m, ea & ~3, w & ~(0xFF << sh) | (v & 0xFF) << sh)


def execute(m, ins) -> list[int]:
    """The op's record (commits it)."""
    v, x, op = m.vx, m.x, ins.op
    cfg, a, b = x[ins.rd], x[ins.rs1], x[ins.rs2]
    m.n += 1
    toks = []

    def track(val, signed=False):
        t = s32(val) if signed else abs(s32(val))
        v.vmx = max(v.vmx, t)

    if op == "VSET":
        v.vp, v.vw = a & 0xFFFFF, b & 0x7FFF
        return [S.tok("NT")]
    if op == "VGET":
        r = [v.vmx, v.vsm, v.varg][a & 3]
        if ins.rd:
            x[ins.rd] = r & M32
            return S.value_record(r)
        return [S.tok("NT")]
    if op == "VMAT":
        n, k, off, mode, keep = cfg & 127, cfg >> 7 & 31, cfg >> 12 & 255, cfg >> 20 & 1, cfg >> 22 & 1
        if not keep:
            v.vmx, v.vbest, v.vrow = 0, None, 0
        for r in range(n):
            acc = 0
            for j in range(k):
                cells = m.wrom_cells(v.vw + r * k + j)
                acc += sum(v.va[(off + 8 * j + c) & 255] * (cells[c] - 8)
                           for c in range(8))
            o = mulh(acc * s32(rd_word(m, v.vp + 4 * r)), b)
            if mode == 0:
                wr_word(m, a + 4 * r, o)
                track(o)
                toks += word_toks(o)
            else:
                if v.vbest is None or o > v.vbest[0]:
                    v.vbest = (o, v.vrow)
                v.varg = v.vbest[1]
                v.vrow += 1
                toks.append(ve(o))
        v.vw, v.vp = (v.vw + n * k) & 0x7FFF, v.vp + 4 * n
    elif op == "VMATR":
        n, ln, off, st = cfg & 127, cfg >> 7 & 63, cfg >> 13 & 255, 4 * (cfg >> 21 & 255)
        sh, signed, keep = 12 if cfg >> 29 & 1 else 0, cfg >> 30 & 1, cfg >> 31 & 1
        if not keep:
            v.vmx = -(1 << 31) if signed else 0
        for r in range(n):
            acc = sum(v.va[(off + j) & 255] * s8(rd_byte(m, v.vp + r * st + j)) for j in range(ln))
            o = s32((acc * s32(b)) >> sh)
            wr_word(m, a + 4 * r, o)
            track(o, signed)
            toks += word_toks(o)
    elif op == "VQ":
        n, dst, off, st, keep = cfg & 255, cfg >> 8 & 3, cfg >> 10 & 255, cfg >> 18 & 255, cfg >> 26 & 1
        if not keep:
            v.vsm = 0
        qs = []
        for i in range(n):
            p = s32(rd_word(m, a + 4 * i)) * s32(b)
            q = (abs(p) + (1 << 23)) >> 24
            q = max(-127, min(127, q if p >= 0 else -q))
            qs.append(q)
            v.vsm += q * q
            if dst == 0:
                v.va[(off + i) & 255] = q
            elif dst == 1:
                wr_byte(m, v.vp + i, q)
            else:
                wr_byte(m, v.vp + i * st, q)
                w = rd_word(m, (v.vp + i * st) & ~3)
                toks += [ve(w), ve(w >> 16), ve(0)]
        if dst != 2:
            toks += [ve((qs[i] & 255) | (qs[i + 1] & 255 if i + 1 < n else 0) << 8)
                     for i in range(0, n, 2)]
    elif op == "VEW":
        n, sub, keep, hs = cfg & 255, cfg >> 8 & 7, cfg >> 11 & 1, cfg >> 12 & 15
        if not keep:
            v.vmx = 0
        out = []
        if sub == 2:  # ROPE
            for i in range(0, n, 2):
                p, q = s32(rd_word(m, a + 4 * i)) >> 6, s32(rd_word(m, a + 4 * i + 4)) >> 6
                c = sx16(rd_half(m, b + 2 * (i % hs))), sx16(rd_half(m, b + 2 * (i % hs) + 2))
                out += [(p * c[0] - q * c[1]) >> 8, (p * c[1] + q * c[0]) >> 8]
        for i in range(n):
            ai = s32(rd_word(m, a + 4 * i))
            if sub == 0:
                o = ai + s32(rd_word(m, v.vp + 4 * i))
            elif sub == 1:
                if ai >= 8 << 16:
                    s = ai >> 4
                elif ai < -(8 << 16):
                    s = 0
                else:
                    s = s32(rd_word(m, b + 4 * ((ai >> 11) + 256)))
                o = (s * (s32(rd_word(m, v.vp + 4 * i)) >> 8)) >> 4
            elif sub == 2:
                o = out[i]
            else:
                o = (s8(rd_byte(m, v.vp + i)) * s32(b)) >> 16
            o = s32(o)
            wr_word(m, a + 4 * i, o)
            track(o)
            toks += word_toks(o)
        v.vp += 4 * n if sub in (0, 1) else n if sub == 3 else 0
    elif op == "VEXP":
        n, off = cfg & 255, cfg >> 8 & 255
        v.vsm, ps = 0, []
        for i in range(n):
            d = mulh(v.vmx - s32(rd_word(m, a + 4 * i)), b) & M32
            p = rd_byte(m, v.vp + d) if d < 256 else 0
            v.va[(off + i) & 255] = s8(p)
            v.vsm += s8(p)
            ps.append(p)
        toks += [ve(ps[i] | (ps[i + 1] if i + 1 < n else 0) << 8) for i in range(0, n, 2)]
    elif op == "VRED":
        n, keep = cfg & 255, cfg >> 8 & 1
        if not keep:
            v.vmx = 0
        for i in range(n):
            track(rd_word(m, a + 4 * i))
        toks += [ve(v.vmx)] * (-(-n // 8))
    if len(toks) > MAX_STREAM:
        raise ValueError(f"{op}: a {len(toks)}-token stream (max {MAX_STREAM})")
    return toks + [S.tok("NT")]


def sx16(v):
    return v - 65536 if v >> 15 else v
