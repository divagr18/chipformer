"""VX: a streaming vector unit for the RV core (reference semantics and trace records only).

Custom-2 (opcode 0x5B, R-type, funct7 = 0). A vector op is one instruction whose record streams
its results, then commits with NT. rd is read as the op's config word (the register file's RD read
port) and is not written, except by VGET. Every token carries at most 16 bits of result (a VE
token, or a VB token: the same with "this element is the tracker's new max"), so the trace costs
what has to come back down the stack; the trackers cost no tokens of their own.

Sized for the circuit: no multiply wider than 24 x 16 bits (the scalings take a 16-bit window of
a value, t = clamp16(v >> sh), times a 15-bit factor; the program picks sh and the factor), dot
products of at most 64 int8 x int4 or 32 int8 x int8 lanes a token, adds and compares of 32 bits.
`check` raises on any operand outside those widths.

State: VA, 256 signed bytes (the activation vector, read 64 at a time by VMAT and 32 by VMATR);
vp, a RAM / DROM byte pointer and vw, a WROM word pointer (VSET; VMAT advances both); the trackers
vmx (max |v| or max v, kept by VB tokens), vsm (a sum: VRED), varg (VMAT's argmax row).

shift(x, s) is x << s for s >= 0, x >> -s below; clamp16 clamps to [-32768, 32767].

  f3 0  VMAT  rd=cfg rs1=out         rows r < n of WROM int4 words at vw (k words a row):
              acc = VA[0:8k] . row; out[r] = acc * S[r] (mod 2^32), S[r] the word at vp + 4r
              (< 2^15). Mode 0: out[r] to RAM at out + 4r, max |out| tracked; mode 1: argmax.
              Tokens a row: 2 per 8 words (mode 0), 1 per 8 words (mode 1).
              cfg: n[0:7] k[7:12] mode[20] keep[22]. vw += n k, vp += 4 n
  f3 1  VMATR rd=cfg rs1=out rs2=s   rows r < n of int8 bytes at vp + r stride (len <= 32 bytes):
              out[r] = (acc * s) >> (12 if sh12 else 0), |s| < 2^17, two VE per row.
              cfg: n[0:7] len[7:13] off[13:21] stride/4[21:29] sh12[29] signed[30] keep[31]
  f3 2  VQ    rd=cfg rs1=src rs2=r   q_i = clamp(round(clamp16(src[i] >> sh) * r / 2^15), +-127),
              r < 2^15; dst 0: VA[off + i], 1: RAM bytes at vp (word aligned), one VE per 2;
              dst 2: RAM bytes at vp + i stride, a merged word each (VE VE VE, like SB).
              cfg: n[0:8] dst[8:10] off[10:18] stride[18:26] sh[26:31]
  f3 3  VEW   rd=cfg rs1=a rs2=p     i < n int32 words a[i], two VE each, in place:
              op 0 ADD  a[i] += shift(clamp16(b[i] >> sh) * p, so), b: words at vp, p < 2^15
              op 1 GLU  y = shift(clamp16(a[i] >> sh) * p, so) (a in Q16), s = SILU(y): the int32
                        table at vp, entry y / 2048 + 256 (Q12), min(y >> 4, 2^23) from 8.0, 0
                        below -8.0; a[i] = s * clamp16(b[i] >> shb) >> 8, b: words at vq
              op 2 ROPE pairs (a[i], a[i+1]): u = clamp16(a >> sh), w = clamp16(a' >> sh); by the
                        int16 (cos, sin) pairs at p, (i mod hs): shift(u c - w s, sh - 14), ...
              op 3 EMB  a[i] = (int8 at vp + i) * p >> 16, p < 2^28
              max |a[i]| tracked. vp += 4 n (ADD), n (EMB); GLU: vp is the table, vq advances.
              cfg: n[0:8] op[8:11] keep[11] hs[12:16] sh[16:21] so[21:27] (signed) shb[27:32]
  f3 4  VEXP  rd=cfg rs1=s rs2=c     i < n: e = min(vmx - s[i], 65535), d = e * c >> ksh (c < 2^16);
              VA[off + i] = T[d] (uint8 table at vp) if d < 256 else 0. One VE per 2.
              cfg: n[0:8] off[8:16] ksh[16:21]
  f3 5  VGET  rd rs1=sel             rd = vmx / vsm / varg / bit length of vmx (sel 0..3): a value
              record
  f3 6  VRED  rd=cfg rs1=a           mode 0: vmx = max |a[i]| over i < n int32 words, 8 a token;
              mode 1: vsm = sum VA[off + i]^2, 32 a token; mode 2: vsm = sum VA[off + i], one token.
              cfg: n[0:8] keep[8] mode[9:11] off[11:19]
  f3 7  VSET  rd rs1 rs2             vp = x[rs1], vw = x[rs2], vq = x[rd] (the register reads): NT
Streams are at most 64 tokens (software splits longer vectors into ops).
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


def sx16(v):
    v &= 0xFFFF
    return v - 65536 if v >> 15 else v


def clamp16(v):
    return max(-32768, min(32767, v))


def shift(x, s):
    return x << s if s >= 0 else x >> -s


def check(ok, what):
    if not ok:
        raise ValueError(f"VX operand out of range: {what}")


def ve(v, best=False):
    return S.tok("VB" if best else "VE", v & 0xFFFF)


class VX:
    def __init__(self):
        self.va = [0] * 256
        self.vp = self.vw = self.vq = 0
        self.vmx = self.vsm = self.varg = 0
        self.vbest = None  # the argmax value so far
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

    def tracked(val, signed=False):
        """Two tokens of a word result; the second is VB when it is the tracker's new max."""
        t = s32(val) if signed else abs(s32(val))
        best = t > v.vmx
        v.vmx = max(v.vmx, t)
        return [ve(val), ve(val >> 16, best)]

    if op == "VSET":
        v.vp, v.vw, v.vq = a & 0xFFFFF, b & 0x7FFF, cfg & 0xFFFFF
        return [S.tok("NT")]
    if op == "VGET":
        r = [v.vmx, v.vsm, v.varg, max(v.vmx, 0).bit_length()][a & 3]
        if ins.rd:
            x[ins.rd] = r & M32
            return S.value_record(r)
        return [S.tok("NT")]
    if op == "VMAT":
        n, k, mode, keep = cfg & 127, cfg >> 7 & 31, cfg >> 20 & 1, cfg >> 22 & 1
        chunks = -(-k // 8)
        if not keep:
            v.vmx, v.vbest, v.vrow = 0, None, 0
        for r in range(n):
            acc = 0
            for j in range(k):
                cells = m.wrom_cells(v.vw + r * k + j)
                acc += sum(v.va[8 * j + c] * (cells[c] - 8) for c in range(8))
            sr = s32(rd_word(m, v.vp + 4 * r))
            check(0 <= sr < 1 << 15, f"VMAT scale {sr}")
            o = s32(acc * sr)
            if mode == 0:
                wr_word(m, a + 4 * r, o)
                toks += [ve(0)] * (2 * (chunks - 1)) + tracked(o)
            else:
                best = v.vbest is None or o > v.vbest
                if best:
                    v.vbest, v.varg = o, v.vrow
                v.vrow += 1
                toks += [ve(0)] * (chunks - 1) + [ve(o, best)]
        v.vw, v.vp = (v.vw + n * k) & 0x7FFF, v.vp + 4 * n
    elif op == "VMATR":
        n, ln, off, st = cfg & 127, cfg >> 7 & 63, cfg >> 13 & 255, 4 * (cfg >> 21 & 255)
        sh, signed, keep = 12 if cfg >> 29 & 1 else 0, cfg >> 30 & 1, cfg >> 31 & 1
        check(ln <= 32 and abs(s32(b)) < 1 << 17, f"VMATR len {ln} scale {s32(b)}")
        if not keep:
            v.vmx = -(1 << 31) if signed else 0
        for r in range(n):
            acc = sum(v.va[(off + j) & 255] * s8(rd_byte(m, v.vp + r * st + j)) for j in range(ln))
            o = s32((acc * s32(b)) >> sh)
            wr_word(m, a + 4 * r, o)
            toks += tracked(o, signed)
    elif op == "VQ":
        n, dst, off, st, sh = cfg & 255, cfg >> 8 & 3, cfg >> 10 & 255, cfg >> 18 & 255, cfg >> 26 & 31
        check(0 <= s32(b) < 1 << 15, f"VQ factor {s32(b)}")
        qs = []
        for i in range(n):
            p = clamp16(s32(rd_word(m, a + 4 * i)) >> sh) * b
            q = (abs(p) + (1 << 14)) >> 15
            q = max(-127, min(127, q if p >= 0 else -q))
            qs.append(q)
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
        sh, so, shb = cfg >> 16 & 31, sx6(cfg >> 21), cfg >> 27 & 31
        if not keep:
            v.vmx = 0
        out = []
        if sub == 2:  # ROPE
            for i in range(0, n, 2):
                u = clamp16(s32(rd_word(m, a + 4 * i)) >> sh)
                w = clamp16(s32(rd_word(m, a + 4 * i + 4)) >> sh)
                c, s = sx16(rd_half(m, b + 2 * (i % hs))), sx16(rd_half(m, b + 2 * (i % hs) + 2))
                out += [shift(u * c - w * s, sh - 14), shift(u * s + w * c, sh - 14)]
        if sub in (0, 1):
            check(0 <= s32(b) < 1 << 15, f"VEW factor {s32(b)}")
        if sub == 3:
            check(0 <= s32(b) < 1 << 28, f"EMB factor {s32(b)}")
        for i in range(n):
            ai = s32(rd_word(m, a + 4 * i))
            if sub == 0:
                o = ai + shift(clamp16(s32(rd_word(m, v.vp + 4 * i)) >> sh) * b, so)
            elif sub == 1:
                y = shift(clamp16(ai >> sh) * b, so)
                if y >= 8 << 16:
                    s = min(y >> 4, 1 << 23)
                elif y < -(8 << 16):
                    s = 0
                else:
                    s = s32(rd_word(m, v.vp + 4 * ((y >> 11) + 256)))
                o = (s * clamp16(s32(rd_word(m, v.vq + 4 * i)) >> shb)) >> 8
            elif sub == 2:
                o = out[i]
            else:
                o = (s8(rd_byte(m, v.vp + i)) * s32(b)) >> 16
            o = s32(o)
            wr_word(m, a + 4 * i, o)
            toks += tracked(o)
        if sub == 0:
            v.vp += 4 * n
        elif sub == 1:
            v.vq += 4 * n
        elif sub == 3:
            v.vp += n
    elif op == "VEXP":
        n, off, ksh = cfg & 255, cfg >> 8 & 255, cfg >> 16 & 31
        check(0 <= s32(b) < 1 << 16, f"VEXP factor {s32(b)}")
        ps = []
        for i in range(n):
            e = min(v.vmx - s32(rd_word(m, a + 4 * i)), 65535)
            check(e >= 0, "VEXP above the max")
            d = (e * b) >> ksh
            p = rd_byte(m, v.vp + d) if d < 256 else 0
            v.va[(off + i) & 255] = s8(p)
            ps.append(p)
        toks += [ve(ps[i] | (ps[i + 1] if i + 1 < n else 0) << 8) for i in range(0, n, 2)]
    elif op == "VRED":
        n, keep, mode, off = cfg & 255, cfg >> 8 & 1, cfg >> 9 & 3, cfg >> 11 & 255
        if mode == 0:
            if not keep:
                v.vmx = 0
            for i in range(n):
                v.vmx = max(v.vmx, abs(s32(rd_word(m, a + 4 * i))))
            toks += [ve(0)] * (-(-n // 8))
        else:
            vals = [v.va[(off + i) & 255] for i in range(n)]
            v.vsm = sum(q * q for q in vals) if mode == 1 else sum(vals)
            toks += [ve(0)] * (-(-n // 32) if mode == 1 else 1)
    if len(toks) > MAX_STREAM:
        raise ValueError(f"{op}: a {len(toks)}-token stream (max {MAX_STREAM})")
    return toks + [S.tok("NT")]


def sx6(v):
    v &= 63
    return v - 64 if v >> 5 else v
