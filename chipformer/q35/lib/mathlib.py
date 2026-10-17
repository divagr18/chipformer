"""BIOS maths library: routines on the existing bignum macros, with no circuit changes.

The library is CHIP-8 code assembled after the BIOS (bcd.SRC), so the BIOS's fixed entries keep
their addresses. It uses only the BIOS's macros (BADD, BSUB, BMUL0, BMUL, BTRIM, BDIV, BADB) and
plain CHIP-8 instructions. Entries are called like the BIOS's (call_writes): the arguments'
digits at A_ADDR / B_ADDR, their lengths at LEN / LEN + 1, MODE, then PC = the entry. The entries
are a jump table right after the BIOS (OPS), so a circuit that compiles their addresses keeps
working when the routines move. Each entry ends with PRINT (00FD), printing V0 cells from I as
digits (10 prints ".", 11 "...", 14 "e").

    pow     A ^ B, B <= 255: exact up to 48 digits
    fact    A!, A <= 39: exact (up to 47 digits)
    npr     A! / (A - B)!: exact up to 48 digits; ncr: A! / (B! (A - B)!), exact while the
            running product fits MAX_DIGITS (results up to about 10^22)
    idiv    A div B; mod: A mod B; gcd; lcm (up to 48 digits)
    spf     the smallest prime factor of A (A itself if A is prime, A if A < 2), by trial
            division by 2, 3 and 6k +- 1 below 1000: complete for A < 10^6, otherwise "..."
            when A has no factor below 1000
    to_base A in base B (2-16), digits MSB first. Digits 10-15 (a-f) need the hex print
            variant (HEX_NOTE); from_base: A's digits (values 0..B-1, MSB first) in base B
    isqrt   floor(sqrt(A)); sqrt: the integer if A is a perfect square, else the integer part,
            ".", DEC decimals (truncated) and "..."
    log10, ln: of an integer A >= 1, DEC decimals (truncated) and "...", or the integer when
            exact (log10 of a power of 10, ln 1); log2 and logb (log_B A): the same, the
            integer when A = B^n
    exp10, exp: 10^x and e^x for x = A / 10^LEN[3], negative if LEN[4] = 1. 10^x for integer x
            prints exactly; otherwise 11 significant digits (truncated) and "...", as
            "d.dddddddddde<n>" from 10^11 up

A result the macros can't hold (an operand longer than MAX_DIGITS) prints "...".

Memory beyond the BIOS's 0x600-0x6E2 (P6):
    OUT      0x000-0x04F: the print buffer, and scratch before the print (below the font)
    SLOT k   bignums, digits MSB first: slots 0-3 at 0x0A0 + 0x40 k (48 digits, the length
             at +0x3F), slots 4-5 at 0x1A0 + 0x20 (k - 4) (24 digits, the length at +0x1F)
    CNT      0x690-0x69F: Briggs factor counts; TMP 0x6E8: FX33 scratch (+3: a digit count);
             VARS 0x6F0-0x6FF
Code: the jump table right after the BIOS (fixed addresses), pads (BADD / BSUB jump targets sit
below 0x400), then routines packed into 0x2AE-0x5FF and 0x700-0xFFF. All of it is about twice
the free space, so the library is built in profiles (PROFILES), each with every entry in the
jump table (an entry outside the profile prints "...").
"""

import re
from decimal import ROUND_HALF_EVEN, Decimal, getcontext

from chipformer.q35.lib import bcd
from chipformer.q35.lib.asm import assemble

P6 = 0x600
A, B, LEN, R, MODE, W = bcd.A_ADDR, bcd.B_ADDR, bcd.LEN, bcd.R_ADDR, bcd.MODE, bcd.W_ADDR
LA, LB, FA, SGN = LEN, LEN + 1, LEN + 3, LEN + 4
oA, oB, oR, oW = A - P6, B - P6, R - P6, W - P6  # offsets in page 0x600
MD, DEC = bcd.MAX_DIGITS, bcd.DEC
TMP, VARS, CNT, OUT = 0x6E8, 0x6F0, 0x690, 0x000
NDIG = TMP + 3  # v2o: the digit count
LOW_END, HIGH, TOP = bcd.A_ADDR, 0x700, 0x1000
BASES = {0: OUT, 6: P6, 7: 0x0A0, 8: 0x1A0}  # copy-routine pages: OUT, P6, slots
LO = {7: 0x3F, 8: 0x1F}  # where a slot keeps its length
PREC = 17  # Briggs: significant digits of y, and the fraction digits of the log table
LEVELS = 16  # Briggs levels k = 0..15 (factors 1 + 10^-k)
HEX_NOTE = ("to_base above 10 is exact in memory, but nibbles 10 and 11 print as '.' and "
            "'...': readable output needs a hex print variant (MODE bit 16 -> a-f)")
OP_NAMES = ("pow", "fact", "npr", "ncr", "idiv", "mod", "gcd", "lcm", "spf", "to_base",
            "from_base", "isqrt", "sqrt", "log10", "ln", "log2", "logb", "exp10", "exp")


def _table():
    getcontext().prec = 60
    q = Decimal(10) ** PREC
    logs = [((1 + Decimal(10) ** -k).log10() * q).quantize(1, ROUND_HALF_EVEN)
            for k in range(LEVELS)]
    consts = {"ln10": (Decimal(10).ln() * q).quantize(1, ROUND_HALF_EVEN),
              "le": (Decimal(1).exp().log10() * q * 10).quantize(1, ROUND_HALF_EVEN)}
    return [str(int(v)).zfill(PREC) for v in logs], {k: str(int(v)) for k, v in consts.items()}


LOGS, CONSTS = _table()  # LOGS[k] = log10(1 + 10^-k) * 10^PREC, PREC digits


# ---------------------------------------------------------------- slots and call-site helpers
def page(k: int) -> int:
    return 7 if k < 4 else 8


def off(k: int) -> int:
    return 0x40 * k if k < 4 else 0x20 * (k - 4)


def SLEN(k):  # the address of slot k's length
    return BASES[page(k)] + off(k) + LO[page(k)]


def SADDR(k):
    return BASES[page(k)] + off(k)


def LDA(k):  # slot k -> A, LA ("..." if longer than MAX_DIGITS)
    return f"ld vb, {off(k)}\ncall ld{page(k)}a"


def LDB(k):
    return f"ld vb, {off(k)}\ncall ld{page(k)}b"


def STR(k):  # the result (page 6 + VB, VD long) -> slot k
    return f"ld vc, {off(k)}\ncall st{page(k)}"


def AS(k):  # A -> slot k
    return f"ld vc, {off(k)}\ncall as{page(k)}"


def BS(k):
    return f"ld vc, {off(k)}\ncall bs{page(k)}"


def ONE(k):
    return f"ld vc, {off(k)}\ncall one{page(k)}"


def ZERO(k):
    return f"ld vc, {off(k)}\ncall zero{page(k)}"


def ISZ(k):  # VF = 1 if slot k is 0
    return f"ld vb, {off(k)}\ncall isz{page(k)}"


def MV(j, k):
    return f"ld vb, {off(j)}\nld vc, {off(k)}\ncall mv{page(j)}{page(k)}"


def PSLOT(k):
    return f"ld vb, {off(k)}\njp ps{page(k)}"


class Src:
    """The library source: blocks of code, plus copy routines and pads made on demand."""

    def __init__(self):
        self.blocks, self.cps, self.pads, self.feature = [], {}, {}, "core"

    def __call__(self, text: str):
        """Add code; a line ";;" splits it into blocks (a core block is only kept if a kept
        block uses one of its labels)."""
        block = []
        for x in text.strip("\n").splitlines():
            if x.strip() == ";;":
                self.blocks.append((self.feature, block))
                block = []
            else:
                block.append(x.strip())
        self.blocks.append((self.feature, block))

    def cp(self, sp: int, dp: int) -> str:
        """A routine copying VD cells, in chunks of 8, from page sp + VB to page dp + VC
        (pages: BASES; clobbers V0-V7, VB, VC, VD, VF, I)."""
        name = f"cp{sp:x}{dp:x}"
        self.cps[name] = (BASES[sp], BASES[dp])
        return name

    def pad(self, target: str) -> str:
        """A jump target below 0x400 for BADD / BSUB, forwarding to ``target``."""
        return self.pads.setdefault(target, f"pad_{len(self.pads)}")

    def source(self, features, swap=None) -> str:
        """The source of a profile: the blocks of these features, and what they use.
        ``swap`` maps a block's first label to a function rewriting its lines."""
        swap = swap or {}

        def sw(b):
            m = re.match(r"([A-Za-z_][\w.]*):", b[0]) if b else None
            return swap[m[1]](b) if m and m[1] in swap else b

        blocks = [sw(b) for f, b in self.blocks if f in features and f != "core"]
        pool = [sw(b) for f, b in self.blocks if f == "core"]

        def code(bs):  # the identifiers the code uses (comments left out)
            return set(re.findall(r"[A-Za-z_][\w.]*", "\n".join(
                ln.split(";")[0] for b in bs for ln in b)))

        used = code(blocks) | {"ovf"}
        while True:  # keep the core blocks the kept code uses, until nothing changes
            new = [b for b in pool if code([b]) & used and b not in blocks
                   and set(re.findall(r"^([A-Za-z_][\w.]*):", "\n".join(b), re.M)) & used]
            if not new:
                break
            blocks += new
            used |= code(new)
        text = "\n".join(ln for b in blocks for ln in b)
        for name, (s_, d) in self.cps.items():
            if re.search(rf"\b{name}\b", text):
                blocks.append([f"{name}: ld i, {s_}", "add i, vb", "ld v7, [i]", f"ld i, {d}",
                               "add i, vc", "ld [i], v7", "add vb, 8", "add vc, 8",
                               "ld v0, 8", "sub vd, v0", "se vf, 1", "ret", "se vd, 0",
                               f"jp {name}", "ret"])
        defined = set(re.findall(r"^([A-Za-z_][\w.]*):", text, re.M))
        head = ["lib_start:",
                *(f"jt_{op}: jp {op if op in defined else 'ovf'}" for op in OP_NAMES),
                *(f"{p}: jp {t}" for t, p in self.pads.items() if re.search(rf"\b{p}\b", text)),
                "pads_end:"]
        low, high, at = [], [], bcd.LABELS["zeros"] + 10 + size(head)
        for b in sorted(blocks, key=size, reverse=True):  # first fit, the biggest first
            n = size(b)
            if at + n <= LOW_END:
                low += b
                at += n
            else:
                high += b
        return "\n".join([*head, *low, "low_end:", f"org {HIGH}", *high, "lib_end:"])


def size(lines: list[str]) -> int:
    n = 0
    for raw in lines:
        line = raw.split(";")[0].strip()
        while (m := re.match(r"([A-Za-z_][\w.]*):\s*(.*)", line)):
            line = m[2].strip()
        if line:
            op = line.split()[0].upper()
            n += len(line.split(None, 1)[1].split(",")) if op == "DB" else 2
    return n


s = Src()

# ================================================================ helpers
s(f"""
ovf:    ld i, {OUT}             ; too big for the macros: print "..."
        ld v0, 11
        ld [i], v0
        ld v0, 1
        print
;;
p_zero: ld v0, 0                ; print the digit V0
p_dig:  ld i, {OUT}
        ld [i], v0
        ld v0, 1
        print
p_one:  ld v0, 1
        jp p_dig
;;
pres:   ld i, {P6}              ; print the result (page 6 + VB, VD long)
        add i, vb
        ld v0, vd
        print
;;
sq:     db {", ".join(str(q * q) for q in range(10))}
""")

# d2v: V9 = the value of the digits at page 6 + V5 (their count at page 6 + V4), V8 = 1 if
# it is above 255. Clobbers V0-V5, VF.
s.feature = "num"
s(f"""
a2v:    ld v4, {LA - P6}
        ld v5, {oA}
        jp d2v
b2v:    ld v4, {LB - P6}
        ld v5, {oB}
d2v:    ld v8, 0
        ld v9, 0
        ld i, {P6}
        add i, v4
        ld v0, [i]
        ld v3, v0
d2v_lp: ld i, {P6}
        add i, v5
        ld v0, [i]
        ld v1, v9
        add v1, v9
        se vf, 0
        ld v8, 1
        ld v9, v1
        add v9, v9
        se vf, 0
        ld v8, 1
        add v9, v9
        se vf, 0
        ld v8, 1
        add v9, v1
        se vf, 0
        ld v8, 1
        add v9, v0
        se vf, 0
        ld v8, 1
        add v5, 1
        add v3, 0xFF
        se v3, 0
        jp d2v_lp
        ret
""")

s.feature = "core"

# v2a / v2b: the digits of V3 (1-3, no leading zeros) to A / B, their count to LA / LB;
# v2o: to OUT, the count in NDIG. Clobbers V0-V2, V4-V6, VF.
s(f"""
v2a:    ld v4, {oA}
        ld v5, {LA - P6}
        jp v2d
v2b:    ld v4, {oB}
        ld v5, {LB - P6}
v2d:    call v2o
        ld i, {OUT}
        ld v2, [i]
        ld i, {P6}
        add i, v4
        ld [i], v2
        ld i, {NDIG}
        ld v0, [i]
        ld i, {P6}
        add i, v5
        ld [i], v0
        ret
v2o:    ld i, {TMP}
        ld b, v3
        ld v2, [i]
        ld v6, 3
        se v0, 0
        jp v2o_s
        ld v0, v1
        ld v1, v2
        ld v6, 2
        se v0, 0
        jp v2o_s
        ld v0, v1
        ld v6, 1
v2o_s:  ld i, {OUT}
        ld [i], v2
        ld v0, v6
        ld i, {NDIG}
        ld [i], v0
        ret
""")

# mul_ab: R = A * B, trimmed: VB = its page-6 offset, VD = length (clobbers V0, V1, V6, VA)
s(f"""
mul_ab: ld i, {LA}
        ld v1, [i]
        ld va, v0
        add va, v1
        ld v6, 0
        bmul0 v6
        bmul v6
        btrim
        ld vd, v0
        ld vb, va
        sub vb, vd
        add vb, {oR}
        ret
""")


def add_like(name: str, macro: str):
    """name: R = A + B (badd) or A - B (bsub), trimmed: VB, VD; VF = 1 (bsub: A >= B, else 0
    and R = A - B mod 10^n). Clobbers V0, V1, V6, VA."""
    after = f"{name}_t"
    s(f"""
{name}: ld i, {LA}
        ld v1, [i]
        ld va, v0
        sub v0, v1
        se vf, 1
        ld va, v1
        add va, 1
        ld v6, 0
        {macro} {s.pad(after)}
        jp {name}_n
{after}: ld v6, 1
{name}_n: btrim
        ld vd, v0
        ld vb, va
        sub vb, vd
        add vb, {oR}
        ld vf, v6
        ret
""")


add_like("add_ab", "badd")
add_like("sub_ab", "bsub")

# the result -> A / B
s(f"""
r_a:    ld i, {LA}
        ld vc, {oA}
        jp r_ab
r_b:    ld i, {LB}
        ld vc, {oB}
r_ab:   ld v0, vd
        ld [i], v0
        jp {s.cp(6, 6)}
""")

# slot helpers for pages 7 and 8 (VB = the source slot's offset, VC = the destination's)
for p in (7, 8):
    base, lo = BASES[p], BASES[p] + LO[p]
    s(f"""
ld{p}a: ld vc, {oA}
        ld v2, {LA - P6}
        jp ld{p}_c
ld{p}b: ld vc, {oB}
        ld v2, {LB - P6}
ld{p}_c: ld i, {lo}
        add i, vb
        ld v0, [i]
        ld i, {P6}
        add i, v2
        ld [i], v0
        ld v1, {MD}
        sub v1, v0
        se vf, 1
        jp ovf
        ld vd, v0
        jp {s.cp(p, 6)}
;;
st{p}:  ld i, {lo}
        add i, vc
        ld v0, vd
        ld [i], v0
        jp {s.cp(6, p)}
;;
as{p}:  ld vb, {oA}
        ld i, {LA}
        jp as{p}_c
bs{p}:  ld vb, {oB}
        ld i, {LB}
as{p}_c: ld v0, [i]
        ld vd, v0
        jp st{p}
;;
one{p}: ld v0, 1
        jp one{p}_c
zero{p}: ld v0, 0
one{p}_c: ld i, {base}
        add i, vc
        ld [i], v0
        ld v0, 1
        ld i, {lo}
        add i, vc
        ld [i], v0
        ret
;;
isz{p}: ld i, {lo}
        add i, vb
        ld v0, [i]
        ld vf, 0
        se v0, 1
        ret
        ld i, {base}
        add i, vb
        ld v0, [i]
        se v0, 0
        ret
        ld vf, 1
        ret
;;
ps{p}:  ld i, {lo}
        add i, vb
        ld v0, [i]
        ld i, {base}
        add i, vb
        print
""")
for p, q in ((7, 7), (8, 7)):
    s(f"""
mv{p}{q}: ld i, {BASES[p] + LO[p]}
        add i, vb
        ld v0, [i]
        ld vd, v0
        ld i, {BASES[q] + LO[q]}
        add i, vc
        ld [i], v0
        jp {s.cp(p, q)}
""")

# ================================================================ dvm: divmod B
# Long division on BDIV / BADB, stopped at the last integer digit: the dividend goes to W
# padded on the left to LAe = max(la, 10, LB) digits, and LEN[0] = LAe - 10, so BDIV's flag
# 3 ("DEC decimals done") fires at t = LAe. Flag 2 (an exact window) at t < LAe continues.
# Digits for t > LEN[0] land at P[t + 1]: a 10-cell move closes that gap.
# dvm_r: the dividend at page 6 + VB, VD long (A, or a result in R); dvm_s: in the slot at
# page 7 + VB. Out: the quotient at page 6 + VB (VD long, trimmed, in R); the remainder at
# page 6 + VARS[0] (VARS[1] long, trimmed, in W). LEN[0] is overwritten. Keeps V8, V9, VE.
# "..." if the dividend is longer than MAX_DIGITS or B is 0.
s(f"""
dvm_s:  ld i, {BASES[7] + LO[7]}
        add i, vb
        ld v0, [i]
        ld vd, v0
        ld v0, 1
        jp dvm_c
dvm_r:  ld v0, 0
dvm_c:  ld i, {VARS + 3}
        ld [i], v0
        ld i, {B}
        ld v0, [i]
        se v0, 0
        jp dvm_go
        jp ovf
dvm_go: ld i, zeros
        ld v7, [i]
        ld i, {W}
        ld [i], v7
        ld i, {W + 8}
        ld [i], v7
        ld i, {W + 16}
        ld [i], v7
        ld v3, vd               ; la
        ld v1, {MD}
        sub v1, v3
        se vf, 1
        jp ovf
        ld va, v3               ; LAe = max(la, 10, LB)
        ld v1, 10
        ld v2, va
        sub v2, v1
        se vf, 1
        ld va, v1
        ld i, {LB}
        ld v0, [i]
        ld v2, va
        sub v2, v0
        se vf, 1
        ld va, v0
        ld vc, va
        sub vc, v3
        add vc, {oW + 1}        ; the dividend at W + 1 + (LAe - la)
        ld i, {VARS + 3}
        ld v0, [i]
        se v0, 0
        jp dvm_cs
        call {s.cp(6, 6)}
        jp dvm_cd
dvm_cs: call {s.cp(7, 6)}
dvm_cd: ld v0, va
        add v0, 0xF6            ; LEN[0] = LAe - 10
        ld vc, v0
        ld i, {LA}
        ld [i], v0
        ld i, {LB}
        ld v0, [i]
        ld v5, v0               ; t = LB
dvm_lp: bdiv v5
        se vf, 1
        jp dvm_ck
        badb v5
        se vf, 0
        jp dvm_ck
        jp dvm_lp
dvm_ck: se vf, 2
        jp dvm_ok
        se v5, va
        jp dvm_nx
        jp dvm_ok
dvm_nx: add v5, 1
        jp dvm_lp
dvm_ok: ld i, {R + 2}           ; close the gap: P[la' + 2 ..] -> P[la' + 1 ..]
        add i, vc
        ld v7, [i]
        ld i, {R + 1}
        add i, vc
        ld [i], v7
        ld i, {R + 10}
        add i, vc
        ld v1, [i]
        ld i, {R + 9}
        add i, vc
        ld [i], v1
        ld i, {LB}
        ld v0, [i]
        ld vd, v0               ; the remainder: W[LAe - LB + 1 ..], LB digits, trimmed
        ld vb, va
        sub vb, vd
        add vb, {oW + 1}
dvm_rt: ld v0, 1
        sub v0, vd
        se vf, 0
        jp dvm_re
        ld i, {P6}
        add i, vb
        ld v0, [i]
        se v0, 0
        jp dvm_re
        add vb, 1
        add vd, 0xFF
        jp dvm_rt
dvm_re: ld v0, vb
        ld v1, vd
        ld i, {VARS}
        ld [i], v1
        ld i, {LB}
        ld v0, [i]
        ld v1, v0               ; the quotient: P[LB .. LAe], trimmed
        ld v0, va
        sub v0, v1
        add v0, 1
        ld i, {R}
        add i, v1
        btrim
        ld vd, v0
        ld vb, va
        add vb, {oR + 1}
        sub vb, vd
        ret
a_b:    ld i, {LA}              ; A -> B, LB = LA
        ld v0, [i]
        ld vb, {oA}
        ld vc, {oB}
        jp ab_c
b_a:    ld i, {LB}              ; B -> A, LA = LB
        ld v0, [i]
        ld vb, {oB}
        ld vc, {oA}
ab_c:   ld vd, v0
        ld i, {LA}
        se vc, {oA}
        ld i, {LB}
        ld [i], v0
        jp {s.cp(6, 6)}
""")

s.feature = "num"
s(f"""
w_b:    ld i, {VARS}            ; the remainder -> B, LB
        ld v1, [i]
        ld vb, v0
        ld vd, v1
        ld v0, v1
        ld i, {LB}
        ld [i], v0
        ld vc, {oB}
        jp {s.cp(6, 6)}
dvm_a:  ld i, {LA}              ; A divmod B
        ld v0, [i]
        ld vd, v0
        ld vb, {oA}
        jp dvm_r
a_one:  ld v0, 1                ; A = 1
        ld i, {A}
        ld [i], v0
        ld i, {LA}
        ld [i], v0
        ret
pr_a:   ld i, {LA}              ; print A
        ld v0, [i]
        ld i, {A}
        print
""")
s.feature = "core"

# ================================================================ pow
# pow_core: slot 1 = slot 0 ^ VE (VE <= 255) by left-to-right squaring, the running value in
# A; V8 = 1 if an operand would pass MAX_DIGITS (the last square or product may reach 48
# digits)
s(f"""
pow_core: ld v8, 0
        se ve, 0
        jp pc_nz
        {ONE(1)}
        ret
pc_nz:  {LDA(0)}
        ld v9, 8
pc_top: shl ve, ve              ; the leading 1 is res = base
        add v9, 0xFF
        se vf, 1
        jp pc_top
        se v9, 0
        jp pc_st
        {AS(1)}
        ret
pc_st:  add v9, 0xFF
        call a_b                ; res^2
        call mul_ab
        shl ve, ve
        se vf, 1
        jp pc_nb
        call pc_fit             ; res * base
        se v8, 0
        ret
        {LDB(0)}
        call mul_ab
pc_nb:  se v9, 0
        jp pc_go
        {STR(1)}
        ret
pc_go:  call pc_fit
        se v8, 0
        ret
        jp pc_st
pc_fit: ld v0, {MD}             ; the result -> A if it fits, else V8 = 1
        sub v0, vd
        se vf, 1
        jp pc_ov
        jp r_a
pc_ov:  ld v8, 1
        ret
""")

s.feature = "num"
s(f"""
pow:    call b2v
        se v8, 0
        jp ovf
        ld ve, v9
        {AS(0)}
        call pow_core
        se v8, 0
        jp ovf
        {PSLOT(1)}
""")

# ================================================================ fact, npr, ncr
# prod: A = the product of the factors V8 .. VE (hi down to lo); when the next product would
# pass MAX_DIGITS, A is parked in slot 2 and a second product starts; the result is their
# product (up to 48 digits)
s(f"""
prod:   call a_one
        ld i, {VARS + 5}
        ld v0, 0
        ld [i], v0              ; nothing parked
pr_lp:  ld v1, v8
        ld v0, ve
        subn v1, v0             ; VF = hi >= lo
        se vf, 1
        jp pr_end
        ld v3, ve
        call v2b
        call mul_ab
        ld v0, {MD}
        sub v0, vd              ; VF = the product fits MAX_DIGITS
        se vf, 1
        jp pr_park
        call r_a
pr_nx:  add ve, 0xFF
        jp pr_lp
pr_park: ld i, {VARS + 5}       ; park A in slot 2, A = the factor
        ld v0, [i]
        se v0, 0
        jp ovf
        ld v0, 1
        ld [i], v0
        {AS(2)}
        call b_a
        jp pr_nx
pr_end: ld i, {VARS + 5}
        ld v0, [i]
        se v0, 1
        jp pr_a
        {LDB(2)}
        call mul_ab
        jp pres
fact:   call a2v
        se v8, 0
        jp ovf
        ld v0, 39
        sub v0, v9
        se vf, 1
        jp ovf
        ld ve, v9
        ld v8, 2
        jp prod
nr_in:  call b2v                ; VE = r, V9 = n, V0 = n - r (or print 0 if r > n)
        se v8, 0
        jp ovf
        ld ve, v9
        call a2v
        se v8, 0
        jp ovf
        ld v0, v9
        sub v0, ve
        se vf, 1
        jp p_zero
        ret
npr:    call nr_in
        ld v8, v0
        add v8, 1
        ld ve, v9
        jp prod
ncr:    call nr_in              ; A = A (n - r' + i) / i for i = 1 .. r' = min(r, n - r)
        ld v1, v0
        sub v1, ve              ; VF = n - r >= r
        se vf, 1
        ld ve, v0
        ld v8, v9
        sub v8, ve              ; n - r'
        ld v9, 1
        call a_one
nc_lp:  ld v0, ve
        sub v0, v9              ; VF = r' >= i
        se vf, 1
        jp pr_a
        ld v3, v8
        add v3, v9
        call v2b
        call mul_ab
        ld v3, v9
        call v2b
        call dvm_r
        call r_a
        add v9, 1
        jp nc_lp
""")

# ================================================================ division family
# gcd_c: A = gcd(A, B) by Euclid in place: (A, B) = (B, A mod B) until B = 0
s(f"""
idiv:   call dvm_a
        jp pres
mod:    call dvm_a
        ld i, {VARS}
        ld v1, [i]
        ld vb, v0
        ld vd, v1
        jp pres
gcd_c:  ld i, {LB}
        ld v0, [i]
        se v0, 1
        jp gc_st
        ld i, {B}
        ld v0, [i]
        se v0, 0
        jp gc_st
        ret
gc_st:  call dvm_a
        call b_a
        call w_b
        jp gcd_c
gcd:    call gcd_c
        jp pr_a
lcm:    {AS(4)}
        {BS(5)}
        call gcd_c
        ld i, {A}
        ld v0, [i]
        se v0, 0
        jp lc_nz
        jp p_zero
lc_nz:  call a_b                ; a / gcd * b
        {LDA(4)}
        call dvm_a
        call r_a
        {LDB(5)}
        call mul_ab
        jp pres
""")

# ================================================================ spf
# trial division by d = 2, 3, 5, 7, 11, 13, ... (steps 1, 2, then 2 and 4 in turn) while
# d < 1000; d is kept as digits in B (bump adds V9 to it). A quotient shorter than d means
# d^2 > A: A is prime.
s(f"""
spf:    call a2v
        se v8, 0
        jp sp_go
        ld v0, 1
        sub v0, v9
        se vf, 1
        jp sp_go
        ld v0, v9
        jp p_dig
sp_go:  ld i, {LA}
        ld v0, [i]
        ld i, {VARS + 4}
        ld [i], v0              ; A's length (dvm overwrites LEN[0])
        ld v0, 2
        ld i, {B}
        ld [i], v0
        ld v0, 1
        ld i, {LB}
        ld [i], v0
        ld ve, 0                ; the step state
sp_lp:  ld i, {VARS + 4}
        ld v0, [i]
        ld vd, v0
        ld vb, {oA}
        call dvm_r
        ld i, {VARS}
        ld v1, [i]
        se v1, 1
        jp sp_nz
        ld i, {P6}
        add i, v0
        ld v0, [i]
        se v0, 0
        jp sp_nz
        ld i, {LB}              ; d divides A: print d
        ld v0, [i]
        ld i, {B}
        print
sp_nz:  ld i, {LB}
        ld v0, [i]
        subn v0, vd             ; VF = quotient length >= d's
        se vf, 1
        jp sp_prime
        ld v9, 1                ; the next d
        se ve, 0
        jp sp_s1
        ld ve, 1
        jp sp_bp
sp_s1:  ld v9, 2
        se ve, 1
        jp sp_s2
        ld ve, 2
        jp sp_bp
sp_s2:  se ve, 2
        jp sp_s3
        ld ve, 3
        jp sp_bp
sp_s3:  ld v9, 4
        ld ve, 2
sp_bp:  call bump
        ld i, {LB}
        ld v0, [i]
        se v0, 4
        jp sp_lp
        ld i, {VARS + 4}        ; d reached 1000: prime if A < 10^6
        ld v0, [i]
        ld v1, 6
        sub v1, v0
        se vf, 1
        jp ovf
sp_prime: ld i, {VARS + 4}
        ld v0, [i]
        ld i, {A}
        print
bump:   ld i, {LB}              ; B += V9 (a small number), growing B by a digit on a carry
        ld v0, [i]
        ld v4, v0
bp_lp:  add v4, 0xFF
        ld i, {B}
        add i, v4
        ld v0, [i]
        add v0, v9
        ld v9, 0
        ld v1, v0
        ld v2, 10
        sub v1, v2
        se vf, 1
        jp bp_st
        ld v0, v1
        ld v9, 1
bp_st:  ld [i], v0
        se v9, 1
        ret
        se v4, 0
        jp bp_lp
        ld i, {LB}              ; a carry out of B[0]: shift right, B[0] = 1
        ld v0, [i]
        ld v4, v0
bp_sh:  add v4, 0xFF
        ld i, {B}
        add i, v4
        ld v0, [i]
        ld i, {B + 1}
        add i, v4
        ld [i], v0
        se v4, 0
        jp bp_sh
        ld v0, 1
        ld i, {B}
        ld [i], v0
        ld i, {LB}
        ld v0, [i]
        add v0, 1
        ld [i], v0
        ret
""")

# ================================================================ base conversion
# kpow: for k = V9, K = k^m is the largest power <= 255: V3 = K, VARS[7] = m, and
# VARS[8 + j] = k^j
s(f"""
kpow:   ld v3, 1
        ld v2, 0
kp_lp:  ld i, {VARS + 8}
        add i, v2
        ld v0, v3
        ld [i], v0
        add v2, 1
        ld v0, 0
        ld v1, v9
kp_mul: add v0, v3
        se vf, 0
        jp kp_end
        add v1, 0xFF
        se v1, 0
        jp kp_mul
        ld v3, v0
        jp kp_lp
kp_end: add v2, 0xFF
        ld v0, v2
        ld i, {VARS + 7}
        ld [i], v0
        ret
k_in:   call b2v                ; V9 = k in 2..16, else "..."
        se v8, 0
        jp ovf
        ld v0, 1
        sub v0, v9
        se vf, 0
        jp ovf
        ld v0, 16
        sub v0, v9
        se vf, 1
        jp ovf
        jp kpow
""")
# to_base: chunks of m digits by dvm with K = k^m (each quotient is the next dividend, in
# place in R), written right to left ending at OUT[79]
s(f"""
to_base: call k_in
        call v2b                ; B = K
        ld i, {LA}
        ld v0, [i]
        ld vd, v0
        ld vb, {oA}
        ld ve, 79
tb_lp:  call dvm_r
        ld i, {VARS}            ; r = the remainder's value (<= 254)
        ld v1, [i]
        ld v2, v0
        ld v3, v1
        ld v4, 0
tb_rv:  ld i, {P6}
        add i, v2
        ld v0, [i]
        ld v1, v4
        add v4, v4
        add v4, v4
        add v4, v1
        add v4, v4
        add v4, v0
        add v2, 1
        add v3, 0xFF
        se v3, 0
        jp tb_rv
        ld i, {VARS + 7}
        ld v0, [i]
        ld v2, v0               ; j = m - 1 .. 0: digit = how many k^j fit
        ld v5, ve
        sub v5, v2
        add v5, 1
tb_dg:  add v2, 0xFF
        ld i, {VARS + 8}
        add i, v2
        ld v0, [i]
        ld v1, v0
        ld v6, 0
tb_sb:  ld v0, v4
        sub v0, v1
        se vf, 1
        jp tb_put
        ld v4, v0
        add v6, 1
        jp tb_sb
tb_put: ld i, {OUT}
        add i, v5
        ld v0, v6
        ld [i], v0
        add v5, 1
        se v2, 0
        jp tb_dg
        ld i, {VARS + 7}
        ld v0, [i]
        sub ve, v0
        se vd, 1                ; the quotient (VB, VD) is 0: done
        jp tb_lp
        ld i, {P6}
        add i, vb
        ld v0, [i]
        se v0, 0
        jp tb_lp
        add ve, 1               ; skip leading zeros, keeping one digit
        ld v1, 79
tb_tz:  se ve, v1
        jp tb_t2
        jp tb_pr
tb_t2:  ld i, {OUT}
        add i, ve
        ld v0, [i]
        se v0, 0
        jp tb_pr
        add ve, 1
        jp tb_tz
tb_pr:  ld i, {OUT}
        add i, ve
        ld v0, 80
        sub v0, ve
        print
""")

# from_base: N = N * K + v over chunks of m digits (v by 8-bit Horner); the digits are kept in
# slot 4 (A is reused), N in slot 1
s(f"""
from_base: {AS(4)}
        call k_in
        ld i, {VARS + 6}
        ld v0, v3
        ld [i], v0              ; VARS[6] = K
        {ZERO(1)}
        ld ve, 0
        ld i, {SLEN(4)}         ; the first chunk: LA mod m digits (m if that is 0)
        ld v0, [i]
        ld v2, v0
        ld i, {VARS + 7}
        ld v0, [i]
        ld v1, v0
fb_md:  ld v0, v2
        sub v0, v1
        se vf, 1
        jp fb_m1
        ld v2, v0
        jp fb_md
fb_m1:  se v2, 0
        jp fb_c
        ld v2, v1
fb_c:   ld v8, v2
        ld v4, 0
fb_hv:  ld i, {SADDR(4)}
        add i, ve
        ld v0, [i]
        ld v5, v0
        ld v0, 0
        ld v1, v9
fb_hm:  add v0, v4
        add v1, 0xFF
        se v1, 0
        jp fb_hm
        add v0, v5
        ld v4, v0
        add ve, 1
        add v8, 0xFF
        se v8, 0
        jp fb_hv
        ld i, {VARS + 5}
        ld v0, v4
        ld [i], v0
        {ISZ(1)}
        se vf, 1
        jp fb_big
        ld v3, v4
        call v2a
        {AS(1)}
        jp fb_nx
fb_big: {LDA(1)}
        ld i, {VARS + 6}
        ld v0, [i]
        ld v3, v0
        call v2b
        call mul_ab
        call r_a
        ld i, {VARS + 5}
        ld v0, [i]
        ld v3, v0
        call v2b
        call add_ab
        {STR(1)}
fb_nx:  ld i, {SLEN(4)}
        ld v0, [i]
        se v0, ve
        jp fb_more
        {PSLOT(1)}
fb_more: ld i, {VARS + 7}
        ld v0, [i]
        ld v2, v0
        jp fb_c
""")

s.feature = "root"
# ================================================================ sqrt, isqrt
# Digit by digit, in place on W like long division: the radicand's pairs from W[SQW], the
# divisor D = 20 p in B (p = the root so far). Each digit: BDIV / BADB give q = min(9,
# floor(Rm / D)) with the window Rm - q D; then the window -= q^2, and while that is negative
# (10's complement: no carry out of the window's top) the window += D + 2q - 1 and q -= 1.
# Then D = 10 D + 20 q (append a 0, add 2q at the tens; D is a multiple of 20, so it never
# carries out). LEN[0] = t - 10 makes BDIV stop after this digit (flag 2 / 3). W holds
# indices 0..34: before t passes 34 the window shifts left (sq_shift). Registers: VE = t,
# VD = LD, VC = digits done, VB = digits wanted, VA = integer digits, V8 = q.
SQW = 8  # W index of the first radicand digit (the pad keeps t >= 10 from the first BDIV on)
s(f"""
sqrt:   ld v0, 1
        jp sq_go
isqrt:  ld v0, 0
sq_go:  ld i, {VARS + 4}
        ld [i], v0
        ld i, {LA}
        ld v0, [i]
        se v0, 1
        jp sq_nz
        ld i, {A}
        ld v0, [i]
        se v0, 0
        jp sq_nz
        jp p_zero
sq_nz:  ld i, zeros
        ld v7, [i]
        ld i, {W}
        ld [i], v7
        ld i, {W + 8}
        ld [i], v7
        ld i, {W + 16}
        ld [i], v7
        ld i, {W + 24}
        ld [i], v7
        ld i, {W + 32}
        ld [i], v7
        ld i, {LA}
        ld v0, [i]
        ld vd, v0
        ld v1, v0
        ld v2, 1
        and v1, v2              ; odd: a 0 first
        ld vc, v1
        add vc, {oW + SQW}
        ld va, v0
        add va, v1
        shr va, va              ; the integer digits (pairs)
        ld vb, {oA}
        call {s.cp(6, 6)}
        ld i, zeros             ; the copy's tail
        ld v7, [i]
        ld i, {W + SQW}
        add i, va
        add i, va
        ld [i], v7
        ld vb, va
        ld i, {VARS + 4}
        ld v0, [i]
        se v0, 0
        add vb, {DEC}           ; sqrt: DEC more digits
        ld i, {W + SQW}         ; the first digit: q0^2 <= x = the first pair
        ld v1, [i]
        ld v2, v0
        add v2, v2
        add v2, v2
        add v2, v0
        add v2, v2
        add v2, v1
        ld v8, 9
sq_q0:  ld i, sq
        add i, v8
        ld v0, [i]
        ld v1, v2
        sub v1, v0
        se vf, 0
        jp sq_q0k
        add v8, 0xFF
        jp sq_q0
sq_q0k: ld i, {TMP}
        ld b, v1
        ld v2, [i]
        ld v0, v1
        ld v1, v2
        ld i, {W + SQW}
        ld [i], v1
        ld v3, v8               ; D = 20 q0
        add v3, v8
        ld i, {TMP}
        ld b, v3
        ld v2, [i]
        se v1, 0
        jp sq_d3
        ld v0, v2
        ld v1, 0
        ld i, {B}
        ld [i], v1
        ld vd, 2
        jp sq_d
sq_d3:  ld v0, v1
        ld v1, v2
        ld v2, 0
        ld i, {B}
        ld [i], v2
        ld vd, 3
sq_d:   ld v0, vd
        ld i, {LB}
        ld [i], v0
        ld v0, v8
        ld i, {OUT}
        ld [i], v0
        ld vc, 1
        ld ve, {SQW + 1}
sq_lp:  se vc, vb
        jp sq_more
        jp sq_fin
sq_more: se vc, va
        jp sq_go2
        call sq_isz             ; all the integer digits: a perfect square?
        se vf, 1
        jp sq_dot
        ld i, {OUT}
        ld v0, va
        print
sq_dot: ld i, {OUT}
        add i, va
        ld v0, 10
        ld [i], v0
sq_go2: ld v0, 32
        sub v0, ve
        se vf, 0
        jp sq_t
        call sq_shift
sq_t:   add ve, 2
        ld v0, ve
        add v0, 0xF6
        ld i, {LA}
        ld [i], v0
sq_bd:  bdiv ve
        se vf, 1
        jp sq_bq
        badb ve
        se vf, 0
        jp sq_bq
        jp sq_bd
sq_bq:  ld i, {R + 1}
        add i, ve
        ld v0, [i]
        ld v8, v0
        se v8, 0
        jp sq_sub
        jp sq_app
sq_sub: ld v7, ve               ; the window's top
        sub v7, vd
        ld i, sq
        add i, v8
        ld v0, [i]
        ld i, {TMP}
        ld b, v0
        ld v2, [i]              ; v1 v2 = q^2's tens, ones
        ld v4, ve
        ld v5, v2
        ld v6, 0
qs_lp:  add v5, v6
        ld v6, 0
        ld i, {W}
        add i, v4
        ld v0, [i]
        sub v0, v5
        se vf, 0
        jp qs_ok
        add v0, 10
        ld v6, 1
qs_ok:  ld [i], v0
        se v4, v7
        jp qs_nx
        jp qs_end
qs_nx:  add v4, 0xFF
        ld v5, v1
        ld v1, 0
        se v5, 0
        jp qs_lp
        se v6, 0
        jp qs_lp
qs_end: se v6, 1
        jp sq_app
        call sq_fix
sq_app: ld v1, vc               ; digit q -> OUT[k + (k >= integer digits)]
        sub v1, va
        ld v0, vc
        add v0, vf
        ld i, {OUT}
        add i, v0
        ld v0, v8
        ld [i], v0
        ld i, {B}               ; D = 10 D + 20 q
        add i, vd
        ld v0, 0
        ld [i], v0
        add vd, 1
        ld v0, vd
        ld i, {LB}
        ld [i], v0
        ld v5, v8
        add v5, v8
        ld v4, vd
        add v4, 0xFE
        ld v2, 10
dl_lp:  se v5, 0
        jp dl_go
        jp dl_end
dl_go:  ld i, {B}
        add i, v4
        ld v0, [i]
        add v0, v5
        ld v5, 0
dl_c:   ld v1, v0
        sub v1, v2
        se vf, 1
        jp dl_st
        ld v0, v1
        add v5, 1
        jp dl_c
dl_st:  ld [i], v0
        add v4, 0xFF
        jp dl_lp
dl_end: add vc, 1
        jp sq_lp
sq_fin: ld i, {VARS + 4}
        ld v0, [i]
        se v0, 0
        jp sq_fd
        ld i, {OUT}
        ld v0, va
        print
sq_fd:  ld v0, va
        add v0, {DEC + 1}
        ld i, {OUT}
        add i, v0
        ld v0, 11
        ld [i], v0
        ld v0, va
        add v0, {DEC + 2}
        ld i, {OUT}
        print
sq_fix: ld v5, v8               ; window += D + 2q - 1, q -= 1, until it carries out
        add v5, v8
        add v5, 0xFF
        ld i, {TMP}
        ld b, v5
        ld v2, [i]
        ld v3, v2
        ld v2, 10
        ld v4, ve
        ld v9, vd
        add v9, 0xFF
        ld v6, 0
fx_lp:  ld i, {B}
        add i, v9
        ld v0, [i]
        ld v5, v0
        ld i, {W}
        add i, v4
        ld v0, [i]
        add v0, v5
        add v0, v3
        add v0, v6
        ld v6, 0
fx_c:   ld v5, v0
        sub v5, v2
        se vf, 1
        jp fx_st
        ld v0, v5
        add v6, 1
        jp fx_c
fx_st:  ld [i], v0
        ld v3, v1
        ld v1, 0
        add v4, 0xFF
        add v9, 0xFF
        se v4, v7
        jp fx_lp
        ld i, {W}
        add i, v4
        ld v0, [i]
        add v0, v6
        ld v5, v0
        sub v5, v2
        se vf, 1
        jp fx_neg
        ld v0, v5
        ld [i], v0
        add v8, 0xFF
        ret
fx_neg: ld [i], v0
        add v8, 0xFF
        jp sq_fix
sq_isz: ld v4, ve               ; VF = 1 if the window is all zero
        sub v4, vd
iz_lp:  ld i, {W}
        add i, v4
        ld v0, [i]
        ld vf, 0
        se v0, 0
        ret
        se v4, ve
        jp iz_nx
        ld vf, 1
        ret
iz_nx:  add v4, 1
        jp iz_lp
sq_shift: ld v4, ve             ; S = min(t - LD - 1, t - 12): W[S ..] -> W[0 ..], t -= S
        sub v4, vd
        add v4, 0xFF
        ld v5, ve
        add v5, 0xF4
        ld v0, v4
        sub v0, v5
        se vf, 1
        jp sh_s
        ld v4, v5
sh_s:   ld v9, v4
        ld v0, vb
        ld v1, vc
        ld v2, vd
        ld i, {VARS + 8}
        ld [i], v2
        ld vb, v9
        add vb, {oW}
        ld vc, {oW}
        ld vd, 35
        sub vd, v9
        call {s.cp(6, 6)}
        ld v0, 0
        ld v1, 0
        ld v2, 0
        ld v3, 0
        ld v6, 35
        sub v6, v9
        ld v5, v9
zs_lp:  ld i, {W}
        add i, v6
        ld [i], v3
        add v6, 4
        ld v4, 4
        sub v5, v4
        se vf, 1
        jp zs_end
        se v5, 0
        jp zs_lp
zs_end: ld i, {VARS + 8}
        ld v2, [i]
        ld vb, v0
        ld vc, v1
        ld vd, v2
        sub ve, v9
        ret
""")

s.feature = "log"
# ================================================================ logs
# lg_core: log10 of A (an integer >= 1) by Briggs pseudo-division. y = A's first PREC digits
# read as A[0].A[1..] in [1, 10); for k = 0 .. LEVELS-1: while y (1 + 10^-k) < 10:
# y += floor(y / 10^k) (a BADD of y and y's first PREC - k digits), CNT[k] += 1. Then
# log10(y) = 1 - sum CNT[k] log10(1 + 10^-k) (within about 10^-15). Out: VARS[12] = LA - 1
# (the integer part), VARS[13] = 1 if A is a power of 10, slot 2 = the PREC fraction digits
# (leading zeros kept), A saved in slot 4. Uses slots 1, 2, 4 and OUT (scratch).
s(f"""
y_ab:   ld i, {R + 1}           ; A = B = R[1 .. PREC + 1)
        ld v7, [i]
        ld i, {A}
        ld [i], v7
        ld i, {B}
        ld [i], v7
        ld i, {R + 9}
        ld v7, [i]
        ld i, {A + 8}
        ld [i], v7
        ld i, {B + 8}
        ld [i], v7
        ld i, {R + 17}
        ld v0, [i]
        ld i, {A + 16}
        ld [i], v0
        ld i, {B + 16}
        ld [i], v0
        ret
cnt0:   ld i, zeros             ; CNT[0 .. 16) = 0
        ld v7, [i]
        ld i, {CNT}
        ld [i], v7
        ld i, {CNT + 8}
        ld [i], v7
        ret
zab:    ld i, zeros             ; A[0 .. 24) = B[0 .. 24) = 0
        ld v7, [i]
        ld i, {A}
        ld [i], v7
        ld i, {A + 8}
        ld [i], v7
        ld i, {A + 16}
        ld [i], v7
        ld i, {B}
        ld [i], v7
        ld i, {B + 8}
        ld [i], v7
        ld i, {B + 16}
        ld [i], v7
        ret
lg_lk:  ld v8, 8                ; A = 0 and L_k (PREC digits at lg_t + V9)
        ld i, lg_t
        add i, v9
        ld v7, [i]
        ld i, {A + 1}
        ld [i], v7
        ld i, lg_t
        add i, v9
        add i, v8
        ld v7, [i]
        ld i, {A + 9}
        ld [i], v7
        ld i, lg_t
        add i, v9
        add i, v8
        add i, v8
        ld v0, [i]
        ld i, {A + 17}
        ld [i], v0
        ld v0, 0
        ld i, {A}
        ld [i], v0
        ld v0, {PREC + 1}
        ld i, {LA}
        ld [i], v0
        ret
lg_bk:  ld v8, 8                ; B = L_k (PREC digits at lg_t + V9)
        ld i, lg_t
        add i, v9
        ld v7, [i]
        ld i, {B}
        ld [i], v7
        ld i, lg_t
        add i, v9
        add i, v8
        ld v7, [i]
        ld i, {B + 8}
        ld [i], v7
        ld i, lg_t
        add i, v9
        add i, v8
        add i, v8
        ld v0, [i]
        ld i, {B + 16}
        ld [i], v0
        ld v0, {PREC}
        ld i, {LB}
        ld [i], v0
        ret
lg_core: ld i, {LA}             ; log 0: "..."
        ld v0, [i]
        se v0, 1
        jp lg_nz
        ld i, {A}
        ld v0, [i]
        se v0, 0
        jp lg_nz
        jp ovf
lg_nz:  {AS(4)}
        ld i, {LA}
        ld v0, [i]
        add v0, 0xFF
        ld i, {VARS + 12}
        ld [i], v0
        ld v8, 1                ; a power of 10?
        ld i, {A}
        ld v0, [i]
        se v0, 1
        ld v8, 0
        ld ve, 1
lg_p10: ld i, {LA}
        ld v0, [i]
        se v0, ve
        jp lg_p1
        jp lg_p2
lg_p1:  ld i, {A}
        add i, ve
        ld v0, [i]
        se v0, 0
        ld v8, 0
        add ve, 1
        jp lg_p10
lg_p2:  ld v0, v8
        ld i, {VARS + 13}
        ld [i], v0
        ld i, {LA}              ; y = A's first min(LA, PREC) digits, zero padded (in OUT)
        ld v0, [i]
        ld vd, v0
        ld v1, {PREC}
        ld v2, v1
        sub v2, v0
        se vf, 1
        ld vd, v1
        ld v9, vd
        ld vb, {oA}
        ld vc, 0
        call {s.cp(6, 0)}
        ld i, zeros
        ld v7, [i]
        ld i, {OUT}
        add i, v9
        ld [i], v7
        ld i, {OUT + 8}
        add i, v9
        ld [i], v7
        call zab
        ld vd, {PREC}
        ld vb, 0
        ld vc, {oA}
        call {s.cp(0, 6)}
        ld vd, {PREC}
        ld vb, 0
        ld vc, {oB}
        call {s.cp(0, 6)}
        ld v0, {PREC}
        ld v1, {PREC}
        ld i, {LA}
        ld [i], v1
        call cnt0
        ld ve, 0
lg_lp:  badd {s.pad("lg_ck")}
lg_ck:  ld i, {R}
        ld v0, [i]
        se v0, 0
        jp lg_rej
        call y_ab
        ld i, {CNT}
        add i, ve
        ld v0, [i]
        add v0, 1
        ld [i], v0
        jp lg_lp
lg_rej: add ve, 1
        se ve, {LEVELS}
        jp lg_nb
        jp lg_sum
lg_nb:  ld v0, {PREC}
        sub v0, ve
        ld i, {LB}
        ld [i], v0
        jp lg_lp
lg_sum: ld i, zeros             ; S = sum CNT[k] L_k, summed in R by BMUL rows:
        ld v7, [i]              ; R[1 ..] += (0 L_k) * n, R[0] = the carry, no copies
        ld i, {R}
        ld [i], v7
        ld i, {R + 8}
        ld [i], v7
        ld i, {R + 16}
        ld [i], v7
        ld ve, 0
        ld v9, 0
ls_lp:  ld i, {CNT}
        add i, ve
        ld v0, [i]
        ld vd, v0               ; (lg_lk clobbers V0-V7)
ls_cnt: se vd, 0
        jp ls_do
        jp ls_nx
ls_do:  ld v0, vd               ; at most 9 per row
        ld v1, 9
        sub v1, v0
        se vf, 1
        ld v0, 9
        sub vd, v0
        ld i, {B}
        ld [i], v0
        ld v0, 1
        ld i, {LB}
        ld [i], v0
        call lg_lk
        ld v6, 0
        bmul v6
        jp ls_cnt
ls_nx:  add v9, {PREC}
        add ve, 1
        se ve, {LEVELS}
        jp ls_lp
        ld vb, {oR}             ; f = 10^PREC - S (S: R[0 .. PREC + 2))
        ld vc, {oB}
        ld vd, {PREC + 2}
        call {s.cp(6, 6)}
        ld v0, {PREC + 2}
        ld i, {LB}
        ld [i], v0
        ld i, zeros
        ld v7, [i]
        ld i, {A}
        ld [i], v7
        ld i, {A + 8}
        ld [i], v7
        ld i, {A + 16}
        ld [i], v7
        ld v0, 1
        ld i, {A}
        ld [i], v0
        ld v0, {PREC + 1}
        ld i, {LA}
        ld [i], v0
        bsub {s.pad("lg_f")}
        ld i, zeros             ; S > 10^PREC (the log of 1 + tiny): f = 0
        ld v7, [i]
        ld i, {R + 3}
        ld [i], v7
        ld i, {R + 11}
        ld [i], v7
        ld i, {R + 19}
        ld [i], v7
lg_f:   ld vb, {oR + 3}
        ld vd, {PREC}
        {STR(2)}
        ret
""")
s("lg_t:   db " + ", ".join(d for v in LOGS for d in v))
s("c_ln:   db " + ", ".join(CONSTS["ln10"]))
s("c_le:   db " + ", ".join(CONSTS["le"]))
assert len(CONSTS["ln10"]) == len(CONSTS["le"]) == PREC + 1

# b_cst: B = the PREC + 1 digits at page ... (V9 = 0: c_ln, 1: c_le), LB = PREC + 1
s(f"""
b_ln:   ld v8, 8                ; B = the constant at c_ln
        ld i, c_ln
        ld v7, [i]
        ld i, {B}
        ld [i], v7
        ld i, c_ln
        add i, v8
        ld v7, [i]
        ld i, {B + 8}
        ld [i], v7
        ld i, c_ln
        add i, v8
        add i, v8
        ld v7, [i]
        ld i, {B + 16}
        ld [i], v7
        jp b_cn
b_le:   ld v8, 8                ; B = the constant at c_le
        ld i, c_le
        ld v7, [i]
        ld i, {B}
        ld [i], v7
        ld i, c_le
        add i, v8
        ld v7, [i]
        ld i, {B + 8}
        ld [i], v7
        ld i, c_le
        add i, v8
        add i, v8
        ld v7, [i]
        ld i, {B + 16}
        ld [i], v7
        jp b_cn
b_cn:   ld v0, {PREC + 1}
        ld i, {LB}
        ld [i], v0
        ret
""")

# v19: A = the PREC + 2 digits of c (VARS[12], 2 digits) and f (slot 2), LA = PREC + 2;
# t19: VB, VD = A without its leading zeros (for STR)
s(f"""
v19:    ld i, {VARS + 12}
        ld v0, [i]
        ld i, {TMP}
        ld b, v0
        ld v2, [i]
        ld v0, v1
        ld v1, v2
        ld i, {A}
        ld [i], v1
        ld vd, {PREC}
        ld vb, {off(2)}
        ld vc, {oA + 2}
        call {s.cp(page(2), 6)}
        ld v0, {PREC + 2}
        ld i, {LA}
        ld [i], v0
        ret
t19:    ld vb, {oA}
t19_l:  ld v0, {oA + PREC + 1}
        se vb, v0
        jp t19_c
        jp t19_g
t19_c:  ld i, {P6}
        add i, vb
        ld v0, [i]
        se v0, 0
        jp t19_g
        add vb, 1
        jp t19_l
t19_g:  ld vd, {oA + PREC + 2}
        sub vd, vb
        ret
""")

# lg_out: OUT = the digits of V3 and "."; VC = where the decimals go, V9 = the digit count;
# lg_end: "..." after DEC decimals, print
s(f"""
lg_out: call v2o
        ld i, {NDIG}
        ld v0, [i]
        ld v9, v0
        ld i, {OUT}
        add i, v9
        ld v0, 10
        ld [i], v0
        ld vc, v9
        add vc, 1
        ret
lg_end: ld v0, v9
        add v0, {DEC + 1}
        ld i, {OUT}
        add i, v0
        ld v1, 11
        ld v0, v1
        ld [i], v0
        ld v0, v9
        add v0, {DEC + 2}
        ld i, {OUT}
        print
lg_int: ld i, {VARS + 12}       ; print VARS[12] (log10 of a power of 10)
        ld v0, [i]
        ld v3, v0
        call v2o
        ld i, {NDIG}
        ld v0, [i]
        ld i, {OUT}
        print
log10:  call lg_core
        ld i, {VARS + 13}
        ld v0, [i]
        se v0, 0
        jp lg_int
        ld i, {VARS + 12}
        ld v0, [i]
        ld v3, v0
        call lg_out
        ld vd, {DEC}
        ld vb, {off(2)}
        call {s.cp(page(2), 0)}
        jp lg_end
ln:     call lg_core
        ld i, {VARS + 13}       ; ln 1 = 0
        ld v0, [i]
        se v0, 1
        jp ln_go
        ld i, {VARS + 12}
        ld v0, [i]
        se v0, 0
        jp ln_go
        jp p_zero
ln_go:  call v19
        call b_ln
        call mul_ab             ; R[0 .. 37): the integer part R[0 .. 3), decimals R[3 ..)
        ld i, {R}
        ld v2, [i]
        ld v3, v0               ; 100 R0 + 10 R1 + R2
        add v3, v3
        add v3, v3
        add v3, v0
        add v3, v3
        add v3, v1
        ld v4, v3
        add v3, v3
        add v3, v3
        add v3, v4
        add v3, v3
        add v3, v2
        call lg_out
        ld vd, {DEC}
        ld vb, {oR + 3}
        call {s.cp(6, 0)}
        jp lg_end
log2:   ld v0, 2                ; log2 = logb with B = 2
        ld i, {B}
        ld [i], v0
        ld v0, 1
        ld i, {LB}
        ld [i], v0
        jp logb
eq14_r: ld i, {SLEN(1)}         ; VF = 1 if slot 1 == slot 4
        ld v0, [i]
        ld v2, v0
        ld i, {SLEN(4)}
        ld v0, [i]
        ld vf, 0
        se v0, v2
        ret
        ld v3, 0
eq_lp:  ld i, {SADDR(1)}
        add i, v3
        ld v0, [i]
        ld v1, v0
        ld i, {SADDR(4)}
        add i, v3
        ld v0, [i]
        ld vf, 0
        se v0, v1
        ret
        add v3, 1
        se v3, v2
        jp eq_lp
        ld vf, 1
        ret
""")

# logb: log_B A = log10 A / log10 B; the integer n when A = B^n (a candidate n from the
# quotient's remainder: below 10^-8 of the divisor gives n, within 10^-8 of it n + 1; then B^n
# is checked against A), otherwise the BIOS's division of the two logs prints 10 decimals
s(f"""
logb:   {BS(0)}                 ; b (the base for the check)
        {AS(3)}                 ; a, until log10 b is done
        {LDA(0)}
        call lg_core
        call v19
        call t19
        {STR(5)}
        {LDA(3)}
        call lg_core            ; log10 a; slot 4 = a
        call v19
        call t19
        {STR(3)}
        {LDB(5)}
        ld vb, {off(3)}
        call dvm_s              ; v_a div v_b
        ld v3, vd               ; the quotient (1-2 digits) -> VE
        ld i, {P6}
        add i, vb
        ld v1, [i]
        ld ve, v0
        se v3, 1
        jp lb_q2
        jp lb_r
lb_q2:  se v3, 2
        jp lb_div
        ld v4, ve
        add ve, ve
        add ve, ve
        add ve, v4
        add ve, ve
        add ve, v1
lb_r:   ld i, {VARS}            ; remainder length <= LB - 8: n
        ld v1, [i]
        ld i, {LB}
        ld v0, [i]
        add v0, 0xF8
        sub v0, v1
        se vf, 1
        jp lb_up
        jp lb_chk
lb_up:  ld i, {VARS}            ; d = v_b - r, length <= L(v_b) - 8: n + 1
        ld v1, [i]
        ld vb, v0
        ld vd, v1
        ld v0, v1
        ld i, {LB}
        ld [i], v0
        ld vc, {oB}
        call {s.cp(6, 6)}
        {LDA(5)}
        call sub_ab
        ld i, {SLEN(5)}
        ld v0, [i]
        add v0, 0xF8
        sub v0, vd
        se vf, 1
        jp lb_div
        add ve, 1
lb_chk: ld v0, ve
        ld i, {VARS + 11}
        ld [i], v0
        call pow_core           ; b ^ n == a ?
        se v8, 0
        jp lb_div
        call eq14_r
        se vf, 1
        jp lb_div
        ld i, {VARS + 11}
        ld v0, [i]
        ld v3, v0
        call v2o
        ld i, {NDIG}
        ld v0, [i]
        ld i, {OUT}
        print
lb_div: ld i, {SLEN(3)}         ; the BIOS's division of v_a by v_b, 10 decimals
        ld v0, [i]
        ld i, {LA}
        ld [i], v0
        ld vd, v0
        ld vb, {off(3)}
        ld vc, {oW + 1}
        call {s.cp(page(3), 6)}
        {LDB(5)}
        jp div
""")

s.feature = "exp"
# ================================================================ exp10, exp
# x = A / 10^FA (negative if SGN); exp: t = x log10(e) = A * c_le / 10^(FA + 18). The digits
# go to OUT (as scratch, Z) with the units digit at Z[31]: the integer part Z[29..32)
# (anything above it is too big), the fraction Z[32 .. 32 + PREC). c = floor(t),
# y = t - c in [0, 1): 10^y = prod (1 + 10^-k)^n_k with n_k from subtracting L_k from y
# greedily (BSUB), multiplied up by shift-and-add (BADD) as in lg_core; z = 10^y is PREC
# digits in A. VE = |c|, VARS[13] = c < 0.
Z = OUT
s(f"""
exp10:  ld v0, 0
        jp ex_go
exp:    ld v0, 1
ex_go:  ld i, {VARS + 4}
        ld [i], v0
        ld i, zeros
        ld v7, [i]
        ld v8, 0
ex_z:   ld i, {Z}
        add i, v8
        ld [i], v7
        add v8, 8
        se v8, 64
        jp ex_z
        ld i, {FA}
        ld v0, [i]
        ld v9, v0               ; fraction digits
        ld v1, 14
        sub v1, v0
        se vf, 1
        jp ovf
        ld i, {VARS + 4}
        ld v0, [i]
        se v0, 0
        jp ex_e
        ld i, {LA}
        ld v0, [i]
        ld vd, v0
        ld vb, {oA}
        jp ex_cp
ex_e:   call b_le
        call mul_ab
        add v9, {PREC + 1}
ex_cp:  ld vc, 32
        sub vc, vd
        add vc, v9
        call {s.cp(6, 0)}
        ld i, zeros             ; the copy's tail
        ld v7, [i]
        ld i, {Z + 32}
        add i, v9
        ld [i], v7
        ld v8, 0                ; anything above Z[29]: too big
        ld v9, 0
ex_hi:  ld i, {Z}
        add i, v9
        ld v0, [i]
        se v0, 0
        ld v8, 1
        add v9, 1
        se v9, 30
        jp ex_hi
        ld i, {Z + 30}
        ld v1, [i]
        ld v3, v0               ; ip = 10 Z30 + Z31
        add v3, v3
        add v3, v3
        add v3, v0
        add v3, v3
        add v3, v1
        ld vd, 0                ; F == 0 ?
        ld v9, 0
ex_fz:  ld i, {Z + 32}
        add i, v9
        ld v0, [i]
        se v0, 0
        ld vd, 1
        add v9, 1
        se v9, {PREC}
        jp ex_fz
        ld i, {SGN}
        ld v0, [i]
        ld v1, 1
        and v0, v1
        ld i, {VARS + 13}
        ld [i], v0
        ld ve, v3
        se v0, 1
        jp ex_pos
        se v8, 0
        jp ex_tiny
        se vd, 0
        jp ex_nf
        jp ex_exact             ; -x an integer: c = -ip, exact
ex_nf:  add ve, 1               ; c = -(ip + 1), y = 1 - F
        call zab
        ld v0, 1
        ld i, {A}
        ld [i], v0
        ld v0, {PREC + 1}
        ld v1, {PREC}
        ld i, {LA}
        ld [i], v1
        ld vd, {PREC}
        ld vb, 32
        ld vc, {oB}
        call {s.cp(0, 6)}
        bsub {s.pad("ex_ny")}
ex_ny:  ld vd, {PREC}           ; y = R[2 ..)
        ld vb, {oR + 2}
        ld vc, 32
        call {s.cp(6, 0)}
        jp ex_y
ex_pos: se v8, 0
        jp ovf
        se vd, 0
        jp ex_y
ex_exact: se ve, 0              ; 10^c exactly (x = 0: 1)
        jp ex_x1
        jp p_one
ex_x1:  ld v0, 57
        sub v0, ve
        se vf, 1
        jp ovf
        ld i, zeros
        ld v7, [i]
        ld v8, 0
ex_xz:  ld i, {OUT}
        add i, v8
        ld [i], v7
        add v8, 8
        se v8, 64
        jp ex_xz
        ld i, {VARS + 13}
        ld v0, [i]
        se v0, 1
        jp ex_xp
        ld i, {OUT + 1}         ; 0.0..01: "0", ".", c - 1 zeros, "1"
        ld v0, 10
        ld [i], v0
        ld i, {OUT + 1}
        add i, ve
        ld v0, 1
        ld [i], v0
        ld v0, ve
        add v0, 2
        ld i, {OUT}
        print
ex_xp:  ld i, {OUT}             ; 1 and c zeros
        ld v0, 1
        ld [i], v0
        ld v0, ve
        add v0, 1
        print
ex_tiny: ld i, zeros            ; 0.0000000000...
        ld v7, [i]
        ld i, {OUT}
        ld [i], v7
        ld i, {OUT + 8}
        ld [i], v7
        ld i, {OUT + 1}
        ld v0, 10
        ld [i], v0
        ld i, {OUT + 12}
        ld v0, 11
        ld [i], v0
        ld i, {OUT}
        ld v0, 13
        print
ex_y:   call zab                ; A = y (PREC digits at Z + 32), LA = PREC
        ld vd, {PREC}
        ld vb, 32
        ld vc, {oA}
        call {s.cp(0, 6)}
        ld v0, {PREC}
        ld i, {LA}
        ld [i], v0
        call cnt0
        ld vd, 0                ; k
        ld v9, 0
eb_lp:  call lg_bk
eb_try: bsub {s.pad("eb_acc")}
        jp eb_rej
eb_acc: ld i, {R + 1}           ; y -= L_k
        ld v7, [i]
        ld i, {A}
        ld [i], v7
        ld i, {R + 9}
        ld v7, [i]
        ld i, {A + 8}
        ld [i], v7
        ld i, {R + 17}
        ld v0, [i]
        ld i, {A + 16}
        ld [i], v0
        ld i, {CNT}
        add i, vd
        ld v0, [i]
        add v0, 1
        ld [i], v0
        jp eb_try
eb_rej: add v9, {PREC}
        add vd, 1
        se vd, {LEVELS}
        jp eb_lp
        call zab                ; z = 1, multiplied up
        ld v0, 1
        ld i, {A}
        ld [i], v0
        ld i, {B}
        ld [i], v0
        ld v0, {PREC}
        ld i, {LA}
        ld [i], v0
        ld vd, 0
ez_lp:  ld v0, {PREC}
        sub v0, vd
        ld i, {LB}
        ld [i], v0
        ld i, {CNT}
        add i, vd
        ld v0, [i]
        ld v9, v0
ez_n:   se v9, 0
        jp ez_do
        jp ez_nx
ez_do:  badd {s.pad("ez_ok")}
ez_ok:  call y_ab
        add v9, 0xFF
        jp ez_n
ez_nx:  add vd, 1
        se vd, {LEVELS}
        jp ez_lp
        ld i, {VARS + 13}
        ld v0, [i]
        se v0, 1
        jp ex_fpos
        ld v0, 21               ; c < 0: "0." and |c| - 1 zeros, then 11 digits
        sub v0, ve
        se vf, 1
        jp ex_tiny
        ld i, {OUT}
        ld v0, 0
        ld v1, 10
        ld [i], v1
        ld v9, 2
        ld v1, ve
ex_nz:  add v1, 0xFF
        se v1, 0
        jp ex_nz0
        jp ex_dig
ex_nz0: ld i, {OUT}
        add i, v9
        ld v0, 0
        ld [i], v0
        add v9, 1
        jp ex_nz
ex_dig: ld vd, 11               ; z's first 11 digits, "..."
        ld vb, {oA}
        ld vc, v9
        call {s.cp(6, 0)}
        add v9, 11
        jp ex_ell
ex_fpos: ld v0, 10
        sub v0, ve
        se vf, 1
        jp ex_sci
        ld vd, ve               ; c + 1 digits, ".", 10 - c digits, "..."
        add vd, 1
        ld vb, {oA}
        ld vc, 0
        call {s.cp(6, 0)}
        ld v9, 11
        ld v0, 10
        se ve, v0
        jp ex_pt
        jp ex_ell
ex_pt:  ld i, {OUT + 1}
        add i, ve
        ld v0, 10
        ld [i], v0
        ld vd, 10
        sub vd, ve
        ld vb, ve
        add vb, {oA + 1}
        ld vc, ve
        add vc, 2
        call {s.cp(6, 0)}
        ld v9, 12
ex_ell: ld i, {OUT}
        add i, v9
        ld v0, 11
        ld [i], v0
        add v9, 1
        ld i, {OUT}
        ld v0, v9
        print
ex_sci: ld i, {A}               ; z0 "." z1..z10 "e" c
        ld v0, [i]
        ld v1, 10
        ld i, {OUT}
        ld [i], v1
        ld vd, 10
        ld vb, {oA + 1}
        ld vc, 2
        call {s.cp(6, 0)}
        ld i, {OUT + 12}
        ld v0, 14
        ld [i], v0
        ld i, {TMP}
        ld b, ve
        ld v2, [i]
        ld v0, v1
        ld v1, v2
        ld i, {OUT + 13}
        ld [i], v1
        ld i, {OUT}
        ld v0, 15
        print
""")

# ================================================================ assembly
PROFILES = {"numbers": ("core", "num", "root"), "analysis": ("core", "log", "exp")}
SRCS, ROMS, LABELSETS = {}, {}, {}
for _name, _feat in PROFILES.items():
    SRCS[_name] = s.source(_feat)
    ROMS[_name], LABELSETS[_name] = assemble(bcd.SRC + "\n" + SRCS[_name])
    _L = LABELSETS[_name]
    assert ROMS[_name][:len(bcd.ROM)] == bcd.ROM, "the BIOS must keep its addresses"
    assert _L["pads_end"] <= 0x400, (_name, hex(_L["pads_end"]))
    assert _L["low_end"] <= LOW_END, (_name, hex(_L["low_end"]))
    assert _L["lib_end"] <= TOP, (_name, hex(_L["lib_end"]))
OPS = {op: LABELSETS["numbers"][f"jt_{op}"] for op in OP_NAMES}
assert all(LABELSETS[p][f"jt_{op}"] == OPS[op] for p in PROFILES for op in OP_NAMES)
PROFILE_OF = {op: next(p for p in PROFILES if LABELSETS[p].get(op)) for op in OP_NAMES}


def call_writes(op: str, a, b="0", mode: int = 0, frac: int = 0,
                neg: bool = False) -> list[tuple[int, int]]:
    """The CALL record for a library entry: A's and B's digits (a string, or a list of digit
    values), their lengths, MODE, A's fraction digits and sign, then PC."""
    from chipformer import spec

    da = [int(ch) for ch in a] if isinstance(a, str) else list(a)
    db = [int(ch) for ch in b] if isinstance(b, str) else list(b)
    assert 1 <= len(da) <= MD and 1 <= len(db) <= MD, (a, b)
    w = [(A + i, d) for i, d in enumerate(da)] + [(B + i, d) for i, d in enumerate(db)]
    return w + [(LA, len(da)), (LB, len(db)), (MODE, mode), (FA, frac), (SGN, int(neg)),
                (spec.PC, OPS[op])]


def run(op: str, a, b="0", mode: int = 0, frac: int = 0, neg: bool = False,
        limit: int = 400_000, machine=None):
    """Run one call on the reference (BridgeMachine with the library ROM; or ``machine`` =
    (a machine class, a ROM)): (the printed cells as text, the trace tokens from the CALL
    record to the print, the machine)."""
    from chipformer import spec
    from chipformer.q35 import bridge as Bq
    from chipformer.trace import encode_writes

    cls, rom = machine or (Bq.BridgeMachine, ROMS[PROFILE_OF[op]])
    bm = cls(rom, bcd.QUIRKS)
    bm.text(3)
    n0 = len(bm.tokens)
    w = call_writes(op, a, b, mode, frac, neg)
    bm.emu.apply(w)
    bm._emit(encode_writes(w) + [spec.STEP], True)
    bm.halted = False
    steps = 0
    while not bm.halted:
        bm.step()
        steps += 1
        if steps > limit:
            raise RuntimeError(f"{op}({a}, {b}) did not halt")
    sym = {Bq.SPACE: " ", Bq.MINUS: " -", Bq.PERIOD: ".", Bq.ELLIPSIS: "...", Bq.NL: "",
           Bq.TR_OPEN: "", Bq.TR_CLOSE: "", Bq.DOLLAR: "$", Bq.DDOLLAR: "$$"}
    out = "".join(sym[t] if t in sym else "0123456789abcdef"[Bq.HEX.index(t)]
                  for t, g in zip(bm.tokens[n0:], bm.generated[n0:]) if g and Bq.is_text(t))
    return out, len(bm.tokens) - n0, bm
