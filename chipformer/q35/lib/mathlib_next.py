"""The maths library with the proposed macros (macros_next.py): what they would save.

The same routines as mathlib.py, with the loops the macros replace swapped out:
    BMOV   r_a / r_b, a_b / b_a, w_b, and the y copy of exp's subtraction phase
    BSHAD  lg_core's pseudo-division loop and exp's multiply-up loop
    BSDIV  to_base's division by K = k^m <= 25 (one record per chunk of m digits)
Run with run() (NextBridgeMachine). Reference only: the circuit has none of these yet.
"""

from chipformer.q35.lib import bcd
from chipformer.q35.lib import mathlib as M
from chipformer.q35.lib.asm import assemble
from chipformer.q35.lib.macros_next import BMOV, BSDIV, BSHAD, NextBridgeMachine, encode


def db(kind, x=0, y=0):
    op = encode(kind, x, y)
    return f"db {op >> 8}, {op & 0xFF}"


def between(start: str, end: str | None, new: str):
    """Swap: lines from the one starting with start up to (not including) the one starting
    with end (None: the block's end) become new."""
    def f(lines):
        i = next(j for j, ln in enumerate(lines) if ln.startswith(start))
        k = len(lines) if end is None else next(
            j for j in range(i + 1, len(lines)) if lines[j].startswith(end))
        return lines[:i] + [x.strip() for x in new.strip("\n").splitlines()] + lines[k:]
    return f


def chain(*fs):
    def f(lines):
        for g in fs:
            lines = g(lines)
        return lines
    return f


P6, A, B, LA, LB, R, VARS, CNT, OUT, PREC = (M.P6, M.A, M.B, M.LA, M.LB, M.R, M.VARS, M.CNT,
                                             M.OUT, M.PREC)
SWAP = {
    "r_a": between("r_a:", None, f"""
r_a:    ld v0, vd               ; the result (page 6 + VB, VD long) -> A / B: one BMOV
        ld i, {P6}
        add i, vb
        {db(BMOV, 0)}
        ret
r_b:    ld v0, vd
        ld i, {P6}
        add i, vb
        {db(BMOV, 1)}
        ret
"""),
    "dvm_s": between("a_b:", None, f"""
a_b:    ld i, {LA}
        ld v0, [i]
        ld i, {A}
        {db(BMOV, 1)}
        ret
b_a:    ld i, {LB}
        ld v0, [i]
        ld i, {B}
        {db(BMOV, 0)}
        ret
"""),
    "w_b": between("w_b:", "dvm_a:", f"""
w_b:    ld i, {VARS}
        ld v1, [i]
        ld i, {P6}
        add i, v0
        ld v0, v1
        {db(BMOV, 1)}
        ret
"""),
    "y_ab": between("lg_lp:", "lg_sum:", f"""
lg_lp:  {db(BSHAD, 0xE, 0)}            ; the whole pseudo-division: CNT[k], k = VE = 0 ..
"""),
    "exp10": chain(between("eb_acc:", "eb_rej:", f"""
eb_acc: ld i, {R + 1}           ; y -= L_k: R[1 ..] -> A
        ld v0, {PREC}
        {db(BMOV, 0)}
        ld i, {CNT}
        add i, vd
        ld v0, [i]
        add v0, 1
        ld [i], v0
        jp eb_try
"""), between("ez_lp:", "ld i, {}".format(VARS + 13), f"""
ez_lp:  {db(BSHAD, 0xD, 1)}            ; z = prod (1 + 10^-k)^CNT[k], k = VD = 0 ..
""")),
    "to_base": between("to_base:", None, f"""
to_base: call k_in              ; K = k^m <= 25 (V3, VARS[7] = m, powers at VARS[8 ..])
        ld v3, 1
        ld v2, 0
tn_kp:  ld i, {VARS + 8}
        add i, v2
        ld v0, v3
        ld [i], v0
        add v2, 1
        ld v0, 0
        ld v1, v9
tn_km:  add v0, v3
        se vf, 0
        jp tn_ke
        add v1, 0xFF
        se v1, 0
        jp tn_km
        ld v1, 25
        sub v1, v0
        se vf, 1
        jp tn_ke
        ld v3, v0
        jp tn_kp
tn_ke:  add v2, 0xFF
        ld v0, v2
        ld i, {VARS + 7}
        ld [i], v0
        ld vc, v3
        ld ve, 79
tn_lp:  {db(BSDIV, 0xC)}               ; A = A div K, VF = the chunk
        ld v4, vf
        ld i, {VARS + 7}
        ld v0, [i]
        ld v2, v0
        ld v5, ve
        sub v5, v2
        add v5, 1
tn_dg:  add v2, 0xFF
        ld i, {VARS + 8}
        add i, v2
        ld v0, [i]
        ld v1, v0
        ld v6, 0
tn_sb:  ld v0, v4
        sub v0, v1
        se vf, 1
        jp tn_put
        ld v4, v0
        add v6, 1
        jp tn_sb
tn_put: ld i, {OUT}
        add i, v5
        ld v0, v6
        ld [i], v0
        add v5, 1
        se v2, 0
        jp tn_dg
        ld i, {VARS + 7}
        ld v0, [i]
        sub ve, v0
        ld i, {LA}              ; A == 0: done
        ld v0, [i]
        se v0, 1
        jp tn_lp
        ld i, {A}
        ld v0, [i]
        se v0, 0
        jp tn_lp
        add ve, 1               ; skip leading zeros, keeping one digit
        ld v1, 79
tn_tz:  se ve, v1
        jp tn_t2
        jp tn_pr
tn_t2:  ld i, {OUT}
        add i, ve
        ld v0, [i]
        se v0, 0
        jp tn_pr
        add ve, 1
        jp tn_tz
tn_pr:  ld i, {OUT}
        add i, ve
        ld v0, 80
        sub v0, ve
        print
"""),
}
PROFILES = {f"{p}_next": f for p, f in M.PROFILES.items()}
SRCS, ROMS, LABELSETS = {}, {}, {}
for _name, _feat in PROFILES.items():
    SRCS[_name] = M.s.source(_feat, SWAP)
    ROMS[_name], LABELSETS[_name] = assemble(bcd.SRC + "\n" + SRCS[_name])
    assert ROMS[_name][:len(bcd.ROM)] == bcd.ROM
    assert all(LABELSETS[_name][f"jt_{op}"] == M.OPS[op] for op in M.OP_NAMES)
    assert LABELSETS[_name]["pads_end"] <= 0x400 and LABELSETS[_name]["lib_end"] <= M.TOP


def run(op: str, a, b="0", mode: int = 0, frac: int = 0, neg: bool = False,
        limit: int = 400_000):
    """As mathlib.run, with the proposed macros."""
    return M.run(op, a, b, mode, frac, neg, limit,
                 machine=(NextBridgeMachine, ROMS[M.PROFILE_OF[op] + "_next"]))
