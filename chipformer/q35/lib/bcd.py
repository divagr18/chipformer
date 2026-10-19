"""Exact decimal arithmetic for the computer inside the model ("the BIOS").

Calling convention (what the circuit's CALL record writes, see bridge.py):
  A = digits of the first argument, most significant first, at A_ADDR (LA of them at LEN)
  B = digits of the second argument at B_ADDR (LB at LEN + 1)
  PC = the routine's entry
Each routine leaves the result's digits at R_ADDR and ends with PRINT (00FD), which prints V0
digits from I and halts. Leading zeros are stripped. Nothing ticks during a call, so the routines
never use the timers or wait for keys.
"""

from pathlib import Path

from chipformer import spec
from chipformer.q35.lib.asm import assemble

A_ADDR, B_ADDR, LEN, R_ADDR = 0x600, 0x620, 0x640, 0x660
# 0: a tool call (<tool_response>); 1: an inline call (" digits"), | 2 inside "$", | 4 inside
# "$$", | 8 the result is negative (set by sub's BRSUB)
MODE = LEN + 2
DSTOP = LEN + 3  # BDIV / BADB stop at t = LA + DSTOP (div: DEC; mod, gcd: 0)
# div: W = 0, A's digits (the CALL record writes them at W_ADDR + 1), DEC zeros; the quotient goes
# to R_ADDR as P[t] (t <= LA), "." (10) at P[LA + 1], P[t + 1] (t > LA); not ending: frac
W_ADDR = 0x6BF
DEC = 10  # decimals of an inexact quotient
TMP = 0x6E4  # gcd: the remainder while A and B move (after W; 0x700.. holds eval's code)
T10, MUL, DIV, MOD = 0x700, 0x710, 0x780, 0x800
MAX_DIGITS = 24
QUIRKS = ("loadstore",)  # FX55/FX65 leave I unchanged


SRC = f"""
idle:   jp idle                 ; PC rests here between calls (not a PRINT)

; ---------------------------------------------------------------- add: R = A + B
add:    badd trim               ; macro: R = A + B, v0 = length, i = R, then trim
        db 0, 0                 ; (entries are compiled into the weights)

; ---------------------------------------------------------------- sub: R = A - B
sub:    bsub trim               ; macro: R = A - B, v0 = length, i = R; trim unless A < B
        jp sub_neg

; ---------------------------------------------------------------- mul: R = A * B
mul:    ld v1, 0                ; rows done
        bmul0 v1                ; macro: the first schoolbook row (no BZERO)
        bmul v1                 ; macro: the next row; repeats until the last row,
                                ; which sets v0 = LA + LB, i = R

; ---------------------------------------------------------------- print mem[i..i+v0)
trim:   btrim                   ; macro: drop a leading 0 (keeping one digit); repeats
        print

; ---------------------------------------------------------------- A - B < 0: B - A
; MODE |= 8: the print starts with " -" (a tool call: "<tool_response> -")
sub_neg: brsub trim

; ---------------------------------------------------------------- div: A / B
; the quotient if it ends within DEC decimals, else the reduced fraction p/q (frac, in low
; memory); "..." if B starts with 0
div:    bzero                   ; P[0 .. LA+LB) = 0
        ld i, {LEN}
        ld v1, [i]              ; v0 = LA, v1 = LB
        ld va, v0
        ld vb, v1
        ld i, {B_ADDR}
        ld v0, [i]
        sne v0, 0
        jp div_z
        ld i, {W_ADDR + 1}
        bld a, va               ; A = A's digits (div works on W; frac needs them)
        ld i, {R_ADDR + 1}
        add i, va
        ld v0, 10
        ld [i], v0              ; P[LA + 1] = "."
        ld i, {DSTOP}
        ld [i], v0              ; DS = DEC (= 10, as the ".")
        ld i, zeros
        ld v9, [i]              ; v0..v9 = 0
        ld i, {W_ADDR}
        ld [i], v0              ; W[0] = 0
        ld i, {W_ADDR + 1}
        add i, va
        ld [i], v9              ; W[LA+1 .. LA+DEC] = 0
        ld vc, vb               ; t = LB
        ld v0, va
        add v0, {DEC}
        sub v0, vc              ; VF = [LA + DEC >= LB]
        se vf, 0
        jp div_lp
        ld vc, va
        add vc, {DEC}           ; B is too long: the quotient is 0.000..., t = LA + DEC
        jp div_rem
div_lp: bdiv vc                 ; repeats itself, one quotient digit per record
        se vf, 1
        jp div_end              ; VF = 2 (exact) or 3 (DEC decimals)
        badb vc                 ; repeats itself while the window is negative
        se vf, 0
        jp div_end
        jp div_lp
div_end: se vf, 3
        jp div_ex
div_rem: jp frac                 ; no end within DEC decimals: the reduced fraction p/q
div_ex: ld ve, vc
        se vc, va
        add ve, 1               ; t > LA: P[t + 1]; t == LA: no "."
div_pr: ld v6, vb               ; first digit: P[LB] (P[LA] if LB > LA), then skip zeros
        ld v0, va
        sub v0, v6
        sne vf, 0
        ld v6, va
dp_lp:  se v6, va
        jp dp_chk
        jp dp_go
dp_chk: ld i, {R_ADDR}
        add i, v6
        ld v0, [i]
        se v0, 0
        jp dp_go
        add v6, 1
        jp dp_lp
dp_go:  ld i, {R_ADDR}
        add i, v6
        ld v0, ve
        sub v0, v6
        add v0, 1
        print
div_z:  ld i, {R_ADDR}            ; B starts with 0: "..." (every result prints something)
        ld v0, 11
        ld [i], v0
        ld v0, 1
        print
zeros:  db 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

"""

# mod and gcd (v16) next to v18's evaluator: the 4 KB are full, so their code fills four gaps
# (eval.asm's {MODGCD1..4}): the end of eval's first part (below A_ADDR), two holes in the data
# page (after LEN's cells, after R's 48 digits; nothing writes them), the end of eval's second
# part (below TAPE) and the end of chain mode's code (below H). Asserts check each one.
GAP_LEN, GAP_R = LEN + 4, R_ADDR + 48  # data-page holes: up to R_ADDR, up to W_ADDR
MODGCD1 = f"""
; ---------------------------------------------------------------- mod: A mod B
; (A at W + 1, as for div); "..." if B starts with 0
mod:    call args
        sne v0, 0
        jp div_z
        call rem
rem_pr: ld i, {W_ADDR}            ; print W[vd .. ve]
        add i, vd
        ld v0, ve
        sub v0, vd
        add v0, 1
        print
mg1_end:
"""
MODGCD2 = f"""
        org {GAP_LEN}
; ---------------------------------------------------------------- the shared parts
args:   ld v0, 0                ; DS = 0 (the integer part only), W[0] = 0,
        ld i, {DSTOP}           ; va = LA, vb = LB, v0 = B[0]
        ld [i], v0
        ld i, {W_ADDR}
        ld [i], v0
        ld i, {LEN}
        ld v1, [i]
        ld va, v0
        ld vb, v1
        ld i, {B_ADDR}
        ld v0, [i]
        ret
args_end:
        org {GAP_R}
; ---------------------------------------------------------------- gcd: gcd(A, B), by Euclid
gcd:    call args
        se v0, 0
        jp gcd_lp
        ld vd, 1                ; B = 0: the answer is A
        ld ve, va
        call strip
        jp rem_pr
gcd_lp: call rem                ; W[vd .. ve] = A mod B
        ld i, {W_ADDR}
        add i, vd
        ld v0, [i]
        sne v0, 0
        jp gcd_b                ; 0: the answer is B
        ld vc, ve               ; vc = the remainder's length (the next LB)
        sub vc, vd
        add vc, 1
        ld i, {W_ADDR}          ; the moves go through A with v18's BLD / BSTA macros
        add i, vd
        bld a, vc               ; A = the remainder
        ld i, {TMP}
        bsta                    ; TMP = the remainder
        ld i, {B_ADDR}
        jp gcd_c
gcd1_end:
"""
MODGCD3 = f"""
gcd_c:  bld a, vb               ; A = B, LA = LB (the next A)
        ld i, {W_ADDR + 1}
        bsta                    ; W + 1 = B
        ld i, {TMP}
        bld b, vc               ; B = the remainder, LB = its length
        ld va, vb
        ld vb, vc
        jp gcd_lp
gcd_b:  jp gcd_e                 ; (low memory: frac's gcd returns instead)
"""
MODGCD4 = f"""
; rem: A mod B for A at W[1 .. va] (W[0] = 0), B at B[0 .. vb) (B[0] != 0), LEN = LA, LB and
; DS = 0: the remainder, without leading zeros (one digit if it is 0), is W[vd .. ve]. BDIV and
; BADB run the integer part of A / B and leave the remainder in the last window.
rem:    ld v0, va
        sub v0, vb              ; VF = [LA >= LB]
        se vf, 0
        jp rem_go
        ld vd, 1                ; B is longer: the remainder is A
        ld ve, va
        jp strip
rem_go: ld vc, vb               ; t = LB
rem_lp: bdiv vc                 ; repeats itself, one quotient digit per record
        se vf, 1
        jp rem_end              ; VF = 2 (a 0 remainder) or 3, at t = LA
        badb vc                 ; repeats itself while the window is negative
        se vf, 0
        jp rem_end
        jp rem_lp
rem_end: ld ve, va
        ld vd, ve
        sub vd, vb
        add vd, 1               ; the window's top digit is 0 (the remainder is below B)
strip:  se vd, ve               ; skip zeros in W[vd .. ve), keeping the last digit
        jp st_chk
        ret
st_chk: ld i, {W_ADDR}
        add i, vd
        ld v0, [i]
        se v0, 0
        ret
        add vd, 1
        jp strip
"""

# ---------------------------------------------------------------- eval: a whole expression (v18)
# The CALL record writes the tape (one code per token of the expression: bridge.TAPE_CODE, digits
# 0..9, the symbols . + - * / ( ) ^ as 10..17, fused tokens such as ")/(" as 18.., which the
# dispatch table expands), its length and MODE. eval (eval.asm) parses it by shunting-yard into
# exact fractions n / d and prints the value: the integer, or n / d as div prints it, or "..."
# when it can't (more than MAX_DIGITS digits anywhere, a division by 0, a malformed expression,
# more than NVAL pending values). A value lives in a slot: H + 32 s = [sign, n length, n's place
# (0: its digits follow, else a number on the tape), d length (0: d = 1), n's digits], D + 32 s
# = d's digits; BLD / BSTA (macros.py) move digits around.
# Memory: eval's code from 0x2AE to A_ADDR and from CODE2 to TAPE, then its data, then DBUF (10^k).
CODE2 = 0x700
TAPE, TLEN, MSAVE, CUR, DXY, SGN, EXP = 0xC00, 0xC30, 0xC31, 0xC32, 0xC34, 0xC36, 0xC38
OPST, XTAB, H, D = 0xC50, 0xC80, 0xE00, 0xF00  # operator stack, ROM tables, value slots
NVAL = 6  # value slots (offsets 0 .. 160); T6 and T7 are temporaries
T6, T7 = 192, 224
END = 63  # after the tape
H1, H3, H4, SGN1, EXP1, LEN1, MAXD1 = H + 1, H + 3, H + 4, SGN + 1, EXP + 1, LEN + 1, MAX_DIGITS + 1
TAPE_M1 = TAPE - 1  # a number's place p: its digits from TAPE + p - 1
# chain mode (chain.asm): its sign parity, accumulator lengths, the first number, accumulators
GP, L0, L1, PEND, FOP, ACC0, ACC1, FST, CODE3 = (0xC3A, 0xC3B, 0xC3C, 0xC3D, 0xC40, 0x0A0, 0x0C0,
                                                OPST, 0xCA0)
VMAX, W1 = 32 * NVAL, W_ADDR + 1
T6_DL, T6_D, T7_NL, T7_N = H + T6 + 3, D + T6, H + T7 + 1, H + T7 + 4
SYM_SUB = dict(zip(".+-*/()^", ("s_pt", "s_add", "s_sub", "s_mul", "s_div", "s_lp", "s_rp", "s_pow")))


def _dispatch() -> tuple[str, str, str]:
    """The tape codes' handlers (each symbol of a fused token in turn), the jump table on codes
    and the one on operators (reduce)."""
    from chipformer.q35 import bridge

    handlers, jtab = [], []
    for code in range(64):
        if code < 10:
            jtab.append("        jp dig")
        elif code in bridge.EXPANSION:
            calls = [f"        call {SYM_SUB[ch]}" for ch in bridge.EXPANSION[code]]
            handlers.append(f"h{code}:" + "\n".join(calls)[3:] + "\n        jp ev_lp")
            jtab.append(f"        jp h{code}")
        else:
            jtab.append("        jp h_end" if code == END else "        jp err")
    ops = {4: "r_add", 5: "r_sub", 8: "r_mul", 9: "r_div", 16: "r_pow"}
    rtab = [f"        jp {ops.get(code, 'err')}" for code in range(17)]
    return "\n".join(handlers), "\n".join(jtab), "\n".join(rtab)


HANDLERS, JTAB, RTAB = _dispatch()
_NAMES = {k: v for k, v in globals().items() if k.isupper()}
EVAL_SRC = (Path(__file__).with_name("eval.asm").read_text(encoding="utf-8")
            .format(**_NAMES, DBUF_DB=", ".join(["1"] + ["0"] * MAX_DIGITS),                    **dict(zip(("CHAIN1", "CHAIN2"), Path(__file__).with_name("chain.asm")
                               .read_text(encoding="utf-8").format(**_NAMES)
                               .split("; ---- part 2")))))

# ---------------------------------------------------------------- frac: a reduced fraction
# Below PROGRAM_START (0x0E0 .. 0x1FF: after the font and chain mode's accumulators) the 4 KB
# still had room: the boot record writes this segment too (ROM.segments, ref/emu.Chip8).
# div jumps here when A / B does not end within DEC decimals: p = A / g, q = B / g for
# g = gcd(A, B) (v16's gcd, which returns to frac instead of printing), printed as "p/q" (12 is
# "/"), MODE's sign as for the quotient. A's and B's digits wait in the tape's cells (eval is done
# with them when it prints through div).
LOW_ORG = 0x0E0
SAVE_A, SAVE_B = TAPE, TAPE + MAX_DIGITS
FRAC_F, LA0, LB0, LP = LEN + 28, LEN + 29, LEN + 30, LEN + 31  # (after args' code in the hole)
OUT = W_ADDR  # the print: p, "/", q (W and TMP are free by then)
SLASH = 12  # the print's code for "/"
LOW_SRC = f"""
frac:   ld i, {SAVE_A}
        bsta                    ; SAVE_A = A (div saved it in A at its start)
        ld i, {B_ADDR}
        bld a, vb
        ld i, {SAVE_B}
        bsta                    ; SAVE_B = B
        ld v0, va
        ld v1, vb
        ld i, {LA0}
        ld [i], v1              ; LA0, LB0
        ld i, {SAVE_A}
        bld a, va
        ld i, {W_ADDR + 1}
        bsta                    ; W + 1 = A, as gcd takes it
        ld v0, 1
        ld i, {FRAC_F}
        ld [i], v0
        call gcd                ; B = g, LB its length
        ld v0, 0
        ld i, {FRAC_F}
        ld [i], v0
        ld i, {LA0}
        ld v0, [i]
        ld i, {SAVE_A}
        bld a, v0
        ld i, {W_ADDR + 1}
        bsta                    ; W + 1 = A
        call quo                ; I = p's first digit, v0 = its digits
        bld a, v0
        ld i, {LP}
        ld [i], v0              ; LP = lp
        ld i, {SAVE_A}
        bsta                    ; SAVE_A = p
        ld i, {LB0}
        ld v0, [i]
        ld i, {SAVE_B}
        bld a, v0
        ld i, {W_ADDR + 1}
        bsta                    ; W + 1 = B
        call quo                ; I = q's first digit, v0 = its digits
        ld v3, v0
        bld a, v3               ; A = q
        ld i, {LP}
        ld v0, [i]
        ld v2, v0               ; v2 = lp, v3 = lq
        ld i, {OUT + 1}
        add i, v2
        bsta                    ; OUT + lp + 1 = q
        ld i, {OUT}
        add i, v2
        ld v0, {SLASH}
        ld [i], v0              ; "/"
        ld i, {SAVE_A}
        bld a, v2
        ld i, {OUT}
        bsta                    ; OUT = p
        ld v0, v2
        add v0, v3
        add v0, 1
        ld i, {OUT}
        print

; quo: A / B for A at W + 1 and B dividing it: I = the quotient's first digit (P[LB .. LA]
; without leading zeros), v0 = its digits
quo:    call args
        call rem
        ld v6, vb
q_lp:   se v6, va
        jp q_chk
        jp q_go
q_chk:  ld i, {R_ADDR}
        add i, v6
        ld v0, [i]
        se v0, 0
        jp q_go
        add v6, 1
        jp q_lp
q_go:   ld v0, va
        sub v0, v6
        add v0, 1
        ld i, {R_ADDR}
        add i, v6
        ret

; gcd's end: the gcd op prints B; frac's call returns with it
gcd_e:  ld i, {FRAC_F}
        ld v0, [i]
        se v0, 0
        ret
        ld i, {B_ADDR}
        ld v0, vb
        print
low_end:
        org {spec.PROGRAM_START}
"""


class Image(bytes):
    """A ROM (loaded at PROGRAM_START) and its segments below it: [(address, bytes)]."""

    segments: tuple = ()


_out, LABELS = assemble(LOW_SRC + SRC + EVAL_SRC, start=LOW_ORG)
ROM = Image(_out[spec.PROGRAM_START - LOW_ORG:])
ROM.segments = ((LOW_ORG, bytes(_out[:LABELS["low_end"] - LOW_ORG])),)
assert LABELS["idle"] == spec.PROGRAM_START and LABELS["low_end"] <= spec.PROGRAM_START
OPS = {op: LABELS[op] for op in ("add", "mul", "sub", "div", "mod", "gcd", "eval")}
# The entries are compiled into the circuit's CALL record (chip8_q35.py), so they are part of the
# weights: moving a routine means rebuilding every checkpoint. Add new code after the entries.
assert {k: v for k, v in OPS.items() if k in ("add", "sub", "mul")} == {
    "add": 0x202, "sub": 0x206, "mul": 0x20A}, OPS
W_OPS = ("div", "mod", "gcd")  # A goes to W + 1: they work on it in place
assert LABELS["sm_t"] < 0x400 and LABELS["am_t"] < 0x400, "macro targets must sit below 0x400"
# compiled into the weights (the expression CALL record): eval's entry, the tape and its length
# (q35-int: div, mod and gcd come first, so eval sits later than v18's 0x2CA)
EVAL_ENTRY = 0x2CC
assert (OPS["eval"], TAPE, TLEN) == (EVAL_ENTRY, 0xC00, 0xC30), (hex(OPS["eval"]), TAPE, TLEN)
assert W_ADDR + 1 + MAX_DIGITS + DEC <= TMP and TMP + MAX_DIGITS <= CODE2, "gcd's TMP"
assert LABELS["code1_end"] <= A_ADDR and LABELS["eval_end"] <= TAPE, "eval's code overflows"
assert LABELS["dbuf"] + MAX_DIGITS < CODE3 and spec.FONT_BASE + 80 <= ACC0 < ACC1 + 32 <= 0x200
assert LABELS["chain_end"] <= H, "chain mode's code overflows"
# mod / gcd's pieces in the data page: clear of LEN's cells, R's 48 digits (mul) and W
assert DSTOP < GAP_LEN and LABELS["args_end"] <= FRAC_F, "args overflows its hole"
assert LABELS["gcd1_end"] <= W_ADDR, "gcd overflows its hole"


class EvalError(Exception):
    pass


def eval_value(tape: list[int]):
    """What eval computes, as (sign, n, d) with n, d strings (d None for 1), or EvalError: the
    reference for eval.asm, including its limits."""
    from chipformer.q35 import bridge

    syms = []
    for code in tape:
        syms += [(None, code)] if code < 10 else [(s, None) for s in bridge.EXPANSION[code]]
    vals, ops = [], []
    num, dec, operand = None, None, False  # digits, decimals (None: no ".")

    def fits(s):
        s = s.lstrip("0") or "0"
        if len(s) > MAX_DIGITS:
            raise EvalError("too long")
        return s

    def commit():
        nonlocal num, dec, operand
        if len(vals) >= NVAL:
            raise EvalError("too many values")
        k = dec or 0
        vals.append([0, fits(num), "1" + "0" * k if k else None])
        num, dec, operand = None, None, True

    def mul(a, b):
        return fits(str(int(a) * int(b)))

    def reduce():
        op = ops.pop()
        if op == "neg":
            if not vals:
                raise EvalError("no operand")
            vals[-1][0] ^= 1
            return
        if len(vals) < 2:
            raise EvalError("no operand")
        y = vals.pop()
        x = vals[-1]
        sx, nx, dx = x
        sy, ny, dy = y
        if op in "+-":
            if op == "-":
                sy ^= 1
            if dx is None and dy is None:
                t6, t7, d = nx, ny, None
            else:
                t6 = nx if dy is None else mul(nx, dy)
                t7 = ny if dx is None else mul(ny, dx)
                d = dx if dy is None else dy if dx is None else mul(dx, dy)
            if sx == sy:
                n, s = fits(str(int(t6) + int(t7))), sx
            else:
                a, b = int(t6), int(t7)
                n, s = fits(str(abs(a - b))), sx if a >= b else sy
            x[:] = [s, n, d]
        elif op == "*":
            x[:] = [sx ^ sy, mul(nx, ny), dx if dy is None else dy if dx is None else mul(dx, dy)]
        elif op == "/":
            if int(ny) == 0:
                raise EvalError("division by 0")
            x[:] = [sx ^ sy, nx if dy is None else mul(nx, dy), ny if dx is None else mul(dx, ny)]
        else:  # ^
            if dy is not None or len(ny) > 2:
                raise EvalError("exponent")
            e = int(ny)
            if e == 0:
                x[:] = [0, "1", None]
                return
            n, d = nx, dx
            for _ in range(e - 1):
                n = mul(n, nx)
                d = None if dx is None else mul(d, dx)
            s = sx if e % 2 else 0
            if sy:
                if int(n) == 0:
                    raise EvalError("division by 0")
                n, d = d or "1", n
            x[:] = [s, n, d]

    PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "neg": 3, "^": 4}
    for s, d in syms:
        if s is None:
            num = (num or "") + str(d)
            if dec is not None:
                dec += 1
            continue
        if s == ".":
            if num is None or dec is not None:
                raise EvalError("bad .")
            dec = 0
            continue
        if num is not None:
            commit()
        if s == "(":
            if operand:
                push_binary("*", ops, PREC, reduce)
            ops.append("(")
            operand = False
        elif s == ")":
            while ops and ops[-1] != "(":
                reduce()
            if ops:
                ops.pop()
            operand = True
        elif not operand:
            if s == "-":
                ops.append("neg")
            elif s != "+":
                raise EvalError("operator without an operand")
        else:
            push_binary(s, ops, PREC, reduce)
            operand = False
    if num is not None:
        commit()
    while ops:
        if ops[-1] == "(":
            ops.pop()
        else:
            reduce()
    if len(vals) != 1:
        raise EvalError("not one value")
    return tuple(vals[0])


def push_binary(s, ops, prec, reduce):
    while ops and ops[-1] != "(" and (prec[ops[-1]] > prec[s]
                                       or (prec[ops[-1]] == prec[s] and s != "^")):
        reduce()
    ops.append(s)


def eval_text(tape: list[int]) -> tuple[bool, str]:
    """(negative, digits) as eval prints them: the integer, the quotient as div prints it, or
    "..." when eval gives up."""
    from chipformer.q35 import evals

    try:
        s, n, d = eval_value(tape)
    except EvalError:
        return False, "..."
    neg = bool(s) and int(n) != 0
    if d is None:
        return neg, n
    return neg, evals.div_text(int(n), int(d))


def eval_writes(tape: list[int], mode: int) -> list[tuple[int, int]]:
    """The CALL record of an expression: the tape, its length, the mode, then PC = eval."""
    return [(TAPE + i, c) for i, c in enumerate(tape)] + [(TLEN, len(tape)), (MODE, mode),
                                                          (spec.PC, OPS["eval"])]


def call_writes(op: str, a: str, b: str, mode: int = 0) -> list[tuple[int, int]]:
    """The writes of a CALL record: the arguments' digits, their lengths, the mode, then PC."""
    da, db = [int(ch) for ch in a], [int(ch) for ch in b]
    assert 1 <= len(da) <= MAX_DIGITS and 1 <= len(db) <= MAX_DIGITS, (a, b)
    a_at = W_ADDR + 1 if op in W_OPS else A_ADDR
    w = [(a_at + i, d) for i, d in enumerate(da)] + [(B_ADDR + i, d) for i, d in enumerate(db)]
    return w + [(LEN, len(da)), (LEN + 1, len(db)), (MODE, mode), (spec.PC, OPS[op])]
