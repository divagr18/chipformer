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
# div: W = 0, A's digits (the CALL record writes them at W_ADDR + 1), DEC zeros; the quotient goes
# to R_ADDR as P[t] (t <= LA), "." (10) at P[LA + 1], P[t + 1] (t > LA), "..." (11) if inexact
W_ADDR = 0x6BF
DEC = 10  # decimals of an inexact quotient
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
; inline: MODE |= 8 (the print starts with " -"); a tool call prints nothing (v0 = 0)
sub_neg: brsub trim

; ---------------------------------------------------------------- div: A / B
; digits of an exact quotient, else DEC decimals and "..."; nothing if B starts with 0
div:    bzero                   ; P[0 .. LA+LB) = 0
        ld i, {LEN}
        ld v1, [i]              ; v0 = LA, v1 = LB
        ld va, v0
        ld vb, v1
        ld i, {B_ADDR}
        ld v0, [i]
        sne v0, 0
        jp div_z
        ld i, {R_ADDR + 1}
        add i, va
        ld v0, 10
        ld [i], v0              ; P[LA + 1] = "."
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
div_rem: ld i, {R_ADDR + 2}
        add i, vc
        ld v0, 11
        ld [i], v0              ; "..." after the last decimal
        ld ve, vc
        add ve, 2               ; the last index printed
        jp div_pr
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
div_z:  ld v0, 0
        print
zeros:  db 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

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
            .format(**_NAMES, DBUF_DB=", ".join(["1"] + ["0"] * MAX_DIGITS),
                    **dict(zip(("CHAIN1", "CHAIN2"), Path(__file__).with_name("chain.asm")
                               .read_text(encoding="utf-8").format(**_NAMES)
                               .split("; ---- part 2")))))

ROM, LABELS = assemble(SRC + EVAL_SRC)
OPS = {"add": LABELS["add"], "mul": LABELS["mul"], "sub": LABELS["sub"], "div": LABELS["div"],
       "eval": LABELS["eval"]}
# The entries are compiled into the circuit's CALL record (chip8_q35.py), so they are part of the
# weights: moving a routine means rebuilding every checkpoint. Add new code after the entries.
assert {k: v for k, v in OPS.items() if k in ("add", "sub", "mul")} == {
    "add": 0x202, "sub": 0x206, "mul": 0x20A}, OPS
assert LABELS["sm_t"] < 0x400 and LABELS["am_t"] < 0x400, "macro targets must sit below 0x400"
# compiled into the v18 weights (the expression CALL record): eval's entry, the tape and its length
assert (OPS["eval"], TAPE, TLEN) == (0x2CA, 0xC00, 0xC30), (hex(OPS["eval"]), TAPE, TLEN)
assert LABELS["code1_end"] <= A_ADDR and LABELS["eval_end"] <= TAPE, "eval's code overflows"
assert LABELS["dbuf"] + MAX_DIGITS < CODE3 and spec.FONT_BASE + 80 <= ACC0 < ACC1 + 32 <= 0x200
assert LABELS["chain_end"] <= H, "chain mode's code overflows"


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
    a_at = W_ADDR + 1 if op == "div" else A_ADDR  # div works in place on W
    w = [(a_at + i, d) for i, d in enumerate(da)] + [(B_ADDR + i, d) for i, d in enumerate(db)]
    return w + [(LEN, len(da)), (LEN + 1, len(db)), (MODE, mode), (spec.PC, OPS[op])]
