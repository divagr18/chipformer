"""Exact decimal arithmetic for the computer inside the model ("the BIOS").

Calling convention (what the circuit's CALL record writes, see bridge.py):
  A = digits of the first argument, most significant first, at A_ADDR (LA of them at LEN)
  B = digits of the second argument at B_ADDR (LB at LEN + 1)
  PC = the routine's entry
Each routine leaves the result's digits at R_ADDR and ends with PRINT (00FD), which prints V0
digits from I and halts. Leading zeros are stripped. Nothing ticks during a call, so the routines
never use the timers or wait for keys.
"""

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

ROM, LABELS = assemble(SRC)
OPS = {"add": LABELS["add"], "mul": LABELS["mul"], "sub": LABELS["sub"], "div": LABELS["div"]}
# The entries are compiled into the circuit's CALL record (chip8_q35.py), so they are part of the
# weights: moving a routine means rebuilding every checkpoint. Add new code after the entries.
assert {k: v for k, v in OPS.items() if k != "div"} == {"add": 0x202, "sub": 0x206,
                                                          "mul": 0x20A}, OPS


def call_writes(op: str, a: str, b: str, mode: int = 0) -> list[tuple[int, int]]:
    """The writes of a CALL record: the arguments' digits, their lengths, the mode, then PC."""
    da, db = [int(ch) for ch in a], [int(ch) for ch in b]
    assert 1 <= len(da) <= MAX_DIGITS and 1 <= len(db) <= MAX_DIGITS, (a, b)
    a_at = W_ADDR + 1 if op == "div" else A_ADDR  # div works in place on W
    w = [(a_at + i, d) for i, d in enumerate(da)] + [(B_ADDR + i, d) for i, d in enumerate(db)]
    return w + [(LEN, len(da)), (LEN + 1, len(db)), (MODE, mode), (spec.PC, OPS[op])]
