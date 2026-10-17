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
DSTOP = LEN + 3  # BDIV / BADB stop at t = LA + DSTOP (div: DEC; mod, gcd: 0)
# div: W = 0, A's digits (the CALL record writes them at W_ADDR + 1), DEC zeros; the quotient goes
# to R_ADDR as P[t] (t <= LA), "." (10) at P[LA + 1], P[t + 1] (t > LA), "..." (11) if inexact
W_ADDR = 0x6BF
DEC = 10  # decimals of an inexact quotient
TMP = 0x700  # gcd: the remainder while A and B move
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

; ---------------------------------------------------------------- mod: A mod B
; (A at W + 1, as for div); nothing if B starts with 0
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
        ld vc, ve               ; vc = the remainder's length (the next LB), ve = LB (the next LA)
        sub vc, vd
        add vc, 1
        ld ve, vb
        ld i, {W_ADDR}          ; TMP = the remainder (copies move 6, 12 or 24 digits)
        add i, vd
        ld v0, 6
        sub v0, vc
        se vf, 0
        jp gr_6
        ld vb, [i]
        ld i, {TMP}
        ld [i], vb
        ld v0, 12
        sub v0, vc
        se vf, 1
        call g_r2
        jp gr_d
gr_6:   ld v5, [i]
        ld i, {TMP}
        ld [i], v5
gr_d:   ld i, {B_ADDR}          ; W + 1 = B (the next A)
        ld v0, 6
        sub v0, ve
        se vf, 0
        jp gb_6
        ld vb, [i]
        ld i, {W_ADDR + 1}
        ld [i], vb
        ld v0, 12
        sub v0, ve
        se vf, 1
        call g_b2
        jp gb_d
gb_6:   ld v5, [i]
        ld i, {W_ADDR + 1}
        ld [i], v5
gb_d:   ld i, {TMP}             ; B = TMP (the next B)
        ld v0, 6
        sub v0, vc
        se vf, 0
        jp gt_6
        ld vb, [i]
        ld i, {B_ADDR}
        ld [i], vb
        ld v0, 12
        sub v0, vc
        se vf, 1
        call g_t2
        jp gt_d
gt_6:   ld v5, [i]
        ld i, {B_ADDR}
        ld [i], v5
gt_d:   ld v0, ve
        ld v1, vc
        ld i, {LEN}
        ld [i], v1              ; LA, LB
        ld va, ve
        ld vb, vc
        jp gcd_lp
gcd_b:  ld i, {B_ADDR}
        ld v0, vb
        print
g_r2:   ld i, {W_ADDR + 12}     ; the second 12 digits of a copy
        add i, vd
        ld vb, [i]
        ld i, {TMP + 12}
        ld [i], vb
        ret
g_b2:   ld i, {B_ADDR + 12}
        ld vb, [i]
        ld i, {W_ADDR + 13}
        ld [i], vb
        ret
g_t2:   ld i, {TMP + 12}
        ld vb, [i]
        ld i, {B_ADDR + 12}
        ld [i], vb
        ret

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
zeros:  db 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

"""

ROM, LABELS = assemble(SRC)
OPS = {op: LABELS[op] for op in ("add", "mul", "sub", "div", "mod", "gcd")}
# The entries are compiled into the circuit's CALL record (chip8_q35.py), so they are part of the
# weights: moving a routine means rebuilding every checkpoint. Add new code after the entries.
assert {k: OPS[k] for k in ("add", "sub", "mul")} == {"add": 0x202, "sub": 0x206, "mul": 0x20A}, OPS
W_OPS = ("div", "mod", "gcd")  # A goes to W + 1: they work on it in place


def call_writes(op: str, a: str, b: str, mode: int = 0) -> list[tuple[int, int]]:
    """The writes of a CALL record: the arguments' digits, their lengths, the mode, then PC."""
    da, db = [int(ch) for ch in a], [int(ch) for ch in b]
    assert 1 <= len(da) <= MAX_DIGITS and 1 <= len(db) <= MAX_DIGITS, (a, b)
    a_at = W_ADDR + 1 if op in W_OPS else A_ADDR
    w = [(a_at + i, d) for i, d in enumerate(da)] + [(B_ADDR + i, d) for i, d in enumerate(db)]
    return w + [(LEN, len(da)), (LEN + 1, len(db)), (MODE, mode), (spec.PC, OPS[op])]
