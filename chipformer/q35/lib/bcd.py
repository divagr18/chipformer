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
# "$$", | 8 the result is negative (set by BRSUB and the signed routines), | 16 / 32 A / B
# are negative (set by the CALL record)
MODE = LEN + 2
DSTOP = LEN + 3  # BDIV / BADB stop at t = LA + DSTOP (div: DEC; mod, gcd: 0)
IA = LEN + 4  # the digits of A before its point (LA without one), then B's at IA + 1
# div: W = 0, A's digits (the CALL record writes them at W_ADDR + 1), DEC zeros; the quotient goes
# to R_ADDR as P[t] (t <= LA), "." (10) at P[LA + 1], P[t + 1] (t > LA), "..." (11) if inexact
W_ADDR = 0x6BF
DEC = 10  # decimals of an inexact quotient
TMP = 0x700  # gcd: the remainder while A and B move
EXT_ORG = 0x720  # where the signed / decimal routines start
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

; signed or decimal add / sub (as_x): where BADD / BSUB / BRSUB jump (vd: negative)
x_fr:   ld vb, 16               ; BRSUB: |B| - |A|, the other sign
        xor vd, vb
x_fs:   jp x_fin

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
de_t:   se ve, va               ; no trailing zeros after the "." (A = 0, or A padded)
        jp de_t1
        jp div_pr
de_t1:  ld i, {R_ADDR}
        add i, ve
        ld v0, [i]
        se v0, 0
        jp div_pr
        add ve, 255
        ld v0, va
        add v0, 1
        se ve, v0
        jp de_t
        add ve, 255             ; only the "." was left
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
        ld v0, [i]              ; the first digit
        ld v1, ve
        sub v1, v6              ; the digits - 1
        se v1, 0
        jp dp_pr
        se v0, 0
        jp dp_pr
        ld i, {MODE}            ; 0 has no sign
        ld v0, [i]
        ld v2, 7
        and v0, v2
        ld [i], v0
dp_pr:  ld i, {R_ADDR}
        add i, v6
        ld v0, v1
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

        org {EXT_ORG}             ; (above W and TMP: room below 0x600 is short)
; ---------------------------------------------------------------- signed / decimal operands
; The CALL record picks these entries when A or B has a sign (MODE bit 16 / 32) or a point
; (IA / IB: the digits before it; LA / LB without one). Results print normalised: no leading
; zeros before the point, no trailing zeros after it (nor the point itself), "-" if below 0.
xargs:  ld i, {LEN}             ; va = LA, vb = LB, v6 = fa, v7 = fb, v2 = MODE,
        ld v5, [i]              ; v8 = the signs (16: A negative, 32: B negative)
        ld va, v0
        ld vb, v1
        ld v6, v0
        sub v6, v4
        ld v7, v1
        sub v7, v5
        ld v8, 48
        and v8, v2
        ret

; add / sub: zeros after the shorter fraction, then |A| + |B| or |A| - |B|
sub_x:  ld vc, 32               ; a - b = a + (-b)
        jp as_x
add_x:  ld vc, 0
as_x:   call xargs
        xor v8, vc
        ld vd, 16               ; the result's sign: A's (x_fr flips it)
        and vd, v8
        ld vc, v7               ; f = fb ...
        sub v7, v6              ; VF = [fb >= fa]
        se vf, 1
        jp as_b
        ld v6, va               ; A gets fb - fa zeros
        add va, v7
        call zp_a
        ld v0, {MAX_DIGITS}
        sub v0, va
        jp as_w
as_b:   ld vc, v6               ; ... or fa, and B gets fa - fb zeros
        ld v0, 0
        sub v0, v7
        ld v7, v0
        ld v6, vb
        add vb, v7
        call zp_b
        ld v0, {MAX_DIGITS}
        sub v0, vb
as_w:   se vf, 1
        jp div_z                ; too long: nothing
        ld v0, va
        ld v1, vb
        ld i, {LEN}
        ld [i], v1
        ld ve, va               ; the result's length: max(LA, LB) + 1
        ld v0, vb
        sub v0, va
        se vf, 0
        ld ve, vb
        add ve, 1
        se v8, 0
        jp as_t
as_s:   badd x_fs               ; same signs: |A| + |B|
as_t:   se v8, 48
        jp as_d
        jp as_s
as_d:   bsub x_fs               ; |A| - |B| >= 0: A's sign
        brsub x_fr              ; else |B| - |A|: the other sign

; mul: f = fa + fb, negative if exactly one of A, B is (v8 + 16 has bit 32)
mul_x:  call xargs
        ld vc, v6
        add vc, v7
        ld vd, v8
        add vd, 16
        ld v0, 32
        and vd, v0
        ld ve, va
        add ve, vb
        ld v1, 0
        bmul0 v1
        bmul v1
        jp x_fin

; div: A * 10^fb / (B * 10^fa) (zeros after A or B, B's leading zeros go), the sign, then div
div_x:  call xargs
        ld v9, v2               ; (MODE: zp_ and bshift use v0 .. v7)
        sub v7, v6              ; VF = [fb >= fa]
        se vf, 1
        jp dx_b
        ld v6, va               ; A (at W + 1) gets fb - fa zeros
        add va, v7
        call zp_w
        ld v0, {MAX_DIGITS}
        sub v0, va
        jp dx_w
dx_b:   ld v0, 0                ; B gets fa - fb zeros
        sub v0, v7
        ld v7, v0
        ld v6, vb
        add vb, v7
        call zp_b
        ld v0, {MAX_DIGITS}
        sub v0, vb
dx_w:   se vf, 1
        jp div_z
        ld vc, 0                ; vc: B's leading zeros
dx_z:   ld i, {B_ADDR}
        add i, vc
        ld v0, [i]
        se v0, 0
        jp dx_s
        add vc, 1
        se vc, vb
        jp dx_z
        jp div_z                ; B = 0
dx_s:   sub vb, vc
        se vc, 0
        call bshift
        ld v0, va
        ld v1, vb
        ld v2, v9
        ld v3, v8               ; MODE & 7, | 8 if exactly one of A, B is negative
        add v3, 16
        ld v4, 32
        and v3, v4
        ld v4, 7
        and v2, v4
        se v3, 0
        add v2, 8
        ld i, {LEN}
        ld [i], v2              ; LA, LB, MODE
        jp div

; bshift: B[0 ..) = B[vc ..) (vb digits), 6 at a time
bshift: ld v6, vc
        ld v7, 0
bs_lp:  ld i, {B_ADDR}
        add i, v6
        ld v5, [i]
        ld i, {B_ADDR}
        add i, v7
        ld [i], v5
        add v6, 6
        add v7, 6
        ld v0, v7
        sub v0, vb              ; VF = [done >= LB]
        se vf, 1
        jp bs_lp
        ret

; zp_a / zp_b / zp_w: v7 zeros at A / B / W + 1, from offset v6 (6 at a time: each region has
; room after its 24 digits)
zp_a:   se v7, 0
        jp za_go
        ret
za_go:  ld i, zeros
        ld v5, [i]
        ld i, {A_ADDR}
        add i, v6
        ld [i], v5
        add v6, 6
        ld v0, 6
        sub v0, v7
        se vf, 0
        ret
        add v7, 250
        jp za_go
zp_b:   se v7, 0
        jp zb_go
        ret
zb_go:  ld i, zeros
        ld v5, [i]
        ld i, {B_ADDR}
        add i, v6
        ld [i], v5
        add v6, 6
        ld v0, 6
        sub v0, v7
        se vf, 0
        ret
        add v7, 250
        jp zb_go
zp_w:   se v7, 0
        jp zw_go
        ret
zw_go:  ld i, zeros
        ld v5, [i]
        ld i, {W_ADDR + 1}
        add i, v6
        ld [i], v5
        add v6, 6
        ld v0, 6
        sub v0, v7
        se vf, 0
        ret
        add v7, 250
        jp zw_go

; x_fin: the result's v0 digits end at R + ve, the last vc of them after the point; vd: below 0
x_fin:  ld va, v0
xf_t:   se vc, 0                ; trailing zeros after the point go
        jp xf_t1
        jp xf_l
xf_t1:  ld i, {R_ADDR - 1}
        add i, ve
        ld v0, [i]
        se v0, 0
        jp xf_l
        add ve, 255
        add va, 255
        add vc, 255
        jp xf_t
xf_l:   ld v6, ve               ; leading zeros before the point go (BTRIM keeps one digit)
        sub v6, va
        ld i, {R_ADDR}
        add i, v6
        ld v0, va
        sub v0, vc
        btrim
        ld va, v0
        add va, vc
        se vc, 0
        call x_dot
        ld v6, ve               ; v6: where the print starts
        sub v6, va
        ld i, {R_ADDR}
        add i, v6
        ld v0, [i]
        se va, 1                ; a single 0 has no sign
        jp xf_s
        se v0, 0
        jp xf_s
        ld vd, 0
xf_s:   ld i, {MODE}
        ld v0, [i]
        ld v1, 7
        and v0, v1
        se vd, 0
        add v0, 8
        ld [i], v0
        ld i, {R_ADDR}
        add i, v6
        ld v0, va
        print

; x_dot: a "." (10) before the last vc digits: they move right one cell, 6 at a time from the end
x_dot:  ld v7, ve               ; v7: the fraction's start, v8: a chunk's, v9: digits from v8 on
        sub v7, vc
        ld v8, v7
        ld v9, vc
xd_n:   ld v0, 6
        sub v0, v9
        se vf, 0
        jp xd_m
        add v8, 6
        add v9, 250
        jp xd_n
xd_m:   ld i, {R_ADDR}
        add i, v8
        ld v5, [i]
        ld i, {R_ADDR + 1}
        add i, v8
        ld [i], v5
        se v8, v7
        jp xd_p
        ld i, {R_ADDR}
        add i, v7
        ld v0, 10
        ld [i], v0
        add va, 1
        add ve, 1
        ret
xd_p:   add v8, 250
        jp xd_m


"""

ROM, LABELS = assemble(SRC)
OPS = {op: LABELS[op] for op in ("add", "mul", "sub", "div", "mod", "gcd")}
# The entries are compiled into the circuit's CALL record (chip8_q35.py), so they are part of the
# weights: moving a routine means rebuilding every checkpoint. Add new code after the entries.
assert {k: OPS[k] for k in ("add", "sub", "mul")} == {"add": 0x202, "sub": 0x206, "mul": 0x20A}, OPS
# the entries for signed or decimal operands (gcd ignores signs; mod takes neither)
OPS_X = {"add": LABELS["add_x"], "sub": LABELS["sub_x"], "mul": LABELS["mul_x"],
         "div": LABELS["div_x"], "mod": OPS["mod"], "gcd": OPS["gcd"]}
W_OPS = ("div", "mod", "gcd")  # A goes to W + 1: they work on it in place
assert max(LABELS["x_fs"], LABELS["x_fr"]) < 0x400  # macro jump targets
assert LABELS["zeros"] + 10 <= A_ADDR and 0x200 + len(ROM) <= 0x1000, hex(LABELS["zeros"])
assert EXT_ORG >= TMP + 24 and EXT_ORG >= W_ADDR + 2 + MAX_DIGITS + DEC


def call_writes(op: str, a: str, b: str, mode: int = 0) -> list[tuple[int, int]]:
    """The writes of a CALL record: the arguments' digits, their lengths, the mode, the digits
    before each point (IA, IB), then PC.

    a and b are numbers as written ("-12.5"): their signs go to MODE bits 16 / 32, and a sign
    or a point anywhere picks the routine's OPS_X entry."""
    def parse(s):
        neg, s = s.startswith("-"), s.lstrip("-")
        whole, _, frac = s.partition(".")
        return neg, "." in s, [int(ch) for ch in whole + frac], len(whole)

    (na, pa, da, ia), (nb, pb, db, ib) = parse(a), parse(b)
    assert 1 <= len(da) <= MAX_DIGITS and 1 <= len(db) <= MAX_DIGITS, (a, b)
    a_at = W_ADDR + 1 if op in W_OPS else A_ADDR
    w = [(a_at + i, d) for i, d in enumerate(da)] + [(B_ADDR + i, d) for i, d in enumerate(db)]
    entry = OPS_X[op] if na or nb or pa or pb else OPS[op]
    mode |= 16 * na | 32 * nb
    return w + [(LEN, len(da)), (LEN + 1, len(db)), (MODE, mode), (IA, ia), (IA + 1, ib),
                (spec.PC, entry)]
