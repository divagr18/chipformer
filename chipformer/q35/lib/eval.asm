
; ================================================================ eval: a whole expression (v18)
; (bcd.py fills in the names in braces and generates the dispatch tables; see its notes)
;
; The tape: one code per token (digits 0..9, symbols 10..17, fused tokens 18..), END after it.
; A value lives in slot s (offset 32 s): H + 32 s = [sign, n length, d length (0: d = 1),
; n's digits], D + 32 s = d's digits. Slots 0 .. NVAL-1 are the value stack, T6 and T7 temporaries.
;
; registers while parsing: ve = tape index, vd = 32 * values, va = operators (vc = the top),
; vb = where the number being read starts (255: none; v8 = its "." + 1, or 0), v9 = 1 after an
; operand;
; in reduce v6 / v7 = the offsets of X / Y (X op Y -> X); v0..v5 are scratch.
; operators: "(" 0, + 4, - 5, * 8, / 9, negation 12, ^ 16 (precedence = code / 4)

; macros with jump targets first (they must sit below 0x400): R = A op B with the leading zeros
; skipped, so v0 = the length and I = its first digit
mulm:   ld v4, 0
        bmul0 v4
        bmul v4
        btrim
        ret
addm:   badd am_t
am_t:   btrim
        ret
subm:   ld v3, 1                ; |A - B|, v3 = [A >= B]
        bsub sm_t
        ld v3, 0
        brsub sm_t
sm_t:   btrim
        ret

; ---------------------------------------------------------------- the parser
eval:   ld i, {MODE}
        ld v0, [i]
        ld i, {MSAVE}
        ld [i], v0
        ld v1, 1
        or v0, v1               ; MODE bit 0 while it runs (so BRSUB sets v0)
        ld i, {MODE}
        ld [i], v0
        ld i, {TLEN}
        ld v0, [i]
        ld i, {TAPE}
        add i, v0
        ld v0, {END}
        ld [i], v0
        ld ve, 0
        ld vd, 0
        ld va, 0
        ld vb, 255
        ld v8, 0
        ld v9, 0
ev_lp:  ld i, {TAPE}
        add i, ve
        ld v0, [i]
        add ve, 1
        add v0, v0
        jp v0, jtab
dig:    se vb, 255
        jp ev_lp
        ld vb, ve               ; a number starts
        add vb, 255
        jp ev_lp

s_pt:   sne vb, 255             ; "."
        jp err
        se v8, 0
        jp err
        ld v8, ve
        ret
s_lp:   se vb, 255              ; "(" (right after an operand: a product)
        call commit
        se v9, 0
        call imul
        ld v0, 0
        jp push
imul:   ld v0, 8
        jp binop
s_rp:   se vb, 255              ; ")" (none open: ignored)
        call commit
rp_lp:  sne va, 0
        jp rp_d
        sne vc, 0
        jp rp_p
        call reduce
        jp rp_lp
rp_p:   call pop
rp_d:   ld v9, 1
        ret
s_add:  ld v0, 4
        jp s_op
s_sub:  ld v0, 5
        jp s_op
s_mul:  ld v0, 8
        jp s_op
s_div:  ld v0, 9
        jp s_op
s_pow:  ld v0, 16
s_op:   ld v7, v0
        se vb, 255
        call commit
        ld v0, v7
        se v9, 0
        jp binop
        sne v0, 5               ; not after an operand: "-" negates, "+" is nothing
        jp s_neg
        sne v0, 4
        ret
        jp err
s_neg:  ld v0, 12
        jp push
binop:  ld i, {CUR}             ; reduce what binds at least as tight, then push v0
        ld [i], v0
        shr v0, v0
        shr v0, v0
        ld i, {PCUR}
        ld [i], v0
bo_lp:  sne va, 0
        jp bo_push
        sne vc, 0
        jp bo_push              ; "("
        ld v1, vc
        shr v1, v1
        shr v1, v1              ; the top's precedence
        ld i, {PCUR}
        ld v0, [i]
        ld v2, v1
        sub v2, v0
        sne vf, 0
        jp bo_push              ; looser: push
        se v2, 0
        jp bo_red               ; tighter: reduce
        sne v0, 4               ; as tight: reduce, unless right-associative "^"
        jp bo_push
bo_red: call reduce
        jp bo_lp
bo_push: ld i, {CUR}
        ld v0, [i]
push:   ld i, {OPST}            ; vc = the top
        add i, va
        ld [i], v0
        add va, 1
        ld vc, v0
        ld v9, 0
        ret
pop:    add va, 255             ; (keeps v0)
        sne va, 0
        ret
        ld v2, v0
        ld v1, va
        add v1, 255
        ld i, {OPST}
        add i, v1
        ld v0, [i]
        ld vc, v0
        ld v0, v2
        ret

commit: sne vd, {VMAX}          ; push the number on the tape from vb to ve - 2
        jp err
        ld v5, ve
        add v5, 255
        sub v5, vb              ; v5 = its tokens
        ld v4, 0                ; v4 = its decimals
        sne v8, 0
        jp cm_z
        ld v4, ve               ; move the decimals left over the "."
        add v4, 255
        sub v4, v8
        ld i, {TAPE}
        add i, v8
        bld a, v4
        ld v0, v8
        add v0, 255
        ld i, {TAPE}
        add i, v0
        bsta
        add v5, 255
cm_z:   sne v5, 1               ; skip leading zeros
        jp cm_c
        ld i, {TAPE}
        add i, vb
        ld v0, [i]
        se v0, 0
        jp cm_c
        add vb, 1
        add v5, 255
        jp cm_z
cm_c:   ld v0, v5
        ld v1, {MAXD1}
        sub v0, v1
        se vf, 0
        jp err
        ld i, {TAPE}
        add i, vb
        bld a, v5
        ld i, {H3}
        add i, vd
        bsta
        ld v0, 0                ; header: sign 0, n length, d length
        ld v1, v5
        ld v2, v4
        sne v2, 0
        jp cm_e
        add v2, 1
        ld i, dbuf              ; d = 10^decimals
        bld a, v2
        ld i, {D}
        add i, vd
        bsta
cm_e:   ld i, {H}
        add i, vd
        ld [i], v2
        add vd, 32
        ld vb, 255
        ld v8, 0
        ld v9, 1
        ret

; ---------------------------------------------------------------- the end: reduce all, print
h_end:  se vb, 255
        call commit
he_lp:  sne va, 0
        jp he_d
        sne vc, 0
        jp he_p
        call reduce
        jp he_lp
he_p:   call pop
        jp he_lp
he_d:   se vd, 32
        jp err
        ld i, {H}
        ld v3, [i]              ; v0 sign, v1 n length, v2 d length, v3 n's first digit
        se v1, 1
        jp he_s
        se v3, 0
        jp he_s
        ld v0, 0                ; 0: no sign, no d
        ld v2, 0
he_s:   ld v5, v1
        ld v6, v2
        ld v7, v0
        call setmode
        se v6, 0
        jp he_div
        ld i, {H3}
        ld v0, v5
        print
he_div: ld i, {H3}              ; n / d: div prints it
        bld a, v5
        ld i, {W1}
        bsta
        ld i, {D}
        bld b, v6
        jp div
setmode: ld i, {MSAVE}          ; MODE as the CALL wrote it, | 8 if v7 (negative)
        ld v0, [i]
        ld v1, 0
        se v7, 0
        ld v1, 8
        or v0, v1
        ld i, {MODE}
        ld [i], v0
        ret
err:    ld i, {MSAVE}           ; prints "..."
        ld v0, [i]
        ld i, {MODE}
        ld [i], v0
        ld v0, 11
        ld i, {R_ADDR}
        ld [i], v0
        ld v0, 1
        print

; ---------------------------------------------------------------- reduce: the top operator
reduce: ld v0, vc
        call pop
        sne v0, 12
        jp r_neg
        sne vd, 32
        jp err
        sne vd, 0
        jp err
        ld v6, vd
        add v6, 192
        ld v7, vd
        add v7, 224
        add v0, v0
        jp v0, rtab
r_neg:  sne vd, 0
        jp err
        ld v3, vd
        add v3, 224
        ld i, {H}
        add i, v3
        ld v0, [i]
        ld v1, 1
        xor v0, v1
        ld [i], v0
        ret
pop1:   add vd, 224
        ret
code1_end:
        org {CODE2}             ; (the data from A_ADDR to the end of W sits in between)

hdrs:   ld i, {H}               ; v3 = sign(Y), DXY = d lengths (X, Y); v0 = sign(X), I = X's header
        add i, v7
        ld v2, [i]
        ld v3, v0
        ld v5, v2
        ld i, {H}
        add i, v6
        ld v2, [i]
        ld v4, v0
        ld v0, v2
        ld v1, v5
        ld i, {DXY}
        ld [i], v1
        ld v0, v4
        ld i, {H}
        add i, v6
        ret
r_mul:  call hdrs
        xor v0, v3
        ld [i], v0              ; sign(X) ^= sign(Y)
        ld v3, v6
        call na
        ld v3, v7
        call nb
        call mulm
        ld v3, v6
        call rs_n               ; n(X) = n(X) n(Y)
        ld i, {DXY}
        ld v1, [i]
        sne v1, 0
        jp pop1                 ; d(Y) = 1
        se v0, 0
        jp rm_dd
        call dcp                ; d(X) = 1: d(X) = d(Y)
        jp pop1
rm_dd:  ld v3, v6
        call da
        ld v3, v7
        call db
        call mulm
        ld v3, v6
        call rs_d
        jp pop1

r_div:  ld i, {H1}              ; Y = 0: "..."
        add i, v7
        ld v2, [i]
        se v0, 1
        jp dv_ok
        sne v2, 0
        jp err
dv_ok:  call hdrs
        xor v0, v3
        ld [i], v0
        ld i, {DXY}
        ld v1, [i]
        sne v1, 0
        jp dv_d
        ld v3, v6               ; n(X) = n(X) d(Y)
        call na
        ld v3, v7
        call db
        call mulm
        ld v3, v6
        call rs_n
dv_d:   ld i, {DXY}
        ld v0, [i]
        se v0, 0
        jp dv_dd
        call ndcp               ; d(X) = 1: d(X) = n(Y)
        jp pop1
dv_dd:  ld v3, v6               ; d(X) = d(X) n(Y)
        call da
        ld v3, v7
        call nb
        call mulm
        ld v3, v6
        call rs_d
        jp pop1

r_sub:  ld i, {H}               ; X - Y = X + (-Y)
        add i, v7
        ld v0, [i]
        ld v1, 1
        xor v0, v1
        ld [i], v0
r_add:  call hdrs
        ld v1, v3
        ld i, {SGN}
        ld [i], v1              ; SGN = sign(X), sign(Y)
        ld i, {DXY}
        ld v1, [i]
        or v0, v1
        se v0, 0
        jp ad_fr
        ld v3, v6               ; whole numbers
        call na
        ld v3, v7
        call nb
        jp sadd
ad_fr:  ld v3, v6               ; T6 = n(X) d(Y)
        call na
        ld i, {DXY}
        ld v1, [i]
        sne v1, 0
        jp a1_n
        ld v3, v7
        call db
        call mulm
        ld v3, {T6}
        call rs_n
        jp a2
a1_n:   ld v3, {T6}
        call ast_n
a2:     ld v3, v7               ; T7 = n(Y) d(X)
        call na
        ld i, {DXY}
        ld v0, [i]
        sne v0, 0
        jp a2_n
        ld v3, v6
        call db
        call mulm
        ld v3, {T7}
        call rs_n
        jp a3
a2_n:   ld v3, {T7}
        call ast_n
a3:     ld i, {DXY}             ; d(X) = d(X) d(Y)
        ld v1, [i]
        sne v1, 0
        jp a4
        se v0, 0
        jp a3_m
        call dcp
        jp a4
a3_m:   ld v3, v6
        call da
        ld v3, v7
        call db
        call mulm
        ld v3, v6
        call rs_d
a4:     ld v3, {T6}
        call na
        ld v3, {T7}
        call nb
sadd:   ld i, {SGN}             ; n(X) = sign(X) A + sign(Y) B
        ld v1, [i]
        se v0, v1
        jp sa_df
        call addm
        ld v3, v6
        call rs_n
        jp pop1
sa_df:  call subm
        ld v5, v3
        ld v3, v6
        call rs_n
        se v5, 0
        jp pop1                 ; |A| >= |B|: X's sign
        ld i, {SGN1}
        ld v0, [i]
        ld i, {H}
        add i, v6
        ld [i], v0              ; else Y's
        jp pop1

r_pow:  ld i, {H}               ; Y: a whole number of at most 2 digits
        add i, v7
        ld v4, [i]              ; v0 sign, v1 length, v2 d length, v3 v4 digits
        se v2, 0
        jp err
        ld v5, v0
        ld v0, v1
        ld v2, 3
        sub v0, v2
        se vf, 0
        jp err
        se v1, 1
        jp pw_2
        ld v4, v3
        ld v3, 0
pw_2:   ld v0, v3               ; e = 10 v3 + v4
        shl v0, v0
        ld v1, v0
        shl v1, v1
        shl v1, v1
        add v0, v1
        add v0, v4
        ld v1, v5
        ld i, {EXP}
        ld [i], v1              ; EXP = e, EXP + 1 = [e < 0]
        ld v3, v6               ; T6 = X
        call na
        ld v3, {T6}
        call ast_n
        ld i, {H2}
        add i, v6
        ld v0, [i]
        ld i, {T6_DL}
        ld [i], v0
        sne v0, 0
        jp pw_x
        ld i, {D}
        add i, v6
        bld a, v0
        ld i, {T6_D}
        bsta
pw_x:   ld i, {EXP}
        ld v0, [i]
        se v0, 0
        jp pw_s
        ld v0, 0                ; e = 0: X = 1
        ld v1, 1
        ld v2, 0
        ld v3, 1
        ld i, {H}
        add i, v6
        ld [i], v3
        jp pop1
pw_s:   ld v1, 1                ; even e: no sign
        and v0, v1
        se v0, 0
        jp pw_lp
        ld i, {H}
        add i, v6
        ld [i], v0
pw_lp:  ld i, {EXP}
        ld v0, [i]
        add v0, 255
        ld [i], v0
        sne v0, 0
        jp pw_e
        ld v3, v6               ; X = X * T6
        call na
        ld v3, {T6}
        call nb
        call mulm
        ld v3, v6
        call rs_n
        ld i, {T6_DL}
        ld v0, [i]
        sne v0, 0
        jp pw_lp
        ld v3, v6
        call da
        ld v3, {T6}
        call db
        call mulm
        ld v3, v6
        call rs_d
        jp pw_lp
pw_e:   ld i, {EXP1}
        ld v0, [i]
        se v0, 0
        call recip
        jp pop1
recip:  ld i, {H1}              ; X = 1 / X; X = 0: "..."
        add i, v6
        ld v2, [i]              ; v0 n length, v1 d length, v2 n's first digit
        se v0, 1
        jp rc_ok
        sne v2, 0
        jp err
rc_ok:  ld v3, v6               ; T7 = n(X)
        call na
        ld v3, {T7}
        call ast_n
        ld i, {H2}
        add i, v6
        ld v0, [i]
        se v0, 0
        jp rc_d
        ld v0, 1                ; d(X) = 1: n(X) = 1
        ld i, {H1}
        add i, v6
        ld [i], v0
        ld i, {H3}
        add i, v6
        ld [i], v0
        jp rc_n
rc_d:   ld i, {D}               ; n(X) = d(X)
        add i, v6
        bld a, v0
        ld i, {H3}
        add i, v6
        bsta
        ld i, {H1}
        add i, v6
        ld [i], v0
rc_n:   ld i, {T7_NL}           ; d(X) = T7
        ld v0, [i]
        ld i, {T7_N}
        bld a, v0
        ld i, {D}
        add i, v6
        bsta
        ld i, {H2}
        add i, v6
        ld [i], v0
        ret

; ---------------------------------------------------------------- moves (slot offset v3)
na:     ld i, {H1}              ; A = n(slot)
        add i, v3
        ld v0, [i]
        ld i, {H3}
        add i, v3
        bld a, v0
        ret
nb:     ld i, {H1}              ; B = n(slot)
        add i, v3
        ld v0, [i]
        ld i, {H3}
        add i, v3
        bld b, v0
        ret
da:     ld i, {H2}              ; A = d(slot)
        add i, v3
        ld v0, [i]
        ld i, {D}
        add i, v3
        bld a, v0
        ret
db:     ld i, {H2}              ; B = d(slot)
        add i, v3
        ld v0, [i]
        ld i, {D}
        add i, v3
        bld b, v0
        ret
ast_n:  ld i, {LEN}             ; n(slot) = A
        ld v0, [i]
        ld i, {H3}
        add i, v3
        bsta
        ld i, {H1}
        add i, v3
        ld [i], v0
        ret
rs_n:   call ovf                ; n(slot) = R
        bld a, v0
        ld i, {H3}
        add i, v3
        bsta
        ld i, {H1}
        add i, v3
        ld [i], v0
        ret
rs_d:   call ovf                ; d(slot) = R
        bld a, v0
        ld i, {D}
        add i, v3
        bsta
        ld i, {H2}
        add i, v3
        ld [i], v0
        ret
ovf:    ld v1, v0               ; over MAX_DIGITS digits: "..."
        ld v2, {MAXD1}
        sub v1, v2
        se vf, 0
        jp err
        ret
dcp:    ld i, {H2}              ; d(X) = d(Y)
        add i, v7
        ld v0, [i]
        ld i, {D}
        add i, v7
        bld a, v0
        ld i, {D}
        add i, v6
        bsta
        ld i, {H2}
        add i, v6
        ld [i], v0
        ret
ndcp:   ld i, {H1}              ; d(X) = n(Y)
        add i, v7
        ld v0, [i]
        ld i, {H3}
        add i, v7
        bld a, v0
        ld i, {D}
        add i, v6
        bsta
        ld i, {H2}
        add i, v6
        ld [i], v0
        ret

; ---------------------------------------------------------------- dispatch
{HANDLERS}
jtab:
{JTAB}
rtab:
{RTAB}
eval_end:

        org {XTAB}
dbuf:   db {DBUF_DB}
