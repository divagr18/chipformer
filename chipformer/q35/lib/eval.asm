
; ================================================================ eval: a whole expression (v18)
; (bcd.py fills in the names in braces and generates the dispatch tables; see its notes)
;
; The tape: one code per token (digits 0..9, symbols 10..17, fused tokens 18..), END after it.
; A value lives in slot s (offset 32 s): H + 32 s = [sign, n length, n's place, d length (0: d = 1),
; n's digits]; n's place 0: its digits follow (H4 + 32 s), else they are a number on the tape
; (TAPE + place - 1); D + 32 s = d's digits. Slots 0 .. NVAL-1 are the value stack, T6 and T7
; temporaries.
;
; registers while parsing: ve = tape index, vd = 32 * values, va = operators (vc = the top),
; vb = where the number being read starts (255: none; v8 = its "." + 1, or 0), v9 = 1 after an
; operand; in reduce v6 / v7 = the offsets of X / Y (X op Y -> X); v0..v5 are scratch.
; operators: "(" 0, + 4, - 5, * 8, / 9, negation 12, ^ 16. An operator reduces the ones on top
; of the stack while they are >= its threshold (+ - 4, * / 8, ^ 17: right-associative).

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
        jp f_init               ; chain mode first (it falls back to g_init)
g_init: ld ve, 0                ; the general evaluator
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

s_pt:   sne vd, 1               ; "." (chain mode: the general evaluator)
        jp g_init
        sne vb, 255
        jp err
        se v8, 0
        jp err
        ld v8, ve
        ret
s_lp:   sne vd, 1               ; "(" (right after an operand: a product)
        jp f_lp
        ld v7, 0
        se vb, 255
        call commit
        se v9, 0
        call imul
        ld v0, 0
        jp push
imul:   ld v0, 8
        ld v1, 8
        ld i, {CUR}
        ld [i], v1
        jp bo_lp
s_rp:   sne vd, 1               ; ")" (none open: ignored)
        jp f_rp
        ld v7, 0
        se vb, 255
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
        ld v1, 4
        jp s_op
s_sub:  ld v0, 5
        ld v1, 4
        jp s_op
s_mul:  ld v0, 8
        ld v1, 8
        jp s_op
s_div:  ld v0, 9
        ld v1, 8
        jp s_op
s_pow:  sne vd, 1
        jp g_init
        ld v0, 16
        ld v1, 17
s_op:   sne vd, 1
        jp f_op
        ld i, {CUR}             ; CUR = the operator, its threshold
        ld [i], v1
        ld v7, v0
        se vb, 255
        call commit
        se v9, 0
        jp bo_lp
        sne v7, 5               ; not after an operand: "-" negates, "+" is nothing
        jp s_neg
        sne v7, 4
        ret
        jp err
s_neg:  ld v0, 12
        jp push
bo_lp:  ld i, {CUR}             ; reduce the operators on top >= the threshold, then push
        ld v1, [i]
        sne va, 0
        jp push
        ld v2, vc
        sub v2, v1
        sne vf, 0
        jp push
        call reduce
        jp bo_lp
push:   ld v1, v0               ; push v0: memory keeps the one below, vc the top
        ld v0, vc
        ld i, {OPST}
        add i, va
        ld [i], v0
        add va, 1
        ld vc, v1
        ld v9, 0
        ret
pop:    add va, 255             ; vc = the one below
        ld i, {OPST}
        add i, va
        ld v0, [i]
        ld vc, v0
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
        ld v2, v4
        sne v2, 0
        jp cm_n
        add v2, 1
        ld i, dbuf              ; d = 10^decimals
        bld a, v2
        ld i, {D}
        add i, vd
        bsta
cm_n:   ld v3, 0                ; a negation right before it (not before "^") folds into it
        sne va, 0
        jp cm_h
        se vc, 12
        jp cm_h
        sne v7, 16
        jp cm_h
        ld v3, 1
        call pop
cm_h:   ld v0, v3               ; header: sign, n length, place, d length
        ld v1, v5
        ld v3, v2
        ld v2, vb
        add v2, 1
        ld i, {H}
        add i, vd
        ld [i], v3
        add vd, 32
        ld vb, 255
        ld v8, 0
        ld v9, 1
        ret

; ---------------------------------------------------------------- the end: reduce all, print
h_end:  sne vd, 1
        jp f_end
        ld v7, 0
        se vb, 255
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
        ld v3, 0
        call na                 ; A = n (v0 = its length)
        ld i, {H}
        ld v3, [i]              ; v0 sign, v1 n length, v2 place, v3 d length
        ld v7, v0
        ld v5, v1
        ld v6, v3
        ld i, {A_ADDR}
        ld v0, [i]
        se v5, 1
        jp he_s
        se v0, 0
        jp he_s
        ld v7, 0                ; 0: no sign, no d
        ld v6, 0
he_s:   call setmode
        se v6, 0
        jp he_div
        ld i, {A_ADDR}
        ld v0, v5
        print
he_div: ld i, {W1}              ; n / d: div prints it
        bsta
        ld i, {D}
        bld b, v6
        jp div
setmode: ld i, {MSAVE}          ; MODE as the CALL wrote it, | 8 if v7 (negative)
        ld v0, [i]
        ld v1, 0
        sne v7, 0
        jp sm_m
        ld v1, 1                ; a tool response has no "-": "..." instead
        and v1, v0
        sne v1, 0
        jp err
        ld v1, 8
sm_m:
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
reduce: ld v5, vc
        call pop
        ld v0, v5
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
{CHAIN1}
code1_end:
        org {CODE2}             ; (the data from A_ADDR to the end of W sits in between)

hdrs:   ld i, {H}               ; v3 = sign(Y), DXY = d lengths (X, Y); v0 = sign(X), I = X's header
        add i, v7
        ld v3, [i]
        ld v4, v0
        ld v5, v3
        ld i, {H}
        add i, v6
        ld v3, [i]
        ld v2, v0
        ld v0, v3
        ld v1, v5
        ld i, {DXY}
        ld [i], v1
        ld v0, v2
        ld v3, v4
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

r_div:  ld v3, v7               ; Y = 0: "..."
        call nb
        se v0, 1
        jp dv_ok
        ld i, {B_ADDR}
        ld v0, [i]
        sne v0, 0
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
        ld v3, [i]              ; v0 sign, v1 length, v2 place, v3 d length
        se v3, 0
        jp err
        ld v5, v0
        ld v0, v1
        ld v2, 3
        sub v0, v2
        se vf, 0
        jp err
        ld v3, v7
        call nb                 ; B = its digits, v0 = how many
        ld v4, v0
        ld i, {B_ADDR}
        ld v1, [i]
        se v4, 1
        jp pw_3
        ld v1, v0               ; one digit: e = B[0]
        ld v0, 0
pw_3:   ld v2, v0               ; e = 10 v0 + v1
        shl v2, v2
        ld v3, v2
        shl v3, v3
        shl v3, v3
        add v2, v3
        add v2, v1
        ld v0, v2
        ld v1, v5
        ld i, {EXP}
        ld [i], v1              ; EXP = e, EXP + 1 = [e < 0]
        ld v3, v6               ; T6 = X
        call na
        ld v3, {T6}
        call ast_n
        ld i, {H3}
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
        ld v3, 0
        ld v4, 1
        ld i, {H}
        add i, v6
        ld [i], v4
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
recip:  ld v3, v6               ; X = 1 / X; X = 0: "..."
        call na                 ; T7 = n(X)
        se v0, 1
        jp rc_ok
        ld i, {A_ADDR}
        ld v0, [i]
        sne v0, 0
        jp err
rc_ok:  ld v3, {T7}
        call ast_n
        ld i, {H3}
        add i, v6
        ld v0, [i]
        se v0, 0
        jp rc_d
        ld v0, 1                ; d(X) = 1: n(X) = 1
        ld i, {H4}
        add i, v6
        ld [i], v0
        ld v0, 1
        ld v1, 0
        jp rc_h
rc_d:   ld i, {D}               ; n(X) = d(X)
        add i, v6
        bld a, v0
        ld i, {H4}
        add i, v6
        bsta
        ld v1, 0
rc_h:   ld i, {H1}
        add i, v6
        ld [i], v1              ; n length, place 0
        ld i, {T7_NL}           ; d(X) = T7
        ld v0, [i]
        ld i, {T7_N}
        bld a, v0
        ld i, {D}
        add i, v6
        bsta
        ld i, {H3}
        add i, v6
        ld [i], v0
        ret

; ---------------------------------------------------------------- moves (slot offset v3)
na:     ld i, {H1}              ; A = n(slot), v0 = its length
        add i, v3
        ld v1, [i]
        sne v1, 0
        jp na_s
        ld i, {TAPE_M1}
        add i, v1
        bld a, v0
        ret
na_s:   ld i, {H4}
        add i, v3
        bld a, v0
        ret
nb:     ld i, {H1}              ; B = n(slot), v0 = its length
        add i, v3
        ld v1, [i]
        sne v1, 0
        jp nb_s
        ld i, {TAPE_M1}
        add i, v1
        bld b, v0
        ret
nb_s:   ld i, {H4}
        add i, v3
        bld b, v0
        ret
da:     ld i, {H3}              ; A = d(slot)
        add i, v3
        ld v0, [i]
        ld i, {D}
        add i, v3
        bld a, v0
        ret
db:     ld i, {H3}              ; B = d(slot)
        add i, v3
        ld v0, [i]
        ld i, {D}
        add i, v3
        bld b, v0
        ret
ast_n:  ld i, {LEN}             ; n(slot) = A
        ld v0, [i]
        ld i, {H4}
        add i, v3
        bsta
        ld v1, 0
        ld i, {H1}
        add i, v3
        ld [i], v1
        ret
rs_n:   ld v1, v0               ; n(slot) = R (over MAX_DIGITS digits: "...")
        ld v2, {MAXD1}
        sub v1, v2
        se vf, 0
        jp err
        bld a, v0
        ld i, {H4}
        add i, v3
        bsta
        ld v1, 0
        ld i, {H1}
        add i, v3
        ld [i], v1
        ret
rs_d:   ld v1, v0               ; d(slot) = R
        ld v2, {MAXD1}
        sub v1, v2
        se vf, 0
        jp err
        bld a, v0
        ld i, {D}
        add i, v3
        bsta
        ld i, {H3}
        add i, v3
        ld [i], v0
        ret
dcp:    ld i, {H3}              ; d(X) = d(Y)
        add i, v7
        ld v0, [i]
        ld i, {D}
        add i, v7
        bld a, v0
        ld i, {D}
        add i, v6
        bsta
        ld i, {H3}
        add i, v6
        ld [i], v0
        ret
ndcp:   ld v3, v7               ; d(X) = n(Y)
        call na
        ld i, {D}
        add i, v6
        bsta
        ld i, {H3}
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

        org {CODE3}
{CHAIN2}
chain_end:
