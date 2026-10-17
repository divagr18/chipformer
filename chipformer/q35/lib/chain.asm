
; ================================================================ eval's chain mode (v18)
; Most expressions are chains of one precedence: only * and / or only + and -, with brackets
; and signs. Their value is (-1)^GP ACC0 / ACC1 (products: every number multiplies the
; numerator ACC0 or the denominator ACC1) or ACC0 - ACC1 (sums: every number adds to the
; positive or the negative terms), so each number costs one macro and no operator stack.
; Anything else (a ".", "^", the other kind of operator, a 0 in a product, an implicit product,
; more than MAX_DIGITS digits) restarts the general evaluator (g_init), which is the reference.
;
; registers: vd = 1 (chain mode), v7 = mode (0 undecided, 1 products, 2 sums), vc = the current
; bracket's bits and v8 = the next operand's (bit 0: negated (sums), bit 1: a divisor
; (products)), va = open brackets (their vc at FST), v6 = the accumulator A holds (its copy in
; memory is stale; 255: none), vb, ve, v9 as in the general evaluator. The first number waits
; in PEND until the first operator decides the mode.

f_init: ld v0, 0
        ld v1, 0
        ld v2, 0
        ld v3, 0
        ld v4, 0
        ld i, {GP}
        ld [i], v4              ; GP, L0, L1 = 0, PEND = none
        ld vd, 1
        ld v6, 255
        ld v7, 0
        ld vc, 0
        ld v8, 0
        ld v9, 0
        ld va, 0
        ld vb, 255
        ld ve, 0
        jp ev_lp

fcons:  sne vb, 255             ; the number that just ended, if any
        ret
        se v9, 0
        jp g_init               ; right after an operand
        ld v5, ve
        add v5, 255
        sub v5, vb              ; v5 = its digits (vb: the first)
        ld i, {TAPE}
        add i, vb
        ld v0, [i]
        sne v0, 0
        call fc_z               ; leading zeros
        ld v0, v5
        ld v1, {MAXD1}
        sub v0, v1
        se vf, 0
        jp g_init
        ld v2, vc               ; v2 = its bits
        xor v2, v8
        sne v7, 0
        jp fc_p
        sne v7, 1
        shr v2, v2              ; products: bit 1 picks the accumulator
        ld v0, 1
        and v2, v0
        call facc
fc_e:   ld vb, 255
        ld v8, 0
        ld v9, 1
        ret
fc_p:   ld v0, 1                ; undecided: PEND = (first digit, digits, negated)
        and v2, v0
        ld v0, vb
        ld v1, v5
        ld i, {PEND}
        ld [i], v2
        jp fc_e
fc_z:   sne v5, 1
        ret
        ld i, {TAPE}
        add i, vb
        ld v0, [i]
        se v0, 0
        ret
        add vb, 1
        add v5, 255
        jp fc_z

facc:   ld i, {TAPE}            ; ACC[v2] op= the number (v5 digits from tape index vb)
        add i, vb
        bld b, v5
        sne v7, 1
        call fa_0
        ld i, {L0}
        add i, v2
        ld v0, [i]
        sne v0, 0
        jp fa_new
        se v6, v2
        call fa_ld
        sne v7, 1
        call mulm
        sne v7, 2
        call addm
        ld v1, v0
        ld v3, {MAXD1}
        sub v1, v3
        se vf, 0
        jp g_init
        bld a, v0
        jp fa_len
fa_new: se v6, 255              ; empty: ACC = the number
        call fa_sp
        ld i, {B_ADDR}
        bld a, v5
        ld v0, v5
fa_len: ld v6, v2               ; A holds ACC[v2]
        ld i, {L0}
        add i, v2
        ld [i], v0
        ret
fa_ld:  se v6, 255              ; A = ACC[v2] (v0 digits), the other one back to memory
        call fa_sp
        ld i, {ACC0}
        se v2, 0
        ld i, {ACC1}
        bld a, v0
        ret
fa_sp:  ld i, {ACC0}            ; ACC[v6] = A
        se v6, 0
        ld i, {ACC1}
        bsta
        ret
fa_0:   se v5, 1                ; products: a 0 goes to the general evaluator
        ret
        ld i, {B_ADDR}
        ld v0, [i]
        sne v0, 0
        jp g_init
        ret

; ---- part 2
f_lp:   call fcons              ; "("
        se v9, 0
        jp g_init               ; a product without an operator
        ld v0, vc
        ld i, {FST}
        add i, va
        ld [i], v0
        add va, 1
        xor vc, v8
        ld v8, 0
        ret
f_rp:   call fcons              ; ")"
        sne va, 0
        jp fr_d
        add va, 255
        ld i, {FST}
        add i, va
        ld v0, [i]
        ld vc, v0
fr_d:   ld v8, 0
        ld v9, 1
        ret
f_op:   ld i, {FOP}             ; an operator (v0 = 4 + 5 - 8 * 9 /)
        ld [i], v0
        call fcons
        ld i, {FOP}
        ld v0, [i]
        ld v3, v0
        se v9, 0
        jp fo_bin
        sne v3, 5               ; not after an operand: "-" negates, "+" is nothing
        jp fo_neg
        sne v3, 4
        ret
        jp g_init
fo_neg: ld v0, 1
        xor v8, v0
        ld i, {GP}
        ld v0, [i]
        ld v1, 1
        xor v0, v1
        ld [i], v0
        ret
fo_bin: ld v1, v3               ; its mode: 1 (* /) or 2 (+ -)
        ld v2, 8
        sub v1, v2
        ld v2, 2
        se vf, 0
        ld v2, 1
        se v7, 0
        jp fo_m
        ld v7, v2               ; the first operator decides
        ld i, {PEND}
        ld v2, [i]              ; v0 = its first digit, v1 = digits, v2 = negated
        sne v1, 0
        jp g_init
        ld vb, v0
        ld v5, v1
        se v7, 2
        ld v2, 0                ; products: the numerator
        call facc
        ld vb, 255
        ld i, {FOP}
        ld v0, [i]
        ld v3, v0
        jp fo_p
fo_m:   se v7, v2
        jp g_init
fo_p:   ld v8, 0                ; the next operand: "-" negates it, "/" divides by it
        sne v3, 5
        ld v8, 1
        sne v3, 9
        ld v8, 2
        ld v9, 0
        ret

f_end:  call fcons              ; the end
        se v6, 255
        call fa_sp
        se v7, 0
        jp fe_m
        ld i, {PEND}            ; one number: its sign is GP
        ld v1, [i]
        sne v1, 0
        jp g_init
        ld i, {TAPE}
        add i, v0
        bld a, v1
        ld v5, v1
        ld v6, 0
        ld i, {GP}
        ld v0, [i]
        ld v7, v0
        jp fe_pr
fe_m:   ld i, {GP}
        ld v2, [i]              ; v0 = GP, v1 = L0, v2 = L1
        ld v5, v1
        ld v6, v2
        se v7, 1
        jp fe_sum
        ld v7, v0               ; products: (-1)^GP ACC0 / ACC1
        ld i, {ACC0}
        bld a, v5
        jp fe_pr
fe_sum: ld v7, 0                ; sums: ACC0 - ACC1
        se v6, 0
        jp fe_s2
        ld i, {ACC0}
        bld a, v5
        jp fe_pr
fe_s2:  ld i, {ACC1}
        bld b, v6
        sne v5, 0
        jp fe_neg
        ld i, {ACC0}
        bld a, v5
        call subm
        ld v7, 1
        se v3, 0
        ld v7, 0
        bld a, v0
        ld v5, v0
        ld v6, 0
        jp fe_pr
fe_neg: ld i, {ACC1}            ; no positive terms: -ACC1
        bld a, v6
        ld v5, v6
        ld v6, 0
        ld v7, 1
fe_pr:  ld i, {A_ADDR}          ; A = the value's n (v5 digits), v6 = d's (0: none), v7 = sign
        ld v0, [i]
        se v5, 1
        jp fe_s
        se v0, 0
        jp fe_s
        ld v7, 0                ; 0: no sign, no d
        ld v6, 0
fe_s:   call setmode
        se v6, 0
        jp fe_div
        ld i, {A_ADDR}
        ld v0, v5
        print
fe_div: ld i, {W1}              ; n / d: div prints it
        bsta
        ld i, {ACC1}
        bld b, v6
        jp div
