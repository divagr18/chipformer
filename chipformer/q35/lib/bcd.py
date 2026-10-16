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
T10, MUL, DIV, MOD = 0x700, 0x710, 0x780, 0x800
MAX_DIGITS = 24
QUIRKS = ("loadstore",)  # FX55/FX65 leave I unchanged

SRC = f"""
idle:   jp idle                 ; PC rests here between calls (not a PRINT)

; ---------------------------------------------------------------- add: R = A + B
add:    ld i, {LEN}
        ld v1, [i]              ; v0 = LA, v1 = LB
        ld v2, v1               ; v2 = digits of B left
        ld v1, v0               ; v1 = digits of A left
        ld v7, v1               ; v7 = n = max(LA, LB) + 1
        ld v6, v2
        sub v6, v1              ; vf = LB >= LA
        se vf, 0
        ld v7, v2
        add v7, 1
        ld v3, v7               ; v3 = result digits left
        ld v4, 0                ; carry
add_lp: ld v5, v4
        ld v4, 0
        se v1, 0
        call add_a
        se v2, 0
        call add_b
        ld v6, 9
        sub v6, v5              ; vf = 9 >= v5
        se vf, 0
        jp add_st
        add v5, 0xf6            ; v5 -= 10
        ld v4, 1
add_st: add v3, 0xff
        ld i, {R_ADDR}
        add i, v3
        ld v0, v5
        ld [i], v0
        se v3, 0
        jp add_lp
        jp print

add_a:  add v1, 0xff
        ld i, {A_ADDR}
        add i, v1
        ld v0, [i]
        add v5, v0
        ret
add_b:  add v2, 0xff
        ld i, {B_ADDR}
        add i, v2
        ld v0, [i]
        add v5, v0
        ret

; ---------------------------------------------------------------- mul: R = A * B
mul:    ld i, {LEN}
        ld v1, [i]
        ld v8, v0               ; LA
        ld v9, v1               ; LB
        ld v7, v8
        add v7, v9              ; n = LA + LB
        ld v6, v7               ; zero R[0..n)
        ld v0, 0
mul_z:  add v6, 0xff
        ld i, {R_ADDR}
        add i, v6
        ld [i], v0
        se v6, 0
        jp mul_z
        ld v1, v8               ; i + 1, counting down
mul_i:  add v1, 0xff            ; v1 = i
        ld i, {A_ADDR}
        add i, v1
        ld v0, [i]
        ld i, {T10}
        add i, v0
        ld v0, [i]
        ld va, v0               ; va = 10 * A[i]
        ld v3, 0                ; carry
        ld v2, v9               ; j + 1
mul_j:  add v2, 0xff            ; v2 = j
        ld i, {B_ADDR}
        add i, v2
        ld v0, [i]
        add v0, va
        ld i, {MUL}
        add i, v0
        ld v0, [i]              ; A[i] * B[j]
        ld v5, v0
        add v5, v3              ; + carry
        ld v6, v1
        add v6, v2
        add v6, 1               ; k = i + j + 1
        ld i, {R_ADDR}
        add i, v6
        ld v0, [i]
        add v5, v0              ; t = R[k] + A[i] B[j] + carry  (<= 99)
        ld i, {MOD}
        add i, v5
        ld v0, [i]
        ld i, {R_ADDR}
        add i, v6
        ld [i], v0              ; R[k] = t % 10
        ld i, {DIV}
        add i, v5
        ld v0, [i]
        ld v3, v0               ; carry = t / 10
        se v2, 0
        jp mul_j
        ld i, {R_ADDR}
        add i, v1
        ld v0, v3
        ld [i], v0              ; R[i] = carry
        se v1, 0
        jp mul_i

; ---------------------------------------------------------------- print R[0..v7), no leading 0s
print:  ld v6, 0
pr_lp:  ld v0, v7
        add v0, 0xff
        se v6, v0               ; keep at least one digit
        jp pr_chk
        jp pr_go
pr_chk: ld i, {R_ADDR}
        add i, v6
        ld v0, [i]
        se v0, 0
        jp pr_go
        add v6, 1
        jp pr_lp
pr_go:  ld i, {R_ADDR}
        add i, v6
        ld v0, v7
        sub v0, v6
        print

        org {T10}
        db {", ".join(str(10 * a) for a in range(10))}
        org {MUL}
        db {", ".join(str((t // 10) * (t % 10)) for t in range(100))}
        org {DIV}
        db {", ".join(str(t // 10) for t in range(100))}
        org {MOD}
        db {", ".join(str(t % 10) for t in range(100))}
"""

ROM, LABELS = assemble(SRC)
OPS = {"add": LABELS["add"], "mul": LABELS["mul"]}


def call_writes(op: str, a: str, b: str) -> list[tuple[int, int]]:
    """The writes of a CALL record: the arguments' digits, their lengths, then PC."""
    da, db = [int(ch) for ch in a], [int(ch) for ch in b]
    assert 1 <= len(da) <= MAX_DIGITS and 1 <= len(db) <= MAX_DIGITS, (a, b)
    w = [(A_ADDR + i, d) for i, d in enumerate(da)] + [(B_ADDR + i, d) for i, d in enumerate(db)]
    return w + [(LEN, len(da)), (LEN + 1, len(db)), (spec.PC, OPS[op])]
