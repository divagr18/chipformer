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
add:    badd                    ; macro: R = A + B, v7 = length
        jp print1

; ---------------------------------------------------------------- sub: R = A - B (A >= B)
sub:    bsub                    ; macro: R = A - B, v7 = length
        jp print1

; ---------------------------------------------------------------- mul: R = A * B
mul:    bzero                   ; macro: R = 0, v7 = LA + LB
        ld i, {LEN}
        ld v1, [i]              ; v1 = LB
mul_r:  add v1, 0xff            ; row = LB-1 .. 0
        bmac v1                 ; macro: one schoolbook row with b = B[v1]
        se v1, 0
        jp mul_r

; ---------------------------------------------------------------- print R[0..v7), short path
; Only R[0] can be 0 after add or mul of plain numbers, so check one or two digits and fall
; back to the general loop when both are 0 (zero arguments, or sub).
print1: ld i, {R_ADDR}
        ld v0, [i]
        se v0, 0
        jp pr_all               ; R[0] != 0: print everything
        ld i, {R_ADDR + 1}
        ld v0, [i]
        sne v0, 0
        jp print                ; R[1] == 0 too
        ld v0, v7
        add v0, 0xff
        print                   ; R[1..v7)
pr_all: ld v0, v7
        print

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

"""

ROM, LABELS = assemble(SRC)
OPS = {"add": LABELS["add"], "mul": LABELS["mul"], "sub": LABELS["sub"]}
# The entries are compiled into the circuit's CALL record (chip8_q35.py), so they are part of the
# weights: moving a routine means rebuilding every checkpoint. Add new code after the entries.
assert OPS == {"add": 0x202, "sub": 0x206, "mul": 0x20A}, OPS


def call_writes(op: str, a: str, b: str) -> list[tuple[int, int]]:
    """The writes of a CALL record: the arguments' digits, their lengths, then PC."""
    da, db = [int(ch) for ch in a], [int(ch) for ch in b]
    assert 1 <= len(da) <= MAX_DIGITS and 1 <= len(db) <= MAX_DIGITS, (a, b)
    w = [(A_ADDR + i, d) for i, d in enumerate(da)] + [(B_ADDR + i, d) for i, d in enumerate(db)]
    return w + [(LEN, len(da)), (LEN + 1, len(db)), (spec.PC, OPS[op])]
