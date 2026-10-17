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
T10, MUL, DIV, MOD = 0x700, 0x710, 0x780, 0x800
MAX_DIGITS = 24
QUIRKS = ("loadstore",)  # FX55/FX65 leave I unchanged


SRC = """
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

"""

ROM, LABELS = assemble(SRC)
OPS = {"add": LABELS["add"], "mul": LABELS["mul"], "sub": LABELS["sub"]}
# The entries are compiled into the circuit's CALL record (chip8_q35.py), so they are part of the
# weights: moving a routine means rebuilding every checkpoint. Add new code after the entries.
assert OPS == {"add": 0x202, "sub": 0x206, "mul": 0x20A}, OPS


def call_writes(op: str, a: str, b: str, mode: int = 0) -> list[tuple[int, int]]:
    """The writes of a CALL record: the arguments' digits, their lengths, the mode, then PC."""
    da, db = [int(ch) for ch in a], [int(ch) for ch in b]
    assert 1 <= len(da) <= MAX_DIGITS and 1 <= len(db) <= MAX_DIGITS, (a, b)
    w = [(A_ADDR + i, d) for i, d in enumerate(da)] + [(B_ADDR + i, d) for i, d in enumerate(db)]
    return w + [(LEN, len(da)), (LEN + 1, len(db)), (MODE, mode), (spec.PC, OPS[op])]
