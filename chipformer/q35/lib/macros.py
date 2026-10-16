"""Bignum macro instructions: the circuit computes a whole row of digits in one record.

They live in CHIP-8's unused 5XYN space (5XY0 is the usual skip) and work on the BIOS layout
(bcd.py): digits most significant first at A_ADDR / B_ADDR, lengths at LEN / LEN+1, result at
R_ADDR. Writes go least significant first so a carry flows from one write to the next.

    5XY1  BADD   R[0..n) = A + B, n = max(LA, LB) + 1, then V7 = n
    5XY2  BZERO  R[0..n) = 0, n = LA + LB, then V7 = n
    5XY3  BMAC   one schoolbook row with b = B[VX]:
                 for i = LA-1 .. 0:  R[i+VX+1] = (R[i+VX+1] + A[i] * b + carry) mod 10, carry = ..div 10
                 then R[VX] = carry

Each record's values come from the state before the record plus the carry chain inside it, so a
multiplication costs about LB records of LA + 2 writes instead of thousands of instructions.
"""

from chipformer import spec
from chipformer.q35.lib import bcd

BADD, BZERO, BMAC = 0x1, 0x2, 0x3


def macro_writes(cells: list[int], op: int):
    """The record for opcode ``op`` if it is a macro, else None."""
    if op >> 12 != 5 or op & 0xF not in (BADD, BZERO, BMAC):
        return None
    la, lb = cells[bcd.LEN], cells[bcd.LEN + 1]
    A = [cells[bcd.A_ADDR + i] for i in range(la)]
    B = [cells[bcd.B_ADDR + i] for i in range(lb)]
    pc = (cells[spec.PC] + 2) & 0xFFF
    kind = op & 0xF
    w = []
    if kind == BADD:
        n = max(la, lb) + 1
        carry = 0
        for k in range(n):  # w-th write is digit k from the right
            a = A[la - 1 - k] if k < la else 0
            b = B[lb - 1 - k] if k < lb else 0
            s = a + b + carry
            w.append((bcd.R_ADDR + n - 1 - k, s % 10))
            carry = s // 10
        w.append((spec.V + 7, n))
    elif kind == BZERO:
        n = la + lb
        w = [(bcd.R_ADDR + k, 0) for k in range(n)] + [(spec.V + 7, n)]
    else:
        row = cells[spec.V + (op >> 8 & 0xF)]
        b = B[row] if row < lb else 0
        carry = 0
        for i in range(la - 1, -1, -1):
            k = i + row + 1
            t = cells[bcd.R_ADDR + k] + A[i] * b + carry
            w.append((bcd.R_ADDR + k, t % 10))
            carry = t // 10
        w.append((bcd.R_ADDR + row, carry))
    return w + [(spec.PC, pc)]
