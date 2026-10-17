"""Bignum macro instructions: the circuit computes a whole row of digits in one record.

They live in CHIP-8's unused 5XYN space (5XY0 is the usual skip) and work on the BIOS layout
(bcd.py): digits most significant first at A_ADDR / B_ADDR, lengths at LEN / LEN+1, result at
R_ADDR. Writes go least significant first so a carry flows from one write to the next. T is a
jump target in the instruction itself, T = 0x200 + 2 * XY.

    5XY1  BADD   R[0..n) = A + B, n = max(LA, LB) + 1, then V0 = n - z, I = R_ADDR + z
                 (z = [R[0] == 0]: the print skips a leading zero); PC = T
    5XY2  BZERO  R[0..n) = 0, n = LA + LB, then V7 = n
    5XY4  BSUB   R[0..n) = A - B mod 10^n, n = max(LA, LB) + 1, then V0 and I as BADD;
                 PC = T if A >= B, else the next instruction
    5XY7  BRSUB  R[0..n) = B - A (for B >= A), then V0 and I as BADD (V0 = 0 for a tool call:
                 MODE bit 1 clear), MODE |= 8 (the print starts with " -"); PC = T
    5X03  BMUL   one schoolbook row of A * B; VX = rows done, row r = LB - 1 - VX:
                 for i = LA-1 .. 0:  R[i+r+1] = (R[i+r+1] + A[i] * B[r] + carry) mod 10, carry = ..div 10
                 then R[r] = carry, VX += 1, and after the last row V0 = LA + LB - z,
                 I = R_ADDR + z (z = [R[0] == 0]); PC stays (the next row) until the last row
    5X13  BMUL0  the first row (VX = 0): the same with R read as 0 (no BZERO needed); PC = the
                 next instruction (a BMUL), or the one after it if LB = 1
    5008  BTRIM  if mem[I] == 0, V0 >= 2 and I is in the result: V0 -= 1, I += 1, PC stays;
                 else PC = the next instruction (strips leading zeros before a PRINT)

A record whose PC stays has no PC write (the circuit fetches from the unchanged PC).

Each record's values come from the state before the record plus the carry chain inside it, so a
multiplication costs LB records of LA + 4 writes.

Long division works in place on W (bcd.W_ADDR: a 0, the digits of A, then DEC zeros), with the
divisor B and the quotient's digits in P (bcd.R_ADDR; P[t] for t <= LA, P[t + 1] after the "."):

    5XY5  BDIV   one quotient digit, t = VX: the remainder window W[t-LB .. t] -= q * B, P[.] = q,
                 with q estimated from the window's top 3 digits and B's top 2 (q <= true + 2)
    5XY6  BADB   the add-back after an estimate that was too big: W[t-LB .. t] += B, P[.] -= 1

Both end with VF = 1 (negative: add back), 2 (done, exact), 3 (done, DEC decimals and a
remainder) or 0 (next digit, VX = t + 1). BDIV repeats itself (PC unchanged) on 0 and BADB on 1,
so a division needs loop instructions only around its add-backs.
"""

from chipformer import spec
from chipformer.q35.lib import bcd

BADD, BZERO, BMUL, BSUB, BDIV, BADB, BRSUB, BTRIM = 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8
KINDS = (BADD, BZERO, BMUL, BSUB, BDIV, BADB, BRSUB, BTRIM)
R_CELLS = 2 * bcd.MAX_DIGITS  # the result region BTRIM can read


def target(op: int) -> int:
    return 0x200 + 2 * (op >> 4 & 0xFF)


def div_estimate(W: list[int], B: list[int], t: int) -> int:
    """BDIV's quotient digit: never below the true one, at most 2 above it (for B[0] != 0)."""
    lb = len(B)
    if lb == 1:
        r3, b2, delta = 10 * W[t - 1] + W[t], B[0], 0
    else:
        r3 = 100 * W[t - lb] + 10 * W[t - lb + 1] + W[t - lb + 2]
        b2, delta = 10 * B[0] + B[1], int(lb >= 3)
    return min(9, (r3 + delta) // b2) if b2 else 9


def _div_record(cells: list[int], op: int, kind: int) -> list[tuple[int, int]]:
    la, lb = cells[bcd.LEN], cells[bcd.LEN + 1]
    B = [cells[bcd.B_ADDR + i] for i in range(lb)]
    x = op >> 8 & 0xF
    t = cells[spec.V + x]
    W = [cells[(bcd.W_ADDR + i) & 0xFFF] for i in range(64)]
    p_addr = bcd.R_ADDR + t + int(t > la)
    q = div_estimate(W, B, t) if kind == BDIV else 0
    # the quotient digit goes first: later tokens of the record read it from the token stream
    w = [(p_addr, q if kind == BDIV else (cells[p_addr] - 1) & 0xFF)]
    carry, digits = 0, []
    for k in range(lb):  # least significant first
        i = lb - 1 - k
        if kind == BDIV:
            v = W[t - k] - q * B[i] - carry
            d = v % 10
            carry = (d - v) // 10  # the borrow
        else:
            v = W[t - k] + B[i] + carry
            d, carry = v % 10, v // 10
        w.append((bcd.W_ADDR + t - k, d))
        digits.append(d)
    v = W[t - lb] + (carry if kind == BADB else -carry)
    d = v % 10
    w.append((bcd.W_ADDR + t - lb, d))
    digits.append(d)
    # BDIV: negative if the top digit borrowed; BADB: still negative unless it carried out
    neg = v < 0 if kind == BDIV else v < 10
    zero = not neg and not any(digits)
    flag = 1 if neg else 2 if zero and t >= la else 3 if t == la + bcd.DEC else 0
    w.append((spec.V + x, t + (flag == 0)))
    w.append((spec.V + 15, flag))
    pc = cells[spec.PC]
    again = flag == (0 if kind == BDIV else 1)
    return w + [(spec.PC, pc if again else (pc + 2) & 0xFFF)]


def macro_writes(cells: list[int], op: int):
    """The record for opcode ``op`` if it is a macro, else None."""
    if op >> 12 != 5 or op & 0xF not in KINDS:
        return None
    if op & 0xF in (BDIV, BADB):
        return _div_record(cells, op, op & 0xF)
    la, lb = cells[bcd.LEN], cells[bcd.LEN + 1]
    A = [cells[bcd.A_ADDR + i] for i in range(la)]
    B = [cells[bcd.B_ADDR + i] for i in range(lb)]
    pc = cells[spec.PC]
    nxt = (pc + 2) & 0xFFF
    kind = op & 0xF
    w = []
    if kind in (BADD, BSUB, BRSUB):
        n = max(la, lb) + 1
        carry = 0
        for k in range(n):  # w-th write is digit k from the right
            a = A[la - 1 - k] if k < la else 0
            b = B[lb - 1 - k] if k < lb else 0
            if kind == BRSUB:
                a, b = b, a
            if kind == BADD:
                s = a + b + carry
                w.append((bcd.R_ADDR + n - 1 - k, s % 10))
                carry = s // 10
            else:  # carry is the borrow
                s = a - b - carry
                w.append((bcd.R_ADDR + n - 1 - k, s % 10))
                carry = int(s < 0)
        mode = cells[bcd.MODE]
        z = int(w[-1][1] == 0)  # the top digit, R[0]
        w.append((spec.V, n - z if kind != BRSUB or mode & 1 else 0))
        w.append((spec.I, bcd.R_ADDR + z))
        if kind == BRSUB:
            w.append((bcd.MODE, mode | 8))
        new_pc = nxt if kind == BSUB and carry else target(op)
    elif kind == BZERO:
        n = la + lb
        w = [(bcd.R_ADDR + k, 0) for k in range(n)] + [(spec.V + 7, n)]
        new_pc = nxt
    elif kind == BMUL:
        x, first = op >> 8 & 0xF, op >> 4 & 0xF == 1
        d = cells[spec.V + x]
        row = lb - 1 - d
        b = B[row] if 0 <= row < lb else 0
        carry = 0
        for i in range(la - 1, -1, -1):
            k = i + row + 1
            t = (0 if first else cells[bcd.R_ADDR + k]) + A[i] * b + carry
            w.append((bcd.R_ADDR + k, t % 10))
            carry = t // 10
        w.append((bcd.R_ADDR + row, carry))
        w.append((spec.V + x, (d + 1) & 0xFF))
        last = d + 1 == lb
        if last:
            z = int(carry == 0)  # R[0]
            w += [(spec.V, la + lb - z), (spec.I, bcd.R_ADDR + z)]
        new_pc = ((nxt + 2) & 0xFFF if last else nxt) if first else nxt if last else pc
    else:  # BTRIM
        i, v0 = cells[spec.I], cells[spec.V]
        z = bcd.R_ADDR <= i < bcd.R_ADDR + R_CELLS and cells[i] & 15 == 0 and v0 >= 2
        if z:
            w = [(spec.V, v0 - 1), (spec.I, i + 1)]
        new_pc = pc if z else nxt
    return w if new_pc == pc else w + [(spec.PC, new_pc)]
