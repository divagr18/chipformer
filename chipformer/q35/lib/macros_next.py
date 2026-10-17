"""Proposed macros for the maths library (reference semantics only: no circuit yet).

Profiling the BIOS-only library (mathlib.py) on the reference shows where trace tokens go:
copies of a result into A or B (35-55% of pow, fact, gcd; the y copy is 45% of log10), the
Briggs shift-and-add loop (log, exp), and per-digit short division (base conversion). These
three one-record macros remove those loops. They sit in the free 5XYN space like macros.py's
and follow the same rules: every write is computed from the state before the record, digit
sums stay below 256 (bf16-exact), and a record has at most 64 writes.

    5X0B  BMOV   A (X = 0) or B (X = 1) = the V0 cells at I, LEN[X] = V0, VF = 0. I must lie
                 in page 0x600 (A, B, R or W: one bank) with 1 <= V0 <= MAX_DIGITS, else
                 only VF = 1. Writes: the digits MSB first, LEN[X], VF, PC.
    5X09  BSDIV  A = A div VX in place for 2 <= VX <= 25 (10 r + digit <= 249), MSB first,
                 VF = A mod VX; the quotient's leading zeros (at most 2) are dropped from A
                 and LA, keeping one digit. VX out of range: only VF = 255.
    5XYE  BSHAD  Briggs' step on y = A (LA digits, A[0] the units digit), k = VX:
                 Y = 0 (log): if y + floor(y / 10^k) < 10^LA: A = it, CNT[k] += 1;
                       else VX = k + 1
                 Y = 1 (exp): if CNT[k] > 0: A = y + floor(y / 10^k), CNT[k] -= 1;
                       else VX = k + 1
                 PC stays (the record repeats) until VX would reach LEVELS, then PC = next.
                 Writes: the digits LSB first and CNT[k], or VX.

NextBridgeMachine runs a trace with them on top of the existing macros, for library code that
uses them (mathlib's NEXT profiles).
"""

from chipformer import spec
from chipformer.q35 import bridge as Bq
from chipformer.q35.lib import bcd
from chipformer.q35.lib.macros import macro_writes

BSDIV, BMOV, BSHAD = 0x9, 0xB, 0xE
KINDS = (BSDIV, BMOV, BSHAD)
LEVELS = 16
CNT = 0x690  # BSHAD's counts (mathlib.CNT)
PAGE, PAGE_END = 0x600, bcd.W_ADDR + 1 + bcd.MAX_DIGITS + bcd.DEC  # BMOV's sources


def encode(kind: int, x: int = 0, y: int = 0) -> int:
    return 0x5000 | x << 8 | y << 4 | kind


def next_writes(cells: list[int], op: int):
    """The record for opcode ``op`` if it is one of these macros, else None."""
    if op >> 12 != 5 or op & 0xF not in KINDS:
        return None
    kind, x, y = op & 0xF, op >> 8 & 0xF, op >> 4 & 0xF
    pc = cells[spec.PC]
    nxt = (pc + 2) & 0xFFF
    la = cells[bcd.LEN]
    w = []
    if kind == BMOV:
        n, src = cells[spec.V], cells[spec.I]
        if x > 1 or not 1 <= n <= bcd.MAX_DIGITS or not PAGE <= src <= PAGE_END - n:
            return [(spec.V + 15, 1), (spec.PC, nxt)]
        dst = bcd.A_ADDR if x == 0 else bcd.B_ADDR
        w = [(dst + j, cells[src + j]) for j in range(n)]
        return w + [(bcd.LEN + x, n), (spec.V + 15, 0), (spec.PC, nxt)]
    if kind == BSDIV:
        d = cells[spec.V + x]
        if not 2 <= d <= 25:
            return [(spec.V + 15, 255), (spec.PC, nxt)]
        q, r = [], 0
        for i in range(la):
            v = 10 * r + cells[bcd.A_ADDR + i]
            q.append(v // d)
            r = v % d
        z = 0
        while z < 2 and z < la - 1 and q[z] == 0:
            z += 1
        w = [(bcd.A_ADDR + j, q[j + z]) for j in range(la - z)]
        if z:
            w.append((bcd.LEN, la - z))
        return w + [(spec.V + 15, r), (spec.PC, nxt)]
    # BSHAD
    k, mode = cells[spec.V + x], y & 1
    cnt = cells[CNT + k] if k < LEVELS else 0
    A = [cells[bcd.A_ADDR + i] for i in range(la)]
    digits, carry = [], 0
    for i in range(la - 1, -1, -1):  # y + floor(y / 10^k), least significant first
        s = A[i] + (A[i - k] if i >= k else 0) + carry
        digits.append((bcd.A_ADDR + i, s % 10))
        carry = s // 10
    take = cnt > 0 if mode else carry == 0
    if k < LEVELS and take:
        return digits + [(CNT + k, cnt - 1 if mode else cnt + 1)]
    w = [(spec.V + x, k + 1)]
    return w if k + 1 < LEVELS else w + [(spec.PC, nxt)]


class NextBridgeMachine(Bq.BridgeMachine):
    """BridgeMachine with the proposed macros as well."""

    def step(self):
        from chipformer.ref.emu import fetch
        from chipformer.trace import encode_writes

        if not self.halted:
            w = next_writes(self.emu.cells, fetch(self.emu.cells))
            if w is not None:
                self.emu.apply(w)
                self._emit(encode_writes(w) + [spec.STEP], True)
                return w
        return super().step()


assert macro_writes([0] * spec.NUM_CELLS, encode(BMOV)) is None, "5XYB is taken"
