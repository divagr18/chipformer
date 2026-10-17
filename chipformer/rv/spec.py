"""Tokens, records and the address map of the RV32 machine.

A trace is one record per instruction. Machine state changes only at a record's last token (its
*commit*); the PC is implied: PC = 4 * (anchor + n), where the anchor is the payload of the latest
J or R token and n counts the commits since then (an R refresh is forced before n reaches 64).

Token families (chip vocabulary ids from 0):

  V   15-bit chunk of a value, not final
  F0  final; the value is this payload sign-extended from bit 14 (a 1-token value record)
  F1  final; after one V: value bits 15..29, sign-extended from bit 29
  F2  final; after two V: value bits 30..31
  J   final; jump: anchor = payload (a word index), n = 0
  R   final; anchor refresh: anchor = payload, n = 0, nothing executes
  NT, ST0, ST1, SS0, SS1, HALT   final controls (ST/SS: the RAM value head, address bit 4)
  GO  not final: the start of a run (after a state snapshot)

Records:

  rd-writing instruction (ALU, MUL, load, LUI, AUIPC)  F0 | V F1 | V V F2 (by the value's size)
  the same with rd = x0, not-taken branch, FENCE      NT
  taken branch, JAL, JALR                              J  (the link rd = PC + 4 is in-circuit)
  SW                                                   ST0 / ST1
  SB, SH                                               V V V SS0 / SS1 (the merged word)
  ECALL (exit), EBREAK                                 HALT
  before the instruction, when n == 63                 R

Memory (P0 sizes; see docs/rv32.md for the P1/P2 maps):

  RAM  0x0000_0000 + 16 KB   words; data and .rodata from 0x100, the stack below 0x4000
  ROM  0x0001_0000 + 64 KB   code only, pre-decoded (rom_fields), filled from a snapshot

RAM sits at the bottom so data and stack pointers fit a 1-token value record (F0 holds
-2^14 .. 2^14 - 1): that saves about 0.23 tokens per instruction on C code.
"""

P = 15  # payload bits of V, F0, F1, J, R
NP = 1 << P

V, F0, F1, J, R = 0, NP, 2 * NP, 3 * NP + 4, 4 * NP + 4
F2 = 3 * NP  # 4 tokens
CTRL = 5 * NP + 4
NT, ST0, ST1, SS0, SS1, HALT, GO = range(CTRL, CTRL + 7)
VOCAB = CTRL + 7

FAMILIES = {"V": (V, NP), "F0": (F0, NP), "F1": (F1, NP), "F2": (F2, 4), "J": (J, NP),
            "R": (R, NP)}
CONTROLS = {"NT": NT, "ST0": ST0, "ST1": ST1, "SS0": SS0, "SS1": SS1, "HALT": HALT, "GO": GO}
FINAL = {"F0", "F1", "F2", "J", "R", "NT", "ST0", "ST1", "SS0", "SS1", "HALT"}

ROM_BASE, ROM_WORDS = 0x10000, 1 << 14  # 64 KB
RAM_BASE, RAM_WORDS = 0x0, 1 << 12  # 16 KB (one sublayer)
DATA_BASE = 0x100  # .data starts here (address 0 stays a null pointer)
SP_INIT = RAM_BASE + 4 * RAM_WORDS
N_MAX = 63  # the R refresh comes before the instruction when n == N_MAX


def family(t: int) -> tuple[str, int]:
    """(family or control name, payload) of a token."""
    for name, (base, size) in FAMILIES.items():
        if base <= t < base + size:
            return name, t - base
    for name, tok in CONTROLS.items():
        if t == tok:
            return name, 0
    raise ValueError(f"not an rv token: {t}")


def tok(name: str, payload: int = 0) -> int:
    if name in CONTROLS:
        return CONTROLS[name]
    base, size = FAMILIES[name]
    assert 0 <= payload < size, (name, payload)
    return base + payload


def is_final(t: int) -> bool:
    return family(t)[0] in FINAL


def value_record(v: int) -> list[int]:
    """Tokens of a 32-bit value: as few chunks as its sign-extended size needs."""
    v &= 0xFFFFFFFF
    s = v - (1 << 32) if v >> 31 else v
    c0, c1, c2 = v & 0x7FFF, v >> 15 & 0x7FFF, v >> 30
    if -(1 << 14) <= s < 1 << 14:
        return [tok("F0", c0)]
    if -(1 << 29) <= s < 1 << 29:
        return [tok("V", c0), tok("F1", c1)]
    return [tok("V", c0), tok("V", c1), tok("F2", c2)]


def decode_value(tokens: list[int]) -> int:
    """Inverse of value_record (the commit's view: the final token and the V before it)."""
    names = [family(t) for t in tokens]
    (fam, p) = names[-1]
    if fam == "F0":
        v = p | (0xFFFF8000 if p >> 14 else 0)
    elif fam == "F1":
        v = names[-2][1] | p << 15 | (0xC0000000 if p >> 14 else 0)
    elif fam == "F2":
        v = names[-3][1] | names[-2][1] << 15 | p << 30
    else:
        raise ValueError(fam)
    return v & 0xFFFFFFFF


def sign32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >> 31 else v


def name(t: int) -> str:
    fam, p = family(t)
    return fam if fam in CONTROLS else f"{fam}({p})"
