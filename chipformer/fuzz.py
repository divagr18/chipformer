"""Random CHIP-8 programs that keep executing and cover every opcode.

Layout: a straight-line body of random instructions (skips allowed, calls to
one subroutine), a jump back to the start, then the subroutine. Stores
(FX33/FX55) may land anywhere in RAM, including the program itself, which
exercises self-modifying code.
"""

import random

SUB_LEN = 4


def random_instruction(rng: random.Random, sub_addr: int) -> int:
    x, y, n, nn = rng.randrange(16), rng.randrange(16), rng.randrange(16), rng.randrange(256)
    kind = rng.choice([
        "00E0", "2NNN", "3XNN", "4XNN", "5XY0", "6XNN", "7XNN", "8XY", "9XY0", "ANNN",
        "BNNN", "CXNN", "DXYN", "DXYN", "EX9E", "EXA1", "FX07", "FX0A", "FX15", "FX18",
        "FX1E", "FX29", "FX33", "FX55", "FX65", "6XNN", "7XNN", "8XY",
    ])
    if kind == "00E0":
        return 0x00E0
    if kind == "2NNN":
        return 0x2000 | sub_addr
    if kind == "BNNN":  # jump to the start of the body plus V0 (V0 small keeps it in range)
        return 0xB200
    if kind == "8XY":
        return 0x8000 | x << 8 | y << 4 | rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 0xE])
    if kind == "ANNN":
        return 0xA000 | rng.choice([rng.randrange(0x200, 0x1000), rng.randrange(0x300, 0x400)])
    code = {
        "3XNN": 0x3000 | x << 8 | nn, "4XNN": 0x4000 | x << 8 | nn,
        "5XY0": 0x5000 | x << 8 | y << 4, "9XY0": 0x9000 | x << 8 | y << 4,
        "6XNN": 0x6000 | x << 8 | nn, "7XNN": 0x7000 | x << 8 | nn,
        "CXNN": 0xC000 | x << 8 | nn, "DXYN": 0xD000 | x << 8 | y << 4 | n,
        "EX9E": 0xE09E | x << 8, "EXA1": 0xE0A1 | x << 8,
    }.get(kind)
    if code is not None:
        return code
    return 0xF000 | x << 8 | int(kind.removeprefix("FX"), 16)


def random_quirks(seed: int) -> tuple[str, ...]:
    """A random quirk profile (each flag on with probability 1/2)."""
    from chipformer.spec import QUIRK_NAMES

    rng = random.Random(seed * 7919 + 1)
    return tuple(q for q in QUIRK_NAMES if rng.random() < 0.5)


def random_program(seed: int, body: int = 120) -> bytes:
    rng = random.Random(seed)
    sub_addr = 0x200 + 2 * (body + 1)
    words = [random_instruction(rng, sub_addr) for _ in range(body)]
    words.append(0x1200)
    words += [random_instruction(rng, sub_addr) for _ in range(SUB_LEN - 1)]
    words = [w if w != 0x2000 | sub_addr or i < body else 0x6000 for i, w in enumerate(words)]
    words.append(0x00EE)
    return b"".join(w.to_bytes(2, "big") for w in words)
