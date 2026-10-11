"""Machine layout and token vocabulary shared by every component.

All machine state lives in one flat array of cells. A cell holds an integer in
0..MAX_VALUE. The trace the transformer reads and writes is a sequence of
write events ``[ADDR a][VAL v]`` grouped into records terminated by ``STEP``.
See docs/trace_format.md for the full specification.
"""

# --- cell address map -------------------------------------------------------
RAM = 0x0000
RAM_SIZE = 0x1000
V = 0x1000  # V0..VF
I = 0x1010  # 12-bit index register
PC = 0x1011  # 12-bit program counter
SP = 0x1012  # stack pointer, 0..15
DT = 0x1013  # delay timer
ST = 0x1014  # sound timer
RNG = 0x1015  # 12-bit LFSR state
CLS = 0x1016  # writing this cell clears the display
QUIRKS = 0x1017  # 6 quirk flags (0/1), meaning as Octo's flags of the same name; see docs/quirks.md
QUIRK_NAMES = ("shift", "loadstore", "clip", "logic", "jump", "vforder")
Q_SHIFT, Q_LOADSTORE, Q_CLIP, Q_LOGIC, Q_JUMP, Q_VFORDER = (QUIRKS + i for i in range(6))
STACK = 0x1020  # 16 return addresses
KEYS = 0x1030  # 16 key states, 0 or 1
DISPLAY = 0x1100  # 32 rows x 8 bytes, MSB is the leftmost pixel
NULL = 0x1200  # sink for writes that must exist but have no effect
NUM_CELLS = 0x1201

DISPLAY_W = 64
DISPLAY_H = 32
DISPLAY_ROW_BYTES = DISPLAY_W // 8

# --- values -----------------------------------------------------------------
NUM_VALUES = 0x1000
MAX_VALUE = NUM_VALUES - 1

# --- token vocabulary -------------------------------------------------------
VAL_BASE = 0  # VAL v  -> v
ADDR_BASE = NUM_VALUES  # ADDR a -> ADDR_BASE + a
STEP = ADDR_BASE + NUM_CELLS
TICK = STEP + 1
BOOT = STEP + 2
VOCAB_SIZE = BOOT + 1

# --- machine constants ------------------------------------------------------
PROGRAM_START = 0x200
FONT_BASE = 0x050
RNG_SEED = 0xACE
RNG_MASK = 0x829  # Galois taps for x^12 + x^6 + x^4 + x + 1 (period 4095)

# Named quirk profiles: the set of enabled flags. All flags off = Octo's defaults.
PROFILES = {
    "octo": (),
    "chip48": ("shift", "loadstore", "clip"),  # HP48 CHIP-48 / SCHIP-era games, e.g. Pong
    "cosmac": ("logic", "clip"),  # original COSMAC VIP interpreter
}


def quirk_cells(quirks: "str | tuple[str, ...] | list[str] | None") -> dict[int, int]:
    """Cell values for a profile name or a list of enabled flag names."""
    if quirks is None:
        quirks = ()
    elif isinstance(quirks, str):
        quirks = PROFILES[quirks] if quirks in PROFILES else tuple(q for q in quirks.split(",") if q)
    unknown = set(quirks) - set(QUIRK_NAMES)
    if unknown:
        raise ValueError(f"unknown quirks {sorted(unknown)}; known: {QUIRK_NAMES}")
    return {QUIRKS + i: int(name in quirks) for i, name in enumerate(QUIRK_NAMES)}

FONT = bytes([
    0xF0, 0x90, 0x90, 0x90, 0xF0,  # 0
    0x20, 0x60, 0x20, 0x20, 0x70,  # 1
    0xF0, 0x10, 0xF0, 0x80, 0xF0,  # 2
    0xF0, 0x10, 0xF0, 0x10, 0xF0,  # 3
    0x90, 0x90, 0xF0, 0x10, 0x10,  # 4
    0xF0, 0x80, 0xF0, 0x10, 0xF0,  # 5
    0xF0, 0x80, 0xF0, 0x90, 0xF0,  # 6
    0xF0, 0x10, 0x20, 0x40, 0x40,  # 7
    0xF0, 0x90, 0xF0, 0x90, 0xF0,  # 8
    0xF0, 0x90, 0xF0, 0x10, 0xF0,  # 9
    0xF0, 0x90, 0xF0, 0x90, 0x90,  # A
    0xE0, 0x90, 0xE0, 0x90, 0xE0,  # B
    0xF0, 0x80, 0x80, 0x80, 0xF0,  # C
    0xE0, 0x90, 0x90, 0x90, 0xE0,  # D
    0xF0, 0x80, 0xF0, 0x80, 0xF0,  # E
    0xF0, 0x80, 0xF0, 0x80, 0x80,  # F
])


def lfsr_step(s: int) -> int:
    lsb = s & 1
    s >>= 1
    if lsb:
        s ^= RNG_MASK
    return s


def val_token(v: int) -> int:
    assert 0 <= v <= MAX_VALUE, v
    return VAL_BASE + v


def addr_token(a: int) -> int:
    assert 0 <= a < NUM_CELLS, a
    return ADDR_BASE + a


def is_val(tok: int) -> bool:
    return VAL_BASE <= tok < ADDR_BASE


def is_addr(tok: int) -> bool:
    return ADDR_BASE <= tok < STEP


def token_name(tok: int) -> str:
    if is_val(tok):
        return f"={tok - VAL_BASE:#x}"
    if is_addr(tok):
        return f"@{tok - ADDR_BASE:#06x}"
    return {STEP: "STEP", TICK: "TICK", BOOT: "BOOT"}.get(tok, f"?{tok}")
