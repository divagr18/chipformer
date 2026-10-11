"""Reference CHIP-8 interpreter operating directly on the flat cell array.

Every state change is expressed as a list of ``(addr, value)`` writes computed
from the state *before* the instruction, then applied. The trace format is
therefore exactly the emulator's semantics: replaying the writes reproduces
the state.

Quirks are machine state: six flag cells (spec.QUIRKS) with the meaning of
Octo's flags of the same name, set when the ROM is loaded (docs/quirks.md):
  - shift:     8XY6/8XYE shift VX in place (off: VX = VY shifted)
  - loadstore: FX55/FX65 leave I unchanged (off: I += X + 1)
  - clip:      sprites clip at the screen edges (off: they wrap around)
  - logic:     8XY1/2/3 reset VF to 0
  - jump:      BXNN jumps to XNN + VX (off: BNNN jumps to NNN + V0)
  - vforder:   for 8XY4-E the result is written after VF (wins when X = F)
Fixed behaviour: sprites start at (VX % 64, VY % 32); FX0A completes as soon
as any key is held (lowest key wins); the stack pointer wraps modulo 16.
"""

from chipformer import spec
from chipformer.spec import (
    CLS,
    DISPLAY,
    DT,
    KEYS,
    NULL,
    PC,
    Q_CLIP,
    Q_JUMP,
    Q_LOADSTORE,
    Q_LOGIC,
    Q_SHIFT,
    Q_VFORDER,
    RNG,
    SP,
    ST,
    STACK,
    I,
    V,
)

Writes = list[tuple[int, int]]


def fetch(cells: list[int]) -> int:
    pc = cells[PC]
    return (cells[pc] << 8) | cells[(pc + 1) & 0xFFF]


def opcode_class(op: int) -> str:
    """Human-readable opcode family, e.g. '8XY4' or 'DXYN'."""
    hi = op >> 12
    if op == 0x00E0:
        return "00E0"
    if op == 0x00EE:
        return "00EE"
    if hi == 0x0:
        return "0NNN"
    if hi == 0x8:
        return f"8XY{op & 0xF:X}"
    if hi in (0xE, 0xF):
        return f"{hi:X}X{op & 0xFF:02X}"
    return {1: "1NNN", 2: "2NNN", 3: "3XNN", 4: "4XNN", 5: "5XY0", 6: "6XNN", 7: "7XNN",
            9: "9XY0", 0xA: "ANNN", 0xB: "BNNN", 0xC: "CXNN", 0xD: "DXYN"}[hi]


class Chip8:
    def __init__(self, rom: bytes, quirks: "str | tuple[str, ...] | None" = None):
        if len(rom) > spec.RAM_SIZE - spec.PROGRAM_START:
            raise ValueError(f"ROM too large: {len(rom)} bytes")
        self.cells = [0] * spec.NUM_CELLS
        self.cells[spec.FONT_BASE:spec.FONT_BASE + len(spec.FONT)] = spec.FONT
        self.cells[spec.PROGRAM_START:spec.PROGRAM_START + len(rom)] = rom
        self.cells[PC] = spec.PROGRAM_START
        self.cells[RNG] = spec.RNG_SEED
        for a, v in spec.quirk_cells(quirks).items():
            self.cells[a] = v

    # --- state transitions --------------------------------------------------

    def apply(self, writes: Writes) -> None:
        apply_writes(self.cells, writes)

    def step(self) -> Writes:
        w = instruction_writes(self.cells)
        self.apply(w)
        return w

    def tick(self) -> Writes:
        w = tick_writes(self.cells)
        self.apply(w)
        return w

    def set_key(self, key: int, down: bool) -> Writes:
        w = [(KEYS + key, int(down))]
        self.apply(w)
        return w

    # --- views --------------------------------------------------------------

    def state_writes(self) -> Writes:
        return state_writes(self.cells)

    def pixels(self) -> list[list[int]]:
        return display_pixels(self.cells)


def apply_writes(cells: list[int], writes: Writes) -> None:
    for a, v in writes:
        assert 0 <= v <= spec.MAX_VALUE, (a, v)
        if a == CLS:
            cells[DISPLAY:DISPLAY + spec.DISPLAY_H * spec.DISPLAY_ROW_BYTES] = (
                [0] * (spec.DISPLAY_H * spec.DISPLAY_ROW_BYTES))
        cells[a] = v


def state_writes(cells: list[int]) -> Writes:
    """A state dump: replaying it on a zeroed state reproduces ``cells``.

    Only nonzero cells are written (reads of never-written cells return 0).
    CLS always comes first, so a "latest CLS" exists and the display bytes
    written after it survive.
    """
    return [(CLS, cells[CLS])] + [
        (a, cells[a]) for a in range(spec.NUM_CELLS) if a not in (CLS, NULL) and cells[a]]


def display_pixels(cells: list[int]) -> list[list[int]]:
    rows = []
    for y in range(spec.DISPLAY_H):
        row = []
        for b in range(spec.DISPLAY_ROW_BYTES):
            byte = cells[DISPLAY + y * spec.DISPLAY_ROW_BYTES + b]
            row.extend((byte >> (7 - i)) & 1 for i in range(8))
        rows.append(row)
    return rows


def tick_writes(cells: list[int]) -> Writes:
    return [(DT, max(cells[DT] - 1, 0)), (ST, max(cells[ST] - 1, 0))]


def instruction_writes(c: list[int]) -> Writes:
    """Writes for one instruction, computed purely from the current state.

    The PC write is always last, so every earlier write sees the old PC.
    """
    pc = c[PC]
    op = fetch(c)
    nxt = (pc + 2) & 0xFFF
    skip = (pc + 4) & 0xFFF
    x, y, n = (op >> 8) & 0xF, (op >> 4) & 0xF, op & 0xF
    nn, nnn = op & 0xFF, op & 0xFFF
    vx, vy = c[V + x], c[V + y]
    hi = op >> 12
    w: Writes = []

    if op == 0x00E0:
        w = [(CLS, 0)]
    elif op == 0x00EE:
        sp = (c[SP] - 1) & 0xF
        w = [(SP, sp)]
        nxt = c[STACK + sp]
    elif hi == 0x0:
        pass  # 0NNN: machine-code routine, ignored
    elif hi == 0x1:
        nxt = nnn
    elif hi == 0x2:
        sp = c[SP]
        w = [(STACK + sp, nxt), (SP, (sp + 1) & 0xF)]
        nxt = nnn
    elif hi == 0x3:
        nxt = skip if vx == nn else nxt
    elif hi == 0x4:
        nxt = skip if vx != nn else nxt
    elif hi == 0x5 and n == 0:
        nxt = skip if vx == vy else nxt
    elif hi == 0x9 and n == 0:
        nxt = skip if vx != vy else nxt
    elif hi == 0x6:
        w = [(V + x, nn)]
    elif hi == 0x7:
        w = [(V + x, (vx + nn) & 0xFF)]
    elif hi == 0x8:
        w = alu_writes(c, x, n, vx, vy)
    elif hi == 0xA:
        w = [(I, nnn)]
    elif hi == 0xB:
        nxt = (nnn + (vx if c[Q_JUMP] else c[V])) & 0xFFF
    elif hi == 0xC:
        s = spec.lfsr_step(c[RNG])
        w = [(V + x, s & nn & 0xFF), (RNG, s)]
    elif hi == 0xD:
        w = draw_writes(c, vx, vy, n)
    elif hi == 0xE and nn == 0x9E:
        nxt = skip if c[KEYS + (vx & 0xF)] else nxt
    elif hi == 0xE and nn == 0xA1:
        nxt = skip if not c[KEYS + (vx & 0xF)] else nxt
    elif hi == 0xF:
        ix = c[I]
        if nn == 0x07:
            w = [(V + x, c[DT])]
        elif nn == 0x0A:
            pressed = [k for k in range(16) if c[KEYS + k]]
            if pressed:
                w = [(V + x, pressed[0])]
            else:
                w = [(V + x, vx)]
                nxt = pc
        elif nn == 0x15:
            w = [(DT, vx)]
        elif nn == 0x18:
            w = [(ST, vx)]
        elif nn == 0x1E:
            w = [(I, (ix + vx) & 0xFFF)]
        elif nn == 0x29:
            w = [(I, spec.FONT_BASE + 5 * (vx & 0xF))]
        elif nn == 0x33:
            w = [((ix + j) & 0xFFF, d) for j, d in enumerate((vx // 100, vx // 10 % 10, vx % 10))]
        elif nn == 0x55:
            w = [((ix + j) & 0xFFF, c[V + j]) for j in range(x + 1)]
        elif nn == 0x65:
            w = [(V + j, c[(ix + j) & 0xFFF]) for j in range(x + 1)]
        if nn in (0x55, 0x65) and not c[Q_LOADSTORE]:
            w.append((I, (ix + x + 1) & 0xFFF))

    w.append((PC, nxt))
    return w


def alu_writes(c: list[int], x: int, n: int, vx: int, vy: int) -> Writes:
    vf = V + 0xF
    if n in (0x1, 0x2, 0x3):
        result = {0x1: vx | vy, 0x2: vx & vy, 0x3: vx ^ vy}[n]
        return [(V + x, result)] + ([(vf, 0)] if c[Q_LOGIC] else [])
    src = vx if c[Q_SHIFT] else vy
    if n == 0x0:
        return [(V + x, vy)]
    if n == 0x4:
        s = vx + vy
        result, flag = s & 0xFF, s >> 8
    elif n == 0x5:
        result, flag = (vx - vy) & 0xFF, int(vx >= vy)
    elif n == 0x6:
        result, flag = src >> 1, src & 1
    elif n == 0x7:
        result, flag = (vy - vx) & 0xFF, int(vy >= vx)
    elif n == 0xE:
        result, flag = (src << 1) & 0xFF, src >> 7
    else:
        return []
    if c[Q_VFORDER]:
        return [(vf, flag), (V + x, result)]
    return [(V + x, result), (vf, flag)]


def draw_writes(c: list[int], vx: int, vy: int, n: int) -> Writes:
    """DXYN as a fixed-length record: two display writes per sprite row
    (left and right byte the row can touch), then VF.

    With clipping, writes that fall off-screen go to NULL so the record length
    depends only on N, which keeps the transformer's micro-step bookkeeping
    trivial. Without it, rows and bytes wrap around the screen.
    """
    clip = c[Q_CLIP]
    x0, y0 = vx % spec.DISPLAY_W, vy % spec.DISPLAY_H
    b, o = x0 >> 3, x0 & 7
    w: Writes = []
    collision = 0
    for i in range(n):
        s = c[(c[I] + i) & 0xFFF]
        parts = ((b, s >> o), (b + 1, (s << (8 - o)) & 0xFF))
        row = y0 + i
        for col, bits in parts:
            if not clip:
                row, col = row % spec.DISPLAY_H, col % spec.DISPLAY_ROW_BYTES
            if row < spec.DISPLAY_H and col < spec.DISPLAY_ROW_BYTES:
                a = DISPLAY + row * spec.DISPLAY_ROW_BYTES + col
                old = c[a]
                w.append((a, old ^ bits))
                collision |= old & bits
            else:
                w.append((NULL, 0))
    w.append((V + 0xF, int(collision != 0)))
    return w
