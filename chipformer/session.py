"""Host loop: drive any next-token model as a CHIP-8 machine.

The host only ever (1) feeds the boot prompt, (2) injects TICK tokens and key
writes right after a STEP, and (3) keeps a mirror of the cells built from the
model's own writes, for the screen. It never computes machine semantics.
"""

from collections.abc import Callable

from chipformer import spec
from chipformer.ref.emu import Chip8, apply_writes, display_pixels
from chipformer.trace import boot_tokens, encode_writes

MAX_RECORD = 2 * 40 + 2  # longer than any valid record


class Session:
    def __init__(self, feed: Callable[[int], int], rom: bytes, quirks=None):
        """``feed(token)`` processes one position and returns the predicted next token."""
        self.feed = feed
        self.cells = [0] * spec.NUM_CELLS
        self.tokens: list[int] = []
        self.pred = -1
        boot = Chip8(rom, quirks)
        apply_writes(self.cells, boot.state_writes())
        for t in boot_tokens(boot.cells):
            self._push(t)

    def _push(self, tok: int) -> None:
        self.tokens.append(tok)
        self.pred = self.feed(tok)

    def _inject(self, toks: list[int]) -> None:
        for t in toks:
            self._push(t)

    def _generate_record(self) -> list[tuple[int, int]]:
        """Let the model write until it emits STEP; apply its writes to the mirror."""
        writes = []
        for _ in range(MAX_RECORD):
            tok = self.pred
            if tok == spec.STEP:
                self._push(tok)
                apply_writes(self.cells, writes)
                return writes
            if not spec.is_addr(tok):
                raise RuntimeError(f"model emitted {spec.token_name(tok)} where an ADDR or STEP "
                                   f"was expected (position {len(self.tokens)})")
            self._push(tok)
            val = self.pred
            if not spec.is_val(val):
                raise RuntimeError(f"model emitted {spec.token_name(val)} after an ADDR")
            self._push(val)
            writes.append((tok - spec.ADDR_BASE, val - spec.VAL_BASE))
        raise RuntimeError("record never ended")

    def step(self) -> list[tuple[int, int]]:
        return self._generate_record()

    def tick(self) -> list[tuple[int, int]]:
        self._inject([spec.TICK])
        return self._generate_record()

    def set_key(self, key: int, down: bool) -> None:
        if bool(self.cells[spec.KEYS + key]) == down:
            return
        w = [(spec.KEYS + key, int(down))]
        self._inject(encode_writes(w) + [spec.STEP])
        apply_writes(self.cells, w)

    def pixels(self) -> list[list[int]]:
        return display_pixels(self.cells)
