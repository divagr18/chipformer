"""The read bridge: the computer prints text that Qwen reads, inside one token stream.

Token space of the v2 circuit: the 8708 chip tokens, then a few *text classes* standing for
Qwen's own text tokens (every other text token is OTHER). In a merged model each Qwen text token
row carries the features of its class, so the circuit sees text but only acts on these.

SCHIP's 00FD ("exit") is redefined as print-and-halt: the record that would execute it prints

    \\n <tool_response> \\n  h(mem[I]) h(mem[I+1]) ... h(mem[I+V0-1])  \\n </tool_response> \\n

as real text tokens (h = the low nibble as a hex digit, so BCD digits print as decimals), then
the machine stops and Qwen continues. Nothing in the machine state changes.
"""

from chipformer import spec
from chipformer.ref.emu import fetch
from chipformer.trace import Machine

TEXT_BASE = spec.VOCAB_SIZE  # 8708
HEX = list(range(TEXT_BASE, TEXT_BASE + 16))  # '0'..'9', 'a'..'f'
NL, TR_OPEN, TR_CLOSE, TC_OPEN, TC_CLOSE, OTHER = range(TEXT_BASE + 16, TEXT_BASE + 22)
N_TEXT = 22
PRINT_OP = 0x00FD
MAX_PRINT = 58  # digits per print (the write counter tops out at 64)

# Qwen3.5 token ids of the classes (single tokens in its tokenizer)
QWEN_TEXT = {**{TEXT_BASE + i: c for i, c in enumerate("0123456789abcdef")},
             NL: "\n", TR_OPEN: "<tool_response>", TR_CLOSE: "</tool_response>",
             TC_OPEN: "<tool_call>", TC_CLOSE: "</tool_call>"}


def is_text(t: int) -> bool:
    return t >= TEXT_BASE


def readout(cells: list[int]) -> list[int]:
    i, n = cells[spec.I], min(cells[spec.V], MAX_PRINT)
    digits = [HEX[cells[(i + k) & 0xFFF] & 15] for k in range(n)]
    return [NL, TR_OPEN, NL, *digits, NL, TR_CLOSE, NL]


class BridgeMachine(Machine):
    """Reference trace for the v2 circuit: machine records, prints, and Qwen's text."""

    def __init__(self, rom: bytes, quirks=None):
        super().__init__(rom, quirks)
        self.halted = False

    def step(self):
        if self.halted:
            raise RuntimeError("machine halted")
        if fetch(self.emu.cells) == PRINT_OP:
            self._emit(readout(self.emu.cells), True)
            self.halted = True
            return []
        return super().step()

    def text(self, n: int, cls: int = OTHER) -> None:
        """Qwen writes n text tokens (the circuit must hand over)."""
        self._emit([cls] * n, False)


def owner_expected(m: BridgeMachine) -> list[int | None]:
    """1 where the circuit must predict the next token, 0 where Qwen does, None for host-only."""
    out = []
    toks, gen = m.tokens, m.generated
    for p, t in enumerate(toks):
        if not is_text(t):
            out.append(1)
        elif p + 1 < len(toks) and gen[p + 1]:
            out.append(1)  # inside a print
        else:
            out.append(0)
    return out
