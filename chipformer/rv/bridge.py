"""The P3 bridge between Qwen's text and the RV32 machine (the reference the circuit must match).

Qwen calls a tool with its native syntax:

    <tool_call>\\n<function=calc>\\n<parameter=expression>\\n123*456\\n</parameter>\\n...</tool_call>

- Between ``<tool_call>`` and ``</tool_call>`` every text token is captured: row i of the text
  buffer (TEXT_ADDR + 16 i, i = 1 .. CAP_MAX) holds the i-th token's UTF-8 bytes, NUL-padded.
  The buffer is cleared at ``<tool_call>``.
- At ``</tool_call>`` the circuit takes over: it predicts GO, and the machine runs from where it
  parked (the firmware's ``exit`` ecall, HALT, is the park; GO resumes after it).
- Every putchar is a BYTE record followed by its echo: the Qwen token of that byte (or, for
  256.., a special token such as ``<tool_response>``), which Qwen reads as ordinary text. HALT
  is echoed as a newline, and Qwen predicts again from that newline on.
- The circuit owns (predicts) the positions from ``</tool_call>`` through the last echo but
  one; Qwen never sees the chip tokens of a run (decoding with transparent chip tokens).

A token of the circuit's stream is a chip token (< spec.VOCAB) or ``spec.TEXT_TOKEN | qwen_id``.
"""

from functools import cached_property
from pathlib import Path

from chipformer.rv import ref
from chipformer.rv import spec as S

# special tokens the firmware prints with putchar(256 + k)
SPECIALS = ["<tool_response>", "</tool_response>", "<think>", "</think>"]


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte-level BPE alphabet: byte -> the character that stands for it."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Text:
    """Qwen's vocabulary as the bridge sees it: each token's bytes, and the printable tokens."""

    def __init__(self, tokenizer_json: str | Path) -> None:
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(tokenizer_json))
        vocab = self.tok.get_vocab(with_added_tokens=True)
        self.n = max(vocab.values()) + 1
        dec = {c: b for b, c in bytes_to_unicode().items()}
        added = {a.content for a in self.tok.get_added_tokens_decoder().values()} \
            if hasattr(self.tok, "get_added_tokens_decoder") else set()
        self.bytes = [b""] * self.n
        for s, i in vocab.items():
            if s in added or not all(ch in dec for ch in s):
                self.bytes[i] = s.encode()
            else:
                self.bytes[i] = bytes(dec[ch] for ch in s)
        ids = {s: i for s, i in vocab.items()}
        self.TCO, self.TCC = ids["<tool_call>"], ids["</tool_call>"]
        u = bytes_to_unicode()
        self.printable = [ids[u[b]] for b in range(256)] + [ids[s] for s in SPECIALS]
        self.pindex = {q: p for p, q in enumerate(self.printable)}
        self.NL = self.printable[10]

    def printable_index(self, qid: int) -> int | None:
        return self.pindex.get(qid)

    def row_bytes(self, qid: int) -> bytes:
        """A token's text-buffer row: its first TEXT_ROW bytes, NUL-padded."""
        b = self.bytes[qid] if 0 <= qid < self.n else b""
        return b[:S.TEXT_ROW].ljust(S.TEXT_ROW, bytes(1))

    def encode(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False).ids

    def decode_bytes(self, qids: list[int]) -> bytes:
        return b"".join(self.bytes[q] for q in qids)

    @cached_property
    def byte_token(self) -> dict[int, int]:
        return {b: q for b, q in enumerate(self.printable[:256])}


def default_tokenizer() -> Path:
    """A Qwen3.5 tokenizer.json from the local HF cache."""
    import glob
    import os

    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    for name in ("Qwen3.5-4B", "Qwen3.5-0.8B", "Qwen3.5-9B"):
        hits = sorted(glob.glob(str(hub / f"models--Qwen--{name}" / "snapshots" / "*" /
                                    "tokenizer.json")))
        if hits:
            return Path(hits[-1])
    raise FileNotFoundError("no Qwen3.5 tokenizer.json in the HF cache")


def text_token(qid: int) -> int:
    return S.TEXT_TOKEN | qid


def park(m: "ref.Machine", max_records: int = 10**6) -> list[int]:
    """Run a freshly loaded firmware to its first park (exit ecall); returns that trace."""
    toks = m.run(max_records)
    assert m.halted and m.fault is None, (m.fault, m.out)
    m.halted = False
    return toks


class Session:
    """The token stream of a conversation with the machine, and who predicts each next token.

    ``toks[i]`` is the i-th token; ``own[i]`` is True where the circuit predicts ``toks[i + 1]``.
    The machine starts parked (its state is the snapshot the host injects)."""

    def __init__(self, text: Text, machine: "ref.Machine") -> None:
        self.T, self.m = text, machine
        machine.text = [bytes(S.TEXT_ROW)] * S.TEXT_ROWS
        self.toks: list[int] = []
        self.own: list[bool] = []
        self.armed, self.cap = False, 0
        self.runs: list[list[int]] = []  # each run's echoed Qwen tokens

    def _push(self, t: int, owner: bool) -> None:
        self.toks.append(t)
        self.own.append(owner)

    def history(self, qids: list[int]) -> None:
        """Prefilled text (an earlier turn, its tool responses already written out): captured
        like any text, but no run follows its </tool_call>s (the circuit owns those positions
        and predicts GO, but the stream goes on with the history's own tokens)."""
        self.qwen(qids, run=False)

    def qwen(self, qids: list[int], max_records: int = 10**6, run: bool = True) -> None:
        """Tokens Qwen writes (a run follows each </tool_call>)."""
        for q in qids:
            if q == self.T.TCO:
                self.armed, self.cap = True, 0
                self.m.text = [bytes(S.TEXT_ROW)] * S.TEXT_ROWS
            elif q == self.T.TCC:
                self.armed = False
            elif self.armed:
                self.cap += 1
                rows = list(self.m.text)
                rows[min(self.cap, S.CAP_MAX)] = self.T.row_bytes(q)
                self.m.text = rows
            self._push(text_token(q), q == self.T.TCC)
            if q == self.T.TCC and run:
                self.run(max_records)

    def run(self, max_records: int = 10**6) -> list[int]:
        """GO, the machine's records until HALT, with every putchar and the HALT echoed."""
        m, echo = self.m, []
        self._push(S.tok("GO"), True)
        m.halted = False
        n = 0
        while n < max_records:
            rec = m.record()
            n += 1
            for t in rec:
                self._push(t, True)
                fam, p = S.family(t)
                if fam == "BYTE":
                    q = self.T.printable[p] if p < len(self.T.printable) else self.T.NL
                    echo.append(q)
                    self._push(text_token(q), True)
            if rec[-1] == S.tok("HALT"):
                echo.append(self.T.NL)
                self._push(text_token(self.T.NL), False)
                break
        else:
            raise RuntimeError("the run did not halt")
        if m.fault is None:
            m.halted = False  # parked: the next GO resumes after the exit ecall
        self.runs.append(echo)
        return echo

    def qwen_text(self) -> list[int]:
        """The Qwen tokens of the stream (what Qwen sees: chip tokens are transparent)."""
        return [t - S.TEXT_TOKEN for t in self.toks if S.is_text(t)]
