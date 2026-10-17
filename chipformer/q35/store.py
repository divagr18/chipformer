"""The object store (v1): a small filesystem of named texts inside the model.

Qwen uses it with native tool calls:

    put(name, text)     store text under name (replaces it)
    append(name, text)  add text at the end (creates the file if it is missing)
    get(name)           the stored text, token for token
    list()              the names, one per line
    delete(name)        forget a name

The circuit captures the name and the text *as Qwen writes them* inside the tool call (no extra
trace tokens), keeps them in Gated DeltaNet state, and answers at ``</tool_call>`` with a few
chip records (register and device writes) and a ``<tool_response>`` it prints itself. Any Qwen
token can be stored and printed: every vocabulary row carries an 18-bit code of its own id
(``ID_BITS``), so the circuit reads a text token's id and can predict any token.

Layout (``Device``):

  content   chunk g holds the text of one put / append call, up to CMAX tokens (rows of
            8 tokens; Qwen3.5-4B's two value heads per key head hold 4 each)
  meta      chunk g: (length, next chunk, has next), a file is a chain of chunks
  dir       entry k: (name tokens, name length, alive), heads of its chain in dirH / dirT

The registers live in unused cells between the keys and the display, the ports there too: a
write to a port stores a row of a device table (the row is chosen by a register).

``StoreMachine`` is the reference: it extends ``bridge.BridgeMachine`` with Qwen's tool calls and
the circuit's answers, token for token.
"""

from dataclasses import dataclass, field

from chipformer import spec
from chipformer.q35 import bridge as B

ID_BITS = 18  # Qwen3.5's vocabulary (248320 ids) in 18 bits
QWEN_VOCAB = 248320
SPECIAL = 248044  # <|endoftext|> and up: never stored or printed by the store
NMAX = 6  # name tokens kept (a name's trailing newline counts)
# text tokens per call (one chunk); a longer text keeps its first CMAX. The circuit counts them
# with a row that halves per token (2^-n), and 2^-(n+14) must stay a normal float: n <= 112
CMAX = 104
CHUNK = 128  # address stride of a chunk
NFILES_MAX = 127

# registers and ports (cells 0x1040.. are unused by CHIP-8)
NEXTG, NFILES, SK, GCUR, TMP, TMPL = range(0x1040, 0x1046)
DIRW, DIRH, DIRT, METAW, METAN = range(0x1050, 0x1055)
REGS = {"NEXTG": NEXTG, "NFILES": NFILES, "SK": SK, "GCUR": GCUR, "TMP": TMP, "TMPL": TMPL}
REG_BITS = {NEXTG: 10, NFILES: 7, SK: 7, GCUR: 10, TMP: 10, TMPL: 7}
PORTS = {"DIRW": DIRW, "DIRH": DIRH, "DIRT": DIRT, "METAW": METAW, "METAN": METAN}

OPS = ("put", "append", "get", "list", "delete")

# Qwen3.5 token ids (tests check them against the tokenizer)
Q_NL, Q_EQ, Q_FUNCTION, Q_PARAMETER, Q_LTSLASH = 198, 28, 1628, 15704, 510
Q_LT, Q_GT = 27, 29
# the function name: "<function=put>" is "<" "function" "=" "put" ">", "=get" / "=list" are one token
OP_BARE = {"put": 611, "append": 4928, "delete": 4394, "get": 447, "list": 1551}
OP_JOINED = {"get": 27362, "list": 39619}
PARAM_BARE = {"name": 591, "text": 1272}
PARAM_JOINED = {"name": 37943, "text": 46696}
MSG = {"ok": [547], "not_found": [1856, 1669], "empty": [3092]}  # "ok", "not" " found", "empty"
TOOLS = [
    {"type": "function", "function": {
        "name": "put", "description": "Store a text under a name in your memory (replaces it).",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "a short name"},
            "text": {"type": "string", "description": f"the text (up to {CMAX} tokens)"}},
            "required": ["name", "text"]}}},
    {"type": "function", "function": {
        "name": "append", "description": "Add a text at the end of a stored text.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "text": {"type": "string"}},
            "required": ["name", "text"]}}},
    {"type": "function", "function": {
        "name": "get", "description": "The text stored under a name, exactly.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                       "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "list", "description": "The stored names.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "delete", "description": "Forget a stored name.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                       "required": ["name"]}}},
]


def qtok(qid: int, cls: int | None = None) -> int:
    """A Qwen text token as the IR sees it: its class below bit 16, its id above."""
    return (qid << 16) | (B.OTHER if cls is None else cls)


def qid_of(t: int) -> int | None:
    return t >> 16 if t >> 16 else None


def cls_of(t: int) -> int:
    return t & 0xFFFF


# --- the vocabulary: per-token features of Qwen's rows -------------------------------------


def default_tokenizer() -> str:
    """A Qwen3.5 tokenizer.json: $QWEN_TOKENIZER, or one of the usual checkpoint folders."""
    import os
    from pathlib import Path

    cands = [os.environ.get("QWEN_TOKENIZER", "")]
    for root in (Path(__file__).resolve().parents[2], Path("D:/weights"), Path.home() / "weights"):
        for sub in ("qwen3.5-0.8b-chip8", "q4-v15", "q4-v14", "q4-v13", "q35-tiny"):
            cands.append(str(root / "dist" / sub / "tokenizer.json"))
    for c in cands:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError("no Qwen3.5 tokenizer.json found; set QWEN_TOKENIZER")


class Vocab:
    """Qwen token strings, their text classes and the store's per-row features."""

    def __init__(self, tokenizer_json: str | None = None):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tokenizer_json or default_tokenizer()))
        self.tok = tok
        self.strings = {i: tok.decode([i]) for i in range(SPECIAL)}
        self.class_of: dict[int, int] = {}
        for cls, s in B.QWEN_TEXT.items():
            (i,) = tok.encode(s, add_special_tokens=False).ids
            self.class_of[i] = cls
        for cls, ss in B.EXTRA_TEXT.items():
            for s in ss:
                (i,) = tok.encode(s, add_special_tokens=False).ids
                self.class_of[i] = cls

    def has_nl(self, qid: int) -> bool:
        return "\n" in self.strings.get(qid, "")

    def cls(self, qid: int) -> int:
        return self.class_of.get(qid, B.OTHER)

    def tok_ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def tokens(self, qids: list[int]) -> list[int]:
        return [qtok(q, self.cls(q)) for q in qids]


def id_code(qid: int | None) -> list[int]:
    """The 18 id bits of a token row (0 for chip tokens and for the special tokens)."""
    if qid is None or qid >= SPECIAL:
        return [0] * ID_BITS
    return [qid >> i & 1 for i in range(ID_BITS)]


# --- the reference ---------------------------------------------------------------------------


@dataclass
class Entry:
    name: tuple
    length: int
    alive: int


@dataclass
class Device:
    content: dict = field(default_factory=dict)  # (chunk, index) -> qid
    meta: dict = field(default_factory=dict)  # chunk -> (length, next, has_next)
    dir: dict = field(default_factory=dict)  # entry -> Entry
    dir_h: dict = field(default_factory=dict)
    dir_t: dict = field(default_factory=dict)


class StoreScan:
    """What the circuit captures from Qwen's tool calls (the reference for its capture).

    Capture happens only inside a tool call (from <tool_call> to </tool_call>), so the store's
    own prints, which may contain any text, never capture anything."""

    def __init__(self, vocab: Vocab, n_chunks: int):
        self.v = vocab
        self.n_chunks = n_chunks
        self.prev = self.prev2 = None
        self.in_call = False
        self.op = None
        self.pk = None  # "name" / "text" while inside a parameter
        self.nlc = 0  # newline tokens since the parameter's name
        self.name: list[int] = []  # name value tokens (all of them; NMAX are kept)
        self.n_text = 0  # text value tokens

    def feed(self, t: int, nextg: int, dev: Device) -> bool:
        """Feed one token; returns True at a store call's </tool_call>."""
        q, c = qid_of(t), cls_of(t)
        prev, prev2 = self.prev, self.prev2
        self.prev, self.prev2 = q, prev
        if c == B.TC_OPEN:
            self.in_call = True
            self.op, self.pk, self.nlc, self.name, self.n_text = None, None, 0, [], 0
            return False
        if c == B.TC_CLOSE or t in (spec.STEP, spec.BOOT):
            was = self.in_call
            self.in_call = False
            return c == B.TC_CLOSE and was and self.op is not None
        if q is None or not self.in_call:  # chip tokens, and text outside tool calls
            return False
        for op, i in OP_BARE.items():
            if q == i and prev == Q_EQ and prev2 == Q_FUNCTION:
                self.op = op
        for op, i in OP_JOINED.items():
            if q == i and prev == Q_FUNCTION:
                self.op = op
        for p, i in PARAM_BARE.items():
            if q == i and prev == Q_EQ and prev2 == Q_PARAMETER:
                self._param(p)
                return False
        for p, i in PARAM_JOINED.items():
            if q == i and prev == Q_PARAMETER:
                self._param(p)
                return False
        closer = q == Q_PARAMETER and prev == Q_LTSLASH
        nl = self.v.has_nl(q)
        if self.pk is not None and not closer:
            if nl:
                self.nlc += 1
            if self.nlc >= 2 or (self.nlc >= 1 and not nl):
                if self.pk == "name":
                    self.name.append(q)
                else:
                    if self.n_text < CMAX and nextg < self.n_chunks:
                        dev.content[nextg, self.n_text] = q if q < SPECIAL else 0
                    self.n_text += 1
        if closer:
            self.pk = None
        return False

    def _param(self, p):
        self.pk, self.nlc = p, 0
        if p == "name":
            self.name = []
        else:
            self.n_text = 0

    def name_key(self) -> tuple:
        """The name as the directory compares it: (length, the first NMAX tokens)."""
        n = max(min(len(self.name), NMAX + 2) - 1, 0)  # the closer's "</" was counted
        toks = [q if q < SPECIAL else 0 for q in self.name[:NMAX]]
        return n, tuple(toks + [0] * (NMAX - len(toks)))

    def text_len(self) -> int:
        return max(min(self.n_text, CMAX + 1) - 1, 0)


class StoreMachine(B.BridgeMachine):
    """Reference trace with the object store: Qwen's text and tool calls, the circuit's
    capture and answers."""

    def __init__(self, vocab: Vocab, rom=None, quirks=None, n_chunks: int = 640):
        from chipformer.q35.lib import bcd

        super().__init__(rom or bcd.ROM, quirks or bcd.QUIRKS)
        self.v = vocab
        self.dev = Device()
        self.n_chunks = n_chunks
        self.sscan = StoreScan(vocab, n_chunks)
        self.answers: list[tuple] = []
        for t in self.tokens:
            self.sscan.feed(t, 0, self.dev)

    @classmethod
    def resume(cls, prev: "StoreMachine") -> "StoreMachine":
        """A new session (its own boot record and text) whose store is ``prev``'s: what a
        restored snapshot gives (store_snapshot.py)."""
        import copy

        m = cls(prev.v, n_chunks=prev.n_chunks)
        m.dev = copy.deepcopy(prev.dev)
        for a in REGS.values():
            m.emu.cells[a] = prev.emu.cells[a]
        return m

    # --- the stream ------------------------------------------------------------------------

    def _emit(self, tokens: list[int], generated: bool) -> None:
        for t in tokens:
            fire = self.sscan.feed(t, self.emu.cells[NEXTG], self.dev)
            super()._emit([t], generated)
            if fire and not generated:
                self._respond()

    def qwen_text(self, qids: list[int]) -> None:
        """Qwen writes these tokens (no arithmetic is answered inside the store's tests)."""
        self._emit(self.v.tokens(qids), False)

    def call(self, op: str, name: list[int] | None = None, text: list[int] | None = None):
        """Qwen's native tool call, as Qwen3.5's template spells it; returns the answer."""
        t = [248058, Q_NL, Q_LT, Q_FUNCTION]  # <tool_call>
        t += [OP_JOINED[op]] if op in OP_JOINED else [Q_EQ, OP_BARE[op]]
        t += [Q_GT, Q_NL]
        for p, val in (("name", name), ("text", text)):
            if val is None:
                continue
            t += [Q_LT, Q_PARAMETER, PARAM_JOINED[p], Q_GT, Q_NL, *val, Q_NL, Q_LTSLASH,
                  Q_PARAMETER, Q_GT, Q_NL]
        t += [Q_LTSLASH, Q_FUNCTION, Q_GT, Q_NL, 248059]  # </tool_call>
        n = len(self.answers)
        self._emit(self.v.tokens(t), False)
        assert len(self.answers) == n + 1, "the call did not reach the store"
        return self.answers[-1]

    # --- the circuit's answer --------------------------------------------------------------

    def _print_cls(self, cls: int) -> None:
        self._emit([qtok(full_qid(cls), cls)], True)

    def _print_qid(self, q: int) -> None:
        self._emit([qtok(q, self.v.cls(q))], True)

    def _write(self, a: int, v: int) -> None:
        cells, dev = self.emu.cells, self.dev
        self.emu.apply([(a, v)])
        if a == DIRW:
            n, toks = self.sscan.name_key()
            dev.dir[cells[SK]] = Entry(toks, n, v & 1)
        elif a == DIRH:
            dev.dir_h[cells[SK]] = v
        elif a == DIRT:
            dev.dir_t[cells[SK]] = v
        elif a == METAW:
            if cells[NEXTG] < self.n_chunks:
                dev.meta[cells[NEXTG]] = (v, 0, 0)
        elif a == METAN:
            if cells[TMP] < self.n_chunks:
                dev.meta[cells[TMP]] = (cells[TMPL], v, 1)
        self._emit([spec.addr_token(a), spec.val_token(v)], True)

    def _match(self, k: int) -> bool:
        e = self.dev.dir.get(k)
        n, toks = self.sscan.name_key()
        return e is not None and e.alive == 1 and e.name == toks and e.length == n

    def _respond(self) -> None:
        op, cells, dev = self.sscan.op, self.emu.cells, self.dev
        nf, g = cells[NFILES], cells[NEXTG]
        header = (B.NL, B.TR_OPEN, B.NL)

        def head():
            for cls in header:
                self._print_cls(cls)

        def msg(name):
            for q in MSG[name]:
                self._print_qid(q)
            self._print_cls(B.NL)

        def close():
            self._print_cls(B.TR_CLOSE)
            self._print_cls(B.NL)

        if op == "list":
            head()
            if nf == 0:
                msg("empty")
            else:
                self._write(SK, 0)
                while cells[SK] != nf:
                    e = dev.dir.get(cells[SK])
                    if e is not None and e.alive:
                        for q in e.name[:min(e.length, NMAX)]:
                            self._print_qid(q)
                    self._write(SK, cells[SK] + 1)
            close()
            self.answers.append(("list",))
            return
        self._write(SK, 0)
        while not (self._match(cells[SK]) or cells[SK] == nf):
            self._write(SK, cells[SK] + 1)
        found = self._match(cells[SK])
        k = cells[SK]
        if op in ("put", "append") and not (op == "append" and found):
            if not found:
                self._write(NFILES, nf + 1)
            self._write(DIRW, 1)
            self._write(DIRH, g)
            self._write(DIRT, g)
        elif op == "append":
            self._write(TMP, dev.dir_t.get(k, 0))
            self._write(TMPL, dev.meta.get(cells[TMP], (0, 0, 0))[0])
            self._write(METAN, g)
            self._write(DIRT, g)
        if op in ("put", "append"):
            self._write(METAW, self.sscan.text_len())
            self._write(NEXTG, g + 1)
            head()
            msg("ok")
        elif op == "delete":
            if found:
                self._write(DIRW, 0)
            head()
            msg("ok" if found else "not_found")
        elif op == "get":
            head()
            if not found:
                msg("not_found")
            else:
                self._write(GCUR, dev.dir_h.get(k, 0))
                while True:
                    n, nxt, has = dev.meta.get(cells[GCUR], (0, 0, 0))
                    for i in range(n):
                        self._print_qid(dev.content.get((cells[GCUR], i), 0))
                    if not has:
                        break
                    self._write(GCUR, nxt)
        close()
        self.answers.append((op, found))

    def stored(self, name_key) -> list[int] | None:
        """The text of a file (for tests): its chunks' tokens in order."""
        for k, e in self.dev.dir.items():
            if e.alive and (e.length, e.name) == name_key:
                out, g = [], self.dev.dir_h[k]
                while True:
                    n, nxt, has = self.dev.meta.get(g, (0, 0, 0))
                    out += [self.dev.content.get((g, i), 0) for i in range(n)]
                    if not has:
                        return out
                    g = nxt
        return None


def full_qid(cls: int) -> int:
    """The Qwen id of a printable class (its one spelling)."""
    return QWEN_CLASS_IDS[cls]


# the ids of the classes the store prints (Qwen3.5 tokenizer; checked by the tests)
QWEN_CLASS_IDS = {B.NL: Q_NL, B.TR_OPEN: 248066, B.TR_CLOSE: 248067, B.TC_OPEN: 248058,
                  B.TC_CLOSE: 248059}
