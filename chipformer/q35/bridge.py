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
from chipformer.trace import Machine, encode_writes

TEXT_BASE = spec.VOCAB_SIZE  # 8708
HEX = list(range(TEXT_BASE, TEXT_BASE + 16))  # '0'..'9', 'a'..'f'
NL, TR_OPEN, TR_CLOSE, TC_OPEN, TC_CLOSE, OTHER, FN_ADD, FN_MUL, FN_SUB = range(
    TEXT_BASE + 16, TEXT_BASE + 25)
# inline arithmetic ("37 \\times 43 =", see inline_scan)
SPACE, EQ, PLUS, MINUS, TIMES, BSL, TIMESW, COMMA, DOLLAR, LPAREN, WS = range(TEXT_BASE + 25,
                                                                           TEXT_BASE + 36)
# "$" splits by spelling: the circuit prints "$", "$$" and "." to close an inline result, so those
# classes may have no other spellings; " $", " $$" and "\$" (currency) are their own classes
SDOLLAR, DDOLLAR, SDDOLLAR, CURR, PERIOD = range(TEXT_BASE + 36, TEXT_BASE + 41)
DOLLARS = (DOLLAR, SDOLLAR, DDOLLAR, SDDOLLAR, CURR)  # all neutral, all may precede a number
# " -" starts a negative inline result, so its other spellings ("-", "−") are their own class
MINUS2 = TEXT_BASE + 41
# division: "/" (" /", "÷"), the tool name "div" (also "\div"), and "..." after an inexact quotient
DIV, FN_DIV, ELLIPSIS = range(TEXT_BASE + 42, TEXT_BASE + 45)
# signs: "(-" (a sign after the same tokens as "-") and " (-" (always one); ")" after a digit
# (closing a signed number); mod (" mod", " %"; "mod" names the tool and follows "\\" or "\\b");
# gcd ("gcd", " gcd": the tool name and an operator written first, "gcd(A, B) =")
LPMINUS, SLPMINUS, RPAREN, MODOP, FN_MOD, FN_GCD = range(TEXT_BASE + 45, TEXT_BASE + 51)
N_TEXT = 51
FUNCS = {"add": FN_ADD, "mul": FN_MUL, "sub": FN_SUB, "div": FN_DIV, "mod": FN_MOD,
         "gcd": FN_GCD}  # tool names it answers
PRINT_OP = 0x00FD
MAX_PRINT = 58  # digits per print (the write counter tops out at 64)
MAX_ARG = 24  # digits per tool-call argument

# Qwen3.5 token ids of the classes (single tokens in its tokenizer)
QWEN_TEXT = {**{TEXT_BASE + i: c for i, c in enumerate("0123456789abcdef")},
             NL: "\n", TR_OPEN: "<tool_response>", TR_CLOSE: "</tool_response>",
             TC_OPEN: "<tool_call>", TC_CLOSE: "</tool_call>", FN_ADD: "add", FN_MUL: "mul",
             FN_SUB: "sub", SPACE: " ", EQ: " =", PLUS: " +", MINUS: " -", TIMES: " \u00d7",
             BSL: " \\", TIMESW: "times", COMMA: ",", DOLLAR: "$", LPAREN: "(",
             WS: "\n\n", SDOLLAR: " $", DDOLLAR: "$$", SDDOLLAR: " $$", CURR: " \\$",
             PERIOD: ".", MINUS2: "-", DIV: "/", FN_DIV: "div", ELLIPSIS: "...",
             LPMINUS: "(-", SLPMINUS: " (-", RPAREN: ")", MODOP: " mod", FN_MOD: "mod",
             FN_GCD: "gcd"}
# other single tokens Qwen generates for the same thing: it writes <function=add> as "=add" and
# <function=mul> as "=m" "ul", not the way its tokenizer splits the text
# (never for a class the circuit prints: the tied lm_head rows would tie)
# (and <function=gcd> as "=g" "cd")
EXTRA_TEXT = {FN_ADD: ["=add"], FN_MUL: ["ul"], FN_SUB: ["=sub", "ub"], EQ: ["=", ")="],
              PLUS: ["+"], MINUS2: [" \u2212", "\u2212"], DIV: [" /", "\u00f7", ")/"],
              TIMES: ["\u00d7", " *", "*"], BSL: ["\\"], MODOP: [" %"], FN_GCD: [" gcd", "cd"],
              TIMESW: ["cdot"], CURR: ["\\$"], LPAREN: [" ("],
              WS: ["\n\n\n", "\n\n\n\n", "  ", "   ", "    ", "     ", " \n", " \n\n"]}
INLINE_OPS = {PLUS: "add", MINUS: "sub", MINUS2: "sub", TIMES: "mul", DIV: "div", MODOP: "mod",
              FN_GCD: "gcd"}
# and after a backslash: TIMESW ("\times", "\cdot") mul, FN_DIV ("\div") div, FN_MOD ("\mod", also
# "\bmod": "b" has the hex digit's class) mod
BSL_OPS = {TIMESW: "mul", FN_DIV: "div", FN_MOD: "mod"}
PREFIX = ("gcd",)  # written before A: "gcd(A, B) ="
SIGNED = ("mul", "div", "gcd")  # the others only answer A and B without signs
SIGNS = (MINUS, MINUS2, LPMINUS)  # a sign when it follows SIGN_PREV (" (-" always is one)
SAFE = (SPACE, NL, WS, LPAREN, *DOLLARS)  # may precede a first number (and so may chip tokens)
OPISH = (PLUS, MINUS, MINUS2, TIMES, DIV, TIMESW, FN_DIV, MODOP, FN_MOD, FN_GCD)
SIGN_PREV = SAFE + (EQ, COMMA) + OPISH
# Inline calls need at least this many digits in A and B together. Qwen gets small ones right,
# and right after a trace its conv window holds chip tokens instead of the "=", so a short
# printed result can read to it like the start of a new expression. Qwen also adds and
# subtracts numbers of a few digits reliably, so those stay its own.
INLINE_MIN_DIGITS = {"mul": 4, "add": 7, "sub": 7, "div": 4, "mod": 4, "gcd": 4}


def is_text(t: int) -> bool:
    return t >= TEXT_BASE


def readout(cells: list[int]) -> list[int]:
    from chipformer.q35.lib import bcd

    i, n = cells[spec.I], min(cells[spec.V], MAX_PRINT)
    # 10 and 11 print as "." and "..." (a quotient's decimals), so no hex digits a / b
    shown = {**dict(enumerate(HEX)), 10: PERIOD, 11: ELLIPSIS}
    digits = [shown[cells[(i + k) & 0xFFF] & 15] for k in range(n)]
    mode = cells[bcd.MODE]
    if mode & 1:  # an inline call: " digits" (" -digits" below 0) and a closer (the math's
        close = DDOLLAR if mode & 4 else DOLLAR if mode & 2 else PERIOD  # "$"/"$$", else ".")
        return [MINUS if mode & 8 else SPACE, *digits, close] if digits else [SPACE]
    # a tool call: " -" instead of the newline before a negative result
    return [NL, TR_OPEN, MINUS if mode & 8 and digits else NL, *digits, NL, TR_CLOSE, NL]


class InlineScan:
    """Which "A op B =" the circuit answers: the reference for its capture logic.

    Text since the last *breaker* token (anything but digits, spaces, operators, signs, "=",
    "$", "(", a backslash, or a comma or ")" right after a digit) must be exactly: digits, one
    operator, digits, then "=" (for gcd: the operator first, "gcd(A, B) ="), with at least
    INLINE_MIN_DIGITS[op] digits in the two numbers together. A comma between digits joins them
    (1,234). The first number must start after a space, newline, other whitespace, "$", "(" or
    a chip token, so "2.5 * 3 =" and "x^2 + 1 =" are left alone.

    A sign is "-" (any spelling) or "(-" right after one of SIGN_PREV, or " (-"; it is no
    operator, and the number right after it is negative. mul, div and gcd take signed numbers
    (the result is negative when exactly one of A, B is, for mul and div); the others do not
    fire on them. Chip tokens are breakers too. Fed every token of the stream, in order.
    """

    def __init__(self):
        self.prev = self.prev2 = None  # None: no token yet (treated like a chip token)
        self.p = self.nops = self.d = 0
        self.lens = {1: 0, 2: 0}
        self.neg = {1: False, 2: False}
        self.digits = {}
        self.op, self.op_ok, self.a_safe, self.in_call = None, False, False, False
        self.opener = 0  # what the first number follows: 1 "$", 2 "$$" (maybe with a space)

    def negative(self) -> bool:
        """Whether the result of the call being made prints with a minus (MODE bit 8)."""
        return self.op in ("mul", "div") and self.neg[1] != self.neg[2]

    def mode(self) -> int:
        """The CALL record's MODE cell for an inline call: 1 | 2 inside "$" | 4 inside "$$"
        | 8 negative."""
        return (1 | (2 if self.opener else 0) | (4 if self.opener == 2 else 0)
                | (8 if self.negative() else 0))

    def feed(self, t: int):
        """Returns (op, a, b) if the circuit takes over after token t, else None."""
        prev, prev2 = self.prev, self.prev2
        self.prev, self.prev2 = t, prev
        text = is_text(t)
        dig = text and t in HEX[:10]

        def isd(x):
            return x is not None and x in HEX[:10]

        def safe(x):
            return x is not None and (not is_text(x) or x in SAFE)

        def sign_prev(x):
            return safe(x) or x in SIGN_PREV

        if t in (TC_OPEN, TC_CLOSE) or t in (spec.STEP, spec.BOOT):
            self.in_call = t == TC_OPEN
        if t == TC_OPEN:
            self.op = None  # a tool call needs its own function name
        cont = dig and (isd(prev) or (prev == COMMA and isd(prev2)))
        run_start = dig and not cont
        neg = run_start and ((prev in SIGNS and sign_prev(prev2)) or prev == SLPMINUS)
        bs_op = t in BSL_OPS and (prev == BSL or (t == FN_MOD and prev == HEX[11]
                                                  and prev2 == BSL))
        sign_minus = t in (MINUS, MINUS2) and sign_prev(prev)
        op_tok = (t in INLINE_OPS or bs_op) and not sign_minus
        neutral = text and (dig or op_tok or sign_minus
                            or t in (SPACE, LPAREN, BSL, EQ, LPMINUS, SLPMINUS, *DOLLARS)
                            or (t in (COMMA, RPAREN) and isd(prev))
                            or (t == HEX[11] and prev == BSL))
        brk = not neutral
        after_eq = prev == EQ and not brk and not self.in_call  # a fresh window after "="
        if t == TC_OPEN or (brk and not self.in_call) or after_eq:
            self.p = 0
        elif run_start:
            self.p += 1
            if self.p == 1:
                self.a_safe = safe(prev) or neg

                def kind(x):
                    return 1 if x in (DOLLAR, SDOLLAR) else 2 if x in (DDOLLAR, SDDOLLAR) else 0

                after = prev in (SPACE, *SIGNS, SLPMINUS)
                self.opener = kind(prev) or (kind(prev2) if after else 0)
            if self.p in (1, 2):
                self.neg[self.p] = neg
        if run_start:
            self.d = 0
        elif cont:
            self.d += 1
        if brk or after_eq:
            self.nops = 0
        elif op_tok:
            self.nops += 1
        if dig and self.p in (1, 2):
            self.lens[self.p] = self.d + 1 if self.d < MAX_ARG else 0
            if self.d < MAX_ARG:
                self.digits[self.p, self.d] = HEX.index(t)
        fn = {cls: name for name, cls in FUNCS.items()}.get(t)
        if fn and self.in_call:
            self.op, self.op_ok = fn, self.p == 1
        if op_tok and not self.in_call:
            self.op = BSL_OPS[t] if bs_op else INLINE_OPS[t]
            self.op_ok = self.p == (0 if self.op in PREFIX else 1)
        if (t == EQ and not self.in_call and self.p == 2 and self.nops == 1 and self.op_ok
                and self.a_safe and self.lens[1] and self.lens[2] and self.op
                and self.lens[1] + self.lens[2] >= INLINE_MIN_DIGITS[self.op]
                and (self.op in SIGNED or not (self.neg[1] or self.neg[2]))):
            a = "".join(str(self.digits[1, w]) for w in range(self.lens[1]))
            b = "".join(str(self.digits[2, w]) for w in range(self.lens[2]))
            return self.op, a, b
        return None


class BridgeMachine(Machine):
    """Reference trace for the v2 circuit: machine records, prints, and Qwen's text."""

    def __init__(self, rom: bytes, quirks=None):
        super().__init__(rom, quirks)
        self.halted = False
        self.scan = InlineScan()
        for t in self.tokens:
            self.scan.feed(t)

    def _emit(self, tokens: list[int], generated: bool) -> None:
        self.fired = None
        for t in tokens:
            self.fired = self.scan.feed(t)
        super()._emit(tokens, generated)

    def qwen(self, classes: list[int]) -> list[tuple[str, str, str]]:
        """Qwen writes these text tokens one at a time; wherever the circuit answers an inline
        "A op B =", it takes over (STEP, the CALL record, the routine, the print)."""
        calls = []
        for t in classes:
            self._emit([t], False)
            if self.fired:
                calls.append(self.fired)
                self._call(*self.fired, mode=self.scan.mode())
                while not self.halted:
                    self.step()
        return calls

    def step(self):
        if self.halted:
            raise RuntimeError("machine halted")
        op = fetch(self.emu.cells)
        if op == PRINT_OP:
            self._emit(readout(self.emu.cells), True)
            self.halted = True
            return []
        from chipformer.q35.lib.macros import macro_writes

        w = macro_writes(self.emu.cells, op)
        if w is not None:
            self.emu.apply(w)
            self._emit(encode_writes(w) + [spec.STEP], True)
            return w
        return super().step()

    def text(self, n: int, cls: int = OTHER) -> None:
        """Qwen writes n text tokens (the circuit must hand over)."""
        self._emit([cls] * n, False)

    def tool_call(self, op: str, a: str, b: str) -> None:
        """Qwen calls a library routine with its native syntax; the circuit then writes the
        CALL record (the arguments into RAM, PC = the routine) and the machine runs it."""

        def lit(n):
            return [OTHER] * n

        def num(s):  # a minus is its own token: Qwen writes "\n-1418261\n"
            return [MINUS2] * s.startswith("-") + [HEX[int(ch)] for ch in s.lstrip("-")]

        text = ([TC_OPEN, NL] + lit(2) + [FUNCS[op]] + lit(1) + [NL] + lit(3) + [NL] + num(a)
                + [NL] + lit(3) + [NL] + lit(3) + [NL] + num(b) + [NL] + lit(3) + [NL] + lit(3)
                + [NL, TC_CLOSE])
        self._emit(text, False)
        self._call(op, a.lstrip("-"), b.lstrip("-"), mode=8 * self.scan.negative())

    def _call(self, op: str, a: str, b: str, mode: int) -> None:
        from chipformer.q35.lib import bcd

        w = bcd.call_writes(op, a, b, mode)
        self.emu.apply(w)
        self._emit([spec.STEP] * (mode & 1) + encode_writes(w) + [spec.STEP], True)
        self.halted = False


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
