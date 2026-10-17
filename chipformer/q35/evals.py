"""Math evals: does the computer in the weights beat the base model?

    python -m chipformer.q35.evals run COND DATA --n 100 [--model PATH] [--out dist/evals]
    python -m chipformer.q35.evals summary [--out dist/evals]

Conditions (same user prompt everywhere; only the model, the tools and the thinking flag change):

  base           base Qwen3.5-4B, no tools, non-thinking
  base_think     base, no tools, thinking mode (bigger budget)
  merged         merged 4B: BIOS boot record + tools; the circuit runs every call (one decode)
  merged_plain   merged 4B with no BIOS and no tools (should match ``base``: a fidelity check)
  pytool_inline  base + tools; a Python executor writes exactly what the circuit would print
                 (same token stream as ``merged``, minus the chip tokens)
  pytool         base + tools in the standard agent loop (tool turn in Qwen's chat template):
                 the "external calculator" upper bound

Datasets: ``gsm8k`` (openai/gsm8k test), ``gsmhard`` (reasoning-machines/gsm-hard, the same 1319
problems with larger numbers; the same seeded indices are used for both), ``arith`` (synthetic
multi-digit add / mul and mixed expressions). Problem ids are stable, so conditions are paired.

The tool set is ``demo.TOOLS``; ``PY_TOOLS`` gives each tool's exact meaning for checking.
Results go to OUT/DATA.COND.jsonl (one line per problem, resumable) and ``summary`` prints
accuracy (Wilson 95% CI), paired differences (McNemar), tool-call rate, call correctness,
machine exactness, budget cutoffs and timings.
"""

import argparse
import itertools
import json
import math
import random
import re
import sys
import time
from collections import deque
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

try:  # chip exactness: no bf16 split-K reductions in GEMMs (they silently break the circuit)
    import torch

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
except ImportError:  # summary / rescore need no torch
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.set_int_max_str_digits(0)  # transcripts can hold runaway digit strings
SUFFIX = "\n\nGive the final answer as a single number in \\boxed{} at the end."
KINDS = ("base", "merged", "merged_inline", "merged_plain", "pytool_inline", "pytool", "oracle",
         "oraclec")
# oracle / oraclec: base Qwen; wherever its text matches the circuit's inline grammar
# (bridge.InlineScan) at "=", the exact result is injected as text " R " (oraclec: " R,RRR ")
BIOS_KINDS = ("merged", "merged_inline")  # the prompt starts with the BIOS boot record
TOOL_KINDS = ("merged", "pytool_inline", "pytool")
FLAGS = ("think", "nb", "ne", "nudge", "nudge2", "nudge3", "fp", "tc", "mix")  # suffixes: thinking mode, <|im_start|> banned, nudge,
# ne = <|endoftext|> banned too (Qwen ends its turn with <|im_end|>; after a printed
# </tool_response> it sometimes writes <|endoftext|> instead of the answer)
# nudge3 = the inline nudge with division (or, with mix, products and quotients inline)
# fp = RoPE positions frozen on chip tokens (BatchDecoder freeze_chip_positions)
# tc = chip tokens invisible to Qwen's layers (BatchDecoder transparent_chip)
CONDS = ("base", "base_think", "merged", "merged_nb", "merged_nudge", "merged_nb_nudge",
         "merged_inline", "merged_inline_nudge", "merged_inline_fp", "merged_inline_nudge_fp", "merged_inline_fp_tc", "merged_inline_nudge_fp_tc",
         "base_nudge", "oracle_nudge", "oraclec_nudge", "oracle_nudge-close", "oracle_nudge2",
         "oracle_nudge2-close", "merged_plain", "pytool_inline", "pytool_inline_nudge", "pytool", "pytool_nudge")
IM_START = 248045
ENDOFTEXT = 248044
NUDGE2 = ("Exact arithmetic: whenever you write a calculation like `a \\times b =`, `a + b =` or "
          "`a - b =` with whole numbers, its exact result is filled in right after the `=`. That "
          "number is final and correct: never write another number after it or recompute it; just "
          "continue your sentence after it. So never do long multiplication or addition by hand: "
          "write the whole calculation and let the result appear.")
NUDGE = ("You have tools for exact arithmetic ({names}). Use them for every arithmetic step "
         "they cover instead of calculating by hand.")
NUDGE3 = ("Exact arithmetic: whenever you write a calculation like `a \\times b =`, `a / b =`, "
          "`a + b =` or `a - b =` with whole numbers, its exact result is filled in right after the "
          "`=` (a quotient that does not end shows 10 decimals, then `...`). So never do long "
          "multiplication or long division by hand: write the whole calculation, e.g. "
          "`918274 / 37 =`, and let the result appear.")
# mix (a tool kind): products inline, sums and differences of big numbers through the tools
NUDGE_MIX = ("Exact arithmetic: whenever you write a multiplication of whole numbers like "
             "`a \\times b =`, its exact result is filled in right after the `=`, so never "
             "multiply long numbers by hand: write the whole product and let the result appear. "
             "For additions and subtractions of large numbers, call the add and sub tools.")
NUDGE_MIX3 = ("Exact arithmetic: whenever you write a multiplication or division of whole numbers "
              "like `a \\times b =` or `a / b =`, its exact result is filled in right after the "
              "`=` (a quotient that does not end shows 10 decimals, then `...`), so never multiply "
              "or divide long numbers by hand: write the whole calculation and let the result "
              "appear. For additions and subtractions of large numbers, call the add and sub tools.")


def parse_cond(cond: str) -> tuple[str, set]:
    """'merged_nb_nudge' -> ('merged', {'nb', 'nudge'}); a '-tag' (model version) is ignored."""
    parts, flags = cond.split("-")[0].split("_"), set()
    while parts and parts[-1] in FLAGS:
        flags.add(parts.pop())
    kind = "_".join(parts)
    if kind not in KINDS:
        raise argparse.ArgumentTypeError(f"unknown condition {cond}")
    return kind, flags


def cond_arg(cond: str) -> str:
    parse_cond(cond)
    return cond


BUDGET = {False: 2048, True: 8192}  # Qwen-written tokens (thinking or not); machine tokens extra
THINK_EXTRA = 512  # answer tokens after a forced </think>
THINK_FORCE = ("\n\nConsidering the limited time by the user, I have to give the solution based on "
               "the thinking directly now.\n</think>\n\n")


def _div(a, b):
    if b == 0:
        raise ZeroDivisionError
    q = Fraction(a) / Fraction(b)
    return q.numerator if q.denominator == 1 else q


def div_text(a: int, b: int) -> str:
    """How the computer prints a / b (lib/bcd.py div): the exact quotient, or its first DEC
    decimals and "..." when it does not end."""
    from chipformer.q35.lib import bcd

    q, r = divmod(a, b)
    if not r:
        return str(q)
    digs = []
    for _ in range(bcd.DEC):
        r *= 10
        digs.append(str(r // b))
        r %= b
        if not r:
            return f"{q}." + "".join(digs)
    return f"{q}." + "".join(digs) + "..."


# exact meaning of each tool (by name); demo.TOOLS decides which ones the model sees
PY_TOOLS = {"add": lambda a, b: a + b, "mul": lambda a, b: a * b,
            "sub": lambda a, b: a - b, "div": _div}


def tools():
    from chipformer.q35 import demo

    return demo.TOOLS


def tool_names():
    return [t["function"]["name"] for t in tools()]


# ---------------------------------------------------------------------------------- datasets

def _num(s):
    s = str(s).replace(",", "").strip()
    try:
        v = Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return v


def load_data(name: str, n: int, seed: int = 0, idx=None) -> list[dict]:
    """``n`` seeded problems, or exactly the problems ``idx`` (dataset indices)."""
    if name == "arith":
        probs = arith(max(idx) + 1 if idx else n, seed)
        return [probs[i] for i in idx] if idx else probs
    from datasets import load_dataset

    if idx is None:
        idx = sorted(random.Random(seed).sample(range(1319), n))
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        out = []
        for i in idx:
            r = ds[i]
            gold = _num(r["answer"].split("####")[-1])
            steps = [_num(v) for v in re.findall(r"<<[^=<>]*=([^<>]*)>>", r["answer"])]
            out.append({"id": f"gsm8k-{i}", "question": r["question"], "gold": str(gold),
                        "steps": [str(v) for v in steps if v is not None]})
        return out
    if name == "gsmhard":
        ds = load_dataset("reasoning-machines/gsm-hard", split="train")
        out = []
        for i in idx:
            r = ds[i]
            out.append({"id": f"gsmhard-{i}", "question": r["input"],
                        "gold": repr(float(r["target"]))})
        return out
    raise ValueError(name)


ARITH_KINDS = ["add", "mul", "sub", "add3", "muladd", "mulmul", "sumprod", "mulsub"]


def arith(n: int, seed: int = 0) -> list[dict]:
    """Seeded synthetic arithmetic with +, - (non-negative results) and *.

    Each item lists the calls a perfect solver makes (``calls``: [op, a, b], operands in any
    order) for checking tool-call arguments."""
    rng = random.Random(seed)

    def r(lo, hi):
        d = rng.randint(lo, hi)
        return rng.randrange(10 ** (d - 1), 10**d)

    out = []
    for k in range(n):
        kind = ARITH_KINDS[k % len(ARITH_KINDS)]
        if kind == "add":
            a, b = r(6, 12), r(5, 12)
            q, gold, calls = f"{a} + {b}", a + b, [("add", a, b)]
        elif kind == "mul":
            a, b = r(4, 6), r(3, 5)
            q, gold, calls = f"{a} * {b}", a * b, [("mul", a, b)]
        elif kind == "sub":
            a, b = r(8, 12), r(6, 11)
            a, b = max(a, b), min(a, b)
            q, gold, calls = f"{a} - {b}", a - b, [("sub", a, b)]
        elif kind == "mulsub":
            a, b, c = r(3, 5), r(2, 4), r(3, 6)
            if c > a * b:
                c = c % (a * b)
            q, gold, calls = f"{a} * {b} - {c}", a * b - c, [("mul", a, b), ("sub", a * b, c)]
        elif kind == "add3":
            a, b, c = r(5, 9), r(5, 9), r(4, 9)
            q, gold, calls = f"{a} + {b} + {c}", a + b + c, [("add", a, b), ("add", a + b, c)]
        elif kind == "muladd":
            a, b, c = r(3, 5), r(2, 4), r(4, 8)
            q, gold, calls = f"{a} * {b} + {c}", a * b + c, [("mul", a, b), ("add", a * b, c)]
        elif kind == "mulmul":
            a, b, c = r(2, 4), r(2, 3), r(2, 3)
            q, gold, calls = f"{a} * {b} * {c}", a * b * c, [("mul", a, b), ("mul", a * b, c)]
        elif kind == "sumprod":
            a, b, c, d = r(2, 4), r(2, 3), r(2, 4), r(2, 3)
            q, gold = f"{a} * {b} + {c} * {d}", a * b + c * d
            calls = [("mul", a, b), ("mul", c, d), ("add", a * b, c * d)]
        tmpl = rng.choice(["What is {q}?", "Compute {q}.", "Calculate {q} exactly."])
        out.append({"id": f"arith-{k}", "question": tmpl.format(q=q), "gold": str(gold),
                    "kind": kind, "calls": [[o, str(x), str(y)] for o, x, y in calls]})
    return out


# ---------------------------------------------------------------------------------- parsing

CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\s]*)>(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>\s]*)>\s*(.*?)\s*</parameter>", re.DOTALL)
RESP_RE = re.compile(r"\s*(?:<\|im_end\|>\s*<\|im_start\|>user\s*)?<tool_response>\s*(.*?)\s*"
                     r"</tool_response>", re.DOTALL)
MARK = re.compile(r"\[machine:\d+\]")


def parse_calls(text: str) -> list[dict]:
    """Tool calls in order, each with its arguments and the response that follows it."""
    text = MARK.sub("", text)
    calls = []
    for m in CALL_RE.finditer(text):
        args = dict(PARAM_RE.findall(m[2]))
        r = RESP_RE.match(text, m.end())
        calls.append({"op": m[1], "args": args, "response": r[1] if r else None})
    return calls


def int_arg(s):
    s = (s or "").strip()
    return int(s) if re.fullmatch(r"\d+", s) else None


def num_arg(s):
    try:
        return Fraction(Decimal((s or "").strip().replace(",", "")))
    except (InvalidOperation, ValueError):
        return None


def fmt(v) -> str:
    if isinstance(v, Fraction):
        if v.denominator == 1:
            return str(v.numerator)
        return str(round(Decimal(v.numerator) / Decimal(v.denominator), 10).normalize())
    return str(v)


def true_result(op: str, args: dict):
    """Exact value of a call by the tool's definition, or None if it is not a valid call."""
    a, b = args.get("a"), args.get("b")
    if op not in PY_TOOLS or a is None or b is None:
        return None
    x, y = num_arg(a), num_arg(b)
    if x is None or y is None:
        return None
    try:
        return PY_TOOLS[op](x, y)
    except ZeroDivisionError:
        return None


def _balanced(s: str, i: int) -> str:
    depth, j = 1, i
    while j < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i:j - 1]


NUM = re.compile(r"-?\d+(?:\.\d+)?")


def to_number(s: str):
    s = re.sub(r"\\(?:text|mathrm|textbf|mbox)\{([^{}]*)\}", r" \1 ", s)
    s = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = s.replace("\\$", " ").replace("$", " ").replace("\\,", "").replace("\\!", "")
    s = s.replace("**", " ").replace("\\%", " ").replace("%", " ")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"(?<=\d)\s(?=\d{3}\b)", "", s)  # 1 234 567
    f = re.fullmatch(r"\s*\(?(-?\d+(?:\.\d+)?)\)?\s*/\s*\(?(-?\d+(?:\.\d+)?)\)?\s*", s)
    if f and float(f[2]) != 0:
        return Fraction(Decimal(f[1])) / Fraction(Decimal(f[2]))
    nums = NUM.findall(s)
    return nums


def extract_answer(text: str, think: bool):
    """The final answer: the last \\boxed{} after the thinking and outside tool calls, else the
    last number. Returns (value as Fraction or None, where it came from)."""
    if think:
        text = "<think>" + text
    if "<think>" in text and "</think>" not in text.split("<think>")[-1]:
        return None, "unclosed_think"
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<tool_call>.*?(</tool_call>|$)", " ", text, flags=re.DOTALL)
    text = re.sub(r"<tool_response>.*?(</tool_response>|$)", " ", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^|]*\|>(assistant|user)?", " ", MARK.sub(" ", text))
    boxes = [_balanced(text, m.end()) for m in re.finditer(r"\\boxed\s*\{", text)]
    for src, cand in [("boxed", boxes[-1:]), ("last_number", [text])]:
        for c in cand:
            v = to_number(c)
            if isinstance(v, Fraction):
                return v, src
            if v:
                return Fraction(Decimal(v[0] if src == "boxed" else v[-1])), src
    return None, "none"


def is_correct(pred, gold: str) -> bool:
    if pred is None:
        return False
    g = Fraction(Decimal(gold)) if "/" not in gold else Fraction(gold)
    if abs(pred - g) <= Fraction(1, 10**6) * max(1, abs(g)):
        return True
    return g.denominator != 1 and abs(pred - g) < Fraction(1, 200)  # 2-decimal answers


# ---------------------------------------------------------------------------------- decoding

class Ctx:
    """What the decode loop needs to know about token ids."""

    def __init__(self, tok, chip=None):
        self.tok = tok
        ids = tok.convert_tokens_to_ids
        self.stop = {ids("<|im_end|>"), ids("<|endoftext|>")}
        self.tc_close = ids("</tool_call>")
        self.tr_close = ids("</tool_response>")
        self.nl = tok("\n", add_special_tokens=False).input_ids[0]
        self.space = tok(" ", add_special_tokens=False).input_ids[0]
        self.digits = {tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)}
        # v11 closes an inline result with "$", "$$" or "." (one token each)
        self.closers = {i[0] for i in (tok(c, add_special_tokens=False).input_ids
                                       for c in ("$", "$$", ".")) if len(i) == 1}
        self.chip = chip  # (offset, n) or None
        self.think_close = tok.convert_tokens_to_ids("</think>")
        self.think_force = None  # tokens that end an over-budget thinking block (run sets it)

    def is_chip(self, t: int) -> bool:
        return self.chip is not None and self.chip[0] <= t < self.chip[0] + self.chip[1]

    def ids(self, s: str) -> list[int]:
        return self.tok(s, add_special_tokens=False).input_ids

    def show(self, out: list[int]) -> str:
        """Decoded text with each run of chip tokens shown as [machine:N]."""
        parts, run, text = [], 0, []
        for t in out + [None]:
            if t is not None and self.is_chip(t):
                if text:
                    parts.append(self.tok.decode(text, skip_special_tokens=False))
                    text = []
                run += 1
                continue
            if run:
                parts.append(f"[machine:{run}]")
                run = 0
            if t is not None:
                text.append(t)
        if text:
            parts.append(self.tok.decode(text, skip_special_tokens=False))
        return "".join(parts)


def oracle(ctx: Ctx, cmap: dict, fmt_: str):
    """Hook for the oracle conditions: feeds every token's class to bridge.InlineScan (the
    circuit's capture rule) and, when it fires at "=", returns " R " as text tokens (nothing for
    A - B < 0), so base Qwen sees what the circuit would print, without chip tokens."""
    from chipformer.q35 import bridge as B

    states: dict = {}
    digit = {str(d): ctx.tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)}
    comma = ctx.tok(",", add_special_tokens=False).input_ids[0]

    def run(out: list[int]) -> list[int]:
        st = states.get(id(out))
        if st is None or st[2] is not out:
            st = states[id(out)] = [B.InlineScan(), 0, out]
        fired = None
        for t in out[st[1]:]:
            fired = st[0].feed(cmap.get(t, B.OTHER))
        st[1] = len(out)
        if not fired:
            return []
        op, a, b = fired
        v = PY_TOOLS[op](int(a), int(b))
        if v < 0:
            return []
        if fmt_ == "{close}":  # close the math or the sentence after the number
            text = ctx.tok.decode(out, skip_special_tokens=True)
            if text.count("$$") % 2:
                r = f" {v}$$"
            elif text.replace("$$", "").count("$") % 2:
                r = f" {v}$"
            else:
                r = f" {v}."
        else:
            r = fmt_.format(r=v, rc=f"{v:,}")
        return [ctx.space if c == " " else comma if c == "," else digit[c] if c in digit
                else ctx.tok(c, add_special_tokens=False).input_ids[0] for c in r]

    run.every = True
    return run


def executor(style: str, ctx: Ctx):
    """Python tool executor for the pytool conditions: called at each </tool_call>, returns the
    tokens to inject."""
    def run(out: list[int]) -> list[int]:
        calls = parse_calls(ctx.show(out))
        if not calls:
            return []
        c = calls[-1]
        v = true_result(c["op"], c["args"])
        res = fmt(v) if v is not None else "error: invalid call"
        if style == "inline":  # what the circuit prints: \n<tool_response>\n R \n</tool_response>\n
            return ctx.ids("\n<tool_response>\n" + res + "\n</tool_response>\n")
        return ctx.ids("<|im_end|>\n<|im_start|>user\n<tool_response>\n" + res
                       + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n"
                       "</think>\n\n")
    return run


class _Row:
    """One problem in flight: its tokens, the tokens still to force (a Python tool response), and
    the count of Qwen-written tokens (machine tokens, the circuit's print and injected tokens are
    not counted against the budget)."""

    def __init__(self, i: int, n_prompt: int):
        self.i, self.n_prompt, self.t0 = i, n_prompt, time.time()
        self.out: list[int] = []
        self.forced: deque = deque()
        self.n_qwen = self.n_inj = 0
        self.printing = False
        self.stop = None
        self.extra, self.think_forced = 0, False
        self.inj_at: list[int] = []  # where each injection starts in out

    def push(self, t: int, forced: bool, ctx, budget: int, max_len: int, hook) -> bool:
        """Record token t; returns True when the row is finished."""
        self.out.append(t)
        if self.n_prompt + len(self.out) >= max_len - 1:
            self.stop = "context"
            return True
        if forced:
            self.n_inj += 1
            return False
        if t in ctx.stop:
            self.stop = "eos"
            return True
        # the circuit prints "\n<tool_response>\n R \n</tool_response>\n" after a tool call and
        # " R" after an inline "A op B =" (or nothing for A - B < 0); those are not Qwen's tokens
        if ctx.is_chip(t):
            self.printing = True
        elif self.printing is True:  # the first text token after a machine run
            self.printing = ("tool" if t == ctx.nl else "inline" if t == ctx.space else False)
            self.n_qwen += not self.printing
        elif self.printing == "tool":
            if t == ctx.tr_close:
                self.printing = "end"
        elif self.printing == "end":
            self.printing = False  # the newline after </tool_response> is the circuit's
        elif self.printing == "inline" and t in ctx.digits:
            pass
        elif self.printing == "inline" and (t == ctx.space or t in ctx.closers):
            self.printing = False  # v9 closes the number with a space, v11 with $ / $$ / .
        else:
            self.printing = False
            self.n_qwen += 1
        if self.n_qwen >= budget + self.extra:
            if ctx.think_force and not self.think_forced and ctx.think_close not in self.out:
                # thinking hit the budget: close it (Qwen's recommended phrase) and let it answer
                self.forced.extend(ctx.think_force)
                self.extra += THINK_EXTRA
                self.think_forced = True
                return False
            self.stop = "budget"
            return True
        if hook is not None and (getattr(hook, "every", False) or t == ctx.tc_close):
            inj = hook(self.out)
            if inj:
                self.inj_at.append(len(self.out))
            self.forced.extend(inj)
        return False

    def result(self) -> dict:
        return {"out": self.out, "n_qwen": self.n_qwen, "n_injected": self.n_inj,
                "stop": self.stop, "think_forced": self.think_forced, "inj_at": self.inj_at,
                "seconds": time.time() - self.t0}


def decode_all(bd, ctx, prompts: list[list[int]], budget: int, hook, on_done) -> None:
    """Greedy decoding of every prompt through ``batch.BatchDecoder``'s rows (per-row positions,
    batch-1 prefill, continuous refill), with per-row stop rules and token forcing: after a
    </tool_call>, a ``hook`` (the Python executor) may queue tokens that are fed instead of the
    model's predictions. Calls ``on_done(i, result)`` as each problem finishes."""
    import torch

    queue = deque(range(len(prompts)))
    n = bd.batch
    bd.prefill(prompts[0])  # the batched cache copies a prefilled cache's shapes
    level = bd._prepare(n) if hasattr(bd, "_prepare") else bd._level(n)
    rows, tok, pos, step = level[:4]
    chip = level[4] if len(level) > 4 else None  # step for all-chip-token inputs (skip_qwen)
    # fp: RoPE positions frozen on chip tokens (a separate per-row buffer in BatchDecoder)
    rpos = bd.rpos[n] if getattr(bd, "freeze", False) else None
    first = bd.prefill(prompts[queue[0]])
    last = [0] * n  # each row's input token
    live: list[_Row | None] = [None] * n

    def admit(b, first=None):
        while queue:
            i = queue.popleft()
            t = bd.prefill(prompts[i]) if first is None else first
            first = None
            r = _Row(i, len(prompts[i]))
            if r.push(t, False, ctx, budget, bd.max_len, hook):
                on_done(i, r.result())
                continue
            rows.load_row(b, bd.small, len(prompts[i]))
            bd.mark_prompt(n, b, prompts[i])  # _tc: its chip slots
            tok[b, 0] = t
            pos[b] = len(prompts[i])
            if rpos is not None:
                rpos[b] = sum(not ctx.is_chip(x) for x in prompts[i])
            live[b], last[b] = r, t
            return
        live[b] = None
        tok[b, 0] = 0  # idle row: decodes garbage at slot 0
        pos[b] = 0

    admit(0, first)
    for b in range(1, n):
        admit(b)
    with torch.no_grad():
        while any(r is not None for r in live):
            busy = [b for b in range(n) if live[b] is not None]
            use_chip = chip is not None and all(ctx.is_chip(last[b]) for b in busy)
            pred = (chip if use_chip else step)(tok, pos).tolist()
            pos.add_(1)
            if rpos is not None:  # the input was text: the next RoPE position moves on
                rpos.add_(bd._is_text(tok[:, 0]))
            nt, done = [], []
            for b, p in enumerate(pred):
                r = live[b]
                if r is None:
                    nt.append(0)
                    continue
                forced = bool(r.forced)
                t = r.forced.popleft() if forced else p
                nt.append(t)
                if r.push(t, forced, ctx, budget, bd.max_len, hook):
                    done.append(b)
            tok.copy_(torch.tensor(nt, device=tok.device)[:, None])
            last = nt
            idle = [b for b in range(n) if live[b] is None]
            if idle:
                pos[idle] = 0
                if rpos is not None:
                    rpos[idle] = 0
            for b in done:
                on_done(live[b].i, live[b].result())
                admit(b)


# ---------------------------------------------------------------------------------- scoring

def _key(op, a, b):
    return (a, b) if op in ("sub", "div") else frozenset((a, b))


USER_TURN = re.compile(r"<\|im_start\|>\s*user(?!\s*<tool_response>)")


INLINE_RE = re.compile(r"(\d[\d,]*)[\s$(]*(\+|-|−|×|\*|\\times|\\cdot|/|÷|\\div)[\s$(]*"
                       r"(\d[\d,]*)[\s$(]*=\s*$")
INLINE_OPS = {"+": "add", "-": "sub", "−": "sub", "×": "mul", "*": "mul",
              "\\times": "mul", "\\cdot": "mul", "/": "div", "÷": "div", "\\div": "div"}


def inline_fires(text: str, pred) -> list[dict]:
    """Every machine run outside a tool call: the "A op B =" before it (as written), the digits
    right after it, the true value, and whether the result reappears later / is the answer.

    The circuit prints exactly the digits of the result (nothing for A - B < 0) and hands back;
    Qwen may append digits of its own (v8 prints no closing space). So the machine's part is the
    first len(true) digits: ``exact`` = the digits start with the true value, ``appended`` = the
    digits Qwen added after it (for A - B < 0, everything shown is Qwen's)."""
    out = []
    # v14 quotients: digits, then "." and decimals, then "..." if inexact
    for m in re.finditer(r"\[machine:\d+\]( ?-?)((?:\d|,(?=\d))*(?:\.\d+(?:\.\.\.)?)?)", text):
        before = MARK.sub("", text[:m.start()])
        if before.rstrip().endswith("</tool_call>") or m.end() == len(text):
            continue  # a tool call, or a run cut off by the budget / context before its print
        e = INLINE_RE.search(before[-200:])
        # v13+ prints A - B < 0 as " -digits" (before: nothing)
        shown = ("-" if "-" in m[1] else "") + m[2].replace(",", "")
        machine = shown
        f = {"expr": e[0].strip() if e else before[-40:], "printed": shown[:60]}
        if e:
            a, b = int(e[1].replace(",", "")), int(e[3].replace(",", ""))
            op = INLINE_OPS[e[2]]
            if op == "div" and not b:
                continue
            v = PY_TOOLS[op](a, b)
            t = (div_text(a, b) if op == "div" else
                 str(v) if v >= 0 or shown.startswith("-") else "")
            exact = shown.startswith(t)
            machine = t if exact else shown
            f |= {"op": op, "a": a, "b": b, "true": t if op == "div" else str(v), "exact": exact,
                  "appended": shown[len(t):][:60] if exact else ""}
        later = MARK.sub("", text[m.end():]).replace(",", "")
        # a number Qwen writes right after the print (v9 " R ", v11 " R$" / " R$$" / " R.")
        ech = re.match(r"(?:\$\$|\$|\.)?[ \t]*\$?[ \t]*(\d+)", later)
        f["echo"] = (None if not ech or not machine or f.get("appended") else
                     "same" if ech[1] == machine else "diff")
        f["reused"] = bool(machine) and machine in later
        f["is_answer"] = (bool(machine) and len(machine) < 60 and pred is not None
                          and Fraction(Decimal(machine.removesuffix("..."))) == pred)
        out.append(f)
    return out


def judge(p: dict, text: str, think: bool) -> dict:
    """Score a transcript: the answer, its correctness, and every tool call. A user turn that
    Qwen writes itself (after a tool response, instead of an assistant turn) ends the answer:
    a chat runtime would hand the turn back there."""
    full = text
    m = USER_TURN.search(text)
    lenient = None
    if m:  # lenient: a boxed answer Qwen wrote inside the user turn it made up
        rest = text[m.start():]
        boxes = [_balanced(rest, b.end()) for b in re.finditer(r"\\boxed\s*\{", rest)]
        v = to_number(boxes[-1]) if boxes else None
        lenient = v if isinstance(v, Fraction) else (Fraction(Decimal(v[0])) if v else None)
        text = text[:m.start()]
    pred, src = extract_answer(text, think)
    fake_resp = False
    if pred is None and lenient is None:
        # lenient: Qwen's last number, counting <tool_response> blocks it wrote itself (with
        # <|im_start|> banned it sometimes answers inside one instead of continuing)
        own = re.sub(r"<tool_call>.*?(</tool_call>|$)", " ", MARK.sub(" ", text), flags=re.DOTALL)
        own = re.sub(r"<\|[^|]*\|>", " ", own)
        nums = NUM.findall(re.sub(r"(?<=\d),(?=\d{3}\b)", "", own))
        if nums:
            lenient, fake_resp = Fraction(Decimal(nums[-1])), True
    gold_steps = {Fraction(s) for s in p.get("steps", [])} | {Fraction(Decimal(p["gold"]))}
    want_calls = [(o, _key(o, Fraction(x), Fraction(y))) for o, x, y in p.get("calls", [])]
    rec_calls = []
    for c in parse_calls(text):
        v = true_result(c["op"], c["args"])
        ia, ib = int_arg(c["args"].get("a")), int_arg(c["args"].get("b"))
        valid = (c["op"] in tool_names() and ia is not None and ib is not None
                 and (c["op"] != "sub" or ia >= ib))
        resp = (c["response"] or "").strip()
        rv = num_arg(resp) if resp else None
        e = {"op": c["op"], "a": c["args"].get("a"), "b": c["args"].get("b"), "valid": valid,
             "response": c["response"], "true": fmt(v) if v is not None else None,
             "exact": (resp == div_text(ia, ib) if c["op"] == "div" and valid and ib
                       else rv is not None and v is not None and rv == v)}
        if "steps" in p and v is not None:
            e["in_gold_steps"] = v in gold_steps
        if want_calls and v is not None:
            key = _key(c["op"], num_arg(c["args"]["a"]), num_arg(c["args"]["b"]))
            e["planned"] = (c["op"], key) in want_calls
        rec_calls.append(e)
    fires = inline_fires(full, pred)
    return {"pred": fmt(pred) if pred is not None else None, "source": src, "inline": fires,
            "correct": is_correct(pred, p["gold"]), "user_turn": bool(m), "fake_resp": fake_resp,
            "correct_lenient": is_correct(pred, p["gold"]) or is_correct(lenient, p["gold"]),
            "calls": rec_calls}


def score(p: dict, cond: str, ctx: Ctx, r: dict, seconds: float, think: bool) -> dict:
    out = r["out"]
    if parse_cond(cond)[0] in ("oracle", "oraclec") and r.get("inj_at"):
        # mark each injection like a machine run, so inline_fires reads it the same way
        cuts = [0, *r["inj_at"], len(out)]
        text = "[machine:0]".join(ctx.show(out[a:b]) for a, b in itertools.pairwise(cuts))
    else:
        text = ctx.show(out)
    return {"id": p["id"], "cond": cond, "question": p["question"], "gold": p["gold"],
            **judge(p, text, think), "stop": r["stop"],
            "n_qwen": r["n_qwen"], "n_chip": sum(1 for t in r["out"] if ctx.is_chip(t)),
            "n_out": len(r["out"]), "n_injected": r.get("n_injected", 0),
            "think_forced": r.get("think_forced", False),
            "seconds": round(seconds, 2), "text": text,
            "ids": r["out"] if ctx.chip else None}  # generated ids (merged: for circuit replays)


def rescore(args) -> None:
    """Re-judge every stored transcript with the current scoring code (no model needed)."""
    for f in sorted(Path(args.out).glob("*.jsonl")):
        data, cond = f.name.split(".")[:2]
        recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line]
        idx = sorted({int(r["id"].rsplit("-", 1)[1]) for r in recs})
        probs = {p["id"]: p for p in load_data(data, 0, idx=idx)}
        changed = 0
        for r in recs:
            new = judge(probs[r["id"]], r["text"], "think" in parse_cond(cond)[1])
            changed += new["correct"] != r["correct"]
            r.update(new)
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                     encoding="utf-8")
        print(f"{f.name}: {len(recs)} records, {changed} changed verdicts")


# ---------------------------------------------------------------------------------- running

def default_model(cond: str) -> str:
    if cond.startswith("merged"):
        return str(ROOT / "dist" / "q4-v5")
    snaps = sorted(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*"))
    return str(snaps[-1])


def build_prompt(tok, question: str, cond: str, bios: list[int]) -> list[int]:
    kind, flags = parse_cond(cond)
    msgs = [{"role": "user", "content": question + SUFFIX}]
    if flags & {"nudge", "nudge2", "nudge3", "mix"}:  # tools: use them; others: inline
        if "nudge3" in flags:  # with division
            nudge = NUDGE_MIX3 if "mix" in flags else NUDGE3
        elif "mix" in flags:
            nudge = NUDGE_MIX
        elif "nudge2" in flags:  # the inline nudge, plus: the filled-in number is final
            nudge = NUDGE2
        elif kind in TOOL_KINDS:
            nudge = NUDGE.format(names=", ".join(tool_names()))
        else:
            from chipformer.q35 import demo

            nudge = demo.NUDGE
        msgs.insert(0, {"role": "system", "content": nudge})
    ids = tok.apply_chat_template(msgs, tools=tools() if kind in TOOL_KINDS else None,
                                  add_generation_prompt=True, return_tensors="pt",
                                  enable_thinking="think" in flags)["input_ids"][0].tolist()
    return (bios if kind in BIOS_KINDS else []) + ids


class BannedHead(torch.nn.Module):
    """The output head with -inf added at banned token ids (generation_config bad_words_ids of
    length 1), so the argmax inside the compiled step never picks them."""

    def __init__(self, head, banned: list[int], n: int):
        super().__init__()
        self.inner, self.weight = head, head.weight
        mask = torch.zeros(n, dtype=head.weight.dtype, device=head.weight.device)
        mask[banned] = float("-inf")
        self.register_buffer("mask", mask)

    def forward(self, h):
        return self.inner(h) + self.mask


SELFTEST = [("mul", 4, 4), ("mul", 12345, 678), ("add", 987654321, 123456789),
            ("sub", 1000000, 1), ("sub", 5000, 4321), ("div", 1000000, 7), ("div", 8352, 48)]


def selftest(bd, ctx, tok, bios) -> None:
    """Before a merged run: Qwen calls each tool on known operands and every call it makes
    must print the exact result. Catches a BIOS that does not match the circuit (the routine
    entry points are compiled into the weights) or inexact kernels."""
    sym = {"add": "+", "mul": "*", "sub": "-", "div": "/"}
    cases = [c for c in SELFTEST if c[0] in tool_names()]
    prompts = [build_prompt(tok, f"Use the {op} tool to compute {a} {sym[op]} {b}.", "merged", bios)
               for op, a, b in cases]
    got = {}
    decode_all(bd, ctx, prompts, 400, None, lambda i, r: got.__setitem__(i, r))
    bad, n = [], 0
    for i, (op, a, b) in enumerate(cases):
        for c in parse_calls(ctx.show(got[i]["out"])):
            v = true_result(c["op"], c["args"])
            if v is None:
                continue
            n += 1
            want = (div_text(int_arg(c["args"]["a"]), int_arg(c["args"]["b"]))
                    if c["op"] == "div" else fmt(v))  # div truncates: 10 decimals, then "..."
            if (c["response"] or "").strip() != want:
                bad.append((c["op"], c["args"], c["response"], want))
    print(f"selftest: {n} calls, {len(bad)} wrong {bad}", flush=True)
    if bad or n == 0:
        raise SystemExit("selftest failed: the BIOS does not match this checkpoint's circuit?")


def run(args) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    cond = args.cond + (f"-{args.tag}" if args.tag else "")
    kind, flags = parse_cond(cond)
    think = "think" in flags
    probs = load_data(args.data, args.n, args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.data}.{cond}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    todo = [p for p in probs if p["id"] not in done]
    if args.shard:
        k, m = map(int, args.shard.split("/"))
        todo = todo[k::m]
    print(f"{path}: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    mpath = args.model or default_model(cond)
    tok = AutoTokenizer.from_pretrained(mpath)
    model = AutoModelForImageTextToText.from_pretrained(mpath, dtype=torch.bfloat16)
    model = model.cuda().eval()
    chip, bios, bios_id = None, [], None
    meta_path = Path(mpath) / "chipformer.json"
    if meta_path.exists():
        from chipformer.q35.lib import bcd
        from chipformer.ref.emu import Chip8
        from chipformer.trace import boot_tokens

        meta = json.loads(meta_path.read_text())
        chip = (meta["chip_token_offset"], meta["chip_vocab"])
        if kind in BIOS_KINDS:
            import hashlib

            bios = [t + chip[0] for t in boot_tokens(Chip8(bcd.ROM, bcd.QUIRKS).cells)]
            bios_id = f"{hashlib.sha1(bcd.ROM).hexdigest()[:10]} ({len(bcd.ROM)} B, OPS {bcd.OPS})"
            print(f"BIOS {bios_id}, boot record {len(bios)} tokens", flush=True)
    ctx = Ctx(tok, chip)
    budget = args.budget or BUDGET[think]
    if think:
        ctx.think_force = ctx.ids(THINK_FORCE)
    hook = {"pytool_inline": executor("inline", ctx),
            "pytool": executor("template", ctx)}.get(kind)
    if kind in ("oracle", "oraclec"):
        from chipformer.q35.spikes_replay import class_map

        cmeta = json.loads((Path(args.class_model) / "chipformer.json").read_text())
        fmt_ = args.oracle_fmt or (" {rc} " if kind == "oraclec" else " {r} ")
        print(f"oracle format {fmt_!r}", flush=True)
        hook = oracle(ctx, class_map(tok, cmeta), fmt_)
    banned = []
    if "nb" in flags or kind == "merged_inline":
        gc = Path(mpath) / "generation_config.json"
        bad = json.loads(gc.read_text()).get("bad_words_ids", []) if gc.exists() else []
        banned = sorted({w[0] for w in bad if len(w) == 1} | {IM_START}
                        | ({ENDOFTEXT} if "ne" in flags else set()))
        model.lm_head = BannedHead(model.lm_head, banned, model.lm_head.weight.shape[0])
        print(f"banned token ids {banned}", flush=True)

    def emit(rec):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps({k: rec[k] for k in ("id", "gold", "pred", "correct", "stop", "n_qwen",
                                               "n_chip", "seconds")}
                         | {"calls": [(c["op"], c["a"], c["b"], c["response"], c["exact"])
                                      for c in rec["calls"]]}), flush=True)

    prompts = [build_prompt(tok, p["question"], cond, bios) for p in todo]
    import torch._inductor.config

    from chipformer.q35.batch import BatchDecoder

    torch._inductor.config.shape_padding = False  # its padded weight copies don't fit 24 GB

    max_len = args.max_len or (max(map(len, prompts)) + budget + (3072 if chip else 1024 if think else 512))
    max_len = -(-max_len // 1024) * 1024
    if args.fast and chip:  # exact: chip layers read only their nonzero blocks, and steps on
        from chipformer.q35 import fast  # chip-token inputs skip Qwen's (no-op) layers

        print("compact", fast.compact(model, qwen=False), flush=True)
    bd = BatchDecoder(model, batch=min(args.batch, len(todo)), max_len=max_len,
                      eos=sorted(ctx.stop), sizes=[min(args.batch, len(todo))],
                      compile=not args.no_compile,
                      **({"skip_qwen": True} if args.fast and chip else {}),
                      **({"freeze_chip_positions": True} if "fp" in flags else {}),
                      **({"transparent_chip": True} if "tc" in flags else {}))
    if kind == "merged" and not args.no_selftest:  # (tool mode)
        selftest(bd, ctx, tok, bios)
    t0 = time.time()
    print(f"batch {bd.batch}, max_len {max_len}, budget {budget}", flush=True)

    def done(i, r):
        emit(score(todo[i], cond, ctx, r, r["seconds"], think)
             | {"batch": bd.batch, "model": mpath, "bios": bios_id, "banned": banned})

    decode_all(bd, ctx, prompts, budget, hook, done)
    wall = time.time() - t0
    with (out_dir / "runs.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"data": args.data, "cond": cond, "n": len(todo), "batch": bd.batch,
                            "wall_s": round(wall, 1), "s_per_problem": round(wall / len(todo), 2),
                            "model": mpath, "bios": bios_id,
                            "gpu": torch.cuda.get_device_name()}) + "\n")
    print(f"done {len(todo)} in {wall:.0f}s ({wall / len(todo):.1f} s/problem)", flush=True)


# ---------------------------------------------------------------------------------- summary

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def load_results(out: Path) -> dict:
    res = {}
    for f in sorted(out.glob("*.jsonl")):
        data, cond = f.name.split(".")[:2]
        recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line]
        res[(data, cond)] = {r["id"]: r for r in recs}
    return res


def failure(r: dict) -> str:
    if r["correct"]:
        return "ok"
    if r["stop"] != "eos":
        return "cutoff"
    if any(c["valid"] and not c["exact"] for c in r["calls"]):
        return "machine_wrong"
    if any(not c["valid"] for c in r["calls"]):
        return "invalid_call"
    if r.get("user_turn"):
        return "user_turn"
    if r.get("fake_resp") and r.get("correct_lenient"):
        return "answer_in_own_tool_response"
    if r["pred"] is None:
        return "no_answer"
    return "qwen_wrong_tools" if r["calls"] else "qwen_wrong"


PAIRS = [("merged_inline_nudge-v11", "base_nudge"), ("merged_inline_nudge_fp-v11", "base_nudge"),
         ("merged_inline_nudge_fp-v11", "merged_inline_nudge-v11"), ("merged_inline-v11", "base"),
         ("merged_inline_nudge-v11", "oracle_nudge-close"),
         ("merged_inline_fp-v11", "merged_inline-v11"), ("merged_inline_fp-v11", "base"),
         ("oracle_nudge", "base_nudge"), ("oraclec_nudge", "base_nudge"),
         ("oracle_nudge-close", "base_nudge"), ("oracle_nudge2", "base_nudge"),
         ("oracle_nudge2-close", "base_nudge"),
         ("merged_inline_nudge", "oracle_nudge"), ("merged_inline_fp", "merged_inline"), ("merged_inline_nudge_fp", "merged_inline_nudge"),
         ("merged_inline_nudge_fp", "base_nudge"), ("merged_inline", "base"), ("merged_nb-v9", "base"), ("merged_inline", "merged_nb-v9"),
         ("merged_inline-v8", "base"), ("merged_inline_nudge-v8", "base_nudge"), ("merged_inline_nudge", "base"),
         ("merged_inline_nudge", "base_nudge"), ("merged", "base"), ("merged_nb", "base"), ("merged_nb", "merged"),
         ("merged_nudge", "base"), ("merged_nb_nudge", "base"), ("pytool_nudge", "base"),
         ("merged", "pytool_inline"), ("merged", "pytool"),
         ("pytool", "base"), ("pytool_inline", "base"), ("base_think", "base"),
         ("merged", "base_think"), ("merged_plain", "base")]
# pairs whose token streams should match (same Qwen text, merged vs base weights)
SAME_TEXT = [("merged", "pytool_inline"), ("merged_plain", "base")]


def _plain(r):
    return MARK.sub("", r["text"])


def _wall(out: Path) -> dict:
    """Wall seconds per problem from runs.log (all rows of a batch share the GPU)."""
    tot = {}
    log = out / "runs.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            x = json.loads(line)
            w, n = tot.get((x["data"], x["cond"]), (0.0, 0))
            tot[(x["data"], x["cond"])] = (w + x["wall_s"], n + x["n"])
    return {k: w / n for k, (w, n) in tot.items() if n}


NL = chr(10)


def _clean(gold: str) -> bool:
    g = Fraction(Decimal(gold))
    return g >= 0 and g.denominator == 1


def _looping(fires: list[dict], n: int = 3) -> bool:
    ex = [f["expr"] for f in fires]
    return any(len(set(ex[i:i + n])) == 1 for i in range(len(ex) - n + 1))


def summary(args) -> None:
    out = Path(args.out)
    res = load_results(out)
    wall = _wall(out)
    lines = [("Accuracy with Wilson 95% CI. tool-call rate = problems with >= 1 call. valid = "
             "the tool exists and both arguments are non-negative integers. machine exact = "
             "printed result == true result (valid calls). args ok = the call is a planned step "
             "(arith) or its result is one of the reference solution's intermediate values "
             "(gsm8k). s/prob = wall time per problem at the run's batch size.")]
    datas = sorted({d for d, _ in res})
    for data in datas:
        conds = [c for c in CONDS if (data, c) in res] + sorted(
            c for d, c in res if d == data and c not in CONDS)
        lines.append(f"\n## {data}\n")
        lines.append("| cond | N | acc | 95% CI | lenient | tool-call rate | calls/prob | "
                     "valid calls | machine exact | args ok | cutoffs | Qwen tok | chip tok | "
                     "s/prob |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in conds:
            rs = list(res[(data, c)].values())
            n, k = len(rs), sum(r["correct"] for r in rs)
            lo, hi = wilson(k, n)
            calls = [x for r in rs for x in r["calls"]]
            valid = [x for x in calls if x["valid"]]
            ex = [x["exact"] for x in valid if x["response"] is not None]
            argok = [x.get("planned", x.get("in_gold_steps")) for x in calls]
            argok = [a for a in argok if a is not None]

            def pct(xs):
                return f"{100 * sum(xs) / len(xs):.0f}% ({sum(xs)}/{len(xs)})" if xs else "-"

            w = wall.get((data, c))
            lines.append(
                f"| {c} | {n} | {100 * k / n:.1f}% | {100 * lo:.0f}-{100 * hi:.0f} | "
                f"{100 * sum(r.get('correct_lenient', r['correct']) for r in rs) / n:.1f}% | "
                f"{pct([bool(r['calls']) for r in rs])} | {len(calls) / n:.2f} | "
                f"{pct([x['valid'] for x in calls])} | {pct(ex)} | {pct(argok)} | "
                f"{sum(r['stop'] != 'eos' for r in rs)} | "
                f"{sum(r['n_qwen'] for r in rs) / n:.0f} | {sum(r['n_chip'] for r in rs) / n:.0f} | "
                f"{f'{w:.1f}' if w else '-'} |")
        inl = [c for c in conds if any(r.get("inline") for r in res[(data, c)].values())]
        if inl:
            lines.append("\nInline arithmetic (machine runs outside tool calls): exact = printed "
                         "== value of the \"A op B =\" as written (empty for A - B < 0); "
                         "unparsed = no \"A op B =\" found before the run.\n")
            lines.append("| cond | problems with a fire | fires/prob | machine exact | "
                         "Qwen digits glued on (after R / A<B fill) | Qwen re-writes a number "
                         "(same / different) | unparsed | result reused later | "
                         "result is the answer | answer from a fire | "
                         "looping (3+ same fires in a row) |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for c in inl:
                rs = list(res[(data, c)].values())
                fs = [f for r in rs for f in r.get("inline", [])]
                ex = [f["exact"] for f in fs if "exact" in f]
                lines.append(
                    f"| {c} | {sum(bool(r.get('inline')) for r in rs)}/{len(rs)} | "
                    f"{len(fs) / len(rs):.2f} | {sum(ex)}/{len(ex)} | "
                    f"{sum(bool(f.get('appended')) and f.get('true', '0')[0] != '-' for f in fs)} / "
                    f"{sum(bool(f.get('appended')) and f.get('true', '0')[0] == '-' for f in fs)} | "
                    f"{sum(f.get('echo') == 'same' for f in fs)} / "
                    f"{sum(f.get('echo') == 'diff' for f in fs)} | "
                    f"{sum('exact' not in f for f in fs)} | {sum(f['reused'] for f in fs)}/{len(fs)} | "
                    f"{sum(f['is_answer'] for f in fs)}/{len(fs)} | "
                    f"{sum(any(f['is_answer'] for f in r.get('inline', [])) for r in rs)}/{len(rs)} | "
                    f"{sum(_looping(r.get('inline', [])) for r in rs)} |")
        lines.append("\nFailures (wrong answers by cause):\n")
        for c in conds:
            cnt = _count(failure(r) for r in res[(data, c)].values())
            lines.append(f"- {c}: " + (", ".join(f"{k} {v}" for k, v in sorted(cnt.items())
                                                  if k != "ok") or "none"))
        if data == "gsmhard":
            lines.append(NL + "GSM-Hard subsets: many targets are negative or fractional (label "
                         "artifacts of swapping in large numbers); 'clean' = non-negative integer "
                         "target." + NL)
            lines.append("| cond | clean acc | clean 95% CI | other acc |")
            lines.append("|---|---|---|---|")
            for c in conds:
                rs = list(res[(data, c)].values())
                cl = [r["correct"] for r in rs if _clean(r["gold"])]
                ot = [r["correct"] for r in rs if not _clean(r["gold"])]
                lo, hi = wilson(sum(cl), len(cl))
                lines.append(f"| {c} | {sum(cl)}/{len(cl)} = {100 * sum(cl) / max(1, len(cl)):.1f}% "
                             f"| {100 * lo:.0f}-{100 * hi:.0f} | {sum(ot)}/{len(ot)} |")
        if data == "arith":
            lines.append("\n| cond | " + " | ".join(ARITH_KINDS) + " |")
            lines.append("|---|" + "---|" * len(ARITH_KINDS))
            for c in conds:
                accs = []
                for j, kind in enumerate(ARITH_KINDS):
                    rs = [r for r in res[(data, c)].values()
                          if int(r["id"].split("-")[1]) % len(ARITH_KINDS) == j]
                    accs.append(f"{sum(r['correct'] for r in rs)}/{len(rs)}")
                lines.append(f"| {c} | " + " | ".join(accs) + " |")
        rows = []
        for a, b in PAIRS:
            if (data, a) not in res or (data, b) not in res:
                continue
            ids = sorted(set(res[(data, a)]) & set(res[(data, b)]))
            if not ids:
                continue
            ra, rb = res[(data, a)], res[(data, b)]
            xa = [ra[i]["correct"] for i in ids]
            xb = [rb[i]["correct"] for i in ids]
            n = len(ids)
            bo = sum(x and not y for x, y in zip(xa, xb))
            co = sum(y and not x for x, y in zip(xa, xb))
            d = (bo - co) / n
            se = math.sqrt(max(0.0, (bo + co) / n - d * d) / n)
            same = "-"
            if (a, b) in SAME_TEXT:
                same = f"{sum(_plain(ra[i]) == _plain(rb[i]) for i in ids)}/{n}"
            rows.append(f"| {a} vs {b} | {n} | {100 * sum(xa) / n:.1f}% | "
                        f"{100 * sum(xb) / n:.1f}% | {100 * d:+.1f} | "
                        f"{100 * (d - 1.96 * se):+.1f}..{100 * (d + 1.96 * se):+.1f} | {bo} | {co} | "
                        f"{mcnemar(bo, co):.3g} | {same} |")
        if rows:
            lines.append("\n| paired (A vs B) | N | A | B | A-B (pts) | 95% CI | A only | B only | "
                         "McNemar p | same text |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            lines += rows
    text = "\n".join(lines)
    print(text)
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")


def _count(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("cond", type=cond_arg, help=f"one of {', '.join(CONDS)} (or KIND_FLAGS)")
    r.add_argument("data", choices=("gsm8k", "gsmhard", "arith"))
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--model")
    r.add_argument("--out", default=str(ROOT / "dist" / "evals"))
    r.add_argument("--budget", type=int, help="Qwen-written tokens (default 2048, thinking 8192)")
    r.add_argument("--max-len", type=int, help="cache length (default: prompt + budget + room)")
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--shard", help="k/m: run every m-th remaining problem from k")
    r.add_argument("--no-compile", action="store_true")
    r.add_argument("--no-selftest", action="store_true")
    r.add_argument("--fast", action="store_true", help="merged: compact chip layers + skip_qwen")
    r.add_argument("--tag", help="model tag appended to the condition name (e.g. v6)")
    r.add_argument("--class-model", default=str(ROOT / "dist" / "q4-v9"),
                   help="oracle: merged checkpoint whose text classes define the inline grammar")
    r.add_argument("--oracle-fmt", help="oracle: injected text, {r} = result, {rc} = with commas "
                   "(default ' {r} ', oraclec ' {rc} '); name the variant with --tag")
    for name in ("summary", "rescore"):
        s = sub.add_parser(name)
        s.add_argument("--out", default=str(ROOT / "dist" / "evals"))
    args = ap.parse_args()
    {"run": run, "summary": summary, "rescore": rescore}[args.cmd](args)


if __name__ == "__main__":
    main()
