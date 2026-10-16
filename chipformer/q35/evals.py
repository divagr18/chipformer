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
import json
import math
import random
import re
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_VOCAB = 248320
SUFFIX = "\n\nGive the final answer as a single number in \\boxed{} at the end."
CONDS = ("base", "base_think", "merged", "merged_plain", "pytool_inline", "pytool")
TOOL_CONDS = ("merged", "pytool_inline", "pytool")
BUDGET = {False: 1536, True: 8192}  # Qwen-written tokens (thinking or not); machine tokens extra


def _div(a, b):
    if b == 0:
        raise ZeroDivisionError
    q = Fraction(a) / Fraction(b)
    return q.numerator if q.denominator == 1 else q


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


def load_data(name: str, n: int, seed: int = 0) -> list[dict]:
    if name == "arith":
        return arith(n, seed)
    from datasets import load_dataset

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


def arith(n: int, seed: int = 0) -> list[dict]:
    """Seeded synthetic arithmetic with only + and *, so the tool set covers every step.

    Each item lists the calls a perfect solver makes (``calls``: [op, a, b], operands in any
    order) for checking tool-call arguments."""
    rng = random.Random(seed)

    def r(lo, hi):
        d = rng.randint(lo, hi)
        return rng.randrange(10 ** (d - 1), 10**d)

    out = []
    kinds = ["add", "mul", "add3", "muladd", "mulmul", "sumprod"]
    for k in range(n):
        kind = kinds[k % len(kinds)]
        if kind == "add":
            a, b = r(6, 12), r(5, 12)
            q, gold, calls = f"{a} + {b}", a + b, [("add", a, b)]
        elif kind == "mul":
            a, b = r(4, 6), r(3, 5)
            q, gold, calls = f"{a} * {b}", a * b, [("mul", a, b)]
        elif kind == "add3":
            a, b, c = r(5, 9), r(5, 9), r(4, 9)
            q, gold, calls = f"{a} + {b} + {c}", a + b + c, [("add", a, b), ("add", a + b, c)]
        elif kind == "muladd":
            a, b, c = r(3, 5), r(2, 4), r(4, 8)
            q, gold, calls = f"{a} * {b} + {c}", a * b + c, [("mul", a, b), ("add", a * b, c)]
        elif kind == "mulmul":
            a, b, c = r(2, 4), r(2, 3), r(2, 3)
            q, gold, calls = f"{a} * {b} * {c}", a * b * c, [("mul", a, b), ("mul", a * b, c)]
        else:
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

class Decoder:
    """Greedy batch-1 decoding with a static cache and a CUDA-graph step (like fast.Runner, but
    over the whole vocabulary). ``prefill`` takes any number of tokens at the current position."""

    def __init__(self, model, max_len: int, vocab: int, compile: bool = True):
        import torch
        from transformers import StaticCache

        self.torch = torch
        cfg = getattr(model.config, "text_config", model.config)
        self.cache = StaticCache(config=cfg, max_cache_len=max_len)
        self.max_len = max_len
        inner = model.model
        self.lm = getattr(inner, "language_model", inner)
        self.head = model.lm_head.weight[:vocab]
        self.dev = self.head.device
        self.pos = 0
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.cpos = torch.zeros(1, dtype=torch.long, device=self.dev)
        self._step = (torch.compile(self._forward, mode="reduce-overhead", fullgraph=False)
                      if compile else self._forward)

    def _forward(self, tok, cpos):
        h = self.lm(input_ids=tok, past_key_values=self.cache, cache_position=cpos,
                    use_cache=True).last_hidden_state[0, -1]
        return (self.head @ h).argmax(-1)

    def prefill(self, toks: list[int]) -> int:
        torch = self.torch
        with torch.no_grad():
            nxt = None
            for s in range(0, len(toks), 4096):
                chunk = toks[s:s + 4096]
                x = torch.tensor([chunk], device=self.dev)
                cpos = torch.arange(self.pos, self.pos + len(chunk), device=self.dev)
                h = self.lm(input_ids=x, past_key_values=self.cache, cache_position=cpos,
                            use_cache=True).last_hidden_state[0, -1]
                self.pos += len(chunk)
                nxt = int((self.head @ h).argmax())
        return nxt

    def feed(self, token: int) -> int:
        with self.torch.no_grad():
            self.tok.fill_(token)
            self.cpos.fill_(self.pos)
            nxt = self._step(self.tok, self.cpos)
            self.pos += 1
            return int(nxt)

    def reset(self) -> None:
        self.cache.reset()
        self.pos = 0


class Ctx:
    """What the decode loop needs to know about token ids."""

    def __init__(self, tok, chip=None):
        self.tok = tok
        ids = tok.convert_tokens_to_ids
        self.stop = {ids("<|im_end|>"), ids("<|endoftext|>")}
        self.tc_close = ids("</tool_call>")
        self.tr_close = ids("</tool_response>")
        self.chip = chip  # (offset, n) or None

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
        return "".join(parts)


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


def decode(dec: Decoder, ctx: Ctx, prompt: list[int], budget: int, hook=None) -> dict:
    """Greedy decode until a stop token, ``budget`` Qwen-written tokens, or the cache end.
    Machine tokens (chip tokens and the circuit's print) and injected tokens are not counted."""
    dec.reset()
    nxt = dec.prefill(prompt)
    out, n_qwen, n_inj, printing, stop = [], 0, 0, False, "budget"
    while True:
        out.append(nxt)
        if nxt in ctx.stop:
            stop = "eos"
            break
        if ctx.is_chip(nxt):
            printing = True
        elif printing:
            if nxt == ctx.tr_close:
                printing = "end"
        else:
            n_qwen += 1
        if printing == "end" and nxt != ctx.tr_close:
            printing = False  # the newline after </tool_response> was the circuit's
        if n_qwen >= budget:
            break
        inject = hook(out) if hook and nxt == ctx.tc_close else []
        if len(prompt) + len(out) + len(inject) + 1 >= dec.max_len:
            stop = "context"
            break
        if inject:
            out += inject
            n_inj += len(inject)
            nxt = dec.prefill([nxt] + inject)  # </tool_call> and the response, then predict
            continue
        nxt = dec.feed(nxt)
    return {"out": out, "n_qwen": n_qwen, "n_injected": n_inj, "stop": stop}


def hf_generate_batch(model, tok, prompts: list[list[int]], max_new: int) -> list[list[int]]:
    """Plain HF batched greedy generation with left padding (text-only conditions)."""
    import torch

    pad = tok.pad_token_id if tok.pad_token_id is not None else 248044
    n = max(len(p) for p in prompts)
    x = torch.tensor([[pad] * (n - len(p)) + p for p in prompts], device="cuda")
    m = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in prompts], device="cuda")
    with torch.no_grad():
        g = model.generate(input_ids=x, attention_mask=m, max_new_tokens=max_new,
                           do_sample=False, pad_token_id=pad)
    outs = []
    stops = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    for row in g[:, n:].tolist():
        k = next((i for i, t in enumerate(row) if t in stops), None)
        outs.append(row if k is None else row[:k + 1])
    return outs


# ---------------------------------------------------------------------------------- scoring

def score(p: dict, cond: str, ctx: Ctx, r: dict, seconds: float, think: bool) -> dict:
    text = ctx.show(r["out"])
    pred, src = extract_answer(text, think)
    calls = parse_calls(text)
    gold_steps = {Fraction(s) for s in p.get("steps", [])} | {Fraction(Decimal(p["gold"]))}
    want_calls = [(o, {Fraction(x), Fraction(y)}) for o, x, y in p.get("calls", [])]
    rec_calls = []
    for c in calls:
        v = true_result(c["op"], c["args"])
        valid = (c["op"] in tool_names() and int_arg(c["args"].get("a")) is not None
                 and int_arg(c["args"].get("b")) is not None)
        resp = (c["response"] or "").strip()
        rv = num_arg(resp) if resp else None
        e = {"op": c["op"], "a": c["args"].get("a"), "b": c["args"].get("b"), "valid": valid,
             "response": c["response"], "true": fmt(v) if v is not None else None,
             "exact": (rv is not None and v is not None and rv == v)}
        if "steps" in p and v is not None:
            e["in_gold_steps"] = v in gold_steps
        if want_calls and v is not None:
            args = {num_arg(c["args"]["a"]), num_arg(c["args"]["b"])}
            e["planned"] = any(o == c["op"] and s == args for o, s in want_calls)
        rec_calls.append(e)
    n_chip = sum(1 for t in r["out"] if ctx.is_chip(t))
    return {"id": p["id"], "cond": cond, "question": p["question"], "gold": p["gold"],
            "pred": fmt(pred) if pred is not None else None, "source": src,
            "correct": is_correct(pred, p["gold"]), "stop": r["stop"],
            "n_qwen": r["n_qwen"], "n_chip": n_chip, "n_out": len(r["out"]),
            "n_injected": r.get("n_injected", 0), "calls": rec_calls,
            "seconds": round(seconds, 2), "text": text}


# ---------------------------------------------------------------------------------- running

def default_model(cond: str) -> str:
    if cond.startswith("merged"):
        return str(ROOT / "dist" / "q4-v3")
    snaps = sorted(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*"))
    return str(snaps[-1])


def build_prompt(tok, question: str, cond: str, bios: list[int]) -> list[int]:
    think = cond.endswith("_think")
    ids = tok.apply_chat_template([{"role": "user", "content": question + SUFFIX}],
                                  tools=tools() if cond in TOOL_CONDS else None,
                                  add_generation_prompt=True, return_tensors="pt",
                                  enable_thinking=think)["input_ids"][0].tolist()
    return (bios if cond == "merged" else []) + ids


def run(args) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    cond, think = args.cond, args.cond.endswith("_think")
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
    chip, bios, vocab = None, [], BASE_VOCAB
    meta_path = Path(mpath) / "chipformer.json"
    if meta_path.exists():
        from chipformer.q35.lib import bcd
        from chipformer.ref.emu import Chip8
        from chipformer.trace import boot_tokens

        meta = json.loads(meta_path.read_text())
        chip = (meta["chip_token_offset"], meta["chip_vocab"])
        vocab = chip[0] + chip[1]
        if cond == "merged":
            bios = [t + chip[0] for t in boot_tokens(Chip8(bcd.ROM, bcd.QUIRKS).cells)]
    ctx = Ctx(tok, chip)
    budget = args.budget or BUDGET[think]
    hook = {"pytool_inline": executor("inline", ctx),
            "pytool": executor("template", ctx)}.get(cond)

    def emit(rec):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps({k: rec[k] for k in ("id", "gold", "pred", "correct", "stop", "n_qwen",
                                               "n_chip", "seconds")}
                         | {"calls": [(c["op"], c["a"], c["b"], c["response"], c["exact"])
                                      for c in rec["calls"]]}), flush=True)

    prompts = [build_prompt(tok, p["question"], cond, bios) for p in todo]
    if args.batch > 1 and cond in ("base", "base_think", "merged_plain"):
        tok.padding_side = "left"
        for s in range(0, len(todo), args.batch):
            t = time.time()
            outs = hf_generate_batch(model, tok, prompts[s:s + args.batch], budget)
            dt = (time.time() - t) / len(outs)
            for p, o in zip(todo[s:s + args.batch], outs):
                stop = "eos" if o and o[-1] in ctx.stop else "budget"
                r = {"out": o, "n_qwen": len(o) - (stop == "eos"), "stop": stop}
                emit(score(p, cond, ctx, r, dt, think) | {"batched": args.batch})
        return
    if cond == "merged" and args.batch > 1:
        raise SystemExit("batched merged decoding: use chipformer.q35.batch when it exists")
    dec = Decoder(model, args.max_len, vocab, compile=not args.no_compile)
    for p, ids in zip(todo, prompts):
        t = time.time()
        r = decode(dec, ctx, ids, budget, hook)
        torch.cuda.synchronize()
        emit(score(p, cond, ctx, r, time.time() - t, think))


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
    if r["pred"] is None:
        return "no_answer"
    return "qwen_wrong_tools" if r["calls"] else "qwen_wrong"


def summary(args) -> None:
    res = load_results(Path(args.out))
    lines = []
    datas = sorted({d for d, _ in res})
    for data in datas:
        conds = [c for c in CONDS if (data, c) in res]
        lines.append(f"\n## {data}\n")
        lines.append("| cond | N | acc | 95% CI | tool-call rate | calls/prob | valid calls | "
                     "machine exact | args ok | cutoffs | Qwen tok | chip tok | s/prob |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
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

            lines.append(
                f"| {c} | {n} | {100 * k / n:.1f}% | {100 * lo:.0f}-{100 * hi:.0f} | "
                f"{pct([bool(r['calls']) for r in rs])} | {len(calls) / n:.2f} | "
                f"{pct([x['valid'] for x in calls])} | {pct(ex)} | {pct(argok)} | "
                f"{sum(r['stop'] != 'eos' for r in rs)} | "
                f"{sum(r['n_qwen'] for r in rs) / n:.0f} | {sum(r['n_chip'] for r in rs) / n:.0f} | "
                f"{sum(r['seconds'] for r in rs) / n:.1f} |")
        lines.append("\nFailures: " + "; ".join(
            f"{c}: " + ", ".join(f"{k} {v}" for k, v in sorted(_count(
                failure(r) for r in res[(data, c)].values()).items()) if k != "ok")
            for c in conds))
        pairs = [(a, b) for a in conds for b in conds if a < b]
        if pairs:
            lines.append("\n| paired (A vs B) | N | A | B | A-B | 95% CI | A only | B only | "
                         "McNemar p |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
        for a, b in pairs:
            ids = sorted(set(res[(data, a)]) & set(res[(data, b)]))
            if not ids:
                continue
            xa = [res[(data, a)][i]["correct"] for i in ids]
            xb = [res[(data, b)][i]["correct"] for i in ids]
            n = len(ids)
            bo = sum(x and not y for x, y in zip(xa, xb))
            co = sum(y and not x for x, y in zip(xa, xb))
            d = (bo - co) / n
            se = math.sqrt(max(0.0, (bo + co) / n - d * d) / n)
            lines.append(f"| {a} vs {b} | {n} | {100 * sum(xa) / n:.1f}% | "
                         f"{100 * sum(xb) / n:.1f}% | {100 * d:+.1f} | "
                         f"{100 * (d - 1.96 * se):+.1f}..{100 * (d + 1.96 * se):+.1f} | {bo} | {co} | "
                         f"{mcnemar(bo, co):.3g} |")
    text = "\n".join(lines)
    print(text)
    Path(args.out, "summary.md").write_text(text + "\n", encoding="utf-8")


def _count(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("cond", choices=CONDS)
    r.add_argument("data", choices=("gsm8k", "gsmhard", "arith"))
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--model")
    r.add_argument("--out", default=str(ROOT / "dist" / "evals"))
    r.add_argument("--budget", type=int, help="Qwen-written tokens (default 1536, thinking 8192)")
    r.add_argument("--max-len", type=int, default=32768, help="static cache length")
    r.add_argument("--batch", type=int, default=1)
    r.add_argument("--shard", help="k/m: run every m-th remaining problem from k")
    r.add_argument("--no-compile", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("--out", default=str(ROOT / "dist" / "evals"))
    args = ap.parse_args()
    (run if args.cmd == "run" else summary)(args)


if __name__ == "__main__":
    main()
