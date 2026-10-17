"""Standard arithmetic benchmarks: merged (the computer in the weights) vs base Qwen3.5-4B.

    python -m chipformer.q35.arith_bench run COND DATA [DATA ...] [--model PATH] [--batch 16]
    python -m chipformer.q35.arith_bench replay COND DATA [DATA ...]   # chip exactness (IR)
    python -m chipformer.q35.arith_bench identity --model MERGED       # BatchDecoder == generate
    python -m chipformer.q35.arith_bench summary

Datasets (problems sampled with a fixed seed, so every condition sees the same ones):

  dmx_add_or_sub_big, dmx_mul_big, dmx_mixed_longer, dmx_add_sub_multiple_longer
  dmi_add_or_sub, dmi_mul, dmi_add_sub_multiple, dmi_mixed
  dmxc_add_or_sub_big, dmxc_mul_big
      the same extrapolate test files, sampled from their clean problems only
      DeepMind Mathematics Dataset v1.0 test files (extrapolate / interpolate); the files
      alternate question / answer lines
  nupa_add, nupa_sub, nupa_mulh, nupa_mule
      NUPA_text test.json, integer add / sub / multiply_hard / multiply_easy, sampled evenly
      over digit lengths 3-20 (groups 3-6, 7-10, 11-15, 16-20). NUPA's "Directly return the
      answer as an integer ..." preamble and its "= answer" are dropped; the task sentence
      ("Add two numbers: 830 + 70") is kept verbatim
  bbh_mat2       BIG-Bench Hard multistep_arithmetic_two (the trailing " =" is dropped)
  eleu_<subset>  EleutherAI/arithmetic (the "Q: ... A:" wrapper is dropped)

Conditions, prompts, decoding and judging are evals.py's (same SUFFIX, budgets, BIOS, bans,
BatchDecoder): this file only adds the loaders, a multi-dataset runner (one model load per
condition), an exact answer check (pred == gold as rationals: no tolerance, decimals and
fractions compared exactly), the circuit replay and the summary.

``clean`` = every number in the problem is a non-negative integer, there is no division and
the answer is a non-negative integer (what the computer's add / sub / mul can do end to end).
"""

import argparse
import json
import math
import random
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

from chipformer.q35 import evals as E

_fmt = E.fmt


def _safe_fmt(v) -> str:
    """evals.fmt with enough decimal precision for huge values (runaway digit strings in a
    transcript would otherwise raise InvalidOperation and end the run)."""
    import decimal

    with decimal.localcontext() as c:
        c.prec = 100_000
        try:
            return _fmt(v)
        except (decimal.InvalidOperation, ValueError, OverflowError):
            return str(v)


E.fmt = _safe_fmt  # judge() looks fmt up in its module at call time
ROOT = Path(__file__).resolve().parents[2]
HOME = Path.home()
DATA = HOME / "data"
OUT = ROOT / "dist" / "arith"
sys.set_int_max_str_digits(0)

DM_MODULES = {"dmx_add_or_sub_big": "extrapolate/arithmetic__add_or_sub_big",
              "dmx_mul_big": "extrapolate/arithmetic__mul_big",
              "dmx_mixed_longer": "extrapolate/arithmetic__mixed_longer",
              "dmx_add_sub_multiple_longer": "extrapolate/arithmetic__add_sub_multiple_longer",
              "dmx_div_big": "extrapolate/arithmetic__div_big",
              "dmx_mul_div_multiple_longer": "extrapolate/arithmetic__mul_div_multiple_longer",
              "dmi_add_or_sub": "interpolate/arithmetic__add_or_sub",
              "dmi_mul": "interpolate/arithmetic__mul",
              "dmi_add_sub_multiple": "interpolate/arithmetic__add_sub_multiple",
              "dmi_mixed": "interpolate/arithmetic__mixed"}
NUPA_TASKS = {"nupa_add": ("add_Integer_Integer_Integer", "add", "+"),
              "nupa_sub": ("sub_Integer_Integer_Integer", "sub", "-"),
              "nupa_mulh": ("multiply_hard_Integer_Integer_Integer", "mul", "*"),
              "nupa_mule": ("multiply_easy_Integer_Integer_Integer", "mul", "*")}
NUPA_GROUPS = [(3, 6), (7, 10), (11, 15), (16, 20)]
ELEU = ["two_digit_addition", "two_digit_subtraction", "two_digit_multiplication",
        "three_digit_addition", "three_digit_subtraction", "four_digit_addition",
        "four_digit_subtraction", "five_digit_addition", "five_digit_subtraction",
        "single_digit_three_ops"]
ELEU_OPS = {"addition": ("add", "plus"), "subtraction": ("sub", "minus"),
            "multiplication": ("mul", "times")}
DMXC = ["dmxc_add_or_sub_big", "dmxc_mul_big"]
DM_DIV = ["dmx_div_big", "dmx_mul_div_multiple_longer", "dmxc_div_big"]  # v14: division
ALL = [*DM_MODULES, *DMXC, "dmxc_div_big", *NUPA_TASKS, "bbh_mat2", *(f"eleu_{s}" for s in ELEU)]
DEFAULT_N = {"dmx": 100, "dmxc": 100, "dmi": 100, "nupa": 400, "bbh": 250, "eleu": 50}
CONDS = ["base", "base_nudge", "merged_nb", "merged_inline_fp", "merged_inline_nudge_fp",
         "merged_inline_fp_tc", "merged_inline_nudge_fp_tc",
         "merged_inline", "merged_inline_nudge", "pytool"]


def family(data: str) -> str:
    return data.split("_")[0]


# ---------------------------------------------------------------------------------- datasets

def _neg_literal(q: str) -> bool:
    """A negative number literal: a '-' right before a digit whose previous non-space character
    is not a digit or ')' (DM / BBH write binary minus as 'a - b')."""
    for m in re.finditer(r"-(?=\d)", q):
        before = q[:m.start()].rstrip()
        if not before or not (before[-1].isdigit() or before[-1] == ")"):
            return True
    return False


def is_clean(q: str, gold: str) -> bool:
    g = Fraction(gold) if "/" in gold else Fraction(Decimal(gold))
    if g < 0 or g.denominator != 1:
        return False
    return not (re.search(r"\d\.\d", q) or "/" in q or _neg_literal(q))


def load_dm(data: str, n: int, seed: int) -> list[dict]:
    """dmx_/dmi_: a seeded sample of the module's test file; dmxc_: of its clean problems only
    (the add / sub / mul the computer covers, without decimals or negatives)."""
    clean_only = data.startswith("dmxc_")
    mod = DM_MODULES["dmx_" + data[5:] if clean_only else data]
    f = DATA / "dm" / "mathematics_dataset-v1.0" / (mod + ".txt")
    lines = f.read_text(encoding="utf-8").splitlines()
    pairs = [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines) - 1, 2)]
    pool = [i for i, (q, a) in enumerate(pairs) if is_clean(q, a)] if clean_only else         range(len(pairs))
    idx = sorted(random.Random(seed).sample(pool, min(n, len(pool))))
    return [{"id": f"{data}-{i}", "question": pairs[i][0], "gold": pairs[i][1],
             "group": data.split("_", 1)[1], "clean": is_clean(pairs[i][0], pairs[i][1])}
            for i in idx]


_NUPA = None


def load_nupa(data: str, n: int, seed: int) -> list[dict]:
    """n problems, spread evenly over the digit lengths 3-20 (n // 18 per length, at least 1)."""
    global _NUPA
    cache = DATA / "nupa" / "int_tasks.json"
    if _NUPA is None:
        if not cache.exists():  # keep only the integer tasks (test.json is 690 MB)
            full = json.loads((DATA / "nupa" / "test.json").read_text(encoding="utf-8"))
            keep = {k: full[k] for k, _, _ in NUPA_TASKS.values()}
            cache.write_text(json.dumps(keep), encoding="utf-8")
        _NUPA = json.loads(cache.read_text(encoding="utf-8"))
    key, op, sym = NUPA_TASKS[data]
    lens = list(range(3, 21))
    per = max(1, n // len(lens))
    out = []
    for L in lens:
        items = _NUPA[key][str(L)]
        order = list(range(len(items)))  # per-length shuffle: a larger n extends a smaller one
        random.Random(f"{seed}-{data}-{L}").shuffle(order)
        for j in sorted(order[:per]):
            s = items[j]
            task = s.split(" . ", 1)[1]  # drop "Directly return the answer ... like 123 ."
            m = re.fullmatch(r"(.*?:\s*)(\d+) (\S) (\d+) = (\d+)", task)
            assert m and m[3] == sym, s
            g = next(f"{lo}-{hi}" for lo, hi in NUPA_GROUPS if lo <= L <= hi)
            out.append({"id": f"{data}-L{L}-{j}", "question": f"{m[1]}{m[2]} {m[3]} {m[4]}",
                        "gold": m[5], "group": f"len {g}", "length": L, "clean": True,
                        "calls": [[op, m[2], m[4]]]})
    return out


def load_bbh(data: str, n: int, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    t = pq.read_table(DATA / "bbh" / "mat2.parquet").to_pylist()
    idx = sorted(random.Random(seed).sample(range(len(t)), min(n, len(t))))
    out = []
    for i in idx:
        q = re.sub(r"\s*=\s*$", "", t[i]["input"].strip())
        out.append({"id": f"{data}-{i}", "question": q, "gold": t[i]["target"].strip(),
                    "group": "multistep_arithmetic_two", "clean": is_clean(q, t[i]["target"])})
    return out


def load_eleu(data: str, n: int, seed: int) -> list[dict]:
    sub = data[5:]
    rows = [json.loads(x) for x in (DATA / "eleuther" / f"{sub}.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]
    idx = sorted(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
    out = []
    for i in idx:
        q = re.sub(r"^\s*Q:\s*|\s*A:\s*$", "", rows[i]["context"]).strip()
        gold = rows[i]["completion"].strip()
        p = {"id": f"{data}-{i}", "question": q, "gold": gold, "group": sub,
             "clean": is_clean(q, gold) and "three_ops" not in sub}
        for word, (op, w) in ELEU_OPS.items():
            m = re.fullmatch(rf"What is (\d+) {w} (\d+)\?", q)
            if word in sub and m:
                p["calls"] = [[op, m[1], m[2]]]
        out.append(p)
    return out


def load(data: str, n: int | None = None, seed: int = 0) -> list[dict]:
    n = n or DEFAULT_N[family(data)]
    fam = family(data)
    if fam in ("dmx", "dmi", "dmxc"):
        return load_dm(data, n, seed)
    if fam == "nupa":
        return load_nupa(data, n, seed)
    if fam == "bbh":
        return load_bbh(data, n, seed)
    if fam == "eleu":
        return load_eleu(data, n, seed)
    raise ValueError(data)


# ---------------------------------------------------------------------------------- judging

def gold_value(gold: str) -> Fraction:
    gold = gold.replace(",", "").strip()
    return Fraction(gold) if "/" in gold else Fraction(Decimal(gold))


def _norm(s: str) -> str:
    s = re.sub(r"\\(?:text|mathrm|textbf|mbox|mathbf)\{([^{}]*)\}", r" \1 ", s)
    s = s.replace("\u2212", "-").replace("\\left", "").replace("\\right", "").replace("{,}", ",")
    for x in ("\\$", "$", "\\,", "\\!", "\\;", "\\ ", "~", "**"):
        s = s.replace(x, " ")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)  # thousands separators
    s = re.sub(r"(?<=\d)\s+(?=\d{3}(?!\d))", "", s)  # 1 234 567
    s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
    return s.strip()


NUMR = r"-?\s*\d+(?:\.\d+)?"


def parse_value(s: str):
    """A boxed answer as an exact rational (decimals and a/b fractions), else None."""
    s = _norm(s)
    if "=" in s:
        s = s.rsplit("=", 1)[1]
    s = s.strip().rstrip(".").strip()
    m = re.fullmatch(rf"(-)?\s*\(?\s*({NUMR})\s*\)?\s*/\s*\(?\s*({NUMR})\s*\)?", s)
    if m:
        a, b = Fraction(Decimal(m[2].replace(" ", ""))), Fraction(Decimal(m[3].replace(" ", "")))
        if b == 0:
            return None
        return (-1 if m[1] else 1) * a / b
    m = re.fullmatch(rf"\(?\s*({NUMR})\s*\)?", s)
    if m:
        return Fraction(Decimal(m[1].replace(" ", "")))
    nums = re.findall(NUMR, s)
    if len(nums) == 1:
        return Fraction(Decimal(nums[0].replace(" ", "")))
    return None


def strict_pred(text: str):
    """The answer as evals.judge reads it (user-turn cut, last \\boxed{} after the thinking and
    outside tool calls, else the last number), parsed exactly (no rounding)."""
    m = E.USER_TURN.search(text)
    if m:
        text = text[:m.start()]
    t = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    t = re.sub(r"<tool_call>.*?(</tool_call>|$)", " ", t, flags=re.DOTALL)
    t = re.sub(r"<tool_response>.*?(</tool_response>|$)", " ", t, flags=re.DOTALL)
    t = re.sub(r"<\|[^|]*\|>(assistant|user)?", " ", E.MARK.sub(" ", t))
    boxes = [E._balanced(t, b.end()) for b in re.finditer(r"\\boxed\s*\{", t)]
    if boxes:
        try:
            return parse_value(boxes[-1]), "boxed"
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None, "boxed_unparsed"
    nums = re.findall(r"-?\d+(?:\.\d+)?", _norm(t))
    if nums:
        return Fraction(Decimal(nums[-1])), "last_number"
    return None, "none"


def exact_fields(p: dict, text: str) -> dict:
    v, src = strict_pred(text)
    g = gold_value(p["gold"])
    return {"pred_exact": (str(v.numerator) if v.denominator == 1 else
                           f"{v.numerator}/{v.denominator}") if v is not None else None,
            "src_exact": src, "exact": v is not None and v == g}


# ---------------------------------------------------------------------------------- running

def _judgeable(p: dict) -> dict:
    """evals.judge reads the gold with Decimal(): give it a fraction gold as a 30-digit decimal
    (it only feeds the tolerant ``correct``; ``exact`` uses the true gold)."""
    if "/" not in p["gold"]:
        return p
    import decimal

    g = gold_value(p["gold"])
    with decimal.localcontext() as c:
        c.prec = 40
        dec = Decimal(g.numerator) / Decimal(g.denominator)
    return p | {"gold": str(dec)}


def rescore_errors(args) -> None:
    """Re-judge records whose scoring failed (score_error) from their stored ids / text."""
    out = Path(args.out)
    for f in sorted(out.glob("*.jsonl")):
        parts = f.name.split(".")
        if len(parts) != 3:
            continue
        recs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        bad = [r for r in recs if r.get("score_error")]
        if not bad:
            continue
        probs = {p["id"]: p for p in load(parts[0], 100_000)}
        for r in bad:
            p = probs[r["id"]]
            j = E.judge(_judgeable(p), r["text"], False)
            r.update(j)
            r.update(exact_fields(p, r["text"]))
            r["source"] = j["source"]
            r["gold"] = p["gold"]
            r.pop("score_error")
            r["rescored"] = True
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                     encoding="utf-8")
        print(f"{f.name}: rescored {len(bad)}", flush=True)


def snapshot() -> str:
    snap = HOME / "snap.txt"
    if snap.exists():
        return snap.read_text().strip().splitlines()[-1].strip()
    return E.default_model("base")


def setup(cond: str, mpath: str, batch: int, max_len: int, fast: bool, compile_: bool,
          view: int = 0):
    """evals.run's model setup: tokenizer, model, BIOS, Ctx, hook, ban, BatchDecoder."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    kind, flags = E.parse_cond(cond)
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
        if kind in E.BIOS_KINDS:
            import hashlib

            bios = [t + chip[0] for t in boot_tokens(Chip8(bcd.ROM, bcd.QUIRKS).cells)]
            bios_id = f"{hashlib.sha1(bcd.ROM).hexdigest()[:10]} ({len(bcd.ROM)} B, OPS {bcd.OPS})"
            print(f"BIOS {bios_id}, boot record {len(bios)} tokens", flush=True)
    ctx = E.Ctx(tok, chip)
    hook = {"pytool_inline": E.executor("inline", ctx), "pytool": E.executor("template", ctx)}.get(kind)
    banned = []
    if "nb" in flags or "ne" in flags or kind == "merged_inline":
        gc = Path(mpath) / "generation_config.json"
        bad = json.loads(gc.read_text()).get("bad_words_ids", []) if gc.exists() else []
        # ne: <|endoftext|> too (Qwen otherwise ends a tool-mode turn right after the response)
        banned = sorted({w[0] for w in bad if len(w) == 1} | {E.IM_START}
                        | ({getattr(E, "ENDOFTEXT", 248044)} if "ne" in flags else set()))
        model.lm_head = E.BannedHead(model.lm_head, banned, model.lm_head.weight.shape[0])
        print(f"banned token ids {banned}", flush=True)
    import torch._inductor.config

    from chipformer.q35.batch import BatchDecoder

    torch._inductor.config.shape_padding = False
    if fast and chip:
        from chipformer.q35 import fast as F

        print("compact", F.compact(model, qwen=False), flush=True)
    bd = BatchDecoder(model, batch=batch, max_len=max_len, eos=sorted(ctx.stop), sizes=[batch],
                      compile=compile_, view=view,
                      **({"skip_qwen": True} if fast and chip else {}),
                      **({"freeze_chip_positions": True} if "fp" in flags else {}),
                      **({"transparent_chip": True} if "tc" in flags else {}))
    return tok, model, ctx, hook, bios, bios_id, banned, bd


def decode_views(bd, ctx, prompts: list[list[int]], budget: int, hook, on_done) -> None:
    """evals.decode_all (same rows, stop rules, forcing, chip steps), except that each step's
    attention reads only the smallest captured KV view covering every live row (BatchDecoder
    ``view``), instead of the whole static cache. Without views it is decode_all."""
    from collections import deque

    import torch

    if not getattr(bd, "views", None) or len(bd.views) < 2:
        return E.decode_all(bd, ctx, prompts, budget, hook, on_done)
    queue = deque(range(len(prompts)))
    n = bd.batch
    bd.prefill(prompts[0])
    level = bd._prepare(n)
    rows, tok, pos = level[:3]
    has_chip = level[4] is not None
    graphs = bd.view_graphs[n]
    # fp: RoPE positions frozen on chip tokens (a separate per-row buffer in BatchDecoder)
    rpos = bd.rpos[n] if getattr(bd, "freeze", False) else None
    first = bd.prefill(prompts[queue[0]])
    last = [0] * n
    live: list = [None] * n
    length = [0] * n  # tokens in each row's cache after its next input

    def admit(b, first=None):
        while queue:
            i = queue.popleft()
            t = bd.prefill(prompts[i]) if first is None else first
            first = None
            r = E._Row(i, len(prompts[i]))
            if r.push(t, False, ctx, budget, bd.max_len, hook):
                on_done(i, r.result())
                continue
            rows.load_row(b, bd.small, len(prompts[i]))
            if hasattr(bd, "mark_prompt"):  # _tc: its chip slots (no-op otherwise)
                bd.mark_prompt(n, b, prompts[i])
            tok[b, 0] = t
            pos[b] = len(prompts[i])
            if rpos is not None:
                rpos[b] = sum(not ctx.is_chip(x) for x in prompts[i])
            live[b], last[b], length[b] = r, t, len(prompts[i]) + 1
            return
        live[b] = None
        tok[b, 0] = 0
        pos[b] = 0
        length[b] = 0

    admit(0, first)
    for b in range(1, n):
        admit(b)
    with torch.no_grad():
        while any(r is not None for r in live):
            busy = [b for b in range(n) if live[b] is not None]
            use_chip = has_chip and all(ctx.is_chip(last[b]) for b in busy)
            need = max(length[b] for b in busy)
            v = next(x for x in bd.views if x >= need)
            step_fn, chip_fn = graphs[v]
            pred = (chip_fn if use_chip else step_fn)(tok, pos).tolist()
            bd.stats["view_sum"] += v
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
                length[b] += 1
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


def default_model(cond: str) -> str:
    return str(ROOT / "dist" / "q4-v10") if cond.startswith("merged") else snapshot()


def run(args) -> None:
    import torch

    cond = args.cond + (f"-{args.tag}" if args.tag else "")
    kind, _ = E.parse_cond(cond)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for data in args.data:
        path = out / f"{data}.{cond}.jsonl"
        done = set()
        if path.exists():
            recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()
                    if x.strip()]
            # --redo-context: rerun problems that hit the cache length (the newer record is
            # appended; readers keep the last record per id)
            done = {r["id"] for r in recs if not (args.redo_context and r["stop"] == "context")}
        probs = load(data, args.n, args.seed)
        if args.per_data:  # the first K of the (sorted, seeded) sample: a subset of the full run
            probs = probs[:args.per_data]
        probs = [p for p in probs if p["id"] not in done]
        print(f"{path.name}: {len(done)} done, {len(probs)} to go", flush=True)
        todo += [(data, p) for p in probs]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        return
    mpath = args.model or default_model(cond)
    budget = args.budget
    batch = min(args.batch, len(todo))
    # the cache length is fixed per condition (one compile); prompts here are short
    max_len = args.max_len or (budget + 1024 + (4096 if cond.startswith("merged") else 512))
    max_len = -(-max_len // 1024) * 1024
    tok, _model, ctx, hook, bios, bios_id, banned, bd = setup(
        cond, mpath, batch, max_len, args.fast, not args.no_compile, args.view)
    prompts = [E.build_prompt(tok, p["question"], cond, bios) for _, p in todo]
    assert max(map(len, prompts)) + budget + 512 < max_len, "prompt too long for max_len"
    if kind == "merged" and not args.no_selftest:
        E.selftest(bd, ctx, tok, bios)
    print(f"batch {bd.batch}, max_len {max_len}, budget {budget}, {len(todo)} problems",
          flush=True)
    t0 = time.time()
    counts = {}

    def done(i, r):
        data, p = todo[i]
        try:
            rec = E.score(_judgeable(p), cond, ctx, r, r["seconds"], False)
            rec |= exact_fields(p, rec["text"])
            rec["gold"] = p["gold"]
        except (ArithmeticError, ValueError, TypeError, KeyError, IndexError) as e:
            # never lose the run to one transcript's scoring
            print(f"scoring error {p['id']}: {e!r}", flush=True)
            rec = {"id": p["id"], "cond": cond, "question": p["question"], "gold": p["gold"],
                   "pred": None, "source": "score_error", "inline": [], "correct": False,
                   "calls": [], "stop": r["stop"], "n_qwen": r["n_qwen"],
                   "n_chip": sum(1 for t in r["out"] if ctx.is_chip(t)), "n_out": len(r["out"]),
                   "seconds": round(r["seconds"], 2), "text": ctx.show(r["out"]),
                   "pred_exact": None, "src_exact": "score_error", "exact": False,
                   "score_error": repr(e)}
        rec |= {"data": data, "group": p["group"], "clean": p["clean"],
                "n_prompt": len(prompts[i]), "batch": bd.batch, "model": mpath,
                "bios": bios_id, "banned": banned, "ids": r["out"], "inj_at": r.get("inj_at")}
        if "length" in p:
            rec["length"] = p["length"]
        with (out / f"{data}.{cond}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        counts[data] = counts.get(data, 0) + 1
        print(json.dumps({k: rec[k] for k in ("id", "gold", "pred_exact", "exact", "stop",
                                               "n_qwen", "n_chip", "seconds")}
                         | {"calls": len(rec["calls"]), "fires": len(rec["inline"])}), flush=True)

    decode_views(bd, ctx, prompts, budget, hook, done)
    wall = time.time() - t0
    with (out / "runs.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"cond": cond, "data": counts, "n": len(todo), "batch": bd.batch,
                            "wall_s": round(wall, 1), "s_per_problem": round(wall / len(todo), 2),
                            "view": args.view, "stats": bd.stats, "model": mpath, "bios": bios_id, "budget": budget,
                            "gpu": torch.cuda.get_device_name()}) + "\n")
    print(f"done {len(todo)} in {wall:.0f}s ({wall / len(todo):.1f} s/problem)", flush=True)


# ---------------------------------------------------------------------------------- checks

def identity(args) -> None:
    """BatchDecoder (the config the runs use) vs HF generate, greedy, on a few problems."""
    import torch

    cond = args.cond
    probs = load("dmx_mul_big", 100)[:2] + load("nupa_mulh", 400)[100:102] + load("bbh_mat2")[:1]
    mpath = args.model or default_model(cond)
    max_len = 1024 + 4096 + args.max_new
    max_len = -(-max_len // 1024) * 1024
    tok, model, ctx, _hook, bios, _, _banned, bd = setup(cond, mpath, args.batch, max_len,
                                                      True, True)
    prompts = [E.build_prompt(tok, p["question"], cond, bios) for p in probs]
    got = {}
    E.decode_all(bd, ctx, prompts, args.max_new, None, lambda i, r: got.__setitem__(i, r["out"]))
    for i, p in enumerate(prompts):
        x = torch.tensor([p], device="cuda")
        with torch.no_grad():
            g = model.generate(x, max_new_tokens=len(got[i]), do_sample=False,
                               cache_implementation="static")[0, len(p):].tolist()
        a = got[i]
        k = next((j for j in range(min(len(a), len(g))) if a[j] != g[j]), min(len(a), len(g)))
        n_chip = sum(ctx.is_chip(t) for t in a)
        print(json.dumps({"id": probs[i]["id"], "batch_len": len(a), "gen_len": len(g),
                          "identical": a == g[:len(a)], "first_diff": k, "chip_tokens": n_chip,
                          "diff_ctx": ctx.show(a[max(0, k - 20):k + 5]) if a != g[:len(a)] else ""}),
              flush=True)


def tfcheck(args) -> None:
    """Teacher-forced check of stored records against HF's plain forward (eager, batch 1): at
    every generated position, is the stored token the HF model's argmax? Where it is not, the
    top-2 logit gap shows whether it is a bf16 near-tie (kernel rounding) or a real
    disagreement. Positions with a forced token (pytool) are skipped."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    cond = args.cond
    mpath = args.model or default_model(cond)
    tok = AutoTokenizer.from_pretrained(mpath)
    model = AutoModelForImageTextToText.from_pretrained(mpath, dtype=torch.bfloat16).cuda().eval()
    from chipformer.q35.fast import exact_matmuls

    exact_matmuls()
    kind, flags = E.parse_cond(cond)
    chip, bios = None, []
    meta_path = Path(mpath) / "chipformer.json"
    if meta_path.exists():
        from chipformer.q35.lib import bcd
        from chipformer.ref.emu import Chip8
        from chipformer.trace import boot_tokens

        meta = json.loads(meta_path.read_text())
        chip = (meta["chip_token_offset"], meta["chip_vocab"])
        if kind in E.BIOS_KINDS:
            bios = [t + chip[0] for t in boot_tokens(Chip8(bcd.ROM, bcd.QUIRKS).cells)]
    banned = []
    if "nb" in flags or "ne" in flags or kind == "merged_inline":
        gc = Path(mpath) / "generation_config.json"
        bad = json.loads(gc.read_text()).get("bad_words_ids", []) if gc.exists() else []
        # ne: <|endoftext|> too (Qwen otherwise ends a tool-mode turn right after the response)
        banned = sorted({w[0] for w in bad if len(w) == 1} | {E.IM_START}
                        | ({getattr(E, "ENDOFTEXT", 248044)} if "ne" in flags else set()))
    out = Path(args.out)
    for data in args.data:
        src = out / f"{data}.{cond}.jsonl"
        recs = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
        for r in recs[:args.k]:
            prompt = E.build_prompt(tok, r["question"], cond, bios)
            gen = r["ids"] if r.get("ids") is not None else tok(
                r["text"], add_special_tokens=False).input_ids
            ids = prompt + gen
            with torch.no_grad():
                lg = model(torch.tensor([ids], device="cuda")).logits[0].float()
            if banned:
                lg[:, banned] = float("-inf")
            lg = lg[len(prompt) - 1:len(ids) - 1]
            top2 = lg.topk(2, dim=-1)
            want = torch.tensor(gen, device="cuda")
            arg = top2.indices[:, 0]
            miss = (arg != want).nonzero().flatten().tolist()
            gaps = [round(float(lg[p, arg[p]] - lg[p, want[p]]), 3) for p in miss]
            chipmiss = [p for p in miss if chip and chip[0] <= gen[p] < chip[0] + chip[1]]
            print(json.dumps({"id": r["id"], "cond": cond, "n_gen": len(gen),
                              "argmax_agree": len(gen) - len(miss), "disagree": len(miss),
                              "first": miss[:1], "gap_at_disagree": gaps[:5],
                              "max_gap": max(gaps) if gaps else 0.0,
                              "chip_token_disagree": len(chipmiss)}), flush=True)


def replay(args) -> None:
    """Replay every merged record through the circuit IR: at every position the circuit owns,
    the model's token must be the IR's prediction."""
    import numpy as np
    from transformers import AutoTokenizer

    from chipformer.q35 import bridge as B
    from chipformer.q35 import check as Q
    from chipformer.q35 import chip8_q35, ir
    from chipformer.q35.lib import bcd
    from chipformer.q35.spikes_replay import class_map
    from chipformer.ref.emu import Chip8
    from chipformer.trace import boot_tokens

    mpath = args.model or default_model("merged")
    meta = json.loads((Path(mpath) / "chipformer.json").read_text())
    off, nchip = meta["chip_token_offset"], meta["chip_vocab"]
    tok = AutoTokenizer.from_pretrained(mpath)
    cmap = class_map(tok, meta)
    circuit = chip8_q35.build()
    bios = [t + off for t in boot_tokens(Chip8(bcd.ROM, bcd.QUIRKS).cells)]
    out = Path(args.out)
    for data in args.data:
        src = out / f"{data}.{args.cond}.jsonl"
        if not src.exists():
            continue
        dst = out / f"{data}.{args.cond}.replay.jsonl"
        have = set()
        if dst.exists():
            have = {json.loads(x)["id"] for x in dst.read_text(encoding="utf-8").splitlines() if x}
        recs = []
        for x in src.read_text(encoding="utf-8").splitlines():
            try:  # the run may be appending the last line right now
                recs.append(json.loads(x))
            except json.JSONDecodeError:
                pass
        if args.shard:
            k, m = map(int, args.shard.split("/"))
            recs = recs[k::m]
        for r in recs:
            if r["id"] in have or r.get("ids") is None:
                continue
            gen = r["ids"]
            if not any(off <= t < off + nchip for t in gen):
                res = {"id": r["id"], "chip_tokens": 0, "owned": 0, "mismatch": 0, "skipped": True}
            else:
                t0 = time.time()
                prompt = E.build_prompt(tok, r["question"], r["cond"], bios)
                ids = prompt + gen
                stream = [t - off if off <= t < off + nchip else cmap.get(t, B.OTHER)
                          for t in ids]
                pred, m = Q.predictions(circuit, stream)
                owner = ir.ev(circuit.owner, m)
                rng = range(len(prompt) - 1, len(ids) - 1)
                owned = [p for p in rng if owner[p] == 1]
                bad = [p for p in owned if pred[p] != stream[p + 1]]
                # chip tokens Qwen wrote where the circuit does not own the position
                stray = [p for p in rng if owner[p] != 1 and off <= ids[p + 1] < off + nchip]
                res = {"id": r["id"], "chip_tokens": sum(off <= t < off + nchip for t in gen),
                       "owned": len(owned), "mismatch": len(bad), "stray_chip": len(stray),
                       "first": [(p - len(prompt), int(stream[p + 1]), int(pred[p]))
                                 for p in bad[:3]],
                       "seconds": round(time.time() - t0, 1)}
                del m
            with dst.open("a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
            print(data, json.dumps(res), flush=True)
        np.zeros(1)


# ---------------------------------------------------------------------------------- summary

def wilson(k, n):
    return E.wilson(k, n)


def paired(ra: dict, rb: dict, key="exact"):
    ids = sorted(set(ra) & set(rb))
    if not ids:
        return None
    xa = [ra[i][key] for i in ids]
    xb = [rb[i][key] for i in ids]
    n = len(ids)
    bo = sum(x and not y for x, y in zip(xa, xb))
    co = sum(y and not x for x, y in zip(xa, xb))
    d = (bo - co) / n
    se = math.sqrt(max(0.0, (bo + co) / n - d * d) / n)
    return n, d, d - 1.96 * se, d + 1.96 * se, bo, co, E.mcnemar(bo, co)


def machine_stats(rs: list[dict]):
    """Every result the machine printed: tool responses of valid calls and parsed inline fires.
    exact = printed == A op B."""
    tool = [c["exact"] for r in rs for c in r["calls"] if c["valid"] and c["response"] is not None]
    inl = [f["exact"] for r in rs for f in r.get("inline", []) if "exact" in f]
    unparsed = sum("exact" not in f for r in rs for f in r.get("inline", []))
    return tool, inl, unparsed


def load_all(out: Path):
    res, rep = {}, {}
    for f in sorted(out.glob("*.jsonl")):
        parts = f.name.split(".")
        if len(parts) == 3:
            recs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
            res[(parts[0], parts[1])] = {r["id"]: r for r in recs}
        elif len(parts) == 4 and parts[2] == "replay":
            recs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
            rep[(parts[0], parts[1])] = {r["id"]: r for r in recs}
    return res, rep


def _pct(k, n):
    return f"{100 * k / n:.1f}%" if n else "-"


def _table(res, rep, keys: list[tuple], conds: list[str], title: str, wall: dict) -> list[str]:
    """One table over the pooled records of ``keys`` (datasets), per condition."""
    L = [f"\n### {title}\n",
         ("| cond | N | exact acc | 95% CI | clean acc (n) | vs base (pts, 95% CI, McNemar p) | "
          "vs base_nudge | tool-call rate | inline-fire rate | answer = a printed result | "
          "machine exact (tool / inline) | IR replay mismatches | cutoffs | loops | Qwen tok | "
          "chip tok | s/prob |"),
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    pooled = {}
    for c in conds:
        d = {}
        for k in keys:
            d |= res.get((k, c), {})
        pooled[c] = d
    for c in conds:
        rs = list(pooled[c].values())
        if not rs:
            continue
        n, k = len(rs), sum(r["exact"] for r in rs)
        lo, hi = wilson(k, n)
        cl = [r["exact"] for r in rs if r["clean"]]
        cmp = []
        for b in ("base", "base_nudge"):
            pr = paired(pooled[c], pooled.get(b, {})) if c != b else None
            cmp.append(f"{100 * pr[1]:+.1f} ({100 * pr[2]:+.1f}..{100 * pr[3]:+.1f}), "
                       f"p={pr[6]:.2g} [{pr[4]}/{pr[5]}]" if pr else "-")
        tool, inl, unp = machine_stats(rs)
        rp = [rep.get((key, c), {}) for key in keys]
        rpr = [x for dct in rp for i, x in dct.items() if i in pooled[c]]
        rp_s = "-"
        if rpr:
            checked = [x for x in rpr if not x.get("skipped")]
            rp_s = (f"{sum(x['mismatch'] for x in checked)} in "
                    f"{sum(x['owned'] for x in checked)} owned pos. ({len(checked)} recs"
                    f"{'' if len(rpr) == n else f', {len(rpr)}/{n} done'})")
        ws = [wall[(key, c)] for key in keys if (key, c) in wall]
        L.append(
            f"| {c} | {n} | {_pct(k, n)} | {100 * lo:.0f}-{100 * hi:.0f} | "
            f"{_pct(sum(cl), len(cl))} ({len(cl)}) | {cmp[0]} | {cmp[1]} | "
            f"{_pct(sum(bool(r['calls']) for r in rs), n)} | "
            f"{_pct(sum(bool(r.get('inline')) for r in rs), n)} | "
            f"{_pct(sum(_from_machine(r) for r in rs), n)} | "
            f"{sum(tool)}/{len(tool)} / {sum(inl)}/{len(inl)}{f' (+{unp} unparsed)' if unp else ''} | "
            f"{rp_s} | {sum(r['stop'] != 'eos' for r in rs)} | "
            f"{sum(E._looping(r.get('inline', [])) for r in rs)} | "
            f"{sum(r['n_qwen'] for r in rs) / n:.0f} | {sum(r['n_chip'] for r in rs) / n:.0f} | "
            f"{f'{sum(ws) / len(ws):.1f}' if ws else '-'} |")
    return L


def _cause(r: dict) -> str:
    """Why a wrong answer is wrong (the first that applies)."""
    tool, inl, _ = machine_stats([r])
    if not all(tool) or not all(inl):
        return "wrong print"
    loop = E._looping(r.get("inline", []))
    if r["stop"] != "eos":
        return "cutoff (looping)" if loop else "cutoff"
    if r.get("pred_exact") is None:
        return "no answer"
    if loop:
        return "looping, then wrong"
    g = str(gold_value(r["gold"]))
    printed = {f.get("true") for f in r.get("inline", []) if f.get("exact")} | {
        (c.get("response") or "").strip() for c in r["calls"] if c.get("exact")}
    if g in printed:
        return "gold was printed, answer differs"
    return "Qwen's own error"


CAUSES = ["wrong print", "cutoff (looping)", "cutoff", "looping, then wrong", "no answer",
          "gold was printed, answer differs", "Qwen's own error"]


def _fail_table(res, keys, conds) -> list[str]:
    L = ["\nWrong answers by cause (pooled; loops = 3+ identical inline fires in a row):\n",
         "| cond | wrong | " + " | ".join(CAUSES) + " |", "|---|---|" + "---|" * len(CAUSES)]
    for c in conds:
        rs = [r for k in keys for r in res.get((k, c), {}).values()]
        if not rs:
            continue
        bad = [_cause(r) for r in rs if not r["exact"]]
        L.append(f"| {c} | {len(bad)}/{len(rs)} | " + " | ".join(
            str(bad.count(x)) for x in CAUSES) + " |")
    return L


def _from_machine(r: dict) -> bool:
    """The final answer is a number the machine (or the Python tool) printed."""
    p = r.get("pred_exact")
    if p is None:
        return False
    return (any(f.get("is_answer") for f in r.get("inline", []))
            or any((c.get("response") or "").strip() == p for c in r["calls"]))


def _wall(out: Path) -> dict:
    """(data, cond) -> wall seconds per problem of the run(s) that made it (a run that covers
    several datasets shares its wall time by problem count)."""
    tot = {}
    log = out / "runs.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            x = json.loads(line)
            for d, n in x["data"].items():
                w, m = tot.get((d, x["cond"]), (0.0, 0))
                tot[(d, x["cond"])] = (w + x["s_per_problem"] * n, m + n)
    return {k: w / n for k, (w, n) in tot.items() if n}


def _groups(res, keys, conds):
    """Accuracy per group (DM module / NUPA length bucket / Eleuther subset)."""
    groups = {}
    for k in keys:
        for c in conds:
            for r in res.get((k, c), {}).values():
                groups.setdefault(f"{k}:{r['group']}", set())
    L = ["\n| group | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for g in sorted(groups, key=lambda s: (s.split(":")[0], [int(x) if x.isdigit() else x
                                                            for x in re.split(r"(\d+)", s)])):
        k, name = g.split(":", 1)
        cells = []
        for c in conds:
            rs = [r for r in res.get((k, c), {}).values() if r["group"] == name]
            cells.append(f"{sum(r['exact'] for r in rs)}/{len(rs)}" if rs else "-")
        L.append(f"| {g} | " + " | ".join(cells) + " |")
    return L


BENCH = [("DeepMind Mathematics, extrapolate (big / longer) test modules",
          [k for k in DM_MODULES if k.startswith("dmx") and k not in DM_DIV]),
         (("DeepMind Mathematics, extrapolate big modules, clean problems only (non-negative "
           "integers, no decimals)"), DMXC),
         ("DeepMind Mathematics, extrapolate division modules (div_big, mul_div_multiple_longer, "
          "clean div_big)", DM_DIV),
         ("DeepMind Mathematics, interpolate test modules", [k for k in DM_MODULES if k.startswith("dmi")]),
         ("NUPA integer add / sub / multiply (lengths 3-20)", list(NUPA_TASKS)),
         ("BBH multistep_arithmetic_two", ["bbh_mat2"]),
         ("EleutherAI/arithmetic", [f"eleu_{s}" for s in ELEU])]


def summary(args) -> None:
    out = Path(args.out)
    res, rep = load_all(out)
    wall = _wall(out)
    conds = [c for c in CONDS if any(cc == c for _, cc in res)] + sorted(
        {cc for _, cc in res if cc not in CONDS})
    L = ["# Arithmetic benchmarks: merged (computer in the weights) vs base Qwen3.5-4B\n",
         PREAMBLE]
    for title, keys in BENCH:
        if not any((k, c) in res for k in keys for c in conds):
            continue
        L += _bench(res, rep, wall, conds, title, keys)
    text = "\n".join(L)
    print(text)
    (out / "SUMMARY.md").write_text(text + "\n", encoding="utf-8")


PREAMBLE = (
    "Non-thinking, greedy, 2048 Qwen-written tokens per problem (the machine's tokens are not "
    "counted), evals.py's prompt suffix (final answer in \\boxed{}). Problems are seeded "
    "samples, identical across conditions (paired). exact acc = boxed (else last) number equals "
    "the gold exactly as a rational (no tolerance). clean = all numbers in the problem and the "
    "answer are non-negative integers and there is no division. Paired deltas: A - B in points "
    "with a 95% CI, exact McNemar p, [A-only/B-only wins]. machine exact = printed result == "
    "A op B over every valid tool call / parsed inline fire. IR replay = positions owned by the "
    "circuit where the model's token differs from the circuit IR's prediction (spikes_replay). "
    "cutoffs = budget or context stops; loops = 3+ identical inline fires in a row. s/prob = "
    "wall time per problem at the run's batch size (base / base_nudge / pytool / merged_nb: L4, "
    "batch 16 / 16 / 16 / 8; v11 merged_inline*: L40S, batch 32; v13: L4 / A10G, batch 8). "
    "Models: merged_nb = q4-v12 (commit 62ed528); merged_inline* = q4-v11 (afd8230; v12 only "
    "changes tool-call capture); -v13 = q4-v13 (f270d1b: inline A - B < 0 prints its negative "
    "result, inline add/sub need >= 7 digits). _fp = RoPE positions frozen on chip tokens; "
    "_tc = chip tokens invisible to Qwen's layers; _mix = NUDGE_MIX (products inline, big "
    "add/sub through the tools).\n")


def _bench(res, rep, wall, conds, title, keys) -> list[str]:
    L = [f"\n## {title}"]
    L += _table(res, rep, keys, conds, "all modules pooled" if len(keys) > 1 else title, wall)
    L += _fail_table(res, keys, conds)
    if len(keys) > 1:
        for k in keys:
            if any((k, c) in res for c in conds):
                L += _table(res, rep, [k], conds, k, wall)
                L += _fail_table(res, [k], conds)
    L.append("\nPer group (exact correct / N):")
    L += _groups(res, keys, conds)
    return L


def _tags(r: dict) -> set:
    fires = r.get("inline", [])
    t = set()
    if any(f.get("true", "0").startswith("-") for f in fires):
        t.add("negative fire")
    if E._looping(fires):
        t.add("loop")
    if r["stop"] != "eos":
        t.add("cutoff")
    return t


def losses(args) -> None:
    """Paired wins / losses of A vs B on the given datasets, each problem tagged with what
    happened in A's transcript (negative inline fire, loop, cutoff)."""
    res, _ = load_all(Path(args.out))
    L = [f"\n{args.a} vs {args.b}\n",
         ("| data | N | A | B | A-only | B-only (losses) | losses with negative fire | with loop | "
          "with cutoff | none of these | McNemar p |"),
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for data in [*args.data, "pooled"]:
        keys = args.data if data == "pooled" else [data]
        ra = {i: r for k in keys for i, r in res.get((k, args.a), {}).items()}
        rb = {i: r for k in keys for i, r in res.get((k, args.b), {}).items()}
        ids = sorted(set(ra) & set(rb))
        if not ids:
            continue
        win = [i for i in ids if ra[i]["exact"] and not rb[i]["exact"]]
        lose = [i for i in ids if rb[i]["exact"] and not ra[i]["exact"]]
        tags = [_tags(ra[i]) for i in lose]
        L.append(f"| {data} | {len(ids)} | {sum(ra[i]['exact'] for i in ids)} | "
                 f"{sum(rb[i]['exact'] for i in ids)} | {len(win)} | {len(lose)} | "
                 f"{sum('negative fire' in t for t in tags)} | {sum('loop' in t for t in tags)} | "
                 f"{sum('cutoff' in t for t in tags)} | {sum(not t for t in tags)} | "
                 f"{E.mcnemar(len(win), len(lose)):.2g} |")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("cond", type=E.cond_arg)
    r.add_argument("data", nargs="+", choices=ALL)
    r.add_argument("--n", type=int, help="problems per dataset (default per family)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--model")
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--budget", type=int, default=2048)
    r.add_argument("--max-len", type=int)
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--limit", type=int, help="only the first N problems still to do")
    r.add_argument("--per-data", type=int, help="only the first K problems of each dataset")
    r.add_argument("--redo-context", action="store_true",
                   help="rerun problems whose stored record stopped at the cache length")
    r.add_argument("--no-compile", action="store_true")
    r.add_argument("--no-selftest", action="store_true")
    r.add_argument("--fast", action="store_true", default=True)
    r.add_argument("--tag")
    r.add_argument("--view", type=int, default=1024,
                   help="attention KV view granularity (bit-identical; 0 = whole cache)")
    i = sub.add_parser("identity")
    i.add_argument("--cond", default="merged_inline")
    i.add_argument("--model")
    i.add_argument("--batch", type=int, default=8)
    i.add_argument("--max-new", type=int, default=1024)
    t = sub.add_parser("tfcheck")
    t.add_argument("cond")
    t.add_argument("data", nargs="+", choices=ALL)
    t.add_argument("--model")
    t.add_argument("--out", default=str(OUT))
    t.add_argument("--k", type=int, default=5)
    p = sub.add_parser("replay")
    p.add_argument("cond")
    p.add_argument("data", nargs="+", choices=ALL)
    p.add_argument("--model")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--shard")
    lo = sub.add_parser("losses", help="paired wins / losses of A vs B, losses tagged")
    lo.add_argument("a")
    lo.add_argument("b")
    lo.add_argument("data", nargs="+", choices=ALL)
    lo.add_argument("--out", default=str(OUT))
    e = sub.add_parser("rescore-errors")
    e.add_argument("--out", default=str(OUT))
    s = sub.add_parser("summary")
    s.add_argument("--out", default=str(OUT))
    ld = sub.add_parser("show", help="print a few loaded problems")
    ld.add_argument("data", nargs="+", choices=ALL)
    ld.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    if args.cmd == "show":
        for d in args.data:
            ps = load(d)
            print(d, len(ps), "clean", sum(p["clean"] for p in ps))
            for p in ps[:args.k]:
                print("  ", json.dumps(p))
        return
    {"run": run, "summary": summary, "replay": replay, "identity": identity,
     "tfcheck": tfcheck, "rescore-errors": rescore_errors,
     "losses": losses}[args.cmd](args)


if __name__ == "__main__":
    main()
