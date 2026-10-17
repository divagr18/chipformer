"""Text benchmarks: is a merged Qwen3.5 + CHIP-8 checkpoint as capable as its base on pure text?

    python -m chipformer.q35.text_eval run MODEL --name NAME [--attn sdpa|eager] [--tokenizer BASE]
    python -m chipformer.q35.text_eval table [dist/evals/text] [--ref base]

``run`` evaluates one checkpoint and writes ``OUT/NAME/TASK.json`` (per-item records; a task whose
file exists is skipped, so a killed run resumes). ``table`` compares every run against ``--ref``.
All runs share one tokenizer (pass the base's) and identical batches, so items line up.

Tasks (all prompts in Qwen's chat template with thinking off, except the ARC/HellaSwag contexts):
  mmlu       cais/mmlu, 1000 seeded test questions, 0-shot; prediction = argmax over A/B/C/D
             logits at the first answer token
  arc        ARC-Challenge test (1172), continuation log-likelihood, acc and acc_norm
  hellaswag  1000 seeded validation items, continuation log-likelihood, acc_norm
  gsm8k      200 seeded test problems, greedy, no tools, final answer in \\boxed{}
  ifeval     google/IFEval (541), greedy, scored by lm-eval's checker (strict and loose)
  niah       needle (a 7-digit "magic number") in a wikitext-103 haystack at 8k/16k/32k tokens,
             chunked prefill; greedy answer and the teacher-forced log-prob of the needle
  ppl        8 disjoint 32k-token wikitext-103 windows: per-token NLL by position and greedy
             next-token agreement (a dense, long-context fidelity measure)

The noise floor is the base model under another attention implementation (eager vs sdpa), which
changes nothing but the rounding inside Qwen's full-attention layers.
"""

import argparse
import itertools
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

TASKS = ("mmlu", "arc", "hellaswag", "gsm8k", "ifeval", "niah", "ppl")
TEXT_VOCAB = 248320  # Qwen's own vocabulary; the merged checkpoint's chip tokens come after it
TOKEN_BUDGET = 4096  # tokens per log-likelihood batch
GEN_BATCH = 16
CHUNK = 1024  # prefill chunk for the long-context check, same for every model


# --- model ------------------------------------------------------------------------------------


def load(path, attn, mem_gb):
    from transformers import AutoModelForImageTextToText

    if mem_gb:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, mem_gb / total))
    model = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation=attn)
    del model.model.visual  # text only; frees ~0.7 GB
    return model.to("cuda").eval()


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                   return_tensors="pt", enable_thinking=False)["input_ids"][0].tolist()


def batches_by_budget(lengths, budget=TOKEN_BUDGET, max_b=64):
    """Indices sorted by length, grouped so that batch * max_len <= budget."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[cur[0]] > budget or len(cur) == max_b:
            out.append(cur)
            cur = []
        cur.append(i)
    return out + [cur] if cur else out


@torch.no_grad()
def target_logprobs(model, seqs, queries, pad):
    """For each sequence, log-probs (full vocabulary) of chosen tokens at chosen positions.

    ``queries[i]`` is a list of (position, [token ids]); returns, per sequence, a list of
    (logprobs of those ids, argmax id) per query. Right padding, so real positions are exact.
    """
    out = [None] * len(seqs)
    for b in batches_by_budget([len(s) for s in seqs]):
        n = max(len(seqs[i]) for i in b)
        ids = torch.full((len(b), n), pad, dtype=torch.long)
        mask = torch.zeros((len(b), n), dtype=torch.long)
        for r, i in enumerate(b):
            ids[r, :len(seqs[i])] = torch.tensor(seqs[i])
            mask[r, :len(seqs[i])] = 1
        h = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(),
                        use_cache=False).last_hidden_state
        rows = [(r, p, toks) for r, i in enumerate(b) for p, toks in queries[i]]
        sel = h[torch.tensor([r for r, _, _ in rows]), torch.tensor([p for _, p, _ in rows])]
        vals, tops = [], []
        for c in range(0, len(rows), 256):  # bounded memory for the vocabulary-wide softmax
            lp = torch.log_softmax(model.lm_head(sel[c:c + 256]).float(), -1)
            tops += lp.argmax(-1).tolist()
            rr = [k - c for k in range(c, min(c + 256, len(rows))) for _ in rows[k][2]]
            tt = [t for k in range(c, min(c + 256, len(rows))) for t in rows[k][2]]
            vals += lp[rr, tt].tolist()
        k = v = 0
        for i in b:
            res = []
            for _, toks in queries[i]:
                res.append((vals[v:v + len(toks)], tops[k]))
                k, v = k + 1, v + len(toks)
            out[i] = res
    return out


@torch.no_grad()
def generate(model, prompts, max_new, eos, pad, batch=GEN_BATCH):
    """Greedy, batched with left padding; batches of similar prompt length, same for every
    model. Returns the new tokens of each prompt, cut after the first EOS."""
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    out = [None] * len(prompts)
    for s in range(0, len(order), batch):
        b = order[s:s + batch]
        n = max(len(prompts[i]) for i in b)
        ids = torch.full((len(b), n), pad, dtype=torch.long)
        mask = torch.zeros((len(b), n), dtype=torch.long)
        for r, i in enumerate(b):
            ids[r, n - len(prompts[i]):] = torch.tensor(prompts[i])
            mask[r, n - len(prompts[i]):] = 1
        model.model.rope_deltas = None
        g = model.generate(input_ids=ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=max_new,
                           do_sample=False, temperature=None, top_p=None, top_k=None,
                           eos_token_id=eos, pad_token_id=pad)[:, n:].tolist()
        for r, i in enumerate(b):
            toks = g[r]
            cut = next((j for j, t in enumerate(toks) if t in eos), None)
            out[i] = toks if cut is None else toks[:cut + 1]
        print(f"  gen {s + len(b)}/{len(prompts)}", flush=True)
    return out


def gen_record(tok, toks, eos):
    text = tok.decode([t for t in toks if t < TEXT_VOCAB], skip_special_tokens=True)
    return {"text": text, "tokens": toks, "n": len(toks), "finished": bool(toks) and toks[-1] in eos,
            "chip_tokens": sum(t >= TEXT_VOCAB for t in toks)}


# --- tasks ------------------------------------------------------------------------------------


def task_mmlu(model, tok, ctx, n=1000, seed=0):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = sorted(random.Random(seed).sample(range(len(ds)), n))
    letters = [tok.encode(c, add_special_tokens=False) for c in "ABCD"]
    assert all(len(x) == 1 for x in letters)
    letters = [x[0] for x in letters]
    seqs, items = [], []
    for i in idx:
        d = ds[i]
        q = (f"The following is a multiple choice question about {d['subject'].replace('_', ' ')}."
             f"\n\n{d['question'].strip()}\n"
             + "".join(f"{c}. {t}\n" for c, t in zip("ABCD", d["choices"]))
             + "\nAnswer with the letter of the correct option only.")
        seqs.append(chat(tok, q))
        items.append({"idx": i, "subject": d["subject"], "gold": d["answer"]})
    res = target_logprobs(model, seqs, [[(len(s) - 1, letters)] for s in seqs], ctx["pad"])
    for it, ((lp, top),) in zip(items, res):
        it.update(lp=lp, pred=int(np.argmax(lp)), top_is_letter=top in letters)
        it["correct"] = it["pred"] == it["gold"]
    return items


def choice_items(model, tok, ctx, docs):
    """docs: (context string, [continuation strings], gold). Continuation log-likelihoods."""
    seqs, queries, meta = [], [], []
    for c, conts, gold in docs:
        cids = tok.encode(c, add_special_tokens=False)
        meta.append((gold, [len(x) for x in conts]))
        for x in conts:
            xids = tok.encode(x, add_special_tokens=False)
            seqs.append(cids + xids)
            queries.append([(len(cids) - 1 + j, [t]) for j, t in enumerate(xids)])
    res = target_logprobs(model, seqs, queries, ctx["pad"])
    items, k = [], 0
    for gold, chars in meta:
        lp, greedy = [], []
        for _ in chars:
            lp.append(sum(r[0][0] for r in res[k]))
            greedy.append(all(r[1] == q[1][0] for r, q in zip(res[k], queries[k])))
            k += 1
        norm = [v / n for v, n in zip(lp, chars)]
        items.append({"gold": gold, "lp": lp, "pred": int(np.argmax(lp)),
                      "pred_norm": int(np.argmax(norm)), "greedy": greedy})
        items[-1]["correct"] = items[-1]["pred"] == gold
        items[-1]["correct_norm"] = items[-1]["pred_norm"] == gold
    return items


def task_arc(model, tok, ctx):
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    docs = [(f"Question: {d['question']}\nAnswer:", [" " + t for t in d["choices"]["text"]],
             d["choices"]["label"].index(d["answerKey"])) for d in ds]
    items = choice_items(model, tok, ctx, docs)
    for d, it in zip(ds, items):
        it["id"] = d["id"]
    return items


def _hs(text):
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def task_hellaswag(model, tok, ctx, n=1000, seed=0):
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    idx = sorted(random.Random(seed).sample(range(len(ds)), n))
    docs = []
    for i in idx:
        d = ds[i]
        c = _hs(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize())
        docs.append((c, [" " + _hs(e) for e in d["endings"]], int(d["label"])))
    items = choice_items(model, tok, ctx, docs)
    for i, it in zip(idx, items):
        it["idx"] = i
    return items


def gsm_answer(text):
    boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    s = boxed[-1] if boxed else text
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", s.replace("$", ""))
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same_number(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def task_gsm8k(model, tok, ctx, n=200, seed=0, max_new=1024):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = sorted(random.Random(seed).sample(range(len(ds)), n))
    prompts = [chat(tok, ds[i]["question"] + "\nPlease reason step by step, and put your final "
                    "answer within \\boxed{}.") for i in idx]
    gens = generate(model, prompts, max_new, ctx["eos"], ctx["pad"])
    items = []
    for i, g in zip(idx, gens):
        gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
        r = gen_record(tok, g, ctx["eos"])
        r.update(idx=i, gold=gold, pred=gsm_answer(r["text"]))
        r["correct"] = same_number(r["pred"], gold)
        items.append(r)
    return items


def ifeval_utils():
    """lm-eval's IFEval checker, imported without lm_eval/__init__ (which pulls in the whole
    harness). Needs langdetect, immutabledict, nltk (punkt_tab)."""
    import importlib.util
    import types

    root = Path(importlib.util.find_spec("lm_eval").submodule_search_locations[0])
    for name, p in (("lm_eval", root), ("lm_eval.tasks", root / "tasks")):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [str(p)]
            sys.modules[name] = m
    from lm_eval.tasks.ifeval import utils

    return utils


def task_ifeval(model, tok, ctx, max_new=1280):
    from datasets import load_dataset

    utils = ifeval_utils()
    ds = load_dataset("google/IFEval", split="train")
    gens = generate(model, [chat(tok, d["prompt"]) for d in ds], max_new, ctx["eos"], ctx["pad"])
    items = []
    for d, g in zip(ds, gens):
        r = gen_record(tok, g, ctx["eos"])
        s = utils.process_results(d, [r["text"]])
        r.update(key=d["key"], strict=s["prompt_level_strict_acc"],
                 loose=s["prompt_level_loose_acc"], inst_strict=s["inst_level_strict_acc"],
                 inst_loose=s["inst_level_loose_acc"])
        r["correct"] = r["strict"]
        items.append(r)
    return items


CITIES = ["Lisbon", "Nairobi", "Osaka", "Quito", "Tbilisi", "Winnipeg", "Hanoi", "Bergen",
          "Cusco", "Dakar", "Perth", "Tallinn"]


def haystack_ids(tok):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "".join(t for t in ds["text"] if t.strip() and not t.strip().startswith("="))
    return tok.encode(text, add_special_tokens=False)


@torch.no_grad()
def prefill(model, ids, keep=1):
    """Chunked prefill with a fresh cache; logits (text vocab) of the last ``keep`` positions."""
    from transformers import DynamicCache

    model.model.rope_deltas = None  # stale after a batched generate(); text-only deltas are 0
    cache = DynamicCache(config=model.config)
    n = len(ids) - keep
    cuts = list(range(0, n, CHUNK)) + [n, len(ids)]
    for a, b in itertools.pairwise(cuts):
        x = torch.tensor([ids[a:b]], device="cuda")
        out = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=keep)
    return out.logits[0, :, :TEXT_VOCAB].float(), cache


@torch.no_grad()
def greedy_from(model, logits, cache, n, eos):
    toks = []
    t = int(logits[-1].argmax())
    for _ in range(n):
        toks.append(t)
        if t in eos:
            break
        out = model(input_ids=torch.tensor([[t]], device="cuda"), past_key_values=cache,
                    use_cache=True)
        t = int(out.logits[0, -1].argmax())  # full vocabulary: a chip token would show up
    return toks


def task_niah(model, tok, ctx, lengths=(8192, 16384, 32768), depths=(0.1, 0.3, 0.5, 0.7, 0.9),
              seeds=4):
    hay = haystack_ids(tok)
    items = []
    for L in lengths:
        for depth in depths:
            for s in range(seeds):
                rng = random.Random(L * 100 + int(depth * 10) * 10 + s)
                city, num = rng.choice(CITIES), str(rng.randrange(10**6, 10**7))
                question = (f"What is the special magic number for {city} mentioned in the text? "
                            "Answer with the number only.")
                head = "Read the text below and answer the question that follows.\n\n"
                over = len(chat(tok, head + "\n\n" + question)) + 20
                off = rng.randrange(0, len(hay) - L)
                body = hay[off:off + L - over]
                p = int(depth * len(body))
                left, right = tok.decode(body[:p]), tok.decode(body[p:])
                cut = left.rfind(". ")
                if cut > len(left) - 2000:  # put the needle at a sentence boundary
                    left, right = left[:cut + 2], left[cut + 2:] + right
                needle = f"The special magic number for {city} is {num}. "
                ids = chat(tok, head + left + needle + right + "\n\n" + question)
                ans = tok.encode(num, add_special_tokens=False)
                logits, _ = prefill(model, ids + ans, keep=len(ans) + 1)
                lp = torch.log_softmax(logits, -1)
                gold_lp = float(sum(lp[j, t] for j, t in enumerate(ans)))
                logits, cache = prefill(model, ids)
                g = greedy_from(model, logits, cache, 16, ctx["eos"])
                del cache
                r = gen_record(tok, g, ctx["eos"])
                r.update(length=len(ids), target=L, depth=depth, seed=s, city=city, gold=num,
                         gold_logprob=gold_lp, correct=num in r["text"])
                items.append(r)
                print(f"  niah L={len(ids)} d={depth} s={s} ok={r['correct']} "
                      f"lp={gold_lp:.3f} {r['text']!r}", flush=True)
    return items


@torch.no_grad()
def task_ppl(model, tok, ctx, length=32768, docs=8):
    """Long-context language modelling: per-token NLL and greedy next token over ``docs``
    disjoint ``length``-token windows of wikitext-103 test, chunked prefill."""
    from transformers import DynamicCache

    hay = haystack_ids(tok)
    items = []
    for d in range(docs):
        ids = hay[d * length:(d + 1) * length]
        model.model.rope_deltas = None
        cache = DynamicCache(config=model.config)
        nll, top = [], []
        for a in range(0, length, CHUNK):
            x = torch.tensor([ids[a:a + CHUNK]], device="cuda")
            h = model.model(input_ids=x, past_key_values=cache, use_cache=True).last_hidden_state[0]
            for c in range(0, h.shape[0], 256):
                lp = torch.log_softmax(model.lm_head(h[c:c + 256]).float(), -1)
                p0 = a + c
                tgt = torch.tensor(ids[p0 + 1:p0 + 1 + lp.shape[0]], device="cuda")
                nll += (-lp[torch.arange(len(tgt)), tgt]).tolist()
                top += lp.argmax(-1)[:len(tgt)].tolist()
        del cache
        items.append({"offset": d * length, "nll": [round(v, 5) for v in nll], "top1": top,
                      "target": ids[1:], "correct": float(np.mean(np.array(top) == ids[1:]))})
        print(f"  ppl doc {d}: ppl {math.exp(np.mean(nll)):.3f} top1 {items[-1]['correct']:.4f}",
              flush=True)
    return items


PPL_BUCKETS = ((0, 2048), (2048, 8192), (8192, 16384), (16384, 32768))


def ppl_table(data, names, ref, summary):
    """Rows for the long-context perplexity task: PPL per position bucket, signed mean dNLL vs
    ``ref`` with a bootstrap CI over 1024-token blocks, |dNLL|, greedy next-token agreement."""
    lines = []
    for lo, hi in PPL_BUCKETS:
        cells = []
        for n in names:
            if "ppl" not in data[n]:
                cells.append("-")
                continue
            v = np.concatenate([np.array(it["nll"][lo:hi]) for it in data[n]["ppl"]])
            s = {"ppl": float(np.exp(v.mean())), "tokens": len(v)}
            cell = f"{s['ppl']:.3f}"
            if n != ref:
                r = [np.array(it["nll"][lo:hi]) for it in data[ref]["ppl"]]
                m = [np.array(it["nll"][lo:hi]) for it in data[n]["ppl"]]
                blocks = np.array([(y[k:k + 1024] - x[k:k + 1024]).mean() for x, y in zip(r, m)
                                   for k in range(0, len(x), 1024)])
                boot = np.random.default_rng(0).choice(blocks, (10000, len(blocks))).mean(1)
                dl, dh = np.percentile(boot, [2.5, 97.5])
                d = np.concatenate([y - x for x, y in zip(r, m)])
                agree = np.concatenate([np.array(a["top1"][lo:hi]) == np.array(b["top1"][lo:hi])
                                        for a, b in zip(data[ref]["ppl"], data[n]["ppl"])])
                s.update(dnll=float(d.mean()), dnll_ci95=[float(dl), float(dh)],
                         abs_dnll=float(np.abs(d).mean()), top1_agreement=float(agree.mean()))
                cell += (f"<br>dNLL {d.mean():+.2e} [{dl:+.1e}, {dh:+.1e}], mean abs dNLL "
                         f"{np.abs(d).mean():.2e}, top-1 agree {100 * agree.mean():.2f}%")
            summary.setdefault(n, {}).setdefault("ppl", {})[f"{lo}-{hi}"] = s
            cells.append(cell)
        lines.append(f"| ppl | positions {lo}-{hi} | " + " | ".join(cells) + " |")
    return lines


def stop_tokens(tok):
    """End of turn and end of text stop generation (as in the merged generation_config.json);
    the base snapshot ships no generation config."""
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    return {"eos": eos, "pad": eos[1]}


# --- run --------------------------------------------------------------------------------------


def run(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    ctx = stop_tokens(tok)
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    model = None
    for task in args.tasks.split(","):
        f = out / f"{task}.json"
        if f.exists():
            print(f"{args.name}/{task}: done already", flush=True)
            continue
        if model is None:
            model = load(args.model, args.attn, args.mem_gb)
        print(f"{args.name}/{task} ...", flush=True)
        t = time.time()
        torch.cuda.reset_peak_memory_stats()
        items = globals()[f"task_{task}"](model, tok, ctx)
        rec = {"model": args.model, "attn": args.attn, "task": task, "seconds": time.time() - t,
               "peak_gb": torch.cuda.max_memory_allocated() / 2**30, "items": items}
        f.write_text(json.dumps(rec, ensure_ascii=False))
        acc = np.mean([it["correct"] for it in items])
        print(f"{args.name}/{task}: acc {acc:.4f} n={len(items)} {rec['seconds']:.0f}s "
              f"peak {rec['peak_gb']:.1f} GB", flush=True)


# --- statistics -------------------------------------------------------------------------------


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def paired_diff(a, b, n_boot=10000, seed=0):
    """Mean of a - b with a paired bootstrap 95% CI, and the exact McNemar p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    d = (a[idx] - b[idx]).mean(1)
    n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    m = n01 + n10
    p = min(1.0, 2 * sum(math.comb(m, k) for k in range(min(n01, n10) + 1)) / 2**m) if m else 1.0
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float((a - b).mean()), float(lo), float(hi), p


def choice_kl(p_lp, q_lp):
    """KL(p || q) between the normalised choice distributions of two runs, per item."""
    p = torch.log_softmax(torch.tensor(p_lp, dtype=torch.float64), -1)
    q = torch.log_softmax(torch.tensor(q_lp, dtype=torch.float64), -1)
    return float((p.exp() * (p - q)).sum())


def first_divergence(x, y):
    return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q),
                None if len(x) == len(y) else min(len(x), len(y)))


METRICS = {  # task -> (metric name, item key) list; the first is the headline
    "mmlu": [("acc", "correct")],
    "arc": [("acc_norm", "correct_norm"), ("acc", "correct")],
    "hellaswag": [("acc_norm", "correct_norm"), ("acc", "correct")],
    "gsm8k": [("acc", "correct")],
    "ifeval": [("prompt_strict", "strict"), ("prompt_loose", "loose")],
    "niah": [("acc", "correct")],
}


def compare_items(task, ref, run):
    """Fine-grained agreement of ``run`` with ``ref`` beyond accuracy."""
    out = {}
    if task in ("mmlu", "arc", "hellaswag"):
        out["same_pred"] = [a["pred"] == b["pred"] for a, b in zip(ref, run)]
        out["mean_choice_kl"] = float(np.mean([choice_kl(a["lp"], b["lp"])
                                               for a, b in zip(ref, run)]))
        out["mean_abs_dlogprob"] = float(np.mean([abs(x - y) for a, b in zip(ref, run)
                                                  for x, y in zip(a["lp"], b["lp"])]))
    elif task == "niah":
        out["same_pred"] = [a["correct"] == b["correct"] for a, b in zip(ref, run)]
        out["mean_abs_dlogprob"] = float(np.mean([abs(a["gold_logprob"] - b["gold_logprob"])
                                                  for a, b in zip(ref, run)]))
        out["identical_text"] = float(np.mean([a["tokens"] == b["tokens"] for a, b in zip(ref, run)]))
    else:
        key = "pred" if task == "gsm8k" else "strict"
        out["same_pred"] = [a[key] == b[key] for a, b in zip(ref, run)]
        out["identical_text"] = float(np.mean([a["tokens"] == b["tokens"] for a, b in zip(ref, run)]))
        fd = [first_divergence(a["tokens"], b["tokens"]) for a, b in zip(ref, run)]
        fd = [x for x in fd if x is not None]
        out["median_first_divergence"] = float(np.median(fd)) if fd else None
        out["unfinished"] = int(sum(not b["finished"] for b in run))
        out["chip_tokens"] = int(sum(b["chip_tokens"] for b in run))
    return out


def table(args):
    root = Path(args.dir)
    names = [args.ref] + [p.name for p in sorted(root.iterdir())
                          if p.is_dir() and p.name != args.ref]
    data = {n: {f.stem: json.loads(f.read_text(encoding="utf-8"))["items"]
                for f in (root / n).glob("*.json")} for n in names}
    summary, lines = {}, []
    lines.append("| task | metric | " + " | ".join(names) + " |")
    lines.append("|---|---|" + "---|" * len(names))
    for task in TASKS:
        if task not in data[args.ref] or task == "ppl":
            continue
        ref = data[args.ref][task]
        for metric, key in METRICS[task]:
            cells = []
            for n in names:
                if task not in data[n]:
                    cells.append("-")
                    continue
                v = [bool(it[key]) for it in data[n][task]]
                lo, hi = wilson(sum(v), len(v))
                s = {"acc": float(np.mean(v)), "ci95": [lo, hi], "n": len(v)}
                cell = f"{100 * s['acc']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"
                if n != args.ref:
                    d, dlo, dhi, p = paired_diff(v, [bool(it[key]) for it in ref])
                    s.update(diff=d, diff_ci95=[dlo, dhi], mcnemar_p=p)
                    cell += f"<br>diff {100 * d:+.1f} [{100 * dlo:+.1f}, {100 * dhi:+.1f}] p={p:.2f}"
                summary.setdefault(n, {}).setdefault(task, {})[metric] = s
                cells.append(cell)
            lines.append(f"| {task} | {metric} (n={len(ref)}) | " + " | ".join(cells) + " |")
        cells = []
        for n in names:
            if n == args.ref or task not in data[n]:
                cells.append("ref" if n == args.ref else "-")
                continue
            c = compare_items(task, ref, data[n][task])
            same = c.pop("same_pred")
            lo, hi = wilson(sum(same), len(same))
            c["agreement"] = {"frac": float(np.mean(same)), "ci95": [lo, hi]}
            summary[n][task]["vs_ref"] = c
            extra = ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
                              for k, v in c.items() if k != "agreement")
            cells.append(f"{100 * np.mean(same):.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]<br>{extra}")
        lines.append(f"| {task} | agreement with {args.ref} | " + " | ".join(cells) + " |")
    if "ppl" in data[args.ref]:
        lines += ppl_table(data, names, args.ref, summary)
    md = "\n".join(lines)
    print(md)
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    (root / "table.md").write_text(md + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument("--name", required=True)
    r.add_argument("--attn", default="sdpa")
    r.add_argument("--tokenizer", default=None, help="shared tokenizer (the base checkpoint)")
    r.add_argument("--tasks", default=",".join(TASKS))
    r.add_argument("--out", default="dist/evals/text")
    r.add_argument("--mem-gb", type=float, default=20.0, help="per-process GPU memory cap")
    t = sub.add_parser("table")
    t.add_argument("dir", nargs="?", default="dist/evals/text")
    t.add_argument("--ref", default="base")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else table(args)


if __name__ == "__main__":
    main()
