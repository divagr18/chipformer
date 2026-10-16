"""Check a merged Qwen3.5 + CHIP-8 checkpoint against its base model.

    python -m chipformer.q35.report BASE MERGED [--device cuda] [--out report.json]

1. CHIP-8: teacher-forced predictions on game traces, pure and inside a chat (mixed context).
2. Leakage: the largest chip-token logit at text positions.
3. Text: greedy generations and logits vs the base model, next to the noise floor between two
   attention implementations of the base model itself.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from chipformer.check import run_machine

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = [
    ("chat", "Explain why the sky is blue in two sentences."),
    ("raw", 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n'),
    ("chat", "What is 37 * 43? Think step by step."),
    ("chat", "Traduis en français : The weather is lovely today and I want to go for a walk."),
    ("chat", "写一首关于秋天的短诗。"),
    ("chat", "List three differences between TCP and UDP."),
    ("raw", "The CHIP-8 is a virtual machine from the 1970s that"),
    ("raw", "Q: If a train leaves at 3pm travelling 60 km/h, how far has it gone by 5:30pm?\nA:"),
    ("chat", "Write a haiku about a lighthouse."),
    ("chat", "Give me a Python one-liner that reverses the words in a sentence."),
    ("chat", "What causes the seasons on Earth?"),
    ("raw", "In 1969, Apollo 11"),
    ("chat", "Summarise the plot of Romeo and Juliet in three sentences."),
    ("chat", "Solve for x: 3x + 7 = 22."),
    ("raw", "SELECT name, COUNT(*) FROM orders"),
    ("chat", "¿Cuál es la capital de Australia y por qué no es Sídney?"),
]


def game_trace(gid: str, steps: int, seed: int = 2):
    games = json.loads((ROOT / "games" / "manifest.json").read_text(encoding="utf-8"))["games"]
    g = next(g for g in games if g["id"] == gid)
    rom = (ROOT / "games" / g["file"]).read_bytes()
    return run_machine(rom, steps, ipf=g["ipf"], seed=seed, key_rate=0.3, quirks=tuple(g["quirks"]))


def load(path, device, impl="sdpa"):
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation=impl).to(device).eval()


def encode(tok, kind, text):
    if kind == "chat":
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       add_generation_prompt=True, return_tensors="pt",
                                       enable_thinking=False)["input_ids"][0].tolist()
    return tok(text).input_ids


@torch.no_grad()
def text_run(model, prompts, device, n_new):
    gens, logits = [], []
    for ids in prompts:
        x = torch.tensor([ids], device=device)
        g = model.generate(x, max_new_tokens=n_new, do_sample=False)
        gens.append(g[0, len(ids):].tolist())
        logits.append(model(input_ids=g).logits[0, :, :248320].float().cpu())
    return gens, logits


def compare(a, b):
    ga, la = a
    gb, lb = b
    same = [x == y for x, y in zip(ga, gb)]
    first = [next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
             for x, y in zip(ga, gb)]
    dmax, kl, top1 = [], [], []
    for x, y, s in zip(la, lb, same):
        if not s:
            continue
        dmax.append(float((x - y).abs().max()))
        p, q = torch.log_softmax(x, -1), torch.log_softmax(y, -1)
        kl.append(float((p.exp() * (p - q)).sum(-1).mean()))
        top1.append(float((x.argmax(-1) == y.argmax(-1)).float().mean()))
    return {"identical_generations": f"{sum(same)}/{len(same)}", "first_divergence": first,
            "max_abs_dlogit": max(dmax, default=None), "mean_kl": float(np.mean(kl)) if kl else None,
            "top1_agreement": float(np.mean(top1)) if top1 else None}


@torch.no_grad()
def chip_check(model, off, tokens, generated, device, prefix=(), suffix=()):
    ids = list(prefix) + [t + off for t in tokens] + list(suffix)
    logits = model(input_ids=torch.tensor([ids], device=device)).logits[0].float().cpu()
    s = len(prefix)
    pred = logits[s : s + len(tokens)].argmax(-1).numpy() - off
    want = np.array(tokens[1:] + [-1])
    gen = np.array(generated[1:] + [False])
    text_pos = list(range(s)) + list(range(s + len(tokens), len(ids)))
    leak = float(logits[text_pos, off:off + 8708].max()) if text_pos else None
    return {"checked": int(gen.sum()), "mismatches": int((gen & (pred != want)).sum()),
            "min_predicted_logit": float(logits[s : s + len(tokens)].max(-1).values[gen].min()),
            "max_chip_logit_at_text": leak}


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("merged")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    meta = json.loads((Path(args.merged) / "chipformer.json").read_text())
    off = meta["chip_token_offset"]
    tok = AutoTokenizer.from_pretrained(args.merged)
    prompts = [encode(tok, k, t) for k, t in PROMPTS]
    rep = {"meta": meta}

    t = time.time()
    model = load(args.merged, args.device)
    rep["params_B"] = sum(p.numel() for p in model.parameters()) / 1e9
    for gid in ("brix", "tetris", "space-invaders"):
        try:
            m = game_trace(gid, 400)
        except StopIteration:
            continue
        rep[f"chip_{gid}"] = chip_check(model, off, m.tokens, m.generated, args.device)
    m = game_trace("tetris", 200, seed=5)
    pre = encode(tok, "chat", "Run this CHIP-8 program for me.")
    post = tok("\nThe machine is running Tetris now. What should I do next?",
               add_special_tokens=False).input_ids
    rep["chip_mixed"] = chip_check(model, off, m.tokens, m.generated, args.device, pre, post)
    merged = text_run(model, prompts, args.device, args.new)
    rep["sample_merged"] = tok.decode(merged[0][2])
    del model
    base = load(args.base, args.device)
    ref = text_run(base, prompts, args.device, args.new)
    rep["sample_base"] = tok.decode(ref[0][2])
    del base
    eager = text_run(load(args.base, args.device, "eager"), prompts, args.device, args.new)
    rep["noise_floor_sdpa_vs_eager"] = compare(ref, eager)
    rep["base_vs_merged"] = compare(ref, merged)
    rep["seconds"] = time.time() - t
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
