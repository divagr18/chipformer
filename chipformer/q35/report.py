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


@torch.no_grad()
def post_trace(model, tok, off, device, lengths=(0, 2000, 10000)):
    """Does Qwen remember the conversation after a CHIP-8 run in the middle of it?

    Compares the merged model on [prompt, trace, question] with itself on [prompt, question]
    where the question keeps the positions it has after the trace, so only the trace's content
    differs. Reports KL on the answer's next-token distributions and both greedy answers.
    """
    pre = tok.apply_chat_template(
        [{"role": "user", "content": "Remember this: my locker code is 4817 and my cat is called "
          "Biscuit. I'll ask about it later. First, run this CHIP-8 program."},
         {"role": "assistant", "content": "Sure, running it now."}],
        return_tensors="pt", enable_thinking=False)["input_ids"][0].tolist()
    q = tok.apply_chat_template(
        [{"role": "user", "content": "What is my locker code and what is my cat called?"}],
        add_generation_prompt=True, return_tensors="pt", enable_thinking=False)["input_ids"][0]
    q = q.tolist()
    out = {}
    m = game_trace("tetris", 4000, seed=7)
    for n in lengths:
        trace = [t + off for t in m.tokens[:n]]
        if trace and m.tokens[n - 1] != 8705:  # end on a STEP
            last = max(i for i in range(n) if m.tokens[i] == 8705)
            trace = trace[: last + 1]
        full = torch.tensor([pre + trace + q], device=device)
        gap = torch.tensor([pre + q], device=device)
        pos = torch.cat([torch.arange(len(pre)), torch.arange(len(q)) + len(pre) + len(trace)])
        a = model.generate(full, max_new_tokens=24, do_sample=False)[0, full.shape[1]:]
        b = model.generate(gap, position_ids=pos[None].to(device), max_new_tokens=24,
                           do_sample=False)[0, gap.shape[1]:] if trace else a
        la = model(input_ids=torch.cat([full, a[None]], 1)).logits[0, -len(a) - 1:-1].float()
        pos_b = torch.cat([pos, torch.arange(len(a)) + pos[-1] + 1])[None].to(device)
        lb = model(input_ids=torch.cat([gap, a[None]], 1), position_ids=pos_b).logits[
            0, -len(a) - 1:-1].float()
        p, r = torch.log_softmax(la[:, :248320], -1), torch.log_softmax(lb[:, :248320], -1)
        out[n] = {"trace_tokens": len(trace), "kl": float((p.exp() * (p - r)).sum(-1).mean()),
                  "answer_with_trace": tok.decode(a, skip_special_tokens=True),
                  "answer_without": tok.decode(b, skip_special_tokens=True)}
    return out


PRINT_ROM = bytes([0xA2, 0x0A, 0x60, 4, 0x00, 0xFD, 0x12, 0x06, 0, 0, 4, 8, 1, 7])  # prints 4817


@torch.no_grad()
def bridge_demo(model, tok, meta, device, rom: bytes = PRINT_ROM, n_new: int = 120):
    """One generate(): a chat turn hands over to a CHIP-8 program, which runs, prints its result
    as text, halts, and Qwen continues. Checks the machine part against the reference."""
    from chipformer.q35 import bridge
    from chipformer.ref.emu import Chip8
    from chipformer.trace import boot_tokens

    off = meta["chip_token_offset"]
    text_ids = {int(k): v for k, v in meta["text_ids"].items()}
    to_id = {**text_ids}
    pre = tok.apply_chat_template(
        [{"role": "user", "content": "Run the program and tell me the number it prints, "
          "then that number plus one."}],
        add_generation_prompt=True, return_tensors="pt", enable_thinking=False)["input_ids"][0]
    pre = pre.tolist() + tok("Running it now.\n", add_special_tokens=False).input_ids
    boot = boot_tokens(Chip8(rom).cells)
    ids = pre + [t + off for t in boot]
    out = model.generate(torch.tensor([ids], device=device), max_new_tokens=n_new,
                         do_sample=False)[0, len(ids):].tolist()
    # reference: the machine records and the print, as model token ids
    bm = bridge.BridgeMachine(rom)
    while not bm.halted:
        bm.step()
    want = [to_id[t] if bridge.is_text(t) else t + off for t in bm.tokens[len(boot):]]
    got_machine = out[:len(want)]
    return {"machine_and_print_exact": got_machine == want,
            "printed": tok.decode(out[:len(want)][-(len(want) - want.index(text_ids[bridge.NL])):]),
            "continuation": tok.decode(out[len(want):], skip_special_tokens=True),
            "n_machine_tokens": len(want)}


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
    rep["post_trace"] = post_trace(model, tok, off, args.device)
    if meta.get("layout") == "v2":
        rep["bridge"] = bridge_demo(model, tok, meta, args.device)
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
