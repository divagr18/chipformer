"""Qwen uses the computer in its weights: exact arithmetic through its own tool calls.

    python -m chipformer.q35.demo MERGED [--base BASE] [--n 10] [--inline]

The prompt starts with the BIOS (bcd.py's routines, as a boot record of chip tokens, which
Qwen's layers ignore) and declares ``add`` / ``mul`` tools in Qwen's usual chat template. When
Qwen calls one, the circuit takes over at </tool_call>: it writes the arguments into the
machine's memory, runs the routine, prints the result inside <tool_response> and hands back.
There is no host program and no tool executor: one ``generate()`` call does everything.

With --inline no tools are declared: wherever Qwen writes "A op B =" in its answer, the circuit
takes over at "=" and prints the result as plain text.
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch

TOOLS = [{"type": "function", "function": {
    "name": name, "description": desc,
    "parameters": {"type": "object", "required": ["a", "b"], "properties": {
        "a": {"type": "integer", "description": "first operand"},
        "b": {"type": "integer", "description": "second operand"}}}}}
    for name, desc in [("mul", ("Multiply two non-negative integers exactly. Use it for any "
                               "multiplication.")),
                       ("add", "Add two non-negative integers exactly. Use it for any addition."),
                       ("sub", ("Subtract b from a exactly (a must be >= b). Use it for any "
                               "subtraction.")),
                       ("div", ("Divide a by b exactly: the quotient, or its first 10 decimals "
                                "followed by ... when it does not end. Use it for any "
                                "division."))]]


def problems(n: int, seed: int = 0, hard: bool = False) -> list[tuple[str, str, int]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        op = rng.choice(["add", "mul", "mul"] if hard else ["add", "mul"])
        if hard:
            a, b = rng.randrange(10**5, 10**rng.randint(6, 10)), rng.randrange(10**3, 10**rng.randint(4, 7))
        else:
            a, b = rng.randrange(10**rng.randint(3, 7)), rng.randrange(10**rng.randint(2, 6))
        sym = "+" if op == "add" else "*"
        q = rng.choice(["What is {a} {s} {b}?", "Compute {a} {s} {b}.",
                        "I need {a} {s} {b} exactly for an invoice."]).format(a=a, b=b, s=sym)
        out.append((q, op, a + b if op == "add" else a * b))
    return out


def load(path):
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    return model, tok


NUDGE = ("Exact arithmetic: whenever you write a calculation like `a \\times b =`, `a + b =` "
         "or `a - b =` with whole numbers, its exact result is filled in right after the `=`. "
         "So never do long multiplication or addition by hand: write the whole calculation and "
         "let the result appear.")


@torch.no_grad()
def ask(model, tok, question: str, prefix: list[int], tools, max_new: int, system=None) -> str:
    msgs = [{"role": "system", "content": system}] if system else []
    ids = tok.apply_chat_template(msgs + [{"role": "user", "content": question}], tools=tools,
                                  add_generation_prompt=True, return_tensors="pt",
                                  enable_thinking=False)["input_ids"][0].tolist()
    x = torch.tensor([prefix + ids], device="cuda")
    out = model.generate(x, max_new_tokens=max_new, do_sample=False,
                         cache_implementation="static")
    return out[0, x.shape[1]:].tolist()


def answer_number(text: str):
    nums = re.findall(r"\d[\d,]*", text.replace("**", ""))
    return int(nums[-1].replace(",", "")) if nums else None


def main() -> None:
    from chipformer.q35.lib import bcd
    from chipformer.ref.emu import Chip8
    from chipformer.trace import boot_tokens

    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("--base")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=4000)
    ap.add_argument("--out")
    ap.add_argument("--inline", action="store_true", help="no tools: inline arithmetic only")
    ap.add_argument("--nudge", action="store_true", help="system prompt: write whole calculations")
    ap.add_argument("--hard", action="store_true", help="6-10 digit x 4-7 digit problems")
    args = ap.parse_args()
    meta = json.loads((Path(args.merged) / "chipformer.json").read_text())
    off = meta["chip_token_offset"]
    bios = [t + off for t in boot_tokens(Chip8(bcd.ROM, bcd.QUIRKS).cells)]
    probs = problems(args.n, hard=args.hard)
    rep = {"merged": [], "base": []}
    model, tok = load(args.merged)
    for q, op, want in probs:
        t = time.time()
        out = ask(model, tok, q, bios, None if args.inline else TOOLS, args.max_new,
                  NUDGE if args.nudge else None)
        text = tok.decode(out, skip_special_tokens=False)
        n_chip = sum(1 for i in out if off <= i < off + meta["chip_vocab"])
        shown = re.sub(r"(<\|chip[^|]*\|>)+", " [machine] ", text)
        resp = re.search(r"<tool_response>\s*(\d+)\s*</tool_response>", text)
        tail = text.split("</tool_response>")[-1]
        got = answer_number(tail) if resp else answer_number(text)
        inline = re.findall(r"(?:<\|chip[^|]*\|>)+ (\d+)", text)
        rep["merged"].append({"q": q, "want": want, "tool_result": int(resp[1]) if resp else None,
                              "inline_results": [int(v) for v in inline],
                              "answer": got, "ok": got == want, "chip_tokens": n_chip,
                              "seconds": round(time.time() - t, 1), "text": shown[-400:]})
        print(json.dumps(rep["merged"][-1], ensure_ascii=False), flush=True)
    del model
    if args.base:
        model, tok = load(args.base)
        for q, op, want in probs:
            out = ask(model, tok, q, [], None, args.max_new, NUDGE if args.nudge else None)
            text = tok.decode(out, skip_special_tokens=True)
            got = answer_number(text)
            rep["base"].append({"q": q, "want": want, "answer": got, "ok": got == want})
            print("base", json.dumps(rep["base"][-1]), flush=True)
    for k in ("merged", "base"):
        if rep[k]:
            print(k, "accuracy", sum(r["ok"] for r in rep[k]), "/", len(rep[k]))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
