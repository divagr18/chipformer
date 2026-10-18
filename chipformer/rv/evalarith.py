"""Arithmetic eval of the rv1 checkpoint: Qwen with the RV32 calc tool inside its weights.

    python -m chipformer.rv.evalarith MODEL --n 40 --out dist/rv-arith.jsonl [--replay]
    python -m chipformer.rv.evalarith BASE --base --n 40 --out dist/rv-arith-pytool.jsonl
    python -m chipformer.rv.evalarith BASE --base --no-tool --n 40 --out dist/rv-arith-base.jsonl

Problems: q35.evals' seeded synthetic arithmetic (the same ids as the CHIP-8 runs). The prompt
is evals' tool nudge plus the question and its boxed-answer suffix. Scoring is exact (the
integer answer must equal the gold one). ``--base`` runs base Qwen3.5 with the same tool, its
responses computed by the host in the firmware's exact format (pytool), or with ``--no-tool``
no tool at all (the nudge then asks for exact arithmetic by hand). ``--replay`` checks every
merged run: the reference machine and the IR circuit must give the same chip tokens and echoes.
Records are appended per problem (resumable).
"""

import argparse
import json
import re
import time
from fractions import Fraction
from pathlib import Path

import torch

from chipformer.q35 import evals as E
from chipformer.rv import bridge as BR
from chipformer.rv import elf, ref
from chipformer.rv.chat import CALC_TOOL, RvChat, replay

NUDGE = ("You have a calc tool for exact arithmetic. Use it for every arithmetic step instead "
         "of calculating by hand.")
NUDGE_NONE = "Do every arithmetic step exactly."


def calc_response(expr: str) -> str:
    """The firmware's tool response for an expression (computed by the host)."""
    e = expr.strip().replace(",", "").replace("_", "").replace("×", "*")
    if not re.fullmatch(r"[0-9+\-*() \t\n]*", e) or not e.strip():
        v = "error"
    else:
        try:
            v = str(eval(e, {"__builtins__": {}}))  # noqa: S307 - digits and + - * ( ) only
        except Exception:  # noqa: BLE001 - a malformed expression prints "error" like the firmware
            v = "error"
    return f"\n<tool_response>\n{v}\n</tool_response>\n<think>\n\n</think>\n\n"


def score(text: str, gold: str) -> tuple[bool, str | None]:
    pred, _ = E.extract_answer(text, False)
    ok = pred is not None and pred == Fraction(int(gold))
    return ok, None if pred is None else str(pred)


@torch.no_grad()
def run_base(model, tok, ids, tool: bool, max_new: int):
    """Greedy decoding of base Qwen; with ``tool`` the host answers calc calls."""
    dev = next(model.parameters()).device
    tco, tcc = tok.convert_tokens_to_ids("<tool_call>"), tok.convert_tokens_to_ids("</tool_call>")
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    seq = list(ids)
    gen = []
    calls = 0
    while len(gen) < max_new:
        out = model.generate(torch.tensor([seq], device=dev), max_new_tokens=max_new - len(gen),
                             do_sample=False, eos_token_id=sorted(eos | ({tcc} if tool else set())))
        new = out[0, len(seq):].tolist()
        gen += new
        seq += new
        if not new or new[-1] in eos or not tool or new[-1] != tcc:
            break
        txt = tok.decode(gen, skip_special_tokens=False)
        m = re.findall(r"<parameter=[^>]*>(.*?)</parameter>", txt.split("<tool_call>")[-1], re.S)
        resp = tok(calc_response(m[0] if m else ""), add_special_tokens=False).input_ids
        gen += resp
        seq += resp
        calls += 1
    del tco
    return gen, calls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--firmware", default="tests/data/rv/c-calc.elf")
    ap.add_argument("--max-new", type=int, default=1500)
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--no-tool", action="store_true")
    ap.add_argument("--replay", action="store_true")
    args = ap.parse_args()
    probs = E.load_data("arith", args.n, args.seed)
    out = Path(args.out)
    done = {json.loads(x)["id"] for x in out.read_text().splitlines()} if out.exists() else set()
    if args.base:
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
        model = model.to("cuda").eval()
    else:
        chat = RvChat(args.model)
        tok = chat.tok
    for p in probs:
        if p["id"] in done:
            continue
        tool = not args.no_tool
        msgs = [{"role": "system", "content": NUDGE if tool else NUDGE_NONE},
                {"role": "user", "content": p["question"] + E.SUFFIX}]
        t0 = time.time()
        rec = {"id": p["id"], "gold": p["gold"], "question": p["question"]}
        if args.base:
            ids = tok.apply_chat_template(msgs, tools=[CALC_TOOL] if tool else None,
                                          add_generation_prompt=True, enable_thinking=False,
                                          tokenize=True)
            ids = ids if isinstance(ids, list) else ids["input_ids"]
            gen, calls = run_base(model, tok, ids, tool, args.max_new)
            text = tok.decode(gen, skip_special_tokens=False)
            rec.update(calls=calls, n_text=len(gen), n_chip=0)
        else:
            ids = chat.prompt_ids(msgs)
            m = ref.Machine.from_program(elf.program(args.firmware))
            BR.park(m)
            m2 = ref.Machine.from_program(elf.program(args.firmware))
            BR.park(m2)
            lock = BR.Session(chat.T, m2)  # every circuit token checked against the reference
            lock.history(ids)
            r = chat.generate(ids, m, args.max_new, check_head=True, lock=lock)
            gen = r["tokens"]
            text = chat.text_of(gen)
            outs = [chat.T.decode_bytes(e).decode("utf-8", "replace") for e in lock.runs]
            rec.update(calls=text.count("<tool_response>"), n_text=r["text_tokens"],
                       n_chip=r["chip_tokens"], lock_error=r["lock_error"],
                       head_mismatch=r["head_mismatch"], gen_seconds=round(r["seconds"], 1),
                       results=[o.split("\n")[2] if o.count("\n") > 2 else o for o in outs])
            if args.replay:
                rec["replay"] = replay(chat, ids, gen, args.firmware)
        ok, pred = score(text, p["gold"])
        rec.update(correct=ok, pred=pred, seconds=round(time.time() - t0, 1), text=text)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        extra = {k: rec[k] for k in ("lock_error", "head_mismatch", "replay") if k in rec}
        print(json.dumps({k: rec[k] for k in ("id", "gold", "pred", "correct", "calls", "n_text",
                                              "n_chip", "seconds")} | extra), flush=True)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    print(f"{sum(r['correct'] for r in rows)}/{len(rows)} correct")


if __name__ == "__main__":
    main()
