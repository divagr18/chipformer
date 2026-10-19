"""Arithmetic evals of the rv1 checkpoint: Qwen with the RV32 calc tool inside its weights.

    python -m chipformer.rv.evalarith MODEL --data arith --n 40 --out DIR
    python -m chipformer.rv.evalarith BASE --cond pytool --data dmx_mul_big --n 100 --out DIR
    python -m chipformer.rv.evalarith BASE --cond base_nudge --data dmx_mul_big --out DIR

Problems: ``arith`` is q35.evals' seeded synthetic set (the CHIP-8 runs' ids); any other name is
a q35.arith_bench dataset (DeepMind modules, NUPA, ...), same seeds. The prompt is a system nudge
(the calc tool, or exact arithmetic by hand without a tool), the question and evals' boxed-answer
suffix; scoring is arith_bench's exact rational match. Conditions:

  rv1         the merged checkpoint; the machine answers calc calls (lockstep-checked against the
              reference machine; the token ids are kept for rv/replay.py)
  pytool      base Qwen3.5 with the same tool, its responses computed by the host in the
              firmware's exact format (calc.expected: // floor, %, **, gcd)
  base_nudge  base Qwen3.5 without tools

Qwen writes at most ``--budget`` tokens (4096). Records are appended per problem (resumable):
DIR/<data>.<cond>.jsonl.
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch

from chipformer.q35 import arith_bench as AB
from chipformer.q35 import evals as E
from chipformer.rv import bridge as BR
from chipformer.rv import calc, elf, ref
from chipformer.rv.chat import CALC_TOOL, RvChat

NUDGE = ("You have a calc tool for exact arithmetic. Use it for every arithmetic step instead "
         "of calculating by hand.")
NUDGE_NONE = "Do every arithmetic step exactly."


def calc_response(expr: str) -> str:
    """The firmware's tool response for an expression (computed by the host)."""
    e = expr.replace("\n", "")
    v = calc.expected(e) if re.fullmatch(r"[0-9.+\-*/%^(), gcd]*", e) and e.strip() else "error\n"
    return f"\n<tool_response>\n{v}</tool_response>\n<think>\n\n</think>\n\n"


class BridgeHost:
    """pytool's host as the machine itself: a reference bridge.Session on the calc firmware,
    fed base Qwen's own tokens (the prompt as history). At </tool_call> it runs the call exactly
    as rv1's machine would (the same capture window, firmware and echo tokens) and returns the
    response's Qwen tokens to force."""

    def __init__(self, T, firmware, prompt: list[int]):
        m = ref.Machine.from_program(elf.program(firmware))
        BR.park(m)
        self.ses = BR.Session(T, m)
        self.ses.history(prompt)
        self.tcc = T.TCC

    def feed(self, t: int) -> list[int]:
        """Qwen's token t; the response tokens to force (empty unless t ran a call)."""
        from chipformer.rv import spec as S

        k = len(self.ses.toks)
        try:
            self.ses.qwen([t])
        except RuntimeError:  # the run did not halt: an error response (rv1 would loop on)
            p = self.ses.T.printable
            body = [10, 256, 10, *b"error (did not halt)\n", 257, 10, 258, 10, 10, 259, 10, 10]
            return [p[b] for b in body]
        return [x - S.TEXT_TOKEN for x in self.ses.toks[k + 1:] if S.is_text(x)]


def problems(data: str, n: int, seed: int) -> list[dict]:
    return E.load_data("arith", n, seed) if data == "arith" else AB.load(data, n, seed)


@torch.no_grad()
def run_base(model, tok, ids, tool: bool, budget: int, host: BridgeHost | None = None):
    """Greedy decoding of base Qwen; with ``tool`` the host answers calc calls (``host``: the
    machine's own responses, else calc.expected's)."""
    dev = next(model.parameters()).device
    tcc = tok.convert_tokens_to_ids("</tool_call>")
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    seq, gen, own, calls = list(ids), [], 0, 0
    while own < budget:
        out = model.generate(torch.tensor([seq], device=dev), max_new_tokens=budget - own,
                             do_sample=False, eos_token_id=sorted(eos | ({tcc} if tool else set())))
        new = out[0, len(seq):].tolist()
        gen += new
        seq += new
        own += len(new)
        resp = []
        if host is not None:
            for t in new:
                resp = host.feed(t)
        if not new or new[-1] in eos or not tool or new[-1] != tcc:
            break
        if host is None:
            txt = tok.decode(gen, skip_special_tokens=False)
            m = re.findall(r"<parameter=[^>]*>(.*?)</parameter>", txt.split("<tool_call>")[-1],
                           re.S)
            resp = tok(calc_response(m[0] if m else ""), add_special_tokens=False).input_ids
        gen += resp
        seq += resp
        calls += 1
    return gen, calls, own


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--cond", default="rv1", choices=("rv1", "pytool", "base_nudge"))
    ap.add_argument("--data", default="arith")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--firmware", default="tests/data/rv/calc-bridge.elf")
    ap.add_argument("--budget", type=int, default=4096, help="Qwen tokens per problem")
    ap.add_argument("--max-chip", type=int, default=200000)
    ap.add_argument("--chip-path", default="executor", choices=("executor", "hf"),
                    help="rv1's chip layers: the circuit's Triton executor, or their HF modules")
    ap.add_argument("--batch", type=int, default=1, help="rv1: conversations decoded at once")
    ap.add_argument("--max-len", type=int, default=6144, help="rv1 --batch: Qwen KV slots")
    ap.add_argument("--shard", default=None, help="i/k: every k-th problem from the i-th")
    ap.add_argument("--ids", default=None, help="comma-separated problem ids (a subset)")
    ap.add_argument("--host", default="bridge", choices=("bridge", "expected"),
                    help="pytool's responses: the calc firmware on a reference bridge Session "
                         "(rv1's machine exactly), or calc.expected (Python, BPE-tokenized)")
    args = ap.parse_args()
    n = args.n or (40 if args.data == "arith" else None)
    probs = problems(args.data, n, args.seed)
    out = Path(args.out) / f"{args.data}.{args.cond}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["id"] for x in out.read_text().splitlines()} if out.exists() else set()
    tool = args.cond != "base_nudge"
    if args.cond == "rv1":
        chat = RvChat(args.model, executor=args.chip_path == "executor")
        tok = chat.tok
    else:
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
        model = model.to("cuda").eval()
        if tool and args.host == "bridge":
            T = BR.Text(Path(args.model) / "tokenizer.json")

    def make_host(prompt):
        return BridgeHost(T, args.firmware, prompt)
    if args.shard:
        i, k = map(int, args.shard.split("/"))
        probs = probs[i::k]
    if args.ids:
        want = set(args.ids.split(","))
        probs = [p for p in probs if p["id"] in want]
    todo = [p for p in probs if p["id"] not in done]
    t_all = time.time()

    def msgs_of(p):
        return [{"role": "system", "content": NUDGE if tool else NUDGE_NONE},
                {"role": "user", "content": p["question"] + E.SUFFIX}]

    def rv1_job(p):
        ids = chat.prompt_ids(msgs_of(p))
        m = ref.Machine.from_program(elf.program(args.firmware))
        BR.park(m)
        m2 = ref.Machine.from_program(elf.program(args.firmware))
        BR.park(m2)
        lock = BR.Session(chat.T, m2)  # every circuit token checked against the reference
        lock.history(ids)
        return ids, m, lock

    def rv1_fields(ids, lock, r):
        gen = r["tokens"]
        outs = [chat.T.decode_bytes(e).decode("utf-8", "replace") for e in lock.runs]
        return chat.text_of(gen), dict(
            prompt_ids=ids, gen_ids=gen, calls=len(lock.runs),
            n_qwen=r["text_tokens"] - sum(len(e) for e in lock.runs),
            n_chip=r["chip_tokens"], lock_error=r["lock_error"],
            head_mismatch=r["head_mismatch"], gen_seconds=round(r["seconds"], 1),
            chip_path=args.chip_path, chip_seconds=r["chip_seconds"],
            text_seconds=r["text_seconds"], batch=args.batch,
            results=[o.split("\n")[2] if o.count("\n") > 2 else o for o in outs])

    def emit(p, text, fields, seconds):
        rec = {"id": p["id"], "gold": p["gold"], "question": p["question"], "cond": args.cond}
        rec.update(fields)
        rec.update(AB.exact_fields(p, text))
        rec.update(seconds=round(seconds, 1), text=text)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        keys = ("id", "gold", "pred_exact", "exact", "calls", "n_qwen", "n_chip", "seconds",
                "lock_error", "head_mismatch")
        print(json.dumps({k: rec[k] for k in keys if k in rec}), flush=True)

    if args.cond == "rv1" and args.batch > 1:
        from chipformer.rv.rvbatch import Job, RvBatch

        rb = RvBatch(chat, args.batch, args.max_len)

        def jobs():
            for p in todo:
                ids, m, lock = rv1_job(p)
                yield Job(p["id"], ids, m, lock, {"p": p})

        def finished(job, r):
            text, fields = rv1_fields(job.ids, job.lock, r)
            emit(job.extra["p"], text, fields, r["seconds"])

        st = rb.run(jobs(), args.budget, args.max_chip, finished)
        print(json.dumps({"batch_stats": st}), flush=True)
    if args.cond != "rv1" and args.batch > 1:  # base Qwen: q35's BatchDecoder
        from chipformer.q35.batch import BatchDecoder

        prompts = []
        for p in todo:
            ids = tok.apply_chat_template(msgs_of(p), tools=[CALC_TOOL] if tool else None,
                                          add_generation_prompt=True, enable_thinking=False,
                                          tokenize=True)
            prompts.append(ids if isinstance(ids, list) else ids["input_ids"])
        tcc = tok.convert_tokens_to_ids("</tool_call>")
        state = [{"forced": [], "own": 0, "gen": [], "calls": 0,
                  "host": make_host(pr) if tool and args.host == "bridge" else None}
                 for pr in prompts]

        def hook(i, t):
            """pytool: after </tool_call> the next tokens are the host's response (forced);
            the row stops after ``budget`` of Qwen's own tokens, as run_base does."""
            s = state[i]
            if s["forced"]:
                t = s["forced"].pop(0)
                s["gen"].append(t)
                return t, False
            s["own"] += 1
            s["gen"].append(t)
            if s["host"] is not None:
                s["forced"] = s["host"].feed(t)
                s["calls"] += bool(s["forced"])
            elif tool and t == tcc:
                txt = tok.decode(s["gen"], skip_special_tokens=False)
                mm = re.findall(r"<parameter=[^>]*>(.*?)</parameter>",
                                txt.split("<tool_call>")[-1], re.S)
                s["forced"] = tok(calc_response(mm[0] if mm else ""),
                                  add_special_tokens=False).input_ids
                s["calls"] += 1
            return t, s["own"] >= args.budget and not s["forced"]

        t0 = time.time()
        max_len = -(-(max(map(len, prompts)) + 2 * args.budget) // 1024) * 1024
        bd = BatchDecoder(model, batch=args.batch, max_len=max_len)

        def fin(i, toks):
            s = state[i]
            text = tok.decode(toks, skip_special_tokens=False)
            emit(todo[i], text, dict(calls=s["calls"], n_qwen=s["own"], n_chip=0,
                                     batch=args.batch, host=args.host if tool else None),
                 time.time() - t0)

        if prompts:
            bd.run(prompts, 2 * args.budget, on_done=fin, hook=hook)
    for p in todo if args.batch == 1 else []:
        t0 = time.time()
        if args.cond == "rv1":
            ids, m, lock = rv1_job(p)
            r = chat.generate(ids, m, args.budget, max_chip=args.max_chip, check_head=True,
                              lock=lock)
            text, fields = rv1_fields(ids, lock, r)
        else:
            ids = tok.apply_chat_template(msgs_of(p), tools=[CALC_TOOL] if tool else None,
                                          add_generation_prompt=True, enable_thinking=False,
                                          tokenize=True)
            ids = ids if isinstance(ids, list) else ids["input_ids"]
            host = make_host(ids) if tool and args.host == "bridge" else None
            gen, calls, own = run_base(model, tok, ids, tool, args.budget, host)
            text = tok.decode(gen, skip_special_tokens=False)
            fields = dict(calls=calls, n_qwen=own, n_chip=0, host=args.host if tool else None)
        emit(p, text, fields, time.time() - t0)
    wall = time.time() - t_all
    if todo:
        print(json.dumps({"problems": len(todo), "wall_s": round(wall, 1),
                          "problems_per_hour": round(len(todo) / wall * 3600, 1),
                          "batch": args.batch}), flush=True)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    print(f"{out.name}: {sum(r['exact'] for r in rows)}/{len(rows)} exact")


if __name__ == "__main__":
    main()
