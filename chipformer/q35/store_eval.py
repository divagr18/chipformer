"""GPU evals of the object store on a merged checkpoint (stock HF modules, greedy decoding).

    python -m chipformer.q35.store_eval exact   MERGED            # teacher-forced: the circuit's
                                                                # answers == the reference
    python -m chipformer.q35.store_eval recall  MERGED BASE --n 50 --pad 10000 100000
    python -m chipformer.q35.store_eval session MERGED            # snapshot -> fresh session

``recall``: Qwen stores n random 20-token strings with ``put`` (one user turn each), then a long
unrelated chat, then for each string a separate question; exact match of its final answer.
Against base Qwen told the same strings in its context, and base Qwen with the strings right
before the question (no padding).
"""

import argparse
import re
import copy
import json
import random
import time
from pathlib import Path

import torch

from chipformer.q35 import store as S

SYSTEM = ("You have a small memory store: put saves a text under a name, get returns it "
          "exactly, list shows the names, append adds to a text, delete removes a name. Use it "
          "whenever you are asked to save or recall something.")
IM_START, IM_END, EOT = 248045, 248046, 248044


class Runner:
    """Greedy decoding with a stock DynamicCache (chunked prefill)."""

    def __init__(self, path, device="cuda"):
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16).to(device).eval()
        self.dev = device
        self.lm_head = self.model.lm_head
        meta = Path(path) / "chipformer.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else None

    def ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    @torch.no_grad()
    def feed(self, cache, ids: list[int], chunk: int = 2048):
        """Prefill ``ids`` into ``cache``; the logits of the last position."""
        from transformers import DynamicCache

        cache = cache if cache is not None else DynamicCache(config=self.model.config)
        out = None
        for i in range(0, len(ids), chunk):
            x = torch.tensor([ids[i:i + chunk]], device=self.dev)
            out = self.model(input_ids=x, past_key_values=cache, use_cache=True,
                             logits_to_keep=1)
        return cache, out.logits[0, -1].float()

    @torch.no_grad()
    def generate(self, cache, ids: list[int], max_new: int, stop=(IM_END,)):
        """Greedy, with <|im_start|> and <|endoftext|> banned (Qwen ends a turn with <|im_end|>;
        evals' "nb" and "ne")."""
        cache, logits = self.feed(cache, ids)
        out = []
        for _ in range(max_new):
            logits[IM_START] = float("-inf")
            logits[EOT] = float("-inf")
            t = int(logits.argmax())
            out.append(t)
            cache, logits = self.feed(cache, [t])  # (the cache holds every token, the end too)
            if t in stop:
                break
        return cache, out

    def decode(self, ids: list[int]) -> str:
        lo = self.meta["chip_token_offset"] if self.meta else 1 << 30
        hi = lo + (self.meta["chip_vocab"] if self.meta else 0)
        return self.tok.decode([t for t in ids if not lo <= t < hi], skip_special_tokens=False)


def turn(role: str, text: str) -> str:
    return f"<|im_start|>{role}\n{text}<|im_end|>\n"


def opening(r: Runner, tools: bool) -> list[int]:
    msgs = [{"role": "system", "content": SYSTEM}] if tools else []
    s = r.tok.apply_chat_template(msgs + [{"role": "user", "content": "hi"}],
                                  tools=S.TOOLS if tools else None, tokenize=False,
                                  add_generation_prompt=False, enable_thinking=False)
    return r.ids(s[:s.rindex("<|im_start|>user")])  # the system part


ASSISTANT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def strings(rng: random.Random, r: Runner, n: int) -> list[str]:
    """n random strings of about 20 tokens: hex ids, uuids, code lines, word salad."""
    words = ("alpha bravo kernel tensor lattice orbit cobalt maple vortex quartz ember delta "
             "harbor pixel signal falcon cedar nimbus").split()
    out = []
    while len(out) < n:
        kind = rng.randrange(4)
        if kind == 0:
            s = "".join(rng.choice("0123456789abcdef") for _ in range(rng.randint(24, 34)))
        elif kind == 1:
            h = "".join(rng.choice("0123456789abcdef") for _ in range(32))
            s = f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
        elif kind == 2:
            v = rng.choice(["x", "acc", "w_2", "buf"])
            s = (f"{v} = np.{rng.choice(['dot', 'clip', 'roll'])}({v}[{rng.randint(1, 9)}:], "
                 f"{rng.randint(10, 999)}) + {rng.random():.4f}")
        else:
            s = " ".join(rng.choice(words) for _ in range(rng.randint(8, 12))) + f" {rng.randint(100, 9999)}"
        if 16 <= len(r.ids(s)) <= 24:
            out.append(s)
    return out


def fires_inline(r: Runner, ids: list[int]) -> bool:
    """Would the circuit answer an inline "A op B =" in this text? Prefilled history must not
    contain one: the prompt would lack the circuit's answer and leave it mid-call."""
    from chipformer.q35.bridge import InlineScan

    vocab = getattr(r, "_vocab", None)
    if vocab is None:
        vocab = r._vocab = S.Vocab(Path(r.tok.name_or_path) / "tokenizer.json")
    scan = InlineScan()
    return any(scan.feed(vocab.cls(q) if q < S.SPECIAL else 0) for q in ids)


def padding(rng: random.Random, r: Runner, n_tokens: int) -> list[int]:
    """Unrelated chat turns (GSM8K questions and answers when available) of about n tokens, none
    of which the circuit's inline arithmetic would answer (see fires_inline)."""
    pairs = []
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="train")
        pairs = [(x["question"], x["answer"]) for x in ds]
    except Exception:
        words = "the a of to and in is it for on with as was at by an be this that from".split()
        for _ in range(5000):
            q = " ".join(rng.choice(words) for _ in range(30)) + "?"
            pairs.append((q, " ".join(rng.choice(words) for _ in range(60)) + "."))
    rng.shuffle(pairs)
    out = []
    for q, a in pairs:
        a = re.sub(r"<<[^>]*>>", "", a)  # GSM8K's calculator annotations
        ids = r.ids(turn("user", q) + turn("assistant", a))
        if fires_inline(r, ids):
            continue
        out += ids
        if len(out) >= n_tokens:
            break
    return out


def is_exact(name: str, ans: str, want: str) -> bool:
    """The answer is the stored text (allowing quotes, backticks, a final period and a leading
    "name:")."""
    a = ans.strip()
    if a.startswith(name + ":"):
        a = a[len(name) + 1:].strip()
    return a.strip("`\"' ") == want or a.strip("`\"' .") == want


def final_answer(r: Runner, gen: list[int]) -> str:
    text = r.decode(gen)
    if "</tool_response>" in text:
        text = text[text.rindex("</tool_response>") + len("</tool_response>"):]
    return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


def tool_text(r: Runner, gen: list[int]) -> str | None:
    text = r.decode(gen)
    if "<tool_response>" not in text:
        return None
    a = text.index("<tool_response>") + len("<tool_response>\n")
    return text[a:text.index("</tool_response>", a)].rstrip("\n")


def recall(args):
    rng = random.Random(args.seed)
    merged = Runner(args.merged)
    items = strings(rng, merged, args.n)
    names = [f"s{i + 1}" for i in range(args.n)]
    res = {"n": args.n, "strings": items, "conditions": {}}
    t0 = time.time()
    # --- merged: Qwen stores them itself
    ctx = opening(merged, True)
    cache, _ = merged.feed(None, ctx)
    puts = []
    for name, s in zip(names, items):
        q = merged.ids(turn("user", f"Save this exactly under the name {name}: {s}") + ASSISTANT)
        cache, gen = merged.generate(cache, q, 400)
        ctx += q + gen + merged.ids("\n")
        cache, _ = merged.feed(cache, merged.ids("\n"))
        puts.append({"name": name, "said": merged.decode(gen)[-300:],
                     "stored_ok": tool_text(merged, gen) == "ok"})
    res["puts"] = puts
    print(f"stored {sum(p['stored_ok'] for p in puts)}/{args.n} ({time.time() - t0:.0f}s)",
          flush=True)
    for pad in args.pad:
        pad_ids = padding(random.Random(args.seed + pad), merged, pad)
        c2, _ = merged.feed(copy.deepcopy(cache), pad_ids)
        rows = []
        for name, s in zip(names, items):
            q = merged.ids(turn("user", f"What is stored under the name {name}? Look it up and "
                                        "reply with exactly that text, nothing else.") + ASSISTANT)
            _, gen = merged.generate(copy.deepcopy(c2), q, 300)
            ans, tt = final_answer(merged, gen), tool_text(merged, gen)
            rows.append({"name": name, "want": s, "answer": ans, "tool": tt,
                         "exact": is_exact(name, ans, s), "tool_exact": tt == s})
        res["conditions"][f"merged_pad{pad}"] = rows
        print(f"merged pad {pad}: exact {sum(x['exact'] for x in rows)}/{len(rows)}, "
              f"tool exact {sum(x['tool_exact'] for x in rows)} ({time.time() - t0:.0f}s)",
              flush=True)
    del merged, cache
    torch.cuda.empty_cache()
    # --- base: the strings in its context
    base = Runner(args.base)
    ctx = opening(base, False)
    for name, s in zip(names, items):
        ctx += base.ids(turn("user", f"Remember this exactly; its name is {name}: {s}")
                        + turn("assistant", "Noted."))
    cache, _ = base.feed(None, ctx)
    for pad in args.pad:
        pad_ids = padding(random.Random(args.seed + pad), base, pad)
        c2, _ = base.feed(copy.deepcopy(cache), pad_ids)
        rows = []
        for name, s in zip(names, items):
            q = base.ids(turn("user", f"What exactly was {name}? Reply with exactly that text, "
                                      "nothing else.") + ASSISTANT)
            _, gen = base.generate(copy.deepcopy(c2), q, 120)
            ans = final_answer(base, gen)
            rows.append({"name": name, "want": s, "answer": ans,
                         "exact": is_exact(name, ans, s)})
        res["conditions"][f"base_pad{pad}"] = rows
        print(f"base pad {pad}: exact {sum(x['exact'] for x in rows)}/{len(rows)}", flush=True)
    rows = []  # the string right before its question
    for name, s in zip(names, items):
        q = opening(base, False) + base.ids(turn("user", f"{name}: {s}\nWhat exactly is {name}? "
                                                   "Reply with exactly that text, nothing else.")
                                            + ASSISTANT)
        _, gen = base.generate(None, q, 120)
        ans = final_answer(base, gen)
        rows.append({"name": name, "want": s, "answer": ans,
                     "exact": is_exact(name, ans, s)})
    res["conditions"]["base_in_context"] = rows
    print(f"base in context: exact {sum(x['exact'] for x in rows)}/{len(rows)}", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))


def session(args):
    """Session 1 saves notes; its store is snapshotted; session 2 starts fresh and recalls."""
    from chipformer.q35 import store_snapshot as snap

    r = Runner(args.merged)
    heads = r.meta["store"]["persistent_heads"]
    notes = [("release", "Ship v2.1 on Friday 14:00 UTC; Priya signs off; freeze main at noon."),
             ("wifi", "SSID garden-5G, password 7Tq!pw-93xz"),
             ("todo", "renew the TLS cert, email Marco the Q3 numbers")]
    ctx = opening(r, True)
    cache, _ = r.feed(None, ctx)
    log = []
    for name, text in notes:
        q = r.ids(turn("user", f"Please save this under the name {name}: {text}") + ASSISTANT)
        cache, gen = r.generate(cache, q, 400)
        log.append(("user", f"save {name}"), )
        log.append(("assistant", r.decode(gen)))
        cache, _ = r.feed(cache, r.ids("\n"))
    state = snap.snapshot(cache, heads)
    path = Path(args.out).with_suffix(".snapshot.pt")
    snap.save(state, path)
    size = path.stat().st_size
    del cache
    # session 2: a new conversation, nothing of session 1 in Qwen's context
    cache = snap.restore(r.model, snap.load(path))
    q = opening(r, True) + r.ids(turn("user", "In an earlier session I asked you to save some "
                                              "notes. Which notes do you have? Then tell me the "
                                              "wifi password and when the release ships.")
                                 + ASSISTANT)
    gen_all = []
    cache, gen = r.generate(cache, q, 600)
    gen_all += gen
    out = {"session1": log, "snapshot_bytes": size, "heads": len(heads),
           "session2": r.decode(gen_all)}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


def exact(args):
    """Teacher-forced: at every position the reference says the circuit predicts, the merged
    model's argmax is that token (store calls with random tokens, long text in between)."""
    from chipformer.q35 import bridge as B

    r = Runner(args.merged)
    v = S.Vocab(Path(args.merged) / "tokenizer.json")
    rng = random.Random(args.seed)
    sm = S.StoreMachine(v)  # (the trace starts with the BIOS boot record)
    lo = r.meta["chip_token_offset"]

    def rnd(k):
        return [rng.randrange(S.SPECIAL) for _ in range(k)]

    def plain(k):  # text that starts nothing (no digits, operators or tags)
        return [q for q in rnd(4 * k) if v.cls(q) == B.OTHER][:k]

    names = [rnd(2) for _ in range(5)]
    sm.qwen_text(plain(30))
    for i in range(args.calls):
        op = rng.choice(["put", "append", "get", "get", "list", "delete"])
        nm = rng.choice(names)
        if op in ("put", "append"):
            sm.call(op, nm, rnd(rng.choice([3, 20, 60, S.CMAX + 4])))
        elif op == "list":
            sm.call("list")
        else:
            sm.call(op, nm)
        sm.qwen_text(plain(rng.randint(5, args.gap)))
    from chipformer import spec

    # chip tokens -> their merged ids; text tokens -> their Qwen id
    ids = [S.qid_of(t) if S.qid_of(t) is not None else
           (t + lo if t < spec.VOCAB_SIZE else S.full_qid(t)) for t in sm.tokens]
    pred = []
    with torch.no_grad():
        from transformers import DynamicCache

        cache = DynamicCache(config=r.model.config)
        for i in range(0, len(ids), 1024):
            x = torch.tensor([ids[i:i + 1024]], device=r.dev)
            lg = r.model(input_ids=x, past_key_values=cache, use_cache=True).logits[0]
            lg[:, IM_START] = float("-inf")
            pred += lg.argmax(-1).tolist()
    bad = [p for p in range(len(ids) - 1) if sm.generated[p + 1] and pred[p] != ids[p + 1]]
    n = sum(sm.generated[1:])
    print(f"{len(ids)} tokens, {n} predicted by the circuit, {len(bad)} wrong", flush=True)
    Path(args.out).write_text(json.dumps({"tokens": len(ids), "checked": n, "wrong": len(bad),
                                          "first_wrong": bad[:20]}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("recall")
    a.add_argument("merged")
    a.add_argument("base")
    a.add_argument("--n", type=int, default=50)
    a.add_argument("--pad", type=int, nargs="+", default=[10000, 100000])
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default="store_recall.json")
    b = sub.add_parser("session")
    b.add_argument("merged")
    b.add_argument("--out", default="store_session.json")
    e = sub.add_parser("exact")
    e.add_argument("merged")
    e.add_argument("--calls", type=int, default=40)
    e.add_argument("--gap", type=int, default=400)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default="store_exact.json")
    args = ap.parse_args()
    {"recall": recall, "session": session, "exact": exact}[args.cmd](args)


if __name__ == "__main__":
    main()
