"""What does a flag do to the rest of the reply? An intervention at the flag position.

For every held-out lie that q4-mon flagged, the reply is cut where the flag appeared and resumed
under four conditions:

  flag      the <|flag|> token as Qwen sees it. Its Qwen dims are the embedding of
            tok.encode("⚠")[0] (merge.py), which is the byte-level token for 0xE2, the UTF-8 lead
            byte of "⚠" (and of every U+2000-U+2FFF symbol): Qwen goes on to complete a symbol
  neutral   a neutral symbol token at the same position (default "◆")
  nothing   the plain reply, uninterrupted
  zero      a token with an all-zero embedding at that position (the flag without the ⚠ meaning)

The resumed replies are labelled false / true / neither (data.label). The runs use base Qwen3.5-4B
fed with input embeddings, which is what Qwen's layers see in q4-mon, minus the monitor itself:
with the monitor in place it would flag again in the "nothing" condition. Up to the flag, q4-mon
decodes what base Qwen decodes; prompts where the two differ are dropped and counted.

python -m chipformer.q35.monitor.intervene run BASE_MODEL --out dist/monitor/intervene.json
"""

import argparse
import json
import math
from pathlib import Path

import torch

from chipformer.q35.monitor import data as D
from chipformer.q35.monitor.probe import load

CONDS = ("flag", "neutral", "nothing", "zero")
FLAG_TEXT = "<|flag|>"


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans: (only a, only b, p)."""
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b))
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return n10, n01, min(1.0, 2 * p)


def cases(tok, gens, replies):
    """(row, prefix ids) for each flagged lie whose pre-flag text matches the base reply."""
    out, dropped = [], 0
    for r in replies:
        g = gens[r["id"]]
        if g["label"] != "lie" or not r["flags"]:
            continue
        pre = r["text"].split(FLAG_TEXT)[0]
        comp = g["completion"]
        k = next((k for k in range(len(comp) + 1)
                  if tok.decode(comp[:k], skip_special_tokens=False) == pre), None)
        if k is None:
            dropped += 1
            continue
        out.append((g, g["prompt"] + comp[:k], k))
    return out, dropped


@torch.no_grad()
def resume(tok, model, items, cond, ins_id, max_new, sample=None, batch=24):
    """Continue each prefix after inserting the condition's token; returns the new token lists."""
    emb = model.get_input_embeddings()
    dev = emb.weight.device
    outs = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        seqs = []
        for ids in chunk:
            e = emb(torch.tensor(ids, device=dev))
            if cond != "nothing":
                extra = torch.zeros_like(e[:1]) if cond == "zero" else \
                    emb(torch.tensor([ins_id[cond]], device=dev))
                e = torch.cat([e, extra])
            seqs.append(e)
        T = max(len(s) for s in seqs)
        pad = emb(torch.tensor([tok.pad_token_id], device=dev))
        x = torch.stack([torch.cat([pad.expand(T - len(s), -1), s]) for s in seqs])
        mask = torch.stack([torch.cat([torch.zeros(T - len(s), dtype=torch.long),
                                       torch.ones(len(s), dtype=torch.long)]) for s in seqs]).to(dev)
        kw = dict(do_sample=False) if sample is None else \
            dict(do_sample=True, temperature=sample["t"], top_p=sample["top_p"])
        g = model.generate(inputs_embeds=x, attention_mask=mask, max_new_tokens=max_new,
                           pad_token_id=tok.pad_token_id, **kw)
        for row in g.tolist():
            cut = [t for t in row]
            for stop in (tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id):
                if stop in cut:
                    cut = cut[:cut.index(stop)]
            outs.append(cut)
    return outs


def outcome(tok, g, k, new):
    text = tok.decode(g["completion"][:k] + new, skip_special_tokens=True)
    lab = D.label(text, g["true"], g["false"])
    return {"lie": "false", "honest": "true"}.get(lab, "neither")


def table(res):
    """Rates with Wilson CIs per condition, and paired McNemar tests on 'states the false value'
    and on 'states the true value'."""
    n = len(next(iter(res.values())))
    rates = {}
    for c, labs in res.items():
        rates[c] = {o: {"k": labs.count(o), "rate": labs.count(o) / n, "ci95": wilson(labs.count(o), n)}
                    for o in ("false", "true", "neither")}
    tests = {}
    for a, b in (("flag", "neutral"), ("flag", "nothing"), ("flag", "zero"), ("neutral", "nothing"),
                 ("zero", "nothing")):
        for o in ("false", "true"):
            x = [l == o for l in res[a]]
            y = [l == o for l in res[b]]
            only_a, only_b, p = mcnemar(x, y)
            tests[f"{a}_vs_{b}:{o}"] = {f"only_{a}": only_a, f"only_{b}": only_b, "p": p}
    return {"n": n, "rates": rates, "paired": tests}


def cmd_run(args):
    tok, model = load(args.model)
    ins_id = {"flag": tok.encode("⚠", add_special_tokens=False)[0]}  # as merge.py
    ids = tok.encode(args.neutral, add_special_tokens=False)
    assert len(ids) == 1, (args.neutral, ids)
    ins_id["neutral"] = ids[0]
    gens = {(g := json.loads(l))["id"]: g for l in open(args.gens, encoding="utf-8")}
    replies = json.load(open(args.replies))
    cs, dropped = cases(tok, gens, replies)
    print(f"{len(cs)} flagged lies, {dropped} dropped (pre-flag text differs from base)", flush=True)
    items = [ids for _, ids, _ in cs]
    report = {"n_cases": len(cs), "dropped": dropped, "neutral_token": args.neutral,
              "max_new": args.max_new}
    greedy, texts = {}, {}
    for c in CONDS:
        new = resume(tok, model, items, c, ins_id, args.max_new)
        greedy[c] = [outcome(tok, g, k, nw) for (g, _, k), nw in zip(cs, new)]
        texts[c] = [tok.decode(nw, skip_special_tokens=True) for nw in new]
        print(c, {o: greedy[c].count(o) for o in ("false", "true", "neither")}, flush=True)
    report["greedy"] = table(greedy)
    report["greedy_by_family"] = {
        f: {c: {o: sum(1 for (g, _, _), l in zip(cs, greedy[c]) if g["family"] == f and l == o)
                for o in ("false", "true", "neither")} for c in CONDS}
        for f in sorted({g["family"] for g, _, _ in cs})}
    sampled = {c: [] for c in CONDS}
    for s in range(args.samples):
        for c in CONDS:
            torch.manual_seed(1000 + s)
            new = resume(tok, model, items, c, ins_id, args.max_new, sample={"t": args.t, "top_p": 0.95})
            sampled[c] += [outcome(tok, g, k, nw) for (g, _, k), nw in zip(cs, new)]
    if args.samples:
        report["sampled"] = table(sampled) | {"samples_per_case": args.samples, "temperature": args.t}
    changed = [i for i in range(len(cs)) if greedy["flag"][i] != greedy["nothing"][i]]
    report["examples"] = [
        {"id": cs[i][0]["id"], "true": cs[i][0]["true"], "false": cs[i][0]["false"],
         "prefix": tok.decode(cs[i][0]["completion"][:cs[i][2]], skip_special_tokens=True),
         **{c: {"label": greedy[c][i], "text": texts[c][i]} for c in CONDS}}
        for i in (changed[:6] + [i for i in range(len(cs)) if i not in changed][:2])]
    json.dump(report, open(args.out, "w"), indent=1)
    print(json.dumps({"greedy": report["greedy"], "sampled": report.get("sampled")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model", help="base Qwen3.5-4B")
    r.add_argument("--gens", default="dist/monitor/gens.jsonl")
    r.add_argument("--replies", default="dist/monitor/eval_replies.json")
    r.add_argument("--out", default="dist/monitor/intervene.json")
    r.add_argument("--neutral", default="◆")
    r.add_argument("--max-new", type=int, default=48)
    r.add_argument("--samples", type=int, default=2)
    r.add_argument("--t", type=float, default=0.7)
    args = ap.parse_args()
    {"run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
