"""Spike: replay a merged model's generation through the circuit IR and compare, token by token.

    python -m chipformer.q35.spikes_replay MERGED --data gsmhard --idx 28 --cond merged_inline

Generates greedily with HF generate (batch 1), maps the tokens to the circuit's alphabet (chip
tokens, text classes, OTHER), evaluates the IR on that stream, and reports every position the
circuit owns where the model's token differs from the IR's prediction, plus each inline fire.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch


def class_map(tok, meta):
    from chipformer.q35 import bridge as B

    m = {int(i): int(cls) for cls, i in meta["text_ids"].items()}
    for cls, strings in B.EXTRA_TEXT.items():
        for s in strings:
            (i,) = tok.encode(s, add_special_tokens=False)
            m[i] = cls
    return m


def main():
    from chipformer.q35 import bridge as B
    from chipformer.q35 import check as Q
    from chipformer.q35 import chip8_q35, evals
    from chipformer.q35.demo import load
    from chipformer.q35.lib import bcd
    from chipformer.ref.emu import Chip8
    from chipformer.trace import boot_tokens

    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("--data", default="gsmhard")
    ap.add_argument("--idx", type=int, nargs="+", default=[28])
    ap.add_argument("--cond", default="merged_inline")
    ap.add_argument("--max-new", type=int, default=1500)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ids", help="replay these saved prompt+generated ids (.npy) instead")
    ap.add_argument("--prompt-len", type=int, help="with --ids: the prompt is this many ids")
    args = ap.parse_args()
    meta = json.loads((Path(args.merged) / "chipformer.json").read_text())
    off, nchip = meta["chip_token_offset"], meta["chip_vocab"]
    bios = [t + off for t in boot_tokens(Chip8(bcd.ROM, bcd.QUIRKS).cells)]
    if args.ids:
        from transformers import AutoTokenizer

        model, tok = None, AutoTokenizer.from_pretrained(args.merged)
    else:
        model, tok = load(args.merged)
    cmap = class_map(tok, meta)
    circuit = chip8_q35.build()
    data = evals.load_data(args.data, args.n)
    for i in args.idx:
        prompt = evals.build_prompt(tok, data[i]["question"], args.cond, bios)
        if args.ids:
            saved = np.load(args.ids).tolist()
            if args.prompt_len:
                prompt = saved[:args.prompt_len]
            assert saved[:len(prompt)] == prompt, "saved ids do not start with this prompt"
            gen = saved[len(prompt):]
        else:
            x = torch.tensor([prompt], device="cuda")
            with torch.no_grad():
                out = model.generate(x, max_new_tokens=args.max_new, do_sample=False)
            gen = out[0, len(prompt):].tolist()
        ids = prompt + gen
        stream = [t - off if off <= t < off + nchip else cmap.get(t, B.OTHER) for t in ids]
        pred, m = Q.predictions(circuit, stream)
        from chipformer.q35 import ir

        owner = ir.ev(circuit.owner, m)
        bad = [p for p in range(len(prompt) - 1, len(ids) - 1)
               if owner[p] == 1 and pred[p] != stream[p + 1]]
        text = tok.decode(gen)
        shown = re.sub(r"(<\|chip[^|]*\|>)+", "[m]", text)
        fires = re.findall(r"([\d,]+)\s*(\\times|\*|×|\+|-)\s*([\d,]+)\s*=\[m\]\s*(\d*)", shown)
        wrong = []
        for a, op, b, r in fires:
            a_, b_ = int(a.replace(",", "")), int(b.replace(",", ""))
            want = {"+": a_ + b_, "-": a_ - b_}.get(op, a_ * b_)
            if r != str(want):
                wrong.append((a, op, b, r, want))
        print(json.dumps({"idx": i, "n_gen": len(gen), "ir_mismatch_at_owned": len(bad),
                          "first": [(p - len(prompt), int(stream[p + 1]), int(pred[p]))
                                    for p in bad[:5]],
                          "fires": len(fires), "wrong_fires": wrong}), flush=True)
        if bad:
            p = bad[0]
            print("context:", repr(tok.decode(ids[max(len(prompt), p - 40):p + 2])), flush=True)
        if not args.ids:
            np.save(Path(args.merged).parent / f"replay_{args.data}_{i}.npy", np.array(ids))


if __name__ == "__main__":
    main()
