"""Spike: does freezing RoPE positions on chip tokens stop Qwen repeating a line after an inline call?

    python -m chipformer.q35.spikes_pos MERGED --idx 0 1 2 --max-new 1200
"""

import argparse
import json
import re
from pathlib import Path

import torch


@torch.no_grad()
def decode(model, prompt, chip_lo, chip_hi, max_new, freeze, stop):
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    ids = list(prompt)
    pos, p = [], 0
    for t in prompt:
        pos.append(p)
        if not (freeze and chip_lo <= t < chip_hi):
            p += 1
    x = torch.tensor([ids], device="cuda")
    out = model(input_ids=x, position_ids=torch.tensor([pos], device="cuda"), past_key_values=cache,
                use_cache=True)
    gen = []
    for _ in range(max_new):
        t = int(out.logits[0, -1].float().argmax())
        gen.append(t)
        if t in stop:
            break
        pos_t = p
        if not (freeze and chip_lo <= t < chip_hi):
            p += 1
        out = model(input_ids=torch.tensor([[t]], device="cuda"),
                    position_ids=torch.tensor([[pos_t]], device="cuda"), past_key_values=cache,
                    use_cache=True)
    return gen


def main():
    from chipformer.q35 import evals
    from chipformer.q35.demo import NUDGE, load
    from chipformer.q35.lib import bcd
    from chipformer.ref.emu import Chip8
    from chipformer.trace import boot_tokens

    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("--idx", type=int, nargs="+", default=[0])
    ap.add_argument("--max-new", type=int, default=1200)
    ap.add_argument("--nudge", action="store_true")
    args = ap.parse_args()
    meta = json.loads((Path(args.merged) / "chipformer.json").read_text())
    off, nchip = meta["chip_token_offset"], meta["chip_vocab"]
    bios = [t + off for t in boot_tokens(Chip8(bcd.ROM, bcd.QUIRKS).cells)]
    model, tok = load(args.merged)
    data = evals.load_data("gsmhard", max(args.idx) + 1)
    stop = {248046, 248044}
    for i in args.idx:
        q = data[i]["question"]
        msgs = ([{"role": "system", "content": NUDGE}] if args.nudge else []) + [
            {"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      enable_thinking=False)["input_ids"][0].tolist()
        for freeze in (False, True):
            gen = decode(model, bios + ids, off, off + nchip, args.max_new, freeze, stop)
            text = tok.decode(gen)
            shown = re.sub(r"(<\|chip[^|]*\|>)+", "[m]", text)
            lines = [ln for ln in shown.split("\n") if ln.strip()]
            rep = max((lines.count(ln) for ln in lines), default=0)
            print(json.dumps({"idx": i, "freeze": freeze, "n": len(gen), "max_line_repeat": rep,
                              "gold": data[i].get("gold"),
                              "tail": shown[-300:]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
