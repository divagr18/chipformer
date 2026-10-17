"""Spike: does freezing RoPE positions on chip tokens stop Qwen repeating a line after an inline call?

    python -m chipformer.q35.spikes_pos MERGED --ids gsmhard-82 gsmhard-306 --max-new 1500

Decodes each problem twice with a manual greedy loop: positions advancing on every token (what
HF generate does), and positions frozen on chip tokens (Qwen's text sees contiguous positions).
Reports the most repeated line that holds an inline call.
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
    pos, p = [], 0
    for t in prompt:
        pos.append(p)
        if not (freeze and chip_lo <= t < chip_hi):
            p += 1
    out = model(input_ids=torch.tensor([prompt], device="cuda"),
                position_ids=torch.tensor([pos], device="cuda"), past_key_values=cache,
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
    from chipformer.q35.demo import load
    from chipformer.q35.lib import bcd
    from chipformer.ref.emu import Chip8
    from chipformer.trace import boot_tokens

    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("--ids", nargs="+", default=["gsmhard-82"])
    ap.add_argument("--cond", default="merged_inline")
    ap.add_argument("--max-new", type=int, default=1500)
    args = ap.parse_args()
    meta = json.loads((Path(args.merged) / "chipformer.json").read_text())
    off, nchip = meta["chip_token_offset"], meta["chip_vocab"]
    bios = [t + off for t in boot_tokens(Chip8(bcd.ROM, bcd.QUIRKS).cells)]
    model, tok = load(args.merged)
    data = {r["id"]: r for r in evals.load_data("gsmhard", 200)}
    stop = {248046, 248044, 248045}
    for i in args.ids:
        prompt = evals.build_prompt(tok, data[i]["question"], args.cond, bios)
        for freeze in (False, True):
            gen = decode(model, prompt, off, off + nchip, args.max_new, freeze, stop)
            shown = re.sub(r"(<\|chip[^|]*\|>)+", "[m]", tok.decode(gen))
            lines = [ln for ln in shown.split("\n") if "[m]" in ln]
            rep = max((lines.count(ln) for ln in lines), default=0)
            print(json.dumps({"id": i, "freeze": freeze, "n": len(gen), "fires": shown.count("[m]"),
                              "max_line_repeat": rep, "gold": data[i].get("gold"),
                              "tail": shown[-240:]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
