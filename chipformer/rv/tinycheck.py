"""P1.5 checks of rv-tiny (a stock Qwen3_5ForCausalLM):

    python -m chipformer.rv.tinycheck [--save dist/rv-tiny] [--device cuda]

1. The residual before the exit layer equals the IR evaluator's on every circuit dim, at every
   position (bf16; one prefill with the snapshot in the cache).
2. Free-running ``generate`` from GO reproduces the reference trace token for token.
"""

import argparse
import time

import torch

from chipformer.q35 import ir
from chipformer.rv import core, elf, firmware, ref, tiny
from chipformer.rv import spec as S
from chipformer.rv.measure import DATA
from chipformer.rv.programs import program


def residual_check(model, c, prog) -> int:
    dev = next(model.parameters()).device
    toks = ref.Machine.from_program(prog).run(10**6)
    core.snapshot(c, ref.Machine.from_program(prog))
    want = ir.evaluate(c, toks, check=False)
    cache = firmware.load_cache(model, c, ref.Machine.from_program(prog))
    with torch.no_grad():
        o = model(input_ids=torch.tensor([toks], device=dev), past_key_values=cache,
                  use_cache=True, output_hidden_states=True)
    h = o.hidden_states[c.n_layers() - 1][0].double().cpu().numpy()
    lay = model.layout
    dims = [d for d in range(1, len(c.dims))
            if c.dims[d].level < 2 * c.n_layers() - 1 and d not in c.keep]
    bad = [c.dims[d].name for d in dims if (h[:, lay.dim(d)] != want[:, d]).any()]
    assert not bad, bad[:5]
    return len(dims) * len(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--progs", default="signext,call,memcpy,rv32ui-lb,rv32um-mulh,c-crc32p,"
                                       "c-strops")
    args = ap.parse_args()
    t = time.time()
    model, c = tiny.build()
    n = sum(p.numel() for p in model.parameters())
    print(f"built rv-tiny: hidden {model.layout.hidden}, {c.n_layers() + 1} layers, "
          f"{n / 1e6:.0f}M parameters ({time.time() - t:.0f} s)", flush=True)
    if args.save:
        import json
        from pathlib import Path

        model.save_pretrained(args.save)
        meta = {"vocab": S.VOCAB, "pad": S.VOCAB, "families": {k: list(v) for k, v in
                                                                 S.FAMILIES.items()},
                "controls": S.CONTROLS, "circuit": c.stats(), "hidden": model.layout.hidden}
        (Path(args.save) / "chipformer.json").write_text(json.dumps(meta, indent=1))
        print(f"saved {args.save}", flush=True)
    model = model.to(args.device)
    for name in args.progs.split(","):
        prog = program(name) if "-" not in name else elf.program(DATA / f"{name}.elf")
        t = time.time()
        cells = residual_check(model, c, prog)
        want = ref.Machine.from_program(prog).run(10**6)
        got = firmware.run(model, c, ref.Machine.from_program(prog), max_new=len(want) + 5)
        ok = got == want[1:]
        out = firmware.output_text(got)
        print(f"{name:14s} residual == IR ({cells} cells); generate == reference: {ok} "
              f"({len(got)} tokens{', output ' + repr(out[:40]) if out else ''}; "
              f"{time.time() - t:.1f} s)", flush=True)
        assert ok


if __name__ == "__main__":
    main()
