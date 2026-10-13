"""Write a Hugging Face model repo for chipformer (transformers, trust_remote_code).

    uv run python -m chipformer.hf_export --out dist/hf

The weights are compiled fresh from the circuits (0.3 s), so the export never depends on a
stale model/ directory. The repo holds config.json (with auto_map), model.safetensors (fp64),
the configuration/modeling code from hf/, the sparse model.chipformer used by the browser
engine, a model card and a Tetris example.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from chipformer import spec
from chipformer.dsl.lib import MAX_POS

ROOT = Path(__file__).resolve().parent.parent
HF_SRC = ROOT / "hf"
GAMES = ROOT / "games"
ROLLOVER_AT = 250_000  # same as the engine
LINKS = {
    "github": "https://github.com/divagr18/chipformer",
    "space": "https://huggingface.co/spaces/divagr1925/chipformer",
    "demo": "https://divagr.com/chipformer",
}
EXAMPLE_GAME = "tetris"


def hf_config(w: dict[str, np.ndarray], config: dict) -> dict:
    layers = []
    for i, lay in enumerate(config["layers"]):
        heads = [[int(w[f"layers.{i}.attn.{h}.wq"].shape[0]), int(w[f"layers.{i}.attn.{h}.wv"].shape[0])]
                 for h in range(lay["heads"])]
        layers.append({"heads": heads, "ffn_units": lay["ffn_units"], "head_names": lay["head_names"]})
    return {
        "architectures": ["ChipformerForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_chipformer.ChipformerConfig",
            "AutoModelForCausalLM": "modeling_chipformer.ChipformerForCausalLM",
        },
        "model_type": "chipformer",
        "d_model": config["d_model"],
        "vocab_size": config["vocab_size"],
        "beta": config["beta"],
        "layers": layers,
        "machine": config["machine"],
        "max_position": MAX_POS,
        "rollover_at": ROLLOVER_AT,
        "params": config["params"],
        "nonzero_params": config["nonzero_params"],
        "dtype": "float64",
        "torch_dtype": "float64",
        "tie_word_embeddings": False,
    }


def example_game() -> dict:
    games = json.loads((GAMES / "manifest.json").read_text())
    games = games["games"] if isinstance(games, dict) else games
    return next(g for g in games if g["id"] == EXAMPLE_GAME)


def model_card(config: dict, game: dict) -> str:
    heads = sum(lay["heads"] for lay in config["layers"])
    units = sum(lay["ffn_units"] for lay in config["layers"])
    keys = game["keys"]
    return f"""---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
- chip-8
- emulator
- mechanistic-interpretability
- compiled-transformer
- custom_code
---

# chipformer

**Tetris, running entirely inside a transformer.**

This model *is* a CHIP-8 computer. Give it a game as the prompt and greedy decoding runs the game:
every token it predicts is one write to the machine's memory (a register, the program counter, a
row of pixels). Paint those writes onto a 64×32 screen and you are playing Tetris. Your keypresses
and the 60 Hz clock go into the context window. Everything else, every frame, every collision,
every point on the scoreboard, comes out of the model.

There is no emulator behind it. It is the same loop every language model runs: read the context,
predict the next token, append it, repeat.

- One set of weights plays Tetris, Space Invaders, Brix, Pong and 100+ other games. Switching games
  means switching the prompt.
- An ordinary decoder-only architecture: {config['n_layers']} layers, d_model {config['d_model']},
  {heads} attention heads, {units:,} ReGLU units, vocabulary {config['vocab_size']:,}, float64.

**Play it in your browser:** [{LINKS['demo'].removeprefix('https://')}]({LINKS['demo']}) ·
[Hugging Face Space]({LINKS['space']}) · Source: [GitHub]({LINKS['github']})

## How it works

- **The ROM is the prompt.** The prompt is `BOOT`, then one `ADDR a, VAL v` token pair per nonzero
  byte of machine state, then `STEP`.
- **Every output token is a write.** Each instruction becomes a short record of writes such as
  `@V3 =1F @PC =22A STEP`. The screen is drawn from these writes.
- **The host injects the outside world.** A `TICK` token arrives once per 60 Hz frame, and keypresses
  are injected as writes. Every other token is the model's greedy prediction.
- **Attention is memory.** To read register V3, a head attends to the most recent write to V3.
  Keys use a parabola over the address plus a recency term, so the hardmax lands exactly there.
- **ReLU is the ALU.** Addition, comparison, XOR and opcode decoding are all built from ReLUs,
  using identities like step(x) = ReLU(x+1) - ReLU(x).

The architecture is ordinary:
- token embedding plus a position embedding from [t, t²]
- causal softmax attention (β = {config['beta']:g}) with residual connections
- ReGLU feed-forward layers
- a linear unembedding
- no normalization

`modeling_chipformer.py` contains nothing CHIP-8 specific apart from the host loop, which only feeds
tokens in and reads the screen back.

**Where the weights come from.** They are compiled from a description of the CHIP-8 circuit
(written in a small DSL in the GitHub repo) rather than learned, which is what makes every
instruction exact. {config['nonzero_params']:,} of the {config['params']:,} parameters are nonzero.

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

repo = "divagr1925/chipformer"
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.float64)

rom = open(hf_hub_download(repo, "roms/{game['id']}.ch8"), "rb").read()
game = model.play(rom, quirks="{','.join(game['quirks'])}", ipf={game['ipf']})
for frame in range(120):
    game.frame(keys={{{keys['left']}}} if frame % 20 < 4 else ())  # CHIP-8 key {keys['left']} = left
print(game.ascii())
```

`play_tetris.py` animates the game in a terminal: `python play_tetris.py`.

The model is a normal `PreTrainedModel`, so you can also call it directly:
- `model(input_ids, past_key_values=cache)` returns next-token logits over the {config['vocab_size']:,}-token vocabulary.
- `game.tokens` (with `keep_tokens=True`) gives you the full trace.

### Notes

- **Use float64.** The memory-read heads compare attention scores of about 10¹³, and these are only
  exact in float64. fp16 and bf16 will break the machine.
- **Speed.** Decoding is greedy, one token at a time with a KV cache. On a CPU in PyTorch that is
  about 200 tokens/s, or 2–3 frames per second of Tetris (the game wants 60).
  - `game = model.play(...)` multiplies by sparse copies of the big feed-forward matrices, since only
    0.4% of the weights are nonzero. The outputs are identical; pass `sparse=False` to run the dense
    weights exactly as loaded.
  - The Rust/wasm engine in the GitHub repo uses the same weights with sparse matmuls and exact
    attention shortcuts. It runs the games at full speed in a browser.
- **Context.** Positions must stay below {MAX_POS:,} (2²²). `ChipformerGame` restarts the context
  from a state dump every {ROLLOVER_AT:,} tokens, as the engine does.
- **Coverage.** The weights implement the original CHIP-8 instruction set with Octo's quirk flags.
  SCHIP/XO-CHIP opcodes and the vBlank display-wait quirk are not supported.

## Files

| File | |
|---|---|
| `model.safetensors` | dense float64 weights (what `transformers` loads) |
| `model.chipformer` | the same weights, sparse (nonzeros only), for the Rust/wasm engine |
| `configuration_chipformer.py`, `modeling_chipformer.py` | the architecture plus the host loop |
| `play_tetris.py` | a terminal demo |
| `roms/{game['id']}.ch8` | {game['title']} by {', '.join(game['authors'])} ({game['year']}), {game['license']} |
"""


EXAMPLE = '''"""Watch chipformer play Tetris in your terminal. Every pixel is a token the model predicted.

    python play_tetris.py                 # from the Hugging Face Hub
    python play_tetris.py path/to/repo    # from a local copy
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.stdout.reconfigure(encoding="utf-8")  # the screen is drawn with block characters
REPO = sys.argv[1] if len(sys.argv) > 1 else "divagr1925/chipformer"
FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 400
LEFT, RIGHT, DOWN, ROTATE = __KEYS__

model = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True, dtype=torch.float64)
if Path(REPO).is_dir():
    rom = Path(REPO, "roms/tetris.ch8").read_bytes()
else:
    from huggingface_hub import hf_hub_download

    rom = Path(hf_hub_download(REPO, "roms/tetris.ch8")).read_bytes()

game = model.play(rom, quirks="__QUIRKS__", ipf=__IPF__)
# a little scripted play: rotate (a tap), slide, drop
plan = {0: [ROTATE], 1: [], 6: [LEFT], 7: [], 8: [LEFT], 9: [], 14: [DOWN]}
t0, fed0 = time.time(), game.fed
print("\\x1b[2J", end="")
for f in range(FRAMES):
    keys = plan.get(f % 40, [] if f % 40 < 14 else [DOWN])
    game.frame(keys)
    rate = (game.fed - fed0) / (time.time() - t0)
    print(f"\\x1b[H{game.ascii()}\\nframe {f + 1}/{FRAMES}  ·  {game.instructions} instructions  ·  "
          f"{rate:,.0f} tokens/s  ·  position {game.position:,}", flush=True)
'''


def export(out: Path) -> None:
    from safetensors.numpy import save_file

    from chipformer.chip8 import build
    from chipformer.compile import compile_graph, save_sparse

    out.mkdir(parents=True, exist_ok=True)
    w, config = compile_graph(build())
    assert config["vocab_size"] == spec.VOCAB_SIZE
    save_file({k: np.ascontiguousarray(v) for k, v in w.items()}, out / "model.safetensors",
              metadata={"format": "pt"})
    save_sparse(w, config, out / "model.chipformer")
    (out / "config.json").write_text(json.dumps(hf_config(w, config), indent=1))
    for f in ("configuration_chipformer.py", "modeling_chipformer.py"):
        shutil.copy(HF_SRC / f, out / f)

    game = example_game()
    (out / "roms").mkdir(exist_ok=True)
    shutil.copy(GAMES / game["file"], out / "roms" / f"{game['id']}.ch8")
    keys = game["keys"]
    (out / "play_tetris.py").write_text(
        EXAMPLE.replace("__KEYS__", f"{keys['left']}, {keys['right']}, {keys['down']}, {keys['a']}")
        .replace("__QUIRKS__", ",".join(game["quirks"])).replace("__IPF__", str(game["ipf"])),
        encoding="utf-8")
    (out / "README.md").write_text(model_card(config, game), encoding="utf-8")
    (out / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        "*.chipformer filter=lfs diff=lfs merge=lfs -text\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export chipformer as a Hugging Face model repo")
    ap.add_argument("--out", type=Path, default=Path("dist/hf"))
    out = ap.parse_args().out
    export(out)
    files = [f for f in out.rglob("*") if f.is_file()]
    print(f"{out}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
