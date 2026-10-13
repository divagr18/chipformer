"""Publish the demo assets into the website, or assemble the Hugging Face Space.

    uv run python -m chipformer.web build --blog D:/Blog
    uv run python -m chipformer.web space --out dist/space [--build-wasm]

`build` builds the WebAssembly engine, then writes:
  <blog>/public/chipformer-assets/   engine.wasm, model.chipformer, roms/, thumbs/, ATTRIBUTION.md
  <blog>/components/chipformer/data.ts   generated game list, machine constants, model facts
`space` writes a self-contained static site: the sources in space/ plus the same assets and a
generated data.js. It reuses the built engine.wasm unless --build-wasm is given.
The weights repo stays the source of truth; the blog and the Space only receive build outputs.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from chipformer.games import GAMES, load_manifest, write_attribution

ROOT = Path(__file__).resolve().parent.parent
WASM = ROOT / "engine" / "target" / "wasm32-unknown-unknown" / "release" / "chipformer_engine.wasm"
PHOSPHOR, SCREEN_BG = (221, 245, 158), (11, 16, 13)
THUMB_INSTR = {"blinky": 20000, "super-pong": 800, "tetris": 120000}  # a more telling frame


def build_wasm() -> None:
    subprocess.run(["cargo", "build", "--release", "--lib", "--target", "wasm32-unknown-unknown",
                    "--no-default-features"], cwd=ROOT / "engine", check=True)


def thumbnail(game: dict, out: Path, instr: int = 2500) -> None:
    """Render a frame of the game from a verified engine run, in the site's phosphor palette."""
    from PIL import Image

    from chipformer import spec
    from chipformer.check import verify_trace
    from chipformer.ref.emu import apply_writes, display_pixels
    from chipformer.trace import parse

    exe = ROOT / "engine" / "target" / "release" / ("engine.exe" if os.name == "nt" else "engine")
    dump = out.with_suffix(".u32")
    subprocess.run([str(exe), "run", str(ROOT / "model"), str(GAMES / game["file"]),
                    "--instr", str(instr), "--ipf", str(game["ipf"]), "--quirks", ",".join(game["quirks"]),
                    "--seed", "11", "--dump", str(dump)], check=True, capture_output=True)
    tokens = np.fromfile(dump, dtype="<u4").tolist()
    dump.unlink()
    _, err = verify_trace(tokens)
    if err:
        raise RuntimeError(f"{game['id']}: trace does not verify: {err}")
    cells = [0] * spec.NUM_CELLS
    for rec in parse(tokens, allow_reboot=True):
        apply_writes(cells, rec.writes)
    img = Image.new("RGB", (spec.DISPLAY_W, spec.DISPLAY_H))
    img.putdata([PHOSPHOR if p else SCREEN_BG for row in display_pixels(cells) for p in row])
    img.resize((spec.DISPLAY_W * 8, spec.DISPLAY_H * 8), Image.NEAREST).save(out)


def publish_assets(assets: Path) -> dict:
    """Copy the engine, model, featured ROMs, thumbnails and credits into `assets`; return the
    generated site data (games, layers, model facts, machine constants)."""
    from chipformer.compile import load_sparse

    for sub in ("roms", "thumbs"):
        (assets / sub).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(WASM, assets / "engine.wasm")
    shutil.copyfile(ROOT / "model" / "model.chipformer", assets / "model.chipformer")
    _, config = load_sparse(ROOT / "model" / "model.chipformer")

    manifest = load_manifest()
    featured = sorted((g for g in manifest if "featured" in g), key=lambda g: g["featured"])
    games = []
    for g in featured:
        shutil.copyfile(GAMES / g["file"], assets / "roms" / f"{g['id']}.ch8")
        thumbnail(g, assets / "thumbs" / f"{g['id']}.png", THUMB_INSTR.get(g["id"], 2500))
        games.append({k: g.get(k) for k in ("id", "title", "authors", "year", "description", "license",
                                           "source", "quirks", "ipf", "keys", "controls", "tap")} | {"gameOver": g.get("game_over")})
    write_attribution(featured, assets / "ATTRIBUTION.md")

    digest = hashlib.sha1((assets / "engine.wasm").read_bytes()
                          + (assets / "model.chipformer").read_bytes()).hexdigest()[:10]
    m = config["machine"]
    facts = {
        "layers": config["n_layers"], "dModel": config["d_model"],
        "heads": sum(lay["heads"] for lay in config["layers"]),
        "ffnUnits": sum(lay["ffn_units"] for lay in config["layers"]),
        "params": config["params"], "nonzero": config["nonzero_params"],
        "vocab": config["vocab_size"], "modelBytes": (assets / "model.chipformer").stat().st_size,
        "wasmBytes": (assets / "engine.wasm").stat().st_size,
    }
    return {
        "digest": digest,
        "machine": {k: m[k] for k in ("VAL_BASE", "ADDR_BASE", "STEP", "TICK", "BOOT", "NULL", "CLS", "PC",
                                      "RNG", "KEYS", "DISPLAY", "QUIRKS", "QUIRK_NAMES")},
        "facts": facts,
        "library": {"total": len(manifest), "verified": sum(g.get("status") == "ok" for g in manifest)},
        "layers": [{"heads": lay["head_names"], "units": lay["ffn_units"]} for lay in config["layers"]],
        "games": games,
    }


def build(blog: Path) -> None:
    build_wasm()
    assets = blog / "public" / "chipformer-assets"
    d = publish_assets(assets)
    digest, machine, facts, library, layers, games = (
        d[k] for k in ("digest", "machine", "facts", "library", "layers", "games"))
    ts = "\n".join([
        "// Generated by chipformer/web.py in the chipformer repo. Do not edit by hand.",
        "",
        'export const ASSET_BASE = "/chipformer-assets";',
        f'export const ASSET_VERSION = "{digest}";',
        "",
        f"export const MACHINE = {json.dumps(machine, indent=2)} as const;",
        "",
        f"export const MODEL_FACTS = {json.dumps(facts, indent=2)} as const;",
        "",
        f"export const LIBRARY = {json.dumps(library)} as const;",
        "",
        "/** Per layer: the names of its attention heads (in model order) and its FFN unit count. */",
        f"export const LAYERS: {{ heads: string[]; units: number }}[] = {json.dumps(layers)};",
        "",
        "export type KeyRole = \"up\" | \"down\" | \"left\" | \"right\" | \"a\" | \"b\" | \"player2Up\" | \"player2Down\";",
        "",
        "export type Game = {",
        "  id: string;",
        "  title: string;",
        "  authors: string[];",
        "  year: string | null;",
        "  description: string;",
        "  license: string;",
        "  source: string;",
        "  quirks: string[];",
        "  ipf: number;",
        "  keys: Partial<Record<KeyRole, number>>;",
        "  controls: string;",
        "  tap: KeyRole[];",
        "  gameOver: { rows: number[]; cols: number[]; seconds: number; message: string } | null;",
        "};",
        "",
        f"export const GAMES: Game[] = {json.dumps(games, indent=2, ensure_ascii=False)};",
        "",
    ])
    (blog / "components" / "chipformer").mkdir(parents=True, exist_ok=True)
    (blog / "components" / "chipformer" / "data.ts").write_text(ts, encoding="utf-8")
    print(f"published {len(games)} games, engine {facts['wasmBytes'] / 1e3:.0f} kB, "
          f"model {facts['modelBytes'] / 1e6:.2f} MB -> {assets}")


def space(out: Path, rebuild_wasm: bool = False) -> None:
    """Assemble the static Hugging Face Space: space/ sources + assets + a generated data.js."""
    if rebuild_wasm or not WASM.exists():
        build_wasm()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "space", out)
    d = publish_assets(out / "assets")
    js = "\n".join([
        "// Generated by chipformer/web.py (python -m chipformer.web space). Do not edit by hand.",
        "",
        'export const ASSET_BASE = "./assets";',
        f'export const ASSET_VERSION = "{d["digest"]}";',
        f"export const MACHINE = {json.dumps(d['machine'])};",
        f"export const MODEL_FACTS = {json.dumps(d['facts'])};",
        f"export const LIBRARY = {json.dumps(d['library'])};",
        "/** Per layer: the names of its attention heads (in model order) and its FFN unit count. */",
        f"export const LAYERS = {json.dumps(d['layers'])};",
        f"export const GAMES = {json.dumps(d['games'], indent=1, ensure_ascii=False)};",
        "",
    ])
    (out / "data.js").write_text(js, encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"space: {len(d['games'])} games, {size / 1e6:.2f} MB -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "space"])
    ap.add_argument("--blog", type=Path, help="build: the website repo to publish into")
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "space", help="space: output directory")
    ap.add_argument("--build-wasm", action="store_true", help="space: rebuild engine.wasm first")
    args = ap.parse_args()
    if args.command == "build":
        if not args.blog:
            ap.error("build needs --blog")
        build(args.blog)
    else:
        space(args.out, args.build_wasm)


if __name__ == "__main__":
    main()
