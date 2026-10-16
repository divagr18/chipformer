"""Record a game played by a q35 checkpoint and render it to video at real-time speed.

    python -m chipformer.q35.record play MODEL tetris --seconds 60 --out run.npz   (GPU box)
    python -m chipformer.q35.record render run.npz run.mp4                         (needs ffmpeg)

``play`` runs the stock model with greedy decoding through ``fast.Runner``; the host only injects
TICKs every ``ipf`` instructions and key writes chosen by a simple bot from the screen. The trace
is verified record by record before it is saved. ``render`` replays the model's own writes and
shows one frame per TICK, i.e. at the speed the game would run on a real CHIP-8.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from chipformer import spec
from chipformer.ref.emu import apply_writes, display_pixels

ROOT = Path(__file__).resolve().parents[2]


class TetrisBot:
    """Port of the browser demo's recorder bot: probe rotations, place by a heuristic score."""

    W0, COLS, ROWS, TOP = 27, 10, 31, 6

    def __init__(self, keys):
        self.k = keys
        self.was_top = False
        self.state, self.wait = "idle", 0
        self.stack = [[0] * 10 for _ in range(31)]

    def grid(self, px):
        return [[px[y][self.W0 + x] for x in range(self.COLS)] for y in range(self.ROWS)]

    @staticmethod
    def evaluate(st, cells, x):
        w = max(c[1] for c in cells) + 1
        h = max(c[0] for c in cells) + 1
        if x + w > 10:
            return None
        y = -h

        def fits(yy):
            return all(yy + dy < 31 and (yy + dy < 0 or not st[yy + dy][x + dx])
                       for dy, dx in cells)

        while fits(y + 1):
            y += 1
        if any(y + dy < 0 for dy, _ in cells):
            return None
        g = [r[:] for r in st]
        for dy, dx in cells:
            g[y + dy][x + dx] = 1
        lines = 0
        r = 30
        while r >= 0:
            if all(g[r]):
                del g[r]
                g.insert(0, [0] * 10)
                lines += 1
            else:
                r -= 1
        hs = [next((31 - r for r in range(31) if g[r][c]), 0) for c in range(10)]
        holes = 0
        for c in range(10):
            seen = False
            for r in range(31):
                if g[r][c]:
                    seen = True
                elif seen:
                    holes += 1
        bump = sum(abs(hs[i] - hs[i + 1]) for i in range(9))
        return -0.51 * sum(hs) + 0.76 * lines - 0.72 * holes - 0.18 * bump

    def frame(self, px) -> set:
        """Keys to hold this frame (closed loop on the screen, like the browser bot)."""
        g = self.grid(px)
        top = [(y, x) for y in range(self.TOP) for x in range(10) if g[y][x]]
        spawned = bool(top) and not self.was_top
        self.was_top = bool(top)
        if spawned:
            self.stack = [r[:] if y >= self.TOP else [0] * 10 for y, r in enumerate(g)]
            self.state, self.wait = "probe", 0
            self.shapes = []
        if self.state == "idle":
            return set()
        piece = [(y, x) for y in range(31) for x in range(10)
                 if g[y][x] and not self.stack[y][x] and y < 16]
        if self.wait > 0:
            self.wait -= 1
            return set()
        if self.state == "probe":  # record each rotation's shape, then pick the best placement
            if len(piece) == 4:
                y0, x0 = min(c[0] for c in piece), min(c[1] for c in piece)
                self.shapes.append(sorted((y - y0, x - x0) for y, x in piece))
            if len(self.shapes) < 4:
                self.wait = 3
                return {self.k["a"]}
            best = None
            for r, cs in enumerate(self.shapes):
                for x in range(10):
                    v = self.evaluate(self.stack, cs, x)
                    if v is not None and (best is None or v > best[0]):
                        best = (v, r, x)
            if best is None:
                self.state = "drop"
                return set()
            _, self.rot, self.target = best
            self.state = "rotate"
            return set()
        if self.state == "rotate":
            if self.rot > 0:
                self.rot -= 1
                self.wait = 3
                return {self.k["a"]}
            self.state = "move"
        if self.state == "move":
            if len(piece) != 4:
                return set()
            x = min(c[1] for c in piece)
            if x < self.target:
                return {self.k["right"]}
            if x > self.target:
                return {self.k["left"]}
            self.state = "drop"
        if self.state == "drop":
            return {self.k["down"]}
        return set()


def play(model_dir: str, game: str, seconds: float, out: str, max_len: int = 65536) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    from chipformer.check import verify_trace
    from chipformer.q35.fast import Runner

    meta_path = Path(model_dir) / "chipformer.json"
    merged = meta_path.exists()
    cls = AutoModelForImageTextToText if merged else AutoModelForCausalLM
    model = cls.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    off = json.loads(meta_path.read_text())["chip_token_offset"] if merged else 0
    games = {g["id"]: g for g in json.loads((ROOT / "games" / "manifest.json").read_text(
        encoding="utf-8"))["games"]}
    g = games[game]
    runner = Runner(model, max_len=max_len, token_offset=off)
    s = runner.session((ROOT / "games" / g["file"]).read_bytes(), tuple(g["quirks"]))
    bot = TetrisBot(g["keys"]) if game == "tetris" else None
    held: set = set()
    frames = int(seconds * 60)
    t = time.time()
    for f in range(frames):
        want = bot.frame(s.pixels()) if bot else set()
        for k in held - want:
            s.set_key(k, False)
        for k in want - held:
            s.set_key(k, True)
        held = want
        s.tick()
        for _ in range(g["ipf"]):
            s.step()
        if f % 600 == 0:
            el = time.time() - t
            print(f"frame {f}/{frames}  {len(s.tokens)} tokens  {el:.0f}s", flush=True)
    n, err = verify_trace(s.tokens)
    print(f"verified {n} records, error: {err}, {time.time() - t:.0f}s")
    np.savez_compressed(out, tokens=np.array(s.tokens, dtype=np.uint32), game=game,
                        model=str(model_dir), verified=err is None)


def render(trace: str, out: str, scale: int = 12) -> None:
    from chipformer.trace import parse

    data = np.load(trace)
    tokens = data["tokens"].tolist()
    cells = [0] * spec.NUM_CELLS
    frames, prev = [], None
    for r in parse(tokens, allow_reboot=True):
        if r.kind == "boot":
            cells = [0] * spec.NUM_CELLS
        apply_writes(cells, r.writes)
        if r.kind == "tick":
            px = np.array(display_pixels(cells), dtype=np.uint8)
            shown = px if prev is None else np.maximum(px, prev)  # soften CHIP-8 flicker
            frames.append(shown)
            prev = px
    h, w = 32 * scale, 64 * scale
    ink, lit = np.array([25, 34, 31], np.uint8), np.array([221, 245, 158], np.uint8)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "60", "-i", "-",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out],
                            stdin=subprocess.PIPE)
    for px in frames:
        img = np.where(px[..., None].astype(bool), lit, ink)
        img = img.repeat(scale, 0).repeat(scale, 1)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"{out}: {len(frames)} frames ({len(frames) / 60:.1f} s at 60 fps)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("model")
    p.add_argument("game")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--out", default="run.npz")
    r = sub.add_parser("render")
    r.add_argument("trace")
    r.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "play":
        play(a.model, a.game, a.seconds, a.out)
    else:
        render(a.trace, a.out)


if __name__ == "__main__":
    main()
