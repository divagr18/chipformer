"""Opcode histogram and trace statistics for a ROM under random input.

    uv run python -m chipformer.ref.profile roms/pong.ch8 --steps 200000
"""

import argparse
import random
from collections import Counter
from pathlib import Path

from chipformer.ref.emu import fetch, opcode_class
from chipformer.trace import Machine


def profile(rom: bytes, steps: int, ipf: int = 10, seed: int = 0):
    rng = random.Random(seed)
    m = Machine(rom)
    boot_len = len(m.tokens)
    ops: Counter[str] = Counter()
    for i in range(steps):
        if i % ipf == 0:
            m.tick()
            if rng.random() < 0.1:
                m.set_key(rng.randrange(16), rng.random() < 0.5)
        ops[opcode_class(fetch(m.emu.cells))] += 1
        m.step()
    return m, ops, boot_len


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rom", type=Path)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--ipf", type=int, default=10)
    args = ap.parse_args()

    m, ops, boot_len = profile(args.rom.read_bytes(), args.steps, args.ipf)
    total = sum(ops.values())
    print(f"{'opcode':8} {'count':>9} {'share':>7}")
    for op, cnt in ops.most_common():
        print(f"{op:8} {cnt:9,} {cnt / total:7.2%}")
    gen = len(m.tokens) - boot_len
    print(f"\nboot prompt: {boot_len:,} tokens")
    print(f"generated:   {gen:,} tokens over {total:,} instructions "
          f"({gen / total:.2f} tokens/instruction incl. ticks and input)")
    print(f"at {args.ipf} instr/frame, 60 Hz needs ~{gen / total * args.ipf * 60:,.0f} tok/s")


if __name__ == "__main__":
    main()
