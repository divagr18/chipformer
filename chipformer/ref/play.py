"""Play a ROM on the reference emulator.

    uv run --extra play python -m chipformer.ref.play roms/pong.ch8

Keypad layout (CHIP-8 -> keyboard):
    1 2 3 C      1 2 3 4
    4 5 6 D  ->  Q W E R
    7 8 9 E      A S D F
    A 0 B F      Z X C V
"""

import argparse
import sys
from pathlib import Path

from chipformer import spec
from chipformer.trace import Machine, replay

KEYMAP = {
    "1": 0x1, "2": 0x2, "3": 0x3, "4": 0xC,
    "q": 0x4, "w": 0x5, "e": 0x6, "r": 0xD,
    "a": 0x7, "s": 0x8, "d": 0x9, "f": 0xE,
    "z": 0xA, "x": 0x0, "c": 0xB, "v": 0xF,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rom", type=Path)
    ap.add_argument("--ipf", type=int, default=10, help="instructions per 60 Hz frame")
    ap.add_argument("--scale", type=int, default=12)
    ap.add_argument("--check", action="store_true", help="replay the trace on exit and verify")
    args = ap.parse_args()

    import pygame

    m = Machine(args.rom.read_bytes())
    pygame.init()
    screen = pygame.display.set_mode((spec.DISPLAY_W * args.scale, spec.DISPLAY_H * args.scale))
    pygame.display.set_caption(f"chipformer reference - {args.rom.name}")
    clock = pygame.time.Clock()
    keycodes = {pygame.key.key_code(k): v for k, v in KEYMAP.items()}

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type in (pygame.KEYDOWN, pygame.KEYUP) and ev.key in keycodes:
                m.set_key(keycodes[ev.key], ev.type == pygame.KEYDOWN)
        m.tick()
        for _ in range(args.ipf):
            m.step()
        screen.fill((16, 16, 24))
        for y, row in enumerate(m.emu.pixels()):
            for x, p in enumerate(row):
                if p:
                    screen.fill((230, 230, 210),
                                (x * args.scale, y * args.scale, args.scale, args.scale))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

    print(f"trace: {len(m.tokens):,} tokens")
    if args.check:
        ok = replay(m.tokens) == m.emu.cells
        print("replay check:", "OK" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
