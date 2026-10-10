import random
from pathlib import Path

import pytest

from chipformer import spec
from chipformer.ref.emu import Chip8
from chipformer.spec import DISPLAY, KEYS, PC, SP, STACK, I, V
from chipformer.trace import Machine, parse, replay

ROMS = Path(__file__).resolve().parent.parent / "roms"


def prog(*words: int) -> bytes:
    return b"".join(w.to_bytes(2, "big") for w in words)


def run(rom: bytes, steps: int) -> Chip8:
    emu = Chip8(rom)
    for _ in range(steps):
        emu.step()
    return emu


def test_lfsr_is_maximal():
    s, seen = spec.RNG_SEED, set()
    while s not in seen:
        seen.add(s)
        s = spec.lfsr_step(s)
    assert len(seen) == 4095 and 0 not in seen


def test_vocab_disjoint():
    assert spec.addr_token(0) == spec.val_token(spec.MAX_VALUE) + 1
    assert spec.STEP == spec.addr_token(spec.NUM_CELLS - 1) + 1
    assert spec.VOCAB_SIZE == spec.BOOT + 1


def test_alu_and_flags():
    emu = run(prog(0x60FF, 0x6102, 0x8014), 3)  # V0=FF, V1=2, V0+=V1
    assert emu.cells[V + 0] == 0x01 and emu.cells[V + 0xF] == 1
    emu = run(prog(0x6005, 0x6107, 0x8015), 3)  # 5-7 borrows
    assert emu.cells[V + 0] == 0xFE and emu.cells[V + 0xF] == 0
    emu = run(prog(0x6F05, 0x6107, 0x8F15), 3)  # flag wins when X=F
    assert emu.cells[V + 0xF] == 0
    emu = run(prog(0x6081, 0x800E), 2)
    assert emu.cells[V + 0] == 0x02 and emu.cells[V + 0xF] == 1


def test_call_return_and_skips():
    #  200: call 206 | 202: 6A01 | 204: jump 204 | 206: 3000 (skip) | 208: 6B01 | 20A: ret
    emu = run(prog(0x2206, 0x6A01, 0x1204, 0x3000, 0x6B01, 0x00EE), 3)  # call, skip, ret
    assert emu.cells[PC] == 0x202 and emu.cells[SP] == 0 and emu.cells[STACK] == 0x202
    assert emu.cells[V + 0xB] == 0
    emu.step()
    assert emu.cells[V + 0xA] == 1


def test_bcd_store_load():
    emu = run(prog(0x60EA, 0xA300, 0xF033, 0xF265), 4)  # 234 -> 2,3,4
    assert emu.cells[0x300:0x303] == [2, 3, 4]
    assert emu.cells[V:V + 3] == [2, 3, 4] and emu.cells[I] == 0x300


def test_draw_straddle_collision_and_clip():
    # draw font '0' at x=4,y=0 twice -> second draw erases and collides
    emu = run(prog(0x6004, 0x6100, 0xF129,0xD015, 0xD015), 4)
    assert emu.cells[V + 0xF] == 0
    assert emu.cells[DISPLAY] == 0x0F and emu.cells[DISPLAY + 1] == 0x00  # F0 >> 4
    emu.step()
    assert emu.cells[V + 0xF] == 1 and not any(emu.cells[DISPLAY:DISPLAY + 256])
    # clipped at bottom-right: x=62, y=30, 5 rows -> only 2 rows, 2 pixels each visible
    emu = run(prog(0x603E, 0x611E, 0xA050, 0xD015), 4)
    assert emu.cells[DISPLAY + 30 * 8 + 7] == 0x03 and emu.cells[DISPLAY + 31 * 8 + 7] == 0x02


def test_keys_and_wait():
    emu = Chip8(prog(0xF30A, 0x1202))
    emu.step()
    assert emu.cells[PC] == 0x200  # no key: stays
    emu.set_key(7, True)
    emu.step()
    assert emu.cells[PC] == 0x202 and emu.cells[V + 3] == 7
    assert emu.cells[KEYS + 7] == 1


def random_trace(rom: bytes, steps: int, seed: int = 0) -> Machine:
    rng = random.Random(seed)
    m = Machine(rom)
    for i in range(steps):
        if i % 10 == 0:
            m.tick()
            m.set_key(rng.randrange(16), rng.random() < 0.5)
        m.step()
    return m


def test_trace_replay_matches_emulator():
    rng = random.Random(1)
    rom = bytes(rng.randrange(256) for _ in range(1024))  # random code exercises every opcode
    m = random_trace(rom, 3000)
    parse(m.tokens)
    assert replay(m.tokens) == m.emu.cells


def test_trace_replay_every_record():
    m = Machine(prog(0x6004, 0x6100, 0xA050, 0xD015, 0x7001, 0x1206))
    for _ in range(50):
        m.step()
        assert replay(m.tokens) == m.emu.cells


def test_boot_prompt_restores_state():
    m = random_trace(bytes(random.Random(2).randrange(256) for _ in range(512)), 500)
    restart = Machine(b"")
    from chipformer.trace import boot_tokens

    assert replay(boot_tokens(m.emu.cells)) == m.emu.cells
    assert restart.emu.cells != m.emu.cells


@pytest.mark.parametrize("rom", sorted((ROMS / "test").glob("*.ch8")) if (ROMS / "test").exists()
                         else [], ids=lambda p: p.name)
def test_rom_trace_roundtrip(rom: Path):
    m = random_trace(rom.read_bytes(), 20_000)
    assert replay(m.tokens) == m.emu.cells
