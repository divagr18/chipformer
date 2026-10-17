"""The RV32 reference: ISA emulator == qemu-riscv32, traced machine == ISA emulator, records."""

import json
import os
import random
from pathlib import Path

import pytest

from chipformer.rv import elf, ref
from chipformer.rv import spec as S
from chipformer.rv.programs import ALL, program

DATA = Path(__file__).parent / "data" / "rv"
QEMU = json.loads((DATA / "qemu.json").read_text())
LONG = {"c-sudoku", "c-fib"}  # 8.4M / 2.1M instructions: RV_SLOW=1


def run_rv32(path, sp):
    cpu = ref.Rv32()
    elf.load_into(cpu, path)
    cpu.x[2] = sp
    while not cpu.halted and cpu.steps < 2 * 10**7:
        cpu.step()
    return cpu


def slow(name):
    if name in LONG and not os.environ.get("RV_SLOW"):
        pytest.skip("long run (RV_SLOW=1)")


@pytest.mark.parametrize("name", sorted(QEMU))
def test_against_qemu(name):
    """Same code at qemu's addresses: exit code and output equal qemu's (built by
    chipformer/rv/env/build_tests.sh in WSL)."""
    slow(name)
    cpu = run_rv32(DATA / f"{name}.qemu.elf", 0x7FFF0000)
    assert cpu.exit_code & 0xFF == QEMU[name]["exit"]
    assert cpu.out.decode("latin-1") == QEMU[name]["stdout"]
    if name.startswith("rv32u"):
        assert QEMU[name]["exit"] == 0  # every compliance test passes


@pytest.mark.parametrize("name", sorted(QEMU))
def test_machine_equals_isa(name):
    """The traced machine (records, implied PC, ROM/RAM map) ends in the ISA emulator's state and
    prints what qemu printed."""
    slow(name)
    m = ref.Machine.from_program(elf.program(DATA / f"{name}.elf"))
    toks = m.run(10**8)
    cpu = run_rv32(DATA / f"{name}.elf", S.SP_INIT)
    assert m.halted and m.fault is None and m.x == cpu.x
    assert ref.trace_stats(toks)["instructions"] == cpu.steps
    assert m.exit_code & 0xFF == QEMU[name]["exit"]
    assert m.out.decode("latin-1") == QEMU[name]["stdout"]


@pytest.mark.parametrize("name", list(ALL))
def test_asm_programs(name):
    p = program(name)
    m = ref.Machine.from_program(p)
    m.run()
    cpu = ref.Rv32()
    for i, w in enumerate(p.text):
        cpu.store(S.ROM_BASE + 4 * i, w, 4)
    for i, b in enumerate(p.data):
        cpu.mem[p.data_base + i] = b
    cpu.x[2], cpu.pc = S.SP_INIT, p.entry
    while not cpu.halted:
        cpu.step()
    assert m.x == cpu.x


def test_value_records_roundtrip():
    rng = random.Random(0)
    vals = [0, 1, -1, 2**14 - 1, -2**14, 2**14, -2**14 - 1, 2**29 - 1, -2**29, 2**29, 2**31 - 1,
            -2**31] + [rng.getrandbits(32) for _ in range(2000)]
    for v in vals:
        rec = S.value_record(v)
        assert S.decode_value(rec) == v & 0xFFFFFFFF
        assert [S.is_final(t) for t in rec] == [False] * (len(rec) - 1) + [True]


def test_rom_fields_cover_rv32i():
    for name in sorted(QEMU):
        p = elf.program(DATA / f"{name}.elf")
        for i, w in enumerate(p.text):
            if w:
                uop, rdw, _, _, imm = ref.rom_fields(w, S.ROM_BASE + 4 * i)
                assert 0 <= uop < 256 and 0 <= rdw < 32 and 0 <= imm < 2**32
