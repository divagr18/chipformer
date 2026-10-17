"""Load a static RV32 ELF (gcc, env/rom.ld or env/qemu.ld) into an asm.Program or an Rv32."""

import struct
from pathlib import Path

from chipformer.rv import spec as S
from chipformer.rv.asm import Program


def segments(path) -> tuple[int, list[tuple[int, bytes]]]:
    """(entry, [(vaddr, bytes incl. zero-filled bss)]) of the PT_LOAD segments."""
    b = Path(path).read_bytes()
    assert b[:4] == b"\x7fELF" and b[4] == 1, "ELF32 only"
    entry, phoff = struct.unpack_from("<II", b, 0x18)
    phentsize, phnum = struct.unpack_from("<HH", b, 0x2A)
    segs = []
    for i in range(phnum):
        p_type, off, vaddr, _, filesz, memsz, _, _ = struct.unpack_from("<8I", b, phoff + i * phentsize)
        if p_type == 1 and memsz:
            segs.append((vaddr, b[off:off + filesz] + bytes(memsz - filesz)))
    return entry, segs


def program(path) -> Program:
    """A ROM-map ELF as a Program: code words from ROM_BASE, data bytes from RAM_BASE."""
    entry, segs = segments(path)
    text = bytearray()
    data = bytearray()
    for vaddr, blob in segs:
        if vaddr >= S.ROM_BASE:
            end = vaddr - S.ROM_BASE + len(blob)
            text += bytes(max(0, end - len(text)))
            text[vaddr - S.ROM_BASE:end] = blob
        else:
            assert S.RAM_BASE <= vaddr and vaddr + len(blob) <= S.RAM_BASE + 4 * S.RAM_WORDS, \
                f"segment at {vaddr:#x} outside ROM/RAM"
            end = vaddr - S.RAM_BASE + len(blob)
            data += bytes(max(0, end - len(data)))
            data[vaddr - S.RAM_BASE:end] = blob
    text += bytes(-len(text) % 4)
    words = [int.from_bytes(text[i:i + 4], "little") for i in range(0, len(text), 4)]
    return Program(words, bytes(data), {"_start": entry}, data_base=S.RAM_BASE)


def load_into(cpu, path) -> None:
    """Load any static ELF into a ref.Rv32 (flat memory) and set its PC."""
    entry, segs = segments(path)
    for vaddr, blob in segs:
        for i, x in enumerate(blob):
            cpu.mem[vaddr + i] = x
    cpu.pc = entry
