"""A small two-pass CHIP-8 assembler (Cowgod mnemonics, plus PRINT for the bridge's 00FD).

    label:              define a label (its address)
    db 1, 2, 0x03       raw bytes
    org 0x600           set the address (gaps are zero-filled)
    ; comment

Instructions: CLS RET PRINT JP CALL SE SNE LD ADD OR AND XOR SUB SHR SUBN SHL RND DRW SKP SKNP.
Operands: V0..VF, I, [I], DT, ST, K, F, B, numbers (decimal / 0x / 0b) and labels.
"""

import re

START = 0x200


def _num(tok: str, labels: dict[str, int]) -> int:
    tok = tok.strip()
    if tok in labels:
        return labels[tok]
    return int(tok, 0)


def _reg(tok: str) -> int | None:
    m = re.fullmatch(r"[vV]([0-9a-fA-F])", tok.strip())
    return int(m[1], 16) if m else None


def assemble(src: str, start: int = START) -> tuple[bytes, dict[str, int]]:
    """Returns (bytes from ``start``, labels)."""
    lines = []
    for raw in src.splitlines():
        line = raw.split(";")[0].strip()
        while line:
            m = re.match(r"([A-Za-z_][\w.]*):\s*(.*)", line)
            if not m:
                break
            lines.append(("label", m[1]))
            line = m[2].strip()
        if line:
            op, _, rest = line.partition(" ")
            args = [a.strip() for a in rest.split(",")] if rest.strip() else []
            lines.append((op.upper(), args))
    labels: dict[str, int] = {}
    for pass_ in range(2):
        pc = start
        out = bytearray()
        for op, args in lines:
            if op == "label":
                labels[args] = pc
                continue
            if op == "ORG":
                target = _num(args[0], labels)
                assert target >= pc, f"org {target:#x} goes backwards from {pc:#x}"
                out += bytes(target - pc)
                pc = target
                continue
            if op == "DB":
                data = [_num(a, labels) & 0xFF for a in args] if pass_ else [0] * len(args)
                out += bytes(data)
                pc += len(data)
                continue
            word = _encode(op, args, labels) if pass_ else 0
            out += bytes([word >> 8, word & 0xFF])
            pc += 2
    return bytes(out), labels


def _encode(op: str, args: list[str], labels: dict[str, int]) -> int:
    r = [_reg(a) for a in args]
    a = [x.upper() for x in args]

    def n(i, bits=12):
        v = _num(args[i], labels)
        assert 0 <= v < 1 << bits, (op, args, v)
        return v

    if op == "CLS":
        return 0x00E0
    if op == "RET":
        return 0x00EE
    if op == "PRINT":
        return 0x00FD
    if op in ("BADD", "BSUB", "BRSUB"):  # bignum macros (macros.py), with a jump target
        t = n(0)
        assert 0x200 <= t < 0x400 and t % 2 == 0, (op, args)
        return 0x5000 | (t - 0x200) // 2 << 4 | {"BADD": 1, "BSUB": 4, "BRSUB": 7}[op]
    if op == "BZERO":
        return 0x5002
    if op == "BMUL":
        return 0x5003 | r[0] << 8
    if op == "BDIV":
        return 0x5005 | r[0] << 8
    if op == "BADB":
        return 0x5006 | r[0] << 8
    if op == "BMUL0":
        return 0x5013 | r[0] << 8
    if op == "BTRIM":
        return 0x5008
    if op == "JP":
        if len(args) == 2:
            return 0xB000 | n(1)
        return 0x1000 | n(0)
    if op == "CALL":
        return 0x2000 | n(0)
    if op in ("SE", "SNE"):
        x = r[0]
        if r[1] is not None:
            return (0x5000 if op == "SE" else 0x9000) | x << 8 | r[1] << 4
        return (0x3000 if op == "SE" else 0x4000) | x << 8 | n(1, 8)
    if op == "LD":
        if a[0] == "I":
            return 0xA000 | n(1)
        if a[0] == "[I]":
            return 0xF055 | r[1] << 8
        if a[1] == "[I]":
            return 0xF065 | r[0] << 8
        if a[0] == "DT":
            return 0xF015 | r[1] << 8
        if a[0] == "ST":
            return 0xF018 | r[1] << 8
        if a[0] == "F":
            return 0xF029 | r[1] << 8
        if a[0] == "B":
            return 0xF033 | r[1] << 8
        if a[1] == "DT":
            return 0xF007 | r[0] << 8
        if a[1] == "K":
            return 0xF00A | r[0] << 8
        if r[1] is not None:
            return 0x8000 | r[0] << 8 | r[1] << 4
        return 0x6000 | r[0] << 8 | n(1, 8)
    if op == "ADD":
        if a[0] == "I":
            return 0xF01E | r[1] << 8
        if r[1] is not None:
            return 0x8004 | r[0] << 8 | r[1] << 4
        return 0x7000 | r[0] << 8 | n(1, 8)
    alu = {"OR": 1, "AND": 2, "XOR": 3, "SUB": 5, "SHR": 6, "SUBN": 7, "SHL": 0xE}
    if op in alu:
        y = r[1] if len(r) > 1 and r[1] is not None else 0
        return 0x8000 | r[0] << 8 | y << 4 | alu[op]
    if op == "RND":
        return 0xC000 | r[0] << 8 | n(1, 8)
    if op == "DRW":
        return 0xD000 | r[0] << 8 | r[1] << 4 | n(2, 4)
    if op == "SKP":
        return 0xE09E | r[0] << 8
    if op == "SKNP":
        return 0xE0A1 | r[0] << 8
    raise ValueError(f"unknown instruction {op} {args}")
