"""Reference RV32IM: an ISA emulator, the pre-decoded ROM, and the trace (records) the circuit
must produce.

``Rv32``      plain RV32IM semantics (cross-checked against qemu-riscv32 on gcc-built ELFs).
``rom_fields``  the circuit's view of an instruction word: (uop, rdw, rs1, rs2, imm). The ROM is a
              compiled image, so static facts are folded in: rdw = rd if the instruction writes
              rd else 0, AUIPC/LUI are constants (imm = the value), branch and JAL targets are
              absolute word indices.
``Machine``   the traced machine: ``record()`` returns the next record's tokens and commits it.
"""

from dataclasses import dataclass, field

from chipformer.rv import spec as S

UOPS = ["ILLEGAL", "CONST", "ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI", "SLLI", "SRLI",
        "SRAI", "ADD", "SUB", "SLL", "SLT", "SLTU", "XOR", "SRL", "SRA", "OR", "AND",
        "LB", "LH", "LW", "LBU", "LHU", "SB", "SH", "SW",
        "BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU", "JAL", "JALR",
        "ECALL", "EBREAK", "FENCE",
        "MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU"]
U = {n: i for i, n in enumerate(UOPS)}
WRITES_RD = {"CONST", "ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI", "SLLI", "SRLI", "SRAI",
             "ADD", "SUB", "SLL", "SLT", "SLTU", "XOR", "SRL", "SRA", "OR", "AND",
             "LB", "LH", "LW", "LBU", "LHU", "JAL", "JALR",
             "MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU"}
IMM_OPS = {"ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI", "SLLI", "SRLI", "SRAI"}
BRANCH_OPS = {"BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"}
M32 = 0xFFFFFFFF


def sx(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


@dataclass
class Inst:
    op: str
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0  # sign-extended immediate (branch/jump: byte offset)


def decode(w: int) -> Inst:
    opc, rd, f3, rs1, rs2, f7 = w & 0x7F, w >> 7 & 31, w >> 12 & 7, w >> 15 & 31, w >> 20 & 31, w >> 25
    i_imm = sx(w >> 20, 12)
    s_imm = sx((w >> 25) << 5 | (w >> 7 & 31), 12)
    b_imm = sx((w >> 31 & 1) << 12 | (w >> 7 & 1) << 11 | (w >> 25 & 63) << 5 | (w >> 8 & 15) << 1, 13)
    j_imm = sx((w >> 31 & 1) << 20 | (w >> 12 & 0xFF) << 12 | (w >> 20 & 1) << 11
               | (w >> 21 & 0x3FF) << 1, 21)
    if opc == 0x37:
        return Inst("LUI", rd, imm=sx(w & 0xFFFFF000, 32))
    if opc == 0x17:
        return Inst("AUIPC", rd, imm=sx(w & 0xFFFFF000, 32))
    if opc == 0x6F:
        return Inst("JAL", rd, imm=j_imm)
    if opc == 0x67 and f3 == 0:
        return Inst("JALR", rd, rs1, imm=i_imm)
    if opc == 0x63 and f3 not in (2, 3):
        return Inst(["BEQ", "BNE", "", "", "BLT", "BGE", "BLTU", "BGEU"][f3], 0, rs1, rs2, b_imm)
    if opc == 0x03 and f3 in (0, 1, 2, 4, 5):
        return Inst(["LB", "LH", "LW", "", "LBU", "LHU"][f3], rd, rs1, imm=i_imm)
    if opc == 0x23 and f3 in (0, 1, 2):
        return Inst(["SB", "SH", "SW"][f3], 0, rs1, rs2, s_imm)
    if opc == 0x13:
        if f3 == 1 and f7 == 0:
            return Inst("SLLI", rd, rs1, imm=rs2)
        if f3 == 5 and f7 in (0, 0x20):
            return Inst("SRAI" if f7 else "SRLI", rd, rs1, imm=rs2)
        if f3 not in (1, 5):
            return Inst(["ADDI", "", "SLTI", "SLTIU", "XORI", "", "ORI", "ANDI"][f3], rd, rs1, imm=i_imm)
    if opc == 0x33:
        if f7 == 0:
            return Inst(["ADD", "SLL", "SLT", "SLTU", "XOR", "SRL", "OR", "AND"][f3], rd, rs1, rs2)
        if f7 == 0x20 and f3 in (0, 5):
            return Inst("SUB" if f3 == 0 else "SRA", rd, rs1, rs2)
        if f7 == 1:
            return Inst(["MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU"][f3], rd, rs1, rs2)
    if w == 0x00000073:
        return Inst("ECALL")
    if w == 0x00100073:
        return Inst("EBREAK")
    if opc == 0x0F:
        return Inst("FENCE")
    return Inst("ILLEGAL")


def alu(op: str, a: int, b: int) -> int:
    """Register-register / register-immediate results (a, b as unsigned 32-bit)."""
    sa, sb = sx(a, 32), sx(b, 32)
    if op in ("ADD", "ADDI"):
        r = a + b
    elif op == "SUB":
        r = a - b
    elif op in ("SLL", "SLLI"):
        r = a << (b & 31)
    elif op in ("SRL", "SRLI"):
        r = a >> (b & 31)
    elif op in ("SRA", "SRAI"):
        r = sa >> (b & 31)
    elif op in ("SLT", "SLTI"):
        r = int(sa < sb)
    elif op in ("SLTU", "SLTIU"):
        r = int(a < b)
    elif op in ("XOR", "XORI"):
        r = a ^ b
    elif op in ("OR", "ORI"):
        r = a | b
    elif op in ("AND", "ANDI"):
        r = a & b
    elif op == "MUL":
        r = a * b
    elif op == "MULH":
        r = (sa * sb) >> 32
    elif op == "MULHSU":
        r = (sa * b) >> 32
    elif op == "MULHU":
        r = (a * b) >> 32
    elif op == "DIV":
        r = -1 if sb == 0 else (sa if (sa, sb) == (-1 << 31, -1) else int(sa / sb))
    elif op == "DIVU":
        r = M32 if b == 0 else a // b
    elif op == "REM":
        r = sa if sb == 0 else (0 if (sa, sb) == (-1 << 31, -1) else sa - sb * int(sa / sb))
    elif op == "REMU":
        r = a if b == 0 else a % b
    else:
        raise ValueError(op)
    return r & M32


def taken(op: str, a: int, b: int) -> bool:
    sa, sb = sx(a, 32), sx(b, 32)
    return {"BEQ": a == b, "BNE": a != b, "BLT": sa < sb, "BGE": sa >= sb,
            "BLTU": a < b, "BGEU": a >= b}[op]


class Fault(Exception):
    pass


@dataclass
class Rv32:
    """Plain RV32IM with a flat byte memory (dict pages) and ecall exit/putchar."""

    mem: dict[int, int] = field(default_factory=dict)
    x: list[int] = field(default_factory=lambda: [0] * 32)
    pc: int = 0
    out: bytearray = field(default_factory=bytearray)
    halted: bool = False
    exit_code: int | None = None
    steps: int = 0

    def load(self, addr: int, n: int) -> int:
        return int.from_bytes(bytes(self.mem.get(addr + i, 0) for i in range(n)), "little")

    def store(self, addr: int, v: int, n: int) -> None:
        for i in range(n):
            self.mem[addr + i] = v >> 8 * i & 0xFF

    def step(self) -> Inst:
        w = self.load(self.pc, 4)
        ins = decode(w)
        op, x = ins.op, self.x
        nxt = self.pc + 4
        a, b = x[ins.rs1], x[ins.rs2]
        res = None
        if op == "LUI":
            res = ins.imm & M32
        elif op == "AUIPC":
            res = (self.pc + ins.imm) & M32
        elif op in ("JAL", "JALR"):
            res = nxt
            nxt = (self.pc + ins.imm) & M32 if op == "JAL" else (a + ins.imm) & M32 & ~1
        elif op in BRANCH_OPS:
            if taken(op, a, b):
                nxt = (self.pc + ins.imm) & M32
        elif op == "EBREAK":
            self.halted = True
        elif op in ("LB", "LH", "LW", "LBU", "LHU"):
            ea = (a + ins.imm) & M32
            n = {"LB": 1, "LBU": 1, "LH": 2, "LHU": 2, "LW": 4}[op]
            v = self.load(ea, n)
            res = (sx(v, 8 * n) if op in ("LB", "LH") else v) & M32
        elif op in ("SB", "SH", "SW"):
            ea = (a + ins.imm) & M32
            self.store(ea, b, {"SB": 1, "SH": 2, "SW": 4}[op])
        elif op == "ECALL":
            if x[17] == 93:
                self.halted, self.exit_code = True, x[10]
            elif x[17] in (1, 64):  # putchar(a0) / write(1, a1, a2)
                if x[17] == 1:
                    self.out.append(x[10] & 0xFF)
                else:
                    self.out += bytes(self.load(x[11] + i, 1) for i in range(x[12]))
                    res = None
            else:
                self.halted = True
        elif op == "FENCE":
            pass
        elif op == "ILLEGAL":
            raise Fault(f"illegal instruction {w:#010x} at {self.pc:#x}")
        elif op in IMM_OPS:
            res = alu(op, a, ins.imm & M32)
        else:
            res = alu(op, a, b)
        if res is not None and ins.rd:
            x[ins.rd] = res
        self.pc = nxt & M32
        self.steps += 1
        return ins


# --- the circuit's ROM ------------------------------------------------------------------------


def rom_fields(w: int, pc: int) -> tuple[int, int, int, int, int]:
    """(uop, rdw, rs1, rs2, imm & 0xFFFFFFFF) of the instruction word w at pc."""
    ins = decode(w)
    op = ins.op
    imm = ins.imm
    if op in ("LUI", "AUIPC"):
        imm = ins.imm if op == "LUI" else pc + ins.imm
        op = "CONST"
    elif op == "JAL" or op in BRANCH_OPS:
        imm = (((pc + ins.imm) & M32) - S.ROM_BASE) >> 2  # target word (from ROM_BASE)
    rdw = ins.rd if op in WRITES_RD else 0
    return U[op], rdw, ins.rs1, ins.rs2, imm & M32


# --- the traced machine -----------------------------------------------------------------------


@dataclass
class Machine:
    """State the trace commits: registers, RAM words, anchor, n. ROM is read-only."""

    rom: list[int]  # instruction words from ROM_BASE
    ram: list[int]  # words from RAM_BASE
    x: list[int]
    anchor: int  # PC word index of the latest J / R
    n: int = 0
    halted: bool = False
    steps: int = 0  # instructions executed

    @classmethod
    def from_program(cls, prog, entry: int | None = None) -> "Machine":
        ram = [0] * S.RAM_WORDS
        off = prog.data_base - S.RAM_BASE
        data = bytes(off) + prog.data
        data += bytes(-len(data) % 4)
        for i in range(0, len(data), 4):
            ram[i // 4] = int.from_bytes(data[i:i + 4], "little")
        x = [0] * 32
        x[2] = S.SP_INIT
        rom = list(prog.text) + [0] * (S.ROM_WORDS - len(prog.text))
        e = prog.entry if entry is None else entry
        return cls(rom, ram, x, (e - S.ROM_BASE) >> 2)

    @property
    def pc(self) -> int:
        return S.ROM_BASE + 4 * (self.anchor + self.n)

    def ram_index(self, ea: int) -> int:
        i = (ea - S.RAM_BASE) >> 2
        if not 0 <= i < S.RAM_WORDS:
            raise Fault(f"address {ea:#x} outside RAM")
        return i

    def record(self) -> list[int]:
        """The next record's tokens; commits it (state as after its last token)."""
        if self.n == S.N_MAX:
            w = self.anchor + self.n
            self.anchor, self.n = w, 0
            return [S.tok("R", w)]
        pc = self.pc
        word = self.rom[(pc - S.ROM_BASE) >> 2]
        ins = decode(word)
        op, x = ins.op, self.x
        a, b = x[ins.rs1], x[ins.rs2]
        self.steps += 1

        def commit_rd(v):
            if ins.rd:
                x[ins.rd] = v & M32
                return S.value_record(v)
            return [S.tok("NT")]

        def jump(target):
            if target & 3:
                raise Fault(f"misaligned jump to {target:#x}")
            self.anchor, self.n = (target - S.ROM_BASE) >> 2, 0
            return [S.tok("J", self.anchor)]

        self.n += 1  # (a jump resets it below)
        if op == "LUI":
            return commit_rd(ins.imm)
        if op == "AUIPC":
            return commit_rd(pc + ins.imm)
        if op in ("JAL", "JALR"):
            target = (pc + ins.imm) & M32 if op == "JAL" else (a + ins.imm) & M32 & ~1
            if ins.rd:
                x[ins.rd] = (pc + 4) & M32
            return jump(target)
        if op in BRANCH_OPS:
            if taken(op, a, b):
                return jump((pc + ins.imm) & M32)
            return [S.tok("NT")]
        if op in ("LB", "LH", "LW", "LBU", "LHU"):
            ea = (a + ins.imm) & M32
            n = {"LB": 1, "LBU": 1, "LH": 2, "LHU": 2, "LW": 4}[op]
            if ea % n:
                raise Fault(f"misaligned load at {ea:#x}")
            wv = self.ram[self.ram_index(ea)] >> 8 * (ea & 3) & ((1 << 8 * n) - 1)
            return commit_rd(sx(wv, 8 * n) if op in ("LB", "LH") else wv)
        if op in ("SB", "SH", "SW"):
            ea = (a + ins.imm) & M32
            n = {"SB": 1, "SH": 2, "SW": 4}[op]
            if ea % n:
                raise Fault(f"misaligned store at {ea:#x}")
            i = self.ram_index(ea)
            w4 = i >> 4 & 1
            if op == "SW":
                self.ram[i] = b
                return [S.tok(f"ST{w4}")]
            sh, mask = 8 * (ea & 3), (1 << 8 * n) - 1
            m = self.ram[i] & ~(mask << sh) & M32 | (b & mask) << sh
            self.ram[i] = m
            return [S.tok("V", m & 0x7FFF), S.tok("V", m >> 15 & 0x7FFF), S.tok("V", m >> 30),
                    S.tok(f"SS{w4}")]
        if op in ("ECALL", "EBREAK", "ILLEGAL"):
            self.halted = True
            return [S.tok("HALT")]
        if op == "FENCE":
            return [S.tok("NT")]
        if op in ("DIV", "DIVU", "REM", "REMU"):
            raise Fault(f"{op} is not in the P0 circuit")
        if op in IMM_OPS:
            return commit_rd(alu(op, a, ins.imm & M32))
        return commit_rd(alu(op, a, b))

    def run(self, max_records: int = 10**6) -> list[int]:
        """The whole trace from GO to HALT."""
        toks = [S.tok("GO")]
        while not self.halted and len(toks) < max_records:
            toks += self.record()
        return toks


def trace_stats(toks: list[int]) -> dict:
    fams = [S.family(t)[0] for t in toks]
    instr = sum(1 for f in fams if f in S.FINAL and f != "R")
    return {"tokens": len(toks), "instructions": instr,
            "tokens_per_instruction": len(toks) / max(instr, 1),
            "refreshes": fams.count("R")}
