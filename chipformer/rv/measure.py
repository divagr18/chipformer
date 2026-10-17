"""P0 measurements of the RV32 core: depth, capacity per sublayer, nonzeros, tokens per
instruction (15- vs 16-bit payloads), vocabulary, and speed estimates.

    python -m chipformer.rv.measure [--nonzeros] > dist/rv/measure.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from chipformer.q35.ir import FFN, HD, Head
from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "rv"
C_PROGS = ["c-crc32", "c-sieve", "c-sort", "c-strops"]
# dynamic instruction mix of small C programs (dist/rv_tool/REPORT.md, P0.6), percent
MIX_SMALL_C = {"ALU": 47.0, "LW": 13.7, "LBH": 1.4, "SW": 8.9, "SBH": 2.3, "BRT": 11.3,
               "BRNT": 10.1, "JAL": 1.4, "JALR": 2.2, "U": 0.1, "MUL": 0.2, "DIV": 1.5}
DENSE_4B = {"hidden": 3072, "key_heads": 16, "value_heads": 32, "intermediate": 9216}


def per_sublayer(c):
    units, heads = Counter(), Counter()
    for n in c.nodes:
        if isinstance(n, FFN):
            units[n.level] += len(n.terms)
        else:
            heads[n.level] += 1
    names = Counter()
    for n in c.nodes:
        if isinstance(n, Head):
            names[(n.level, n.name.rstrip("0123456789").rstrip("."))] += 1
    return ({f"L{(lv + 1) // 2}.{'attn' if lv % 2 else 'mlp'}": u for lv, u in sorted(units.items())},
            {f"L{(lv + 1) // 2}.attn": h for lv, h in sorted(heads.items())},
            {f"L{(lv + 1) // 2}.{nm}": k for (lv, nm), k in sorted(names.items())})


def value_lengths(vals, payload_bits):
    """Tokens of each value record with chunks of payload_bits (sign-extended final chunk)."""
    out = Counter()
    for v in vals:
        s = v - (1 << 32) if v >> 31 else v
        k = 1
        while not -(1 << (payload_bits * k - 1)) <= s < 1 << (payload_bits * k - 1):
            k += 1
        out[min(k, -(-32 // payload_bits))] += 1
    return out


def trace_classes():
    """Per-class record counts and rd values over the C programs' traces."""
    cls, vals = Counter(), []
    for name in C_PROGS:
        m = ref.Machine.from_program(elf.program(DATA / f"{name}.elf"))
        while not m.halted:
            if m.n == S.N_MAX:
                m.record()
                cls["R"] += 1
                continue
            ins = ref.decode(m.rom[(m.pc - S.ROM_BASE) >> 2])
            before = list(m.x)
            m.record()
            op = ins.op
            if op in ref.BRANCH_OPS:
                cls["BR"] += 1
            elif op in ("JAL", "JALR", "SW", "SB", "SH", "ECALL", "EBREAK", "FENCE"):
                cls[op] += 1
            elif ins.rd:
                cls["value"] += 1
                vals.append(m.x[ins.rd])
            else:
                cls["NT"] += 1
            del before
    return cls, vals


def tokens_per_instruction(vbar: float, refresh: float = 0.0) -> float:
    """Expected tokens per instruction for the small-C mix (value records cost vbar)."""
    t = {"ALU": vbar, "LW": vbar, "LBH": vbar, "U": vbar, "MUL": vbar, "DIV": vbar,
         "SW": 1, "SBH": 4, "BRT": 1, "BRNT": 1, "JAL": 1, "JALR": 1}
    return sum(MIX_SMALL_C[k] * t[k] for k in MIX_SMALL_C) / sum(MIX_SMALL_C.values()) + refresh


def payload_options(vals, hidden: int = 3072) -> dict:
    """Value-chunk widths: tokens per instruction (small-C mix) against vocabulary rows.

    12-bit chunks need a 2-token jump (the ROM word index is wider than 12 bits); 16-bit chunks
    let an SB/SH write its merged word as V + final (2 tokens instead of 4) but need 16-bit SS
    families too."""
    out = {}
    for pb in (12, 15, 16):
        lens = value_lengths(vals, pb)
        vbar = sum(k * n for k, n in lens.items()) / max(sum(lens.values()), 1)
        jump = 2 if pb < 14 else 1
        t = {"ALU": vbar, "LW": vbar, "LBH": vbar, "U": vbar, "MUL": vbar, "DIV": vbar,
             "SW": 1, "SBH": 2 if pb == 16 else 4, "BRT": jump, "BRNT": 1, "JAL": jump,
             "JALR": jump}
        tpi = sum(MIX_SMALL_C[k] * t[k] for k in MIX_SMALL_C) / sum(MIX_SMALL_C.values())
        rows = {12: 5 * 4096 + 256 + 7, 15: 5 * (1 << 15) + 4 + 7, 16: 7 * (1 << 16) + 5}[pb]
        out[pb] = {"value_record_lengths": dict(sorted(lens.items())), "vbar": round(vbar, 3),
                   "tokens_per_instruction_small_c_mix": round(tpi, 3), "vocab_rows": rows,
                   "embedding_gb_bf16": round(rows * hidden * 2 / 1e9, 2)}
    return out


def nonzeros(c):
    """Nonzero weights of the emitted chip layers (standalone layout; independent of width)."""
    from chipformer.q35.emit import emit_layers
    from chipformer.q35.tiny import BASE, pick_layout

    lay = pick_layout(len(c.dims))
    tc = dict(BASE, hidden_size=lay.hidden, linear_num_key_heads=c.target.key_heads,
              linear_num_value_heads=c.target.value_heads, intermediate_size=c.target.intermediate)
    sd = emit_layers(c, lay, tc, 0, "model.")
    per = Counter()
    for k, t in sd.items():
        per[int(k.split(".")[2])] += int((t != 0).sum())
    return dict(sorted(per.items())), lay.hidden


def dense_params_per_layer(h, kh, vh, inter):
    qkv = (2 * kh * HD + vh * HD) * h
    return qkv + vh * HD * h + 2 * vh * h + h * vh * HD + 3 * inter * h + 4 * (2 * kh * HD + vh * HD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nonzeros", action="store_true")
    args = ap.parse_args()
    c = core.build()
    units, heads, names = per_sublayer(c)
    cls, vals = trace_classes()
    n_instr = sum(v for k, v in cls.items() if k != "R")
    res = {"stats": c.stats(), "units_per_sublayer": units, "heads_per_sublayer": heads,
           "heads_by_table": names, "trace_classes": dict(cls)}
    res["payload_options"] = payload_options(vals)
    per_prog = {}
    toks = 0
    for n in C_PROGS:
        st = ref.trace_stats(ref.Machine.from_program(elf.program(DATA / f"{n}.elf")).run(10**7))
        per_prog[n] = {k: st[k] for k in ("instructions", "tokens", "tokens_per_instruction")}
        toks += st["tokens"]
    res["c_programs"] = per_prog
    res["measured_tokens_per_instruction_c"] = toks / n_instr
    dense = dense_params_per_layer(DENSE_4B["hidden"], 16, 32, 9216)
    res["dense_params_per_chip_layer_4b"] = dense
    res["dense_chip_params_4b"] = dense * c.n_layers()
    cap = {"rom_bytes_per_sublayer": 16 * 1024 * 4, "ram_bytes_per_sublayer": 16 * 2 * 128 * 4}
    res["capacity"] = cap
    if args.nonzeros:
        nz, h = nonzeros(c)
        res["nonzeros_per_layer"] = nz
        res["nonzeros_total"] = sum(nz.values())
        res["nonzeros_hidden"] = h
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
