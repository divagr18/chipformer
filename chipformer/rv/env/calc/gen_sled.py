"""Write sled.S: per digit-file register r, BPRD r, 127 .. BPRD r, 0 then ret (bn.h bn_print
jumps in at the top digit to print, one token per digit)."""

import sys

out = ["    .section .text", "    .globl bn_sleds"]
for r in range(8):
    out += [f"bn_sled{r}:"] + [f"    .insn i 0x2B, 0, x{r}, x0, {k}" for k in range(127, -1, -1)]
    out += ["    ret"]
out += ["    .section .rodata", "    .align 2", "bn_sleds:"]
out += [f"    .word bn_sled{r}" for r in range(8)]
open(sys.argv[1], "w").write("\n".join(out) + "\n")
