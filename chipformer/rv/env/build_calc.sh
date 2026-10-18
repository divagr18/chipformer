#!/bin/bash
# Build the P3 calc firmware:  build_calc.sh OUT_ELF
# Freestanding (no libc), -march=rv32im: calc.c has no division, and the objdump check below
# makes sure gcc emitted none (DIV / REM are an illegal-instruction fault in the core).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1
GCC=riscv64-unknown-elf-gcc
$GCC -march=rv32im -mabi=ilp32 -O2 -ffreestanding -fno-builtin -I"$HERE" \
  -nostdlib -nostartfiles -static -T"$HERE/rom.ld" -o "$OUT" \
  "$HERE/crt0.S" "$HERE/progs/calc.c" -lgcc
riscv64-unknown-elf-size "$OUT"
n=$(riscv64-unknown-elf-objdump -d "$OUT" | grep -cE '\s(div|divu|rem|remu)\s' || true)
echo "div/rem instructions: $n"
[ "$n" = 0 ]
