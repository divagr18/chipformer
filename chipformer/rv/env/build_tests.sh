#!/bin/bash
# Build riscv-tests rv32ui and the C programs for the chipformer RV32 machine (run in WSL):
#   build_tests.sh RISCV_TESTS_SRC OUT
# For each test: OUT/<name>.elf (ROM/RAM map, for the reference and the circuit) and
# OUT/<name>.qemu.elf (the same code at qemu-mappable addresses); OUT/qemu.json holds qemu's exit
# code and stdout for every .qemu.elf.
set -euo pipefail
SRC=$1
OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd)
CC="riscv64-unknown-elf-gcc -march=rv32i -mabi=ilp32 -nostdlib -nostartfiles -static -O2"
mkdir -p "$OUT"
SKIP="fence_i ma_data"  # self-modifying code (ROM is read-only); misaligned access (traps)
for s in "$SRC"/isa/rv32ui/*.S; do
  n=$(basename "$s" .S)
  case " $SKIP " in *" $n "*) continue;; esac
  for map in rom qemu; do
    sfx=$([ $map = rom ] && echo "" || echo ".qemu")
    $CC -I"$HERE" -I"$SRC/isa/macros/scalar" -T"$HERE/$map.ld" -o "$OUT/rv32ui-$n$sfx.elf" "$s"
  done
done
for c in "$HERE"/progs/*.c; do
  n=$(basename "$c" .c)
  for map in rom qemu; do
    sfx=$([ $map = rom ] && echo "" || echo ".qemu")
    $CC -ffreestanding -fno-builtin -I"$HERE" -T"$HERE/$map.ld" -o "$OUT/c-$n$sfx.elf" \
      "$HERE/crt0.S" "$c" -lgcc
  done
done
python3 - "$OUT" <<'EOF'
import json, subprocess, sys, glob, os
out = sys.argv[1]
res = {}
for f in sorted(glob.glob(os.path.join(out, "*.qemu.elf"))):
    p = subprocess.run(["qemu-riscv32", f], capture_output=True, timeout=60)
    res[os.path.basename(f)[:-len(".qemu.elf")]] = {"exit": p.returncode,
                                                    "stdout": p.stdout.decode("latin-1")}
json.dump(res, open(os.path.join(out, "qemu.json"), "w"), indent=1)
print(len(res), "programs;", sum(r["exit"] == 0 for r in res.values()), "exit 0")
EOF
