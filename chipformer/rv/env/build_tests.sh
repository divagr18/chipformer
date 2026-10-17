#!/bin/bash
# Build riscv-tests (rv32ui, rv32um, rv32uzba) and the C programs for the chipformer RV32 machine
# (run in WSL):
#   build_tests.sh RISCV_TESTS_SRC MINI_GMP_DIR OUT
# For each program: OUT/<name>.elf (ROM/DROM/RAM map, for the reference and the circuit) and
# OUT/<name>.qemu.elf (the same code at qemu-mappable addresses, putchar as write(2));
# OUT/qemu.json holds qemu's exit code and stdout for every .qemu.elf.
# C: gcc -march=rv32im_zba, picolibc (rv32im).
set -euo pipefail
SRC=$1
GMP=$2
OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
PICO=/usr/lib/picolibc/riscv64-unknown-elf
GCC=riscv64-unknown-elf-gcc
MARCH=rv32im_zba
LIBGCC=$($GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
LIBS="-Wl,--start-group $PICO/lib/release/rv32im/ilp32/libc.a $LIBGCC -Wl,--end-group"
QEMU="qemu-riscv32 -cpu rv32,zba=true"
mkdir -p "$OUT"
rm -f "$OUT"/*.elf

# compliance tests: rv32ui (but fence_i: self-modifying code; ma_data: misaligned), rv32um,
# rv32uzba
for s in "$SRC"/isa/rv32ui/*.S "$SRC"/isa/rv32um/*.S "$SRC"/isa/rv32uzba/*.S; do
  n=$(basename "$s" .S)
  suite=$(basename "$(dirname "$s")")
  case " fence_i ma_data " in *" $n "*) continue;; esac
  for map in rom qemu; do
    sfx=$([ $map = rom ] && echo "" || echo ".qemu")
    $GCC -march=$MARCH -mabi=ilp32 -nostdlib -nostartfiles -static -I"$HERE" \
      -I"$SRC/isa/macros/scalar" -T"$HERE/$map.ld" -o "$OUT/$suite-$n$sfx.elf" "$s"
  done
done

# C programs (gmpmul links mini-gmp)
for c in "$HERE"/progs/*.c; do
  n=$(basename "$c" .c)
  extra=""
  [ "$n" = gmpmul ] && extra="-I$GMP $GMP/mini-gmp.c"
  for map in rom qemu; do
    sfx=$([ $map = rom ] && echo "" || echo ".qemu")
    def=$([ $map = rom ] && echo "" || echo "-DQEMU")
    $GCC -march=$MARCH -mabi=ilp32 -O2 $def -isystem "$PICO/include" -I"$HERE" \
      -nostdlib -nostartfiles -static -T"$HERE/$map.ld" -o "$OUT/c-$n$sfx.elf" \
      "$HERE/crt0.S" "$c" "$HERE/sys.c" $extra $LIBS
  done
done

python3 - "$OUT" "$QEMU" <<'EOF'
import json, subprocess, sys, glob, os
out, qemu = sys.argv[1], sys.argv[2].split()
res = {}
for f in sorted(glob.glob(os.path.join(out, "*.qemu.elf"))):
    p = subprocess.run(qemu + [f], capture_output=True, timeout=600)
    res[os.path.basename(f)[:-len(".qemu.elf")]] = {"exit": p.returncode,
                                                    "stdout": p.stdout.decode("latin-1")}
json.dump(res, open(os.path.join(out, "qemu.json"), "w"), indent=1)
print(len(res), "programs;", sum(r["exit"] == 0 for r in res.values()), "exit 0")
for k, r in res.items():
    if k.startswith("c-"):
        print(k, r["exit"], r["stdout"].strip()[:80])
EOF
