#!/bin/bash
# Build the calc firmware (run in WSL):  build_calc.sh OUT
# OUT/calc.elf: the ROM / DROM / RAM map, the expression buffer calc_in at RAM 0x100.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ENV=$HERE/..
OUT=$(realpath "$1"); NAME=${2:-calc}
PICO=/usr/lib/picolibc/riscv64-unknown-elf
GCC=riscv64-unknown-elf-gcc
MARCH=rv32im_zba
LIBGCC=$($GCC -march=rv32im -mabi=ilp32 -print-libgcc-file-name)
B=$(mktemp -d)
python3 "$HERE/gen_sled.py" "$B/sled.S"
# calc_in first in RAM (0x100)
sed 's/  .data : {/  .calcin : { *(.calcin) } > RAM\n  .data : {/' "$ENV/rom.ld" > "$B/calc.ld"
$GCC -march=$MARCH -mabi=ilp32 ${CFLAGS:--O2} -isystem "$PICO/include" -I"$ENV" -I"$HERE" \
  -nostdlib -nostartfiles -static -T"$B/calc.ld" -o "$OUT/$NAME.elf" \
  "$ENV/crt0.S" "$HERE/calc.c" "$ENV/sys.c" "$B/sled.S" \
  -Wl,--start-group "$PICO/lib/release/rv32im/ilp32/libc.a" "$LIBGCC" -Wl,--end-group
riscv64-unknown-elf-size -A "$OUT/$NAME.elf" | grep -E '^\.(text|rodata|calcin|data|bss)'
riscv64-unknown-elf-nm "$OUT/$NAME.elf" | grep -E ' calc_in$'
rm -rf "$B"
