#!/bin/bash
# Build MicroPython for the chipformer RV32 machine, one firmware per script (run in WSL):
#   build_mpy.sh OUT script.py [script.py ...]
# The configuration is dist/rv_tool's best build under 256 KB: the tree TREE carries its 3 source
# patches (FIRST-set parse prediction, fast lexer, qstr bucket index), the FIRST-set table sits in
# rodata (FS: the table dumped by rv_tool's build_fsrom.sh), 5 hot files at -O3, word-wise libc
# (fastlib.c), computed goto, the map lookup cache, no conservative GC clear, parse chunk 512;
# -march=rv32im_zba. OUT/mpy-<script>.elf is the ROM/DROM/RAM map, OUT/mpy-<script>.qemu.elf
# the same code for qemu-riscv32; OUT/qemu.json gets their exit codes and output.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=${TREE:-/root/build/mp-all3}
FS=${FS:-$TREE/ports/rv32sim/build-v-combo-hot5O3/fs_table.h}
OUT=$(realpath "$1"); shift
P=$TREE/ports/chipformer
mkdir -p "$P" "$OUT"
cp "$HERE"/mpy/*.c "$HERE"/mpy/*.h "$HERE"/mpy/Makefile "$P"/
COPT="-DRV_FIRST_SETS -DRV_FS_ROM -DMICROPY_OPT_COMPUTED_GOTO=1 -DMICROPY_OPT_MAP_LOOKUP_CACHE=1"
COPT="$COPT -DMICROPY_GC_CONSERVATIVE_CLEAR=0 -DMICROPY_ALLOC_PARSE_CHUNK_INIT=512"
HOT="py/parse py/lexer py/qstr py/map py/vm"
for s in "$@"; do
  n=$(basename "$s" .py)
  for map in rom qemu; do
    B=build-$n-$map
    mkdir -p "$P/$B"
    cp "$FS" "$P/$B/fs_table.h"
    make -C "$P" -j16 CONFIG=basic MAP=$map SCRIPT="$(realpath "$s")" ENV="$HERE" BUILD=$B \
      HOT="$HOT" COPT="$COPT" > "/tmp/mpy-$n-$map.log" 2>&1 || { tail -40 "/tmp/mpy-$n-$map.log"; exit 1; }
    sfx=$([ $map = rom ] && echo "" || echo ".qemu")
    cp "$P/$B/firmware.elf" "$OUT/mpy-$n$sfx.elf"
    grep -E '^\.(text|rodata|data|bss)' "/tmp/mpy-$n-$map.log" | tr '\n' ' '; echo " ($n $map)"
  done
done
python3 - "$OUT" "$@" <<'PYEOF'
import json, os, subprocess, sys
out, scripts = sys.argv[1], sys.argv[2:]
path = os.path.join(out, "qemu.json")
res = json.load(open(path)) if os.path.exists(path) else {}
for s in scripts:
    n = "mpy-" + os.path.basename(s)[:-3]
    p = subprocess.run(["qemu-riscv32", "-cpu", "rv32,zba=true", os.path.join(out, n + ".qemu.elf")],
                       capture_output=True, timeout=600)
    res[n] = {"exit": p.returncode, "stdout": p.stdout.decode("latin-1")}
    print(n, p.returncode, repr(res[n]["stdout"][:200]))
json.dump(res, open(path, "w"), indent=1)
PYEOF
