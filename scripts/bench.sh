#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BIN="zig-out/bin/microgpt_zig"
ASM="zig-out/bin/microgpt_zig.s"

mkdir -p zig-out/bin

echo "== build: ReleaseFast/native =="
zig build -Doptimize=ReleaseFast -Dtarget=native

echo "== emit assembly =="
zig build-exe src/main.zig \
    -O ReleaseFast \
    -mcpu=native \
    -femit-bin="$BIN" \
    -femit-asm="$ASM"

echo "== benchmark =="
"$BIN"

echo
echo "== SIMD instruction counts =="
for op in fmla fmadd fmul fadd ld1 st1; do
    count=$(rg -c "\\b${op}\\b" "$ASM" || true)
    printf '%-6s %s\n' "$op" "$count"
done

echo
echo "Assembly: $ASM"
