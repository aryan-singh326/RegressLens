"""Static check that AVX2 code cannot leak into the portable code path.

Two ways an AVX2 instruction can reach a CPU without AVX2:

1. A baseline object file contains VEX-encoded instructions itself
   (an -mavx2 flag reached a file it should not have).
2. An inline function or template is instantiated both in the AVX2
   object and in a baseline object. The linker keeps one COMDAT copy,
   and if it keeps the AVX2-compiled one, baseline code calls it.

This script checks both on a CMake build tree (GNU binutils; Linux CI):
every object except kernels_avx2.cpp.o must have zero VEX instructions,
and the weak/COMDAT symbols defined by kernels_avx2.cpp.o must not be
defined by any other object linked into the extension.

Usage: python tools/check_isa_isolation.py BUILD_DIR
"""
import pathlib
import re
import subprocess
import sys

VEX = re.compile(r"\t(v[a-z0-9]+)\s")  # VEX/EVEX mnemonics all start with 'v'


def objects(build_dir):
    objs = [p for p in pathlib.Path(build_dir).rglob("*.o")
            if "regresslens_kernels.dir" in str(p) or "_native.dir" in str(p)]
    avx2 = [p for p in objs if p.name.startswith("kernels_avx2")]
    rest = [p for p in objs if not p.name.startswith("kernels_avx2")]
    return avx2, rest


def vex_instructions(obj):
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(obj)],
                         capture_output=True, text=True, check=True).stdout
    return [m.group(1) for m in VEX.finditer(out)]


def weak_symbols(obj):
    out = subprocess.run(["nm", "--defined-only", str(obj)],
                         capture_output=True, text=True, check=True).stdout
    syms = set()
    for line in out.splitlines():
        parts = line.split()
        # 'W' = weak code symbol (COMDAT functions). Weak *data* ('V',
        # e.g. DW.ref.__gxx_personality_v0) holds no instructions.
        if len(parts) == 3 and parts[1] in ("W", "u"):
            syms.add(parts[2])
    return syms


def main(build_dir):
    avx2, rest = objects(build_dir)
    if not avx2:
        print("no kernels_avx2 object found (non-x86 build?); nothing to check")
        return 0
    if not rest:
        print("error: no baseline objects found under", build_dir)
        return 2
    failures = 0
    for obj in rest:
        vex = vex_instructions(obj)
        if vex:
            failures += 1
            print(f"FAIL {obj.name}: {len(vex)} VEX instructions, e.g. {sorted(set(vex))[:5]}")
        else:
            print(f"ok   {obj.name}: no VEX instructions")
    avx2_weak = weak_symbols(avx2[0])
    for obj in rest:
        shared = avx2_weak & weak_symbols(obj)
        if shared:
            failures += 1
            print(f"FAIL COMDAT overlap between {avx2[0].name} and {obj.name}:")
            for s in sorted(shared)[:10]:
                print("       ", s)
    if not failures:
        print(f"ok   {avx2[0].name}: {len(avx2_weak)} weak symbols, none shared "
              "with baseline objects")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "build"))
