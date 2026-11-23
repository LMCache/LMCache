#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail fast when triton-cpu codegen emits instructions this host cannot run.

The CPU device E2E once died sixty seconds in, inside a vLLM worker, with
``Fatal Python error: Illegal instruction``. Root cause was the pinned
triton-cpu wheel: it built its LLVM target machine from the CPU *model* with an
empty feature string, so codegen ignored the CPUID/XCR0-sanitised
``llvm.get_cpu_features()`` and emitted AVX512 code on a host that never had
that state enabled. The wheel is now pinned to a rebuild carrying
triton-lang/triton-cpu#314 (see LMCache/vllm-cpu), and this check is what keeps
a bad pin from coming back: it compiles the same masked load/store path in ~2 s
and rejects any kernel that uses instructions outside the detected feature set.

Exit codes: 0 aligned, 1 mismatch (the job fails printing the offending
instructions and the installed triton-cpu provenance).
"""

# Standard
import os
import platform
import re
import subprocess
import sys
import tempfile

_PROBE = r"""
import torch
import triton
import triton.language as tl


@triton.jit
def _isa_probe(out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < BLOCK
    values = tl.load(out_ptr + offsets, mask=mask, other=-1)
    tl.store(out_ptr + offsets, values, mask=mask)


out = torch.zeros(1024, dtype=torch.int32)
_isa_probe[(1,)](out, BLOCK=1024)
assert int(out[0]) == 0 and int(out[-1]) == 0, out[:4]
print("probe kernel ran")
"""

# Mnemonics that only exist in the EVEX (AVX512) encoding space. AVX2 code uses
# vmovdqu/vpmaskmovd instead.
_EVEX_ONLY = re.compile(
    r"\b(vmovdqu(8|16|32|64)|vpternlog[dfq]|vpermi?2[dfq]|vpermt2[dfq]|vptestm[db]|"
    r"vpconflict[dfq]|vpmovm[2b]|vpbroadcastm|vcompressp[dfq]|vexpandp[dfq]|"
    r"vp2intersect[dfq]|vpmadd52[lh]uq|vpdpbusd|vpdpwssd)\b"
)
_K_MASK = re.compile(r"%k[0-7]\b")
_ZMM = re.compile(r"%zmm")
_YMM = re.compile(r"%ymm")


def _run_probe(dump_dir: str) -> str:
    os.makedirs(dump_dir, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "TRITON_CACHE_DIR": os.path.join(dump_dir, "cache"),
            "TRITON_KERNEL_DUMP": "1",
            "TRITON_DUMP_DIR": dump_dir,
            "TRITON_ALWAYS_COMPILE": "1",
        }
    )
    probe_path = os.path.join(dump_dir, "probe.py")
    with open(probe_path, "w") as handle:
        handle.write(_PROBE)
    result = subprocess.run(
        [sys.executable, probe_path],
        env=env,
        capture_output=True,
        text=True,
        cwd=dump_dir,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit("probe kernel failed to compile or run")
    asm_paths: list[str] = []
    for root, _, files in os.walk(dump_dir):
        asm_paths.extend(
            os.path.join(root, name) for name in files if name.endswith(".asm")
        )
    if not asm_paths:
        raise SystemExit("no generated assembly found; is TRITON_KERNEL_DUMP working?")
    with open(asm_paths[0]) as handle:
        return handle.read()


def _violations(asm: str, features: set) -> list:
    problems = []
    if "avx512f" not in features:
        for pattern, label in (
            (_ZMM, "%zmm registers"),
            (_K_MASK, "AVX512 mask registers"),
        ):
            found = pattern.findall(asm)
            if found:
                problems.append(f"{label} in {len(found)} place(s)")
        evex = _EVEX_ONLY.findall(asm)
        if evex:
            problems.append(f"EVEX-only mnemonics: {sorted(set(evex))[:4]}")
    if "avx2" not in features and _YMM.search(asm):
        problems.append("%ymm registers")
    return problems


def main() -> int:
    if platform.machine() != "x86_64":
        print(f"ISA check only applies to x86_64, skipping on {platform.machine()}")
        return 0
    try:
        # Third Party
        from triton._C.libtriton import cpu
    except ImportError:
        print("triton-cpu is not installed, skipping the ISA check")
        return 0

    cpu_name = cpu.llvm.get_cpu_name()
    features = set(cpu.llvm.get_cpu_features())
    print(f"triton-cpu {cpu_name}: {', '.join(sorted(features))}")

    with tempfile.TemporaryDirectory() as tmp:
        asm = _run_probe(os.path.join(tmp, "host"))
        problems = _violations(asm, features)
        if problems:
            print(
                "::error::triton-cpu codegen exceeds the host ISA: "
                + "; ".join(problems)
            )
            print(
                "::error::the pinned triton-cpu wheel ignores "
                "llvm.get_cpu_features(); it must carry "
                "triton-lang/triton-cpu#314. See LMCache/vllm-cpu "
                "build_triton_cpu_wheel.yml."
            )
            return 1
        print("host codegen is ISA-legal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
