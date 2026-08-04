# CPU nightly is broken end to end: three defects in `LMCache/vllm-cpu`

**Date:** 2026-08-04(investigating the 2026-08-03 failure)
**Reported from:** PR [#4240](https://github.com/LMCache/LMCache/pull/4240)
**Failing run:** [30825438265](https://github.com/LMCache/LMCache/actions/runs/30825438265)
**Auto-filed tracking issue:** [#4400](https://github.com/LMCache/LMCache/issues/4400)
**Verdict:** Not an LMCache regression, and **not** an unfixable upstream
problem either. `vllm-cpu-nightly` is built and published by
[`LMCache/vllm-cpu`](https://github.com/LMCache/vllm-cpu), so all three root
causes below are ours to fix.

---

## TL;DR

The macOS e2e failure that surfaced in PR #4240 is one symptom of a CPU nightly
pipeline that has been quietly broken for weeks. Three independent defects, all
in `LMCache/vllm-cpu`:

| # | Defect | Effect | Since |
| --- | --- | --- | --- |
| 1 | Build scripts hardcode `pip install torch==2.11.0` while the wheel's declared torch follows upstream `requirements/cpu.txt` (now `2.13.0`) | macOS wheel imports fine, then crashes as soon as a CPU worker starts | wheel of 2026-07-30 |
| 2 | Linux wheel grew past PyPI's 100 MiB per-file limit -> `400 Bad Request` on upload | **no Linux wheel published since 2026-07-19** | 2026-07-20 |
| 3 | `csrc/cpu/sgl-kernels/fla.cpp` uses `constexpr` + `sqrt`, which Apple clang rejects | macOS build fails too, so nothing at all is published now | 2026-08-03 |

Consequence: `LMCache/vllm-cpu`'s nightly workflow has been **red for 12
consecutive days** (2026-07-23 .. 2026-08-03) and nobody noticed, because
nothing consumes its failure signal.

Two things this incident also proves about the LMCache-side nightly verify:

1. An import-level check is not enough. Steps 9-12 of the failing job (install
   vLLM, install LMCache, both server bench modes) all **passed**; only the
   step that truly boots a vLLM engine failed. The pre-#4240 check would have
   stamped this wheel as "verified".
2. LMCache's own code is not implicated anywhere. Both `server_bench` legs
   exercise the LMCache KV transfer path and both were green.

---

## Symptom

`.github/scripts/cpu_vllm_e2e_test.sh` starts `vllm serve` and waits up to
`VLLM_READY_TIMEOUT` (300s) for readiness. The server never becomes ready
because `EngineCore` aborts during executor init:

```
WARNING [cpu.py:459] Failed to import from vllm._C: ImportError(
  'dlopen(.../site-packages/vllm/_C.abi3.so, 0x0002): Symbol not found:
   __ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE
   Referenced from: .../site-packages/vllm/_C.abi3.so
   Expected in:     .../site-packages/torch/lib/libc10.dylib')

ERROR [multiproc_executor.py:912] WorkerProc failed to start.
ERROR [multiproc_executor.py:912]   File ".../vllm/v1/worker/cpu_worker.py", line 71, in __init__
ERROR [multiproc_executor.py:912]     torch.ops._C.init_cpu_memory_env([memory_node])
ERROR [multiproc_executor.py:912] AttributeError: '_OpNamespace' '_C' object has no attribute 'init_cpu_memory_env'

RuntimeError: Engine core initialization failed. See root cause above.
  Failed core proc(s): {'EngineCore': 1}
```

Note the two-stage nature. The `dlopen` failure is only logged as a `WARNING` —
vLLM tolerates a missing `_C` on some platforms — so the run continues until
`cpu_worker.py:71` unconditionally calls `torch.ops._C.init_cpu_memory_env(...)`.
Because `_C` never registered, the op namespace is empty and the call raises
`AttributeError`. **The `AttributeError` is a symptom; the `dlopen` symbol
failure is the cause.**

### Failure is confined to engine startup

Step-level results for `macos-latest / opt-125m`
(job [91725839895](https://github.com/LMCache/LMCache/actions/runs/30825438265/job/91725839895)):

| # | Step | Result |
| --- | --- | --- |
| 9 | Install vLLM CPU (prebuilt nightly from PyPI) | success |
| 10 | Install lmcache (CPU-only, no vLLM) | success |
| 11 | Server bench — LMCache-driven | success |
| 12 | Server bench — Engine-driven | success |
| 13 | Prepare model — opt-125m | success |
| 14 | **vLLM e2e — Engine-driven (pickle)** | **failure** |
| 15-16 | remaining e2e legs | skipped |

Step 9 succeeds even though the wheel is broken, because
`install_vllm_cpu.sh` ends with a plain import:

```bash
python -c "import vllm, torch; \
print('vllm:', vllm.__version__, 'torch:', torch.__version__, \
      'cuda:', torch.cuda.is_available())"
```

That printed `vllm: 0.26.1.dev202607300835 torch: 2.13.0 cuda: False` and
exited 0. Importing the Python package never dlopens the extension.

---

## Evidence that this is not a PR regression

The same commit produced both a green and a red run in the same push,
differing only in which vLLM wheel got installed:

| Run | Trigger | vLLM version | ubuntu-22.04 (x2) | macos-latest (x2) |
| --- | --- | --- | --- | --- |
| [30825437380](https://github.com/LMCache/LMCache/actions/runs/30825437380) | `cpu_device.yml` own `pull_request` (macOS **pinned**) | `0.25.2.dev202607230832` | pass | **pass** |
| [30825438265](https://github.com/LMCache/LMCache/actions/runs/30825438265) | nightly reuse (`nightly_verify: true`, pin **lifted**) | `0.26.1.dev202607300835` | pass, pass | **fail, fail** |

Identical runner images, identical test code, identical models. PR #4240 did
not introduce the defect; it removed the pin that was hiding it.

---

## Defect 1 — build-time torch does not match declared torch

The missing symbol demangles to:

```
c10::impl::cow::materialize_cow_storage(c10::StorageImpl&)
```

a PyTorch-internal C++ ABI entry point. `_C.abi3.so` has an undefined
reference to that exact mangled name and `torch==2.13.0`'s `libc10.dylib` does
not export it. (`abi3` refers to the *CPython* stable ABI; libtorch has no
stable C++ ABI, so a torch extension must be built against the torch it runs
on.)

Why the mismatch happens — `scripts/build_and_publish_vllm_cpu_nightly_macos.sh`:

```bash
# line 169
pip install torch==2.11.0 >/dev/null
```

and the Linux script has the same hardcoding:

```bash
# scripts/build_and_publish_vllm_cpu_nightly.sh, line 191
pip install torch==2.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu >/dev/null
```

Meanwhile the *declared* dependency is whatever upstream vLLM's
`requirements/cpu.txt` says — the script only strips the `+cpu` local label so
PyPI will accept it:

```bash
sed -i.bak -E 's/torch==([0-9.]+)\+cpu/torch==\1/g' requirements/cpu.txt
```

So the build compiles against a **hardcoded** torch while the wheel advertises
a **tracked** one. As long as upstream said `2.11.0` the two agreed by
coincidence. Upstream bumped to `2.13.0` between 2026-07-23 and 2026-07-30, and
the hardcoded2.11.0 did not follow:

| Wheel | declared torch | compiled against | result |
| --- | --- | --- | --- |
| `0.25.2.dev202607230832` (07-23) | `2.11.0` | 2.11.0 | consistent, works |
| `0.26.1.dev202607300835` (07-30) | **`2.13.0`** | 2.11.0 | **ABI break** |

### Fix

Install exactly what the wheel will declare, and fail the build if that is
impossible — never publish a wheel whose ABI contradicts its metadata:

```bash
# Compile against the torch the wheel will declare. Reading it from the
# (already patched) requirements file means an upstream bump is picked up
# automatically instead of silently producing a mismatched binary.
TORCH_REQ="$(grep -E '^torch==' requirements/cpu.txt | head -n1)"
: "${TORCH_REQ:?could not read torch requirement from requirements/cpu.txt}"
pip install "${TORCH_REQ}" >/dev/null
```

Apply to both scripts (the Linux one keeps its `--extra-index-url`).

Optional hardening in the same scripts: after building, assert the extension
actually loads before uploading:

```bash
python -c "import vllm._C"   # dlopens the extension; fails on ABI mismatch
```

---

## Defect 2 — Linux wheel exceeds PyPI's 100 MiB file limit

`build-ubuntu` has failed every night since 2026-07-20while still *building*
successfully. From the 2026-08-03 log:

```
[10:22:44] Wheels staged on host:
-rw-r--r-- 1 root root 105M  vllm_cpu_nightly-0.26.1.dev202608030949-cp312-cp312-manylinux_2_28_x86_64.whl
[10:22:44] Uploading to PyPI via twine...
ERROR    HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
```

PyPI's default per-file limit is 100 MiB (104,857,600 bytes). Published Linux
wheel sizes:

| Date | Version | Size | Headroom |
| --- | --- | --- | --- |
| 2026-07-14 | `0.25.2.dev202607140808` | 96.55 MiB | +3.45 MiB |
| 2026-07-18 | `0.25.2.dev202607180753` | 96.73 MiB | +3.27 MiB |
| **2026-07-19** | `0.25.2.dev202607190821` | **96.72 MiB** | +3.28 MiB — **last one published** |
| 2026-08-03 | `0.26.1.dev202608030949` | **105 MiB** | **-5 MiB, rejected** |

The wheel had been running3 MiB below a hard limit for weeks and finally
crossed it. Nothing alerted, so Linux consumers have silently been pinned to
the 2026-07-19 buildever since.

The macOS wheels are ~9 MiB, which is why macOS was never affected by this.

### Fix (pick one)

1. **Request a file-size limit increase** for the `vllm-cpu-nightly` project on
   PyPI. Simplest, but needs a manual request and will need repeating.
2. **Shrink the wheel.** 105 MiB for a CPU-only build suggests unstripped debug
   info and/or a wide kernel matrix. Worth checking
   `-C strip=debuginfo` / `CMAKE_BUILD_TYPE=Release` / `strip -x` on the
   produced `.so` before packaging.
3. **Publish to GitHub Releases instead of PyPI**, which is what the LMCache
   main repo already does for its own nightlies
   (`gh release create nightly ... dist/cu130/*.whl` in `nightly_build.yml`).
   Consumers switch to `--find-links`. No size ceiling, and it removes the
   PYPI_TOKEN dependency.

Recommended: (2) as the real fix, (1) as an immediate unblock.

Also worth fixing regardless: the Rust extension fails on Linux and is merely
warned about —

```
build_rust: optional Rust extension vllm.vllm-rs failed
`cargo build --manifest-path rust/src/cmd/Cargo.toml ... --bin vllm-rs` failed with code 101
```

It is declared optional, so it does not fail the build, but it means the Linux
wheel ships without `vllm-rs`. Decide whether that is acceptable or should be
fatal.

---

## Defect 3 — macOS build no longer compiles

Since 2026-08-03 the macOS job fails as well, so *nothing* is being published:

```
csrc/cpu/sgl-kernels/fla.cpp:104:21: error: constexpr variable 'scale' must be
  initialized by a constant expression
csrc/cpu/sgl-kernels/fla.cpp:104:35: note: non-constexpr function 'sqrt' cannot
  be used in a constant expression
12 warnings and 8 errors generated.
ninja: build stopped: subcommand failed.
subprocess.CalledProcessError: Command '['cmake', '--build', '.', '-j=3',
  '--target=spinloop', '--target=fs_io_C', '--target=_C']' returned non-zero exit status 1
```

`std::sqrt` is not `constexpr` in libc++ before C++26; GCC accepts it as a
builtin, which is why Linux compiles the same source fine. This is genuine
upstream vLLM source that is not portable to Apple clang.

### Fix

- Short term: patch it in `vllm-cpu`'s build script (same place the script
  already patches `pyproject.toml` and `requirements/cpu.txt`), turning
  `constexpr` into `const` for that variable.
- Proper: report upstream to `vllm-project/vllm` — the fix is either `const`,
  or computing the reciprocal square root at runtime. This is the **only** part
  of this investigation that genuinely belongs to upstream vLLM.

---

## Why none of this was caught earlier

Worth writing down, because the process failure is more interesting than any
single defect:

- `LMCache/vllm-cpu`'s nightly has no failure notification. 12 consecutive red
  runs produced no issue, no comment, no alert. (LMCache's own nightly verify
  now does exactly this — the same pattern is worth porting to `vllm-cpu`.)
- The LMCache-side check was import-only, so even a fatally broken wheel
  reported "verified" (see the step table above).
- Linux resolves to the newest *available* wheel, which has been the 2026-07-19
  build for two weeks. The Linux leg therefore passes every night while
  validating nothing new — a green check that carries no information.

Net effect: **neither platform was really validating "today's CPU nightly"** —
macOS because the newest build is broken, Linux because there is no newest
build.

---

## Action plan

### In `LMCache/vllm-cpu` (owner of all three defects)

1. Read the torch requirement from `requirements/cpu.txt` instead of
   hardcoding `2.11.0`, in **both** build scripts. (Defect 1)
2. Add `python -c "import vllm._C"` as a pre-upload gate so an ABI-mismatched
   wheel can never be published again. (Defect 1)
3. Get the Linux wheel under 100 MiB, or move publishing to GitHub Releases.
   (Defect 2)
4. Patch `fla.cpp`'s `constexpr` so macOS builds again. (Defect 3)
5. Give that workflow a failure notification. Twelve silent red nights is the
   real bug behind the other four.

### In `LMCache/LMCache`

6. Add an ABI self-check to `.github/scripts/install_vllm_cpu.sh`
   (`import vllm._C` after the existing import). Moves the failure into the
   install step, so the recorded reason becomes "vLLM CPU nightly install
   failed on ..." instead of the misleading "CPU device tests failed on ...",
   and saves the model download plus the 300s readiness wait per leg.
7. Decide how the nightly should behave until `vllm-cpu` is fixed:
   - **A (recommended):** change nothing. macOS is correctly reported failed,
     ubuntu is still pinned, issue #4400 stays open, and the run goes green by
     itself once `vllm-cpu` is fixed. The red check only blocks PR #4240
     because of the *temporary* `pull_request` trigger, which is being removed
     before merge anyway.
   - **B:** `continue-on-error: ${{ inputs.nightly_verify && runner.os == 'macOS' }}`
     to keep the nightly workflow green while still recording the failure.
     **Blocking prerequisite:** `nightly-collect-verify-result.py` derives its
     verdict from `job.status`, and `continue-on-error: true` forces
     `job.status` to `success` — adopting B without first switching to an
     aggregation of per-step `outcome` values would record macOS as **passing**
     and wrongly pin it as tested.
   - **C:** drop macOS from the nightly e2e matrix. Saves 10x-billed runners,
     but with Linux stuck on a stale wheel it leaves the nightly validating
     nothing current.
   - **D:** re-pin macOS. Green, but a no-op — it will never notice the fix.

### In `vllm-project/vllm`

8. Report the `fla.cpp` `constexpr sqrt` portability break. Nothing else from
   this investigation is upstream's.

---

## Secondary issue found while investigating (fixed)

The CSV history lost the version of a failed platform. Aggregator, before:

```bash
    else
        pin_status=failed
        pin_version=""          # <-- real version discarded
        pin_reason="${OS_REASON[$os]}"
    fi
```

`nightly-collect-verify-result.py` deliberately keeps the version for failed
legs, and the tracking issue did show `0.26.1.dev202607300835`, but the
append-only CSV recorded `unknown`:

```
Verified vLLM version: unknown<- macOS (failed)
Verified vLLM version: 0.25.2.dev202607190821     <- ubuntu (tested)
```

Since `tested_vllm_versions.csv` is the long-term record used to answer "which
builds have broken before", a failure row with `unknown` is useless. Fixed by
resolving the version once, before branching on status:

```bash
    pin_version="${OS_VERSION[$os]:-}"
    if [ "${OS_STATUS[$os]}" = "ok" ]; then
        pin_status=tested
        pin_reason=""
    else
        pin_status=failed
        pin_reason="${OS_REASON[$os]}"
    fi
```

A failure row now carries the build that broke:

```json
{"vllm_version": "0.26.1.dev202607300835", "status": "failed",
 "reason": "CPU device tests failed on macos-latest / opt-125m",
 "os_platform": "macos-latest", "ci_platform": "github_actions"}
```

Pin semantics are unchanged: `latest_tested_vllm_<os>.txt` is still written
only for `status=tested`, and the pin script's `unknown` fallback still applies
when no leg reported a version at all.

Related, minor and still open: the issue body's "Failure details" lists only
the first failing leg per OS (by design, to avoid flooding), so
`opt-125m=failed` is visible in the Legs column but has no detail line.

---

## What worked correctly

Worth recording, because the machinery added in PR #4240 did its job:

- Per-OS folding produced the right verdicts:
  ```
  macos-latest  status=failed  version=0.26.1.dev202607300835  legs: deepseek-v2-lite=failed opt-125m=failed
  ubuntu-22.04  status=ok      version=0.25.2.dev202607190821  legs: deepseek-v2-lite=ok opt-125m=ok
  === Summary: 1 / 2 platforms passed ===
  ```
- ubuntu was still pinned as tested despite macOS being broken (per-OS pin).
- The tracking issue was created with correct labels and a per-leg table
  ([#4400](https://github.com/LMCache/LMCache/issues/4400)).
- The failure path of `pin-tested-vllm.sh` no longer crashes on a runner
  without vLLM installed.
- The e2e upgrade caught a wheel that the previous import-only check would have
  marked verified.

---

## References

- Failing jobs: [macos / deepseek-v2-lite](https://github.com/LMCache/LMCache/actions/runs/30825438265/job/91725839719), [macos / opt-125m](https://github.com/LMCache/LMCache/actions/runs/30825438265/job/91725839895)
- Green control run (macOS pinned): [30825437380](https://github.com/LMCache/LMCache/actions/runs/30825437380)
- Builder workflow: [`LMCache/vllm-cpu` nightly.yml](https://github.com/LMCache/vllm-cpu/blob/main/.github/workflows/nightly.yml)
- Builder failures: [08-03 (both jobs red)](https://github.com/LMCache/vllm-cpu/actions/runs/30802995445), [07-30 (linux red, macos green)](https://github.com/LMCache/vllm-cpu/actions/runs/30526854501)
- Build scripts: `scripts/build_and_publish_vllm_cpu_nightly.sh`,
  `scripts/build_and_publish_vllm_cpu_nightly_macos.sh`
- Consumer side: `.github/scripts/install_vllm_cpu.sh`,
  `.github/scripts/cpu_vllm_e2e_test.sh` (`VLLM_READY_TIMEOUT`, default 300s)
