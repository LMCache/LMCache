# CPU / Weak-GPU Hello-World

A hardware-free on-ramp to LMCache. Every other example needs a datacenter GPU;
this one runs LMCache — and, in Tier 1, a real language model — **entirely on
CPU**, using a small **open-weight, Apache-2.0** model. If you have a weak or
consumer GPU, the same steps apply and LMCache simply offloads KV to CPU/disk.

The goal is to *see LMCache actually cache and reuse KV*, on a laptop, in a few
minutes.

> This CPU path is the same one LMCache's own CI exercises on GPU-less Ubuntu
> and macOS runners (`.github/workflows/cpu_device.yml`); this example packages
> it as a friendly, self-contained walkthrough.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Linux or macOS | No GPU / CUDA / nvcc required. |
| Python 3.10–3.13 | `uv` recommended but not required. |
| ~4 GB free disk | For the vLLM CPU wheel and the ~0.5 B model (Tier 1). |
| Internet | Tier 1 downloads the model from Hugging Face on first run. |
| Docker | Tier 2 only (dashboards). |

## Install (once)

```bash
# from anywhere inside the repo
examples/cpu_hello_world/install_cpu.sh
# or, with uv:
PIP_BIN="uv pip" examples/cpu_hello_world/install_cpu.sh
```

This installs a CPU vLLM build and LMCache in CPU-only mode (`NO_GPU_EXT=1`).
See the script header for what each step does and why.

## The three tiers

| Tier | Command | Needs | Proves |
|------|---------|-------|--------|
| **0 — Storage smoke** | [`tier0_storage_smoke/run.sh`](tier0_storage_smoke/) | CPU only, **no model** | LMCache stores & retrieves KV on CPU (checksum verified) |
| **1 — TTFT on CPU** | [`tier1_ttft_cpu/run_demo.sh`](tier1_ttft_cpu/) | CPU + tiny model | A real **cache hit** and **TTFT reduction** on a repeated prefix |
| **2 — Live metrics** | [`tier2_observability/`](tier2_observability/) | Docker (+ a Tier 1 engine) | Cache hit rate on a live Grafana dashboard |

Start at Tier 0 (seconds, no download); it confirms the install before you
pull a model.

```bash
examples/cpu_hello_world/install_cpu.sh
examples/cpu_hello_world/tier0_storage_smoke/run.sh
examples/cpu_hello_world/tier1_ttft_cpu/run_demo.sh
```

---

## What you *can* demonstrate on CPU / weak GPU

- **Store & retrieve on CPU** — checksummed, no model (Tier 0).
- **A genuine cache hit** — write/read chunk counters move (Tier 1).
- **TTFT reduction on a repeated prefix** — cold vs. warm (Tier 1); the gap
  grows with longer shared contexts.
- **Prefix-driven hits** — a different prompt does not hit (Tier 1 negative
  control).
- **CPU-RAM (and, as an extension, local-disk) offload targets.**
- **Live cache-hit-rate metrics** on a dashboard (Tier 2).

## What you *cannot* demonstrate on this hardware

These need real GPUs / interconnects and are intentionally out of scope here:

- **CacheBlend** (non-prefix / blended KV reuse) — needs the blend engine + GPU.
- **fp8 / quantized serialization** — needs GPU kernels.
- **NIXL / RDMA / P2P** KV transfer and **PD (prefill/decode) disaggregation** —
  need multiple GPUs and a fast fabric.
- **MoE "10×" throughput** headlines — need real GPU decode throughput.

For those, see the GPU-based examples (`blend_kv_v1/`, `serde/fp8/`,
`disagg_prefill/`, `p2p/`).

---

## Model & license

Tier 1 defaults to **`Qwen/Qwen2.5-0.5B-Instruct`** — Apache-2.0, ungated,
freely redistributable, and from the same model family as LMCache's quickstart.
Other small permissive options you can pass via `MODEL=...`:

| Model | Params | License |
|-------|--------|---------|
| `Qwen/Qwen2.5-0.5B-Instruct` (default) | 0.5 B | Apache-2.0 |
| `Qwen/Qwen3-0.6B` | 0.6 B | Apache-2.0 |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 0.36 B | Apache-2.0 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1 B | Apache-2.0 |

> **Note:** LMCache's CI uses `facebook/opt-125m`, whose OPT license is
> **non-commercial / non-redistributable**. It is deliberately **not** used
> here so this example stays freely shareable. Verify any model's license
> before redistributing.

LMCache itself is architecture-agnostic (it caches KV-tensor chunks, not
model-specific state), so any small model your vLLM build supports should work.

## Troubleshooting

- **`vllm serve` won't start / "PackageNotFoundError: vllm".** The CPU wheel
  registers under `vllm-cpu-nightly`; `install_cpu.sh` writes a `+cpu`
  dist-info alias to fix this. Re-run the installer.
- **No cache hit / zero chunks.** Your prompt is shorter than one chunk. Keep
  the small `--chunk-size` (16) and a long shared prefix, and make sure the two
  requests share that prefix.
- **vLLM is slow to come up.** CPU startup can take a minute+; the driver waits
  up to 10 minutes (`VLLM_READY_TIMEOUT`).
- **macOS.** Use the default engine-driven transport (POSIX shared-memory
  transports are Linux-only). No other change needed.
- **No pinned/DMA memory on CPU.** LMCache falls back to regular host memory
  (`lmcache/python_ops_fallback.py`) — functionally correct, just without the
  page-locked-memory speedup.
