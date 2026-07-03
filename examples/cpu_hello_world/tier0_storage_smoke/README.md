# Tier 0 — Storage smoke test (CPU, no model, no GPU)

The lowest-barrier proof that LMCache works on your machine. It starts an
`lmcache server` with a CPU cache and runs `lmcache bench server --mode cpu`,
which **stores KV chunks, retrieves them, and byte-compares the result**. No
inference engine, no model download, no accelerator.

## Run

```bash
# from the repo root, after examples/cpu_hello_world/install_cpu.sh
examples/cpu_hello_world/tier0_storage_smoke/run.sh
```

## What success looks like

```
==> Running: lmcache bench server --mode cpu (3 requests)
... CHECKSUM MATCH OK ...
... CHECKSUM MATCH OK ...
... CHECKSUM MATCH OK ...
==> PASS: 3/3 requests stored and retrieved with matching checksums on CPU
```

## What this proves

- LMCache's **store → retrieve** path is functional on plain host memory.
- The CPU-only build (`NO_GPU_EXT=1`) and its pure-Python memory fallback
  (`lmcache/python_ops_fallback.py`) are working.

## What it does *not* show

This exercises the storage layer in isolation — it does **not** involve a
language model or measure TTFT. For that, continue to
[Tier 1](../tier1_ttft_cpu/).

## Notes

- This mirrors the `server_bench` step of LMCache's own CPU CI
  (`.github/scripts/cpu_server_bench_test.sh`).
- `--transfer-mode engine_driven` is the default; `lmcache_driven` also works.
  POSIX shared-memory transports are Linux-only.
- An alternative, model-free storage check is
  `python -m lmcache.v1.basic_check --mode test_storage_manager` (see
  [`examples/basic_check/`](../../basic_check/)).
