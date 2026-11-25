# Tier 2 — Live cache metrics (optional)

Once Tier 1 works, you can watch LMCache's behavior on a live dashboard. This
tier adds nothing new to LMCache itself — it points the existing observability
stack at the CPU (or weak-GPU) engine from Tier 1.

## What you get

The [`examples/observability/`](../../observability/) stack (OpenTelemetry →
Prometheus + Tempo → Grafana, all via `docker compose`) ships a pre-provisioned
**LMCache** dashboard showing cache hit rate and StorageManager read/write
throughput, plus per-request trace waterfalls
(`request → lookup → retrieve → store`).

## Run

1. Start the dashboards (CPU/laptop-friendly, Docker only):

   ```bash
   cd examples/observability
   docker compose up -d
   # Grafana: http://localhost:3000  (anonymous admin, no login)
   ```

2. Start the LMCache server **with observability exporters enabled** and the
   Tier 1 vLLM engine pointed at it (see `examples/observability/start-server.sh`
   for the exact server flags), then generate load — either re-run the Tier 1
   driver a few times, or use the built-in workload:

   ```bash
   lmcache bench engine --workload long-doc-qa   # after an engine is up
   ```

3. Open Grafana and watch the **hit rate** climb as repeated prefixes are
   served from cache.

## Hardware notes

- The **dashboards** run fine on a laptop with no GPU.
- The **trace/metric source** is a working LMCache + vLLM engine. Use the Tier 1
  CPU engine, or — if you have a **weak/consumer GPU** — drop the CPU env vars
  from the Tier 1 command and instead pass a low `--gpu-memory-utilization`
  (e.g. `0.3`) with a small `--max-model-len`; LMCache will offload KV to CPU
  RAM/disk, which is the whole point on a small GPU.

## Not shown here

The dashboard includes a collapsed **CacheBlend** row. CacheBlend (non-prefix
KV reuse) needs the blend engine and GPU scatter/re-RoPE, so it is **not**
demonstrable on this CPU/weak-GPU path — see the parent
[README](../README.md#what-you-cannot-demonstrate-on-this-hardware).
