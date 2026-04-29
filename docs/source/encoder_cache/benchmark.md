# Encoder Cache End-to-End Benchmark

This benchmark exercises the LMCache Encoder Cache (EC) connector with vLLM
on a real video-multimodal model and reports the time saved by reusing
encoder outputs across requests.

## Setup

| Component | Version / Path |
|---|---|
| Hardware | 1× NVIDIA H100 80GB (SM_90) |
| vLLM | benyebai/vllm @ `fix/lmcache-ec-connector-module-clean` (vLLM PR #38668) |
| LMCache | benyebai/LMCache @ `ec-localdisk-storage-manager` (this PR) |
| Model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Precision | bf16 |
| Video | Big Buck Bunny (10:34, 720p, 60 MB MP4) — Internet Archive |
| Prompt | "Describe what happens in this video in detail." |
| Decode budget | 32 tokens |
| `--media-io-kwargs` | `{"video": {"num_frames": 128}}` (heavier-encoder run) |

## EC connector configuration

vLLM is launched with the LMCache EC connector in `ec_both` mode (this
instance both produces EC entries and consumes them):

```json
{
  "ec_connector": "LMCacheECConnector",
  "ec_role": "ec_both",
  "ec_connector_module_path": "vllm.distributed.ec_transfer.ec_connector.lmcache_connector"
}
```

LMCache backend (`lmcache_ec.yaml`):

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 2          # GiB
local_disk: "file:///data2/sammshen/ec_e2e/ec_cache"
max_local_disk_size: 16        # GiB
```

## Methodology

The same chat-completion request is sent N+1 times to the same vLLM server.
The first call (cold) populates the EC cache: vLLM runs the vision encoder,
produces encoder outputs, and the EC connector writes them to the LMCache
storage backend. Subsequent calls (warm) hit the EC cache: the connector's
`start_load_caches` finds the entry, populates vLLM's `encoder_cache` dict
directly, and the encoder is **not run** for that request.

Time-to-first-token (TTFT) is the primary metric because the encoder runs
during prefill — any encoder savings show up as a TTFT delta. Total wall
time is reported for completeness; decode tokens-per-second is unaffected
by EC since the LM stack is identical.

## Results

Two configurations were measured, varying only `num_frames` (the number
of frames vLLM samples from the video before feeding the visual tower).
Larger `num_frames` makes the vision encoder a bigger share of prefill,
which is what EC caches.

### `num_frames = 32` (vLLM default)

EC entry size on disk: **34.3 MB** (one `bf16` tensor per `mm_hash`).

| Phase | TTFT (s) | Wall time (s) | Tokens out | Notes |
|---|---:|---:|---:|---|
| cold (populates EC) | 3.923 | 4.244 | 32 | encoder ran, `EC put` log |
| warm[0] (EC hit) | 3.038 | 3.374 | 32 | encoder skipped |
| warm[1] (EC hit) | 3.211 | 3.539 | 32 | encoder skipped |

- **warm TTFT (mean): 3.125 s**
- **speedup: 1.26×  (≈ 798 ms / request)**

### `num_frames = 128`

EC entry size on disk: **130.8 MB** per `mm_hash` (≈ 4× the 32-frame entry).

| Phase | TTFT (s) | Wall time (s) | Tokens out | Notes |
|---|---:|---:|---:|---|
| cold (populates EC) | 5.895 | 6.215 | 31 | encoder ran, `EC put` log |
| warm[0] (EC hit) | 3.438 | 3.761 | 31 | encoder skipped |
| warm[1] (EC hit) | 3.368 | 3.691 | 32 | encoder skipped |
| warm[2] (EC hit) | 3.321 | 3.647 | 32 | encoder skipped |

- **warm TTFT (mean): 3.375 s**
- **speedup: 1.75×  (≈ 2.52 s / request)**

The win grows with `num_frames` because the encoder workload scales
linearly with frame count while the rest of prefill (LM forward over
the resulting visual tokens + the short text prompt) scales sublinearly.

## Verifying EC hits

Three independent signals confirm the warm runs actually hit the cache,
not just incidental warm-up jitter:

1. **vLLM's own metric.** After the warm runs, the `loggers.py` line
   reports `MM cache hit rate: 66.7%` — exactly 2 of 3 (or 3 of 4 with
   one extra warm) `has_cache_item` queries hit. The cold request is the
   one miss.
2. **LMCache log line.** Cold runs emit
   `LMCache INFO: EC put: stored N bytes for mm_hash=H` exactly once per
   distinct video; warm runs emit no `EC put` (write path skipped).
3. **On-disk cache file.** Under `local_disk`, a single
   `<model>@1@0@<chunk_hash>@bfloat16.pt` file appears after the first
   request and is reused by every subsequent request. The `@1@0@` prefix
   reflects the deliberate sentinel `world_size=1, worker_id=0` in the
   EC cache key (see the design doc) — TP ranks share one entry.

## Caveats

- Speedup scales with how expensive the vision encoder is relative to the
  rest of prefill. For short text prompts and long videos (many tokens), the
  encoder dominates and the speedup is large; for short videos with long
  text prefill, the encoder is a smaller share and the speedup shrinks.
- Encoder outputs are stored at the model's encoder dtype. Changing the
  model dtype invalidates EC entries (this is intentional — see
  [`docs/design/v1/encoder-cache.md`](../../design/v1/encoder-cache.md)).
- All TP ranks share one EC entry per `mm_hash` (see the design doc for the
  rationale). With 1× H100 the test is single-rank, but multi-rank serving
  benefits identically and avoids storing N redundant copies.

## Reproducing

The full setup script and benchmark client live under `/data2/sammshen/ec_e2e/`
on the test host:

- `run_server.sh` — launches vLLM with the EC connector configured.
- `bench_client.py` — sends one cold + N warm requests to a running server,
  prints TTFT / wall-time per phase.
- `lmcache_ec.yaml` — LMCache config picked up by the EC connector.

```bash
# Start the server (waits for "Application startup complete")
./run_server.sh &
# Wait for ready, then run the bench
./bench_client.py --video videos/BigBuckBunny.mp4 --repeats 3
```
