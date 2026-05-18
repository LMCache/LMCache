# LMCache MP C++ Mirror

This folder is the first C++ slice of the LMCache multiprocess server path.
It currently contains two related pieces:

- A Python MP server bridge for the hbm-dram-disk flow.
- An experimental `lmcache-mp-server-native` C++ binary for the MP
  controller envelope and HTTP management frontend.

The C++-backed bridge path is:

1. vLLM owns HBM KV blocks and talks to the existing LMCache MP protocol.
2. The Python MP boundary keeps the current ZMQ/msgpack compatibility with vLLM.
3. Existing LMCache CUDA copy code moves data between HBM and CPU buffers.
4. `lmcache_mp_cpp` stores those CPU buffers in a C++ DRAM/disk tiered cache.

The native binary path is:

```text
vLLM/Python MessageQueueClient
  -> existing ZMQ/msgpack request envelope
  -> lmcache-mp-server-native
  -> native controller handlers and C++ tiered cache status
```

This is intentionally not a full rewrite of the Python MP server yet. Replacing
the CUDA IPC tensor registration, KV store/retrieve data path, L2 adapters, and
all cache-control APIs in C++ should be done one component at a time against the
Python behavior. See `docs/native_mp_status.md` for the current parity matrix.

## Build

The Python wrapper builds the shared library on first import:

```bash
PYTHONPATH=LMCache-mp-cpp/python:$PYTHONPATH \
  uv run --python 3.12 python -c "from lmcache_mp_cpp import TieredCache; print('ok')"
```

The build uses `g++` directly and writes artifacts under
`LMCache-mp-cpp/.build/`.

Build the native binary from the repository root:

```bash
cmake -S . -B build-native -DLMCACHE_BUILD_NATIVE_MP=ON
cmake --build build-native --target lmcache-mp-server-native
```

Build the optional CUDA-aware native transfer path:

```bash
cmake -S . -B build-native-cuda \
  -DLMCACHE_BUILD_NATIVE_MP=ON \
  -DLMCACHE_ENABLE_CUDA=ON
cmake --build build-native-cuda --target lmcache-mp-server-native
```

Install the CMake-built binary into a prefix:

```bash
cmake --install build-native --prefix /opt/lmcache-native
```

Build and package the native binary through `setup.py`:

```bash
NO_CUDA_EXT=1 LMCACHE_BUILD_NATIVE_MP=1 \
  uv run --python 3.12 python setup.py build
```

Add `LMCACHE_NATIVE_MP_ENABLE_CUDA=1` to package the CUDA-aware binary as
`lmcache/bin/lmcache-mp-server-native-cuda`.

The main release-artifact workflow now smoke-tests a no-CUDA wheel built with
the opt-in native package flag and verifies the packaged
`lmcache-mp-server-native` binary runs from the wheel contents. A local
packaged no-CUDA binary protocol smoke also verifies valid `STORE` and
`RETRIEVE` requests return safe `(b'', False)` results with
`cuda_transfer_enabled=false`.
The main, cu12.9, and nightly CUDA wheel builds are also configured to set
`LMCACHE_BUILD_NATIVE_MP=1` and `LMCACHE_NATIVE_MP_ENABLE_CUDA=1` during
`cibuildwheel`; their wheel tests import `lmcache.c_ops`, verify
`lmcache/bin/lmcache-mp-server-native-cuda` is executable, and run `--help`
before uploading artifacts. Local CUDA-native wheel packaging has also been
smoke-tested with `LMCACHE_NATIVE_MP_ENABLE_CUDA=1`: the wheel contains an
executable `lmcache/bin/lmcache-mp-server-native-cuda`, and that packaged binary
responds to `--help`.

Build a ThreadSanitizer-instrumented native binary for concurrency checks:

```bash
cmake -S . -B build-native-tsan \
  -DLMCACHE_BUILD_NATIVE_MP=ON \
  -DLMCACHE_ENABLE_TSAN=ON
cmake --build build-native-tsan --target lmcache-mp-server-native
```

## Run The C++-Backed MP Server

Launch the MP server with the C++ tiered storage core:

```bash
PYTHONPATH=LMCache-mp-cpp/python:$PYTHONPATH \
  uv run --python 3.12 python -m lmcache_mp_cpp.server \
    --host localhost \
    --port 5555 \
    --chunk-size 256 \
    --l1-size-gb 4 \
    --eviction-policy LRU \
    --cxx-disk-path /tmp/lmcache-mp-cpp-disk \
    --disable-observability
```

Then point vLLM at the same MP connector configuration used by the Python MP
server, for example:

```bash
LMCACHE_CONFIG_FILE=/path/to/lmcache.yaml \
uv run --python 3.12 --with vllm vllm serve facebook/opt-125m \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":"5555"}}'
```

The default vLLM connector registration path still uses the existing
PyTorch-backed `CudaIPCWrapper`, and the native CUDA path can import those
handles in focused byte-movement tests. When LMCache `use_layerwise` is
enabled, the vLLM registration path now carries `use_layerwise=True` in
`layout_hints` so the native server can record the layerwise lifecycle mode
without changing the wire schema. A focused connector test also covers MLA
rank normalization, where TP=4 changes `(world_size=8, rank=5)` into
`(kv_world_size=2, kv_rank=1)` before the MP adapter builds cache keys. For raw
CUDA IPC experiments, opt in to raw handles explicitly:

```bash
lmcache server --native-cuda --l1-size-gb 1 --eviction-policy LRU
```

Then start vLLM with raw CUDA IPC registration enabled:

```bash
uv run --python 3.12 --with vllm vllm serve facebook/opt-125m \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.port":"5555","lmcache.mp.raw_cuda_ipc":true}}'
```

The same worker-side behavior can be enabled with
`LMCACHE_MP_RAW_CUDA_IPC=1` or `LMCACHE_MP_NATIVE_RAW_CUDA_IPC=1`. This raw
mode is intended for the native C++ CUDA path and still requires real vLLM
parity validation before it can be treated as the default behavior.

## Run The Native Controller-Envelope Server

```bash
lmcache server --native --l1-size-gb 1 --eviction-policy LRU
```

`--native` selects the CUDA-enabled native build by default, because this is
the path that can move vLLM CUDA KV bytes. `--native-cuda` is kept as an
explicit synonym:

```bash
lmcache server --native-cuda --l1-size-gb 1 --eviction-policy LRU
```

The equivalent environment switches are `LMCACHE_MP_NATIVE=1` and
`LMCACHE_MP_NATIVE_CUDA=1`; truthy values `1`, `true`, `yes`, and `on` are
accepted, while falsey values `0`, `false`, `no`, and `off` leave the default
Python path selected. The same truthy/falsey parsing is covered for
`LMCACHE_MP_NATIVE_NO_CUDA`. Focused CLI tests verify the env-var routes use
the native launcher while `--python` still forces the Python HTTP server path.
A prebuilt binary can be selected with
`LMCACHE_MP_NATIVE_CUDA_BINARY=/path/to/binary`.
If the package contains
`lmcache/bin/lmcache-mp-server-native-cuda`, the launcher uses it before
falling back to source-checkout builds. For controller-only no-CUDA checks, use
`--native-no-cuda` or
`LMCACHE_MP_NATIVE_NO_CUDA=1`; that path still honors
`LMCACHE_MP_NATIVE_BINARY`, can use packaged
`lmcache/bin/lmcache-mp-server-native`, and cannot move vLLM CUDA KV bytes.
The native binary also accepts `--log-level LEVEL` to initialize the native
`lmcache` logger level exposed by `/loglevel`; invalid levels fail before the
server starts.
The `lmcache server` native launcher can seed supported startup fields from
`--config-file`, `LMCACHE_CONFIG_FILE`, or the matching LMCache environment
variables: `chunk_size` / `LMCACHE_CHUNK_SIZE`, `max_local_cpu_size` /
`LMCACHE_MAX_LOCAL_CPU_SIZE` for `--l1-size-gb` when the CLI does not provide
one, `cache_policy` / `LMCACHE_CACHE_POLICY` for `--eviction-policy` when the
CLI does not provide one, and `local_disk` / `LMCACHE_LOCAL_DISK` as native
filesystem L2 adapters when `--l2-adapter` is not already set. Native mode
keeps config-file values ahead of conflicting engine environment variables.
Native mode still rejects unsupported config-file or environment-driven server
modes such as remote storage, PD/P2P, runtime plugins, storage plugins, and
non-local CPU tiers instead of silently ignoring them. The direct C++ binary
also accepts flat top-level YAML/JSON for the same supported startup keys through
`--config-file` or `LMCACHE_CONFIG_FILE`; use `lmcache server --native` for full
LMCache config parsing.

Native mode fails before exec for Python-only options that are not implemented
by the native server yet, including non-`blake3` hash algorithms, blend engine
mode, runtime plugin configuration, separate CPU/GPU worker-pool sizes, and
non-`LRU` eviction policies. Non-default Python L1 allocator, lock TTL,
eviction-watermark, eviction-ratio, and L2 store/prefetch policy knobs are also
rejected rather than silently ignored. The same is true for Python
EventBus/OTel/tracing, lookup-hash logging, trace-recording, and standalone
Prometheus-port knobs that native mode does not implement yet.
Use the Python fallback for those server modes.

Force the Python fallback:

```bash
lmcache server --python --l1-size-gb 1 --eviction-policy LRU
```

## Current Scope

Implemented:

- C++ DRAM/disk byte cache with LRU spill and promotion.
- Exact native cache status for resident DRAM bytes and disk-tier bytes,
  including odd-sized spill, promotion, and replacement cases without alignment
  rounding, deterministic duplicate replacement of spilled entries, plus
  rollback when a new store or overwrite cannot complete the required disk
  spill.
- Focused LRU tests verify resident reads refresh the entry before the next
  spill and disk promotion refreshes the promoted entry before spilling the
  next-oldest resident.
- Cumulative LRU spill eviction count in cache stats and native `/status`.
- C++ lock/pin protection for resident entries so LRU spill, explicit remove,
  and safe direct-cache clear skip active entries, with status accounting for
  both locked-entry count, total lock/refcount depth, and locked bytes. The
  MP `CLEAR` request and HTTP `clear-cache` endpoint use the Python-compatible
  force-clear behavior and remove active lookup locks.
- Locked disk-tier reads avoid promotion when other active locks already fill
  the L1 budget, so retrieve can serve the bytes without creating temporary
  over-capacity unlock failures.
- Lookup status and Prometheus counters for aggregate hits/misses, partial
  hits, L1 hits, L2 hits, L2 misses, and derived `cache_hit_rate`.
- Native worker-pool status fields for `worker_count`, `worker_queue_depth`,
  `max_worker_queue_depth`, `response_queue_depth`, and
  `active_worker_count`; `queue_full_count` tracks bounded-queue
  backpressure events.
- Native client-count status fields for `active_client_count` and
  `observed_client_count`. The current native loop counts valid ZMQ client
  identities observed since startup; it does not yet expire identities on
  disconnect.
- Native request-latency status fields for count, total/max microseconds, and
  fixed histogram buckets.
- Native lookup locks tracked by request id, range-scoped
  `FREE_LOOKUP_LOCKS`, `END_SESSION` cleanup, and stale-lock cleanup when a
  lookup id is reused. CUDA-backed tests now cover subset lock release across
  two chunks, verify protocol `CLEAR` force-removes remaining lookup-locked
  chunks, verify request-id reuse replaces the previous lookup lock set without
  accumulating lock depth, verify two request ids can hold the same chunk with
  `locked_entries=1` and `lock_count=2` before per-owner cleanup decrements the
  refcount, verify RETRIEVE only takes a temporary transfer lock while a lookup
  lock for the same chunk remains held, verify `END_SESSION` releases a lookup
  lock so a following `CLEAR` can remove the chunk, and verify a
  partial-missing RETRIEVE releases an acquired transfer lock before returning
  failure. A CUDA-backed unregister test verifies `UNREGISTER_KV_CACHE` removes
  the registered context and later STORE/RETRIEVE requests for that instance
  fail without leaking locks.
- C++ default `blake3` rolling chunk hashes, KV-rank expansion, and ObjectKey
  string compatibility helpers, with focused byte-for-byte tests against Python
  for nontrivial token values, chunk windows, all-rank and worker-specific
  ObjectKeys, and cache-salt variants. A direct native
  `IPCCacheEngineKey` expansion wrapper also compares the server-side
  ObjectKey path against Python for all-rank, worker-specific, and empty
  start/end ranges.
- Python ctypes wrapper for the C++ cache.
- MP-compatible storage-manager bridge used by a C++-backed MP server.
- Native binary build with ZMQ controller-envelope handlers.
- Native HTTP `healthcheck`, `status`, and `clear-cache` endpoints.
- Native HTTP `conf`, version, `loglevel`, `threads`, `periodic-threads`,
  `periodic-threads-health`, `quota`, `metrics`, and `metrics/reset`
  endpoints. The Python launcher passes package version metadata into the
  native process, `loglevel` provides the same plain-text get/set surface as the
  Python common route for native logger names, `threads` reports the native
  HTTP/ZMQ/worker thread summary, `periodic-threads` reports an empty native
  registry, `quota` exposes the Python CRUD/reporting shape for native quota
  metadata plus filesystem-L2 usage, `kvcache/check` exposes the Python
  validation/error surface for native registered contexts plus CUDA checksum
  results for supported block-native NHD/HND/MLA layouts, and the metrics
  endpoint uses Prometheus text output for the native counters and gauges
  currently exposed in `/status`.
- Diagnostic KV checksum computation still returns explicit errors for
  unsupported KV formats. Native quota metadata is not an `IsolatedLRU`
  enforcement implementation; native eviction is still LRU-only today.
- CLI launch path for CUDA-enabled `lmcache server --native`, explicit
  `--native-cuda`, controller-only `--native-no-cuda`, and `--python`
  fallback.
- Protocol version/schema documentation plus Python/native `RequestType`
  constant compatibility checks.
- Native decoding for `REGISTER_KV_CACHE`, `LOOKUP`, `STORE`, `RETRIEVE`,
  `FREE_LOOKUP_LOCKS`, and `UNREGISTER_KV_CACHE` metadata frames, including
  engine/layout-hint validation, scalar `compress_ratio` validation against
  registered KV block-size metadata, acceptance and `/status` reporting of
  layerwise hints for supported per-layer wrapper metadata, rejection of
  unsupported layer-group descriptors, TRT-LLM reshape-hint validation, and
  key/hash expansion for key-bearing requests.
- Native-friendly `CudaIPCWrapper` serialization replaces opaque pickle bytes
  inside the msgpack Ext payload for current clients, while Python deserialization
  keeps a legacy pickle fallback. The native server decodes registered wrapper
  dtype, shape, stride, device UUID, block count, block size, and CUDA IPC
  handle tuple metadata into `/status`.
- Native `STORE`/`RETRIEVE` safe-failure paths validate token range alignment,
  block-id counts, block-id ranges, and retrieve skip-token alignment against
  the registered layout metadata.
- Optional native CUDA IPC transfer layer behind `LMCACHE_ENABLE_CUDA`. It
  preserves CUDA IPC handle bytes from registration and compiles a basic D2H/H2D
  path for homogeneous and heterogeneous per-layer vLLM NHD/HND, layerwise-
  hinted heterogeneous per-layer NHD, compressed NHD, mixed-compression NHD,
  compact 4D NHD, cross-layer NHD/HND, and MLA layouts, plus TRT-LLM's 4D HND
  KV-pool layout. Skip-gated GPU pytests cover default PyTorch CUDA IPC
  NHD/HND, compressed NHD, mixed-compression NHD, compact 4D NHD,
  heterogeneous NHD, layerwise-hinted heterogeneous NHD,
  larger NHD, cross-layer NHD/HND, TRT-LLM 4D, default PyTorch MLA-shaped
  tensors, multi-chunk PyTorch CUDA IPC STORE/LOOKUP/RETRIEVE, and raw CUDA
  IPC NHD/HND STORE/RETRIEVE byte movement. Native CUDA IPC transfers are
  serialized around CUDA runtime handle open/copy/close calls so concurrent
  worker requests cannot overlap process-global CUDA IPC operations. The default
  no-CUDA build still returns safe failed tuples for valid KV transfer
  requests. A focused test also covers aligned `skip_first_n_tokens` retrieval
  so skipped target tokens remain untouched.
- Optional vLLM connector raw CUDA IPC registration through
  `kv_connector_extra_config={"lmcache.mp.raw_cuda_ipc": true}` or worker-side
  `LMCACHE_MP_RAW_CUDA_IPC=1` / `LMCACHE_MP_NATIVE_RAW_CUDA_IPC=1`. The default
  connector behavior remains the existing PyTorch `CudaIPCWrapper`.
- Native filesystem L2 adapter config (`{"type":"fs","base_path":"..."}`),
  key filename compatibility, byte put/get/delete/clear helpers, and `/status`
  reporting.
- Native LOOKUP checks configured filesystem L2 adapters after L1 misses and
  reports filesystem L2 metadata hits through the existing prefetch query
  calls. Native `CLEAR` also clears configured filesystem L2 files, matching
  Python MP's storage-manager-wide force clear semantics for this supported L2
  adapter.
- Native CUDA STORE writes successful chunks through to configured filesystem
  L2 adapters while holding transfer locks, and native CUDA RETRIEVE hydrates
  missing L1 chunks from filesystem L2 before copying bytes back to GPU while
  holding transfer locks. A focused restart test now verifies that bytes stored
  through filesystem L2 by one native process can be loaded and retrieved by a
  second native process pointed at the same L2 directory, and a focused missing
  retrieve test verifies absent keys return a false transfer result without
  leaking locks. A two-chunk partial-missing RETRIEVE test verifies the native
  server releases the already-acquired transfer lock for the first chunk when a
  later chunk is absent.
- Native `REPORT_BLOCK_ALLOCATION` validates `BlockAllocationRecord` payloads
  and exposes report/record accounting in `/status`.
- Golden trace capture/replay covers controller requests, block-allocation
  reports, token-key `LOOKUP`, prefetch query calls, and native status metadata
  for block-allocation accounting. With
  `--include-pytorch-cuda-kv` or `--include-raw-cuda-kv`, it records one
  Python CUDA IPC STORE/LOOKUP/RETRIEVE byte checksum per trace file and
  verifies native replay against it. The trace rows now carry explicit CUDA KV
  `layout`, `kv_layout`, and shape metadata; Python-captured PyTorch NHD,
  HND, compressed NHD, mixed-compression NHD, compact 4D NHD, heterogeneous
  NHD, larger NHD, multi-chunk NHD, wide NHD/HND, cross-layer NHD/HND, TRT-LLM
  4D, and MLA-shaped traces plus raw CUDA NHD/HND traces have native replay
  coverage. `--include-fs-l2-partial-hit` adds a temporary filesystem-L2 seed
  row and verifies a one-hit/one-miss partial lookup hit count during replay.
  CUDA trace rows now also ask native replay to verify that the lookup lock is
  still held after `RETRIEVE` completes and before `FREE_LOOKUP_LOCKS` runs.
  `--include-layerwise-hint` records `use_layerwise=True` in the CUDA KV
  registration row for supported wrapper metadata; layerwise-hinted NHD, HND,
  heterogeneous NHD, compressed NHD, mixed-compression NHD, compact NHD, larger
  NHD, multi-chunk NHD, wide NHD/HND, cross-layer NHD/HND, TRT-LLM 4D, and
  MLA-shaped PyTorch CUDA IPC traces now replay successfully against native
  CUDA. `--cuda-kv-layout ALL` appends all 14 PyTorch CUDA layout cases to one
  trace file for the layerwise-hinted replay matrix, and
  `--cuda-kv-lifecycle-cycles` repeats independent byte-checked
  STORE/LOOKUP/RETRIEVE/FREE/END_SESSION/UNREGISTER cycles per selected
  layout.
- Worker-side MP connector layerwise lifecycle handling now waits once for
  pending async retrieve futures from `wait_for_layer_load`, records failed
  retrieve block ids, keeps per-layer `save_kv_layer` as a no-op, and submits
  stores at `wait_for_save`; a focused unit test covers
  `start_load_kv` -> `wait_for_layer_load` -> `save_kv_layer` ->
  `wait_for_save`.
- Real vLLM smoke coverage through
  `benchmarks/mp_native_vs_python/vllm_native_smoke.py`: with vLLM 0.21.0 and
  `facebook/opt-125m`, one native CUDA MP server serves two vLLM processes, and
  the second process retrieves the four chunks stored by the first process in
  both default PyTorch CUDA IPC and opt-in raw CUDA IPC modes. A follow-up
  default PyTorch CUDA IPC run with `--kv-cache-layout HND` resolved vLLM's
  layout to `HND`, raised native retrieves from 0 to 1, raised cache hits from
  0 to 4, and kept `unsupported_count=0`. A cached
  `Qwen/Qwen2.5-0.5B-Instruct` run also passed, with native retrieves rising
  from 0 to 1, cache hits from 0 to 3, and `unsupported_count=0`.
- A cached `facebook/opt-1.3b` native run with two prompt variants and two
  concurrent reader processes passed with native retrieves rising from 1 after
  the writer to 5 after the readers, cache hits from 12 to 60,
  `transfer_lock_failure_count=0`, `unsupported_count=0`, and no release
  warnings in the native server stderr tail.
- A cached `Qwen/Qwen3-4B` native run with two prompt variants and two
  concurrent reader processes passed with native retrieves rising from 1 after
  the writer to 5 after the readers, cache hits from 7 to 35,
  `transfer_lock_failure_count=0`, `unsupported_count=0`, and clean native
  stderr.
- A cached `Qwen/Qwen3-4B` HND-layout native run resolved vLLM's layout to
  `HND`, raised native retrieves from 0 to 1, raised cache hits from 0 to 6,
  and kept `transfer_lock_failure_count=0`, `unsupported_count=0`, and clean
  native stderr.
- Real `facebook/opt-125m` native smokes with `--use-layerwise` passed for both
  default vLLM layout and explicit `--kv-cache-layout HND`: native retrieves
  rose from 0 after the writer to 1 after the reader, cache hits rose from 0 to
  4, `transfer_lock_failure_count=0`, `unsupported_count=0`, and native stderr
  stayed clean.
- A focused `Qwen/Qwen2.5-0.5B-Instruct` native layerwise HND smoke also
  passed: vLLM resolved `HND`, native retrieves rose from 0 after the writer to
  1 after the reader, cache hits rose from 0 to 3,
  `transfer_lock_failure_count=0`, `unsupported_count=0`, and native stderr
  stayed clean.
- A focused larger-model `Qwen/Qwen3-4B` native layerwise HND smoke also
  passed: vLLM resolved `HND`, native retrieves rose from 0 after the writer to
  1 after the reader, cache hits rose from 0 to 5,
  `transfer_lock_failure_count=0`, `unsupported_count=0`, and native stderr
  stayed clean.
- A broader layerwise HND `facebook/opt-125m` run with two prompt variants, one
  writer, two sequential reader processes, one warmup round, and two measured
  steady-state rounds per process also passed: native retrieves rose from 5
  after the writer to 17 after the second reader, cache hits rose from 60 to
  204, `transfer_lock_failure_count=0`, `unsupported_count=0`, and native
  stderr stayed clean.
- A smaller real-vLLM layerwise HND concurrent-reader run with one writer and
  two simultaneous reader processes also passed: native retrieves rose from 0
  after the writer to 2 after the readers, cache hits rose from 0 to 12,
  `transfer_lock_failure_count=0`, `unsupported_count=0`, and native stderr
  stayed clean.
- Focused four-client native CUDA concurrency tests now cover PyTorch CUDA IPC
  STORE/LOOKUP/RETRIEVE round trips with and without `use_layerwise=True`
  registration hints. A gated native CUDA concurrency stress test runs 8 clients
  through 4 STORE/LOOKUP/RETRIEVE rounds each and asserts 32 stores, 32 lookups,
  32 retrieves, 64 transfer locks, zero transfer-lock failures, zero unsupported
  requests, and no leaked locks.
- A focused CUDA lock lifecycle test holds a lookup lock, rejects a duplicate
  `STORE` for the same ObjectKey, frees the lookup lock, and retrieves the
  original byte-identical KV payload.
- The vLLM smoke harness can enforce clean native server stderr through
  `--require-clean-native-stderr`; the current `facebook/opt-125m` gated run
  reported `clean_native_stderr=true` and no unexpected native stderr lines
  while still proving a reader-side native retrieve.
- The vLLM smoke harness also supports `--kv-cache-layout`, `--use-layerwise`,
  `--batch-size`, and `--prompt-repetitions`; the current longer native run
  used two prompt variants with `prompt_repetitions=40`, increased native
  retrieves from 1 to 3 after the second process, and kept
  `unsupported_count=0`. Per-run status summaries now include
  `cache_hit_rate` derived from the reported hit/miss counters.
- The vLLM smoke harness can also capture metadata-only real MP request traces
  from spawned vLLM worker processes with `--mp-trace-output` and assert the
  expected `REGISTER_KV_CACHE`/`STORE`/`LOOKUP`/`QUERY_PREFETCH_STATUS`/
  `RETRIEVE` lifecycle with `--require-mp-trace-lifecycle`. A focused
  two-reader `facebook/opt-125m` layerwise HND trace captured 66
  request/response rows, including three registrations with
  `use_layerwise=true` and `kv_layout=HND`, one real vLLM STORE, two real vLLM
  RETRIEVEs, lookup/status calls, session cleanup, and unregister calls. This
  is lifecycle metadata evidence, not a byte-replay trace.
- The controller latency benchmark writes Python/native PING, NOOP, missing-key
  LOOKUP, or filesystem-L2 partial-hit LOOKUP latency reports with mean, p50,
  p95, p99, raw sample values, concurrent-client `requests_per_s`, and MP
  server `/proc` resource deltas for CPU, RSS, peak RSS, and thread count. The
  report remains scoped to controller-envelope latency and L2 metadata checks;
  it does not claim KV data-path parity.
- The same harness can run `--compare-python` to produce a startup-inclusive
  Python-vs-native report plus inner `llm.generate()` timing. The opt-125m
  report showed startup-inclusive Python second generation 17.356s, native
  second generation 17.026s, and `native_over_python=0.981`; generate-only
  second generation was Python 0.182s, native 0.431s, and
  `native_over_python=2.367`.
- The compare harness also supports warmup-controlled steady-state TTFT and
  throughput reporting, including mean, p50, p95, p99, and raw measured values
  for generate latency, output-token throughput, and TTFT. `--compare-python`
  also captures real-vLLM worker traces for both Python and native MP and adds
  `mp_request_latency_ms`, including mean, p50, p95, p99, and raw samples for
  actual MP `STORE`, `LOOKUP`, `RETRIEVE`, and other request types observed in
  the worker trace. A Qwen2.5-0.5B compare reported second-reader steady-state
  throughput of 98.195 output tokens/s for Python MP and 35.605 output
  tokens/s for native MP, second-reader mean TTFT of 0.035s for Python MP and
  0.141s for native MP, and native `unsupported_count=0`. A Mistral-7B compare
  reported Python
  66.040 output tokens/s and native 16.845 output tokens/s for the second
  reader (`native_over_python=0.255`), second-reader mean TTFT of 0.037s for
  Python MP and 0.195s for native MP, native retrieves increasing from 2 after
  the writer to 5 after the reader, `transfer_lock_failure_count=0`,
  `unsupported_count=0`, and clean native stderr.
- The vLLM smoke/compare report now includes MP server `/proc` resource
  snapshots and deltas. A focused native opt-125m run reported
  `total_cpu_s_delta=2.88`, `rss_bytes_delta=113246208`,
  `rss_peak_bytes=161480704`, one native retrieve after the reader, and
  `unsupported_count=0`.
- Explicit cache-blend protocol responses with the current Python response
  shapes: empty match lists for lookup variants and safe failed tuples for
  store/retrieve variants. The native handlers validate their current payload
  schemas and count malformed blend payloads in `/status`.
- Deterministic malformed protocol fuzzing through `tools/mp_protocol_fuzz.py`,
  covering malformed envelopes, invalid request-type frames, out-of-range
  request types, and malformed typed payloads while checking post-fuzz
  PING/NOOP liveness and invalid-payload accounting. The longer documented run
  now covers 1,024 deterministic malformed cases with
  `invalid_payload_delta=1022`, `request_count=1023`, and
  `unsupported_count=1023`.
- Focused tests for DRAM/disk spill correctness and MP protocol liveness.
- Malformed core KV `STORE`/`RETRIEVE` payload handling through raw ZMQ
  requests, including `/status` invalid-payload accounting and post-error
  liveness.
- Malformed ZMQ envelopes and out-of-range numeric request types are rejected
  and counted instead of being dispatched through the worker pool.
- Focused CUDA data-path stress coverage for eight repeated keyed
  STORE/LOOKUP/CLEAR/RETRIEVE cycles and four concurrent PyTorch CUDA IPC
  clients against one native server process.
- Gated longer CUDA stress coverage for 32 keyed PyTorch CUDA IPC
  STORE/LOOKUP/CLEAR/RETRIEVE cycles in one native server process.
- Current-source no-TSAN and ThreadSanitizer coverage for the malformed-frame
  plus concurrent PING/NOOP stress path through `tools/mp_tsan_stress.py`,
  including 8 clients, 800 handled requests, invalid-payload accounting for the
  malformed envelope, and a longer TSAN run with 8 clients and 4,000 handled
  requests, a 60-second current-source duration soak with 8 clients and 35,775
  handled requests, and a two-hour current-source duration soak with 8 clients
  and 4,240,701 handled requests. CUDA+TSAN coverage now includes a two-client
  concurrent PyTorch CUDA IPC STORE/LOOKUP/RETRIEVE run, the older
  single-client CUDA data-path run, and the all-layout CUDA lifecycle replay;
  broader prolonged CUDA/TSAN soak coverage is still missing.
- Filesystem-L2 CUDA recovery after both graceful native-server termination and
  SIGKILL before a second native process retrieves the stored bytes.
- Focused shutdown coverage sends SIGTERM to a live native binary and requires
  a clean zero exit after the HTTP frontend is ready.
- Real vLLM reuse smoke with one writer process followed by two reader
  processes against one native server; the current run used two prompt variants,
  increased native retrieves from 1 to 5, and kept `unsupported_count=0`.
- Concurrent real vLLM reuse smoke with one writer followed by two simultaneous
  reader processes against one native server; the current run increased native
  retrieves from 0 to 2, cache hits from 0 to 8, and kept
  `unsupported_count=0`.
- Concurrent HND real vLLM reuse smoke with `facebook/opt-125m`, two prompt
  variants, and two simultaneous readers passed with clean native stderr:
  native retrieves increased from 1 after the writer to 5 after the readers,
  cache hits from 12 to 60, and `unsupported_count=0`.
- Real vLLM reuse smoke with `mistralai/Mistral-7B-Instruct-v0.2` passed with
  one writer and one reader: native retrieves increased from 0 to 1, cache hits
  from 0 to 4, `transfer_lock_failure_count=0`, `unsupported_count=0`, and
  clean native stderr.
- Warmup-controlled native-vs-Python vLLM throughput and request-stat TTFT
  comparison inside one loaded LLM process. The opt-125m run reported Python
  133.692 output tokens/s and native 61.549 output tokens/s for the second
  reader, with mean TTFT of 0.028s for Python MP and 0.087s for native MP; the
  Qwen2.5-0.5B run reported Python 98.195 output tokens/s and native 35.605
  output tokens/s, with mean TTFT of 0.035s for Python MP and 0.141s for native
  MP.

Still Python for now:

- CUDA IPC tensor wrapping on the Python/vLLM connector side.
- Full KV `STORE`/`RETRIEVE` parity in production workloads. The native binary
  now has CUDA-gated PyTorch and raw CUDA IPC transfer paths, but it still needs
  broader vLLM validation beyond the current focused smoke coverage,
  full real vLLM layerwise lifecycle coverage, remaining engine-specific layout
  coverage, and more large golden replay cases.
- Non-filesystem L2 adapters.
  `type="nixl"` is rejected with an explicit native-not-implemented error.
- Full production benchmark sweeps beyond the current controller latency,
  real-vLLM request-latency, three-model steady-state throughput/TTFT, and
  focused larger-model smoke reports.

This keeps the first correctness comparison grounded in the existing Python MP
contract while moving the DRAM/disk storage behavior into C++.
