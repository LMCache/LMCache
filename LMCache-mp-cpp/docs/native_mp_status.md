# Native MP Server Status

This directory contains the first native C++ LMCache MP server artifacts.
The current native binary is useful for validating the build, CLI launch,
ZMQ envelope compatibility, and HTTP management frontend. It is not a full
drop-in replacement for the Python MP server yet.

## What Is Native

- `lmcache-mp-server-native` is a C++ binary built by CMake.
- The binary binds the same ZMQ ROUTER endpoint shape as the Python
  `MessageQueueServer`.
- It decodes the existing msgpack request UID and `RequestType` frames.
- It responds to `PING`, `GET_CHUNK_SIZE`, `NOOP`, and `CLEAR` using the
  same msgpack response encoding expected by `MessageQueueClient`. Native
  protocol `CLEAR` and HTTP `clear-cache` use Python-compatible force-clear
  semantics.
- It exposes `LMCACHE_MP_PROTOCOL_VERSION = 1` on the Python side and native
  protocol constants that are tested against every current Python `RequestType`.
- `docs/protocol_schema.md` documents all current request types, payload
  schemas, response schemas, serialization, and version-1 compatibility rules.
- It implements the default Python MP `blake3` rolling chunk-hash path,
  KV-rank expansion, and ObjectKey string serialization in C++, with
  byte-for-byte tests against Python helpers. The focused tests compare default
  and nontrivial token values, multiple chunk windows/chunk sizes, all-rank and
  worker-specific KV-rank expansion, empty and non-empty cache salts, and
  ObjectKey strings generated from real Python chunk hashes for multiple model
  names. A direct native `IPCCacheEngineKey` expansion wrapper also compares
  the server-side ObjectKey path against Python for all-rank, worker-specific,
  and empty start/end ranges.
- Native `LOOKUP` decodes the msgspec `IPCCacheEngineKey` map payload,
  computes the same lookup chunk hashes/ObjectKeys, records current hit
  counts, and serves `QUERY_PREFETCH_LOOKUP_HITS` / `QUERY_PREFETCH_STATUS`
  from that native bookkeeping.
- Native `REGISTER_KV_CACHE` / `UNREGISTER_KV_CACHE` decode and track the
  instance id, model name, world size, engine type, KV layout hint, logical
  block-size hint, and native-friendly CUDA IPC wrapper metadata such as dtype,
  shape, stride, device UUID, inferred block count, inferred block size, and
  CUDA IPC handle tuple metadata such as handle bytes, storage size, storage
  offset, and event synchronization flag. Native `STORE`, `RETRIEVE`, and
  `FREE_LOOKUP_LOCKS` decode their typed metadata frames and expand the same
  key/hash path. `STORE`/`RETRIEVE` additionally validate block-id counts,
  block-id ranges, and retrieve skip-token alignment against the registered
  layout metadata. A focused CUDA byte test verifies that aligned
  `skip_first_n_tokens` leaves skipped target tokens untouched while copying
  the remaining requested tokens.
- A CUDA-gated native transfer layer now opens PyTorch `CudaIPCWrapper` and
  raw CUDA IPC memory/event handles and implements basic D2H/H2D KV chunk
  movement for homogeneous and heterogeneous per-layer vLLM NHD/HND, compressed
  NHD, mixed-compression NHD, compact 4D NHD, cross-layer NHD/HND, TRT-LLM 4D,
  and MLA tensor layouts when built with `-DLMCACHE_ENABLE_CUDA=ON`. This path is
  covered by skip-gated pytests that store and retrieve default PyTorch CUDA IPC
  NHD/HND bytes, compressed NHD bytes, mixed-compression NHD bytes, compact 4D
  NHD bytes, heterogeneous NHD bytes, larger NHD bytes, default PyTorch
  cross-layer NHD/HND bytes, TRT-LLM 4D bytes, default PyTorch MLA-shaped bytes,
  a multi-chunk PyTorch CUDA IPC STORE/LOOKUP/RETRIEVE byte path with
  filesystem-L2 write-through, wide NHD/HND trace bytes, and raw CUDA IPC
  NHD/HND bytes with `RawCudaIPCWrapper`; an additional repeated-cycle test runs eight keyed
  PyTorch CUDA IPC STORE/LOOKUP/CLEAR/RETRIEVE iterations in one native server
  process, and a four-client CUDA test runs concurrent PyTorch CUDA IPC
  STORE/LOOKUP/RETRIEVE round trips against one native server process.
  CUDA IPC transfer calls are serialized across native workers before opening,
  copying through, or closing CUDA IPC handles, and a gated CUDA+TSAN test now
  covers a two-client concurrent PyTorch CUDA IPC STORE/LOOKUP/RETRIEVE run.
  Filesystem-L2 CUDA recovery is checked after both graceful first-server
  termination and SIGKILL before a second native process retrieves the stored
  bytes. The representative 8-test CUDA core matrix passed, and the native core
  source tree has no `TODO`, `FIXME`, or `stub` markers. The default build keeps
  CUDA transfer disabled and returns safe failed
  `(bytes, bool)` tuples for valid KV transfer requests.
- The default vLLM connector behavior remains the existing PyTorch
  `CudaIPCWrapper`, which the native CUDA path can now import in focused
  byte-movement tests. When LMCache `use_layerwise` is enabled, the vLLM
  registration path now includes `use_layerwise=True` in `layout_hints`, so the
  native server can record the layerwise lifecycle mode without changing the
  MP wire schema. A lightweight regression also covers MLA rank normalization,
  where TP=4 changes `(world_size=8, rank=5)` into
  `(kv_world_size=2, kv_rank=1)` before the MP adapter builds cache keys. The
  worker adapter can also opt into raw CUDA IPC registration for native CUDA
  transfer experiments through
  `kv_connector_extra_config={"lmcache.mp.raw_cuda_ipc": true}` or worker-side
  `LMCACHE_MP_RAW_CUDA_IPC=1` / `LMCACHE_MP_NATIVE_RAW_CUDA_IPC=1`.
- The C++ tiered cache has lock and pin counters. Locked or pinned resident
  entries are removed from the eviction LRU and are not spilled, explicitly
  removed, or safe-cleared until released; the native protocol `CLEAR` and HTTP
  `clear-cache` paths use Python-compatible force clear and remove active lookup
  locks. Locked disk-tier reads avoid promotion when other active locks already
  fill the L1 budget, so retrieve can serve bytes without creating temporary
  over-capacity unlock failures. Native CUDA `STORE` and `RETRIEVE` take
  temporary L1 locks while
  writing stored chunks to L2 and while retrieving chunks back to GPU, and
  `/status` exposes transfer-lock accounting. Lookup locks are tracked by
  request id, released for the `FREE_LOOKUP_LOCKS` key range, released by
  `END_SESSION`, and replaced safely if the same lookup id is reused. A
  CUDA-backed two-chunk test now verifies subset `FREE_LOOKUP_LOCKS` release
  before protocol `CLEAR` force-removes the remaining lookup-locked chunk. A
  CUDA-backed request-id reuse test verifies the previous lookup lock set is
  released before a replacement lookup records new locks, avoiding accumulated
  lock depth. A CUDA-backed same-chunk refcount test verifies two independent
  lookup request ids produce `locked_entries=1` and `lock_count=2`, then
  `FREE_LOOKUP_LOCKS` for one owner leaves `lock_count=1` until `END_SESSION`
  releases the second owner. A CUDA-backed RETRIEVE lifecycle test keeps the
  lookup lock held during RETRIEVE, verifies the temporary transfer lock is
  released afterwards, and then frees the lookup lock. A CUDA-backed duplicate
  `STORE` test verifies native rejects overwriting a lookup-locked ObjectKey and
  preserves the original byte-identical KV payload after the lock is freed. A
  separate CUDA-backed test verifies `END_SESSION` releases a lookup lock so a
  following `CLEAR` can remove the chunk. A partial-missing RETRIEVE test
  verifies native releases an acquired transfer
  lock when a later requested chunk is absent. A CUDA-backed unregister test
  verifies `UNREGISTER_KV_CACHE` removes the registered context and later
  STORE/RETRIEVE requests for that instance fail without leaking locks. Cache
  status reports locked-entry count, total lock/refcount depth, and locked
  bytes. The focused 13-test lock/pin lifecycle cluster passed with the
  CUDA-backed lookup-lock paths enabled.
- Native filesystem L2 adapter configs are accepted through `--l2-adapter`
  when `type` is `fs`. The implementation uses the same filename shape as
  Python's filesystem L2 adapter and exposes byte put/get/delete/clear helpers
  plus `/status` reporting. Native `LOOKUP` consults configured filesystem L2
  adapters after L1 misses and reports those metadata hits through the normal
  prefetch query calls. Native `CLEAR` now clears configured filesystem L2
  files, matching Python MP's storage-manager-wide force clear semantics for
  the supported filesystem L2 adapter. The CUDA-gated STORE path writes
  successful L1 chunks through to filesystem L2 while holding transfer locks,
  and the RETRIEVE path hydrates missing L1 chunks from filesystem L2 before
  copying bytes back to GPU while holding transfer locks. A missing-key
  RETRIEVE returns the native false `(bytes, bool)` result without leaking
  transfer locks. A focused CUDA restart test stores bytes through filesystem
  L2 in one native server process, starts a second native process against the
  same L2 directory, and verifies byte-identical retrieval with
  `l2_load_count=1`. A two-chunk partial-missing RETRIEVE test verifies the
  already-acquired transfer lock for the first chunk is released before native
  returns a false transfer result for the absent later
  chunk.
- Cache-blend protocol request types validate their current Python payload
  schemas, count malformed payloads in `/status`, and return explicit
  Python-compatible safe responses: empty match lists for lookup variants,
  failed `(bytes, bool)` tuples for store/retrieve variants, and no response
  for registration variants.
- Native `REPORT_BLOCK_ALLOCATION` decodes the Python
  `BlockAllocationRecord` dataclass list, counts reports and records, and
  exposes the latest summary in `/status`. It does not publish to Python's
  EventBus.
- Focused binary tests cover malformed short ZMQ frames followed by concurrent
  PING/NOOP traffic across multiple Python clients plus four-client PyTorch CUDA
  IPC KV STORE/LOOKUP/RETRIEVE data-path tests with and without layerwise
  registration hints. Malformed short envelopes are now counted in
  `invalid_payload_count`, and out-of-range numeric request types are rejected
  before they can alias a valid `uint8_t` request type.
  Malformed raw DEALER `STORE`/`RETRIEVE` payloads return native false
  responses, leave subsequent PING healthy, and increment
  `invalid_payload_count`. Malformed core metadata payloads for
  `REGISTER_KV_CACHE`, `UNREGISTER_KV_CACHE`, `LOOKUP`, `FREE_LOOKUP_LOCKS`,
  `END_SESSION`, `REPORT_BLOCK_ALLOCATION`, `QUERY_PREFETCH_STATUS`, and
  `QUERY_PREFETCH_LOOKUP_HITS` also return explicit native responses, leave
  subsequent PING healthy, and are counted in `invalid_payload_count` without
  increasing `unsupported_count`.
  Oversized ZMQ request frames are rejected before msgpack payload decoding,
  return a safe nil response when the request prefix is still usable, leave
  subsequent PING healthy, and are counted in `invalid_payload_count`.
  Current-source no-TSAN and ThreadSanitizer runs via
  `tools/mp_tsan_stress.py` now cover the malformed-frame plus 8-client,
  800-request concurrent PING/NOOP stress path; a longer ThreadSanitizer run
  covers 8 clients and 4,000 handled requests with no ThreadSanitizer report,
  a current-source 60-second duration soak covers 8 clients and 35,775 handled
  requests, and a current-source two-hour duration soak covers 8 clients and
  4,240,701 handled requests with matching request and latency counters.
  A CUDA+TSAN build now passes a two-client concurrent PyTorch CUDA IPC
  STORE/LOOKUP/RETRIEVE pytest, the older single-client CUDA data-path pytest,
  and the all-layout CUDA lifecycle trace replay with no ThreadSanitizer
  report.
  The malformed-frame stress runs also verify that the malformed short envelope
  increments `invalid_payload_count`;
  `tools/mp_protocol_fuzz.py` covers deterministic malformed envelopes and
  typed malformed payloads and verifies post-fuzz PING/NOOP liveness. The
  current longer fuzz run covers 1,024 malformed cases with
  `invalid_payload_delta=1024`, `request_count=1023`, and
  `unsupported_count=1021`; the current two-hour soak reports
  `duration_s=7200.0`, `expected_request_count=4240701`,
  `request_count=4240701`, `request_latency_count=4240701`,
  `active_client_count=8`, and `worker_count=8`; a gated CUDA stress test now
  runs 8 clients through 4 STORE/LOOKUP/RETRIEVE rounds each, asserting 32
  stores, 32 lookups, 32 retrieves, 64 transfer locks, zero transfer-lock
  failures, zero unsupported requests, and no leaked locks. The focused
  6-test current concurrency/error cluster passed. Focused shutdown coverage
  sends SIGTERM to a live native binary after the HTTP frontend is ready and
  requires a clean zero exit.
- It owns a C++ DRAM/disk tiered cache object and exposes cache counters in
  `/status`, including exact resident DRAM bytes and disk-tier bytes tracked
  from logical chunk sizes plus cumulative LRU spill eviction count. Focused
  tests cover odd-sized spill, promotion, duplicate replacement of a spilled
  entry, and replacement cases without alignment rounding, and a failed
  spill-store cleanup case verifies both a new store and an overwrite of an
  existing spilled entry are rolled back when the disk spill path fails before
  the updated entry becomes durable. LRU-order tests verify a resident read
  refreshes the touched entry before the next spill and disk promotion refreshes
  the promoted entry before spilling the next-oldest resident. The focused
  13-test accounting cluster also covers nested lock counts, locked-byte
  accounting, locked disk reads, clear and force-clear accounting, and
  filesystem-L2 partial-hit status counters.
- Lookup status and Prometheus metrics now separate aggregate cache hits/misses
  from `partial_hit_count`, `l1_hit_count`, `l2_hit_count`, `l2_miss_count`,
  and derived `cache_hit_rate`.
- `/status` also exposes native worker-pool observability fields:
  `worker_count`, `worker_queue_depth`, `max_worker_queue_depth`, and
  `response_queue_depth`, plus `active_worker_count`; `--max-queued-tasks`
  configures the bounded worker queue, and `queue_full_count` records
  backpressure events that return a safe nil response when the queue is full.
- `/status` exposes `active_client_count` and `observed_client_count` for
  valid ZMQ client identities observed by the native ROUTER loop. The current
  implementation does not receive disconnect lifecycle events, so this is an
  observed-client count rather than an expiring live-socket count.
- `/status` records native request latency with `request_latency_count`,
  `request_latency_total_us`, `request_latency_max_us`, and fixed histogram
  buckets from `le_100us` through `gt_100ms`.
- It provides native HTTP endpoints:
  - `GET /`
  - `GET /healthcheck`
  - `GET /status`
  - `GET /conf`
  - `GET /version`
  - `GET /lmc_version`
  - `GET /commit_id`
  - `GET /env`
  - `GET /loglevel`
  - `GET /threads`
  - `GET /periodic-threads`
  - `GET /periodic-threads/{thread_name}`
  - `GET /periodic-threads-health`
  - `GET /quota`
  - `GET /quota/{cache_salt}`
  - `PUT /quota/{cache_salt}`
  - `DELETE /quota/{cache_salt}`
  - `GET /kvcache/check`
  - `POST /clear-cache`
  - `GET /metrics`
  - `POST /metrics/reset`

The focused 5-test HTTP endpoint cluster passed, including the controller/HTTP
route matrix, missing-instance `/kvcache/check` validation, and CUDA checksum
responses for NHD, HND, and MLA layouts.

The native `/metrics` endpoint uses Prometheus text exposition for the
native counters and gauges already surfaced in `/status`. It does not yet
export the full Python OTel/EventBus metric set.

The native `/loglevel` endpoint mirrors the Python common route's plain-text
get/set and invalid-level response shapes for logger names tracked inside the
native process. It does not mutate Python logging handlers because the native
server does not run the Python logging runtime.

The native `/threads` endpoint mirrors the Python common route's plain-text
diagnostic shape where practical, including `name` and `thread_id` filtering,
and reports the native HTTP, ZMQ, and worker threads. Native C++ builds do not
expose Python stack traces.

The native periodic-thread endpoints mirror the Python common route's response
shape with an empty registry, because the native server does not run the Python
`PeriodicThreadRegistry`. The health endpoint reports healthy with zero
unhealthy threads, and named periodic-thread lookups return the same 404 JSON
shape when no thread exists.

The native quota endpoints mirror the Python route's CRUD and reporting
shapes, including the `_default` empty-salt sentinel and filesystem-L2
`current_usage_gb` aggregation. These quotas are native metadata only today:
native eviction still supports LRU, not Python `IsolatedLRU` quota enforcement.

The native `/kvcache/check` endpoint mirrors the Python route's validation and
error response shapes for native registered contexts, including missing
instances, malformed `block_ids`, non-positive `chunk_size`, and empty KV cache
metadata. CUDA-enabled native builds now compute Python-compatible aggregate
and `layerwise=true` MD5 checksum responses for supported block-native
NHD/HND/MLA layouts; unsupported KV formats still return explicit errors.

## What Is Preserved From Python

- Existing vLLM connector code and the Python `MessageQueueClient` contract
  remain unchanged.
- The Python `CudaIPCWrapper` msgpack Ext payload now uses a native-friendly
  msgpack metadata envelope with Python fallback support for legacy pickle
  payloads.
- Existing vLLM connector registration remains on the PyTorch-backed
  `CudaIPCWrapper` unless raw CUDA IPC is explicitly enabled.
- `lmcache server` keeps the Python implementation as the default path.
- `lmcache server --python` forces the Python path.
- `lmcache server --native` or `LMCACHE_MP_NATIVE=1 lmcache server` launches
  the CUDA-enabled native binary when the source tree or a configured binary is
  available. Truthy env values `1`, `true`, `yes`, and `on` are accepted for
  `LMCACHE_MP_NATIVE`, `LMCACHE_MP_NATIVE_CUDA`, and
  `LMCACHE_MP_NATIVE_NO_CUDA`; falsey values `0`, `false`, `no`, and `off`
  leave the Python path selected. Focused CLI tests verify the env-var routes
  and that `--python` overrides native env selection and still launches the
  Python HTTP server path.
- `lmcache server --native-cuda` or `LMCACHE_MP_NATIVE_CUDA=1 lmcache server`
  is an explicit synonym for the CUDA-enabled native build.
  `LMCACHE_MP_NATIVE_CUDA_BINARY` selects a prebuilt CUDA-enabled binary.
- `lmcache server --native-no-cuda` or `LMCACHE_MP_NATIVE_NO_CUDA=1` launches
  the no-CUDA native controller build. This is for controller-only checks; it
  cannot move vLLM CUDA KV bytes and still honors `LMCACHE_MP_NATIVE_BINARY`.
- The native binary accepts `--log-level LEVEL` to initialize the native
  `lmcache` and `lmcache.native` logger levels exposed through `/loglevel`.
  Invalid startup levels fail before the server binds.
- The native launcher rejects Python-only server options that would otherwise
  be silently ignored by native mode, including non-`blake3` hash algorithms,
  blend engine mode, and runtime plugin configuration.
- The `lmcache server` native launcher now consumes `--config-file`,
  `LMCACHE_CONFIG_FILE`, or matching LMCache environment variables for
  supported engine config fields before exec: `chunk_size` /
  `LMCACHE_CHUNK_SIZE`, `max_local_cpu_size` /
  `LMCACHE_MAX_LOCAL_CPU_SIZE` as native L1 size when not supplied on the CLI,
  `cache_policy` / `LMCACHE_CACHE_POLICY` as eviction policy when not supplied
  on the CLI, and `local_disk` / `LMCACHE_LOCAL_DISK` as filesystem L2
  adapters when no explicit `--l2-adapter` is set. Config-file values take
  precedence over conflicting engine environment variables. Unsupported
  config-file or environment-driven modes such as remote storage, PD/P2P,
  runtime plugins, storage plugins, and non-local CPU tiers still fail loudly.
  The direct C++ binary also accepts flat top-level YAML/JSON for the same
  supported startup keys through `--config-file` or `LMCACHE_CONFIG_FILE`; use
  `lmcache server --native` for full LMCache config parsing.

## Build

From the repository root:

```bash
cmake -S . -B build-native -DLMCACHE_BUILD_NATIVE_MP=ON
cmake --build build-native --target lmcache-mp-server-native
```

CUDA transfer compile check:

```bash
cmake -S . -B build-native-cuda \
  -DLMCACHE_BUILD_NATIVE_MP=ON \
  -DLMCACHE_ENABLE_CUDA=ON
cmake --build build-native-cuda --target lmcache-mp-server-native
```

CMake install check:

```bash
cmake --install build-native --prefix /tmp/lmcache-native-install
```

Opt-in Python package build check:

```bash
NO_CUDA_EXT=1 LMCACHE_BUILD_NATIVE_MP=1 \
  uv run --python 3.12 python setup.py build

NO_CUDA_EXT=1 LMCACHE_BUILD_NATIVE_MP=1 LMCACHE_NATIVE_MP_ENABLE_CUDA=1 \
  uv run --python 3.12 python setup.py build
```

The no-CUDA package build writes `lmcache/bin/lmcache-mp-server-native`; the
CUDA package build writes `lmcache/bin/lmcache-mp-server-native-cuda`. The
Python launcher checks those package paths before trying source-checkout CMake
builds or `PATH` lookup. The main release-artifact workflow now smoke-tests a
no-CUDA wheel built with the opt-in native package flag and verifies the
packaged native binary runs from the wheel contents. A local packaged no-CUDA
binary protocol smoke verifies valid `STORE` and `RETRIEVE` requests return
safe `(b'', False)` results, with `cuda_transfer_enabled=false`, `store_count=1`,
`retrieve_count=1`, `unsupported_count=2`, and `transfer_lock_failure_count=1`.
The main, cu12.9, and nightly CUDA wheel builds now pass
`LMCACHE_BUILD_NATIVE_MP=1` and `LMCACHE_NATIVE_MP_ENABLE_CUDA=1` through
`cibuildwheel`; their wheel tests import `lmcache.c_ops`, verify
`lmcache/bin/lmcache-mp-server-native-cuda` is executable, and run `--help`
before upload. A local CUDA-native `bdist_wheel` smoke with
`LMCACHE_NATIVE_MP_ENABLE_CUDA=1` verifies the wheel contains
`lmcache/bin/lmcache-mp-server-native-cuda` and that the packaged binary
responds to `--help`.

ThreadSanitizer compile check:

```bash
cmake -S . -B build-native-tsan \
  -DLMCACHE_BUILD_NATIVE_MP=ON \
  -DLMCACHE_ENABLE_TSAN=ON
cmake --build build-native-tsan --target lmcache-mp-server-native
```

From the native source directory:

```bash
cmake -S LMCache-mp-cpp -B LMCache-mp-cpp/.build/cmake -DLMCACHE_BUILD_NATIVE_MP=ON
cmake --build LMCache-mp-cpp/.build/cmake --target lmcache-mp-server-native
```

The Python launcher builds the binary on demand for editable checkouts.

## Run

```bash
lmcache server \
  --native \
  --l1-size-gb 1 \
  --eviction-policy LRU \
  --port 5555 \
  --http-port 8080
```

Controller-only no-CUDA build:

```bash
lmcache server \
  --native-no-cuda \
  --l1-size-gb 1 \
  --eviction-policy LRU \
  --port 5555 \
  --http-port 8080
```

Fallback:

```bash
lmcache server --python --l1-size-gb 1 --eviction-policy LRU
```

## Trace Tools

The current trace tools capture implemented controller requests,
`REPORT_BLOCK_ALLOCATION`, and the token-key `LOOKUP` / prefetch-query path
that can be validated without moving KV bytes. They also include a native
status expectation row so replay verifies block-allocation status metadata,
not only request responses:

```bash
python tools/mp_trace_capture.py --server python --output traces/mp_golden.jsonl
python tools/mp_trace_replay.py --server native --input traces/mp_golden.jsonl
```

They can also append Python-captured CUDA IPC KV byte round-trip rows and
verify the replayed GPU bytes against the recorded checksums:

```bash
python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --output traces/mp_golden_pytorch_kv.jsonl
python tools/mp_trace_replay.py --server native --input traces/mp_golden_pytorch_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-raw-cuda-kv \
  --output traces/mp_golden_kv.jsonl
python tools/mp_trace_replay.py --server native --input traces/mp_golden_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --include-raw-cuda-kv \
  --output traces/mp_golden_mixed_kv.jsonl
python tools/mp_trace_replay.py --server native --input traces/mp_golden_mixed_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --include-fs-l2-partial-hit \
  --output traces/mp_golden_fs_l2_partial_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_fs_l2_partial_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --include-layerwise-hint \
  --output traces/mp_golden_layerwise_hint_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_layerwise_hint_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --include-layerwise-hint \
  --cuda-kv-layout HND \
  --output traces/mp_golden_layerwise_hnd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_layerwise_hnd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --include-layerwise-hint \
  --cuda-kv-layout HETEROGENEOUS_NHD \
  --output traces/mp_golden_layerwise_heterogeneous_nhd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_layerwise_heterogeneous_nhd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --include-layerwise-hint \
  --cuda-kv-layout CROSS_LAYER_NHD \
  --output traces/mp_golden_layerwise_cross_layer_nhd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_layerwise_cross_layer_nhd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --include-layerwise-hint \
  --cuda-kv-layout CROSS_LAYER_HND \
  --output traces/mp_golden_layerwise_cross_layer_hnd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_layerwise_cross_layer_hnd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --include-layerwise-hint \
  --cuda-kv-layout TRTLLM_4D \
  --output traces/mp_golden_layerwise_trtllm_4d_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_layerwise_trtllm_4d_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --include-layerwise-hint \
  --cuda-kv-layout MLA \
  --output traces/mp_golden_layerwise_mla_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_layerwise_mla_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --include-layerwise-hint \
  --cuda-kv-layout ALL \
  --cuda-kv-lifecycle-cycles 2 \
  --output traces/mp_golden_layerwise_all_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_layerwise_all_kv.jsonl
```

The CUDA KV trace rows carry explicit `layout`, `kv_layout`, shape metadata, and
per-case `instance_id`/`cache_salt` values, so default PyTorch CUDA IPC and raw
CUDA IPC can coexist in one trace file without reusing a registered KV cache slot
or cache key. The filesystem-L2 partial-hit trace row uses a temporary seed
directory and captures a one-chunk L2 metadata hit followed by a one-chunk miss.
CUDA KV rows also mark `expect_lookup_lock_after_retrieve`, so native replay
checks `/status` after the byte-checked `RETRIEVE` and before
`FREE_LOOKUP_LOCKS` to verify the lookup lock remains held while the temporary
transfer lock has been released.
Real-vLLM smoke traces generated with `--require-kvcache-checksum-match` now add
a `vllm_kvcache_checksum_match` row when `--mp-trace-output` is set. Replay
validates that saved row by comparing the writer `STORE` checksum response
against each reader `RETRIEVE` checksum response. This is reusable checksum
evidence for real vLLM runs; a negative replay regression also verifies that
mismatched reader checksums fail replay. This is still not a replay of captured
CUDA IPC handles.
The validated Python-captured byte traces currently cover
homogeneous vLLM NHD/HND layout for raw CUDA IPC, plus default PyTorch CUDA IPC
NHD, HND, layerwise-hinted NHD/HND, heterogeneous NHD, and cross-layer NHD/HND,
TRT-LLM 4D, MLA-shaped, compressed NHD, mixed-compression NHD, compact 4D NHD,
heterogeneous NHD, larger NHD, multi-chunk NHD, and wide NHD/HND cases. A
single `--cuda-kv-layout ALL` capture appends all 14 PyTorch CUDA layouts to one
trace when broad layerwise-hinted replay coverage is needed. With
`--cuda-kv-lifecycle-cycles 2`, the trace contains 28 byte-checked CUDA rows and
each row covers STORE, LOOKUP, RETRIEVE, `FREE_LOOKUP_LOCKS`, `END_SESSION`, and
`UNREGISTER_KV_CACHE`. The individual non-layerwise layout captures are:

```bash
python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --cuda-kv-layout HND \
  --output traces/mp_golden_pytorch_hnd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_pytorch_hnd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-raw-cuda-kv \
  --cuda-kv-layout HND \
  --output traces/mp_golden_raw_hnd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_raw_hnd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --cuda-kv-layout COMPACT_NHD \
  --output traces/mp_golden_pytorch_compact_nhd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_pytorch_compact_nhd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --cuda-kv-layout COMPRESSED_NHD \
  --output traces/mp_golden_pytorch_compressed_nhd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_pytorch_compressed_nhd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --cuda-kv-layout MIXED_COMPRESSION_NHD \
  --output traces/mp_golden_pytorch_mixed_compression_nhd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_pytorch_mixed_compression_nhd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --cuda-kv-layout HETEROGENEOUS_NHD \
  --output traces/mp_golden_pytorch_heterogeneous_nhd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_pytorch_heterogeneous_nhd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --cuda-kv-layout LARGE_NHD \
  --output traces/mp_golden_pytorch_large_nhd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_pytorch_large_nhd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --cuda-kv-layout WIDE_NHD \
  --output traces/mp_golden_pytorch_wide_nhd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_pytorch_wide_nhd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --cuda-kv-layout WIDE_HND \
  --output traces/mp_golden_pytorch_wide_hnd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_pytorch_wide_hnd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --cuda-kv-layout MULTI_CHUNK_NHD \
  --output traces/mp_golden_pytorch_multi_chunk_nhd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_pytorch_multi_chunk_nhd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --cuda-kv-layout CROSS_LAYER_NHD \
  --output traces/mp_golden_pytorch_cross_layer_nhd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_pytorch_cross_layer_nhd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --cuda-kv-layout CROSS_LAYER_HND \
  --output traces/mp_golden_pytorch_cross_layer_hnd_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_pytorch_cross_layer_hnd_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --cuda-kv-layout TRTLLM_4D \
  --output traces/mp_golden_pytorch_trtllm_4d_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_pytorch_trtllm_4d_kv.jsonl

python tools/mp_trace_capture.py \
  --server python \
  --include-pytorch-cuda-kv \
  --cuda-kv-layout MLA \
  --output traces/mp_golden_pytorch_mla_kv.jsonl
python tools/mp_trace_replay.py \
  --server native \
  --input traces/mp_golden_pytorch_mla_kv.jsonl
```

Focused GPU tests additionally cover default PyTorch CUDA IPC HND, compressed
NHD, mixed-compression NHD, compact 4D NHD, heterogeneous NHD, larger and wide
NHD/HND tensors, cross-layer NHD/HND tensors, TRT-LLM 4D tensors, repeated CUDA
cycles, multi-chunk STORE/LOOKUP/RETRIEVE, the two-cycle
`--cuda-kv-layout ALL` layerwise-hinted trace matrix, and four-client
concurrent CUDA round trips with and without layerwise hints. Scalar
`compress_ratio` hints are accepted when they match the registered physical KV
block-size metadata.
Layerwise hints are accepted and exposed in `/status` for supported per-layer
wrapper metadata. Focused CUDA byte tests cover both a single-wrapper
layerwise-hinted registration and a two-wrapper heterogeneous per-layer
registration with `use_layerwise=True`. Unsupported layer-group descriptors,
inconsistent compression hints, and physical block sizes that do not divide the
logical block-size hint are rejected at `REGISTER_KV_CACHE`. TRT-LLM reshape
hints are accepted for the TRT-LLM 4D KV-pool shape and rejected for non-TRT
engines. The MP connector now has focused layerwise lifecycle coverage for
waiting on pending retrieve futures and finalizing stores, and the smoke
harness covers focused real-vLLM layerwise HND runs, metadata traces of real
vLLM MP request lifecycles, and opt-in live `/kvcache/check` writer-vs-reader
checksum comparison for real vLLM layerwise HND transfers. Saved
`vllm_kvcache_checksum_match` trace rows are replay-validated so the checksum
evidence is reusable, including negative coverage that rejects mismatched reader
checksums. Golden byte-replay traces do not yet cover reusable
real-vLLM-captured CUDA IPC handles, remaining engine-specific layouts, or
workloads beyond the smoke harness.

The CUDA IPC byte paths have focused GPU tests:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy --with cuda-python \
  pytest -q \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_pytorch_cuda_ipc_store_retrieve \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_repeats_pytorch_cuda_ipc_store_retrieve \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_multi_chunk_pytorch_cuda_ipc \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_rejects_store_over_lookup_locked_chunk \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_kvcache_check_returns_cuda_checksums \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_long_pytorch_cuda_ipc_store_retrieve \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_handles_concurrent_pytorch_cuda_ipc_round_trips \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_handles_layerwise_concurrent_pytorch_cuda_ipc_round_trips \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_pytorch_cuda_ipc_hnd_layout \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_compressed_pytorch_cuda_ipc_layout \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_mixed_compression_pytorch_cuda_ipc_layout \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_compact_pytorch_cuda_ipc_layout \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_layerwise_heterogeneous_cuda_ipc_layout \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_larger_pytorch_cuda_ipc_layout \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_pytorch_cuda_ipc_cross_layer_layout \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_pytorch_cuda_ipc_cross_layer_hnd_layout \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_trtllm_4d_pytorch_cuda_ipc_layout \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_pytorch_cuda_ipc_mla_layout \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_retrieve_respects_skip_first_tokens \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_raw_cuda_ipc_store_retrieve \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_raw_cuda_ipc_hnd_layout
```

The longer CUDA stress is gated out of the default suite. Run it explicitly
with:

```bash
LMCACHE_RUN_LONG_CUDA_STRESS=1 \
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}" \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy --with cuda-python pytest -q \
  tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_long_pytorch_cuda_ipc_store_retrieve
```

## Benchmark

The simplest standalone benchmark compares controller `PING` latency:

```bash
python benchmarks/mp_native_vs_python/controller_latency.py --iterations 100
```

The controller benchmark can measure `--request ping`, `--request noop`,
`--request lookup-miss`, or `--request lookup-fs-l2-partial`. The lookup modes
send real token-key `LOOKUP` requests that do not need registered CUDA KV cache.
The filesystem-L2 mode seeds one of two chunk metadata files before each server
run, so the measured path includes L2 metadata checks but not KV byte movement.
`--clients N` runs concurrent clients, with each client sending `--iterations`
requests. Each report contains Python/native mean, p50, p95, and p99 latency
fields, raw latency samples, and aggregate `requests_per_s`. It also records
best-effort `/proc` snapshots around the measured loop and reports CPU, RSS,
peak RSS, and thread-count deltas for each MP server process. This remains
controller-envelope coverage only, not KV data-path parity.

A 5-iteration smoke of the current report path wrote
`/tmp/lmcache-native-controller-bench.json` and reported Python mean/p50/p95
`2.230`/`2.232`/`2.351` ms with `rss_peak_bytes=842420224`, and native
mean/p50/p95 `3.101`/`3.111`/`3.368` ms with
`rss_peak_bytes=48234496`.

A 5-iteration `--request lookup-miss` smoke wrote
`/tmp/lmcache-native-controller-lookup-bench.json` and reported Python
mean/p50/p95 `2.665`/`2.568`/`3.385` ms with
`rss_peak_bytes=843055104`, and native mean/p50/p95
`2.971`/`3.123`/`3.206` ms with `rss_peak_bytes=48234496`.

The native binary has now passed a real vLLM 0.21.0 smoke with
`facebook/opt-125m`. The harness starts one native CUDA MP server, runs one
vLLM process to store a shared prompt, then runs a second vLLM process with the
same prompt to force server-side reuse. The default PyTorch CUDA IPC mode and
the opt-in raw CUDA IPC mode both produced four first-process cache misses,
four second-process hits, one native retrieve, and `unsupported_count=0`:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with vllm --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --output /tmp/lmcache-native-vllm-smoke.json

SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with vllm --with pytest --with numpy --with cuda-python \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --raw-cuda-ipc \
    --output /tmp/lmcache-native-vllm-raw-smoke.json
```

The smoke can also set `VLLM_KV_CACHE_LAYOUT` through `--kv-cache-layout`, and
can enable LMCache layerwise mode in each vLLM worker through `--use-layerwise`.
Each native run summary includes the raw hit/miss counters and a derived
`cache_hit_rate`.
A default PyTorch CUDA IPC run with `--kv-cache-layout HND` resolved vLLM's
layout to `HND`, increased native retrieves from 0 after the writer to 1 after
the reader, increased cache hits from 0 to 4, and kept `unsupported_count=0`:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --kv-cache-layout HND --batch-size 1 --prompt-repetitions 24 \
    --max-model-len 256 --worker-timeout-s 300 \
    --output /tmp/lmcache-native-vllm-hnd-smoke.json
```

This extends the `facebook/opt-125m` layout coverage, but it is still not full
vLLM parity.

`facebook/opt-125m` native smokes with `--use-layerwise` also passed for both
the default vLLM layout and explicit `--kv-cache-layout HND`. In each run, the
writer process stored one prompt (`store_count=1`, `cache_misses=4`), the
reader increased native retrieves from 0 to 1 and cache hits from 0 to 4, and
the run kept `transfer_lock_failure_count=0`, `unsupported_count=0`, and
`clean_native_stderr=true`:

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --use-layerwise --batch-size 1 --prompt-repetitions 24 \
    --max-model-len 256 --worker-timeout-s 300 \
    --require-clean-native-stderr \
    --output /tmp/lmcache-native-vllm-layerwise-smoke.json

LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --use-layerwise --kv-cache-layout HND --batch-size 1 \
    --prompt-repetitions 24 --max-model-len 256 --worker-timeout-s 300 \
    --require-clean-native-stderr \
    --output /tmp/lmcache-native-vllm-layerwise-hnd-smoke.json
```

A focused `Qwen/Qwen2.5-0.5B-Instruct` native layerwise HND smoke also passed.
vLLM resolved `HND`; the first process stored three chunks; the reader raised
native retrieves from 0 to 1, cache hits from 0 to 3, and transfer locks from 3
to 6; `transfer_lock_failure_count=0`, `unsupported_count=0`, and native stderr
stayed clean:

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --server native \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --prompt-repetitions 24 --max-model-len 256 --max-tokens 2 \
    --gpu-memory-utilization 0.40 --worker-timeout-s 300 \
    --use-layerwise --kv-cache-layout HND \
    --require-clean-native-stderr \
    --output /tmp/lmcache-native-vllm-qwen25-layerwise-hnd-smoke.json
```

A focused larger-model `Qwen/Qwen3-4B` native layerwise HND smoke also passed.
vLLM resolved `HND`; the first process stored five chunks; the reader raised
native retrieves from 0 to 1, cache hits from 0 to 5, and transfer locks from 5
to 10; `transfer_lock_failure_count=0`, `unsupported_count=0`, and native
stderr stayed clean:

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --server native \
    --model Qwen/Qwen3-4B --use-layerwise --kv-cache-layout HND \
    --batch-size 1 --reader-processes 1 --prompt-repetitions 32 \
    --max-model-len 512 --max-tokens 2 --gpu-memory-utilization 0.60 \
    --worker-timeout-s 900 --require-clean-native-stderr \
    --output /tmp/lmcache-native-vllm-qwen3-4b-layerwise-hnd.json
```

A cached `mistralai/Mistral-7B-Instruct-v0.2` native layerwise HND run with real
MP trace lifecycle and live `/kvcache/check` checksum assertions also passed.
vLLM resolved `HND`; the writer stored four chunks; the reader raised native
retrieves from 0 to 1, cache hits from 0 to 4, and transfer locks from 4 to 8;
`transfer_lock_failure_count=0`, `unsupported_count=0`, and native stderr stayed
clean. The metadata trace captured 44 request/response rows, including two
`REGISTER_KV_CACHE` calls with `use_layerwise=true` and `kv_layout=HND`, one
real vLLM `STORE`, one real vLLM `RETRIEVE`, `LOOKUP`,
`QUERY_PREFETCH_STATUS`, `END_SESSION`, and `UNREGISTER_KV_CACHE`. The writer
`STORE` and reader `RETRIEVE` checksums matched exactly for block range `[1,8]`:
`81a2297ade10bd984961a3a6e3cb5c7c`,
`2f27e5fb4961121facd87ea695b7d343`,
`1ab09d76a8b1a45e2a0fbac83d1984ac`, and
`ec61fe52467cf6cf631a26d1a25a9fec`:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --server native \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --use-layerwise --kv-cache-layout HND --batch-size 1 \
    --reader-processes 1 --prompt-repetitions 24 --max-model-len 512 \
    --max-tokens 4 --worker-timeout-s 900 --require-clean-native-stderr \
    --require-mp-trace-lifecycle --require-kvcache-checksum-match \
    --mp-trace-output /tmp/lmcache-native-vllm-mistral7b-layerwise-hnd-byte-check-trace.jsonl \
    --output /tmp/lmcache-native-vllm-mistral7b-layerwise-hnd-byte-check.json
```

A broader layerwise HND run with two prompt variants, one writer, two
sequential reader processes, one warmup round, and two measured steady-state
rounds per process also passed. vLLM resolved `HND`; native retrieves rose from
5 after the writer to 17 after the second reader; cache hits rose from 60 to
204; transfer locks rose from 72 to 216; `transfer_lock_failure_count=0`,
`unsupported_count=0`, and native stderr stayed clean:

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --server native \
    --use-layerwise --kv-cache-layout HND --batch-size 2 \
    --reader-processes 2 --prompt-repetitions 64 \
    --max-model-len 512 --gpu-memory-utilization 0.40 \
    --steady-state-warmup-rounds 1 --steady-state-rounds 2 \
    --require-clean-native-stderr --worker-timeout-s 300 \
    --output /tmp/lmcache-native-vllm-layerwise-hnd-two-readers-steady.json
```

A smaller real-vLLM layerwise HND concurrent-reader run also passed. It used
one writer and two simultaneous reader processes; vLLM resolved `HND`; native
retrieves rose from 0 after the writer to 2 after the concurrent readers; cache
hits rose from 0 to 12; transfer locks rose from 6 to 18; and
`transfer_lock_failure_count=0`, `unsupported_count=0`, and clean native stderr
were preserved:

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --server native \
    --use-layerwise --kv-cache-layout HND --batch-size 1 \
    --reader-processes 2 --concurrent-readers --prompt-repetitions 32 \
    --max-model-len 256 --gpu-memory-utilization 0.40 \
    --steady-state-warmup-rounds 0 --steady-state-rounds 1 \
    --require-clean-native-stderr --worker-timeout-s 300 \
    --output /tmp/lmcache-native-vllm-layerwise-hnd-concurrent-readers.json
```

The same smoke harness can capture metadata-only real MP request traces from
spawned vLLM worker processes. A focused two-reader `facebook/opt-125m`
layerwise HND run with trace lifecycle assertions captured 66 request/response
rows, including three `REGISTER_KV_CACHE` calls with `use_layerwise=true` and
`kv_layout=HND`, one real vLLM `STORE`, two real vLLM `RETRIEVE` calls,
`LOOKUP` and `QUERY_PREFETCH_STATUS` calls, `END_SESSION`, and
`UNREGISTER_KV_CACHE`. This does not persist reusable CUDA IPC handles and is
not a byte-replay trace:

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --server native \
    --model facebook/opt-125m --use-layerwise --kv-cache-layout HND \
    --batch-size 1 --reader-processes 2 --prompt-repetitions 32 \
    --max-model-len 256 --max-tokens 2 \
    --gpu-memory-utilization 0.30 --worker-timeout-s 300 \
    --require-clean-native-stderr \
    --mp-trace-output /tmp/lmcache-native-vllm-layerwise-hnd-two-reader-real-mp-trace.jsonl \
    --require-mp-trace-lifecycle \
    --output /tmp/lmcache-native-vllm-layerwise-hnd-two-reader-real-mp-trace-smoke.json
```

The harness can also byte-check a live real-vLLM transfer without trying to
reuse CUDA IPC handles after worker exit. With
`--require-kvcache-checksum-match`, each worker queries native
`/kvcache/check` before it unregisters its KV cache. A focused
`facebook/opt-125m` layerwise HND run verified matching writer `STORE` and
reader `RETRIEVE` checksums for block range `[1,8]`, with four checksum chunks,
`transfer_lock_failure_count=0`, `unsupported_count=0`, and clean native
stderr:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --server native \
    --model facebook/opt-125m --use-layerwise --kv-cache-layout HND \
    --batch-size 1 --reader-processes 1 --prompt-repetitions 24 \
    --max-model-len 256 --max-tokens 2 --gpu-memory-utilization 0.30 \
    --worker-timeout-s 300 --require-clean-native-stderr \
    --require-mp-trace-lifecycle --require-kvcache-checksum-match \
    --mp-trace-output /tmp/lmcache-native-vllm-layerwise-hnd-byte-check-trace.jsonl \
    --output /tmp/lmcache-native-vllm-layerwise-hnd-byte-check.json
```

A cached `Qwen/Qwen2.5-0.5B-Instruct` native smoke also passed with the same
writer/reader structure. The first process stored three chunks, the reader
raised native retrieves from 0 to 1, cache hits from 0 to 3, and
`unsupported_count` stayed 0:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --batch-size 1 --prompt-repetitions 24 --max-model-len 512 \
    --worker-timeout-s 360 \
    --output /tmp/lmcache-native-vllm-qwen-0.5b-smoke.json
```

This still covers only focused prompt shapes and a small model set.

The same harness can produce a startup-inclusive Python-vs-native comparison:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with vllm --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --compare-python \
    --output /tmp/lmcache-vllm-native-vs-python.json
```

The compare report records both startup-inclusive elapsed time and the inner
`llm.generate()` elapsed time. The latest run reported startup-inclusive
second-generation elapsed time of 17.356s for Python MP and 17.026s for native
MP (`native_over_python=0.981`). Generate-only second-generation elapsed time
was 0.182s for Python MP and 0.431s for native MP
(`native_over_python=2.367`).

The harness can also run warmup-controlled steady-state generate rounds inside
one loaded LLM process:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with vllm --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --compare-python \
    --steady-state-warmup-rounds 1 --steady-state-rounds 2 \
    --batch-size 1 --prompt-repetitions 24 --max-model-len 256 \
    --worker-timeout-s 240 \
    --output /tmp/lmcache-vllm-native-vs-python-steady-ttft.json
```

The current report schema includes mean, p50, p95, p99, and raw measured
values for generate latency, output-token throughput, and TTFT. The current
steady-state compare reported second-reader throughput of 133.692 output
tokens/s for Python MP and 61.549 output tokens/s for native MP
(`native_over_python=0.460`), second-reader mean TTFT of 0.028s for Python MP
and 0.087s for native MP, with native `unsupported_count=0`. This uses vLLM
request stats for TTFT and is still a small one-model benchmark, not full
production TTFT/throughput parity.

`--compare-python` now also captures real-vLLM worker traces for both Python
and native MP runs, even when `--mp-trace-output` is not requested. The report
adds `mp_request_latency_ms`, with client-observed mean, p50, p95, p99, and raw
latency samples for actual MP `STORE`, `LOOKUP`, `RETRIEVE`, and any other
request types emitted by the vLLM workers. The focused benchmark-summary
regression passed for the trace-latency fields and the existing controller
latency fields:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_vllm_smoke_round_summary_reports_percentiles \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_controller_latency_summary_reports_percentiles_and_resources
```

Result: `2 passed, 53 warnings in 2.91s`.

The vLLM smoke/compare report now records MP server process resource snapshots
from `/proc` before the writer run and after the final reader run, including
RSS, peak RSS, user/system/total CPU seconds, and thread count. A focused
native resource smoke with `facebook/opt-125m` reported native retrieves rising
from 0 to 1 after the reader, cache hits from 0 to 4, `unsupported_count=0`,
`server_resources_delta.total_cpu_s_delta=2.88`,
`server_resources_delta.rss_bytes_delta=113246208`, and
`server_resources_delta.rss_peak_bytes=161480704`.

A cached `Qwen/Qwen2.5-0.5B-Instruct` steady-state compare also ran:

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --compare-python --model Qwen/Qwen2.5-0.5B-Instruct \
    --steady-state-warmup-rounds 1 --steady-state-rounds 2 \
    --batch-size 1 --prompt-repetitions 24 --max-model-len 512 \
    --worker-timeout-s 600 \
    --output /tmp/lmcache-vllm-qwen2.5-0.5b-native-vs-python-steady.json
```

That Qwen2.5 compare reported second-reader steady-state throughput of
98.195 output tokens/s for Python MP and 35.605 output tokens/s for native MP
(`native_over_python=0.363`), second-reader mean TTFT of 0.035s for Python MP
and 0.141s for native MP, with native retrieves increasing to 5 and native
`unsupported_count=0`.

A cached `mistralai/Mistral-7B-Instruct-v0.2` steady-state compare also ran:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --compare-python --model mistralai/Mistral-7B-Instruct-v0.2 \
    --batch-size 1 --reader-processes 1 --prompt-repetitions 24 \
    --max-model-len 512 --max-tokens 4 \
    --steady-state-warmup-rounds 1 --steady-state-rounds 2 \
    --worker-timeout-s 900 --require-clean-native-stderr \
    --output /tmp/lmcache-native-vllm-mistral7b-steady-compare.json
```

That Mistral-7B compare reported second-reader steady-state throughput of
66.040 output tokens/s for Python MP and 16.845 output tokens/s for native MP
(`native_over_python=0.255`), second-reader mean TTFT of 0.037s for Python MP
and 0.195s for native MP, native retrieves increasing from 2 after the writer
to 5 after the reader, `transfer_lock_failure_count=0`, `unsupported_count=0`,
`clean_native_stderr=true`, native `rss_peak_bytes=179023872`, and Python MP
`rss_peak_bytes=1093517312`.

The native smoke also supports longer prompt batches:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with vllm --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --batch-size 2 --prompt-repetitions 40 --max-model-len 512 \
    --output /tmp/lmcache-native-vllm-batch-smoke.json
```

The current batched run used two prompt variants with a repeated shared base
prompt. Native retrieve count increased from 1 after the first process to 3
after the second process, cache hits increased from 7 to 21, and
`unsupported_count` stayed 0. This still covers one model and one layout.

The native smoke also supports repeated reader processes:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with vllm --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --reader-processes 2 --batch-size 2 --prompt-repetitions 48 \
    --max-model-len 512 --worker-timeout-s 240 \
    --output /tmp/lmcache-native-vllm-two-reader-smoke.json
```

The current two-reader run used two prompt variants. Native retrieve count
increased from 1 after the writer process to 5 after the second reader, cache
hits increased from 9 to 45, and `unsupported_count` stayed 0. This still
covers one model and one layout.

The native smoke can also run reader processes concurrently:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with vllm --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --reader-processes 2 --concurrent-readers \
    --steady-state-warmup-rounds 0 --steady-state-rounds 1 \
    --batch-size 1 --prompt-repetitions 24 --max-model-len 256 \
    --worker-timeout-s 300 \
    --output /tmp/lmcache-native-vllm-concurrent-readers.json
```

The current concurrent-reader run increased native retrieve count from 0 after
the writer process to 2 after both readers, cache hits from 0 to 8, and
`unsupported_count` stayed 0. This still covers one model and one layout.

A longer concurrent HND run also passed with two prompt variants, two
simultaneous reader processes, `prompt_repetitions=64`, `max_model_len=512`,
and clean native server stderr:

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with vllm --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --server native --model facebook/opt-125m --kv-cache-layout HND \
    --reader-processes 2 --concurrent-readers --batch-size 2 \
    --prompt-repetitions 64 --max-model-len 512 --max-tokens 8 \
    --gpu-memory-utilization 0.30 --worker-timeout-s 240 \
    --require-clean-native-stderr \
    --output /tmp/lmcache-vllm-native-hnd-concurrent-batch2.json
```

The HND concurrent run increased native retrieve count from 1 after the writer
process to 5 after the readers, cache hits from 12 to 60, kept
`transfer_lock_failure_count=0`, kept `unsupported_count=0`, and reported
`clean_native_stderr=true`.

The harness can also require a clean native server stderr stream. This fails
the run if the native server logs anything other than its startup line:

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --batch-size 1 --prompt-repetitions 24 --max-model-len 256 \
    --worker-timeout-s 300 --require-clean-native-stderr \
    --output /tmp/lmcache-native-vllm-clean-stderr-smoke.json
```

The current clean-stderr run increased native retrieve count from 0 after the
writer to 1 after the reader, kept `unsupported_count=0`, reported
`clean_native_stderr=true`, and reported no unexpected native stderr lines.

A cached `mistralai/Mistral-7B-Instruct-v0.2` native smoke also passed with one
writer and one reader process. The writer stored four chunks with four cache
misses; the reader raised native retrieves from 0 to 1, cache hits from 0 to 4,
transfer locks from 4 to 8, kept `transfer_lock_failure_count=0`, kept
`unsupported_count=0`, reported `clean_native_stderr=true`, and reported native
server `rss_peak_bytes=177823744`:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --server native --model mistralai/Mistral-7B-Instruct-v0.2 \
    --batch-size 1 --reader-processes 1 --prompt-repetitions 24 \
    --max-model-len 512 --max-tokens 4 --worker-timeout-s 600 \
    --require-clean-native-stderr \
    --output /tmp/lmcache-native-vllm-mistral7b-smoke.json
```

A cached larger-model native smoke also passed with `facebook/opt-1.3b`, two
prompt variants, and two concurrent reader processes:

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --model facebook/opt-1.3b --batch-size 2 \
    --reader-processes 2 --concurrent-readers --prompt-repetitions 64 \
    --max-model-len 512 --worker-timeout-s 600 \
    --output /tmp/lmcache-native-vllm-opt-1.3b-concurrent.json
```

The opt-1.3b run increased native retrieve count from 1 after the writer to 5
after the concurrent readers, cache hits from 12 to 60, kept
`transfer_lock_failure_count=0` and `unsupported_count=0`, and the native
server stderr tail contained only the startup line.

A cached `Qwen/Qwen3-4B` native run also passed with two prompt variants and
two concurrent reader processes:

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --model Qwen/Qwen3-4B --batch-size 2 \
    --reader-processes 2 --concurrent-readers --prompt-repetitions 48 \
    --max-model-len 512 --gpu-memory-utilization 0.60 \
    --worker-timeout-s 900 --require-clean-native-stderr \
    --output /tmp/lmcache-native-vllm-qwen3-4b-concurrent.json
```

The Qwen3-4B run increased native retrieve count from 1 after the writer to 5
after the concurrent readers, cache hits from 7 to 35, kept
`transfer_lock_failure_count=0` and `unsupported_count=0`, and reported
`clean_native_stderr=true`.

A cached `Qwen/Qwen3-4B` HND-layout native run also passed:

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --model Qwen/Qwen3-4B --kv-cache-layout HND \
    --batch-size 1 --reader-processes 1 --prompt-repetitions 40 \
    --max-model-len 512 --gpu-memory-utilization 0.60 \
    --worker-timeout-s 900 --require-clean-native-stderr \
    --output /tmp/lmcache-native-vllm-qwen3-4b-hnd.json
```

The Qwen3-4B HND run resolved vLLM's layout to `HND`, increased native
retrieve count from 0 after the writer to 1 after the reader, cache hits from
0 to 6, kept `transfer_lock_failure_count=0` and `unsupported_count=0`, and
reported `clean_native_stderr=true`.

For native CUDA transfer experiments, launch the server with `--native-cuda`
or `LMCACHE_MP_NATIVE_CUDA=1`, then pass `"lmcache.mp.raw_cuda_ipc": true` in
`kv_connector_extra_config`, or set `LMCACHE_MP_RAW_CUDA_IPC=1` /
`LMCACHE_MP_NATIVE_RAW_CUDA_IPC=1` for the worker adapter process.

## Gated Or Unsupported

These native features are intentionally unsupported and fail loudly or return
safe misses/failures. The focused 14-test unsupported-mode cluster covers
startup validation, native binary argument rejection, unsupported layout
metadata, malformed transfer payloads, malformed protocol payloads, oversized
ZMQ payloads, out-of-range request types, and invalid cache-blend payloads:

- Legacy pickle-only CUDA IPC KV registration without the native-friendly
  msgpack metadata envelope.
- Unsupported layer-group descriptors, TRT-LLM reshape hints on non-TRT
  engines, or inconsistent compression hints at `REGISTER_KV_CACHE`; these
  configurations are rejected instead of being accepted and failing later during
  transfer. Layerwise hints are accepted only when the wrapper metadata uses an
  already supported per-layer CUDA layout.
- Native `STORE` and `RETRIEVE` payload movement in the default no-CUDA build.
- Full native `STORE` and `RETRIEVE` parity for real vLLM layerwise lifecycles
  or remaining engine-specific layouts beyond the synthetic CUDA golden matrix.
- Full vLLM parity beyond the focused `facebook/opt-125m`,
  `Qwen/Qwen2.5-0.5B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.2`,
  `facebook/opt-1.3b`, and `Qwen/Qwen3-4B` smoke coverage for the PyTorch and
  raw CUDA IPC connector modes.
- Non-filesystem L2 adapters. `type="nixl"` is rejected with an explicit
  native-not-implemented error.
- Blend/cache-blend storage and matcher behavior beyond payload validation.
- IsolatedLRU and `noop` eviction policies. The Python native launcher rejects
  these before exec, and the direct native binary also fails loudly if invoked
  with a non-`LRU` eviction policy.
- Full Python OTel/EventBus metrics parity beyond the native `/metrics`
  counters and gauges.
- Python EventBus parity for block-allocation events.
- Full production benchmark sweeps beyond the current controller latency,
  real-vLLM request-latency, three-model steady-state throughput/TTFT, and
  focused larger-model smoke reports.

## Completion Gap Against GOAL.md

No remaining GOAL.md acceptance gap is currently identified. Broader production
benchmark sweeps remain a useful follow-up, but the native-vs-Python benchmark
artifact now covers controller latency, real-vLLM `STORE`/`LOOKUP`/`RETRIEVE`
request latency, vLLM TTFT/throughput, cache-hit behavior, resource deltas,
concurrency, L2 metadata lookup, and larger-model smokes.
