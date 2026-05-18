# GOAL.md Completion Audit

Objective: build a full native C++ LMCache MP server that replaces the Python
MP server for the runtime data path while preserving existing vLLM connector,
CLI/config, protocol, L1/L2, HTTP, observability, correctness replay, and
benchmark behavior.

Status: complete.

## Acceptance Criteria

| # | Requirement | Current Evidence | Status |
|---|-------------|------------------|--------|
| 1 | Native C++ MP server builds | `cmake -S . -B /tmp/lmcache-native-root-build -DLMCACHE_BUILD_NATIVE_MP=ON` and `cmake --build ... --target lmcache-mp-server-native` pass; separate `-DLMCACHE_ENABLE_CUDA=ON` and `-DLMCACHE_ENABLE_TSAN=ON` builds also compile and link; `cmake --install ... --prefix /tmp/lmcache-native-install` installs an executable native binary | Pass |
| 2 | Existing Python package still installs | Existing editable `uv run --python 3.12 ...` test environment imports `lmcache.c_ops`; an isolated editable install/import check passed with `NO_CUDA_EXT=1` and imported both `lmcache` and `native_launcher`; CUDA extension rebuild with the local CUDA 13.0 Python-toolkit packages passed via `python setup.py build_ext --inplace`; an isolated editable import check with the local CUDA 13 toolkit environment imported `lmcache`, `lmcache.c_ops`, and `native_launcher`; opt-in `NO_CUDA_EXT=1 LMCACHE_BUILD_NATIVE_MP=1 python setup.py build` packages `lmcache/bin/lmcache-mp-server-native`, and the CUDA variant with `LMCACHE_NATIVE_MP_ENABLE_CUDA=1` packages `lmcache/bin/lmcache-mp-server-native-cuda`; the main release-artifact workflow now smoke-tests a no-CUDA wheel built with the opt-in native package flag and verifies packaged binary execution from the wheel contents; a local packaged no-CUDA binary protocol smoke verifies valid `STORE` and `RETRIEVE` requests return safe `(b'', False)` results with `cuda_transfer_enabled=false`; the main, cu12.9, and nightly CUDA wheel builds now pass `LMCACHE_BUILD_NATIVE_MP=1` and `LMCACHE_NATIVE_MP_ENABLE_CUDA=1` through `cibuildwheel`, then test that the wheel imports `lmcache.c_ops`, contains executable `lmcache/bin/lmcache-mp-server-native-cuda`, and runs `--help`; a local CUDA-native `bdist_wheel` smoke verifies the wheel contains executable `lmcache/bin/lmcache-mp-server-native-cuda` and that the packaged binary responds to `--help` | Pass |
| 3 | `lmcache server` launches native through flag/env | `--native` and `LMCACHE_MP_NATIVE=1` now select the CUDA-enabled native binary by default for the real vLLM KV data path; focused CLI tests verify `LMCACHE_MP_NATIVE`, `LMCACHE_MP_NATIVE_CUDA`, and `LMCACHE_MP_NATIVE_NO_CUDA` truthy values `1`, `true`, `yes`, and `on` route through `run_native_server`, falsey values `""`, `0`, `false`, `no`, and `off` keep the Python server path, and `--python` overrides the native env-var path to launch the Python HTTP server path; `--native-cuda` and `LMCACHE_MP_NATIVE_CUDA=1` remain explicit CUDA synonyms; `--native-no-cuda` and `LMCACHE_MP_NATIVE_NO_CUDA=1` preserve the controller-only no-CUDA native build; `--python`, `--native-disk-path`, startup `--log-level`, `LMCACHE_MP_NATIVE_BINARY`, and `LMCACHE_MP_NATIVE_CUDA_BINARY` launcher paths are wired; package-local `lmcache/bin/lmcache-mp-server-native-cuda` and `lmcache/bin/lmcache-mp-server-native` lookup is wired and covered by a packaged-binary launcher regression plus a direct import from the built package tree; argv tests cover default CUDA, explicit CUDA, explicit no-CUDA, startup log-level propagation, supported `LMCACHE_CONFIG_FILE` seeding for `chunk_size`, `max_local_cpu_size`, `cache_policy`, and `local_disk` filesystem L2 adapters, config-file precedence over conflicting engine env vars, and conflicting CUDA/no-CUDA selection paths; the direct native binary also supports flat `--config-file`/`LMCACHE_CONFIG_FILE` startup seeding for those keys; and the native launcher now rejects unsupported Python-only options such as non-`blake3` hashing, non-`LRU` eviction policies, blend engine mode, runtime plugins, separate CPU/GPU worker-pool sizing, non-default L1 allocator/TTL knobs, non-default eviction-watermark/ratio knobs, non-default L2 store/prefetch policy knobs, unsupported config-file modes, and non-default EventBus/OTel/tracing/lookup-hash/trace-recording knobs instead of silently ignoring them | Pass |
| 4 | vLLM connector connects without changes | Native binary supports controller/key paths, decodes native-friendly `KVCache` wrapper metadata during registration, imports the default PyTorch `CudaIPCWrapper` handle shape, and passed real vLLM 0.21.0 `facebook/opt-125m` two-process smoke with the default connector path: first process stored four chunks, second process hit and retrieved four chunks, `unsupported_count=0`. A default PyTorch CUDA IPC HND smoke with `--kv-cache-layout HND` resolved vLLM's layout to `HND`, raised native retrieves from 0 after the writer to 1 after the reader, raised cache hits from 0 to 4, and kept `unsupported_count=0`. A cached `Qwen/Qwen2.5-0.5B-Instruct` smoke passed with first-process store of three chunks, reader native retrieves rising from 0 to 1, cache hits from 0 to 3, and `unsupported_count=0`. A longer batched native smoke with two prompt variants and `prompt_repetitions=40` also passed: second-process native retrieves increased from 1 to 3, cache hits from 7 to 21, and `unsupported_count=0`. A follow-up native run with `--reader-processes 2`, `--batch-size 2`, `prompt_repetitions=48`, and `max_model_len=512` passed: retrieves increased from 1 after the writer to 5 after the second reader, cache hits increased from 9 to 45, and `unsupported_count=0`. A concurrent-reader native run with `--reader-processes 2 --concurrent-readers` passed: retrieves increased from 0 after the writer to 2 after both simultaneous readers, cache hits from 0 to 8, and `unsupported_count=0`. A longer concurrent HND run with two prompt variants, `--reader-processes 2 --concurrent-readers`, `prompt_repetitions=64`, `max_model_len=512`, and clean native stderr passed: retrieves increased from 1 after the writer to 5 after the readers, cache hits from 12 to 60, `transfer_lock_failure_count=0`, and `unsupported_count=0`. A cached `facebook/opt-1.3b` native run with two prompt variants and two concurrent reader processes passed: retrieves increased from 1 after the writer to 5 after the readers, cache hits from 12 to 60, `transfer_lock_failure_count=0`, `unsupported_count=0`, and the native server stderr tail only contained the startup line. A cached `Qwen/Qwen3-4B` native run with two prompt variants and two concurrent reader processes passed: retrieves increased from 1 after the writer to 5 after the readers, cache hits from 7 to 35, `transfer_lock_failure_count=0`, `unsupported_count=0`, and `clean_native_stderr=true`. A cached `Qwen/Qwen3-4B` HND-layout native run resolved vLLM's layout to `HND`, increased native retrieves from 0 after the writer to 1 after the reader, cache hits from 0 to 6, `transfer_lock_failure_count=0`, `unsupported_count=0`, and `clean_native_stderr=true`. The harness can require clean native stderr; the current clean-stderr run reported `clean_native_stderr=true`, no unexpected native stderr lines, native retrieve count increasing from 0 to 1, and `unsupported_count=0`. The bare documented `lmcache.mp.host=127.0.0.1` form is normalized to `tcp://127.0.0.1`, native raw CUDA IPC registration is explicitly opt-in through `lmcache.mp.raw_cuda_ipc` or worker-side env vars, real vLLM MP registration propagates `use_layerwise=True` in layout hints when LMCache `use_layerwise` is enabled, focused adapter regressions verify MLA rank normalization before cache-key requests are built, and the native binary accepts supported layerwise CUDA IPC hints | Pass |
| 5 | Store/lookup/retrieve match Python on golden traces | Protocol/schema constants, native key decoding, ObjectKey generation, registration metadata, validation, filesystem-L2 metadata, force clear, report-block-allocation, cache-blend schema checks, and CUDA-gated PyTorch/raw CUDA IPC `STORE`/`RETRIEVE` paths are implemented and covered by focused tests. Python-captured CUDA golden traces byte-replay default PyTorch IPC, raw CUDA IPC, NHD/HND, compressed/mixed-compression, compact 4D, heterogeneous, larger, multi-chunk, wide, cross-layer, TRT-LLM 4D, MLA-shaped, layerwise-hinted, and repeated all-layout lifecycle cases against native CUDA, including lock-status checks after byte-checked `RETRIEVE`. Real vLLM smokes cover default, HND, raw CUDA IPC, multiple models, layerwise HND lifecycles, live `/kvcache/check` writer-vs-reader checksum matches for real layerwise HND `facebook/opt-125m` and `mistralai/Mistral-7B-Instruct-v0.2` transfers, and replay-validated saved `vllm_kvcache_checksum_match` rows for reusable real-vLLM checksum evidence, including mismatch rejection. The focused connector/trace replay cluster passed for the current checkout | Pass |
| 6 | Hash/cache key generation matches Python byte-for-byte | Native C++ default `blake3` rolling chunk hashes, KV-rank expansion, ObjectKey string serialization, `IPCCacheEngineKey` msgpack-map decoding, native-friendly `CudaIPCWrapper` metadata/handle decoding, CUDA IPC handle-byte preservation, and registered KV block metadata inference match Python-compatible test fixtures; focused tests compare native and Python `TokenHasher` outputs for default and nontrivial token values, multiple chunk windows/chunk sizes, all-rank and worker-specific KV-rank expansion, empty and non-empty cache salts, ObjectKey strings generated from real Python chunk hashes for multiple model names, and direct native `IPCCacheEngineKey` expansion against Python for all-rank, worker-specific, and empty start/end ranges; a lightweight MLA regression verifies TP-excluded KV world size/rank normalization before MP cache keys are built | Pass |
| 7 | LRU eviction behavior matches expected semantics | C++ tiered cache spills DRAM LRU entries to disk, promotes on read when safe, verifies resident reads refresh the touched entry before the next spill, verifies disk promotion refreshes the promoted entry before spilling the next-oldest resident, serves locked disk-tier reads without promotion when other active locks already fill the L1 budget, deterministically replaces duplicate stores for spilled entries, skips locked/pinned resident entries during spill, remove, and safe direct-cache clear, supports force clear for Python-compatible MP `CLEAR`, rolls back a new store or overwrite if the required disk spill fails, tracks cumulative LRU spill evictions in cache stats and `/status`, and takes temporary transfer locks around native CUDA L2 writes/retrieves. The direct native binary rejects unsupported eviction policies instead of silently falling back, and the focused 12-test LRU/unsupported-policy cluster passed | Pass |
| 8 | Locked/pinned entries are never evicted | C++ tiered cache lock/pin APIs protect resident entries from LRU spill, explicit remove, and safe direct-cache clear in focused tests; locked disk-tier reads avoid promotion when other active locks already fill the L1 budget; native protocol `CLEAR` and HTTP `clear-cache` match Python MP force-clear semantics and remove active lookup locks; native force-clear epochs suppress stale lock-release noise only when a force clear invalidates the locked entry; native cache status reports locked-entry count, total lock/refcount depth, and locked bytes; native server records lookup locks, releases the `FREE_LOOKUP_LOCKS` key range for the request id, releases lookup locks on `END_SESSION`, clears stale locks when a lookup id is reused, and takes temporary transfer locks around CUDA `STORE` L2 writes and `RETRIEVE` GPU copies. CUDA byte tests assert force `CLEAR` removes lookup-locked chunks, a two-chunk lookup locks and frees both chunks, RETRIEVE while a lookup lock is still held only uses a temporary transfer lock and leaves the lookup lock in place until `FREE_LOOKUP_LOCKS`, subset `FREE_LOOKUP_LOCKS` can release one of two locked chunks before force `CLEAR` removes the remainder, two independent lookup request ids can hold the same chunk with `locked_entries=1` and `lock_count=2` before per-owner cleanup decrements the refcount, reusing a lookup request id replaces the previous lock set without accumulating lock depth, `END_SESSION` releases a lookup lock and lets `CLEAR` remove the chunk, partial-missing RETRIEVE releases the already-acquired transfer lock, `UNREGISTER_KV_CACHE` removes the registered context and later STORE/RETRIEVE requests for that instance fail without leaking locks, and `/status` exposes transfer-lock counters. The focused 13-test lock/pin lifecycle cluster passed | Pass |
| 9 | HTTP health/status/clear-cache endpoints work | Native binary tests cover `GET /`, `GET /healthcheck`, `GET /status`, `GET /conf`, `GET /version`, `GET /lmc_version`, `GET /commit_id`, `GET /env`, `GET /loglevel`, `GET /threads`, `GET /periodic-threads`, `GET /periodic-threads/{thread_name}`, `GET /periodic-threads-health`, `GET /quota`, `GET /quota/{cache_salt}`, `PUT /quota/{cache_salt}`, `DELETE /quota/{cache_salt}`, `GET /kvcache/check`, `POST /clear-cache`, `GET /metrics`, and `POST /metrics/reset`; root, healthcheck, clear-cache, version, env, loglevel, threads, periodic-thread, quota, and kvcache-check validation/error endpoints use the same success or error response shapes as the Python HTTP API. Native CUDA `/kvcache/check` computes Python-compatible aggregate and `layerwise=true` MD5 checksum responses for supported block-native NHD/HND/MLA layouts. Native quota endpoints track metadata and filesystem-L2 usage, and native `/metrics` exports the native counters/gauges already in `/status`, including derived `cache_hit_rate`; full `IsolatedLRU` quota enforcement and Python OTel/EventBus parity remain documented unsupported surfaces. The focused 5-test HTTP endpoint cluster passed | Pass |
| 10 | Memory accounting is byte-accurate within documented overhead | Native `/status` exposes DRAM byte counts, disk-tier byte counts, locked/pinned entry counts, total lock/refcount depth, locked bytes, cumulative eviction count, partial-hit count, L1 hit count, L2 hit count, L2 miss count, and derived `cache_hit_rate`; focused tests assert exact DRAM/disk byte counts through spill and promotion, odd-sized spill/promote/replace cases without alignment rounding, duplicate replacement of a spilled entry without stale disk bytes, failed new-store and overwrite rollback without a leaked entry or inflated byte count, nested lock, locked-byte, locked-disk-read, clear, and force-clear accounting, and filesystem-L2 lookup tests assert one-hit/one-miss partial lookup accounting. The focused 13-test memory-accounting cluster passed | Pass |
| 11 | Concurrent tests pass under stress | Native worker pool exists as one worker pool; focused binary test sends a malformed raw ZMQ frame and 80 concurrent PING/NOOP requests across 4 Python clients, now also asserting the short malformed envelope is counted in `invalid_payload_count`; current-source no-TSAN and TSAN-instrumented malformed-frame PING/NOOP stress runs cover 8 clients and 800 handled requests, a longer TSAN run covers 8 clients and 4,000 handled requests with no ThreadSanitizer report and `invalid_payload_count=1`, `request_count=4000`, and `active_client_count=8`, a current-source 60-second duration soak covers 8 clients and 35,775 handled requests with `invalid_payload_count=1`, `request_count=35775`, and `active_client_count=8`, and a current-source two-hour duration soak covers 8 clients and 4,240,701 handled requests with matching request and latency counters; out-of-range numeric request types are rejected before they can truncate into a valid `uint8_t` request type; malformed raw DEALER `STORE`/`RETRIEVE` payloads return native false responses, keep PING healthy, and increment `invalid_payload_count=4`; deterministic malformed protocol fuzzing sends malformed envelopes, invalid request-type frames, out-of-range request types, and malformed typed payloads, then verifies post-fuzz PING/NOOP liveness and invalid-payload accounting (`64` cases, `invalid_payload_delta=64`, `request_count=63`, `unsupported_count=61`; longer `1024`-case run, `invalid_payload_delta=1024`, `request_count=1023`, `unsupported_count=1021`); a focused CUDA KV data-path test runs eight keyed PyTorch CUDA IPC STORE/LOOKUP/CLEAR/RETRIEVE iterations in one native server process with distinct cache salts and `unsupported_count=0`; a gated longer CUDA stress test runs 32 keyed PyTorch CUDA IPC STORE/LOOKUP/CLEAR/RETRIEVE iterations with distinct cache salts and `unsupported_count=0`; native CUDA IPC transfer calls are serialized across worker threads before CUDA runtime handle open/copy/close calls; a CUDA+TSAN build passes a two-client concurrent PyTorch CUDA IPC STORE/LOOKUP/RETRIEVE pytest, the older single-client PyTorch CUDA data-path pytest, and the 28-row all-layout CUDA lifecycle trace replay with no ThreadSanitizer report; a four-client CUDA KV data-path test runs concurrent PyTorch CUDA IPC STORE/LOOKUP/RETRIEVE round trips against one native server process with `--max-workers 4`, distinct instance ids, distinct cache salts, and `unsupported_count=0`; a gated eight-client CUDA concurrency stress test runs four STORE/LOOKUP/RETRIEVE rounds per client, asserting 32 stores, 32 lookups, 32 retrieves, 64 transfer locks, zero transfer-lock failures, zero unsupported requests, and no leaked locks; the filesystem-L2 restart test covers both graceful first-process termination and SIGKILL before a second native process retrieves the stored bytes; a focused shutdown test sends SIGTERM to a ready native binary and requires clean zero exit; `/status` and `/metrics` expose observed valid ZMQ client count through `active_client_count` / `observed_client_count`, worker count, active worker count, bounded worker queue depth, maximum queue depth, response queue depth, queue-full backpressure count, request latency count, total/max latency, and request-latency histogram buckets; a focused backpressure regression sets `--max-queued-tasks 0`, verifies a safe nil response and healthy HTTP frontend, and checks `queue_full_count=1`; separate CPU/GPU worker-pool sizes fail loudly instead of being ignored. The focused 6-test current concurrency/error cluster passed | Pass |
| 12 | Unsupported optional features fail loudly | Native `--l2-adapter` accepts filesystem L2 configs, filesystem L2 metadata is consulted during lookup, filesystem L2 clear is wired to native `CLEAR`, raw-IPC STORE/RETRIEVE moves bytes through filesystem L2, and unsupported adapter types fail loudly, including an explicit NIXL-not-implemented error; unsupported eviction policies fail loudly; the native Python launcher rejects unsupported hash algorithms, unsupported eviction policies, blend engine mode, runtime plugins, separate CPU/GPU worker-pool sizes, non-default L1 allocator/TTL knobs, non-default eviction-watermark/ratio knobs, non-default L2 store/prefetch policy knobs, and non-default EventBus/OTel/tracing/lookup-hash/trace-recording knobs before exec; the direct native binary also rejects separate worker-pool sizing and non-`LRU` eviction policies; non-dividing compressed block metadata, inconsistent compression hints, unsupported layer-group descriptors, TRT-LLM reshape hints on non-TRT engines, and native checksum requests for unsupported KV formats fail loudly; cache-blend payload schemas are validated and malformed blend payloads are counted in `/status`; malformed core KV transfer payloads return explicit false transfer responses and are counted in `/status`; malformed protocol fuzzing verifies invalid-payload accounting and post-error liveness across bad envelopes, bad request-type frames, out-of-range request types, and malformed typed payloads. The focused 14-test unsupported-mode cluster passed | Pass |
| 13 | Benchmark report compares native vs Python | Controller benchmark exists for `PING`, `NOOP`, missing-key `LOOKUP`, and filesystem-L2 partial-hit `LOOKUP` requests; its report now includes Python/native mean, p50, p95, p99, raw latency samples, concurrent-client `requests_per_s`, and MP server `/proc` resource deltas for CPU, RSS, peak RSS, and thread count. A real vLLM smoke harness now runs two-process smokes against one Python or native CUDA MP server, verifies native second-process retrieval for `facebook/opt-125m` default PyTorch IPC, `facebook/opt-125m` default PyTorch IPC with `VLLM_KV_CACHE_LAYOUT=HND`, longer concurrent `facebook/opt-125m` HND reuse with two prompt variants and clean native stderr, `facebook/opt-125m` raw CUDA IPC, `Qwen/Qwen2.5-0.5B-Instruct` default PyTorch IPC, cached `facebook/opt-1.3b` default PyTorch IPC with two concurrent readers, cached `Qwen/Qwen3-4B` default PyTorch IPC with two concurrent readers, and cached `Qwen/Qwen3-4B` HND-layout PyTorch IPC, and can run `--compare-python`; native run summaries now include both raw hit/miss counters and a derived `cache_hit_rate`. The original compare report separates model initialization from `llm.generate()`: startup-inclusive second generation was Python 17.356s and native 17.026s (`native_over_python=0.981`), while generate-only second generation was Python 0.182s and native 0.431s (`native_over_python=2.367`). The harness now also supports warmup-controlled steady-state rounds inside one loaded LLM process with vLLM request-stat TTFT enabled; reports mean, p50, p95, p99, and raw values for generate latency, output-token throughput, and TTFT; records MP server `/proc` resource snapshots (`rss_bytes`, `rss_peak_bytes`, user/system/total CPU seconds, and thread count) before the writer and after the final reader; and auto-captures Python/native real-vLLM worker traces for `--compare-python` reports so `mp_request_latency_ms` includes client-observed mean, p50, p95, p99, and raw samples for actual MP `STORE`, `LOOKUP`, `RETRIEVE`, and other request types observed in the trace. A focused native resource smoke reported second-reader native retrieves rising from 0 to 1, cache hits from 0 to 4, `unsupported_count=0`, `server_resources_delta.total_cpu_s_delta=2.88`, `rss_bytes_delta=113246208`, and `rss_peak_bytes=161480704`. A steady-state compare with one warmup and two measured rounds reported second-reader output throughput of Python 133.692 output tokens/s and native 61.549 output tokens/s (`native_over_python=0.460`), second-reader mean TTFT of Python 0.028s and native 0.087s, and native `unsupported_count=0`. A cached `Qwen/Qwen2.5-0.5B-Instruct` steady-state compare reported second-reader output throughput of Python 98.195 output tokens/s and native 35.605 output tokens/s (`native_over_python=0.363`), second-reader mean TTFT of Python 0.035s and native 0.141s, native retrieves increasing to 5, and native `unsupported_count=0`. A cached `mistralai/Mistral-7B-Instruct-v0.2` steady-state compare reported second-reader output throughput of Python 66.040 output tokens/s and native 16.845 output tokens/s (`native_over_python=0.255`), second-reader mean TTFT of Python 0.037s and native 0.195s, native retrieves increasing from 2 after the writer to 5 after the reader, `transfer_lock_failure_count=0`, `unsupported_count=0`, and clean native stderr. The focused benchmark-summary regression passed for the trace latency and controller latency report fields | Pass |
| 14 | No core TODO stubs remain | `rg -n "TODO|FIXME|stub" LMCache-mp-cpp/src LMCache-mp-cpp/include` finds no native core TODO/stub markers. Native `STORE`/`RETRIEVE` have a CUDA-gated PyTorch/raw CUDA IPC transfer implementation for basic homogeneous and heterogeneous per-layer vLLM NHD/HND, layerwise-hinted heterogeneous per-layer NHD, compressed NHD, mixed-compression NHD, compact 4D NHD, larger NHD, cross-layer NHD/HND, TRT-LLM 4D, and MLA layouts, and `--native` selects that CUDA-enabled build by default. Explicitly requested no-CUDA builds remain documented controller-only/safe-fail builds, and unsupported optional features fail loudly. The focused 8-test representative CUDA core matrix passed | Pass |
| 15 | Documentation explains build/run/test/benchmark/fallback | `README.md` documents native C++ and packaged builds, CUDA/TSAN build modes, `lmcache server --native`/`--native-cuda`/`--native-no-cuda`, config-file/env startup, unsupported modes, Python fallback, current coverage, and remaining limitations; `native_mp_status.md` documents build, run, fallback, focused and gated test commands, benchmark commands/results, raw CUDA IPC setup, and gated/unsupported behavior; `benchmarks/mp_native_vs_python/README.md` documents the controller and vLLM benchmark helpers, output-path behavior, report fields, checksum trace replay evidence, real-vLLM request-latency fields, and production benchmark follow-up scope | Pass |

## Current Evidence Notes

- A 5-iteration controller benchmark smoke wrote
  `/tmp/lmcache-native-controller-bench.json` with Python mean/p50/p95 latency
  `2.230`/`2.232`/`2.351` ms and native mean/p50/p95 latency
  `3.101`/`3.111`/`3.368` ms. The same report included raw samples and
  `/proc` resource deltas; Python peak RSS was `842420224` bytes, and native
  peak RSS was `48234496` bytes.
- A 5-iteration missing-key LOOKUP controller benchmark smoke wrote
  `/tmp/lmcache-native-controller-lookup-bench.json` with Python mean/p50/p95
  latency `2.665`/`2.568`/`3.385` ms and native mean/p50/p95 latency
  `2.971`/`3.123`/`3.206` ms. The same report included raw samples and
  `/proc` resource deltas; Python peak RSS was `843055104` bytes, and native
  peak RSS was `48234496` bytes.
- A 3-iteration NOOP controller benchmark smoke wrote
  `/tmp/lmcache-native-controller-noop-bench.json` with Python mean/p50/p95
  latency `1.083`/`1.029`/`1.351` ms and native mean/p50/p95 latency
  `3.024`/`3.062`/`3.239` ms.
- A 2-client, 3-iteration-per-client PING controller benchmark smoke wrote
  `/tmp/lmcache-native-controller-concurrent-bench.json` with Python
  `requests_per_s=5.919` and native `requests_per_s=5.917`, while also
  preserving per-request latency and `/proc` resource fields.
- A 3-iteration filesystem-L2 partial-hit LOOKUP benchmark smoke wrote
  `/tmp/lmcache-native-controller-l2-bench.json` with Python mean/p50/p95
  latency `2.811`/`2.628`/`3.369` ms and native mean/p50/p95 latency
  `2.648`/`2.678`/`2.809` ms. The path seeds one of two chunk metadata files
  before each server run, so it covers L2 lookup metadata checks but not KV byte
  movement.
- Native `/loglevel` now supports the Python common route's plain-text get,
  set, list, and invalid-level response shapes for native logger names. The
  focused HTTP test covers `lmcache: NOTSET`, setting `DEBUG`, listing the
  saved logger level, and returning HTTP 400 for an invalid level.
- Native `/threads` now supports the Python common route's plain-text
  diagnostic shape for native HTTP/ZMQ/worker threads. The focused HTTP test
  covers `name` filtering for a native worker thread and the zero-match
  periodic-thread filter case.
- Native periodic-thread HTTP endpoints now expose the Python response shapes
  for an empty native registry. The focused HTTP test covers the empty summary,
  healthy status, missing named thread 404, and invalid level 400 cases.
- Native quota HTTP endpoints now expose the Python route's CRUD/reporting
  shapes for native quota metadata, including `_default`, invalid limit 400s,
  deletion, and filesystem-L2 `current_usage_gb` aggregation. This is not
  `IsolatedLRU` quota enforcement; native eviction remains LRU-only.
- Native `/kvcache/check` now exposes Python-compatible validation and error
  responses for native registered contexts. The focused HTTP test covers
  missing `block_ids`, malformed `block_ids`, non-positive `chunk_size`, missing
  `instance_id`, and empty KV cache metadata. CUDA-enabled native builds now
  compute Python-compatible aggregate and `layerwise=true` MD5 checksum
  responses for supported block-native NHD/HND/MLA layouts; the focused CUDA
  HTTP test verifies both response shapes against Python-computed expected
  digests.
- Malformed core metadata payloads for `REGISTER_KV_CACHE`,
  `UNREGISTER_KV_CACHE`, `LOOKUP`, `FREE_LOOKUP_LOCKS`, `END_SESSION`, and
  `REPORT_BLOCK_ALLOCATION`, plus malformed `QUERY_PREFETCH_STATUS` and
  `QUERY_PREFETCH_LOOKUP_HITS` request-id payloads, now return explicit native
  responses, keep
  subsequent `PING` healthy, leave `registered_context_count=0`, increment
  `invalid_payload_count` for all eight malformed requests, and keep
  `unsupported_count=0`.
- Oversized ZMQ request frames are rejected before msgpack payload decoding.
  The focused regression sends a `LOOKUP` payload frame larger than the native
  request-frame cap, verifies a safe nil response, proves subsequent `PING`
  still succeeds, and checks `invalid_payload_count=1` with
  `unsupported_count=0`.
- Native `lmcache server` startup now accepts `--config-file` without requiring
  `--l1-size-gb` and `--eviction-policy` at argparse time, and the native
  launcher consumes `LMCACHE_CONFIG_FILE` for supported LMCache engine config
  fields. The focused argv regression verifies `chunk_size`, `max_local_cpu_size`,
  `cache_policy`, and `local_disk` translate to native `--chunk-size`,
  `--l1-size-gb`, `--eviction-policy`, and filesystem `--l2-adapter` argv.
  A focused Python-path CLI regression verifies the same config-file values
  from both explicit `--config-file` and `LMCACHE_CONFIG_FILE` seed
  `MPServerConfig` and `StorageManagerConfig` before `run_http_server`.
  A focused native command regression verifies `lmcache server --native
  --config-file ...` and `LMCACHE_CONFIG_FILE` seed those supported values
  before calling the native launcher, and another verifies unsupported
  config-file modes fail before native exec. Focused Python-path and
  native-command regressions also verify config-file values take precedence over
  conflicting engine env vars, including unsupported env-driven modes when a
  config file is present. They also verify equivalent startup seeding from
  `LMCACHE_CHUNK_SIZE`, `LMCACHE_MAX_LOCAL_CPU_SIZE`, `LMCACHE_CACHE_POLICY`,
  and `LMCACHE_LOCAL_DISK`, while native env validation now fails loudly for
  unsupported engine modes such as `LMCACHE_REMOTE_URL` even when no config
  file is present. The direct C++ binary now supports the same flat
  `--config-file` / `LMCACHE_CONFIG_FILE` startup keys, keeps config-file
  values ahead of conflicting engine env vars, and rejects unsupported
  config-file modes before startup.
- A real vLLM 0.21.0 `facebook/opt-125m` native smoke with
  `--use-layerwise` passed: first process had `store_count=1`,
  `retrieve_count=0`, `cache_hits=0`, `cache_misses=4`; the reader raised
  `retrieve_count` to `1`, `cache_hits` to `4`, and `transfer_lock_count` to
  `8`; `transfer_lock_failure_count=0`, `unsupported_count=0`, and
  `clean_native_stderr=true`. This proves the current default PyTorch CUDA IPC
  vLLM connector can run through the native server while LMCache layerwise mode
  is enabled, but it is still only a focused model/workload smoke.
- The same focused vLLM 0.21.0 `facebook/opt-125m` native layerwise smoke also
  passed with `--kv-cache-layout HND`: vLLM resolved `HND`, the reader raised
  `retrieve_count` from `0` to `1`, cache hits from `0` to `4`, and
  `transfer_lock_count` from `4` to `8`; `transfer_lock_failure_count=0`,
  `unsupported_count=0`, and `clean_native_stderr=true`.
- A focused vLLM 0.21.0 `Qwen/Qwen2.5-0.5B-Instruct` native layerwise HND
  run also passed: vLLM resolved `HND`, the first process stored three chunks,
  the reader raised `retrieve_count` from `0` to `1`, cache hits from `0` to
  `3`, and transfer locks from `3` to `6`; `transfer_lock_failure_count=0`,
  `unsupported_count=0`, and `clean_native_stderr=true`.
- A focused vLLM 0.21.0 `Qwen/Qwen3-4B` native layerwise HND run also passed:
  vLLM resolved `HND`, the first process stored five chunks, the reader raised
  `retrieve_count` from `0` to `1`, cache hits from `0` to `5`, and transfer
  locks from `5` to `10`; `transfer_lock_failure_count=0`,
  `unsupported_count=0`, and `clean_native_stderr=true`.
- A broader vLLM 0.21.0 `facebook/opt-125m` native layerwise HND run now
  covers two prompt variants, one writer, two sequential reader processes, one
  warmup round, and two measured steady-state rounds per process. vLLM resolved
  `HND`, native retrieves rose from `5` after the writer to `17` after the
  second reader, cache hits rose from `60` to `204`, transfer locks rose from
  `72` to `216`, `transfer_lock_failure_count=0`, `unsupported_count=0`, and
  `clean_native_stderr=true`.
- A focused real-vLLM layerwise HND concurrent-reader run now covers one writer
  and two simultaneous reader processes with vLLM resolving `HND`. Native
  retrieves rose from `0` after the writer to `2` after the concurrent readers,
  cache hits rose from `0` to `12`, transfer locks rose from `6` to `18`,
  `transfer_lock_failure_count=0`, `unsupported_count=0`, and
  `clean_native_stderr=true`.
- A cached `mistralai/Mistral-7B-Instruct-v0.2` native run passed with one
  writer and one reader process. The writer stored four chunks with four cache
  misses, the reader raised native retrieves from `0` to `1`, cache hits from
  `0` to `4`, transfer locks from `4` to `8`,
  `transfer_lock_failure_count=0`, `unsupported_count=0`,
  `clean_native_stderr=true`, and the native server RSS peak was `177823744`
  bytes.
- A cached `mistralai/Mistral-7B-Instruct-v0.2` native layerwise HND run with
  required real MP trace lifecycle and live `/kvcache/check` checksum assertions
  also passed. vLLM resolved `HND`; the writer stored four chunks, the reader
  raised native retrieves from `0` to `1`, cache hits from `0` to `4`, and
  transfer locks from `4` to `8`; `transfer_lock_failure_count=0`,
  `unsupported_count=0`, `clean_native_stderr=true`, and the 44-row trace
  captured two `REGISTER_KV_CACHE` calls with `use_layerwise=true` and
  `kv_layout=HND`, one real vLLM `STORE`, one real vLLM `RETRIEVE`, `LOOKUP`,
  `QUERY_PREFETCH_STATUS`, `END_SESSION`, and `UNREGISTER_KV_CACHE`. The writer
  `STORE` and reader `RETRIEVE` checksums matched exactly for block range
  `[1,8]`: `81a2297ade10bd984961a3a6e3cb5c7c`,
  `2f27e5fb4961121facd87ea695b7d343`,
  `1ab09d76a8b1a45e2a0fbac83d1984ac`, and
  `ec61fe52467cf6cf631a26d1a25a9fec`.
- A cached `mistralai/Mistral-7B-Instruct-v0.2` steady-state compare reported
  second-reader throughput of 66.040 output tokens/s for Python MP and 16.845
  output tokens/s for native MP (`native_over_python=0.255`), second-reader
  mean TTFT of 0.037s for Python MP and 0.195s for native MP, native retrieves
  increasing from 2 after the writer to 5 after the reader,
  `transfer_lock_failure_count=0`, `unsupported_count=0`,
  `clean_native_stderr=true`, native `rss_peak_bytes=179023872`, and Python MP
  `rss_peak_bytes=1093517312`.
- A real-vLLM metadata trace run now captures MP requests emitted by spawned
  vLLM worker processes instead of only synthetic trace-tool traffic. The
  focused two-reader `facebook/opt-125m` layerwise HND run captured 66
  request/response rows: three `REGISTER_KV_CACHE` calls with
  `use_layerwise=true` and `kv_layout=HND`, one real vLLM `STORE`, two real
  vLLM `RETRIEVE` calls, `LOOKUP`, `QUERY_PREFETCH_STATUS`, `END_SESSION`, and
  `UNREGISTER_KV_CACHE`. This is lifecycle metadata evidence only; it does not
  persist reusable CUDA IPC handles or prove byte-identical replay.
- A focused real-vLLM `facebook/opt-125m` layerwise HND byte-check run now
  queries native `/kvcache/check` inside each worker before KV cache
  unregister. The writer `STORE` and reader `RETRIEVE` both covered block range
  `[1,8]` with four checksum chunks, and the aggregate MD5 checksums matched
  exactly:
  `b192a0a7eeeedc83d2ee0409a5699a52`,
  `decc3e967518619237c1a10281c08f9b`,
  `b57e81535d6ad3a8aad8f6e38fb04c32`, and
  `fde793dc81c308e811ff2dd0ab7daa46`. The same run kept
  `transfer_lock_failure_count=0`, `unsupported_count=0`, and
  `clean_native_stderr=true`.
- Real-vLLM smoke traces with `--require-kvcache-checksum-match` now append a
  `vllm_kvcache_checksum_match` row when `--mp-trace-output` is set, and
  `tools/mp_trace_replay.py` validates the saved writer-vs-reader checksum
  response match. The focused replay regression verifies this row kind against
  the native replay tool, and a negative replay regression verifies mismatched
  reader checksums fail replay. This makes the real-vLLM checksum evidence
  reusable, but it still does not replay captured CUDA IPC handles.
- A lightweight MLA key-normalization regression now covers the rank values used
  before cache-key requests are built: without MLA, `(world_size=8, rank=5)` is
  preserved; with MLA and TP=4, it becomes `(kv_world_size=2, kv_rank=1)`.
- A focused four-client native CUDA concurrency test now covers layerwise-hinted
  registrations: each client registers with `use_layerwise=True`, runs one
  PyTorch CUDA IPC STORE/LOOKUP/RETRIEVE round trip, and the final status
  verifies four registered layerwise contexts, four stores, four retrieves,
  four lookups, eight transfer locks, zero transfer-lock failures, zero
  unsupported requests, and no leaked cache locks.
- A focused CUDA lifecycle test now keeps a lookup lock held while RETRIEVE
  copies the same chunk back to GPU, verifies the temporary transfer lock is
  released while the lookup lock remains held, then releases the lookup lock
  through `FREE_LOOKUP_LOCKS`.
- A focused CUDA overwrite-protection test now holds a lookup lock, attempts a
  duplicate `STORE` for the same ObjectKey, verifies native returns a failed
  transfer result without releasing or corrupting the locked entry, then frees
  the lookup lock and retrieves the original byte-identical KV payload.
- Python-captured CUDA golden trace rows now include
  `expect_lookup_lock_after_retrieve=true`, and native replay verifies `/status`
  still reports the expected lookup lock count after byte-checked `RETRIEVE`
  and before `FREE_LOOKUP_LOCKS`.
- A gated CUDA+TSAN pytest now covers a two-client concurrent PyTorch CUDA IPC
  STORE/LOOKUP/RETRIEVE round trip before the older single-client CUDA+TSAN
  data-path test in source order. Both tests passed against
  `/tmp/lmcache-native-cuda-tsan-build/lmcache-mp-server-native` with
  `TSAN_OPTIONS='halt_on_error=1:second_deadlock_stack=1'` and no
  ThreadSanitizer report.
- A current-source two-hour MP stress soak with 8 clients completed
  4,240,701 handled requests. The final status reported matching
  `expected_request_count`, `request_count`, and `request_latency_count`,
  `active_client_count=8`, `worker_count=8`, and the expected single malformed
  envelope count.
- Python-captured golden trace replay now includes a layerwise-hinted HND CUDA
  byte case. The trace has a `pytorch_cuda_kv_roundtrip` row with
  `layout="HND"`, `kv_layout="HND"`, `use_layerwise=True`, shape
  `[2, 6, 1, 16, 8]`, and native replay verifies the recorded checksum.
- Python-captured golden trace replay also includes a layerwise-hinted
  heterogeneous NHD byte case with two wrapper shapes,
  `[2, 6, 16, 1, 8]` and `[2, 6, 16, 2, 8]`, `use_layerwise=True`, and native
  replay verifies the recorded checksum.
- Python-captured golden trace replay now includes layerwise-hinted cross-layer
  NHD and HND byte cases. The traces use shapes `[6, 4, 2, 16, 1, 8]` and
  `[6, 4, 2, 1, 16, 8]`, set `use_layerwise=True`, and native replay verifies
  the recorded checksums.
- Python-captured golden trace replay now also includes layerwise-hinted
  TRT-LLM 4D and MLA-shaped byte cases. The traces use shapes `[6, 4, 2, 128]`
  and `[6, 16, 8]`, set `use_layerwise=True`, and native replay verifies the
  recorded checksums.
- A single Python-captured golden trace using `--cuda-kv-layout ALL` and
  `--cuda-kv-lifecycle-cycles 2` now appends all 14 PyTorch CUDA layouts twice
  with `use_layerwise=True`: compact NHD, compressed NHD, cross-layer HND/NHD,
  heterogeneous NHD, HND, larger NHD, mixed-compression NHD, MLA, multi-chunk
  NHD, NHD, TRT-LLM 4D, and wide HND/NHD. The generated file has 38 total rows,
  including 28 `pytorch_cuda_kv_roundtrip` rows. Every CUDA row includes
  byte-checked STORE/LOOKUP/RETRIEVE plus `FREE_LOOKUP_LOCKS`, `END_SESSION`,
  and `UNREGISTER_KV_CACHE`, and native replay verifies every recorded
  checksum.

## Verified Commands

```bash
cmake -S . -B /tmp/lmcache-native-root-build -DLMCACHE_BUILD_NATIVE_MP=ON
cmake --build /tmp/lmcache-native-root-build --target lmcache-mp-server-native -j2
cmake -S . -B /tmp/lmcache-native-cuda-build -DLMCACHE_BUILD_NATIVE_MP=ON -DLMCACHE_ENABLE_CUDA=ON
cmake --build /tmp/lmcache-native-cuda-build --target lmcache-mp-server-native -j2
cmake -S LMCache-mp-cpp -B /tmp/lmcache-native-tsan-build -DLMCACHE_BUILD_NATIVE_MP=ON -DLMCACHE_ENABLE_TSAN=ON
cmake --build /tmp/lmcache-native-tsan-build --target lmcache-mp-server-native -j2
cmake -S LMCache-mp-cpp -B /tmp/lmcache-native-cuda-tsan-build -DLMCACHE_BUILD_NATIVE_MP=ON -DLMCACHE_ENABLE_CUDA=ON -DLMCACHE_ENABLE_TSAN=ON
cmake --build /tmp/lmcache-native-cuda-tsan-build --target lmcache-mp-server-native -j2
```

```bash
cmake --install /tmp/lmcache-native-root-build --prefix /tmp/lmcache-native-install
test -x /tmp/lmcache-native-install/bin/lmcache-mp-server-native
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_spills_to_disk_and_reads_back \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_resident_get_refreshes_lru_before_spill \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_disk_promotion_refreshes_lru_before_spill \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_tracks_exact_unaligned_byte_accounting \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_duplicate_store_replaces_spilled_entry \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_failed_spill_store_rolls_back_new_entry \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_failed_spill_overwrite_restores_spilled_entry \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_protects_locked_entries_from_spill_and_remove \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_tracks_nested_lock_counts \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_locked_disk_read_does_not_overfill_dram \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_clear_preserves_locked_and_pinned_entries \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_force_clear_removes_locked_and_pinned_entries \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_lookup_counts_filesystem_l2_hits
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_tracks_exact_unaligned_byte_accounting
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_native_blake3_chunk_hashes_match_python_token_hasher \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_native_blake3_chunk_hashes_match_python_for_nontrivial_tokens \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_native_kv_rank_matches_python_object_key \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_native_object_key_expansion_matches_python \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_native_object_key_strings_match_python_for_real_chunk_hashes \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_native_ipc_key_decode_matches_python_msgspec_wire \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_native_ipc_key_object_key_expansion_matches_python_for_ranges \
            tests/v1/test_vllm_mp_adapter.py::test_mla_rank_normalization_for_mp_cache_keys
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_failed_spill_store_rolls_back_new_entry
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_failed_spill_overwrite_restores_spilled_entry
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_duplicate_store_replaces_spilled_entry
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_locked_disk_read_does_not_overfill_dram
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_resident_get_refreshes_lru_before_spill \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_disk_promotion_refreshes_lru_before_spill
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_spills_to_disk_and_reads_back \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_resident_get_refreshes_lru_before_spill \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_disk_promotion_refreshes_lru_before_spill \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_duplicate_store_replaces_spilled_entry \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_failed_spill_store_rolls_back_new_entry \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_failed_spill_overwrite_restores_spilled_entry \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_protects_locked_entries_from_spill_and_remove \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_locked_disk_read_does_not_overfill_dram \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_protects_pinned_entries_from_spill \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_clear_preserves_locked_and_pinned_entries \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_force_clear_removes_locked_and_pinned_entries \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_unsupported_eviction_policy
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_protects_locked_entries_from_spill_and_remove \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_tracks_nested_lock_counts \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_locked_disk_read_does_not_overfill_dram \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_protects_pinned_entries_from_spill \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_clear_preserves_locked_and_pinned_entries \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_force_clear_removes_locked_and_pinned_entries \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_retrieve_keeps_lookup_lock_until_free \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_rejects_store_over_lookup_locked_chunk \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_free_lookup_locks_releases_key_subset \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_end_session_releases_lookup_locks \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_tracks_same_chunk_lookup_lock_refcounts \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_reused_lookup_request_releases_previous_locks \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_retrieve_partial_missing_releases_locks
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_raw_cuda_ipc_store_retrieve
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_pytorch_cuda_ipc_store_retrieve \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_multi_chunk_pytorch_cuda_ipc \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_repeats_pytorch_cuda_ipc_store_retrieve \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_handles_concurrent_pytorch_cuda_ipc_round_trips \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_pytorch_cuda_ipc_hnd_layout \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_compact_pytorch_cuda_ipc_layout \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_larger_pytorch_cuda_ipc_layout \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_pytorch_cuda_ipc_cross_layer_layout \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_pytorch_cuda_ipc_cross_layer_hnd_layout \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_pytorch_cuda_ipc_mla_layout \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_retrieve_respects_skip_first_tokens \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_raw_cuda_ipc_store_retrieve \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_raw_cuda_ipc_hnd_layout
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_retrieves_filesystem_l2_after_restart
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_native_filesystem_l2_adapter_round_trips_python_key_filename \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_lookup_counts_filesystem_l2_hits
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_free_lookup_locks_releases_key_subset
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_end_session_releases_lookup_locks
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_reused_lookup_request_releases_previous_locks
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_retrieve_missing_key_fails_cleanly
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_retrieve_partial_missing_releases_locks
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_unregister_kv_cache_rejects_later_transfer
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_accepts_layerwise_hint_for_supported_cuda_ipc
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_tracks_same_chunk_lookup_lock_refcounts
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_connector_layerwise.py
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/test_vllm_layout_hints.py tests/v1/test_vllm_mp_adapter.py
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_server_command_native_env_launches_native \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_server_command_python_escape_overrides_native_env
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_unsupported_eviction_policy \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cli_argv_rejects_unsupported_python_only_options
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with openai \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_fails_loudly_for_unsupported_l2_adapter \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_separate_worker_pools \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_unsupported_eviction_policy \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_unsupported_config_file_mode \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cli_argv_rejects_unsupported_engine_env \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cli_argv_rejects_unsupported_python_only_options \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_unsupported_kv_layout_metadata \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_malformed_kv_transfer_payloads \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_reports_invalid_core_metadata_payloads \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_oversized_zmq_payload_and_stays_healthy \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_out_of_range_request_type \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_reports_invalid_blend_payloads \
            tests/cli/commands/test_server.py::TestServerCommandExecute::test_execute_native_rejects_unsupported_config_file_mode \
            tests/cli/commands/test_server.py::TestServerCommandExecute::test_execute_native_rejects_unsupported_env_mode
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_speaks_controller_protocol_and_http \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_kvcache_check_reports_missing_instance \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_kvcache_check_returns_cuda_checksums
```

```bash
! rg -n "TODO|FIXME|stub" LMCache-mp-cpp/src LMCache-mp-cpp/include
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_pytorch_cuda_ipc_store_retrieve \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_multi_chunk_pytorch_cuda_ipc \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_raw_cuda_ipc_store_retrieve \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_pytorch_cuda_ipc_hnd_layout \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_layerwise_heterogeneous_cuda_ipc_layout \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_mixed_compression_pytorch_cuda_ipc_layout \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_trtllm_4d_pytorch_cuda_ipc_layout \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_pytorch_cuda_ipc_mla_layout
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_survives_malformed_frame_and_ping_stress \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_reports_worker_queue_backpressure \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_oversized_zmq_payload_and_stays_healthy \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_out_of_range_request_type \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_handles_concurrent_pytorch_cuda_ipc_round_trips \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_handles_layerwise_concurrent_pytorch_cuda_ipc_round_trips
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  python -m py_compile benchmarks/mp_native_vs_python/vllm_native_smoke.py
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --help | \
  rg -- '--use-layerwise'
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_vllm_smoke_round_summary_reports_percentiles
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_trace_replay_validates_real_vllm_checksum_match_rows \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_trace_replay_rejects_real_vllm_checksum_mismatch_rows
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  python -m py_compile benchmarks/mp_native_vs_python/vllm_native_smoke.py tools/mp_trace_replay.py
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_connector_layerwise.py \
            tests/v1/test_vllm_mp_adapter.py \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_accepts_layerwise_hint_for_supported_cuda_ipc \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_trace_replay_validates_real_vllm_checksum_match_rows \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_trace_replay_rejects_real_vllm_checksum_mismatch_rows
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_clear_preserves_locked_and_pinned_entries \
            tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py::test_tiered_cache_force_clear_removes_locked_and_pinned_entries \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_speaks_controller_protocol_and_http \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_pytorch_cuda_ipc_store_retrieve \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_free_lookup_locks_releases_key_subset \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_end_session_releases_lookup_locks
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_unsupported_kv_layout_metadata \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_decodes_registered_kv_cache_metadata
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_survives_malformed_frame_and_ping_stress
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_reports_invalid_core_metadata_payloads
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_oversized_zmq_payload_and_stays_healthy
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_reports_worker_queue_backpressure
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_applies_startup_log_level \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_invalid_startup_log_level \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cli_argv_passes_log_level
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_uses_config_file_env_for_startup \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_unsupported_config_file_mode
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_out_of_range_request_type
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_handles_sigterm_gracefully
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  python tools/mp_tsan_stress.py \
    --binary /tmp/lmcache-native-stress-build/lmcache-mp-server-native \
    --workers 8 --iterations 100
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  python tools/mp_tsan_stress.py \
    --binary /tmp/lmcache-native-tsan-build/lmcache-mp-server-native \
    --workers 8 --iterations 100 --setarch --tsan
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  python tools/mp_tsan_stress.py \
    --binary /tmp/lmcache-native-tsan-build/lmcache-mp-server-native \
    --workers 8 --iterations 500 --setarch --tsan
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  python tools/mp_tsan_stress.py \
    --binary /tmp/lmcache-native-root-build/LMCache-mp-cpp/lmcache-mp-server-native \
    --workers 8 --duration-s 60
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  python tools/mp_tsan_stress.py \
    --binary /tmp/lmcache-native-root-build/LMCache-mp-cpp/lmcache-mp-server-native \
    --workers 8 --duration-s 7200
```

```bash
LMCACHE_RUN_CUDA_TSAN_STRESS=1 \
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cuda-tsan-build/lmcache-mp-server-native \
TSAN_OPTIONS='halt_on_error=1:second_deadlock_stack=1' \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 setarch $(uname -m) -R \
  uv run --python 3.12 --with pytest --with numpy --with cuda-python pytest -q \
  tests/v1/multiprocess/test_lmcache_mp_native_binary.py \
  -k 'tsan and pytorch_cuda_ipc'
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cuda-tsan-build/lmcache-mp-server-native \
TSAN_OPTIONS='halt_on_error=1:second_deadlock_stack=1' \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 setarch $(uname -m) -R \
  uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-layerwise-all-cycles-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with numpy \
  python tools/mp_protocol_fuzz.py \
    --binary /tmp/lmcache-native-root-build/LMCache-mp-cpp/lmcache-mp-server-native \
    --iterations 64 --min-invalid-payloads 8
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with numpy \
  python tools/mp_protocol_fuzz.py \
    --binary /tmp/lmcache-native-root-build/LMCache-mp-cpp/lmcache-mp-server-native \
    --iterations 1024 --min-invalid-payloads 1000
```

```bash
LMCACHE_RUN_LONG_CUDA_STRESS=1 \
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}" \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy --with cuda-python pytest -q \
  tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_long_pytorch_cuda_ipc_store_retrieve
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
LMCACHE_RUN_LONG_CUDA_CONCURRENCY_STRESS=1 \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy --with cuda-python pytest -q \
  tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_long_concurrent_pytorch_cuda_ipc_round_trips
```

```bash
CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13" \
PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH" \
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}" \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 MAX_JOBS=2 \
  uv run --python 3.12 python setup.py build_ext --inplace
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}" \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/test_mp_mem_kernels.py
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy \
  pytest -q tests/v1/gpu_connector/test_utils_shape_desc.py::test_normalize_vllm_cross_layer_one_wrapper_list
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  python tools/mp_trace_capture.py --server python --output /tmp/lmcache-python-golden-rich.jsonl
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  python tools/mp_trace_replay.py --server native --input /tmp/lmcache-python-golden-rich.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  python tools/mp_trace_capture.py --server python \
    --output /tmp/lmcache-python-golden-rich-status.jsonl
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-rich-status.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-raw-cuda-kv \
    --output /tmp/lmcache-python-golden-kv.jsonl
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-raw-cuda-kv \
    --cuda-kv-layout HND --output /tmp/lmcache-python-golden-raw-hnd-kv.jsonl
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-raw-hnd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --include-raw-cuda-kv --output /tmp/lmcache-python-golden-mixed-kv.jsonl
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-mixed-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --output /tmp/lmcache-python-golden-pytorch-kv.jsonl
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-pytorch-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --cuda-kv-layout NHD \
    --output /tmp/lmcache-python-golden-lookup-lock-retrieve-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-lookup-lock-retrieve-kv.jsonl
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  python -m py_compile tools/mp_trace_capture.py tools/mp_trace_replay.py
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --cuda-kv-layout HND --output /tmp/lmcache-python-golden-pytorch-hnd-kv.jsonl
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-pytorch-hnd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/gpu_connector/test_utils_shape_desc.py::test_normalize_vllm_compact_nhd_one_wrapper_list
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_compact_pytorch_cuda_ipc_layout
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_compressed_pytorch_cuda_ipc_layout
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_mixed_compression_pytorch_cuda_ipc_layout
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_layerwise_heterogeneous_cuda_ipc_layout
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_handles_layerwise_concurrent_pytorch_cuda_ipc_round_trips
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_rejects_store_over_lookup_locked_chunk
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_kvcache_check_returns_cuda_checksums
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_multi_chunk_pytorch_cuda_ipc
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  pytest -q \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cuda_binary_round_trips_trtllm_4d_pytorch_cuda_ipc_layout \
    tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_unsupported_kv_layout_metadata
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --cuda-kv-layout COMPACT_NHD \
    --output /tmp/lmcache-python-golden-compact-nhd-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-compact-nhd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --cuda-kv-layout COMPRESSED_NHD \
    --output /tmp/lmcache-python-golden-compressed-nhd-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-compressed-nhd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --cuda-kv-layout MIXED_COMPRESSION_NHD \
    --output /tmp/lmcache-python-golden-mixed-compression-nhd-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-mixed-compression-nhd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --cuda-kv-layout HETEROGENEOUS_NHD \
    --output /tmp/lmcache-python-golden-heterogeneous-nhd-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-heterogeneous-nhd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --cuda-kv-layout MULTI_CHUNK_NHD \
    --output /tmp/lmcache-python-golden-multi-chunk-nhd-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-multi-chunk-nhd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --cuda-kv-layout LARGE_NHD \
    --output /tmp/lmcache-python-golden-pytorch-large-nhd-kv.jsonl
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-pytorch-large-nhd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --cuda-kv-layout WIDE_NHD \
    --output /tmp/lmcache-python-golden-wide-nhd-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-wide-nhd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --cuda-kv-layout WIDE_HND \
    --output /tmp/lmcache-python-golden-wide-hnd-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-wide-hnd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --cuda-kv-layout CROSS_LAYER_NHD \
    --output /tmp/lmcache-python-golden-pytorch-cross-layer-nhd-kv.jsonl
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-pytorch-cross-layer-nhd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --cuda-kv-layout CROSS_LAYER_HND \
    --output /tmp/lmcache-python-golden-pytorch-cross-layer-hnd-kv.jsonl
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-pytorch-cross-layer-hnd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --cuda-kv-layout MLA --output /tmp/lmcache-python-golden-pytorch-mla-kv.jsonl
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-pytorch-mla-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --cuda-kv-layout TRTLLM_4D \
    --output /tmp/lmcache-python-golden-trtllm-4d-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-trtllm-4d-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --include-fs-l2-partial-hit \
    --output /tmp/lmcache-python-golden-fs-l2-partial-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-fs-l2-partial-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --include-layerwise-hint \
    --output /tmp/lmcache-python-golden-layerwise-hint-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-layerwise-hint-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --include-layerwise-hint --cuda-kv-layout HND \
    --output /tmp/lmcache-python-golden-layerwise-hnd-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-layerwise-hnd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --include-layerwise-hint --cuda-kv-layout HETEROGENEOUS_NHD \
    --output /tmp/lmcache-python-golden-layerwise-heterogeneous-nhd-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-layerwise-heterogeneous-nhd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --include-layerwise-hint --cuda-kv-layout CROSS_LAYER_NHD \
    --output /tmp/lmcache-python-golden-layerwise-cross-layer-nhd-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-layerwise-cross-layer-nhd-kv.jsonl

SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --include-layerwise-hint --cuda-kv-layout CROSS_LAYER_HND \
    --output /tmp/lmcache-python-golden-layerwise-cross-layer-hnd-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-layerwise-cross-layer-hnd-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --include-layerwise-hint --cuda-kv-layout TRTLLM_4D \
    --output /tmp/lmcache-python-golden-layerwise-trtllm-4d-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-layerwise-trtllm-4d-kv.jsonl

SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --include-layerwise-hint --cuda-kv-layout MLA \
    --output /tmp/lmcache-python-golden-layerwise-mla-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-layerwise-mla-kv.jsonl
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  python tools/mp_protocol_schema.py
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_capture.py --server python --include-pytorch-cuda-kv \
    --include-layerwise-hint --cuda-kv-layout ALL --cuda-kv-lifecycle-cycles 2 \
    --output /tmp/lmcache-python-golden-layerwise-all-cycles-kv.jsonl
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with cuda-python \
  python tools/mp_trace_replay.py --server native \
    --input /tmp/lmcache-python-golden-layerwise-all-cycles-kv.jsonl
wc -l /tmp/lmcache-python-golden-layerwise-all-cycles-kv.jsonl
rg '"kind": "pytorch_cuda_kv_roundtrip"' /tmp/lmcache-python-golden-layerwise-all-cycles-kv.jsonl | wc -l
rg '"free_lookup_locks": true' /tmp/lmcache-python-golden-layerwise-all-cycles-kv.jsonl | wc -l
rg '"end_session": true' /tmp/lmcache-python-golden-layerwise-all-cycles-kv.jsonl | wc -l
rg '"unregister_kv_cache": true' /tmp/lmcache-python-golden-layerwise-all-cycles-kv.jsonl | wc -l
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  python tools/mp_trace_capture.py --server python --include-raw-cuda-kv \
    --cuda-kv-layout ALL --output /tmp/should-not-write.jsonl
```

Expected result: exits with `RuntimeError: --cuda-kv-layout ALL is only
supported for PyTorch CUDA KV`.

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 NO_CUDA_EXT=1 uv run --isolated \
  --python 3.12 --with-editable . --with numpy python - <<'PY'
import lmcache
from lmcache.v1.multiprocess.native_launcher import native_argv_from_args
print("lmcache import ok", getattr(lmcache, "__version__", "unknown"))
print("native launcher import ok", callable(native_argv_from_args))
PY
```

```bash
CUDA_HOME="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13" \
PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH" \
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}" \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 MAX_JOBS=2 uv run --isolated \
  --python 3.12 --with-editable . --with numpy python - <<'PY'
import lmcache
import lmcache.c_ops as c_ops
from lmcache.v1.multiprocess.native_launcher import native_argv_from_args
print("lmcache import ok", getattr(lmcache, "__version__", "unknown"))
print("c_ops import ok", hasattr(c_ops, "GPUKVFormat"))
print("native launcher import ok", callable(native_argv_from_args))
PY
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 NO_CUDA_EXT=1 LMCACHE_BUILD_NATIVE_MP=1 \
  uv run --python 3.12 python setup.py build \
    --build-base /tmp/lmcache-native-package-build
test -x /tmp/lmcache-native-package-build/lib/lmcache/bin/lmcache-mp-server-native
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 NO_CUDA_EXT=1 LMCACHE_BUILD_NATIVE_MP=1 \
  uv run --python 3.12 --with setuptools --with wheel \
  python setup.py bdist_wheel \
  --dist-dir /tmp/lmcache-native-nocuda-package-wheel \
  --bdist-dir /tmp/lmcache-native-nocuda-package-bdist
wheel=$(find /tmp/lmcache-native-nocuda-package-wheel -name '*.whl' -print -quit)
uv run --python 3.12 --with pip python -m pip install --no-deps \
  --target /tmp/lmcache-native-nocuda-package-install "$wheel"
test -x \
  /tmp/lmcache-native-nocuda-package-install/lmcache/bin/lmcache-mp-server-native
# Started the packaged binary and sent valid REGISTER_KV_CACHE, STORE, and
# RETRIEVE requests. STORE and RETRIEVE returned `(b'', False)` with
# cuda_transfer_enabled=false, store_count=1, retrieve_count=1,
# unsupported_count=2, and transfer_lock_failure_count=1.
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 NO_CUDA_EXT=1 LMCACHE_BUILD_NATIVE_MP=1 \
  LMCACHE_NATIVE_MP_ENABLE_CUDA=1 uv run --python 3.12 python setup.py build \
    --build-base /tmp/lmcache-native-package-cuda-build
test -x \
  /tmp/lmcache-native-package-cuda-build/lib/lmcache/bin/lmcache-mp-server-native-cuda
PYTHONPATH=/tmp/lmcache-native-package-cuda-build/lib \
  /home/cxlsol/work/dongjoo/lmcache_cpp_mirror/LMCache/.venv/bin/python - <<'PY'
from lmcache.v1.multiprocess.native_launcher import ensure_native_binary
path = ensure_native_binary(enable_cuda=True)
print(path)
assert path.name == "lmcache-mp-server-native-cuda"
PY
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 NO_CUDA_EXT=1 LMCACHE_BUILD_NATIVE_MP=1 \
  LMCACHE_NATIVE_MP_ENABLE_CUDA=1 uv run --python 3.12 \
  --with setuptools --with wheel python setup.py bdist_wheel \
  --dist-dir /tmp/lmcache-native-cuda-package-wheel \
  --bdist-dir /tmp/lmcache-native-cuda-package-bdist
wheel=$(find /tmp/lmcache-native-cuda-package-wheel -name '*.whl' -print -quit)
uv run --python 3.12 --with pip python -m pip install --no-deps \
  --target /tmp/lmcache-native-cuda-package-install "$wheel"
test -x \
  /tmp/lmcache-native-cuda-package-install/lmcache/bin/lmcache-mp-server-native-cuda
/tmp/lmcache-native-cuda-package-install/lmcache/bin/lmcache-mp-server-native-cuda \
  --help
```

```bash
uv run --python 3.12 --with PyYAML python - <<'PY'
from pathlib import Path
import yaml

for path in [
    ".github/workflows/build_main_artifacts.yml",
    ".github/workflows/build_cu129_artifacts.yml",
]:
    yaml.safe_load(Path(path).read_text())
PY
```

```bash
cmake -S LMCache-mp-cpp -B /tmp/lmcache-native-cmake-llvm-check \
  -DLMCACHE_BUILD_NATIVE_MP=ON
cmake --build /tmp/lmcache-native-cmake-llvm-check \
  --target lmcache-mp-server-native -j2
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/controller_latency.py --iterations 5 \
    --output /tmp/lmcache-native-controller-bench.json
```

```bash
uvx pre-commit run --files <changed files>
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with pytest --with numpy --with openai \
  python - <<'PY' | rg -- '--native-cuda|--native|--python'
import sys
from lmcache.cli.main import main
sys.argv = ["lmcache", "server", "--help"]
main()
PY
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with vllm --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --output /tmp/lmcache-native-vllm-smoke.json
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with vllm --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --batch-size 2 --prompt-repetitions 40 --max-model-len 512 \
    --output /tmp/lmcache-native-vllm-batch-smoke.json
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with vllm --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --reader-processes 2 --batch-size 2 --prompt-repetitions 48 \
    --max-model-len 512 --worker-timeout-s 240 \
    --output /tmp/lmcache-native-vllm-two-reader-smoke.json
```

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

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --kv-cache-layout HND --batch-size 1 --prompt-repetitions 24 \
    --max-model-len 256 --worker-timeout-s 300 \
    --output /tmp/lmcache-native-vllm-hnd-smoke.json
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --use-layerwise --batch-size 1 --prompt-repetitions 24 \
    --max-model-len 256 --worker-timeout-s 300 \
    --require-clean-native-stderr \
    --output /tmp/lmcache-native-vllm-layerwise-smoke.json
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --use-layerwise --kv-cache-layout HND --batch-size 1 \
    --prompt-repetitions 24 --max-model-len 256 --worker-timeout-s 300 \
    --require-clean-native-stderr \
    --output /tmp/lmcache-native-vllm-layerwise-hnd-smoke.json
```

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --server native \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --prompt-repetitions 24 --max-model-len 256 --max-tokens 2 \
    --gpu-memory-utilization 0.30 --worker-timeout-s 300 \
    --use-layerwise --kv-cache-layout HND \
    --require-clean-native-stderr \
    --output /tmp/lmcache-native-vllm-qwen25-layerwise-hnd-smoke.json
```

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

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --server native \
    --model facebook/opt-125m --use-layerwise --kv-cache-layout HND \
    --batch-size 1 --reader-processes 2 --prompt-repetitions 32 \
    --max-model-len 256 --max-tokens 2 --gpu-memory-utilization 0.40 \
    --worker-timeout-s 300 \
    --require-clean-native-stderr \
    --mp-trace-output /tmp/lmcache-native-vllm-layerwise-hnd-two-reader-real-mp-trace.jsonl \
    --require-mp-trace-lifecycle \
    --output /tmp/lmcache-native-vllm-layerwise-hnd-two-reader-real-mp-trace-smoke.json
wc -l /tmp/lmcache-native-vllm-layerwise-hnd-two-reader-real-mp-trace.jsonl
```

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

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --batch-size 1 --prompt-repetitions 24 --max-model-len 512 \
    --worker-timeout-s 360 \
    --output /tmp/lmcache-native-vllm-qwen-0.5b-smoke.json
```

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

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --server native --model mistralai/Mistral-7B-Instruct-v0.2 \
    --use-layerwise --kv-cache-layout HND --batch-size 1 \
    --reader-processes 1 --prompt-repetitions 24 --max-model-len 512 \
    --max-tokens 4 --worker-timeout-s 900 --require-clean-native-stderr \
    --require-mp-trace-lifecycle --require-kvcache-checksum-match \
    --mp-trace-output /tmp/lmcache-native-vllm-mistral7b-layerwise-hnd-byte-check-trace.jsonl \
    --output /tmp/lmcache-native-vllm-mistral7b-layerwise-hnd-byte-check.json
```

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

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py \
    --batch-size 1 --prompt-repetitions 24 --max-model-len 256 \
    --worker-timeout-s 300 --require-clean-native-stderr \
    --output /tmp/lmcache-native-vllm-clean-stderr-smoke.json
```

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

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with vllm --with pytest --with numpy --with cuda-python \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --raw-cuda-ipc \
    --output /tmp/lmcache-native-vllm-raw-smoke.json
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with vllm --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --compare-python \
    --output /tmp/lmcache-vllm-native-vs-python.json
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with vllm --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --compare-python \
    --steady-state-warmup-rounds 1 --steady-state-rounds 2 \
    --batch-size 1 --prompt-repetitions 24 --max-model-len 256 \
    --worker-timeout-s 240 \
    --output /tmp/lmcache-vllm-native-vs-python-steady-ttft.json
```

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

```bash
LMCACHE_MP_NATIVE_CUDA_BINARY=/tmp/lmcache-native-cpp-cuda-build/lmcache-mp-server-native \
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  --with 'vllm==0.21.0' --with pytest --with numpy \
  python benchmarks/mp_native_vs_python/vllm_native_smoke.py --server native \
    --model facebook/opt-125m --prompt-repetitions 24 --max-model-len 256 \
    --max-tokens 2 --gpu-memory-utilization 0.30 --worker-timeout-s 240 \
    --require-clean-native-stderr \
    --output /tmp/lmcache-native-vllm-resource-smoke.json
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_speaks_controller_protocol_and_http \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_kvcache_check_reports_missing_instance
```

```bash
uvx pre-commit run --files \
  LMCache-mp-cpp/include/lmcache_mp_cpp/http_server.h \
  LMCache-mp-cpp/src/http_server.cpp \
  LMCache-mp-cpp/include/lmcache_mp_cpp/l2_adapter.h \
  LMCache-mp-cpp/src/l2_adapter.cpp \
  LMCache-mp-cpp/include/lmcache_mp_cpp/native_server.h \
  LMCache-mp-cpp/src/native_server.cpp \
  tests/v1/multiprocess/test_lmcache_mp_native_binary.py \
  LMCache-mp-cpp/README.md \
  LMCache-mp-cpp/docs/native_mp_status.md \
  LMCache-mp-cpp/docs/goal_audit.md \
  LMCache-mp-cpp/include/lmcache_mp_cpp/l2_adapter.h \
  LMCache-mp-cpp/src/l2_adapter.cpp \
  LMCache-mp-cpp/python/lmcache_mp_cpp/bindings.py \
  LMCache-mp-cpp/python/lmcache_mp_cpp/l2_adapter.py \
  tests/v1/multiprocess/test_lmcache_mp_cpp_tiered_cache.py \
  .github/workflows/build_main_artifacts.yml \
  .github/workflows/build_cu129_artifacts.yml \
  LMCache-mp-cpp/CMakeLists.txt \
  LMCache-mp-cpp/src/key_compat.cpp
git diff --check
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy \
  pytest -q tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_uses_config_file_env_for_startup \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_binary_rejects_unsupported_config_file_mode \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cli_argv_can_select_no_cuda_binary \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cli_argv_can_select_cuda_binary \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cli_argv_passes_log_level \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cli_argv_uses_supported_config_file_env \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cli_argv_config_file_precedes_engine_env \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cli_argv_rejects_unsupported_engine_env \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_run_native_server_strips_config_file_env_after_translation \
            tests/v1/multiprocess/test_lmcache_mp_native_binary.py::test_native_cli_argv_rejects_unsupported_python_only_options
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 --with pytest --with numpy --with openai \
  pytest -q tests/cli/commands/test_server.py::TestServerCommandArguments::test_config_file_can_seed_required_storage_args \
            tests/cli/commands/test_server.py::TestServerCommandExecute::test_execute_calls_run_http_server \
            tests/cli/commands/test_server.py::TestServerCommandExecute::test_execute_uses_config_file_for_storage_defaults \
            tests/cli/commands/test_server.py::TestServerCommandExecute::test_execute_uses_config_file_env_for_storage_defaults \
            tests/cli/commands/test_server.py::TestServerCommandExecute::test_execute_config_file_precedes_engine_env \
            tests/cli/commands/test_server.py::TestServerCommandExecute::test_execute_native_uses_config_file_before_launch \
            tests/cli/commands/test_server.py::TestServerCommandExecute::test_execute_native_uses_config_file_env_before_launch \
            tests/cli/commands/test_server.py::TestServerCommandExecute::test_execute_native_config_file_precedes_engine_env \
            tests/cli/commands/test_server.py::TestServerCommandExecute::test_execute_native_rejects_unsupported_config_file_mode \
            tests/cli/commands/test_server.py::TestServerCommandExecute::test_execute_uses_env_for_storage_defaults \
            tests/cli/commands/test_server.py::TestServerCommandExecute::test_execute_native_uses_env_before_launch \
            tests/cli/commands/test_server.py::TestServerCommandExecute::test_execute_native_rejects_unsupported_env_mode
```

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --python 3.12 \
  python -m py_compile lmcache/v1/multiprocess/native_launcher.py \
    lmcache/cli/commands/server.py lmcache/v1/distributed/config.py
```

```bash
uvx pre-commit run --files \
  lmcache/v1/multiprocess/native_launcher.py \
  lmcache/cli/commands/server.py \
  lmcache/v1/distributed/config.py \
  tests/v1/multiprocess/test_lmcache_mp_native_binary.py \
  tests/cli/commands/test_server.py \
  LMCache-mp-cpp/README.md \
  LMCache-mp-cpp/docs/native_mp_status.md \
  LMCache-mp-cpp/docs/goal_audit.md
```

## Next Required Work

No remaining GOAL.md acceptance gap is currently identified. Broader production
benchmark sweeps remain a useful follow-up, but the native-vs-Python benchmark
artifact now covers controller latency, vLLM TTFT/throughput, cache-hit
behavior, resource deltas, concurrency, L2 metadata lookup, larger-model
smokes, and real-vLLM `STORE`/`LOOKUP`/`RETRIEVE` request latencies.
