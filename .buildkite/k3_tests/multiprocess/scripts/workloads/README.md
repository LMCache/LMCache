# Multiprocess workloads

Workloads are grouped by their inference-engine compatibility. The
multiprocess harness normally owns the LMCache and inference-engine lifecycle;
a workload owns its requests and assertions.

`common/` contains workloads that use the shared engine-adapter contract:

- `lm-eval.sh`: two `lm_eval` runs with output or score consistency checks.
- `long-doc-qa.sh`: LMCache-versus-baseline long-document performance checks.
- `high-concurrency.sh`: concurrent random-prefill completion validation.
- `deadlock.sh`: high-concurrency deadlock regression.
- `long-doc-qa-l2.sh`: L2 restart, data-flow, and performance validation.
  Its checks live in `long-doc-qa-l2-verify.sh`, which is not a workload: it
  runs every check before exiting so a threshold failure still leaves the
  `/metrics` snapshot and a per-check summary in `RESULTS_DIR`, and it runs
  on CPU with stub inputs (`tests/tools/test_long_doc_qa_l2_verify.py`).
- `restart-recovery.sh`: restart LMCache after a heartbeat-length outage, verify
  the original workers re-register, and require the second random-prefill bench
  to complete at least 80% of the first round's writes. Also verify store, retrieve,
  engine-loaded chunks and identical output. Fast restarts are not covered.
- `multi-chunk-size.sh`: two engines connected to one MP server register
  different model-scoped chunk sizes and independently complete an L1
  store/retrieve round trip.
- `http-api.sh`: LMCache HTTP API and CLI coverage with a live engine.

`verification/` contains shared assertions and request helpers that are sourced
by executable workloads but are not discoverable workloads themselves.

`vllm/` and `sglang/` contain workloads that depend on engine-specific
commands, response extensions, endpoints, logs, or model launch flags. For
example, each engine has its own `dsv4-flash-tp.sh` and `kimi-linear-tp.sh`
lifecycle while sourcing shared prompt construction and L1 round-trip checks
from `verification/`. `mp-autostart-tp2.sh` verifies vLLM-specific connector
auto-start behavior and therefore remains under `vllm/` rather than `common/`.
`client-wheel.sh` builds the `lmcache-client` wheel, runs vLLM on it from a
second venv against a server from the full `lmcache`, and checks a store and
an L1 retrieve in both transfer modes with the shared round-trip helpers.
`sleep-mode.sh` is under `vllm/` for the same reason: it drives vLLM's
`/sleep` and `/wake_up` development endpoints. Its checks live in
`sleep_mode.py`, which is not a workload: two sleep/wake-up cycles, during
which the LMCache server must drop and re-create its KV cache registration,
GPU memory must be released and come back, and the prompt must hit LMCache
with the output from before sleep. `LMCACHE_MP_TRANSFER_MODE` selects which
registration of the server's `/status` the checks read.

The two P2P workloads under `vllm/` launch their own coordinator and LMCache
servers on localhost:

- `p2p.sh`: two engines, each with its own server; the second engine must be
  served the first engine's KV over P2P.
- `p2p-write.sh`: one engine whose server runs with `--enable-p2p-write`,
  plus a headless peer server with no engine. Prompts fill the main server's
  small L1 past the usage threshold; the early ones must stay local, the later
  ones must land on the peer, and all of them must be served from LMCache on
  replay. Requests and checks are in `p2p_write_verify.py`; its verdict runs on
  CPU with recorded rows (`tests/ci/test_p2p_write_verify.py`).

The harness converts a test name from underscores to hyphens and discovers the
script automatically. For example, `long_doc_qa` resolves to
`common/long-doc-qa.sh`, while `cache_stats` with `INFERENCE_ENGINE=vllm`
resolves to `vllm/cache-stats.sh`. Only test names that share an implementation
need an explicit alias in `../workload-discovery.sh`.

Common workloads use `ENGINE_PORT`, `ENGINE_BASELINE_PORT`, `MODEL`, and
`RESULTS_DIR`. Engine-specific launch behavior must be exposed by an adapter in
`../engines/`. Each engine's CI matrix should run the common workloads plus the
workloads in its engine-named directory. Tests remain separate Buildkite steps
because their GPU counts, models, timeouts, and retry policies differ.
An adapter may temporarily skip unsupported common workloads through
`ENGINE_COMMON_WORKLOAD_BLACKLIST`; this is an exception list, not a workload
allowlist. Transfer-mode and request-transport support is declared separately
by the adapter, so a common pipeline matrix can safely include engines with
different capabilities.

CPU E2E reuses `common/restart-recovery.sh` and `vllm/fault-tolerance.sh`
in its opt-125m / LMCache-driven / gRPC leg. `RR_PREPARE_ONLY=1` captures the first
benchmark (4 requests × 512 tokens) and current registrations before fault
injection. `FT_REUSE_SERVICES=1` preserves the running engine;
`RR_SERVER_STOPPED=1` then resumes recovery from that same outage.
`RR_EXCLUDE_REGISTRATIONS` points to the status snapshot taken before the earlier
vLLM restart, so retired Worker IDs are excluded from the expected recovery set.
Both workloads use `MP_TEST_PID_FILE` (server PID on line 1);
`MP_TEST_LMCACHE_LAUNCH_SCRIPT` supplies the exact server launch command.
Recovery waits for an actual completed store before measuring the second bench:
server-side registration can precede completion of the Worker's recovery callback.
Benchmark summaries, counter deltas, outputs and registration snapshots are retained.

When running the CPU workload from a separate checkout, ensure the installed
LMCache CLI points to that checkout or to the intended server wheel. The
OPT-125M random-prefill benchmark sends chat requests, so the engine needs a chat
template. For this fixed BF16 model, the CPU harness supplies
`RR_TOKENS_PER_GB=29127` (36,864 KV bytes per token) because an offline engine can
register a snapshot path that differs from the served model ID.
