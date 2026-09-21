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
- `restart-recovery.sh`: LMCache server restart and worker re-registration.
- `http-api.sh`: LMCache HTTP API and CLI coverage with a live engine.

`vllm/` contains workloads that currently depend on vLLM-specific commands,
response extensions, endpoints, logs, or model launch flags. A future
`sglang/` directory holds the corresponding SGLang-only coverage. For example,
`mp-autostart-tp2.sh` verifies vLLM-specific connector auto-start behavior and
therefore remains under `vllm/` rather than `common/`.

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
