# Multiprocess inference-engine adapters

The multiprocess harness owns the LMCache server and process lifecycle. Engine
adapters own only inference-engine setup, launch arguments, readiness probes,
and engine-specific cleanup.

Select an adapter with `INFERENCE_ENGINE` (default: `vllm`). An adapter named
`<engine>.sh` provides:

- `ENGINE_NAME`
- `ENGINE_DEFAULT_MODEL`
- `ENGINE_COMMON_WORKLOAD_BLACKLIST` (array, empty by default)
- `ENGINE_SUPPORTED_TRANSFER_MODES` (array, optional; unrestricted if omitted)
- `ENGINE_SUPPORTED_REQUEST_TRANSPORTS` (array, optional; unrestricted if omitted)
- `engine_setup_environment <repo-root>`
- `engine_configure_defaults`
- `engine_configure_workload <test-name>` (optional)
- `engine_prepare_launch <device-index>`
- `engine_add_lmcache_server_environment <array-name>`
- `engine_launch <lmcache|baseline> <port> <device-index> <log-file>`
- `engine_launch_profile <profile> <lmcache|baseline> <port> <device-index> <log-file>` (optional)
- `engine_ready_urls <port>`
- `engine_clear_local_cache <port>` (optional; must preserve LMCache contents)
- `engine_count_preemptions <log-file>` (required only by preemption workloads)
- `engine_print_timeout_diagnostics <log-file>` (optional)
- `engine_cleanup_processes` (optional)

`engine_launch` stores the background process ID in `ENGINE_PID`. Workloads
receive only `ENGINE_PORT`, `ENGINE_BASELINE_PORT`, `MODEL`, and OpenAI-compatible
HTTP endpoints, so they can be reused by another serving engine.

Tests use named launch profiles when their topology differs from the default.
The test owns engine-neutral scenario parameters and assertions; each adapter
maps the profile to its engine-specific command line.

The workload tree defines support automatically. Tests under `../workloads/common/`
are shared by every engine. Tests under `../workloads/<engine>/` are available
only when that adapter is selected. No adapter-side test allowlist is required.
An adapter can temporarily exclude a common workload by adding its normalized
underscore name to `ENGINE_COMMON_WORKLOAD_BLACKLIST`. Blacklisted workloads
are skipped before dependencies are installed or servers are launched.

The supported-mode and supported-transport arrays filter shared Buildkite
matrix combinations at the same early preflight stage. This lets a new engine
join `x-common-inference-engines` without duplicating pipeline steps merely
because it implements fewer transfer modes or transports.

Use `engine_configure_workload` for engine-specific defaults instead of adding
engine-specific copies of common pipeline steps. For example, an adapter can
set long-document performance thresholds or enable retrieval verification for
`lm_eval`. Workloads can call `engine_clear_local_cache` when they must remove
an engine-local prefix/radix hit while retaining LMCache data.
