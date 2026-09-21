# Multiprocess integration tests

This suite validates an inference engine connected to a standalone LMCache
multiprocess server. Tests run directly in a Buildkite Kubernetes pod; they do
not use Docker Compose.

## Execution flow

Each Buildkite step runs one test so its model, GPU count, timeout, and retry
policy can be configured independently:

```text
pipeline.yml
  -> run.sh <test-name>
     -> engines/<engine>.sh + workload-discovery.sh: preflight capabilities
     -> engines/<engine>.sh: install and configure the engine
     -> scripts/run-single-test.sh
        -> resolve workloads/common or workloads/<engine>
        -> scripts/launch-processes.sh
           -> start LMCache
           -> start the engine through its adapter
           -> optionally start a baseline engine
        -> scripts/wait-for-servers.sh
           -> probe adapter-provided readiness URLs
        -> scripts/workloads/{common|<engine>}/*.sh
        -> scripts/cleanup.sh
```

`INFERENCE_ENGINE` selects the adapter and defaults to `vllm`. For example:

```bash
INFERENCE_ENGINE=vllm \
LMCACHE_REQUEST_TRANSPORT=zmq \
.buildkite/k3_tests/multiprocess/run.sh long_doc_qa
```

The request transport can be `zmq` or `grpc`. The harness maps these values to
the `tcp://` and `grpc://` connector addresses respectively.

## Directory layout

```text
multiprocess/
├── pipeline.yml                 # Buildkite steps and pod resources
├── run.sh                       # Pod entry point and environment setup
└── scripts/
    ├── engines/
    │   ├── README.md            # Adapter API details
    │   ├── sglang.sh            # SGLang implementation
    │   └── vllm.sh              # vLLM implementation
    ├── workloads/
    │   ├── README.md            # Workload classification
    │   ├── helpers.sh           # Shared workload environment
    │   ├── common/              # Engine-neutral requests and assertions
    │   └── vllm/                # vLLM-specific workloads
    ├── launch-processes.sh      # Shared process launcher
    ├── wait-for-servers.sh      # Readiness checks and diagnostics
    ├── cleanup.sh               # PID, port, log, and result cleanup
    ├── workload-discovery.sh    # Workload resolution and capability preflight
    └── run-single-test.sh       # Per-test configuration and workload discovery
```

Common workloads use `ENGINE_PORT`, `ENGINE_BASELINE_PORT`, `ENGINE_NAME`,
`MODEL`, and OpenAI-compatible HTTP APIs. Code that depends on an engine's
private response fields, endpoints, logs, or command-line flags belongs under
`workloads/<engine>/`.

Some common scenarios need non-default launch settings. They select a named
launch profile, such as `deadlock`; each adapter translates that profile into
its own command-line arguments. This keeps the workload assertions independent
of how the engine process is started.

## Adding an inference engine

### 1. Add the adapter

Create `scripts/engines/<engine>.sh`. The adapter owns installation, defaults,
launch commands, readiness probes, and engine-specific cleanup. A minimal
adapter has this shape:

```bash
#!/usr/bin/env bash

export ENGINE_NAME="ExampleEngine"
export ENGINE_DEFAULT_MODEL="example/model"

ENGINE_SUPPORTED_TRANSFER_MODES=(lmcache_driven)
ENGINE_SUPPORTED_REQUEST_TRANSPORTS=(zmq grpc)

# Use normalized underscore names. Keep this empty when all common workloads
# are supported.
ENGINE_COMMON_WORKLOAD_BLACKLIST=(deadlock)

engine_setup_environment() {
    local repo_root="$1"
    # Install the engine and LMCache integration into the active environment.
}

engine_configure_defaults() {
    export ENGINE_PORT="${ENGINE_PORT:-8000}"
    export ENGINE_BASELINE_PORT="${ENGINE_BASELINE_PORT:-9000}"
    export GPU_FOR_ENGINE="${GPU_FOR_ENGINE:-0}"
    export GPU_FOR_BASELINE="${GPU_FOR_BASELINE:-1}"
    export ENGINE_LOG_FILE="${ENGINE_LOG_FILE:-/tmp/example_engine.log}"
    export ENGINE_BASELINE_LOG_FILE="${ENGINE_BASELINE_LOG_FILE:-/tmp/example_engine_baseline.log}"
}

engine_configure_workload() {
    local test_name="$1"
    if [[ "$test_name" == "long_doc_qa" ]]; then
        export MAX_TTFT_SLOWDOWN_PCT="${MAX_TTFT_SLOWDOWN_PCT:-0}"
    fi
}

engine_prepare_launch() {
    local device_index="$1"
    # Compute launch arguments that depend on the selected device.
}

engine_add_lmcache_server_environment() {
    local environment_name="$1"
    local -n environment="$environment_name"
    # Add environment variables needed by the LMCache server, if any.
}

engine_launch() {
    local mode="$1"       # lmcache or baseline
    local port="$2"
    local device_index="$3"
    local log_file="$4"

    example-engine serve \
        --model "$MODEL" \
        --port "$port" \
        > "$log_file" 2>&1 &
    ENGINE_PID=$!
}

engine_ready_urls() {
    local port="$1"
    printf 'http://127.0.0.1:%s/health\n' "$port"
}

engine_clear_local_cache() {
    local port="$1"
    # Optional: clear only the engine-local prefix/radix cache.
}
```

`engine_launch` must start the process in the background and store its PID in
`ENGINE_PID`. In `lmcache` mode it must enable the engine's LMCache connector
using `LMCACHE_REQUEST_SCHEME`, `LMCACHE_PORT`, and any required engine-side
configuration. In `baseline` mode it must start the same engine without
LMCache.

`ENGINE_SUPPORTED_TRANSFER_MODES` and
`ENGINE_SUPPORTED_REQUEST_TRANSPORTS` describe the matrix combinations the
adapter can run. When either array is omitted, that capability is treated as
unrestricted for compatibility with external adapters. Unsupported
combinations and blacklisted common workloads are skipped before
`engine_setup_environment`, so they do not install packages or start servers.

Use the optional `engine_configure_workload <test-name>` hook for engine-specific
defaults such as performance thresholds. Keep user- or pipeline-provided values
authoritative by assigning defaults with `${VAR:-default}`. A workload that
must distinguish an LMCache retrieval from an engine-local hit can opt in with
`VERIFY_LMCACHE_RETRIEVAL=true`; its adapter implements
`engine_clear_local_cache <port>` without clearing LMCache itself.

The SGLang adapter currently exercises the unified radix-cache integration in
`lmcache_driven` mode. Its CI coverage reuses the common long-document,
accuracy, and HTTP API workloads. Server restart recovery,
`engine_driven`, and special deadlock launch profiles remain excluded until the
unified connector implements those contracts.

Workload support requires no adapter allowlist. A common workload is available
to every adapter; an engine-specific workload is available when its script
exists under `workloads/<engine>/`. Missing implementations fail before any
server is launched. An adapter can list temporary common-workload exceptions in
`ENGINE_COMMON_WORKLOAD_BLACKLIST`; these workloads are skipped before launch.

### 2. Implement required launch profiles

If the engine supports a common workload with a special topology, implement
`engine_launch_profile` in its adapter. For example:

```bash
engine_launch_profile() {
    local profile="$1"
    local mode="$2"
    local port="$3"
    local device_index="$4"
    local log_file="$5"

    case "$profile" in
        deadlock)
            # Start this engine with the TP=2 and batching settings required by
            # the common deadlock workload, then assign ENGINE_PID.
            ;;
        *)
            echo "Unsupported launch profile: $profile" >&2
            return 1
            ;;
    esac
}
```

The profile contains only engine-specific translation. Models, LMCache server
settings, request generation, and assertions remain in the shared harness or
common workload.

### 3. Add engine-specific workloads when needed

Place private engine coverage in `scripts/workloads/<engine>/`. Use an
engine-specific workload only when the same assertion cannot be expressed via
the shared adapter contract or OpenAI-compatible API. Examples include checking
an engine-specific response extension or resetting its private prefix cache.

Name the script after the test, replacing underscores with hyphens. For
example, `run.sh example_feature` automatically resolves:

```text
scripts/workloads/<engine>/example-feature.sh
```

If the scenario is portable, put the same filename under `workloads/common/`.
No adapter allowlist or dispatch entry is needed. Add configuration in
`run-single-test.sh` only when the test needs a custom model, launch profile,
baseline server, or LMCache setting. Add an alias to
`workload-discovery.sh` only when multiple test names intentionally share one
workload script.

### 4. Add Buildkite steps

Add the engine to `x-common-inference-engines`. Shared steps expand across the
engine matrix, while the adapter's capability declarations and blacklist remove
unsupported combinations at preflight. Do not add duplicate shared-workload
steps for engine-specific defaults; use `engine_configure_workload` instead.
Add separate steps only for engine-private workloads and choose pod resources
for each test:

```yaml
- label: ":test_tube: {{matrix.inference_engine}} / example_feature / {{matrix.request_transport}}"
  command: .buildkite/k3_tests/multiprocess/run.sh example_feature
  matrix:
    setup:
      inference_engine:
        - example
      request_transport:
        - zmq
  env:
    INFERENCE_ENGINE: "{{matrix.inference_engine}}"
    LMCACHE_REQUEST_TRANSPORT: "{{matrix.request_transport}}"
  agents: { queue: "k8s" }
```

Do not run all workloads sequentially in one pod merely because they share an
engine. Separate steps preserve test isolation and allow the deadlock, TP,
baseline, and large-model cases to request different resources.

## Validation

At minimum, validate a new adapter with both transports and one common
workload:

```bash
bash -n .buildkite/k3_tests/multiprocess/scripts/engines/<engine>.sh

INFERENCE_ENGINE=<engine> LMCACHE_REQUEST_TRANSPORT=zmq \
    .buildkite/k3_tests/multiprocess/run.sh high_concurrency

INFERENCE_ENGINE=<engine> LMCACHE_REQUEST_TRANSPORT=grpc \
    .buildkite/k3_tests/multiprocess/run.sh high_concurrency
```

Also verify that missing workload implementations fail before starting LMCache
or the engine, and inspect the collected `build_*.log` files when startup fails.
