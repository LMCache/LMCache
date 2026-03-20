# `lmcache server` CLI Command

**Status:** Proposal  |  **Date:** 2026-03-19

## Goal

Add `lmcache server` as a CLI subcommand that replaces the standalone
`lmcache_server` and `python3 -m lmcache.v1.multiprocess.http_server` entry
points. Users get one unified command to start the LMCache cache server.

## Current State

Today there are three separate entry points in `pyproject.toml`:

| Entry point | Module | Protocol |
|---|---|---|
| `lmcache_v0_server` | `lmcache.server.__main__:main` | Raw TCP (legacy v0) |
| `lmcache_server` | `lmcache.v1.server.__main__:main` | Raw TCP (v1) |
| *(none — module path)* | `lmcache.v1.multiprocess.http_server` | ZMQ + HTTP |

The multiprocess HTTP server is the primary production entry point but requires
`python3 -m lmcache.v1.multiprocess.http_server ...` to launch.

## Design

### Usage

```bash
# Full HTTP + ZMQ server (default)
lmcache server \
    --engine-type blend --host 0.0.0.0 --port 5555 \
    --l1-size-gb 60 --eviction-policy LRU

# ZMQ-only (no HTTP frontend)
lmcache server \
    --no-http \
    --host 0.0.0.0 --port 5555 \
    --l1-size-gb 60 --eviction-policy LRU
```

### Command Behavior

`lmcache server` wraps the existing `lmcache.v1.multiprocess.http_server`
startup flow:

1. Parse arguments (reusing existing `add_*_args()` helpers)
2. Convert to config objects (`MPServerConfig`, `HTTPFrontendConfig`,
   `StorageManagerConfig`, `PrometheusConfig`, `TelemetryConfig`)
3. Call `run_http_server()` (or `run_cache_server()` if `--no-http`)
4. Run in foreground; Ctrl-C to stop

The command is a **thin wrapper** — it delegates entirely to the existing
multiprocess server machinery. No server logic is duplicated.

### Arguments

Arguments are added by calling the existing helper functions directly on the
subcommand's parser:

| Helper | Arguments added |
|---|---|
| `add_mp_server_args(parser)` | `--host`, `--port`, `--chunk-size`, `--max-workers`, `--hash-algorithm`, `--engine-type` |
| `add_storage_manager_args(parser)` | `--l1-size-gb`, `--l1-use-lazy`, `--l1-init-size-gb`, `--l1-align-bytes`, `--l1-write-ttl-seconds`, `--l1-read-ttl-seconds`, `--eviction-policy`, `--eviction-trigger-watermark`, `--eviction-ratio`, `--l2-store-policy`, `--l2-prefetch-policy`, L2 adapter args |
| `add_http_frontend_args(parser)` | `--http-host`, `--http-port` |
| `add_prometheus_args(parser)` | `--disable-prometheus`, `--prometheus-port`, `--prometheus-log-interval` |
| `add_telemetry_args(parser)` | `--enable-telemetry`, `--telemetry-max-queue-size`, `--telemetry-processor` |

One new argument is added by the command itself:

- `--no-http` (flag, default: False) — Disable the HTTP frontend and run
  ZMQ-only via `run_cache_server()` instead of `run_http_server()`.

### Implementation

#### File: `lmcache/cli/commands/server.py`

```python
class ServerCommand(BaseCommand):
    def name(self) -> str:
        return "server"

    def help(self) -> str:
        return "Start the LMCache cache server."

    def add_arguments(self, parser):
        add_mp_server_args(parser)
        add_storage_manager_args(parser)
        add_http_frontend_args(parser)
        add_prometheus_args(parser)
        add_telemetry_args(parser)
        parser.add_argument(
            "--no-http", action="store_true", default=False,
            help="Disable HTTP frontend (ZMQ-only mode).",
        )

    def execute(self, args):
        mp_config = parse_args_to_mp_server_config(args)
        storage_config = parse_args_to_config(args)
        prometheus_config = parse_args_to_prometheus_config(args)
        telemetry_config = parse_args_to_telemetry_config(args)

        if args.no_http:
            run_cache_server(mp_config, storage_config,
                             prometheus_config, telemetry_config)
        else:
            http_config = parse_args_to_http_frontend_config(args)
            run_http_server(http_config, mp_config, storage_config,
                            prometheus_config, telemetry_config)
```

#### Registration

In `lmcache/cli/commands/__init__.py`:

```python
from lmcache.cli.commands.server import ServerCommand

ALL_COMMANDS: list[BaseCommand] = [
    MockCommand(),
    ServerCommand(),
]
```

### BaseCommand Considerations

The `server` command is a long-running process, not a query-and-report
command. It does **not** use the metrics system (`create_metrics` / `emit`).
This is fine — `BaseCommand.execute()` has no requirement to use metrics.

The `--format` and `--output` flags added automatically by
`BaseCommand.register()` are harmless but unused. If this becomes confusing
for users, we can override `register()` in `ServerCommand` to skip calling
`_add_output_args()`. This is a minor polish item, not blocking.

### Signal Handling

The existing `run_http_server()` / `run_cache_server()` functions already
handle graceful shutdown on SIGINT/SIGTERM. The CLI dispatcher in `main.py`
also catches `KeyboardInterrupt` and exits with code 130. No additional
signal handling is needed.

### Deprecation of Old Entry Points

Per the [commands.md](commands.md) design doc, `lmcache_server` is kept as
a deprecated alias for 2 minor releases. We add a deprecation warning to
its `main()`:

```python
import warnings
warnings.warn(
    "lmcache_server is deprecated. Use 'lmcache server' instead.",
    DeprecationWarning, stacklevel=2,
)
```

The `lmcache_v0_server` entry point is legacy (v0 protocol) and out of
scope for this change.

## Testing

- **Unit test:** Verify `ServerCommand` registers correctly and parses all
  expected arguments (mock `run_http_server` / `run_cache_server`).
- **Integration test:** Start `lmcache server` in a subprocess, verify the
  ZMQ and HTTP ports become reachable, send Ctrl-C and verify clean exit.
- **`--no-http` test:** Start with `--no-http`, verify only ZMQ port is open.

## Non-Goals

- No changes to the server internals (ZMQ, HTTP, cache engine).
- No configuration file support in this phase (future work).
- No documentation changes yet (per user request).
