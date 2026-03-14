# LMCache CLI Framework & Metrics System Design

**Status:** Proposal  |  **Date:** 2026-03-14

## Scope

This document covers the **CLI framework** (pluggable command discovery) and the
**hierarchical metrics logging system**. It is the implementation plan for Phase 1
of the [CLI design](commands.md), minus the actual server/ping/describe commands
(those come later). A `lmcache mock` command is included as a working example.

---

## 1. Explicit Command Registration

### Goal

Adding a new subcommand (e.g., `lmcache describe`) requires:

1. Creating a new file in `lmcache/cli/commands/` with a `BaseCommand` subclass.
2. Adding one import + one entry to `ALL_COMMANDS` in `commands/__init__.py`.

### Mechanism

```python
# lmcache/cli/commands/my_cmd.py
from lmcache.cli.commands.base import BaseCommand

class MyCommand(BaseCommand):
    def name(self) -> str:
        return "my-cmd"

    def help(self) -> str:
        return "Short help text."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--flag", ...)

    def handler(self, args) -> None:
        ...  # command logic
```

```python
# lmcache/cli/commands/__init__.py  (add import + list entry)
from lmcache.cli.commands.my_cmd import MyCommand

ALL_COMMANDS: list[BaseCommand] = [
    ...,
    MyCommand(),
]
```

`BaseCommand` enforces that all four abstract methods (`name`, `help`,
`add_arguments`, `handler`) are implemented — instantiation fails otherwise.
The concrete `register()` method (inherited, not typically overridden) wires
everything up automatically.

### How command discovery works

1. `lmcache <cmd> ...` invokes `main()` in `main.py`.
2. `main.py` imports `ALL_COMMANDS` from `commands/__init__.py`.
3. At import time, `__init__.py` imports each command class and instantiates
   it into the `ALL_COMMANDS` list.  Instantiation validates that all abstract
   methods are implemented (`TypeError` on failure).
4. `main.py` iterates `ALL_COMMANDS` and calls `cmd.register(subparsers)`.
5. `BaseCommand.register()` creates an argparse subparser (using `name()` and
   `help()`), calls `add_arguments()` to wire up flags, and sets
   `parser.set_defaults(func=self.handler)`.
6. After parsing, `main.py` dispatches via `args.func(args)`, which calls the
   matched command's `handler()`.

### How to add a new subcommand

**Step 1.** Create `lmcache/cli/commands/describe.py`:

```python
from lmcache.cli.commands.base import BaseCommand

class DescribeCommand(BaseCommand):
    def name(self) -> str:
        return "describe"

    def help(self) -> str:
        return "Describe a running KV cache server."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--url", required=True)

    def handler(self, args) -> None:
        ...  # implementation
```

**Step 2.** Register it in `lmcache/cli/commands/__init__.py`:

```python
from lmcache.cli.commands.describe import DescribeCommand

ALL_COMMANDS: list[BaseCommand] = [
    MockCommand(),
    DescribeCommand(),   # <-- add here
]
```

That's it — `lmcache describe --url ...` is now available.

### File layout

```
lmcache/cli/
├── __init__.py          # empty
├── main.py              # main() entry point
├── metrics.py           # Metrics (Section 2)
├── commands/
│   ├── __init__.py      # ALL_COMMANDS registry
│   ├── base.py          # BaseCommand ABC, add_output_arg()
│   └── mock.py          # lmcache mock  (example command)
├── config.py            # CLIConfig (centralized config system)
└── corpora/             # built-in prompt corpora (future)
```

### Entry point (pyproject.toml)

```toml
[project.scripts]
lmcache = "lmcache.cli.main:main"
```

---

## 2. Hierarchical Metrics Logger

### Goal

A lightweight, dependency-free metrics collector that:

1. Accepts metrics organized into **sections** (categories).
2. Renders a fixed-width ASCII table to the terminal — matching the style used
   throughout the CLI design doc and `vllm bench serve` output.
3. Serializes the same data to JSON for file output and programmatic consumption.

### API

Each metric has a **machine key** (used in JSON output) and a **human-readable
label** (used in terminal output). Sections work the same way.

```python
from lmcache.cli.metrics import Metrics

metrics = Metrics(title="Bench KV Cache Result (30s)")

# Title can be changed after construction
metrics.title("Bench KV Cache Result (60s)")

# Create named sections (machine key + display label)
metrics.create_section("ops", "Operations (ops/s)")
metrics.create_section("hit_rate", "Hit Rate")
metrics.create_section("correctness", "Correctness")

# Add metrics to sections via dict-like access
metrics["ops"].add("store", "Store", 41.3)
metrics["ops"].add("retrieve", "Retrieve", 127.3)
metrics["hit_rate"].add("l1", "L1", "92.3%")
metrics["correctness"].add("checksums", "Checksums", "5060/5060 OK")

# Terminal output (uses human-readable labels)
metrics.print()

# JSON output (uses machine keys)
metrics.to_json("result.json")
# Or get dict directly
data: dict = metrics.to_dict()
```

### Terminal output format

```
========= Bench KV Cache Result (30s) =========
--------------Operations (ops/s)----------------
Store:                                   41.3
Retrieve:                                127.3
-----------------Hit Rate-----------------------
L1:                                      92.3%
--------------Correctness-----------------------
Checksums:                               5060/5060 OK
================================================
```

Design choices:
- **Fixed total width** of 48 characters (configurable via `width` param).
- Title row is centered within `=` borders.
- Section headers are centered within `-` borders.
- Key-value lines are left-aligned label, right-aligned value.
- Values are formatted automatically: floats get 2 decimal places, strings are
  printed as-is, `None` is printed as `N/A`.

### JSON output format

JSON uses machine keys, not display labels:

```json
{
  "title": "Bench KV Cache Result (30s)",
  "metrics": {
    "ops": {
      "store": 41.3,
      "retrieve": 127.3
    },
    "hit_rate": {
      "l1": "92.3%"
    },
    "correctness": {
      "checksums": "5060/5060 OK"
    }
  }
}
```

### Flat metrics (no section)

For top-level metrics that don't belong to a section, use `metrics.add()`
directly:

```python
metrics = Metrics(title="Ping KV Cache")
metrics.add("status", "Status", "OK")
metrics.add("rtt_ms", "Round trip time (ms)", 0.42)
```

Produces:

```
======= Ping KV Cache =======
Status:                  OK
Round trip time (ms):    0.42
==============================
```

These go into a default unnamed section — no header line is rendered, and in
JSON the entries appear at the top level of `"metrics"`.

### Configurable output style

The terminal rendering style is **pluggable**. The default is `vllm` (matching
`vllm bench serve`). Override via:

- Constructor: `Metrics(title="...", style="rich_panel")`
- Environment variable: `LMCACHE_CLI_METRICS_STYLE=rich_panel` (uses the
  centralized config system in `config.py` with `LMCACHE_CLI_` prefix)
- Constructor takes precedence over env var.

Supported styles (initial):
- `vllm` — `=`/`-` dividers, plain ASCII (default)
- Future styles can be added by subclassing `MetricsStyle` in `metrics.py`.

### Implementation notes

- `Metrics` holds an ordered list of `Section` objects. Each `Section` stores
  a machine key, a display label, and a list of `(key, label, value)` entries.
- `metrics["name"]` returns the `Section` with that machine key. `KeyError`
  if `create_section()` was not called first.
- `metrics.add(key, label, value)` appends to a default unnamed section
  (created implicitly on first use).
- `print()` writes to `sys.stdout` by default; accepts an optional `file` param.
- `to_dict()` returns `{"title": ..., "metrics": ...}` with sections as nested
  dicts keyed by machine key. The unnamed section's entries are placed at the
  top level of `"metrics"`.
- `to_json(path)` calls `to_dict()` → `json.dump()`.
- Rendering is delegated to a `MetricsStyle` object (strategy pattern). Each
  style implements `render(title, sections, width) -> str`. The renderer
  receives display labels, not machine keys.
- No external dependencies beyond the Python standard library.

---

## 3. `lmcache mock` — Example Command

A mock command that demonstrates the full framework: argument parsing, metrics
logging, and both terminal and JSON output. It doesn't connect to any server.

```bash
$ lmcache mock --name test-run --num-items 5

============ Mock Result ============
----------- Input Parameters --------
Name:                        test-run
Num items:                          5
------------ Mock Metrics -----------
Items processed:                    5
Total time (ms):                12.34
Throughput (items/s):          405.19
------------- Validation ------------
Status:                            OK
=====================================

# JSON output uses machine keys
$ lmcache mock --name test-run --num-items 5 --output result.json
(same terminal output)
Metrics saved to result.json
# result.json → {"title": "Mock Result", "metrics": {"input": {"name": "test-run", ...}, ...}}
```

This command lives in `lmcache/cli/commands/mock.py` and serves as a reference
implementation for future commands.

---

## 4. Shared CLI Conventions

### `--output` flag

All commands support `--output <path>` to save metrics as JSON. Added via a shared
helper `add_output_arg(parser)` in `base.py`.

### `--url` flag

Each subcommand configures its own `--url` flag as needed (ZMQ vs HTTP
semantics vary per command).

### Error handling

Commands print errors to stderr and return exit code 1. The dispatcher catches
exceptions from `args.func(args)` and prints a clean error message.
