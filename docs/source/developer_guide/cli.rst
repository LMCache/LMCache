Extending the CLI
=================

This guide explains how to add new subcommands to the ``lmcache`` CLI.

Architecture Overview
---------------------

The CLI uses explicit command registration:

1. Each command is a class inheriting from ``BaseCommand`` in
   ``lmcache/cli/commands/base.py``.
2. Commands are instantiated and listed in ``ALL_COMMANDS`` in
   ``lmcache/cli/commands/__init__.py``.
3. At startup, ``main.py`` iterates ``ALL_COMMANDS`` and calls
   ``cmd.register(subparsers)`` to wire up argparse.

``BaseCommand`` is an abstract class with four required methods. Forgetting any
of them raises ``TypeError`` at instantiation time.

File Layout
-----------

.. code-block:: text

   lmcache/cli/
   ├── __init__.py
   ├── main.py              # Entry point
   ├── config.py            # CLIConfig (env var configuration)
   ├── metrics/             # Metrics system
   │   ├── __init__.py      # Re-exports
   │   ├── metrics.py       # Metrics collector
   │   ├── section.py       # Section data class
   │   ├── handler.py       # StreamHandler, FileHandler
   │   └── formatter.py     # VllmFormatter, JsonFormatter
   └── commands/
       ├── __init__.py      # ALL_COMMANDS registry
       ├── base.py          # BaseCommand ABC, add_output_args()
       └── mock.py          # Example command


Step-by-Step: Adding a New Command
-----------------------------------

**Step 1.** Create ``lmcache/cli/commands/describe.py``:

.. code-block:: python

   # SPDX-License-Identifier: Apache-2.0
   import argparse

   from lmcache.cli.commands.base import BaseCommand, add_output_args

   class DescribeCommand(BaseCommand):

       def name(self) -> str:
           return "describe"

       def help(self) -> str:
           return "Describe a running KV cache server."

       def add_arguments(self, parser: argparse.ArgumentParser) -> None:
           parser.add_argument("--url", required=True,
                               help="LMCache HTTP server URL (e.g. http://localhost:8000)")
           add_output_args(parser)

       def handler(self, args: argparse.Namespace) -> None:
           # Connect to server, gather info...
           metrics = self.create_metrics("Describe KV Cache", args)
           metrics.add("status", "Status", "OK")
           metrics.add("chunks", "Cached chunks", 1024)
           metrics.emit()

**Step 2.** Register it in ``lmcache/cli/commands/__init__.py``:

.. code-block:: python

   from lmcache.cli.commands.describe import DescribeCommand

   ALL_COMMANDS: list[BaseCommand] = [
       MockCommand(),
       DescribeCommand(),   # add here
   ]

That's it --- ``lmcache describe --url http://localhost:8000`` is now available.


Using the Metrics System
------------------------

The metrics system uses a **handler + formatter** architecture:

- **Metrics** — the collector. Holds sections and entries.
- **Handler** — the destination (stdout, file, etc.).
- **Formatter** — the rendering (ASCII table, JSON, etc.).

``BaseCommand.create_metrics()`` sets up default handlers automatically, so
command authors just build metrics and call ``emit()``:

.. code-block:: python

   def handler(self, args: argparse.Namespace) -> None:
       # create_metrics() auto-registers:
       #   - StreamHandler → stdout (formatter chosen by --format, default: vllm)
       #   - FileHandler   → if --output is set (always JSON)
       metrics = self.create_metrics("Bench KV Cache Result", args)

       # Create named sections
       metrics.add_section("ops", "Operations (ops/s)")
       metrics["ops"].add("store", "Store", 41.3)
       metrics["ops"].add("retrieve", "Retrieve", 127.3)

       # Top-level metrics (no section header)
       metrics.add("status", "Status", "OK")

       # Trigger all handlers
       metrics.emit()

The ``--format`` and ``--output`` flags are available via the shared helper:

.. code-block:: python

   from lmcache.cli.commands.base import add_output_args

   def add_arguments(self, parser):
       # ... your args ...
       add_output_args(parser)
       # Adds:
       #   --format FORMAT  Stdout format (default: vllm). Available: vllm, json.
       #   --output PATH    Save metrics to a JSON file at PATH.
