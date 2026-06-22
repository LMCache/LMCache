Extending the CLI
=================

This guide explains how to add new subcommands to the ``lmcache`` CLI.

.. note::

   For the full extension guide -- including N-level nested subcommands
   via ``CompositeCommand`` and the auto-discovery rules -- see
   :doc:`/extension/cli`.

Architecture Overview
---------------------

The CLI uses **auto-discovery** of subcommand classes:

1. Each command is a class inheriting from ``BaseCommand``.
2. Commands live in their own module (or sub-package) under
   ``lmcache/cli/commands/``. They are picked up at startup by
   ``discover_subclasses()`` (in
   ``lmcache/v1/utils/subclass_discovery.py``) and exposed via
   ``ALL_COMMANDS`` in ``commands/__init__.py``.
3. The entry point iterates ``ALL_COMMANDS`` and calls
   ``cmd.register(subparsers)`` to wire up argparse.

No manual registration is required -- creating the module is enough.
Helper modules whose names start with an underscore (e.g.
``_helpers.py``) are excluded from discovery.

``BaseCommand`` is an abstract class with a small set of required methods
(name, help, argument registration, and execute). Forgetting any of them
raises ``TypeError`` at instantiation time. Commands that only group
sub-subcommands inherit from ``CompositeCommand`` instead -- see
:doc:`/extension/cli` for that pattern.

Step-by-Step: Adding a New Command
-----------------------------------

**Step 1.** Create a new module under ``lmcache/cli/commands/`` and
define a class that subclasses ``BaseCommand``:

.. code-block:: python

   # SPDX-License-Identifier: Apache-2.0
   # lmcache/cli/commands/describe.py
   import argparse

   from lmcache.cli.commands.base import BaseCommand

   class DescribeCommand(BaseCommand):

       def name(self) -> str:
           return "describe"

       def help(self) -> str:
           return "Describe a running KV cache server."

       def add_arguments(self, parser: argparse.ArgumentParser) -> None:
           parser.add_argument("--url", required=True,
                               help="LMCache HTTP server URL (e.g. http://localhost:8000)")

       def execute(self, args: argparse.Namespace) -> None:
           # Connect to server, gather info...
           metrics = self.create_metrics("Describe KV Cache", args)
           metrics.add("status", "Status", "OK")
           metrics.add("chunks", "Cached chunks", 1024)
           metrics.emit()

**Step 2.** That's it -- the command is discovered automatically. No
edits to ``commands/__init__.py`` are required.
``lmcache describe --url http://localhost:8000`` is now available.

For nested subcommands (e.g. ``lmcache tool cache-simulator simulate``),
follow the ``CompositeCommand`` pattern documented in
:doc:`/extension/cli`.

Using the Metrics System
------------------------

The metrics system uses a **handler + formatter** architecture:

- **Metrics** — the collector. Holds sections and entries.
- **Handler** — the destination (stdout, file, etc.).
- **Formatter** — the rendering (ASCII table, JSON, etc.).

``BaseCommand.create_metrics()`` sets up default handlers automatically, so
command authors just build metrics and call ``emit()``:

.. code-block:: python

   def execute(self, args: argparse.Namespace) -> None:
       # create_metrics() auto-registers:
       #   - StreamHandler → stdout (formatter chosen by --format, default: terminal)
       #   - FileHandler   → if --output is set (same format as --format)
       metrics = self.create_metrics("Bench KV Cache Result", args)

       # Create named sections
       metrics.add_section("ops", "Operations (ops/s)")
       metrics["ops"].add("store", "Store", 41.3)
       metrics["ops"].add("retrieve", "Retrieve", 127.3)

       # Top-level metrics (no section header)
       metrics.add("status", "Status", "OK")

       # Trigger all handlers
       metrics.emit()

The ``--format`` and ``--output`` flags are added automatically by
``BaseCommand.register()`` — subcommands do not need to add them manually.
