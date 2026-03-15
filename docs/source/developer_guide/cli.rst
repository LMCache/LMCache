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
   ├── metrics.py           # Metrics class + rendering styles
   └── commands/
       ├── __init__.py      # ALL_COMMANDS registry
       ├── base.py          # BaseCommand ABC, add_output_arg()
       └── mock.py          # Example command


Step-by-Step: Adding a New Command
-----------------------------------

**Step 1.** Create ``lmcache/cli/commands/describe.py``:

.. code-block:: python

   # SPDX-License-Identifier: Apache-2.0
   import argparse

   from lmcache.cli.commands.base import BaseCommand
   from lmcache.cli.metrics import Metrics

   class DescribeCommand(BaseCommand):

       def name(self) -> str:
           return "describe"

       def help(self) -> str:
           return "Describe a running KV cache server."

       def add_arguments(self, parser: argparse.ArgumentParser) -> None:
           parser.add_argument("--url", required=True,
                               help="LMCache HTTP server URL (e.g. http://localhost:8000)")

       def handler(self, args: argparse.Namespace) -> None:
           # Connect to server, gather info...
           metrics = Metrics(title="Describe KV Cache")
           metrics.add("status", "Status", "OK")
           metrics.add("chunks", "Cached chunks", 1024)
           metrics.print()

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

The ``Metrics`` class provides hierarchical metrics with separate machine keys
(for JSON) and human-readable labels (for terminal display).

.. code-block:: python

   from lmcache.cli.metrics import Metrics

   metrics = Metrics(title="Bench KV Cache Result")

   # Create named sections
   metrics.create_section("ops", "Operations (ops/s)")
   metrics["ops"].add("store", "Store", 41.3)
   metrics["ops"].add("retrieve", "Retrieve", 127.3)

   # Top-level metrics (no section header)
   metrics.add("status", "Status", "OK")

   # Change title after construction
   metrics.title("Updated Title")

   # Output
   metrics.print()                   # terminal (human labels)
   metrics.to_json("result.json")    # file (machine keys)
   data = metrics.to_dict()          # dict (machine keys)

The ``--output`` flag is available via the shared helper:

.. code-block:: python

   from lmcache.cli.commands.base import add_output_arg

   def add_arguments(self, parser):
       # ... your args ...
       add_output_arg(parser)

   def handler(self, args):
       metrics = Metrics(title="My Result")
       # ... populate metrics ...
       metrics.print()
       if args.output:
           metrics.to_json(args.output)
