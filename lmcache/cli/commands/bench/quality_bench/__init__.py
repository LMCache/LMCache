# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench quality`` command group.

Measures whether CacheBlend changes the model's answers, in two stages:

* ``run`` sends byte-identical prompts to one or more arms (a full
  recompute reference, optionally vLLM prefix caching, and CacheBlend) and
  saves their answers to a results directory.
* ``analyze`` scores the saved answers and reports per-question rows and a
  verdict.

Arms can be run one at a time into the same directory, so they never need
to be served at once. The subcommands are auto-discovered from the
``*_command`` modules in this package; the runtime lives in ``core`` so that
discovery imports only these lightweight wrappers.
"""

# First Party
from lmcache.cli.commands.base import CompositeCommand


class QualityBenchCommand(CompositeCommand):
    """Measure CacheBlend answer quality against full recompute."""

    def name(self) -> str:
        return "quality"

    def help(self) -> str:
        return "Measure CacheBlend answer quality against full recompute."
