# SPDX-License-Identifier: Apache-2.0
"""Legacy path and dependencies of ``memory_layout`` and ``sliding_window``."""

# Standard
import subprocess
import sys
import textwrap

# First Party
from lmcache.v1.common import memory_layout
from lmcache.v1.distributed import api as plugin_path


def test_plugin_path_exports_the_same_class() -> None:
    assert plugin_path.MemoryLayoutDesc is memory_layout.MemoryLayoutDesc


def test_import_adds_no_endpoint_modules() -> None:
    # `import lmcache` itself loads lmcache.v1.multiprocess.transfer_mode.
    script = textwrap.dedent(
        """
        import sys

        import torch

        import lmcache

        before = set(sys.modules)
        from lmcache.v1.common.memory_layout import MemoryLayoutDesc
        from lmcache.v1.common.sliding_window import recalculate_blocks_to_skip

        MemoryLayoutDesc([torch.Size([2, 4])], [torch.float16])
        recalculate_blocks_to_skip(4, 2, 7)
        endpoints = (
            "lmcache.v1.client",
            "lmcache.v1.distributed",
            "lmcache.v1.multiprocess",
        )
        added = sorted(m for m in set(sys.modules) - before if m.startswith(endpoints))
        assert added == [], added
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
