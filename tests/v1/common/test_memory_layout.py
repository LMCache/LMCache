# SPDX-License-Identifier: Apache-2.0
"""Legacy path and dependencies of ``lmcache.v1.common.memory_layout``."""

# Standard
import json
import subprocess
import sys
import textwrap

# First Party
from lmcache.v1.common import memory_layout
from lmcache.v1.distributed import api as plugin_path

_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    import torch

    from lmcache.v1.common.memory_layout import MemoryLayoutDesc

    MemoryLayoutDesc([torch.Size([2, 4])], [torch.float16])
    loaded = [m for m in sys.modules if m.startswith("lmcache.v1.distributed")]
    print(json.dumps(sorted(loaded)))
    """
)


def test_plugin_path_exports_the_same_class() -> None:
    assert plugin_path.MemoryLayoutDesc is memory_layout.MemoryLayoutDesc


def test_import_does_not_load_the_distributed_package() -> None:
    # A fresh interpreter: this session has already imported lmcache.v1.distributed.
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
