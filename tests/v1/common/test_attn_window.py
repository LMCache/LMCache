# SPDX-License-Identifier: Apache-2.0
"""Legacy path and dependencies of ``lmcache.v1.common.attn_window``."""

# Standard
import subprocess
import sys
import textwrap

# First Party
from lmcache.v1.common import attn_window
from lmcache.v1.distributed import api as plugin_path


def test_plugin_path_exports_the_same_objects():
    for name in ("AttnWindowDesc", "GroupKind", "DEFAULT_ATTN_WINDOW_DESC"):
        assert getattr(plugin_path, name) is getattr(attn_window, name), name


def test_import_does_not_load_the_distributed_package():
    script = textwrap.dedent(
        """
        import sys

        from lmcache.v1.common.attn_window import AttnWindowDesc

        AttnWindowDesc(num_chunks_in_sw=[-1, 2], group_kinds=("attention", "aux"))
        loaded = [m for m in sys.modules if m.startswith("lmcache.v1.distributed")]
        assert loaded == [], loaded
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
