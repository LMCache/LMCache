# SPDX-License-Identifier: Apache-2.0
"""Import boundaries between shared transfer support and in-process connectors."""

# Standard
import subprocess
import sys
import textwrap


def test_shared_mp_imports_do_not_load_inprocess_connectors() -> None:
    """Shared layout and MP transfer imports work without the old engine package."""
    script = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import sys

        class RejectInprocess(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "lmcache.inprocess" or fullname.startswith(
                    "lmcache.inprocess."
                ):
                    raise AssertionError(f"Shared import loaded {fullname}")
                return None

        sys.meta_path.insert(0, RejectInprocess())
        for name in (
            "lmcache.v1.gpu_connector",
            "lmcache.v1.gpu_connector.kv_format",
            "lmcache.v1.gpu_connector.utils",
            "lmcache.v1.gpu_connector.gpu_ops",
            "lmcache.v1.gpu_connector.gds_context",
            "lmcache.v1.platform.base.cache_context",
            "lmcache.v1.multiprocess.object_group_transfer",
        ):
            importlib.import_module(name)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
