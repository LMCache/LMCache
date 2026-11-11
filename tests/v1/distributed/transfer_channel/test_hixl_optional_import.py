# SPDX-License-Identifier: Apache-2.0
"""Check HIXL dependency loading through the public factory in fresh processes."""

# Standard
from pathlib import Path
import os
import subprocess
import sys
import textwrap

# Third Party
import pytest


def _run_probe(tmp_path: Path, script: str, package_source: str) -> None:
    package = tmp_path / "lmcache_ascend"
    package.mkdir()
    (package / "__init__.py").write_text(package_source, encoding="utf-8")
    env = os.environ.copy()
    # Use this checkout even when LMCache has not been installed editable.
    paths = [str(tmp_path), str(Path(__file__).resolve().parents[4])]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["LMCACHE_TRACK_USAGE"] = "false"
    env["LMCACHE_HIXL_ENGINE_ID"] = "127.0.0.1:19100"
    for suffix in ("DEVICE_ID", "CONNECT_TIMEOUT_MS", "COMM_LISTEN_PORT"):
        env.pop("LMCACHE_HIXL_" + suffix, None)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_discovery_does_not_load_ascend_dependencies(tmp_path: Path) -> None:
    _run_probe(
        tmp_path,
        """
        import sys
        from lmcache.v1.distributed.transfer_channel import (
            initialize_transfer_channel_context,
        )
        assert callable(initialize_transfer_channel_context)
        assert "lmcache_ascend" not in sys.modules
        assert "lmcache_ascend.hixl_npu_comms" not in sys.modules
        assert "lmcache_ascend.v1.hixl_runtime" not in sys.modules
        """,
        'raise AssertionError("Unselected Ascend dependency was imported")',
    )


@pytest.mark.parametrize("error_type", ["ModuleNotFoundError", "OSError"])
def test_selected_hixl_reports_dependency_failure(
    tmp_path: Path,
    error_type: str,
) -> None:
    _run_probe(
        tmp_path,
        f"""
        from lmcache.v1.distributed.internal_api import L1MemoryDesc
        from lmcache.v1.distributed.transfer_channel import (
            get_transfer_channel_context,
            initialize_transfer_channel_context,
        )
        try:
            initialize_transfer_channel_context(
                "hixl", L1MemoryDesc(ptr=4096, size=4096, align_bytes=256),
                "127.0.0.1:19000", "127.0.0.1:19000",
            )
        except RuntimeError as error:
            assert "LMCache-Ascend" in str(error)
            assert isinstance(error.__cause__, {error_type})
        else:
            raise AssertionError("Selected HIXL did not report dependency failure")
        try:
            get_transfer_channel_context()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Failed setup left a global Context")
        """,
        f'raise {error_type}("Ascend dependency unavailable")',
    )
