# SPDX-License-Identifier: Apache-2.0
"""Exercise optional Mooncake loading through the public context lifecycle."""

# Standard
from pathlib import Path
import os
import subprocess
import sys
import textwrap

# Third Party
import pytest


def _run_probe(
    tmp_path: Path,
    script: str,
    engine_source: str = "",
    package_source: str = "",
) -> None:
    """Use a fresh interpreter so prior imports cannot hide SDK loading."""
    package = tmp_path / "mooncake"
    package.mkdir()
    (package / "__init__.py").write_text(package_source, encoding="utf-8")
    (package / "engine.py").write_text(textwrap.dedent(engine_source), encoding="utf-8")
    env = os.environ.copy()
    paths = [str(tmp_path), str(Path(__file__).resolve().parents[4])]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
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


def test_discovery_does_not_load_mooncake_sdk(tmp_path: Path) -> None:
    """Discover factories without executing an installed SDK's module code."""
    _run_probe(
        tmp_path,
        """
        import sys
        from lmcache.v1.distributed.transfer_channel import (
            initialize_transfer_channel_context,
        )

        assert callable(initialize_transfer_channel_context)
        assert "mooncake" not in sys.modules
        assert "mooncake.engine" not in sys.modules
        """,
        engine_source='raise AssertionError("Unselected SDK was imported")',
    )


def test_selected_mooncake_loads_sdk_and_registers_memory(tmp_path: Path) -> None:
    """Select the real factory with an SDK double and close via the public API."""
    _run_probe(
        tmp_path,
        """
        import os
        import sys
        import zmq
        from lmcache.v1.distributed.internal_api import L1MemoryDesc
        from lmcache.v1.distributed.transfer_channel import (
            delete_transfer_channel_context,
            get_transfer_channel_context,
            initialize_transfer_channel_context,
        )

        os.environ["MC_TE_PROTOCOL"] = "tcp"
        os.environ["MC_TE_DEVICE"] = ""
        assert "mooncake.engine" not in sys.modules
        context = initialize_transfer_channel_context(
            "mooncake_te", L1MemoryDesc(ptr=4096, size=4096, align_bytes=256),
            "127.0.0.1:0", "127.0.0.1:0",
        )
        try:
            import mooncake.engine as sdk

            assert get_transfer_channel_context() is context
            assert context.get_num_connected_clients() == 0
            assert isinstance(context.mooncake_te_engine, sdk.TransferEngine)
            assert sdk.calls == [
                "import", "construct",
                ("initialize", "127.0.0.1", "P2PHANDSHAKE", "tcp", ""),
                ("register", 4096, 4096),
            ]
        finally:
            delete_transfer_channel_context()
            zmq.Context.instance().term()
        assert sdk.calls[-1] == ("unregister", 4096)
        """,
        engine_source="""
        calls = ["import"]

        class TransferEngine:
            def __init__(self) -> None:
                calls.append("construct")

            def initialize(self, host, metadata, protocol, device) -> int:
                calls.append(("initialize", host, metadata, protocol, device))
                return 0

            def get_rpc_port(self) -> int:
                return 18591

            def register_memory(self, pointer: int, size: int) -> int:
                calls.append(("register", pointer, size))
                return 0

            def unregister_memory(self, pointer: int) -> int:
                calls.append(("unregister", pointer))
                return 0
        """,
    )


@pytest.mark.parametrize(
    ("package_source", "engine_source", "error_type", "error_message"),
    [
        (
            'raise ModuleNotFoundError("No module named mooncake", name="mooncake")',
            "",
            "ModuleNotFoundError",
            "No module named mooncake",
        ),
        (
            "",
            'raise ModuleNotFoundError("Missing SDK dependency", '
            'name="sdk_dependency")',
            "ModuleNotFoundError",
            "Missing SDK dependency",
        ),
        (
            "",
            'raise ImportError("SDK native library could not be loaded")',
            "ImportError",
            "SDK native library could not be loaded",
        ),
    ],
    ids=["sdk-absent", "transitive-dependency-absent", "native-library-unavailable"],
)
def test_selected_mooncake_reports_import_failure(
    tmp_path: Path,
    package_source: str,
    engine_source: str,
    error_type: str,
    error_message: str,
) -> None:
    """Defer SDK failures until selection and preserve the original exception."""
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
                "mooncake_te", L1MemoryDesc(ptr=0, size=4096, align_bytes=256),
                "127.0.0.1:0", "127.0.0.1:0",
            )
        except ImportError as error:
            assert type(error).__name__ == {error_type!r}
            assert str(error) == {error_message!r}
        else:
            raise AssertionError("Selected backend did not report its SDK failure")
        try:
            get_transfer_channel_context()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Failed SDK import left a global context")
        """,
        engine_source=engine_source,
        package_source=package_source,
    )
