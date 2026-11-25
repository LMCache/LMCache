# SPDX-License-Identifier: Apache-2.0
"""Peer control and shared transports work without the worker client package."""

# Standard
import subprocess
import sys
import textwrap

# Third Party
import pytest

# First Party
from tests.import_utils import lmcache_subprocess_env


@pytest.mark.parametrize(
    "module",
    [
        "lmcache.v1.distributed.p2p",
        "lmcache.v1.distributed.l2_adapters.p2p_l2_adapter",
        "lmcache.v1.transport.grpc.utils",
    ],
)
def test_peer_transport_without_worker_client(module: str) -> None:
    """Import peer APIs with the worker client unavailable, including auth setup."""
    script = textwrap.dedent(
        """
        import importlib
        import sys

        sys.modules["lmcache.v1.client"] = None
        module = importlib.import_module(sys.argv[1])
        if sys.argv[1] == "lmcache.v1.distributed.p2p":
            client = module.P2PClient("grpc://127.0.0.1:1", api_key="peer-test")
            client.close()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, module],
        env=lmcache_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
