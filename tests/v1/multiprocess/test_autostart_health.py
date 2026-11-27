# SPDX-License-Identifier: Apache-2.0
"""Exercise auto-start health checks through the gRPC request transport."""

# Standard
import subprocess
import sys
import textwrap

# Third Party
import pytest


@pytest.mark.parametrize("start_server", [True, False])
def test_autostart_health_real_transport(start_server: bool) -> None:
    """PING(None) succeeds on a live server and closes promptly on timeout.

    A subprocess deadline also catches regressions that hang client.close().
    No fake client or protocol serializer is substituted.
    """
    script = textwrap.dedent("""
        import sys
        from lmcache.integration.vllm.mp_server_launcher import is_mp_server_healthy
        from lmcache.v1.multiprocess.config import MPServerConfig
        from lmcache.v1.multiprocess.request_handler import request_handler
        from lmcache.v1.multiprocess.transport.grpc_impl import server_builder

        class Health:
            @request_handler()
            def ping(self, instance_id: int | None) -> bool:
                return instance_id is None

        enabled = sys.argv[1] == 'True'
        config = MPServerConfig(host="127.0.0.1", port=0, grpc_server_mode="sync")
        server = server_builder.build_grpc_request_server([Health()], config)
        server.start()
        endpoint = f"grpc://127.0.0.1:{server.bound_port}"
        if not enabled:
            server.close()
        try:
            assert is_mp_server_healthy(endpoint, None, timeout=1) is enabled
        finally:
            if enabled:
                server.close()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(start_server)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
