# SPDX-License-Identifier: Apache-2.0
"""Exercise auto-start health checks through the real ZMQ wire protocol."""

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
        import zmq
        from lmcache.integration.vllm.mp_server_launcher import is_mp_server_healthy
        from lmcache.v1.multiprocess.mq import MessageQueueServer
        from lmcache.v1.multiprocess.protocol import (
            RequestType, get_handler_type, get_payload_classes,
        )

        def ping(instance_id: int | None) -> bool:
            assert instance_id is None
            return True

        enabled = sys.argv[1] == 'True'
        context = zmq.Context()
        endpoint = 'inproc://autostart-health-regression'
        server = None
        try:
            if enabled:
                server = MessageQueueServer(endpoint, context)
                server.add_handler(
                    RequestType.PING, get_payload_classes(RequestType.PING),
                    get_handler_type(RequestType.PING), ping,
                )
                server.add_normal_thread_pool([RequestType.PING], max_workers=1)
                server.start()
            assert is_mp_server_healthy(endpoint, context, timeout=0.5) is enabled
        finally:
            if server is not None:
                server.close()
            context.term()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(start_server)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
