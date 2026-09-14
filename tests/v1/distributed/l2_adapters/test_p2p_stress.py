# SPDX-License-Identifier: Apache-2.0
"""Peer-churn regression tests for the P2P L2 adapter over a real MQ."""

# Standard
from unittest.mock import MagicMock, patch
import time

# Third Party
import zmq

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters import p2p_l2_adapter as p2p_mod
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import (
    P2PL2Adapter,
    P2PL2AdapterConfig,
)
from lmcache.v1.multiprocess.config import CoordinatorConfig, P2PConfig
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.mq import MessageQueueServer
from lmcache.v1.multiprocess.protocol import get_payload_classes


def test_peer_disappears_while_lookup_is_pending() -> None:
    """A pending lookup becomes an all-miss result after its peer disappears."""
    lookup_timeout_s = 0.05
    peer_context = MagicMock()
    peer_context.storage_manager.submit_prefetch_task.return_value = MagicMock(
        l1_found_indices=()
    )
    peer_context.storage_manager.query_prefetch_status.return_value = None
    controller = P2PController(
        peer_context,
        P2PConfig(),
        CoordinatorConfig(),
        instance_id="peer",
    )

    zmq_context = zmq.Context.instance()
    mq_server = MessageQueueServer("tcp://127.0.0.1:0", zmq_context)
    peer_mq_url = mq_server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
    handler_specs = controller.get_handlers()
    for spec in handler_specs:
        mq_server.add_blocking_handler(
            spec.request_type,
            get_payload_classes(spec.request_type),
            spec.handler,
        )
    mq_server.add_normal_thread_pool(
        [spec.request_type for spec in handler_specs], max_workers=4
    )
    mq_server.start()
    server_closed = False

    transfer_context = MagicMock()
    transfer_context.get_transfer_channel_client.return_value = MagicMock()
    notifier = MagicMock()

    try:
        with (
            patch.object(
                p2p_mod,
                "get_transfer_channel_context",
                return_value=transfer_context,
            ),
            patch.object(p2p_mod, "PeriodicEventNotifier") as mock_notifier,
        ):
            mock_notifier.get.return_value = notifier
            adapter = P2PL2Adapter(
                P2PL2AdapterConfig(
                    peer_mq_server_url=peer_mq_url,
                    peer_transfer_channel_server_url="unused:0",
                    lookup_timeout_s=lookup_timeout_s,
                )
            )
            try:
                keys = [
                    ObjectKey(
                        chunk_hash=ObjectKey.IntHash2Bytes(1),
                        model_name="test_model",
                        kv_rank=0,
                    )
                ]
                layout = MemoryLayoutDesc(shapes=[], dtypes=[])
                task_id = adapter.submit_lookup_and_lock_task(keys, {0: layout})

                assert adapter.query_lookup_and_lock_result(task_id) is None
                assert adapter.report_status()["in_flight_lookups"] == 1

                mq_server.close()
                server_closed = True
                time.sleep(lookup_timeout_s + 0.01)

                query_started = time.monotonic()
                bitmap = adapter.query_lookup_and_lock_result(task_id)
                query_elapsed = time.monotonic() - query_started

                assert bitmap is not None
                assert bitmap.popcount() == 0
                # An expired task must not wait for the 3-second MQ RPC timeout.
                assert query_elapsed < 1.0
                assert adapter.report_status()["in_flight_lookups"] == 0
            finally:
                adapter.close()
    finally:
        if not server_closed:
            mq_server.close()
        controller.close()
