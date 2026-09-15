# SPDX-License-Identifier: Apache-2.0
"""Tests for how the MP connector reports a finished request's end.

The server's commit policy decides from ``SessionEndInfo`` whether the
request's final window is copied to L2. The connector has to read the
finish reason and stop token while the vLLM ``Request`` is still around, and
in lazy-offload mode hold them until ``END_SESSION`` is finally sent from
``update_connector_output``.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.v1.request import RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
    _build_session_end_info,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPWorkerMetadata,
)
from lmcache.v1.multiprocess.custom_types import SessionEndInfo  # noqa: E402

IM_END = 151645
"""Qwen's ``<|im_end|>``, the token a chat turn ends on."""


def _finished_request(
    status: RequestStatus = RequestStatus.FINISHED_STOPPED,
    stop_reason: int | str | None = None,
    output_token_ids: list[int] | None = None,
    request_id: str = "req-1",
) -> SimpleNamespace:
    """A duck-typed finished vLLM ``Request`` with what the connector reads.

    Args:
        status: The request's final status.
        stop_reason: vLLM's ``stop_reason``; ``None`` for a stop on EOS.
        output_token_ids: Generated tokens; defaults to a short answer that
            ends on ``IM_END``.
        request_id: The request id.

    Returns:
        The fake request.
    """
    if output_token_ids is None:
        output_token_ids = [10, 11, IM_END]
    return SimpleNamespace(
        request_id=request_id,
        status=status,
        stop_reason=stop_reason,
        output_token_ids=output_token_ids,
        kv_transfer_params=None,
    )


class TestBuildSessionEndInfo:
    """Tests for reading the finish reason and stop token off a request."""

    def test_eos_stop_falls_back_to_the_last_generated_token(self):
        """vLLM leaves ``stop_reason`` None for the model's own EOS."""
        info = _build_session_end_info(_finished_request(stop_reason=None))

        assert info == SessionEndInfo(finish_reason="stop", stop_token_id=IM_END)

    def test_explicit_stop_token_id_is_reported_as_is(self):
        """A ``stop_token_ids`` stop carries the token in ``stop_reason``."""
        info = _build_session_end_info(
            _finished_request(stop_reason=42, output_token_ids=[10, 42])
        )

        assert info.stop_token_id == 42

    def test_stop_string_falls_back_to_the_last_generated_token(self):
        """A stop *string* is not a token id."""
        info = _build_session_end_info(_finished_request(stop_reason="\n\n"))

        assert info.stop_token_id == IM_END

    @pytest.mark.parametrize(
        ("status", "reason"),
        [
            (RequestStatus.FINISHED_LENGTH_CAPPED, "length"),
            (RequestStatus.FINISHED_ABORTED, "abort"),
            (RequestStatus.FINISHED_IGNORED, "length"),
        ],
    )
    def test_non_stop_finish_reasons_are_reported(
        self, status: RequestStatus, reason: str
    ):
        """The policy refuses these; the connector must not hide them.

        Args:
            status: The request's final status.
            reason: The finish reason vLLM maps it to.
        """
        info = _build_session_end_info(_finished_request(status=status))

        assert info.finish_reason == reason

    def test_request_without_output_reports_no_token(self):
        """Nothing generated, nothing stopped on."""
        info = _build_session_end_info(_finished_request(output_token_ids=[]))

        assert info.stop_token_id == -1


def _scheduler_connector(lazy_offload: bool) -> SimpleNamespace:
    """The slice of a scheduler-side connector ``request_finished`` touches.

    Args:
        lazy_offload: Whether END_SESSION is deferred to store completion.

    Returns:
        A namespace usable as ``self`` for the unbound connector methods.
    """
    gpu_block_pool = MagicMock()
    gpu_block_pool.blocks = {}
    return SimpleNamespace(
        lazy_offload=lazy_offload,
        scheduler_adapter=MagicMock(),
        _pending_store=MagicMock(),
        _pending_end_info={},
        _gpu_block_pool=gpu_block_pool,
        _cleanup_request_tracker=MagicMock(),
    )


class TestEndSessionDelivery:
    """Tests for when ``END_SESSION`` carries the end info to the server."""

    def test_eager_mode_ends_the_session_at_request_finished(self):
        """Without lazy offload the info travels immediately."""
        connector = _scheduler_connector(lazy_offload=False)
        request = _finished_request()

        LMCacheMPConnector.request_finished(connector, request, block_ids=[])

        connector.scheduler_adapter.end_session.assert_called_once_with(
            "req-1", SessionEndInfo(finish_reason="stop", stop_token_id=IM_END)
        )
        assert connector._pending_end_info == {}

    def test_lazy_mode_parks_the_info_until_the_store_completes(self):
        """The Request is gone by the time END_SESSION is sent."""
        connector = _scheduler_connector(lazy_offload=True)
        connector.scheduler_adapter.update_pending_store_count.return_value = True
        connector._pending_store.get_request_gpu_block_ids.return_value = []
        request = _finished_request()

        LMCacheMPConnector.request_finished(connector, request, block_ids=[])

        connector.scheduler_adapter.end_session.assert_not_called()
        assert connector._pending_end_info == {
            "req-1": SessionEndInfo(finish_reason="stop", stop_token_id=IM_END)
        }

        output = SimpleNamespace(
            kv_connector_worker_meta=LMCacheMPWorkerMetadata(
                completed_store_requests={"req-1": 1}
            )
        )
        LMCacheMPConnector.update_connector_output(connector, output)

        connector.scheduler_adapter.end_session.assert_called_once_with(
            "req-1", SessionEndInfo(finish_reason="stop", stop_token_id=IM_END)
        )
        assert connector._pending_end_info == {}

    def test_lazy_mode_store_completion_without_parked_info_sends_defaults(self):
        """A completion for an unknown request still ends its session."""
        connector = _scheduler_connector(lazy_offload=True)
        connector.scheduler_adapter.update_pending_store_count.return_value = True
        connector._pending_store.get_request_gpu_block_ids.return_value = []

        output = SimpleNamespace(
            kv_connector_worker_meta=LMCacheMPWorkerMetadata(
                completed_store_requests={"req-9": 1}
            )
        )
        LMCacheMPConnector.update_connector_output(connector, output)

        connector.scheduler_adapter.end_session.assert_called_once_with(
            "req-9", SessionEndInfo()
        )
