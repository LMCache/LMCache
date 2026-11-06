# SPDX-License-Identifier: Apache-2.0
"""Tests for how the MP connector reports a finished request's end.

The server's commit policy decides from ``SessionEndInfo`` whether the
request's final window is copied to L2. The connector has to read the
finish reason and stop token while the vLLM ``Request`` is still around, and
in lazy-offload mode hold them until ``END_SESSION`` is finally sent from
``update_connector_output``.
"""

# Standard
from functools import partial
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm import SamplingParams  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm import lmcache_mp_connector  # noqa: E402
from lmcache.integration.vllm.lazy_offload_manager import (  # noqa: E402
    LazyOffloadActions,
)
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

SPECIAL = 99
"""A special token the fake tokenizer drops when skipping special tokens."""


class _FakeTokenizer:
    """Decodes each token id to a fixed piece of text."""

    PIECES = {
        10: "Thought",
        11: ": done",
        12: "\nObs",
        13: "ervation",
        14: ":",
        15: " ok",
        SPECIAL: "<x>",
        IM_END: "",
    }

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Concatenate the pieces of ``ids``.

        Args:
            ids: Token ids.
            skip_special_tokens: Drop ``SPECIAL`` when True.

        Returns:
            The decoded text.
        """
        return "".join(
            self.PIECES[i] for i in ids if not (skip_special_tokens and i == SPECIAL)
        )


def _no_tokenizer_expected() -> None:
    """Tokenizer getter for cases that must not load the tokenizer."""
    pytest.fail("the tokenizer was loaded for a request that cannot need it")


def _finished_request(
    status: RequestStatus = RequestStatus.FINISHED_STOPPED,
    stop_reason: int | str | None = None,
    output_token_ids: list[int] | None = None,
    request_id: str = "req-1",
    stop: list[str] | None = None,
    min_tokens: int = 0,
) -> SimpleNamespace:
    """A duck-typed finished vLLM ``Request`` with what the connector reads.

    Args:
        status: The request's final status.
        stop_reason: vLLM's ``stop_reason``; ``None`` for a stop on EOS.
        output_token_ids: Generated tokens; defaults to a short answer that
            ends on ``IM_END``.
        request_id: The request id.
        stop: The request's stop strings.
        min_tokens: The request's ``min_tokens``.

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
        sampling_params=SamplingParams(stop=stop, min_tokens=min_tokens),
        kv_transfer_params=None,
    )


class TestBuildSessionEndInfo:
    """Tests for reading the finish reason and stop token off a request."""

    def test_eos_stop_falls_back_to_the_last_generated_token(self):
        """vLLM leaves ``stop_reason`` None for the model's own EOS."""
        info = _build_session_end_info(
            _finished_request(stop_reason=None), _no_tokenizer_expected
        )

        assert info == SessionEndInfo(finish_reason="stop", stop_token_id=IM_END)

    def test_explicit_stop_token_id_is_reported_as_is(self):
        """A ``stop_token_ids`` stop carries the token in ``stop_reason``."""
        info = _build_session_end_info(
            _finished_request(stop_reason=42, output_token_ids=[10, 42]),
            _no_tokenizer_expected,
        )

        assert info.stop_token_id == 42

    def test_stop_string_in_stop_reason_reports_no_stop_token(self):
        """A stop *string* is not a token id."""
        info = _build_session_end_info(
            _finished_request(stop_reason="\n\n"), _no_tokenizer_expected
        )

        assert info == SessionEndInfo(finish_reason="stop", stop_token_id=-1)

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
        info = _build_session_end_info(
            _finished_request(status=status), _no_tokenizer_expected
        )

        assert info.finish_reason == reason

    def test_request_without_output_reports_no_token(self):
        """Nothing generated, nothing stopped on."""
        info = _build_session_end_info(
            _finished_request(output_token_ids=[]), _no_tokenizer_expected
        )

        assert info.stop_token_id == -1


class TestStopStringStop:
    """Tests for a stop-string stop, which reaches the engine as an abort.

    vLLM matches stop strings in the frontend and then aborts the request in
    the engine, so the connector has to tell it apart from a client abort.
    """

    @pytest.mark.parametrize(
        "output_token_ids",
        [
            pytest.param([10, 11, 12, 13, 14], id="stop-string-last"),
            pytest.param([10, 11, 12, 13, 14, 15], id="tokens-after-it"),
        ],
    )
    def test_aborted_request_whose_output_has_a_stop_string_stopped(
        self, output_token_ids: list[int]
    ):
        """A stop string in the output means the frontend stopped it there.

        The engine may have generated tokens past the stop string before the
        abort reached it.

        Args:
            output_token_ids: The engine's generated tokens.
        """
        info = _build_session_end_info(
            _finished_request(
                status=RequestStatus.FINISHED_ABORTED,
                output_token_ids=output_token_ids,
                stop=["Observation:"],
            ),
            _FakeTokenizer,
        )

        assert info == SessionEndInfo(finish_reason="stop", stop_token_id=-1)

    def test_aborted_request_without_its_stop_string_is_an_abort(self):
        """Stop strings configured but never produced: a client abort."""
        info = _build_session_end_info(
            _finished_request(
                status=RequestStatus.FINISHED_ABORTED,
                output_token_ids=[10, 11, 12],
                stop=["Observation:"],
            ),
            _FakeTokenizer,
        )

        assert info == SessionEndInfo(finish_reason="abort", stop_token_id=12)

    def test_stop_string_inside_min_tokens_does_not_count(self):
        """vLLM ignores a stop string that ends within ``min_tokens``."""
        info = _build_session_end_info(
            _finished_request(
                status=RequestStatus.FINISHED_ABORTED,
                output_token_ids=[10, 11, 15, 15],
                stop=[": done"],
                min_tokens=2,
            ),
            _FakeTokenizer,
        )

        assert info.finish_reason == "abort"

    def test_stop_string_ending_after_min_tokens_counts(self):
        """One that starts within ``min_tokens`` but ends after it stops."""
        info = _build_session_end_info(
            _finished_request(
                status=RequestStatus.FINISHED_ABORTED,
                output_token_ids=[10, 11, 12, 13, 14],
                stop=["Observation:"],
                min_tokens=3,
            ),
            _FakeTokenizer,
        )

        assert info.finish_reason == "stop"

    def test_special_tokens_are_skipped_like_the_frontend_does(self):
        """``skip_special_tokens`` (on by default) drops them from the text."""
        info = _build_session_end_info(
            _finished_request(
                status=RequestStatus.FINISHED_ABORTED,
                output_token_ids=[12, 13, SPECIAL, 14],
                stop=["Observation:"],
            ),
            _FakeTokenizer,
        )

        assert info.finish_reason == "stop"

    def test_aborted_request_without_stop_strings_loads_no_tokenizer(self):
        """A plain abort never needs the tokenizer."""
        info = _build_session_end_info(
            _finished_request(status=RequestStatus.FINISHED_ABORTED),
            _no_tokenizer_expected,
        )

        assert info.finish_reason == "abort"

    def test_finished_request_with_stop_strings_loads_no_tokenizer(self):
        """Only an abort can hide a stop-string stop."""
        info = _build_session_end_info(
            _finished_request(stop=["Observation:"]), _no_tokenizer_expected
        )

        assert info == SessionEndInfo(finish_reason="stop", stop_token_id=IM_END)

    def test_without_a_tokenizer_an_abort_stays_an_abort(self):
        """vLLM run with ``skip_tokenizer_init`` has no tokenizer to check."""
        info = _build_session_end_info(
            _finished_request(
                status=RequestStatus.FINISHED_ABORTED,
                output_token_ids=[10, 11, 12, 13, 14],
                stop=["Observation:"],
            ),
            lambda: None,
        )

        assert info.finish_reason == "abort"


def _scheduler_connector(lazy_offload: bool) -> SimpleNamespace:
    """The slice of a scheduler-side connector ``request_finished`` touches.

    Args:
        lazy_offload: Whether END_SESSION is deferred to store completion.

    Returns:
        A namespace usable as ``self`` for the unbound connector methods.
    """
    manager = MagicMock()
    # The store is still in flight when the request finishes, so the manager
    # does not release the session yet; it does once the store completes.
    manager.on_request_finished.return_value = LazyOffloadActions()
    manager.on_store_results.return_value = LazyOffloadActions()
    connector = SimpleNamespace(
        lazy_offload=lazy_offload,
        scheduler_adapter=MagicMock(),
        _lazy_offload_manager=manager,
        _pending_end_info={},
        _kv_cache_events=None,
        _cleanup_request_tracker=MagicMock(),
        _can_store=True,
        _vllm_config=SimpleNamespace(model_config=SimpleNamespace()),
    )
    connector._end_lazy_sessions = partial(
        LMCacheMPConnector._end_lazy_sessions,
        cast(LMCacheMPConnector, connector),
    )
    return connector


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

    def test_stop_string_stop_ends_the_session_as_a_stop(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The connector reads the model's tokenizer from the vLLM config.

        Args:
            monkeypatch: Replaces vLLM's tokenizer loader.
        """
        connector = _scheduler_connector(lazy_offload=False)
        loaded_for: list[object] = []

        def load(model_config: object) -> _FakeTokenizer:
            loaded_for.append(model_config)
            return _FakeTokenizer()

        monkeypatch.setattr(lmcache_mp_connector, "cached_tokenizer_from_config", load)
        request = _finished_request(
            status=RequestStatus.FINISHED_ABORTED,
            output_token_ids=[10, 11, 12, 13, 14],
            stop=["Observation:"],
        )

        LMCacheMPConnector.request_finished(
            cast(LMCacheMPConnector, connector), request, block_ids=[]
        )

        connector.scheduler_adapter.end_session.assert_called_once_with(
            "req-1", SessionEndInfo(finish_reason="stop", stop_token_id=-1)
        )
        assert loaded_for == [connector._vllm_config.model_config]

    def test_lazy_mode_parks_the_info_until_the_store_completes(self):
        """The Request is gone by the time END_SESSION is sent."""
        connector = _scheduler_connector(lazy_offload=True)
        request = _finished_request()

        LMCacheMPConnector.request_finished(connector, request, block_ids=[])

        connector.scheduler_adapter.end_session.assert_not_called()
        assert connector._pending_end_info == {
            "req-1": SessionEndInfo(finish_reason="stop", stop_token_id=IM_END)
        }

        connector._lazy_offload_manager.on_store_results.return_value = (
            LazyOffloadActions(sessions_to_end=["req-1"])
        )
        output = SimpleNamespace(
            kv_cache_events=None,
            kv_connector_worker_meta=LMCacheMPWorkerMetadata(
                completed_store_requests={"req-1": 1}
            ),
        )
        LMCacheMPConnector.update_connector_output(connector, output)

        connector.scheduler_adapter.end_session.assert_called_once_with(
            "req-1", SessionEndInfo(finish_reason="stop", stop_token_id=IM_END)
        )
        assert connector._pending_end_info == {}

    def test_lazy_mode_store_completion_without_parked_info_sends_defaults(self):
        """A completion for an unknown request still ends its session."""
        connector = _scheduler_connector(lazy_offload=True)
        connector._lazy_offload_manager.on_store_results.return_value = (
            LazyOffloadActions(sessions_to_end=["req-9"])
        )

        output = SimpleNamespace(
            kv_cache_events=None,
            kv_connector_worker_meta=LMCacheMPWorkerMetadata(
                completed_store_requests={"req-9": 1}
            ),
        )
        LMCacheMPConnector.update_connector_output(connector, output)

        connector.scheduler_adapter.end_session.assert_called_once_with(
            "req-9", SessionEndInfo()
        )
