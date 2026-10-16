# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``token_ids`` field of the ``MP_LOOKUP`` event.

With trace collection enabled, the field carries a request's prompt token
ids exactly once, on its first lookup; otherwise it is empty.  The context
is real apart from the storage manager and event bus; the lookup handler
test mocks the context and checks only the wiring.
"""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.lookup import LookupModule

TOKEN_IDS = (11, 22, 33)


def _key(request_id: str = "req-1") -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_ids=TOKEN_IDS,
        start=0,
        end=0,
        request_id=request_id,
        num_kv_readers=1,
    )


def _storage_manager_config() -> MagicMock:
    config = MagicMock()
    config.l1_manager_config.gds_l1_config = None
    memory_config = config.l1_manager_config.memory_config
    memory_config.shm_name = ""
    memory_config.use_lazy = False
    memory_config.devdax_path = None
    memory_config.size_in_bytes = 0
    return config


@pytest.fixture
def context():
    """A real context with a real session manager; storage is patched out."""
    with (
        patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
        patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
    ):
        ctx = MPCacheServerContext(_storage_manager_config(), trace_collect=True)
        yield ctx
        ctx.close()


class TestTraceTokenIds:
    def test_disabled_returns_empty(self):
        with (
            patch("lmcache.v1.multiprocess.engine_context.StorageManager"),
            patch("lmcache.v1.multiprocess.engine_context.get_event_bus"),
        ):
            ctx = MPCacheServerContext(_storage_manager_config())
            try:
                assert ctx.trace_collect is False
                assert ctx.trace_token_ids(_key()) == ()
            finally:
                ctx.close()

    def test_first_lookup_returns_prompt_without_creating_session(self, context):
        key = _key()

        assert context.trace_token_ids(key) is key.token_ids
        assert context.session_manager.get(key.request_id) is None

    def test_session_without_lookup_key_returns_prompt(self, context):
        key = _key()
        context.session_manager.get_or_create(key.request_id)

        assert context.trace_token_ids(key) is key.token_ids

    def test_repeat_lookup_returns_empty(self, context):
        """A session that already has a lookup key is a re-submitted request."""
        key = _key()
        context.session_manager.get_or_create(key.request_id).begin_lookup(key, ())

        assert context.trace_token_ids(key) == ()

    def test_sessions_are_independent(self, context):
        first, second = _key("req-1"), _key("req-2")
        context.session_manager.get_or_create(first.request_id).begin_lookup(first, ())

        assert context.trace_token_ids(first) == ()
        assert context.trace_token_ids(second) is second.token_ids


def test_lookup_event_carries_trace_token_ids():
    """The lookup handler puts the context's answer on ``MP_LOOKUP``."""
    ctx = MagicMock()
    ctx.chunk_size = 256
    ctx.trace_token_ids.return_value = TOKEN_IDS
    ctx.event_bus.has_subscribers.side_effect = (
        lambda event_type: event_type is EventType.MP_LOOKUP
    )
    ctx.layout_desc_registry.find.return_value = MagicMock(dtypes=[], shapes=[])
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1], world_size=1
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {}
    ctx.token_hasher.compute_chunk_hashes.return_value = [b"h"]
    key = _key()

    LookupModule(ctx).lookup(key, tp_size=1)

    (event,) = [
        call.args[0]
        for call in ctx.event_bus.publish.call_args_list
        if call.args[0].event_type is EventType.MP_LOOKUP
    ]
    assert event.metadata["token_ids"] is TOKEN_IDS
    ctx.trace_token_ids.assert_called_once_with(key)
