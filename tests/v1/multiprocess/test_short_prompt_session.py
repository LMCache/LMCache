# SPDX-License-Identifier: Apache-2.0
"""Session lifecycle for requests whose prompt is shorter than one chunk.

A prompt below ``chunk_size`` yields no chunk hashes, so ``lookup`` takes its
``empty_chunk_hashes`` early exit.  These tests pin the contract that such a
request is a *completed* lookup with nothing to fetch: it ends quietly and
touches nothing, while ``end_session`` keeps warning for a session that is
genuinely missing or genuinely missing its lookup key.

The storage manager and layout registry are mocked -- the behavior under test
is session bookkeeping, not prefetching -- but the session manager and token
hasher are real, since the defect was in how those two interact.
``fold_unfold_ranked`` is patched out because it needs the native kernel.

Warnings are captured by patching ``logger.warning`` rather than with
``caplog``: LMCache's loggers do not propagate to the root handler, so
``caplog`` silently records nothing and every "no warning" assertion would
pass vacuously.  The captured value is the lazy ``%`` format string, which is
what the ``*_WARNING`` fragments below match against.
"""

# Standard
from collections.abc import Iterator
from unittest.mock import MagicMock, patch
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, PrefetchHandle
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher
import lmcache.v1.multiprocess.modules.lookup as lookup_module

CHUNK_SIZE = 4
REQUEST_ID = "req-1"

NO_LOOKUP_KEY_WARNING = "has no lookup ipc key"
NO_SESSION_WARNING = "not found, skipping touch"


@pytest.fixture
def warnings() -> Iterator[MagicMock]:
    """Patch the lookup module's ``logger.warning`` and yield the mock."""
    with patch.object(lookup_module.logger, "warning") as mock_warning:
        yield mock_warning


def _messages(warnings: MagicMock) -> list[str]:
    """Return the format string of every captured warning call."""
    return [call.args[0] for call in warnings.call_args_list]


def _make_ctx(
    num_groups: int = 1,
    world_size: int = 1,
    layout_found: bool = True,
    group_layouts_found: bool = True,
) -> MagicMock:
    """Build an engine context with a real session manager and token hasher.

    Args:
        num_groups: Object groups per chunk.
        world_size: kv_rank shards per chunk.
        layout_found: False triggers the ``no_gpu_context`` early exit.
        group_layouts_found: False triggers the ``no_group_layout_descs``
            early exit.

    Returns:
        A mock context wired for ``LookupModule``.
    """
    ctx = MagicMock()
    ctx.chunk_size = CHUNK_SIZE
    ctx.token_hasher = TokenHasher(chunk_size=CHUNK_SIZE, hash_algorithm="blake3")
    ctx.session_manager = SessionManager(
        ctx.token_hasher, ttl=600, cleanup_interval=None
    )
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.find.return_value = MagicMock() if layout_found else None
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1] * num_groups, world_size=world_size
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = (
        {group_id: MagicMock() for group_id in range(num_groups)}
        if group_layouts_found
        else {}
    )
    return ctx


def _lookup_key(
    token_ids: list[int],
    world_size: int = 1,
) -> IPCCacheServerKey:
    """A scheduler-side lookup key (``worker_id=None`` spans every KV rank)."""
    return IPCCacheServerKey(
        model_name="m",
        world_size=world_size,
        worker_id=None,
        token_ids=tuple(token_ids),
        start=0,
        end=len(token_ids),
        request_id=REQUEST_ID,
        num_kv_readers=1,
    )


def _run_request(
    ctx: MagicMock,
    token_ids: list[int],
    world_size: int = 1,
    found_count: int = 0,
    grown_token_ids: list[int] | None = None,
) -> None:
    """Drive one request through lookup, prefetch polling, and end_session.

    Args:
        ctx: Context from ``_make_ctx``.
        token_ids: Prompt tokens sent with the lookup.
        world_size: kv_rank shards per chunk.
        found_count: Chunk prefix the (patched) fold reports as hit.
        grown_token_ids: When given, the token sequence a store records on the
            session between polling and ``end_session``, standing in for a
            prompt that grows past a chunk boundary during generation.  The
            lookup-key fallback is applied exactly as
            ``MPCacheServerContext.resolve_object_keys`` applies it.
    """
    num_chunks = len(token_ids) // CHUNK_SIZE
    ctx.storage_manager.submit_prefetch_task.return_value = PrefetchHandle(
        prefetch_request_id=0,
        external_request_id=REQUEST_ID,
        l1_found_indices=(),
        l1_hit_chunks=found_count,
        total_requested_keys=num_chunks * world_size,
        submit_time=time.monotonic(),
    )

    module = LookupModule(ctx)
    with patch.object(
        lookup_module, "fold_unfold_ranked", return_value=(found_count, MagicMock())
    ):
        module.lookup(_lookup_key(token_ids, world_size), tp_size=1)
        module.query_prefetch_status(REQUEST_ID)
        if grown_token_ids is not None:
            store_key = _lookup_key(grown_token_ids, world_size)
            session = ctx.session_manager.get_or_create(REQUEST_ID)
            session.set_tokens(grown_token_ids)
            if session.lookup_ipc_key is None:
                session.lookup_ipc_key = store_key.no_worker_id_version()
        module.end_session(REQUEST_ID)


def _touched_keys(ctx: MagicMock) -> list:
    """Return the object keys passed to the single expected touch call."""
    ctx.storage_manager.touch_l1_keys.assert_called_once()
    return ctx.storage_manager.touch_l1_keys.call_args.args[0]


# =============================================================================
# The reproduced case
# =============================================================================


def test_sub_chunk_prompt_ends_without_warning(warnings: MagicMock):
    """A prompt below chunk_size ends quietly: nothing of it was cacheable."""
    ctx = _make_ctx()

    _run_request(ctx, token_ids=list(range(3)))

    assert _messages(warnings) == []


def test_sub_chunk_prompt_skips_the_touch(warnings: MagicMock):
    """No chunk-aligned key exists, so no L1 recency update is dispatched."""
    ctx = _make_ctx()

    _run_request(ctx, token_ids=list(range(3)))

    ctx.storage_manager.touch_l1_keys.assert_not_called()


def test_sub_chunk_lookup_records_its_key_on_the_session(warnings: MagicMock):
    """The lookup happened, so the session carries the key that proves it."""
    ctx = _make_ctx()
    module = LookupModule(ctx)

    module.lookup(_lookup_key(list(range(3))), tp_size=1)

    session = ctx.session_manager.get(REQUEST_ID)
    assert session is not None
    assert session.lookup_ipc_key is not None


def test_sub_chunk_prompt_still_reports_its_early_exit(warnings: MagicMock):
    """Quieting the warning must not cost the early-exit attribution."""
    ctx = _make_ctx()

    _run_request(ctx, token_ids=list(range(3)))

    end_events = [
        call.args[0].metadata
        for call in ctx.event_bus.publish.call_args_list
        if call.args[0].metadata
        and call.args[0].metadata.get("early_exit_reason") is not None
    ]
    assert end_events, "no MP_LOOKUP_PREFETCH_END event was published"
    assert end_events[-1]["early_exit_reason"] == "empty_chunk_hashes"
    assert end_events[-1]["hit_tokens"] == 0


# =============================================================================
# The warnings that must survive
# =============================================================================


def test_missing_session_still_warns(warnings: MagicMock):
    """A request whose session vanished before cleanup is still unexpected."""
    ctx = _make_ctx()
    module = LookupModule(ctx)

    module.end_session("never-looked-up")

    assert any(NO_SESSION_WARNING in message for message in _messages(warnings))
    ctx.storage_manager.touch_l1_keys.assert_not_called()


def test_cacheable_chunks_without_lookup_key_still_warns(warnings: MagicMock):
    """The fix must not silence a genuinely lost lookup key."""
    ctx = _make_ctx()
    module = LookupModule(ctx)
    session = ctx.session_manager.get_or_create(REQUEST_ID)
    session.set_tokens(list(range(8)))  # two full chunks -- these were cacheable

    module.end_session(REQUEST_ID)

    assert any(NO_LOOKUP_KEY_WARNING in message for message in _messages(warnings))
    ctx.storage_manager.touch_l1_keys.assert_not_called()


def test_no_gpu_context_still_warns(warnings: MagicMock):
    """The error early exit keeps its session warning.

    It returns before the layout descriptor is resolved, so unlike the
    sub-chunk path it cannot record a lookup key -- and it should not go
    quiet: it has already logged an error, and the session it leaves behind
    really is incompletely initialized.
    """
    ctx = _make_ctx(layout_found=False)

    _run_request(ctx, token_ids=list(range(8)))

    assert any(NO_LOOKUP_KEY_WARNING in message for message in _messages(warnings))
    ctx.storage_manager.touch_l1_keys.assert_not_called()


# =============================================================================
# Unchanged paths
# =============================================================================


@pytest.mark.parametrize(
    ("num_tokens", "expected_keys"),
    [(4, 1), (9, 2), (12, 3)],
    ids=["one_chunk", "two_chunks_plus_remainder", "three_chunks"],
)
def test_chunk_aligned_prompt_touches_every_chunk(
    num_tokens: int,
    expected_keys: int,
    warnings: MagicMock,
):
    """A prompt with complete chunks touches one key per chunk, as before."""
    ctx = _make_ctx()

    _run_request(ctx, token_ids=list(range(num_tokens)))

    assert _messages(warnings) == []
    assert len(_touched_keys(ctx)) == expected_keys


def test_touch_expands_over_world_size(warnings: MagicMock):
    """Each chunk is touched once per kv_rank when worker_id is None."""
    ctx = _make_ctx(world_size=2)

    _run_request(ctx, token_ids=list(range(8)), world_size=2)

    assert _messages(warnings) == []
    assert len(_touched_keys(ctx)) == 4  # 2 chunks * 2 workers


def test_sub_chunk_prompt_that_grows_touches_the_completed_chunk(
    warnings: MagicMock,
):
    """A short prompt that crosses a chunk boundary while generating still
    touches the chunk it stored.

    Recording the lookup key up front makes the store path's fallback a no-op
    instead of the only thing that sets the key; the resolved key set must be
    the same either way, because object keys are built from the session's
    hashes and the key's model/world/salt -- never from the key's tokens.
    """
    ctx = _make_ctx()

    _run_request(
        ctx,
        token_ids=list(range(3)),
        grown_token_ids=list(range(6)),  # one complete chunk now exists
    )

    assert _messages(warnings) == []
    assert len(_touched_keys(ctx)) == 1


def test_no_group_layout_descs_keeps_its_initialized_session(warnings: MagicMock):
    """This early exit returns after begin_lookup, so it already ends clean."""
    ctx = _make_ctx(group_layouts_found=False)

    _run_request(ctx, token_ids=list(range(8)))

    assert _messages(warnings) == []
    assert len(_touched_keys(ctx)) == 2


def test_free_lookup_locks_on_sub_chunk_request_releases_nothing(
    warnings: MagicMock,
):
    """A sub-chunk request locked no object, so cleanup is a silent no-op."""
    ctx = _make_ctx()
    token_ids = list(range(3))
    ctx.storage_manager.submit_prefetch_task.return_value = PrefetchHandle(
        prefetch_request_id=0,
        external_request_id=REQUEST_ID,
        l1_found_indices=(),
        l1_hit_chunks=0,
        total_requested_keys=0,
        submit_time=time.monotonic(),
    )

    module = LookupModule(ctx)
    with patch.object(
        lookup_module, "fold_unfold_ranked", return_value=(0, MagicMock())
    ):
        module.lookup(_lookup_key(token_ids), tp_size=1)
        module.query_prefetch_status(REQUEST_ID)
        module.free_lookup_locks(_lookup_key(token_ids), tp_size=1)

    assert _messages(warnings) == []
    ctx.storage_manager.finish_read_prefetched.assert_not_called()
