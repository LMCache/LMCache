# SPDX-License-Identifier: Apache-2.0
"""``lmcache.mp.sparse_state_every_n_chunks`` through the scheduler connector.

The scheduler names the chunk ends whose recurrent state a STORE keeps. A
chunk end holds a state only where a scheduler step ended on it, so the kept
ends follow the steps and not a fixed grid of chunk ends.
"""

# Standard
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.config import KVTransferConfig, VllmConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm import lmcache_mp_connector as connector_mod  # noqa: E402
from lmcache.integration.vllm import (  # noqa: E402
    vllm_multi_process_adapter as adapter_mod,
)
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)
from lmcache.lmcache_native import Bitmap  # noqa: E402
from lmcache.v1.client.worker_transfer import (  # noqa: E402
    LMCacheDrivenTransferContext,
)
from lmcache.v1.common.sparse_state import (  # noqa: E402
    SPARSE_STATE_KEEP_HINT,
    kept_chunk_ends,
)
from lmcache.v1.distributed.api import FULL_ATTENTION_WINDOW_CHUNKS  # noqa: E402
from lmcache.v1.distributed.bitmap_ops import fold_unfold_grouped  # noqa: E402

pytestmark = pytest.mark.no_shared_allocator

# One chunk is one attention block (group 0) and four state blocks (group 1).
CHUNK = 8
STATE_BLOCK = 2
# A 200,000-token prompt over 9216-token chunks is 21.7 chunks; a request
# sharing its first 190,000 tokens shares 20.6 chunks.
PROMPT = 174
SHARED = 165


def _config(every_n: int) -> VllmConfig:
    return cast(
        VllmConfig,
        SimpleNamespace(
            model_config=SimpleNamespace(model="test-model", use_mla=False),
            parallel_config=SimpleNamespace(
                world_size=1, rank=0, tensor_parallel_size=1, pipeline_parallel_size=1
            ),
            cache_config=SimpleNamespace(block_size=CHUNK, enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheMPConnector",
                kv_role="kv_both",
                kv_connector_extra_config=(
                    {"lmcache.mp.sparse_state_every_n_chunks": every_n}
                    if every_n
                    else {}
                ),
            ),
        ),
    )


def _schedule(
    num_tokens: int,
    block_ids: tuple[list[int], ...],
    *,
    new: bool,
    num_computed_tokens: int,
) -> SchedulerOutput:
    return cast(
        SchedulerOutput,
        SimpleNamespace(
            scheduled_new_reqs=(
                [
                    SimpleNamespace(
                        req_id="request",
                        block_ids=block_ids,
                        num_computed_tokens=num_computed_tokens,
                    )
                ]
                if new
                else []
            ),
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[] if new else ["request"],
                new_block_ids=[] if new else [block_ids],
                resumed_req_ids=set(),
                num_computed_tokens=[] if new else [num_computed_tokens],
            ),
            num_scheduled_tokens={"request": num_tokens},
            total_num_scheduled_tokens=num_tokens,
        ),
    )


MakeScheduler = Callable[[int], LMCacheMPConnector]


@pytest.fixture
def make_scheduler(monkeypatch: pytest.MonkeyPatch) -> Iterator[MakeScheduler]:
    """Build real scheduler connectors over a stubbed scheduler adapter."""
    adapter = MagicMock(lmcache_tokens_per_chunk=CHUNK)
    adapter.check_lookup_result.return_value = 0
    monkeypatch.setattr(
        connector_mod, "LMCacheMPSchedulerAdapter", lambda **kw: adapter
    )
    monkeypatch.setattr(connector_mod, "print_banner_once", MagicMock())
    monkeypatch.setattr(connector_mod, "vllm_layout_hints", lambda config: {})
    made: list[LMCacheMPConnector] = []

    def make(every_n: int) -> LMCacheMPConnector:
        scheduler = LMCacheMPConnector(_config(every_n), KVConnectorRole.SCHEDULER)
        # Group 0: attention, one block per chunk. Group 1: align-mode Mamba
        # state, four blocks per chunk, as __init__ derives from the groups.
        scheduler._group_tokens_per_block = [CHUNK, STATE_BLOCK]
        scheduler._windowed_group_tokens = {1: STATE_BLOCK}
        made.append(scheduler)
        return scheduler

    yield make
    for scheduler in made:
        scheduler.shutdown()


def _prefill(
    scheduler: LMCacheMPConnector, prompt: int, step: int
) -> tuple[list[int], list[int]]:
    """Prefill a prompt in steps of ``step`` tokens.

    vLLM snapshots the recurrent state on the block where each step ends and
    leaves the blocks a step skips null (block 0).

    Returns:
        The chunk ends (in chunks) of the stored chunks, and of the chunks
        whose STORE keeps the recurrent state (every chunk that holds one
        when the option is off).
    """
    request = cast(
        Request,
        SimpleNamespace(
            request_id="request",
            cache_salt="",
            lora_request=None,
            all_token_ids=list(range(prompt)),
            num_prompt_tokens=prompt,
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            kv_transfer_params=None,
        ),
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    stored: list[int] = []
    states: list[int] = []
    next_block = 1
    done = 0
    while done < prompt:
        tokens = min(step, prompt - done)
        end = done + tokens
        attention = -(-end // CHUNK) - -(-done // CHUNK)
        state = -(-end // STATE_BLOCK) - -(-done // STATE_BLOCK)
        block_ids = (
            list(range(next_block, next_block + attention)),
            [0] * (state - 1) + [next_block + attention],
        )
        next_block += attention + 1
        if done == 0:
            scheduler.update_state_after_alloc(
                request, MagicMock(get_block_ids=lambda ids=block_ids: ids), 0
            )
        metadata = scheduler.build_connector_meta(
            _schedule(tokens, block_ids, new=done == 0, num_computed_tokens=done)
        )
        for meta in metadata.requests:
            assert meta.direction == "STORE"
            ends = range(meta.op.start + CHUNK, meta.op.end + 1, CHUNK)
            stored.extend(chunk_end // CHUNK for chunk_end in ends)
            kept = kept_chunk_ends(meta.request_configs)
            if kept is None:
                # Option off: the server stores every state that is not null.
                per_chunk = CHUNK // STATE_BLOCK
                kept = frozenset(
                    chunk_end
                    for i, chunk_end in enumerate(ends)
                    if meta.op.block_ids[1][(i + 1) * per_chunk - 1] != 0
                )
            assert kept <= set(ends)
            states.extend(sorted(chunk_end // CHUNK for chunk_end in kept))
        done = end
    return stored, states


def _hit_chunks(stored: list[int], states: list[int], shared_tokens: int) -> int:
    """Chunks a request sharing ``shared_tokens`` hits, by the server's
    window-aware lookup over the stored attention and state objects."""
    num_chunks = shared_tokens // CHUNK
    rows = []
    for ends in (states, stored):
        row = Bitmap(num_chunks)
        for end in ends:
            if end <= num_chunks:
                row.set(end - 1)
        rows.append(row)
    hit, _retain = fold_unfold_grouped(rows, [1, FULL_ATTENTION_WINDOW_CHUNKS])
    return hit


@pytest.mark.parametrize(
    ("step", "kept", "hit", "hit_every_state"),
    [
        # One chunk per step: every chunk end holds a state.
        (CHUNK, [8, 16, 21], 16, 20),
        # 1.75 chunks (16,128 of 9216): states at 7, 14 and 21 chunks.
        (14, [14, 21], 14, 14),
        (3 * CHUNK, [9, 18, 21], 18, 18),
        # States at 5, 10, 15 and 20 chunks; none at the prompt's last chunk.
        (5 * CHUNK, [10, 20], 20, 20),
    ],
)
def test_kept_states_follow_the_steps(
    make_scheduler: MakeScheduler,
    step: int,
    kept: list[int],
    hit: int,
    hit_every_state: int,
) -> None:
    stored, states = _prefill(make_scheduler(8), PROMPT, step)

    assert stored == list(range(1, PROMPT // CHUNK + 1))
    assert states == kept
    assert _hit_chunks(stored, states, SHARED) == hit
    # A request that extends the prompt hits up to its last kept state.
    assert _hit_chunks(stored, states, PROMPT) == kept[-1]

    stored, states = _prefill(make_scheduler(0), PROMPT, step)
    assert _hit_chunks(stored, states, SHARED) == hit_every_state


def test_prompt_end_state_is_kept_for_a_repeat(make_scheduler: MakeScheduler) -> None:
    # A 95,000-token prompt over 9216-token chunks is 10.3 chunks.
    prompt = 82

    stored, states = _prefill(make_scheduler(8), prompt, CHUNK)

    assert states == [8, 10]
    assert _hit_chunks(stored, states, prompt) == 10


def test_option_off_sends_no_hint(make_scheduler: MakeScheduler) -> None:
    scheduler = make_scheduler(0)
    request = cast(
        Request,
        SimpleNamespace(
            request_id="request",
            cache_salt="",
            lora_request=None,
            all_token_ids=list(range(CHUNK)),
            num_prompt_tokens=CHUNK,
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            kv_transfer_params=None,
        ),
    )
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    block_ids = ([1], [0, 0, 0, 2])
    scheduler.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: block_ids), 0
    )

    metadata = scheduler.build_connector_meta(
        _schedule(CHUNK, block_ids, new=True, num_computed_tokens=0)
    )

    (store,) = metadata.requests
    assert SPARSE_STATE_KEEP_HINT not in (store.request_configs or {})


@pytest.mark.parametrize("lmcache_driven", [True, False])
def test_option_requires_the_lmcache_driven_transfer_mode(
    monkeypatch: pytest.MonkeyPatch, lmcache_driven: bool
) -> None:
    """Only the lmcache-driven server store skips the chunks the scheduler
    does not keep, so an engine-driven worker refuses the option."""
    transfer = MagicMock(spec=LMCacheDrivenTransferContext if lmcache_driven else None)
    monkeypatch.setattr(connector_mod, "vllm_layout_hints", lambda config: {})
    monkeypatch.setattr(adapter_mod.RequestClientFactory, "create", MagicMock())
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: CHUNK)
    monkeypatch.setattr(adapter_mod, "get_experimental", lambda *a, **kw: set())
    monkeypatch.setattr(adapter_mod, "HeartbeatThread", MagicMock())
    monkeypatch.setattr(adapter_mod, "set_ipc_policy", MagicMock())
    monkeypatch.setattr(
        adapter_mod, "create_transfer_context", lambda *a, **kw: transfer
    )
    monkeypatch.setattr(
        adapter_mod.RequestTelemetryFactory, "create", lambda **kw: MagicMock()
    )
    kv_caches = {"layer": torch.empty(2, 8, CHUNK, 1, 8)}

    worker = LMCacheMPConnector(_config(8), KVConnectorRole.WORKER)
    try:
        if lmcache_driven:
            worker.register_kv_caches(kv_caches)
        else:
            with pytest.raises(ValueError, match="sparse_state_every_n_chunks"):
                worker.register_kv_caches(kv_caches)
    finally:
        worker.shutdown()

    # Without the option the engine-driven worker registers.
    plain = LMCacheMPConnector(_config(0), KVConnectorRole.WORKER)
    try:
        plain.register_kv_caches(kv_caches)
    finally:
        plain.shutdown()
