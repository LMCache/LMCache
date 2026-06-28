# SPDX-License-Identifier: Apache-2.0

"""Tier-1 end-to-end test for Dynamo KV-event publishing.

Starts a *real* LMCache MP cache server in a spawned subprocess with
``enable_dynamo_kv_events=True``, drives real store + eviction through the
public message-queue client, and asserts that an independent ZMQ ``SUB`` peer
receives correctly shaped vLLM ``BlockStored`` / ``BlockRemoved`` events
decoded with vLLM's own structs. No Dynamo and no model are involved.

Reuses the proven harness patterns from ``test_cache_server.py`` (subprocess
server management, ``ClientContext`` GPU tensors, ``REGISTER_KV_CACHE`` +
single-key ``STORE``), shrunk to a minimal KV geometry so it stays lightweight.

This test exercises real native/CUDA paths; it requires a GPU host with the
native extension built. It deliberately adds no environment skips.
"""

# Standard
import os
import time

# Third Party
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVEventBatch
import msgspec
import torch
import zmq

# First Party
from lmcache.utils import EngineType
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

# Local
# Local (shared harness)
from .dynamo_e2e_harness import (  # noqa: F401  -- pytest fixture used by the test below
    BLOCKS_PER_KEY,
    DEFAULT_TIMEOUT,
    KV_BLOCK_SIZE,
    MODEL_NAME,
    NUM_KEYS,
    _make_key,
    _make_kv_cache,
    _store_key,
    _wrap_kv_cache,
    server,
)

_DECODER = msgspec.msgpack.Decoder(type=KVEventBatch)


def _drain(
    sub: zmq.Socket,
    idle_ms: int = 800,
    max_total_s: float = 10.0,
) -> tuple[list[int], list[object], list[int | None]]:
    """Drain frames until ``idle_ms`` of silence or a hard deadline.

    Returns the per-frame sequence numbers in receive order, the flattened
    list of decoded events across all received batches, and the per-batch
    ``data_parallel_rank`` values (one per received frame).
    """
    seqs: list[int] = []
    events: list[object] = []
    dp_ranks: list[int | None] = []
    deadline = time.monotonic() + max_total_s
    while time.monotonic() < deadline:
        if sub.poll(idle_ms):
            _topic, seq_bytes, payload = sub.recv_multipart()
            seqs.append(int.from_bytes(seq_bytes, "big"))
            batch = _DECODER.decode(payload)
            events.extend(batch.events)
            dp_ranks.append(batch.data_parallel_rank)
        else:
            break
    return seqs, events, dp_ranks


def test_store_and_evict_emit_real_kv_events(
    server: tuple[str, str],  # noqa: F811  -- shadows the imported fixture by design
) -> None:
    """Real store -> BlockStored, real CLEAR-eviction -> BlockRemoved."""
    if not torch.cuda.is_available():
        raise RuntimeError("this end-to-end test requires a CUDA device")

    mq_url, zmq_bind = server
    device = torch.device("cuda:0")
    instance_id = os.getpid()

    zmq_ctx = zmq.Context()
    sub = zmq_ctx.socket(zmq.SUB)
    sub.connect(zmq_bind)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    # PUB slow-joiner: let the SUB connection settle before any store happens.
    time.sleep(1.0)

    client = MessageQueueClient(server_url=mq_url, context=zmq.Context.instance())
    kv_tensors = _make_kv_cache(device)

    try:
        # Register the KV cache (lmcache-driven path).
        client.submit_request(
            RequestType.REGISTER_KV_CACHE,
            [
                instance_id,
                _wrap_kv_cache(kv_tensors),
                MODEL_NAME,
                1,
                EngineType.VLLM,
                {},
                [],
            ],
            get_response_class(RequestType.REGISTER_KV_CACHE),
        ).result(timeout=DEFAULT_TIMEOUT)

        # Real stores: NUM_KEYS single-chunk keys, each a full CHUNK_SIZE span
        # that yields BLOCKS_PER_KEY full KV blocks.
        keys = [_make_key(i) for i in range(NUM_KEYS)]
        event = torch.cuda.Event(interprocess=True)
        event.record()
        for i, key in enumerate(keys):
            block_ids = list(range(i * BLOCKS_PER_KEY, (i + 1) * BLOCKS_PER_KEY))
            _store_key(client, key, instance_id, block_ids, event)

        store_seqs, store_events, store_dp_ranks = _drain(sub)

        stored = [e for e in store_events if isinstance(e, BlockStored)]
        assert stored, "expected at least one BlockStored event"

        # Field-level checks on the BlockStored for the first key (sequence
        # start, so parent_block_hash must be None).
        first_tokens = list(keys[0].token_ids)
        first_stored = next(
            (e for e in stored if list(e.token_ids) == first_tokens), None
        )
        assert first_stored is not None, "no BlockStored matched key 0's tokens"
        assert first_stored.block_size == KV_BLOCK_SIZE
        assert first_stored.parent_block_hash is None
        assert first_stored.medium == "GPU"
        assert len(first_stored.block_hashes) == BLOCKS_PER_KEY
        assert all(isinstance(h, int) for h in first_stored.block_hashes)
        # i64 range check (Dynamo block hashes are signed 64-bit).
        assert all(-(2**63) <= h < 2**63 for h in first_stored.block_hashes)

        assert store_seqs, "expected sequenced store frames"

        # Trigger real eviction of everything via CLEAR.
        client.submit_request(
            RequestType.CLEAR, [], get_response_class(RequestType.CLEAR)
        ).result(timeout=DEFAULT_TIMEOUT)

        evict_seqs, evict_events, evict_dp_ranks = _drain(sub)

        removed = [e for e in evict_events if isinstance(e, BlockRemoved)]
        assert removed, "expected at least one BlockRemoved event after CLEAR"

        # Every emitted batch carries the configured data_parallel_rank (0).
        all_dp_ranks = store_dp_ranks + evict_dp_ranks
        assert all_dp_ranks, "expected at least one batch frame"
        assert all(rank == 0 for rank in all_dp_ranks), (
            f"every batch's data_parallel_rank must be 0, saw {set(all_dp_ranks)}"
        )

        # Every removed block hash should match a previously stored block hash.
        stored_hashes = {h for e in stored for h in e.block_hashes}
        removed_hashes = {h for e in removed for h in e.block_hashes}
        assert removed_hashes, "BlockRemoved carried no block hashes"
        assert removed_hashes & stored_hashes, (
            "removed block hashes do not intersect any stored block hashes"
        )

        # Sequence numbers are globally monotonic across the whole session.
        all_seqs = store_seqs + evict_seqs
        assert all_seqs == sorted(all_seqs), "seq numbers not non-decreasing"
        assert len(set(all_seqs)) == len(all_seqs), "seq numbers not strictly unique"
    finally:
        try:
            client.submit_request(
                RequestType.UNREGISTER_KV_CACHE,
                [instance_id],
                get_response_class(RequestType.UNREGISTER_KV_CACHE),
            ).result(timeout=DEFAULT_TIMEOUT)
        except Exception:
            pass
        client.close()
        sub.close(linger=0)
        zmq_ctx.term()
        del kv_tensors
        torch.cuda.empty_cache()
