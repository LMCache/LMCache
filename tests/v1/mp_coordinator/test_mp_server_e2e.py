# SPDX-License-Identifier: Apache-2.0
"""What a real MP server reports to the coordinator, end to end.

Each case starts real ``lmcache server`` processes, CPU-only, against a
coordinator on a thread, drives them through a KV client and the HTTP APIs,
and checks two things: the batches the server sent, read from its own events
trace, and what the coordinator's directory made of them. See
``mp_server_harness`` for the pieces.

They start subprocesses and take about ten seconds each, so unit test runs
skip them. The coordinator integration pipeline,
``.buildkite/coordinator_controllers/``, runs them with ``RUN_MP_E2E=1``.
"""

# Standard
import os
import pathlib

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.distributed.api import Tier
from lmcache.v1.mp_coordinator.api import CacheEventBatch, CacheEventType

# Local
from .mp_server_harness import (
    L1_BACKEND,
    L1_SIZE_GB,
    Placement,
    directory,
    encoded,
    keys_of,
    kv_client,
    local_l2,
    running_coordinator,
    running_server,
    settle,
    shared_l2,
    steady,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MP_E2E") != "1",
    reason="end-to-end: starts real MP servers; set RUN_MP_E2E=1 to run",
)

MODEL = "e2e-model"
TOKENS = list(range(1, 9))
"""Two whole chunks."""

L1 = (Tier.L1, L1_BACKEND, False)
LOCAL_L2 = (Tier.L2, "mock", False)
SHARED_L2 = (Tier.L2, "fs", True)


def _entries(
    batches: list[CacheEventBatch], event_type: CacheEventType
) -> dict[tuple[Tier, str, bool], list[tuple[object, ...]]]:
    """Each backend's entries of one event type, in the order they were sent.

    Args:
        batches: A server's batches.
        event_type: The type to keep.

    Returns:
        Per ``(tier, backend, shared)``, one ``(key, size, tokens, offset)``
        per entry.
    """
    entries: dict[tuple[Tier, str, bool], list[tuple[object, ...]]] = {}
    for batch in batches:
        if batch.event_type != event_type:
            continue
        entries.setdefault((batch.tier, batch.backend, batch.shared), []).extend(
            (e.key.to_object_key(), e.size_bytes, e.token_ids, e.token_offset)
            for e in batch.entries
        )
    return entries


def test_a_store_is_reported_once_per_backend(tmp_path: pathlib.Path) -> None:
    """MP-1: one STORE batch per backend the chunks landed in, carrying the
    backend's tier, name and ``shared`` flag, and every chunk's size, tokens
    and offset."""
    with running_coordinator() as url:
        backends = [local_l2(), shared_l2(tmp_path / "pool")]
        with running_server(url, tmp_path, "mp-1", backends) as server:
            with kv_client(server.rpc_url, MODEL) as client:
                client.store(TOKENS, cache_salt="alice")
                size = client.chunk_bytes
            keys = keys_of(MODEL, TOKENS, "alice")
            expected = {
                Placement("mp-1", tier, backend, shared, key, size)
                for tier, backend, shared in (L1, LOCAL_L2, SHARED_L2)
                for key in keys
            }
            assert settle(lambda: directory(url), expected) == expected

    stores = [b for b in server.batches() if b.event_type == CacheEventType.STORE]
    assert len(stores) == 3
    chunks = [
        (key, size, TOKENS[offset : offset + 4], offset)
        for key, offset in zip(keys, (0, 4), strict=True)
    ]
    assert _entries(stores, CacheEventType.STORE) == {
        L1: chunks,
        LOCAL_L2: chunks,
        SHARED_L2: chunks,
    }


def test_a_delete_is_reported_for_the_tier_and_backend_it_left(
    tmp_path: pathlib.Path,
) -> None:
    """MP-2: deleting a placement through the server's API sends a DELETE
    batch for that tier and backend alone, and the coordinator drops only
    that placement."""
    with running_coordinator() as url:
        backends = [local_l2(), shared_l2(tmp_path / "pool")]
        with running_server(url, tmp_path, "mp-1", backends) as server:
            with kv_client(server.rpc_url, MODEL) as client:
                client.store(TOKENS)
                size = client.chunk_bytes
            first, second = keys_of(MODEL, TOKENS)
            held = {
                Placement("mp-1", tier, backend, shared, key, size)
                for tier, backend, shared in (L1, LOCAL_L2, SHARED_L2)
                for key in (first, second)
            }
            assert settle(lambda: directory(url), held) == held

            for key, tier, adapter in (
                (first, "l1", None),
                (first, "l2", "mock"),
                (second, "l2", "fs"),
            ):
                response = httpx.request(
                    "DELETE",
                    f"{server.http_url}/cache/objects",
                    json={"keys": encoded([key]), "tier": tier, "adapter": adapter},
                    timeout=5,
                )
                assert response.json() == {"deleted": 1, "skipped": 0, "ok": True}

            gone = {
                Placement("mp-1", *L1, first, size),
                Placement("mp-1", *LOCAL_L2, first, size),
                Placement("mp-1", *SHARED_L2, second, size),
            }
            assert settle(lambda: directory(url), held - gone) == held - gone

    deletes = _entries(server.batches(), CacheEventType.DELETE)
    assert deletes == {
        L1: [(first, 0, [], -1)],
        LOCAL_L2: [(first, 0, [], -1)],
        SHARED_L2: [(second, 0, [], -1)],
    }


def test_l1_eviction_is_reported_and_the_directory_follows(
    tmp_path: pathlib.Path,
) -> None:
    """MP-2: chunks the server evicts from a full L1 come back as L1 DELETE
    batches. The coordinator's L1 view ends equal to what the server's own
    events leave there, and the L2 copies are untouched."""
    with running_coordinator() as url:
        # 1 MB of L1 against 32 KiB chunks: a few sequences overflow it.
        with running_server(
            url, tmp_path, "mp-1", [local_l2()], l1_size_gb=0.001
        ) as server:
            with kv_client(server.rpc_url, MODEL, heads=8, head_size=128) as client:
                for sequence in range(6):
                    client.store([100 * sequence + t for t in range(1, 57)])

            def held(tier: Tier) -> set[object]:
                return {p.key for p in directory(url) if p.tier == tier}

            # Eviction runs on the server's own tick, about once a second:
            # wait until the coordinator has heard of some, then until the
            # view holds still across a tick.
            evicting = settle(lambda: len(held(Tier.L1)) < len(held(Tier.L2)), True)
            assert evicting, "the coordinator heard of no eviction"
            l1 = steady(lambda: held(Tier.L1), interval_s=1.5)
            l2 = held(Tier.L2)

    stores = _entries(server.batches(), CacheEventType.STORE)
    deletes = _entries(server.batches(), CacheEventType.DELETE)
    stored = {entry[0] for entry in stores[L1]}
    evicted = {entry[0] for entry in deletes[L1]}
    assert evicted and evicted <= stored
    assert l1 == stored - evicted
    assert l2 == {entry[0] for entry in stores[LOCAL_L2]} == stored
    assert LOCAL_L2 not in deletes


def test_a_coordinator_delete_reaches_the_server_and_comes_back(
    tmp_path: pathlib.Path,
) -> None:
    """MP-6: ``POST /cache/delete`` on the coordinator makes the server remove
    the sequence's chunks from both tiers and report the DELETEs, which empty
    the coordinator's directory."""
    with running_coordinator() as url:
        with running_server(url, tmp_path, "mp-1", [local_l2()]) as server:
            with kv_client(server.rpc_url, MODEL) as client:
                client.store(TOKENS, cache_salt="alice")
            keys = keys_of(MODEL, TOKENS, "alice")
            assert settle(lambda: len(directory(url)), 4) == 4

            response = httpx.post(
                f"{url}/cache/delete",
                json={
                    "instance_id": "mp-1",
                    "model_name": MODEL,
                    "world_size": 1,
                    "token_ids": TOKENS,
                    "cache_salt": "alice",
                    "tier": "all",
                },
                timeout=5,
            )
            assert response.status_code == httpx.codes.OK, response.text
            assert response.json()["requested"] == 2
            nothing: set[Placement] = set()
            assert settle(lambda: directory(url), nothing) == nothing

    deletes = _entries(server.batches(), CacheEventType.DELETE)
    assert {backend: {e[0] for e in rows} for backend, rows in deletes.items()} == {
        L1: set(keys),
        LOCAL_L2: set(keys),
    }


def test_capacity_is_declared_per_compartment_and_again_on_reregistration(
    tmp_path: pathlib.Path,
) -> None:
    """MP-3: on registering, the server declares every compartment's capacity
    in one ``config`` batch each, all under one revision. Registering again
    declares them again under the next revision."""
    with running_coordinator() as url:
        backends = [local_l2(), shared_l2(tmp_path / "pool")]
        with running_server(url, tmp_path, "mp-1", backends) as server:
            # Forgotten by the coordinator, the server's next heartbeat gets
            # a 404, and it registers again.
            response = httpx.delete(f"{url}/instances/mp-1", timeout=5)
            assert response.status_code == httpx.codes.NO_CONTENT

            def declarations() -> int:
                return sum(
                    "Registered with coordinator" in line
                    for line in server.log_path.read_text().splitlines()
                )

            assert settle(declarations, 2) == 2

    configs = [
        (b.tier, b.backend, b.shared, b.capacity_bytes, b.capacity_revision)
        for b in server.batches()
        if b.event_type == CacheEventType.CONFIG
    ]
    l1_bytes = int(L1_SIZE_GB * (1 << 30))
    compartments = [
        (*L1, l1_bytes),
        (*LOCAL_L2, 64 << 20),
        # The fs adapter declares no capacity.
        (*SHARED_L2, 0),
    ]
    assert configs == [(*c, 1) for c in compartments] + [(*c, 2) for c in compartments]


def test_a_restart_is_a_new_incarnation_that_reports_afresh(
    tmp_path: pathlib.Path,
) -> None:
    """MP-4: a restarted server reports under a higher incarnation with its
    sequence back at 1, and does not report its earlier placements again.
    The coordinator fences the old L1 placements and keeps the L2 ones."""
    with running_coordinator() as url:
        pool = shared_l2(tmp_path / "pool")
        with running_server(url, tmp_path, "mp-1", [pool]) as server:
            with kv_client(server.rpc_url, MODEL) as client:
                client.store(TOKENS)
                size = client.chunk_bytes
            keys = keys_of(MODEL, TOKENS)
            assert settle(lambda: len(directory(url)), 4) == 4

            server.restart()
            l2 = {Placement("mp-1", *SHARED_L2, key, size) for key in keys}
            assert settle(lambda: directory(url), l2) == l2

    first, second = sorted({b.incarnation for b in server.batches()})
    assert first < second
    runs = {
        incarnation: [b for b in server.batches() if b.incarnation == incarnation]
        for incarnation in (first, second)
    }
    assert [b.seq for b in runs[second]] == list(range(1, len(runs[second]) + 1))
    assert {b.event_type for b in runs[second]} == {CacheEventType.CONFIG}


def test_a_prefetch_through_the_coordinator_warms_l1(tmp_path: pathlib.Path) -> None:
    """MP-7: a prefetch submitted to the coordinator runs on the server, which
    loads the chunks from L2 back into L1 and reports them. Its status, read
    through the coordinator, says every key was found."""
    with running_coordinator() as url:
        with running_server(url, tmp_path, "mp-1", [local_l2()]) as server:
            # The client stays open: a prefetch needs the model's layout,
            # which the server holds only while a worker is registered.
            with kv_client(server.rpc_url, MODEL) as client:
                client.store(TOKENS, cache_salt="alice")
                size = client.chunk_bytes
                keys = keys_of(MODEL, TOKENS, "alice")
                l1 = {Placement("mp-1", *L1, key, size) for key in keys}
                l2 = {Placement("mp-1", *LOCAL_L2, key, size) for key in keys}
                assert settle(lambda: directory(url), l1 | l2) == l1 | l2

                response = httpx.request(
                    "DELETE",
                    f"{server.http_url}/cache/objects",
                    json={"keys": encoded(keys), "tier": "l1"},
                    timeout=5,
                )
                assert response.json()["deleted"] == 2
                assert settle(lambda: directory(url), l2) == l2

                response = httpx.post(
                    f"{url}/cache/prefetches",
                    json={
                        "instance_id": "mp-1",
                        "model_name": MODEL,
                        "world_size": 1,
                        "token_ids": TOKENS,
                        "cache_salt": "alice",
                    },
                    timeout=5,
                )
                assert response.status_code == httpx.codes.OK, response.text
                request_id = response.json()["request_id"]

                def status() -> dict[str, object]:
                    return httpx.get(
                        f"{url}/cache/prefetches/mp-1/{request_id}", timeout=5
                    ).json()

                # The poll that sees the job finished also retires it, so
                # wait for the whole finished body in one comparison.
                finished = {
                    "request_id": request_id,
                    "status": "completed",
                    "found_keys": 2,
                    "total_keys": 2,
                }
                assert settle(status, finished) == finished
                assert settle(lambda: directory(url), l1 | l2) == l1 | l2
