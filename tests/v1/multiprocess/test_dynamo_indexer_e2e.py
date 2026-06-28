# SPDX-License-Identifier: Apache-2.0

"""Tier-2 end-to-end test: a real Dynamo KV indexer consumes our KV events.

Starts a *real* LMCache MP cache server (Dynamo KV-event publishing on) and a
*real* standalone Dynamo KV indexer (``python -m dynamo.indexer``) in a
separate, isolated virtualenv. The indexer subscribes to the server's ZMQ PUB
endpoint via its HTTP ``/register`` API, then we drive real store + eviction
through the public MQ client and assert that the indexer's radix tree actually
ingests our blocks (``/dump``) and scores prefix overlap (``/query``), keyed by
our ``instance_id`` and ``dp_rank=0``.

The indexer binary lives in ``.venv-dynamo`` (``ai-dynamo-runtime``); when that
venv is absent the whole module is skipped (``requires dynamo.indexer
binary``). The LMCache environment's dependencies are never touched.

Reuses the harness shape of ``test_dynamo_kv_events_e2e.py`` (subprocess server,
minimal GPU KV geometry, ``REGISTER_KV_CACHE`` + single-key ``STORE``,
``CLEAR``-triggered eviction).
"""

# Standard
from pathlib import Path
from typing import Any, Generator, cast
import itertools
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

# Third Party
import pytest
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
    DP_RANK,
    KV_BLOCK_SIZE,
    MODEL_NAME,
    NUM_KEYS,
    _free_port,
    _make_key,
    _make_kv_cache,
    _store_key,
    _wrap_kv_cache,
    server,
)

# Isolated venv holding the Dynamo indexer binary (see Tier-2 setup).
_VENV_PYTHON = Path(__file__).resolve().parents[3] / ".venv-dynamo" / "bin" / "python"


def _indexer_available() -> bool:
    """True if the isolated venv can run ``python -m dynamo.indexer``."""
    if not _VENV_PYTHON.exists():
        return False
    try:
        proc = subprocess.run(
            [str(_VENV_PYTHON), "-m", "dynamo.indexer", "--help"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


pytestmark = pytest.mark.skipif(
    not _indexer_available(),
    reason="requires dynamo.indexer binary",
)


# --------------------------------------------------------------------------- #
# Indexer HTTP helpers.
# --------------------------------------------------------------------------- #


def _http(
    method: str, base: str, path: str, body: dict[str, object] | None = None
) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")


def _dump(base: str) -> dict[str, object]:
    status, text = _http("GET", base, "/dump")
    assert status == 200, f"/dump returned {status}: {text}"
    return json.loads(text)


def _query(base: str, token_ids: list[int]) -> dict[str, object]:
    status, text = _http(
        "POST", base, "/query", {"token_ids": token_ids, "model_name": MODEL_NAME}
    )
    assert status == 200, f"/query returned {status}: {text}"
    return json.loads(text)


def _dump_block_hashes(dump: dict[str, object], instance_id: int) -> set[int]:
    """Block hashes recorded in the radix tree under ``instance_id`` / dp 0."""
    entry = dump.get(f"{MODEL_NAME}:default")
    if not isinstance(entry, dict):
        return set()
    hashes: set[int] = set()
    for ev in entry.get("events", []):
        if ev.get("worker_id") != instance_id:
            continue
        event = ev.get("event", {})
        if event.get("dp_rank") != DP_RANK:
            continue
        stored = event.get("data", {}).get("stored")
        if not stored:
            continue
        for block in stored.get("blocks", []):
            hashes.add(block["block_hash"])
    return hashes


def _query_score(query: dict[str, object], instance_id: int) -> int:
    """Overlap score (matched tokens) for ``instance_id`` at dp 0, else 0."""
    scores = cast(dict[str, Any], query.get("scores", {}))
    per_instance = scores.get(str(instance_id), {})
    return int(per_instance.get(str(DP_RANK), 0))


def _poll_until(predicate, timeout_s: float = 5.0, interval_s: float = 0.2) -> bool:
    """Poll ``predicate`` until it is truthy or ``timeout_s`` elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def indexer() -> Generator[str, None, None]:
    """Start the Dynamo KV indexer in the isolated venv; yield its base URL."""
    port = _free_port()
    proc = subprocess.Popen(
        [
            str(_VENV_PYTHON),
            "-m",
            "dynamo.indexer",
            "--port",
            str(port),
            "--block-size",
            str(KV_BLOCK_SIZE),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    # Wait for the HTTP server to accept connections.
    ready = _poll_until(lambda: _http("GET", base, "/health")[0] == 200, timeout_s=15.0)
    if not ready:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.fail("dynamo indexer HTTP server did not become ready")

    try:
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_indexer_ingests_and_evicts_our_blocks(
    server: tuple[str, str],  # noqa: F811  -- shadows the imported fixture by design
    indexer: str,
) -> None:
    """Real store -> indexer radix tree; CLEAR -> blocks drop out."""
    if not torch.cuda.is_available():
        raise RuntimeError("this end-to-end test requires a CUDA device")

    mq_url, zmq_endpoint = server
    device = torch.device("cuda:0")
    instance_id = os.getpid()

    # Register our server's PUB endpoint with the indexer before any store.
    # Readiness is established below with a probe store, not by registration
    # ordering alone (the ZMQ SUB subscription propagates asynchronously).
    status, text = _http(
        "POST",
        indexer,
        "/register",
        {
            "instance_id": instance_id,
            "endpoint": zmq_endpoint,
            "model_name": MODEL_NAME,
            "block_size": KV_BLOCK_SIZE,
            "dp_rank": DP_RANK,
        },
    )
    assert status == 201, f"/register failed: {status} {text}"

    client = MessageQueueClient(server_url=mq_url, context=zmq.Context.instance())
    kv_tensors = _make_kv_cache(device)

    try:
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

        # Readiness probe. The ZMQ PUB/SUB link has a slow-joiner window:
        # events published before the indexer's SUB subscription has
        # propagated are silently dropped (the emitter has no replay channel).
        # Rather than guessing with a fixed sleep, store throwaway probe keys
        # until one actually lands in the indexer -- the first block we observe
        # proves the link is live. Each probe needs a *fresh* key: re-storing
        # an already-cached key is a no-op that never re-publishes (the storage
        # manager reserves only objects that do not yet exist).
        probe_event = torch.cuda.Event(interprocess=True)
        probe_event.record()
        probe_block_ids = list(
            range(NUM_KEYS * BLOCKS_PER_KEY, (NUM_KEYS + 1) * BLOCKS_PER_KEY)
        )
        probe_indices = itertools.count(NUM_KEYS)

        def _probe_link_live() -> bool:
            _store_key(
                client,
                _make_key(next(probe_indices), prefix="indexer_request"),
                instance_id,
                probe_block_ids,
                probe_event,
            )
            return len(_dump_block_hashes(_dump(indexer), instance_id)) > 0

        assert _poll_until(_probe_link_live, timeout_s=20.0, interval_s=0.5), (
            "indexer never received a probe block; ZMQ link never came up"
        )

        # Link confirmed live -> the real stores below will not be lost.
        keys = [_make_key(i, prefix="indexer_request") for i in range(NUM_KEYS)]
        event = torch.cuda.Event(interprocess=True)
        event.record()
        for i, key in enumerate(keys):
            block_ids = list(range(i * BLOCKS_PER_KEY, (i + 1) * BLOCKS_PER_KEY))
            _store_key(client, key, instance_id, block_ids, event)

        # Probe blocks already make the dump non-empty, so assert on the real
        # key specifically: poll until keys[0]'s prefix becomes queryable.
        ingested = _poll_until(
            lambda: (
                _query_score(_query(indexer, list(keys[0].token_ids)), instance_id) > 0
            ),
            timeout_s=5.0,
        )
        assert ingested, "indexer never ingested our real stored blocks"

        tree_hashes = _dump_block_hashes(_dump(indexer), instance_id)
        assert tree_hashes, "no blocks under our (instance_id, dp_rank=0)"

        # /query: stored token prefix must have a positive overlap score
        # nested under (instance_id, dp_rank=0).
        query = _query(indexer, list(keys[0].token_ids))
        assert _query_score(query, instance_id) > 0, (
            f"expected positive overlap score, got {query.get('scores')}"
        )

        # Trigger real eviction of everything via CLEAR.
        client.submit_request(
            RequestType.CLEAR, [], get_response_class(RequestType.CLEAR)
        ).result(timeout=DEFAULT_TIMEOUT)

        # Poll /dump until our blocks disappear from the tree.
        cleared = _poll_until(
            lambda: len(_dump_block_hashes(_dump(indexer), instance_id)) == 0,
            timeout_s=5.0,
        )
        assert cleared, "blocks did not drop out of the radix tree after CLEAR"

        # /query overlap for the same prefix must fall back to 0.
        query_after = _query(indexer, list(keys[0].token_ids))
        assert _query_score(query_after, instance_id) == 0, (
            f"expected zero overlap after CLEAR, got {query_after.get('scores')}"
        )
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
        del kv_tensors
        torch.cuda.empty_cache()
