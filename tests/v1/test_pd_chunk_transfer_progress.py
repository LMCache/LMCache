# SPDX-License-Identifier: Apache-2.0
"""Regression: PD transfer progress must survive chunked-prefill metadata rebuilds.

Under chunked prefill, the scheduler rebuilds ``ReqMeta`` / ``DisaggSpec`` each
step. Scheduler→worker metadata is one-way, so worker-side mutations to
``DisaggSpec.num_transferred_tokens`` are lost on the next step (reset to 0).

``wait_for_save`` gates PD store with::

    skip = min(save_spec.skip_leading_tokens, disagg_spec.num_transferred_tokens)

Without a worker-local watermark, the second chunk always sees
``num_transferred_tokens=0`` and re-transfers from token 0 (or, when LocalCPU
already advanced ``skip_leading_tokens``, incorrectly skips PD entirely).

This test locks in:

1. After a real PD store, progress is remembered on the worker.
2. When the next step arrives with ``num_transferred_tokens=0``, the watermark
   is restored and the already-transferred prefix is not stored again.
3. LocalCPU ``skip_leading_tokens`` alone must never seed PD progress (full
   LocalCPU hit with zero PD progress must still attempt PD store).
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    DisaggSpec,
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    SaveSpec,
)


class _FakeParent:
    def __init__(self, metadata: LMCacheConnectorMetadata) -> None:
        self._connector_metadata = metadata

    def _get_connector_metadata(self) -> LMCacheConnectorMetadata:
        return self._connector_metadata


class _FakeEngine:
    def __init__(self) -> None:
        self.unpinned: list[str] = []
        self.store_calls: list[dict] = []

    def lookup_unpin(self, req_id: str) -> None:
        self.unpinned.append(req_id)

    def store(self, token_ids, **kwargs) -> None:
        self.store_calls.append(
            {
                "req_id": kwargs.get("req_id"),
                "token_len": len(token_ids),
                "offset": kwargs.get("offset"),
                "mask_true": int(kwargs["mask"].sum().item())
                if kwargs.get("mask") is not None
                else None,
                "transfer_spec": kwargs.get("transfer_spec"),
            }
        )


def _make_disagg_request(
    req_id: str,
    *,
    token_len: int,
    skip_leading: int,
    transferred: int,
    is_last_prefill: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        req_id=req_id,
        token_ids=list(range(token_len)),
        slot_mapping=torch.arange(token_len, dtype=torch.long),
        save_spec=SaveSpec(skip_leading_tokens=skip_leading, can_save=True),
        disagg_spec=DisaggSpec(
            req_id=req_id,
            receiver_id="decoder-0",
            receiver_host="127.0.0.1",
            receiver_init_port=5555,
            receiver_alloc_port=5556,
            num_transferred_tokens=transferred,
        ),
        is_last_prefill=is_last_prefill,
        request_configs=None,
    )


def _make_connector(
    requests: list[SimpleNamespace],
    *,
    chunk_size: int = 256,
) -> tuple[LMCacheConnectorV1Impl, _FakeEngine]:
    metadata = LMCacheConnectorMetadata(requests=requests)  # type: ignore[arg-type]
    engine = _FakeEngine()
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent(metadata)
    connector._manager = SimpleNamespace(  # type: ignore[assignment]
        lmcache_engine=engine
    )
    connector.kv_role = "kv_producer"
    connector.use_layerwise = False
    connector.enable_blending = False
    connector.device = "cpu"
    connector._lmcache_chunk_size = chunk_size
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector.config = SimpleNamespace(pd_bidirectional=False)
    connector._pd_transferred_tokens = {}
    return connector, engine


@pytest.fixture
def last_pp_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """``wait_for_save`` only advances watermarks on the last PP rank."""
    pp = MagicMock()
    pp.is_last_rank = True
    monkeypatch.setattr(
        "lmcache.integration.vllm.vllm_v1_adapter.get_pp_group",
        lambda: pp,
    )


def test_pd_progress_persists_across_chunked_prefill_steps(
    last_pp_rank: None,
) -> None:
    """Second chunk must not re-store the prefix already transferred."""
    chunk = 256
    # Step 1: first aligned chunk (0..512) with no prior PD progress.
    req = _make_disagg_request(
        "req-pd",
        token_len=512,
        skip_leading=0,
        transferred=0,
        is_last_prefill=False,
    )
    connector, engine = _make_connector([req], chunk_size=chunk)

    connector.wait_for_save()

    assert len(engine.store_calls) == 1
    assert engine.store_calls[0]["offset"] == 0
    assert connector._pd_transferred_tokens["req-pd"] == 512
    assert req.disagg_spec.num_transferred_tokens == 512

    # Step 2: scheduler rebuilds DisaggSpec with transferred=0 (metadata loss),
    # while LocalCPU skip may already cover the first chunk.
    req2 = _make_disagg_request(
        "req-pd",
        token_len=1024,
        skip_leading=512,
        transferred=0,
        is_last_prefill=False,
    )
    connector._parent = _FakeParent(
        LMCacheConnectorMetadata(requests=[req2])  # type: ignore[arg-type]
    )
    engine.store_calls.clear()

    connector.wait_for_save()

    assert req2.disagg_spec.num_transferred_tokens == 1024
    assert connector._pd_transferred_tokens["req-pd"] == 1024
    assert len(engine.store_calls) == 1
    # Restored watermark (512) must gate the store offset — not re-send 0..512.
    assert engine.store_calls[0]["offset"] == 512


def test_local_cpu_skip_does_not_seed_pd_progress(last_pp_rank: None) -> None:
    """Full LocalCPU hit must still PD-store when peer progress is zero."""
    token_len = 512
    req = _make_disagg_request(
        "req-local-hit",
        token_len=token_len,
        skip_leading=token_len,  # LocalCPU already has everything
        transferred=0,  # but nothing sent to decoder yet
        is_last_prefill=True,
    )
    connector, engine = _make_connector([req], chunk_size=256)

    connector.wait_for_save()

    # Must NOT skip: min(skip_leading, transferred) without restore would be 0,
    # and skip_leading alone must not inflate transferred.
    assert len(engine.store_calls) == 1
    assert engine.store_calls[0]["offset"] == 0
    assert connector._pd_transferred_tokens["req-local-hit"] == token_len
    # After store, PD progress advances; it was never seeded from skip_leading
    # before the store.
    assert req.disagg_spec.num_transferred_tokens == token_len


def test_get_finished_clears_pd_watermark(last_pp_rank: None) -> None:
    connector, _engine = _make_connector([], chunk_size=256)
    connector._pd_transferred_tokens["req-done"] = 1024
    connector._pd_transferred_tokens["req-keep"] = 256

    connector.get_finished({"req-done"})

    assert "req-done" not in connector._pd_transferred_tokens
    assert connector._pd_transferred_tokens["req-keep"] == 256
