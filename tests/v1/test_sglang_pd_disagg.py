# SPDX-License-Identifier: Apache-2.0
# tests/v1/test_sglang_pd_disagg.py
#
# Unit tests for the SGLang PD (prefill/decode) disaggregation glue in
# lmcache.integration.sglang.sglang_adapter: the DisaggSpec routing struct,
# its from_dict parser, and that LMCachePDConnector forwards transfer_spec
# into engine.store while a plain store omits it.

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
# DisaggSpec lives in a dependency-free module so its parsing/validation can be
# tested without a full SGLang (or torch/GPU) install.
from lmcache.integration.sglang.pd_types import DisaggSpec


def _import_adapter():
    """Import the sglang adapter or skip: it imports `sglang` at module scope,
    which is bind-mounted at runtime and absent from the LMCache test image."""
    return pytest.importorskip(
        "lmcache.integration.sglang.sglang_adapter",
        reason="sglang is not installed",
    )


def _valid_spec_dict() -> dict:
    return {
        "req_id": "req-123",
        "receiver_host": "10.0.0.7",
        "receiver_init_port": [5600, 5601],
        "receiver_alloc_port": [5700, 5701],
    }


class TestDisaggSpecFromDict:
    def test_parses_required_fields(self):
        spec = DisaggSpec.from_dict(_valid_spec_dict())
        assert spec.req_id == "req-123"
        assert spec.receiver_host == "10.0.0.7"
        assert spec.receiver_init_port == [5600, 5601]
        assert spec.receiver_alloc_port == [5700, 5701]
        # Defaults: query ports empty, single-shot store marks last prefill.
        assert spec.receiver_query_port == []
        assert spec.is_last_prefill is True

    def test_parses_optional_fields(self):
        spec_dict = _valid_spec_dict()
        spec_dict["receiver_query_port"] = [5800, 5801]
        spec_dict["is_last_prefill"] = False
        spec = DisaggSpec.from_dict(spec_dict)
        assert spec.receiver_query_port == [5800, 5801]
        assert spec.is_last_prefill is False

    @pytest.mark.parametrize(
        "missing_key",
        ["req_id", "receiver_host", "receiver_init_port", "receiver_alloc_port"],
    )
    def test_missing_required_key_raises(self, missing_key):
        spec_dict = _valid_spec_dict()
        del spec_dict[missing_key]
        with pytest.raises(ValueError, match="missing required keys"):
            DisaggSpec.from_dict(spec_dict)

    @pytest.mark.parametrize(
        "bad_ports",
        [5600, "5600", [5600, "x"], None],
    )
    def test_non_list_ports_raise(self, bad_ports):
        spec_dict = _valid_spec_dict()
        spec_dict["receiver_init_port"] = bad_ports
        with pytest.raises(ValueError, match="must be a list\\[int\\]"):
            DisaggSpec.from_dict(spec_dict)

    def test_ports_are_copied_not_aliased(self):
        spec_dict = _valid_spec_dict()
        spec = DisaggSpec.from_dict(spec_dict)
        spec_dict["receiver_init_port"].append(9999)
        assert spec.receiver_init_port == [5600, 5601]


class TestStoreMetadataTransferSpec:
    def test_default_transfer_spec_is_none(self):
        StoreMetadata = _import_adapter().StoreMetadata
        md = StoreMetadata(
            last_node=None,
            token_ids=[1, 2, 3],
            kv_indices=MagicMock(),
            offset=0,
        )
        assert md.transfer_spec is None


class TestLMCachePDConnectorForwarding:
    """LMCachePDConnector.store_kv must forward transfer_spec into
    engine.store; a None spec must still store (local-only)."""

    def _make_connector(self):
        # Bypass __init__ (which builds a real LMCache engine) and inject the
        # minimal state store_kv touches.
        LMCachePDConnector = _import_adapter().LMCachePDConnector
        conn = LMCachePDConnector.__new__(LMCachePDConnector)
        conn.lmcache_engine = MagicMock()
        conn.kvcaches = [MagicMock()]
        return conn

    def _store_metadata(self, transfer_spec):
        # kv_indices must survive .to(int64).to(device); a MagicMock returns
        # itself from chained calls, and len() is stubbed to match token_ids.
        StoreMetadata = _import_adapter().StoreMetadata
        kv_indices = MagicMock()
        kv_indices.to.return_value = kv_indices
        kv_indices.__len__.return_value = 3
        return StoreMetadata(
            last_node=None,
            token_ids=[1, 2, 3],
            kv_indices=kv_indices,
            offset=0,
            request_id="req-123",
            transfer_spec=transfer_spec,
        )

    def test_forwards_transfer_spec_to_engine_store(self, monkeypatch):
        # torch.tensor(...).to(...) and torch.ones_like(...) are exercised by
        # the base store_kv; keep them real but tiny.
        conn = self._make_connector()
        spec = DisaggSpec.from_dict(_valid_spec_dict())
        conn.store_kv(self._store_metadata(spec))

        conn.lmcache_engine.store.assert_called_once()
        _, kwargs = conn.lmcache_engine.store.call_args
        assert kwargs["transfer_spec"] is spec

    def test_none_transfer_spec_still_stores(self):
        conn = self._make_connector()
        conn.store_kv(self._store_metadata(None))

        conn.lmcache_engine.store.assert_called_once()
        _, kwargs = conn.lmcache_engine.store.call_args
        assert kwargs["transfer_spec"] is None
