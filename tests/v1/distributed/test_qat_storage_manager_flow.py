# SPDX-License-Identifier: Apache-2.0
"""
StorageManager-level integration test for QAT + DramL2Adapter.

Verifies the full wiring WITHOUT vLLM:
  StorageManager config → StoreController → SerdeL2AdapterWrapper
  → AccelCompressSerializer → DramL2Adapter store → skip_l1 policy deletes L1
  → prefetch → DramL2Adapter load → AccelCompressDeserializer → verify

Requires:
  - KVCLIP_QZIP_LIB_PATH pointing to libkvclip_qzip.so
  - No GPU needed (uses lazy CPU allocator)

Run:
  KVCLIP_QZIP_LIB_PATH=/path/to/libkvclip_qzip.so \
    python -m pytest tests/v1/distributed/test_qat_storage_manager_flow.py -v -s
"""

# Standard
import os
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.dram_l2_adapter import DramL2AdapterConfig
from lmcache.v1.distributed.serde import SerdeConfig
from lmcache.v1.distributed.storage_manager import StorageManager

# Skip if QAT library not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("KVCLIP_QZIP_LIB_PATH"),
    reason="KVCLIP_QZIP_LIB_PATH not set",
)


# =============================================================================
# Helpers
# =============================================================================


def _make_object_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="qat_test_model",
        kv_rank=0,
    )


def _make_layout() -> MemoryLayoutDesc:
    """KV layout: [2, num_layers, num_tokens, hidden_dim] in bf16.

    2 = key + value, 2 layers, 16 tokens, 256 hidden_dim (4 heads × 64 dim).
    Total per object: 2×2×16×256×2 = 32768 bytes.
    """
    return MemoryLayoutDesc(
        shapes=[torch.Size([2, 2, 16, 256])],
        dtypes=[torch.bfloat16],
    )


def _make_storage_manager() -> StorageManager:
    """Build StorageManager with DramL2Adapter + accel_kv_compress + skip_l1."""
    adapter_cfg = DramL2AdapterConfig(max_size_gb=0.1)  # 100 MB
    adapter_cfg.serde_config = SerdeConfig(
        type="accel_kv_compress",
        kwargs={
            "backend": "qat",
            "byte_reorder": True,
            "truncate_bits": 0,  # lossless for exact verification
            "element_size": 2,
            "max_workers": 2,
        },
    )

    cfg = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=64 * 1024 * 1024,  # 64 MB
                use_lazy=True,
                init_size_in_bytes=32 * 1024 * 1024,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=[adapter_cfg]),
        store_policy="skip_l1",
    )
    return StorageManager(cfg)


def _wait_for_condition(
    predicate,
    timeout: float = 30.0,
    poll_interval: float = 0.05,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def _wait_for_prefetch(
    sm: StorageManager,
    handle,
    timeout: float = 30.0,
) -> Bitmap | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = sm.query_prefetch_status(handle)
        if result is not None:
            return result
        time.sleep(0.05)
    return None


def _finish_read_until_clean(sm: StorageManager, keys: list[ObjectKey]) -> None:
    for _ in range(4):
        sm.finish_read_prefetched(keys)
        ok = _wait_for_condition(
            lambda: (
                sm.report_status()["l1_manager"]["read_locked_count"] == 0
                and sm.report_status()["l1_manager"]["write_locked_count"] == 0
                and sm.report_status()["l1_manager"]["temporary_count"] == 0
            ),
            timeout=30.0,
        )
        if ok:
            return


# =============================================================================
# Tests
# =============================================================================


class TestStorageManagerQatFlow:
    """End-to-end StorageManager test with QAT compression + DramL2Adapter."""

    def test_store_compress_delete_l1_then_prefetch(self):
        """Full cycle: write L1 → compress+store DramL2 → delete L1 → prefetch.

        Verifies:
        1. reserve_write + finish_write injects KV data into L1
        2. StoreController compresses via QAT and stores to DramL2
        3. skip_l1 policy deletes raw from L1 after store
        4. prefetch loads from DramL2, decompresses, data matches original
        """
        sm = _make_storage_manager()
        layout = _make_layout()
        keys = [_make_object_key(i) for i in range(3)]

        try:
            # Step 1: Write KV data to L1
            ret = sm.reserve_write(keys, layout, mode="new")
            assert len(ret) == len(keys)

            original_by_key = {}
            for i, key in enumerate(keys):
                obj = ret[key]
                assert obj.tensor is not None
                # Fill with deterministic random data
                torch.manual_seed(42 + i)
                data = torch.randn(
                    obj.tensor.shape,
                    dtype=obj.tensor.dtype,
                    device=obj.tensor.device,
                )
                obj.tensor.copy_(data)
                original_by_key[key] = data.detach().clone()

            sm.finish_write(list(ret.keys()))

            # Step 2: Wait for store to complete (compress + store to DramL2)
            ok = _wait_for_condition(
                lambda: (
                    sm.report_status()["store_controller"]["in_flight_task_count"] == 0
                    and sm.report_status()["store_controller"]["pending_keys_count"] == 0
                    and sm.report_status()["l1_manager"]["write_locked_count"] == 0
                    and sm.report_status()["l1_manager"]["read_locked_count"] == 0
                    and sm.report_status()["l1_manager"]["temporary_count"] == 0
                ),
                timeout=60.0,
            )
            assert ok, (
                f"Store did not complete: {sm.report_status()['store_controller']}"
            )

            # Step 3: Verify skip_l1 policy deleted raw data from L1
            # (with skip_l1, L1 should be empty after store completes)
            ok = _wait_for_condition(
                lambda: sm.report_status()["l1_manager"]["total_object_count"] == 0,
                timeout=10.0,
            )
            assert ok, (
                f"L1 not cleared by skip_l1 policy: "
                f"{sm.report_status()['l1_manager']}"
            )

            # Step 4: Prefetch from DramL2 (decompress)
            handle = sm.submit_prefetch_task(keys, layout)
            hit_bitmap = _wait_for_prefetch(sm, handle, timeout=60.0)
            assert hit_bitmap is not None
            hits = hit_bitmap.count_leading_ones()
            assert hits == len(keys), f"Expected {len(keys)} hits, got {hits}"

            # Step 5: Read and verify data matches original
            with sm.read_prefetched_results(keys) as objs:
                assert objs is not None
                assert len(objs) == len(keys)

                for key, obj in zip(keys, objs, strict=True):
                    assert obj.tensor is not None
                    recovered = obj.tensor
                    original = original_by_key[key]

                    # With truncate_bits=0 and byte_reorder (self-inverse),
                    # roundtrip should be bit-exact
                    assert torch.equal(recovered, original), (
                        f"Data mismatch for key {key}!\n"
                        f"Max diff: {(recovered.float() - original.float()).abs().max().item()}"
                    )

            _finish_read_until_clean(sm, keys)
            print("\n  StorageManager QAT flow: PASS (bit-exact roundtrip)")

        finally:
            sm.close()

    def test_store_with_truncation_lossy(self):
        """Same flow but with truncate_bits=2 — verify data is close but lossy."""
        adapter_cfg = DramL2AdapterConfig(max_size_gb=0.1)
        adapter_cfg.serde_config = SerdeConfig(
            type="accel_kv_compress",
            kwargs={
                "backend": "qat",
                "byte_reorder": True,
                "truncate_bits": 2,
                "element_size": 2,
                "max_workers": 2,
            },
        )

        cfg = StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=64 * 1024 * 1024,
                    use_lazy=True,
                    init_size_in_bytes=32 * 1024 * 1024,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=[adapter_cfg]),
            store_policy="skip_l1",
        )
        sm = StorageManager(cfg)
        layout = _make_layout()
        keys = [_make_object_key(100 + i) for i in range(3)]

        try:
            ret = sm.reserve_write(keys, layout, mode="new")
            original_by_key = {}
            for i, key in enumerate(keys):
                obj = ret[key]
                torch.manual_seed(100 + i)
                data = torch.randn(
                    obj.tensor.shape, dtype=obj.tensor.dtype,
                    device=obj.tensor.device,
                )
                obj.tensor.copy_(data)
                original_by_key[key] = data.detach().clone()

            sm.finish_write(list(ret.keys()))

            ok = _wait_for_condition(
                lambda: (
                    sm.report_status()["store_controller"]["in_flight_task_count"] == 0
                    and sm.report_status()["l1_manager"]["total_object_count"] == 0
                ),
                timeout=60.0,
            )
            assert ok

            handle = sm.submit_prefetch_task(keys, layout)
            hit_bitmap = _wait_for_prefetch(sm, handle, timeout=60.0)
            assert hit_bitmap is not None
            assert hit_bitmap.count_leading_ones() == len(keys)

            with sm.read_prefetched_results(keys) as objs:
                for key, obj in zip(keys, objs, strict=True):
                    recovered = obj.tensor.float().flatten()
                    original = original_by_key[key].float().flatten()

                    # Lossy: correlation should be high but not bit-exact
                    corr = torch.corrcoef(torch.stack([original, recovered]))[0, 1].item()
                    assert corr > 0.95, f"Low correlation {corr} for key {key}"

            _finish_read_until_clean(sm, keys)
            print(f"\n  Lossy (truncate_bits=2): correlation > 0.95 PASS")

        finally:
            sm.close()


class TestCliConfigParsing:
    """Validate CLI JSON parsing for DramL2Adapter + accel_kv_compress serde."""

    def test_cli_json_parses_to_correct_config(self):
        """--l2-adapter JSON + --l2-store-policy skip_l1 parse correctly."""
        import json
        from lmcache.v1.distributed.config import get_arg_parser
        from lmcache.v1.distributed.l2_adapters.config import (
            parse_args_to_l2_adapters_config,
        )

        cli_args = [
            "--l1-size-gb", "1.0",
            "--eviction-policy", "LRU",
            "--l2-adapter", json.dumps({
                "type": "dram",
                "max_size_gb": 8.0,
                "serde": {
                    "type": "accel_kv_compress",
                    "backend": "qat",
                    "byte_reorder": True,
                    "truncate_bits": 2,
                    "element_size": 2,
                    "max_workers": 4,
                },
            }),
            "--l2-store-policy", "skip_l1",
        ]

        parser = get_arg_parser()
        args = parser.parse_args(cli_args)

        assert args.l2_store_policy == "skip_l1"

        l2_cfg = parse_args_to_l2_adapters_config(args)
        assert len(l2_cfg.adapters) == 1

        adapter = l2_cfg.adapters[0]
        assert type(adapter).__name__ == "DramL2AdapterConfig"
        assert adapter.max_size_gb == 8.0
        assert adapter.serde_config is not None
        assert adapter.serde_config.type == "accel_kv_compress"
        assert adapter.serde_config.kwargs["backend"] == "qat"
        assert adapter.serde_config.kwargs["byte_reorder"] is True
        assert adapter.serde_config.kwargs["truncate_bits"] == 2
        assert adapter.serde_config.kwargs["element_size"] == 2
        assert adapter.serde_config.kwargs["max_workers"] == 4
