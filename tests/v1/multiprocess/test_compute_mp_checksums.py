# SPDX-License-Identifier: Apache-2.0
# Standard

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.http_apis.cache_api import _compute_mp_checksums


class TestComputeMpChecksums:
    """Test cases for _compute_mp_checksums helper."""

    @staticmethod
    def _make_cpu_kv_tensors(
        num_layers: int = 2,
        num_blocks: int = 4,
        block_size: int = 4,
        num_heads: int = 2,
        head_size: int = 8,
        dtype: torch.dtype = torch.float32,
        kv_size: int = 2,
    ) -> list[torch.Tensor]:
        """Create deterministic CPU KV tensors.

        ``kv_size`` is 2 for standard K/V layouts and 1 for
        MLA, matching the leading dimension accepted by
        :func:`_compute_mp_checksums`.
        """
        torch.manual_seed(0)
        return [
            torch.randn(
                kv_size,
                num_blocks,
                block_size,
                num_heads,
                head_size,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

    def test_non_layerwise_returns_expected_keys(self):
        """Non-layerwise result has the right structure."""
        kv = self._make_cpu_kv_tensors()
        result = _compute_mp_checksums(
            kv,
            slot_indices=[0, 1, 2, 3],
            chunk_size=2,
            layerwise=False,
        )
        assert result["status"] == "success"
        assert result["layerwise"] is False
        assert result["chunk_size"] == 2
        assert result["num_chunks"] == 2
        assert len(result["chunk_checksums"]) == 2

    def test_layerwise_returns_expected_keys(self):
        """Layerwise result has per-layer checksum lists."""
        kv = self._make_cpu_kv_tensors(num_layers=3)
        result = _compute_mp_checksums(
            kv,
            slot_indices=[0, 1, 2, 3],
            chunk_size=4,
            layerwise=True,
        )
        assert result["status"] == "success"
        assert result["layerwise"] is True
        assert result["num_chunks"] == 1
        checksums = result["chunk_checksums"]
        assert len(checksums) == 3
        for i in range(3):
            assert "layer_%d" % i in checksums
            assert len(checksums["layer_%d" % i]) == 1

    def test_deterministic_output(self):
        """Same input produces the same checksums."""
        kv = self._make_cpu_kv_tensors()
        slots = [0, 1]
        r1 = _compute_mp_checksums(
            kv,
            slots,
            chunk_size=2,
            layerwise=False,
        )
        r2 = _compute_mp_checksums(
            kv,
            slots,
            chunk_size=2,
            layerwise=False,
        )
        assert r1["chunk_checksums"] == r2["chunk_checksums"]

    def test_chunk_boundary(self):
        """Partial last chunk is handled correctly."""
        kv = self._make_cpu_kv_tensors()
        # 3 slots with chunk_size=2 -> 2 chunks (2 + 1)
        result = _compute_mp_checksums(
            kv,
            slot_indices=[0, 1, 2],
            chunk_size=2,
            layerwise=False,
        )
        assert result["num_chunks"] == 2
        assert len(result["chunk_checksums"]) == 2

    def test_single_slot_single_chunk(self):
        """Single slot produces exactly one chunk."""
        kv = self._make_cpu_kv_tensors()
        result = _compute_mp_checksums(
            kv,
            slot_indices=[0],
            chunk_size=1,
            layerwise=False,
        )
        assert result["num_chunks"] == 1
        assert len(result["chunk_checksums"]) == 1

    def test_checksums_are_valid_md5(self):
        """All returned checksums are 32-char hex strings."""
        kv = self._make_cpu_kv_tensors()
        result = _compute_mp_checksums(
            kv,
            slot_indices=[0, 1],
            chunk_size=1,
            layerwise=False,
        )
        for cksum in result["chunk_checksums"]:
            assert len(cksum) == 32
            int(cksum, 16)  # must be valid hex

    def test_bfloat16_does_not_crash(self):
        """bfloat16 tensors must not raise TypeError."""
        kv = self._make_cpu_kv_tensors(dtype=torch.bfloat16)
        result = _compute_mp_checksums(
            kv,
            slot_indices=[0, 1, 2, 3],
            chunk_size=2,
            layerwise=False,
        )
        assert result["status"] == "success"
        assert len(result["chunk_checksums"]) == 2

    def test_bfloat16_layerwise(self):
        """bfloat16 works in layerwise mode too."""
        kv = self._make_cpu_kv_tensors(
            num_layers=2,
            dtype=torch.bfloat16,
        )
        result = _compute_mp_checksums(
            kv,
            slot_indices=[0, 1],
            chunk_size=2,
            layerwise=True,
        )
        assert result["layerwise"] is True
        assert "layer_0" in result["chunk_checksums"]
        assert "layer_1" in result["chunk_checksums"]

    def test_bfloat16_deterministic(self):
        """bfloat16 checksums are deterministic across calls."""
        kv = self._make_cpu_kv_tensors(dtype=torch.bfloat16)
        slots = [0, 1]
        r1 = _compute_mp_checksums(
            kv,
            slots,
            chunk_size=2,
            layerwise=False,
        )
        r2 = _compute_mp_checksums(
            kv,
            slots,
            chunk_size=2,
            layerwise=False,
        )
        assert r1["chunk_checksums"] == r2["chunk_checksums"]

    def test_mla_kv_size_one(self):
        """MLA-style tensors (kv_size=1) must not crash reshape.

        Regression for Bugbot #3147088004: a hardcoded ``2`` in
        ``kv.reshape`` used to break for MLA layouts. We now
        derive the leading dim from ``kv.shape[0]``.
        """
        kv = self._make_cpu_kv_tensors(kv_size=1)
        assert kv[0].shape[0] == 1
        # non-layerwise path
        result = _compute_mp_checksums(
            kv,
            slot_indices=[0, 1, 2, 3],
            chunk_size=2,
            layerwise=False,
        )
        assert result["status"] == "success"
        assert len(result["chunk_checksums"]) == 2
        # layerwise path
        lw = _compute_mp_checksums(
            kv,
            slot_indices=[0, 1, 2, 3],
            chunk_size=2,
            layerwise=True,
        )
        assert lw["layerwise"] is True
        assert "layer_0" in lw["chunk_checksums"]

    def test_different_dtypes_produce_different_checksums(self):
        """float32 and bfloat16 should produce different checksums."""
        kv_f32 = self._make_cpu_kv_tensors(dtype=torch.float32)
        kv_bf16 = self._make_cpu_kv_tensors(dtype=torch.bfloat16)
        r_f32 = _compute_mp_checksums(
            kv_f32,
            slot_indices=[0, 1],
            chunk_size=2,
            layerwise=False,
        )
        r_bf16 = _compute_mp_checksums(
            kv_bf16,
            slot_indices=[0, 1],
            chunk_size=2,
            layerwise=False,
        )
        # bfloat16 is cast to float32 internally, but the
        # underlying data differs due to precision loss, so
        # checksums should differ.
        assert r_f32["chunk_checksums"] != r_bf16["chunk_checksums"]
