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

    @staticmethod
    def _make_mla_3d_kv_tensors(
        num_layers: int = 2,
        num_blocks: int = 4,
        block_size: int = 4,
        head_size: int = 8,
        dtype: torch.dtype = torch.float32,
    ) -> list[torch.Tensor]:
        """Create MLA-style 3D CPU tensors: ``[NB, BS, HS]``."""
        torch.manual_seed(0)
        return [
            torch.randn(
                num_blocks,
                block_size,
                head_size,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

    @staticmethod
    def _make_4d_kv_tensors(
        num_layers: int = 2,
        num_blocks: int = 4,
        block_size: int = 4,
        num_heads: int = 2,
        head_size: int = 8,
        dtype: torch.dtype = torch.float32,
    ) -> list[torch.Tensor]:
        """Create 4D CPU tensors: ``[NB, BS, NH, HS]``."""
        torch.manual_seed(0)
        return [
            torch.randn(
                num_blocks,
                block_size,
                num_heads,
                head_size,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

    def test_mla_3d_tensor_no_crash(self):
        """Real MLA 3D tensors ``[NB, BS, HS]`` must not crash.

        Regression for Bugbot ``#3151480582``: the old reshape
        hard-coded a 5D layout and silently produced wrong
        checksums for 3D MLA tensors coming from vLLM.
        """
        kv = self._make_mla_3d_kv_tensors()
        assert kv[0].ndim == 3
        result = _compute_mp_checksums(
            kv,
            slot_indices=[0, 1, 2, 3],
            chunk_size=2,
            layerwise=False,
        )
        assert result["status"] == "success"
        assert result["num_chunks"] == 2
        lw = _compute_mp_checksums(
            kv,
            slot_indices=[0, 1, 2, 3],
            chunk_size=2,
            layerwise=True,
        )
        assert lw["layerwise"] is True
        assert "layer_0" in lw["chunk_checksums"]
        assert "layer_1" in lw["chunk_checksums"]

    def test_4d_tensor_no_crash(self):
        """4D tensors ``[NB, BS, NH, HS]`` must be supported."""
        kv = self._make_4d_kv_tensors()
        assert kv[0].ndim == 4
        result = _compute_mp_checksums(
            kv,
            slot_indices=[0, 1, 2, 3],
            chunk_size=2,
            layerwise=False,
        )
        assert result["status"] == "success"
        assert result["num_chunks"] == 2

    def test_mla_3d_matches_equivalent_5d(self):
        """3D MLA and equivalent 5D ``kv_size=1`` yield same data.

        The 5D form ``[1, NB, BS, 1, HS]`` carries the exact
        same bytes as the 3D form ``[NB, BS, HS]``; the
        checksum computation should therefore produce the
        same result for matching slots.
        """
        torch.manual_seed(7)
        nb, bs, hs = 4, 4, 8
        base = torch.randn(nb, bs, hs, dtype=torch.float32)
        kv_3d = [base.clone()]
        kv_5d = [base.view(1, nb, bs, 1, hs).clone()]
        r_3d = _compute_mp_checksums(
            kv_3d,
            slot_indices=[0, 1, 5, 7],
            chunk_size=2,
            layerwise=False,
        )
        r_5d = _compute_mp_checksums(
            kv_5d,
            slot_indices=[0, 1, 5, 7],
            chunk_size=2,
            layerwise=False,
        )
        assert r_3d["chunk_checksums"] == r_5d["chunk_checksums"]

    def test_unsupported_ndim_raises(self):
        """An unsupported ndim should raise ``ValueError``."""
        bad = [torch.randn(2, 3, dtype=torch.float32)]
        try:
            _compute_mp_checksums(
                bad,
                slot_indices=[0],
                chunk_size=1,
                layerwise=False,
            )
        except ValueError:
            return
        raise AssertionError("expected ValueError for 2D tensor")


class TestSlotTensorDeviceAlignment:
    """Regression for the Bugbot review: the slot index tensor
    passed to :func:`extract_kv_at_slots` must live on the same
    device as the per-layer KV tensor, to avoid implicit H2D
    copies (or failures on PyTorch/CUDA builds that forbid
    mixed-device indexing).

    The tests run on CPU; we verify the invariants via a
    monkeypatched stub that records every ``slot_tensor.device``
    it observes.
    """

    @staticmethod
    def _make_kv_list(num_layers: int) -> list[torch.Tensor]:
        # Same shape layout as TestComputeMpChecksums; small
        # random tensors are enough for device-alignment checks.
        kv_shape = (2, 2, 4, 2, 4)  # [2, NB, BS, NH, HS]
        torch.manual_seed(123)
        return [torch.randn(*kv_shape, dtype=torch.float32) for _ in range(num_layers)]

    def test_slot_tensor_device_follows_kv_device(self, monkeypatch):
        # First Party
        from lmcache.v1.multiprocess.http_apis import cache_api as mp_api

        seen_devices: list[torch.device] = []

        def fake_extract(kv_tensor, slot_tensor):
            seen_devices.append(slot_tensor.device)
            assert slot_tensor.device == kv_tensor.device, (
                "slot_tensor.device must match kv_tensor.device"
            )
            flat = kv_tensor.reshape(kv_tensor.shape[0], -1, *kv_tensor.shape[3:])
            return flat[:, slot_tensor]

        monkeypatch.setattr(mp_api, "extract_kv_at_slots", fake_extract)

        kvs = self._make_kv_list(num_layers=3)
        result = _compute_mp_checksums(
            kvs, slot_indices=[0, 1, 2, 3], chunk_size=2, layerwise=False
        )
        assert result["status"] == "success"
        assert len(seen_devices) == 3
        assert all(d == torch.device("cpu") for d in seen_devices)

    def test_slot_tensor_is_cached_per_device(self, monkeypatch):
        """With N layers on the same device, the slot tensor is
        built once and the same object is reused."""
        # First Party
        from lmcache.v1.multiprocess.http_apis import cache_api as mp_api

        ids_seen: set[int] = set()

        def fake_extract(kv_tensor, slot_tensor):
            ids_seen.add(id(slot_tensor))
            flat = kv_tensor.reshape(kv_tensor.shape[0], -1, *kv_tensor.shape[3:])
            return flat[:, slot_tensor]

        monkeypatch.setattr(mp_api, "extract_kv_at_slots", fake_extract)

        kvs = self._make_kv_list(num_layers=4)
        _compute_mp_checksums(
            kvs, slot_indices=[0, 1, 2], chunk_size=2, layerwise=False
        )
        assert len(ids_seen) == 1
