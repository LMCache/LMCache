# SPDX-License-Identifier: Apache-2.0
"""Unit tests for lmcache.v1.utils.kv_slot_ops."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.utils.kv_slot_ops import (
    extract_kv_at_slots,
    slice_by_slot_dim,
)


class TestExtractKvAtSlots:
    """Behavioural tests covering the three supported layouts."""

    def test_mha_5d(self):
        """MHA layout: [2, NB, BS, NH, HS] -> [2, NS, NH, HS]."""
        nb, bs, nh, hs = 3, 4, 2, 8
        kv = torch.arange(2 * nb * bs * nh * hs, dtype=torch.float32).reshape(
            2, nb, bs, nh, hs
        )
        slots = torch.tensor([0, 5, 11], dtype=torch.long)

        out = extract_kv_at_slots(kv, slots)

        assert out.shape == (2, 3, nh, hs)
        # Reference: flatten (NB,BS) then index directly.
        ref = kv.reshape(2, nb * bs, nh, hs)[:, slots, :, :]
        assert torch.equal(out, ref)

    def test_4d(self):
        """4D layout: [NB, BS, NH, HS] -> [NS, NH, HS]."""
        nb, bs, nh, hs = 2, 3, 2, 4
        kv = torch.arange(nb * bs * nh * hs, dtype=torch.float32).reshape(
            nb, bs, nh, hs
        )
        slots = torch.tensor([0, 2, 5], dtype=torch.long)

        out = extract_kv_at_slots(kv, slots)

        assert out.shape == (3, nh, hs)
        ref = kv.reshape(nb * bs, nh, hs)[slots, :, :]
        assert torch.equal(out, ref)

    def test_mla_3d(self):
        """MLA layout: [NB, BS, HS] -> [NS, HS]."""
        nb, bs, hs = 2, 4, 6
        kv = torch.arange(nb * bs * hs, dtype=torch.float32).reshape(nb, bs, hs)
        slots = torch.tensor([0, 3, 7], dtype=torch.long)

        out = extract_kv_at_slots(kv, slots)

        assert out.shape == (3, hs)
        ref = kv.reshape(nb * bs, hs)[slots, :]
        assert torch.equal(out, ref)

    def test_unsupported_ndim_raises(self):
        """Unknown layouts should fail loudly, not silently corrupt."""
        bad = torch.randn(2, 3, dtype=torch.float32)
        with pytest.raises(ValueError, match="Unsupported kv_tensor"):
            extract_kv_at_slots(bad, torch.tensor([0], dtype=torch.long))


class TestSliceBySlotDim:
    """Slicing must hit the slot dimension regardless of layout."""

    def test_slice_4d_result_mha(self):
        """MHA extract output [2, NS, NH, HS] sliced on dim=1."""
        t = torch.arange(2 * 4 * 2 * 3, dtype=torch.float32).reshape(2, 4, 2, 3)
        out = slice_by_slot_dim(t, 1, 3)
        assert out.shape == (2, 2, 2, 3)
        assert torch.equal(out, t[:, 1:3, :, :])

    def test_slice_3d_result_4d(self):
        """4D extract output [NS, NH, HS] sliced on dim=0."""
        t = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
        out = slice_by_slot_dim(t, 1, 3)
        assert out.shape == (2, 2, 3)
        assert torch.equal(out, t[1:3, :, :])

    def test_slice_2d_result_mla(self):
        """MLA extract output [NS, HS] sliced on dim=0."""
        t = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
        out = slice_by_slot_dim(t, 1, 3)
        assert out.shape == (2, 3)
        assert torch.equal(out, t[1:3, :])


class TestRoundTripWithExtract:
    """Extract + slice should give the same contiguous bytes
    regardless of input layout, as long as (NB, BS) packing of
    slot_indices is preserved."""

    def test_3d_equals_5d_bytes_for_matching_data(self):
        nb, bs, hs = 2, 4, 6
        base = torch.arange(nb * bs * hs, dtype=torch.float32).reshape(nb, bs, hs)
        # Equivalent 5D view with KV=1, NH=1.
        kv_3d = base.clone()
        kv_5d = base.view(1, nb, bs, 1, hs).clone()
        slots = torch.tensor([0, 1, 5, 7], dtype=torch.long)

        out_3d = extract_kv_at_slots(kv_3d, slots)  # [4, 6]
        out_5d = extract_kv_at_slots(kv_5d, slots)  # [1, 4, 1, 6]

        # After slicing a 2-slot window, the raw bytes match.
        b_3d = slice_by_slot_dim(out_3d, 1, 3).contiguous().numpy().tobytes()
        b_5d = slice_by_slot_dim(out_5d, 1, 3).contiguous().numpy().tobytes()
        assert b_3d == b_5d
