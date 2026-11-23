# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the ATOM -> LMCache conversion layer.

The descriptor shapes follow the external-connector design (v2): ATOM hands
over named tensors plus kernel groups, chunk hashes plus block ids, and asks
for a chunk hasher. These conversions are pure functions, so the whole boundary
between the projects can be exercised without an AMD GPU.
"""

# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.atom.hasher import ChunkHasher, TokenChainHasher
from lmcache.integration.atom.layout import (
    AtomKernelGroup,
    AtomStoreLayout,
    engine_group_infos_from_atom,
    page_kernel_groups,
)
from lmcache.integration.atom.transfer import (
    AtomTransferRequest,
    transfer_spec_from_atom,
)

_MODEL_TENSORS = {"layer.0": torch.empty(1), "layer.1": torch.empty(1)}


def _layout(*groups: AtomKernelGroup) -> AtomStoreLayout:
    return AtomStoreLayout(tensors=_MODEL_TENSORS, groups=groups)


class TestPageKernelGroups:
    def test_each_tensor_group_becomes_engine_group_zero(self):
        groups = page_kernel_groups([["layer.0"], ["layer.1"]], tokens_per_block=16)

        assert [group.engine_group_id for group in groups] == [0, 0]
        assert [group.tensor_names for group in groups] == [("layer.0",), ("layer.1",)]
        assert {group.tokens_per_block for group in groups} == {16}

    def test_empty_group_list_is_rejected(self):
        with pytest.raises(ValueError, match="no PAGE tensor groups"):
            page_kernel_groups([], tokens_per_block=16)

    def test_an_empty_group_is_rejected(self):
        with pytest.raises(ValueError, match="group 1 is empty"):
            page_kernel_groups([["layer.0"], []], tokens_per_block=16)


class TestEngineGroupInfos:
    def test_names_are_resolved_to_registration_indices(self):
        infos = engine_group_infos_from_atom(
            _layout(
                AtomKernelGroup(
                    tensor_names=("layer.1",),
                    engine_group_id=0,
                    tokens_per_block=16,
                )
            )
        )

        # layer.1 is the second registered tensor.
        assert infos[0].layer_indices == (1,)

    def test_fields_are_mapped_one_to_one(self):
        infos = engine_group_infos_from_atom(
            _layout(
                AtomKernelGroup(
                    tensor_names=("layer.0",),
                    engine_group_id=0,
                    tokens_per_block=16,
                ),
                AtomKernelGroup(
                    tensor_names=("layer.1",),
                    engine_group_id=1,
                    tokens_per_block=8,
                    sw_size_tokens=128,
                    recurrent_state=True,
                    extra_object_group_tag=3,
                ),
            )
        )

        assert [info.engine_group_id for info in infos] == [0, 1]
        assert [info.tokens_per_block for info in infos] == [16, 8]
        assert infos[1].sw_size_tokens == 128
        assert infos[1].recurrent_state is True
        assert infos[1].extra_object_group_tag == 3

    def test_replication_factors_ride_on_the_group(self):
        infos = engine_group_infos_from_atom(
            _layout(
                AtomKernelGroup(
                    tensor_names=("layer.0",),
                    engine_group_id=0,
                    tokens_per_block=16,
                    tp_replicas=8,
                    dcp_shards=2,
                ),
                AtomKernelGroup(
                    tensor_names=("layer.1",),
                    engine_group_id=1,
                    tokens_per_block=4,
                    tp_replicas=2,
                ),
            )
        )

        assert [(i.tp_replicas, i.dcp_shards) for i in infos] == [(8, 2), (2, 1)]

    def test_dcp_cannot_exceed_tp_for_one_group(self):
        with pytest.raises(
            ValueError, match="engine group 1.*dcp_shards \\(4\\).*tp_replicas \\(2\\)"
        ):
            engine_group_infos_from_atom(
                _layout(
                    AtomKernelGroup(
                        tensor_names=("layer.0",),
                        engine_group_id=0,
                        tokens_per_block=16,
                    ),
                    AtomKernelGroup(
                        tensor_names=("layer.1",),
                        engine_group_id=1,
                        tokens_per_block=16,
                        tp_replicas=2,
                        dcp_shards=4,
                    ),
                )
            )

    def test_a_zero_factor_is_rejected(self):
        with pytest.raises(
            ValueError, match="engine group 0: tp_replicas must be >= 1"
        ):
            engine_group_infos_from_atom(
                _layout(
                    AtomKernelGroup(
                        tensor_names=("layer.0",),
                        engine_group_id=0,
                        tokens_per_block=16,
                        tp_replicas=0,
                    )
                )
            )

    def test_an_unknown_tensor_name_is_rejected(self):
        with pytest.raises(ValueError, match="names tensor 'layer 9'"):
            engine_group_infos_from_atom(
                _layout(
                    AtomKernelGroup(
                        tensor_names=("layer 9",),
                        engine_group_id=0,
                        tokens_per_block=16,
                    )
                )
            )

    def test_one_tensor_cannot_join_two_groups(self):
        with pytest.raises(ValueError, match="appears in kernel groups 0 and 1"):
            engine_group_infos_from_atom(
                _layout(
                    AtomKernelGroup(
                        tensor_names=("layer.0",),
                        engine_group_id=0,
                        tokens_per_block=16,
                    ),
                    AtomKernelGroup(
                        tensor_names=("layer.0",),
                        engine_group_id=1,
                        tokens_per_block=8,
                    ),
                )
            )

    def test_engine_group_ids_must_be_dense(self):
        with pytest.raises(ValueError, match="dense from 0"):
            engine_group_infos_from_atom(
                _layout(
                    AtomKernelGroup(
                        tensor_names=("layer.0",),
                        engine_group_id=1,
                        tokens_per_block=16,
                    )
                )
            )

    def test_no_groups_and_no_tensors_are_rejected(self):
        with pytest.raises(ValueError, match="no kernel groups"):
            engine_group_infos_from_atom(AtomStoreLayout(tensors=_MODEL_TENSORS))
        with pytest.raises(ValueError, match="no tensors"):
            engine_group_infos_from_atom(
                AtomStoreLayout(
                    tensors={},
                    groups=(
                        AtomKernelGroup(
                            tensor_names=("layer.0",),
                            engine_group_id=0,
                            tokens_per_block=16,
                        ),
                    ),
                )
            )


class TestNativeStateGroups:
    """State image groups register the same way PAGE groups do."""

    def test_state_groups_share_the_single_conversion_path(self):
        layout = AtomStoreLayout(
            tensors={"page.0": object(), "image.0": object()},
            groups=(
                AtomKernelGroup(
                    tensor_names=("page.0",),
                    engine_group_id=0,
                    tokens_per_block=16,
                    tp_replicas=2,
                ),
                AtomKernelGroup(
                    tensor_names=("image.0",),
                    engine_group_id=1,
                    tokens_per_block=16,
                    tp_replicas=1,
                    dcp_shards=1,
                ),
            ),
            native_state=object(),
        )

        infos = engine_group_infos_from_atom(layout)

        assert [info.engine_group_id for info in infos] == [0, 1]
        assert [info.tp_replicas for info in infos] == [2, 1]
        assert [list(info.layer_indices) for info in infos] == [[0], [1]]


class TestTransferSpec:
    HASHES = bytes([7]) * 64  # two chunks

    def _request(self, **overrides: Any) -> Any:
        body: dict[str, Any] = {
            "hashes": self.HASHES,
            "block_ids": [[0, 1]],
            "start": 0,
            "end": 16,
        }
        body.update(overrides)
        return SimpleNamespace(**body)

    def test_the_dataclass_shape_is_accepted(self):
        """ATOM may build the descriptor itself instead of a namespace."""
        spec = transfer_spec_from_atom(
            AtomTransferRequest(hashes=bytes([1]) * 32, block_ids=[0], start=0, end=8)
        )

        assert spec.chunk_hashes == [bytes([1]) * 32]
        assert spec.block_ids == [[0]]

    def test_the_hash_blob_is_split_per_chunk(self):
        spec = transfer_spec_from_atom(self._request())

        assert spec.chunk_hashes == [bytes([7]) * 32, bytes([7]) * 32]

    def test_a_partial_hash_is_rejected(self):
        with pytest.raises(ValueError, match="not a multiple of 32"):
            transfer_spec_from_atom(self._request(hashes=b"x" * 40))

    def test_the_hash_count_matches_the_chunk_size(self):
        """Hashes cover ``[0, end)``; the server rejects any other count."""
        spec = transfer_spec_from_atom(
            self._request(hashes=bytes([7]) * 64, end=512), chunk_tokens=256
        )

        assert len(spec.chunk_hashes) == 2

    def test_the_hash_chain_covers_the_prefix_even_when_start_is_not_zero(self):
        """Hashes cover ``[0, end)``; ``start`` only selects the block ids."""
        spec = transfer_spec_from_atom(
            self._request(hashes=bytes([7]) * 64, start=256, end=512),
            chunk_tokens=256,
        )

        assert len(spec.chunk_hashes) == 2

    def test_a_hash_count_that_does_not_match_the_range_is_rejected(self):
        with pytest.raises(ValueError, match="chunk hash"):
            transfer_spec_from_atom(
                self._request(hashes=bytes([7]) * 32, end=512), chunk_tokens=256
            )

    def test_a_non_positive_chunk_size_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            transfer_spec_from_atom(self._request(), chunk_tokens=0)

    def test_flat_block_ids_are_one_engine_group(self):
        spec = transfer_spec_from_atom(
            self._request(block_ids=[0, 1]), num_engine_groups=1
        )

        assert spec.block_ids == [[0, 1]]

    def test_flat_block_ids_with_two_groups_are_rejected(self):
        with pytest.raises(ValueError, match="1 block-id list\\(s\\) but 2"):
            transfer_spec_from_atom(
                self._request(block_ids=[0, 1]), num_engine_groups=2
            )

    def test_the_salt_reaches_the_spec(self):
        spec = transfer_spec_from_atom(self._request(cache_salt="tenant-a"))

        assert spec.cache_salt == "tenant-a"

    def test_a_salt_with_a_separator_is_rejected(self):
        with pytest.raises(ValueError, match="cache_salt must not contain"):
            transfer_spec_from_atom(self._request(cache_salt="a@b"))

    def test_a_non_empty_range_without_hashes_is_rejected(self):
        with pytest.raises(ValueError, match="carries no chunk hashes"):
            transfer_spec_from_atom(self._request(hashes=b""))

    def test_a_range_without_blocks_is_rejected(self):
        with pytest.raises(ValueError, match="names no blocks"):
            transfer_spec_from_atom(self._request(block_ids=[[]]))

    def test_start_past_end_is_rejected(self):
        with pytest.raises(ValueError, match="is past its end"):
            transfer_spec_from_atom(self._request(start=8, end=4))

    def test_token_ids_stay_empty(self):
        """The external API carries hashes; the backend keys by them."""
        spec = transfer_spec_from_atom(self._request())

        assert spec.token_ids == []


class TestChunkHasher:
    def test_it_names_its_scheme_and_chunk_size(self):
        hasher = TokenChainHasher(chunk_size=4, hash_algorithm="blake3")

        assert hasher.chunk_tokens == 4
        assert hasher.scheme == "lmcache-blake3-4-v1"
        assert isinstance(hasher, ChunkHasher)

    def test_chain_returns_one_hash_per_chunk_call(self):
        hasher = TokenChainHasher(chunk_size=4)

        first = hasher.chain([1, 2, 3, 4], 0, None)
        second = hasher.chain([5, 6, 7, 8], 1, first)

        assert isinstance(first, bytes) and len(first) == 32
        assert second != first

    def test_a_short_chunk_is_rejected(self):
        hasher = TokenChainHasher(chunk_size=4)

        with pytest.raises(ValueError, match="expected 4"):
            hasher.chain([1, 2, 3], 0, None)

    def test_the_stats_report_its_identity(self):
        hasher = TokenChainHasher(chunk_size=8)

        assert hasher.stats()["chunk_tokens"] == 8
        assert hasher.stats()["scheme"] == hasher.scheme
