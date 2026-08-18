# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch

# First Party
from lmcache.cli.commands.bench.l2_adapter_bench.data import (
    create_l1_memory_desc,
    make_aligned_tensor,
    make_memory_objects,
    make_object_keys,
)


def test_make_aligned_tensor_returns_aligned_buffer() -> None:
    tensor = make_aligned_tensor(4096 * 3, align_bytes=4096)

    assert tensor.numel() == 4096 * 3
    assert tensor.dtype == torch.uint8
    assert tensor.data_ptr() % 4096 == 0


def test_create_l1_memory_desc_uses_requested_alignment() -> None:
    tensor = make_aligned_tensor(8192, align_bytes=4096)

    desc = create_l1_memory_desc(tensor, align_bytes=4096)

    assert desc.ptr == tensor.data_ptr()
    assert desc.size == 8192
    assert desc.align_bytes == 4096


def test_make_object_keys_spread_hash_prefixes_deterministically() -> None:
    keys = make_object_keys(512, key_offset=100)

    assert keys == make_object_keys(512, key_offset=100)
    assert len({key.chunk_hash for key in keys}) == len(keys)
    assert all(len(key.chunk_hash) == 16 for key in keys)

    level_one_prefixes = {key.chunk_hash[:1] for key in keys}
    level_two_prefixes = {key.chunk_hash[:2] for key in keys}
    assert len(level_one_prefixes) > 1
    assert len(level_two_prefixes) > len(level_one_prefixes)


def test_make_memory_objects_uses_shared_l1_range() -> None:
    buffer = make_aligned_tensor(4096, align_bytes=1024)

    objects = make_memory_objects(
        buffer,
        num_keys=2,
        data_size=1024,
        base_offset=1024,
    )

    assert len(objects) == 2
    assert objects[0].raw_data.data_ptr() == buffer.data_ptr() + 1024
    assert objects[1].raw_data.data_ptr() == buffer.data_ptr() + 2048
    assert torch.all(objects[0].raw_data == 0)
    assert torch.all(objects[1].raw_data == 1)


def test_make_memory_objects_can_use_different_fill_pattern() -> None:
    buffer = make_aligned_tensor(2048, align_bytes=1024)

    objects = make_memory_objects(
        buffer,
        num_keys=2,
        data_size=1024,
        base_offset=0,
        fill_offset=1,
    )

    assert torch.all(objects[0].raw_data == 1)
    assert torch.all(objects[1].raw_data == 2)
