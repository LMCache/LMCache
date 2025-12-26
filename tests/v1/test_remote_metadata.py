# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.protocol import RemoteMetadata


@pytest.mark.parametrize("num_groups", [1, 2, 3])
def test_serialize_and_deserialize(num_groups):
    all_shapes = [
        torch.Size([1, 2, 3, 4]),
        torch.Size([5, 6, 7, 8]),
        torch.Size([9, 10, 11, 12]),
    ]
    all_dtypes = [torch.uint8, torch.float16, torch.float32]

    shapes = all_shapes[:num_groups]
    dtypes = all_dtypes[:num_groups]
    origin_metadata = RemoteMetadata(
        100,
        shapes,
        dtypes,
        MemoryFormat.KV_MLA_FMT,
    )

    fmt = "i" * (2 + num_groups * 5)
    meta_bytes = origin_metadata.serialize(fmt)
    assert len(meta_bytes) == (2 + num_groups * 5) * 4
    new_metadata = RemoteMetadata.deserialize(meta_bytes, fmt)
    assert origin_metadata.length == new_metadata.length
    assert origin_metadata.shapes == new_metadata.shapes
    assert origin_metadata.dtypes == new_metadata.dtypes
    assert origin_metadata.fmt == new_metadata.fmt
