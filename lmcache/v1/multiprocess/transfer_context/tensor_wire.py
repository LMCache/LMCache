# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for the transport-owned inline tensor wire codec."""

# First Party
from lmcache.v1.transport.tensor_wire import (
    ChunkIndices,
    GroupedTensorEnvelope,
    TensorBuffer,
    TensorEnvelope,
    TensorFrame,
    TensorFrames,
    TensorGroup,
    TensorGroups,
    TensorShape,
    decode_grouped_tensors,
    decode_tensors,
    encode_grouped_tensors,
    encode_tensors,
)

__all__ = [
    "ChunkIndices",
    "GroupedTensorEnvelope",
    "TensorBuffer",
    "TensorEnvelope",
    "TensorFrame",
    "TensorFrames",
    "TensorGroup",
    "TensorGroups",
    "TensorShape",
    "decode_grouped_tensors",
    "decode_tensors",
    "encode_grouped_tensors",
    "encode_tensors",
]
