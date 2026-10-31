# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.inprocess.v1.storage_backend.legacy_serde.cachegen_basics import (
    CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK,
)
from lmcache.inprocess.v1.storage_backend.legacy_serde.cachegen_encoder import (
    encode_ntokens,
)

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

NLAYERS, NCHANNELS, NTOKENS, LP = 2, 8, 256, 17


def skewed_cdf():
    """CDF where symbol 0 holds almost all of the probability and symbols
    1..15 the minimum, so every other symbol costs about 16 bits."""
    values = [0] + [0xFFFF - LP + k for k in range(1, LP)]
    cdf = torch.tensor(
        [v - 0x10000 if v > 0x7FFF else v for v in values], dtype=torch.int16
    )
    return cdf.expand(NLAYERS, NCHANNELS, LP).contiguous().to(torch_device_type)


def encode(symbols):
    output_buffer = torch.zeros(
        (NLAYERS, NCHANNELS, CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK),
        dtype=torch.uint8,
        device=torch_device_type,
    )
    output_lengths = torch.zeros(
        (NLAYERS, NCHANNELS), dtype=torch.int32, device=torch_device_type
    )
    return encode_ntokens(
        skewed_cdf(), symbols.to(torch_device_type), output_buffer, output_lengths
    )


def test_encode_ntokens_rejects_stream_longer_than_slot():
    """A stream that needs more bytes than its slot holds used to be stored
    truncated (and its neighbours corrupted); it must now raise instead."""
    symbols = torch.zeros((NLAYERS, NTOKENS, NCHANNELS), dtype=torch.int8)
    symbols[0, :, 0] = 7  # ~16 bits per token: about 512 bytes for 256 tokens
    with pytest.raises(RuntimeError, match="stream slot holds"):
        encode(symbols)


def test_encode_ntokens_accepts_streams_that_fit():
    """With the same CDF, symbols it expects stay far below the slot size."""
    symbols = torch.zeros((NLAYERS, NTOKENS, NCHANNELS), dtype=torch.int8)
    byte_tensor = encode(symbols)
    assert byte_tensor.numel() > 0
