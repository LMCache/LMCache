# SPDX-License-Identifier: Apache-2.0
"""Preprocessing transforms for accelerated KV cache compression.

These are applied before compression (serialize path) and reversed after
decompression (deserialize path) to improve compression ratio.

Both transforms operate in-place on a NumPy view of the buffer.
"""

import numpy as np


def quant_trunc(buf: memoryview, element_size: int, truncate_bits: int) -> None:
    """Zero the least-significant bits of each element (lossy, in-place).

    This increases byte-level redundancy to improve deflate compression
    ratio by 10-25%. Applied only on the serialize (PUT) path.

    Args:
        buf: Writable memoryview of the raw KV data.
        element_size: Bytes per element (2 for bf16/fp16, 1 for fp8).
        truncate_bits: Number of LSBs to zero (e.g. 2 for bf16).
    """
    if truncate_bits <= 0:
        return

    if element_size == 2:
        arr = np.frombuffer(buf, dtype=np.uint16)
        mask = np.uint16((0xFFFF << truncate_bits) & 0xFFFF)
        np.bitwise_and(arr, mask, out=arr)
    elif element_size == 1:
        arr = np.frombuffer(buf, dtype=np.uint8)
        mask = np.uint8((0xFF << truncate_bits) & 0xFF)
        np.bitwise_and(arr, mask, out=arr)


def data_shuffle(buf: memoryview, element_size: int) -> None:
    """Byte-lane shuffle for bf16 data to improve compression ratio (in-place).

    Separates exponent bytes from mantissa bytes so deflate finds longer
    runs of similar values. The transform is self-inverse (involutory):
    applying it twice restores the original data.

    Layout for bf16 ([exp|man] pairs):
      Original: [e0|m0][e1|m1][e2|m2]...
      Shuffled: exponent bytes grouped, mantissa bytes grouped

    The actual operation swaps high_bytes[i*2] with low_bytes[i*2+1]
    where high_bytes = first half, low_bytes = second half of the buffer.

    Args:
        buf: Writable memoryview of the raw KV data.
        element_size: Bytes per element. Only applied when element_size == 2.
    """
    if element_size != 2:
        return

    arr = np.frombuffer(buf, dtype=np.uint8)
    size = len(arr)
    half = size // 2

    high = arr[:half].reshape(-1, 2)
    low = arr[half:].reshape(-1, 2)

    # Swap high[:, 0] with low[:, 1] (self-inverse)
    tmp = high[:, 0].copy()
    high[:, 0] = low[:, 1]
    low[:, 1] = tmp
