# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import ThreadPoolExecutor

# Third Party
from numba import njit
import numpy as np
import torch


@njit(cache=True)
def _encode_single_channel(
    cdf_layer_c,  # np.uint32 [lp]
    sym_channel,  # np.uint8 [n_tokens]
    out_buf_lc,  # np.uint8 [buffer_size]
):
    """Core arithmetic encoding for a single (layer, channel).
    Returns number of bytes written."""
    MASK32 = 0xFFFFFFFF
    precision = 16
    max_symbol = len(cdf_layer_c) - 2
    n_tokens = len(sym_channel)

    low, high = 0, MASK32
    pending_bits = 0
    output_reg, output_reg_len = 0, 0
    ptr = 0
    buf_size = len(out_buf_lc)

    # Inline flush_bit to avoid closure (numba does not support nonlocal)
    for token_idx in range(n_tokens):
        sym = int(sym_channel[token_idx])
        c_low = int(cdf_layer_c[sym])
        c_high = 0x10000 if sym == max_symbol else int(cdf_layer_c[sym + 1])

        span = (high - low + 1) & MASK32
        if span == 0:
            span = 0x100000000

        high = (low + ((span * c_high) >> precision) - 1) & MASK32
        low = (low + ((span * c_low) >> precision)) & MASK32

        while True:
            if (high & 0x80000000) == (low & 0x80000000):
                # flush_bit(bit)
                bit = (high >> 31) & 1
                output_reg = (output_reg << 1) | bit
                output_reg_len += 1
                if output_reg_len == 8:
                    if ptr < buf_size:
                        out_buf_lc[ptr] = output_reg & 0xFF
                        ptr += 1
                    output_reg, output_reg_len = 0, 0
                # flush pending bits
                for _ in range(pending_bits):
                    output_reg = (output_reg << 1) | (1 - bit)
                    output_reg_len += 1
                    if output_reg_len == 8:
                        if ptr < buf_size:
                            out_buf_lc[ptr] = output_reg & 0xFF
                            ptr += 1
                        output_reg, output_reg_len = 0, 0
                pending_bits = 0
                low = (low << 1) & MASK32
                high = ((high << 1) | 1) & MASK32
            elif (low & 0x40000000) != 0 and (high & 0x40000000) == 0:
                pending_bits += 1
                low = (low << 1) & 0x7FFFFFFF
                high = ((high << 1) | 0x80000001) & MASK32
            else:
                break

    # Final flushing sequence
    pending_bits += 1
    bit = 1 if (low & 0x40000000) != 0 else 0
    output_reg = (output_reg << 1) | bit
    output_reg_len += 1
    if output_reg_len == 8:
        if ptr < buf_size:
            out_buf_lc[ptr] = output_reg & 0xFF
            ptr += 1
        output_reg, output_reg_len = 0, 0
    for _ in range(pending_bits):
        output_reg = (output_reg << 1) | (1 - bit)
        output_reg_len += 1
        if output_reg_len == 8:
            if ptr < buf_size:
                out_buf_lc[ptr] = output_reg & 0xFF
                ptr += 1
            output_reg, output_reg_len = 0, 0
    pending_bits = 0  # noqa: F841

    if output_reg_len > 0:
        if ptr < buf_size:
            out_buf_lc[ptr] = (output_reg << (8 - output_reg_len)) & 0xFF
            ptr += 1

    return ptr


def encode_fast_new(cdf, input_sym, output_buffer, output_lengths):
    """
    Python equivalent of C++ Arithmetic Encoder.
    Strictly emulates 32-bit unsigned overflow for high/low.
    """
    cdf_np = cdf.cpu().numpy().view(np.uint16).astype(np.uint32)
    sym_np = input_sym.cpu().numpy().astype(np.uint8)

    n_layers, n_tokens, n_channels = sym_np.shape
    out_buf_np = np.zeros(output_buffer.shape, dtype=np.uint8)
    out_len_np = np.zeros(output_lengths.shape, dtype=np.int32)

    def encode_one(args):
        layer_idx, c = args
        length = _encode_single_channel(
            cdf_np[layer_idx, c],
            sym_np[layer_idx, :, c],
            out_buf_np[layer_idx, c],
        )
        out_len_np[layer_idx, c] = length

    tasks = [(layer_idx, c) for layer_idx in range(n_layers) for c in range(n_channels)]

    with ThreadPoolExecutor() as executor:
        list(executor.map(encode_one, tasks))

    output_buffer.copy_(torch.from_numpy(out_buf_np))
    output_lengths.copy_(torch.from_numpy(out_len_np))


@njit(cache=True)
def _decode_single_channel(
    cdf_layer_c,
    bs_np,
    start_off,
    end_off,
    n_tokens,
    out_layer_c,
):
    MASK32 = 0xFFFFFFFF
    precision = 16
    max_symbol = len(cdf_layer_c) - 2

    v_val = 0
    if start_off + 4 <= len(bs_np):
        v_val = (
            (int(bs_np[start_off]) << 24)
            | (int(bs_np[start_off + 1]) << 16)
            | (int(bs_np[start_off + 2]) << 8)
            | int(bs_np[start_off + 3])
        ) & MASK32

    low, high = 0, MASK32
    byte_buffer_offset = start_off + 4
    bit_idx = 1
    byte_buffer = int(bs_np[byte_buffer_offset]) if byte_buffer_offset < end_off else 0

    for i in range(n_tokens):
        span = (high - low + 1) & MASK32
        if span == 0:
            span = 0x100000000

        v_minus_l = (v_val - low) & MASK32
        count = ((v_minus_l + 1) * 0x10000 - 1) // span
        count = count & 0xFFFF

        left = 0
        right = max_symbol + 1
        while left + 1 < right:
            m = (left + right) // 2
            if int(cdf_layer_c[m]) < count:
                left = m
            elif int(cdf_layer_c[m]) > count:
                right = m
            else:
                left = m
                break

        out_layer_c[i] = left

        if i == n_tokens - 1:
            break

        sym_i = left
        c_low = int(cdf_layer_c[sym_i])
        c_high = 0x10000 if sym_i == max_symbol else int(cdf_layer_c[sym_i + 1])

        high = (low + ((span * c_high) >> precision) - 1) & MASK32
        low = (low + ((span * c_low) >> precision)) & MASK32

        while True:
            if low >= 0x80000000 or high < 0x80000000:
                v_val = ((v_val << 1) | ((byte_buffer >> (8 - bit_idx)) & 1)) & MASK32
                low = (low << 1) & MASK32
                high = ((high << 1) | 1) & MASK32
                bit_idx += 1
            elif low >= 0x40000000 and high < 0xC0000000:
                v_val = (v_val - 0x40000000) & MASK32
                v_val = ((v_val << 1) | ((byte_buffer >> (8 - bit_idx)) & 1)) & MASK32
                low = (low << 1) & 0x7FFFFFFF
                high = ((high << 1) | 0x80000001) & MASK32
                bit_idx += 1
            else:
                break

            if bit_idx == 9:
                bit_idx = 1
                byte_buffer_offset += 1
                byte_buffer = (
                    int(bs_np[byte_buffer_offset])
                    if byte_buffer_offset < end_off
                    else 0
                )


# Standard


def decode_fast_new(cdf, bytestreams, lengths, output):
    """
    Python implementation of Arithmetic Decoding.
    Strictly aligned with CUDA decode_with_accessor_kernel.
    bytestreams shape: [nlayers, nchannels, buffer_size]
    """
    cdf_np = cdf.cpu().numpy().view(np.uint16).astype(np.uint32)
    bs_np = bytestreams.cpu().numpy().astype(np.uint8)
    len_np = lengths.cpu().numpy().astype(np.int32)

    n_layers, n_tokens, n_channels = output.shape
    out_np = np.zeros(output.shape, dtype=np.uint8)

    def decode_one(args):
        layer_idx, c = args
        curr_len = int(len_np[layer_idx, c])
        # For decode_fast_new, each channel has its own contiguous buffer,
        # so start_off=0 and end_off=curr_len within channel_bs
        channel_bs = bs_np[layer_idx, c]  # shape [buffer_size]
        _decode_single_channel(
            cdf_np[layer_idx, c],
            channel_bs,
            0,
            curr_len,
            n_tokens,
            out_np[layer_idx, :, c],
        )

    tasks = [(layer_idx, c) for layer_idx in range(n_layers) for c in range(n_channels)]

    with ThreadPoolExecutor() as executor:
        list(executor.map(decode_one, tasks))

    if output is not None:
        output.copy_(torch.from_numpy(out_np))


def decode_fast_prefsum(cdf, bytestreams, lengths_prefsum, output):
    """
    Python equivalent of C++ decode_fast_prefsum.
    bytestreams shape: [total_bytes] (1D, all channels packed)
    """
    cdf_np = cdf.cpu().numpy().view(np.uint16).astype(np.uint32)
    pref_np = lengths_prefsum.cpu().numpy().astype(np.int64).flatten()

    # WA: CUDA kernel reads out-of-bound in two ways:
    # 1. max(prefsum) may equal len(bytestreams) (off-by-one on exclusive-end)
    # 2. v_val init reads 4 bytes starting at start_off, may exceed bytestreams
    # Pad with zeros to make all reads safe.
    max_prefsum = int(pref_np.max())
    pad_size = max(0, max_prefsum + 4 - bytestreams.shape[0])
    if pad_size > 0:
        bytestreams = torch.nn.functional.pad(bytestreams, (0, pad_size), value=0)

    bs_np = bytestreams.cpu().numpy().astype(np.uint8)  # must be after padding

    n_layers, n_tokens, n_channels = output.shape
    out_np = np.zeros(output.shape, dtype=np.uint8)

    def decode_one(args):
        layer_idx, c = args
        cid = layer_idx * n_channels + c
        start_off = 0 if cid == 0 else int(pref_np[cid - 1])
        end_off = int(pref_np[cid])
        _decode_single_channel(
            cdf_np[layer_idx, c],
            bs_np,
            start_off,
            end_off,
            n_tokens,
            out_np[layer_idx, :, c],
        )

    tasks = [(layer_idx, c) for layer_idx in range(n_layers) for c in range(n_channels)]

    with ThreadPoolExecutor() as executor:
        list(executor.map(decode_one, tasks))

    output.copy_(torch.from_numpy(out_np))


def calculate_cdf(input_tensor: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Equivalent to CUDA calculate_cdf.

    Calculates the CDF across tokens for each (layer, channel) pair.

    Args:
        input_tensor: 3D tensor with shape [nlayers, ntokens, nchannels].
        num_bins: Maximum number of bins (i.e., Lp - 1).

    Returns:
        int16 tensor with shape [nlayers, nchannels, num_bins + 1]
        containing normalized CDF values.
    """
    nlayers, ntokens, nchannels = input_tensor.shape
    device = input_tensor.device

    # Compute per-(layer, channel) histogram via scatter_add.
    # Permute to [nlayers, nchannels, ntokens] then flatten first two dims.
    input_perm = input_tensor.permute(0, 2, 1).reshape(-1, ntokens).long()
    src = torch.ones_like(input_perm)
    counts = torch.zeros(nlayers * nchannels, num_bins, dtype=torch.long, device=device)
    counts.scatter_add_(1, input_perm.clamp(0, num_bins - 1), src)
    counts = counts.reshape(nlayers, nchannels, num_bins)

    # Build CDF: cdf[..., 0] = 0, cdf[..., i] = sum(counts[..., 0:i])
    cdf = torch.zeros(nlayers, nchannels, num_bins + 1, dtype=torch.long, device=device)
    cdf[:, :, 1:] = torch.cumsum(counts, dim=2)

    # Total count per (layer, channel)
    total = cdf[:, :, -1:]  # [nlayers, nchannels, 1]

    # Normalize: (0xFFFF - num_bins) * cdf / total + bin_index
    max_uint16_value = 0xFFFF - num_bins
    bin_offsets = torch.arange(num_bins + 1, dtype=torch.long, device=device)

    safe_total = total.clamp(min=1)
    normalized = (max_uint16_value * cdf) // safe_total + bin_offsets

    # Where total is 0, use just the bin offsets
    normalized = torch.where(
        total > 0, normalized, bin_offsets.unsqueeze(0).unsqueeze(0)
    )

    return normalized.to(torch.int16)
