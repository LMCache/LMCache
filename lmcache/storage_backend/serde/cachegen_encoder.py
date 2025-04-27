from typing import Tuple

import torch

import lmcache.c_ops as lmc_ops
import lmcache.storage_backend.serde.cachegen_basics as CGBasics
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.cachegen_basics import (
    CacheGenConfig, CacheGenGPUBytestream, CacheGenGPUEncoderOutput)
from lmcache.storage_backend.serde.serde import Serializer
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


@_lmcache_nvtx_annotate
def torch_quant(bins: int,
                qA: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a float tensor to fixed number of bins

    Input:
        bins: number of bins
        qA: the input tensor

    Returns:
        xq: the quantized tensor, in float32
        max1: the maximum value of the tensor
    """
    MAX = bins // 2 - 1
    C = MAX
    max1 = torch.amax(torch.abs(qA), dim=-1, keepdim=True)
    # Avoid division by zero if max1 is zero
    max1 = torch.where(max1 == 0, torch.tensor(1.0, device=max1.device), max1)
    xq = torch.round(qA * (C / max1)).to(torch.int8)

    return xq, max1


@_lmcache_nvtx_annotate
def torch_quant_vectorized(
        bins: torch.Tensor,
        input_groups: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize each group of a tensor to fixed number of bins

    Input:
        bins: number of bins for different layers, with shape [nlayer]
        input_groups: with shape [nlayers, ntokens, nchannels]

    Returns:
        quantized groups: [nlayers, ntokens, nchannels]
        maxes: [nlayers, ntokens, 1]
    """
    MAX = (bins // 2 - 1)[:, None, None]  # shape [nlayers, 1, 1]
    max1 = torch.amax(torch.abs(input_groups), dim=-1,
                      keepdim=True)  # shape [nlayers, ntokens, 1]
    # Avoid division by zero
    max1 = torch.where(max1 == 0, torch.tensor(1.0, device=max1.device), max1)
    factor = MAX / max1  # shape [nlayers, ntokens, 1]
    # Add MAX to shift range from [-MAX, MAX] to [0, 2*MAX]
    # for uint8 representation in CDF calculation?
    # Original CacheGen seemed to expect signed int8, but CDF calculation
    # needs non-negative indices.
    # Let's keep the original logic for now: `tmp[0] + bins // 2 - 1`
    # in the old quantize suggests
    # the range should be centered around zero before shifting.
    # The `+ MAX` here seems to shift to non-negative, matching the
    # `num_classes=max_val + 1` in compute_cdf.
    # Let's stick with `+ MAX` for now.
    xq = torch.round(input_groups * factor + MAX).to(
        torch.int8)  # shape [nlayers, ntokens, nchannels]

    return xq, max1


@_lmcache_nvtx_annotate
def concat_max(max1):
    """
    Given a dict of max tensors, concatenate them into a single tensor
    """
    # TODO: this function can be optimized, we don't really need this
    maxes = []
    for i in range(len(max1)):
        maxes.append(max1[i].unsqueeze(0))
    return torch.cat(maxes, dim=0)


def _split_kv(tensor: torch.Tensor,
              fmt: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a blob KV tensor into K and V tensors, reshaping them
    to the expected format.

    Input:
        tensor: The KV tensor blob. Expected shapes:
                vLLM: [2, num_layers, num_tokens, num_heads, head_size]
                HF:   [2, num_layers, num_heads, num_tokens, head_size]
        fmt: The format ('vllm' or 'huggingface')

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: K and V tensors, both with shape
            [num_layers, num_tokens, num_channels],
#             where num_channels = num_heads * head_size
    """
    num_layers = tensor.shape[1]
    k_tensor = tensor[0]  # Shape [num_layers, ...]
    v_tensor = tensor[1]  # Shape [num_layers, ...]

    if fmt == "vllm":
        # Input K/V shape: [num_layers, num_tokens, num_heads, head_size]
        num_tokens = tensor.shape[2]
        num_heads = tensor.shape[3]
        head_size = tensor.shape[4]
        num_channels = num_heads * head_size
        # Reshape directly
        fp_k = k_tensor.reshape(num_layers, num_tokens, num_channels)
        fp_v = v_tensor.reshape(num_layers, num_tokens, num_channels)
    elif fmt == "huggingface":
        # Input K/V shape: [num_layers, num_heads, num_tokens, head_size]
        num_heads = tensor.shape[2]
        num_tokens = tensor.shape[3]
        head_size = tensor.shape[4]
        num_channels = num_heads * head_size
        # Permute heads and tokens, then reshape
        # [num_layers, num_heads, num_tokens, head_size] ->
        # [num_layers, num_tokens, num_heads, head_size]
        k_permuted = k_tensor.permute(0, 2, 1, 3)
        v_permuted = v_tensor.permute(0, 2, 1, 3)
        fp_k = k_permuted.reshape(num_layers, num_tokens, num_channels)
        fp_v = v_permuted.reshape(num_layers, num_tokens, num_channels)
    else:
        raise ValueError(f"Invalid format: {fmt}")

    return fp_k, fp_v


@_lmcache_nvtx_annotate
def _convert_to_int_and_normalize(cdf_float, needs_normalization):
    """
    Convert floatingpoint CDF to integers. See README for more info.

    The idea is the following:
    When we get the cdf here, it is (assumed to be) between 0 and 1, i.e,
      cdf in [0, 1)
    (note that 1 should not be included.)
    We now want to convert this to int16 but make sure we do not get
    the same value twice, as this would break the arithmetic coder
    (you need a strictly monotonically increasing function).
    So, if needs_normalization==True, we multiply the input CDF
    with 2**16 - (Lp - 1). This means that now,
      cdf in [0, 2**16 - (Lp - 1)].
    Then, in a final step, we add an arange(Lp), which is just a line with
    slope one. This ensure that for sure, we will get unique, strictly
    monotonically increasing CDFs, which are in [0, 2**16)
    """
    PRECISION = 16
    Lp = cdf_float.shape[-1]
    factor = torch.tensor(2, dtype=torch.float32,
                          device=cdf_float.device).pow_(PRECISION)
    new_max_value = factor
    if needs_normalization:
        new_max_value = new_max_value - (Lp - 1)
    cdf_float = cdf_float.mul(new_max_value)
    cdf_float = cdf_float.round()
    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    if needs_normalization:
        r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
        cdf.add_(r)
    return cdf


# This class seems unused, the logic is directly in
# encode_function now. Keeping for reference?
# class CacheGenEncoderImpl:
#     ... (omitted for brevity)

# @_lmcache_nvtx_annotate
# def collect_bytes(output_buffer, output_lengths) -> torch.Tensor:
#     """
#     Collect a byte tensor from the output_buffer + output_lengths
#     """
#     output_buffer_size = output_buffer.shape[-1]
#     flattened_lengths = output_lengths.flatten()
#     # Ensure lengths are non-negative and do not exceed buffer size
#     flattened_lengths = torch.clamp(flattened_lengths, 0, output_buffer_size)
#     flattened_buffer = output_buffer.flatten()
#
#     # Calculate cumulative sum of (buffer_size - length)
#     # to find start indices in flattened buffer
#     # This seems overly complex and potentially incorrect
#     # if buffer isn't full.
#     # A simpler approach: use cumsum of flattened_lengths to get end indices,
#     # then derive start indices and use repeat_interleave on arange.
#
#     # Alternative approach:
#     total_bytes = flattened_lengths.sum()
#     if total_bytes == 0:
#         return torch.empty(0, dtype=torch.uint8, device=output_buffer.device)
#
#     # Create indices for each layer/channel pair
#     layer_channel_indices = torch.arange(output_lengths.numel(),
#                                          device=output_buffer.device)
#     # Calculate the starting row index in the flattened
#     # buffer for each layer/channel
#     row_starts = layer_channel_indices * output_buffer_size
#
#     # Create indices within each row based on lengths
#     # Example: lengths = [3, 2], buffer_size = 5
#     # row_starts = [0, 5]
#     # repeated_starts = [0, 0, 0, 5, 5]
#     # arange_lengths = [0, 1, 2, 0, 1]
#     # indices = [0, 1, 2, 5, 6]
#     repeated_starts = row_starts.repeat_interleave(flattened_lengths)
#     arange_lengths = torch.cat([
#         torch.arange(l, device=output_buffer.device)
#         for l in flattened_lengths
#     ])
#
#     indices = repeated_starts + arange_lengths
#
#     return flattened_buffer[indices]


@_lmcache_nvtx_annotate
def encode_ntokens(cdf_int, encode_input, output_buffer,
                   output_lengths) -> torch.Tensor:
    """Encode a batch of ntokens.

    :param cdf_int: int16 tensor on GPU with shape [nlayers, nchannels, Lp]
                     Here nlayers is 2 * original_num_layers (K and V stacked)
    :param encode_input: int8 tensor on GPU with shape
                         [nlayers, ntokens, nchannels]
    :param output_buffer: uint8 tensor on GPU with shape
                          [nlayers, nchannels, BUFFER_SIZE]
    :param output_lengths: int32 tensor on GPU with shape [nlayers, nchannels]

    :return byte_tensor: the byte tensor
    """
    lmc_ops.encode_fast_new(
        cdf_int,
        encode_input,
        output_buffer,
        output_lengths,
    )
    # The original collect_bytes is commented out as it's likely
    # unused/incorrect.
    # Assuming encode_fast_new might directly return bytes or we need a new way.
    # For now, let's try to reconstruct it based on the expected output format.
    # It seems encode_fast_new populates output_buffer and output_lengths.
    # We need to extract the relevant bytes based on lengths.

    # Re-implementing a potentially correct collect_bytes logic here:
    output_buffer_size = output_buffer.shape[-1]
    flattened_lengths = output_lengths.flatten()
    # Ensure lengths are non-negative and do not exceed buffer size
    flattened_lengths = torch.clamp(flattened_lengths, 0, output_buffer_size)

    total_bytes = flattened_lengths.sum()
    if total_bytes == 0:
        return torch.empty(0, dtype=torch.uint8, device=output_buffer.device)

    # Create indices for each layer/channel pair
    layer_channel_indices = torch.arange(output_lengths.numel(),
                                         device=output_buffer.device)
    # Calculate the starting row index in the *original*
    # output_buffer for each layer/channel
    row_starts = layer_channel_indices * output_buffer_size

    # Create indices within each row based on lengths
    repeated_starts = row_starts.repeat_interleave(flattened_lengths)
    arange_lengths = torch.cat([
        torch.arange(length, device=output_buffer.device)
        for length in flattened_lengths
    ])

    # These are the indices in the *flattened* version of output_buffer
    indices_in_flat_buffer = repeated_starts + arange_lengths

    # Select the bytes from the flattened buffer
    byte_tensor = output_buffer.flatten()[indices_in_flat_buffer]

    return byte_tensor


@_lmcache_nvtx_annotate
def encode_function(
    kv: torch.Tensor,
    fmt: str,
    config: CacheGenConfig,
    key_bins: torch.Tensor,
    value_bins: torch.Tensor,
    chunk_size: int,  # This is the number of tokens in the input kv tensor
) -> CacheGenGPUEncoderOutput:
    """
    Given the original key value cache tensor blob, encode the KV cache.

    Input:
        kv: The KV tensor blob. Expected shapes:
            vLLM: [2, num_layers, num_tokens, num_heads, head_size]
            HF:   [2, num_layers, num_heads, num_tokens, head_size]
        fmt: The format ('vllm' or 'huggingface')
        config: CacheGen configuration
        key_bins: Tensor of quantization bins for key layers [num_layers]
        value_bins: Tensor of quantization bins for value layers [num_layers]
        chunk_size: Number of tokens in the input kv tensor.
    """
    # Determine shapes based on format
    if fmt == "vllm":
        num_tokens = kv.shape[2]
        num_heads = kv.shape[3]
        head_size = kv.shape[4]
    elif fmt == "huggingface":
        num_heads = kv.shape[2]
        num_tokens = kv.shape[3]
        head_size = kv.shape[4]
    else:
        raise ValueError(f"Invalid format: {fmt}")

    assert num_tokens == chunk_size, (
        f"Input tensor token dimension ({num_tokens}) does not match "
        f"chunk_size ({chunk_size})")

    # Split and reshape K, V to [num_layers, num_tokens, num_channels]
    fp_k, fp_v = _split_kv(kv, fmt)

    num_layers_orig = fp_k.shape[0]  # Original number of layers
    nchannels = num_heads * head_size
    nlayers_stacked = num_layers_orig * 2  # K and V are stacked for processing

    # Quantize K and V separately
    # Input shapes: [num_layers, num_tokens, nchannels]
    # Output shapes: quantized=[num_layers, ntokens, nchannels],
    #                maxes=[num_layers, ntokens, 1]
    new_key, max_tensors_key = torch_quant_vectorized(key_bins, fp_k)
    new_value, max_tensors_value = torch_quant_vectorized(value_bins, fp_v)

    # Stack quantized K and V for encoding input
    # Shape: [2 * num_layers, num_tokens, nchannels]
    encode_input = torch.cat((new_key, new_value), dim=0)

    # Calculate CDFs separately for K and V
    # Input shapes: [num_layers, num_tokens, nchannels]
    # Output shapes: [num_layers, nchannels, num_bins + 1]
    new_cdf_key = lmc_ops.calculate_cdf(new_key,
                                        int(key_bins.max().item()) +
                                        1)  # Pass max number of bins + 1
    new_cdf_value = lmc_ops.calculate_cdf(new_value,
                                          int(value_bins.max().item()) +
                                          1)  # Pass max number of bins + 1

    # Stack CDFs
    # Shape: [2 * num_layers, nchannels, num_bins + 1]
    # Needs normalization before passing to encode_ntokens
    # which expects int16
    # Assuming needs_normalization=True based on
    # _convert_to_int_and_normalize logic
    cdf_float = torch.cat([new_cdf_key, new_cdf_value], dim=0)
    # TODO: Verify if normalization is needed here or if
    #       encode_fast_new handles float CDFs
    # Assuming encode_fast_new needs normalized int16 CDFs
    # based on old code structure
    cdf_int = _convert_to_int_and_normalize(cdf_float,
                                            needs_normalization=True)

    # Prepare buffers for encoding
    output_buffer = torch.zeros(
        (nlayers_stacked, nchannels,
         CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK),
        dtype=torch.uint8,
        device=encode_input.device,
    )
    output_lengths = torch.zeros((nlayers_stacked, nchannels),
                                 dtype=torch.int32,
                                 device=encode_input.device)

    # Encode in smaller chunks if necessary (due to GPU buffer limits)
    # data_chunks = []
    # for i in range(0, chunk_size, CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK):
    #     start = i
    #     end = min(i + CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK, chunk_size)
    #
    #     # Slice the encode_input for the current token chunk
    #     current_encode_input = encode_input[:, start:end, :]
    #     current_ntokens = end - start
    #
    #     # Reset buffers for each smaller chunk? No, the lengths
    # track progress.
    #     # output_buffer and output_lengths seem sized for the whole chunk,
    #     # but encode_ntokens might operate on smaller token batches within it.
    #     # Let's assume encode_ntokens handles the sub-chunking logic if
    #     # needed,
    #     # or that CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK is the actual limit
    #     # per call.
    #
    #     # Ensure buffer is large enough for the current sub-chunk
    #     # This check seems redundant if
    #     # CACHEGEN_GPU_BUFFER_SIZE_PER_CHUNK is sized correctly
    #     # assert output_buffer.shape[2] >= current_ntokens, "Output buffer too
    #     # small for token chunk"
    #
    #     bytestream = encode_ntokens(
    #         cdf_int,  # Full CDF for all layers/channels
    #         current_encode_input,  # Input for current token range
    #         output_buffer,  # Reused buffer
    #         output_lengths,  # Updated by encode_ntokens
    #     )
    #
    #     # Store the results for this sub-chunk
    #     data_chunks.append(
    #         CacheGenGPUBytestream(
    #             bytestream=bytestream.clone(
    #             ),  # Clone necessary? byte_tensor is created fresh.
    #             bytestream_lengths=output_lengths.clone(
    #             ),  # Must clone lengths as they are updated in-place
    #             ntokens=current_ntokens,
    #         ))
    #     # Reset lengths for next sub-chunk? No, encode_fast_new appends.
    #     # The logic in collect_bytes assumes lengths are final *per call*
    #     # to encode_ntokens.
    #     # This implies encode_ntokens should *reset* lengths internally or we
    #     # need to handle it.
    #     # Let's assume encode_ntokens gives the lengths *for that call*.
    #     # We need to accumulate the bytes and lengths across sub-chunks.
    #
    # # --- Refined logic for sub-chunking ---
    # all_bytestreams = []
    # all_lengths = []
    # total_encoded_bytes = 0
    # output_buffer.zero_()  # Ensure buffer is clean
    # output_lengths.zero_()  # Ensure lengths are clean
    #
    # for i in range(0, chunk_size, CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK):
    #     start = i
    #     end = min(i + CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK, chunk_size)
    #     current_encode_input = encode_input[:, start:end, :]
    #     current_ntokens = end - start
    #
    #     # Create temporary buffers for this specific sub-chunk call
    #     # This avoids issues with reusing buffers across calls if
    #     # encode_ntokens has side effects
    #     current_output_buffer = torch.zeros_like(output_buffer)
    #     current_output_lengths = torch.zeros_like(output_lengths)
    #
    #     bytestream = encode_ntokens(
    #         cdf_int,
    #         current_encode_input,
    #         current_output_buffer,  # Use temp buffer
    #         current_output_lengths  # Use temp lengths
    #     )
    #     all_bytestreams.append(bytestream)
    #     all_lengths.append(
    #         current_output_lengths.clone())
    #         # Store lengths for this sub-chunk
    #     total_encoded_bytes += bytestream.numel()
    #
    # # Combine results from sub-chunks
    # final_bytestream = torch.cat(
    #     all_bytestreams) if all_bytestreams else torch.empty(
    #         0, dtype=torch.uint8, device=kv.device)
    # # We need a way to reconstruct the data from the final_bytestream.
    # # The original CacheGenGPUEncoderOutput structure implies separate
    # # bytestreams per sub-chunk.
    # # Let's revert to the previous structure but ensure cloning is correct.

    data_chunks = []
    output_buffer.zero_()  # Ensure buffer is clean
    output_lengths.zero_()  # Ensure lengths are clean

    for i in range(0, chunk_size, CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK):
        start = i
        end = min(i + CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK, chunk_size)
        current_encode_input = encode_input[:, start:end, :]
        current_ntokens = end - start

        # Reset lengths before each call, as encode_ntokens calculates lengths
        # for *this* input
        output_lengths.zero_()

        bytestream = encode_ntokens(
            cdf_int,
            current_encode_input,
            output_buffer,  # Buffer can be reused if large enough
            output_lengths  # Lengths are specific to this call
        )

        data_chunks.append(
            CacheGenGPUBytestream(
                bytestream=bytestream,  # bytestream is newly created tensor
                bytestream_lengths=output_lengths.clone(
                ),  # Clone lengths *after* the call
                ntokens=current_ntokens,
            ))

    return CacheGenGPUEncoderOutput(
        data_chunks=data_chunks,  # List of bytestreams per sub-chunk
        cdf=cdf_int,  # Should this be float or int? Decoder needs float? Let's
        # store int for now.
        max_tensors_key=max_tensors_key,
        max_tensors_value=max_tensors_value,
        num_heads=num_heads,
        head_size=head_size,
    )


class CacheGenSerializer(Serializer):

    def __init__(self, config: LMCacheEngineConfig,
                 metadata: LMCacheEngineMetadata):
        self.cachegen_config = CacheGenConfig.from_model_name(
            metadata.model_name)
        # Fallback for models not explicitly defined
        if self.cachegen_config is None:
            logger.warning(
                f"CacheGenConfig not found for model {metadata.model_name}. "
                f"Using default.")
            # Provide a default config - this needs actual values
            # For now, let's raise an error or use dummy values that might
            # fail
            # raise ValueError(f"CacheGenConfig not found for model "
            #                  f"{metadata.model_name}")
            # Using dummy values - replace with actual defaults if possible
            self.cachegen_config = CacheGenConfig(
                nlayers=metadata.
                num_layers,  # Assuming metadata has num_layers
                kspecs=[CGBasics.QuantizationSpec(0, metadata.num_layers, 16)
                        ],  # Default: 16 bins for all key layers
                vspecs=[CGBasics.QuantizationSpec(0, metadata.num_layers, 16)
                        ]  # Default: 16 bins for all value layers
            )

        self.chunk_size = config.chunk_size
        self.fmt = metadata.fmt
        # Ensure bins are created on the correct device later if needed
        self._key_bins = None
        self._value_bins = None
        self._bins_device = None

    def _ensure_bins_on_device(self, device: torch.device):
        if (self._key_bins is None or self._value_bins is None
                or self._bins_device != device):
            logger.debug(f"Creating/moving CacheGen bins to device: {device}")
            key_bins_cpu = torch.zeros(self.cachegen_config.nlayers,
                                       dtype=torch.int32)
            for spec in self.cachegen_config.kspecs:
                key_bins_cpu[spec.start_layer:spec.end_layer] = spec.bins
            self._key_bins = key_bins_cpu.to(device)

            value_bins_cpu = torch.zeros(self.cachegen_config.nlayers,
                                         dtype=torch.int32)
            for spec in self.cachegen_config.vspecs:
                value_bins_cpu[spec.start_layer:spec.end_layer] = spec.bins
            self._value_bins = value_bins_cpu.to(device)
            self._bins_device = device

    @property
    def key_bins(self) -> torch.Tensor:
        if self._key_bins is None:
            raise RuntimeError("Bins accessed before device was set.")
        return self._key_bins

    @property
    def value_bins(self) -> torch.Tensor:
        if self._value_bins is None:
            raise RuntimeError("Bins accessed before device was set.")
        return self._value_bins

    @_lmcache_nvtx_annotate
    def to_bytes(self, tensor: torch.Tensor) -> bytes:
        """
        Serialize a pytorch tensor (KV cache blob) to bytes
        using CacheGen encoding.

        Input:
            tensor: The input KV cache tensor blob. Expected shapes:
                    vLLM: [2, num_layers, num_tokens, num_heads, head_size]
                    HF:   [2, num_layers, num_heads, num_tokens, head_size]
                    The tensor should be on a CUDA device.

        Returns:
            bytes: The serialized bytes representing the encoded KV cache.
        """
        if not tensor.is_cuda:
            raise ValueError(
                "CacheGenSerializer requires input tensor to be on CUDA device."
            )

        # Ensure quantization bins are on the same device as the tensor
        self._ensure_bins_on_device(tensor.device)

        # Determine number of tokens based on format
        if self.fmt == "vllm":
            ntokens = tensor.shape[2]
        elif self.fmt == "huggingface":
            ntokens = tensor.shape[3]
        else:
            raise ValueError(f"Invalid format: {self.fmt}")

        # The old permutation logic is removed; handled in _split_kv now.

        # Call the encoding function
        output_dict = encode_function(
            tensor,  # Pass the tensor directly
            self.fmt,
            self.cachegen_config,
            self.key_bins,
            self.value_bins,
            ntokens,  # Pass the number of tokens
        )
        return output_dict.to_bytes()
