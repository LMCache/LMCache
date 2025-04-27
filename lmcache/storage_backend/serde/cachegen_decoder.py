from typing import List, Optional

import torch

import lmcache.c_ops as lmc_ops
import lmcache.storage_backend.serde.cachegen_basics as CGBasics
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend.serde.cachegen_basics import (
    CacheGenConfig, CacheGenGPUBytestream, CacheGenGPUEncoderOutput)
from lmcache.storage_backend.serde.serde import Deserializer
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


@_lmcache_nvtx_annotate
def quant(bins: int, xq: torch.Tensor, max1: float):
    """Helper for dequantization logic (seems unused directly)."""
    C = bins // 2 - 1
    # Avoid division by zero
    C_tensor = torch.tensor(C, dtype=xq.dtype, device=xq.device)
    # Handle potential C=0 case (e.g., bins=2) although unlikely for
    # typical quantization
    if C == 0:
        return torch.zeros_like(xq)
    x = xq / C_tensor * max1
    return x


def do_dequantize(t: torch.Tensor, bins: torch.Tensor,
                  maxtensors: torch.Tensor):
    """
    Dequantize the tensor.
    t: Quantized tensor [nlayers, ntokens, nchannels] (int8 expected, shifted)
    bins: Quantization bins per layer [nlayers]
    maxtensors: Max values used during quantization [nlayers, ntokens, 1]
    """
    # Ensure bins match the device of the tensor
    if bins.device != t.device:
        bins = bins.to(t.device)

    C = (bins // 2 - 1)[:, None, None]  # Shape [nlayers, 1, 1]
    # Shift back to center around zero
    # Convert t to float before operations to avoid overflow/precision issues
    t_float = t.float() - C.float()

    # Handle potential C=0 case (bins=2) to avoid division by zero
    # Where C is 0, the division result should be 0
    # (or handled appropriately)
    # We can set C to 1 where it's 0, as t_float will also be 0 or -1 in\n    # that range.
    C_safe = torch.where(C == 0, torch.tensor(1.0, device=C.device), C.float())

    t_float = t_float / C_safe
    t_float = t_float * maxtensors.float()  # Ensure maxtensors is float
    return t_float


# @_lmcache_nvtx_annotate
# def recombine_bytes(bytes_tensor, output_lengths) -> torch.Tensor:
#     """Helper function, potentially for reconstructing buffer\n    (seems unused directly)."""
#     # This function seems complex and might not be robust.
#     # It assumes a specific way bytes are packed, which might not hold.
#     # If decode_fast_prefsum works correctly, this might not be needed.
#     # Keeping it for reference, but marked as potentially problematic.
#     logger.warning("recombine_bytes is potentially unused or unreliable.")
#     output_buffer_size = CGBasics.CACHEGEN_GPU_BUFFER_SIZE_PER_CHUNK
#     # Assuming fixed buffer size used in encoding
#
#     # Ensure lengths are valid
#     valid_lengths = torch.clamp(output_lengths.flatten(), 0,
#                                 output_buffer_size)
#
#     # Calculate starting offsets within the flattened theoretical buffer
#     layer_channel_offsets = torch.arange(
#         output_lengths.numel(),
#         device=output_lengths.device) * output_buffer_size
#
#     # Create indices for the actual packed bytes_tensor
#     packed_indices = torch.arange(bytes_tensor.numel(),
#                                   device=bytes_tensor.device)
#
#     # Create indices representing where these bytes *would* be in\n    # the full buffer
#     # This requires knowing the structure used by collect_bytes in the encoder
#     # Let's assume collect_bytes logic:
#     repeated_starts = layer_channel_offsets.repeat_interleave(valid_lengths)
#     arange_lengths = torch.cat(
#         [torch.arange(l, device=output_lengths.device) for l in\n#          valid_lengths])
#     full_buffer_indices = repeated_starts + arange_lengths
#
#     # Create the full buffer and scatter the bytes
#     full_buffer = torch.zeros(output_lengths.numel() * output_buffer_size,
#                               dtype=torch.uint8,
#                               device=bytes_tensor.device)
#     # Ensure indices are within bounds before scattering
#     valid_packed_indices = packed_indices[packed_indices <
#                                           full_buffer_indices.numel()]
#     valid_full_buffer_indices = full_buffer_indices[full_buffer_indices <
#                                                     full_buffer.numel()]
#
#     # Ensure shapes match for scattering
#     scatter_len = min(valid_packed_indices.numel(),
#                       valid_full_buffer_indices.numel())
#     if scatter_len > 0:
#         full_buffer[valid_full_buffer_indices[:scatter_len]] = bytes_tensor[
#             valid_packed_indices[:scatter_len]]
#     else:
#         logger.warning("No valid indices for scattering in recombine_bytes.")
#
#     return full_buffer.reshape(output_lengths.shape[0],
#                                output_lengths.shape[1], output_buffer_size)


@_lmcache_nvtx_annotate
def decode_chunk(
    cdf: torch.Tensor,
    data_chunk: CacheGenGPUBytestream,
    target_buffer: torch.Tensor,
) -> None:
    """
    Decode a single data chunk into the target buffer using cumulative lengths.

    Args:
        cdf: The CDF tensor [2 * nlayers, nchannels, num_bins + 1],\n           int16 expected.
        data_chunk: Contains bytestream and lengths for this chunk.
        target_buffer: The output buffer slice for this chunk
                       [2 * nlayers, ntokens_in_chunk, nchannels], uint8\n                       expected.
    """
    bytes_tensor = data_chunk.bytestream
    if bytes_tensor is None or bytes_tensor.numel() == 0:
        logger.warning("Empty bytestream in decode_chunk, skipping.")
        target_buffer.zero_()  # Ensure buffer is zeroed if no data
        return

    # Ensure lengths are on the correct device
    bytestream_lengths = data_chunk.bytestream_lengths.to(cdf.device)

    # Calculate cumulative sum of lengths for decode_fast_prefsum
    # Shape: [2 * nlayers, nchannels]
    length_prefsum = (bytestream_lengths.flatten().cumsum(0).reshape(
        bytestream_lengths.shape))

    lmc_ops.decode_fast_prefsum(cdf, bytes_tensor, length_prefsum,
                                target_buffer)


@_lmcache_nvtx_annotate
def decode_function_gpu(
        cdf: torch.Tensor,
        data_chunks: List[CacheGenGPUBytestream],
        layers_in_key: int,
        chunk_size:
    int,  # Total number of tokens expected across all data_chunks
        output_buffer: torch.Tensor,  # Pre-allocated buffer
):
    """
    Decode the KV cache from encoded data chunks into a pre-allocated buffer.

    Inputs:
        cdf: The CDF tensor [2 * nlayers, nchannels, num_bins + 1],\n           int16 expected.
        data_chunks: List of encoded data chunks.
        layers_in_key: Number of original layers (before K/V stacking).
        chunk_size: Total number of tokens across all data_chunks.
        output_buffer: Pre-allocated uint8 buffer on GPU,
                       Shape: [2 * nlayers, chunk_size, nchannels].

    Outputs:
        Tuple[torch.Tensor, torch.Tensor]: Decoded key and value tensors,
            both with shape [nlayers, ntokens, nchannels], still\n             quantized (uint8).
    """
    nlayers_stacked, nchannels, _ = cdf.shape
    assert output_buffer.shape == (nlayers_stacked, chunk_size, nchannels),
        f"Output buffer shape mismatch: expected {(nlayers_stacked, chunk_size, nchannels)}, got {output_buffer.shape}"
    assert nlayers_stacked == 2 * layers_in_key, "CDF layer dimension mismatch"

    output_buffer.zero_()  # Ensure buffer is clean

    start_token_idx = 0
    for data_chunk in data_chunks:
        ntokens_in_chunk = data_chunk.ntokens
        if ntokens_in_chunk == 0:
            continue

        end_token_idx = start_token_idx + ntokens_in_chunk
        if end_token_idx > chunk_size:
            logger.error(
                (f"Token count mismatch: cumulative tokens ({end_token_idx}) "
                 f"exceed expected chunk size ({chunk_size}). Truncating.")
            )
            ntokens_in_chunk = chunk_size - start_token_idx
            end_token_idx = chunk_size
            if ntokens_in_chunk <= 0:
                break  # Stop if we've somehow exceeded the buffer

        # Get the slice of the output buffer for this chunk
        buffer_slice = output_buffer[:, start_token_idx:end_token_idx, :]

        decode_chunk(cdf, data_chunk, buffer_slice)
        start_token_idx = end_token_idx

    if start_token_idx != chunk_size:
        logger.warning(
            (f"Decoded tokens ({start_token_idx}) do not match expected "
             f"chunk size ({chunk_size}). The rest of the buffer will be zero."
             ))

    # output_buffer now contains the quantized data
    # Reshape and split into K and V
    # Shape: [2, nlayers, chunk_size, nchannels]
    out_reshaped = output_buffer.reshape(
        (2, layers_in_key, chunk_size, nchannels))
    key_quantized = out_reshaped[0]  # Shape: [nlayers, chunk_size, nchannels]
    value_quantized = out_reshaped[
        1]  # Shape: [nlayers, chunk_size, nchannels]

    # Return the quantized tensors
    return key_quantized, value_quantized


class CacheGenDeserializer(Deserializer):

    def __init__(self, config: LMCacheEngineConfig,
                 metadata: LMCacheEngineMetadata, dtype):
        self.dtype = dtype
        self.cachegen_config = CacheGenConfig.from_model_name(
            metadata.model_name)
        # Fallback for models not explicitly defined
        if self.cachegen_config is None:
            logger.warning(
                (f"CacheGenConfig not found for model {metadata.model_name}. "
                 "Using default."))
            # Provide a default config - this needs actual values
            # Assuming metadata has num_layers
            num_layers = getattr(metadata, 'num_layers',
                                 32)  # Default to 32 if not found
            self.cachegen_config = CacheGenConfig(
                nlayers=num_layers,
                kspecs=[CGBasics.QuantizationSpec(0, num_layers, 16)
                        ],  # Default: 16 bins for all key layers
                vspecs=[CGBasics.QuantizationSpec(0, num_layers, 16)
                        ]  # Default: 16 bins for all value layers
            )

        self.chunk_size = config.chunk_size
        self.output_buffer: Optional[torch.Tensor] = None
        self.fmt = metadata.fmt
        # Delay bin creation until device is known
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
            # Attempt to create on default CUDA device if not set
            if torch.cuda.is_available():
                self._ensure_bins_on_device(torch.device('cuda'))
            else:
                raise RuntimeError(
                    ("Bins accessed before device was set and CUDA not "
                     "available."))
        return self._key_bins

    @property
    def value_bins(self) -> torch.Tensor:
        if self._value_bins is None:
            if torch.cuda.is_available():
                self._ensure_bins_on_device(torch.device('cuda'))
            else:
                raise RuntimeError(
                    ("Bins accessed before device was set and CUDA not "
                     "available."))
        return self._value_bins

    def _ensure_output_buffer(self, nlayers_stacked: int, nchannels: int,
                              ntokens: int, device: torch.device):
        """Ensure the output buffer exists, has the right shape and device."""
        required_shape = (nlayers_stacked, ntokens, nchannels)
        if (self.output_buffer is None
                or self.output_buffer.shape != required_shape
                or self.output_buffer.device != device):
            logger.debug(
                (f"Creating decode output buffer with shape {required_shape} "
                 f"on device {device}"))
            self.output_buffer = torch.zeros(required_shape,
                                             dtype=torch.uint8,
                                             device=device)
        # Return the buffer (or a slice if ntokens < self.chunk_size,
        # though current logic uses full size)
        # Let's always return the full buffer required for the current
        # decode operation
        return self.output_buffer

    @_lmcache_nvtx_annotate
    def from_bytes(self, bs: bytes) -> torch.Tensor:
        """
        Deserialize bytes into a KV cache tensor blob using CacheGen decoding.

        Output tensor shape: [2, num_layers, ...] matching the format
        specified by self.fmt.
        """
        if not bs:
            logger.warning("Received empty bytes, returning empty tensor.")
            # Return an empty tensor with expected rank but zero dimensions
            # where appropriate
            # This is tricky without knowing num_layers etc. beforehand.
            # Returning a completely empty tensor might be safer.
            return torch.empty(0, dtype=self.dtype)

        encoder_output = CacheGenGPUEncoderOutput.from_bytes(bs)

        # Determine target device (prefer GPU if available)
        target_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        if target_device.type == 'cpu':
            # TODO: Implement CPU decoding path if needed
            raise RuntimeError(
                "CacheGenDeserializer currently requires CUDA for decoding.")

        # Move necessary data from encoder_output to target device
        encoder_output.move_to(target_device)

        # Ensure bins are on the target device
        self._ensure_bins_on_device(target_device)

        # Determine shapes and sizes from the decoded metadata
        ntokens = sum(chunk.ntokens for chunk in encoder_output.data_chunks)
        if ntokens == 0:
            logger.warning(
                "Decoded data chunks contain zero tokens, returning empty "
                "tensor.")
            return torch.empty(0, dtype=self.dtype, device=target_device)

        layers_in_key = self.cachegen_config.nlayers  # Use config nlayers
        nlayers_stacked, nchannels, _ = encoder_output.cdf.shape
        num_heads = encoder_output.num_heads
        head_size = encoder_output.head_size

        # Validate shapes
        if nlayers_stacked != 2 * layers_in_key:
            raise ValueError(
                (f"Inconsistent layer count: CDF={nlayers_stacked//2}, "
                 f"Config={layers_in_key}"))
        if nchannels != num_heads * head_size:
            raise ValueError((f"Inconsistent channel count: CDF={nchannels}, "
                              f"HeadInfo={num_heads*head_size}"))

        # Get or create the output buffer
        decode_buffer = self._ensure_output_buffer(nlayers_stacked, nchannels,
                                                   ntokens, target_device)

        # Decode into the buffer (populates decode_buffer in-place)
        # decode_function_gpu returns quantized K, V separately
        key_q, value_q = decode_function_gpu(
            encoder_output.cdf,  # Shape [2*nlayers, nchannels, bins+1], int16
            encoder_output.data_chunks,
            layers_in_key,
            ntokens,
            decode_buffer,  # Shape [2*nlayers, ntokens, nchannels], uint8
        )
        # key_q, value_q shape: [nlayers, ntokens, nchannels], uint8

        # Dequantize
        # max tensors shape: [nlayers, ntokens, 1]
        key = do_dequantize(key_q, self.key_bins,
                            encoder_output.max_tensors_key)
        value = do_dequantize(value_q, self.value_bins,
                              encoder_output.max_tensors_value)
        # key, value shape: [nlayers, ntokens, nchannels], float

        # Stack K and V
        # Shape: [2, nlayers, ntokens, nchannels]
        blob = torch.stack([key, value], dim=0)

        # Reshape to include heads and head_size
        # Shape: [2, nlayers, ntokens, num_heads, head_size]
        blob_reshaped = blob.reshape((
            2,
            layers_in_key,
            ntokens,
            num_heads,
            head_size,
        ))

        # Permute based on the expected format, convert to final dtype
        if self.fmt == "vllm":
            # Expected: [2, nlayers, ntokens, num_heads, head_size]
            # No permutation needed from blob_reshaped
            final_blob = blob_reshaped
        elif self.fmt == "huggingface":
            # Expected: [2, nlayers, num_heads, ntokens, head_size]
            # Permute ntokens and num_heads dimensions (2 and 3)
            final_blob = blob_reshaped.permute(0, 1, 3, 2, 4)
        else:
            raise RuntimeError("Unknown format %s" % self.fmt)

        return final_blob.contiguous().to(self.dtype)
