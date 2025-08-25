# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import List, Tuple
import io
import pickle

# Third Party
from transformers import AutoConfig
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)

CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK = 256


@dataclass
class CacheGenGPUBytestream:
    bytestream: torch.Tensor
    bytestream_lengths: torch.Tensor  # [nlayers, nchannels, bytestream_length]
    ntokens: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)


@dataclass
class CacheGenGPUEncoderOutput:
    data_chunks: List[CacheGenGPUBytestream]
    cdf: torch.Tensor
    max_tensors_key: torch.Tensor
    max_tensors_value: torch.Tensor
    num_heads: int
    head_size: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    @_lmcache_nvtx_annotate
    def to_bytes(self) -> bytes:
        """Save the output to a file"""
        with io.BytesIO() as f:
            pickle.dump(self, f)
            return f.getvalue()

    @staticmethod
    @_lmcache_nvtx_annotate
    def from_bytes(bs: bytes) -> "CacheGenGPUEncoderOutput":
        with io.BytesIO(bs) as f:
            return pickle.load(f)

    def debug_print_device(self):
        logger.debug(f"bytestream device: {self.data_chunks[0].bytestream.device}")
        logger.debug(
            f"bytestream_lengths device: "
            f"{self.data_chunks[0].bytestream_lengths.device}"
        )
        logger.debug(f"cdf device: {self.cdf.device}")
        logger.debug(f"max_tensors_key device: {self.max_tensors_key.device}")
        logger.debug(f"max_tensors_value device: {self.max_tensors_value.device}")


@dataclass
class QuantizationSpec:
    start_layer: int
    end_layer: int
    bins: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)


@dataclass
class CacheGenConfig:
    # TODO: move this class to another file like "cachegen_basics.py"
    nlayers: int
    kspecs: List[QuantizationSpec]
    vspecs: List[QuantizationSpec]

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    @staticmethod
    def from_model_name(model_name: str) -> "CacheGenConfig":
        family_7b = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "lmsys/longchat-7b-16k",
            "Qwen/Qwen-7B",
        ]
        family_8b = ["meta-llama/Llama-3.1-8B-Instruct"]
        family_9b = ["THUDM/glm-4-9b-chat"]
        if model_name in family_7b:
            return CacheGenConfig(
                nlayers=32,
                kspecs=[
                    QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                    QuantizationSpec(start_layer=10, end_layer=32, bins=16),
                ],
                vspecs=[
                    QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                    QuantizationSpec(start_layer=2, end_layer=32, bins=16),
                ],
            )
        elif model_name in family_8b:
            return CacheGenConfig(
                nlayers=32,
                kspecs=[
                    QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                    QuantizationSpec(start_layer=10, end_layer=32, bins=16),
                ],
                vspecs=[
                    QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                    QuantizationSpec(start_layer=2, end_layer=32, bins=16),
                ],
            )
        # TODO(Jiayi): needs tuning for better quality
        elif model_name in family_9b:
            return CacheGenConfig(
                nlayers=40,
                kspecs=[
                    QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                    QuantizationSpec(start_layer=10, end_layer=40, bins=16),
                ],
                vspecs=[
                    QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                    QuantizationSpec(start_layer=2, end_layer=40, bins=16),
                ],
            )
        elif model_name == "test_model":
            return CacheGenConfig(
                nlayers=32,
                kspecs=[
                    QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                    QuantizationSpec(start_layer=10, end_layer=32, bins=16),
                ],
                vspecs=[
                    QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                    QuantizationSpec(start_layer=2, end_layer=32, bins=16),
                ],
            )
        else:
            try:
                config = AutoConfig.from_pretrained(model_name)
                # Default name caught by num_hidden_layers
                if config.num_hidden_layers is None:
                    raise ValueError(
                        f"num_hidden_layers is None for model {model_name}"
                    )
                if config.num_hidden_layers < 10:
                    return CacheGenConfig(
                        nlayers=config.num_hidden_layers,
                        kspecs=[
                            QuantizationSpec(
                                start_layer=0,
                                end_layer=config.num_hidden_layers,
                                bins=32,
                            ),
                        ],
                        vspecs=[
                            QuantizationSpec(
                                start_layer=0,
                                end_layer=config.num_hidden_layers,
                                bins=32,
                            ),
                        ],
                    )
                else:
                    return CacheGenConfig(
                        nlayers=config.num_hidden_layers,
                        kspecs=[
                            QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                            QuantizationSpec(
                                start_layer=10,
                                end_layer=config.num_hidden_layers,
                                bins=16,
                            ),
                        ],
                        vspecs=[
                            QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                            QuantizationSpec(
                                start_layer=2,
                                end_layer=config.num_hidden_layers,
                                bins=16,
                            ),
                        ],
                    )
            except Exception as e:
                raise ValueError(
                    f"Model {model_name} not supported by CacheGenConfig"
                ) from e


def _split_kv(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Split a blob KV tensor to K and V tensors with the merged heads

    Input:
        tensor: the KV tensor with shape
            [num_layers, 2, num_tokens, num_heads, head_size]

    Returns:
        K and V tensors with shape
            [num_layers, num_tokens, num_channels]
    """
    num_layers, _, num_tokens, num_heads, head_size = tensor.shape
    return torch.unbind(
        tensor.reshape(num_layers, 2, num_tokens, num_heads * head_size), dim=1
    )


@_lmcache_nvtx_annotate
def torch_quant_vectorized(
    bins: torch.Tensor, input_groups: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    max1 = torch.amax(
        torch.abs(input_groups), dim=-1, keepdim=True
    )  # shape [nlayers, ntokens, 1]
    factor = MAX / max1  # shape [nlayers, ntokens, 1]
    xq = torch.round(input_groups * factor + MAX).to(
        torch.int8
    )  # shape [nlayers, ntokens, nchannels]

    return xq, max1


@_lmcache_nvtx_annotate
def collect_bytes(output_buffer, output_lengths) -> torch.Tensor:
    """
    Collect a byte tensor from the output_buffer + output_lengths
    """
    output_buffer_size = output_buffer.shape[-1]
    flattened_lengths = output_lengths.flatten()
    flattened_buffer = output_buffer.flatten()
    summed_length = (output_buffer_size - flattened_lengths).cumsum(0)
    summed_length = summed_length.roll(1)
    summed_length[0] = 0
    indexes = summed_length.repeat_interleave(flattened_lengths)
    indexes = indexes + torch.arange(len(indexes), device=indexes.device)
    return flattened_buffer[indexes]


@_lmcache_nvtx_annotate
def encode_ntokens(
    cdf_int, encode_input, output_buffer, output_lengths
) -> torch.Tensor:
    """Encode a batch of ntokens.

    :param cdf_int: int16 tensor on GPU with shape [nlayers, nchannels, Lp]
    :param encode_input: int8 tensor on GPU with shape
    :param [nlayers, ntokens, nchannels]
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
    byte_tensor = collect_bytes(output_buffer, output_lengths)
    return byte_tensor


@_lmcache_nvtx_annotate
def encode_function(
    kv: torch.Tensor,
    config: CacheGenConfig,
    key_bins: torch.Tensor,
    value_bins: torch.Tensor,
    chunk_size: int,
) -> CacheGenGPUEncoderOutput:
    """
    Given the path to the original key value cache, encode the KV cache
    """
    num_heads, head_size = kv.shape[-2:]
    fp_k, fp_v = _split_kv(kv)
    nchannels = num_heads * head_size
    nlayers = fp_k.shape[0] + fp_v.shape[0]

    new_key, max_tensors_key = torch_quant_vectorized(key_bins, fp_k)
    new_value, max_tensors_value = torch_quant_vectorized(value_bins, fp_v)
    encode_input = torch.cat((new_key, new_value), dim=0).reshape(
        nlayers, chunk_size, nchannels
    )

    new_cdf_key = lmc_ops.calculate_cdf(new_key, int(key_bins.max()))
    new_cdf_value = lmc_ops.calculate_cdf(new_value, int(value_bins.max()))
    cdf_int = torch.cat([new_cdf_key, new_cdf_value])

    output_buffer = torch.zeros(
        (nlayers, nchannels, CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK),
        dtype=torch.uint8,
        device=encode_input.device,
    )
    output_lengths = torch.zeros(
        (nlayers, nchannels), dtype=torch.int32, device=encode_input.device
    )

    data_chunks = []
    for i in range(0, chunk_size, CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK):
        start = i
        end = min(i + CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK, chunk_size)
        bytestream = encode_ntokens(
            cdf_int,
            encode_input[:, start:end, :],
            output_buffer,
            output_lengths,
        )
        data_chunks.append(
            CacheGenGPUBytestream(
                bytestream=bytestream,
                bytestream_lengths=output_lengths.clone(),
                ntokens=end - start,
            )
        )

    return CacheGenGPUEncoderOutput(
        data_chunks,
        cdf_int,
        max_tensors_key=max_tensors_key,
        max_tensors_value=max_tensors_value,
        num_heads=num_heads,
        head_size=head_size,
    )


@_lmcache_nvtx_annotate
def decode_chunk(
    cdf: torch.Tensor,
    data_chunk: CacheGenGPUBytestream,
    target_buffer: torch.Tensor,
) -> None:
    """
    Write the decode output in target_buffer
    Expected shape: [nlayers (kv in total), ntokens, nchannels]
    """
    bytes_tensor = data_chunk.bytestream
    length_prefsum = (
        data_chunk.bytestream_lengths.flatten()
        .cumsum(0)
        .reshape(data_chunk.bytestream_lengths.shape)
    )
    lmc_ops.decode_fast_prefsum(cdf, bytes_tensor, length_prefsum, target_buffer)


@_lmcache_nvtx_annotate
def decode_function_gpu(
    cdf: torch.Tensor,
    data_chunks: List[CacheGenGPUBytestream],
    layers_in_key: int,
    chunk_size: int,
    output: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: dtype and shape -- still have 128 and 8
    """
    Given the path to the encoded KV bytestream, decode the KV cache

    Inputs:
        cdf: the cdf tensor, in shape [2 * nlayers, nchannels, bins + 1]
        data_chunks: the data_chunks in the encoder's output
        layers_in_key: number of layers in K (or V)
        (K/V should have the same number of layers)
        chunk_size: the chunk_size
        output: output buffer, in shape [ntokens, 2 * nlayers * nchannels]

    Outputs:
        key: the decoded key tensor in the shape of (layers, tokens, nchannels)
        value: the decoded value tensor in the shape of
        (layers, tokens, nchannels)
    """
    nlayers, nchannels, _ = cdf.shape
    output = output.reshape((nlayers, chunk_size, nchannels))

    start = 0
    for data_chunk in data_chunks:
        end = start + data_chunk.ntokens
        decode_chunk(cdf, data_chunk, output[:, start:end, :])
        start = end

    out = output.reshape((2, layers_in_key, chunk_size, nchannels))
    key, value = out.float()

    return key, value


def do_dequantize(
    t: torch.Tensor, bins: torch.Tensor, maxtensors: torch.Tensor
) -> torch.Tensor:
    """
    t: [nlayers, ntokens, nchannels]
    bins: [nlayers]
    maxtensors: [nlayers, ntokens, 1]
    """
    C = (bins // 2 - 1)[:, None, None]
    t = t - C
    t = t / C
    t = t * maxtensors
    return t
