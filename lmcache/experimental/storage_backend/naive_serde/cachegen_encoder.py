from typing import Tuple

import torch

import lmcache.c_ops as lmc_ops
from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.memory_management import (BytesBufferMemoryObj,
                                                    MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.storage_backend.naive_serde import \
    cachegen_basics as CGBasics
from lmcache.experimental.storage_backend.naive_serde.cachegen_basics import (
    CacheGenConfig, CacheGenGPUBytestream, CacheGenGPUEncoderOutput)
from lmcache.experimental.storage_backend.naive_serde.serde import Serializer
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)


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
    factor = MAX / max1  # shape [nlayers, ntokens, 1]
    xq = torch.round(input_groups * factor + MAX).to(
        torch.int8)  # shape [nlayers, ntokens, nchannels]

    return xq, max1


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
    return torch.unbind(tensor.reshape(num_layers, 2, num_tokens,
                                       num_heads * head_size),
                        dim=1)


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
def encode_ntokens(cdf_int, encode_input, output_buffer,
                   output_lengths) -> torch.Tensor:
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
    encode_input = torch.cat((new_key, new_value),
                             dim=0).reshape(nlayers, chunk_size, nchannels)

    new_cdf_key = lmc_ops.calculate_cdf(new_key, int(key_bins.max()))
    new_cdf_value = lmc_ops.calculate_cdf(new_value, int(value_bins.max()))
    cdf_int = torch.cat([new_cdf_key, new_cdf_value])

    output_buffer = torch.zeros(
        (nlayers, nchannels, CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK),
        dtype=torch.uint8,
        device=encode_input.device,
    )
    output_lengths = torch.zeros((nlayers, nchannels),
                                 dtype=torch.int32,
                                 device=encode_input.device)

    data_chunks = []
    for i in range(0, chunk_size, CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK):
        start = i
        end = min(i + CGBasics.CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK, chunk_size)
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
            ))

    return CacheGenGPUEncoderOutput(
        data_chunks,
        cdf_int,
        max_tensors_key=max_tensors_key,
        max_tensors_value=max_tensors_value,
        num_heads=num_heads,
        head_size=head_size,
    )


class CacheGenSerializer(Serializer):

    def __init__(self, config: LMCacheEngineConfig,
                 metadata: LMCacheEngineMetadata,
                 memory_allocator: MemoryAllocatorInterface):
        self.cachegen_config = CacheGenConfig.from_model_name(
            metadata.model_name)
        self.chunk_size = config.chunk_size
        self.fmt = metadata.fmt
        self.key_bins = self.make_key_bins(self.cachegen_config)
        self.value_bins = self.make_value_bins(self.cachegen_config)

        self.memory_allocator = memory_allocator

    def make_key_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.nlayers)
        for spec in config.kspecs:
            ret[spec.start_layer:spec.end_layer] = spec.bins
        return ret.cuda()

    def make_value_bins(self, config: CacheGenConfig) -> torch.Tensor:
        ret = torch.zeros(config.nlayers)
        for spec in config.vspecs:
            ret[spec.start_layer:spec.end_layer] = spec.bins
        return ret.cuda()

    # TODO(Jiayi): A lot of memory copies can be avoided in this function.
    @_lmcache_nvtx_annotate
    def serialize(self, memory_obj: MemoryObj) -> BytesBufferMemoryObj:
        """
        Serialize a KV_BLOB MemoryObj to CACHEGEN_BINARY MemoryObj.

        Input:
            memory_obj: the memory object to be serialized.

        Returns:
            MemoryObj: the serialized binary memory object.
        """

        # TODO(Jiayi): please avoid this copy by directly performing
        # serialization inside gpu connector.
        assert memory_obj.tensor is not None
        tensor = memory_obj.tensor.cuda()

        # Temporary fix for issue #83: encoder will have the default device 0
        # on all the ray workers. Need to set it to the correct device.
        # Also need to figure out why this happens.
        if torch.cuda.current_device != tensor.device:
            torch.cuda.set_device(tensor.device)
        if tensor.device != self.key_bins.device:
            self.key_bins = self.key_bins.to(tensor.device)
        if tensor.device != self.value_bins.device:
            self.value_bins = self.value_bins.to(tensor.device)

        # TODO(Jiayi): remove hardcoded "2"
        """ expecting a tensor of shape 
        [num_layers, 2, num_tokens, num_heads, head_size] """
        ntokens = tensor.shape[2]
        output_dict = encode_function(
            tensor,
            self.cachegen_config,
            self.key_bins,
            self.value_bins,
            ntokens,
        )

        return BytesBufferMemoryObj(output_dict.to_bytes())
