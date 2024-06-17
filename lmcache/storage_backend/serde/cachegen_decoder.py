import io
import pickle
import torchac_cuda
import numpy as np
import torch
from typing import Tuple, List, Any

from lmcache.storage_backend.serde.cachegen_basics import CacheGenConfig, CacheGenEncoderOutput
from lmcache.storage_backend.serde.serde import Deserializer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger

logger = init_logger(__name__)

def quant(bins: int, xq: torch.Tensor, max1: float):
    C = bins // 2 - 1
    x = (xq / C * max1).to(torch.float16)
    return x


def decode_function_gpu(
        cdf: torch.Tensor, 
        bits: bytes, 
        start_indices: torch.Tensor, 
        max_tensors_k: torch.Tensor, 
        max_tensors_v: torch.Tensor, 
        quantization_config: CacheGenConfig, 
        chunk_size: int, 
        output: torch.Tensor, 
        chunk_id):
    # TODO: dtype and shape -- still have 128 and 8
    """
    Given the path to the encoded KV bytestream, decode the KV cache

    Inputs:
        cdf: the cdf tensor, in shape [2 * nlayers, nchannels, bins + 1]
        bits: the encoded key value cache bytestream
        start_indices: start indices of each "bytestream element" in the encoded bytestream.
                       In shape [2 * nlayers * ntokens]
        max_tensors_k: the max tensor for key in shape [nlayers, ntokens, 1]
        max_tensors_v: the max tensor for value in shape [nlayers, ntokens, 1]
        quantization_config: the quantization config
        output: output buffer, in shape [ntokens, 2 * nlayers * nchannels]
        chunk_size: the chunk_size

    Outputs:
        key: the decoded key tensor in the shape of (layers, tokens, nchannels)
        value: the decoded value tensor in the shape of (layers, tokens, nchannels)
    """
    config = quantization_config
    np_array = np.frombuffer(bits, dtype=np.uint8)
    concated_string = torch.from_numpy(np_array)
    nlayers, nchannels, _ = cdf.shape

    start_indices = torch.tensor(start_indices).cuda()
    max_tensors_k = max_tensors_k.cuda()
    max_tensors_v = max_tensors_v.cuda()

    '''
    num_threads = chunk_size
    num_blocks = nlayers
    
    # FIXME(Jiayi): scale*num_thread = chunk_size; num_thread<1000 (32X)
    scale = 1
    '''
    
    num_blocks = nlayers
    
    if chunk_size < 1000:
        num_threads = chunk_size
        scale = 1
    elif chunk_size % 1000 == 0:
        num_threads = 1000
        scale = int(chunk_size/num_threads)
    else:
        raise Exception(f"The current cuda kernel does not support chunk size {chunk_size}") 
    
    torchac_cuda.decode_fast(
            output,
            cdf.unsqueeze(0),
            concated_string,
            start_indices,
            chunk_size,
            num_blocks,
            num_threads,
            scale)

    out = output.reshape((2, max_tensors_k.shape[0], chunk_size, nchannels))
    key, value = out.float()
    for l in range(key.shape[0]):
        if l < config["key_first_layers"]:
            bins = config["key_first_bins"]
        elif l < config["key_second_layers"]:
            bins = config["key_second_bins"]
        else:
            bins = config["key_third_bins"]
        key[l] = quant(bins, key[l] - (bins // 2 - 1), max_tensors_k[l, chunk_id * chunk_size: (chunk_id + 1) * chunk_size])

    for l in range(value.shape[0]):
        if l < config["value_first_layers"]:
            bins = config["value_first_bins"]
        else:
            bins = config["value_second_bins"]
        value[l] = quant(bins, value[l] - (bins // 2 - 1), max_tensors_v[l, chunk_id * chunk_size: (chunk_id + 1) * chunk_size])
    key = key.reshape(
        key.shape[0],
        key.shape[1],
        nchannels)
    value = value.reshape(
        value.shape[0],
        value.shape[1],
        nchannels)
    return key, value

class CacheGenDeserializer(Deserializer):
    def __init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata):
        self.cachegen_config = CacheGenConfig.from_model_name(metadata.model_name)
        self.chunk_size = config.chunk_size
        self.output_buffer = None
        self.fmt = metadata.fmt

    def get_output_buffer(self, nlayers: int, nchannels: int, ntokens: int):
        if self.output_buffer is None or self.output_buffer.shape[1] != 2 * nlayers * nchannels:
            self.output_buffer = torch.zeros((self.chunk_size, 2 * nlayers * nchannels), dtype=torch.int).cuda()
        return self.output_buffer[:ntokens, :]

    def from_bytes(self, bs: bytes) -> torch.Tensor:
        encoder_output = CacheGenEncoderOutput.from_bytes(bs)
        ntokens = encoder_output.max_tensors_key.shape[1]
        key, value = decode_function_gpu(
                encoder_output.cdf,
                encoder_output.bytestream,
                encoder_output.start_indices,
                encoder_output.max_tensors_key,
                encoder_output.max_tensors_value,
                self.cachegen_config,
                ntokens,
                self.get_output_buffer(encoder_output.cdf.shape[0] // 2, encoder_output.cdf.shape[1], ntokens),
                0)

        ''' merge key and value back and reshape '''
        nlayers, ntokens, nchannels = key.shape
        blob = torch.stack([key, value]) # [2, nlayers, ntokens, nchannels] 
        blob = blob.reshape((2, nlayers, ntokens, encoder_output.num_heads, encoder_output.head_size))\
        
        match self.fmt:
            case "vllm":
                return blob.permute((1, 0, 2, 3, 4)).to(torch.bfloat16) # [nlayers, 2, ntokens, num_heads, head_size]
            case "huggingface":
                return blob.permute((1, 0, 3, 2, 4)).to(torch.float16) # [nlayers, 2, num_heads, ntokens, head_size]
