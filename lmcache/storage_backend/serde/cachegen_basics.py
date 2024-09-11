import torch
import io
import pickle
from dataclasses import dataclass
from typing import List
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.logging import init_logger
logger = init_logger(__name__)

CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK = 256

@dataclass
class CacheGenConfig:
    # TODO: move this class to another file like "cachegen_basics.py"
    key_first_layers: int
    key_second_layers: int
    key_third_layers: int
    key_first_bins: int
    key_second_bins: int
    key_third_bins: int
    value_first_layers: int
    value_first_bins: int
    value_second_bins: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    @staticmethod
    def from_model_name(model_name: str) -> "CacheGenConfig":
        family_7b = ["mistralai/Mistral-7B-Instruct-v0.2", "lmsys/longchat-7b-16k"]
        family_9b = ["THUDM/glm-4-9b-chat"]
        if model_name in family_7b:
            return CacheGenConfig(
                key_first_layers=10, 
                key_second_layers=20,
                key_third_layers=32, # total layers
                key_first_bins=32,
                key_second_bins=16,
                key_third_bins=16,
                value_first_layers=2,
                value_first_bins=32,
                value_second_bins=16
            )
        # TODO(Jiayi): needs tuning for better quality
        elif model_name in family_9b:
            return CacheGenConfig(
                key_first_layers=10,
                key_second_layers=20,
                key_third_layers=40,
                key_first_bins=32,
                key_second_bins=16,
                key_third_bins=16,
                value_first_layers=2,
                value_first_bins=32,
                value_second_bins=16
            )
        else:
            raise ValueError(f"Model {model_name} is not supported")

@dataclass
class CacheGenEncoderOutput:
    # TODO: maybe use numpy array so that we can directly tobytes() and frombuffer() to have a better performance
    bytestream: bytes
    start_indices: torch.Tensor
    cdf: torch.Tensor
    max_tensors_key: torch.Tensor
    max_tensors_value: torch.Tensor
    num_heads: int
    head_size: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    def to_bytes(self) -> bytes:
        """ Save the output to a file """
        with io.BytesIO() as f:
            #torch.save(self, f)
            pickle.dump(self, f)
            return f.getvalue()

    @staticmethod
    def from_bytes(bs: bytes) -> "CacheGenEncoderOutput":
        with io.BytesIO(bs) as f:
            return pickle.load(f)

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
        """ Save the output to a file """
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
        logger.debug(f"bytestream_lengths device: {self.data_chunks[0].bytestream_lengths.device}")
        logger.debug(f"cdf device: {self.cdf.device}")
        logger.debug(f"max_tensors_key device: {self.max_tensors_key.device}")
        logger.debug(f"max_tensors_value device: {self.max_tensors_value.device}")