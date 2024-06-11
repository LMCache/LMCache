import torch
import io
import pickle
from dataclasses import dataclass

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

        if model_name in family_7b:
            return CacheGenConfig(
                key_first_layers=10,
                key_second_layers=20,
                key_third_layers=32,
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
