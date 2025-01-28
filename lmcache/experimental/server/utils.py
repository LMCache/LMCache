from typing import List
import torch

from lmcache.experimental.memory_management import MemoryFormat

# TODO(Jiayi): Maybe move the memory management in remote
# cache server to `memory_management.py` as well.
class LMSMemoryObj:
    data: bytearray
    length: int
    fmt: MemoryFormat
    dtype: torch.dtype
    shape: torch.Size