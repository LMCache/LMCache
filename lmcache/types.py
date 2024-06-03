import torch
from typing import Tuple

# Type definition
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
