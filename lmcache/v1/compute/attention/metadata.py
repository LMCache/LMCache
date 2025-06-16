# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from dataclasses import dataclass

# Third Party
import torch


@dataclass
class LMCAttnMetadata:
    pass


@dataclass
class LMCFlashAttnMetadata(LMCAttnMetadata):
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_query_len: torch.Tensor
    max_seq_len: torch.Tensor
