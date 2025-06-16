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
import abc

# Third Party
import torch

# First Party
from lmcache.v1.compute.attention.metadata import LMCFlashAttnMetadata


class AttentionInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "LMCFlashAttnMetadata",
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform forward pass of the attention mechanism.
        """
        raise NotImplementedError
