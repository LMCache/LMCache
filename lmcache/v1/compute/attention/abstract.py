# SPDX-License-Identifier: Apache-2.0
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
