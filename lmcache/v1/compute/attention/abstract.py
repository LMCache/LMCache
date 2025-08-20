# SPDX-License-Identifier: Apache-2.0
# Standard
import abc

# Third Party
import torch

# First Party


class AttentionInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "LMCAttnMetadata",
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform forward pass of the attention mechanism.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def init_attn_metadata(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> "LMCAttnMetadata":
        """
        Initialize attention metadata.
        """
        raise NotImplementedError
