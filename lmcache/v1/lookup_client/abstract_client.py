# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional
import abc

# Third Party
import torch

if TYPE_CHECKING:
    # Third Party
    pass


class LookupClientInterface(metaclass=abc.ABCMeta):
    """Abstract interface for lookup clients."""

    @abc.abstractmethod
    def lookup(self, token_ids: torch.Tensor, request_id: Optional[str] = None) -> int:
        """
        Perform lookup for the given token IDs.

        Args:
            token_ids: The token IDs to lookup

            request_id: The request ID to associate with the lookup

        Returns:
            The number of tokens that can be loaded from cache
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Close the lookup client and clean up resources."""
        raise NotImplementedError

    def supports_producer_reuse(self) -> bool:
        """
        Return whether this lookup client supports producer KV cache reuse.

        Returns:
            True if producer reuse is supported, False otherwise
        """
        return False
