# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Optional, Union
import abc

# Third Party
import torch

if TYPE_CHECKING:
    # Third Party
    pass


class LookupClientInterface(metaclass=abc.ABCMeta):
    """Abstract interface for lookup clients."""

    @abc.abstractmethod
    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
    ) -> Optional[int]:
        """
        Perform lookup for the given token IDs.

        Args:
            token_ids: The token IDs to lookup

            lookup_id: The lookup ID to associate with the lookup

            request_configs: The configs of the request,
            includes tags and the other configs

        Returns:
            The number of tokens that can be loaded from cache.
            None indicates the lookup/prefetch is in progress.
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

    def clear_lookup_status(self, lookup_id: str) -> None:
        """
        Clear temporary lookup status for a given lookup ID.

        Args:
            lookup_id: The lookup ID whose status needs to be cleared.
        """
        return
