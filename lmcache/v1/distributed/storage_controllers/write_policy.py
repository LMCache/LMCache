# SPDX-License-Identifier: Apache-2.0
"""Synchronous write-overflow candidate order, separate from allocation."""

# Standard
from dataclasses import dataclass


@dataclass
class OrderedWritePolicy:
    """Try the primary manager, then explicitly eligible fallback managers.

    ``manager_ids`` contains stable process-local identities, not list positions.
    The router retries only capacity failures and preserves allocation batches.
    """

    manager_ids: tuple[int, ...]

    def select_write_targets(self) -> tuple[int, ...]:
        """Return eligible manager identities in their explicit overflow order."""
        return self.manager_ids
