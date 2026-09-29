# SPDX-License-Identifier: Apache-2.0
"""Placement policy for serving-engine writes into peer L1 managers."""

# Standard
from typing import Protocol

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.storage_controllers.utils import L1ManagerDescriptor


class WritePolicy(Protocol):
    """Select one L1 per key; keys omitted from the plan are not reserved."""

    def select_write_targets(
        self, keys: list[ObjectKey], managers: list[L1ManagerDescriptor]
    ) -> dict[int, list[ObjectKey]]:
        """Return manager-index -> keys; each key may occur at most once."""
        ...


class DefaultWritePolicy:
    """Write to the first DRAM L1, preferring the compatibility tag _default."""

    def select_write_targets(
        self, keys: list[ObjectKey], managers: list[L1ManagerDescriptor]
    ) -> dict[int, list[ObjectKey]]:
        """Select a DRAM target, or the sole L1 for single-backend deployments.

        Args:
            keys: Objects being written by the serving engine.
            managers: Available L1 descriptors in configuration order.

        Returns:
            A single-target plan, or an empty plan when no target is available.
        """
        dram = [d for d in managers if d.config.gds_l1_config is None]
        targets = sorted(dram, key=lambda d: d.config.tag != "_default")
        if not targets and len(managers) == 1:
            targets = managers
        return {targets[0].index: keys} if targets else {}
