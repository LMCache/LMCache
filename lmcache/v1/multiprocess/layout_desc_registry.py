# SPDX-License-Identifier: Apache-2.0
"""Registry for memory layout descriptors used by multiprocess lookup."""

# Standard
import threading

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc


class LayoutDescRegistry:
    """Thread-safe registry mapping model/world pairs to layout descriptors.

    Modules write to this registry when KV caches are registered. Consumers
    such as ``LookupModule`` read from it to find layout descriptors for
    prefetch tasks.
    """

    def __init__(self) -> None:
        # Key: (model_name, world_size) -> owner -> MemoryLayoutDesc.
        # Multiple workers can register the same model/world pair, so
        # ownership has to survive individual unregister calls.
        self._registry: dict[
            tuple[str, int],
            dict[int | None, MemoryLayoutDesc],
        ] = {}
        self._lock = threading.Lock()

    def register(
        self,
        model_name: str,
        world_size: int,
        layout_desc: MemoryLayoutDesc,
        *,
        instance_id: int | None = None,
    ) -> None:
        """Register a layout descriptor for a model/world pair.

        Args:
            model_name: The model name.
            world_size: The world size.
            layout_desc: The memory layout descriptor.
            instance_id: Optional worker instance identifier owning the layout.
        """
        with self._lock:
            owners = self._registry.setdefault((model_name, world_size), {})
            owners[instance_id] = layout_desc

    def unregister(
        self,
        model_name: str,
        world_size: int,
        *,
        instance_id: int | None = None,
    ) -> None:
        """Remove an owned layout descriptor for a model/world pair.

        Args:
            model_name: The model name.
            world_size: The world size.
            instance_id: Optional worker instance identifier used during
                ``register``.
        """
        with self._lock:
            key = (model_name, world_size)
            owners = self._registry.get(key)
            if owners is None:
                return

            owners.pop(instance_id, None)
            if not owners:
                self._registry.pop(key, None)

    def find(self, model_name: str, world_size: int) -> MemoryLayoutDesc | None:
        """Look up a layout descriptor by model/world pair.

        Args:
            model_name: The model name.
            world_size: The world size.

        Returns:
            The layout descriptor if found, otherwise None.

        Note:
            Multiple owners for the same model/world pair are expected to have
            compatible layout descriptors. When several owners exist, this
            method returns the first live descriptor registered for the key.
        """
        with self._lock:
            owners = self._registry.get((model_name, world_size))
            if not owners:
                return None

            return next(iter(owners.values()))
