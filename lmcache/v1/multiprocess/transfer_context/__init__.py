# SPDX-License-Identifier: Apache-2.0
"""Transport package for non-GPU KV data transfer in multiprocess mode.

Re-exports all public symbols from the sub-modules so that existing imports
from ``lmcache.v1.multiprocess.transfer_context`` work without specifying the
sub-module.

The worker-side contexts now live under :mod:`lmcache.v1.client` and stay
re-exported here, because this package path is imported from outside the
repository: the out-of-tree coordinator integration suite builds its KV client
from it. They resolve lazily; see :func:`__getattr__`.
"""

# Standard
from typing import TYPE_CHECKING, Any
import importlib

# First Party
from lmcache.v1.multiprocess.transfer_mode import MPTransferMode

# Local
from .base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
    compute_kv_layout,
    create_engine_driven_context,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from .inline import EngineDrivenContextInline
from .shm import EngineDrivenContextShm, ShmSlotDescriptor

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.client.async_engine_driven import (
        AsyncEngineDrivenTransferContext,
    )
    from lmcache.v1.client.worker_transfer import (
        EngineDrivenTransferContext,
        LMCacheDrivenTransferContext,
        TransferContext,
        create_transfer_context,
    )

WORKER_TRANSFER_NAMES = {
    "AsyncEngineDrivenTransferContext": "lmcache.v1.client.async_engine_driven",
    "EngineDrivenTransferContext": "lmcache.v1.client.worker_transfer",
    "LMCacheDrivenTransferContext": "lmcache.v1.client.worker_transfer",
    "TransferContext": "lmcache.v1.client.worker_transfer",
    "create_transfer_context": "lmcache.v1.client.worker_transfer",
}
"""The public names this package re-exports, and the module each now lives in."""

__all__ = [
    "AsyncEngineDrivenTransferContext",
    "EngineDrivenContext",
    "EngineDrivenContextInline",
    "EngineDrivenContextMetadata",
    "EngineDrivenContextShm",
    "EngineDrivenTransferContext",
    "LMCacheDrivenTransferContext",
    "MPTransferMode",
    "ShmSlotDescriptor",
    "TransferContext",
    "compute_kv_layout",
    "create_engine_driven_context",
    "create_transfer_context",
    "gather_paged_kv_to_cpu",
    "scatter_cpu_to_paged_kv",
]


def __getattr__(name: str) -> Any:
    """Resolve the worker-side contexts from their own modules.

    An eager import here would be circular: those modules import :mod:`.base`
    from this package, so importing them while this package is initialising
    would find a half-built module.

    Args:
        name: The attribute requested from this package.

    Returns:
        The attribute, taken from the module that now defines it.

    Raises:
        AttributeError: If this package defines no such attribute.
    """
    module_name = WORKER_TRANSFER_NAMES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)
