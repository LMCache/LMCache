# SPDX-License-Identifier: Apache-2.0
"""CPU-specific platform primitives.

Importing this package self-registers the POSIX-SHM KV-cache wrapper
factory with :mod:`lmcache.v1.platform._registry`, so the dispatch
in :mod:`lmcache.integration.vllm.vllm_multi_process_adapter` can
pick the right wrapper based on ``tensor.device.type`` without any
if/elif chain.
"""

# Standard
from typing import Any

# Third Party
import torch

# First Party
from lmcache.v1.platform._registry import register_kv_wrapper


def _kv_wrapper_factory(tensor: torch.Tensor) -> Any:
    """Indirect-dispatch wrapper.

    Defers loading :mod:`lmcache.v1.platform.cpu.shm` (which pulls in
    ``multiprocess.custom_types``) until first use, so importing this
    package during ``lmcache/__init__.py``'s bootstrap does not race
    other imports that touch ``torch_dev``.
    """
    # First Party
    from lmcache.v1.platform.cpu.shm import migrate_to_shm_and_wrap

    return migrate_to_shm_and_wrap(tensor)


register_kv_wrapper("cpu", _kv_wrapper_factory)
