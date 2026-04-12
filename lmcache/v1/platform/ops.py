# SPDX-License-Identifier: Apache-2.0
"""Centralized import of CUDA / non-CUDA operations.

Instead of every file doing::

    if torch.cuda.is_available():
        import lmcache.c_ops as lmc_ops
    else:
        import lmcache.non_cuda_equivalents as lmc_ops

All callers should use::

    from lmcache.v1.platform import lmc_ops
"""

# First Party
from lmcache.v1.platform.capabilities import HAS_CUDA

if HAS_CUDA:
    # First Party
    import lmcache.c_ops as lmc_ops
else:
    # First Party
    import lmcache.non_cuda_equivalents as lmc_ops  # type: ignore[assignment]

__all__ = ["lmc_ops"]
