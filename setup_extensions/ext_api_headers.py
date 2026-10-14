# SPDX-License-Identifier: Apache-2.0
"""Native extension API headers shipped in the lmcache wheel.

``build_py`` copies these from ``csrc/`` to ``lmcache/include/`` under the same
relative paths; ``lmcache.get_include()`` returns that directory so separately
built CUDA extensions can reuse LMCache's kernels and call its
``cuda_ops._ext_api`` table. The set must be closed under the headers' local
``#include "..."`` directives (checked by tests/test_ext_api_headers.py).
"""

EXT_API_HEADERS: tuple[str, ...] = (
    "engine_kv_format.h",
    "kv_transfer_types.h",
    "kv_transfer_plan_types.h",
    "cuda/transfer_plan_types.cuh",
    "cuda/mp_mem_kernels_detail.cuh",
    "cuda/lmcache_ext_api.h",
)
