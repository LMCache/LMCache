# SPDX-License-Identifier: Apache-2.0
"""Opt-in engine-local token-drop allocation adapter for pinned vLLM.

Select with vLLM's --scheduler-cls option. This module intentionally does not
replace KVCacheManager.allocate_slots on the class or mutate logical Requests.

Experimental: the custom Scheduler interface is internal to vLLM 0.25.1.
"""

# Future
from __future__ import annotations

# Third Party
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler

# First Party
from lmcache.integration.vllm.experimental.token_drop_adapter import (
    _allocate_with_resident_frontier,
    _check_vllm_version,
)


class _EngineLocalKVCacheManagerProxy:
    """Override only allocation at this Scheduler's manager boundary."""

    def __init__(self, manager: KVCacheManager):
        self._manager = manager

    def __getattr__(self, name):
        return getattr(self._manager, name)

    def allocate_slots(self, request, num_new_tokens, *args, **kwargs):
        # Call the stock unbound method, not a patched module-level function.
        # Reuse S2's resident-length state, validation, and preemption guards.
        return _allocate_with_resident_frontier(
            type(self._manager).allocate_slots,
            self._manager,
            request,
            num_new_tokens,
            *args,
            **kwargs,
        )


class LMCacheTokenDropScheduler(Scheduler):
    """Use the stock vLLM Scheduler with an instance-local manager proxy."""

    def __init__(self, *args, **kwargs):
        _check_vllm_version()
        super().__init__(*args, **kwargs)
        if type(self.kv_cache_manager) is not KVCacheManager:
            raise RuntimeError("Token dropping requires the stock KVCacheManager")
        self.kv_cache_manager = _EngineLocalKVCacheManagerProxy(
            self.kv_cache_manager
        )
