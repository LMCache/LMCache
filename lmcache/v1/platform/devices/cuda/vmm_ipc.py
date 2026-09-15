# SPDX-License-Identifier: Apache-2.0
"""Process-global switch for sharing VMM-allocated KV caches.

``use_vmm_api`` declares that the inference engine allocates its KV
cache through the CUDA VMM API (``cuMemCreate``/``cuMemMap``) -- vLLM's
cumem allocator (``--enable-cumem-allocator``) -- so KV registration
must use VMM IPC (``cuMemExportToShareableHandle``) instead of legacy
CUDA IPC memory handles, which fail on VMM memory.

Orthogonal to and mutually exclusive with ``isolated_ipc``: the VMM
POSIX-fd transport needs a shared filesystem path between the processes
for fd passing (``SCM_RIGHTS`` over an ``AF_UNIX`` socket), which the
zero-share isolated-IPC deployment model rules out.

The switch defaults to ``False``: legacy CUDA IPC covers the default
torch-caching-allocator KV pools, and the fd delivery transport is a
deployment prerequisite the operator must opt into.
"""

_enabled: bool = False


def set_use_vmm_api(enabled: bool) -> None:
    """Set the process-global VMM-API switch."""
    global _enabled
    _enabled = enabled


def is_use_vmm_api() -> bool:
    """Return whether KV registration must use VMM IPC."""
    return _enabled
