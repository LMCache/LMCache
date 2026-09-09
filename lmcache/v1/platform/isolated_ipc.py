# SPDX-License-Identifier: Apache-2.0
"""Process-global switch for IPC across isolated containers.

``isolated_ipc`` declares that the cooperating LMCache processes (vLLM
workers and the LMCache server) may run in containers that share nothing:
no host IPC namespace (``hostIPC``) and no common ``/dev/shm`` tmpfs.
Device backends consult the switch when choosing IPC mechanisms; CUDA
selects the timeline-semaphore event backend instead of CUDA interprocess
event handles (which only resolve across a shared ``/dev/shm``).

The switch defaults to ``False`` because the SGLang, TensorRT-LLM,
CacheBlend, and qstore integrations still create raw CUDA interprocess events.
"""

_enabled: bool = False


def set_isolated_ipc(enabled: bool) -> None:
    """Set the process-global isolated-IPC switch."""
    global _enabled
    _enabled = enabled


def is_isolated_ipc() -> bool:
    """Return whether IPC mechanisms must work across isolated containers."""
    return _enabled
