# SPDX-License-Identifier: Apache-2.0
"""Current process-global IPC policy for platform backends."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass


@dataclass
class IPCPolicy:
    """Runtime IPC switches used by platform backends."""

    isolated_ipc: bool = False
    use_vmm_api: bool = False

    def set(
        self,
        *,
        isolated_ipc: bool | None = None,
        use_vmm_api: bool | None = None,
    ) -> IPCPolicy:
        """Update this policy in place and return it."""
        if isolated_ipc is not None:
            self.isolated_ipc = bool(isolated_ipc)
        if use_vmm_api is not None:
            self.use_vmm_api = bool(use_vmm_api)
        return self

    def with_isolated_ipc(self, enabled: bool = True) -> IPCPolicy:
        """Set isolated-IPC mode and return this policy."""
        return self.set(isolated_ipc=enabled)

    def with_use_vmm_api(self, enabled: bool = True) -> IPCPolicy:
        """Set VMM-API mode and return this policy."""
        return self.set(use_vmm_api=enabled)


current_ipc_policy: IPCPolicy = IPCPolicy()


def set_ipc_policy(
    *,
    isolated_ipc: bool | None = None,
    use_vmm_api: bool | None = None,
) -> IPCPolicy:
    """Update the process-global IPC policy."""
    return current_ipc_policy.set(
        isolated_ipc=isolated_ipc,
        use_vmm_api=use_vmm_api,
    )


def get_ipc_policy() -> IPCPolicy:
    """Return the current process-global IPC policy."""
    return current_ipc_policy


def set_isolated_ipc(enabled: bool) -> IPCPolicy:
    """Set whether IPC must work across isolated containers."""
    return set_ipc_policy(isolated_ipc=enabled)


def is_isolated_ipc() -> bool:
    """Return whether IPC must work across isolated containers."""
    return get_ipc_policy().isolated_ipc


def set_use_vmm_api(enabled: bool) -> IPCPolicy:
    """Set whether KV registration must use VMM IPC."""
    return set_ipc_policy(use_vmm_api=enabled)


def is_use_vmm_api() -> bool:
    """Return whether KV registration must use VMM IPC."""
    return get_ipc_policy().use_vmm_api
