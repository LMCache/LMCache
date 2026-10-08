# SPDX-License-Identifier: Apache-2.0
"""XPU IPC wrapper for the LMCache-driven MP handle path.

:class:`XpuIPCWrapper` ships KV tensors across the multiprocess wire through
PyTorch's storage IPC (``UntypedStorage._share_xpu_`` / ``_new_shared_xpu``)
rather than through a device-specific handle module.

"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar
import ctypes
import os
import sys

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.ipc_policy import is_isolated_ipc

logger = init_logger(__name__)

#: Opt-in switch.
ENV_XPU_HANDLE_TRANSFER = "LMCACHE_XPU_HANDLE_TRANSFER"

#: PID of the externally started LMCache server, or ``"any"``. Read by
#: :func:`grant_server_ptrace_access`.
ENV_MP_SERVER_PID = "LMCACHE_MP_SERVER_PID"

#: ``PR_SET_PTRACER`` from ``<linux/prctl.h>``; the value is ASCII "Yama".
_PR_SET_PTRACER = 0x59616D61

#: Set once the process has attempted the ``prctl`` call, so a registration
#: covering many layers does not repeat it (or repeat its warnings).
_ptracer_grant_attempted = False

#: ``torch.UntypedStorage`` methods the wrapper needs.
_REQUIRED_STORAGE_IPC_SYMBOLS = (
    "_share_xpu_",
    "_new_shared_xpu",
)

_TORCH_XPU_IPC_REQUIREMENT = "torch XPU storage IPC is unavailable."

#: SYCL IPC resolves a peer allocation with ``pidfd_getfd(2)``, which the
#: kernel gates on ptrace access mode ``PTRACE_MODE_ATTACH_REALCREDS``.
#: The importing process must therefore be permitted to ptrace the exporter.
PIDFD_PRECONDITION_HINT = (
    "SYCL IPC on Level Zero imports the worker's KV allocation via "
    "pidfd_getfd(2), so the LMCache server must be allowed to ptrace the "
    "worker. Run both as the same user, and under Yama ptrace_scope=1 "
    "(Ubuntu's default) have the worker call prctl(PR_SET_PTRACER, "
    "<server pid>) before registering its KV caches, or set ptrace_scope=0. "
    "Alternatively grant the server CAP_SYS_PTRACE, which ptrace_scope=2 "
    "requires; ptrace_scope=3 forbids it. Both processes must run on one "
    "Linux >= 5.6 host in the same PID namespace."
)

#: SYCL exposes no allocation-handle API that rendezvous without a process
#: relationship, so the zero-share isolated deployment model cannot be served
#: on XPU at all.
_ISOLATED_IPC_UNSUPPORTED_HINT = (
    "isolated_ipc is enabled, but XPU has no driver-level IPC that works "
    "across fully isolated containers: SYCL IPC needs pidfd_getfd(2) on "
    "the exporting process, which a zero-share deployment cannot grant. "
    "Disable isolated_ipc for XPU and keep the processes in a shared PID "
    "namespace with ptrace permission."
)


def is_xpu_handle_transfer_enabled() -> bool:
    """Return whether the operator opted into the XPU handle path.

    Returns:
        ``True`` when :data:`ENV_XPU_HANDLE_TRANSFER` is set to a truthy
        value.
    """
    return os.environ.get(ENV_XPU_HANDLE_TRANSFER, "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _warn_if_ptrace_restricted() -> None:
    """Warn when Yama will refuse the import and nothing has granted access.

    Called only when :data:`ENV_MP_SERVER_PID` is unset. Reading the sysctl
    cannot prove the import will fail -- the server may hold
    ``CAP_SYS_PTRACE`` -- so this only warns, and surfaces it in the
    worker's log where an operator looks first, rather than leaving a bare
    ``EPERM`` in the server's log.
    """
    try:
        with open("/proc/sys/kernel/yama/ptrace_scope") as handle:
            scope = handle.read().strip()
    except OSError:
        # No Yama LSM compiled in; classic ptrace permissions apply.
        return
    if scope == "0":
        return
    if scope in ("2", "3"):
        # PR_SET_PTRACER is ignored at these scopes, so do not advertise it.
        logger.warning(
            "Yama ptrace_scope=%s, which PR_SET_PTRACER cannot relax, so the "
            "LMCache server will fail to import this worker's XPU handles. "
            "Scope 2 requires the server to hold CAP_SYS_PTRACE; scope 3 "
            "forbids ptrace outright. %s",
            scope,
            PIDFD_PRECONDITION_HINT,
        )
        return
    logger.warning(
        "Yama ptrace_scope=%s and %s is unset, so the LMCache server will "
        "likely fail to import this worker's XPU handles with EPERM. Set "
        "%s=<server pid>, or =any, or run the host with ptrace_scope=0. %s",
        scope,
        ENV_MP_SERVER_PID,
        ENV_MP_SERVER_PID,
        PIDFD_PRECONDITION_HINT,
    )


def grant_server_ptrace_access() -> None:
    """Nominate the LMCache server as this worker's permitted tracer.

    SYCL IPC on Level Zero imports the exported allocation with
    ``pidfd_getfd(2)``, which the kernel gates on ptrace access to the
    exporter. Under the default Yama ``ptrace_scope=1`` the exporting
    process must name its tracer through ``prctl(PR_SET_PTRACER, ...)``.

    Reads the server's PID from :data:`ENV_MP_SERVER_PID`, which the
    operator sets when starting an external standalone server. The literal
    ``"any"`` maps to ``PR_SET_PTRACER_ANY`` for deployments where the PID
    is not knowable; that is still far narrower than lowering the host-wide
    ``ptrace_scope``. When the variable is unset this is a no-op, which is
    correct under ``ptrace_scope=0``.

    Runs at most once per process and never raises: the grant is only one
    of several ways the precondition can be met, so a failure is logged and
    import is left to fail later with :data:`PIDFD_PRECONDITION_HINT`.
    """
    global _ptracer_grant_attempted

    if _ptracer_grant_attempted or not sys.platform.startswith("linux"):
        return
    _ptracer_grant_attempted = True

    raw = os.environ.get(ENV_MP_SERVER_PID, "").strip()
    if not raw:
        _warn_if_ptrace_restricted()
        return

    if raw.lower() == "any":
        target = ctypes.c_ulong(-1).value
    else:
        try:
            target = int(raw)
        except ValueError:
            logger.warning(
                "Ignoring %s=%r: expected a decimal PID or 'any'",
                ENV_MP_SERVER_PID,
                raw,
            )
            return
        if target <= 0:
            logger.warning(
                "Ignoring %s=%r: PID must be positive", ENV_MP_SERVER_PID, raw
            )
            return

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
        libc.prctl.restype = ctypes.c_int
        rc = libc.prctl(_PR_SET_PTRACER, target, 0, 0, 0)
    except OSError as exc:
        logger.warning("prctl(PR_SET_PTRACER) is unavailable: %s", exc)
        return

    if rc != 0:
        logger.warning(
            "prctl(PR_SET_PTRACER, %s) failed: %s; XPU IPC import may be refused. %s",
            raw,
            os.strerror(ctypes.get_errno()),
            PIDFD_PRECONDITION_HINT,
        )
        return
    logger.info("Granted ptrace access to LMCache server %s for XPU IPC", raw)


def is_torch_xpu_available() -> bool:
    """Return whether this torch build reports a visible XPU device."""
    try:
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False


def check_torch_xpu_storage_ipc_support() -> bool:
    """Return whether ``UntypedStorage`` exposes the XPU IPC methods.

    Returns:
        ``True`` when both ``_share_xpu_`` and ``_new_shared_xpu`` are
        callable on :class:`torch.UntypedStorage`.
    """
    return all(
        callable(getattr(torch.UntypedStorage, symbol, None))
        for symbol in _REQUIRED_STORAGE_IPC_SYMBOLS
    )


def is_xpu_memory_ipc_available() -> bool:
    """Return whether the opt-in torch XPU memory IPC API is usable.

    Returns:
        ``True`` when the feature is enabled, an XPU device is visible,
        and torch provides the storage IPC methods.
    """
    if not is_xpu_handle_transfer_enabled() or not is_torch_xpu_available():
        return False
    return check_torch_xpu_storage_ipc_support()


def is_xpu_event_ipc_available() -> bool:
    """Return whether the opt-in torch XPU event IPC API is usable.

    Delegates the capability probe to
    :class:`~lmcache.v1.platform.devices.xpu.event_ipc.XpuEventIPCBackend`
    so the event requirement is expressed in exactly one place.

    Returns:
        ``True`` when the feature is enabled and ``torch.xpu.Event``
        supports interprocess handles.
    """
    if not is_xpu_handle_transfer_enabled() or not is_torch_xpu_available():
        return False
    # First Party
    from lmcache.v1.platform.devices.xpu.event_ipc import (
        check_torch_xpu_event_support,
    )

    return check_torch_xpu_event_support()


def is_xpu_handle_transfer_available() -> bool:
    """Return whether the XPU MP handle mode may be selected.

    Returns:
        ``True`` only when memory IPC and event IPC are both available.
        Block transfer needs no separate gate: ``XpuDeviceOps`` provides
        a torch implementation unconditionally.
    """
    return is_xpu_memory_ipc_available() and is_xpu_event_ipc_available()


class XpuIPCWrapper(DeviceIPCWrapper):
    """Wire wrapper for XPU tensors shared through torch storage IPC.

    The producer exports the tensor's untyped storage, the consumer rebuilds
    it with ``_new_shared_xpu`` and re-applies ``shape`` / ``stride`` /
    ``storage_offset``.

    The exported handle's first slot is the *producer's* device ordinal,
    which is meaningless in the consumer. As on CUDA, it is discarded and
    the consumer resolves its own ordinal from :attr:`device_uuid`, so a
    worker and a server with different device visibility still agree.
    """

    #: ``torch.device.type`` this wrapper handles. Kept as a class-level
    #: constant so external tooling / tests can introspect the binding.
    device_type: ClassVar[str] = "xpu"

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "XpuIPCWrapper":
        """Factory used by
        :func:`~lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Args:
            tensor: An XPU tensor backed by PyTorch's caching allocator.

        Returns:
            A new :class:`XpuIPCWrapper` wrapping ``tensor`` for the
            multiprocess wire.
        """
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        """Export ``tensor``'s storage through torch XPU IPC.

        Args:
            tensor: An XPU tensor to export.

        Raises:
            ValueError: If ``tensor`` is not on an XPU device.
            RuntimeError: If isolated IPC is enabled (unsupported on XPU)
                or the torch XPU storage IPC API is unavailable.
        """
        # First Party
        from lmcache.v1.gpu_connector.kv_format.contiguity import (
            attempt_permute_to_contiguous_view,
        )

        if is_isolated_ipc():
            raise RuntimeError(_ISOLATED_IPC_UNSUPPORTED_HINT)

        # Same layout normalization as the CUDA wrapper: permute any
        # non-contiguous view (e.g. vLLM's NHD-over-HND) so the
        # shape/stride encoded on the wire reflects the physical layout.
        # Offset is preserved by the storage_offset field below.
        tensor = attempt_permute_to_contiguous_view(tensor)
        if tensor.device.type != "xpu":
            raise ValueError(
                f"expected an XPU tensor for XpuIPCWrapper; got {tensor.device.type}"
            )

        if not is_xpu_memory_ipc_available():
            raise RuntimeError(
                "XPU memory IPC is not available. Set "
                f"{ENV_XPU_HANDLE_TRANSFER}=1 with a torch build that provides "
                f"XPU storage IPC ({_TORCH_XPU_IPC_REQUIREMENT}), or use MP "
                "transfer mode 'engine_driven' or 'auto'."
            )

        # Must precede the export: the server calls pidfd_getfd(2) while
        # handling the registration RPC that carries this handle, so the
        # grant has to be in place before that request is sent.
        grant_server_ptrace_access()

        storage = tensor.untyped_storage()
        self.handle = storage._share_xpu_()  # type: ignore[attr-defined]
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())

        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(
            0 if device_index is None else device_index
        )

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor in this process from the IPC handle.

        Returns:
            A tensor viewing the exporter's XPU allocation.

        Raises:
            RuntimeError: If the receiver's torch build lacks the XPU
                storage IPC API.

        Note:
            This may fail if the accelerator is not initialized in the
            importing process; ``torch_dev.init()`` should run first.
            Import also requires ptrace permission over the exporter --
            see :data:`PIDFD_PRECONDITION_HINT`.
        """
        if not is_xpu_memory_ipc_available():
            raise RuntimeError(
                "XPU memory IPC is not available in the receiver process. "
                f"{_TORCH_XPU_IPC_REQUIREMENT} Set "
                f"{ENV_XPU_HANDLE_TRANSFER}=1 in the LMCache server "
                "environment as well as the worker's."
            )

        device_index = self._get_device_index_from_uuid(self.device_uuid)

        # Slot 0 of the handle is the producer's ordinal; substitute the
        # locally resolved one and pass the remaining opaque slots through
        # verbatim, exactly as the CUDA wrapper does.
        try:
            storage = torch.UntypedStorage._new_shared_xpu(  # type: ignore[attr-defined]
                device_index, *self.handle[1:]
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"XPU IPC import failed: {exc}. Please check the ptrace "
                f"precondition: {PIDFD_PRECONDITION_HINT}"
            ) from exc

        t = torch.empty(
            (), device=f"{torch_device_type}:{device_index}", dtype=self.dtype
        )
        t.set_(storage, self.storage_offset, self.shape, self.stride)
        return t
