# SPDX-License-Identifier: Apache-2.0
"""Base class for platform device specification.

Each built-in accelerator sub-package under ``platform/devices`` (for example,
``cuda`` or ``musa``) provides a concrete :class:`DeviceSpec` subclass that
describes how to detect the device and which ops backend to load. External
packages can expose the same class through the ``lmcache.device_plugins``
Python entry-point group.

The :mod:`~lmcache.v1.platform` module discovers these subclasses automatically
from :mod:`lmcache.v1.platform.devices` at import time via
``pkgutil.iter_modules``: it imports each backend sub-package, inspects its
module namespace for
:class:`DeviceSpec` subclasses, instantiates them, and uses the
resulting objects for device detection and backend selection.

No manual registration call is required. Built-in implementations are found
from their sub-package; external implementations are found from installed
package metadata.

:class:`DeviceSpec` itself is instantiable and doubles as the fallback
implementation used when no accelerator sub-package matches the
detected device: all capabilities default to a safe "no-op / False"
behaviour, and ``device_type`` / ``torch_module_name`` default to an
empty string (concrete backends -- including CPU via
:class:`~lmcache.v1.platform.devices.cpu.CpuDeviceSpec` -- override them).
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    from lmcache.v1.platform.base.device_ops import DeviceOps
    from lmcache.v1.platform.base.event_ipc import EventIPCBackend
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper


class DeviceSpec:
    """Description of a hardware accelerator backend.

    Subclasses override the properties / methods below to describe a
    concrete accelerator. Built-in subclasses are auto-discovered from the
    ``platform.devices`` package, while external subclasses are registered as Python
    entry points.

    Instantiating :class:`DeviceSpec` directly yields the fallback
    implementation with "no-op / all False" semantics -- this is the
    behaviour used for CPU-only or unknown device types.
    """

    # Cached pin-memory backend instance (lazy-initialized).
    _pin_backend_cache: PinMemoryBackend | None = None
    # Cached DeviceOps singleton instance (lazy-initialized).
    _ops_cache: DeviceOps | None = None

    @property
    def device_type(self) -> str:
        """Device type string (e.g. ``"cuda"``, ``"musa"``, ``"mlu"``).

        Concrete backends override this; the base returns an empty
        string so a bare ``DeviceSpec()`` instance is never mistaken
        for a real accelerator (CPU is represented by
        :class:`~lmcache.v1.platform.devices.cpu.CpuDeviceSpec`).
        """
        return ""

    @property
    def backend_name(self) -> str:
        """Unique LMCache backend identifier for explicit backend selection.

        ``device_type`` is tied to torch device naming, so multiple specs may
        legitimately share it (for example, two different implementations of
        ``"cuda"``). ``backend_name`` is the LMCache-specific disambiguator for
        those cases and must therefore be unique across all registered specs.

        The base implementation reuses :attr:`device_type`, which keeps today's
        built-in backends unchanged. Specialised backends that share a
        ``device_type`` with another implementation should override this with a
        distinct lowercase name.
        """
        return self.device_type

    @property
    def torch_module_name(self) -> str:
        """Attribute name on the ``torch`` package for the device module.

        For example, ``"cuda"`` corresponds to ``torch.cuda``.  The
        base returns an empty string; concrete backends (including
        :class:`~lmcache.v1.platform.devices.cpu.CpuDeviceSpec`) override it.
        """
        return ""

    @property
    def ops_cls(self) -> type[DeviceOps]:
        """DeviceOps subclass providing this platform's operation surface.

        Lazy by design: the import happens on *access*, not at class-definition
        or DeviceSpec-discovery time, so resolving a spec never drags the torch
        baseline (or a native .so) into the platform package's import graph.
        The base returns the torch/CPU baseline; accelerator specs override.

        Returns:
            type[DeviceOps]: The base DeviceOps torch/CPU baseline class for the
            fallback spec. Accelerator subclasses override this property to
            return their backend-specific DeviceOps subclass.
        """
        # First Party
        from lmcache.v1.platform.base.device_ops import DeviceOps

        return DeviceOps

    def get_ops(self) -> DeviceOps:
        """Return the cached :class:`DeviceOps` singleton for this spec.

        Lazy-initialized on first access.  Calls :meth:`ensure_native`
        so native ops are bound before the instance is used.  The same
        instance is reused process-wide for a given spec.
        """
        ops = self._ops_cache
        if ops is None:
            ops = self.ops_cls()
            ops.ensure_native()
            self._ops_cache = ops
        return ops

    def is_available(self) -> bool:
        """Return ``True`` when the device is usable on this system.

        This method must NOT import from ``lmcache.__init__`` to avoid
        circular dependencies.  Use ``import torch`` directly instead.
        The fallback implementation always returns ``False`` so that
        auto-detection never picks it up.
        """
        return False

    def is_handle_transfer_available(self) -> bool:
        """Return ``True`` when the device is usable for handle transfer."""
        # TODO(chunxiaozheng): implement on subclasses
        return True

    # ------------------------------------------------------------------
    # Stream execution
    # ------------------------------------------------------------------

    def current_stream(self, device: object) -> object:
        """Return the current stream for ``device``.

        Args:
            device: A torch device object owned by this specification.

        Returns:
            The platform's current stream object.

        Raises:
            RuntimeError: If this specification is not the active runtime
                platform.
        """
        return self._get_torch_module().current_stream(device)

    def get_stream_handle(self, stream: object) -> int:
        """Return the native handle consumed by stream-aware native libraries.

        Native stream-handle layouts are platform-specific. Accelerator
        specifications that support such libraries must override this method;
        the base class deliberately fails instead of guessing an attribute on
        an unknown stream implementation.

        Args:
            stream: A platform stream returned by :meth:`current_stream`.

        Returns:
            The native stream handle.

        Raises:
            NotImplementedError: If the platform has no native stream-handle
                adapter.
        """
        raise NotImplementedError(
            f"DeviceSpec for device_type={self.device_type!r} does not provide "
            "a native stream handle."
        )

    def synchronize_stream(self, stream: object) -> None:
        """Wait until work already enqueued on ``stream`` has completed.

        Args:
            stream: A platform stream returned by :meth:`current_stream`.
        """
        stream_object: Any = stream
        stream_object.synchronize()

    def synchronize_device(self, device: object) -> None:
        """Wait until work already enqueued on ``device`` has completed.

        Args:
            device: A torch device object owned by this specification.
        """
        self._get_torch_module().synchronize(device=device)

    def create_stream_event(self, device: object) -> object:
        """Create an event used to observe completion on ``device``'s stream.

        Args:
            device: A torch device object owned by this specification.

        Returns:
            A platform event object.
        """
        return self._get_torch_module().Event()

    def record_stream_event(self, event: object, stream: object) -> None:
        """Record ``event`` after work already queued on ``stream``.

        Args:
            event: An event returned by :meth:`create_stream_event`.
            stream: A platform stream returned by :meth:`current_stream`.
        """
        event_object: Any = event
        event_object.record(stream)

    def is_stream_event_complete(self, event: object) -> bool:
        """Return whether a previously recorded stream event has completed.

        Args:
            event: An event returned by :meth:`create_stream_event`.

        Returns:
            ``True`` when the event has completed.
        """
        event_object: Any = event
        return event_object.query()

    @property
    def event_ipc_backend(self) -> "EventIPCBackend | None":
        """Return the device-event IPC backend for this device, if supported.

        Concrete device specifications must explicitly provide an event IPC
        backend when they support cross-process event synchronization.

        Returns:
            The device's ``EventIPCBackend``, or ``None`` when unsupported.
        """
        return None

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        """PinMemoryBackend subclass for this device, or None for default.

        Subclasses that support host-memory pinning should override this
        property and return the appropriate backend class.  Use a lazy
        import inside the property body to avoid heavy imports at class
        definition time.
        """
        return None

    @property
    def ipc_wrapper_cls(self) -> type[DeviceIPCWrapper] | None:
        """:class:`DeviceIPCWrapper` subclass used to ship KV tensors across
        the multiprocess wire for this device, or ``None`` when the
        device has no IPC wrapper support.

        Subclasses that participate in the LMCache multiprocess KV
        transfer path override this property and return their concrete
        wrapper class.  Use a lazy import inside the property body to
        avoid dragging accelerator-specific modules into the platform
        base import graph.

        The wrapper class is expected to expose a ``wrap(tensor)``
        classmethod that returns a serializable
        :class:`DeviceIPCWrapper` instance ready for the wire.
        """
        return None

    def _get_pin_backend(self) -> PinMemoryBackend:
        """Return the cached pin-memory backend, instantiating on first use."""
        backend = self._pin_backend_cache
        if backend is None:
            backend_cls = self.pin_memory_backend or PinMemoryBackend
            backend = backend_cls()
            self._pin_backend_cache = backend
        return backend

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Pin a host memory region for DMA access.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.
            size: Size in bytes of the region to pin.
            flags: Platform-specific registration flags (e.g.
                ``cudaHostRegisterDefault = 0``).

        Returns:
            True if pinning succeeded, False otherwise.
        """
        return self._get_pin_backend().pin_memory(ptr, size, flags)

    def unpin_memory(self, ptr: int) -> bool:
        """Unpin a previously pinned host memory region.

        Args:
            ptr: Raw pointer (data_ptr) to the memory region.

        Returns:
            True if unpinning succeeded, False otherwise.
        """
        return self._get_pin_backend().unpin_memory(ptr)

    @property
    def is_pin_supported(self) -> bool:
        """Whether the current platform supports memory pinning."""
        return self._get_pin_backend().is_pin_supported

    # ------------------------------------------------------------------
    # Cache context factory
    # ------------------------------------------------------------------

    def create_cache_context(self, *args: Any, **kwargs: Any) -> "BaseCacheContext":
        """Instantiate the ``BaseCacheContext`` implementation for this device.

        Subclasses that ship a ``cache_context`` module (e.g. ``cuda``,
        ``cpu``) must override this hook: perform a lazy import of the
        concrete subclass and forward ``*args`` / ``**kwargs`` verbatim
        so the call site in
        :func:`lmcache.v1.platform.cache_context.create_cache_context`
        stays backend-agnostic.

        The default implementation raises :class:`NotImplementedError`
        so a missing override surfaces loudly instead of silently
        falling back to the CPU path.
        """
        raise NotImplementedError(
            "DeviceSpec for device_type=%r does not provide a "
            "BaseCacheContext implementation." % self.device_type
        )

    def _get_torch_module(self) -> Any:
        """Return this specification's active torch device module.

        Stream execution happens on the process's selected accelerator, so a
        specification for another device type must not accidentally operate on
        it through a similarly shaped torch module.
        """
        # First Party
        from lmcache.v1.platform._device_detect import get_torch_device

        torch_module, active_device_type = get_torch_device()
        if active_device_type != self.device_type:
            raise RuntimeError(
                "Cannot use stream execution for device type "
                f"{self.device_type!r} while {active_device_type!r} is active."
            )
        return torch_module
