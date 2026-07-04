# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for platform device information.

Each accelerator sub-package (``platform/cuda``, ``platform/musa``, ...)
provides a concrete :class:`DeviceInfo` subclass that describes how to
detect the device and which ops backend to load.

The :mod:`~lmcache.v1.platform` module discovers these
subclasses automatically at import time via ``pkgutil.iter_modules``:
it imports each sub-package, inspects its module namespace for
:class:`DeviceInfo` subclasses, instantiates them, and uses the
resulting objects for device detection and backend selection.

No manual registration (e.g. ``DEVICE_INFO = ...``) is required --
simply defining the subclass in the sub-package's ``__init__.py`` is
enough.
"""

# Standard
import abc

# First Party
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend


# TODO(chunxiaozheng): bind `DeviceIPCWrapper` with `DeviceInfo`?
class DeviceInfo(abc.ABC):
    """Abstract description of a hardware accelerator backend.

    Subclasses must override the abstract properties / methods below.
    Defining a concrete subclass in a platform sub-package's
    ``__init__.py`` is sufficient for auto-discovery.

    NOTE: the cpu is a default device, and it is not a real device.
    The cpu is used to provide a default device for the device detection.
    """

    @property
    @abc.abstractmethod
    def device_type(self) -> str:
        """Device type string (e.g. ``"cuda"``, ``"musa"``, ``"mlu"``)."""

    @property
    @abc.abstractmethod
    def torch_module_name(self) -> str:
        """Attribute name on the ``torch`` package for the device module.

        For example, ``"cuda"`` corresponds to ``torch.cuda``.
        """

    @property
    @abc.abstractmethod
    def ops_module(self) -> str | None:
        """Fully-qualified module path for the compiled ops backend.

        Return ``None`` if no custom ops are available (fallback only).
        """

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` when the device is usable on this system.

        This method must NOT import from ``lmcache.__init__`` to avoid
        circular dependencies.  Use ``import torch`` directly instead.
        """

    def is_handle_transfer_available(self) -> bool:
        """Return ``True`` when the device is usable for handle transfer."""
        # TODO(chunxiaozheng): implement on subclasses
        return True

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        """PinMemoryBackend subclass for this device, or None for default.

        Subclasses that support host-memory pinning should override this
        property and return the appropriate backend class.  Use a lazy
        import inside the property body to avoid heavy imports at class
        definition time.
        """
        return None
