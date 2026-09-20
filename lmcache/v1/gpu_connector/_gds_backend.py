# SPDX-License-Identifier: Apache-2.0
"""Backend base class and registry for async GPUDirect Storage wrappers."""

# Standard
from collections.abc import Iterable, MutableMapping, Sequence
from enum import Enum
from types import ModuleType
from typing import Any, ClassVar
import importlib
import pkgutil

# Third Party
import torch

BackendName = str
AUTO_BACKEND_NAME = "auto"


class TorchPlatform(Enum):
    """PyTorch GPU platform required or preferred by a GDS backend."""

    CUDA = "CUDA"
    ROCM = "ROCm"


def current_torch_platforms() -> frozenset[TorchPlatform]:
    """Return the GPU platforms reported by the active PyTorch build."""
    platforms: set[TorchPlatform] = set()
    if torch.version.cuda is not None:
        platforms.add(TorchPlatform.CUDA)
    if torch.version.hip is not None:
        platforms.add(TorchPlatform.ROCM)
    return frozenset(platforms)


def _format_platforms(platforms: Iterable[TorchPlatform]) -> str:
    by_preference = {
        TorchPlatform.ROCM: 0,
        TorchPlatform.CUDA: 1,
    }
    names = [
        platform.value
        for platform in sorted(
            platforms, key=lambda p: by_preference.get(p, len(by_preference))
        )
    ]
    if not names:
        return "any"
    if len(names) == 1:
        return names[0]
    return " or ".join(names)


class GDSAsyncBackend:
    """Base class for one async GDS backend implementation.

    Subclasses live next to their native wrapper functions. They declare their
    user-facing name, compatible PyTorch platforms, auto-selection eligibility,
    and any backend-specific capabilities by overriding methods.
    """

    _backend_classes: ClassVar[dict[str, type["GDSAsyncBackend"]]] = {}

    name: ClassVar[str] = ""
    required_platforms: ClassVar[frozenset[TorchPlatform]] = frozenset()
    auto_platforms: ClassVar[frozenset[TorchPlatform]] = frozenset()
    auto_priority: ClassVar[int] = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register concrete backend subclasses when their module is imported."""
        super().__init_subclass__(**kwargs)
        if not cls.name:
            return
        existing = GDSAsyncBackend._backend_classes.get(cls.name)
        if existing is not None and existing.__module__ != cls.__module__:
            raise ValueError(f"duplicate GDS backend registration: {cls.name}")
        GDSAsyncBackend._backend_classes[cls.name] = cls

    def __init__(self) -> None:
        if not self.name:
            raise TypeError(f"{type(self).__name__} must define a backend name")

    @property
    def module(self) -> ModuleType:
        """Return the module implementing this backend's async IO surface."""
        return importlib.import_module(type(self).__module__)

    def bind_surface(
        self,
        namespace: MutableMapping[str, Any],
        exported_names: Iterable[str],
    ) -> None:
        """Bind this backend's common async IO surface into ``namespace``."""
        module = self.module
        for name in exported_names:
            namespace[name] = getattr(module, name)

    def matches_auto(self, platforms: frozenset[TorchPlatform]) -> bool:
        """Return whether this backend should be considered for auto mode."""
        return bool(
            self.auto_platforms and not self.auto_platforms.isdisjoint(platforms)
        )

    def validate_platform(self, platforms: frozenset[TorchPlatform]) -> None:
        """Raise if the active PyTorch build cannot support this backend."""
        if self.required_platforms and self.required_platforms.isdisjoint(platforms):
            raise ValueError(
                f"{self.name} requires a "
                f"{_format_platforms(self.required_platforms)} PyTorch build"
            )

    def get_device_capacity(self, fd: int, handle: int) -> int:
        """Return finite backing-device capacity for backends that expose it."""
        raise RuntimeError(f"{self.name} does not expose a GDS device capacity query")

    @classmethod
    def registered_backend_classes(
        cls, package_name: str
    ) -> list[type["GDSAsyncBackend"]]:
        """Return registered backend classes defined under ``package_name``."""
        prefix = f"{package_name}."
        return sorted(
            (
                backend_cls
                for backend_cls in cls._backend_classes.values()
                if backend_cls.__module__.startswith(prefix)
            ),
            key=lambda backend_cls: backend_cls.name,
        )


class GDSBackendRegistry:
    """Registry that resolves configured names to async GDS backend objects."""

    def __init__(self, backends: Sequence[GDSAsyncBackend]) -> None:
        by_name: dict[str, GDSAsyncBackend] = {}
        for backend in backends:
            if backend.name in by_name:
                raise ValueError(f"duplicate GDS backend registration: {backend.name}")
            by_name[backend.name] = backend
        if not by_name:
            raise RuntimeError("no async GDS backends are registered")
        self._backends = by_name

    @classmethod
    def discover(cls, package_name: str) -> "GDSBackendRegistry":
        """Import sibling async backend modules and collect their descriptors."""
        package = importlib.import_module(package_name)
        package_path = getattr(package, "__path__", None)
        if package_path is None:
            raise RuntimeError(f"{package_name} is not a package")

        for module_info in pkgutil.iter_modules(package_path, f"{package_name}."):
            module_basename = module_info.name.rsplit(".", 1)[-1]
            if (
                not module_basename.endswith("_async")
                or module_basename == "_gds_async"
            ):
                continue
            importlib.import_module(module_info.name)
        return cls(
            [
                backend_cls()
                for backend_cls in GDSAsyncBackend.registered_backend_classes(
                    package_name
                )
            ]
        )

    def default_backend(self) -> GDSAsyncBackend:
        """Return the import-time backend used for stable module-level bindings."""
        platforms = current_torch_platforms()
        candidates = self._auto_candidates(platforms)
        if candidates:
            return candidates[0]
        return sorted(self._backends.values(), key=lambda backend: backend.name)[0]

    def select(self, name: BackendName) -> GDSAsyncBackend:
        """Resolve and validate a configured backend name."""
        platforms = current_torch_platforms()
        if name == AUTO_BACKEND_NAME:
            candidates = self._auto_candidates(platforms)
            if not candidates:
                auto_platforms = frozenset(
                    platform
                    for backend in self._backends.values()
                    for platform in backend.auto_platforms
                )
                raise ValueError(
                    f"{AUTO_BACKEND_NAME} requires a "
                    f"{_format_platforms(auto_platforms)} PyTorch build"
                )
            backend = candidates[0]
        else:
            try:
                backend = self._backends[name]
            except KeyError as exc:
                raise ValueError(f"unsupported GDS L1 backend: {name}") from exc

        backend.validate_platform(platforms)
        return backend

    def _auto_candidates(
        self, platforms: frozenset[TorchPlatform]
    ) -> list[GDSAsyncBackend]:
        return sorted(
            (
                backend
                for backend in self._backends.values()
                if backend.matches_auto(platforms)
            ),
            key=lambda backend: (-backend.auto_priority, backend.name),
        )
