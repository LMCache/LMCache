# SPDX-License-Identifier: Apache-2.0
"""Protocol and construction contract for pluggable engine modules."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeGuard, TypeVar, cast
import abc

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext

# Method set that makes up the InstanceLivenessTarget contract. A module that
# defines all four is driven by the periodic reaper.
_LIVENESS_TARGET_METHODS = (
    "touch_instance",
    "reap_stale_instances",
    "tracked_instance_count",
    "drop_instance_state",
)

DiscoverableModuleT = TypeVar("DiscoverableModuleT", bound="DiscoverableModule")


class EngineModule(Protocol):
    """Protocol for pluggable engine modules.

    Each module owns transport-neutral business state and operations.
    """

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        ...

    def report_status(self) -> dict:
        """Return module-specific status information."""
        ...

    def close(self) -> None:
        """Release resources owned by this module."""
        ...


class InstanceLivenessTarget(Protocol):
    """A module the periodic reaper drives, in either or both of two roles.

    * **Liveness owner** -- tracks per-worker registrations keyed by
      ``instance_id``, refreshed on PING and scanned for staleness
      (``touch_instance`` / ``reap_stale_instances`` /
      ``tracked_instance_count``). The transfer modules fill this role.
    * **State mirror** -- holds a second reference to a reaped instance's
      resources and releases it on demand (``drop_instance_state``).
      ``BlendModule`` fills this role for its per-instance CB state.

    Every method defaults to a no-op, so an implementer subclasses this
    protocol and overrides only the role it fills. The management module
    drives all targets from the PING handler and the reaper; no caller
    touches a module's private state directly.
    """

    def touch_instance(self, instance_id: int) -> None:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked (already reaped or never
        registered), or for a target that owns no liveness state.

        Args:
            instance_id: The worker's opaque instance ID.
        """
        return

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Evict and clean up workers that have gone silent.

        An instance that has sent at least one PING is judged against
        ``reap_timeout_s``; one that has never pinged (warming up, or dead
        before its first request) is judged against ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances;
                must be >= ``reap_timeout_s``.

        Returns:
            The instance IDs reaped during this scan; empty for a target
            that owns no liveness state.
        """
        return []

    def tracked_instance_count(self) -> int:
        """Return the number of currently tracked instances (0 if none)."""
        return 0

    def drop_instance_state(self, instance_id: int) -> None:
        """Release any state mirrored for a reaped instance.

        Called for every reaped ``instance_id``. A no-op unless the target
        keeps a second reference to that instance's resources (only mirrors
        such as ``BlendModule`` override this).

        Args:
            instance_id: The reaped worker's instance ID.
        """
        return


class ModuleBuildContext:
    """Shared configuration and constructed dependencies for module factories."""

    def __init__(
        self,
        engine_context: MPCacheServerContext,
        mp_config: MPServerConfig,
        coordinator_config: CoordinatorConfig,
        module_configs: Mapping[type[DiscoverableModule], Mapping[str, Any]]
        | None = None,
    ) -> None:
        self.engine_context = engine_context
        self.mp_config = mp_config
        self.coordinator_config = coordinator_config
        self.liveness_targets: list[InstanceLivenessTarget] = []
        self.experimental: list[str] = []
        self._module_configs = {
            module_type: dict(config)
            for module_type, config in (module_configs or {}).items()
        }
        self._built: dict[type[DiscoverableModule], EngineModule] = {}

    @property
    def built(self) -> list[EngineModule]:
        """Return modules in build order for dependent handler overrides."""
        return list(self._built.values())

    @property
    def built_types(self) -> tuple[type[DiscoverableModule], ...]:
        """Return constructed module classes in build order."""
        return tuple(self._built)

    def config_for(self, module_type: type[DiscoverableModule]) -> Mapping[str, Any]:
        """Return ``--server-module`` configuration for a module class."""
        return self._module_configs.get(module_type, {})

    def require(self, module_type: type[DiscoverableModuleT]) -> DiscoverableModuleT:
        """Return an already-built dependency with the requested module name.

        A configured external subclass may replace a built-in class while
        retaining its ``module_name``. In that case, a dependency declared
        with the built-in class resolves to the external instance.
        """
        module = self._built.get(module_type)
        if module is None:
            module = next(
                (
                    built_module
                    for built_type, built_module in self._built.items()
                    if built_type.module_name == module_type.module_name
                ),
                None,
            )
        if module is None:
            raise ValueError(
                f"Module {module_type.module_name!r} is not built yet; built so "
                f"far: {[cls.module_name for cls in self._built]}"
            )
        return cast(DiscoverableModuleT, module)

    def register(
        self,
        module_type: type[DiscoverableModule],
        module: EngineModule,
    ) -> None:
        """Record a module and collect its optional liveness contract."""
        if module_type in self._built:
            raise ValueError(f"Module {module_type.module_name!r} is already built")
        self._built[module_type] = module
        if _is_liveness_target(module):
            self.liveness_targets.append(module)

    def add_experimental(self, feature: str) -> None:
        """Advertise an experimental feature contributed by a constructed module.

        The collected names are returned to clients by the management
        module's ``get_experimental`` handler.

        Args:
            feature: Feature identifier (e.g. ``TRANSFER_QUERY``).
        """
        if feature not in self.experimental:
            self.experimental.append(feature)


class DiscoverableModule(abc.ABC):
    """Construction contract shared by built-in and external modules.

    Dependencies are module class references. ``deferred`` modules are built
    after normal modules, and ``create`` returns ``None`` when disabled.
    """

    module_name: ClassVar[str] = ""
    module_dependencies: ClassVar[tuple[type[DiscoverableModule], ...]] = ()
    deferred: ClassVar[bool] = False

    @classmethod
    @abc.abstractmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Construct this module, or return ``None`` if it does not apply."""
        raise NotImplementedError


def _is_liveness_target(module: EngineModule) -> TypeGuard[InstanceLivenessTarget]:
    """Return whether a module exposes the instance-liveness target contract."""
    return all(
        callable(getattr(module, name, None)) for name in _LIVENESS_TARGET_METHODS
    )


def order_modules(
    module_classes: list[type[DiscoverableModule]],
    known_modules: "set[type[DiscoverableModule]] | None" = None,
) -> list[type[DiscoverableModule]]:
    """Return one build phase in deterministic dependency order.

    Dependencies outside this phase must still appear in ``known_modules``;
    this supports a deferred phase whose dependencies were already built.
    A dependency class resolves through its ``module_name`` so a configured
    external subclass can take the place of a same-named built-in class.
    """
    if known_modules is None:
        known_modules = set(module_classes)

    names: dict[str, type[DiscoverableModule]] = {}
    for module_type in sorted(
        known_modules,
        key=_module_sort_key,
    ):
        name = module_type.module_name
        if not name:
            raise ValueError(f"{module_type.__name__} does not define module_name")
        previous = names.get(name)
        if previous is not None and previous is not module_type:
            raise ValueError(
                f"Duplicate module_name {name!r}: {previous.__name__} and "
                f"{module_type.__name__}"
            )
        names[name] = module_type

    module_set = set(module_classes)
    graph: dict[type[DiscoverableModule], set[type[DiscoverableModule]]] = {}
    for module_type in sorted(module_classes, key=_module_sort_key):
        dependencies: set[type[DiscoverableModule]] = set()
        for declared_dependency in module_type.module_dependencies:
            dependency = names.get(declared_dependency.module_name)
            if dependency is None:
                raise ValueError(
                    f"{module_type.__name__} depends on "
                    f"{declared_dependency.__name__}, which is not a "
                    "discoverable module."
                )
            dependencies.add(dependency)
        graph[module_type] = dependencies & module_set

    try:
        return list(TopologicalSorter(graph).static_order())
    except CycleError as exc:
        raise ValueError("Module dependency cycle") from exc


def _module_sort_key(
    module_type: type[DiscoverableModule],
) -> tuple[str, str, str]:
    return (
        module_type.module_name,
        module_type.__module__,
        module_type.__qualname__,
    )


def discover_modules() -> list[type[DiscoverableModule]]:
    """Discover concrete modules under the built-in modules package."""
    # First Party
    from lmcache.v1.utils.subclass_discovery import discover_subclasses

    return list(
        discover_subclasses(
            "lmcache.v1.multiprocess.modules",
            DiscoverableModule,  # type: ignore[type-abstract]
            module_filter=lambda name: not name.startswith("_"),
            require_defined_in_module=False,
            on_import_error=_raise_import_error,
            levels=[],
        )
    )


def _raise_import_error(module_name: str, exc: Exception) -> None:
    """Turn a module import failure into a loud startup error."""
    raise ImportError(f"Failed to import engine module {module_name}: {exc}") from exc
