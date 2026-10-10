# SPDX-License-Identifier: Apache-2.0
"""Protocol and construction contract for pluggable engine modules."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeGuard, TypeVar, cast
import abc
import heapq

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

EngineModuleT = TypeVar("EngineModuleT", bound="EngineModule")


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
    """Standard inputs handed to every :meth:`DiscoverableModule.create`.

    Modules never import each other for wiring; a module reads its siblings
    off this context instead. The creator records each module as it is built.

    Attributes:
        engine_context: The shared engine context.
        mp_config: Parsed multiprocess server configuration.
        coordinator_config: Parsed coordinator configuration.
        liveness_targets: Liveness targets collected so far, in build order.
    """

    def __init__(
        self,
        engine_context: MPCacheServerContext,
        mp_config: MPServerConfig,
        coordinator_config: CoordinatorConfig,
    ) -> None:
        self.engine_context = engine_context
        self.mp_config = mp_config
        self.coordinator_config = coordinator_config
        self.liveness_targets: list[InstanceLivenessTarget] = []
        self._built: dict[str, tuple[int, EngineModule]] = {}
        self._build_index = 0

    @property
    def built(self) -> list[EngineModule]:
        """Return the modules built so far, in close order.

        Close order is the reverse of build order, so a module is torn down
        before the dependencies it holds references to. ``ManagementModule`` is
        built last (deferred, after the plugins) and therefore closes first,
        which stops its reaper before the transfer modules release state.
        """
        return [
            module
            for _, module in sorted(
                self._built.values(), key=lambda entry: entry[0], reverse=True
            )
        ]

    @property
    def module_names(self) -> list[str]:
        """Return the names of the modules built so far."""
        return sorted(self._built)

    def require(
        self, module_name: str, module_type: type[EngineModuleT]
    ) -> EngineModuleT:
        """Return an already-built sibling module, checked against its type.

        Raises:
            ValueError: If no module with that name is built yet, or the
                name maps to a different class than ``module_type``.
        """
        entry = self._built.get(module_name)
        if entry is None:
            raise ValueError(
                "Module %r is not built yet; built so far: %r"
                % (module_name, sorted(self._built))
            )
        module = cast(EngineModuleT, entry[1])
        if not isinstance(module, module_type):
            raise ValueError(
                "Module %r is a %s, not a %s"
                % (module_name, type(module).__name__, module_type.__name__)
            )
        return module

    def register(self, module: EngineModule) -> None:
        """Record a constructed module, collecting it if it is a liveness target.

        The creator calls this after each successful ``create()``.

        Raises:
            ValueError: If the module declares no ``module_name``.
        """
        name = getattr(type(module), "module_name", "")
        if not name:
            raise ValueError(
                f"{type(module).__name__} does not define module_name; every "
                "DiscoverableModule must."
            )
        self._built[name] = (self._build_index, module)
        self._build_index += 1
        if _is_liveness_target(module):
            self.liveness_targets.append(module)

    def add_liveness_target(self, module: EngineModule) -> None:
        """Collect an out-of-tree module that satisfies the liveness contract."""
        if _is_liveness_target(module):
            self.liveness_targets.append(module)


class DiscoverableModule(abc.ABC):
    """Base class for engine modules the server discovers by scanning.

    :func:`discover_modules` walks ``lmcache.v1.multiprocess.modules`` and
    collects every concrete subclass, so adding a module means adding a file
    with a subclass -- no registry list to edit.

    A subclass declares:

    * ``module_name`` -- the key :meth:`ModuleBuildContext.require` resolves
      sibling modules by, and the key the builder records it under.
    * ``module_dependencies`` -- the ``module_name`` values of sibling modules
      it must be built *after*. The builder topologically sorts the discovered
      modules so every dependency is constructed first; a module that calls
      :meth:`ModuleBuildContext.require` must name that sibling here, otherwise
      ``require`` would raise because the sibling is not built yet.
    * ``deferred`` -- build this module *after* the out-of-tree
      ``--server-module`` plugins rather than before them. A module that
      consumes plugin contributions at construction time sets this;
      ``ManagementModule`` reads the liveness targets those plugins register.

    Every subclass implements :meth:`create`, which returns ``None`` when the
    module does not apply to the current configuration. That keeps enablement
    logic inside the module that owns it.
    """

    module_name: ClassVar[str] = ""
    module_dependencies: ClassVar[list[str]] = []
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
    known_names: "set[str] | None" = None,
) -> list[type[DiscoverableModule]]:
    """Topologically sort modules so each is built after its dependencies.

    A module lists the ``module_name`` values it requires in
    ``module_dependencies``; those are constructed first. Dependencies that
    are not part of ``module_classes`` are treated as already satisfied and
    skipped -- this is how a deferred module can depend on a sibling that is
    built in an earlier phase.

    The result is deterministic: among modules that are equally free to
    build, they are ordered by ``module_name``.

    Args:
        module_classes: The modules to order (one build phase).
        known_names: The set of every discoverable ``module_name``; a
            dependency that is not in this set is a typo and is rejected.
            Defaults to the names in ``module_classes``.

    Raises:
        ValueError: If a module names a dependency that is not discoverable,
            or if the dependencies form a cycle.
    """
    if known_names is None:
        known_names = {cls.module_name for cls in module_classes}

    for cls in module_classes:
        for dep in cls.module_dependencies:
            if dep not in known_names:
                raise ValueError(
                    f"{cls.__name__} depends on {dep!r}, which is not a "
                    "discoverable module (check the module_name spelling)."
                )

    by_name = {cls.module_name: cls for cls in module_classes}
    in_degree: dict[str, int] = {name: 0 for name in by_name}
    dependents: dict[str, list[type[DiscoverableModule]]] = {
        name: [] for name in by_name
    }
    for cls in module_classes:
        for dep in cls.module_dependencies:
            if dep in by_name:  # an in-phase dependency to honor
                in_degree[cls.module_name] += 1
                dependents[dep].append(cls)

    ready: list[str] = [name for name, deg in in_degree.items() if deg == 0]
    heapq.heapify(ready)
    ordered: list[type[DiscoverableModule]] = []
    while ready:
        name = heapq.heappop(ready)
        ordered.append(by_name[name])
        for dependent in dependents[name]:
            in_degree[dependent.module_name] -= 1
            if in_degree[dependent.module_name] == 0:
                heapq.heappush(ready, dependent.module_name)

    if len(ordered) != len(module_classes):
        unresolved = sorted(
            cls.__name__
            for cls in module_classes
            if cls.module_name not in {o.module_name for o in ordered}
        )
        raise ValueError(f"Module dependency cycle among: {', '.join(unresolved)}")
    return ordered


def discover_modules() -> list[type[DiscoverableModule]]:
    """Scan the built-in modules package and return the module classes.

    Walks ``lmcache.v1.multiprocess.modules`` -- including the ``blend`` and
    ``experimental`` sub-packages -- and returns the concrete
    :class:`DiscoverableModule` subclasses. Ordering for construction is
    applied later by the builder via :func:`order_modules`; this function
    only discovers.

    Raises:
        ImportError: If a module file fails to import. Surfacing it is
            deliberate: silently dropping a module would start the server
            with a missing capability.
    """
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
