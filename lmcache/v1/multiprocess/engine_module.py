# SPDX-License-Identifier: Apache-2.0
"""Protocol and construction contract for pluggable engine modules."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeGuard, TypeVar, cast
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

    A build context is the single seam between the module creator and the
    modules it constructs. Modules never import each other to learn about
    their siblings; they read what they need from here.

    Attributes:
        engine_context: The shared engine context.
        mp_config: Parsed multiprocess server configuration.
        coordinator_config: Parsed coordinator configuration.
        liveness_targets: Instance-liveness targets collected from the modules
            built so far, in build order.
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

    @property
    def built(self) -> list[EngineModule]:
        """Return the modules built so far, in close order."""
        return self.built_in_close_order()

    @property
    def module_names(self) -> list[str]:
        """Return the names of the modules built so far."""
        return sorted(self._built)

    def built_in_close_order(self) -> list[EngineModule]:
        """Return the modules built so far, sorted by close-order rank."""
        return [module for _, module in sorted(self._built.values(), key=_by_rank)]

    def require(
        self, module_name: str, module_type: type[EngineModuleT]
    ) -> EngineModuleT:
        """Return an already-built module by name, checked against its type.

        Args:
            module_name: The ``module_name`` of a sibling built earlier.
            module_type: The class the caller expects, used to verify that
                ``module_name`` still refers to the same module.

        Returns:
            The sibling module instance.

        Raises:
            ValueError: If no module with that name has been built yet.
                Modules declare their ordering precisely so this cannot
                happen for a well-formed configuration.
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
        """Record a constructed module and collect it if it is a liveness target.

        The module creator calls this after each successful ``create()``, so
        modules do not have to announce themselves.

        Args:
            module: The freshly constructed module.

        Raises:
            ValueError: If the module does not declare a ``module_name``.
        """
        name = getattr(type(module), "module_name", "")
        if not name:
            raise ValueError(
                f"{type(module).__name__} does not define module_name; every "
                "DiscoverableModule must."
            )
        self._built[name] = (getattr(type(module), "module_order", 0), module)
        if _is_liveness_target(module):
            self.liveness_targets.append(module)

    def add_liveness_target(self, module: EngineModule) -> None:
        """Collect an out-of-tree module as a liveness target.

        Out-of-tree modules are not registered (they have no close-order rank
        in this table) but they can still satisfy the liveness contract, so
        they are folded into the target list directly.

        Args:
            module: An out-of-tree module that may be a liveness target.
        """
        if _is_liveness_target(module):
            self.liveness_targets.append(module)


def _by_rank(entry: tuple[int, EngineModule]) -> int:
    """Sort key extracting the close-order rank."""
    return entry[0]


class DiscoverableModule(abc.ABC):
    """Base class for engine modules the server discovers by scanning.

    Subclassing this is what makes a module discoverable:
    :func:`discover_modules` walks ``lmcache.v1.multiprocess.modules`` and
    collects every concrete subclass, so adding a module means adding a file
    with a subclass -- no registry list to edit.

    A subclass declares two class attributes and implements
    :meth:`create`:

    * ``module_name`` -- stable key used by :meth:`ModuleBuildContext.require`.
    * ``module_order`` -- close-order rank; lower closes earlier.

    ``create()`` returns ``None`` to signal that the module does not apply to
    the current configuration (for example a transfer module under a mode that
    excludes it). That keeps enablement logic inside the module that owns it
    instead of in the creator.

    A module whose construction needs the *out-of-tree* plugin modules (today
    only :class:`ManagementModule`, which consumes their liveness targets) sets
    ``deferred = True`` so the creator builds it after plugins are loaded.
    That is separate from ``module_order``, which only governs closing.

    Example:
        class MyModule(DiscoverableModule):
            module_name = "my_module"
            module_order = 45

            @classmethod
            def create(cls, build_ctx: ModuleBuildContext) -> MyModule | None:
                if not build_ctx.mp_config.enable_my_module:
                    return None
                return cls(build_ctx.engine_context)
    """

    module_name: ClassVar[str] = ""
    module_order: ClassVar[int] = 0
    deferred: ClassVar[bool] = False

    @classmethod
    @abc.abstractmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Construct this module for the given configuration.

        Args:
            build_ctx: Standard construction inputs, including any sibling
                modules already built.

        Returns:
            The constructed module, or ``None`` when the module does not
            apply to this configuration.

        Raises:
            ValueError: If the configuration requests this module but the
                request cannot be satisfied.
        """
        raise NotImplementedError


def _is_liveness_target(module: EngineModule) -> TypeGuard[InstanceLivenessTarget]:
    """Return whether a module exposes the instance-liveness target contract."""
    return all(
        callable(getattr(module, name, None)) for name in _LIVENESS_TARGET_METHODS
    )


def discover_modules() -> list[type[DiscoverableModule]]:
    """Discover every built-in engine module by scanning the modules package.

    Walks ``lmcache.v1.multiprocess.modules`` (including sub-packages, so
    ``blend`` and ``experimental`` are covered) and returns the concrete
    :class:`DiscoverableModule` subclasses, ordered by ``module_order``.

    Module imports are eager, which costs nothing here: the modules package
    is already imported by the server entry point before this runs.

    Returns:
        Concrete module classes sorted by close order.

    Raises:
        ImportError: If a module file exists but fails to import. Surfacing it
            is deliberate -- silently dropping a module would start a server
            with a missing capability.
    """
    # First Party
    from lmcache.v1.utils.subclass_discovery import discover_subclasses

    modules = list(
        discover_subclasses(
            "lmcache.v1.multiprocess.modules",
            DiscoverableModule,  # type: ignore[type-abstract]
            module_filter=lambda name: not name.startswith("_"),
            require_defined_in_module=False,
            on_import_error=_raise_import_error,
            levels=[],
        )
    )
    return sorted(modules, key=lambda cls: cls.module_order)


def _raise_import_error(module_name: str, exc: Exception) -> None:
    """Re-raise a module import failure during discovery.

    Args:
        module_name: The dotted path that failed to import.
        exc: The original import error.

    Raises:
        ImportError: Always, chaining the original error.
    """
    raise ImportError(f"Failed to import engine module {module_name}: {exc}") from exc
