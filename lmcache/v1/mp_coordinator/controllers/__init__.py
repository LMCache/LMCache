# SPDX-License-Identifier: Apache-2.0
"""Discovery for the coordinator's controllers.

Controllers are found by scanning this package rather than listed at the
call site, so dropping a new one in this directory is the whole of adding
it: startup builds it, hands it out by type, and routes whatever durable
state it advertises.
"""

# Standard
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, TypeVar, cast
import importlib
import inspect
import pkgutil

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import PersistenceType
from lmcache.v1.mp_coordinator.controllers.base import Controller
from lmcache.v1.mp_coordinator.persistence.store import DurableComponent

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig

logger = init_logger(__name__)

ControllerT = TypeVar("ControllerT", bound=Controller)


class ControllerRegistry:
    """The controllers a coordinator runs, addressed by type.

    Handing this around instead of one field per controller keeps adding
    a controller to a single change: writing it.
    """

    def __init__(self, controllers: Sequence[Controller]) -> None:
        """Index ``controllers`` by their concrete type.

        Args:
            controllers: One instance per controller class.

        Raises:
            ValueError: If a type appears twice; callers address them by
                type, so a duplicate has no unambiguous answer.
        """
        self._by_type: dict[type[Controller], Controller] = {}
        for controller in controllers:
            if type(controller) in self._by_type:
                raise ValueError(f"duplicate controller {type(controller).__name__}")
            self._by_type[type(controller)] = controller

    def get(self, controller_type: type[ControllerT]) -> ControllerT:
        """Return the controller of ``controller_type``.

        Args:
            controller_type: The class to look up.

        Returns:
            The single instance of that class.

        Raises:
            KeyError: If no such controller was discovered, which means
                the class is not a ``Controller`` under ``controllers/``.
        """
        try:
            # The index is keyed by concrete type, so the value is that type;
            # the mapping's own annotation cannot say so.
            return cast(ControllerT, self._by_type[controller_type])
        except KeyError:
            raise KeyError(
                f"no {controller_type.__name__} was discovered under "
                f"{__name__}; is it a Controller subclass in that package?"
            ) from None

    def durable_components(self) -> dict[PersistenceType, list[DurableComponent]]:
        """Group every controller's durable state by where it persists.

        Returns:
            A list per :class:`PersistenceType`, empty where nothing
            applies. Controllers holding nothing durable contribute
            nothing, so a caller need not know which those are.
        """
        collected: dict[PersistenceType, list[DurableComponent]] = {
            persistence_type: [] for persistence_type in PersistenceType
        }
        for controller in self._by_type.values():
            for component in controller.get_durable_components():
                collected[component.persistence_type].append(component)
        return collected


def build_controllers(config: "MPCoordinatorConfig") -> ControllerRegistry:
    """Construct every controller in this package.

    Each class is built by its ``from_config``, which by default ignores
    the configuration -- so only a controller that reads it writes one.

    Args:
        config: Passed to each ``from_config``.

    Returns:
        The registry, built in class-name order so a checkpoint's
        sections do not depend on filesystem order.
    """
    controllers = [
        controller_type.from_config(config) for controller_type in _controller_types()
    ]
    logger.debug(
        "Discovered %d controller(s): %s",
        len(controllers),
        ", ".join(type(controller).__name__ for controller in controllers),
    )
    return ControllerRegistry(controllers)


# -- Internals ----------------------------------------------------------------


def _controller_types() -> Iterator[type[Controller]]:
    """Yield the ``Controller`` classes defined in this package.

    Only classes a module defines itself are yielded, so one imported for
    use elsewhere is not built a second time.
    """
    for module_info in sorted(
        pkgutil.iter_modules(__path__), key=lambda info: info.name
    ):
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        for _, candidate in sorted(inspect.getmembers(module, inspect.isclass)):
            if (
                candidate.__module__ == module.__name__
                and not inspect.isabstract(candidate)
                and issubclass(candidate, Controller)
            ):
                yield candidate
