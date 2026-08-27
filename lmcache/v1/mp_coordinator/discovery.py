# SPDX-License-Identifier: Apache-2.0
"""Building the coordinator's collaborators by scanning a package.

Views and controllers are found the same way and held the same way; only
what they may depend on differs. Sharing the machinery keeps the two
from drifting into being wired differently, which is how one of them
ends up forgotten.
"""

# Standard
from collections.abc import Callable, Iterator, Sequence
from typing import Generic, TypeVar, cast
import importlib
import inspect
import pkgutil

# First Party
from lmcache.v1.mp_coordinator.persistence.durable_component import (
    Durability,
    DurableComponent,
    PersistenceType,
)

MemberT = TypeVar("MemberT")
LookupT = TypeVar("LookupT")


class Registry(Generic[MemberT]):
    """Collaborators of one kind, addressed by type.

    Instances are built on first request, so one can ask for another
    while itself being built and neither has to be constructed first.
    """

    def __init__(
        self,
        member_types: Sequence[type[MemberT]],
        build: Callable[[type[MemberT], "Registry[MemberT]"], MemberT],
    ) -> None:
        """Index ``member_types`` without building any of them yet.

        Args:
            member_types: The classes to make available.
            build: Constructs one class, given this registry so it can
                resolve peers.

        Raises:
            ValueError: If a class appears twice; callers address them by
                type, so a duplicate has no unambiguous answer.
        """
        self._types: set[type[MemberT]] = set()
        for member_type in member_types:
            if member_type in self._types:
                raise ValueError(f"duplicate {member_type.__name__}")
            self._types.add(member_type)
        self._build = build
        self._built: dict[type[MemberT], MemberT] = {}
        self._building: set[type[MemberT]] = set()

    def get(self, member_type: type[LookupT]) -> LookupT:
        """Return the instance of ``member_type``, building it once.

        Args:
            member_type: The class to look up.

        Returns:
            The single instance of that class.

        Raises:
            KeyError: If the class was not discovered.
            ValueError: If the classes depend on each other in a cycle,
                which would otherwise recurse until the stack ran out.
        """
        wanted = cast("type[MemberT]", member_type)
        built = self._built.get(wanted)
        if built is not None:
            # The index is keyed by concrete type, so the value is that
            # type; the mapping's own annotation cannot say so.
            return cast(LookupT, built)
        if wanted not in self._types:
            raise KeyError(
                f"{member_type.__name__} was not discovered; is it defined "
                f"in the package that is scanned for it?"
            )
        if wanted in self._building:
            raise ValueError(
                f"{member_type.__name__} depends on itself, directly or "
                f"through {', '.join(t.__name__ for t in self._building)}"
            )
        self._building.add(wanted)
        try:
            instance = self._build(wanted, self)
        finally:
            self._building.discard(wanted)
        self._built[wanted] = instance
        return cast(LookupT, instance)

    def all(self) -> list[MemberT]:
        """Return every member, building any not built yet.

        Returns:
            One instance per discovered class, ordered by class name so a
            checkpoint's sections do not depend on filesystem order.
        """
        return [
            self.get(cast("type[MemberT]", t))
            for t in sorted(self._types, key=lambda t: t.__name__)
        ]

    def durable_components(self) -> dict[PersistenceType, list[DurableComponent]]:
        """Group every member's durable state by where it persists.

        Returns:
            A list per :class:`PersistenceType`, empty where nothing
            applies. A member holding nothing durable contributes
            nothing, so a caller need not know which those are.
        """
        collected: dict[PersistenceType, list[DurableComponent]] = {
            persistence_type: [] for persistence_type in PersistenceType
        }
        for member in self.all():
            # Holding durable state is a property of the class, not of the
            # package, so the protocol decides rather than the base.
            if not isinstance(member, Durability):
                continue
            for component in member.get_durable_components():
                collected[component.persistence_type].append(component)
        return collected


def discover(
    package_path: Sequence[str], package_name: str, base: type[MemberT]
) -> Iterator[type[MemberT]]:
    """Yield the ``base`` subclasses each module of a package defines.

    Only classes a module defines itself are yielded, so one imported for
    use elsewhere is not built a second time, and ``base`` is skipped --
    it is the definition of a member, not one.

    Args:
        package_path: The package's ``__path__``.
        package_name: The package's ``__name__``.
        base: The marker every member subclasses.

    Yields:
        One class per member found, in module then class-name order.
    """
    for module_info in sorted(
        pkgutil.iter_modules(package_path), key=lambda info: info.name
    ):
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        for _, candidate in sorted(inspect.getmembers(module, inspect.isclass)):
            if (
                candidate.__module__ == module.__name__
                and candidate is not base
                and not inspect.isabstract(candidate)
                and issubclass(candidate, base)
            ):
                yield candidate
