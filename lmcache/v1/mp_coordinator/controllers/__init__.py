# SPDX-License-Identifier: Apache-2.0
"""Discovery for the coordinator's controllers.

Controllers are found by scanning this package rather than listed at the
call site, so dropping a new one in this directory is the whole of adding
it: startup builds it, hands it out by type, and routes whatever durable
state it advertises.

A controller that ships elsewhere is named in
``MPCoordinatorConfig.extra_config`` under ``controller_packages`` and
scanned the same way. One that should not be built is named under
``disabled_controllers`` -- the other half of the same idea, so replacing a
built-in controller does not mean forking the tree it lives in.
"""

# Standard
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.controllers.base import Controller
from lmcache.v1.mp_coordinator.discovery import Registry, discover

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
    from lmcache.v1.mp_coordinator.views.base import View

logger = init_logger(__name__)

_PACKAGES_KEY = "controller_packages"
_DISABLED_KEY = "disabled_controllers"


def build_controllers(
    config: "MPCoordinatorConfig", views: "Registry[View]"
) -> Registry[Controller]:
    """Discover and construct every controller available to the coordinator.

    Args:
        config: Passed to each ``from_config``, and read for the
            out-of-tree packages to scan alongside this one and for any
            controller to leave unbuilt.
        views: The fleet's read models, which a controller may depend on.

    Returns:
        A registry with every discovered controller built, less those
        disabled.

    Raises:
        ValueError: If ``extra_config["controller_packages"]`` or
            ``extra_config["disabled_controllers"]`` is not a list of names,
            or if a disabled name matches nothing discovered.
        ModuleNotFoundError: If one of the package names does not import.
    """
    packages = (__name__, *_named_packages(config.extra_config, _PACKAGES_KEY))
    discovered = list(discover(packages, Controller))
    registry: Registry[Controller] = Registry(
        _remove_disabled(discovered, config.extra_config),
        # A controller cannot reach a peer, so the registry it is
        # handed for that is ignored here.
        build=lambda controller_type, _peers: controller_type.from_config(
            config, views
        ),
    )
    built = registry.all()
    logger.debug(
        "Discovered %d controller(s): %s",
        len(built),
        ", ".join(type(controller).__name__ for controller in built),
    )
    return registry


def _named_packages(extra_config: Mapping[str, object], key: str) -> Sequence[str]:
    """Return the names an operator listed under ``key``, validated.

    Args:
        extra_config: The coordinator's untyped settings.
        key: Which list to read.

    Returns:
        The names in the order given, empty when the key is absent.

    Raises:
        ValueError: If the value is not a list of strings. It arrives as
            JSON from the command line, so its shape is the operator's to
            get wrong and worth naming precisely.
    """
    named = extra_config.get(key, ())
    if isinstance(named, str) or not isinstance(named, (list, tuple)):
        raise ValueError(
            f"extra_config[{key!r}] must be a list of names, got {type(named).__name__}"
        )
    for name in named:
        if not isinstance(name, str):
            raise ValueError(
                f"extra_config[{key!r}] must contain strings, got {type(name).__name__}"
            )
    return tuple(named)


def _remove_disabled(
    discovered: Sequence[type[Controller]], extra_config: Mapping[str, object]
) -> list[type[Controller]]:
    """Drop the controllers an operator asked not to be built.

    Naming a class here is how a built-in controller gets out of the way of
    one that has taken over its job: the built-in package is always scanned,
    so without this the two would both be built, both hold state under the
    same artifact sections, and both answer for the same endpoints.

    Args:
        discovered: Every controller class found by scanning.
        extra_config: The coordinator's untyped settings.

    Returns:
        The classes to build, in discovery order.

    Raises:
        ValueError: If the setting is malformed, or names a class that was
            not discovered -- an operator believing they disabled something
            they did not is worse than a boot failure that says so.
    """
    disabled = _named_packages(extra_config, _DISABLED_KEY)
    if not disabled:
        return list(discovered)
    by_name = {controller.__name__: controller for controller in discovered}
    unknown = [name for name in disabled if name not in by_name]
    if unknown:
        raise ValueError(
            f"extra_config[{_DISABLED_KEY!r}] names {unknown}, which "
            f"{'was' if len(unknown) == 1 else 'were'} not discovered; "
            f"available: {sorted(by_name)}"
        )
    logger.info("Disabled controller(s): %s", ", ".join(disabled))
    return [c for c in discovered if c.__name__ not in set(disabled)]


__all__ = ["Controller", "build_controllers"]
