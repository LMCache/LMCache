# SPDX-License-Identifier: Apache-2.0
"""Discovery for the coordinator's controllers.

Controllers are found by scanning this package rather than listed at the
call site, so dropping a new one in this directory is the whole of adding
it: startup builds it, hands it out by type, and routes whatever durable
state it advertises.

A controller that ships elsewhere is named in
``MPCoordinatorConfig.extra_config`` under ``controller_packages`` and
scanned the same way.
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

PACKAGES_KEY = "controller_packages"


def build_controllers(
    config: "MPCoordinatorConfig", views: "Registry[View]"
) -> Registry[Controller]:
    """Discover and construct every controller available to the coordinator.

    Args:
        config: Passed to each ``from_config``, and read for the
            out-of-tree packages to scan alongside this one.
        views: The fleet's read models, which a controller may depend on.

    Returns:
        A registry with every discovered controller built.

    Raises:
        ValueError: If ``extra_config[PACKAGES_KEY]`` is not a list of
            importable names.
        ModuleNotFoundError: If one of those names does not import.
    """
    packages = (__name__, *_named_packages(config.extra_config))
    registry: Registry[Controller] = Registry(
        list(discover(packages, Controller)),
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


def _named_packages(extra_config: Mapping[str, object]) -> Sequence[str]:
    """Return the out-of-tree packages an operator named, validated.

    Args:
        extra_config: The coordinator's untyped settings.

    Returns:
        The names in the order given, empty when the key is absent.

    Raises:
        ValueError: If the value is not a list of strings. It arrives as
            JSON from the command line, so its shape is the operator's to
            get wrong and worth naming precisely.
    """
    named = extra_config.get(PACKAGES_KEY, ())
    if isinstance(named, str) or not isinstance(named, (list, tuple)):
        raise ValueError(
            f"extra_config[{PACKAGES_KEY!r}] must be a list of importable "
            f"package names, got {type(named).__name__}"
        )
    for name in named:
        if not isinstance(name, str):
            raise ValueError(
                f"extra_config[{PACKAGES_KEY!r}] must contain strings, "
                f"got {type(name).__name__}"
            )
    return tuple(named)


__all__ = ["PACKAGES_KEY", "Controller", "build_controllers"]
