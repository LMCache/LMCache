# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import Iterable, List
import importlib

# Third Party
from fastapi import APIRouter

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def discover_api_routers(
    package_name: str,
    suffix: str = "_api",
    exclude: Iterable[str] = (),
) -> List[APIRouter]:
    """Discover every ``router`` attribute that is an
    :class:`~fastapi.APIRouter` among the submodules of *package_name*.

    The scan location is taken from the *imported* package's
    ``__path__`` rather than reconstructed from a ``__file__`` path.
    Reconstructing a path (e.g. ``Path(other_pkg.__file__).parent /
    "common"``) is unreliable under editable installs (PEP 660), where
    a package's ``__file__`` can resolve to a different location than
    the one the import system uses for its submodules.

    Args:
        package_name: Fully-qualified package whose submodules are
            scanned (e.g. ``"lmcache.v1.multiprocess.http_apis"``).
        suffix: Only modules whose name ends with this string are
            considered.  Defaults to ``"_api"``.
        exclude: Iterable of module base names to skip (e.g.
            ``{"run_script_api"}``).  Useful when a host package only
            wants a subset of the available routers.

    Returns:
        A list of discovered :class:`~fastapi.APIRouter` instances,
        ordered by module name.

    Raises:
        ModuleNotFoundError: If *package_name* cannot be imported.
    """
    excluded = set(exclude)
    routers: List[APIRouter] = []
    # Ensure modules created/installed after interpreter start are visible
    # to the import machinery's cached file finders.
    importlib.invalidate_caches()
    package = importlib.import_module(package_name)
    logger.debug(
        "Scanning %s for API routers in %s", package_name, list(package.__path__)
    )
    seen: set[str] = set()
    for location in package.__path__:
        for entry in sorted(Path(location).iterdir()):
            module_name = entry.stem
            if entry.suffix != ".py" or module_name == "__init__":
                continue
            if module_name in seen:
                continue
            seen.add(module_name)
            if not module_name.endswith(suffix):
                continue
            if module_name in excluded:
                logger.info("Skipping excluded API module: %s", module_name)
                continue
            module = importlib.import_module(f"{package_name}.{module_name}")
            if hasattr(module, "router") and isinstance(module.router, APIRouter):
                routers.append(module.router)
                logger.info(
                    "Discovered API module: %s (%d routes)",
                    module_name,
                    len(module.router.routes),
                )
    return routers
