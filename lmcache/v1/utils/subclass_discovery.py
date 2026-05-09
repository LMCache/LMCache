# SPDX-License-Identifier: Apache-2.0
"""Generic helper for plugin-style auto-discovery of concrete subclasses.

This module centralises the boilerplate that several packages used to
duplicate (CLI commands, controller-benchmark handlers, lookup-client
record strategies, health-monitor checks, remote connector adapters,
...).  Each of those packages walks its own submodules with
``pkgutil.iter_modules``, imports them, then iterates classes via
``inspect.getmembers`` to locate concrete subclasses of a given base
class.  ``discover_subclasses`` captures that pattern in a single
well-tested place so callers stay tiny and behave consistently.
"""

# Standard
from types import ModuleType
from typing import Callable, Iterator, Optional, TypeVar, Union
import importlib
import inspect
import pkgutil

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

T = TypeVar("T")


def _resolve_package(package: Union[ModuleType, str]) -> ModuleType:
    if isinstance(package, str):
        return importlib.import_module(package)
    return package


def discover_subclasses(
    package: Union[ModuleType, str],
    base_class: type[T],
    *,
    module_filter: Optional[Callable[[str], bool]] = None,
    include_abstract: bool = False,
    require_defined_in_module: bool = True,
    on_import_error: Optional[Callable[[str, Exception], None]] = None,
) -> Iterator[type[T]]:
    """Yield concrete subclasses of *base_class* found in direct
    submodules of *package*.

    Each subclass is yielded **at most once**, even when re-exported
    from several modules.

    Args:
        package: The package to scan, either as a module object or its
            fully-qualified dotted name.
        base_class: The base class whose concrete subclasses to collect.
        module_filter: Optional predicate over the *short* module name
            (i.e. without the package prefix).  Modules for which the
            predicate returns ``False`` are skipped.  Defaults to
            ``None`` which keeps every module.
        include_abstract: When ``False`` (default) classes with
            unimplemented abstract methods are skipped.
        require_defined_in_module: When ``True`` (default) classes that
            were merely imported (re-exported) into a module are
            ignored; only classes whose ``__module__`` matches the
            module being scanned are yielded.  Set to ``False`` to keep
            the historical behaviour of accepting re-exported classes.
        on_import_error: Optional callback invoked as
            ``on_import_error(full_module_name, exc)`` when a submodule
            fails to import.  When omitted the error is logged at
            ``warning`` level and discovery proceeds with the next
            module.
    """
    pkg = _resolve_package(package)
    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is None:
        raise TypeError(
            "discover_subclasses requires a package (with __path__), got %r" % (pkg,)
        )

    seen: set[type] = set()
    for _, short_name, _ in pkgutil.iter_modules(pkg_path):
        if module_filter is not None and not module_filter(short_name):
            continue
        full_name = "%s.%s" % (pkg.__name__, short_name)
        try:
            module = importlib.import_module(full_name)
        except Exception as exc:
            if on_import_error is not None:
                on_import_error(full_name, exc)
            else:
                logger.warning(
                    "Failed to import module %s during subclass discovery: %s",
                    full_name,
                    exc,
                )
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, base_class) or obj is base_class:
                continue
            if not include_abstract and inspect.isabstract(obj):
                continue
            if require_defined_in_module and obj.__module__ != module.__name__:
                continue
            if obj in seen:
                continue
            seen.add(obj)
            yield obj
