# SPDX-License-Identifier: Apache-2.0
"""Policy-driven extension build orchestrator.

:class:`BuildPolicy` auto-discovers available backend strategies,
selects the best one via explicit env var or auto-detection with fallback,
and drives the common C++ + backend extension build pipeline.
"""

# Standard
from types import ModuleType
from typing import Callable, Iterator, Optional, TypeVar, Union
import importlib
import inspect
import pkgutil
import sys

# First Party
from setup_extensions.common_cpp import build_common_cpp
from setup_extensions.strategies import BuildStrategy

# ---------------------------------------------------------------------------
# Generic subclass discovery (filesystem-based, from _discovery.py)
# ---------------------------------------------------------------------------

T = TypeVar("T")


def discover_subclasses(
    package: Union[ModuleType, str],
    base_class: type[T],
    *,
    on_import_error: Optional[Callable[[str, Exception], None]] = None,
) -> Iterator[type[T]]:
    """Yield concrete subclasses of *base_class* found in direct
    submodules of *package*.

    Args:
        package: The package to scan (module or dotted name).
        base_class: The base class whose concrete subclasses to collect.
        on_import_error: Optional callback ``(full_module_name, exc)``
            invoked on import failures.  When omitted the error is
            printed to stderr and discovery continues.
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    pkg_path = getattr(package, "__path__", None)
    if pkg_path is None:
        raise TypeError(
            "discover_subclasses requires a package (with __path__), "
            "got %r" % (package,)
        )

    seen: set[type] = set()
    for _, short_name, _ in pkgutil.iter_modules(pkg_path):
        full_name = "%s.%s" % (package.__name__, short_name)
        try:
            module = importlib.import_module(full_name)
        except Exception as exc:
            if on_import_error is not None:
                on_import_error(full_name, exc)
            else:
                print(
                    "warning: failed to import %s: %s" % (full_name, exc),
                    file=sys.stderr,
                )
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, base_class) or obj is base_class:
                continue
            if inspect.isabstract(obj):
                continue
            if obj in seen:
                continue
            seen.add(obj)
            yield obj


# ---------------------------------------------------------------------------
# Strategy auto-discovery (filesystem-based, no hard-coded module list)
# ---------------------------------------------------------------------------


def _discover_strategies() -> list[BuildStrategy]:
    """Auto-discover all backend strategies.

    Walks ``setup_extensions.strategies`` via ``discover_subclasses``,
    which uses ``pkgutil.iter_modules`` to find submodules at the
    filesystem level.  Adding a new ``.py`` file under that package
    with a concrete ``BuildStrategy`` subclass is all that is needed —
    no module list to maintain.
    """
    strategies: list[BuildStrategy] = []
    for cls in discover_subclasses(
        "setup_extensions.strategies",
        BuildStrategy,  # type: ignore[type-abstract]
    ):
        strategies.append(cls())
    return strategies


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------


class BuildPolicy:
    """Selects and builds extensions using platform-aware backend strategies.

    Resolution order:
        1. If an explicit ``BUILD_WITH_*`` env var is set, use that backend
           unconditionally (no fallback). A warning is emitted when its
           toolchain cannot be auto-detected, but the build proceeds so
           that the underlying compiler produces the authoritative error.
        2. Otherwise auto-detect with fallback through candidates.
        3. If nothing is available, warn and continue without extensions.
    """

    def __init__(self) -> None:
        self._strategies = _discover_strategies()

    def resolve_strategy(self) -> Optional[BuildStrategy]:
        """Resolve the active backend strategy.

        Returns ``None`` when building sdist, native extensions are
        disabled, GPU extensions are disabled, or no backend was
        detected.
        """
        if (
            BuildStrategy.is_building_sdist()
            or BuildStrategy.is_native_ext_disabled()
            or BuildStrategy.is_gpu_ext_disabled()
        ):
            return None

        # ---------------------------------------------------------------
        # Phase 1: explicit env var selection
        # ---------------------------------------------------------------
        explicitly_requested = [
            s for s in self._strategies if s.is_explicitly_requested()
        ]
        if len(explicitly_requested) > 1:
            names = ", ".join(s.name for s in explicitly_requested)
            raise RuntimeError("Multiple backends explicitly requested: %s" % names)
        if explicitly_requested:
            strategy = explicitly_requested[0]
            print("Using explicitly requested backend: %s" % strategy.name)
            if not strategy.detect():
                print(
                    "warning: backend '%s' was explicitly requested but its "
                    "toolchain was not auto-detected; proceeding anyway"
                    % strategy.name,
                    file=sys.stderr,
                )
            return strategy

        # ---------------------------------------------------------------
        # Phase 2: auto-detect with fallback
        # ---------------------------------------------------------------
        print("No backend explicitly selected, auto-detecting...")
        for strategy in self._strategies:
            if strategy.detect():
                print("Auto-detected backend: %s" % strategy.name)
                return strategy

        # ---------------------------------------------------------------
        # Phase 3: nothing found
        # ---------------------------------------------------------------
        print(
            "warning: no backend detected, building without extensions",
            file=sys.stderr,
        )
        return None

    @staticmethod
    def collect_extensions(
        strategy: Optional[BuildStrategy],
    ) -> tuple[list, dict, Optional[str]]:
        """Build all extensions and return requirements file name.

        Args:
            strategy: Resolved backend strategy, or ``None``.

        Returns:
            ``(ext_modules, cmdclass, requirements_file)`` tuple.
        """
        if BuildStrategy.is_building_sdist():
            print("Not building extensions for sdist")
            return [], {}, None

        if BuildStrategy.is_native_ext_disabled():
            return [], {}, None

        # ---- common C++ flags from strategy (or defaults) ----
        cpp_flags = (
            strategy.common_cpp_flags() if strategy else ["-D_GLIBCXX_USE_CXX11_ABI=1"]
        )
        fs_flags = strategy.fs_cpp_flags() if strategy else cpp_flags

        # ---- build common C++ extensions ----
        ext_modules, cmdclass = build_common_cpp(cpp_flags, fs_flags)

        # ---- build backend-specific extensions ----
        if strategy and not BuildStrategy.is_gpu_ext_disabled():
            em, cc = strategy.build()
            ext_modules.extend(em)
            cmdclass.update(cc)

        # ---- requirements ----
        req_file = strategy.requirements_file() if strategy else None

        return ext_modules, cmdclass, req_file
