# SPDX-License-Identifier: Apache-2.0
"""Synthetic, definition-decoupled mechanism tests for the platform registry.

This file holds the purely synthetic tests that exercise the registry's
discovery and indexing *mechanism* without referencing any real production
concrete or base class (beyond ``PlatformBase`` itself). It imports no
``torch``, no ``lmcache.c_ops``, and no device sub-package module, so it stays
collectable and runnable in a minimal environment.
"""

# Standard
from collections.abc import Iterator
from types import ModuleType
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.platform import _registry as platform_registry
from lmcache.v1.platform.base._base import PlatformBase
from lmcache.v1.utils.subclass_discovery import discover_subclasses


class _FakeBaseModule(ModuleType):
    ImportedMarked: type[PlatformBase]


# ---------------------------------------------------------------------------
# TEST 1a — Synthetic base-class discovery mechanism
# ---------------------------------------------------------------------------


def test_collect_base_classes_uses_platformbase_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that ``_collect_base_classes`` discovers exactly the locally
    defined ``PlatformBase`` subclasses in scanned modules.

    All classes are synthetic — this test has zero dependency on any real
    production class definition beyond ``PlatformBase`` itself.

    Checks:
    - ``FakeBaseA`` and ``FakeBaseB`` (locally-defined subclasses) are found.
    - ``Helper`` (non-subclass) is excluded.
    - ``ImportedMarked`` (``__module__`` mismatch, re-exported) is excluded.
    - ``PlatformBase`` itself is never collected.
    """
    saved = platform_registry.snapshot()
    # Build a fake 'base' module with locally-defined PlatformBase subclasses,
    # a non-subclass Helper, and a re-exported subclass (ImportedMarked).
    fake_mod = _FakeBaseModule("lmcache.v1.platform.base.fake")
    exec(
        "\n".join(
            [
                "from lmcache.v1.platform.base._base import PlatformBase",
                "class FakeBaseA(PlatformBase):",
                "    pass",
                "class FakeBaseB(PlatformBase):",
                "    pass",
                "class Helper:",
                "    pass",
            ]
        ),
        fake_mod.__dict__,
    )
    # ImportedMarked is a PlatformBase subclass but defined outside the module
    # (__module__ != fake_mod.__name__), so it must be excluded.
    imported_marked = type("ImportedMarked", (PlatformBase,), {})
    fake_mod.ImportedMarked = imported_marked

    def fake_iter_modules(_: object) -> Iterator[tuple[None, str, bool]]:
        return iter([(None, "fake", False)])

    def fake_import_module(name: str) -> ModuleType:
        if name != "lmcache.v1.platform.base.fake":
            raise AssertionError("unexpected module import %s" % name)
        return fake_mod

    monkeypatch.setattr(platform_registry.pkgutil, "iter_modules", fake_iter_modules)
    monkeypatch.setattr(
        platform_registry.importlib, "import_module", fake_import_module
    )

    try:
        platform_registry.reset_for_tests()
        base_classes = platform_registry._collect_base_classes()
        # Only the two locally-defined PlatformBase subclasses are discovered
        assert set(base_classes) == {fake_mod.FakeBaseA, fake_mod.FakeBaseB}
        # Non-subclass must not appear
        assert fake_mod.Helper not in base_classes
        # Re-exported subclass must not appear
        assert imported_marked not in base_classes
        # PlatformBase itself is never collected
        assert PlatformBase not in base_classes
    finally:
        platform_registry.restore(saved)


# ---------------------------------------------------------------------------
# TEST 1b — Synthetic subclass discovery + registry indexing
# ---------------------------------------------------------------------------


def test_synthetic_discover_and_index_subclasses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify subclass discovery and registry indexing with a purely synthetic
    platform package tree.

    All classes are synthetic — no real production concrete or base class is
    imported or referenced here. The test builds an in-memory package tree
    registered via ``sys.modules`` and patches ``pkgutil.iter_modules`` to
    return synthetic module listings.

    Synthetic package layout. With ``levels=[2, 2]`` the scan window is exactly
    depth 2, so only the depth-2 leaf modules are scanned for classes; the
    depth-1 sub-package ``__init__`` modules are NOT scanned for classes — they
    are only traversed to reach the depth-2 leaves:

        synth_platform/
            cpu/             <- depth-1 sub-package (traversed, not scanned)
                cpu_a.py     <- depth-2 leaf: FakeCpuA (FakeBaseA subclass)
                cpu_b.py     <- depth-2 leaf: FakeCpuB (FakeBaseB subclass)
            cuda/
                cuda_a.py    <- depth-2 leaf: FakeCudaA (FakeBaseA subclass)

    Checks:
    - ``discover_subclasses(synth_pkg, FakeBaseA, levels=[2,2])``
      → ``{FakeCpuA, FakeCudaA}``
      (FakeCpuB excluded because it subclasses FakeBaseB, not FakeBaseA).
    - ``discover_subclasses(synth_pkg, FakeBaseB, levels=[2,2])``
      → ``{FakeCpuB}``
      (FakeCpuA and FakeCudaA excluded; the base class itself is never yielded).
    - Registry ``get_all_impls`` and ``get_impl`` return the correct
      ``{device_type: cls}`` index for each base class.
    - Different bases do not cross-contaminate the registry.
    - An unregistered ``(base, device_type)`` pair raises ``ValueError``.
    """
    # --- 1. Synthetic base classes ------------------------------------------
    # Use type() to create clean synthetic PlatformBase subclasses.
    FakeBaseA = type("FakeBaseA", (PlatformBase,), {})
    FakeBaseB = type("FakeBaseB", (PlatformBase,), {})

    # --- 2. Synthetic device subclasses -------------------------------------
    # __module__ must match the module name they live in so that
    # discover_subclasses with require_defined_in_module=True accepts them.
    FakeCpuA = type(
        "FakeCpuA",
        (FakeBaseA,),
        {"device_type": "cpu", "__module__": "synth_platform.cpu.cpu_a"},
    )
    FakeCudaA = type(
        "FakeCudaA",
        (FakeBaseA,),
        {"device_type": "cuda", "__module__": "synth_platform.cuda.cuda_a"},
    )
    FakeCpuB = type(
        "FakeCpuB",
        (FakeBaseB,),
        {"device_type": "cpu", "__module__": "synth_platform.cpu.cpu_b"},
    )

    # --- 3. Build the in-memory synthetic package tree ----------------------
    # Each package level needs a unique __path__ list object so the patched
    # iter_modules can dispatch by identity (see fake_iter_modules below).
    _synth_root_path: list[str] = []
    _synth_cpu_path: list[str] = []
    _synth_cuda_path: list[str] = []

    synth_pkg = ModuleType("synth_platform")
    synth_pkg.__path__ = _synth_root_path  # type: ignore[assignment]

    synth_cpu_pkg = ModuleType("synth_platform.cpu")
    synth_cpu_pkg.__path__ = _synth_cpu_path  # type: ignore[assignment]

    synth_cuda_pkg = ModuleType("synth_platform.cuda")
    synth_cuda_pkg.__path__ = _synth_cuda_path  # type: ignore[assignment]

    # Leaf modules: each holds the corresponding device subclass.
    synth_cpu_a = ModuleType("synth_platform.cpu.cpu_a")
    synth_cpu_a.FakeCpuA = FakeCpuA  # type: ignore[attr-defined]

    synth_cpu_b = ModuleType("synth_platform.cpu.cpu_b")
    synth_cpu_b.FakeCpuB = FakeCpuB  # type: ignore[attr-defined]

    synth_cuda_a = ModuleType("synth_platform.cuda.cuda_a")
    synth_cuda_a.FakeCudaA = FakeCudaA  # type: ignore[attr-defined]

    # Register all synthetic modules so importlib.import_module finds them.
    for _mod in (
        synth_pkg,
        synth_cpu_pkg,
        synth_cuda_pkg,
        synth_cpu_a,
        synth_cpu_b,
        synth_cuda_a,
    ):
        monkeypatch.setitem(sys.modules, _mod.__name__, _mod)

    # --- 4. Patch pkgutil.iter_modules for the synthetic paths -------------
    # Only intercept calls originating from our synthetic __path__ lists;
    # fall back to the real implementation for all other paths so unrelated
    # code (e.g. other fixture teardown) is unaffected.
    _real_iter_modules = platform_registry.pkgutil.iter_modules

    def fake_iter_modules(
        path: object, prefix: str = ""
    ) -> Iterator[tuple[None, str, bool]]:
        if path is _synth_root_path:
            # depth-1: two device sub-packages
            return iter([(None, "cpu", True), (None, "cuda", True)])
        if path is _synth_cpu_path:
            # depth-2: leaf modules under cpu/
            return iter([(None, "cpu_a", False), (None, "cpu_b", False)])
        if path is _synth_cuda_path:
            # depth-2: leaf module under cuda/
            return iter([(None, "cuda_a", False)])
        # Real paths: delegate to the original implementation
        return _real_iter_modules(path, prefix)  # type: ignore[arg-type, return-value]

    monkeypatch.setattr(platform_registry.pkgutil, "iter_modules", fake_iter_modules)

    # --- 5. Verify discover_subclasses directly -----------------------------
    # levels=[2, 2] mirrors the real registry: only depth-2 leaf modules are
    # scanned; sub-packages at depth 1 are traversed but not scanned directly.
    found_a: set[type] = set(
        discover_subclasses(synth_pkg, FakeBaseA, levels=[2, 2], include_abstract=False)
    )
    found_b: set[type] = set(
        discover_subclasses(synth_pkg, FakeBaseB, levels=[2, 2], include_abstract=False)
    )
    # FakeBaseA subclasses: FakeCpuA + FakeCudaA; FakeCpuB (FakeBaseB sub) excluded
    assert found_a == {FakeCpuA, FakeCudaA}
    assert FakeCpuB not in found_a
    assert FakeBaseA not in found_a  # base class is never yielded

    # FakeBaseB subclasses: only FakeCpuB; FakeCpuA and FakeCudaA excluded
    assert found_b == {FakeCpuB}
    assert FakeCpuA not in found_b
    assert FakeCudaA not in found_b
    assert FakeBaseB not in found_b  # base class is never yielded

    # --- 6. Verify registry indexing via get_all_impls / get_impl -----------
    # Populate the registry directly using the discovered classes to verify
    # that the {device_type: cls} index is correct and that different bases
    # do not cross-contaminate one another.
    saved = platform_registry.snapshot()
    try:
        registry_state: dict[type, dict[str, type]] = {
            FakeBaseA: {cls.device_type: cls for cls in found_a},  # type: ignore[attr-defined]
            FakeBaseB: {cls.device_type: cls for cls in found_b},  # type: ignore[attr-defined]
        }
        platform_registry.restore(
            {
                "registry": registry_state,
                "availability": {},
                "discovered": True,
            }
        )

        # FakeBaseA → {cpu: FakeCpuA, cuda: FakeCudaA}
        assert platform_registry.get_all_impls(FakeBaseA) == {
            "cpu": FakeCpuA,
            "cuda": FakeCudaA,
        }
        assert platform_registry.get_impl(FakeBaseA, "cpu") is FakeCpuA
        assert platform_registry.get_impl(FakeBaseA, "cuda") is FakeCudaA

        # FakeBaseB → {cpu: FakeCpuB}; independent of FakeBaseA entries
        assert platform_registry.get_all_impls(FakeBaseB) == {"cpu": FakeCpuB}
        assert platform_registry.get_impl(FakeBaseB, "cpu") is FakeCpuB

        # No cross-contamination: cuda is not registered under FakeBaseB
        with pytest.raises(ValueError):
            platform_registry.get_impl(FakeBaseB, "cuda")

        # Completely unregistered base raises ValueError
        with pytest.raises(ValueError):
            platform_registry.get_impl(PlatformBase, "cpu")

    finally:
        platform_registry.restore(saved)
