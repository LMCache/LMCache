# SPDX-License-Identifier: Apache-2.0

"""Structural conformance tests for ``L1ManagerInterface``.

These bind the Protocol to ``L1Manager`` so that method-set or signature drift
between the two fails CI (the guard the sibling design relies on -- a
``runtime_checkable`` ``issubclass`` alone only checks method *names*).
"""

# Standard
import inspect

# Third Party
import pytest

try:
    # First Party
    from lmcache.v1.distributed.l1_manager import L1Manager
    from lmcache.v1.distributed.l1_protocol import L1ManagerInterface
except ImportError:
    pytest.skip(
        "L1Manager / L1ManagerInterface unavailable (native ext missing)",
        allow_module_level=True,
    )


def _interface_methods() -> set[str]:
    """Public method names declared on the Protocol."""
    return {
        name
        for name, val in vars(L1ManagerInterface).items()
        if callable(val) and not name.startswith("_")
    }


def _l1_manager_methods() -> set[str]:
    """Public method names on the concrete L1Manager."""
    return {
        name
        for name in dir(L1Manager)
        if not name.startswith("_") and callable(getattr(L1Manager, name))
    }


def _unwrap(fn):
    """Recover the wrapped function from ``l1_mgr_synchronized``.

    That decorator is a plain closure without ``functools.wraps``, so
    ``inspect.signature`` would otherwise see ``(self, *args, **kwargs)``. The
    original function is the ``func`` free variable of the wrapper closure.
    """
    code = getattr(fn, "__code__", None)
    while code is not None and fn.__closure__ and "func" in code.co_freevars:
        fn = fn.__closure__[code.co_freevars.index("func")].cell_contents
        code = getattr(fn, "__code__", None)
    return fn


def _params(fn) -> list[tuple[str, object, object]]:
    """Parameter (name, kind, default) list excluding ``self``.

    Annotations are excluded on purpose: L1Manager leaves some param/return
    annotations off, so only the call shape (names/kinds/defaults) is compared.
    """
    return [
        (p.name, p.kind, p.default)
        for p in inspect.signature(fn).parameters.values()
        if p.name != "self"
    ]


def test_interface_matches_l1_manager_surface():
    """The Protocol declares exactly L1Manager's public method set."""
    methods = _interface_methods()
    assert methods == _l1_manager_methods()
    assert len(methods) == 17  # tripwire against wholesale drift


def test_signatures_match_l1_manager():
    """Each Protocol method's call shape matches L1Manager's."""
    for name in _interface_methods():
        proto = _params(getattr(L1ManagerInterface, name))
        impl = _params(_unwrap(getattr(L1Manager, name)))
        assert proto == impl, name


def test_l1_manager_conforms_to_interface():
    """The stock L1Manager satisfies the interface (structural, no instance)."""
    assert issubclass(L1Manager, L1ManagerInterface)


def test_incomplete_class_does_not_conform():
    """A class missing interface methods is rejected (negative control)."""

    class Partial:
        def close(self) -> None: ...

    assert not issubclass(Partial, L1ManagerInterface)
