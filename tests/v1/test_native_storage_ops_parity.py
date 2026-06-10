# SPDX-License-Identifier: Apache-2.0
"""
Verify that every public class/method in the Python fallback
``lmcache.native_storage_ops`` matches the C++ extension
``native_storage_ops.cpython-*.so``.

Three parity dimensions only — purely structural, no
behavior testing:

1. Class existence    - every class in the .so has a fallback class.
2. Method existence   - every public method (and selected dunders) on the
                        .so class exists on the fallback class.
3. Method signature   - parameter names, count, and defaults match.
                        ``Bitmap.__init__`` is overloaded in pybind11; the
                        Python single-signature is accepted as long as it
                        is call-compatible with EVERY .so overload.

Requires the compiled extension (``.so``) to be importable.  Automatically
skipped on CPU-only / non-CUDA CI where only the Python fallback is present.

"""

# Standard
from typing import Any
import importlib
import importlib.util
import inspect
import os
import re

# Third Party
import pytest

# Strategy for getting both the .so and the .py fallback in the same process:
#
# When the C++ extension is built, ``lmcache.native_storage_ops`` lives next
# to ``native_storage_ops.py`` as ``native_storage_ops.cpython-*.so``.
# Python's import machinery prefers the .so over the .py, so a normal
# ``import lmcache.native_storage_ops`` returns the compiled extension when
# it is available and falls back to the .py module otherwise.
#
# To compare the two we therefore:
#   1. ``import lmcache.native_storage_ops`` → ``so_mod``.  This is the
#      production module.  If the .so was not built, this still imports
#      the .py fallback; we detect that case and skip the whole suite.
#   2. Locate ``native_storage_ops.py`` on disk and load it explicitly via
#      ``importlib.util.spec_from_file_location`` → ``fallback``.  Doing it
#      by file path bypasses the import-machinery preference and guarantees
#      we get the Python implementation regardless of whether the .so is
#      already loaded.
try:
    # First Party
    import lmcache.native_storage_ops as so_mod  # type: ignore[import-not-found]

    _so_file = getattr(so_mod, "__file__", "") or ""
    _is_extension = _so_file.endswith((".so", ".pyd"))

    # Locate the .py fallback file.  ``find_spec`` is unreliable here because
    # it returns the .so spec when the extension is loaded; we instead walk
    # the package's __path__ entries looking for the literal .py file.
    _fallback_path: str | None = None
    # First Party
    import lmcache as _lmcache_pkg

    for _candidate_dir in list(_lmcache_pkg.__path__):
        _candidate = os.path.join(_candidate_dir, "native_storage_ops.py")
        if os.path.isfile(_candidate):
            _fallback_path = _candidate
            break

    if not _is_extension or _fallback_path is None:
        raise ImportError(
            "native_storage_ops .so not built or .py fallback file not found"
        )

    _fallback_spec = importlib.util.spec_from_file_location(
        "_native_storage_ops_fallback", _fallback_path
    )
    if _fallback_spec is None or _fallback_spec.loader is None:
        raise ImportError("could not build a spec for the .py fallback")
    fallback = importlib.util.module_from_spec(_fallback_spec)
    _fallback_spec.loader.exec_module(fallback)

    HAS_SO = True
except (ImportError, AttributeError, OSError):
    so_mod = None  # type: ignore[assignment]
    fallback = None  # type: ignore[assignment]
    HAS_SO = False


# Five classes that make up the public API of native_storage_ops.
TARGET_CLASS_NAMES: tuple[str, ...] = (
    "TTLLock",
    "Bitmap",
    "ParallelPatternMatcher",
    "PeriodicEventNotifier",
    "RangePatternMatcher",
)

# Dunder methods we explicitly check (they implement real business logic
# rather than Python boilerplate).  Other dunders like ``__class__``,
# ``__hash__``, ``__init_subclass__`` are skipped to avoid noise.
TESTED_DUNDERS: frozenset[str] = frozenset(
    {"__init__", "__and__", "__or__", "__invert__", "__repr__"}
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for parsing docstrings
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_default(value_str: str) -> Any:
    """Normalize a default-value string for comparison.

    pybind11 docstrings render booleans lowercase (``false``/``true``); the
    Python ``inspect`` representation renders them as ``False``/``True``.
    Both spellings collapse to the same Python object after this pass.
    """
    s = value_str.strip()
    if s.lower() == "false":
        return False
    if s.lower() == "true":
        return True
    if s in ("None", "none"):
        return None
    if (s.startswith("'") and s.endswith("'")) or (
        s.startswith('"') and s.endswith('"')
    ):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _split_params(params_str: str) -> list[str]:
    """Split a ``"a: int, b: list[int] = [1, 2]"`` string at top-level commas."""
    params: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in params_str:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            params.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        params.append("".join(current).strip())
    return params


def _parse_param_list(params_str: str) -> list[tuple[str, bool, Any]]:
    """Parse ``"a: int, b: int = 5"`` → ``[("a", False, None), ("b", True, 5)]``."""
    if not params_str.strip():
        return []
    result: list[tuple[str, bool, Any]] = []
    for raw in _split_params(params_str):
        colon_idx = raw.find(":")
        if colon_idx == -1:
            name = raw.split("=")[0].strip()
        else:
            name = raw[:colon_idx].strip()
        has_default = "=" in raw
        default_value: Any = None
        if has_default:
            default_value = _normalize_default(raw.split("=", 1)[1])
        result.append((name, has_default, default_value))
    return result


def _strip_self_param(
    params: list[tuple[str, bool, Any]],
) -> list[tuple[str, bool, Any]]:
    """Drop a leading ``self``/``cls`` param if present.

    pybind11 method docstrings include ``self: ClassName`` as the first
    parameter; ``inspect.signature`` on a Python method *also* includes
    ``self``.  We strip both so the comparisons line up regardless of how
    each side was extracted.
    """
    if params and params[0][0] in ("self", "cls"):
        return params[1:]
    return params


def _parse_pybind_overloads(
    method: Any,
) -> list[list[tuple[str, bool, Any]]] | None:
    """Parse the docstring of a pybind11-bound method into a list of overload
    signatures.  Always returns a list (one entry for non-overloaded methods,
    multiple for overloaded ones), or ``None`` if parsing fails.

    pybind11 emits one of two shapes:

      Single signature:
          method_name(self: Cls, arg: int) -> ret

      Overloaded:
          method_name(*args, **kwargs)
          Overloaded function.

          1. method_name(self: Cls, size: int) -> None
          2. method_name(self: Cls, size: int, prefix_bits: int) -> None
    """
    doc = getattr(method, "__doc__", None)
    if not doc:
        return None

    # Overloaded form — collect every "N. name(...)" line.
    if "Overloaded function." in doc:
        overloads: list[list[tuple[str, bool, Any]]] = []
        for line in doc.split("\n"):
            stripped = line.strip()
            m = re.match(r"\d+\.\s*\w+\((.*)\)\s*->", stripped)
            if m:
                overloads.append(_strip_self_param(_parse_param_list(m.group(1))))
        return overloads if overloads else None

    # Single-signature form — first line.
    first = doc.strip().split("\n")[0]
    m = re.match(r"\w+\((.*)\)\s*->", first)
    if m is None:
        return None
    return [_strip_self_param(_parse_param_list(m.group(1)))]


def _get_python_params(func: Any) -> list[tuple[str, bool, Any]]:
    """Extract ``[(name, has_default, default_value)]`` via inspect.signature."""
    sig = inspect.signature(func)
    out: list[tuple[str, bool, Any]] = []
    for p in sig.parameters.values():
        if p.name in ("self", "cls"):
            continue
        has_default = p.default is not inspect.Parameter.empty
        default_value = _normalize_default(repr(p.default)) if has_default else None
        out.append((p.name, has_default, default_value))
    return out


def _has_real_names(params: list[tuple[str, bool, Any]]) -> bool:
    """pybind11 falls back to ``arg0``/``arg1``/... when no ``py::arg()`` was
    given.  In that case names are not meaningful and only count is checked.
    """
    return bool(params) and not any(
        re.match(r"^arg\d+$", name) for name, _, _ in params
    )


def _public_methods(cls: type) -> dict[str, Any]:
    """Return ``{name: callable}`` for the methods we want to compare."""
    out: dict[str, Any] = {}
    for name, obj in inspect.getmembers(cls):
        if not callable(obj):
            continue
        if name in TESTED_DUNDERS or not name.startswith("_"):
            out[name] = obj
    return out


def _params_compatible(
    py_params: list[tuple[str, bool, Any]],
    so_params: list[tuple[str, bool, Any]],
    *,
    check_names: bool,
) -> tuple[bool, str]:
    """Return ``(ok, reason)``.

    "Compatible" means: a Python caller using exactly the .so's parameter
    spelling (positional or keyword) would also be a valid call to the
    Python implementation.

    Rules
    -----
    * The Python signature MUST accept at least as many positional args as
      the .so exposes.
    * For each .so param, the Python side MUST have a same-position
      parameter.  If ``check_names`` is True, names must also match.
    * If the .so param has a default, the Python side MUST also have a
      default with the same value (otherwise callers relying on the
      default break).  The Python side MAY add a default where the .so
      has none (backward-compatible widening).
    * The Python side MAY have additional trailing parameters as long as
      they all carry defaults (so the .so call shape still works).
    """
    if len(py_params) < len(so_params):
        return (
            False,
            f"param count: .so has {len(so_params)}, fallback has {len(py_params)}",
        )

    for i, (so_p, py_p) in enumerate(zip(so_params, py_params, strict=False)):
        so_name, so_has_def, so_default = so_p
        py_name, py_has_def, py_default = py_p
        if check_names and so_name != py_name:
            return False, f"param #{i} name: .so='{so_name}', fallback='{py_name}'"
        if so_has_def:
            if not py_has_def:
                return (
                    False,
                    f"param #{i} ('{py_name}'): .so has default={so_default!r} "
                    f"but fallback has no default",
                )
            if so_default != py_default:
                return (
                    False,
                    f"param #{i} ('{py_name}'): default mismatch "
                    f".so={so_default!r}, fallback={py_default!r}",
                )

    # Trailing extras on the Python side must all be optional.
    for j in range(len(so_params), len(py_params)):
        py_name, py_has_def, _ = py_params[j]
        if not py_has_def:
            return (
                False,
                f"fallback has extra required parameter '{py_name}' at "
                f"position #{j} not present in .so",
            )

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────
_so_classes: dict[str, type] = (
    {
        name: getattr(so_mod, name)
        for name in TARGET_CLASS_NAMES
        if hasattr(so_mod, name)
    }
    if HAS_SO
    else {}
)
_fallback_classes: dict[str, type] = {
    name: getattr(fallback, name)
    for name in TARGET_CLASS_NAMES
    if hasattr(fallback, name)
}
_shared_class_names: list[str] = sorted(set(_so_classes) & set(_fallback_classes))


def _shared_method_names(cls_name: str) -> list[str]:
    if cls_name not in _so_classes or cls_name not in _fallback_classes:
        return []
    so_methods = _public_methods(_so_classes[cls_name])
    py_methods = _public_methods(_fallback_classes[cls_name])
    return sorted(set(so_methods) & set(py_methods))


_method_param_ids: list[tuple[str, str]] = [
    (cls_name, m_name)
    for cls_name in _shared_class_names
    for m_name in _shared_method_names(cls_name)
]


# ─────────────────────────────────────────────────────────────────────────────
# Structural parity tests
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not HAS_SO, reason="native_storage_ops .so not available")
def test_all_so_classes_have_fallback() -> None:
    """Every target class exposed by the .so must exist in the Python fallback."""
    missing = sorted(set(_so_classes) - set(_fallback_classes))
    assert not missing, (
        f".so classes missing from native_storage_ops fallback: {missing}"
    )


@pytest.mark.skipif(not HAS_SO, reason="native_storage_ops .so not available")
@pytest.mark.parametrize(
    "cls_name",
    _shared_class_names if _shared_class_names else ["__placeholder__"],
)
def test_class_method_existence(cls_name: str) -> None:
    """Every public method (and tested dunder) on the .so class must also be
    defined on the fallback class.  Extra methods on the fallback are fine
    (e.g. helpers).
    """
    if cls_name == "__placeholder__":
        pytest.skip("No shared classes between .so and fallback")
    so_methods = _public_methods(_so_classes[cls_name])
    py_methods = _public_methods(_fallback_classes[cls_name])
    missing = sorted(set(so_methods) - set(py_methods))
    assert not missing, (
        f"{cls_name}: methods present in .so but missing from fallback: {missing}"
    )


@pytest.mark.skipif(not HAS_SO, reason="native_storage_ops .so not available")
@pytest.mark.parametrize(
    ("cls_name", "method_name"),
    _method_param_ids if _method_param_ids else [("__placeholder__", "x")],
)
def test_method_signature_parity(cls_name: str, method_name: str) -> None:
    """Each shared method's signature on the fallback must be call-compatible
    with every .so overload.

    "Call-compatible" means: a caller writing code against the .so signature
    will also successfully call the fallback.  Adding optional keyword
    parameters in the fallback is fine; removing or renaming required ones
    is not.
    """
    if cls_name == "__placeholder__":
        pytest.skip("No shared methods to compare")

    so_method = _public_methods(_so_classes[cls_name])[method_name]
    py_method = _public_methods(_fallback_classes[cls_name])[method_name]

    so_overloads = _parse_pybind_overloads(so_method)
    if so_overloads is None:
        pytest.skip(
            f"{cls_name}.{method_name}: cannot parse .so signature from docstring"
        )

    try:
        py_params = _get_python_params(py_method)
    except (ValueError, TypeError):
        pytest.skip(f"{cls_name}.{method_name}: cannot inspect fallback signature")

    # The Python single signature must be call-compatible with at least one
    # .so overload, AND we expect it to cover ALL of them — otherwise the
    # fallback silently drops a public calling convention.
    failures: list[str] = []
    # mypy: so_overloads cannot be None here due to skip above
    assert so_overloads is not None
    for so_params in so_overloads:
        check_names = _has_real_names(so_params)
        ok, reason = _params_compatible(py_params, so_params, check_names=check_names)
        if not ok:
            failures.append(f"  overload {[p[0] for p in so_params]}: {reason}")

    assert not failures, (
        f"{cls_name}.{method_name}: fallback signature "
        f"{[p[0] for p in py_params]} is not call-compatible with all "
        f".so overloads:\n" + "\n".join(failures)
    )
