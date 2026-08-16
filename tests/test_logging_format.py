# SPDX-License-Identifier: Apache-2.0
"""Guard logging calls against passing more arguments than they can render.

``logging`` formats a record lazily as ``msg % args``.  A call that passes more
positional arguments than its format string consumes raises ``TypeError``
*inside* the logging machinery; logging swallows that and prints
``--- Logging error ---`` to stderr, so the intended message is never emitted.

``PLE1205``/``PLE1206`` are enabled in ``pyproject.toml`` and cover plain string
literals, but ruff cannot count placeholders inside an f-string.  This test
walks the AST so the f-string variant -- a stray comma where implicit string
concatenation was intended -- stays covered as well.
"""

# Standard
from pathlib import Path
from typing import List, Optional
import ast
import re

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "lmcache"

# Mirrors the `exclude` list in [tool.ruff] of pyproject.toml.
EXCLUDED = ("lmcache_mp_connector_",)

LOGGER_NAMES = frozenset({"logger", "log", "logging", "self"})
LOG_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical"}
)

# A `%`-style conversion specifier, e.g. `%s`, `%-5.2f`, `%%`.
CONVERSION = re.compile(r"%[-#0 +]*(?:\*|\d+)?(?:\.(?:\*|\d+))?[hlL]?([a-zA-Z%])")


def _literal_text(node: ast.expr) -> Optional[str]:
    """Return the literal text of a string node, or None if it is not one.

    Args:
        node: The AST node holding the logging call's message argument.

    Returns:
        The concatenation of the node's literal string parts, with the
        interpolated slots of an f-string omitted (they contribute no `%`
        conversions), or None when the node is not a string literal at all --
        for example a name, a call, or a `"..." % value` expression whose
        argument count cannot be checked statically.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
        return "".join(parts)
    return None


def _placeholder_count(text: str) -> Optional[int]:
    """Count the arguments a `%`-style format string consumes.

    Args:
        text: The literal text of the message argument.

    Returns:
        The number of positional arguments the string consumes, or None when
        the string uses mapping keys (`%(name)s`) or a `*` width, since those
        consume a single mapping or a variable number of arguments.
    """
    if "%(" in text or "%*" in text:
        return None
    count = 0
    for match in CONVERSION.finditer(text):
        if match.group(1) == "%":
            continue
        count += 1
    return count


def _violations(tree: ast.Module) -> List[str]:
    """Collect logging calls whose argument count cannot render.

    Args:
        tree: The parsed module to inspect.

    Returns:
        A list of human-readable descriptions, one per offending call.
    """
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in LOG_METHODS:
            continue
        root = func.value
        while isinstance(root, ast.Attribute):
            root = root.value
        if not isinstance(root, ast.Name) or root.id not in LOGGER_NAMES:
            continue
        if not node.args or any(isinstance(a, ast.Starred) for a in node.args):
            continue
        text = _literal_text(node.args[0])
        if text is None:
            continue
        expected = _placeholder_count(text)
        if expected is None:
            continue
        supplied = len(node.args) - 1
        if supplied != expected:
            found.append(
                f"line {node.lineno}: {func.attr}() consumes {expected} "
                f"argument(s) but {supplied} were passed"
            )
    return found


def _source_files() -> List[Path]:
    """Return the package sources this guard applies to.

    Returns:
        Every `.py` file under `lmcache/`, minus the paths excluded from ruff.
    """
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if not any(marker in path.name for marker in EXCLUDED)
    )


def test_logging_arguments_match_format_string():
    """Every logging call must pass exactly as many arguments as it renders."""
    reported = []
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(PACKAGE_ROOT.parent)
        reported.extend(f"{rel}:{entry}" for entry in _violations(tree))
    assert not reported, "logging calls that drop their message:\n  " + "\n  ".join(
        reported
    )


def test_guard_detects_a_stray_comma_in_an_f_string_call():
    """The guard catches the f-string variant ruff's PLE1205 cannot see."""
    tree = ast.parse(
        "logger.warning(\n"
        '    f"Mock object is None on {i}",\n'
        '    f" out of {len(mock_objs)} objects",\n'
        ")\n"
    )
    assert len(_violations(tree)) == 1


def test_guard_accepts_a_well_formed_call():
    """A message whose placeholders match its arguments is not reported."""
    tree = ast.parse('logger.error("closing %s failed: %s", url, exc)')
    assert _violations(tree) == []
