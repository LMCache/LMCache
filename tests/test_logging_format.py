# SPDX-License-Identifier: Apache-2.0
"""Guards logging calls in ``lmcache/`` against format-string/argument mismatches.

``logging`` renders a record lazily as ``msg % args``. When the two disagree the
formatting raises ``TypeError`` *inside* the logging machinery, so the intended
message is replaced by a ``--- Logging error ---`` traceback on stderr. These
calls sit on failure paths, which is exactly where losing the message hurts most.

Ruff's ``PLE1205``/``PLE1206`` cover the static-string case. They cannot see
through an f-string, so ``logger.warning(f"a {x}", " and b")`` -- a stray comma
where implicit string concatenation was intended -- stays invisible to the
linter. This test closes that gap while the codebase migrates from f-string
logging to %-format.
"""

# Standard
from pathlib import Path
from typing import List, Optional, Tuple
import ast
import re

LMCACHE_ROOT = Path(__file__).resolve().parents[1] / "lmcache"

LOG_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal"}
)

# Excluded from linting in pyproject.toml; generated vLLM connector shims.
EXCLUDED = ("lmcache_mp_connector_",)

# %[(key)][flags][width][.precision][length]conversion
PERCENT_SPEC = re.compile(
    r"%(?:\((?P<key>[^)]*)\))?"
    r"[#0\- +]*"
    r"(?P<width>\*|\d+)?"
    r"(?:\.(?P<precision>\*|\d+))?"
    r"[hlL]?"
    r"(?P<conversion>[diouxXeEfFgGcrsa%])"
)


def iter_source_files() -> List[Path]:
    """Return every Python file in the ``lmcache`` package that CI lints."""
    return [
        path
        for path in sorted(LMCACHE_ROOT.rglob("*.py"))
        if not any(marker in path.name for marker in EXCLUDED)
    ]


def resolve_logger_method(call: ast.Call) -> Optional[str]:
    """Return the log level for ``call`` if it looks like a logging call.

    Matches ``<receiver>.<level>(...)`` where the receiver's name contains
    "log" (``logger``, ``logging``, ``self.logger``, ``_LOG``, ...). Narrow on
    purpose: a false positive here would fail CI on unrelated code.
    """
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr not in LOG_METHODS:
        return None

    receiver = func.value
    if isinstance(receiver, ast.Name):
        name = receiver.id
    elif isinstance(receiver, ast.Attribute):
        name = receiver.attr
    else:
        return None

    return func.attr if "log" in name.lower() else None


def literal_format_string(node: ast.expr) -> Optional[str]:
    """Return the statically known part of a format string, or ``None``.

    For an f-string only the literal segments are returned; interpolations
    cannot contribute ``%`` placeholders.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    return None


def count_placeholders(fmt: str) -> Tuple[int, int, int]:
    """Return ``(positional, mapping, star)`` placeholder counts for ``fmt``."""
    positional = mapping = star = 0
    for match in PERCENT_SPEC.finditer(fmt):
        if match.group("conversion") == "%":
            continue
        if match.group("key") is not None:
            mapping += 1
        else:
            positional += 1
        star += (match.group("width") == "*") + (match.group("precision") == "*")
    return positional, mapping, star


def find_violations(source_file: Path) -> List[str]:
    """Return a human-readable description of each mismatched call in a file."""
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or resolve_logger_method(node) is None:
            continue
        if not node.args or any(isinstance(arg, ast.Starred) for arg in node.args):
            continue

        fmt = literal_format_string(node.args[0])
        if fmt is None:
            continue

        expected, mapping, star = count_placeholders(fmt)
        # Mapping style takes a single dict; star width/precision consumes extra
        # positional args. Neither is statically checkable here.
        if mapping or star:
            continue

        supplied = len(node.args) - 1
        if expected != supplied:
            location = f"{source_file.relative_to(LMCACHE_ROOT.parent)}:{node.lineno}"
            violations.append(
                f"{location} expects {expected} arg(s) but {supplied} "
                f"were supplied: {fmt!r}"
            )
    return violations


def test_logging_calls_match_their_format_string() -> None:
    """Every logging call supplies exactly as many args as its format consumes."""
    violations = [
        violation
        for source_file in iter_source_files()
        for violation in find_violations(source_file)
    ]

    assert not violations, (
        "Logging calls would raise TypeError at runtime:\n" + "\n".join(violations)
    )
