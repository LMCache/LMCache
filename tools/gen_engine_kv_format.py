#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the Python ``EngineKVFormat`` enum from the shared X-macro spec.

The enum members are single-sourced in ``csrc/engine_kv_format.def`` (the same
file the C++ ``enum class`` and the pybind11 registrations are generated from).
This script renders a static, typed Python mirror at
``lmcache/_engine_kv_format.py`` so that:

- the pure-Python fallback (``lmcache.python_ops_fallback``) cannot drift from
  the compiled ``EngineKVFormat``, and
- static type checkers and IDEs see real enum members (a runtime ``IntEnum``
  built by parsing the ``.def`` would not).

Run directly (``python tools/gen_engine_kv_format.py``) or via the
``gen-engine-kv-format`` pre-commit hook. The generated file is committed; CI
re-runs this script and fails if the committed output is stale.
"""

# Standard
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
DEF_PATH = REPO_ROOT / "csrc" / "engine_kv_format.def"
OUT_PATH = REPO_ROOT / "lmcache" / "_engine_kv_format.py"

# Matches a single X-macro entry, e.g. ``X(NL_X_NB_BS_HS, 3)``, tolerating an
# optional trailing ``// comment``.
_MEMBER_RE = re.compile(r"^\s*X\(\s*(\w+)\s*,\s*(\d+)\s*\)\s*(?://.*)?$")
# Matches a ``// ...`` doc comment line (carried over as a Python ``#`` comment).
_COMMENT_RE = re.compile(r"^\s*//\s?(.*)$")

_HEADER = """\
# SPDX-License-Identifier: Apache-2.0
#
# AUTO-GENERATED FROM csrc/engine_kv_format.def -- DO NOT EDIT.
# Regenerate with: python tools/gen_engine_kv_format.py
# (or `pre-commit run gen-engine-kv-format --all-files`).
#
# This is the static, typed Python mirror of the C++ EngineKVFormat enum. The
# members are single-sourced in csrc/engine_kv_format.def; edit that file (one
# X(...) line) and regenerate, never edit this file by hand.

# Standard
from enum import IntEnum


class EngineKVFormat(IntEnum):
    \"\"\"Enumeration of different engine KV cache memory layouts.\"\"\"
"""


def parse_members(def_text: str) -> list[tuple[str, int, list[str]]]:
    """Parse the ``.def`` into ordered ``(name, value, comment_lines)`` tuples.

    Args:
        def_text: Full text of ``csrc/engine_kv_format.def``.

    Returns:
        One tuple per ``X(...)`` entry, in file order. ``comment_lines`` holds
        the ``//`` doc-comment lines immediately preceding the entry (the file
        header block, terminated by a blank line, is not attached to any entry).

    Raises:
        ValueError: If no members are found, or a name or value is duplicated.
    """
    members: list[tuple[str, int, list[str]]] = []
    seen_values: set[int] = set()
    seen_names: set[str] = set()
    pending_comments: list[str] = []
    for line in def_text.splitlines():
        member_match = _MEMBER_RE.match(line)
        if member_match is not None:
            name = member_match.group(1)
            value = int(member_match.group(2))
            if name in seen_names:
                raise ValueError(f"Duplicate EngineKVFormat name {name}")
            if value in seen_values:
                raise ValueError(f"Duplicate EngineKVFormat value {value} for {name}")
            seen_names.add(name)
            seen_values.add(value)
            members.append((name, value, pending_comments))
            pending_comments = []
            continue
        comment_match = _COMMENT_RE.match(line)
        if comment_match is not None:
            pending_comments.append(comment_match.group(1).rstrip())
        else:
            # Blank line or other content resets the pending comment block, so
            # only comments directly above an X(...) entry attach to it.
            pending_comments = []
    if not members:
        raise ValueError(f"No X(...) members found in {DEF_PATH}")
    return members


def render(members: list[tuple[str, int, list[str]]]) -> str:
    """Render the generated module text from parsed members.

    Args:
        members: Output of :func:`parse_members`.

    Returns:
        The full contents of ``lmcache/_engine_kv_format.py``.
    """
    lines: list[str] = [_HEADER.rstrip("\n")]
    for name, value, comments in members:
        lines.append("")
        for comment in comments:
            lines.append(f"    # {comment}" if comment else "    #")
        lines.append(f"    {name} = {value}")
    # Backward-compat alias (pre-#3673 name). Mirrors the GPUKVFormat alias
    # registered in the pybind modules, so callers and the c_ops/fallback parity
    # check see the same enum surface in both.
    lines.append("")
    lines.append("")
    lines.append("# Backward-compat alias for the pre-#3673 GPUKVFormat name.")
    lines.append("GPUKVFormat = EngineKVFormat")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Regenerate ``lmcache/_engine_kv_format.py`` from the ``.def`` spec.

    Returns:
        Process exit code (0 on success).
    """
    members = parse_members(DEF_PATH.read_text(encoding="utf-8"))
    OUT_PATH.write_text(render(members), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
