# SPDX-License-Identifier: Apache-2.0
"""Classify pull requests that mainly change deprecated modules.

The workflow supplies the changed-file metadata returned by GitHub's pull
request files API. Policy remains in ``.github/deprecated-modules.toml`` so
maintainers can add or remove deprecated paths without editing this logic.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence
import argparse
import json
import tomllib

COMMENT_MARKER = "<!-- lmcache-deprecated-module-guard -->"


@dataclass(frozen=True)
class DeprecatedModule:
    """One deprecated module and the maintained alternative contributors use."""

    name: str
    paths: tuple[str, ...]
    replacement: str


@dataclass(frozen=True)
class Policy:
    """Repository policy for classifying deprecated-module changes."""

    minimum_deprecated_line_fraction: float
    minimum_deprecated_file_fraction: float
    minimum_deprecated_changed_lines: int
    allow_changes_label: str
    test_paths: tuple[str, ...]
    test_suffixes: tuple[str, ...]
    modules: tuple[DeprecatedModule, ...]


@dataclass(frozen=True)
class ChangedFile:
    """The changed-line metadata GitHub reports for one pull-request file."""

    filename: str
    status: str
    additions: int
    deletions: int
    previous_filename: str | None = None

    @property
    def changed_lines(self) -> int:
        """Return the additions and deletions GitHub attributes to the file."""
        return self.additions + self.deletions


@dataclass(frozen=True)
class ModuleMatch:
    """Deprecated changes attributed to one configured module."""

    module: DeprecatedModule
    changed_lines: int
    files: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    """Whether the pull request should be closed and the evidence for it."""

    should_close: bool
    should_comment: bool
    deprecated_lines: int
    total_lines: int
    deprecated_line_fraction: float
    deprecated_files: int
    total_files: int
    deprecated_file_fraction: float
    minimum_deprecated_line_fraction: float
    minimum_deprecated_file_fraction: float
    allow_changes_label: str
    allow_changes_applied: bool
    is_dependabot: bool
    minimum_deprecated_changed_lines: int
    matches: tuple[ModuleMatch, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for the workflow."""
        return asdict(self)


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _module_from_config(value: object, index: int) -> DeprecatedModule:
    if not isinstance(value, dict):
        raise ValueError(f"modules[{index}] must be a table")
    name = _nonempty_string(value.get("name"), f"modules[{index}].name")
    replacement = _nonempty_string(
        value.get("replacement"), f"modules[{index}].replacement"
    )
    raw_paths = value.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError(f"modules[{index}].paths must be a non-empty list")
    paths = tuple(
        _nonempty_string(path, f"modules[{index}].paths").strip("/")
        for path in raw_paths
    )
    if any(not path for path in paths):
        raise ValueError(f"modules[{index}].paths cannot contain the repository root")
    return DeprecatedModule(name=name, paths=paths, replacement=replacement)


def _string_list(raw: object, field: str, *, strip_slashes: bool) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a list")
    values = tuple(
        _nonempty_string(value, field).strip("/")
        if strip_slashes
        else _nonempty_string(value, field)
        for value in raw
    )
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"{field} must contain non-empty strings")
    return values


def load_policy(path: Path) -> Policy:
    """Load and validate the deprecated-module policy.

    Args:
        path: TOML policy file to read.

    Returns:
        The validated threshold and deprecated modules.

    Raises:
        ValueError: If the threshold or a module entry is malformed.
    """
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    line_fraction = _fraction(
        raw.get("minimum_deprecated_line_fraction"),
        "minimum_deprecated_line_fraction",
    )
    file_fraction = _fraction(
        raw.get("minimum_deprecated_file_fraction"),
        "minimum_deprecated_file_fraction",
    )
    minimum_lines = raw.get("minimum_deprecated_changed_lines")
    if type(minimum_lines) is not int or minimum_lines < 1:
        raise ValueError("minimum_deprecated_changed_lines must be a positive integer")
    allow_changes_label = _nonempty_string(
        raw.get("allow_changes_label"), "allow_changes_label"
    )
    test_paths = _string_list(raw.get("test_paths"), "test_paths", strip_slashes=True)
    test_suffixes = _string_list(
        raw.get("test_suffixes"), "test_suffixes", strip_slashes=False
    )

    raw_modules = raw.get("modules")
    if not isinstance(raw_modules, list) or not raw_modules:
        raise ValueError("modules must be a non-empty list")
    modules = tuple(
        _module_from_config(module, index) for index, module in enumerate(raw_modules)
    )
    names = [module.name for module in modules]
    if len(names) != len(set(names)):
        raise ValueError("module names must be unique")
    return Policy(
        minimum_deprecated_line_fraction=line_fraction,
        minimum_deprecated_file_fraction=file_fraction,
        minimum_deprecated_changed_lines=minimum_lines,
        allow_changes_label=allow_changes_label,
        test_paths=test_paths,
        test_suffixes=test_suffixes,
        modules=modules,
    )


def _belongs_to(filename: str, module: DeprecatedModule) -> bool:
    normalized = filename.strip("/")
    return any(
        normalized == path or normalized.startswith(f"{path}/") for path in module.paths
    )


def _fraction(value: object, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    fraction = float(value)
    if not 0 < fraction <= 1:
        raise ValueError(f"{field} must be in (0, 1]")
    return fraction


def _is_test_file(filename: str, policy: Policy) -> bool:
    normalized = filename.strip("/")
    return any(
        normalized == path or normalized.startswith(f"{path}/")
        for path in policy.test_paths
    ) or normalized.endswith(policy.test_suffixes)


def classify_changes(
    policy: Policy,
    files: Sequence[ChangedFile],
    labels: Sequence[str] = (),
    *,
    author: str = "",
) -> Decision:
    """Classify a pull request from GitHub's changed-file metadata.

    Files outside configured test paths and suffixes contribute to the line
    and file totals. Deprecated files contribute to both deprecated counts,
    except for whole-file removals, deletion-only modifications, and pure
    renames involving a deprecated path. A rename with edits is classified
    by its destination path.

    Args:
        policy: Deprecated modules and closing threshold.
        files: Changed files returned by GitHub's pull-request files API.
        labels: Current pull-request label names. The configured allow-changes
            label prevents automatic closure.
        author: GitHub login of the pull-request author. Dependabot PRs
            receive warnings without automatic closure.

    Returns:
        A decision with aggregate line and file counts plus per-module details.
    """
    total_lines = 0
    total_files = 0
    deprecated_lines = 0
    deprecated_files = 0
    lines_by_module = {module.name: 0 for module in policy.modules}
    files_by_module: dict[str, list[str]] = {
        module.name: [] for module in policy.modules
    }

    for changed_file in files:
        if _is_test_file(changed_file.filename, policy):
            continue
        matching = [
            module
            for module in policy.modules
            if _belongs_to(changed_file.filename, module)
        ]
        previous_filename = changed_file.previous_filename
        if (
            changed_file.status == "renamed"
            and changed_file.changed_lines == 0
            and (
                matching
                or (
                    previous_filename is not None
                    and any(
                        _belongs_to(previous_filename, module)
                        for module in policy.modules
                    )
                )
            )
        ):
            continue
        if len(matching) > 1:
            names = ", ".join(module.name for module in matching)
            raise ValueError(
                f"{changed_file.filename!r} matches multiple deprecated modules: "
                f"{names}"
            )
        # Removing deprecated code is cleanup, not continued development. It
        # affects neither side of either ratio.
        if matching and (
            changed_file.status == "removed"
            or (
                changed_file.status == "modified"
                and changed_file.additions == 0
                and changed_file.deletions > 0
            )
        ):
            continue

        total_lines += changed_file.changed_lines
        total_files += 1
        if not matching:
            continue
        module = matching[0]
        deprecated_lines += changed_file.changed_lines
        deprecated_files += 1
        lines_by_module[module.name] += changed_file.changed_lines
        files_by_module[module.name].append(changed_file.filename)

    line_fraction = deprecated_lines / total_lines if total_lines else 0.0
    file_fraction = deprecated_files / total_files if total_files else 0.0
    allow_changes_applied = policy.allow_changes_label in labels
    matches = tuple(
        ModuleMatch(
            module=module,
            changed_lines=lines_by_module[module.name],
            files=tuple(files_by_module[module.name]),
        )
        for module in policy.modules
        if files_by_module[module.name]
    )
    exceeds_threshold = (
        line_fraction >= policy.minimum_deprecated_line_fraction
        or file_fraction >= policy.minimum_deprecated_file_fraction
    )
    is_dependabot = author == "dependabot[bot]"
    return Decision(
        should_close=(
            bool(matches)
            and exceeds_threshold
            and deprecated_lines >= policy.minimum_deprecated_changed_lines
            and not allow_changes_applied
            and not is_dependabot
        ),
        should_comment=bool(matches),
        deprecated_lines=deprecated_lines,
        total_lines=total_lines,
        deprecated_line_fraction=line_fraction,
        deprecated_files=deprecated_files,
        total_files=total_files,
        deprecated_file_fraction=file_fraction,
        minimum_deprecated_line_fraction=policy.minimum_deprecated_line_fraction,
        minimum_deprecated_file_fraction=policy.minimum_deprecated_file_fraction,
        allow_changes_label=policy.allow_changes_label,
        allow_changes_applied=allow_changes_applied,
        is_dependabot=is_dependabot,
        minimum_deprecated_changed_lines=policy.minimum_deprecated_changed_lines,
        matches=matches,
    )


def render_comment(decision: Decision) -> str:
    """Render the deprecated-module warning for a pull request.

    Args:
        decision: A matching decision returned by :func:`classify_changes`.

    Returns:
        Markdown containing the classification, maintained alternatives, and
        whether the pull request was closed.

    Raises:
        ValueError: If the decision has no deprecated-module changes.
    """
    if not decision.should_comment:
        raise ValueError("cannot render a comment without deprecated changes")

    line_percentage = decision.deprecated_line_fraction * 100
    file_percentage = decision.deprecated_file_fraction * 100
    module_lines = "\n".join(
        f"- **{match.module.name}**: {match.changed_lines} changed lines across "
        f"{len(match.files)} file(s). {match.module.replacement}"
        for match in decision.matches
    )
    line_threshold = decision.minimum_deprecated_line_fraction * 100
    file_threshold = decision.minimum_deprecated_file_fraction * 100
    exceeded_dimensions = []
    if decision.deprecated_line_fraction >= decision.minimum_deprecated_line_fraction:
        exceeded_dimensions.append("changed lines")
    if decision.deprecated_file_fraction >= decision.minimum_deprecated_file_fraction:
        exceeded_dimensions.append("changed files")
    exceeded = " and ".join(exceeded_dimensions)
    if decision.should_close:
        outcome = (
            f"The {exceeded} dimension meets its configured threshold, so the "
            "pull request was automatically closed."
        )
    elif decision.is_dependabot:
        outcome = "The pull request remains open because Dependabot updates are exempt."
    elif decision.deprecated_lines < decision.minimum_deprecated_changed_lines:
        outcome = (
            "The pull request remains open because it changes fewer than "
            f"{decision.minimum_deprecated_changed_lines} deprecated lines."
        )
    elif decision.allow_changes_applied and exceeded_dimensions:
        outcome = (
            f"The {exceeded} dimension meets its configured threshold, but the "
            f"`{decision.allow_changes_label}` label keeps the pull request open."
        )
    else:
        outcome = (
            "Both dimensions are below their thresholds, so the pull request "
            "remains open."
        )

    return f"""{COMMENT_MARKER}
Thanks for the contribution. This pull request changes deprecated modules:

{module_lines}

- Changed lines: {decision.deprecated_lines}/{decision.total_lines}
  ({line_percentage:.1f}%; threshold {line_threshold:.1f}%)
- Changed files: {decision.deprecated_files}/{decision.total_files}
  ({file_percentage:.1f}%; threshold {file_threshold:.1f}%)

These modules receive only necessary bug fixes and removal work; please avoid
adding new features to them. {outcome}

The deprecated-module registry is maintained in
`.github/deprecated-modules.toml`. For an exceptional change, a maintainer can apply the
`{decision.allow_changes_label}` label and reopen the pull request.
Test files, deletion-only changes, and pure renames of deprecated files are
excluded from both ratios.
"""


def _changed_file(value: object, index: int) -> ChangedFile:
    if not isinstance(value, Mapping):
        raise ValueError(f"files[{index}] must be an object")
    filename = _nonempty_string(value.get("filename"), f"files[{index}].filename")
    status = _nonempty_string(value.get("status"), f"files[{index}].status")
    additions = value.get("additions")
    deletions = value.get("deletions")
    if not isinstance(additions, int) or isinstance(additions, bool) or additions < 0:
        raise ValueError(f"files[{index}].additions must be a non-negative integer")
    if not isinstance(deletions, int) or isinstance(deletions, bool) or deletions < 0:
        raise ValueError(f"files[{index}].deletions must be a non-negative integer")
    previous_filename = value.get("previous_filename")
    if previous_filename is not None:
        previous_filename = _nonempty_string(
            previous_filename, f"files[{index}].previous_filename"
        )
    return ChangedFile(filename, status, additions, deletions, previous_filename)


def _load_changed_files(path: Path) -> tuple[ChangedFile, ...]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("changed-files input must be a JSON list")
    return tuple(_changed_file(value, index) for index, value in enumerate(raw))


def _load_labels(path: Path) -> tuple[str, ...]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not all(isinstance(label, str) for label in raw):
        raise ValueError("labels input must be a JSON list of strings")
    return tuple(raw)


def main() -> None:
    """Classify a PR snapshot and write workflow-readable decision files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--files", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--author", default="")
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--comment", required=True, type=Path)
    args = parser.parse_args()

    decision = classify_changes(
        load_policy(args.config),
        _load_changed_files(args.files),
        _load_labels(args.labels),
        author=args.author,
    )
    args.decision.write_text(json.dumps(decision.to_dict(), indent=2) + "\n")
    args.comment.write_text(render_comment(decision) if decision.should_comment else "")


if __name__ == "__main__":
    main()
