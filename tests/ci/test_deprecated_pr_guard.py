# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the deprecated-module pull-request guard."""

# Standard
from pathlib import Path
from types import ModuleType
import importlib.util
import sys

# Third Party
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / ".github/scripts/deprecated_pr_guard.py"
_CONFIG = _ROOT / ".github/deprecated-modules.toml"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_lmcache_deprecated_pr_guard", _SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the classifier from {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_module()

_TEST_POLICY = guard.Policy(
    minimum_deprecated_line_fraction=0.5,
    minimum_deprecated_file_fraction=0.5,
    minimum_deprecated_changed_lines=10,
    allow_changes_label="allow-deprecated-changes",
    test_paths=("tests",),
    test_suffixes=("_test.go",),
    modules=(
        guard.DeprecatedModule("Kubernetes operator", ("operator",), "Use DaemonSet."),
        guard.DeprecatedModule(
            "ZMQ request transport",
            ("lmcache/v1/client/transport/zmq_impl",),
            "Use the gRPC request transport.",
        ),
    ),
)


def test_repository_policy_loads() -> None:
    """Validate the configured policy independently of classification examples."""
    guard.load_policy(_CONFIG)


def _file(
    filename: str,
    additions: int,
    deletions: int = 0,
    status: str = "modified",
) -> object:
    return guard.ChangedFile(
        filename=filename,
        status=status,
        additions=additions,
        deletions=deletions,
    )


def test_majority_deprecated_changes_close_the_pull_request() -> None:
    """Reaching a configured deprecated-change threshold closes the PR."""
    policy = _TEST_POLICY

    decision = guard.classify_changes(
        policy,
        [
            _file("operator/controllers/cache_controller.go", 30, 20),
            _file("lmcache/v1/cache_engine.py", 30, 20),
        ],
    )

    assert decision.should_close
    assert decision.should_comment
    assert decision.deprecated_lines == 50
    assert decision.total_lines == 100
    assert decision.deprecated_line_fraction == 0.5
    assert decision.deprecated_files == 1
    assert decision.total_files == 2
    assert decision.deprecated_file_fraction == 0.5
    assert [match.module.name for match in decision.matches] == ["Kubernetes operator"]


@pytest.mark.parametrize(
    ("files", "should_comment"),
    [
        (
            [
                _file("operator/controllers/cache_controller.go", 10),
                _file("lmcache/v1/cache_engine.py", 10),
                _file("lmcache/v1/storage_backend.py", 10),
                _file("lmcache/v1/lookup_client.py", 10),
            ],
            True,
        ),
        (
            [
                _file(
                    "operator/controllers/cache_controller.go",
                    additions=0,
                    deletions=100,
                    status="removed",
                )
            ],
            False,
        ),
    ],
)
def test_minor_or_removed_deprecated_changes_remain_open(
    files: list[object], should_comment: bool
) -> None:
    """Small deprecated changes and complete removals do not close the PR."""
    decision = guard.classify_changes(_TEST_POLICY, files)

    assert not decision.should_close
    assert decision.should_comment is should_comment
    if not should_comment:
        assert decision.total_lines == 0
        assert decision.total_files == 0


def test_test_files_do_not_affect_the_deprecated_code_ratio() -> None:
    """Test additions affect neither side of the production-code ratio."""
    policy = _TEST_POLICY

    decision = guard.classify_changes(
        policy,
        [
            _file("operator/controllers/cache_controller.go", 10),
            _file("operator/controllers/cache_controller_test.go", 100),
            _file("tests/v1/test_cache_engine.py", 100),
            _file("lmcache/v1/cache_engine.py", 10),
            _file("lmcache/v1/storage_backend.py", 10),
            _file("lmcache/v1/lookup_client.py", 10),
        ],
    )

    assert decision.total_lines == 40
    assert decision.deprecated_lines == 10
    assert decision.deprecated_line_fraction == 0.25
    assert decision.total_files == 4
    assert decision.deprecated_files == 1
    assert decision.deprecated_file_fraction == 0.25
    assert decision.should_comment
    assert not decision.should_close
    assert "pull request remains open" in guard.render_comment(decision)


def test_deprecated_file_fraction_can_close_the_pull_request() -> None:
    """Many small deprecated files trigger the file-count threshold."""
    policy = _TEST_POLICY

    decision = guard.classify_changes(
        policy,
        [
            _file("operator/controllers/a.go", 4),
            _file("operator/controllers/b.go", 4),
            _file("operator/controllers/c.go", 4),
            _file("lmcache/v1/cache_engine.py", 100),
        ],
    )

    assert decision.deprecated_line_fraction == pytest.approx(12 / 112)
    assert decision.deprecated_file_fraction == 0.75
    assert decision.should_close
    assert "changed files dimension" in guard.render_comment(decision)


def test_comment_names_the_module_and_replacement() -> None:
    """The automatic reply tells contributors why the PR was closed."""
    policy = _TEST_POLICY
    decision = guard.classify_changes(
        policy,
        [_file("lmcache/v1/client/transport/zmq_impl/mq_client.py", 12)],
    )

    comment = guard.render_comment(decision)

    assert "ZMQ request transport" in comment
    assert "Changed lines: 12/12" in comment
    assert "Changed files: 1/1" in comment
    assert "Use the gRPC request transport" in comment
    assert "automatically closed" in comment
    assert "`allow-deprecated-changes` label" in comment
    assert guard.COMMENT_MARKER in comment


def test_allow_changes_label_keeps_the_pull_request_open() -> None:
    """The configured label keeps an otherwise rejected PR open."""
    policy = _TEST_POLICY

    decision = guard.classify_changes(
        policy,
        [_file("operator/controllers/cache_controller.go", 20)],
        labels=[policy.allow_changes_label],
    )

    assert decision.allow_changes_applied
    assert decision.should_comment
    assert not decision.should_close


@pytest.mark.parametrize(
    ("additions", "deletions", "author", "should_comment"),
    [
        (1, 0, "contributor", True),
        (40, 0, "dependabot[bot]", True),
        (0, 40, "contributor", False),
    ],
)
def test_small_changes_dependency_updates_and_cleanup_remain_open(
    additions: int, deletions: int, author: str, should_comment: bool
) -> None:
    decision = guard.classify_changes(
        _TEST_POLICY,
        [_file("operator/go.mod", additions, deletions)],
        author=author,
    )

    assert not decision.should_close
    assert decision.should_comment is should_comment
    if should_comment:
        assert "pull request remains open" in guard.render_comment(decision)
    else:
        assert decision.total_lines == decision.total_files == 0


@pytest.mark.parametrize(
    ("maintained_deletions", "expected_ratios"),
    [(None, (0.5, 1 / 3)), (40, (0.2, 0.5))],
)
def test_one_ratio_at_threshold_is_enough(
    maintained_deletions: int | None, expected_ratios: tuple[float, float]
) -> None:
    files = [_file("operator/controller.go", 10)]
    if maintained_deletions is None:
        files.extend([_file("lmcache/a.py", 5), _file("lmcache/b.py", 5)])
    else:
        files.append(_file("lmcache/old.py", 0, maintained_deletions, "removed"))

    decision = guard.classify_changes(_TEST_POLICY, files)

    assert (decision.deprecated_line_fraction, decision.deprecated_file_fraction) == (
        pytest.approx(expected_ratios)
    )
    assert decision.should_close


def test_similarly_named_directory_is_not_deprecated() -> None:
    decision = guard.classify_changes(
        _TEST_POLICY, [_file("operator_utils/helper.go", 20)]
    )

    assert decision.total_files == 1
    assert decision.deprecated_files == 0
    assert not decision.should_comment
    assert not decision.should_close


@pytest.mark.parametrize("changed_lines", [0, 20])
def test_rename_out_of_deprecated_module(changed_lines: int) -> None:
    """An edited move out counts only toward the overall change totals."""
    moved = guard.ChangedFile(
        filename="lmcache/controller.go",
        previous_filename="operator/controller.go",
        status="renamed",
        additions=changed_lines,
        deletions=0,
    )

    decision = guard.classify_changes(_TEST_POLICY, [moved])

    assert decision.total_files == int(changed_lines > 0)
    assert decision.total_lines == changed_lines
    assert decision.deprecated_files == decision.deprecated_lines == 0
    assert not decision.should_close
    assert not decision.should_comment


@pytest.mark.parametrize("old_path", ["lmcache/controller.go", "operator/old.go"])
def test_edited_rename_to_deprecated_path_counts(old_path: str) -> None:
    moved = guard.ChangedFile(
        filename="operator/controller.go",
        previous_filename=old_path,
        status="renamed",
        additions=20,
        deletions=0,
    )

    decision = guard.classify_changes(_TEST_POLICY, [moved])

    assert decision.deprecated_files == decision.total_files == 1
    assert decision.deprecated_lines == decision.total_lines == 20
    assert decision.should_close
