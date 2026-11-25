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
    policy = guard.load_policy(_CONFIG)

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
    decision = guard.classify_changes(guard.load_policy(_CONFIG), files)

    assert not decision.should_close
    assert decision.should_comment is should_comment
    if not should_comment:
        assert decision.total_lines == 0
        assert decision.total_files == 0


def test_test_files_do_not_affect_the_deprecated_code_ratio() -> None:
    """Test additions affect neither side of the production-code ratio."""
    policy = guard.load_policy(_CONFIG)

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
    policy = guard.load_policy(_CONFIG)

    decision = guard.classify_changes(
        policy,
        [
            _file("operator/controllers/a.go", 1),
            _file("operator/controllers/b.go", 1),
            _file("operator/controllers/c.go", 1),
            _file("lmcache/v1/cache_engine.py", 100),
        ],
    )

    assert decision.deprecated_line_fraction == pytest.approx(3 / 103)
    assert decision.deprecated_file_fraction == 0.75
    assert decision.should_close
    assert "changed files dimension" in guard.render_comment(decision)


def test_comment_names_the_module_and_replacement() -> None:
    """The automatic reply tells contributors why the PR was closed."""
    policy = guard.load_policy(_CONFIG)
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
    policy = guard.load_policy(_CONFIG)

    decision = guard.classify_changes(
        policy,
        [_file("operator/controllers/cache_controller.go", 20)],
        labels=[policy.allow_changes_label],
    )

    assert decision.allow_changes_applied
    assert decision.should_comment
    assert not decision.should_close
