# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBLN nightly lane's rbln-serve image resolver.

Run without loading the torch-dependent tests/conftest.py:
    uv run --no-sync --python 3.12 python -m pytest -q \
        --confcutdir=tests/ci tests/ci/test_rbln_nightly_image.py
"""

# Standard
from pathlib import Path
from types import ModuleType
import importlib.util

# Third Party
import pytest

RESOLVER = (
    Path(__file__).resolve().parents[2]
    / ".buildkite/k3_tests/rbln/resolve_nightly_image.py"
)


@pytest.fixture(scope="module")
def resolver() -> ModuleType:
    """Load the resolver script, which lives outside any package."""
    spec = importlib.util.spec_from_file_location("resolve_nightly_image", RESOLVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _version_set(
    tag: str | None,
    awarded: str,
    digest: str | None = "sha256:abc",
    other_achievements: tuple[tuple[str, str], ...] = (),
) -> dict[str, object]:
    """Build a ``/api/version-sets`` item verified at ``awarded``."""
    achievements = [{"name": "nightly-verified", "created_at": awarded}]
    achievements += [
        {"name": name, "created_at": created_at}
        for name, created_at in other_achievements
    ]
    return {
        "image_ref": tag and f"harbor.example/rbln-serve:{tag}",
        "image_digest": digest,
        "achievements": achievements,
    }


def test_returns_image_pinned_by_digest(resolver: ModuleType) -> None:
    """The result names the verified image by both tag and digest."""
    version_sets = [_version_set("a", "2026-10-28T17:53:30.844763+00:00")]

    assert (
        resolver.latest_verified_image(version_sets)
        == "harbor.example/rbln-serve:a@sha256:abc"
    )


def test_picks_newest_award_not_list_order(resolver: ModuleType) -> None:
    """The most recently awarded version set wins wherever it is listed."""
    version_sets = [
        _version_set("listed-first", "2026-10-17T15:23:49.377370+00:00"),
        _version_set("verified-last", "2026-10-28T17:53:30.844763+00:00"),
        _version_set("listed-last", "2026-10-12T01:17:01.683906+00:00"),
    ]

    assert resolver.latest_verified_image(version_sets).startswith(
        "harbor.example/rbln-serve:verified-last@"
    )


def test_ignores_other_achievement_times(resolver: ModuleType) -> None:
    """Only the nightly-verified award time decides the winner."""
    version_sets = [
        _version_set(
            "old",
            "2026-10-12T01:17:01.683906+00:00",
            other_achievements=(("perf-verified", "2026-10-30T00:00:00+00:00"),),
        ),
        _version_set("new", "2026-10-17T15:23:49.377370+00:00"),
    ]

    assert resolver.latest_verified_image(version_sets).startswith(
        "harbor.example/rbln-serve:new@"
    )


def test_skips_version_sets_without_image(resolver: ModuleType) -> None:
    """Version sets missing an image ref or digest are never chosen."""
    version_sets = [
        _version_set(None, "2026-10-28T00:00:00+00:00"),
        _version_set("no-digest", "2026-10-27T00:00:00+00:00", digest=None),
        _version_set("with-image", "2026-10-12T00:00:00+00:00"),
    ]

    assert resolver.latest_verified_image(version_sets).startswith(
        "harbor.example/rbln-serve:with-image@"
    )


def test_fails_when_no_version_set_has_image(resolver: ModuleType) -> None:
    """The lookup fails rather than printing an empty image."""
    version_sets = [_version_set(None, "2026-10-28T00:00:00+00:00")]

    with pytest.raises(SystemExit):
        resolver.latest_verified_image(version_sets)
