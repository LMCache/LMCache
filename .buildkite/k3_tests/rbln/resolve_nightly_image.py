# SPDX-License-Identifier: Apache-2.0
"""Print the rbln-serve image that most recently passed rbln-serve nightly CI.

rbln-serve's CI awards ``nightly-verified`` to each version set (driver +
rbln-serve image) that passes its nightly run. The RBLN nightly lane runs the
LMCache smoke on that image, so vLLM-RBLN changes show up before the PR lane's
pinned tag is bumped. Exits non-zero when no verified version set has an image.

Usage: python resolve_nightly_image.py
"""

# Standard
from datetime import datetime
from typing import TypedDict
import json
import urllib.parse
import urllib.request

ACHIEVEMENT = "nightly-verified"
VERSION_SETS_URL = "https://rbln-serve.udc.rbln.in/api/version-sets"


class Achievement(TypedDict):
    """An achievement awarded to a version set."""

    name: str
    created_at: str


class VersionSet(TypedDict):
    """The fields of a ``/api/version-sets`` item that this script reads."""

    image_ref: str | None
    image_digest: str | None
    achievements: list[Achievement]


def _awarded_at(version_set: VersionSet) -> datetime:
    """Return when ``version_set`` was awarded ``nightly-verified``."""
    return next(
        datetime.fromisoformat(achievement["created_at"])
        for achievement in version_set["achievements"]
        if achievement["name"] == ACHIEVEMENT
    )


def latest_verified_image(version_sets: list[VersionSet]) -> str:
    """Pick the image of the most recently verified version set.

    The newest award wins, not the newest version set, because an older
    version set can pass nightly CI after a newer one.

    Args:
        version_sets: ``items`` of a ``/api/version-sets`` response filtered
            by ``achievement=nightly-verified``.

    Returns:
        ``<image_ref>@<image_digest>``, so the pod pulls the exact image that
        was verified even if its tag is later moved.

    Raises:
        SystemExit: If no version set has both an image ref and a digest.
    """
    candidates = [
        (_awarded_at(vs), f"{vs['image_ref']}@{vs['image_digest']}")
        for vs in version_sets
        if vs["image_ref"] and vs["image_digest"]
    ]
    if not candidates:
        raise SystemExit(f"no {ACHIEVEMENT} version set has an image")
    return max(candidates)[1]


def main() -> None:
    """Fetch the verified version sets and print the latest image.

    Raises:
        urllib.error.URLError: If the version-set API is unreachable.
        SystemExit: If no verified version set has an image.
    """
    query = urllib.parse.urlencode({"achievement": ACHIEVEMENT})
    with urllib.request.urlopen(f"{VERSION_SETS_URL}?{query}", timeout=30) as resp:
        version_sets = json.load(resp)["items"]
    print(latest_verified_image(version_sets))


if __name__ == "__main__":
    main()
