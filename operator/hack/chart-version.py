# SPDX-License-Identifier: Apache-2.0
"""Convert Operator release tags to Helm chart SemVer without changing image tags."""

# Standard
import datetime
import re
import sys


def chart_version(version: str) -> str:
    """Return chart SemVer for a stable, prerelease, or dated nightly version.

    Args:
        version: Operator version, such as v0.5.5, v0.5.6rc1, v0.5.6-rc.1,
            or nightly-2026-09-19. Compact alpha/beta suffixes are also accepted.

    Returns:
        A SemVer string suitable for ``helm package --version``.

    Raises:
        ValueError: If the version is unsupported or contains an invalid date.
    """
    numeric = r"(?:0|[1-9][0-9]*)"
    identifier = rf"(?:{numeric}|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    prerelease = rf"{identifier}(?:\.{identifier})*"
    release = re.fullmatch(
        rf"v?({numeric}\.{numeric}\.{numeric})"
        rf"(?:(alpha|beta|rc)({numeric})|-({prerelease}))?",
        version,
    )
    if release:
        base, channel, sequence, canonical = release.groups()
        if channel is not None:
            return f"{base}-{channel}.{sequence}"
        return base if canonical is None else f"{base}-{canonical}"
    nightly = re.fullmatch(r"nightly-(\d{8}|\d{4}-\d{2}-\d{2})", version)
    if nightly:
        date_text = nightly[1].replace("-", "")
        date = datetime.date(
            int(date_text[:4]), int(date_text[4:6]), int(date_text[6:])
        )
        return f"0.0.0-nightly.{date:%Y%m%d}"
    raise ValueError(f"Unsupported Operator version: {version!r}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: chart-version.py <Operator version>")
    try:
        print(chart_version(sys.argv[1]))
    except ValueError as error:
        sys.exit(str(error))
