#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install CPU nightly URL dependencies and reject stale nightly builds.

The producer stores URL requirements and provenance in dist-info/cpu-nightly.json.
Older pinned wheels have no manifest; their .devYYYYMMDDHHMM version records
the UTC build time. Freshness checks run in nightly verification only.
"""

# Standard
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import argparse
import importlib.metadata as md
import json
import os
import re
import shlex
import subprocess

# Third Party
from packaging.requirements import Requirement


@dataclass(frozen=True)
class NightlyBuild:
    """Provenance and URL dependencies embedded by the CPU wheel producer."""

    source_sha: str
    built_at: datetime
    direct_requirements: list[str]


def read_build(dist: md.Distribution) -> NightlyBuild:
    """Read build provenance, falling back to the legacy version timestamp.

    Args:
        dist: Installed vllm-cpu-nightly distribution.

    Returns:
        Build revision, timezone-aware build time and URL requirements.

    Raises:
        ValueError: Build provenance is missing or malformed.
    """
    manifest = dist.read_text("cpu-nightly.json")
    if manifest is None:
        stamp = re.search(r"\.dev(\d{12})(?:\+.*)?$", dist.version)
        if stamp is None:
            raise ValueError("CPU nightly has no build manifest or timestamp")
        return NightlyBuild(
            "",
            datetime.strptime(stamp[1], "%Y%m%d%H%M").replace(tzinfo=timezone.utc),
            [],
        )
    data = json.loads(manifest)
    source_sha = data["source_sha"]
    built_at = datetime.fromisoformat(data["built_at"].replace("Z", "+00:00"))
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None or built_at.tzinfo is None:
        raise ValueError("CPU nightly build manifest has invalid provenance")
    return NightlyBuild(source_sha, built_at, data["direct_requirements"])


def check_freshness(build: NightlyBuild, now: datetime) -> None:
    """Require a build no older than 48 hours, allowing 15 minutes of clock skew.

    Args:
        build: Installed wheel provenance.
        now: Current timezone-aware UTC time.

    Raises:
        ValueError: The build is stale or its timestamp is in the future.
    """
    age = now - build.built_at
    if age > timedelta(hours=48) or age < -timedelta(minutes=15):
        raise ValueError(
            "CPU nightly is stale or future-dated: "
            f"built_at={build.built_at.isoformat()}, "
            f"age_hours={age.total_seconds() / 3600:.1f}; expected at most 48 hours. "
            "Check https://github.com/LMCache/vllm-cpu/actions/workflows/nightly.yml"
        )


def install_dependencies(build: NightlyBuild) -> None:
    """Install URL requirements applicable to the current runtime.

    Args:
        build: Wheel manifest whose markers select platform and Python ABI.

    Raises:
        subprocess.CalledProcessError: An applicable dependency cannot install.
    """
    applicable = []
    for requirement in build.direct_requirements:
        parsed = Requirement(requirement)
        if parsed.marker is None or parsed.marker.evaluate():
            applicable.append(requirement)
    if applicable:
        subprocess.run(
            [*shlex.split(os.environ.get("PIP_BIN", "pip")), "install", *applicable],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["install-dependencies", "check-freshness"])
    action = parser.parse_args().action
    build = read_build(md.distribution("vllm-cpu-nightly"))
    if action == "install-dependencies":
        install_dependencies(build)
    else:
        check_freshness(build, datetime.now(timezone.utc))
        print(
            f"CPU nightly is fresh: built_at={build.built_at.isoformat()}, "
            f"sha={build.source_sha}"
        )
