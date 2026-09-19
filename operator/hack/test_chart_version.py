# SPDX-License-Identifier: Apache-2.0
"""Test the chart versions published for supported Operator release tags."""

# Standard
from pathlib import Path
import importlib.util
import unittest

SPEC = importlib.util.spec_from_file_location(
    "chart_version", Path(__file__).with_name("chart-version.py")
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ChartVersionTest(unittest.TestCase):
    """Check the public release-to-chart version contract."""

    def test_supported_releases(self) -> None:
        """Stable, prerelease and nightly tags produce valid chart versions."""
        for source, expected in {
            "v0.5.5": "0.5.5",
            "1.0.0": "1.0.0",
            "v0.4.8rc1": "0.4.8-rc.1",
            "v0.5.6alpha1": "0.5.6-alpha.1",
            "v0.5.6beta2": "0.5.6-beta.2",
            "v0.5.6-rc.1": "0.5.6-rc.1",
            "v0.5.6-alpha.1": "0.5.6-alpha.1",
            "0.5.6-preview.1": "0.5.6-preview.1",
            "v0.5.6-0.3.7": "0.5.6-0.3.7",
            "v0.5.6-x.7.z.92": "0.5.6-x.7.z.92",
            "nightly-2026-09-19": "0.0.0-nightly.20260919",
            "nightly-20260919": "0.0.0-nightly.20260919",
        }.items():
            with self.subTest(version=source):
                self.assertEqual(MODULE.chart_version(source), expected)

    def test_invalid_releases(self) -> None:
        """Unsupported tags and impossible dates fail before publishing a package."""
        for version in (
            "latest",
            "nightly",
            "v01.2.3",
            "v0.5.6rc01",
            "v0.5.6alpha01",
            "v0.5.6-rc.01",
            "v0.5.6-rc..1",
            "v0.5.6-",
            "v0.5.6+build.1",
            "nightly-2026-02-30",
            "nightly-2026-0919",
        ):
            with self.subTest(version=version), self.assertRaises(ValueError):
                MODULE.chart_version(version)
