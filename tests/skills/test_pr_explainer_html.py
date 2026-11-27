# SPDX-License-Identifier: Apache-2.0
"""Check the reviewer-guide starter and the author's offline browser check."""

# Standard
from pathlib import Path
import subprocess
import sys

# Third Party
import pytest

# First Party
from tests.import_utils import repo_path


@pytest.mark.parametrize(
    ("edit", "diagnostic"),
    [
        (None, None),
        (("</body>", '<img src="neighbor.svg" alt=""></body>'), "external asset"),
        (
            (
                "</body>",
                "<noscript><style>[data-view]{display:none}</style></noscript></body>",
            ),
            "static outcome is hidden",
        ),
        ((" data-view=", " data-unused="), "no panel for full"),
    ],
    ids=["standalone", "local-dependency", "hidden-fallback", "disconnected-controls"],
)
def test_copied_reviewer_guide(
    tmp_path: Path, edit: tuple[str, str] | None, diagnostic: str | None
) -> None:
    """The author can copy the shared starter and catch broken offline behavior."""
    skill = repo_path(".agents", "skills", "create-pr-explainer")
    starter = skill / "assets" / "guide.html"
    assert starter.samefile(
        repo_path(".claude", "skills", "create-pr-explainer", "assets", "guide.html")
    )
    pytest.importorskip("playwright.sync_api")
    guide = tmp_path / "review guide.html"
    source = starter.read_text(encoding="utf-8")
    if edit is not None:
        source = source.replace(*edit)
    guide.write_text(source, encoding="utf-8")
    (tmp_path / "neighbor.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8"
    )
    run = subprocess.run(
        [sys.executable, str(skill / "scripts" / "check_guide.py"), str(guide)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if "Executable doesn't exist" in run.stderr:
        pytest.skip("Install Playwright Chromium to run the authoring checks")
    if diagnostic is None:
        assert run.returncode == 0, run.stderr
    else:
        assert run.returncode != 0
        assert diagnostic in run.stderr, run.stderr
