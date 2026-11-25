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
    ("addition", "diagnostic"),
    [
        ("", None),
        ('<img src="neighbor.svg" alt="">', "external asset"),
        (
            "<noscript><style>[data-view] { display: none }</style></noscript>",
            "static outcome is hidden",
        ),
    ],
    ids=["standalone", "local-dependency", "hidden-fallback"],
)
def test_copied_reviewer_guide(
    tmp_path: Path, addition: str, diagnostic: str | None
) -> None:
    """The author can copy the shared starter and catch broken offline behavior."""
    skill = repo_path(".agents", "skills", "create-pr-explainer")
    starter = skill / "assets" / "guide.html"
    assert starter.samefile(
        repo_path(".claude", "skills", "create-pr-explainer", "assets", "guide.html")
    )
    pytest.importorskip("playwright.sync_api")
    guide = tmp_path / "review guide.html"
    guide.write_text(
        starter.read_text(encoding="utf-8").replace("</body>", addition + "</body>"),
        encoding="utf-8",
    )
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
