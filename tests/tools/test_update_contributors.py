# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/update_contributors.py. No network access is needed."""

# Standard
from pathlib import Path
from types import ModuleType
import importlib.util

# Third Party
import pytest

TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "update_contributors.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("update_contributors", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _table(*rows: str) -> str:
    return "\n".join(
        [
            "# Acknowledgments",
            "",
            tool.TABLE_START,
            "| GitHub handle | Name | Affiliation | Last updated |",
            "| --- | --- | --- | --- |",
            *rows,
            tool.TABLE_END,
            "",
        ]
    )


class TestParseAcknowledgments:
    def test_reads_rows_between_markers(self):
        entries = tool.parse_acknowledgments(
            _table(
                "| @Alice | Alice A. | Example Lab | 2026-11-08 |",
                "| `@bob` |  |  | 2026-11-08 |",
            )
        )
        assert entries["alice"] == tool.Acknowledgment(
            "Alice", "Alice A.", "Example Lab"
        )
        assert entries["bob"] == tool.Acknowledgment("bob", "", "")

    def test_ignores_tables_outside_markers(self):
        text = "| @outside | X | Y | 2026-11-08 |\n" + _table()
        assert tool.parse_acknowledgments(text) == {}

    def test_rejects_duplicate_handles_case_insensitively(self):
        with pytest.raises(ValueError, match="duplicate"):
            tool.parse_acknowledgments(_table("| @alice | | | |", "| @ALICE | | | |"))

    def test_rejects_invalid_handle(self):
        with pytest.raises(ValueError, match="handle"):
            tool.parse_acknowledgments(_table("| @not a handle | | | |"))

    def test_requires_markers(self):
        with pytest.raises(ValueError, match="markers"):
            tool.parse_acknowledgments("# Acknowledgments\n")


class TestIsBot:
    @pytest.mark.parametrize(
        "login,account_type,expected",
        [
            ("dependabot[bot]", "Bot", True),
            ("github-actions[bot]", "User", True),
            ("some-app", "Bot", True),
            ("quaid", "User", False),
        ],
    )
    def test_classifies_accounts(self, login, account_type, expected):
        assert tool.is_bot(login, account_type) is expected


class TestRenderContributors:
    def test_sorted_and_deduplicated(self):
        text = tool.render_contributors(["zed", "Alice", "alice", "bob"])
        listed = [line for line in text.splitlines() if line.startswith("- [@")]
        assert listed == [
            "- [@Alice](https://github.com/Alice)",
            "- [@bob](https://github.com/bob)",
            "- [@zed](https://github.com/zed)",
        ]


class TestRenderThanks:
    def test_one_line_per_person_in_handle_order(self):
        entries = tool.parse_acknowledgments(
            _table(
                "| @alice | Alice A. | Example Lab | 2026-11-08 |",
                "| @carol |  | Example Lab & Other Lab | 2026-11-08 |",
                "| @dave | Dave D. |  | 2026-11-08 |",
            )
        )
        text = tool.render_thanks(["dave", "erin", "carol", "alice", "alice"], entries)
        listed = [line for line in text.splitlines() if line.startswith("- @")]
        assert listed == [
            "- @alice (Alice A., Example Lab)",
            "- @carol (Example Lab & Other Lab)",
            "- @dave (Dave D.)",
            "- @erin",
        ]
        assert "the 4 people who contributed" in text

    def test_no_entries(self):
        text = tool.render_thanks(["solo"], {})
        assert "the 1 person who contributed" in text
        assert "- @solo" in text
