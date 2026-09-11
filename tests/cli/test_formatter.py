# SPDX-License-Identifier: Apache-2.0
"""Tests for the CLI metrics formatters, focused on ``CsvFormatter``."""

# Standard
import csv
import io

# Third Party
import pytest

# First Party
from lmcache.cli.metrics.formatter import (
    CsvFormatter,
    get_formatter,
)
from lmcache.cli.metrics.section import Section


def _flat_section(entries: list[tuple[str, str, object]]) -> Section:
    """Build the unnamed top-level section holding *entries*."""
    section = Section(None, None)
    for key, label, value in entries:
        section.add(key, label, value)
    return section


def _parse(text: str) -> list[list[str]]:
    """Parse formatter CSV output back into rows."""
    return list(csv.reader(io.StringIO(text)))


class TestCsvFormatterRegistration:
    def test_lookup_returns_csv_formatter(self):
        assert isinstance(get_formatter("csv"), CsvFormatter)

    def test_ignores_unrelated_kwargs(self):
        # ``BaseCommand.create_metrics`` forwards ``width``; the CSV
        # formatter must accept being constructed without using it.
        assert isinstance(get_formatter("csv", width=80), CsvFormatter)


class TestCsvFormatterOutput:
    def test_header_is_always_first_row(self):
        rows = _parse(CsvFormatter().format("Title", []))
        assert rows == [["section", "index", "key", "label", "value"]]

    def test_flat_metrics_have_empty_section_and_index(self):
        sections = [_flat_section([("health", "Health", "Healthy")])]
        rows = _parse(CsvFormatter().format("Describe", sections))
        assert rows[1] == ["", "", "health", "Health", "Healthy"]

    def test_named_section_uses_key_and_empty_index(self):
        section = Section("limits", "Limits")
        section.add("limit_gb", "Limit (GB)", 40)
        rows = _parse(CsvFormatter().format("Quota", [section]))
        assert rows[1] == ["limits", "", "limit_gb", "Limit (GB)", "40"]

    def test_list_group_sections_get_incrementing_indices(self):
        model0 = Section("model_0", "Model 0", list_group="models")
        model0.add("model", "Model", "llama")
        model1 = Section("model_1", "Model 1", list_group="models")
        model1.add("model", "Model", "mistral")
        rows = _parse(CsvFormatter().format("Describe", [model0, model1]))
        assert rows[1] == ["models", "0", "model", "Model", "llama"]
        assert rows[2] == ["models", "1", "model", "Model", "mistral"]

    def test_none_value_renders_as_empty_field(self):
        sections = [_flat_section([("l1_capacity_gb", "L1 capacity (GB)", None)])]
        rows = _parse(CsvFormatter().format("Describe", sections))
        assert rows[1] == ["", "", "l1_capacity_gb", "L1 capacity (GB)", ""]

    def test_value_with_comma_is_quoted_and_round_trips(self):
        sections = [_flat_section([("gpu_ids", "GPU IDs", "0, 1")])]
        text = CsvFormatter().format("Describe", sections)
        # Raw output must quote the comma so the cell stays intact...
        assert '"0, 1"' in text
        # ...and parsing back must recover the original value unchanged.
        assert _parse(text)[1][4] == "0, 1"

    def test_no_trailing_newline(self):
        sections = [_flat_section([("status", "Status", "OK")])]
        assert not CsvFormatter().format("Describe", sections).endswith("\n")


def test_unknown_format_raises_value_error():
    with pytest.raises(ValueError, match="csv"):
        get_formatter("does-not-exist")
