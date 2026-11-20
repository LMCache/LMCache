# SPDX-License-Identifier: Apache-2.0
"""Check JSON latency reporting in the long-document QA benchmark."""

# Standard
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
import asyncio
import csv
import importlib.util
import json
import sys

# Third Party
import pytest


@pytest.fixture
def long_doc_qa_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the benchmark without invoking its CLI."""
    path = Path(__file__).resolve().parents[2] / "benchmarks/long_doc_qa/long_doc_qa.py"
    spec = importlib.util.spec_from_file_location("long_doc_qa_stream_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "write_resp", lambda text: None)
    return module


def test_trim_fraction_help_formats_percentages(
    long_doc_qa_module: ModuleType,
) -> None:
    """A literal percent sign must survive argparse's help interpolation."""
    help_text = long_doc_qa_module.create_argument_parser().format_help()
    assert "bottom 10% and top 10%." in " ".join(help_text.split())


@pytest.mark.asyncio
async def test_cli_summary_uses_successful_request_latency(
    long_doc_qa_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The JSON summary uses the same endpoint as the CSV diagnostics."""
    monkeypatch.chdir(tmp_path)
    args = long_doc_qa_module.create_argument_parser().parse_args(
        [
            "--model",
            "test-model",
            "--num-documents",
            "2",
            "--repeat-count",
            "1",
            "--json-output",
        ]
    )

    async def requests(**kwargs: object) -> list[object]:
        return [
            long_doc_qa_module.RequestStats(0, 10.0, 0.2, 12.0, True),
            long_doc_qa_module.RequestStats(1, 10.0, -1, 15.0, False),
        ]

    monkeypatch.setattr(long_doc_qa_module, "test_long_document_qa", requests)
    await long_doc_qa_module.main(args)
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["query_mean_request_latency"] == 2.0
    assert summary["warmup_mean_request_latency"] == 2.0
    assert (tmp_path / "query_round.csv").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("truncated_phase", ["warmup", "query"])
async def test_main_keeps_csvs_when_a_stream_is_truncated(
    long_doc_qa_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    truncated_phase: str,
) -> None:
    """A truncated stream fails the run after both CSVs have been written."""
    monkeypatch.chdir(tmp_path)
    args = long_doc_qa_module.create_argument_parser().parse_args(
        ["--model", "test-model", "--num-documents", "1", "--repeat-count", "1"]
    )
    phases = iter(("prewarmup", "warmup", "query"))

    async def requests(**kwargs: object) -> list[object]:
        truncated = next(phases) == truncated_phase
        return [
            long_doc_qa_module.RequestStats(
                0, 10.0, 0.2, 12.0, not truncated, missing_finish_reason=truncated
            )
        ]

    monkeypatch.setattr(long_doc_qa_module, "test_long_document_qa", requests)
    counts = "warmup=1, query=0" if truncated_phase == "warmup" else "warmup=0, query=1"
    with pytest.raises(RuntimeError, match=counts):
        await long_doc_qa_module.main(args)

    for phase in ("warmup", "query"):
        with (tmp_path / f"{phase}_round.csv").open(newline="") as result:
            rows = list(csv.DictReader(result))
        assert [row["successful"] for row in rows] == [str(phase != truncated_phase)]


@pytest.mark.asyncio
@pytest.mark.parametrize("completions_mode", [False, True])
@pytest.mark.parametrize("finish_reason", [None, "length"])
async def test_stream_completion_is_required_after_content(
    long_doc_qa_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    completions_mode: bool,
    finish_reason: str | None,
) -> None:
    """Usage-only frames do not mark a stream complete."""
    monkeypatch.setattr(long_doc_qa_module, "completions_mode", completions_mode)

    async def create(**kwargs: object) -> AsyncIterator[SimpleNamespace]:
        async def stream() -> AsyncIterator[SimpleNamespace]:
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        text="token",
                        delta=SimpleNamespace(content="token"),
                    )
                ]
            )
            if finish_reason is not None:
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason=finish_reason,
                            text="",
                            delta=SimpleNamespace(content=None),
                        )
                    ]
                )
            yield SimpleNamespace(choices=[])

        return stream()

    completions = SimpleNamespace(create=create)
    client = SimpleNamespace(
        completions=completions, chat=SimpleNamespace(completions=completions)
    )
    result = await long_doc_qa_module.process_single_prompt(
        client, "test-model", "prompt", 0, 1, 1, asyncio.Semaphore(1)
    )
    assert result.successful == (finish_reason is not None)
    assert result.missing_finish_reason == (finish_reason is None)
