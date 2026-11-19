# SPDX-License-Identifier: Apache-2.0
"""Check JSON latency reporting in the long-document QA benchmark."""

# Standard
from pathlib import Path
from types import ModuleType
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
