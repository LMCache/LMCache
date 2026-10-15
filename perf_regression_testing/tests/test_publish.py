# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import json

# First Party
from perf_regression_testing import publish

REPORT = {
    "suite": "weekly",
    "date": "20261015-181909",
    "images": {"vllm_source": "pinned", "mooncake_commit": "3aa0f716", "serving": "x"},
    "runs": [
        {
            "workload": "agentic",
            "deployment": "vllm_native",
            "summary": {"invalid": [], "ttft_p50": 1.6},
            "vs_history": {"metrics": {}, "min_runs": 10},
        },
        {
            "workload": "agentic",
            "deployment": "vllm_mooncake",
            "summary": {"invalid": ["no mooncake image: gh failed"]},
        },
    ],
}


def test_each_run_becomes_one_line_and_lines_accumulate(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    assert publish.append(REPORT, path, "h200-22", "https://bk/12") == 2
    assert publish.append(REPORT, path, "h200-22", None) == 2
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 4
    native, mooncake = lines[0], lines[1]
    assert native["host"] == "h200-22" and native["build_url"] == "https://bk/12"
    assert native["suite"] == "weekly" and native["date"] == "20261015-181909"
    assert native["summary"]["ttft_p50"] == 1.6
    assert native["vs_history"] == {"metrics": {}, "min_runs": 10}
    assert native["images"]["mooncake_commit"] == "3aa0f716"
    assert "serving" not in native["images"]
    # An invalid run is still published, without a history check.
    assert mooncake["summary"]["invalid"] and mooncake["vs_history"] is None
    assert lines[2]["build_url"] is None
