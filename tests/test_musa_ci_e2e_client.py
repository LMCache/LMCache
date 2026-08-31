# SPDX-License-Identifier: Apache-2.0
"""Tests for the dependency-free MUSA CI E2E client."""

# Standard
from pathlib import Path
from types import ModuleType
from typing import Any
import importlib.util
import json
import sys

# Third Party
import pytest


def _load_client() -> ModuleType:
    client_path = (
        Path(__file__).parents[1] / ".buildkite" / "k3_tests" / "musa" / "e2e_client.py"
    )
    spec = importlib.util.spec_from_file_location("musa_e2e_client", client_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_client(
    monkeypatch: pytest.MonkeyPatch,
    client: ModuleType,
    arguments: list[str],
    response: dict[str, Any],
) -> int:
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: response)
    monkeypatch.setattr(sys, "argv", ["e2e_client.py", *arguments])
    return client.main()


def test_completion_normalizes_openai_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Completion output retains text, finish reason, usage, and latency."""
    client = _load_client()
    prompt = tmp_path / "prompt.txt"
    output = tmp_path / "completion.json"
    prompt.write_text("deterministic prompt")
    response = {
        "id": "request-1",
        "choices": [{"text": "stable answer", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    assert (
        _run_client(
            monkeypatch,
            client,
            [
                "completion",
                "--url",
                "http://127.0.0.1/v1/completions",
                "--model",
                "musa-e2e",
                "--prompt-file",
                str(prompt),
                "--max-tokens",
                "8",
                "--seed",
                "0",
                "--temperature",
                "0",
                "--top-k",
                "1",
                "--output",
                str(output),
            ],
            response,
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["text"] == "stable answer"
    assert result["finish_reason"] == "stop"
    assert result["usage"] == response["usage"]
    assert result["elapsed_seconds"] >= 0


def test_chat_completion_normalizes_openai_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Chat output is normalized to the same comparison schema."""
    client = _load_client()
    prompt = tmp_path / "prompt.txt"
    output = tmp_path / "chat.json"
    prompt.write_text("deterministic prompt")
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "stable answer"},
                "finish_reason": "stop",
            }
        ]
    }
    assert (
        _run_client(
            monkeypatch,
            client,
            [
                "chat-completion",
                "--url",
                "http://127.0.0.1/v1/chat/completions",
                "--model",
                "musa-e2e",
                "--prompt-file",
                str(prompt),
                "--max-tokens",
                "8",
                "--seed",
                "0",
                "--temperature",
                "0",
                "--top-k",
                "1",
                "--output",
                str(output),
            ],
            response,
        )
        == 0
    )
    assert json.loads(output.read_text())["text"] == "stable answer"


def test_compare_rejects_different_completion_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Baseline and cache results with different text fail comparison."""
    client = _load_client()
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps({"text": "left"}))
    right.write_text(json.dumps({"text": "right"}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "e2e_client.py",
            "compare",
            "--left",
            str(left),
            "--right",
            str(right),
        ],
    )
    assert client.main() == 1
