# SPDX-License-Identifier: Apache-2.0
"""The connector's outbound credential, `lmcache.mp.api_key_file`.

An engine that only reads and writes cache entries should not have to hold the
fleet key: it can be given the LMCache server's key alone. This setting names
the file holding that key, and the adapters pass it to every request client.
"""

# Standard
from pathlib import Path
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    ParallelStrategy,
)

_KEY = "server-side-key\n"


def _key_file(tmp_path: Path, value: str = _KEY) -> str:
    path = tmp_path / "server-key"
    path.write_text(value, encoding="utf-8")
    return str(path)


class _RecordingFactory:
    """Records the credential each client creation receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def create(self, url: str, *, context: object = None, api_key: str | None = None):
        self.calls.append((url, api_key))
        client = MagicMock()
        # The adapters await a future from the server-size query, not a value.
        future = MagicMock()
        future.result.return_value = 16
        client.get_chunk_size.return_value = future
        return client


class TestConfigResolution:
    def test_absent_setting_keeps_the_environment_fallback(self) -> None:
        assert adapter_mod._connector_request_api_key(None) is None
        assert adapter_mod._connector_request_api_key({}) is None

    def test_a_configured_file_is_read_and_stripped(self, tmp_path: Path) -> None:
        path = _key_file(tmp_path)
        assert (
            adapter_mod._connector_request_api_key({"lmcache.mp.api_key_file": path})
            == "server-side-key"
        )

    def test_a_non_string_path_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty path string"):
            adapter_mod._connector_request_api_key({"lmcache.mp.api_key_file": 3})

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        path = _key_file(tmp_path, "   \n")
        with pytest.raises(ValueError, match="empty"):
            adapter_mod._connector_request_api_key({"lmcache.mp.api_key_file": path})

    def test_a_multi_line_file_is_refused(self, tmp_path: Path) -> None:
        path = _key_file(tmp_path, "one\ntwo\n")
        with pytest.raises(ValueError, match="exactly one line"):
            adapter_mod._connector_request_api_key({"lmcache.mp.api_key_file": path})

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            adapter_mod._connector_request_api_key(
                {"lmcache.mp.api_key_file": str(tmp_path / "nope")}
            )


class TestAdaptersUseIt:
    """The setting has to reach the clients, or it is decoration."""

    def test_scheduler_clients_receive_the_configured_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = _RecordingFactory()
        monkeypatch.setattr(adapter_mod, "RequestClientFactory", factory)
        path = _key_file(tmp_path)

        LMCacheMPSchedulerAdapter(
            ["tcp://one:5555", "tcp://two:5555"],
            context=object(),
            model_name="m",
            vllm_block_size=16,
            parallel_strategy=ParallelStrategy(1, 0, 1, 1, 1),
            extra_config={"lmcache.mp.api_key_file": path},
        )

        assert factory.calls == [
            ("tcp://one:5555", "server-side-key"),
            ("tcp://two:5555", "server-side-key"),
        ]

    def test_scheduler_clients_fall_back_without_the_setting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = _RecordingFactory()
        monkeypatch.setattr(adapter_mod, "RequestClientFactory", factory)

        LMCacheMPSchedulerAdapter(
            ["tcp://one:5555"],
            context=object(),
            model_name="m",
            vllm_block_size=16,
            parallel_strategy=ParallelStrategy(1, 0, 1, 1, 1),
        )

        assert factory.calls == [("tcp://one:5555", None)]

    def test_worker_client_receives_the_configured_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = _RecordingFactory()
        monkeypatch.setattr(adapter_mod, "RequestClientFactory", factory)
        path = _key_file(tmp_path)

        LMCacheMPWorkerAdapter(
            server_url="tcp://one:5555",
            context=object(),
            model_name="m",
            vllm_block_size=16,
            parallel_strategy=ParallelStrategy(1, 0, 1, 1, 1),
            extra_config={"lmcache.mp.api_key_file": path},
        )

        assert factory.calls == [("tcp://one:5555", "server-side-key")]
