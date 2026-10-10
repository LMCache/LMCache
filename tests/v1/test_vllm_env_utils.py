# SPDX-License-Identifier: Apache-2.0

"""Tests for environment flags used by vLLM integrations."""

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.utils import is_env_var_enabled


@pytest.mark.parametrize(
    "value", ["0", "false", "FALSE", " false ", "no", "off", "", " "]
)
def test_env_var_false_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LMCACHE_TEST_FLAG", value)

    assert is_env_var_enabled("LMCACHE_TEST_FLAG") is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_env_var_true_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LMCACHE_TEST_FLAG", value)

    assert is_env_var_enabled("LMCACHE_TEST_FLAG") is True


def test_unset_env_var_defaults_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LMCACHE_TEST_FLAG", raising=False)

    assert is_env_var_enabled("LMCACHE_TEST_FLAG") is False
