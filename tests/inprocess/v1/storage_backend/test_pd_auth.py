# SPDX-License-Identifier: Apache-2.0

"""Tests for the PD shared-token helpers.

Kept free of heavy imports (no torch/models) so the token plumbing can be
checked without a built compute backend.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.inprocess.v1.storage_backend.pd_auth import (
    PD_AUTH_TOKEN_CONFIG_KEY,
    PD_AUTH_TOKEN_ENV_VAR,
    get_pd_auth_token,
    pd_auth_token_matches,
)


class TestGetPdAuthToken:
    def test_absent_everywhere_disables_auth(self, monkeypatch):
        monkeypatch.delenv(PD_AUTH_TOKEN_ENV_VAR, raising=False)
        config = MagicMock()
        config.extra_config = {}
        assert get_pd_auth_token(config) is None

    def test_reads_environment(self, monkeypatch):
        monkeypatch.setenv(PD_AUTH_TOKEN_ENV_VAR, "from-env")
        assert get_pd_auth_token(MagicMock(extra_config={})) == b"from-env"

    def test_config_overrides_environment(self, monkeypatch):
        monkeypatch.setenv(PD_AUTH_TOKEN_ENV_VAR, "from-env")
        config = MagicMock(extra_config={PD_AUTH_TOKEN_CONFIG_KEY: "from-config"})
        assert get_pd_auth_token(config) == b"from-config"

    def test_empty_value_disables_auth(self, monkeypatch):
        monkeypatch.delenv(PD_AUTH_TOKEN_ENV_VAR, raising=False)
        config = MagicMock(extra_config={PD_AUTH_TOKEN_CONFIG_KEY: ""})
        assert get_pd_auth_token(config) is None

    def test_missing_extra_config_attribute_disables_auth(self, monkeypatch):
        monkeypatch.delenv(PD_AUTH_TOKEN_ENV_VAR, raising=False)
        assert get_pd_auth_token(object()) is None


class TestPdAuthTokenMatches:
    def test_disabled_accepts_any_token(self):
        assert pd_auth_token_matches(None, b"")
        assert pd_auth_token_matches(None, b"whatever")

    def test_matching_token(self):
        assert pd_auth_token_matches(b"secret", b"secret")

    def test_mismatched_token(self):
        assert not pd_auth_token_matches(b"secret", b"other")

    def test_missing_token_is_rejected_when_enabled(self):
        assert not pd_auth_token_matches(b"secret", b"")

    def test_prefix_is_rejected(self):
        assert not pd_auth_token_matches(b"secret", b"sec")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
