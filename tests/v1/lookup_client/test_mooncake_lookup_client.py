# SPDX-License-Identifier: Apache-2.0
"""
Tests for MooncakeLookupClient key construction.

Uses a fake ``mooncake.store`` module, so these tests do not require Mooncake.
"""

# Standard
from typing import Any, Optional, cast
import sys
import types

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.mooncake_lookup_client import MooncakeLookupClient
from lmcache.v1.token_database import ChunkedTokenDatabase
from tests.v1.utils import create_test_config, create_test_metadata, generate_tokens

TENANT_A = {"lmcache.tag.tenant": "tenant-a"}
TENANT_B = {"lmcache.tag.tenant": "tenant-b"}


def _install_fake_mooncake_store(
    monkeypatch: pytest.MonkeyPatch,
    stored_keys: set[str],
) -> None:
    """Install a fake mooncake.store whose store holds ``stored_keys``."""

    class FakeMooncakeDistributedStore:
        def setup(self, config: dict[str, str]) -> None:
            pass

        def batch_is_exist(self, keys: list[str]) -> list[int]:
            return [1 if key in stored_keys else 0 for key in keys]

    fake_module = types.ModuleType("mooncake.store")
    cast(Any, fake_module).MooncakeDistributedStore = FakeMooncakeDistributedStore
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", fake_module)


def _stored_keys(
    config: LMCacheEngineConfig,
    tokens: torch.Tensor,
    request_configs: Optional[dict],
) -> set[str]:
    """Key strings the Mooncake backend stores for ``tokens``."""
    token_database = ChunkedTokenDatabase(config, create_test_metadata())
    keys = set()
    for _, _, key in token_database.process_tokens(
        tokens, request_configs=request_configs
    ):
        assert isinstance(key, CacheEngineKey)
        keys.add(key.to_string())
    return keys


class TestMooncakeLookupClientRequestTags:
    """Lookups must build the same tagged keys the Mooncake backend stores."""

    @pytest.mark.parametrize("request_configs", [None, TENANT_A])
    def test_lookup_hits_entries_stored_with_same_tags(
        self, monkeypatch: pytest.MonkeyPatch, request_configs: Optional[dict]
    ) -> None:
        config = create_test_config()
        tokens = generate_tokens(512, "cpu")
        _install_fake_mooncake_store(
            monkeypatch, _stored_keys(config, tokens, request_configs)
        )
        client = MooncakeLookupClient(config, create_test_metadata(), "localhost:50051")

        assert client.lookup(tokens, request_configs=request_configs) == 512

    @pytest.mark.parametrize(
        "stored_configs,lookup_configs",
        [(None, TENANT_A), (TENANT_A, None), (TENANT_A, TENANT_B)],
    )
    def test_lookup_misses_entries_stored_with_other_tags(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stored_configs: Optional[dict],
        lookup_configs: Optional[dict],
    ) -> None:
        config = create_test_config()
        tokens = generate_tokens(512, "cpu")
        _install_fake_mooncake_store(
            monkeypatch, _stored_keys(config, tokens, stored_configs)
        )
        client = MooncakeLookupClient(config, create_test_metadata(), "localhost:50051")

        assert client.lookup(tokens, request_configs=lookup_configs) == 0
