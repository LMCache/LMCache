# SPDX-License-Identifier: Apache-2.0
"""Contract of the per-request load-vs-recompute policy module."""

# Standard
from collections.abc import Mapping
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.kv_load_policy import (
    DefaultKVLoadPolicy,
    KVLoadContext,
    KVLoadPolicy,
    create_kv_load_policy,
)


class ShortPromptsRecompute(KVLoadPolicy):
    """Serves only lookups covering at least ``min_tokens`` tokens."""

    def __init__(self, configs: Mapping[str, Any]) -> None:
        super().__init__(configs)
        self.min_tokens = int(configs.get("test.min_tokens", 0))

    def should_load(self, ctx: KVLoadContext) -> bool:
        return ctx.num_lookup_tokens >= self.min_tokens


class NotAPolicy:
    pass


def _ctx(
    request_configs: Mapping[str, Any] | None = None, num_lookup_tokens: int = 512
) -> KVLoadContext:
    return KVLoadContext(
        request_id="req",
        model_name="m",
        chunk_size=256,
        num_lookup_tokens=num_lookup_tokens,
        engine_computed_tokens=None,
        request_configs=request_configs,
    )


@pytest.mark.parametrize(
    "request_configs, expected",
    [
        (None, True),
        ({}, True),
        ({"lmcache.skip_save": True}, True),
        ({"lmcache.skip_load": False}, True),
        ({"lmcache.skip_load": True}, False),
    ],
)
def test_default_policy_honors_skip_load(
    request_configs: Mapping[str, Any] | None, expected: bool
) -> None:
    """The default policy serves unless the request sets lmcache.skip_load."""
    assert DefaultKVLoadPolicy({}).should_load(_ctx(request_configs)) is expected


def test_factory_builds_default_policy() -> None:
    """The built-in name builds the default policy."""
    assert isinstance(create_kv_load_policy("DEFAULT", {}), DefaultKVLoadPolicy)


def test_factory_loads_custom_policy_with_configs() -> None:
    """A module:ClassName path builds that class with the given configs."""
    policy = create_kv_load_policy(
        f"{__name__}:ShortPromptsRecompute", {"test.min_tokens": 1024}
    )

    assert isinstance(policy, ShortPromptsRecompute)
    assert not policy.should_load(_ctx(num_lookup_tokens=512))
    assert policy.should_load(_ctx(num_lookup_tokens=1024))


@pytest.mark.parametrize("name", ["UNKNOWN", "no_colon.module", ":Cls", "mod:"])
def test_factory_rejects_malformed_names(name: str) -> None:
    """Names that are neither built-in nor module:ClassName are rejected."""
    with pytest.raises(ValueError, match="Unknown"):
        create_kv_load_policy(name, {})


def test_factory_rejects_non_policy_class() -> None:
    """A path resolving to something other than a KVLoadPolicy is rejected."""
    with pytest.raises(ValueError, match="not a KVLoadPolicy"):
        create_kv_load_policy(f"{__name__}:NotAPolicy", {})


def test_factory_propagates_missing_module() -> None:
    """An unimportable module surfaces as ImportError."""
    with pytest.raises(ImportError):
        create_kv_load_policy("no_such_pkg_xyz.mod:Cls", {})
