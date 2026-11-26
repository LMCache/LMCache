# SPDX-License-Identifier: Apache-2.0
"""KV cache dtype in the vLLM MP connector's cache identity.

Two live engines can serve one model with different KV cache dtypes. The server
learns the dtype from the last registration, so the identity has to come from
the request: the connector adds ``lmcache.tag.kv_dtype`` next to every other
identity tag. These tests pin which string each engine sends and that the tag
reaches the request configuration. No GPU needed.

The modules that import vLLM are imported lazily so the k3 unit environment,
which has no vLLM, still collects this file.
"""

# Standard
from types import SimpleNamespace
import importlib.util

# Third Party
import pytest

requires_vllm = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None, reason="requires vLLM"
)

_TAG_KEY = "lmcache.tag.kv_dtype"


def _config(*, cache_dtype: object = "auto", dtype: object = "torch.bfloat16"):
    """Build the vLLM config subset the dtype helper reads."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        model_config=SimpleNamespace(dtype=dtype),
    )


def _dtype_tag(config: SimpleNamespace) -> str:
    """Resolve the tag through the real helper."""
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import _kv_dtype_tag

    return _kv_dtype_tag(config)  # type: ignore[arg-type]


def _apply(tag: str, configs: dict[str, object] | None) -> dict[str, object] | None:
    """Run the connector's tag injection on a minimal tracker stub."""
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

    connector = SimpleNamespace(_kv_dtype_tag=tag)
    tracker = SimpleNamespace(request_id="req", request_configs=configs)
    LMCacheMPConnector._apply_kv_dtype_tag(connector, tracker)  # type: ignore[arg-type]
    return tracker.request_configs


@requires_vllm
def test_auto_uses_the_model_dtype():
    """With ``auto`` the cache is allocated as the model dtype."""
    assert _dtype_tag(_config(dtype="torch.bfloat16")) == "bf16"
    assert _dtype_tag(_config(dtype="torch.float16")) == "f16"


@requires_vllm
def test_an_explicit_cache_dtype_gets_its_identity_code():
    """The per-token-head modes need distinct codes that still fit a name."""
    assert _dtype_tag(_config(cache_dtype="fp8_per_token_head")) == "f8pth"
    assert _dtype_tag(_config(cache_dtype="int8_per_token_head")) == "i8pth"
    assert _dtype_tag(_config(cache_dtype="fp8_e4m3")) == "fp8_e4m3"


@requires_vllm
def test_an_unknown_model_dtype_is_not_dropped():
    """An unknown dtype keeps its name so two engines cannot collide."""
    assert _dtype_tag(_config(dtype="torch.float8_e4m3fn")) == "f8e4m3"
    assert _dtype_tag(_config(dtype="custom_dtype")) == "custom_dtype"


@requires_vllm
def test_the_tag_reaches_the_request_configuration():
    configs = _apply("bf16", {"lmcache.hint.engine_computed_tokens": 4})
    assert configs == {"lmcache.hint.engine_computed_tokens": 4, _TAG_KEY: "bf16"}


@requires_vllm
def test_an_empty_dtype_leaves_the_request_alone():
    """An engine that cannot report a dtype must not tag objects."""
    assert _apply("", {"a": 1}) == {"a": 1}


@requires_vllm
def test_a_conflicting_caller_tag_is_overridden():
    """The serving engine's dtype wins over one the caller invented."""
    configs = _apply("bf16", {_TAG_KEY: "f8pth"})
    assert configs == {_TAG_KEY: "bf16"}


@requires_vllm
def test_a_matching_caller_tag_is_kept_without_noise():
    configs = _apply("bf16", {_TAG_KEY: "bf16"})
    assert configs == {_TAG_KEY: "bf16"}
