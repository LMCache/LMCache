# SPDX-License-Identifier: Apache-2.0
"""Reject vLLM configs that cannot supply reusable multimodal identities."""

# Standard
from types import SimpleNamespace
import importlib

# Third Party
import pytest

pytest.importorskip("vllm", reason="The connectors require vLLM")

# Third Party
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)


class _InitializationStarted(RuntimeError):
    """Stop construction before any services or device resources are created."""


@pytest.fixture(
    params=[
        "lmcache_mp_connector",
        "lmcache_mp_connector_0180",
        "lmcache_mp_connector_0201",
        "vllm_v1_adapter",
    ]
)
def implementation(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=[KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
def role(request: pytest.FixtureRequest) -> KVConnectorRole:
    return request.param


def _make_config(
    mm_cache_gb: float | None, prefix_caching: bool | None
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            multimodal_config=(
                SimpleNamespace(mm_processor_cache_gb=mm_cache_gb)
                if mm_cache_gb is not None
                else None
            )
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        device_config=SimpleNamespace(device="cpu"),
        kv_transfer_config=SimpleNamespace(kv_role="kv_both"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )


def _initialize_connector(
    implementation: str,
    config: SimpleNamespace,
    role: KVConnectorRole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(f"lmcache.integration.vllm.{implementation}")

    def stop_initialization(*args: object, **kwargs: object) -> None:
        raise _InitializationStarted

    if implementation == "vllm_v1_adapter":
        monkeypatch.setattr(module, "print_banner_once", lambda *args: None)
        monkeypatch.setattr(module, "lmcache_get_or_create_config", stop_initialization)
        module.LMCacheConnectorV1Impl(config, role, parent=None)
    else:
        monkeypatch.setattr(module.KVConnectorBase_V1, "__init__", stop_initialization)
        module.LMCacheMPConnector(config, role)


@pytest.mark.parametrize("prefix_caching", [False, None])
def test_rejects_request_scoped_mm_ids_before_initialization(
    implementation: str,
    role: KVConnectorRole,
    prefix_caching: bool | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(mm_cache_gb=0, prefix_caching=prefix_caching)

    with pytest.raises(ValueError, match="--mm-processor-cache-gb"):
        _initialize_connector(implementation, config, role, monkeypatch)


@pytest.mark.parametrize(
    "mm_cache_gb,prefix_caching",
    [(4.0, True), (4.0, False), (0.0, True), (None, False)],
    ids=["defaults", "processor-cache-only", "prefix-cache-only", "text-model"],
)
def test_allows_configs_with_stable_mm_ids_or_text_models(
    implementation: str,
    role: KVConnectorRole,
    mm_cache_gb: float | None,
    prefix_caching: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(mm_cache_gb, prefix_caching)

    with pytest.raises(_InitializationStarted):
        _initialize_connector(implementation, config, role, monkeypatch)


def test_allows_model_config_without_multimodal_config(
    implementation: str,
    role: KVConnectorRole,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(mm_cache_gb=None, prefix_caching=False)
    config.model_config = SimpleNamespace()

    with pytest.raises(_InitializationStarted):
        _initialize_connector(implementation, config, role, monkeypatch)
