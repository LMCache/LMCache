# SPDX-License-Identifier: Apache-2.0

# Standard
from importlib import import_module
from pathlib import Path

# Third Party
import pytest

pytest.importorskip("vllm")


@pytest.mark.parametrize(
    ("legacy_module", "canonical_module", "symbol"),
    [
        (
            "lmcache.integration.vllm.vllm_v1_adapter",
            "lmcache.inprocess.integration.vllm.vllm_v1_adapter",
            "LMCacheConnectorV1Impl",
        ),
        (
            "lmcache.integration.vllm.vllm_service_factory",
            "lmcache.inprocess.integration.vllm.vllm_service_factory",
            "VllmServiceFactory",
        ),
        (
            "lmcache.integration.vllm.vllm_ec_adapter",
            "lmcache.inprocess.integration.vllm.vllm_ec_adapter",
            "LMCacheECConnectorImpl",
        ),
    ],
)
def test_legacy_imports_reexport_canonical_symbols(
    legacy_module: str,
    canonical_module: str,
    symbol: str,
) -> None:
    """Verify legacy vLLM import paths preserve symbol identity."""
    legacy = import_module(legacy_module)
    canonical = import_module(canonical_module)

    assert getattr(legacy, symbol) is getattr(canonical, symbol)


def test_legacy_connector_is_symlink_to_canonical_module() -> None:
    """Verify the legacy connector path points to the canonical module."""
    legacy_path = (
        Path(__file__).parents[4] / "lmcache/integration/vllm/lmcache_connector_v1.py"
    )
    canonical_path = (
        Path(__file__).parents[4]
        / "lmcache/inprocess/integration/vllm/lmcache_connector_v1.py"
    )

    assert legacy_path.is_symlink()
    assert legacy_path.resolve() == canonical_path.resolve()
