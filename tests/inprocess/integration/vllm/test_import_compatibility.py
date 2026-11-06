# SPDX-License-Identifier: Apache-2.0

# Standard
from importlib import import_module

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
