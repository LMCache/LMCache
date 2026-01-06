# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl


def _empty_generator():
    if False:
        yield None


def _one_step_generator():
    yield None


def test_finalize_layerwise_storers_handles_exhausted():
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.layerwise_storers = [_empty_generator()]

    connector._finalize_layerwise_storers("test")


def test_finalize_layerwise_storers_advances_generator():
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    gen = _one_step_generator()
    connector.layerwise_storers = [gen]

    connector._finalize_layerwise_storers("test")

    with pytest.raises(StopIteration):
        next(gen)
