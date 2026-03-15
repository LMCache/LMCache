# SPDX-License-Identifier: Apache-2.0
"""Tests for lmcache.v1.compute.models.utils."""

# Standard
import sys
import types

# First Party
from lmcache.v1.compute.models import utils


class Qwen3ForCausalLM:
    """Minimal stand-in for Qwen3ForCausalLM."""


class CUDAGraphWrapper:
    """Mimics vLLM's CUDAGraphWrapper."""

    def __init__(self, runnable):
        self.runnable = runnable


def test_infer_model_from_vllm_unwraps_cuda_graph_wrapper(monkeypatch):
    """infer_model_from_vllm should unwrap CUDA graph wrappers and return the adapter."""

    class _FakeAdapter:
        def __init__(self, vllm_model, blender, enable_sparse=False):
            self.vllm_model = vllm_model
            self.blender = blender
            self.enable_sparse = enable_sparse

    fake_qwen3_module = types.ModuleType("lmcache.v1.compute.models.qwen3")
    fake_qwen3_module.LMCQwen3Model = _FakeAdapter
    monkeypatch.setitem(
        sys.modules,
        "lmcache.v1.compute.models.qwen3",
        fake_qwen3_module,
    )

    wrapped = CUDAGraphWrapper(Qwen3ForCausalLM())
    adapter = utils.infer_model_from_vllm(
        wrapped,
        blender=object(),
        enable_sparse=True,
    )

    assert isinstance(adapter, _FakeAdapter)
    assert isinstance(adapter.vllm_model, Qwen3ForCausalLM)
    assert adapter.enable_sparse is True
