# SPDX-License-Identifier: Apache-2.0
"""Reproduce issue #2639 (CacheBlend fails with vLLM graph mode).

Usage:
  python scripts/repro_issue_2639.py --mode minimal
  python scripts/repro_issue_2639.py --mode fixed
  python scripts/repro_issue_2639.py --mode real --model /path/to/Qwen3-8B

Modes:
  minimal: reproduces the pre-fix failure contract without requiring vLLM.
  fixed: validates the current LMCache code path without requiring vLLM.
  real: starts the real vLLM + LMCache initialization path. Requires vLLM.
"""

# Standard
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import types
from dataclasses import asdict
from typing import Any

# Make the repo root importable when this script is executed as a file.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# First Party
from lmcache.v1.compute.models.utils import infer_model_from_vllm


class Qwen3ForCausalLM:
    """Fake model with the same class name used by LMCache dispatch."""


class CUDAGraphWrapper:
    """Fake wrapper matching the vLLM wrapper name/shape."""

    def __init__(self, runnable: Any) -> None:
        self.runnable = runnable


def _old_infer_model_from_vllm(vllm_model: Any) -> str:
    """Pre-fix logic from LMCache to demonstrate the original failure."""
    model_name = type(vllm_model).__name__
    if model_name == "LlamaForCausalLM":
        return "llama"
    if model_name == "Qwen2ForCausalLM":
        return "qwen2"
    if model_name == "Qwen3ForCausalLM":
        return "qwen3"
    raise NotImplementedError(f"Model type {model_name} is not supported in LMCache.")


def run_minimal_repro() -> int:
    """Show the exact old failure without requiring vLLM."""
    wrapped_model = CUDAGraphWrapper(Qwen3ForCausalLM())
    print("Running pre-fix reproduction...")
    try:
        _old_infer_model_from_vllm(wrapped_model)
    except Exception as exc:  # noqa: BLE001
        print(f"Expected failure: {type(exc).__name__}: {exc}")
        return 0
    print("Unexpected success; reproduction failed.")
    return 1


def run_fixed_path_demo() -> int:
    """Exercise the current fixed helper path without requiring vLLM."""

    class FakeQwen3Adapter:
        def __init__(
            self, vllm_model: Any, blender: Any, enable_sparse: bool = False
        ) -> None:
            self.vllm_model = vllm_model
            self.blender = blender
            self.enable_sparse = enable_sparse

    fake_qwen3_module = types.ModuleType("lmcache.v1.compute.models.qwen3")
    fake_qwen3_module.LMCQwen3Model = FakeQwen3Adapter
    sys.modules["lmcache.v1.compute.models.qwen3"] = fake_qwen3_module

    wrapped_model = CUDAGraphWrapper(Qwen3ForCausalLM())
    adapter = infer_model_from_vllm(wrapped_model, blender=object(), enable_sparse=True)
    print("Fixed path succeeded.")
    print(f"Adapter type: {type(adapter).__name__}")
    print(f"Wrapped model unwrapped to: {type(adapter.vllm_model).__name__}")
    print(f"enable_sparse: {adapter.enable_sparse}")
    return 0


def run_real_repro(model: str) -> int:
    """Run the real vLLM initialization path that used to fail."""
    try:
        # Third Party
        from vllm import LLM
        from vllm.config import KVTransferConfig
        from vllm.engine.arg_utils import EngineArgs
    except ImportError:
        print("vllm is not installed. Use --mode minimal or --mode fixed instead.")
        return 2

    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = "# #"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"
    os.environ["LMCACHE_EXTRA_CONFIG"] = json.dumps(
        {"lmcache_instance_id": "repro-issue-2639"}
    )

    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )
    args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        enable_prefix_caching=False,
    )

    print("Starting real vLLM + LMCache initialization...")
    print("Before the fix, this would fail with:")
    print("NotImplementedError: Model type CUDAGraphWrapper is not supported in LMCache.")
    llm = LLM(**asdict(args))
    print("Initialization succeeded.")

    shutdown = getattr(llm, "shutdown", None)
    if callable(shutdown):
        shutdown()
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["minimal", "fixed", "real"],
        default="minimal",
        help="How to reproduce issue #2639.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model path/name for --mode real.",
    )
    args = parser.parse_args()

    if args.mode == "minimal":
        return run_minimal_repro()
    if args.mode == "fixed":
        return run_fixed_path_demo()
    if not args.model:
        print("--model is required for --mode real")
        return 2
    return run_real_repro(args.model)


if __name__ == "__main__":
    raise SystemExit(main())
