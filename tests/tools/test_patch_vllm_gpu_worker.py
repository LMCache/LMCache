#!/usr/bin/env python3
"""
Tests for the vLLM GPU worker patch script.
"""

# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import importlib.util

# Third Party
import pytest


def _load_patch_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "patch_vllm_gpu_worker.py"
    spec = importlib.util.spec_from_file_location(
        "patch_vllm_gpu_worker",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_gpu_worker_source() -> str:
    return "\n".join(
        [
            "from vllm.logger import init_logger",
            "",
            "logger = init_logger(__name__)",
            "",
            "def init_worker_distributed_environment(vllm_config):",
            "    ensure_kv_transfer_initialized(vllm_config)",
            "",
            "def initialize_from_config(self):",
            "    ensure_kv_transfer_initialized(self.vllm_config)",
            "",
            "def other(self):",
            "    pass",
            "",
        ]
    )


def test_patch_gpu_worker_idempotent(tmp_path: Path):
    module = _load_patch_module()
    worker_path = tmp_path / "gpu_worker.py"
    worker_path.write_text(_sample_gpu_worker_source(), encoding="utf-8")

    changed = module.patch_gpu_worker(worker_path)
    assert changed is True

    backup_path = worker_path.with_suffix(".py.bak")
    assert backup_path.exists()

    patched = worker_path.read_text(encoding="utf-8")
    assert "from lmcache.integration.vllm.utils import ENGINE_NAME" in patched
    assert "from lmcache.v1.compute.models.utils import VLLMModelTracker" in patched
    assert "VLLMModelTracker.register_model(" in patched
    assert "ensure_kv_transfer_initialized(self.vllm_config)" in patched
    assert "# ensure_kv_transfer_initialized(vllm_config)" in patched

    changed_again = module.patch_gpu_worker(worker_path)
    assert changed_again is False
