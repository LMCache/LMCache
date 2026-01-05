#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Patch vLLM GPU worker for LMCache model tracking.

This script:
  - locates vllm.v1.worker.gpu_worker via import
  - applies the LMCache model registration + KV transfer init changes
  - comments out ensure_kv_transfer_initialized in init_worker_distributed_environment
  - creates a backup of the original file

It is safe to run multiple times.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import argparse
import importlib
import shutil
import sys
import time

_IMPORTS_TO_ADD = [
    "from lmcache.integration.vllm.utils import ENGINE_NAME\n",
    "from lmcache.v1.compute.models.utils import VLLMModelTracker\n",
]


def _find_module_path(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"Unable to resolve file path for {module_name}")
    return Path(module_file).resolve()


def _backup_file(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + ".bak")
    if backup.exists():
        backup = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
    shutil.copy2(path, backup)
    return backup


def _find_function_block(lines: list[str], func_name: str) -> tuple[int, int] | None:
    start = None
    indent = 0
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"def {func_name}("):
            start = idx
            indent = len(line) - len(stripped)
            break
    if start is None:
        return None
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].lstrip()
        if stripped.startswith("def ") and (len(lines[idx]) - len(stripped)) <= indent:
            end = idx
            break
    return start, end


def _ensure_imports(lines: list[str]) -> tuple[list[str], bool]:
    content = "".join(lines)
    if all(line in content for line in _IMPORTS_TO_ADD):
        return lines, False

    insert_at = None
    for idx, line in enumerate(lines):
        if line.startswith("logger = init_logger("):
            insert_at = idx
            break
    if insert_at is None:
        insert_at = 0

    new_lines = lines[:insert_at] + _IMPORTS_TO_ADD + lines[insert_at:]
    return new_lines, True


def _comment_kv_init_in_worker_env(lines: list[str]) -> tuple[list[str], bool]:
    block = _find_function_block(lines, "init_worker_distributed_environment")
    if block is None:
        return lines, False

    start, end = block
    if not (0 <= start < end <= len(lines)):
        raise RuntimeError(
            "Invalid initialize_from_config range in gpu_worker.py: "
            f"start={start}, end={end}, total={len(lines)}"
        )
    changed = False
    for idx in range(start, end):
        line = lines[idx]
        stripped = line.lstrip()
        if "ensure_kv_transfer_initialized(" in stripped and not stripped.startswith(
            "#"
        ):
            leading = line[: len(line) - len(stripped)]
            lines[idx] = f"{leading}# {stripped}"
            changed = True
    return lines, changed


def _patch_initialize_from_config(lines: list[str]) -> tuple[list[str], bool]:
    block = _find_function_block(lines, "initialize_from_config")
    if block is None:
        raise RuntimeError("Unable to find initialize_from_config in gpu_worker.py")

    start, end = block
    ensure_idx = None
    for idx in range(start, end):
        if "ensure_kv_transfer_initialized(" in lines[idx]:
            ensure_idx = idx
            break

    if ensure_idx is None:
        raise RuntimeError(
            "Unable to find ensure_kv_transfer_initialized call "
            "inside initialize_from_config"
        )

    line = lines[ensure_idx]
    indent = line[: len(line) - len(line.lstrip())]
    register_line = (
        f"{indent}VLLMModelTracker.register_model("
        f"ENGINE_NAME, self.model_runner.model)\n"
    )
    ensure_line = f"{indent}ensure_kv_transfer_initialized(self.vllm_config)\n"

    changed = False
    has_registration = any(
        "VLLMModelTracker.register_model(" in line for line in lines[start:end]
    )
    if not has_registration:
        lines.insert(ensure_idx, register_line)
        ensure_idx += 1
        end += 1
        changed = True

    if lines[ensure_idx] != ensure_line:
        lines[ensure_idx] = ensure_line
        changed = True

    return lines, changed


def patch_gpu_worker(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    changed = False

    lines, imports_changed = _ensure_imports(lines)
    changed = changed or imports_changed

    lines, env_changed = _comment_kv_init_in_worker_env(lines)
    changed = changed or env_changed

    lines, init_changed = _patch_initialize_from_config(lines)
    changed = changed or init_changed

    if not changed:
        return False

    _backup_file(path)
    path.write_text("".join(lines), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        default="vllm.v1.worker.gpu_worker",
        help="Module path to patch (default: vllm.v1.worker.gpu_worker)",
    )
    args = parser.parse_args()

    try:
        target_path = _find_module_path(args.module)
    except Exception as exc:
        print(f"Error locating module {args.module}: {exc}", file=sys.stderr)
        return 1

    try:
        did_patch = patch_gpu_worker(target_path)
    except Exception as exc:
        print(f"Error patching {target_path}: {exc}", file=sys.stderr)
        return 1

    if did_patch:
        print(f"Patched {target_path}")
        print("Backup created alongside the original file.")
    else:
        print(f"No changes needed in {target_path}")

    print("Restart vLLM workers after patching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
