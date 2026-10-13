# SPDX-License-Identifier: Apache-2.0
"""Verify the installed PR #5685 image as its default non-root user."""

# Future
from __future__ import annotations

# Standard
from multiprocessing.queues import Queue
from pathlib import Path
import gc
import importlib.metadata
import importlib.util
import json
import os

# Third Party
import torch
import torch.multiprocessing as mp

# First Party
from lmcache import _version
from lmcache.v1.platform.base.ipc_wire import WireIPCData
from lmcache.v1.platform.devices.cuda.ipc_wrapper import CudaIPCWrapper
import lmcache
import lmcache.cuda_ops  # noqa: F401


def read_shared_tensor(descriptor: WireIPCData, output: Queue[float]) -> None:
    """Read a CUDA IPC tensor in a separate process and report its sum.

    Args:
        descriptor: Tensor descriptor produced by LMCache's CUDA IPC wrapper.
        output: Queue carrying the sum to the producer.
    """
    torch.cuda.init()
    wrapper = CudaIPCWrapper.from_wire(descriptor)
    tensor = wrapper.to_tensor()
    output.put(float(tensor.sum()))
    del tensor, wrapper
    gc.collect()
    torch.cuda.synchronize()


if __name__ == "__main__":
    if os.getuid() != 1000 or os.environ["HOME"] != "/home/lmcache":
        raise RuntimeError("The image did not run as its default lmcache user")
    if not str(lmcache.__file__).startswith("/opt/venv/"):
        raise RuntimeError("The checkout is shadowing the installed image package")
    version = importlib.metadata.version("lmcache")
    if _version.commit_id != "g1541a889":
        raise RuntimeError(f"Unexpected image source revision: {_version.commit_id}")
    if importlib.util.find_spec("lmcache.c_ops") is not None:
        raise RuntimeError("The obsolete lmcache.c_ops extension is present")
    print(
        json.dumps(
            {
                "uid": os.getuid(),
                "home": os.environ["HOME"],
                "lmcache": version,
                "commit": _version.commit_id,
                "package": lmcache.__file__,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "capability": torch.cuda.get_device_capability(0),
            }
        ),
        flush=True,
    )
    for directory in (
        Path.home() / ".cache/huggingface/hub",
        Path.home() / ".cache/vllm",
        Path("/dev/shm"),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f"nonroot-image-{os.getpid()}"
        probe.write_text("writable")
        probe.unlink()
    if not torch.empty(1024, pin_memory=True).is_pinned():
        raise RuntimeError("Pinned host memory allocation failed")
    tensor = torch.arange(1024, dtype=torch.float32, device="cuda")
    descriptor = CudaIPCWrapper.wrap(tensor).to_wire()
    context = mp.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=read_shared_tensor, args=(descriptor, output))
    process.start()
    result = output.get(timeout=60)
    process.join(timeout=60)
    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError("CUDA IPC consumer did not exit")
    if process.exitcode != 0 or result != 523776.0:
        raise RuntimeError(f"CUDA IPC round trip failed: {process.exitcode}, {result}")
    output.close()
    output.join_thread()
    print("NONROOT_IMAGE_CUDA_PINNED_MEMORY_IPC_PASS", flush=True)
