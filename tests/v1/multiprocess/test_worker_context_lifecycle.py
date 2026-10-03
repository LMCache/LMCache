# SPDX-License-Identifier: Apache-2.0
"""Actual SHM ownership and data transfer across replacement/server restart."""

# Standard
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPWorkerAdapter,
    ParallelStrategy,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transfer_context import shm as shm_module


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait(condition: Callable[[], bool], timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.1)
    raise AssertionError("condition timed out")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA pinning")
def test_shm_replacements_and_server_restart_release_local_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    port, http_port = _free_port(), _free_port()
    endpoint = f"tcp://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "lmcache.v1.multiprocess.http_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--http-host",
        "127.0.0.1",
        "--http-port",
        str(http_port),
        "--chunk-size",
        "8",
        "--l1-size-gb",
        "0.05",
        "--no-l1-use-lazy",
        "--supported-transfer-mode",
        "engine_driven",
        "--eviction-policy",
        "LRU",
    ]
    attachments, pins, unpins = [], [], []
    attach = shm_module.shared_memory.SharedMemory
    device = shm_module.current_device_spec

    def record_attachment(*args, **kwargs):
        mapping = attach(*args, **kwargs)
        attachments.append(mapping)
        return mapping

    def pin(ptr: int, size: int) -> bool:
        result = device.pin_memory(ptr, size)
        if result:
            pins.append(ptr)
        return result

    def unpin(ptr: int) -> None:
        device.unpin_memory(ptr)
        unpins.append(ptr)

    monkeypatch.setattr(shm_module.shared_memory, "SharedMemory", record_attachment)
    monkeypatch.setattr(
        shm_module,
        "current_device_spec",
        SimpleNamespace(
            pin_memory=pin,
            unpin_memory=unpin,
        ),
    )
    context = zmq.Context()
    server = None
    adapter = None
    closed = False
    logs: list[TextIO] = []

    def start_server() -> subprocess.Popen:
        log = (tmp_path / f"server-{len(logs)}.log").open("w")
        logs.append(log)
        process = subprocess.Popen(
            command + ["--shm-name", f"lmcache_lifecycle_{os.getpid()}_{len(logs)}"],
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        def ready() -> bool:
            assert process.poll() is None, "server exited; inspect server log"
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{http_port}/metrics", timeout=0.2
                ) as response:
                    return response.status == 200
            except (urllib.error.URLError, OSError):
                return False

        _wait(ready, 60)
        return process

    def fd_count() -> int:
        names = {mapping.name for mapping in attachments}
        count = 0
        for fd in os.listdir("/proc/self/fd"):
            try:
                target = os.readlink(f"/proc/self/fd/{fd}")
            except FileNotFoundError:
                continue
            count += any(f"/dev/shm/{name}" in target for name in names)
        return count

    kv = {"layer.0": torch.randn(2, 4, 4, 2, 8, device="cuda")}

    def roundtrip(seed: int) -> None:
        assert adapter is not None
        key = IPCCacheServerKey.from_token_ids(
            "lifecycle-test",
            1,
            0,
            list(range(seed, seed + 8)),
            end=8,
            request_id=f"round-{seed}",
        )
        source = kv["layer.0"][:, :2].clone()
        with adapter.use_transfer_context(blocking=True) as transfer:
            assert isinstance(
                transfer.engine_driven_context, shm_module.EngineDrivenContextShm
            )
            future = transfer.submit_store(
                "store",
                key,
                kv,
                [[0, 1]],
                transfer.create_recorded_event(),
                2,
            )
            assert future.result(timeout=20) is True
            adapter.req_client.lookup(key, 1).result(timeout=20)
            adapter.req_client.wait_prefetch_status(key.request_id, 20).result(
                timeout=25
            )
            assert (
                transfer.submit_retrieve("retrieve", key, kv, [[2, 3]], None, 2).result(
                    timeout=20
                )
                is True
            )
        assert torch.equal(kv["layer.0"][:, 2:4], source)
        adapter.req_client.end_session(key.request_id).result(timeout=20)

    try:
        server = start_server()
        adapter = LMCacheMPWorkerAdapter(
            server_url=endpoint,
            context=context,
            model_name="lifecycle-test",
            vllm_block_size=4,
            parallel_strategy=ParallelStrategy(False, 1, 0, 1, 1, 1),
            mq_timeout=5,
            extra_config={
                "lmcache.mp.mp_transfer_mode": "engine_driven",
                "lmcache.mp.heartbeat_interval": 0.1,
            },
        )
        adapter.register_kv_caches(kv, layout_hints={"kv_layout": "NHD"})
        roundtrip(0)
        baseline_fds, baseline_threads = fd_count(), threading.active_count()
        assert baseline_fds > 0
        for cycle in range(1, 6):
            adapter.register_kv_caches(kv, layout_hints={"kv_layout": "NHD"})
            roundtrip(cycle * 16)
            assert fd_count() == baseline_fds
            assert threading.active_count() <= baseline_threads + 1
            assert os.path.exists(f"/dev/shm/{attachments[-1].name}")
        old_name = attachments[-1].name
        server.terminate()
        server.wait(timeout=20)
        server = start_server()
        _wait(lambda: attachments[-1].name != old_name and adapter.is_healthy)
        roundtrip(128)
        assert fd_count() == baseline_fds
        adapter.shutdown()
        closed = True
        assert fd_count() == 0
        assert pins and len(pins) == len(unpins)
    finally:
        if adapter is not None and not closed:
            adapter.shutdown()
        if server is not None:
            server.terminate()
            server.wait(timeout=20)
        context.destroy(linger=0)
        for log in logs:
            log.close()
