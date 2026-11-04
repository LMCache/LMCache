# SPDX-License-Identifier: Apache-2.0
"""Exercise real client modules with server imports and GPU extensions denied.

No LMCache/torch/engine symbols are stubbed. Full engine integration is outside
this CPU probe; the vLLM engine module and SGLang constructor need the engines.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import argparse
import base64
import gc
import importlib
import importlib.abc
import json
import select
import subprocess
import sys
import traceback


class BoundaryGuard(importlib.abc.MetaPathFinder):
    """Reject any LMCache module outside the physically materialized client."""

    def __init__(self, root: Path, block_native: bool):
        self.root = root
        self.block_native = block_native
        self.denied: list[str] = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "lmcache" and not fullname.startswith("lmcache."):
            return None
        if fullname == "lmcache.lmcache_native" and not self.block_native:
            return None
        name = self.root.joinpath(*fullname.split("."))
        if name.is_dir() or name.with_suffix(".py").exists():
            return None
        self.denied.append(fullname)
        raise ModuleNotFoundError(f"split boundary denies {fullname}", name=fullname)


UNREGISTERED_TIMEOUT = 0.5


def expected_unregistered_error(transport: str, exc: Exception, elapsed: float) -> dict:
    """Accept only the transport's documented reaction to a missing handler."""
    name = f"{type(exc).__module__}.{type(exc).__qualname__}"
    # Both servers reply with an explicit error instead of letting the request time out.
    if elapsed >= UNREGISTERED_TIMEOUT * 0.9:
        raise AssertionError(
            f"missing handler took {elapsed:.3f}s; expected an error reply"
        ) from exc
    if transport == "zmq":
        if not (type(exc) is RuntimeError and "(operation_not_enabled)" in str(exc)):
            raise AssertionError(
                f"ZMQ missing handler must reply operation_not_enabled, got {exc!r}"
            ) from exc
        return {"exception": name, "elapsed_s": round(elapsed, 3), "message": str(exc)}
    # First Party
    from lmcache.v1.multiprocess.transport.errors import MPRequestError

    if not (isinstance(exc, MPRequestError) and exc.code == "UNIMPLEMENTED"):
        raise AssertionError(
            f"gRPC missing handler must be UNIMPLEMENTED, got {exc!r}"
        ) from exc
    return {
        "exception": name,
        "code": exc.code,
        "details": exc.details,
        "elapsed_s": round(elapsed, 3),
        "message": str(exc),
    }


def client_cleanup(transport: str, client) -> dict:
    """After close(): the ZMQ sockets are closed and the shared poll loop is gone."""
    if transport != "zmq":
        return {"close_returned": True}
    # Standard
    import threading
    import time

    # First Party
    from lmcache.v1.multiprocess.transport.zmq_impl.mq import ClientPollingLoop

    mq = client._message_queue_client
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(
        t.name == "mq-client-shared-loop" and t.is_alive()
        for t in threading.enumerate()
    ):
        time.sleep(0.05)
    state = {
        "socket_closed": mq.socket.closed,
        "monitor_closed": mq.monitor.closed,
        "polling_loop_released": ClientPollingLoop._instance is None,
        "polling_thread_alive": any(
            t.name == "mq-client-shared-loop" and t.is_alive()
            for t in threading.enumerate()
        ),
        "pending_futures_after_close": len(mq.pending_futures),
    }
    if not (
        state["socket_closed"]
        and state["monitor_closed"]
        and state["polling_loop_released"]
        and not state["polling_thread_alive"]
    ):
        raise AssertionError(f"ZMQ client not cleaned up: {state}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--block-native", action="store_true")
    parser.add_argument("--zmq")
    parser.add_argument("--grpc")
    args = parser.parse_args()
    root = args.root.resolve()

    def protect_sources(event, values):
        if event == "open" and isinstance(values[0], str):
            filename, mode, flags = values
            if filename.endswith(".py") and str(root) in filename and flags & 3:
                raise RuntimeError(
                    "Probe attempted to change Python source: " + filename
                )

    sys.addaudithook(protect_sources)
    guard = BoundaryGuard(root, args.block_native)
    sys.meta_path.insert(0, guard)
    sys.path.insert(0, str(root))
    results = []

    def run(name, fn):
        try:
            detail = fn()
            results.append({"name": name, "status": "pass", "detail": detail})
        except Exception as exc:
            results.append(
                {
                    "name": name,
                    "status": "fail",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    for module in (
        "lmcache",
        "lmcache.integration.vllm.vllm_multi_process_adapter",
        "lmcache.integration.sglang.unified_lmcache_mp_connector",
        "lmcache.v1.multiprocess.transport.zmq_impl.client",
        "lmcache.v1.multiprocess.transport.grpc_impl.client",
    ):
        run(
            "import:" + module,
            lambda module=module: importlib.import_module(module).__file__,
        )

    def singleton_and_discovery():
        # First Party
        from lmcache.v1.platform import get_device_spec, resolve_device_ops
        from lmcache.v1.platform._device_detect import _build_backend_registry
        import lmcache

        assert lmcache.device_ops is resolve_device_ops(lmcache.torch_device_type)
        assert resolve_device_ops("cpu") is resolve_device_ops("cpu")
        assert get_device_spec("cpu").ipc_wrapper_cls.__name__ == "CpuShmTensorWrapper"
        return {
            "device": lmcache.torch_device_type,
            "backends": sorted(_build_backend_registry()),
            "ops": type(lmcache.device_ops).__name__,
        }

    run("root-singleton-and-builtin-discovery", singleton_and_discovery)

    def rpc_and_codecs():
        # First Party
        from lmcache.v1.multiprocess.rpc import get_rpc_specs
        from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
            get_service_bindings,
        )
        from lmcache.v1.multiprocess.transport.grpc_impl.method_registry import (
            get_method_codec_registry,
        )

        specs = get_rpc_specs()
        codecs = get_method_codec_registry().by_full_name
        assert len(specs) == len(codecs) == 37
        assert {c.operation for c in codecs.values()} == set(specs)
        return {
            "rpc": len(specs),
            "grpc_codecs": len(codecs),
            "grpc_services": len(get_service_bindings()),
        }

    run("all-rpc-hints-and-generated-grpc-codecs", rpc_and_codecs)

    def gather_scatter():
        # Third Party
        import torch

        # First Party
        from lmcache.v1.multiprocess.transfer_context.base import (
            gather_paged_kv_to_cpu,
            scatter_cpu_to_paged_kv,
        )

        shape = (2, 4, 2, 1, 3)
        original = {
            f"layer.{i}": torch.arange(48, dtype=torch.float32).reshape(shape) + i * 100
            for i in range(2)
        }
        expected = torch.stack(
            [t[:, [2, 0], :, :, :].reshape(2, 4, 3) for t in original.values()], dim=1
        )
        chunks = gather_paged_kv_to_cpu(original, [2, 0], 2, {"kv_layout": "NHD"})
        assert len(chunks) == 1
        assert torch.equal(chunks[0], expected), (
            "gather must match independently indexed canonical bytes"
        )
        target = {k: torch.zeros_like(t) for k, t in original.items()}
        scatter_cpu_to_paged_kv(
            target, [2, 0], chunks, 2, layout_hints={"kv_layout": "NHD"}
        )
        for key, value in target.items():
            reference = torch.zeros_like(value)
            reference[:, [2, 0]] = original[key][:, [2, 0]]
            assert torch.equal(value, reference)
        # The function-local import in scatter used to drag the server back in.
        assert "lmcache.v1.multiprocess.object_group_transfer" not in sys.modules
        return {
            "layers": 2,
            "chunk_shape": list(chunks[0].shape),
            "reference": "independent tensor indexing; reordered nonadjacent blocks",
        }

    run("cpu-engine-driven-gather-scatter", gather_scatter)

    def inline_codec():
        # Third Party
        import msgspec
        import torch

        # First Party
        from lmcache.v1.multiprocess.transfer_context.tensor_wire import (
            TensorEnvelope,
            decode_tensors,
            encode_tensors,
        )

        value = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
        restored = decode_tensors(
            encode_tensors([value]),
            expected_shape=value.shape,
            expected_dtype=value.dtype,
            expected_count=1,
        )
        assert torch.equal(restored[0], value)
        encoded = msgspec.msgpack.encode(
            {"version": 1, "tensors": [], "future_field": 1}
        )
        try:
            msgspec.msgpack.decode(encoded, type=TensorEnvelope)
        except msgspec.ValidationError:
            return {"noncontiguous_tensor": "pass", "unknown_field": "rejected"}
        raise AssertionError("strict envelope unexpectedly accepted an unknown field")

    run("inline-tensor-codec-and-strict-schema", inline_codec)

    def eventfd():
        # First Party
        from lmcache.v1.platform import create_event_notifier

        notifier = create_event_notifier()
        try:
            notifier.notify()
            ready, _, _ = select.select([notifier.fileno()], [], [], 1)
            assert ready
            notifier.consume()
            assert not select.select([notifier.fileno()], [], [], 0)[0]
            return type(notifier).__name__
        finally:
            notifier.close()

    run("linux-event-notifier", eventfd)

    def shm():
        # Third Party
        import msgspec
        import torch

        # First Party
        from lmcache.v1.platform.devices.cpu.shm import CpuShmTensorWrapper

        value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        wrapper = CpuShmTensorWrapper.wrap(value)
        encoded = base64.b64encode(msgspec.msgpack.encode(wrapper.to_wire())).decode()
        script = """import sys,base64,msgspec,torch
sys.path.insert(0,sys.argv[1])
from lmcache.v1.platform.base.ipc_wire import CpuShmIPCData
from lmcache.v1.platform.devices.cpu.shm import CpuShmTensorWrapper
wire=msgspec.msgpack.decode(base64.b64decode(sys.argv[2]),type=CpuShmIPCData)
t=CpuShmTensorWrapper.from_wire(wire).to_tensor()
assert torch.equal(t,torch.arange(6,dtype=torch.float32).reshape(2,3))
t[0,0]=91
print('child-shm-write=91')
"""
        out = subprocess.run(
            [sys.executable, "-I", "-B", "-c", script, str(root), encoded],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        assert value[0, 0].item() == 91
        name = wrapper.shm_name
        del wrapper, value
        gc.collect()
        return {"cross_process": out.stdout.strip(), "name": name}

    run("linux-posix-shm-cross-process", shm)

    for transport, endpoint in (("zmq", args.zmq), ("grpc", args.grpc)):
        if not endpoint:
            continue

        def transport_probe(endpoint=endpoint, transport=transport):
            # Standard
            import time

            # First Party
            from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

            client = RequestClientFactory.create(endpoint)
            closed = False
            try:
                assert client.get_chunk_size().result(timeout=5) == 4
                assert client.ping(None).result(timeout=5) is True
                assert client.noop().result(timeout=5) == "split-probe"
                start = time.monotonic()
                try:
                    client.query_prefetch_status("deliberately-unregistered").result(
                        timeout=UNREGISTERED_TIMEOUT
                    )
                except Exception as exc:
                    unregistered = expected_unregistered_error(
                        transport, exc, time.monotonic() - start
                    )
                else:
                    raise AssertionError("unregistered operation succeeded")
                client.close()
                closed = True
                return {
                    "basic_rpc": "pass",
                    "unregistered_error": unregistered["exception"],
                    "message": unregistered["message"],
                    "unregistered": unregistered,
                    "cleanup": client_cleanup(transport, client),
                }
            finally:
                if not closed:
                    client.close()

        run(transport + "-real-request-client", transport_probe)

    if args.grpc:

        def scheduler_startup():
            # Third Party
            import zmq

            # First Party
            from lmcache.integration.vllm.vllm_multi_process_adapter import (
                LMCacheMPSchedulerAdapter,
                ParallelStrategy,
            )

            context = zmq.Context()
            adapter = LMCacheMPSchedulerAdapter(
                [args.grpc],
                context,
                "split-probe",
                2,
                ParallelStrategy(False, 1, 0, 1, 1, 1),
                mq_timeout=5,
            )
            try:
                assert adapter.lmcache_tokens_per_chunk == 4
                assert adapter.blocks_in_chunk == 2
                return {
                    "startup": "pass",
                    "chunk_size": 4,
                    "blocks_in_chunk": 2,
                    "scope": "scheduler adapter, typed management handlers; no engine",
                }
            finally:
                adapter.shutdown()
                context.term()

        run("vllm-scheduler-adapter-grpc-startup", scheduler_startup)

    # Explicitly disclose the limits without fabricating an installed engine.
    host = {}
    for module in ("vllm", "sglang"):
        try:
            host[module] = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            host[module] = False
    loaded = sorted(
        n for n in sys.modules if n == "lmcache" or n.startswith("lmcache.")
    )
    forbidden = sorted(
        set(guard.denied)
        - {
            "lmcache.cuda_ops",
            "lmcache.xpu_ops",
            *(["lmcache.lmcache_native"] if args.block_native else []),
        }
    )
    result = {
        "root": str(root),
        "native_blocked": args.block_native,
        "checks": results,
        "pass": sum(r["status"] == "pass" for r in results),
        "fail": sum(r["status"] == "fail" for r in results),
        "denied_import_attempts": sorted(set(guard.denied)),
        "unexpected_boundary_attempts": forbidden,
        "loaded_lmcache_modules": loaded,
        "host_engines_installed": host,
    }
    print(json.dumps(result, indent=2))
    if result["fail"] or forbidden:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
