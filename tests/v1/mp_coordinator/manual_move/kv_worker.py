# SPDX-License-Identifier: Apache-2.0
"""External validation harness; uses public MP transport and transfer APIs."""

# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
import argparse
import hashlib
import json
import random
import time
import traceback

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    LMCacheDrivenTransferContext,
)
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

NL, NB, BS, NH, HS = 28, 128, 16, 8, 128
TOKENS, CHUNK = 1536, 256
SRC_BLOCKS = random.Random(173).sample(range(NB), TOKENS // BS)
DST_BLOCKS = random.Random(811).sample(range(NB), TOKENS // BS)
STATE: dict[str, Any] = {}
SERVER_URL = "tcp://127.0.0.1:19555"


def expected_layer(rank: int, layer: int, dtype: torch.dtype) -> torch.Tensor:
    """Generate reproducible CPU reference data independently on each host."""
    gen = torch.Generator(device="cpu").manual_seed(19271 + rank * 1000 + layer)
    return torch.randn((2, NB, BS, NH, HS), generator=gen, dtype=torch.float32).to(
        dtype
    )


def key(request_id: str, rank: int | None = None) -> IPCCacheServerKey:
    """Build the current case key for lookup or a per-rank transfer."""
    return IPCCacheServerKey(
        model_name=STATE["model"],
        world_size=STATE["tp"],
        worker_id=rank,
        token_ids=tuple(STATE["tokens"]),
        start=0,
        end=TOKENS,
        request_id=request_id,
        num_kv_readers=1,
    )


def close() -> None:
    """Unregister buffers before releasing GPU memory."""
    client = STATE.get("client")
    if client:
        for rank, ctx in enumerate(STATE["contexts"]):
            client.unregister_kv_cache(1000 + rank).result(timeout=30)
            ctx.close()
        client.close()
    STATE.clear()
    torch_dev.empty_cache()


def compare(blocks: list[int]) -> dict[str, Any]:
    """Compare logical chunks as bytes and report per-chunk SHA-256 hashes."""
    checks, mismatches, byte_count = {}, 0, 0
    source_index = torch.tensor(SRC_BLOCKS)
    read_index = torch.tensor(blocks, device=torch_device_type)
    for rank, layers in enumerate(STATE["buffers"]):
        for li, tensor in enumerate(layers.values()):
            expected = expected_layer(rank, li, STATE["dtype"]).index_select(
                1, source_index
            )
            actual = tensor.index_select(1, read_index).cpu()
            for ci in range(TOKENS // CHUNK):
                lo = ci * (CHUNK // BS)
                x = actual[:, lo : lo + CHUNK // BS].contiguous().view(torch.uint8)
                y = expected[:, lo : lo + CHUNK // BS].contiguous().view(torch.uint8)
                diff = int(torch.count_nonzero(x != y))
                mismatches += diff
                byte_count += x.numel()
                checks[f"rank{rank}/layer{li}/chunk{ci}"] = hashlib.sha256(
                    x.numpy().tobytes()
                ).hexdigest()
    return {
        "mismatched_bytes": mismatches,
        "compared_bytes": byte_count,
        "checksums": checks,
        "rank_count": STATE["tp"],
        "layers": NL,
        "chunks": TOKENS // CHUNK,
    }


def lookup(request_id: str) -> int:
    """Wait for the public lookup result or fail after 30 seconds."""
    client = STATE["client"]
    client.lookup(key(request_id), STATE["tp"]).result(timeout=30)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        n = client.query_prefetch_status(request_id).result(timeout=30)
        if n is not None:
            return n
        time.sleep(0.05)
    raise RuntimeError("lookup did not complete within 30 seconds")


def action(op: str, data: dict[str, Any]) -> dict[str, Any]:
    """Run one validation operation; failures return HTTP 500 to the driver."""
    if op == "prepare":
        close()
        STATE.update(
            tp=data["tp"],
            model=data["model"],
            tokens=data["tokens"],
            dtype=getattr(torch, data["dtype"]),
            contexts=[],
            buffers=[],
        )
        STATE["client"] = RequestClientFactory.create(SERVER_URL)
        for rank in range(STATE["tp"]):
            layers = {
                f"layer_{li}": expected_layer(rank, li, STATE["dtype"]).to(
                    torch_device_type
                )
                for li in range(NL)
            }
            if data["role"] != "source":
                for t in layers.values():
                    t.fill_(-123)
            ctx = LMCacheDrivenTransferContext()
            ctx.register(
                1000 + rank,
                layers,
                STATE["model"],
                STATE["tp"],
                CHUNK // BS,
                STATE["client"],
                30,
                layout_hints={"kv_layout": "NHD"},
                engine_group_infos=[
                    EngineGroupInfo(
                        engine_group_id=0,
                        layer_indices=tuple(range(NL)),
                        tokens_per_block=BS,
                    )
                ],
            )
            STATE["buffers"].append(layers)
            STATE["contexts"].append(ctx)
        return {
            "ready": True,
            "device_type": torch_device_type,
            "model": STATE["model"],
            "role": data["role"],
        }
    if op == "original":
        return compare(SRC_BLOCKS)
    if op == "poison":
        for layers in STATE["buffers"]:
            for t in layers.values():
                t.fill_(-123)
        torch_dev.synchronize()
        return compare(DST_BLOCKS)
    if op == "store":
        rid = "store-" + STATE["model"]
        n = lookup(rid)
        assert n == 0, n
        results = []
        for rank, ctx in enumerate(STATE["contexts"]):
            event = ctx.create_recorded_event()
            result = ctx.submit_store(
                rid,
                key(rid, rank),
                1000 + rank,
                STATE["buffers"][rank],
                [SRC_BLOCKS],
                event,
                CHUNK // BS,
            ).result(timeout=60)
            results.append(result)
        STATE["client"].end_session(rid).result(timeout=30)
        assert all(results), results
        return {"stored": results}
    if op == "retrieve":
        rid = "retrieve-" + STATE["model"] + "-" + str(time.time_ns())
        n = lookup(rid)
        assert n == TOKENS // CHUNK, n
        results = []
        for rank, ctx in enumerate(STATE["contexts"]):
            event = ctx.create_recorded_event()
            result = ctx.submit_retrieve(
                rid,
                key(rid, rank),
                1000 + rank,
                STATE["buffers"][rank],
                [DST_BLOCKS],
                event,
                CHUNK // BS,
            ).result(timeout=60)
            results.append(result)
        STATE["client"].end_session(rid).result(timeout=30)
        torch_dev.synchronize()
        assert all(results), results
        report = compare(DST_BLOCKS)
        # Unaddressed destination blocks must still contain the poison value.
        unused = torch.tensor(
            sorted(set(range(NB)) - set(DST_BLOCKS)), device=torch_device_type
        )
        report["untouched_blocks_preserved"] = all(
            bool(torch.all(t.index_select(1, unused) == -123))
            for layers in STATE["buffers"]
            for t in layers.values()
        )
        report["retrieved"] = results
        return report
    if op == "close":
        close()
        return {"closed": True}
    raise ValueError(op)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        """Dispatch a test operation and return its JSON result."""
        try:
            data = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
            )
            result = action(self.path.strip("/"), data)
            status = 200
        except Exception:
            result = {"error": traceback.format_exc()}
            print(result["error"], flush=True)
            status = 500
        body = json.dumps(result).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    """Serve an explicitly bound, single-threaded test worker."""
    global SERVER_URL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19955)
    parser.add_argument("--server-url", default=SERVER_URL)
    args = parser.parse_args()
    SERVER_URL = args.server_url
    if not torch_dev.is_available():
        parser.error(
            "An accelerator and working LMCache native extensions are required"
        )
    with HTTPServer((args.host, args.port), Handler) as server:
        print("KV worker ready", flush=True)
        try:
            server.serve_forever()
        finally:
            close()


if __name__ == "__main__":
    main()
