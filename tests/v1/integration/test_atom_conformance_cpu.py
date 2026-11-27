# SPDX-License-Identifier: Apache-2.0
"""Conformance run: an ATOM transfer against a real MP server on CPU.

The fake ATOM descriptor/request generator below mirrors the external-connector
API (v2). It produces the layout and transfer request ATOM would send, runs
them through the LMCache conversion, and drives a real ``lmcache server`` over
the CPU transport: register the converted groups, store a token sequence with a
tenant salt, then look it up with the chunk-hash chain the converter produced.

A hit here means the two sides agree on the object identity (model, salt, chunk
hashes, rank fan-out); a miss means the conversion and the server derive
different keys.

The server test needs the CPU device backend, so it skips on a CUDA host. The
conversion and rejection cases run everywhere.
"""

# Standard
from dataclasses import replace
from typing import Any, cast
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import uuid

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache.integration.atom.hasher import TokenChainHasher
from lmcache.integration.atom.layout import (
    AtomKernelGroup,
    AtomStoreLayout,
    engine_group_infos_from_atom,
)
from lmcache.integration.atom.transfer import (
    AtomTransferRequest,
    transfer_spec_from_atom,
)
from lmcache.v1.client.transport.factory import RequestClientFactory
from lmcache.v1.client.worker_transfer import create_transfer_context
from lmcache.v1.common.custom_types import EngineType, IPCCacheServerKey
from lmcache.v1.common.group_view import EngineGroupInfo
from lmcache.v1.common.token_codec import pack_token_ids
from lmcache.v1.platform import torch_device_type

_CHUNK_TOKENS = 256
_CHUNKS = 2
_SALT = "tenant-a"
_MODEL = "atom-conformance"
_RPC_TIMEOUT_S = 30.0

needs_cpu_device = pytest.mark.skipif(
    torch_device_type != "cpu",
    reason="the CPU transport run needs the CPU device backend",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get(url: str) -> str:
    """GET a loopback URL without the environment's HTTP proxy."""
    # Standard
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=5.0) as response:
        return response.read().decode("utf-8", "replace")


def _fake_atom_layout() -> AtomStoreLayout:
    """A PAGE-like layout: one latent tensor in one kernel group."""
    return AtomStoreLayout(
        tensors={"latent": torch.zeros((16, _CHUNK_TOKENS, 16), dtype=torch.float16)},
        groups=(
            AtomKernelGroup(
                tensor_names=("latent",),
                engine_group_id=0,
                tokens_per_block=_CHUNK_TOKENS,
            ),
        ),
    )


def _fake_atom_request(tokens: list[int]) -> AtomTransferRequest:
    """The transfer request ATOM would send for ``tokens``."""
    hasher = TokenChainHasher(chunk_size=_CHUNK_TOKENS)
    hashes = b""
    prev: bytes | None = None
    for index in range(_CHUNKS):
        chunk = tokens[index * _CHUNK_TOKENS : (index + 1) * _CHUNK_TOKENS]
        prev = hasher.chain(chunk, index, prev)
        hashes += prev
    return AtomTransferRequest(
        hashes=hashes,
        block_ids=[list(range(1, _CHUNKS + 1))],
        start=0,
        end=_CHUNK_TOKENS * _CHUNKS,
        cache_salt=_SALT,
    )


class _Server:
    """One CPU ``lmcache server`` subprocess for the block's duration."""

    def __init__(self) -> None:
        self.rpc_port = _free_port()
        self.http_port = _free_port()
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def rpc_url(self) -> str:
        return f"grpc://127.0.0.1:{self.rpc_port}"

    def __enter__(self) -> "_Server":
        env = dict(
            os.environ,
            LMCACHE_DISABLE_BANNER="1",
            LMCACHE_TRACK_USAGE="false",
            DO_NOT_TRACK="1",
        )
        command = [
            sys.executable,
            "-m",
            "lmcache.v1.multiprocess.http_server",
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(self.http_port),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.rpc_port),
            "--chunk-size",
            str(_CHUNK_TOKENS),
            "--l1-size-gb",
            "0.0625",
            "--supported-transfer-mode",
            "lmcache_driven",
        ]
        log = pathlib.Path(f"atom-conformance-{uuid.uuid4().hex}.log")
        with log.open("wb") as sink:
            self._process = subprocess.Popen(
                command, env=env, stdout=sink, stderr=subprocess.STDOUT
            )
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                break
            try:
                if "ok" in _get(f"http://127.0.0.1:{self.http_port}/healthcheck"):
                    return self
            except Exception:
                pass
            time.sleep(0.2)
        log_text = log.read_text(errors="replace")[-4000:]
        self._stop()
        raise RuntimeError(f"the conformance server did not come up:\n{log_text}")

    def __exit__(self, *exc_info: object) -> None:
        self._stop()

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.send_signal(signal.SIGINT)
        try:
            process.wait(15)
        except subprocess.TimeoutExpired:
            process.kill()

    def metric_total(self, family: str) -> float:
        """Sum one counter family over its series."""
        total = 0.0
        for line in _get(f"http://127.0.0.1:{self.http_port}/metrics").splitlines():
            if line.startswith("#") or not line.startswith(family):
                continue
            total += float(line.rsplit(" ", 1)[-1])
        return total


def test_the_converted_groups_are_what_the_server_registers() -> None:
    """The descriptor conversion is the registration payload, field by field."""
    infos = engine_group_infos_from_atom(_fake_atom_layout())

    assert infos == [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0,),
            tokens_per_block=_CHUNK_TOKENS,
        )
    ]


def test_the_converted_hash_chain_is_the_prefix_chain() -> None:
    """ATOM sends the ``[0, end)`` chain; the converter keeps it unsliced."""
    tokens = list(range(_CHUNK_TOKENS * _CHUNKS))
    spec = transfer_spec_from_atom(
        _fake_atom_request(tokens), chunk_tokens=_CHUNK_TOKENS
    )

    assert spec.start == 0 and spec.end == _CHUNK_TOKENS * _CHUNKS
    assert spec.chunk_hashes is not None
    assert len(spec.chunk_hashes) == 32 * _CHUNKS
    assert spec.cache_salt == _SALT


@pytest.mark.parametrize(
    "request_kwargs, match",
    [
        ({"hashes": b"\x00" * 32}, "carries 1 chunk hash"),
        ({"block_ids": []}, "names no block-id list"),
        ({"block_ids": [[]]}, "names no blocks"),
        ({"cache_salt": "a@b"}, "must not contain"),
        ({"start": 512, "end": 256}, "is past its end"),
    ],
)
def test_a_malformed_request_is_rejected_before_submission(
    request_kwargs: dict[str, object], match: str
) -> None:
    """Failure injection at the conversion boundary, not inside the server."""
    tokens = list(range(_CHUNK_TOKENS * _CHUNKS))
    body = _fake_atom_request(tokens)
    request = replace(body, **cast(Any, request_kwargs))

    with pytest.raises(ValueError, match=match):
        transfer_spec_from_atom(request, chunk_tokens=_CHUNK_TOKENS)


@needs_cpu_device
def test_a_store_written_by_atom_is_found_by_the_converted_key() -> None:
    """A real server hit proves both sides derive the same object identity."""
    tokens = list(range(_CHUNK_TOKENS * _CHUNKS))
    spec = transfer_spec_from_atom(
        _fake_atom_request(tokens), chunk_tokens=_CHUNK_TOKENS
    )
    with _Server() as server:
        context = zmq.Context()
        context.setsockopt(zmq.LINGER, 0)
        client = RequestClientFactory.create(server.rpc_url, context=context)
        kv_caches = {
            "latent": torch.randn((16, _CHUNK_TOKENS, 16), dtype=torch.float16)
        }
        transfer = create_transfer_context(
            kv_caches,
            mode="lmcache_driven",
            instance_id=1000,
            req_client=client,
        )
        try:
            transfer.register(
                kv_caches,
                _MODEL,
                1,
                0,
                _RPC_TIMEOUT_S,
                layout_hints={},
                engine_group_infos=engine_group_infos_from_atom(_fake_atom_layout()),
                engine_type=EngineType.ATOM,
                lmcache_chunk_tokens=_CHUNK_TOKENS,
            )
            store_id = uuid.uuid4().hex
            stored = transfer.submit_store(
                store_id,
                IPCCacheServerKey(
                    model_name=_MODEL,
                    world_size=1,
                    worker_id=0,
                    token_bytes=pack_token_ids(tokens),
                    start=0,
                    end=_CHUNK_TOKENS * _CHUNKS,
                    request_id=store_id,
                    cache_salt=_SALT,
                ),
                kv_caches,
                spec.block_ids,
                transfer.create_recorded_event(),
                1,
            ).result(timeout=_RPC_TIMEOUT_S)
            assert stored
            client.end_session(store_id).result(timeout=_RPC_TIMEOUT_S)

            lookup_id = uuid.uuid4().hex
            raw = spec.chunk_hashes
            assert raw is not None
            converted_key = IPCCacheServerKey.from_chunk_hashes(
                model_name=_MODEL,
                world_size=1,
                worker_id=None,
                chunk_hashes=tuple(
                    raw[offset : offset + 32] for offset in range(0, len(raw), 32)
                ),
                start=spec.start,
                end=spec.end,
                request_id=lookup_id,
                cache_salt=spec.cache_salt,
            )
            client.lookup(converted_key, 1).result(timeout=_RPC_TIMEOUT_S)

            assert server.metric_total("lmcache_mp_lookup_hit_tokens_total") == (
                _CHUNK_TOKENS * _CHUNKS
            )
        finally:
            try:
                unregistered = transfer.unregister()
                if unregistered is not None:
                    unregistered.result(timeout=_RPC_TIMEOUT_S)
            finally:
                transfer.close()
                client.close()
                context.term()
