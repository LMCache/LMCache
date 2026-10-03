# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni encoder results backed by the existing LMCache EC engine.

A complete nested result is one EC entry so an eviction cannot produce a partial
hit (for example image embeddings without deepstack features or audio lengths).
The payload uses safetensors and a JSON tree, never pickle.
"""

# Future
from __future__ import annotations

# Standard
import hashlib
import json
import math
import struct
import threading
from typing import TYPE_CHECKING, Any

# Third Party
import torch
from safetensors.torch import load, save

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.ec_engine import ECCacheEngine


def _pack(value: Any) -> bytes:
    tensors: dict[str, torch.Tensor] = {}

    def visit(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            name = str(len(tensors))
            tensors[name] = item.detach().to(device="cpu").contiguous()
            return ["tensor", name]
        if isinstance(item, dict) and all(isinstance(k, str) for k in item):
            return ["dict", [[k, visit(item[k])] for k in sorted(item)]]
        if isinstance(item, (list, tuple)):
            return [
                "tuple" if isinstance(item, tuple) else "list",
                [visit(v) for v in item],
            ]
        if item is None or isinstance(item, (str, bool, int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("non-finite scalar in encoder result")
            return ["scalar", item]
        raise TypeError(f"unsupported encoder cache value: {type(item).__name__}")

    schema = json.dumps(visit(value), separators=(",", ":"), allow_nan=False)
    return save(tensors, metadata={"sglang_omni_v1": schema})


def _unpack(payload: bytes) -> Any:
    header_size = struct.unpack("<Q", payload[:8])[0]
    header = json.loads(payload[8 : 8 + header_size])
    schema = json.loads(header["__metadata__"]["sglang_omni_v1"])
    tensors = load(payload)

    def visit(node: Any) -> Any:
        kind, item = node
        if kind == "tensor":
            return tensors[item]
        if kind == "dict":
            return {k: visit(v) for k, v in item}
        if kind == "list":
            return [visit(v) for v in item]
        if kind == "tuple":
            return tuple(visit(v) for v in item)
        if kind == "scalar":
            return item
        raise ValueError(f"unknown encoder payload node: {kind}")

    return visit(schema)


class SGLangOmniEncoderCache:
    """Implement Omni's encoder ``get``/``put`` contract through an EC engine.

    Args:
        engine: A dedicated EC engine with ``encoder_dtype=torch.uint8``. This
            adapter owns it and closes it on ``close()``.
        namespace: Nonempty model/revision/stage/preprocessing identity. Change
            it when weights, dtype or encoder computation changes.

    Raises:
        ValueError: If namespace is empty.

    Notes:
        Returned payload tensors reside on CPU, like Omni's default encoder
        StageOutputCache. One serialized uint8 tensor is stored per complete
        result. This is experimental in-process EC support, not MP support.
    """

    def __init__(self, engine: ECCacheEngine, namespace: str) -> None:
        if not namespace.strip():
            raise ValueError("encoder cache namespace must not be empty")
        self.engine = engine
        self.namespace = namespace
        self.closed = False
        self.lock = threading.RLock()

    def key_for_inputs(self, model_inputs: dict[str, Any]) -> str:
        """Hash prepared inputs, including tensor content, shapes and dtypes.

        Args:
            model_inputs: Encoder-ready tensors and scalar metadata, excluding
                request IDs, URLs and transport-only metadata.

        Returns:
            A content digest; input dictionary order does not affect the key.

        Raises:
            TypeError: For values outside the tensor/JSON-container contract.
            ValueError: For non-finite scalar metadata.

        This CPU hashing cost is part of the opt-in cache path and must be
        included in end-to-end measurements. A URL alone is not a content key.
        """
        return hashlib.sha256(_pack(model_inputs)).hexdigest()

    def get(self, key: str | None) -> Any | None:
        """Return a complete CPU encoder result, or None on a miss/disabled key.

        Args:
            key: Input content key, or None to bypass caching.

        Raises:
            RuntimeError: If the cache has been closed or EC retrieval fails.
            ValueError: If a stored payload cannot be decoded.
        """
        if key is None:
            return None
        with self.lock:
            if self.closed:
                raise RuntimeError("encoder cache is closed")
            tensor = self.engine.get(self.cache_key(key), device="cpu")
            if tensor is None:
                return None
            if tensor.dtype != torch.uint8 or tensor.ndim != 2 or tensor.shape[0] != 1:
                raise ValueError("invalid serialized encoder result")
            return _unpack(tensor.contiguous().numpy().tobytes())

    def put(self, key: str | None, data: Any) -> bool:
        """Store a complete encoder result; False means bypass/allocation failure.

        Args:
            key: Input content key, or None to bypass caching.
            data: Nested tensors, string-keyed dictionaries, lists, tuples and
                JSON scalars. Tensor dtypes/shapes and container types persist.

        Returns:
            Whether EC accepted the store; this does not assert disk durability.

        Raises:
            TypeError: For unsupported result types.
            ValueError: For non-finite scalar metadata.
            RuntimeError: If the cache has been closed or EC storage fails.
        """
        if key is None:
            return False
        payload = _pack(data)
        tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(1, -1)
        with self.lock:
            if self.closed:
                raise RuntimeError("encoder cache is closed")
            return self.engine.put(self.cache_key(key), tensor)

    def cache_key(self, key: str) -> str:
        """Return an unambiguous versioned EC key for this namespace/input pair."""
        return json.dumps(
            ["sglang-omni-encoder-v1", self.namespace, key], separators=(",", ":")
        )

    def close(self) -> None:
        """Close the owned EC engine once, after any in-flight cache operation."""
        with self.lock:
            if not self.closed:
                self.closed = True
                self.engine.close()


def create_encoder_cache(config_file: str, namespace: str) -> SGLangOmniEncoderCache:
    """Create a dedicated, worker-owned SGLang-Omni EC cache.

    Args:
        config_file: LMCache YAML with local CPU staging and a storage backend.
        namespace: Model/revision/stage/dtype/preprocessing identity.

    Returns:
        An owned cache. Call close from the encoder scheduler's shutdown hook.

    Raises:
        ValueError: For an empty namespace or invalid storage configuration.
        OSError: If the config file cannot be read.

    The dummy KV geometry is only allocator metadata; this cache never stores KV
    or connects to a vLLM engine. Existing EC uses sentinel rank/world size one.
    """
    # First Party
    from lmcache.integration.sglang.utils import lmcache_get_config
    from lmcache.v1.ec_engine import ECCacheEngine
    from lmcache.v1.metadata import LMCacheMetadata

    if not namespace.strip():
        raise ValueError("encoder cache namespace must not be empty")
    config = lmcache_get_config(config_file)
    metadata = LMCacheMetadata(
        model_name="sglang-omni-ec-" + hashlib.sha256(namespace.encode()).hexdigest(),
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.uint8,
        kv_shape=(1, 2, config.chunk_size, 1, 1),
        role="worker",
        chunk_size=config.chunk_size,
    )
    engine = ECCacheEngine(config, metadata, encoder_dtype=torch.uint8)
    return SGLangOmniEncoderCache(engine, namespace)
