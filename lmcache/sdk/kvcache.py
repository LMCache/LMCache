# SPDX-License-Identifier: Apache-2.0
"""SDK helpers for storing and retrieving KV cache tensors over HTTP."""

# Standard
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast
import math

# Third Party
import httpx
import torch

# First Party
from lmcache.v1.multiprocess.http_apis.kv_protocol import (
    PROTOCOL_VERSION,
    STREAM_MEDIA_TYPE,
    RetrieveRequest,
    StoreManifest,
    decode_retrieve_manifest,
    decode_retrieve_shard,
    encode_retrieve_request,
    encode_store_chunk,
    encode_store_manifest,
    iter_decode_frames,
)


class KVCacheSDKError(RuntimeError):
    """Raised when an SDK KV-cache HTTP operation fails."""


@dataclass(frozen=True)
class KVCachePackage:
    """In-memory KV cache tensor plus routing metadata.

    Args:
        kv: 4-D KV tensor in canonical ``[2, num_layers, num_tokens, hidden_dim]``
            layout.
        model_name: Registered LMCache model name for the KV tensor.
        tokens: Token IDs addressed by ``kv``.
        cache_salt: Optional per-namespace isolation salt.
    """

    kv: torch.Tensor
    model_name: str
    tokens: list[int]
    cache_salt: str = ""


@dataclass(frozen=True)
class StoreResult:
    """Result returned by :func:`store`."""

    total_tokens: int
    total_chunks: int
    stored_tokens: int
    stored_chunks: int


@dataclass(frozen=True)
class RetrieveResult:
    """Result returned by :func:`retrieve`."""

    total_tokens: int
    total_chunks: int
    hit_tokens: int
    hit_chunks: int
    package: KVCachePackage


def store(
    package: KVCachePackage,
    url: str,
    *,
    model_name: str = "",
    tokens: Sequence[int] = (),
    cache_salt: str = "",
    timeout: float = 60.0,
) -> StoreResult:
    """Store an in-memory KV cache package into an LMCache HTTP server.

    Args:
        package: In-memory KV package. The tensor must use canonical KV_2LTD
            layout ``[2, num_layers, num_tokens, hidden_dim]``.
        url: Base URL of the LMCache MP HTTP server.
        model_name: Optional model-name override. If empty, the value is read
            from the package metadata.
        tokens: Optional token sequence override. If empty, the sequence is
            read from the package metadata.
        cache_salt: Optional cache-salt override. If empty, the value is read
            from the package metadata when present.
        timeout: HTTP timeout in seconds.

    Returns:
        Store result metadata reported by the server.

    Raises:
        KVCacheSDKError: If the tensor, metadata, or HTTP request is invalid.
    """
    resolved_package = _resolve_package(package, model_name, tokens, cache_salt)
    chunk_size = _fetch_chunk_size(url, timeout)
    _validate_store_package(resolved_package, chunk_size)
    response = httpx.post(
        f"{_normalize_url(url)}/api/kv/store",
        content=_iter_store_frames(resolved_package, chunk_size),
        headers={"Content-Type": STREAM_MEDIA_TYPE},
        timeout=timeout,
    )
    _raise_for_status(response)
    body = _json_object(response)
    return StoreResult(
        total_tokens=_json_int(body, "total_tokens"),
        total_chunks=_json_int(body, "total_chunks"),
        stored_tokens=_json_int(body, "stored_tokens"),
        stored_chunks=_json_int(body, "stored_chunks"),
    )


def retrieve(
    url: str,
    *,
    model_name: str,
    tokens: Sequence[int],
    cache_salt: str = "",
    timeout: float = 60.0,
) -> RetrieveResult:
    """Retrieve KV cache bytes from an LMCache HTTP server into memory.

    Args:
        url: Base URL of the LMCache MP HTTP server.
        model_name: Registered model name to retrieve from.
        tokens: Token sequence to retrieve.
        cache_salt: Optional per-namespace isolation salt.
        timeout: HTTP timeout in seconds.

    Returns:
        Retrieve metadata and the assembled in-memory KV package.

    Raises:
        KVCacheSDKError: If the HTTP request or stream is invalid.
    """
    request = RetrieveRequest(
        model_name=model_name,
        tokens=list(tokens),
        cache_salt=cache_salt,
        protocol_version=PROTOCOL_VERSION,
    )
    with httpx.stream(
        "POST",
        f"{_normalize_url(url)}/api/kv/retrieve",
        content=encode_retrieve_request(request),
        headers={
            "Accept": STREAM_MEDIA_TYPE,
            "Content-Type": "application/json",
        },
        timeout=timeout,
    ) as response:
        _raise_for_status(response)
        frames = iter_decode_frames(response.iter_bytes())
        try:
            manifest = decode_retrieve_manifest(next(frames))
        except StopIteration as exc:
            raise KVCacheSDKError(
                "retrieve response did not include a manifest"
            ) from exc
        dtype = _dtype_from_name(manifest.dtype)
        kv = torch.empty(manifest.shape, dtype=dtype)
        expected_shards = {
            (chunk_index, worker_id)
            for chunk_index in range(manifest.hit_chunks)
            for worker_id in range(manifest.world_size)
        }
        seen_shards: set[tuple[int, int]] = set()
        expected_payload_bytes = math.prod(manifest.shard_shape) * dtype.itemsize
        for frame in frames:
            chunk_index, worker_id, payload = decode_retrieve_shard(frame)
            shard_key = (chunk_index, worker_id)
            if shard_key not in expected_shards:
                raise KVCacheSDKError(
                    "retrieve response included unexpected shard "
                    f"chunk={chunk_index}, worker={worker_id}"
                )
            if shard_key in seen_shards:
                raise KVCacheSDKError(
                    "retrieve response included duplicate shard "
                    f"chunk={chunk_index}, worker={worker_id}"
                )
            if len(payload) != expected_payload_bytes:
                raise KVCacheSDKError(
                    f"retrieve shard {shard_key} has {len(payload)} bytes, "
                    f"expected {expected_payload_bytes}"
                )
            shard = torch.frombuffer(
                bytearray(payload),
                dtype=dtype,
            ).reshape(manifest.shard_shape)
            t_start = chunk_index * manifest.chunk_size
            t_end = t_start + manifest.chunk_size
            d_per_worker = manifest.shard_shape[3]
            d_start = worker_id * d_per_worker
            d_end = d_start + d_per_worker
            kv[:, :, t_start:t_end, d_start:d_end] = shard
            seen_shards.add(shard_key)
        missing_shards = expected_shards - seen_shards
        if missing_shards:
            raise KVCacheSDKError(
                f"retrieve response missing {len(missing_shards)} shard frames"
            )

    package = KVCachePackage(
        kv=kv,
        model_name=manifest.model_name,
        tokens=list(tokens)[: manifest.hit_tokens],
        cache_salt=cache_salt,
    )
    return RetrieveResult(
        total_tokens=manifest.total_tokens,
        total_chunks=manifest.total_chunks,
        hit_tokens=manifest.hit_tokens,
        hit_chunks=manifest.hit_chunks,
        package=package,
    )


def _normalize_url(url: str) -> str:
    """Normalize an LMCache HTTP server base URL."""
    stripped = url.strip()
    if not stripped.startswith(("http://", "https://")):
        stripped = f"http://{stripped}"
    return stripped.rstrip("/")


def _resolve_package(
    package: KVCachePackage,
    model_name: str,
    tokens: Sequence[int],
    cache_salt: str,
) -> KVCachePackage:
    """Apply metadata overrides to an in-memory package."""
    if not isinstance(package, KVCachePackage):
        raise KVCacheSDKError("package must be a KVCachePackage")
    resolved_model_name = model_name or package.model_name
    resolved_tokens = list(tokens) if tokens else package.tokens
    resolved_cache_salt = cache_salt or package.cache_salt
    if not resolved_model_name:
        raise KVCacheSDKError("model_name must be provided or stored in the package")
    if not resolved_tokens:
        raise KVCacheSDKError("tokens must be provided or stored in the package")
    return KVCachePackage(
        kv=package.kv.detach().cpu(),
        model_name=resolved_model_name,
        tokens=resolved_tokens,
        cache_salt=resolved_cache_salt,
    )


def _validate_store_package(package: KVCachePackage, chunk_size: int) -> None:
    """Validate tensor shape against token metadata and server chunk size."""
    if package.kv.ndim != 4:
        raise KVCacheSDKError(
            f"kv tensor must be 4-D [2, L, T, D], got shape {tuple(package.kv.shape)}"
        )
    total_tokens = (len(package.tokens) // chunk_size) * chunk_size
    if total_tokens == 0:
        raise KVCacheSDKError("tokens must contain at least one complete chunk")
    if package.kv.shape[2] != total_tokens:
        raise KVCacheSDKError(
            f"kv tensor token dim {package.kv.shape[2]} does not match "
            f"complete token prefix {total_tokens}"
        )


def _iter_store_frames(package: KVCachePackage, chunk_size: int) -> Iterable[bytes]:
    """Yield protocol frames for storing a KV package."""
    kv = package.kv.contiguous()
    shape = tuple(int(dim) for dim in kv.shape)
    if len(shape) != 4:
        raise KVCacheSDKError(f"kv tensor must be 4-D, got shape {shape}")
    yield encode_store_manifest(
        StoreManifest(
            model_name=package.model_name,
            tokens=package.tokens,
            cache_salt=package.cache_salt,
            shape=(shape[0], shape[1], shape[2], shape[3]),
            dtype=str(kv.dtype),
        )
    )
    total_chunks = shape[2] // chunk_size
    for chunk_index in range(total_chunks):
        start = chunk_index * chunk_size
        end = start + chunk_size
        payload = (
            kv[:, :, start:end, :].contiguous().view(torch.uint8).numpy().tobytes()
        )
        yield encode_store_chunk(chunk_index, payload)


def _fetch_chunk_size(url: str, timeout: float) -> int:
    """Fetch the server chunk size from ``/api/status``."""
    response = httpx.get(f"{_normalize_url(url)}/api/status", timeout=timeout)
    _raise_for_status(response)
    body = _json_object(response)
    return _json_int(body, "chunk_size")


def _raise_for_status(response: httpx.Response) -> None:
    """Raise ``KVCacheSDKError`` for non-success HTTP responses."""
    if response.status_code < 400:
        return
    raise KVCacheSDKError(
        f"{response.request.method} {response.request.url} failed "
        f"with HTTP {response.status_code}: {response.text}"
    )


def _json_object(response: httpx.Response) -> Mapping[str, object]:
    """Parse a JSON response object."""
    try:
        decoded: object = response.json()
    except ValueError as exc:
        raise KVCacheSDKError("server returned invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise KVCacheSDKError("server JSON response must be an object")
    return cast(Mapping[str, object], decoded)


def _json_int(body: Mapping[str, object], key: str) -> int:
    """Read an integer field from a JSON object."""
    value = body.get(key)
    if not isinstance(value, int):
        raise KVCacheSDKError(f"server JSON field {key!r} must be an int")
    return value


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    """Resolve a ``torch`` dtype string such as ``"torch.float16"``."""
    normalized = dtype_name.removeprefix("torch.")
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype):
        raise KVCacheSDKError(f"unsupported KV dtype {dtype_name!r}")
    return dtype
