# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from threading import Lock
from typing import Protocol
from urllib.parse import quote, unquote
import builtins

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

PLUGIN_TYPE = "gcs"
_GCS_HANDLE_PREFIX = "gs://"


@dataclass(frozen=True)
class GCSLocation:
    """Parsed Google Cloud Storage location."""

    bucket_name: str
    object_prefix: str


@dataclass(frozen=True)
class GCSConnectorConfig:
    """Resolved GCS connector configuration."""

    plugin_name: str
    bucket_location: GCSLocation
    project: str | None
    credentials_path: str | None
    create_bucket_if_missing: bool
    metadata_cache_ttl_secs: float


@dataclass(frozen=True)
class _CachedObjectMetadata:
    """Cached object size entry with expiration metadata."""

    size_bytes: int
    expires_at: float


class GCSClientInterface(Protocol):
    """Protocol for GCS client implementations used by the connector."""

    def ensure_bucket(self, bucket_name: str) -> None:
        """Ensure the bucket exists."""

    def bucket_exists(self, bucket_name: str) -> bool:
        """Return whether the bucket exists and is reachable."""

    def get_blob_size(self, bucket_name: str, blob_name: str) -> int:
        """Return blob size in bytes or ``0`` when missing."""

    def list_blobs(self, bucket_name: str, prefix: str) -> list[str]:
        """List blob names under ``prefix``."""

    def upload_blob(self, bucket_name: str, blob_name: str, payload: bytes) -> None:
        """Upload blob bytes."""

    def download_blob(self, bucket_name: str, blob_name: str) -> bytes | None:
        """Download blob bytes or ``None`` when missing."""

    def delete_blob(self, bucket_name: str, blob_name: str) -> bool:
        """Delete a blob. Return ``True`` when deleted or already absent."""


class GCSClient(GCSClientInterface):
    """Thin synchronous wrapper around ``google-cloud-storage``."""

    def __init__(
        self,
        *,
        project: str | None = None,
        credentials_path: str | None = None,
    ) -> None:
        # Third Party
        from google.cloud import storage

        client_kwargs: dict[str, object] = {}
        if project:
            client_kwargs["project"] = project
        if credentials_path:
            # Third Party
            from google.oauth2 import service_account

            client_kwargs["credentials"] = (
                service_account.Credentials.from_service_account_file(credentials_path)
            )

        self._client = storage.Client(**client_kwargs)

    def ensure_bucket(self, bucket_name: str) -> None:
        """Create the bucket when it does not already exist."""
        if self._client.lookup_bucket(bucket_name) is not None:
            return
        self._client.create_bucket(bucket_name)

    def bucket_exists(self, bucket_name: str) -> bool:
        """Return whether the bucket currently exists."""
        return self._client.lookup_bucket(bucket_name) is not None

    def get_blob_size(self, bucket_name: str, blob_name: str) -> int:
        """Return blob size in bytes or ``0`` when missing."""
        blob = self._client.bucket(bucket_name).get_blob(blob_name)
        if blob is None:
            return 0
        size = getattr(blob, "size", 0)
        return int(size) if size is not None else 0

    def list_blobs(self, bucket_name: str, prefix: str) -> list[str]:
        """List blob names for the provided prefix."""
        return [
            blob.name for blob in self._client.list_blobs(bucket_name, prefix=prefix)
        ]

    def upload_blob(self, bucket_name: str, blob_name: str, payload: bytes) -> None:
        """Upload blob bytes."""
        blob = self._client.bucket(bucket_name).blob(blob_name)
        blob.upload_from_string(payload, content_type="application/octet-stream")

    def download_blob(self, bucket_name: str, blob_name: str) -> bytes | None:
        """Download blob bytes or ``None`` when missing."""
        blob = self._client.bucket(bucket_name).get_blob(blob_name)
        if blob is None:
            return None
        return blob.download_as_bytes()

    def delete_blob(self, bucket_name: str, blob_name: str) -> bool:
        """Delete a blob and treat missing objects as success."""
        try:
            self._client.bucket(bucket_name).delete_blob(blob_name)
        except Exception as exc:
            if _is_not_found_error(exc):
                return True
            raise
        return True


class GCSConnector(RemoteConnector):
    """LMCache remote connector backed by Google Cloud Storage."""

    def __init__(
        self,
        local_cpu_backend: LocalCPUBackend,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        connector_config: GCSConnectorConfig,
        gcs_client: GCSClientInterface | None = None,
    ) -> None:
        """Initialize the GCS connector."""
        normalized_config = _normalize_save_chunk_meta_config(config)
        super().__init__(normalized_config, metadata)

        if self.save_chunk_meta:
            raise ValueError("save_chunk_meta must be False for gcs")
        if config.save_unfull_chunk:
            raise ValueError("save_unfull_chunk must be False for gcs")

        self.local_cpu_backend = local_cpu_backend
        self.plugin_name = connector_config.plugin_name
        self.bucket_name = connector_config.bucket_location.bucket_name
        self.object_prefix = connector_config.bucket_location.object_prefix
        self.create_bucket_if_missing = connector_config.create_bucket_if_missing
        self.metadata_cache_ttl_secs = connector_config.metadata_cache_ttl_secs

        self._gcs_client = gcs_client or GCSClient(
            project=connector_config.project,
            credentials_path=connector_config.credentials_path,
        )
        self._metadata_cache: dict[str, _CachedObjectMetadata] = {}
        self._metadata_cache_lock = Lock()
        self._bucket_create_lock = Lock()
        self._bucket_create_checked = False

        logger.info(
            "Initialized GCSConnector for bucket %s with prefix '%s'",
            self.bucket_name,
            self.object_prefix,
        )

    async def exists(self, key: CacheEngineKey) -> bool:
        """Return whether a full LMCache chunk exists for ``key``."""
        return self.exists_sync(key)

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Synchronously return whether a full LMCache chunk exists."""
        object_size = self._get_object_size_bytes(key.to_string())
        return object_size == self.full_chunk_size_bytes

    async def get(self, key: CacheEngineKey) -> MemoryObj | None:
        """Retrieve the full chunk associated with ``key``."""
        memory_objs = await self.batched_get([key])
        return memory_objs[0]

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Store a full chunk in GCS."""
        await self.batched_put([key], [memory_obj])

    async def list(self) -> builtins.list[str]:
        """List LMCache keys currently stored under this connector prefix."""
        return self._list_sync()

    async def close(self) -> None:
        """Release connector-local resources."""
        with self._metadata_cache_lock:
            self._metadata_cache.clear()

    def support_ping(self) -> bool:
        """Report support for lightweight connectivity checks."""
        return True

    async def ping(self) -> int:
        """Check access to the configured bucket."""
        return self._ping_sync()

    def support_batched_put(self) -> bool:
        """Report support for batched uploads."""
        return True

    async def batched_put(
        self,
        keys: builtins.list[CacheEngineKey],
        memory_objs: builtins.list[MemoryObj],
    ) -> None:
        """Upload multiple full chunks in sequence."""
        if len(keys) != len(memory_objs):
            raise ValueError(
                "keys and memory_objs must have the same length for batched_put"
            )

        upload_entries: builtins.list[tuple[str, bytes]] = []
        try:
            for key, memory_obj in zip(keys, memory_objs, strict=True):
                key_str = key.to_string()
                self._validate_full_chunk_for_upload(key_str, memory_obj)
                upload_entries.append((key_str, bytes(memory_obj.byte_array)))

            self._batched_put_sync(upload_entries)
        finally:
            for memory_obj in memory_objs:
                memory_obj.ref_count_down()

    def support_batched_get(self) -> bool:
        """Report support for batch downloads."""
        return True

    async def batched_get(
        self,
        keys: builtins.list[CacheEngineKey],
    ) -> builtins.list[MemoryObj | None]:
        """Download multiple chunks while preserving input order."""
        if not keys:
            return []

        key_strings = [key.to_string() for key in keys]
        object_sizes = self._resolve_object_sizes(key_strings)
        results: builtins.list[MemoryObj | None] = [None] * len(keys)

        try:
            for index, (key_str, object_size) in enumerate(
                zip(key_strings, object_sizes, strict=False)
            ):
                if object_size == 0:
                    continue
                if object_size != self.full_chunk_size_bytes:
                    logger.error(
                        "Size mismatch for %s: GCS has %d bytes, expected %d bytes. "
                        "Rejecting the load because gcs only supports full chunks.",
                        key_str,
                        object_size,
                        self.full_chunk_size_bytes,
                    )
                    continue

                data = self._gcs_client.download_blob(
                    self.bucket_name,
                    self._key_string_to_object_path(key_str),
                )
                if data is None:
                    self._set_cached_object_size(key_str, 0)
                    continue
                if len(data) != self.full_chunk_size_bytes:
                    logger.error(
                        "Downloaded object for %s has %d bytes, expected %d bytes. "
                        "Rejecting the load because gcs only supports full chunks.",
                        key_str,
                        len(data),
                        self.full_chunk_size_bytes,
                    )
                    self._set_cached_object_size(key_str, len(data))
                    continue

                memory_obj = self.local_cpu_backend.allocate(
                    self.meta_shapes,
                    self.meta_dtypes,
                    self.meta_fmt,
                )
                if memory_obj is None:
                    logger.debug("Memory allocation failed while downloading from gcs.")
                    continue

                try:
                    buffer = memory_obj.byte_array.cast("B")
                    if len(buffer) < len(data):
                        raise RuntimeError(
                            "Allocated buffer is smaller than downloaded gcs object"
                        )
                    buffer[: len(data)] = data
                    results[index] = memory_obj
                except Exception:
                    memory_obj.ref_count_down()
                    raise
        except Exception:
            for existing in results:
                if existing is not None:
                    existing.ref_count_down()
            raise

        return results

    def support_batched_contains(self) -> bool:
        """Report support for synchronous prefix contains checks."""
        return True

    def batched_contains(self, keys: builtins.list[CacheEngineKey]) -> int:
        """Return the number of consecutive prefix keys that exist as full chunks."""
        key_strings = [key.to_string() for key in keys]
        object_sizes = self._resolve_object_sizes(key_strings)
        hit_count = 0
        for object_size in object_sizes:
            if object_size != self.full_chunk_size_bytes:
                return hit_count
            hit_count += 1
        return hit_count

    def support_batched_async_contains(self) -> bool:
        """Report support for async prefix contains checks."""
        return True

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: builtins.list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """Asynchronously return the number of consecutive prefix hits."""
        del lookup_id
        del pin
        return self.batched_contains(keys)

    def support_batched_get_non_blocking(self) -> bool:
        """Report support for non-blocking batch loads."""
        return True

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: builtins.list[CacheEngineKey],
    ) -> builtins.list[MemoryObj]:
        """Return the successful prefix of ``batched_get`` results."""
        del lookup_id
        results = await self.batched_get(keys)
        prefix_results: builtins.list[MemoryObj] = []
        found_failure = False
        for result in results:
            if found_failure:
                if result is not None:
                    result.ref_count_down()
                continue
            if result is None:
                found_failure = True
                continue
            prefix_results.append(result)
        return prefix_results

    def remove_sync(self, key: CacheEngineKey) -> bool:
        """Synchronously remove a GCS object for ``key``."""
        key_str = key.to_string()
        try:
            removed = self._gcs_client.delete_blob(
                self.bucket_name,
                self._key_string_to_object_path(key_str),
            )
        except Exception as exc:
            logger.error("Failed to delete %s from gcs: %s", key_str, exc)
            return False

        if removed:
            self._set_cached_object_size(key_str, 0)
        return removed

    def __repr__(self) -> str:
        return (
            f"<GCSConnector bucket_name={self.bucket_name} "
            f"prefix={self.object_prefix!r}>"
        )

    def _ping_sync(self) -> int:
        """Perform the synchronous bucket health check used by ``ping``."""
        try:
            return 0 if self._gcs_client.bucket_exists(self.bucket_name) else 1
        except Exception as exc:
            logger.warning("Failed to ping gcs bucket %s: %s", self.bucket_name, exc)
            return 1

    def _batched_put_sync(
        self,
        upload_entries: Sequence[tuple[str, bytes]],
    ) -> None:
        """Upload all provided chunks."""
        self._ensure_bucket_for_writes()
        uploaded_key_strings: builtins.list[str] = []
        try:
            for key_str, payload in upload_entries:
                self._gcs_client.upload_blob(
                    self.bucket_name,
                    self._key_string_to_object_path(key_str),
                    payload,
                )
                uploaded_key_strings.append(key_str)
                self._set_cached_object_size(key_str, self.full_chunk_size_bytes)
        except Exception:
            if uploaded_key_strings:
                refreshed_sizes = self._fetch_object_sizes_sync(uploaded_key_strings)
                for key_str, size in refreshed_sizes.items():
                    self._set_cached_object_size(key_str, size)
            raise

    def _ensure_bucket_for_writes(self) -> None:
        """Create the bucket on demand when configured to do so."""
        if not self.create_bucket_if_missing or self._bucket_create_checked:
            return

        with self._bucket_create_lock:
            if self._bucket_create_checked:
                return
            self._gcs_client.ensure_bucket(self.bucket_name)
            self._bucket_create_checked = True

    def _validate_full_chunk_for_upload(
        self,
        key_str: str,
        memory_obj: MemoryObj,
    ) -> None:
        """Reject partial or metadata-bearing uploads for the conservative MVP."""
        physical_size = memory_obj.get_physical_size()
        if physical_size != self.full_chunk_size_bytes:
            raise ValueError(
                f"Cannot upload {key_str}: chunk size {physical_size} bytes does not "
                f"match expected full chunk size {self.full_chunk_size_bytes} bytes. "
                "Partial/unfull chunks are not supported by gcs."
            )

    def _list_sync(self) -> builtins.list[str]:
        """Return LMCache key strings discovered under the configured prefix."""
        try:
            blob_names = self._gcs_client.list_blobs(
                self.bucket_name,
                self.object_prefix,
            )
        except Exception as exc:
            if _is_not_found_error(exc):
                return []
            raise

        keys: builtins.list[str] = []
        for blob_name in blob_names:
            relative_path = self._relative_object_path(blob_name)
            if relative_path is None or "/" in relative_path:
                continue
            key_str = decode_gcs_object_name(relative_path)
            keys.append(key_str)
            self._set_cached_object_size(
                key_str,
                self._gcs_client.get_blob_size(self.bucket_name, blob_name),
            )
        return keys

    def _get_object_size_bytes(self, key_str: str) -> int:
        """Return the cached or fetched size for a specific LMCache key string."""
        cached_size = self._get_cached_object_size(key_str)
        if cached_size is not None:
            return cached_size

        object_sizes = self._fetch_object_sizes_sync([key_str])
        object_size = object_sizes.get(key_str, 0)
        self._set_cached_object_size(key_str, object_size)
        return object_size

    def _resolve_object_sizes(self, key_strings: Sequence[str]) -> builtins.list[int]:
        """Resolve cached and uncached object sizes while preserving order."""
        cached_results: dict[str, int] = {}
        uncached_key_strings: builtins.list[str] = []
        for key_str in key_strings:
            cached_size = self._get_cached_object_size(key_str)
            if cached_size is None:
                uncached_key_strings.append(key_str)
            else:
                cached_results[key_str] = cached_size

        fetched_results = self._fetch_object_sizes_sync(uncached_key_strings)
        for key_str, size in fetched_results.items():
            self._set_cached_object_size(key_str, size)

        return [
            cached_results.get(key_str, fetched_results.get(key_str, 0))
            for key_str in key_strings
        ]

    def _fetch_object_sizes_sync(
        self,
        key_strings: Sequence[str],
    ) -> dict[str, int]:
        """Fetch object sizes for the requested key strings."""
        results: dict[str, int] = {}
        for key_str in key_strings:
            try:
                results[key_str] = self._gcs_client.get_blob_size(
                    self.bucket_name,
                    self._key_string_to_object_path(key_str),
                )
            except Exception as exc:
                if _is_not_found_error(exc):
                    results[key_str] = 0
                    continue
                raise
        return results

    def _key_string_to_object_path(self, key_str: str) -> str:
        """Convert an LMCache key string into the stored GCS object path."""
        object_name = encode_gcs_object_name(key_str)
        if not self.object_prefix:
            return object_name
        return f"{self.object_prefix}/{object_name}"

    def _relative_object_path(self, object_path: str) -> str | None:
        """Strip the configured prefix from an object path if it matches."""
        if not self.object_prefix:
            return object_path

        prefix_with_separator = f"{self.object_prefix}/"
        if object_path == self.object_prefix:
            return None
        if not object_path.startswith(prefix_with_separator):
            return None
        return object_path[len(prefix_with_separator) :]

    def _get_cached_object_size(self, key_str: str) -> int | None:
        """Return the non-expired cached object size for ``key_str``."""
        # Standard
        import time

        with self._metadata_cache_lock:
            cached_entry = self._metadata_cache.get(key_str)
            if cached_entry is None:
                return None
            if cached_entry.expires_at < time.monotonic():
                self._metadata_cache.pop(key_str, None)
                return None
            return cached_entry.size_bytes

    def _set_cached_object_size(self, key_str: str, size_bytes: int) -> None:
        """Cache the object size for ``key_str``."""
        # Standard
        import time

        with self._metadata_cache_lock:
            self._metadata_cache[key_str] = _CachedObjectMetadata(
                size_bytes=size_bytes,
                expires_at=time.monotonic() + self.metadata_cache_ttl_secs,
            )


def parse_gcs_bucket_handle(bucket_handle: str) -> GCSLocation:
    """Parse a GCS bucket handle into bucket and prefix components.

    Args:
        bucket_handle: Bucket handle in ``gs://<bucket>[/<prefix>]`` form.

    Returns:
        A :class:`GCSLocation` containing the bucket name and optional object
        prefix.

    Raises:
        ValueError: If ``bucket_handle`` does not start with ``gs://`` or does
            not include a bucket name.
    """
    normalized_handle = bucket_handle.strip()
    if not normalized_handle.startswith(_GCS_HANDLE_PREFIX):
        raise ValueError("bucket_handle must start with 'gs://' for the gcs plugin")

    path = normalized_handle[len(_GCS_HANDLE_PREFIX) :].strip("/")
    path_parts = [part for part in path.split("/") if part]
    if not path_parts:
        raise ValueError("bucket_handle must be in the form 'gs://<bucket>[/<prefix>]'")

    return GCSLocation(
        bucket_name=path_parts[0],
        object_prefix="/".join(path_parts[1:]),
    )


def resolve_gcs_connector_config(
    config: LMCacheEngineConfig,
    plugin_name: str,
) -> GCSConnectorConfig:
    """Resolve the plugin-scoped configuration for a GCS connector.

    Args:
        config: Engine configuration containing ``extra_config`` overrides.
        plugin_name: Fully-qualified remote storage plugin name.

    Returns:
        A :class:`GCSConnectorConfig` built from the plugin-specific settings.

    Raises:
        ValueError: If the required bucket handle setting is missing or invalid.
    """
    extra_config = config.extra_config or {}
    config_prefix = f"remote_storage_plugin.{plugin_name}"

    bucket_handle_obj = extra_config.get(f"{config_prefix}.bucket_handle")
    if not isinstance(bucket_handle_obj, str) or not bucket_handle_obj:
        raise ValueError(
            f"GCS connector '{plugin_name}' requires '{config_prefix}.bucket_handle'"
        )

    project_obj = extra_config.get(f"{config_prefix}.project")
    project = project_obj if isinstance(project_obj, str) and project_obj else None

    credentials_path_obj = extra_config.get(f"{config_prefix}.credentials_path")
    credentials_path = (
        credentials_path_obj
        if isinstance(credentials_path_obj, str) and credentials_path_obj
        else None
    )

    return GCSConnectorConfig(
        plugin_name=plugin_name,
        bucket_location=parse_gcs_bucket_handle(bucket_handle_obj),
        project=project,
        credentials_path=credentials_path,
        create_bucket_if_missing=_coerce_bool(
            extra_config.get(f"{config_prefix}.create_bucket_if_missing", False)
        ),
        metadata_cache_ttl_secs=_coerce_float(
            extra_config.get(f"{config_prefix}.metadata_cache_ttl_secs", 30.0)
        ),
    )


def encode_gcs_object_name(key_str: str) -> str:
    """Encode a serialized LMCache key into a reversible GCS object name.

    Args:
        key_str: Serialized LMCache key string.

    Returns:
        A percent-encoded object name safe to use in GCS paths.
    """
    return quote(key_str, safe="")


def decode_gcs_object_name(object_name: str) -> str:
    """Decode a reversible GCS object name back into an LMCache key string.

    Args:
        object_name: Percent-encoded GCS object name.

    Returns:
        The decoded serialized LMCache key string.
    """
    return unquote(object_name)


def _normalize_save_chunk_meta_config(
    config: LMCacheEngineConfig,
) -> LMCacheEngineConfig:
    """Clone config and default ``save_chunk_meta`` to ``False`` for gcs."""
    if config.extra_config is not None and "save_chunk_meta" in config.extra_config:
        return config

    normalized_config = copy(config)
    normalized_extra_config = (
        dict(config.extra_config) if config.extra_config is not None else {}
    )
    normalized_extra_config["save_chunk_meta"] = False
    normalized_config.extra_config = normalized_extra_config
    return normalized_config


def _coerce_bool(value: object) -> bool:
    """Coerce common truthy config values into ``bool``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_float(value: object) -> float:
    """Coerce numeric config values into ``float``."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError(f"Cannot coerce {value!r} to float")


def _is_not_found_error(exc: Exception) -> bool:
    """Best-effort detection for GCS not-found errors without a hard import."""
    return exc.__class__.__name__ == "NotFound"
