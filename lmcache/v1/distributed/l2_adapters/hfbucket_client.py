# SPDX-License-Identifier: Apache-2.0
"""Shared Hugging Face Buckets client primitives for L2 storage."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote, unquote

# Third Party
from huggingface_hub import HfApi
from packaging.version import InvalidVersion, Version
import huggingface_hub

_HFBUCKET_HANDLE_PREFIX = "hf://buckets/"
_MIN_HUGGINGFACE_HUB_VERSION = Version("1.5.0")


@dataclass(frozen=True)
class HFBucketLocation:
    """Parsed Hugging Face bucket location.

    Args:
        bucket_id: Hugging Face bucket identifier in ``namespace/bucket`` format.
        object_prefix: Optional object prefix inside the bucket.
    """

    bucket_id: str
    object_prefix: str


class HFBucketClientInterface(Protocol):
    """Protocol for synchronous Hugging Face bucket clients."""

    def create_bucket(self, bucket_id: str) -> None:
        """Create a bucket if needed."""

    def bucket_info(self, bucket_id: str) -> object:
        """Fetch lightweight bucket metadata."""

    def get_paths_info(self, bucket_id: str, paths: Sequence[str]) -> list[object]:
        """Fetch exact metadata for bucket paths."""

    def list_tree(self, bucket_id: str, prefix: str) -> list[object]:
        """List bucket entries under the provided prefix."""

    def upload_files(
        self,
        bucket_id: str,
        add: Sequence[tuple[bytes, str]],
    ) -> None:
        """Upload files to a bucket in one batch call."""

    def download_files(
        self,
        bucket_id: str,
        files: Sequence[tuple[str, str]],
    ) -> None:
        """Download files from a bucket to local paths."""

    def delete_files(self, bucket_id: str, delete: Sequence[str]) -> None:
        """Delete files from a bucket in one batch call."""


class HFBucketClient(HFBucketClientInterface):
    """Thin synchronous wrapper around ``huggingface_hub`` bucket APIs."""

    def __init__(self, token: str | None) -> None:
        _validate_huggingface_hub_support()
        self._api = HfApi(token=token)

    def create_bucket(self, bucket_id: str) -> None:
        """Create the bucket if it does not already exist."""
        self._api.create_bucket(bucket_id, exist_ok=True)

    def bucket_info(self, bucket_id: str) -> object:
        """Return bucket metadata for auth and health checks."""
        return self._api.bucket_info(bucket_id)

    def get_paths_info(self, bucket_id: str, paths: Sequence[str]) -> list[object]:
        """Return exact metadata entries for the requested paths."""
        if not paths:
            return []
        return list(self._api.get_bucket_paths_info(bucket_id, list(paths)))

    def list_tree(self, bucket_id: str, prefix: str) -> list[object]:
        """Return recursive bucket tree entries under ``prefix``."""
        return list(
            self._api.list_bucket_tree(bucket_id, recursive=True, prefix=prefix)
        )

    def upload_files(
        self,
        bucket_id: str,
        add: Sequence[tuple[bytes, str]],
    ) -> None:
        """Upload files in a single batch operation."""
        self._api.batch_bucket_files(bucket_id, add=list(add))

    def download_files(
        self,
        bucket_id: str,
        files: Sequence[tuple[str, str]],
    ) -> None:
        """Download files in a single batch operation."""
        self._api.download_bucket_files(bucket_id, files=list(files))

    def delete_files(self, bucket_id: str, delete: Sequence[str]) -> None:
        """Delete files in a single batch operation."""
        self._api.batch_bucket_files(bucket_id, delete=list(delete))


def parse_hfbucket_handle(bucket_handle: str) -> HFBucketLocation:
    """Parse a Hugging Face bucket handle into bucket id and prefix.

    Args:
        bucket_handle: Handle in ``hf://buckets/<namespace>/<bucket>/<prefix>`` form.

    Returns:
        Parsed bucket location containing the bucket id and optional prefix.

    Raises:
        ValueError: If the handle is not a valid Hugging Face bucket handle.
    """
    normalized_handle = bucket_handle.strip()
    if not normalized_handle.startswith(_HFBUCKET_HANDLE_PREFIX):
        raise ValueError(
            "bucket_handle must start with 'hf://buckets/' for the hfbucket plugin"
        )

    path = normalized_handle[len(_HFBUCKET_HANDLE_PREFIX) :].strip("/")
    path_parts = [part for part in path.split("/") if part]
    if len(path_parts) < 2:
        raise ValueError(
            "bucket_handle must be in the form "
            "'hf://buckets/<namespace>/<bucket>[/<prefix>]'"
        )

    return HFBucketLocation(
        bucket_id=f"{path_parts[0]}/{path_parts[1]}",
        object_prefix="/".join(path_parts[2:]),
    )


def encode_hfbucket_object_name(key_str: str) -> str:
    """Encode a serialized LMCache key into a reversible bucket object name."""
    return quote(key_str, safe="")


def decode_hfbucket_object_name(object_name: str) -> str:
    """Decode a reversible bucket object name back into an LMCache key string."""
    return unquote(object_name)


def _validate_huggingface_hub_support() -> None:
    """Raise a clear error when ``huggingface_hub`` lacks Buckets support."""
    version_value = getattr(huggingface_hub, "__version__", None)
    if isinstance(version_value, str):
        try:
            version = Version(version_value)
        except InvalidVersion:
            pass
        else:
            if version < _MIN_HUGGINGFACE_HUB_VERSION:
                raise _build_huggingface_hub_error(
                    "huggingface_hub version is too old for Buckets support; "
                    "install huggingface_hub>=1.5.0"
                )

    required_methods = (
        "batch_bucket_files",
        "download_bucket_files",
        "list_bucket_tree",
        "bucket_info",
        "create_bucket",
        "get_bucket_paths_info",
    )
    missing = [name for name in required_methods if not hasattr(HfApi, name)]
    if missing:
        raise _build_huggingface_hub_error(
            "huggingface_hub does not expose required Buckets APIs: "
            + ", ".join(missing)
        )


def _build_huggingface_hub_error(message: str) -> RuntimeError:
    """Create a consistent runtime error for unsupported ``huggingface_hub``."""
    return RuntimeError(
        f"{message}. LMCache HFBucket support requires huggingface_hub>=1.5.0."
    )
