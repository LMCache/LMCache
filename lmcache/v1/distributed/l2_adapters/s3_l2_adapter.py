# SPDX-License-Identifier: Apache-2.0
"""
S3 L2 adapter using the AWS CRT Python bindings.

Wraps the same awscrt.s3 client used by the non-MP ``S3Connector`` but
exposes the poll-driven ``L2AdapterInterface`` contract instead of the
async/await ``RemoteConnector`` one.

Pattern follows ``FSL2Adapter`` (asyncio loop on a daemon thread + 3
eventfds) and adds refcount-based locking + capacity tracking for
eviction, modelled on ``NativeConnectorL2Adapter``.
"""

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional
from urllib.parse import quote as url_quote
from urllib.parse import urlencode
import asyncio
import ctypes
import json
import threading
import xml.etree.ElementTree as ET

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.distributed.internal_api import L1MemoryDesc
    from lmcache.v1.memory_management import MemoryObj

# Third Party
from awscrt import auth, io, s3
from awscrt.http import HttpHeaders, HttpRequest
from awscrt.io import ClientTlsContext, TlsConnectionOptions, TlsContextOptions

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import KeyEntry, KeyListPage, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers (lifted from s3_connector.py / native_connector_l2_adapter.py)
# ---------------------------------------------------------------------------


def _string_to_object_key(name: str) -> ObjectKey:
    """Reverse of :func:`_object_key_to_string`.

    Expects
    ``<model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>[@<cache_salt>]``.

    Raises:
        ValueError: ``name`` does not match the expected format.
    """
    parts = name.split("@")
    if len(parts) == 4:
        model_name, kv_rank_hex, object_group_id_hex, chunk_hash_hex = parts
        cache_salt = ""
    elif len(parts) == 5:
        (
            model_name,
            kv_rank_hex,
            object_group_id_hex,
            chunk_hash_hex,
            cache_salt,
        ) = parts
    else:
        raise ValueError(f"unparsable S3 object name {name!r}: wrong field count")
    try:
        kv_rank = int(kv_rank_hex, 16)
    except ValueError as exc:
        raise ValueError(
            f"unparsable S3 object name {name!r}: bad kv_rank {kv_rank_hex!r}"
        ) from exc
    try:
        object_group_id = int(object_group_id_hex, 16)
    except ValueError as exc:
        raise ValueError(
            f"unparsable S3 object name {name!r}: "
            f"bad object_group_id {object_group_id_hex!r}"
        ) from exc
    try:
        chunk_hash = bytes.fromhex(chunk_hash_hex)
    except ValueError as exc:
        raise ValueError(
            f"unparsable S3 object name {name!r}: bad chunk_hash {chunk_hash_hex!r}"
        ) from exc
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name=model_name,
        kv_rank=kv_rank,
        object_group_id=object_group_id,
        cache_salt=cache_salt,
    )


def _parse_s3_last_modified(text: Optional[str]) -> Optional[float]:
    """Parse an S3 ``LastModified`` timestamp into a UTC epoch float.

    S3 emits ISO-8601 with a trailing ``Z`` (e.g.
    ``2024-01-02T03:04:05.000Z``). Returns ``None`` on a missing or
    unparsable value so callers can fall back to a neutral ordering
    rather than crash on a malformed listing.
    """
    if not text:
        return None
    try:
        # ``fromisoformat`` accepts ``+00:00`` but historically not the
        # bare ``Z`` suffix on older interpreters — normalize it.
        normalized = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _parse_list_response_xml(
    body: bytes,
) -> tuple[list[tuple[ObjectKey, int]], Optional[str]]:
    """Parse a ListObjectsV2 XML response into ``(entries, next_token)``.

    Entries this adapter can't parse (foreign objects in the bucket)
    are skipped silently. ``next_token`` is ``None`` when the listing
    is not truncated.

    Raises:
        ValueError: the response body is not valid XML.
    """
    entries_with_mtime, next_token = _parse_list_response_xml_with_mtime(body)
    entries = [(obj_key, size) for obj_key, size, _mtime in entries_with_mtime]
    return entries, next_token


def _parse_list_response_xml_with_mtime(
    body: bytes,
) -> tuple[list[tuple[ObjectKey, int, Optional[float]]], Optional[str]]:
    """Like :func:`_parse_list_response_xml` but also surfaces each
    object's ``LastModified`` as a UTC epoch float (or ``None`` when
    absent / unparsable).

    Used by the startup size-seed path, which orders objects
    oldest-modified first as an LRU proxy. Kept separate from
    :func:`_parse_list_response_xml` so the existing ``(ObjectKey, int)``
    listing contract used by ``list_l2_keys`` stays unchanged.

    Raises:
        ValueError: the response body is not valid XML.
    """
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ValueError(f"malformed ListObjectsV2 XML: {exc}") from None

    # Strip the default XML namespace so ``findall("Contents")`` works.
    # The tree is local to this function — do not cache or return it.
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    entries: list[tuple[ObjectKey, int, Optional[float]]] = []
    for contents in root.findall("Contents"):
        key_elem = contents.find("Key")
        size_elem = contents.find("Size")
        mtime_elem = contents.find("LastModified")
        if key_elem is None or key_elem.text is None:
            continue
        try:
            obj_key = _string_to_object_key(key_elem.text)
        except ValueError as exc:
            logger.debug(
                "Skipping unparsable S3 object %r in listing: %s", key_elem.text, exc
            )
            continue
        size = 0
        if size_elem is not None and size_elem.text is not None:
            try:
                size = int(size_elem.text)
            except ValueError:
                pass
        last_modified = _parse_s3_last_modified(
            mtime_elem.text if mtime_elem is not None else None
        )
        entries.append((obj_key, size, last_modified))

    next_token_elem = root.find("NextContinuationToken")
    next_token = (
        next_token_elem.text
        if next_token_elem is not None and next_token_elem.text
        else None
    )
    return entries, next_token


# Sidecar object name holding the serialized LRU access-order index.
# Kept generic; lives in the same bucket as the cached objects. The
# trailing ``@`` shape would make it look like a malformed cache key, so
# the leading underscore plus ``.json`` keeps it out of the cache-key
# namespace (``_string_to_object_key`` rejects it, so a stray listing of
# the sidecar is skipped, not mis-parsed).
LRU_INDEX_OBJECT_NAME = "_lmcache_lru_index.json"

# Bump when the on-disk sidecar layout changes incompatibly.
_LRU_INDEX_SCHEMA_VERSION = 1


def _serialize_lru_index(
    entries: list[tuple[str, int, int]],
) -> bytes:
    """Serialize an LRU index to a compact JSON sidecar payload.

    ``entries`` is ``[(object_name, size_bytes, access_tick), ...]`` in
    LRU order (oldest access first). The payload is intentionally small:
    a flat list of 3-tuples plus a schema version, so a 100k-object
    index stays in the low single-digit MBs.
    """
    payload = {
        "version": _LRU_INDEX_SCHEMA_VERSION,
        # Flat triples keep the JSON ~3x smaller than a list of dicts.
        "entries": [[name, size, tick] for (name, size, tick) in entries],
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _deserialize_lru_index(
    body: bytes,
) -> Optional[list[tuple[str, int, int]]]:
    """Parse a sidecar payload back into an ordered LRU index.

    Returns the list of ``(object_name, size_bytes, access_tick)`` triples
    in stored order, or ``None`` when the payload is missing, truncated,
    corrupt, or carries an unknown schema version. A ``None`` return is
    the signal for the caller to treat the sidecar as absent and fall
    back to the LastModified scan — a partial/corrupt sidecar must never
    crash startup.
    """
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != _LRU_INDEX_SCHEMA_VERSION:
        return None
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        return None

    out: list[tuple[str, int, int]] = []
    for item in raw_entries:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            # Skip individual malformed rows rather than discarding the
            # whole index — a single bad triple shouldn't lose all the
            # access-order fidelity the sidecar exists to preserve.
            continue
        name, size, tick = item
        if not isinstance(name, str):
            continue
        if not isinstance(size, int) or isinstance(size, bool):
            continue
        if not isinstance(tick, int) or isinstance(tick, bool):
            continue
        out.append((name, size, tick))
    return out


def _object_key_to_string(key: ObjectKey) -> str:
    """Serialize an ObjectKey to a deterministic S3 object name.

    Unsalted::

        <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>

    Salted (trailing ``cache_salt``)::

        <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>

    ``@`` in ``model_name`` and ``cache_salt`` is rejected by
    ``ObjectKey.__post_init__``, so the format is unambiguous.
    """
    base = (
        f"{key.model_name}@{key.kv_rank:08x}"
        f"@{key.object_group_id:x}@{key.chunk_hash.hex()}"
    )
    if key.cache_salt:
        return f"{base}@{key.cache_salt}"
    return base


def _format_safe_path(key_str: str) -> str:
    """URL-encode the object name to form a safe HTTP path."""
    return "/" + url_quote(key_str)


def _make_credentials_provider(
    config: "S3L2AdapterConfig",
) -> auth.AwsCredentialsProvider:
    """Build an awscrt credentials provider for the S3 L2 adapter.

    Resolution:

    1. Static keys from ``config.aws_access_key_id`` /
       ``config.aws_secret_access_key`` when both are set.
    2. Otherwise, delegate to ``boto3``. ``botocore``'s default chain
       covers env vars, shared profile, container credentials
       (``AWS_CONTAINER_CREDENTIALS_FULL_URI`` /
       ``AWS_CONTAINER_CREDENTIALS_RELATIVE_URI``), web-identity
       (``AWS_WEB_IDENTITY_TOKEN_FILE`` / ``AWS_ROLE_ARN``), and IMDS
       uniformly, including HTTPS endpoints that the awscrt Python
       binding's default chain cannot reach. The resolved
       ``RefreshableCredentials`` are republished to awscrt via
       ``new_delegate``; every sign call invokes
       ``get_frozen_credentials()`` so rotating short-lived OIDC
       credentials refresh before expiry.

    Args:
        config: S3 L2 adapter configuration.

    Returns:
        An ``AwsCredentialsProvider`` ready to attach to ``S3Request``.

    Raises:
        ImportError: ``boto3`` is required but not installed.
        RuntimeError: ``boto3`` returned no resolvable credentials.
    """
    if config.aws_access_key_id and config.aws_secret_access_key:
        logger.info("S3L2Adapter using explicit AWS credentials")
        return auth.AwsCredentialsProvider.new_static(
            config.aws_access_key_id,
            config.aws_secret_access_key,
        )

    logger.info("S3L2Adapter resolving AWS credentials via boto3 delegate")
    try:
        # Third Party
        import boto3
    except ImportError as e:
        raise ImportError(
            "S3L2Adapter requires boto3 to resolve credentials when "
            "aws_access_key_id / aws_secret_access_key are not set. "
            "Install boto3 or provide static credentials in the adapter "
            "config."
        ) from e

    boto_creds = boto3.Session().get_credentials()
    if boto_creds is None:
        raise RuntimeError("S3L2Adapter: boto3 found no credentials in the environment")

    def fetch() -> auth.AwsCredentials:
        frozen = boto_creds.get_frozen_credentials()
        return auth.AwsCredentials(
            frozen.access_key,
            frozen.secret_key,
            frozen.token,
        )

    return auth.AwsCredentialsProvider.new_delegate(fetch)


class MemoryViewStream:
    """Zero-copy stream adapter over a ``memoryview``-like object."""

    def __init__(self, mv):
        self.mv = memoryview(mv).cast("B")
        self.offset = 0

    def read(self, size=None):
        if size is None:
            size = len(self.mv) - self.offset
        if size < 0:
            size = 0
        end = min(self.offset + size, len(self.mv))
        result = self.mv[self.offset : end]
        self.offset = end
        return result

    def seek(self, offset, whence=0):
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset += offset
        elif whence == 2:
            self.offset = len(self.mv) + offset
        return self.offset

    def tell(self):
        return self.offset

    def __len__(self):
        return len(self.mv)


def _is_connection_error(error_msg: str) -> bool:
    return (
        "CONNECTION_REFUSED" in error_msg
        or "SOCKET" in error_msg
        or "DNS" in error_msg
        or "TIMEOUT" in error_msg
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class S3L2AdapterConfig(L2AdapterConfigBase):
    """Config for the S3 L2 adapter.

    Fields:
    - s3_endpoint (str, required): bucket URL using **virtual-hosted**
      style; accepts either ``"s3://<bucket>.<host>"`` or the bare
      ``"<bucket>.<host>"`` form. The bucket name must be part of the
      host because requests are signed and routed against this Host
      header (path-style addressing is not supported).
    - s3_region (str, required): AWS region used for SigV4.
    - s3_num_io_threads (int): CRT IO threads.
    - s3_prefer_http2 (bool): ALPN negotiate to HTTP/2.
    - s3_enable_s3express (bool): enable S3 Express signing.
    - disable_tls (bool): bypass TLS on the bucket data plane (for
      S3-compatible HTTP endpoints). Does not affect the credentials
      resolver, which may still issue HTTPS calls.
    - aws_access_key_id / aws_secret_access_key (str): optional static
      credentials. When unset, credentials are resolved through boto3
      (env vars, profile, container, web-identity, IMDS).
    - max_capacity_gb (float): aggregate capacity used by
      ``get_usage()``; ``0`` disables aggregate eviction
      (``usage_fraction == -1.0``).
    - s3_seed_usage_on_start (bool): when ``True``, scan the bucket on
      adapter startup and seed both the aggregate byte counter and the
      eviction policy's key index (ordered oldest-``LastModified``
      first) so the LRU cap binds across process restarts. Default
      ``False`` (no behavior change for existing deployments). A scan
      failure is logged and non-fatal.
    - s3_persist_lru_index (bool): when ``True``, periodically write a
      small sidecar object (``_lmcache_lru_index.json``) capturing true
      LRU access order, and restore from it on startup (reconciled
      against the live bucket listing) for higher eviction fidelity than
      the ``LastModified`` proxy alone. Default ``False``. Implies the
      startup seed behavior when restoring.
    - s3_lru_index_checkpoint_interval_s (int): seconds between sidecar
      checkpoints when ``s3_persist_lru_index`` is on. Default 300.
    """

    def __init__(
        self,
        s3_endpoint: str,
        s3_region: str,
        s3_num_io_threads: int = 64,
        s3_prefer_http2: bool = True,
        s3_enable_s3express: bool = False,
        disable_tls: bool = False,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        max_capacity_gb: float = 0.0,
        s3_seed_usage_on_start: bool = False,
        s3_persist_lru_index: bool = False,
        s3_lru_index_checkpoint_interval_s: int = 300,
    ):
        self.s3_endpoint = s3_endpoint
        self.s3_region = s3_region
        self.s3_num_io_threads = s3_num_io_threads
        self.s3_prefer_http2 = s3_prefer_http2
        self.s3_enable_s3express = s3_enable_s3express
        self.disable_tls = disable_tls
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.max_capacity_gb = max_capacity_gb
        self.s3_seed_usage_on_start = s3_seed_usage_on_start
        self.s3_persist_lru_index = s3_persist_lru_index
        self.s3_lru_index_checkpoint_interval_s = s3_lru_index_checkpoint_interval_s

    @classmethod
    def from_dict(cls, d: dict) -> "S3L2AdapterConfig":
        endpoint = d.get("s3_endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("s3_endpoint must be a non-empty string")
        region = d.get("s3_region")
        if not isinstance(region, str) or not region:
            raise ValueError("s3_region must be a non-empty string")

        def _int(key, default):
            v = d.get(key, default)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{key} must be a positive integer")
            return v

        def _bool(key, default):
            v = d.get(key, default)
            if not isinstance(v, bool):
                raise ValueError(f"{key} must be a boolean")
            return v

        def _opt_str(key):
            v = d.get(key, None)
            if v is None:
                return None
            if not isinstance(v, str):
                raise ValueError(f"{key} must be a string")
            return v

        max_cap = d.get("max_capacity_gb", 0.0)
        if not isinstance(max_cap, (int, float)) or isinstance(max_cap, bool):
            raise ValueError("max_capacity_gb must be a number")

        cfg = cls(
            s3_endpoint=endpoint,
            s3_region=region,
            s3_num_io_threads=_int("s3_num_io_threads", 64),
            s3_prefer_http2=_bool("s3_prefer_http2", True),
            s3_enable_s3express=_bool("s3_enable_s3express", False),
            disable_tls=_bool("disable_tls", False),
            aws_access_key_id=_opt_str("aws_access_key_id"),
            aws_secret_access_key=_opt_str("aws_secret_access_key"),
            max_capacity_gb=float(max_cap),
            s3_seed_usage_on_start=_bool("s3_seed_usage_on_start", False),
            s3_persist_lru_index=_bool("s3_persist_lru_index", False),
            s3_lru_index_checkpoint_interval_s=_int(
                "s3_lru_index_checkpoint_interval_s", 300
            ),
        )
        cfg.eviction_config = cls._parse_eviction_config(d)
        return cfg

    @classmethod
    def help(cls) -> str:
        return (
            "S3 L2 adapter config fields:\n"
            "- s3_endpoint (str, required): virtual-hosted bucket URL "
            "('s3://<bucket>.<host>' or '<bucket>.<host>')\n"
            "- s3_region (str, required): AWS region for SigV4\n"
            "- s3_num_io_threads (int): CRT IO threads (default 64)\n"
            "- s3_prefer_http2 (bool): try HTTP/2 via ALPN (default true)\n"
            "- s3_enable_s3express (bool): S3 Express signing (default false)\n"
            "- disable_tls (bool): bypass TLS on the bucket data plane\n"
            "- aws_access_key_id / aws_secret_access_key (str): static creds; "
            "when unset, boto3 resolves credentials\n"
            "- max_capacity_gb (float): capacity for get_usage (0 = disabled)\n"
            "- s3_seed_usage_on_start (bool): scan bucket on startup and "
            "seed usage + eviction LRU index (default false)\n"
            "- s3_persist_lru_index (bool): checkpoint/restore a sidecar LRU "
            "index for cross-restart access-order fidelity (default false)\n"
            "- s3_lru_index_checkpoint_interval_s (int): sidecar checkpoint "
            "interval in seconds (default 300)\n"
            "- eviction (dict): optional, see L2AdapterConfigBase"
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class S3L2Adapter(L2AdapterInterface):
    """S3-backed L2 adapter.

    Concurrency model: one asyncio event loop on a dedicated daemon
    thread. Each ``submit_*`` schedules a coroutine via
    ``run_coroutine_threadsafe``; that coroutine launches parallel
    ``s3.S3Request`` handles, awaits them with ``asyncio.gather``, and
    signals the corresponding eventfd when done.

    Locking: client-side refcount in ``_locked_keys``. ``delete()``
    skips any key whose refcount is > 0 — prevents evicting a key that
    a concurrent load is about to read.

    Circuit breaker: after ``max_connection_failures`` consecutive
    connection-class errors, ``connection_disabled`` is set; all
    subsequent submits short-circuit and record failure without
    touching S3.
    """

    max_connection_failures = 3

    def __init__(self, config: S3L2AdapterConfig):
        super().__init__(max_capacity_bytes=int(config.max_capacity_gb * (1024**3)))
        self._config = config

        endpoint = config.s3_endpoint
        if endpoint.startswith("s3://"):
            endpoint = endpoint[len("s3://") :]
        self._endpoint = endpoint
        self._region = config.s3_region
        self._enable_s3express = config.s3_enable_s3express

        # awscrt client setup (mirrors s3_connector.py:103-153)
        event_loop_group = io.EventLoopGroup(config.s3_num_io_threads)
        host_resolver = io.DefaultHostResolver(event_loop_group)
        client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)

        self._credentials_provider = _make_credentials_provider(config)

        tls_opts = None
        if config.s3_prefer_http2:
            tls_ctx = ClientTlsContext(TlsContextOptions())
            tls_opts = TlsConnectionOptions(tls_ctx)
            try:
                tls_opts.set_alpn_list(["h2", "http/1.1"])
            except Exception:
                tls_opts = None

        signing_config = None
        if self._enable_s3express:
            signing_config = auth.AwsSigningConfig(
                algorithm=auth.AwsSigningAlgorithm.V4_S3EXPRESS,
                region=self._region,
                service="s3",
                credentials_provider=self._credentials_provider,
            )

        tls_mode = (
            s3.S3RequestTlsMode.DISABLED
            if config.disable_tls
            else s3.S3RequestTlsMode.ENABLED
        )
        logger.info("Initializing S3 client for S3L2Adapter")
        self._s3_client = s3.S3Client(
            bootstrap=client_bootstrap,
            region=self._region,
            enable_s3express=self._enable_s3express,
            tls_connection_options=tls_opts,
            tls_mode=tls_mode,
            signing_config=signing_config,
        )

        # 3 distinct cross-platform notifiers for the L2 interface.
        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}

        # Refcounted locks (like NativeConnectorL2Adapter).
        self._locked_keys: dict[ObjectKey, int] = defaultdict(int)

        # Per-key size map — retained so ``delete`` can recover each
        # key's stored size and pass it to ``_notify_keys_deleted``.
        # Aggregate byte accounting lives in the base class via
        # ``_notify_keys_stored``/``_notify_keys_deleted``; we do not
        # maintain a parallel total here.
        self._key_sizes: dict[ObjectKey, int] = {}

        # Cached HEAD-verified object sizes (keyed by S3 object name).
        self._object_size_cache: dict[str, int] = {}

        # Circuit breaker state.
        self._connection_failures = 0
        self._connection_disabled = False

        # LRU access-order tracking for the optional sidecar checkpoint
        # (feature b). The adapter is the single source of truth for what
        # it stored/accessed, so we record a monotonically increasing
        # "tick" per key here rather than reaching into the eviction
        # policy's private ordering. ``_access_ticks`` maps the S3 object
        # name -> (size, last_access_tick); only maintained when
        # ``s3_persist_lru_index`` is enabled to keep the hot path free.
        self._track_access_order = config.s3_persist_lru_index
        self._access_ticks: dict[str, tuple[int, int]] = {}
        self._access_tick_counter = 0

        self._lock = threading.Lock()

        # Background asyncio event loop.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
            name="s3-l2-adapter-loop",
        )
        self._loop_thread.start()

        self._closed = False

        # Background sidecar checkpoint thread (feature b). Created lazily
        # in ``_maybe_start_checkpoint_thread`` after a successful seed so
        # we never checkpoint an empty index over a good one.
        self._checkpoint_stop = threading.Event()
        self._checkpoint_thread: Optional[threading.Thread] = None

        logger.info(
            "Initialized S3L2Adapter (endpoint=%s region=%s "
            "http2=%s s3express=%s tls=%s max_capacity_gb=%.2f "
            "seed_usage_on_start=%s persist_lru_index=%s)",
            self._endpoint,
            self._region,
            config.s3_prefer_http2,
            self._enable_s3express,
            not config.disable_tls,
            config.max_capacity_gb,
            config.s3_seed_usage_on_start,
            config.s3_persist_lru_index,
        )

        # Startup seed (features a + b). Both flags default OFF, so this
        # is a no-op for existing deployments. A failure here must never
        # break adapter startup, so the whole block is best-effort.
        if config.s3_seed_usage_on_start or config.s3_persist_lru_index:
            try:
                self._seed_from_bucket()
            except Exception:
                logger.exception(
                    "S3L2Adapter startup seed failed; continuing with an "
                    "empty in-memory index (eviction will only see objects "
                    "stored by this process generation)"
                )
            if config.s3_persist_lru_index:
                self._maybe_start_checkpoint_thread()

    # ------------------------------------------------------------------
    # Event Fd Interface
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # ------------------------------------------------------------------
    # Store Interface
    # ------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            if self._connection_disabled:
                self._completed_store_tasks[task_id] = L2StoreResult(False, 0)
                disabled = True
            else:
                disabled = False

        if disabled:
            self._store_efd.notify()
            return task_id

        asyncio.run_coroutine_threadsafe(
            self._execute_store(list(keys), list(objects), task_id),
            self._loop,
        )
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    # ------------------------------------------------------------------
    # Lookup and Lock Interface
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
    ) -> L2TaskId:
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            if self._connection_disabled:
                self._completed_lookup_tasks[task_id] = Bitmap(len(keys))
                disabled = True
            else:
                disabled = False

        if disabled:
            self._lookup_efd.notify()
            return task_id

        asyncio.run_coroutine_threadsafe(
            self._execute_lookup(list(keys), task_id),
            self._loop,
        )
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        with self._lock:
            for key in keys:
                if key not in self._locked_keys:
                    continue
                if self._locked_keys[key] <= 1:
                    del self._locked_keys[key]
                else:
                    self._locked_keys[key] -= 1

    # ------------------------------------------------------------------
    # Load Interface
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            if self._connection_disabled:
                self._completed_load_tasks[task_id] = Bitmap(len(keys))
                disabled = True
            else:
                disabled = False

        if disabled:
            self._load_efd.notify()
            return task_id

        asyncio.run_coroutine_threadsafe(
            self._execute_load(list(keys), list(objects), task_id),
            self._loop,
        )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    # ------------------------------------------------------------------
    # LRU access-order tracking (feature b)
    # ------------------------------------------------------------------
    #
    # We override the base ``_notify_keys_*`` hooks to additionally record
    # a per-key access tick when ``s3_persist_lru_index`` is on. The base
    # implementations still run (byte accounting + listener fanout to the
    # eviction policy), so behavior is unchanged when tracking is off.

    def _bump_access_ticks(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        """Record/refresh the access tick for ``keys`` (size-aware)."""
        with self._lock:
            for key, size in zip(keys, sizes, strict=True):
                self._access_tick_counter += 1
                self._access_ticks[_object_key_to_string(key)] = (
                    size,
                    self._access_tick_counter,
                )

    def _touch_access_ticks(self, keys: list[ObjectKey]) -> None:
        """Refresh the access tick for already-tracked ``keys`` on read."""
        with self._lock:
            for key in keys:
                name = _object_key_to_string(key)
                prev = self._access_ticks.get(name)
                if prev is None:
                    continue
                self._access_tick_counter += 1
                self._access_ticks[name] = (prev[0], self._access_tick_counter)

    def _notify_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        super()._notify_keys_stored(keys, sizes)
        if self._track_access_order:
            self._bump_access_ticks(keys, sizes)

    def _notify_keys_accessed(self, keys: list[ObjectKey]) -> None:
        super()._notify_keys_accessed(keys)
        if self._track_access_order:
            self._touch_access_ticks(keys)

    def _notify_keys_deleted(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        super()._notify_keys_deleted(keys, sizes)
        if self._track_access_order:
            with self._lock:
                for key in keys:
                    self._access_ticks.pop(_object_key_to_string(key), None)

    # ------------------------------------------------------------------
    # Eviction Interface
    # ------------------------------------------------------------------

    def delete(self, keys: list[ObjectKey]) -> None:
        if not keys:
            return

        # Filter out locked keys — they're being read right now.
        with self._lock:
            if self._connection_disabled:
                return
            deletable = [k for k in keys if self._locked_keys.get(k, 0) == 0]

        if not deletable:
            return

        fut = asyncio.run_coroutine_threadsafe(
            self._execute_delete(deletable),
            self._loop,
        )
        try:
            deleted_keys, deleted_sizes = fut.result(timeout=30.0)
        except Exception as e:
            logger.warning("S3L2Adapter delete failed: %s", e)
            return

        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    # ``get_usage()`` is inherited from ``L2AdapterInterface``. The base
    # class maintains the aggregate and per-``cache_salt`` byte totals
    # via ``_notify_keys_stored`` / ``_notify_keys_deleted`` and returns
    # an ``AdapterUsage`` snapshot with ``usage_fraction == -1.0`` when
    # ``max_capacity_gb`` was 0 (unlimited / no eviction signal).

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_l2_keys(
        self,
        model_name: Optional[str] = None,
        page_size: int = 500,
        cursor: Optional[str] = None,
    ) -> KeyListPage:
        """List keys from the S3 bucket via ``ListObjectsV2``.

        Args:
            model_name: if set, restrict listing to objects whose name
                starts with ``<model_name>@``.
            page_size: target entries per page. Clamped to S3's
                ``MaxKeys=1000`` ceiling.
            cursor: opaque continuation token from the previous page.

        Raises:
            ValueError: ``page_size`` non-positive, malformed
                ``cursor``, or malformed S3 response.
            RuntimeError: connection is circuit-broken, or the
                underlying CRT request errored out.
        """
        if page_size <= 0:
            raise ValueError(f"page_size must be positive (got {page_size})")
        # ListObjectsV2's ``MaxKeys`` is capped at 1000 by S3. The
        # adapter clamps so callers asking for more just get the S3
        # max plus a continuation token — they don't need to know
        # about the server-side limit.
        max_keys = min(page_size, 1000)
        # ``model_name`` rides along as a ``prefix=`` query param so S3
        # skips non-matching keys server-side. Keys are stored under
        # their literal name (no path-flattening), so the prefix is
        # just ``<model_name>@``.
        prefix: Optional[str] = None
        if model_name is not None:
            prefix = f"{model_name}@"

        with self._lock:
            if self._connection_disabled:
                raise RuntimeError(
                    "S3 connection disabled (circuit-broken); listing unavailable"
                )

        fut = asyncio.run_coroutine_threadsafe(
            self._execute_list(prefix, max_keys, cursor),
            self._loop,
        )
        entries, next_token = fut.result(timeout=30.0)
        page_entries = tuple(
            KeyEntry(key=k.to_encoded_object_key(), size_bytes=sz) for k, sz in entries
        )
        return KeyListPage(entries=page_entries, next_page_token=next_token)

    # ------------------------------------------------------------------
    # Status / Cleanup
    # ------------------------------------------------------------------

    def report_status(self) -> dict:
        with self._lock:
            failures = self._connection_failures
            disabled = self._connection_disabled
        usage = self.get_usage()
        return {
            "is_healthy": self._loop_thread.is_alive() and not disabled,
            "type": "S3L2Adapter",
            "endpoint": self._endpoint,
            "region": self._region,
            "connection_failures": failures,
            "connection_disabled": disabled,
            "current_size_bytes": usage.total_bytes_used,
            "max_capacity_bytes": usage.total_capacity_bytes,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        # Stop the checkpoint thread and write one final sidecar so the
        # latest access order survives a clean shutdown. Best-effort:
        # a failure here must not block close().
        if self._checkpoint_thread is not None:
            self._checkpoint_stop.set()
            try:
                # Short timeout: don't let a slow S3 block shutdown past the
                # container's termination grace period (-> SIGKILL).
                self._checkpoint_lru_index(timeout=5.0)
            except Exception:
                logger.exception("S3L2Adapter final LRU checkpoint failed")
            self._checkpoint_thread.join(timeout=5)

        async def _stop_tasks():
            tasks = [
                t
                for t in asyncio.all_tasks(self._loop)
                if t is not asyncio.current_task()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        if self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(_stop_tasks(), self._loop).result(
                    timeout=5
                )
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._loop_thread.join(timeout=5)
        try:
            self._loop.close()
        except Exception:
            pass

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

        # Drop awscrt references so their native event loops / host
        # resolver threads / epoll fds can be reaped immediately rather
        # than surviving until this adapter is garbage-collected. Without
        # this, spinning up many adapters in a process (e.g. a test
        # module with per-test fixtures) can pile up FDs and exhaust
        # ``ulimit -n`` on CI runners.
        self._s3_client = None
        self._credentials_provider = None
        logger.info("S3L2Adapter closed")

    # ------------------------------------------------------------------
    # Internal: event loop & S3 request helpers
    # ------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _make_request(
        self, method: str, key_str: str, *, body_stream=None, extra_headers=None
    ):
        headers = HttpHeaders()
        headers.add("Host", self._endpoint)
        if extra_headers:
            for k, v in extra_headers:
                headers.add(k, v)
        return HttpRequest(
            method,
            _format_safe_path(key_str),
            headers,
            body_stream=body_stream,
        )

    def _head_request(self, key_str: str):
        req = self._make_request("HEAD", key_str)
        captured = {"len": None, "status": None}

        def on_headers(status_code, headers, **kwargs):
            captured["status"] = status_code
            for name, value in headers:
                if name.lower() == "content-length":
                    try:
                        captured["len"] = int(value)
                    except Exception:
                        pass

        s3_req = s3.S3Request(
            client=self._s3_client,
            type=s3.S3RequestType.DEFAULT,
            request=req,
            operation_name="HeadObject",
            on_headers=on_headers,
            credential_provider=self._credentials_provider,
            region=self._region,
        )
        return s3_req, captured

    def _get_request(self, key_str: str, mem_obj: MemoryObj):
        req = self._make_request("GET", key_str)
        data_ptr = mem_obj.data_ptr

        def on_body(chunk, offset, **kwargs):
            # Write chunk into the caller-provided MemoryObj buffer.
            ctypes.memmove(data_ptr + offset, chunk, len(chunk))

        def on_done(error=None, status_code=None, **kwargs):
            ok = (status_code in (200, 206)) or (status_code is None and error is None)
            if error or not ok:
                raise RuntimeError(
                    f"S3 GET failed for {key_str}: {error or status_code}"
                )

        s3_req = s3.S3Request(
            client=self._s3_client,
            type=s3.S3RequestType.GET_OBJECT,
            request=req,
            on_body=on_body,
            on_done=on_done,
            credential_provider=self._credentials_provider,
            region=self._region,
        )
        return s3_req

    def _put_request(self, key_str: str, mem_obj: MemoryObj):
        stream = MemoryViewStream(mem_obj.byte_array)
        total_len = len(stream)
        req = self._make_request(
            "PUT",
            key_str,
            body_stream=stream,
            extra_headers=[
                ("Content-Length", str(total_len)),
                ("Content-Type", "application/octet-stream"),
            ],
        )
        captured = {"status": None}

        def on_done(error=None, status_code=None, **kwargs):
            captured["status"] = status_code
            if error or status_code not in (200, 201):
                raise RuntimeError(
                    f"S3 PUT failed for {key_str}: {error or status_code}"
                )

        s3_req = s3.S3Request(
            client=self._s3_client,
            type=s3.S3RequestType.PUT_OBJECT,
            request=req,
            on_done=on_done,
            credential_provider=self._credentials_provider,
            region=self._region,
        )
        return s3_req

    def _delete_request(self, key_str: str):
        req = self._make_request("DELETE", key_str)
        captured = {"status": None}

        def on_headers(status_code, headers, **kwargs):
            captured["status"] = status_code

        def on_done(error=None, status_code=None, **kwargs):
            captured["status"] = status_code or captured["status"]
            # 204 is standard for DeleteObject, 200 also tolerated.
            if error or captured["status"] not in (200, 204):
                raise RuntimeError(
                    f"S3 DELETE failed for {key_str}: {error or captured['status']}"
                )

        s3_req = s3.S3Request(
            client=self._s3_client,
            type=s3.S3RequestType.DEFAULT,
            request=req,
            operation_name="DeleteObject",
            on_headers=on_headers,
            on_done=on_done,
            credential_provider=self._credentials_provider,
            region=self._region,
        )
        return s3_req

    def _list_request(
        self,
        prefix: Optional[str],
        max_keys: int,
        continuation_token: Optional[str],
    ):
        """Build a ListObjectsV2 request.

        Returns ``(s3_req, body_chunks, captured)``. The caller awaits
        ``s3_req.finished_future`` and assembles the XML from
        ``body_chunks``.
        """
        params: list[tuple[str, str]] = [
            ("list-type", "2"),
            ("max-keys", str(max_keys)),
        ]
        if prefix:
            params.append(("prefix", prefix))
        if continuation_token:
            params.append(("continuation-token", continuation_token))
        # urlencode handles percent-escaping of values (continuation
        # tokens are typically base64 with ``+``/``/``/``=`` chars).
        path = "/?" + urlencode(params, quote_via=url_quote)

        headers = HttpHeaders()
        headers.add("Host", self._endpoint)
        req = HttpRequest("GET", path, headers)

        body_chunks: list[bytes] = []
        captured: dict[str, Optional[int]] = {"status": None}

        def on_body(chunk, offset, **kwargs):
            body_chunks.append(bytes(chunk))

        def on_headers(status_code, headers, **kwargs):
            captured["status"] = status_code

        def on_done(error=None, status_code=None, **kwargs):
            captured["status"] = status_code or captured["status"]
            if error or captured["status"] != 200:
                raise RuntimeError(
                    f"S3 ListObjectsV2 failed: {error or captured['status']}"
                )

        s3_req = s3.S3Request(
            client=self._s3_client,
            type=s3.S3RequestType.DEFAULT,
            request=req,
            operation_name="ListObjectsV2",
            on_body=on_body,
            on_headers=on_headers,
            on_done=on_done,
            credential_provider=self._credentials_provider,
            region=self._region,
        )
        return s3_req, body_chunks, captured

    def _record_connection_outcome(self, error_msg: Optional[str]) -> None:
        """Update the circuit breaker under the lock."""
        with self._lock:
            if error_msg is None:
                if self._connection_failures > 0:
                    logger.info("S3L2Adapter connection recovered")
                self._connection_failures = 0
                return
            if not _is_connection_error(error_msg):
                return
            self._connection_failures += 1
            logger.error(
                "S3L2Adapter connection error (%d/%d): %s",
                self._connection_failures,
                self.max_connection_failures,
                error_msg,
            )
            if self._connection_failures >= self.max_connection_failures:
                self._connection_disabled = True
                logger.error(
                    "S3L2Adapter disabled after %d consecutive connection failures",
                    self.max_connection_failures,
                )

    # ------------------------------------------------------------------
    # Internal: coroutines
    # ------------------------------------------------------------------

    async def _execute_store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        futures: list[Optional[asyncio.Future]] = []
        indexed: list[tuple[int, ObjectKey, MemoryObj, Optional[str]]] = []
        for i, (key, obj) in enumerate(zip(keys, objects, strict=True)):
            try:
                key_str = _object_key_to_string(key)
                s3_req = self._put_request(key_str, obj)
                futures.append(asyncio.wrap_future(s3_req.finished_future))
                indexed.append((i, key, obj, key_str))
            except Exception:
                logger.exception("S3L2Adapter failed to launch PUT")
                indexed.append((i, key, obj, None))
                futures.append(None)

        # Await all non-None futures.
        results: list = []
        real_futures = [f for f in futures if f is not None]
        real_results = await asyncio.gather(*real_futures, return_exceptions=True)
        real_iter = iter(real_results)
        for f in futures:
            if f is None:
                results.append(RuntimeError("failed to launch S3 PUT"))
            else:
                results.append(next(real_iter))

        success = True
        # Track net-new keys for accounting notification. Same chunk_hash
        # re-stored is identical content (content-addressed), so skipping
        # re-notify here prevents the base class from double-counting
        # bytes for the same object.
        newly_stored_keys: list[ObjectKey] = []
        newly_stored_sizes: list[int] = []
        last_error: Optional[str] = None
        for indexed_entry, result in zip(indexed, results, strict=True):
            i, key, obj, opt_key_str = indexed_entry
            if isinstance(result, Exception):
                success = False
                last_error = str(result)
                continue
            # Use logical size (``get_size``) to match the number of
            # bytes actually PUT to S3 via ``obj.byte_array`` — which
            # excludes any alignment padding in the underlying buffer.
            # ``get_physical_size`` would inflate ``total_bytes_used``
            # relative to the on-wire payload and cause premature
            # aggregate-watermark eviction. Matches the convention used
            # by ``native_connector_l2_adapter`` and ``mock_l2_adapter``.
            size = obj.get_size()
            with self._lock:
                is_new = key not in self._key_sizes
                self._key_sizes[key] = size
                if opt_key_str is not None:
                    self._object_size_cache[opt_key_str] = size
            if is_new:
                newly_stored_keys.append(key)
                newly_stored_sizes.append(size)

        self._record_connection_outcome(last_error if not success else None)

        bytes_transferred = sum(newly_stored_sizes)
        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(
                success, bytes_transferred
            )

        if newly_stored_keys:
            self._notify_keys_stored(newly_stored_keys, newly_stored_sizes)
        self._store_efd.notify()

    async def _execute_lookup(
        self,
        keys: list[ObjectKey],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        futures: list = []
        captured_list: list = []
        key_strings: list[str] = []
        cache_hits: list[Optional[int]] = []

        with self._lock:
            for key in keys:
                key_str = _object_key_to_string(key)
                key_strings.append(key_str)
                cache_hits.append(self._object_size_cache.get(key_str))

        for idx, (key_str, cached_size) in enumerate(
            zip(key_strings, cache_hits, strict=True)
        ):
            if cached_size is not None:
                futures.append(None)
                captured_list.append({"status": 200, "len": cached_size})
                continue
            try:
                s3_req, captured = self._head_request(key_str)
                futures.append(asyncio.wrap_future(s3_req.finished_future))
                captured_list.append(captured)
            except Exception:
                logger.exception("S3L2Adapter failed to launch HEAD")
                futures.append(None)
                captured_list.append({"status": None, "len": None})

        real_futures = [f for f in futures if f is not None]
        real_results = await asyncio.gather(*real_futures, return_exceptions=True)
        real_iter = iter(real_results)
        combined: list = []
        for f in futures:
            if f is None:
                combined.append(None)  # cached or failed-to-launch
            else:
                combined.append(next(real_iter))

        last_error: Optional[str] = None
        any_success = False

        for i, (key, key_str, captured, result) in enumerate(
            zip(keys, key_strings, captured_list, combined, strict=True)
        ):
            status = captured.get("status")
            length = captured.get("len")
            if isinstance(result, Exception):
                # Non-200 surfaces as finished_future exception.
                # 404 is an expected not-found.
                if status == 404:
                    continue
                last_error = str(result)
                continue
            # result is None (cached) or None (success returned by future).
            if status == 200 and length is not None and length > 0:
                bitmap.set(i)
                any_success = True
                with self._lock:
                    self._object_size_cache[key_str] = length
                    self._locked_keys[key] += 1

        if any_success:
            self._record_connection_outcome(None)
        elif last_error is not None:
            self._record_connection_outcome(last_error)

        with self._lock:
            self._completed_lookup_tasks[task_id] = bitmap
        self._lookup_efd.notify()

        accessed = [keys[i] for i in range(len(keys)) if bitmap.test(i)]
        if accessed:
            self._notify_keys_accessed(accessed)

    async def _execute_load(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        futures = []
        launched_indices = []

        for i, (key, obj) in enumerate(zip(keys, objects, strict=True)):
            try:
                key_str = _object_key_to_string(key)
                s3_req = self._get_request(key_str, obj)
                futures.append(asyncio.wrap_future(s3_req.finished_future))
                launched_indices.append(i)
            except Exception:
                logger.exception("S3L2Adapter failed to launch GET")

        results = await asyncio.gather(*futures, return_exceptions=True)
        last_error: Optional[str] = None
        any_success = False

        for idx, result in zip(launched_indices, results, strict=True):
            if isinstance(result, Exception):
                last_error = str(result)
                continue
            bitmap.set(idx)
            any_success = True

        if any_success:
            self._record_connection_outcome(None)
        elif last_error is not None:
            self._record_connection_outcome(last_error)

        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._load_efd.notify()

    async def _execute_list(
        self,
        prefix: Optional[str],
        max_keys: int,
        continuation_token: Optional[str],
    ) -> tuple[list[tuple[ObjectKey, int]], Optional[str]]:
        """Issue one ``ListObjectsV2`` call and parse the response."""
        s3_req, body_chunks, _captured = self._list_request(
            prefix, max_keys, continuation_token
        )
        await asyncio.wrap_future(s3_req.finished_future)
        return _parse_list_response_xml(b"".join(body_chunks))

    async def _execute_delete(
        self, keys: list[ObjectKey]
    ) -> tuple[list[ObjectKey], list[int]]:
        """Run DELETE for each key and drop its size-tracking entry.

        Returns parallel lists of successfully deleted keys and their
        stored sizes, suitable for passing straight to
        ``_notify_keys_deleted``. Keys whose size we never learned
        (delete of an unknown key) are reported with size ``0`` so
        listener fanout still fires while base-class byte accounting
        stays balanced.
        """
        futures = []
        indexed = []
        for key in keys:
            try:
                key_str = _object_key_to_string(key)
                s3_req = self._delete_request(key_str)
                futures.append(asyncio.wrap_future(s3_req.finished_future))
                indexed.append((key, key_str))
            except Exception:
                logger.exception("S3L2Adapter failed to launch DELETE")

        results = await asyncio.gather(*futures, return_exceptions=True)
        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        for (key, key_str), result in zip(indexed, results, strict=True):
            if isinstance(result, Exception):
                logger.warning("S3L2Adapter DELETE failed for %s: %s", key_str, result)
                continue
            with self._lock:
                sz = self._key_sizes.pop(key, None)
                self._object_size_cache.pop(key_str, None)
            deleted_keys.append(key)
            deleted_sizes.append(sz if sz is not None else 0)
        return deleted_keys, deleted_sizes

    async def _execute_list_with_mtime(
        self,
        prefix: Optional[str],
        max_keys: int,
        continuation_token: Optional[str],
    ) -> tuple[list[tuple[ObjectKey, int, Optional[float]]], Optional[str]]:
        """One ``ListObjectsV2`` call, parsed with ``LastModified``."""
        s3_req, body_chunks, _captured = self._list_request(
            prefix, max_keys, continuation_token
        )
        await asyncio.wrap_future(s3_req.finished_future)
        return _parse_list_response_xml_with_mtime(b"".join(body_chunks))

    async def _execute_get_raw(self, key_str: str) -> Optional[bytes]:
        """GET an arbitrary object and return its raw bytes.

        Used for the LRU-index sidecar (not a cache object, so it does
        not flow through the MemoryObj load path). Returns ``None`` on a
        404 / any error — the caller treats a missing sidecar as "fall
        back to scan".
        """
        body_chunks: list[bytes] = []
        captured = {"status": None}

        req = self._make_request("GET", key_str)

        def on_body(chunk, offset, **kwargs):
            body_chunks.append(bytes(chunk))

        def on_headers(status_code, headers, **kwargs):
            captured["status"] = status_code

        def on_done(error=None, status_code=None, **kwargs):
            # Do NOT raise here. A missing sidecar (HTTP 404) is an expected,
            # benign case the caller handles by falling back to a full scan.
            # Raising inside this CRT callback logs a noisy traceback even though
            # finished_future already surfaces any genuine error — so just record
            # the outcome and let the awaiter decide.
            captured["status"] = status_code or captured["status"]
            captured["error"] = error

        s3_req = s3.S3Request(
            client=self._s3_client,
            type=s3.S3RequestType.DEFAULT,
            request=req,
            operation_name="GetObject",
            on_body=on_body,
            on_headers=on_headers,
            on_done=on_done,
            credential_provider=self._credentials_provider,
            region=self._region,
        )
        try:
            await asyncio.wrap_future(s3_req.finished_future)
        except Exception as exc:
            # CRT raises AWS_ERROR_S3_INVALID_RESPONSE_STATUS for a 404 — that's a
            # missing sidecar, not a failure. Return None quietly; caller scans.
            logger.debug("sidecar %s absent/unreadable (%s); will scan", key_str, exc)
            return None
        if captured.get("error") or captured["status"] not in (200, 206):
            logger.debug(
                "sidecar %s status %s; treating as absent",
                key_str,
                captured.get("error") or captured["status"],
            )
            return None
        return b"".join(body_chunks)

    async def _execute_put_raw(self, key_str: str, body: bytes) -> bool:
        """PUT raw bytes to an arbitrary object name (the sidecar)."""
        stream = MemoryViewStream(memoryview(body))
        req = self._make_request(
            "PUT",
            key_str,
            body_stream=stream,
            extra_headers=[
                ("Content-Length", str(len(body))),
                ("Content-Type", "application/json"),
            ],
        )
        captured = {"status": None}

        def on_done(error=None, status_code=None, **kwargs):
            captured["status"] = status_code
            if error or status_code not in (200, 201):
                raise RuntimeError(
                    f"S3 PUT failed for {key_str}: {error or status_code}"
                )

        s3_req = s3.S3Request(
            client=self._s3_client,
            type=s3.S3RequestType.PUT_OBJECT,
            request=req,
            on_done=on_done,
            credential_provider=self._credentials_provider,
            region=self._region,
        )
        try:
            await asyncio.wrap_future(s3_req.finished_future)
        except Exception as exc:
            logger.warning("S3 PUT for sidecar %s failed: %s", key_str, exc)
            return False
        return True

    # ------------------------------------------------------------------
    # Internal: startup seed + sidecar checkpoint (features a & b)
    # ------------------------------------------------------------------

    def _scan_bucket(self) -> list[tuple[ObjectKey, int, Optional[float]]]:
        """List the entire bucket, paginated, returning all cache entries.

        Each entry is ``(ObjectKey, size_bytes, last_modified_epoch)``.
        Foreign / unparsable objects (including the sidecar itself) are
        skipped by the XML parser. Logs progress so a 100k+ object scan
        is observable.
        """
        all_entries: list[tuple[ObjectKey, int, Optional[float]]] = []
        cursor: Optional[str] = None
        page_no = 0
        while True:
            fut = asyncio.run_coroutine_threadsafe(
                self._execute_list_with_mtime(None, 1000, cursor),
                self._loop,
            )
            entries, next_token = fut.result(timeout=60.0)
            all_entries.extend(entries)
            page_no += 1
            if page_no % 20 == 0:
                logger.info(
                    "S3L2Adapter seed scan in progress: %d pages, %d objects",
                    page_no,
                    len(all_entries),
                )
            if not next_token:
                break
            cursor = next_token
        logger.info(
            "S3L2Adapter seed scan complete: %d pages, %d cache objects",
            page_no,
            len(all_entries),
        )
        return all_entries

    def _seed_policy_in_order(self, ordered: list[tuple[ObjectKey, int]]) -> int:
        """Seed usage + the eviction policy from keys in LRU order.

        ``ordered`` is ``[(ObjectKey, size), ...]`` with the **least
        recently used key first**. The eviction policy is populated via
        ``_notify_keys_stored`` (base class -> ``L2EvictionPolicy`` ->
        ``policy.on_keys_created``). ``on_keys_created`` reverses keys
        within a single call, so a single batch notify of the
        **reversed** list reconstructs ``ordered`` exactly in the policy's
        OrderedDict (oldest at the front == first eviction victim) — same
        result as one notify per key, but O(1) lock/listener traffic
        instead of O(N) (a 100k-object bucket would otherwise thrash on
        startup). ``_key_sizes`` is also seeded so a later ``delete`` of a
        seeded key balances the base-class byte accounting.

        Returns the number of keys seeded.
        """
        to_seed_keys: list[ObjectKey] = []
        to_seed_sizes: list[int] = []
        with self._lock:
            for key, size in ordered:
                if key in self._key_sizes:
                    # Already known from a store in this process — don't
                    # double-count or reorder it.
                    continue
                self._key_sizes[key] = size
                self._object_size_cache[_object_key_to_string(key)] = size
                to_seed_keys.append(key)
                to_seed_sizes.append(size)
        if not to_seed_keys:
            return 0

        # Single batch notify. Suppress per-key tick tracking during the
        # batch (it would stamp the reversed order); seed ticks explicitly
        # afterwards in true oldest->newest order.
        was_tracking = self._track_access_order
        self._track_access_order = False
        try:
            self._notify_keys_stored(
                list(reversed(to_seed_keys)), list(reversed(to_seed_sizes))
            )
        finally:
            self._track_access_order = was_tracking

        if was_tracking:
            with self._lock:
                for key, size in zip(to_seed_keys, to_seed_sizes, strict=True):
                    self._access_tick_counter += 1
                    self._access_ticks[_object_key_to_string(key)] = (
                        size,
                        self._access_tick_counter,
                    )
        return len(to_seed_keys)

    def _seed_from_bucket(self) -> None:
        """Seed usage + eviction index from the bucket on startup.

        Strategy:
        1. If ``s3_persist_lru_index`` is on and a valid sidecar exists,
           restore true access-order from it, reconciled against the live
           listing (drop vanished keys; append new keys by LastModified).
        2. Otherwise fall back to a pure LastModified scan (feature a).
        """
        entries = self._scan_bucket()

        restored = False
        if self._config.s3_persist_lru_index:
            restored = self._restore_from_sidecar(entries)

        if not restored:
            self._seed_from_last_modified(entries)

    def _seed_from_last_modified(
        self, entries: list[tuple[ObjectKey, int, Optional[float]]]
    ) -> None:
        """Feature (a): order by LastModified (oldest first) and seed."""
        ordered = _order_entries_by_last_modified(entries)
        seeded = self._seed_policy_in_order(ordered)
        logger.info(
            "S3L2Adapter seeded %d keys from LastModified ordering (total bytes ~%d)",
            seeded,
            sum(sz for _k, sz in ordered),
        )

    def _restore_from_sidecar(
        self, entries: list[tuple[ObjectKey, int, Optional[float]]]
    ) -> bool:
        """Feature (b): restore true access-order from the sidecar.

        Returns ``True`` when a usable sidecar was found and applied,
        ``False`` when it was absent/corrupt (caller falls back to the
        LastModified scan). On a successful restore, the in-memory
        ``_access_ticks`` map is rebuilt so the next checkpoint preserves
        continuity.
        """
        fut = asyncio.run_coroutine_threadsafe(
            self._execute_get_raw(LRU_INDEX_OBJECT_NAME),
            self._loop,
        )
        try:
            body = fut.result(timeout=60.0)
        except Exception as exc:
            logger.warning("S3L2Adapter sidecar GET failed: %s", exc)
            return False
        if body is None:
            logger.info("S3L2Adapter no LRU sidecar found; falling back to scan")
            return False

        index = _deserialize_lru_index(body)
        if index is None:
            logger.warning(
                "S3L2Adapter LRU sidecar corrupt/unknown-version; "
                "treating as absent and falling back to scan"
            )
            return False

        # Reconcile: only keep sidecar entries whose object still exists
        # in the bucket; append live objects missing from the sidecar
        # using their LastModified as the order proxy (placed before the
        # sidecar entries since a never-recorded object is, by access
        # order, colder than anything we have a tick for).
        live_by_name: dict[str, tuple[ObjectKey, int, Optional[float]]] = {}
        for obj_key, size, mtime in entries:
            live_by_name[_object_key_to_string(obj_key)] = (obj_key, size, mtime)

        sidecar_names = {name for (name, _sz, _tick) in index}

        # Sidecar entries are stored oldest-access first already.
        sidecar_ordered: list[tuple[ObjectKey, int, int]] = []
        for name, size, tick in index:
            live = live_by_name.get(name)
            if live is None:
                continue  # object no longer in bucket — drop it
            obj_key, live_size, _mtime = live
            sidecar_ordered.append((obj_key, live_size, tick))

        # Live objects not in the sidecar, ordered by LastModified.
        missing = [
            (obj_key, size, mtime)
            for name, (obj_key, size, mtime) in live_by_name.items()
            if name not in sidecar_names
        ]
        missing_ordered = _order_entries_by_last_modified(missing)

        # Final LRU order: brand-new (never-recorded) objects are coldest,
        # then sidecar entries by ascending access tick.
        final_ordered: list[tuple[ObjectKey, int]] = [
            (k, sz) for (k, sz) in missing_ordered
        ]
        final_ordered.extend((k, sz) for (k, sz, _tick) in sidecar_ordered)

        seeded = self._seed_policy_in_order(final_ordered)

        # Rebuild the in-memory tick map so the next checkpoint continues
        # from a monotonic counter above every restored tick.
        with self._lock:
            self._access_ticks.clear()
            max_tick = 0
            for obj_key, size, tick in sidecar_ordered:
                self._access_ticks[_object_key_to_string(obj_key)] = (size, tick)
                max_tick = max(max_tick, tick)
            # Missing objects get fresh ticks below the sidecar floor so
            # they remain the coldest in subsequent checkpoints until
            # actually touched.
            self._access_tick_counter = max_tick
        logger.info(
            "S3L2Adapter restored LRU index from sidecar: %d from sidecar, "
            "%d new from listing, %d seeded total",
            len(sidecar_ordered),
            len(missing_ordered),
            seeded,
        )
        return True

    def _snapshot_lru_index(self) -> list[tuple[str, int, int]]:
        """Snapshot the access-order index as ordered serializable triples.

        Returns ``[(object_name, size, tick), ...]`` sorted by ascending
        tick (oldest access first) so the sidecar restores in LRU order.
        """
        with self._lock:
            items = [
                (name, size, tick) for name, (size, tick) in self._access_ticks.items()
            ]
        items.sort(key=lambda t: t[2])
        return items

    def _checkpoint_lru_index(self, timeout: float = 60.0) -> bool:
        """Serialize the access-order index and PUT it to the sidecar.

        Best-effort: returns ``False`` (logged) on any failure so the
        checkpoint loop keeps running. ``timeout`` bounds the PUT wait;
        ``close()`` passes a short one so a slow/unresponsive S3 can't
        block shutdown past the container's termination grace period.
        """
        snapshot = self._snapshot_lru_index()
        if not snapshot:
            return False
        try:
            body = _serialize_lru_index(snapshot)
        except Exception:
            logger.exception("S3L2Adapter failed to serialize LRU index")
            return False
        fut = asyncio.run_coroutine_threadsafe(
            self._execute_put_raw(LRU_INDEX_OBJECT_NAME, body),
            self._loop,
        )
        try:
            ok = fut.result(timeout=timeout)
        except Exception as exc:
            logger.warning("S3L2Adapter LRU checkpoint PUT failed: %s", exc)
            return False
        if ok:
            logger.debug(
                "S3L2Adapter checkpointed LRU index (%d entries, %d bytes)",
                len(snapshot),
                len(body),
            )
        return ok

    def _maybe_start_checkpoint_thread(self) -> None:
        """Launch the periodic sidecar checkpoint thread (once)."""
        if self._checkpoint_thread is not None:
            return
        interval = max(1, int(self._config.s3_lru_index_checkpoint_interval_s))

        def _loop() -> None:
            while not self._checkpoint_stop.wait(interval):
                try:
                    self._checkpoint_lru_index()
                except Exception:
                    logger.exception("S3L2Adapter checkpoint loop error")

        self._checkpoint_thread = threading.Thread(
            target=_loop,
            daemon=True,
            name="s3-l2-lru-checkpoint",
        )
        self._checkpoint_thread.start()
        logger.info(
            "S3L2Adapter LRU checkpoint thread started (interval=%ds)", interval
        )


# ---------------------------------------------------------------------------
# Module helpers for seeding (pure; unit-tested without S3)
# ---------------------------------------------------------------------------


def _order_entries_by_last_modified(
    entries: list[tuple[ObjectKey, int, Optional[float]]],
) -> list[tuple[ObjectKey, int]]:
    """Order ``(ObjectKey, size, last_modified)`` triples oldest-first.

    Oldest ``LastModified`` sorts to the front (== first LRU eviction
    victim). Objects with an unknown ``LastModified`` (``None``) are
    treated as oldest (``-inf``) so a malformed timestamp can't protect
    an object from eviction. Returns ``(ObjectKey, size)`` pairs in LRU
    order, ready for :meth:`S3L2Adapter._seed_policy_in_order`.
    """

    def sort_key(item: tuple[ObjectKey, int, Optional[float]]) -> float:
        mtime = item[2]
        return mtime if mtime is not None else float("-inf")

    ordered = sorted(entries, key=sort_key)
    return [(obj_key, size) for obj_key, size, _mtime in ordered]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


register_l2_adapter_type("s3", S3L2AdapterConfig)


def _create_s3_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    return S3L2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("s3", _create_s3_adapter)
