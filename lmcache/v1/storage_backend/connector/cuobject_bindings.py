# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional
import json

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy import of the pybind11 C++ extension.
#
# The extension is built by setup.py as ``lmcache.lmcache_cuobject``
# when the cuObjClient SDK (cuobjclient.h) is found at build time.
# If it was not compiled (e.g. sdist install, missing SDK) the import
# is deferred and a clear error is raised at construction time.
# ---------------------------------------------------------------------------
try:
    # First Party
    from lmcache.lmcache_cuobject import CuObjectClient  # pybind11 C++ class
except ImportError:
    CuObjectClient = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Constants (mirrored from C++ for convenience)
#
# These match the official cuObjClient API Specification v1.0.0.
# ---------------------------------------------------------------------------

# Protocol enum for cuObjClient initialization
# Official: CUOBJ_PROTO_RDMA_DC_V1 = 1001 (RDMA Dynamically Connected v1)
CUOBJ_PROTO_RDMA_DC_V1 = 1001

# Return codes (official: cuObjErr_t)
CU_OBJ_SUCCESS = 0
CU_OBJ_FAIL = 1

# Maximum memory registration size per call (official: 4 GiB)
CUOBJ_MAX_MEMORY_REG_SIZE = 4 * 1024 * 1024 * 1024


@dataclass
class CuObjConfig:
    """Configuration for the cuObject client wrapper."""

    # cuObject transport protocol.
    # Official: CUOBJ_PROTO_RDMA_DC_V1 (1001)
    proto: int = CUOBJ_PROTO_RDMA_DC_V1


class CuObjClientWrapper:
    """Python wrapper around the cuObject pybind11 C++ client.

    Provides:

    * One-time pool registration (``register_pool`` / ``deregister_pool``)
    * Per-request RDMA token generation (``prepare_put`` / ``prepare_get``)
    * RDMA reply verification (``parse_rdma_reply``)
    * Connection status (``is_connected``)

    Thread safety
    -------------
    ``prepare_put`` and ``prepare_get`` call ``cuMemObjGetRDMAToken``
    which writes to a caller-provided output pointer, so concurrent
    calls from different threads each get their own descriptor.
    No internal mutex is needed.
    """

    def __init__(self, config: Optional[CuObjConfig] = None):
        if config is None:
            config = CuObjConfig()
        self._config = config
        self._client = None  # Set early for safe __del__

        if CuObjectClient is None:
            raise ImportError(
                "The lmcache.lmcache_cuobject C++ extension is not "
                "available. Rebuild LMCache with the cuObjClient SDK "
                "installed (CUDA Toolkit >= 13.1.1 with cuObject support)."
            )

        # -- Create the C++ client (build-time linked cuObjClient) -----------
        self._client = CuObjectClient(config.proto)
        logger.info(f"cuObject client initialised (proto={config.proto})")

    # -- Public API ----------------------------------------------------------

    def register_pool(self, ptr: int, size: int) -> tuple[int, int]:
        """Register a contiguous pinned memory pool for RDMA.

        Must be called once at init time with the base address and byte
        size of the ``MixedMemoryAllocator`` / ``PinMemoryAllocator``
        buffer.

        Args:
            ptr: Base address of the memory region.
            size: Byte size.  Must be < 4 GiB (CUOBJ_MAX_MEMORY_REG_SIZE).

        Returns:
            ``(ptr, size)`` tuple that serves as the registration handle
            for ``deregister_pool``.
        """
        result = self._client.register_pool(ptr, size)
        logger.info(f"Registered RDMA memory pool: ptr=0x{ptr:x}, size={size} bytes")
        return result

    def deregister_pool(self, handle: tuple[int, int]) -> None:
        """Deregister a previously registered memory pool.

        Args:
            handle: The ``(ptr, size)`` tuple returned by ``register_pool``.
                    Only ptr is passed to the library (per the official API,
                    cuMemObjPutDescriptor takes only the pointer).
        """
        ptr, size = handle
        rc = self._client.deregister_pool(ptr)
        if rc != CU_OBJ_SUCCESS:
            logger.warning(f"cuMemObjPutDescriptor returned error {rc} (ptr=0x{ptr:x})")
        else:
            logger.info(
                f"Deregistered RDMA memory pool: ptr=0x{ptr:x}, size={size} bytes"
            )

    def prepare_put(self, ptr: int, size: int) -> str:
        """Generate an RDMA token for a PUT operation.

        Calls ``cuMemObjGetRDMAToken`` with ``CUOBJ_PUT``.  The C++
        layer automatically computes the buffer offset from the
        registered pool base address.

        Args:
            ptr: Data pointer of the ``MemoryObj`` to upload (must be
                 within the registered pool).
            size: Byte size of the data.

        Returns:
            The ``x-amz-rdma-token`` header value as a string.
        """
        return self._client.prepare_put(ptr, size)

    def prepare_get(self, ptr: int, size: int) -> str:
        """Generate an RDMA token for a GET operation.

        Calls ``cuMemObjGetRDMAToken`` with ``CUOBJ_GET``.  The C++
        layer automatically computes the buffer offset from the
        registered pool base address.

        Args:
            ptr: Destination pointer where the server will
                 ``RDMA_WRITE`` data (must be within the registered
                 pool).
            size: Expected byte size of the data.

        Returns:
            The ``x-amz-rdma-token`` header value as a string.
        """
        return self._client.prepare_get(ptr, size)

    def is_connected(self) -> bool:
        """Check if the cuObject client is connected and ready.

        Returns:
            True if connected, False otherwise.
        """
        return self._client.is_connected()

    # Keywords that unambiguously indicate RDMA success.
    _SUCCESS_KEYWORDS: frozenset = frozenset(
        {"ok", "success", "complete", "completed", "done"}
    )
    # Prefixes that unambiguously indicate RDMA failure.
    _ERROR_PREFIXES: tuple = ("error", "fail", "fault")

    @staticmethod
    def parse_rdma_reply(reply_header: Optional[str]) -> bool:
        """Parse the ``x-amz-rdma-reply`` response header.

        Returns *True* if the header indicates successful RDMA
        completion, *False* otherwise.

        Supported reply formats
        -----------------------
        * **JSON object** -- must contain a ``"status"`` key whose value
          is one of the recognised success keywords (``ok``, ``success``,
          ``complete``, ``completed``, ``done``).  An ``"error"`` key
          with a truthy value is always treated as failure.
        * **Numeric string** -- ``"0"`` maps to ``CU_OBJ_SUCCESS``;
          any other integer is treated as an error code.
        * **Plain keyword** -- one of the recognised success / error
          keywords listed above.

        Any unrecognised, non-empty value is logged as a warning and
        treated as failure to prevent silent data corruption.
        """
        if not reply_header or not reply_header.strip():
            return False

        stripped = reply_header.strip()
        lower = stripped.lower()

        # -- 1. Try JSON -------------------------------------------------
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return CuObjClientWrapper._parse_json_reply(data)
        except (json.JSONDecodeError, ValueError):
            pass  # Not JSON -- fall through to simpler formats.

        # -- 2. Numeric error code ----------------------------------------
        try:
            code = int(stripped)
            if code == CU_OBJ_SUCCESS or code in (200, 201, 206):
                return True
            logger.warning(f"RDMA reply error code: {code}")
            return False
        except ValueError:
            pass

        # -- 3. Plain-text keywords ---------------------------------------
        if lower in CuObjClientWrapper._SUCCESS_KEYWORDS:
            return True

        if any(lower.startswith(p) for p in CuObjClientWrapper._ERROR_PREFIXES):
            logger.warning(f"RDMA reply indicates failure: {reply_header}")
            return False

        # -- 4. Unrecognised format -- fail safe ---------------------------
        logger.warning(
            f"RDMA reply has unrecognised format, treating as failure: {reply_header!r}"
        )
        return False

    @staticmethod
    def _parse_json_reply(data: dict) -> bool:
        """Interpret a JSON-decoded RDMA reply dict.

        Returns *True* only when the reply unambiguously signals
        success; *False* (with a warning log) otherwise.
        """
        # Explicit error field takes priority.
        err = data.get("error")
        if err:
            logger.warning(f"RDMA reply indicates error: {err}")
            return False

        status = str(data.get("status", "")).strip().lower()
        if status in CuObjClientWrapper._SUCCESS_KEYWORDS:
            return True

        if not status:
            logger.warning(f"RDMA reply JSON missing 'status' field: {data}")
        else:
            msg = data.get("message", data.get("msg", ""))
            logger.warning(
                f"RDMA reply status indicates failure: "
                f"status={status!r}, message={msg!r}"
            )
        return False

    def close(self) -> int:
        """Destroy the cuObject client and release RDMA resources."""
        rc = CU_OBJ_SUCCESS
        if self._client is not None:
            rc = self._client.close()
            self._client = None
            if rc != CU_OBJ_SUCCESS:
                logger.warning(f"cuObject client close() returned error {rc}")
        return rc

    def __del__(self):
        self.close()
