# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional
import ctypes
import ctypes.util
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# cuObject C API type definitions
#
# These are derived from the NVIDIA cuObject documentation (v1.16).
# The actual struct layouts and function signatures should be verified
# against the cuobject.h header shipped with CUDA Toolkit >= 13.1.1.
# ---------------------------------------------------------------------------

# Protocol enum for cuObjClient initialization
CUOBJ_PROTO_DC = 0  # Dynamic Connection (InfiniBand / RoCEv2)

# Return codes
CUOBJ_SUCCESS = 0

# Callback type invoked by cuObjPut/cuObjGet to deliver the RDMA token.
#   int (*rdma_token_cb)(void *user_ctx, const char *token, size_t token_len)
RDMA_TOKEN_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,  # return code
    ctypes.c_void_p,  # user_ctx
    ctypes.c_char_p,  # token data
    ctypes.c_size_t,  # token length
)


class CUObjOps(ctypes.Structure):
    """Callback operations struct passed to cuObjClient constructor.

    The cuObject client library calls ``rdma_token_cb`` after it has
    registered the memory region and generated the RDMA token.  The
    callback is responsible for capturing the token so the caller can
    inject it into the HTTP ``x-amz-rdma-token`` header.

    .. note::
       The exact field layout must be verified against ``cuobject.h``.
       Additional fields (e.g. for error callbacks) may exist; they
       should be added here as ``ctypes.c_void_p`` placeholders.
    """

    _fields_ = [
        ("rdma_token_cb", RDMA_TOKEN_CALLBACK),
        ("user_ctx", ctypes.c_void_p),
    ]


@dataclass
class CuObjConfig:
    """Configuration for the cuObject client wrapper."""

    # Explicit path to the shared library.  When *None* the wrapper
    # falls back to ``ctypes.util.find_library("cuobject_client")``.
    lib_path: Optional[str] = None

    # RDMA NIC device name (e.g. ``"mlx5_0"``).  *None* = auto-select.
    nic_device: Optional[str] = None

    # cuObject transport protocol.  Currently only DC is supported.
    proto: int = CUOBJ_PROTO_DC


class CuObjClientWrapper:
    """Python wrapper around ``libcuobject_client.so``.

    Provides:

    * One-time pool registration (``register_pool`` / ``deregister_pool``)
    * Per-request RDMA token generation (``prepare_put`` / ``prepare_get``)
    * RDMA reply verification (``parse_rdma_reply``)

    Thread safety
    -------------
    ``prepare_put`` and ``prepare_get`` are serialised with a lock so
    that the callback-captured token cannot be clobbered by a concurrent
    call.  ``register_pool`` / ``deregister_pool`` are expected to be
    called only during init / shutdown.
    """

    def __init__(self, config: Optional[CuObjConfig] = None):
        if config is None:
            config = CuObjConfig()
        self._config = config
        self._lock = threading.Lock()
        self._captured_token: Optional[bytes] = None
        self._handle: Optional[ctypes.c_void_p] = None

        # -- Load the shared library -----------------------------------------
        self._lib = self._load_library(config.lib_path)

        # -- Resolve C function symbols --------------------------------------
        self._resolve_symbols()

        # -- Set up the callback that captures RDMA tokens -------------------
        # Must be stored as an instance attribute so the prevent GC of the
        # prevent garbage-collection of the ctypes callback closure.
        self._token_cb = RDMA_TOKEN_CALLBACK(self._on_rdma_token)

        # -- Initialise the cuObject client ----------------------------------
        self._ops = CUObjOps(
            rdma_token_cb=self._token_cb,
            user_ctx=None,
        )
        self._handle = ctypes.c_void_p()
        rc = self._lib.cuObjClientCreate(
            ctypes.byref(self._handle),
            ctypes.byref(self._ops),
            ctypes.c_int(config.proto),
        )
        if rc != CUOBJ_SUCCESS:
            raise RuntimeError(
                f"cuObjClientCreate failed with error code {rc}"
            )
        logger.info(
            "cuObject client initialised "
            f"(proto={config.proto}, nic={config.nic_device})"
        )

    # -- Library loading -----------------------------------------------------

    @staticmethod
    def _load_library(lib_path: Optional[str]) -> ctypes.CDLL:
        """Load ``libcuobject_client.so`` via *ctypes*."""
        if lib_path is None:
            lib_path = ctypes.util.find_library("cuobject_client")
        if lib_path is None:
            raise ImportError(
                "Cannot find libcuobject_client.so. "
                "Ensure CUDA Toolkit >= 13.1.1 is installed and "
                "LD_LIBRARY_PATH includes the cuObject library directory, "
                "or set 'cuobject_lib_path' in extra_config."
            )
        try:
            lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise ImportError(
                f"Failed to load cuObject client library from {lib_path}: "
                f"{exc}"
            ) from exc
        logger.info(f"Loaded cuObject client library from {lib_path}")
        return lib

    def _resolve_symbols(self):
        """Resolve and type-annotate the C API entry points.

        .. note::
           The exact signatures must be verified against ``cuobject.h``.
           The placeholders below reflect the documented API surface.
        """
        lib = self._lib

        # int cuObjClientCreate(void **handle, CUObjOps_t *ops, int proto)
        lib.cuObjClientCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(CUObjOps),
            ctypes.c_int,
        ]
        lib.cuObjClientCreate.restype = ctypes.c_int

        # int cuObjClientDestroy(void *handle)
        lib.cuObjClientDestroy.argtypes = [ctypes.c_void_p]
        lib.cuObjClientDestroy.restype = ctypes.c_int

        # int cuObjRegisterMemory(void *handle, void *ptr, size_t size)
        lib.cuObjRegisterMemory.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.cuObjRegisterMemory.restype = ctypes.c_int

        # int cuObjDeregisterMemory(void *handle, void *ptr, size_t size)
        lib.cuObjDeregisterMemory.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.cuObjDeregisterMemory.restype = ctypes.c_int

        # int cuObjPut(void *handle, void *ptr, size_t size, off_t offset)
        lib.cuObjPut.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int64,
        ]
        lib.cuObjPut.restype = ctypes.c_int

        # int cuObjGet(void *handle, void *ptr, size_t size, off_t offset)
        lib.cuObjGet.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int64,
        ]
        lib.cuObjGet.restype = ctypes.c_int

    # -- RDMA token callback -------------------------------------------------

    def _on_rdma_token(
        self,
        _user_ctx: ctypes.c_void_p,
        token_data: bytes,
        token_len: int,
    ) -> int:
        """Called by the cuObject library with the generated RDMA token."""
        self._captured_token = token_data[:token_len]
        return CUOBJ_SUCCESS

    # -- Public API ----------------------------------------------------------

    def register_pool(self, ptr: int, size: int) -> tuple[int, int]:
        """Register a contiguous pinned memory pool for RDMA.

        Must be called once at init time with the base address and byte
        size of the ``MixedMemoryAllocator`` / ``PinMemoryAllocator``
        buffer.

        Returns:
            ``(ptr, size)`` tuple that serves as the registration handle
            for ``deregister_pool``.
        """
        rc = self._lib.cuObjRegisterMemory(
            self._handle,
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(size),
        )
        if rc != CUOBJ_SUCCESS:
            raise RuntimeError(
                f"cuObjRegisterMemory failed (ptr=0x{ptr:x}, "
                f"size={size}): error {rc}"
            )
        logger.info(
            f"Registered RDMA memory pool: "
            f"ptr=0x{ptr:x}, size={size} bytes"
        )
        return (ptr, size)

    def deregister_pool(self, handle: tuple[int, int]) -> None:
        """Deregister a previously registered memory pool."""
        ptr, size = handle
        rc = self._lib.cuObjDeregisterMemory(
            self._handle,
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(size),
        )
        if rc != CUOBJ_SUCCESS:
            logger.warning(
                f"cuObjDeregisterMemory returned error {rc} "
                f"(ptr=0x{ptr:x}, size={size})"
            )
        else:
            logger.info(
                f"Deregistered RDMA memory pool: "
                f"ptr=0x{ptr:x}, size={size} bytes"
            )

    def prepare_put(self, ptr: int, size: int, offset: int = 0) -> str:
        """Prepare an RDMA-accelerated PUT operation.

        Internally calls ``cuObjPut`` which registers the sub-region
        (within the already-registered pool) and invokes the callback
        with the RDMA token.

        Args:
            ptr: Data pointer of the ``MemoryObj`` to upload.
            size: Byte size of the data.
            offset: Byte offset within the object (default 0).

        Returns:
            The ``x-amz-rdma-token`` header value as a string.
        """
        with self._lock:
            self._captured_token = None
            rc = self._lib.cuObjPut(
                self._handle,
                ctypes.c_void_p(ptr),
                ctypes.c_size_t(size),
                ctypes.c_int64(offset),
            )
            if rc != CUOBJ_SUCCESS:
                raise RuntimeError(
                    f"cuObjPut failed (ptr=0x{ptr:x}, size={size}): "
                    f"error {rc}"
                )
            if self._captured_token is None:
                raise RuntimeError(
                    "cuObjPut succeeded but RDMA token callback "
                    "was not invoked"
                )
            return self._captured_token.decode("ascii")

    def prepare_get(self, ptr: int, size: int, offset: int = 0) -> str:
        """Prepare an RDMA-accelerated GET operation.

        Internally calls ``cuObjGet`` which registers the destination
        buffer and invokes the callback with the RDMA token.

        Args:
            ptr: Destination pointer where the server will
                 ``RDMA_WRITE`` data.
            size: Expected byte size of the data.
            offset: Byte offset (default 0).

        Returns:
            The ``x-amz-rdma-token`` header value as a string.
        """
        with self._lock:
            self._captured_token = None
            rc = self._lib.cuObjGet(
                self._handle,
                ctypes.c_void_p(ptr),
                ctypes.c_size_t(size),
                ctypes.c_int64(offset),
            )
            if rc != CUOBJ_SUCCESS:
                raise RuntimeError(
                    f"cuObjGet failed (ptr=0x{ptr:x}, size={size}): "
                    f"error {rc}"
                )
            if self._captured_token is None:
                raise RuntimeError(
                    "cuObjGet succeeded but RDMA token callback "
                    "was not invoked"
                )
            return self._captured_token.decode("ascii")

    @staticmethod
    def parse_rdma_reply(reply_header: str) -> bool:
        """Parse the ``x-amz-rdma-reply`` response header.

        Returns *True* if the header indicates successful RDMA
        completion, *False* otherwise.

        The exact semantics of the reply value are defined by the
        cuObject protocol; for now we treat any non-empty value as
        success.
        """
        if not reply_header:
            return False
        # TODO: parse structured reply from cuObjServer if needed
        return True

    def close(self) -> None:
        """Destroy the cuObject client and release RDMA resources."""
        if self._handle is not None:
            rc = self._lib.cuObjClientDestroy(self._handle)
            if rc != CUOBJ_SUCCESS:
                logger.warning(
                    f"cuObjClientDestroy returned error {rc}"
                )
            self._handle = None
            logger.info("cuObject client destroyed")

    def __del__(self):
        self.close()
