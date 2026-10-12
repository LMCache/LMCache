# SPDX-License-Identifier: Apache-2.0
"""OpenDS implementation of the shared GDS contracts.

OpenDS is a vendor-neutral accelerator direct storage API modeled on cuFile.
Its aisio backend, ``libopends_aisio.so``, moves data between NVMe and GPU
memory by PCIe P2P DMA through xNVMe. This module reaches it through the
``opends`` Python package that ships with OpenDS.
"""

# Standard
from types import ModuleType
from typing import Any, ClassVar
import os

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.gds_backends._driver import SharedDriver
from lmcache.v1.gpu_connector.gds_backends._file import SLAB_FILE_NAME
from lmcache.v1.gpu_connector.gds_backends.base import (
    GDSBackend,
    GDSHandle,
    Submission,
)


def _opends() -> ModuleType:
    """Import the Python package that ships with OpenDS.

    The import loads ``libopends_aisio.so``, so it runs on first use and
    not when this module is imported.

    Returns:
        The ``opends`` package.

    Raises:
        RuntimeError: If the ``opends`` package of OpenDS 0.3.0 or later is
            not installed.
        OSError: If the package does not find the library.
    """
    try:
        # Third Party
        import opends
    except ImportError as error:
        raise RuntimeError(
            "opends requires the Python package of OpenDS 0.3.0 or later; "
            "install it from the OpenDS source tree (pip install <opends>/python)"
        ) from error
    if not hasattr(opends, "STREAM_FIXED"):
        raise RuntimeError(
            "opends requires the Python package of OpenDS 0.3.0 or later; "
            "the installed opends package is older or another project"
        )
    return opends


class _StreamSubmission(Submission):
    """A Submission whose native storage is an OpenDS ``StreamOp``."""

    __slots__ = ("_op",)

    def __init__(self, op: Any, size: int, file_offset: int, buf_offset: int) -> None:
        super().__init__(size, file_offset, buf_offset)
        self._op = op

    @property
    def bytes_done(self) -> int:
        """Return the byte count as OpenDS writes it, cuFile's convention.

        Returns:
            0 until the stream has reached the IO, then the byte count, -1 for
            an IO error, or the negated ``opends_op_error_t`` code.
        """
        return self._op.nbytes


class Backend(GDSBackend):
    """Own the OpenDS driver session and its registration operations."""

    name = "opends"
    _driver = SharedDriver()
    # The open ``opends.Driver``. ``_driver`` opens and closes it under its lock.
    _session: ClassVar[Any] = None

    def validate_environment(self) -> None:
        if torch.version.cuda is None:
            raise ValueError("opends requires a CUDA PyTorch build")

    def open_slab(self, location: str, size: int, direct_io: bool) -> "AsyncHandle":
        """Open the preallocated slab file; never create or resize it.

        aisio reads through the extent index that xal-server takes when it
        starts. Blocks allocated later are not in the index, and a read of them
        fails or returns other data.

        Args:
            location: Directory that holds ``lmcache_gds_slab.bin``.
            size: Slab size in bytes. The file must have exactly this size and
                no holes.
            direct_io: Must be True. OpenDS registers only ``O_DIRECT``
                descriptors.

        Returns:
            The owning handle of the registered slab.

        Raises:
            ValueError: If ``direct_io`` is False, or the slab is missing,
                sparse, or of another size.
            opends.OpenDSError: If OpenDS does not register the file.
        """
        if not direct_io:
            raise ValueError("opends requires --gds-l1-use-direct-io")
        path = os.path.join(location, SLAB_FILE_NAME)
        stat = os.stat(path) if os.path.isfile(path) else None
        if stat is None or stat.st_size != size or stat.st_blocks * 512 < size:
            raise ValueError(
                f"opends requires a preallocated {size}-byte slab at {path}; "
                "create it with fallocate before xal-server starts"
            )
        self._ensure_driver_open()
        return AsyncHandle(_opends().OpenDSFile(path, "r+", use_direct_io=True), path)

    def register_buffer(self, buf: torch.Tensor) -> None:
        if not buf.is_cuda:
            raise ValueError("register_buffer: tensor must be on CUDA")
        self._ensure_driver_open()
        _opends().register_buffer(buf)

    def deregister_buffer(self, buf: torch.Tensor) -> None:
        _opends().deregister_buffer(buf)

    def register_stream(self, raw_stream: int) -> None:
        """Register with the FIXED flags: a submission's size and offsets are
        final when it is made."""
        self._ensure_driver_open()
        opends = _opends()
        opends.register_stream(raw_stream, opends.STREAM_FIXED)

    def deregister_stream(self, raw_stream: int) -> None:
        _opends().deregister_stream(raw_stream)

    def _open_driver(self) -> None:
        # aisio captures the CUDA context current on this thread and accepts
        # IO from every thread on it. LMCache threads run under PyTorch's
        # primary context, so that is what it captures.
        Backend._session = _opends().Driver()

    def _close_driver(self) -> None:
        Backend._session.close()


class AsyncHandle(GDSHandle):
    """An owning OpenDS slab file with stream-ordered IO."""

    def __init__(self, file: Any, path: str) -> None:
        super().__init__(path)
        self._file = file

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        op = self._file.read_stream(buf_base, size, file_offset, buf_offset, raw_stream)
        return _StreamSubmission(op, size, file_offset, buf_offset)

    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        op = self._file.write_stream(
            buf_base, size, file_offset, buf_offset, raw_stream
        )
        return _StreamSubmission(op, size, file_offset, buf_offset)

    def close(self) -> None:
        """Deregister and close the slab file; repeated calls do nothing."""
        self._file.close()
