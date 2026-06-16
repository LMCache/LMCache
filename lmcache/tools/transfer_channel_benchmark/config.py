# SPDX-License-Identifier: Apache-2.0
"""Configuration for the transfer channel throughput benchmark."""

# Standard
from dataclasses import dataclass

_UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
}

# Defaults expressed as byte counts so they can be used directly as argparse
# defaults (argparse applies ``type=parse_size`` only to string inputs).
DEFAULT_BUFFER_SIZE = 8 * 1024**3
DEFAULT_PAGE_SIZE = 512 * 1024
DEFAULT_OBJECT_SIZE = 10 * 1024**2


def parse_size(value: str) -> int:
    """Parse a human-friendly size string into a byte count.

    Args:
        value: A size such as ``"8GB"``, ``"512KB"``, ``"10MB"`` or a plain
            integer byte count (e.g. ``"1048576"``).

    Returns:
        The size in bytes.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    text = str(value).strip().upper()
    for suffix in ("TB", "GB", "MB", "KB", "T", "G", "M", "K", "B"):
        if text.endswith(suffix) and text[: -len(suffix)].strip():
            return int(float(text[: -len(suffix)].strip()) * _UNITS[suffix])
    return int(text)


@dataclass
class BenchmarkConfig:
    """Resolved configuration for one benchmark process (server or client).

    Attributes:
        role: ``"server"`` or ``"client"``.
        transfer_channel_type: The transfer channel implementation to use
            (e.g. ``"nixl"``).
        nixl_backend: The nixl backend name (e.g. ``"UCX"``). Only used by the
            nixl transfer channel type.
        url: Server role binds its transfer-channel server here; client role
            dials this as the peer (server) advertise url to read from.
        listen_url: Client role binds its own (mandatory) transfer-channel
            server here. It never receives reads in this benchmark.
        control_url: Benchmark catalog side-channel. The server binds a ZMQ REP
            socket here; the client connects to fetch the source object catalog.
        buffer_size: Size in bytes of the server's registered L1 region.
        page_size: Page / alignment size in bytes (the L1 ``align_bytes``).
        object_size: Size in bytes of each transferred object (multiple of
            ``page_size``).
        num_objects: Number of objects transferred per read.
        num_source_objects: Number of source objects the server allocates. The
            client reads a random ``num_objects``-sized subset of these.
        use_lazy: Whether the L1 memory manager uses lazy allocation.
        iters: Number of measured read iterations.
        warmup: Number of warmup read iterations (not measured).
        seed: RNG seed for selecting the read subset.
        verify: Whether to verify transferred bytes against a known pattern.
        server_timeout: Seconds the server stays up serving catalog requests.
    """

    role: str
    transfer_channel_type: str = "nixl"
    nixl_backend: str = "UCX"
    url: str = "127.0.0.1:7600"
    listen_url: str = "0.0.0.0:7601"
    control_url: str = "0.0.0.0:7610"
    buffer_size: int = DEFAULT_BUFFER_SIZE
    page_size: int = DEFAULT_PAGE_SIZE
    object_size: int = DEFAULT_OBJECT_SIZE
    num_objects: int = 100
    num_source_objects: int = 0
    use_lazy: bool = False
    iters: int = 5
    warmup: int = 1
    seed: int = 0
    verify: bool = False
    server_timeout: float = 1800.0

    def __post_init__(self) -> None:
        """Apply derived defaults and validate the configuration.

        Raises:
            ValueError: If any field is out of range or mutually inconsistent.
        """
        if self.num_source_objects <= 0:
            self.num_source_objects = 5 * self.num_objects

        if self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size}")
        if self.object_size <= 0 or self.object_size % self.page_size != 0:
            raise ValueError(
                f"object_size ({self.object_size}) must be a positive multiple "
                f"of page_size ({self.page_size})"
            )
        if self.num_objects < 1:
            raise ValueError(f"num_objects must be >= 1, got {self.num_objects}")
        if self.num_source_objects < self.num_objects:
            raise ValueError(
                f"num_source_objects ({self.num_source_objects}) must be >= "
                f"num_objects ({self.num_objects})"
            )
