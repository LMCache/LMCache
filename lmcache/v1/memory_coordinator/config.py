# SPDX-License-Identifier: Apache-2.0
"""Configuration for the standalone Memory Coordinator service."""

# Standard
from dataclasses import dataclass
import os

_ENV_PREFIX = "LMCACHE_MEMORY_COORDINATOR_"


@dataclass(frozen=True)
class MemoryCoordinatorConfig:
    """Startup configuration for one Memory Coordinator process.

    The coordinator is deliberately single-everything for M0: one process,
    one Uvicorn worker, one region, one immutable layout profile. The
    bearer token is read from ``token_file`` at startup and never appears
    in CLI arguments, logs, or status responses.

    Attributes:
        host: Interface the HTTP server binds to.
        port: Port the HTTP server binds to.
        token_file: Absolute path to the bearer-token file (mounted from a
            Kubernetes Secret in cluster deployments).
        state_file: Absolute startup-latch path on persistent storage shared
            by every replacement coordinator for this region. Remove it only
            after all MP servers are stopped; it contains no recoverable state.
        region_id: Operator-provisioned stable identity of the shared
            physical region.
        capacity_bytes: Logical shared-region capacity in bytes.
        alignment_bytes: Allocation alignment in bytes; positive power of
            two.
        layout_id: Operator-supplied immutable layout-profile fingerprint.
    """

    host: str = "0.0.0.0"
    port: int = 9400
    token_file: str = ""
    state_file: str = ""
    region_id: str = ""
    capacity_bytes: int = 0
    alignment_bytes: int = 2 * 1024 * 1024
    layout_id: str = ""

    def __post_init__(self) -> None:
        """Validate the complete region contract.

        Raises:
            ValueError: A field is empty, out of range, or the alignment is
                not a positive power of two.
        """
        if not self.host.strip():
            raise ValueError("host must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not os.path.isabs(self.token_file):
            raise ValueError("token_file must be an absolute path")
        if not os.path.isabs(self.state_file):
            raise ValueError("state_file must be an absolute persistent path")
        if not self.region_id.strip():
            raise ValueError("region_id must not be empty")
        if self.capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        alignment = self.alignment_bytes
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("alignment_bytes must be a positive power of two")
        if not self.layout_id.strip():
            raise ValueError("layout_id must not be empty")

    @classmethod
    def from_env(cls) -> "MemoryCoordinatorConfig":
        """Build a config from ``LMCACHE_MEMORY_COORDINATOR_*`` variables.

        Returns:
            The parsed and validated configuration.

        Raises:
            ValueError: A variable is malformed or the resulting config is
                invalid.
        """

        def value(name: str, default: str | int) -> str:
            return os.environ.get(_ENV_PREFIX + name, str(default))

        return cls(
            host=value("HOST", cls.host),
            port=int(value("PORT", cls.port)),
            token_file=value("TOKEN_FILE", cls.token_file),
            state_file=value("STATE_FILE", cls.state_file),
            region_id=value("REGION_ID", cls.region_id),
            capacity_bytes=int(value("CAPACITY_BYTES", cls.capacity_bytes)),
            alignment_bytes=int(value("ALIGNMENT_BYTES", cls.alignment_bytes)),
            layout_id=value("LAYOUT_ID", cls.layout_id),
        )


def read_token_file(token_file: str) -> str:
    """Read a nonempty bearer token from an absolute file path.

    Args:
        token_file: Absolute path to the token file.

    Returns:
        The token with surrounding whitespace stripped.

    Raises:
        ValueError: The path is not an absolute regular file, or the file
            is empty after stripping whitespace.
    """
    if not os.path.isabs(token_file) or not os.path.isfile(token_file):
        raise ValueError("token file must be an absolute regular file")
    with open(token_file, encoding="utf-8") as handle:
        token = handle.read().strip()
    if (
        not token
        or not token.isascii()
        or not token.isprintable()
        or any(character.isspace() for character in token)
    ):
        raise ValueError(
            "token file must contain one printable non-whitespace ASCII token"
        )
    return token
