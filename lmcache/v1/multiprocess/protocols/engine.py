# SPDX-License-Identifier: Apache-2.0
"""Python response types used by engine-driven gRPC handlers."""

# Standard
from dataclasses import dataclass, field


@dataclass
class PrepareStoreResponse:
    """Response returned by ``PrepareStore``."""

    context: dict = field(default_factory=dict)


@dataclass
class PrepareRetrieveResponse:
    """Response returned by ``PrepareRetrieve``."""

    success: bool
    data: bytes = b""
    context: dict = field(default_factory=dict)


@dataclass
class RegisterEngineDrivenContextResponse:
    """Response returned by ``RegisterKvCacheEngineDrivenContext``."""

    shm_name: str = ""
    pool_size: int = 0


__all__ = [
    "PrepareRetrieveResponse",
    "PrepareStoreResponse",
    "RegisterEngineDrivenContextResponse",
]
