# SPDX-License-Identifier: Apache-2.0
"""
Abstract base class for request telemetry.

This module provides the interface for tracking request-level events
in LMCache, such as when a request finishes and its associated async
save operations complete.
"""

# Standard
from abc import ABC, abstractmethod
from typing import Any


class RequestTelemetry(ABC):
    """
    Abstract base class for request telemetry.

    This class defines the interface for capturing request-level telemetry
    events. Implementations can log events, emit metrics, or perform other
    actions when specific request lifecycle events occur.

    Example:
        class LoggingTelemetry(RequestTelemetry):
            def on_request_save_finished(
                self,
                request_id: str,
                num_tokens_saved: int,
                save_duration_ms: float,
            ) -> None:
                logger.info(
                    f"Request {request_id} save finished: "
                    f"{num_tokens_saved} tokens in {save_duration_ms}ms"
                )
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def on_request_store_finished(
        self,
        request_ids_set: set[str],
        model_name: str,
        world_size: int,
        kv_rank: int,
    ) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __del__(self) -> None:
        self.close()
