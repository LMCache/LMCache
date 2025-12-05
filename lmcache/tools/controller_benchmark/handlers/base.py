"""Base handler class for benchmark operations (Strategy Pattern)"""

# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Local
    from ..benchmark import TestData, ZMQControllerBenchmark


class OperationHandler(ABC):
    """Base class for benchmark operation handlers (Strategy Pattern)"""

    @property
    @abstractmethod
    def operation_name(self) -> str:
        """Return the operation name for registration"""
        pass

    @abstractmethod
    def create_message(
        self, benchmark: "ZMQControllerBenchmark", test_data: "TestData"
    ) -> Any:
        """Create a message for this operation"""
        pass

    @abstractmethod
    def get_message_count(self, benchmark: "ZMQControllerBenchmark") -> int:
        """Get the number of messages in a single operation"""
        pass

    def use_req_socket(self) -> bool:
        """Whether this operation uses REQ-REP socket"""
        return False
