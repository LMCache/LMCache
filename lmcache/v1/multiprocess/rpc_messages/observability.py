# SPDX-License-Identifier: Apache-2.0
"""Payload contracts for multiprocess observability RPCs."""

# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.multiprocess.custom_types import BlockAllocationRecord
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.rpc_messages.registry import register_rpc_message_types


@dataclass(frozen=True)
class ReportBlockAllocationRequest:
    """Report GPU block-allocation changes."""

    instance_id: int
    model_name: str
    records: list[BlockAllocationRecord]


@dataclass(frozen=True)
class ReportBlockAllocationResponse:
    """Acknowledge a block-allocation report."""


register_rpc_message_types(
    RequestType.REPORT_BLOCK_ALLOCATION,
    ReportBlockAllocationRequest,
    ReportBlockAllocationResponse,
)


__all__ = ["ReportBlockAllocationRequest", "ReportBlockAllocationResponse"]
