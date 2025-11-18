# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional


@dataclass
class WorkerInfo:
    instance_id: str
    worker_id: int
    ip: str
    port: int
    distributed_url: Optional[str]
    registration_time: float
    last_heartbeat_time: float
