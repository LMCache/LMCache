# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import Dict, Optional, Tuple, Union

# Third Party
import msgspec


class MsgBase(msgspec.Struct, tag=True):  # type: ignore
    """Base class for all messages"""

    def describe(self) -> str:
        return ""


# NOTE: The additional layer of abstraction is to
# differentiate among
# (1) WorkerMsg: push-pull (lmcache->controller)
# (2) ControlMessage: req-reply (controller->lmcache)
# (3) OrchMsg: req-reply (ochestrator->controller)
"""Message from LMCache to Controller"""


class WorkerMsg(MsgBase):
    """Message between LMCache and Controller"""

    def describe(self) -> str:
        return ""


class RegisterMsg(WorkerMsg):
    """Message for Registration"""

    instance_id: str
    worker_id: int
    ip: str
    port: int

    def describe(self) -> str:
        return (
            f"Registering instance {self.instance_id}, "
            f"worker {self.worker_id} "
            f"at {self.ip}:{self.port}"
        )


class DeRegisterMsg(WorkerMsg):
    """Message for Deregistration"""

    instance_id: str
    worker_id: int
    ip: str
    port: int

    def describe(self) -> str:
        return (
            f"Deregistering instance {self.instance_id}, "
            f"worker {self.worker_id} "
            f"at {self.ip}:{self.port}"
        )


class KVAdmitMsg(WorkerMsg):
    """Message for KV chunk admission"""

    instance_id: str
    worker_id: int
    key: str
    location: str

    def describe(self) -> str:
        return f"kv_admit {self.key} to {self.instance_id}"


class KVEvictMsg(WorkerMsg):
    """Message for KV chunk eviction"""

    instance_id: str
    worker_id: int
    key: str
    location: str

    def describe(self) -> str:
        return f"kv_evict {self.key} from {self.instance_id}"


"""Control Message from Controller to LMCache"""


class ControlMsg(MsgBase):
    """Message from Controller to LMCache"""

    def describe(self) -> str:
        return ""


class ClearWorkerMsg(ControlMsg):
    """Clear message for a single lmcache worker"""

    locations: Optional[list[str]] = None
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return f"Clear tokens {self.tokens} in locations {self.locations}"


class PinWorkerMsg(ControlMsg):
    """Pin message for a single lmcache worker"""

    locations: Optional[list[str]] = None
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return f"Pin tokens {self.tokens} in locations {self.locations}"


class CompressWorkerMsg(ControlMsg):
    """Compress message for a single lmcache worker"""

    method: str
    locations: Optional[list[str]] = None
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return (
            f"Compress tokens {self.tokens} in "
            f"locations {self.locations} with "
            f"method {self.method}"
        )


class MoveWorkerMsg(ControlMsg):
    """Move message for a single lmcache worker"""

    old_position: Tuple[str, str]
    new_position: Tuple[str, str]
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return (
            f"Move tokens {self.tokens} from {self.old_position} to {self.new_position}"
        )


class HealthWorkerMsg(ControlMsg):
    """Health message for a single lmcache worker"""

    def describe(self) -> str:
        return "Health check"


class CheckFinishWorkerMsg(ControlMsg):
    """Check finish message for a single lmcache worker"""

    worker_event_id: str

    def describe(self) -> str:
        return f"Checking finish for worker event {self.worker_event_id}"


class ControlRetMsg(MsgBase):
    """Return message from LMCache to Controller"""

    def describe(self) -> str:
        return ""


class ClearWorkerRetMsg(ControlRetMsg):
    """Return message for a ClearWorkerMsg"""

    success: bool

    def describe(self) -> str:
        return f"Clear success: {self.success}"


class PinWorkerRetMsg(ControlRetMsg):
    """Pin return message for a single lmcache worker"""

    success: bool

    def describe(self) -> str:
        return f"Pin success: {self.success}"


class CompressWorkerRetMsg(ControlRetMsg):
    """Compress return message for a single lmcache worker"""

    worker_event_id: str

    def describe(self) -> str:
        return f"Compress worker event id: {self.worker_event_id}"


class MoveWorkerRetMsg(ControlRetMsg):
    """Move return message for a single lmcache worker"""

    worker_event_id: str

    def describe(self) -> str:
        return f"Move worker event id: {self.worker_event_id}"


class HealthWorkerRetMsg(ControlRetMsg):
    """Health return message for a single lmcache worker"""

    alive: bool

    def describe(self) -> str:
        return f"Health check alive: {self.alive}"


class CheckFinishWorkerRetMsg(ControlRetMsg):
    """Check finish return message for a single lmcache worker"""

    worker_event_id: str

    def describe(self) -> str:
        return f"Checking finish for worker event {self.worker_event_id}"


"""Orchestration Message from Ochestrator to LMCache"""


class OrchMsg(MsgBase):
    """Message from Ochestrator to Controller"""

    def describe(self) -> str:
        return ""


class QueryInstMsg(OrchMsg):
    """Query instance message"""

    ip: str

    def describe(self) -> str:
        return f"Query instance id of ip {self.ip}"


class LookupMsg(OrchMsg):
    """Lookup message"""

    tokens: list[int]

    def describe(self) -> str:
        return f"Lookup tokens {self.tokens}"


class ClearMsg(OrchMsg):
    """Clear message"""

    instance_id: str
    locations: Optional[list[str]] = None
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return (
            f"Clear tokens {self.tokens} in instance "
            f"{self.instance_id} and "
            f"locations {self.locations}"
        )


class PinMsg(OrchMsg):
    """Pin message"""

    instance_id: str
    locations: Optional[list[str]] = None
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return (
            f"Pin tokens {self.tokens} in instance "
            f"{self.instance_id} and "
            f"locations {self.locations}"
        )


class CompressMsg(OrchMsg):
    """Compress message"""

    instance_id: str
    method: str
    locations: Optional[list[str]] = None
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return (
            f"Compress tokens {self.tokens} in instance "
            f"{self.instance_id} and "
            f"locations {self.locations} with "
            f"method {self.method}"
        )


class MoveMsg(OrchMsg):
    """Move message"""

    old_position: Tuple[str, str]
    new_position: Tuple[str, str]
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return (
            f"Move tokens {self.tokens} from {self.old_position} to {self.new_position}"
        )


class HealthMsg(OrchMsg):
    """Health message"""

    instance_id: str

    def describe(self) -> str:
        return f"Health check for instance {self.instance_id}"


class CheckFinishMsg(OrchMsg):
    """Check finish message"""

    event_id: str

    def describe(self) -> str:
        return f"Checking finish for event {self.event_id}"


class OrchRetMsg(MsgBase):
    """Return message from Controller to Ochestrator"""

    def describe(self) -> str:
        return ""


class QueryInstRetMsg(OrchRetMsg):
    """Query instance return message"""

    instance_id: Optional[str]

    def describe(self) -> str:
        return f"The instance id is {self.instance_id}"


class LookupRetMsg(OrchRetMsg):
    """Lookup return message"""

    layout_info: Dict[str, Tuple[str, int]]

    def describe(self) -> str:
        return f"The layout info is {self.layout_info}"


class ClearRetMsg(OrchRetMsg):
    """Clear return message"""

    success: bool

    def describe(self) -> str:
        return f"Clear success: {self.success}"


class PinRetMsg(OrchRetMsg):
    """Pin return message"""

    success: bool

    def describe(self) -> str:
        return f"Pin success: {self.success}"


class CompressRetMsg(OrchRetMsg):
    """Compress return message"""

    event_id: str

    def describe(self) -> str:
        return f"Compress event id: {self.event_id}"


class MoveRetMsg(OrchRetMsg):
    """Move return message"""

    event_id: str

    def describe(self) -> str:
        return f"Move event id: {self.event_id}"


class HealthRetMsg(OrchRetMsg):
    """Health return message"""

    alive: bool

    def describe(self) -> str:
        return f"Alive: {self.alive}"


class CheckFinishRetMsg(OrchRetMsg):
    """Check finish return message"""

    finished: str

    def describe(self) -> str:
        return f"Event finished: {self.finished}"


class ErrorMsg(MsgBase):
    """Control Error Message"""

    error: str

    def describe(self) -> str:
        return f"Error: {self.error}"


Msg = Union[
    RegisterMsg,
    DeRegisterMsg,
    KVAdmitMsg,
    KVEvictMsg,
    ClearWorkerMsg,
    ClearWorkerRetMsg,
    PinWorkerMsg,
    PinWorkerRetMsg,
    CompressWorkerMsg,
    CompressWorkerRetMsg,
    MoveWorkerMsg,
    MoveWorkerRetMsg,
    HealthWorkerMsg,
    HealthWorkerRetMsg,
    CheckFinishWorkerMsg,
    CheckFinishWorkerRetMsg,
    LookupMsg,
    LookupRetMsg,
    ClearMsg,
    ClearRetMsg,
    PinMsg,
    PinRetMsg,
    CompressMsg,
    CompressRetMsg,
    MoveMsg,
    MoveRetMsg,
    HealthMsg,
    HealthRetMsg,
    CheckFinishMsg,
    CheckFinishRetMsg,
    ErrorMsg,
    QueryInstMsg,
    QueryInstRetMsg,
]
