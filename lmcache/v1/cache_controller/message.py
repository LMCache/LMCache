# SPDX-License-Identifier: Apache-2.0
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
    distributed_url: str  # URL for actual KV cache transfer

    def describe(self) -> str:
        return (
            f"Registering instance {self.instance_id}, "
            f"worker {self.worker_id} "
            f"at {self.ip}:{self.port}"
            f" with distributed URL {self.distributed_url}"
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
    key: int
    location: str

    def describe(self) -> str:
        return f"kv_admit {self.key} to {self.instance_id}"


class KVEvictMsg(WorkerMsg):
    """Message for KV chunk eviction"""

    instance_id: str
    worker_id: int
    key: int
    location: str

    def describe(self) -> str:
        return f"kv_evict {self.key} from {self.instance_id}"


class HeartbeatMsg(RegisterMsg):
    """Message for heartbeat, include register info for re-register"""

    # TODO: add more heartbeat info

    def describe(self) -> str:
        return f"Heartbeat from instance {self.instance_id}, worker {self.worker_id}"


"""Control Message from Controller to LMCache"""


class ControlMsg(MsgBase):
    """Message from Controller to LMCache"""

    def describe(self) -> str:
        return ""


class ClearWorkerMsg(ControlMsg):
    """Clear message for a single lmcache worker"""

    worker_event_id: str
    location: str

    def describe(self) -> str:
        return f"Clear tokens in location {self.location}"


class PinWorkerMsg(ControlMsg):
    """Pin message for a single lmcache worker"""

    worker_event_id: str
    location: str
    tokens: list[int]

    def describe(self) -> str:
        return f"Pin tokens {self.tokens} in location {self.location}"


class CompressWorkerMsg(ControlMsg):
    """Compress message for a single lmcache worker"""

    worker_event_id: str
    method: str
    location: str
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return (
            f"Compress tokens {self.tokens} in "
            f"locations {self.location} with "
            f"method {self.method}"
        )


class DecompressWorkerMsg(ControlMsg):
    """Decompress message for a single lmcache worker"""

    worker_event_id: str
    method: str
    location: str
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return (
            f"Decompress tokens {self.tokens} in "
            f"locations {self.location} with "
            f"method {self.method}"
        )


class MoveWorkerMsg(ControlMsg):
    """Move message for a single lmcache worker"""

    worker_event_id: str
    old_position: str  # location (storage backend name)
    new_position: Tuple[str, str]  # (target_url, location (storage backend name) )
    tokens: Optional[list[int]] = None
    copy: Optional[bool] = True

    def describe(self) -> str:
        return (
            f"Move tokens {self.tokens} from {self.old_position} to {self.new_position}"
        )


class HealthWorkerMsg(ControlMsg):
    """Health message for a single lmcache worker"""

    worker_event_id: str

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

    num_tokens: int

    def describe(self) -> str:
        return f"Number of cleared tokens: {self.num_tokens}"


class PinWorkerRetMsg(ControlRetMsg):
    """Pin return message for a single lmcache worker"""

    num_tokens: int

    def describe(self) -> str:
        return f"Number of pinned tokens: {self.num_tokens}"


class CompressWorkerRetMsg(ControlRetMsg):
    """Compress return message for a single lmcache worker"""

    num_tokens: int

    def describe(self) -> str:
        return f"Compress success: {self.num_tokens}"


class DecompressWorkerRetMsg(ControlRetMsg):
    """Decompress return message for a single lmcache worker"""

    num_tokens: int

    def describe(self) -> str:
        return f"Decompress success: {self.num_tokens}"


class MoveWorkerRetMsg(ControlRetMsg):
    """Move return message for a single lmcache worker"""

    num_tokens: int

    def describe(self) -> str:
        return f"Moving {self.num_tokens} tokens"


class HealthWorkerRetMsg(ControlRetMsg):
    """Health return message for a single lmcache worker"""

    error_code: int

    def describe(self) -> str:
        return f"Health check error code: {self.error_code}"


class CheckFinishWorkerRetMsg(ControlRetMsg):
    """Check finish return message for a single lmcache worker"""

    status: str

    def describe(self) -> str:
        return f"Check finish status: {self.status}"


"""Orchestration Message from Ochestrator to LMCache"""


class OrchMsg(MsgBase):
    """Message from Ochestrator to Controller"""

    def describe(self) -> str:
        return ""


class QueryInstMsg(OrchMsg):
    """Query instance message"""

    event_id: str
    ip: str

    def describe(self) -> str:
        return f"Query instance id of ip {self.ip}"


class LookupMsg(OrchMsg):
    """Lookup message"""

    event_id: str
    tokens: list[int]

    def describe(self) -> str:
        return f"Lookup tokens {self.tokens}"


class ClearMsg(OrchMsg):
    """Clear message"""

    event_id: str
    instance_id: str
    location: str

    def describe(self) -> str:
        return (
            f"Clear tokens in instance {self.instance_id} and locations {self.location}"
        )


class PinMsg(OrchMsg):
    """Pin message"""

    event_id: str
    instance_id: str
    location: str
    tokens: list[int]

    def describe(self) -> str:
        return (
            f"Pin tokens {self.tokens} in instance "
            f"{self.instance_id} and "
            f"location {self.location}"
        )


class CompressMsg(OrchMsg):
    """Compress message"""

    event_id: str
    instance_id: str
    method: str
    location: str
    tokens: Optional[list[int]] = None  # `None` means compress all tokens

    def describe(self) -> str:
        return (
            f"Compress tokens {self.tokens} in instance "
            f"{self.instance_id} and "
            f"locations {self.location} with "
            f"method {self.method}"
        )


class DecompressMsg(OrchMsg):
    """Decompress message"""

    event_id: str
    instance_id: str
    method: str
    location: str
    tokens: Optional[list[int]] = None  # `None` means compress all tokens

    def describe(self) -> str:
        return (
            f"Decompress tokens {self.tokens} in instance "
            f"{self.instance_id} and "
            f"locations {self.location} with "
            f"method {self.method}"
        )


class MoveMsg(OrchMsg):
    """Move message"""

    event_id: str
    old_position: Tuple[str, str]
    new_position: Tuple[str, str]
    tokens: Optional[list[int]] = None
    copy: Optional[bool] = False

    def describe(self) -> str:
        return (
            f"Move tokens {self.tokens} from {self.old_position} to {self.new_position}"
        )


class HealthMsg(OrchMsg):
    """Health message"""

    event_id: str
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

    event_id: str
    instance_id: Optional[str]

    def describe(self) -> str:
        return f"The instance id is {self.instance_id}"


class LookupRetMsg(OrchRetMsg):
    """Lookup return message"""

    event_id: str
    layout_info: Dict[str, Tuple[str, int]]

    def describe(self) -> str:
        return f"The layout info is {self.layout_info}"


class ClearRetMsg(OrchRetMsg):
    """Clear return message"""

    event_id: str
    num_tokens: int

    def describe(self) -> str:
        return f"Number of cleared tokens: {self.num_tokens}"


class PinRetMsg(OrchRetMsg):
    """Pin return message"""

    event_id: str
    num_tokens: int

    def describe(self) -> str:
        return f"Number of pinned tokens: {self.num_tokens}"


class CompressRetMsg(OrchRetMsg):
    """Compress return message"""

    event_id: str
    num_tokens: int

    def describe(self) -> str:
        return f"Compressed {self.num_tokens} tokens"


class DecompressRetMsg(OrchRetMsg):
    """Decompress return message"""

    event_id: str
    num_tokens: int

    def describe(self) -> str:
        return f"Decompressed {self.num_tokens} tokens"


class MoveRetMsg(OrchRetMsg):
    """Move return message"""

    event_id: str
    num_tokens: int

    def describe(self) -> str:
        return f"Moving {self.num_tokens} tokens"


class HealthRetMsg(OrchRetMsg):
    """Health return message"""

    event_id: str
    # worker_id -> error_code
    error_codes: Dict[int, int]

    def describe(self) -> str:
        return f"error_codes: {self.error_codes}"


class CheckFinishRetMsg(OrchRetMsg):
    """Check finish return message"""

    status: str

    def describe(self) -> str:
        return f"Event status: {self.status}"


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
    DecompressWorkerMsg,
    DecompressWorkerRetMsg,
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
    DecompressMsg,
    DecompressRetMsg,
    MoveMsg,
    MoveRetMsg,
    HealthMsg,
    HealthRetMsg,
    CheckFinishMsg,
    CheckFinishRetMsg,
    ErrorMsg,
    QueryInstMsg,
    QueryInstRetMsg,
    HeartbeatMsg,
]
