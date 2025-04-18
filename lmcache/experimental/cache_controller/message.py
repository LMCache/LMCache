from typing import Literal, Optional, Union

import msgspec


class MsgBase(msgspec.Struct, frozen=True, tag=True):  # type: ignore
    """Base class for all messages"""

    def describe(self) -> str:
        return ""


# NOTE: The additional layer of abstraction is to
# differentiate among
# (1) WorkerMsg: push-pull (lmcache->controller)
# (2) ControlMessage: req-reply (controller->lmcache)
# (3) OrchMsg: req-reply (ochestrator->controller)
"""Message from LMCache to Controller"""


class WorkerMsg(MsgBase, frozen=True):
    """Message between LMCache and Controller"""

    def describe(self) -> str:
        return ""


class RegisterMsg(WorkerMsg, frozen=True):
    """Message for Registration"""
    # TODO(Jiayi): instance_id can be replaced with url
    instance_id: str
    worker_id: int
    url: str

    def describe(self) -> str:
        return f"Registering instance {self.instance_id}"


class DeRegisterMsg(WorkerMsg, frozen=True):
    """Message for Deregistration"""
    # TODO(Jiayi): instance_id can be replaced with url
    instance_id: str
    worker_id: int

    def describe(self) -> str:
        return f"Deregistering instance {self.instance_id}"


class KVAdmitMsg(WorkerMsg, frozen=True):
    """Message for KV chunk admission"""
    # TODO(Jiayi): instance_id can be replaced with url
    instance_id: str
    worker_id: int
    key: str
    location: str

    def describe(self) -> str:
        return f"kv_admit {self.key} to {self.instance_id}"


class KVEvictMsg(WorkerMsg, frozen=True):
    """Message for KV chunk eviction"""
    # TODO(Jiayi): instance_id can be replaced with url
    instance_id: str
    worker_id: int
    key: str
    location: str

    def describe(self) -> str:
        return f"kv_evict {self.key} from {self.instance_id}"


"""Control Message from Controller to LMCache"""


class ControlMsg(MsgBase, frozen=True):
    """Message from Controller to LMCache"""

    def describe(self) -> str:
        return ""


class ClearWorkerMsg(ControlMsg, frozen=True):
    """Clear message for a single lmcache worker"""
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return f"Clear tokens {self.tokens}"


class ControlRetMsg(MsgBase, frozen=True):
    """Return message from LMCache to Controller"""

    def describe(self) -> str:
        return ""


class ClearWorkerRetMsg(ControlRetMsg, frozen=True):
    """Return message for a ClearWorkerMsg"""
    success: bool

    def describe(self) -> str:
        return f"Clear success: {self.success}"


"""Orchestration Message from Ochestrator to LMCache"""


class OrchMsg(MsgBase, frozen=True):
    """Message from Ochestrator to Controller"""

    def describe(self) -> str:
        return ""


class LookupMsg(OrchMsg, frozen=True):
    """Lookup message"""
    tokens: list[int]

    def describe(self) -> str:
        return f"Lookup tokens {self.tokens}"


class ClearMsg(OrchMsg, frozen=True):
    """Clear message"""
    instance_id: str
    worker_ids: Optional[list[int]] = None
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return (f"Clear tokens {self.tokens} in instance "
                f"{self.instance_id} on workers {self.worker_ids}")


class OrchRetMsg(MsgBase, frozen=True):
    """Return message from  Controller to Ochestrator"""

    def describe(self) -> str:
        return ""


class LookupRetMsg(OrchRetMsg, frozen=True):
    """Lookup message"""
    best_instance_id: Optional[str]

    def describe(self) -> str:
        return f"The best instance is {self.best_instance_id}"


class ClearRetMsg(OrchRetMsg, frozen=True):
    """Clear message"""
    success: bool

    def describe(self) -> str:
        return f"Clear success: {self.success}"


class ErrorMsg(MsgBase, frozen=True):
    """Control Error Message"""
    error: str

    def describe(self) -> str:
        return f"Error: {self.error}"


Msg = Union[RegisterMsg, DeRegisterMsg, KVAdmitMsg, KVEvictMsg, ClearWorkerMsg,
            ClearWorkerRetMsg, LookupMsg, LookupRetMsg, ClearMsg, ClearRetMsg,
            ErrorMsg]
