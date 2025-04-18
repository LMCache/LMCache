from abc import ABC, abstractmethod
from typing import Literal, Optional

import msgspec


class MsgBase(msgspec.Struct, ABC, frozen=True,
              discriminator="type"):  # type: ignore
    """Abstract base class for all messages"""

    @abstractmethod
    def describe(self) -> str:
        pass


# NOTE: The additional layer of abstraction is to
# differentiate among
# (1) WorkerMsg: push-pull (lmcache->controller)
# (2) ControlMessage: req-reply (controller->lmcache)
# (3) OrchMsg: req-reply (ochestrator->controller)
"""Message from LMCache to Controller"""


class WorkerMsg(MsgBase, frozen=True):
    """Message between LMCache and Controller"""

    @abstractmethod
    def describe(self) -> str:
        pass


class RegisterMsg(WorkerMsg, frozen=True):
    """Message for Registration"""
    # TODO(Jiayi): instance_id can be replaced with url
    instance_id: str
    worker_id: int
    url: str
    type: Literal["register"] = "register"

    def describe(self) -> str:
        return f"Registering instance {self.instance_id}"


class DeRegisterMsg(WorkerMsg, frozen=True):
    """Message for Deregistration"""
    # TODO(Jiayi): instance_id can be replaced with url
    instance_id: str
    worker_id: int
    type: Literal["de_register"] = "de_register"

    def describe(self) -> str:
        return f"Deregistering instance {self.instance_id}"


class KVAdmitMsg(WorkerMsg, frozen=True):
    """Message for KV chunk admission"""
    # TODO(Jiayi): instance_id can be replaced with url
    instance_id: str
    worker_id: int
    key: str
    location: str
    type: Literal["kv_admit"] = "kv_admit"

    def describe(self) -> str:
        return f"kv_admit {self.key} to {self.instance_id}"


class KVEvictMsg(WorkerMsg, frozen=True):
    """Message for KV chunk eviction"""
    # TODO(Jiayi): instance_id can be replaced with url
    instance_id: str
    worker_id: int
    key: str
    location: str
    type: Literal["kv_evict"] = "kv_evict"

    def describe(self) -> str:
        return f"kv_evict {self.key} from {self.instance_id}"


"""Control Message from Controller to LMCache"""


class ControlMsg(MsgBase, frozen=True):
    """Message from Controller to LMCache"""

    @abstractmethod
    def describe(self) -> str:
        pass


class ClearWorkerMsg(ControlMsg, frozen=True):
    """Clear message for a single lmcache worker"""
    tokens: Optional[list[int]] = None
    type: Literal["clear_worker"] = "clear_worker"

    def describe(self) -> str:
        return f"Clear tokens {self.tokens}"


class ControlRetMsg(MsgBase, frozen=True):
    """Return message from LMCache to Controller"""

    @abstractmethod
    def describe(self) -> str:
        pass


class ClearWorkerRetMsg(ControlRetMsg, frozen=True):
    """Return message for a ClearWorkerMsg"""
    success: bool
    type: Literal["clear_worker_ret"] = "clear_worker_ret"

    def describe(self) -> str:
        return f"Clear success: {self.success}"


"""Orchestration Message from Ochestrator to LMCache"""


class OrchMsg(MsgBase, frozen=True):
    """Message from Ochestrator to Controller"""

    @abstractmethod
    def describe(self) -> str:
        pass


class LookupMsg(OrchMsg, frozen=True):
    """Lookup message"""
    tokens: list[int]
    type: Literal["lookup"] = "lookup"

    def describe(self) -> str:
        return f"Lookup tokens {self.tokens}"


class ClearMsg(OrchMsg, frozen=True):
    """Clear message"""
    instance_id: str
    worker_ids: Optional[list[int]] = None
    tokens: Optional[list[int]] = None
    type: Literal["clear"] = "clear"

    def describe(self) -> str:
        return (f"Clear tokens {self.tokens} in instance "
                f"{self.instance_id} on workers {self.worker_ids}")


class OrchRetMsg(MsgBase, frozen=True):
    """Return message from  Controller to Ochestrator"""

    @abstractmethod
    def describe(self) -> str:
        pass


class LookupRetMsg(OrchRetMsg, frozen=True):
    """Lookup message"""
    best_instance_id: Optional[str]
    type: Literal["lookup_ret"] = "lookup_ret"

    def describe(self) -> str:
        return f"The best instance is {self.best_instance_id}"


class ClearRetMsg(OrchRetMsg, frozen=True):
    """Clear message"""
    success: bool
    type: Literal["clear_ret"] = "clear_ret"

    def describe(self) -> str:
        return f"Clear success: {self.success}"


class ErrorMsg(MsgBase, frozen=True):
    """Control Error Message"""
    error: str
    type: Literal["error"] = "error"

    def describe(self) -> str:
        return f"Error: {self.error}"
