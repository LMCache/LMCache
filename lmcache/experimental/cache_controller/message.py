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

from typing import Optional, Union

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
    # TODO(Jiayi): instance_id can be replaced with url
    instance_id: str
    worker_id: int
    url: str

    def describe(self) -> str:
        return f"Registering instance {self.instance_id}"


class DeRegisterMsg(WorkerMsg):
    """Message for Deregistration"""
    # TODO(Jiayi): instance_id can be replaced with url
    instance_id: str
    worker_id: int

    def describe(self) -> str:
        return f"Deregistering instance {self.instance_id}"


class KVAdmitMsg(WorkerMsg):
    """Message for KV chunk admission"""
    # TODO(Jiayi): instance_id can be replaced with url
    instance_id: str
    worker_id: int
    key: str
    location: str

    def describe(self) -> str:
        return f"kv_admit {self.key} to {self.instance_id}"


class KVEvictMsg(WorkerMsg):
    """Message for KV chunk eviction"""
    # TODO(Jiayi): instance_id can be replaced with url
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
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return f"Clear tokens {self.tokens}"


class ControlRetMsg(MsgBase):
    """Return message from LMCache to Controller"""

    def describe(self) -> str:
        return ""


class ClearWorkerRetMsg(ControlRetMsg):
    """Return message for a ClearWorkerMsg"""
    success: bool

    def describe(self) -> str:
        return f"Clear success: {self.success}"


"""Orchestration Message from Ochestrator to LMCache"""


class OrchMsg(MsgBase):
    """Message from Ochestrator to Controller"""

    def describe(self) -> str:
        return ""


class LookupMsg(OrchMsg):
    """Lookup message"""
    tokens: list[int]

    def describe(self) -> str:
        return f"Lookup tokens {self.tokens}"


class ClearMsg(OrchMsg):
    """Clear message"""
    instance_id: str
    worker_ids: Optional[list[int]] = None
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return (f"Clear tokens {self.tokens} in instance "
                f"{self.instance_id} on workers {self.worker_ids}")


class OrchRetMsg(MsgBase):
    """Return message from  Controller to Ochestrator"""

    def describe(self) -> str:
        return ""


class LookupRetMsg(OrchRetMsg):
    """Lookup message"""
    best_instance_id: Optional[str]

    def describe(self) -> str:
        return f"The best instance is {self.best_instance_id}"


class ClearRetMsg(OrchRetMsg):
    """Clear message"""
    success: bool

    def describe(self) -> str:
        return f"Clear success: {self.success}"


class ErrorMsg(MsgBase):
    """Control Error Message"""
    error: str

    def describe(self) -> str:
        return f"Error: {self.error}"


Msg = Union[RegisterMsg, DeRegisterMsg, KVAdmitMsg, KVEvictMsg, ClearWorkerMsg,
            ClearWorkerRetMsg, LookupMsg, LookupRetMsg, ClearMsg, ClearRetMsg,
            ErrorMsg]
