# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import Future
from typing import List, Optional, Sequence
import time

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface

logger = init_logger(__name__)


class AuditBackend(StorageBackendInterface):
    """
    Audit wrapper for StorageBackend that logs operations and measures performance.
    """

    def __init__(self, real_backend: StorageBackendInterface):
        self.real_backend = real_backend
        self.logger = logger.getChild("audit")
        self.logger.info(
            f"[AUDIT_BACKEND] Initialized for backend: {str(real_backend)}"
        )

    def _log_operation(
        self,
        op_name: str,
        start_time: float,
        key: Optional[CacheEngineKey] = None,
        success: bool = True,
        result=None,
        error=None,
        size=None,
    ):
        """Helper method to log operation results."""
        cost = (time.perf_counter() - start_time) * 1000
        backend_name = str(self.real_backend)

        if error:
            self.logger.error(
                f"[AUDIT_BACKEND][{backend_name}]:{op_name}|FAILED|"
                f"Key:{key}|Error:{str(error)}"
            )
        elif success:
            log_msg = (
                f"[AUDIT_BACKEND][{backend_name}]:{op_name}|SUCCESS|Cost:{cost:.2f}ms"
            )
            if key:
                log_msg += f"|Key:{key}"
            if size is not None:
                log_msg += f"|Size:{size}"
            if result is not None:
                log_msg += f"|Result:{result}"
            self.logger.info(log_msg)

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check key existence with audit logging."""
        self.logger.debug(f"[AUDIT_BACKEND] Checking contains for key: {key}")
        start_time = time.perf_counter()
        try:
            result = self.real_backend.contains(key, pin)
            self._log_operation("CONTAINS", start_time, key, True, result)
            return result
        except Exception as e:
            self._log_operation("CONTAINS", start_time, key, False, error=e)
            raise

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Retrieve data with audit logging."""
        self.logger.debug(f"[AUDIT_BACKEND] Getting data for key: {key}")
        start_time = time.perf_counter()
        try:
            result = self.real_backend.get_blocking(key)
            size = len(result.byte_array) if result else 0
            self._log_operation(
                "GET", start_time, key, True, result=result is not None, size=size
            )
            return result
        except Exception as e:
            self._log_operation("GET", start_time, key, False, error=e)
            raise

    def close(self) -> None:
        """Close backend with audit logging."""
        self.logger.debug("[AUDIT_BACKEND] Closing backend")
        start_time = time.perf_counter()
        try:
            self.real_backend.close()
            self._log_operation("CLOSE", start_time, None, True)
        except Exception as e:
            self._log_operation("CLOSE", start_time, None, False, error=e)
            raise

    # Implement other required methods following the same pattern
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        start_time = time.perf_counter()
        try:
            result = self.real_backend.exists_in_put_tasks(key)
            self._log_operation("EXISTS_IN_PUT_TASKS", start_time, key, True, result)
            return result
        except Exception as e:
            self._log_operation("EXISTS_IN_PUT_TASKS", start_time, key, False, error=e)
            raise

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> bool:
        start_time = time.perf_counter()
        try:
            result = self.real_backend.submit_prefetch_task(key)
            self._log_operation("SUBMIT_PREFETCH_TASK", start_time, key, True, result)
            return result
        except Exception as e:
            self._log_operation("SUBMIT_PREFETCH_TASK", start_time, key, False, error=e)
            raise

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        sizes = [len(obj.byte_array) for obj in memory_objs]
        start_time = time.perf_counter()
        try:
            self.real_backend.batched_submit_put_task(keys, memory_objs, transfer_spec)
            self._log_operation(
                "BATCHED_SUBMIT_PUT_TASK", start_time, None, True, size=sum(sizes)
            )
        except Exception as e:
            self._log_operation(
                "BATCHED_SUBMIT_PUT_TASK", start_time, None, False, error=e
            )
            raise

    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        start_time = time.perf_counter()
        try:
            result = self.real_backend.get_non_blocking(key)
            self._log_operation("GET_NON_BLOCKING", start_time, key, True)
            return result
        except Exception as e:
            self._log_operation("GET_NON_BLOCKING", start_time, key, False, error=e)
            raise

    def pin(self, key: CacheEngineKey) -> bool:
        start_time = time.perf_counter()
        try:
            result = self.real_backend.pin(key)
            self._log_operation("PIN", start_time, key, True, result)
            return result
        except Exception as e:
            self._log_operation("PIN", start_time, key, False, error=e)
            raise

    def unpin(self, key: CacheEngineKey) -> bool:
        start_time = time.perf_counter()
        try:
            result = self.real_backend.unpin(key)
            self._log_operation("UNPIN", start_time, key, True, result)
            return result
        except Exception as e:
            self._log_operation("UNPIN", start_time, key, False, error=e)
            raise

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        start_time = time.perf_counter()
        try:
            result = self.real_backend.batched_get_blocking(keys)
            self._log_operation(
                "BATCHED_GET_BLOCKING",
                start_time,
                None,
                True,
                result=len(result) if result is not None else 0,
            )
            return result
        except Exception as e:
            self._log_operation(
                "BATCHED_GET_BLOCKING", start_time, None, False, error=e
            )
            raise

    def remove(self, key: CacheEngineKey, free_obj: bool = True) -> bool:
        start_time = time.perf_counter()
        try:
            result = self.real_backend.remove(key, free_obj)
            self._log_operation("REMOVE", start_time, key, True, result)
            return result
        except Exception as e:
            self._log_operation("REMOVE", start_time, key, False, error=e)
            raise

    def batched_remove(
        self,
        keys: list[CacheEngineKey],
        free_obj: bool = True,
    ) -> int:
        start_time = time.perf_counter()
        try:
            result = self.real_backend.batched_remove(keys, free_obj)
            self._log_operation("BATCHED_REMOVE", start_time, None, True, result)
            return result
        except Exception as e:
            self._log_operation("BATCHED_REMOVE", start_time, None, False, error=e)
            raise
