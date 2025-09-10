# SPDX-License-Identifier: Apache-2.0
# Standard
from threading import Lock
from typing import Dict, List, Optional
import asyncio
import hashlib
import time

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class AuditConnector(RemoteConnector):
    """Audit wrapper for RemoteConnector that verifies data integrity
    and logs operations.

    Features:
    - Wraps any RemoteConnector implementation
    - Configurable checksum verification via URL parameter
    - Logs all operations with timestamps
    - Optional checksum validation for put/get operations
    """

    def __init__(
        self, real_connector: RemoteConnector, lmcache_config: LMCacheEngineConfig
    ):
        self.real_connector = real_connector
        self.verify_checksum = (
            lmcache_config.extra_config is not None
            and "audit_verify_checksum" in lmcache_config.extra_config
            and lmcache_config.extra_config["audit_verify_checksum"]
        )
        self.calc_checksum = (
            lmcache_config.extra_config is not None
            and "audit_calc_checksum" in lmcache_config.extra_config
            and lmcache_config.extra_config["audit_calc_checksum"]
        )
        self.checksum_registry: Dict[CacheEngineKey, str] = {}
        self.registry_lock = Lock() if self.verify_checksum else None

        # Parse audit exclude commands
        self.excluded_cmds = set()
        if (
            lmcache_config.extra_config
            and "audit_exclude_cmds" in lmcache_config.extra_config
        ):
            exclude_cmds = lmcache_config.extra_config["audit_exclude_cmds"]
            if exclude_cmds:
                self.excluded_cmds = {cmd.strip() for cmd in exclude_cmds.split(",")}

        self.logger = logger.getChild("audit")

        # Dynamically replace excluded methods
        self._replace_excluded_methods()

        logger.info(
            f"[REMOTE_AUDIT][{self.real_connector}]:INITIALIZED|"
            f"Calc Checksum:{self.calc_checksum}｜"
            f"Verify Checksum: {self.verify_checksum}|"
            f"Excluded Cmds: {self.excluded_cmds}"
        )

    def _replace_excluded_methods(self):
        """Dynamically replace methods that should be excluded from auditing"""
        for method_name in self.excluded_cmds:
            if hasattr(self.real_connector, method_name):
                # Create a direct pass-through method
                real_method = getattr(self.real_connector, method_name)

                if asyncio.iscoroutinefunction(real_method):

                    def create_async_wrapper(rm):
                        async def async_wrapper(*args, **kwargs):
                            return await rm(*args, **kwargs)

                        return async_wrapper

                    setattr(self, method_name, create_async_wrapper(real_method))
                else:

                    def create_sync_wrapper(rm):
                        def sync_wrapper(*args, **kwargs):
                            return rm(*args, **kwargs)

                        return sync_wrapper

                    setattr(self, method_name, create_sync_wrapper(real_method))

    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum for data validation"""
        return hashlib.sha256(data).hexdigest()

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Store data with optional checksum tracking"""
        data = memory_obj.byte_array
        checksum = self._calculate_checksum(data) if self.calc_checksum else "N/A"
        data_size = len(data)
        self.logger.debug(
            f"[REMOTE_AUDIT][{self.real_connector}]:PUT|START|Size:{data_size}|"
            f"Checksum:{checksum[:8]}|Saved:{len(self.checksum_registry)}|Key:{key}"
        )

        try:
            t1 = time.perf_counter()
            await self.real_connector.put(key, memory_obj)
            t2 = time.perf_counter()
            cost = (t2 - t1) * 1000
            if self.registry_lock:
                with self.registry_lock:
                    self.checksum_registry[key] = checksum
            self.logger.info(
                f"[REMOTE_AUDIT][{self.real_connector}]:PUT|SUCCESS|Size:{data_size}|"
                f"Checksum:{checksum[:8]}|Cost:{cost:.6f}ms|Saved:"
                f"{len(self.checksum_registry)}|Key:{key}"
            )

        except Exception as e:
            self.logger.error(
                f"[REMOTE_AUDIT][{self.real_connector}]:PUT|FAILED|Size:{data_size}|"
                f"Key:{key}|Error: {str(e)}"
            )
            raise

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Retrieve data with optional integrity check"""
        self.logger.debug(
            f"[REMOTE_AUDIT][{self.real_connector}]:GET|START|"
            f"Saved:{len(self.checksum_registry)}|Key:{key}"
        )

        try:
            t1 = time.perf_counter()
            result = await self.real_connector.get(key)
            t2 = time.perf_counter()
            if result is None:
                self.logger.info(
                    f"[REMOTE_AUDIT][{self.real_connector}]:GET|MISS|Key:{key}|"
                    f"Saved: {len(self.checksum_registry)}"
                )
                return None

            current_data = result.byte_array
            current_checksum = (
                self._calculate_checksum(current_data) if self.calc_checksum else "N/A"
            )
            data_size = len(current_data)

            if self.registry_lock:
                with self.registry_lock:
                    expected_checksum = self.checksum_registry.get(key)

                if expected_checksum and current_checksum != expected_checksum:
                    self.logger.error(
                        f"[REMOTE_AUDIT][{self.real_connector}]:"
                        f"GET|MISMATCH|Size:{data_size}|"
                        f"Expected:<{expected_checksum[:8]}>|"
                        f"Actual:<{current_checksum[:8]}>|Key:{key}"
                    )
                    return None

            cost = (t2 - t1) * 1000
            self.logger.info(
                f"[REMOTE_AUDIT][{self.real_connector}]:GET|SUCCESS|"
                f"Checksum:{current_checksum[:8]}|"
                f"Cost:{cost:.6f}ms|Saved:{len(self.checksum_registry)}|Key:{key}"
            )
            return result

        except Exception as e:
            self.logger.error(
                f"[REMOTE_AUDIT][{self.real_connector}]:GET|"
                f"FAILED|Key:{key}|Error: {str(e)}"
            )
            raise

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check key existence with audit log"""
        self.logger.debug(
            f"[REMOTE_AUDIT][{self.real_connector}]:EXISTS|START|Key:{key}"
        )
        t1 = time.perf_counter()
        result = await self.real_connector.exists(key)
        t2 = time.perf_counter()
        cost = (t2 - t1) * 1000
        self.logger.info(
            f"[REMOTE_AUDIT][{self.real_connector}]:EXISTS|{result}|"
            f"Cost:{cost:.6f}ms|"
            f"Key:{key}"
        )
        return result

    def exists_sync(self, key: CacheEngineKey) -> bool:
        """Check key existence with audit log synchronized"""
        self.logger.debug(f"[REMOTE_AUDIT]EXISTS_SYNC|START|Key:{key}")
        result = self.real_connector.exists_sync(key)
        self.logger.info(f"[REMOTE_AUDIT]EXISTS_SYNC|{result}|Key: {key}")
        return result

    async def list(self) -> List[str]:
        """List keys with audit log"""
        self.logger.debug("[REMOTE_AUDIT][{self.real_connector}]:LIST|START")
        t1 = time.perf_counter()
        result = await self.real_connector.list()
        t2 = time.perf_counter()
        cost = (t2 - t1) * 1000
        self.logger.info(
            f"[REMOTE_AUDIT][{self.real_connector}]:LIST|SUCCESS|"
            f"Count:{len(result)}|Cost:{cost:.6f}ms"
        )
        return result

    async def close(self):
        """Cleanup resources with audit log"""
        self.logger.debug(f"[REMOTE_AUDIT][{self.real_connector}]:CLOSE|START")
        await self.real_connector.close()
        self.logger.info(f"[REMOTE_AUDIT][{self.real_connector}]:CLOSE|SUCCESS")

    def support_ping(self) -> bool:
        self.logger.debug(f"[REMOTE_AUDIT][{self.real_connector}]:SUPPORT_PING|START")
        support = self.real_connector.support_ping()
        self.logger.info(
            f"[REMOTE_AUDIT][{self.real_connector}]:SUPPORT_PING|{support}"
        )
        return support

    async def ping(self) -> int:
        self.logger.debug(f"[REMOTE_AUDIT][{self.real_connector}]:PING|START")
        t1 = time.perf_counter()
        error_code = await self.real_connector.ping()
        t2 = time.perf_counter()
        cost = (t2 - t1) * 1000
        self.logger.debug(
            f"[REMOTE_AUDIT][{self.real_connector}]:PING|{error_code}｜"
            f"Cost:{cost:.6f}ms"
        )
        return error_code
