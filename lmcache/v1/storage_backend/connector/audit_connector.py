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
from threading import Lock
from typing import Dict, List, Optional
import hashlib
import time

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
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

    def __init__(self, real_connector: RemoteConnector, verify_checksum: bool = False):
        self.real_connector = real_connector

        self.verify_checksum = verify_checksum
        self.checksum_registry: Dict[CacheEngineKey, str] = {}
        self.registry_lock = Lock() if verify_checksum else None
        self.logger = logger.getChild("audit")
        logger.info(f"[REMOTE_AUDIT]INITIALIZED|Verify Checksum: {verify_checksum}")

    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum for data validation"""
        return hashlib.sha256(data).hexdigest()

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        """Store data with optional checksum tracking"""
        data = memory_obj.byte_array
        checksum = self._calculate_checksum(data)
        data_size = len(data)
        self.logger.debug(
            f"[REMOTE_AUDIT]:PUT|START|Size:{data_size}|"
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
                f"[REMOTE_AUDIT]PUT|SUCCESS|Size:{data_size}|"
                f"Checksum:{checksum[:8]}|Cost:{cost:.6f}ms|Saved:"
                f"{len(self.checksum_registry)}|Key:{key}"
            )

        except Exception as e:
            self.logger.error(
                f"[REMOTE_AUDIT]PUT|FAILED|Size:{data_size}|Key:{key}|Error: {str(e)}"
            )
            raise

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Retrieve data with optional integrity check"""
        self.logger.debug(
            f"[REMOTE_AUDIT]GET|START|Saved:{len(self.checksum_registry)}|Key:{key}"
        )

        try:
            t1 = time.perf_counter()
            result = await self.real_connector.get(key)
            t2 = time.perf_counter()
            if result is None:
                self.logger.info(
                    f"[REMOTE_AUDIT]GET|MISS|Key:{key}|"
                    f"Saved: {len(self.checksum_registry)}"
                )
                return None

            current_data = result.byte_array
            current_checksum = self._calculate_checksum(current_data)
            data_size = len(current_data)

            if self.registry_lock:
                with self.registry_lock:
                    expected_checksum = self.checksum_registry.get(key)

                if expected_checksum and current_checksum != expected_checksum:
                    self.logger.error(
                        f"[REMOTE_AUDIT]GET|MISMATCH|Size:{data_size}|"
                        f"Expected:<{expected_checksum[:8]}>|"
                        f"Actual:<{current_checksum[:8]}>|Key:{key}"
                    )
                    return None

            cost = (t2 - t1) * 1000
            self.logger.info(
                f"[REMOTE_AUDIT]GET|SUCCESS|"
                f"Checksum:{current_checksum[:8]}|"
                f"Cost:{cost:.6f}ms|Saved:{len(self.checksum_registry)}|Key:{key}"
            )
            return result

        except Exception as e:
            self.logger.error(f"[REMOTE_AUDIT]GET|FAILED|Key:{key}|Error: {str(e)}")
            raise

    async def exists(self, key: CacheEngineKey) -> bool:
        """Check key existence with audit log"""
        self.logger.debug(f"[REMOTE_AUDIT]EXISTS|START|Key:{key}")
        result = await self.real_connector.exists(key)
        self.logger.info(f"[REMOTE_AUDIT]EXISTS|{result}|Key: {key}")
        return result

    async def list(self) -> List[str]:
        """List keys with audit log"""
        self.logger.debug("[REMOTE_AUDIT]LIST|START")
        result = await self.real_connector.list()
        self.logger.info(f"[REMOTE_AUDIT]LIST|SUCCESS|Size:{len(result)}")
        return result

    async def close(self):
        """Cleanup resources with audit log"""
        self.logger.debug("[REMOTE_AUDIT]CLOSE|START")
        await self.real_connector.close()
        self.logger.info("[REMOTE_AUDIT]CLOSE|SUCCESS")
