# SPDX-License-Identifier: Apache-2.0

"""
Manages the states of the objects in L1 memory
"""

# Standard
from dataclasses import dataclass
from typing import Iterable
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import TTLLock
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.distributed.api import ObjectKey
from lmcache.v1.multiprocess.distributed.config import L1ObjectManagerConfig
from lmcache.v1.multiprocess.distributed.error import L1ObjectManagerError
from lmcache.v1.multiprocess.distributed.internal_api import L1OperationResult

logger = init_logger(__name__)


# HELPER CLASSES
@dataclass
class L1ObjectEntry:
    """An entry representing an object in L1 memory"""

    memory_obj: MemoryObj | None
    """ The memory object associated with the entry """

    ttl_lock: TTLLock
    """ The TTL lock associated with the entry """

    dirty: bool
    """ Whether the object is reserved for write (dirty) """

    is_temporary: bool
    """ Whether the object is temporary (to be freed after use) """

    @staticmethod
    def new_empty() -> "L1ObjectEntry":
        """Create a new empty L1ObjectEntry instance.

        Returns:
            L1ObjectEntry instance with no associated object.
        """
        global _NON_EXISTENT_OBJECT_ENTRY
        return _NON_EXISTENT_OBJECT_ENTRY

    # Helper functions
    def mark_as_committed(self) -> None:
        """Mark the entry as committed (not dirty)"""
        self.dirty = False

    def mark_as_reserved(self) -> None:
        """Mark the entry as reserved (dirty)"""
        self.dirty = True


# PUBLIC CLASSES
@dataclass(frozen=True)
class L1ObjectState:
    """The immutable state snapshot of the object in L1 memory

    At a high-level, the object allocated in L1 memory can have the following
    states:

    1. Reserved: the object is reserved for write operations (i.e., dirty). The
       reserved object should not be read/accessed by any other operations.

    2. Committed: the object is ready to be read/accessed by other operations.
       There are a few sub-states for the committed object:
       2.1 Locked: the object is being locked from eviction.
       2.2 Temporary: this object is a temporary object that should be freed
           after use. This flag will be used for prefetched objects.
    """

    _memory_obj: MemoryObj | None
    _is_existent: bool
    _is_reserved: bool
    _is_committed: bool
    _is_locked: bool
    _is_temporary: bool

    def __post_init__(self):
        assert not (self._is_reserved and self._is_committed), (
            "An object cannot be both reserved and committed."
        )

    # APIs to query the state
    def exists(self) -> bool:
        return self._is_existent

    def is_reserved(self) -> bool:
        return self._is_reserved

    def is_committed(self) -> bool:
        return self._is_committed

    def is_locked(self) -> bool:
        return self._is_locked

    def is_temporary(self) -> bool:
        return self._is_temporary

    @property
    def memory_obj(self) -> MemoryObj | None:
        """
        Returns:
            None if the object does not exist in L1 memory,
            otherwise returns the MemoryObj instance.
        """
        return self._memory_obj

    # APIs to create the object state
    # Should only be used by L1ObjectManager internally
    @staticmethod
    def new_empty() -> "L1ObjectState":
        """
        Create a new L1ObjectState instance representing a non-existing object.

        Returns:
            L1ObjectState instance with no associated object.
        """
        global _NON_EXISTENT_OBJECT_STATE
        return _NON_EXISTENT_OBJECT_STATE

    @staticmethod
    def from_entry(entry: L1ObjectEntry) -> "L1ObjectState":
        """
        Create a new L1ObjectState instance from the given L1ObjectEntry.

        Args:
            entry: The L1ObjectEntry instance to create the state from.

        Returns:
            L1ObjectState instance representing the state of the object.
        """
        return L1ObjectState(
            _memory_obj=entry.memory_obj,
            _is_existent=True,
            _is_reserved=entry.dirty,
            _is_committed=not entry.dirty,
            _is_locked=entry.ttl_lock.is_locked(),
            _is_temporary=entry.is_temporary,
        )


_NON_EXISTENT_OBJECT_ENTRY = L1ObjectEntry(
    memory_obj=None, ttl_lock=TTLLock(), dirty=True, is_temporary=False
)

_NON_EXISTENT_OBJECT_STATE = L1ObjectState(
    _memory_obj=None,
    _is_existent=False,
    _is_reserved=False,
    _is_committed=False,
    _is_locked=False,
    _is_temporary=False,
)


class L1ObjectManager:
    """
    This class manages the keys and the states of the objects in L1 memory.

    Observability metrics to emit:
    - number of uncommitted keys and their size usage
    - number of committed keys and their size usage
    - number of locked keys
    - number of temporary keys

    Error handling semantics: 'ALL OR NOTHING' or "FORCED"
    - 'ALL OR NOTHING': If the error happens during the function call, the
      function will make sure to revert itself to the state as if it has
      never been called.
    - 'FORCED': If the error happens during the function call, the function
      will try its best to process the "good" ones and skip the "bad" ones.
    The function name will indicate whether it's forced or not.
    - xxx_forced: forced
    - xxx_must: all or nothing
    - xxx (with force parameter): can be forced or not based on the parameter.

    Note that only some of the functions support the 'FORCED' semantics.
    """

    def __init__(self, config: L1ObjectManagerConfig) -> None:
        self._reserved: dict[ObjectKey, L1ObjectEntry] = {}
        self._committed: dict[ObjectKey, L1ObjectEntry] = {}

        # Locks for reserved and committed dicts
        # NOTE: we will always acquire _reserved_lock first, then _committed_lock
        # TODO: we have global locks for now. In the future, we can use finer-grained
        # locks (e.g., per-bucket locks) to improve concurrency.
        self._reserved_lock = threading.Lock()
        self._committed_lock = threading.Lock()

        # TTL:
        self._lock_ttl = config.lock_ttl_seconds

    def _has_key(self, key: ObjectKey) -> int:
        """Thread-safe function to check if the key exists in either reserved
        or committed dicts.

        Args:
            key: The key to check.
        Returns:
            0 if the key does not exist,
            1 if the key exists in reserved dict,
            2 if the key exists in committed dict.

        Note:
            This function will acquire both reserved and committed locks.
        """
        with self._reserved_lock, self._committed_lock:
            return key in self._reserved or key in self._committed

    def _get_entry(self, key: ObjectKey) -> L1ObjectEntry | None:
        """Thread-safe function to get the L1ObjectEntry for the given key.

        Args:
            key: The key to get the entry for.

        Returns:
            The L1ObjectEntry if the key exists, None otherwise.
        """
        with self._reserved_lock, self._committed_lock:
            if key in self._reserved:
                return self._reserved[key]
            elif key in self._committed:
                return self._committed[key]
            else:
                return None

    def prereserve_forced(
        self,
        keys: Iterable[ObjectKey],
    ) -> L1OperationResult:
        """Thread-safe function to pre-reserve a set of keys without having
        associated memory objects.

        When multiple threads trying to reserve on the same set of keys key,
        the expected behavior is each of the thread will get some of the keys.

        This function will skip the keys that are already committed or reserved.
        (i.e., default is 'FORCED' semantics)

        Args:
            keys: The keys to reserve. Cannot be already committed in the manager.

        Returns:
            L1OperationResult with:
            - error: SUCCESS if all keys are reserved successfully,
                     KEYS_ALREADY_EXIST if some of the keys already exist.
            - success_keys: Keys that were successfully pre-reserved.
            - failed_keys: Keys that failed (already exist).
            - failed_reasons: Per-key error codes for failed keys (KEYS_ALREADY_EXIST)
        """
        result = L1OperationResult()

        for key in keys:
            if self._has_key(key):
                result.add_error(key, L1ObjectManagerError.KEYS_ALREADY_EXIST)
            else:
                with self._reserved_lock:
                    self._reserved[key] = L1ObjectEntry.new_empty()
                result.add_success(key)
        return result

    def postreserve_must(
        self,
        keys: Iterable[ObjectKey],
        objects: Iterable[MemoryObj],
    ) -> L1OperationResult:
        """Thread-safe function to post-reserve a set of keys with associated
        memory objects. The keys should be already pre-reserved.

        This function uses "ALL-OR-NOTHING" semantics. If any key fails,
        the function will not associate any memory objects with the keys.

        It's not expected to have multiple thread calling this function on the
        same set of keys. If that happens, only one thread will succeed. this function
        can be lock-free if we move to per-bucket lock.

        Args:
            keys: The keys to post-reserve.
            objects: The memory objects to associate with the keys.

        Returns:
            L1OperationResult with:
            - error: SUCCESS if all keys are post-reserved successfully,
                     KEYS_NOT_RESERVED if some keys are not in reserved state
                     ENTRY_NOT_EMPTY if some keys are already associated with
                     memory objects.
            - success_keys: Keys that were successfully post-reserved.
            - failed_keys: Keys that failed (not found or entry not empty).
            - failed_reasons: Per-key error codes for failed keys.
        """
        result = L1OperationResult()
        with self._reserved_lock:
            for key, obj in zip(keys, objects, strict=False):
                if key not in self._reserved:
                    result.add_error(key, L1ObjectManagerError.KEYS_NOT_RESERVED)
                    break

                entry = self._reserved[key]
                if entry.memory_obj is not None:
                    result.add_error(key, L1ObjectManagerError.ENTRY_NOT_EMPTY)
                    break

                # NOTE: We create a new L1ObjectEntry to avoid modifying the
                # global _NON_EXISTENT_OBJECT_ENTRY instance.
                self._reserved[key] = L1ObjectEntry(
                    memory_obj=obj,
                    ttl_lock=TTLLock(self._lock_ttl),
                    dirty=True,
                    is_temporary=False,
                )
                result.add_success(key)

            if not result.is_successful():
                # Rollback
                for key in result.success_keys:
                    self._reserved[key].memory_obj = None

        # This part does not need to be in the lock
        if not result.is_successful():
            # Mark the remaining keys as skpped
            num_processed = len(result.success_keys) + len(result.failed_keys)
            result.mark_success_as_skipped()
            for key in list(keys)[num_processed:]:
                result.add_skipped(key)

        return result

    def cancel_prereserve_forced(
        self,
        keys: Iterable[ObjectKey],
    ) -> L1OperationResult:
        """Thread-safe function to cancel the pre-reservation of a set of keys.
        If the keys are already associated with memory objects (i.e., committed),
        the cancellation will fail for those keys.

        This function uses "FORCED" semantics. It will try its best to cancel
        the pre-reservation for the given keys and skip the keys that are not
        reserved or already committed.

        It's not expected to have multiple thread calling this function on the
        same set of keys. If that happens, only one thread will succeed. This function
        can be lock-free if we move to per-bucket lock.

        Args:
            keys: The keys to cancel pre-reservation.

        Returns:
            L1OperationResult with:
            - error: SUCCESS if all keys are cancelled successfully,
              KEYS_NOT_RESERVED if some keys are not reserved.
              KEYS_NOT_EMPTY if some keys are already associated with memory objects.
            - success_keys: Keys that were successfully cancelled.
            - failed_keys: Keys that failed (not reserved).
            - failed_reasons: Per-key error codes for failed keys.
        """
        result = L1OperationResult()

        with self._reserved_lock:
            for key in keys:
                if key not in self._reserved:
                    result.add_error(key, L1ObjectManagerError.KEYS_NOT_RESERVED)
                    continue

                entry = self._reserved[key]
                if entry.memory_obj is not None:
                    result.add_error(key, L1ObjectManagerError.ENTRY_NOT_EMPTY)
                else:
                    del self._reserved[key]
                    result.add_success(key)

        return result

    def mark_reserved_must(
        self,
        keys: Iterable[ObjectKey],
    ) -> L1OperationResult:
        """
        Change the existing "committed" keys as "reserved". The input keys
        are expected to be "committed" and "unlocked" and not "temporary".

        If multiple threads try to mark the same set of keys as reserved, only
        one thread will succeed for each key.

        When error happens, the function will have "ALL OR NOTHING" semantics.
        It's expected for the caller to "retry" or "abort" when the function fails.

        Args:
            keys: The keys to mark as "reserved".
            force: Use "FORCED" error handling semantics if True. Otherwise, use
                   "ALL OR NOTHING" semantics (default)

        Returns:
            L1OperationResult with:
            - error: SUCCESS if all keys are marked successfully,
              KEYS_NOT_COMMITTED if some keys are not committed,
              KEYS_ALREADY_LOCKED if some keys are locked.
              KEYS_ARE_TEMPORARY if some keys are temporary.
            - success_keys: Keys that were successfully marked as reserved.
            - failed_keys: Keys that failed to be marked.
            - failed_reasons: Per-key error codes for failed keys.

        Note:
            We don't support `FORCE` semantics here because the caller should never
            have reservation conflicts (i.e., two modules try to update to the same
            key)
        """
        result = L1OperationResult()

        with self._committed_lock, self._reserved_lock:
            for key in keys:
                if key not in self._committed:
                    result.add_error(key, L1ObjectManagerError.KEYS_NOT_COMMITTED)
                    break

                entry = self._committed[key]
                if entry.ttl_lock.is_locked():
                    result.add_error(key, L1ObjectManagerError.KEYS_ALREADY_LOCKED)
                    break

                if entry.is_temporary:
                    result.add_error(key, L1ObjectManagerError.KEYS_ARE_TEMPORARY)
                    break

                # Move the entry from committed to reserved
                entry = self._committed.pop(key)
                entry.mark_as_reserved()
                self._reserved[key] = entry
                result.add_success(key)

            if not result.is_successful():
                # Rollback
                for key in result.success_keys:
                    entry = self._reserved.pop(key)
                    entry.mark_as_committed()
                    self._committed[key] = entry

        if not result.is_successful():
            # Mark the remaining keys as skipped
            num_processed = len(result.success_keys) + len(result.failed_keys)
            result.mark_success_as_skipped()
            for key in list(keys)[num_processed:]:
                result.add_skipped(key)

        return result

    def commit(self, keys: Iterable[ObjectKey], force: bool) -> L1OperationResult:
        """
        Change the state of the keys from "reserved" to "committed". The input
        keys are expected to be "reserved".

        It's not expected to have multiple process calling this function on the
        same set of keys. In the future, this function can be lock-free if we
        move to per-bucket lock.

        Args:
            keys: The keys to commit.
            force: Use "FORCED" error handling semantics if True. Otherwise, use
                   "ALL OR NOTHING" semantics (default)

        Returns:
            L1OperationResult with:
            - error: SUCCESS if all keys are committed successfully,
              KEYS_NOT_RESERVED if some keys are not reserved (either committed
                or not exist or only pre-reserved)
            - success_keys: Keys that were successfully committed.
            - failed_keys: Keys that failed to be committed.
            - failed_reasons: Per-key error codes for failed keys.

        Note:
            There is no default value for `force`, because the caller need to
            carefully think about whether to set it or not.
        """
        result = L1OperationResult()
        rollback_on_failure = not force
        with self._committed_lock, self._reserved_lock:
            for key in keys:
                if key in self._reserved and self._reserved[key].memory_obj is not None:
                    entry = self._reserved.pop(key)
                    entry.mark_as_committed()
                    self._committed[key] = entry
                    result.add_success(key)
                    continue

                # Failed case
                result.add_error(key, L1ObjectManagerError.KEYS_NOT_RESERVED)
                if force:
                    continue
                else:
                    break

            if not result.is_successful() and rollback_on_failure:
                # Rollback
                for key in result.success_keys:
                    entry = self._committed.pop(key)
                    entry.mark_as_reserved()
                    self._reserved[key] = entry

        if not result.is_successful() and rollback_on_failure:
            # Mark the remaining keys as skipped
            num_processed = len(result.success_keys) + len(result.failed_keys)
            result.mark_success_as_skipped()
            for key in list(keys)[num_processed:]:
                result.add_skipped(key)

        return result

    def lock(self, keys: Iterable[ObjectKey], force: bool) -> L1OperationResult:
        """
        Add the lock counter and update the lock TTL for the keys.

        TTLLock itself is thread-safe, so we don't need to acquire the global locks
        Potential race: if other threads are deleting the keys while we are trying
        to access the ttl lock. We don't process this condition since it does not
        impact the correctness of the lock/unlock operation.

        Args:
            keys: The keys to lock. The keys should be committed.
            force: Use "FORCED" error handling semantics if True. Otherwise, use
                   "ALL OR NOTHING" semantics (default)

        Returns:
            L1OperationResult with:
            - error: SUCCESS if all keys are locked successfully,
              KEYS_NOT_COMMITTED if the keys is not in committed state (either
                reserved or not exist).
            - success_keys: Keys that were successfully locked.
            - failed_keys: Keys that failed to be locked.
            - failed_reasons: Per-key error codes for failed keys.
        """
        result = L1OperationResult()
        successful_entries = []

        for key in keys:
            entry = self._committed.get(key, None)
            if entry is not None:
                entry.ttl_lock.lock()
                result.add_success(key)
                successful_entries.append(entry)
                continue

            # Failed case
            result.add_error(key, L1ObjectManagerError.KEYS_NOT_COMMITTED)
            if force:
                continue
            else:
                break

        if not result.is_successful() and not force:
            # Rollback
            for entry in successful_entries:
                entry.ttl_lock.unlock()

            # Mark the remaining keys as skipped
            num_processed = len(result.success_keys) + len(result.failed_keys)
            result.mark_success_as_skipped()
            for key in list(keys)[num_processed:]:
                result.add_skipped(key)

        return result

    def unlock(self, keys: Iterable[ObjectKey], force: bool) -> L1OperationResult:
        """
        Decrease the lock counter for the keys.

        Similar to `lock()`, we don't need to acquire the global locks here.


        Args:
            keys: The keys to unlock. Not necessarily need to be locked.
            force: Use "FORCED" error handling semantics if True. Otherwise, use
                   "ALL OR NOTHING" semantics (default)

        Returns:
            L1OperationResult with:
            - error: SUCCESS if all keys are unlocked successfully,
              KEYS_NOT_COMMITTED if some keys are not committed (i.e., not exist
                or reserved).
            - success_keys: Keys that were successfully unlocked.
            - failed_keys: Keys that failed to be unlocked.
            - failed_reasons: Per-key error codes for failed keys.

        Note:
            It's recommended to use force=True here. Because it's okay to skip the
            keys that are not exist or not committed.
        """
        result = L1OperationResult()
        successful_entries = []

        for key in keys:
            entry = self._committed.get(key, None)
            if entry is not None:
                entry.ttl_lock.unlock()
                result.add_success(key)
                successful_entries.append(entry)
                continue

            # Failed case
            result.add_error(key, L1ObjectManagerError.KEYS_NOT_COMMITTED)
            if force:
                continue
            else:
                break

        if not result.is_successful() and not force:
            # Rollback
            for entry in successful_entries:
                entry.ttl_lock.lock()

            # Mark the remaining keys as skipped
            num_processed = len(result.success_keys) + len(result.failed_keys)
            result.mark_success_as_skipped()
            for key in list(keys)[num_processed:]:
                result.add_skipped(key)

        return result

    def lookup_and_lock_forced(
        self, keys: Iterable[ObjectKey]
    ) -> tuple[L1OperationResult, list[MemoryObj]]:
        """
        Lookup the keys in the "committed" ones. Lock the found ones and return
        the memory objects to the caller.

        This function will ensure that the "lookup and lock" are atomic, which
        means that once the function returns, the caller is guaranteed to have
        the lock on the returned memory objects.

        This function uses "FORCED" semantics. It will try its best to lookup
        and lock the given keys, and skip the keys that are not found or not
        committed.

        Args:
            keys: The keys to lookup and lock.

        Returns:
            A tuple of (L1OperationResult, list[MemoryObj]):
            - L1OperationResult with:
              - error: SUCCESS if key is found and locked,
                       KEYS_NOT_FOUND if some keys are not found,
                       ENTRY_IS_EMPTY if some keys are committed but empty.
              - success_keys: Keys that were found and locked.
              - failed_keys: Keys that were not found or not committed.
              - failed_reasons: Per-key error codes for failed keys.
            - list[MemoryObj]: Memory objects for the successfully locked keys,
              in the same order as success_keys.
        """
        result = L1OperationResult()
        memory_objs: list[MemoryObj] = []
        with self._committed_lock:
            for key in keys:
                entry = self._committed.get(key, None)
                if entry is None:
                    result.add_error(key, L1ObjectManagerError.KEYS_NOT_FOUND)
                    continue

                if entry.memory_obj is None:
                    result.add_error(key, L1ObjectManagerError.ENTRY_IS_EMPTY)
                    continue

                entry.ttl_lock.lock()
                result.add_success(key)
                memory_objs.append(entry.memory_obj)

        return result, memory_objs

    def query_states(self, keys: Iterable[ObjectKey]) -> list[L1ObjectState]:
        """
        Query the object states associated with the given keys.

        Args:
            keys: The keys to query.

        Returns:
            List of L1ObjectState instances for the given keys.

        Note:
            The returned list is a "snapshot" of the object states, and
            the states may change after the function returns.
        """
        ret = []
        for key in keys:
            entry = self._get_entry(key)
            if entry is None:
                ret.append(L1ObjectState.new_empty())
            else:
                ret.append(L1ObjectState.from_entry(entry))
        return ret

    def delete_committed(
        self, keys: Iterable[ObjectKey], force: bool
    ) -> L1OperationResult:
        """
        Remove the keys from the object manager. The keys should be committed but
        not locked.

        Args:
            keys: The keys to delete.
            force: Use "FORCED" error handling semantics if True. Otherwise, use
                   "ALL OR NOTHING" semantics (default)

        Returns:
            L1OperationResult with:
            - error: SUCCESS if all keys are deleted successfully,
              KEYS_NOT_COMMITTED if some keys are not committed.
              KEYS_ALREADY_LOCKED if some keys are locked.
            - success_keys: Keys that were successfully deleted.
            - failed_keys: Keys that failed to be deleted.
            - failed_reasons: Per-key error codes for failed keys.

        Note:
            The function will NOT free the memory associated with the keys. The caller
            needs to manually free.
        """
        result = L1OperationResult()
        deleted_entries: list[L1ObjectEntry] = []
        with self._committed_lock:
            for key in keys:
                if key not in self._committed:
                    result.add_error(key, L1ObjectManagerError.KEYS_NOT_COMMITTED)
                    if force:
                        continue
                    else:
                        break

                entry = self._committed[key]
                if entry.ttl_lock.is_locked():
                    result.add_error(key, L1ObjectManagerError.KEYS_ALREADY_LOCKED)
                    if force:
                        continue
                    else:
                        break

                # Delete the key
                deleted_entries.append(entry)
                del self._committed[key]
                result.add_success(key)

            if not result.is_successful() and not force:
                # Rollback
                for key, entry in zip(
                    result.success_keys, deleted_entries, strict=False
                ):
                    self._committed[key] = entry

        if not result.is_successful() and not force:
            # Mark the remaining keys as skipped
            num_processed = len(result.success_keys) + len(result.failed_keys)
            result.mark_success_as_skipped()
            for key in list(keys)[num_processed:]:
                result.add_skipped(key)

        return result

    def delete_reserved(
        self, keys: Iterable[ObjectKey], force: bool
    ) -> L1OperationResult:
        """
        Drop from the reserved keys.

        Args:
            keys: The keys to drop.
            force: Use "FORCED" error handling semantics if True. Otherwise, use
                   "ALL OR NOTHING" semantics (default)

        Returns:
            L1OperationResult with:
            - error: SUCCESS if all keys are dropped successfully,
              KEYS_NOT_RESERVED if some keys are not reserved.
            - success_keys: Keys that were successfully dropped.
            - failed_keys: Keys that failed to be dropped.
            - failed_reasons: Per-key error codes for failed keys.

        Note:
            The function will NOT free the memory associated with the keys. The caller
            needs to manually free.
        """
        result = L1OperationResult()
        dropped_entries: list[L1ObjectEntry] = []
        with self._reserved_lock:
            for key in keys:
                if key not in self._reserved:
                    result.add_error(key, L1ObjectManagerError.KEYS_NOT_RESERVED)
                    if force:
                        continue
                    else:
                        break

                # Drop the key
                entry = self._reserved.pop(key)
                dropped_entries.append(entry)
                result.add_success(key)

            if not result.is_successful() and not force:
                # Rollback
                for key, entry in zip(
                    result.success_keys, dropped_entries, strict=False
                ):
                    self._reserved[key] = entry

        if not result.is_successful() and not force:
            # Mark the remaining keys as skipped
            num_processed = len(result.success_keys) + len(result.failed_keys)
            result.mark_success_as_skipped()
            for key in list(keys)[num_processed:]:
                result.add_skipped(key)

        return result
