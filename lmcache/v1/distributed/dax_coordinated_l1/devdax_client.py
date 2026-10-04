# SPDX-License-Identifier: Apache-2.0
"""Python lifecycle client for the native DAX-Coordinated L1 core."""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from typing import NoReturn, Protocol

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.distributed.config import (
    DaxCoordinatedL1Config,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_key_codec import (
    key_digest,
    layout_id,
    layout_profile_digest,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_layout import (
    DevDaxPayloadGeometry,
    resolve_payload_geometry,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_model_profile import (
    DevDaxModelProfile,
    resolve_devdax_model_profile,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_region import (
    DaxCoordinatedL1Region,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_types import (
    DaxCoordinatedL1RawResult as RawResult,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_types import (
    DevDaxReservationResult,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_management import MemoryObj

try:
    # First Party
    from lmcache.lmcache_dax_coordinated_l1 import (
        DevDaxBucketIndexCore,
        dax_coordinated_l1_is_formatted,
        dax_coordinated_l1_parameters,
        format_dax_coordinated_l1_region,
    )
except ModuleNotFoundError as exc:
    if exc.name != "lmcache.lmcache_dax_coordinated_l1":
        raise
    _native_import_error = exc

    def _native_extension_unavailable(*args: object, **kwargs: object) -> NoReturn:
        """Reject Device-DAX use on builds without its native module."""
        raise RuntimeError(
            "DAX-Coordinated L1 requires the "
            "lmcache_dax_coordinated_l1 extension; rebuild LMCache on "
            "Linux x86-64 with BUILD_WITH_DAX_COORDINATED_L1=1"
        ) from _native_import_error

    DevDaxBucketIndexCore = _native_extension_unavailable
    dax_coordinated_l1_parameters = _native_extension_unavailable
    dax_coordinated_l1_is_formatted = _native_extension_unavailable
    format_dax_coordinated_l1_region = _native_extension_unavailable

logger = init_logger(__name__)


def _raw_result(native_result: object) -> RawResult:
    """Convert one pybind enum value to its stable string-backed enum."""
    name = str(native_result).rsplit(".", maxsplit=1)[-1]
    return RawResult(name)


_L1_ERRORS = {
    RawResult.SUCCESS: L1Error.SUCCESS,
    RawResult.NOT_FOUND: L1Error.KEY_NOT_EXIST,
    RawResult.TARGET_INVALIDATING: L1Error.KEY_NOT_READABLE,
    RawResult.WRITER_BUSY: L1Error.KEY_NOT_WRITABLE,
    RawResult.OWNER_MISMATCH: L1Error.KEY_NOT_WRITABLE,
    RawResult.NO_FREE_BUCKET: L1Error.OUT_OF_MEMORY,
    RawResult.NO_LOCAL_PAYLOAD_SLOT: L1Error.OUT_OF_MEMORY,
    RawResult.ACTIVE_READER: L1Error.KEY_IS_LOCKED,
}


def raw_result_to_l1_error(result: RawResult) -> L1Error:
    """Map a native DAX-Coordinated L1 result to the existing public L1Error enum."""
    return _L1_ERRORS.get(result, L1Error.KEY_IN_WRONG_STATE)


class _NativePayload(Protocol):
    """Payload fields needed to create a view of a native reservation."""

    payload_offset: int
    payload_length: int
    layout_id: int
    bucket_generation: int
    slot_generation: int


@dataclass
class _ReservationContext:
    """One native token and view, with remaining locks for a read reservation."""

    token: int
    memory_obj: MemoryObj
    count: int = 0


class DaxCoordinatedL1Client:
    """Manage process-local contexts around the shared native lifecycle.

    This class intentionally has no dependency on LMCache listeners, events,
    or controllers. The owning :class:`L1Manager` adapter maps raw outcomes and
    emits lifecycle notifications only after native commit/completion succeeds.
    """

    def __init__(
        self,
        dax_coordinated_l1_config: DaxCoordinatedL1Config,
        memory_config: L1MemoryManagerConfig,
    ) -> None:
        """Map and GPU-register configured payloads before model registration.

        Args:
            dax_coordinated_l1_config: Participant, region, and qualification contract.
            memory_config: Device path, range size, and payload alignment.
        """
        self._config = dax_coordinated_l1_config
        self._rank_placement = dax_coordinated_l1_config.rank_placement
        self._closed = False
        self._lifecycle_lock = RLock()
        self._memory_config = memory_config
        self._region: DaxCoordinatedL1Region | None = None
        self._core: DevDaxBucketIndexCore | None = None
        self._geometry: DevDaxPayloadGeometry | None = None
        self._model_profile: DevDaxModelProfile | None = None
        self._layout_digest = b""
        self._write_contexts: dict[ObjectKey, _ReservationContext] = {}
        self._read_contexts: dict[ObjectKey, list[_ReservationContext]] = {}
        self._layouts: dict[int, MemoryLayoutDesc] = {}
        self._read_views: OrderedDict[
            tuple[ObjectKey, int, int, int, int, int, bool], MemoryObj
        ] = OrderedDict()
        self._read_view_hits = 0
        self._read_view_misses = 0
        self._region = DaxCoordinatedL1Region(self._config, memory_config)

    @property
    def initialized(self) -> bool:
        """Return whether model geometry is formatted and attached."""
        return self._core is not None and self._region is not None

    def initialize_model_layouts(
        self,
        model_name: str,
        kv_world_size: int,
        chunk_size: int,
        layout_descs: list[MemoryLayoutDesc],
    ) -> bool:
        """Bind the payload arena once to one admitted runtime model layout.

        Returns ``False`` on nonzero participants while participant 0 has not yet
        formatted fresh metadata. A later request retries the attach.
        """
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("DAX-Coordinated L1 client is closed")
            profile = resolve_devdax_model_profile(
                model_name,
                kv_world_size,
                chunk_size,
                layout_descs,
                expected_kv_world_size=self._rank_placement.tp_size,
            )
            if self._model_profile is None:
                geometry = resolve_payload_geometry(self._config, profile.payload_bytes)
                self._model_profile = profile
                self._geometry = geometry
            elif profile != self._model_profile:
                raise ValueError(
                    "DAX-Coordinated L1 is already bound to a different model "
                    "name or runtime layout"
                )
            if self.initialized:
                return True
            if self._geometry is None:
                raise RuntimeError("Device-DAX model geometry resolution failed")
            return self._attach_geometry(self._geometry)

    def reserve_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, DevDaxReservationResult]:
        """Add ``read_locks`` local locks per key and return payload views.

        The count belongs to this reservation, not to all hosts' readers.
        L1Manager clamps its public input; direct clients require a positive count.
        Views may be reused, but every call acquires fresh native reservations.
        """
        self._key_ranks(keys)
        if read_locks < 1:
            raise ValueError("read_locks must be positive")
        if not self.initialized and self._geometry is not None:
            self._attach_geometry(self._geometry)
        if not self.initialized:
            return {key: DevDaxReservationResult(RawResult.NOT_FOUND) for key in keys}
        _, core = self._require_initialized()
        count = read_locks
        output: dict[ObjectKey, DevDaxReservationResult] = {}
        digests = [key_digest(key, self._layout_digest) for key in keys]
        native_results = core.reserve_reads(digests, count)
        for key, native in zip(keys, native_results, strict=True):
            result = _raw_result(native.result)
            if result is not RawResult.SUCCESS:
                output[key] = DevDaxReservationResult(result)
                continue
            memory_obj = self._read_memory_obj(key, native)
            context = _ReservationContext(
                token=native.token,
                memory_obj=memory_obj,
                count=count,
            )
            self._read_contexts.setdefault(key, []).append(context)
            output[key] = DevDaxReservationResult(result, memory_obj)
        return output

    def unsafe_read(
        self, keys: list[ObjectKey]
    ) -> dict[ObjectKey, DevDaxReservationResult]:
        """Return payload views for keys already reserved by this process."""
        self._key_ranks(keys)
        output: dict[ObjectKey, DevDaxReservationResult] = {}
        for key in keys:
            contexts = self._read_contexts.get(key, [])
            if not contexts:
                output[key] = DevDaxReservationResult(RawResult.NOT_FOUND)
                continue
            output[key] = DevDaxReservationResult(
                RawResult.SUCCESS, contexts[-1].memory_obj
            )
        return output

    def finish_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, RawResult]:
        """Release ``read_locks`` local locks per key after actual read completion.

        A reservation may be consumed by several workers, one lock at a time.
        Other local reservations and peer activity flags remain protected.
        """
        self._key_ranks(keys)
        if read_locks < 1:
            raise ValueError("read_locks must be positive")
        _, core = self._require_initialized()
        count = read_locks
        output: dict[ObjectKey, RawResult] = {}
        pending: list[tuple[ObjectKey, _ReservationContext, int]] = []
        for key in keys:
            contexts = self._read_contexts.get(key, [])
            if sum(context.count for context in contexts) < count:
                output[key] = RawResult.INVALID_STATE
                continue
            remaining = count
            for context in reversed(contexts):
                released = min(context.count, remaining)
                pending.append((key, context, released))
                remaining -= released
                if remaining == 0:
                    break
        native_results = core.finish_reads(
            [context.token for _, context, _ in pending],
            [released for _, _, released in pending],
        )
        for (key, context, released), native_result in zip(
            pending, native_results, strict=True
        ):
            result = _raw_result(native_result)
            if result is RawResult.SUCCESS:
                context.count -= released
            if key not in output or result is not RawResult.SUCCESS:
                output[key] = result
        for key in output:
            contexts = self._read_contexts.get(key, [])
            remaining_contexts = [context for context in contexts if context.count > 0]
            if remaining_contexts:
                self._read_contexts[key] = remaining_contexts
            else:
                self._read_contexts.pop(key, None)
        return output

    def reserve_write(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> dict[ObjectKey, DevDaxReservationResult]:
        """Reserve new allocations; published keys reject further writes."""
        ranks = self._key_ranks(keys)
        payload_length = get_size_bytes(layout_desc.shapes, layout_desc.dtypes)
        geometry = self._geometry
        if geometry is None:
            raise RuntimeError(
                "DAX-Coordinated L1 model payload geometry "
                "is not initialized; "
                "register model layouts before reserving writes"
            )
        if payload_length > geometry.payload_slot_bytes:
            raise ValueError(
                "write payload exceeds the initialized Device-DAX slot geometry"
            )
        if not self.initialized and not self._attach_geometry(geometry):
            raise RuntimeError("DAX-Coordinated L1 attach is not ready")
        region, core = self._require_initialized()
        stable_layout_id = layout_id(layout_desc)
        self._layouts[stable_layout_id] = layout_desc
        output: dict[ObjectKey, DevDaxReservationResult] = {}
        for key, rank in zip(keys, ranks, strict=True):
            digest = key_digest(key, self._layout_digest)
            native = core.reserve_write(digest, payload_length, stable_layout_id, rank)
            result = _raw_result(native.result)
            if result is not RawResult.SUCCESS:
                output[key] = DevDaxReservationResult(result)
                continue
            memory_obj = region.make_memory_obj(
                native.payload_offset, native.payload_length, layout_desc
            )
            self._write_contexts[key] = _ReservationContext(native.token, memory_obj)
            output[key] = DevDaxReservationResult(result, memory_obj)
        return output

    def finish_write(
        self, keys: list[ObjectKey]
    ) -> dict[ObjectKey, DevDaxReservationResult]:
        """Publish completed D2H payloads and commit READY metadata."""
        self._key_ranks(keys)
        _, core = self._require_initialized()
        output: dict[ObjectKey, DevDaxReservationResult] = {}
        pending: list[tuple[ObjectKey, _ReservationContext]] = []
        for key in keys:
            context = self._write_contexts.pop(key, None)
            if context is None:
                output[key] = DevDaxReservationResult(RawResult.NOT_FOUND)
                continue
            pending.append((key, context))
        native_results = core.finish_writes([context.token for _, context in pending])
        for (key, context), native_result in zip(pending, native_results, strict=True):
            result = _raw_result(native_result)
            output[key] = DevDaxReservationResult(
                result, context.memory_obj if result is RawResult.SUCCESS else None
            )
        return output

    def finish_write_and_reserve_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, DevDaxReservationResult]:
        """Commit writes and add ``read_locks`` local locks before writer release."""
        self._key_ranks(keys)
        if read_locks < 1:
            raise ValueError("read_locks must be positive")
        _, core = self._require_initialized()
        count = read_locks
        output: dict[ObjectKey, DevDaxReservationResult] = {}
        for key in keys:
            write_context = self._write_contexts.get(key)
            if write_context is None:
                output[key] = DevDaxReservationResult(RawResult.NOT_FOUND)
                continue
            native = core.finish_write_and_reserve_read(write_context.token, count)
            result = _raw_result(native.result)
            if result is not RawResult.SUCCESS:
                if result is RawResult.GENERATION_MISMATCH:
                    self._write_contexts.pop(key, None)
                output[key] = DevDaxReservationResult(result)
                continue
            self._write_contexts.pop(key, None)
            context = _ReservationContext(
                native.token,
                write_context.memory_obj,
                count,
            )
            self._read_contexts.setdefault(key, []).append(context)
            output[key] = DevDaxReservationResult(result, write_context.memory_obj)
        return output

    def delete(self, keys: list[ObjectKey]) -> dict[ObjectKey, DevDaxReservationResult]:
        """Invalidate owned targets; returned views describe reclaimed payload."""
        self._key_ranks(keys)
        _, core = self._require_initialized()
        output: dict[ObjectKey, DevDaxReservationResult] = {}
        for key in keys:
            native = core.delete_key(key_digest(key, self._layout_digest))
            result = _raw_result(native.result)
            # Deleted views supply event metadata only; the slot is already reusable.
            memory_obj = (
                self._memory_obj(native) if result is RawResult.SUCCESS else None
            )
            output[key] = DevDaxReservationResult(result, memory_obj)
        return output

    def report_status(self) -> dict[str, int | str | bool]:
        """Return local participant, payload slot, and registration status.

        is_healthy reports local lifecycle/registration health. Use memcheck
        explicitly to validate shared metadata and bucket/slot references.
        """
        status: dict[str, int | str | bool] = {
            "is_healthy": not self._closed
            and self._region is not None
            and self._region.cuda_registered,
            "cuda_registered": self._region is not None
            and self._region.cuda_registered,
            "initialized": self.initialized,
            "region_id": self._config.region_id,
            "region_epoch": self._config.region_epoch,
            "participant_id": self._config.participant_id,
            "participant_count": self._config.participant_count,
            "ownership_mode": self._config.ownership_mode,
            "skip_payload_flush": self._config.skip_payload_flush,
            "memcheck_on_attach": self._config.memcheck_on_attach,
            "payload_put_flush_enabled": not self._config.skip_payload_flush,
            "payload_get_refresh_enabled": not self._config.skip_payload_flush,
            "read_view_cache_entries": len(self._read_views),
            "read_view_cache_limit": self._config.read_view_cache_max_entries,
            "read_view_cache_hits": self._read_view_hits,
            "read_view_cache_misses": self._read_view_misses,
            **self._model_profile_status(),
        }
        if not self.initialized:
            return status
        region, core = self._require_initialized()
        geometry = self._geometry
        assert geometry is not None
        native = core.report_status()
        status.update(
            {
                "is_healthy": not self._closed and region.cuda_registered,
                "devdax_path": self._config.devdax_path,
                "layout_profile_digest": self._layout_digest.hex(),
                "owner_slot_begin": geometry.owner_slot_begin,
                "owner_slot_count": geometry.owner_slot_count,
                "payload_slot_bytes": geometry.payload_slot_bytes,
                "payload_slot_count": geometry.payload_slot_count,
                "payload_slot_free": native.free_slots,
                "payload_slot_used": native.used_slots,
                "active_read_reservations": native.active_read_reservations,
                "active_write_reservations": native.active_write_reservations,
                "visibility_mode": self._config.visibility_mode,
                "cuda_registered": region.cuda_registered,
                "metadata_offset_bytes": region.mapping_offset,
                "payload_offset_bytes": region.payload_mapping_offset,
            }
        )
        status["payload_size_GiB"] = self._config.payload_size_bytes >> 30
        status["payload_size_bytes"] = self._config.payload_size_bytes
        for rank, free in enumerate(native.free_slots_by_rank):
            total = geometry.owner_rank_slot_counts[rank]
            status[f"rank_{rank}_payload_slot_free"] = free
            status[f"rank_{rank}_payload_slot_used"] = total - free
            status[f"rank_{rank}_owner_slot_count"] = total
        return status

    def get_memory_usage(self) -> tuple[int, int]:
        """Return owner-local occupied slot bytes and total slot capacity.

        Count reserved and committed payload slots from the local free lists.
        Peer reads retain the original slot owner's accounting. This
        query does not scan shared buckets.
        """
        geometry = self._geometry
        if geometry is None:
            return 0, self._config.owner_payload_size_bytes
        used_slots = self._core.report_status().used_slots if self._core else 0
        return (
            used_slots * geometry.payload_slot_bytes,
            geometry.owner_slot_count * geometry.payload_slot_bytes,
        )

    def get_l1_memory_desc(self) -> L1MemoryDesc:
        """Describe the complete mapped shared range."""
        if self._rank_placement.tp_size > 1:
            return L1MemoryDesc(
                ptr=0, size=0, align_bytes=self._memory_config.align_bytes
            )
        if self._region is None:
            return L1MemoryDesc(
                ptr=0, size=0, align_bytes=self._memory_config.align_bytes
            )
        pointer, size, alignment = self._region.get_memory_desc()
        return L1MemoryDesc(ptr=pointer, size=size, align_bytes=alignment)

    def memcheck(self) -> bool:
        """Validate shared state combinations and forward/back references."""
        if self._closed:
            return False
        if not self.initialized:
            return True
        _, core = self._require_initialized()
        return core.memcheck()

    def close(self) -> None:
        """Drain DMA and release the sole native index and all payload mappings."""
        with self._lifecycle_lock:
            if self._region is not None:
                self._region.synchronize()
                # The client owns the index; release reservations before unmapping.
                if self._core is not None:
                    self._core.close()
                self._write_contexts.clear()
                self._read_contexts.clear()
                self._read_views.clear()
                self._region.close()
                self._region = None
                self._core = None
            self._closed = True

    def _attach_geometry(self, geometry: DevDaxPayloadGeometry) -> bool:
        """Serialize deferred attach against registration and shutdown."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("DAX-Coordinated L1 client is closed")
            # A LOOKUP retry can race with another instance's KV registration.
            # Keep the core that owns any reservations already issued locally.
            if self.initialized:
                return True
            return self._attach_geometry_locked(geometry)

    def _attach_geometry_locked(self, geometry: DevDaxPayloadGeometry) -> bool:
        """Map metadata and attach while holding the lifecycle lock."""
        model_profile = self._model_profile
        if model_profile is None:
            raise RuntimeError("Device-DAX model profile is not initialized")
        digest = layout_profile_digest(
            self._config,
            geometry,
            self._memory_config.align_bytes,
            model_profile.layout_digest,
        )
        digest = sha256(
            digest + b"shared-index" + self._rank_placement.layout_digest()
        ).digest()

        parameters = dax_coordinated_l1_parameters(
            self._config.region_epoch,
            self._config.buckets_per_level,
            geometry.payload_slot_bytes,
            geometry.payload_slot_count,
            self._memory_config.align_bytes,
            self._visibility_mode_code(self._config.visibility_mode),
            geometry.participant_0_slot_count,
            digest,
            bytes.fromhex(self._config.hardware_qualification_digest),
            sha256(self._config.region_id.encode("utf-8")).digest(),
            self._config.payload_size_bytes,
            self._config.participant_count,
        )

        region = self._region
        if region is None:
            raise RuntimeError("DAX-Coordinated L1 client is closed")
        region.map_metadata(parameters.layout.required_metadata_bytes)
        payload_ranges = region.payload_ranges(geometry)
        if not dax_coordinated_l1_is_formatted(region.base_address):
            if self._config.participant_id != 0:
                logger.info(
                    "Device-DAX metadata is not formatted yet; participant "
                    "%d will retry after participant 0 registers the model",
                    self._config.participant_id,
                )
                return False
            format_dax_coordinated_l1_region(
                region.base_address, region.size, parameters
            )
        # The ready magic is published last by the formatter. The constructor
        # additionally validates epoch, model digest and the complete geometry.
        core = DevDaxBucketIndexCore(
            region.base_address,
            region.size,
            self._config.participant_id,
            parameters,
            payload_ranges,
            self._config.skip_payload_flush,
            self._config.memcheck_on_attach,
        )
        self._core = core
        self._layout_digest = digest
        logger.info(
            "Device-DAX model payload geometry: object=%d slot=%d "
            "slots=%d owner=[%d,%d) payload_used=%d payload_unused=%d "
            "model=%s family=%s chunk_size=%d dtype=%s",
            geometry.model_payload_bytes,
            geometry.payload_slot_bytes,
            geometry.payload_slot_count,
            geometry.owner_slot_begin,
            geometry.owner_slot_begin + geometry.owner_slot_count,
            geometry.payload_bytes_used,
            geometry.payload_bytes_unused,
            model_profile.model_name,
            model_profile.model_family,
            model_profile.chunk_size,
            ",".join(model_profile.dtype_names),
        )
        if self._config.skip_payload_flush:
            logger.warning(
                "Device-DAX DMA payload flush/refresh is disabled by "
                "skip_payload_flush=true. DDIO non-allocating writes, UC payload "
                "memory on every host, DMA-only payload use and cross-host DMA "
                "visibility are required but not verified by the runtime. "
                "All hosts must quiesce and clear pre-existing payload cache lines "
                "before using this qualified DMA-only region."
            )
        return True

    @staticmethod
    def _visibility_mode_code(mode: str) -> int:
        return {
            "x86_clflush_64b_v1": 1,
            "x86_clflushopt_bulk_v1": 2,
        }[mode]

    def _require_initialized(
        self,
    ) -> tuple[DaxCoordinatedL1Region, DevDaxBucketIndexCore]:
        region = self._region
        core = self._core
        if region is None or core is None:
            raise RuntimeError(
                "DAX-Coordinated L1 has not received a model payload layout"
            )
        return region, core

    def _key_ranks(self, keys: list[ObjectKey]) -> list[int]:
        """Validate the whole TP batch before reserving or releasing any slot."""
        if self._closed:
            raise RuntimeError("DAX-Coordinated L1 client is closed")
        ranks = []
        for key in keys:
            ranks.append(self._rank_placement.rank_from_kv_rank(key.kv_rank))
            if key.object_group_id != 0:
                raise ValueError(
                    "TP DAX-Coordinated L1 requires homogeneous object group 0"
                )
            if self._model_profile and key.model_name != self._model_profile.model_name:
                raise ValueError("TP key model differs from registered model")
        return ranks

    def _model_profile_status(self) -> dict[str, int | str]:
        """Return the bounded public projection of the bound model profile."""
        profile = self._model_profile
        if profile is None:
            return {}
        status: dict[str, int | str] = {
            "model_name": profile.model_name,
            "model_family": profile.model_family,
            "kv_world_size": profile.kv_world_size,
            "chunk_size": profile.chunk_size,
            "model_payload_bytes": profile.payload_bytes,
            "model_layout_digest": profile.layout_digest.hex(),
            "model_dtypes": ",".join(profile.dtype_names),
        }
        status["tp_size"] = self._rank_placement.tp_size
        status["payload_region_count"] = len(self._rank_placement.regions)
        status["tp_layout_digest"] = self._rank_placement.layout_digest().hex()
        status["payload_unused_partition_bytes"] = (
            sum(r.payload_size_GiB << 30 for r in self._rank_placement.regions)
            - self._config.payload_size_bytes
        )
        for placement in self._rank_placement.placements():
            prefix = f"rank_{placement.rank}_"
            status.update(
                {
                    prefix + "devdax_path": placement.devdax_path,
                    prefix + "payload_offset_bytes": placement.payload_offset_bytes,
                    prefix + "payload_size_bytes": placement.payload_size_GiB << 30,
                }
            )
        return status

    def _read_memory_obj(self, key: ObjectKey, native: _NativePayload) -> MemoryObj:
        """Reuse a bounded non-owning view after native read validation succeeds.

        The client owns one mapping lifetime. Key/generation and the full view
        geometry prevent reuse across slot replacement or layout discovery.
        Eviction drops only the cache reference; active contexts keep their view
        and native token until DMA completion and finish_read.
        """
        identity = (
            key,
            native.bucket_generation,
            native.slot_generation,
            native.payload_offset,
            native.payload_length,
            native.layout_id,
            native.layout_id in self._layouts,
        )
        memory_obj = self._read_views.get(identity)
        if (
            memory_obj is not None
            and memory_obj.is_valid()
            and memory_obj.get_size() == native.payload_length
        ):
            self._read_views.move_to_end(identity)
            self._read_view_hits += 1
            return memory_obj
        memory_obj = self._memory_obj(native)
        self._read_views[identity] = memory_obj
        self._read_views.move_to_end(identity)
        if len(self._read_views) > self._config.read_view_cache_max_entries:
            self._read_views.popitem(last=False)
        self._read_view_misses += 1
        return memory_obj

    def _memory_obj(self, native: _NativePayload) -> MemoryObj:
        """Build a payload view, using a locally known layout when possible."""
        region, _ = self._require_initialized()
        return region.make_memory_obj(
            native.payload_offset,
            native.payload_length,
            self._layouts.get(native.layout_id),
        )
