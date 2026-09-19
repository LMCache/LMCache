# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import IntEnum, auto
from typing import List, Optional, Union, no_type_check
import asyncio
import ctypes
import os
import random
import string
import time

# Third Party
import eic
import yaml

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

# Capability marker: this core loads libcudart defensively (cuda_lib stays None
# when the library is absent and only an explicit TRANSPORT_GDR is rejected).
# Platform patches that previously shimmed the ctypes load to make this
# connector importable without a CUDA runtime (e.g. lmcache_ascend) must gate
# on it and no-op, because a truthy CDLL stand-in would also defeat the
# cuda_lib is None GDR guard.
_LMCACHE_EIC_CUDART_OPTIONAL = True


class Priorities(IntEnum):
    PEEK = auto()
    PREFETCH = auto()
    GET = auto()
    PUT = auto()


class PerformanceTimer:
    def __init__(self, key, op):
        self.key = key
        self.size = 0
        self.op = op
        self.start_times = {}
        self.elapsed_times = {}

    def set_size(self, size):
        self.size = size

    def start(self, operation_name):
        self.start_times[operation_name] = time.perf_counter()

    # unit: us
    def stop(self, operation_name):
        if operation_name not in self.start_times:
            raise ValueError(f"operation {operation_name} not found")

        end_time = time.perf_counter()
        start_time = self.start_times[operation_name]
        elapsed_time = end_time - start_time
        self.elapsed_times[operation_name] = elapsed_time * 1000000
        del self.start_times[operation_name]
        return elapsed_time

    def get_elapsed_time(self, operation_name):
        return self.elapsed_times.get(operation_name)

    def get_all_elapsed_times(self):
        return self.elapsed_times

    def debug_all_elapsed_times(self):
        logger.debug("== Perf key:%s =========", self.key)
        logger.debug("== Perf op %s size %s =========", self.op, self.size)
        for op, item in self.elapsed_times.items():
            logger.debug("Step: %s cost %.2f us", op, item)
        logger.debug("== Perf op %s size %s =========", self.op, self.size)


class FlexibleDRAMMemoryPool:
    def __init__(self, conn):
        self._init = False
        self.connection = conn
        self.used_mem = {}

    def allocate(self, size):
        ptr = self.connection.allocate_managed_buffer(size)
        if ptr == 0:
            logger.error("fail to allocate dram pool, ptr %s, size %s", ptr, size)
            return ptr
        logger.debug("allocate dram pool: ptr %s, size %s", ptr, size)
        self.used_mem[ptr] = size
        return ptr

    def deallocate(self, ptr):
        size = self.used_mem.get(ptr)
        if size is not None:
            logger.debug("deallocate dram pool: ptr %s, size %s", ptr, size)
            self.connection.free_managed_buffer(ptr, size)
            del self.used_mem[ptr]


def _make_dir(path: str):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        logger.info("create dir '%s' success", path)
    except OSError as e:
        logger.error("create dir '%s' error %s", path, e)


def _transfer_base_ptr(memory_obj: MemoryObj) -> int:
    """Base address of a memory object's backing bytes.

    ``MemoryObj.tensor`` views the whole allocation through the first group's
    shape and dtype; on a multi-group object (e.g. MLA latent/rope/indexer
    groups) that view raises on reshape, the put dies inside the async
    callback, and the store still reports success. The transfer only needs the
    base address, and group 0 starts at offset 0, so use its view for
    multi-group objects and keep the singular view otherwise. Returns 0 when
    there is no addressable buffer; callers must skip that chunk instead of
    handing a null pointer to the EIC client.
    """
    if len(memory_obj.get_shapes()) > 1:
        tensor = memory_obj.get_tensor(0)
    else:
        tensor = memory_obj.tensor
    if tensor is None:
        return 0
    return tensor.data_ptr()


class EICConnector(RemoteConnector):
    """
    The remote url should start with "eic://" and only have one host-port pair
    """

    def __init__(
        self,
        endpoint: str,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: LocalCPUBackend,
    ):
        # initialize base class, which includes some common attributes
        super().__init__(memory_allocator.config, memory_allocator.metadata)

        logger.info("init EICConnector")
        logger.info("try connect to eic: %s", endpoint)

        self.loop = loop
        self.memory_allocator = memory_allocator

        # Initialize pq_executor early to avoid AttributeError
        try:
            self.pq_executor = AsyncPQExecutor(loop)
            logger.info("AsyncPQExecutor initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize AsyncPQExecutor: %s", e)
            raise

        self.cudaError_t = ctypes.c_int
        self.cudaMemcpyDeviceToHost = 2
        self.cudaMemcpyHostToDevice = 1
        # libcudart only backs the CUDA GDR path. Hosts without a CUDA runtime
        # (CPU-only machines, Ascend CANN images) still use RDMA, so a missing
        # library must not abort construction: leave cuda_lib None and reject
        # only an explicit GDR configuration below.
        try:
            self.cuda_lib = ctypes.CDLL("libcudart.so")
            self.cuda_lib.cudaMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            self.cuda_lib.cudaMemcpy.restype = self.cudaError_t
        except OSError:
            self.cuda_lib = None
            logger.warning(
                "libcudart.so could not be loaded; CUDA GDR transport is "
                "unavailable, other transports such as RDMA remain usable"
            )

        self.enable_compare = False

        config_file = os.getenv("LMCACHE_CONFIG_FILE")
        if config_file is None:
            raise ValueError("LMCACHE_CONFIG_FILE environment variable is not set")
        with open(config_file, "r") as fin:
            config = yaml.safe_load(fin)

        remote_url = config.get("remote_url", None)
        logger.info("eic remote_url: %s", remote_url)

        eic_instance_id = config.get("eic_instance_id", None)
        logger.info("eic instance_id: %s", eic_instance_id)

        eic_thread_num = config.get("eic_thread_num", 6)
        logger.info("eic thread_num: %s", eic_thread_num)

        eic_log_dir = config.get("eic_log_dir", None)
        logger.info("eic log_dir: %s", eic_log_dir)

        eic_log_level = config.get("eic_log_level", 1)
        logger.info("eic log_level: %s", eic_log_level)

        eic_trans_type = config.get("eic_trans_type", 3)
        logger.info("eic trans_type: %s", eic_trans_type)

        self.eic_kv_ttl = config.get("eic_kv_ttl", -1)
        logger.info("eic eic_kv_ttl: %s", self.eic_kv_ttl)

        self.eic_kv_ns = config.get("eic_kv_ns", "")
        logger.info("eic eic_kv_ns: %s", self.eic_kv_ns)

        eic_flag_file = config.get("eic_flag_file", None)
        logger.info("eic flag_file: %s", eic_flag_file)

        _make_dir(eic_log_dir)

        self.connection = eic.Client()
        init_option = eic.InitOption()
        init_option.log_dir = eic_log_dir
        init_option.log_level = eic.LogLevel(eic_log_level)
        init_option.transport_type = eic.TransportType(eic_trans_type)
        init_option.flag_file = eic_flag_file
        endpoint = endpoint.removeprefix("eic://").removesuffix("/")
        ret = self.connection.init(eic_instance_id, endpoint, init_option)
        if ret != 0:
            logger.error("fail to init eic client, ret: %s", ret)
            raise RuntimeError(
                f"Failed to initialize eic client with error code: {ret}"
            )
        else:
            logger.info("init eic client success, ret: %s", ret)

        self.trans_type = eic.TransportType(eic_trans_type)

        # Register memory in rdma and gdr scenarios
        if self.trans_type == eic.TransportType.TRANSPORT_GDR:
            if self.cuda_lib is None:
                raise RuntimeError(
                    "eic_trans_type selects TRANSPORT_GDR but libcudart.so was "
                    "not loaded; GDR requires a CUDA runtime"
                )
            if not isinstance(self.memory_allocator, LocalCPUBackend):
                raise RuntimeError("memory_allocator must be LocalCPUBackend")
            allocator = self.memory_allocator.memory_allocator
            if not isinstance(allocator, MixedMemoryAllocator):
                raise RuntimeError(
                    "memory_allocator.memory_allocator must be MixedMemoryAllocator"
                )

            if hasattr(allocator, "pin_allocator") and hasattr(
                allocator.pin_allocator, "buffer"
            ):
                mem_pool = allocator.pin_allocator.buffer
                meminfo = eic.MemoryInfo()
                meminfo.type = eic.MemoryType.MEMORY_CUDA
                meminfo.cuda_id = 0

                vals = eic.IOBuffers()
                vals.append(
                    mem_pool.data_ptr(),
                    mem_pool.numel() * mem_pool.element_size(),
                    True,
                )

                if self.connection.register_memory(vals, meminfo):
                    logger.info("register mixed memory pin buffer success")
                else:
                    logger.error("fail to register mixed memory pin buffer")
                    exit(1)
            else:
                logger.error("mixed memory pin buffer is None")
                exit(1)
        self.prebuilt_connection()

    def prebuilt_connection(self) -> None:
        def random_string(N):
            return self.eic_kv_ns.join(
                random.choices(string.ascii_uppercase + string.digits, k=N)
            )

        try:
            for i in range(2048):
                key = random_string(30)
                self._exists_sync(key)
            logger.info("eic prebuilt connection finish")
        except Exception as e:
            logger.error("Error in prebuilt connection thread: %s", e)

    def delete_sync(self, key: str) -> bool:
        keys = eic.StringVector()
        keys.append(key)
        status_code, _ = self.connection.mdel(keys)
        if status_code != eic.StatusCode.SUCCESS:
            logger.debug("eic delete %s failed, status_code %s", key, status_code)
            return False
        return True

    def _exists_sync(self, key_str: str) -> bool:
        keys = eic.StringVector()
        keys.append(key_str)
        exist_option = eic.ExistOption()
        status_code, exist_outcome = self.connection.mexist(keys, exist_option)
        if status_code != eic.StatusCode.SUCCESS:
            logger.debug("eic exists %s failed, status_code %s", key_str, status_code)
            return False

        err_code = exist_outcome.status_codes[0]
        success = err_code == eic.StatusCode.SUCCESS
        if success:
            logger.debug("eic exists %s success", key_str)
        else:
            logger.debug(
                "eic exists %s failed, status_code %s err_code %s",
                key_str,
                status_code,
                err_code,
            )
        return success

    async def _exists(self, key: CacheEngineKey) -> bool:
        return self._exists_sync(key.to_string())

    async def exists(self, key: CacheEngineKey) -> bool:
        if not hasattr(self, "pq_executor") or self.pq_executor is None:
            logger.error("pq_executor is not initialized in EICConnector")
            raise AttributeError("pq_executor is not initialized")

        return await self.pq_executor.submit_job(
            self._exists, key=key, priority=Priorities.PEEK
        )

    def exists_sync(self, key: CacheEngineKey) -> bool:
        return self._exists_sync(key.to_string())

    async def get_meta(self, key_str: str) -> Optional[RemoteMetadata]:
        perf_timer = PerformanceTimer(key_str, "get_meta")
        perf_timer.start("total_cost")

        # Get Meta: generate meta keys and vals
        meta_keys = eic.StringVector()
        meta_vals = eic.IOBuffers()

        # Get Meta: generate meta buffer tensor
        perf_timer.start("alloc_mem")
        meta_key = key_str + "_meta"
        meta_size = self.remote_metadata_bytes
        meta_bytes = bytearray(meta_size)
        meta_bytes_ptr = self.bytes_get_ptr(meta_bytes)

        meta_keys.append(meta_key)
        meta_vals.append(meta_bytes_ptr, meta_size, False)

        perf_timer.set_size(meta_size)
        perf_timer.stop("alloc_mem")

        # Get Meta: recv meta buffer tensor
        perf_timer.start("eic_mget")
        get_option = eic.GetOption()
        get_option.ns = self.eic_kv_ns
        status_code, meta_vals, get_outcome = self.connection.mget(
            meta_keys, get_option, meta_vals
        )
        err_code = get_outcome.status_codes[0]
        if status_code != eic.StatusCode.SUCCESS or err_code != eic.StatusCode.SUCCESS:
            if err_code == eic.StatusCode.KEY_NOT_EXIST:
                logger.debug(
                    "eic mget meta %s failed, status_code %s err_code %s",
                    key_str,
                    status_code,
                    err_code,
                )
            else:
                logger.error(
                    "eic mget meta %s failed, status_code %s err_code %s",
                    key_str,
                    status_code,
                    err_code,
                )
            return None
        else:
            logger.debug("eic mget meta %s success", key_str)

        perf_timer.stop("eic_mget")

        perf_timer.start("serialize")
        meta = RemoteMetadata.deserialize(meta_bytes[:meta_size])
        perf_timer.stop("serialize")

        perf_timer.stop("total_cost")
        perf_timer.debug_all_elapsed_times()

        return meta

    async def get_data(self, key_str: str, meta: RemoteMetadata) -> Optional[MemoryObj]:
        perf_timer = PerformanceTimer(key_str, "get_data")
        perf_timer.start("total_cost")
        perf_timer.start("alloc_obj")
        memory_obj = self.memory_allocator.allocate(
            meta.shapes,
            meta.dtypes,
            meta.fmt,
        )
        if memory_obj is None:
            logger.error(
                "fail to allocate memory during remote receive key %s length %s",
                key_str,
                meta.length,
            )
            return None
        perf_timer.stop("alloc_obj")

        perf_timer.start("alloc_mem")
        obj_size = memory_obj.get_size()
        data_ptr = _transfer_base_ptr(memory_obj)
        if data_ptr == 0:
            logger.error("Memory object has no address for key %s", key_str)
            memory_obj.ref_count_down()
            return None
        data_keys = eic.StringVector()
        data_vals = eic.IOBuffers()
        data_keys.append(key_str)

        perf_timer.set_size(obj_size)
        perf_timer.stop("alloc_mem")

        try:
            if self.trans_type == eic.TransportType.TRANSPORT_GDR:
                data_vals.append(data_ptr, obj_size, True)
            else:
                data_vals.append(data_ptr, obj_size, False)

            perf_timer.start("eic_mget")
            get_option = eic.GetOption()
            get_option.ns = self.eic_kv_ns
            status_code, data_vals, get_outcome = self.connection.mget(
                data_keys, get_option, data_vals
            )
            err_code = get_outcome.status_codes[0]
            if (
                status_code != eic.StatusCode.SUCCESS
                or err_code != eic.StatusCode.SUCCESS
            ):
                logger.error(
                    "eic mget data %s failed, status_code %s err_code %s",
                    key_str,
                    status_code,
                    err_code,
                )
                memory_obj.ref_count_down()
                return None
            else:
                logger.debug("eic mget data %s success", key_str)
        except Exception as e:
            logger.error(
                "eic mget data %s raised exception: %s", key_str, e, exc_info=True
            )
            memory_obj.ref_count_down()
            return None

        perf_timer.stop("eic_mget")

        perf_timer.stop("total_cost")
        perf_timer.debug_all_elapsed_times()

        return memory_obj

    async def _get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        key_str = key.to_string()
        meta = await self.get_meta(key_str)
        if meta is None:
            return None
        data = await self.get_data(key_str, meta)
        return data

    @_lmcache_nvtx_annotate
    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        if not hasattr(self, "pq_executor") or self.pq_executor is None:
            logger.error("pq_executor is not initialized in EICConnector")
            raise AttributeError("pq_executor is not initialized")

        return await self.pq_executor.submit_job(
            self._get, key=key, priority=Priorities.GET
        )

    def bytes_get_ptr(self, mv: Union[bytearray, memoryview, bytes]) -> int:
        if isinstance(mv, bytes):
            pointer = ctypes.cast(ctypes.c_char_p(mv), ctypes.POINTER(ctypes.c_char))
            ptr = ctypes.addressof(pointer.contents)
            return ptr
        return ctypes.addressof(ctypes.c_char.from_buffer(mv))

    async def _put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        return self._put_sync(key, memory_obj)

    def _put_sync(self, key: CacheEngineKey, memory_obj: MemoryObj):
        key_str = key.to_string()
        logger.debug("eic put %s", key_str)

        perf_timer = PerformanceTimer(key_str, "put_data")
        perf_timer.start("total_cost")
        kv_bytes = memory_obj.byte_array
        kv_shapes = memory_obj.get_shapes()
        kv_dtypes = memory_obj.get_dtypes()
        memory_format = memory_obj.get_memory_format()
        value_size = memory_obj.get_physical_size()

        logger.debug(
            "eic put %s data len %s value_size %s", key_str, len(kv_bytes), value_size
        )

        kv_ptr = _transfer_base_ptr(memory_obj)
        if kv_ptr == 0:
            raise RuntimeError(f"Memory object has no address for key {key_str}")

        perf_timer.start("serialize")

        # generate meta bytes
        remote_meta = RemoteMetadata(
            self.remote_metadata_bytes, kv_shapes, kv_dtypes, memory_format
        )

        logger.debug("eic meta %s remote_meta%s", key_str, remote_meta)

        meta_bytes = remote_meta.serialize()

        perf_timer.stop("serialize")

        perf_timer.start("trans_address")
        # generate meta & data ptr
        meta_ptr = self.bytes_get_ptr(meta_bytes)
        meta_size = len(meta_bytes)
        data_ptr = kv_ptr
        data_size = len(kv_bytes)
        perf_timer.set_size(data_size)
        perf_timer.stop("trans_address")

        logger.debug(
            "eic put %s meta ptr %s len %s data ptr %s len %s",
            key_str,
            meta_ptr,
            meta_size,
            data_ptr,
            data_size,
        )

        perf_timer.start("eic_mset")
        keys = eic.StringVector()
        vals = eic.IOBuffers()

        # set meta key & value
        meta_key = key_str + "_meta"
        keys.append(meta_key)
        vals.append(meta_ptr, meta_size, False)

        # set data key & value
        keys.append(key_str)
        if self.trans_type == eic.TransportType.TRANSPORT_GDR:
            vals.append(data_ptr, data_size, True)
        else:
            vals.append(data_ptr, data_size, False)

        # set options
        set_option = eic.SetOption()
        set_option.ns = self.eic_kv_ns
        set_option.ttl_second = self.eic_kv_ttl

        status_code, set_outcome = self.connection.mset(keys, vals, set_option)
        meta_err_code = set_outcome.status_codes[0]
        data_err_code = set_outcome.status_codes[1]
        # Per-key status_codes are authoritative: PARTIAL_FAILED overall with
        # both of this call's keys succeeding is still a success. Only a
        # whole-call failure or an actual per-key failure is reported.
        if (
            status_code in (eic.StatusCode.SUCCESS, eic.StatusCode.PARTIAL_FAILED)
            and meta_err_code == eic.StatusCode.SUCCESS
            and data_err_code == eic.StatusCode.SUCCESS
        ):
            logger.debug("eic put %s success", key_str)
        else:
            # Swallowing this used to leave the caller believing the write
            # landed: the put task future completed normally and the store
            # reported success. Raise so remote_backend.put_callback observes
            # the failure through the task future.
            logger.error(
                "eic put %s failed, status_code %s, meta err_code %s, data err_code %s",
                key_str,
                status_code,
                meta_err_code,
                data_err_code,
            )
            raise RuntimeError(
                f"eic mset failed for key {key_str}, status_code {status_code}, "
                f"meta err_code {meta_err_code}, data err_code {data_err_code}"
            )

        perf_timer.stop("eic_mset")
        perf_timer.stop("total_cost")
        perf_timer.debug_all_elapsed_times()

    async def _batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        if not keys or not memory_objs:
            return

        # Prepare all keys and values for batch mset
        eic_keys = eic.StringVector()
        eic_vals = eic.IOBuffers()
        # Keep references to meta_bytes to prevent dangling pointers
        meta_list = []
        # chunk key per appended (meta, data) pair, in append order, so the
        # flat per-key mset outcome maps back to its input chunk.
        batched_key_strs: List[str] = []
        skipped_key_strs: List[str] = []
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            key_str = key.to_string()
            logger.debug("eic batched_put processing %s", key_str)

            # Get memory object data
            kv_bytes = memory_obj.byte_array
            kv_shapes = memory_obj.get_shapes()
            kv_dtypes = memory_obj.get_dtypes()
            memory_format = memory_obj.get_memory_format()
            kv_ptr = _transfer_base_ptr(memory_obj)
            if kv_ptr == 0:
                # Skip this chunk but keep preparing the rest of the batch.
                # A return here drops every other chunk. Recorded so the batch
                # result still reports the chunk as failed below; the path is
                # reachable once support_batched_put() reports True.
                logger.error("Memory object has no address for key %s", key_str)
                skipped_key_strs.append(key_str)
                continue

            remote_meta = RemoteMetadata(
                self.remote_metadata_bytes, kv_shapes, kv_dtypes, memory_format
            )
            meta_bytes = remote_meta.serialize()
            meta_list.append(meta_bytes)
            meta_ptr = self.bytes_get_ptr(meta_bytes)
            meta_size = len(meta_bytes)
            data_ptr = kv_ptr
            data_size = len(kv_bytes)

            logger.info(
                "eic batched_put %s, shapes: %s, dtypes: %s, fmt: %s",
                key_str,
                kv_shapes,
                kv_dtypes,
                memory_format,
            )

            # Add meta key & value, then data key & value, as one pair.
            eic_keys.append(key_str + "_meta")
            eic_vals.append(meta_ptr, meta_size, False)
            eic_keys.append(key_str)
            if self.trans_type == eic.TransportType.TRANSPORT_GDR:
                eic_vals.append(data_ptr, data_size, True)
            else:
                eic_vals.append(data_ptr, data_size, False)
            batched_key_strs.append(key_str)

        # Every chunk was skipped before an mset was built: nothing landed and
        # the empty call must not be mistaken for success.
        if not batched_key_strs:
            raise RuntimeError(
                "eic batched_put wrote nothing; all chunks had no address: "
                f"{skipped_key_strs}"
            )

        set_option = eic.SetOption()
        set_option.ns = self.eic_kv_ns
        set_option.ttl_second = self.eic_kv_ttl

        set_status_code, set_outcome = self.connection.mset(
            eic_keys, eic_vals, set_option
        )

        # PARTIAL_FAILED means some chunks landed and the per-key status_codes
        # below are authoritative, so the confirmation loop still runs. A
        # whole-call failure has no per-key detail to inspect.
        if set_status_code not in (
            eic.StatusCode.SUCCESS,
            eic.StatusCode.PARTIAL_FAILED,
        ):
            logger.error(
                "eic batched_put mset failed, status_code %s", set_status_code
            )
            raise RuntimeError(
                f"eic batched_put mset failed, status_code {set_status_code}, "
                f"skipped chunks {skipped_key_strs}"
            )

        # Each input chunk occupies one (meta, data) status pair. A chunk is
        # only confirmed when both landed; collect the rest and raise once so
        # the task future carries the failure instead of completing normally.
        failed = list(skipped_key_strs)
        for pos, key_str in enumerate(batched_key_strs):
            meta_err = set_outcome.status_codes[2 * pos]
            data_err = set_outcome.status_codes[2 * pos + 1]
            if (
                meta_err == eic.StatusCode.SUCCESS
                and data_err == eic.StatusCode.SUCCESS
            ):
                logger.debug("eic batched_put %s success", key_str)
                continue
            logger.error(
                "eic batched_put %s failed, meta err_code %s, data err_code %s",
                key_str,
                meta_err,
                data_err,
            )
            failed.append(key_str)

        if failed:
            raise RuntimeError(
                f"eic batched_put failed for {len(failed)} of "
                f"{len(batched_key_strs) + len(skipped_key_strs)} chunks: {failed}"
            )

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        if not hasattr(self, "pq_executor") or self.pq_executor is None:
            logger.error("pq_executor is not initialized in EICConnector")
            raise AttributeError("pq_executor is not initialized")

        return await self.pq_executor.submit_job(
            self._put, key=key, memory_obj=memory_obj, priority=Priorities.PUT
        )

    def support_batched_put(self) -> bool:
        # _batched_put builds one multi-key mset instead of letting
        # remote_backend fall back to a per-key loop. Production observation:
        # ~2.24x write throughput on an 8-chunk x 182 MB batch.
        return True

    async def batched_put(
        self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]
    ):
        if not hasattr(self, "pq_executor") or self.pq_executor is None:
            logger.error("pq_executor is not initialized in EICConnector")
            raise AttributeError("pq_executor is not initialized")

        await self.pq_executor.submit_job(
            self._batched_put,
            keys=keys,
            memory_objs=memory_objs,
            priority=Priorities.PUT,
        )

    def support_batched_async_contains(self) -> bool:
        return True

    async def _batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if not keys:
            return 0

        # Convert all keys to strings at once
        key_strings = eic.StringVector()
        for key in keys:
            key_strings.append(key.to_string())

        # Use mexist to check all keys at once
        exist_option = eic.ExistOption()
        status_code, exist_outcome = self.connection.mexist(key_strings, exist_option)

        if status_code != eic.StatusCode.SUCCESS:
            logger.error(
                "eic batched_async_contains mexist failed, status_code %s", status_code
            )
            return 0

        # Count consecutive hits from the beginning
        num_hit_counts = 0
        for i, key in enumerate(keys):
            status_code = exist_outcome.status_codes[i]
            if status_code != eic.StatusCode.SUCCESS:
                logger.debug(
                    "eic batched_async_contains %s miss, err_code %s",
                    key.to_string(),
                    status_code,
                )
                break
            num_hit_counts += 1
        return num_hit_counts

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        if not hasattr(self, "pq_executor") or self.pq_executor is None:
            logger.error("pq_executor is not initialized in EICConnector")
            raise AttributeError("pq_executor is not initialized")

        return await self.pq_executor.submit_job(
            self._batched_async_contains,
            lookup_id=lookup_id,
            keys=keys,
            pin=pin,
            priority=Priorities.PEEK,
        )

    def support_batched_get(self) -> bool:
        return True

    async def _batched_get_impl(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """Fetch N chunks with two mget calls instead of 2N serial ones.

        ``connection.mget`` is a synchronous pybind11 call, so awaiting
        ``_get`` coroutines through ``asyncio.gather`` never yields the event
        loop and N keys cost 2N serial round trips (meta plus data per key).
        Production observation on a 28K / 16-chunk load: ~344 ms to ~158 ms
        once each stage became one mget. The two stages cannot merge: data
        buffers are allocated from each meta's shapes/dtypes/fmt.

        Returns one entry per input key; a miss keeps its slot as None so
        callers can index the result against their own key list.
        """
        n = len(keys)
        if n == 0:
            return []
        key_strs = [key.to_string() for key in keys]

        # Stage 1: one mget for every meta key.
        meta_size = self.remote_metadata_bytes
        meta_bufs = [bytearray(meta_size) for _ in range(n)]
        meta_keys = eic.StringVector()
        meta_vals = eic.IOBuffers()
        for i, key_str in enumerate(key_strs):
            meta_keys.append(key_str + "_meta")
            meta_vals.append(self.bytes_get_ptr(meta_bufs[i]), meta_size, False)

        get_option = eic.GetOption()
        get_option.ns = self.eic_kv_ns
        status_code, _, meta_outcome = self.connection.mget(
            meta_keys, get_option, meta_vals
        )
        # Overall status is three-state, and FAILED on a read is an ordinary
        # all-miss, not a fault: an mget of never-written keys returns exactly
        # that. Per-key status_codes below are the only authority on which
        # keys hit.
        if status_code not in (
            eic.StatusCode.SUCCESS,
            eic.StatusCode.PARTIAL_FAILED,
            eic.StatusCode.FAILED,
        ):
            logger.error("eic batched mget meta failed, status_code %s", status_code)
            return [None] * n

        metas: List[Optional[RemoteMetadata]] = [None] * n
        for i in range(n):
            if meta_outcome.status_codes[i] != eic.StatusCode.SUCCESS:
                logger.debug(
                    "eic mget meta %s missed, err_code %s",
                    key_strs[i],
                    meta_outcome.status_codes[i],
                )
                continue
            try:
                metas[i] = RemoteMetadata.deserialize(meta_bufs[i][:meta_size])
            except Exception as e:
                logger.error("eic meta deserialize failed for %s: %s", key_strs[i], e)

        hit_indices = [i for i in range(n) if metas[i] is not None]
        if not hit_indices:
            return [None] * n

        # Stage 2: allocate buffers for every meta hit, then one mget for all
        # data keys. Every allocation enters ``staged`` and is released in the
        # finally unless the consume loop hands it to the caller, so a raise
        # from allocate, the pointer helper, or mget cannot leak the objects
        # staged before it.
        results: List[Optional[MemoryObj]] = [None] * n
        data_keys = eic.StringVector()
        data_vals = eic.IOBuffers()
        staged: List[Optional[tuple[int, MemoryObj]]] = []
        try:
            for i in hit_indices:
                meta = metas[i]
                memory_obj = self.memory_allocator.allocate(
                    meta.shapes, meta.dtypes, meta.fmt
                )
                if memory_obj is None:
                    # Allocation failure is not a miss; leave the slot None and
                    # skip it instead of requesting data with nowhere to land.
                    logger.error(
                        "fail to allocate memory during remote receive key %s",
                        key_strs[i],
                    )
                    continue
                # Take ownership as soon as allocation succeeds so the finally
                # covers a raise from the pointer helper or buffer append, not
                # just the data_ptr == 0 return.
                staged.append((i, memory_obj))
                data_ptr = _transfer_base_ptr(memory_obj)
                if data_ptr == 0:
                    staged.pop()
                    memory_obj.ref_count_down()
                    logger.error(
                        "Memory object has no address for key %s", key_strs[i]
                    )
                    continue
                data_keys.append(key_strs[i])
                data_vals.append(
                    data_ptr,
                    memory_obj.get_size(),
                    self.trans_type == eic.TransportType.TRANSPORT_GDR,
                )

            if not staged:
                return results

            try:
                status_code, _, data_outcome = self.connection.mget(
                    data_keys, get_option, data_vals
                )
            except Exception as e:
                logger.error("eic batched mget data raised exception: %s", e)
                return results

            if status_code not in (
                eic.StatusCode.SUCCESS,
                eic.StatusCode.PARTIAL_FAILED,
                eic.StatusCode.FAILED,
            ):
                logger.error(
                    "eic batched mget data failed, status_code %s", status_code
                )
                return results

            for pos, entry in enumerate(staged):
                i, memory_obj = entry
                if data_outcome.status_codes[pos] == eic.StatusCode.SUCCESS:
                    results[i] = memory_obj
                    # Ownership passes to the caller; skip it in the finally.
                    staged[pos] = None
                else:
                    logger.debug(
                        "eic mget data %s missed, err_code %s",
                        key_strs[i],
                        data_outcome.status_codes[pos],
                    )
            return results
        finally:
            for entry in staged:
                if entry is not None:
                    entry[1].ref_count_down()

    async def _batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        # Two mget calls per stage across all keys instead of 2N serial ones.
        return await self._batched_get_impl(keys)

    async def batched_get(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        if not hasattr(self, "pq_executor") or self.pq_executor is None:
            logger.error("pq_executor is not initialized in EICConnector")
            raise AttributeError("pq_executor is not initialized")

        return await self.pq_executor.submit_job(
            self._batched_get,
            keys=keys,
            priority=Priorities.GET,
        )

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def _batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        # Callers pair the returned objects positionally with the ordered key
        # list (storage_manager zips keys with results), so a hole must not
        # compact later hits forward: return the contiguous prefix of hits and
        # release everything from the first miss onward. A chunk can miss here
        # even though the earlier contains() probe hit (eviction or a partial
        # failure between the two calls), so the prefix is re-derived here.
        results = await self._batched_get_impl(keys)
        prefix: List[MemoryObj] = []
        for memory_obj in results:
            if memory_obj is None:
                break
            prefix.append(memory_obj)
        for memory_obj in results[len(prefix) :]:
            if memory_obj is not None:
                memory_obj.ref_count_down()
        return prefix

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
    ) -> List[MemoryObj]:
        if not hasattr(self, "pq_executor") or self.pq_executor is None:
            logger.error("pq_executor is not initialized in EICConnector")
            raise AttributeError("pq_executor is not initialized")

        return await self.pq_executor.submit_job(
            self._batched_get_non_blocking,
            lookup_id=lookup_id,
            keys=keys,
            priority=Priorities.PREFETCH,
        )

    async def close(self):
        if hasattr(self, "pq_executor") and self.pq_executor is not None:
            await self.pq_executor.shutdown(wait=True)
        if self.connection:
            self.connection = None
        logger.info("closed the eic connection")

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass
