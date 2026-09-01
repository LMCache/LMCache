# SPDX-License-Identifier: Apache-2.0
"""Neuron-specific NIXL staging for LMCache GPU connectors.

This path stages selected paged-KV blocks from Neuron device memory into a
temporary CPU tensor via NIXL, then reuses the existing CPU-side pack logic in
``device_ops.multi_layer_kv_transfer``. It exists to avoid ``tensor.cpu()`` on
Neuron shared-storage tensors, which fails at runtime.
"""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional
import os
import time
import uuid

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.logging import init_logger
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


def _load_nixl_api():
    from nixl._api import nixl_agent, nixl_agent_config

    return nixl_agent, nixl_agent_config


def _load_vllm_nixl_wrapper():
    from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config

    return NixlWrapper, nixl_agent_config


def _parse_backends(backends: Optional[Sequence[str] | str]) -> list[str]:
    if backends is None:
        env = os.environ.get("LMCACHE_NEURON_NIXL_BACKENDS", "")
        if env:
            return [item.strip() for item in env.split(",") if item.strip()]
        return ["LIBFABRIC"]
    if isinstance(backends, str):
        return [item.strip() for item in backends.split(",") if item.strip()]
    return [str(item).strip() for item in backends if str(item).strip()]


def _view_nbytes(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return 0
    max_offset = 0
    for dim, stride in zip(tensor.shape, tensor.stride(), strict=True):
        max_offset += (int(dim) - 1) * int(stride)
    return (max_offset + 1) * tensor.element_size()


def _tensor_device_id(tensor: torch.Tensor) -> int:
    device = tensor.device
    if device.type == "cpu":
        return 0
    idx = getattr(device, "index", None)
    if idx is not None:
        return int(idx)
    get_device = getattr(tensor, "get_device", None)
    if callable(get_device):
        try:
            return max(int(get_device()), 0)
        except Exception:
            return 0
    return 0


def _slice_addr(
    tensor: torch.Tensor,
    indices: tuple[int, ...],
) -> int:
    offset_elems = 0
    for idx, stride in zip(indices, tensor.stride(), strict=True):
        offset_elems += int(idx) * int(stride)
    return int(tensor.data_ptr()) + offset_elems * tensor.element_size()


@dataclass(frozen=True)
class _TransferRegion:
    addr: int
    size_bytes: int
    device_id: int


class NeuronNixlBlockStager:
    """Stage selected KV blocks from Neuron memory into CPU paged tensors."""

    def __init__(self, backends: Optional[Sequence[str] | str] = None):
        self.backends = _parse_backends(backends)
        self._agent: Any = None
        self._src_wrapper: Any = None
        self._dst_wrapper: Any = None

    def transfer_into_key_value(
        self,
        key_value: torch.Tensor,
        layer_tensors: list[torch.Tensor],
        slot_mapping: torch.Tensor,
        engine_kv_format: "lmcache_native.EngineKVFormat",
        block_size: int,
        head_size: int,
    ) -> None:
        if key_value.device.type != "cpu":
            raise ValueError("Neuron NIXL staging requires a CPU destination tensor")
        if not layer_tensors:
            return
        if slot_mapping.numel() == 0:
            return

        selected_blocks, compact_slot_mapping = self._compact_slot_mapping(
            slot_mapping, block_size
        )
        if not selected_blocks:
            return

        compact_num_blocks = len(selected_blocks)
        staged_layers = [
            self._alloc_stage_tensor(
                layer_tensor, compact_num_blocks, block_size, engine_kv_format
            )
            for layer_tensor in layer_tensors
        ]

        for src_layer, dst_layer in zip(layer_tensors, staged_layers, strict=True):
            self._copy_layer_blocks(
                src_layer=src_layer,
                dst_layer=dst_layer,
                selected_blocks=selected_blocks,
                engine_kv_format=engine_kv_format,
                block_size=block_size,
            )

        compact_page_buffer_size = compact_num_blocks * block_size
        device_ops.multi_layer_kv_transfer(
            key_value,
            staged_layers,
            compact_slot_mapping,
            torch.device("cpu"),
            compact_page_buffer_size,
            lmcache_native.TransferDirection.D2H,
            engine_kv_format,
            block_size=block_size,
            head_size=head_size,
        )

    def _ensure_agent(self):
        if self._agent is not None:
            return self._agent
        nixl_agent, nixl_agent_config = _load_nixl_api()
        self._agent = nixl_agent(
            f"lmcache-neuron-nixl-{id(self)}",
            nixl_agent_config(backends=self.backends),
        )
        return self._agent

    def _create_wrapper(self):
        NixlWrapper, nixl_agent_config = _load_vllm_nixl_wrapper()
        config = nixl_agent_config(backends=self.backends)
        return NixlWrapper(str(uuid.uuid4()), config)

    def _ensure_src_wrapper(self):
        if self._src_wrapper is None:
            self._src_wrapper = self._create_wrapper()
        return self._src_wrapper

    def _ensure_dst_wrapper(self):
        if self._dst_wrapper is None:
            self._dst_wrapper = self._create_wrapper()
        return self._dst_wrapper

    def _copy_layer_blocks(
        self,
        src_layer: torch.Tensor,
        dst_layer: torch.Tensor,
        selected_blocks: list[int],
        engine_kv_format: "lmcache_native.EngineKVFormat",
        block_size: int,
    ) -> None:
        device_type = str(src_layer.device).split(":")[0]
        if device_type == "neuron":
            self._copy_layer_blocks_via_vllm_wrapper(
                src_layer=src_layer,
                dst_layer=dst_layer,
                selected_blocks=selected_blocks,
                engine_kv_format=engine_kv_format,
                block_size=block_size,
            )
            return

        agent = self._ensure_agent()
        src_mem_type = "VRAM"
        dst_mem_type = "DRAM"

        src_nbytes = _view_nbytes(src_layer)
        dst_nbytes = _view_nbytes(dst_layer)
        src_reg = agent.register_memory(
            [(int(src_layer.data_ptr()), src_nbytes, _tensor_device_id(src_layer), "")],
            mem_type=src_mem_type,
        )
        dst_reg = agent.register_memory(
            [(int(dst_layer.data_ptr()), dst_nbytes, 0, "")],
            mem_type=dst_mem_type,
        )
        src_regions = self._regions_for_tensor(
            src_layer, selected_blocks, engine_kv_format, block_size
        )
        dst_regions = self._regions_for_tensor(
            dst_layer,
            list(range(len(selected_blocks))),
            engine_kv_format,
            block_size,
        )
        src_descs = agent.get_xfer_descs(
            [(r.addr, r.size_bytes, r.device_id) for r in src_regions],
            mem_type=src_mem_type,
        )
        dst_descs = agent.get_xfer_descs(
            [(r.addr, r.size_bytes, r.device_id) for r in dst_regions],
            mem_type=dst_mem_type,
        )
        src_handle = agent.prep_xfer_dlist("", src_descs, mem_type=src_mem_type)
        dst_handle = agent.prep_xfer_dlist("", dst_descs, mem_type=dst_mem_type)
        xfer_handle = None
        try:
            indices = list(range(len(src_regions)))
            xfer_handle = agent.make_prepped_xfer(
                "READ",
                dst_handle,
                indices,
                src_handle,
                indices,
            )
            state = agent.transfer(xfer_handle)
            deadline = time.monotonic() + 30.0
            while state not in ("DONE", "ERR"):
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "NIXL READ timed out after 30s for staged KV block copy"
                    )
                state = agent.check_xfer_state(xfer_handle)
            if state != "DONE":
                raise RuntimeError(
                    f"NIXL READ failed for staged KV block copy: state={state}"
                )
        finally:
            if xfer_handle is not None:
                agent.release_xfer_handle(xfer_handle)
            agent.deregister_memory(src_reg)
            agent.deregister_memory(dst_reg)

    def _copy_layer_blocks_via_vllm_wrapper(
        self,
        src_layer: torch.Tensor,
        dst_layer: torch.Tensor,
        selected_blocks: list[int],
        engine_kv_format: "lmcache_native.EngineKVFormat",
        block_size: int,
    ) -> None:
        src_wrapper = self._ensure_src_wrapper()
        dst_wrapper = self._ensure_dst_wrapper()
        src_mem_type = "VRAM"
        dst_mem_type = "DRAM"

        src_nbytes = _view_nbytes(src_layer)
        dst_nbytes = _view_nbytes(dst_layer)
        src_reg_descs = src_wrapper.get_reg_descs(
            [(int(src_layer.data_ptr()), src_nbytes, _tensor_device_id(src_layer), "")],
            src_mem_type,
        )
        dst_reg_descs = dst_wrapper.get_reg_descs(
            [(int(dst_layer.data_ptr()), dst_nbytes, 0, "")],
            dst_mem_type,
        )
        src_wrapper.register_memory(src_reg_descs, backends=self.backends)
        dst_wrapper.register_memory(dst_reg_descs, backends=self.backends)

        src_regions = self._regions_for_tensor(
            src_layer, selected_blocks, engine_kv_format, block_size
        )
        dst_regions = self._regions_for_tensor(
            dst_layer,
            list(range(len(selected_blocks))),
            engine_kv_format,
            block_size,
        )

        src_descs = dst_wrapper.get_xfer_descs(
            [(r.addr, r.size_bytes, r.device_id) for r in src_regions],
            src_mem_type,
        )
        dst_descs = dst_wrapper.get_xfer_descs(
            [(r.addr, r.size_bytes, r.device_id) for r in dst_regions],
            dst_mem_type,
        )
        remote_agent_name = None
        src_handle = None
        dst_handle = None
        remote_agent_name = dst_wrapper.add_remote_agent(
            src_wrapper.get_agent_metadata()
        )
        src_handle = dst_wrapper.prep_xfer_dlist(remote_agent_name, src_descs)
        dst_handle = dst_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", dst_descs)
        xfer_handle = None
        try:
            indices = list(range(len(src_regions)))
            xfer_handle = dst_wrapper.make_prepped_xfer(
                "READ",
                dst_handle,
                indices,
                src_handle,
                indices,
                b"lmcache-neuron-staging",
            )
            import time

            state = dst_wrapper.transfer(xfer_handle)
            deadline = time.monotonic() + 30.0
            while state not in ("DONE", "ERR"):
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "NIXL READ timed out after 30s for staged KV block copy"
                    )
                state = dst_wrapper.check_xfer_state(xfer_handle)
            if state != "DONE":
                raise RuntimeError(
                    f"NIXL READ failed for staged KV block copy: state={state}"
                )
        except Exception:
            logger.exception(
                "Neuron NIXL staging failed: selected_blocks=%s src_nbytes=%d "
                "dst_nbytes=%d src_regions=%s dst_regions=%s",
                selected_blocks,
                src_nbytes,
                dst_nbytes,
                [(r.addr, r.size_bytes, r.device_id) for r in src_regions],
                [(r.addr, r.size_bytes, r.device_id) for r in dst_regions],
            )
            raise
        finally:
            if xfer_handle is not None:
                dst_wrapper.release_xfer_handle(xfer_handle)
            if src_handle is not None:
                dst_wrapper.release_dlist_handle(src_handle)
            if dst_handle is not None:
                dst_wrapper.release_dlist_handle(dst_handle)
            if remote_agent_name is not None and hasattr(
                dst_wrapper, "remove_remote_agent"
            ):
                dst_wrapper.remove_remote_agent(remote_agent_name)
            if hasattr(src_wrapper, "deregister_memory"):
                src_wrapper.deregister_memory(src_reg_descs)
            if hasattr(dst_wrapper, "deregister_memory"):
                dst_wrapper.deregister_memory(dst_reg_descs)

    def _compact_slot_mapping(
        self, slot_mapping: torch.Tensor, block_size: int
    ) -> tuple[list[int], torch.Tensor]:
        slots_cpu = slot_mapping.to(dtype=torch.long, device="cpu")
        valid_slots = [int(v) for v in slots_cpu.tolist() if int(v) >= 0]
        if not valid_slots:
            return [], slots_cpu

        selected_blocks = sorted({slot // block_size for slot in valid_slots})
        block_map = {block_id: i for i, block_id in enumerate(selected_blocks)}
        compact = slots_cpu.clone()
        for idx, slot in enumerate(compact.tolist()):
            if slot < 0:
                continue
            old_block = int(slot) // block_size
            offset = int(slot) % block_size
            compact[idx] = block_map[old_block] * block_size + offset
        return selected_blocks, compact

    def _alloc_stage_tensor(
        self,
        layer_tensor: torch.Tensor,
        compact_num_blocks: int,
        block_size: int,
        engine_kv_format: "lmcache_native.EngineKVFormat",
    ) -> torch.Tensor:
        shape = list(layer_tensor.shape)
        fmt = int(engine_kv_format)
        if fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_BS_HS):
            shape[0] = compact_num_blocks * block_size
        elif fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS):
            shape[0] = compact_num_blocks
        elif fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS):
            shape[0] = compact_num_blocks
        elif fmt == int(lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS):
            shape[1] = compact_num_blocks
        else:
            shape[1] = compact_num_blocks * block_size
        return torch.empty(tuple(shape), dtype=layer_tensor.dtype, device="cpu")

    def _regions_for_tensor(
        self,
        tensor: torch.Tensor,
        block_ids: list[int],
        engine_kv_format: "lmcache_native.EngineKVFormat",
        block_size: int,
    ) -> list[_TransferRegion]:
        fmt = int(engine_kv_format)
        elt_size = tensor.element_size()
        device_id = _tensor_device_id(tensor)
        regions: list[_TransferRegion] = []

        if fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_BS_HS):
            hidden = int(tensor.shape[1])
            block_bytes = block_size * hidden * elt_size
            for block_id in block_ids:
                start = block_id * block_size
                regions.append(
                    _TransferRegion(
                        addr=_slice_addr(tensor, (start, 0)),
                        size_bytes=block_bytes,
                        device_id=device_id,
                    )
                )
            return regions

        if fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS):
            hidden = int(tensor.shape[3])
            block_bytes = 2 * block_size * hidden * elt_size
            for block_id in block_ids:
                regions.append(
                    _TransferRegion(
                        addr=_slice_addr(tensor, (block_id, 0, 0, 0)),
                        size_bytes=block_bytes,
                        device_id=device_id,
                    )
                )
            return regions

        if fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS):
            num_heads = int(tensor.shape[2])
            head_size = int(tensor.shape[4])
            block_bytes = 2 * num_heads * block_size * head_size * elt_size
            for block_id in block_ids:
                regions.append(
                    _TransferRegion(
                        addr=_slice_addr(tensor, (block_id, 0, 0, 0, 0)),
                        size_bytes=block_bytes,
                        device_id=device_id,
                    )
                )
            return regions

        if fmt == int(lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS):
            num_heads = int(tensor.shape[2])
            head_size = int(tensor.shape[4])
            block_bytes = num_heads * block_size * head_size * elt_size
            for block_id in block_ids:
                for kv_idx in range(2):
                    regions.append(
                        _TransferRegion(
                            addr=_slice_addr(tensor, (kv_idx, block_id, 0, 0, 0)),
                            size_bytes=block_bytes,
                            device_id=device_id,
                        )
                    )
            return regions

        hidden = int(tensor.shape[2])
        block_bytes = block_size * hidden * elt_size
        for block_id in block_ids:
            start = block_id * block_size
            for kv_idx in range(2):
                regions.append(
                    _TransferRegion(
                        addr=_slice_addr(tensor, (kv_idx, start, 0)),
                        size_bytes=block_bytes,
                        device_id=device_id,
                    )
                )
        return regions
