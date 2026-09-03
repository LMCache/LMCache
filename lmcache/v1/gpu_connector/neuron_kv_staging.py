# SPDX-License-Identifier: Apache-2.0
"""Neuron-specific KV staging for LMCache GPU connectors.

This path stages selected paged-KV blocks between Neuron device memory and
temporary CPU tensors, then reuses the existing CPU-side pack/unpack logic in
``device_ops.multi_layer_kv_transfer``. It exists because Neuron device-to-host
copy (``tensor.to("cpu")``) only succeeds on fully-contiguous tensors: slicing
out individual KV blocks yields a non-contiguous view, and ``.contiguous()`` is
a no-op on Neuron shared-storage tensors, so a naive block-slice copy raises
``Expected self.is_contiguous() to be true``.

The staging therefore gathers the selected blocks into a contiguous device
tensor with ``torch.index_select`` and copies that in one shot (D2H), and
scatters a staged CPU tensor back with ``torch.Tensor.index_copy_`` (H2D). Both
primitives are device-agnostic, so the same code runs on CPU tensors for tests.
"""

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.logging import init_logger
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


class NeuronKVBlockStager:
    """Stage selected KV blocks between Neuron memory and CPU paged tensors."""

    def transfer_into_key_value(
        self,
        key_value: torch.Tensor,
        layer_tensors: list[torch.Tensor],
        slot_mapping: torch.Tensor,
        engine_kv_format: "lmcache_native.EngineKVFormat",
        block_size: int,
        head_size: int,
    ) -> None:
        """Stage selected KV blocks from device memory into a CPU tensor (D2H).

        :param key_value: CPU destination tensor packed by
            ``device_ops.multi_layer_kv_transfer``.
        :param layer_tensors: Per-layer paged KV tensors on the source device.
        :param slot_mapping: Token-to-slot mapping for the request.
        :param engine_kv_format: Layout of the per-layer KV tensors.
        :param block_size: Number of token slots per KV block.
        :param head_size: Size of each attention head.
        :raises ValueError: If ``key_value`` is not a CPU tensor.
        """
        if key_value.device.type != "cpu":
            raise ValueError("Neuron staging requires a CPU destination tensor")
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
            self._gather_to_cpu(
                layer_tensor, selected_blocks, engine_kv_format, block_size
            )
            for layer_tensor in layer_tensors
        ]

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

    def transfer_from_key_value(
        self,
        key_value: torch.Tensor,
        layer_tensors: list[torch.Tensor],
        slot_mapping: torch.Tensor,
        engine_kv_format: "lmcache_native.EngineKVFormat",
        block_size: int,
        head_size: int,
    ) -> None:
        """Scatter staged CPU KV blocks back into device memory (H2D).

        :param key_value: CPU source tensor unpacked by
            ``device_ops.multi_layer_kv_transfer``.
        :param layer_tensors: Per-layer paged KV tensors on the destination
            device; updated in place at the selected block positions.
        :param slot_mapping: Token-to-slot mapping for the request.
        :param engine_kv_format: Layout of the per-layer KV tensors.
        :param block_size: Number of token slots per KV block.
        :param head_size: Size of each attention head.
        :raises ValueError: If ``key_value`` is not a CPU tensor.
        """
        if key_value.device.type != "cpu":
            raise ValueError("Neuron staging requires a CPU source tensor")
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
        compact_page_buffer_size = compact_num_blocks * block_size

        device_ops.multi_layer_kv_transfer(
            key_value,
            staged_layers,
            compact_slot_mapping,
            torch.device("cpu"),
            compact_page_buffer_size,
            lmcache_native.TransferDirection.H2D,
            engine_kv_format,
            block_size=block_size,
            head_size=head_size,
        )

        for layer_tensor, staged in zip(layer_tensors, staged_layers, strict=True):
            dim, indices = self._selection(
                layer_tensor.device, selected_blocks, engine_kv_format, block_size
            )
            layer_tensor.index_copy_(dim, indices, staged.to(layer_tensor.device))

    def _gather_to_cpu(
        self,
        layer_tensor: torch.Tensor,
        selected_blocks: list[int],
        engine_kv_format: "lmcache_native.EngineKVFormat",
        block_size: int,
    ) -> torch.Tensor:
        """Gather selected blocks into a contiguous CPU tensor.

        ``torch.index_select`` produces a contiguous tensor, which is the only
        shape Neuron device-to-host copy accepts.

        :param layer_tensor: A per-layer paged KV tensor on the source device.
        :param selected_blocks: Sorted, de-duplicated KV block ids to gather.
        :param engine_kv_format: Layout of the per-layer KV tensor.
        :param block_size: Number of token slots per KV block.
        :returns: A contiguous CPU tensor holding only the selected blocks.
        """
        dim, indices = self._selection(
            layer_tensor.device, selected_blocks, engine_kv_format, block_size
        )
        return torch.index_select(layer_tensor, dim, indices).to("cpu")

    def _selection(
        self,
        device: torch.device,
        selected_blocks: list[int],
        engine_kv_format: "lmcache_native.EngineKVFormat",
        block_size: int,
    ) -> tuple[int, torch.Tensor]:
        """Compute the gather/scatter dimension and index tensor for a layout.

        Block-structured layouts index the block axis directly; token-structured
        layouts (where blocks map to ``block_size`` consecutive token slots)
        index the token axis.

        :param device: Device the returned index tensor must live on.
        :param selected_blocks: Sorted, de-duplicated KV block ids.
        :param engine_kv_format: Layout of the per-layer KV tensor.
        :param block_size: Number of token slots per KV block.
        :returns: A tuple of the tensor dimension to gather/scatter along and a
            ``torch.long`` index tensor on ``device``.
        """
        fmt = int(engine_kv_format)
        block_axis = {
            int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS): 0,
            int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS): 0,
            int(lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS): 1,
        }
        if fmt in block_axis:
            dim = block_axis[fmt]
            indices = selected_blocks
        elif fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_BS_HS):
            dim = 0
            indices = [
                block_id * block_size + offset
                for block_id in selected_blocks
                for offset in range(block_size)
            ]
        else:
            dim = 1
            indices = [
                block_id * block_size + offset
                for block_id in selected_blocks
                for offset in range(block_size)
            ]
        return dim, torch.tensor(indices, dtype=torch.long, device=device)

    def _compact_slot_mapping(
        self, slot_mapping: torch.Tensor, block_size: int
    ) -> tuple[list[int], torch.Tensor]:
        """Remap a slot mapping onto a compacted, gap-free block range.

        :param slot_mapping: Token-to-slot mapping; negative entries are unused
            (e.g. prefix-cache padding) and skipped.
        :param block_size: Number of token slots per KV block.
        :returns: A tuple of the sorted, de-duplicated source block ids and a
            CPU ``torch.long`` slot mapping rewritten against the compacted
            block layout.
        """
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
        """Allocate a CPU staging tensor sized for the compacted block range.

        :param layer_tensor: A per-layer paged KV tensor whose dtype and
            non-block dimensions are mirrored.
        :param compact_num_blocks: Number of blocks in the compacted range.
        :param block_size: Number of token slots per KV block.
        :param engine_kv_format: Layout of the per-layer KV tensor.
        :returns: An uninitialized CPU tensor matching the staged layout.
        """
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
