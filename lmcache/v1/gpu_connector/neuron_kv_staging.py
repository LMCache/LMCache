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
tensor and copies that in one shot (D2H), then scatters a staged CPU tensor
back the same way (H2D).

The gather/scatter deliberately avoids ``torch.index_select`` and
``torch.Tensor.index_copy_``: on Neuron those cost time proportional to the
*whole* paged cache rather than to the selected rows, so a 0.02 GiB request
against a 27k-block cache took over a minute. Instead the selected indices are
collapsed into maximal runs of consecutive values and each run is moved with
``torch.Tensor.narrow`` plus ``copy_``, which touches only the selected region.
Both primitives are device-agnostic, so the same code runs on CPU tensors for
tests.
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

        dim, indices = self._selection_indices(
            selected_blocks, engine_kv_format, block_size
        )
        runs = self._contiguous_runs(indices)
        for layer_tensor, staged in zip(layer_tensors, staged_layers, strict=True):
            offset = 0
            for start, length in runs:
                for destination, source in self._run_slice_pairs(
                    layer_tensor, staged, dim, start, offset, length
                ):
                    destination.copy_(source.to(layer_tensor.device))
                offset += length

    def _gather_to_cpu(
        self,
        layer_tensor: torch.Tensor,
        selected_blocks: list[int],
        engine_kv_format: "lmcache_native.EngineKVFormat",
        block_size: int,
    ) -> torch.Tensor:
        """Gather selected blocks into a contiguous CPU tensor.

        The selected indices are collapsed into consecutive runs and copied run
        by run into a freshly allocated contiguous device tensor, which is then
        moved to host in a single copy. Contiguity is required because Neuron
        device-to-host copy rejects strided views.

        :param layer_tensor: A per-layer paged KV tensor on the source device.
        :param selected_blocks: Sorted, de-duplicated KV block ids to gather.
        :param engine_kv_format: Layout of the per-layer KV tensor.
        :param block_size: Number of token slots per KV block.
        :returns: A contiguous CPU tensor holding only the selected blocks.
        """
        dim, indices = self._selection_indices(
            selected_blocks, engine_kv_format, block_size
        )
        shape = list(layer_tensor.shape)
        shape[dim] = len(indices)
        staged = torch.empty(tuple(shape), dtype=layer_tensor.dtype, device="cpu")

        offset = 0
        for start, length in self._contiguous_runs(indices):
            for source, destination in self._run_slice_pairs(
                layer_tensor, staged, dim, start, offset, length
            ):
                destination.copy_(source.to("cpu"))
            offset += length
        return staged

    def _selection(
        self,
        device: torch.device,
        selected_blocks: list[int],
        engine_kv_format: "lmcache_native.EngineKVFormat",
        block_size: int,
    ) -> tuple[int, torch.Tensor]:
        """Compute the gather/scatter dimension and index tensor for a layout.

        Retained for callers that need an explicit index tensor. The transfer
        paths deliberately do not use it; see :meth:`_selection_indices`.

        :param device: Device the returned index tensor must live on.
        :param selected_blocks: Sorted, de-duplicated KV block ids.
        :param engine_kv_format: Layout of the per-layer KV tensor.
        :param block_size: Number of token slots per KV block.
        :returns: A tuple of the tensor dimension to gather/scatter along and a
            ``torch.long`` index tensor on ``device``.
        """
        dim, indices = self._selection_indices(
            selected_blocks, engine_kv_format, block_size
        )
        return dim, torch.tensor(indices, dtype=torch.long, device=device)

    def _selection_indices(
        self,
        selected_blocks: list[int],
        engine_kv_format: "lmcache_native.EngineKVFormat",
        block_size: int,
    ) -> tuple[int, list[int]]:
        """Compute the transfer dimension and the indices to move along it.

        Block-structured layouts index the block axis directly; token-structured
        layouts (where blocks map to ``block_size`` consecutive token slots)
        index the token axis.

        :param selected_blocks: Sorted, de-duplicated KV block ids.
        :param engine_kv_format: Layout of the per-layer KV tensor.
        :param block_size: Number of token slots per KV block.
        :returns: A tuple of the tensor dimension and the sorted indices along
            it, as plain Python ints.
        """
        fmt = int(engine_kv_format)
        block_axis = {
            int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS): 0,
            int(lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS): 0,
            int(lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS): 1,
        }
        if fmt in block_axis:
            return block_axis[fmt], list(selected_blocks)
        dim = 0 if fmt == int(lmcache_native.EngineKVFormat.NL_X_NB_BS_HS) else 1
        return dim, [
            block_id * block_size + offset
            for block_id in selected_blocks
            for offset in range(block_size)
        ]

    @staticmethod
    def _run_slice_pairs(
        device_tensor: torch.Tensor,
        host_tensor: torch.Tensor,
        dim: int,
        device_start: int,
        host_start: int,
        length: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Pair up matching device and host slices for one contiguous run.

        Every returned *device* slice is contiguous, which Neuron requires for
        any host transfer. A run along dimension 0 is already a contiguous
        block; a run along dimension 1 is not, so it is split into one slice per
        leading (key/value) index, each of which is contiguous.

        :param device_tensor: The paged KV tensor on the source/destination
            device.
        :param host_tensor: The CPU staging tensor.
        :param dim: Transfer axis, either 0 or 1.
        :param device_start: Start index of the run in ``device_tensor``.
        :param host_start: Start index of the run in ``host_tensor``.
        :param length: Number of indices in the run.
        :returns: A list of ``(device_slice, host_slice)`` pairs of equal shape.
        :raises ValueError: If ``dim`` is neither 0 nor 1.
        """
        if dim == 0:
            return [
                (
                    device_tensor.narrow(0, device_start, length),
                    host_tensor.narrow(0, host_start, length),
                )
            ]
        if dim != 1:
            raise ValueError(f"Unsupported transfer dimension {dim}")
        return [
            (
                device_tensor[leading].narrow(0, device_start, length),
                host_tensor[leading].narrow(0, host_start, length),
            )
            for leading in range(device_tensor.shape[0])
        ]

    @staticmethod
    def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
        """Collapse sorted indices into maximal runs of consecutive values.

        Transfers are issued per run so each one reads a single narrow slice of
        the paged cache rather than indexing across the whole tensor.

        :param indices: Sorted, de-duplicated indices along the transfer axis.
        :returns: A list of ``(start, length)`` pairs covering ``indices`` in
            order.
        """
        runs: list[tuple[int, int]] = []
        for index in indices:
            if runs and index == runs[-1][0] + runs[-1][1]:
                start, length = runs[-1]
                runs[-1] = (start, length + 1)
            else:
                runs.append((index, 1))
        return runs

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
        valid_mask = slots_cpu >= 0
        if not bool(valid_mask.any()):
            return [], slots_cpu

        valid_slots = slots_cpu[valid_mask]
        source_blocks = torch.div(valid_slots, block_size, rounding_mode="floor")
        # ``torch.unique`` returns sorted values, so a block's position in
        # ``selected`` is exactly its index in the compacted range and
        # ``searchsorted`` recovers that mapping without a Python-level loop.
        selected = torch.unique(source_blocks)
        compact_blocks = torch.searchsorted(selected, source_blocks)

        compact = slots_cpu.clone()
        compact[valid_mask] = compact_blocks * block_size + valid_slots % block_size
        return [int(block_id) for block_id in selected.tolist()], compact

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
