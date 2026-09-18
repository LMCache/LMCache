# SPDX-License-Identifier: Apache-2.0
"""Neuron-specific KV staging for LMCache GPU connectors.

This path stages selected paged-KV blocks between Neuron device memory and
temporary CPU tensors, then reuses the existing CPU-side pack/unpack logic in
``device_ops.multi_layer_kv_transfer``. It exists because Neuron device-to-host
copy (``tensor.to("cpu")``) only succeeds on fully-contiguous tensors: slicing
out individual KV blocks yields a non-contiguous view, and ``.contiguous()`` is
a no-op on Neuron shared-storage tensors, so a naive block-slice copy raises
``Expected self.is_contiguous() to be true``.

The staging therefore moves each selected region directly between the paged KV
tensor and a host staging buffer, one ``narrow`` + ``copy_`` per contiguous run
per layer.

Three properties of Neuron copies, measured on a trn2.3xlarge (SDK 2.31, native
``torch.neuron`` backend), determine that shape. They are recorded here because
they are not obvious and they rule out the designs one would otherwise reach for:

1. **A device-to-device copy between views of differently-sized parent tensors
   fails** with ``nrt_tensor_copy status=2``. So the selected blocks *cannot* be
   gathered into a compact device-side buffer and shipped to the host in one
   transfer -- that buffer is necessarily smaller than the paged cache. Only
   same-sized parents work device-to-device.
2. **A direct device-to-host (or host-to-device) copy between views of
   differently-sized parents works and is correct.** No intermediate
   ``.to("cpu")`` temporary is needed, so none is allocated.
3. **Per-copy overhead is small.** A single block-slice transfer costs about
   0.017 ms, and a whole 32-copy chunk of a 16-layer model about 0.7 ms
   (~1.35 GiB/s store, ~1.67 GiB/s retrieve). Issuing one copy per run per layer
   is therefore cheap; minimising the *number* of transfers is not worth
   contorting the design for.

The gather/scatter deliberately avoids ``torch.index_select`` and
``torch.Tensor.index_copy_``: on Neuron those cost time proportional to the
*whole* paged cache rather than to the selected rows, so a 0.02 GiB request
against a 27k-block cache took over a minute. Instead the selected indices are
collapsed into maximal runs of consecutive values and each run is moved with
``torch.Tensor.narrow`` plus ``copy_``, which touches only the selected region.
Both primitives are device-agnostic, so the same code runs on CPU tensors for
tests.

Staging buffers are host-side, cached, and reused across chunks. Chunk geometry
is fixed by ``chunk_size`` and the layer layout, so steady-state serving
reallocates nothing; a differently-shaped request (the final short chunk)
replaces the cache entry.
"""

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.logging import init_logger
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


class NeuronKVBlockStager:
    """Stage selected KV blocks between Neuron memory and CPU paged tensors.

    One instance backs one connector and is reused for every chunk, which is
    what lets the staging buffers persist across transfers. It is not safe to
    share an instance across threads: the buffers are mutated in place.
    """

    def __init__(self) -> None:
        """Create a stager with empty staging-buffer caches."""
        self._store_stage: dict[tuple[object, ...], torch.Tensor] = {}
        self._retrieve_stage: dict[tuple[object, ...], torch.Tensor] = {}

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
        dim, indices = self._selection_indices(
            selected_blocks, engine_kv_format, block_size
        )
        runs = self._contiguous_runs(indices)

        # Copies land straight in the host buffer: device-to-host is legal
        # between differently-sized parents, so no device-side gather buffer and
        # no per-run temporary is needed.
        staged_host = self._stage_buffer(
            self._store_stage, layer_tensors, dim, len(indices)
        )
        self._gather_runs(layer_tensors, staged_host, dim, runs)
        staged_layers = [staged_host[index] for index in range(len(layer_tensors))]

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
        skip_prefix_n_tokens: int,
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
        :param skip_prefix_n_tokens: Number of leading tokens of this chunk that
            vLLM already holds via prefix caching and must not be written. Their
            blocks can be shared with other running requests, so writing them
            races with live readers. Required rather than defaulted: it was
            silently omitted here once already, and nothing failed until loads
            started running.
        :raises ValueError: If ``key_value`` is not a CPU tensor, or if
            ``skip_prefix_n_tokens`` is negative.
        """
        if key_value.device.type != "cpu":
            raise ValueError("Neuron staging requires a CPU source tensor")
        if skip_prefix_n_tokens < 0:
            raise ValueError(
                f"skip_prefix_n_tokens must be non-negative, got {skip_prefix_n_tokens}"
            )
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
        dim, indices = self._selection_indices(
            selected_blocks, engine_kv_format, block_size
        )
        runs = self._contiguous_runs(indices)

        staged_host = self._stage_buffer(
            self._retrieve_stage, layer_tensors, dim, len(indices)
        )

        # Transfers are block-granular but the unpack below fills only the token
        # slots it is asked to. Any staged position it leaves alone would be
        # scattered back as-is, overwriting live device KV with whatever the
        # buffer last held. Seeding the buffer from the device first makes those
        # positions write back what is already there. Two cases leave gaps:
        #
        #  * A partially covered block -- the tail block of any request whose
        #    length is not a multiple of block_size. Its remaining slots hold KV
        #    for tokens vLLM has not generated yet.
        #  * A non-zero skip_prefix_n_tokens. Those leading slots are already
        #    populated by vLLM's prefix cache and their blocks may be shared with
        #    other running requests.
        #
        # The extra host crossing is skipped when neither applies, which is the
        # common case.
        if skip_prefix_n_tokens > 0 or self._has_partial_block(
            slot_mapping, block_size
        ):
            self._gather_runs(layer_tensors, staged_host, dim, runs)

        staged_layers = [staged_host[index] for index in range(len(layer_tensors))]
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
            skip_prefix_n_tokens=skip_prefix_n_tokens,
        )

        self._scatter_runs(layer_tensors, staged_host, dim, runs)

    @staticmethod
    def _has_partial_block(slot_mapping: torch.Tensor, block_size: int) -> bool:
        """Report whether any selected block is only partially covered.

        A block is partially covered when the slot mapping names fewer than
        ``block_size`` of its slots, which happens for the tail block of any
        request whose token count is not a multiple of ``block_size``.

        :param slot_mapping: Token-to-slot mapping for the request; negative
            entries are unused and ignored.
        :param block_size: Number of token slots per KV block.
        :returns: True if at least one selected block is partially covered.
        """
        slots = slot_mapping.to(dtype=torch.long, device="cpu")
        valid = slots[slots >= 0]
        if valid.numel() == 0:
            return False
        blocks = torch.div(valid, block_size, rounding_mode="floor")
        _, counts = torch.unique(blocks, return_counts=True)
        return bool((counts < block_size).any())

    @staticmethod
    def _gather_runs(
        layer_tensors: list[torch.Tensor],
        staged: torch.Tensor,
        dim: int,
        runs: list[tuple[int, int]],
    ) -> None:
        """Copy the selected runs of every layer into the staging buffer.

        :param layer_tensors: Per-layer paged KV tensors to read from.
        :param staged: All-layer staging buffer to fill, indexed by layer on its
            leading dimension.
        :param dim: Transfer axis within a single layer's shape.
        :param runs: ``(start, length)`` pairs along ``dim``, in staging order.
        """
        for layer_index, layer_tensor in enumerate(layer_tensors):
            offset = 0
            for start, length in runs:
                for source, destination in NeuronKVBlockStager._run_slice_pairs(
                    layer_tensor, staged[layer_index], dim, start, offset, length
                ):
                    destination.copy_(source)
                offset += length

    @staticmethod
    def _scatter_runs(
        layer_tensors: list[torch.Tensor],
        staged: torch.Tensor,
        dim: int,
        runs: list[tuple[int, int]],
    ) -> None:
        """Copy the staging buffer back into the selected runs of every layer.

        :param layer_tensors: Per-layer paged KV tensors to write into, updated
            in place.
        :param staged: All-layer staging buffer to read from, indexed by layer on
            its leading dimension.
        :param dim: Transfer axis within a single layer's shape.
        :param runs: ``(start, length)`` pairs along ``dim``, in staging order.
        """
        for layer_index, layer_tensor in enumerate(layer_tensors):
            offset = 0
            for start, length in runs:
                for destination, source in NeuronKVBlockStager._run_slice_pairs(
                    layer_tensor, staged[layer_index], dim, start, offset, length
                ):
                    destination.copy_(source)
                offset += length

    @staticmethod
    def _stage_buffer(
        cache: dict[tuple[object, ...], torch.Tensor],
        layer_tensors: list[torch.Tensor],
        dim: int,
        selected_count: int,
    ) -> torch.Tensor:
        """Return a reusable all-layer host staging buffer, allocated on demand.

        The buffer holds every layer's selected region in one contiguous host
        allocation, shaped ``(num_layers, *layer_shape)`` with the transfer axis
        narrowed to ``selected_count``. It is always on CPU: the paged KV tensor
        is the only device-side participant, because a device-to-device copy
        between differently-sized parents is rejected by the runtime.

        Every position is written before it is read -- by the unpack, by the
        gather, or by the seeding gather for partially covered regions -- so
        returning a buffer still holding a previous chunk's data is safe, and the
        allocation is skipped for every chunk of the same geometry.

        :param cache: Per-direction buffer cache to look in and populate.
        :param layer_tensors: Per-layer paged KV tensors, used for dtype and the
            non-transfer dimensions.
        :param dim: Transfer axis within a single layer's shape.
        :param selected_count: Extent of the transfer axis in the staged buffer.
        :returns: A contiguous CPU tensor of shape ``(num_layers, *layer_shape)``
            with ``layer_shape[dim] == selected_count``.
        """
        layer_shape = list(layer_tensors[0].shape)
        layer_shape[dim] = selected_count
        shape = (len(layer_tensors), *layer_shape)
        dtype = layer_tensors[0].dtype

        key = (dtype, shape)
        buffer = cache.get(key)
        if buffer is None:
            buffer = torch.empty(shape, dtype=dtype, device="cpu")
            cache[key] = buffer
        return buffer

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
