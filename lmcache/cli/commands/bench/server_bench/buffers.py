# SPDX-License-Identifier: Apache-2.0
"""Worker-local KV allocation and physical block access for ServerBench."""

# Standard
from collections.abc import Iterator, Sequence
import hashlib

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo
from lmcache.v1.multiprocess.group_view import EngineGroupInfo


class KVCacheBuffers:
    """Own worker-local KV tensors; keep them alive until transfers close."""

    def __init__(
        self,
        groups: Sequence[KVLayerGroupInfo],
        device: str | torch.device | None = None,
        *,
        block_size: int | None = None,
    ) -> None:
        """Allocate client-validated groups, optionally overriding token block size."""
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.engine_group_infos: list[EngineGroupInfo] = []
        self.block_size: int = (
            groups[0].shape_desc.bs if block_size is None else block_size
        )
        self._block_dims: dict[str, int] = {}
        torch.random.manual_seed(42)
        dev = (
            torch.device(device)
            if device
            else torch.device(torch_device_type, torch_dev.current_device())
        )
        for group_index, group in enumerate(groups):
            sd = group.shape_desc
            shape: tuple[int, ...]
            if sd.kv_size == 1:
                shape = (sd.nb, sd.bs, sd.nh * sd.hs)
                block_dim = 0
            else:
                shape = (sd.kv_size, sd.nb, sd.bs, sd.nh, sd.hs)
                block_dim = 1
            self.engine_group_infos.append(
                EngineGroupInfo(
                    engine_group_id=group_index,
                    layer_indices=tuple(group.layer_indices),
                    tokens_per_block=self.block_size,
                )
            )
            for _ in range(sd.nl):
                if group.dtype.is_floating_point:
                    tensor = torch.randn(shape, dtype=group.dtype, device=dev)
                else:
                    bounds = torch.iinfo(group.dtype)
                    tensor = torch.randint(
                        bounds.min, bounds.max + 1, shape, dtype=group.dtype, device=dev
                    )
                name = f"layer.{len(self.kv_caches)}"
                self.kv_caches[name] = tensor
                self._block_dims[name] = block_dim

    def block_views(self, block_offset: int, num_blocks: int) -> Iterator[torch.Tensor]:
        """Yield writable block-range views in layer order."""
        for name, tensor in self.kv_caches.items():
            yield tensor.narrow(self._block_dims[name], block_offset, num_blocks)

    def zero_blocks(self, block_offset: int, num_blocks: int) -> None:
        """Zero the selected block range in every layer."""
        for view in self.block_views(block_offset, num_blocks):
            view.zero_()

    def checksums(
        self, block_offset: int, num_blocks: int, chunk_size: int
    ) -> list[str]:
        """Hash selected CPU blocks per chunk across layers.

        ``chunk_size`` counts tokens; incomplete trailing chunks are ignored.
        """
        if chunk_size % self.block_size != 0:
            raise ValueError(
                "chunk_size %d must be a multiple of block_size %d"
                % (chunk_size, self.block_size)
            )
        blocks_per_chunk = chunk_size // self.block_size
        checksums: list[str] = []
        for chunk in range(num_blocks // blocks_per_chunk):
            start = block_offset + chunk * blocks_per_chunk
            digest = hashlib.md5()
            for view in self.block_views(start, blocks_per_chunk):
                digest.update(view.contiguous().view(torch.uint8).numpy().tobytes())
            checksums.append(digest.hexdigest())
        return checksums

    def release(self) -> None:
        """Drop owned references without synchronizing or unregistering."""
        self.kv_caches.clear()
        self._block_dims.clear()
