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
from lmcache.v1.platform import synchronize_device


def _block_bytes(tensor: torch.Tensor, dim: int, block_ids: Sequence[int]) -> bytes:
    """Gather block bytes in logical order."""
    indices = torch.tensor(block_ids, dtype=torch.long, device=tensor.device)
    selected = tensor.index_select(dim, indices)
    if selected.dtype == torch.bfloat16:
        selected = selected.float()
    return selected.contiguous().view(torch.uint8).cpu().numpy().tobytes()


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
                tensor_name = f"tensor.{len(self.kv_caches)}"
                self.kv_caches[tensor_name] = tensor
                self._block_dims[tensor_name] = block_dim

    def block_views(
        self, block_ids_per_group: Sequence[Sequence[int]]
    ) -> Iterator[torch.Tensor]:
        """Yield validated block views in group/tensor/logical-block order."""
        for _, tensor, dim, block_ids in self._iter_group_tensors(block_ids_per_group):
            for block_id in block_ids:
                yield tensor.narrow(dim, block_id, 1)

    def fill_deterministic_kv(
        self,
        block_ids_per_group: Sequence[Sequence[int]],
        token_ids: Sequence[int],
        worker_id: int,
    ) -> None:
        """Fill bound blocks with reproducible pseudorandom KV in [1, 127]."""
        prefix = hashlib.sha256()
        seeds = []
        for start in range(0, len(token_ids) - self.block_size + 1, self.block_size):
            prefix.update(
                str(tuple(token_ids[start : start + self.block_size])).encode()
            )
            seeds.append(prefix.hexdigest())
        for tensor_index, tensor, dim, block_ids in self._iter_group_tensors(
            block_ids_per_group
        ):
            generator = torch.Generator(device=tensor.device)
            for logical_block, block_id in enumerate(block_ids):
                identity = f"{seeds[logical_block]}:{worker_id}:{tensor_index}".encode()
                seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
                generator.manual_seed(seed)
                view = tensor.narrow(dim, block_id, 1)
                view.copy_(
                    torch.randint(
                        1,
                        128,
                        view.shape,
                        dtype=tensor.dtype,
                        device=tensor.device,
                        generator=generator,
                    )
                )

    def set_blocks_to_zero(self, block_ids_per_group: Sequence[Sequence[int]]) -> None:
        """Set bound blocks to zero."""
        for view in self.block_views(block_ids_per_group):
            view.zero_()

    def checksum(self, block_ids_per_group: Sequence[Sequence[int]]) -> str:
        """Hash selected blocks across tensors in logical order into one MD5 digest."""
        digest = hashlib.md5()
        for _, tensor, dim, block_ids in self._iter_group_tensors(block_ids_per_group):
            digest.update(_block_bytes(tensor, dim, block_ids))
        return digest.hexdigest()

    def synchronize(self) -> None:
        """Wait for buffer preparation outside transfer timing."""
        synchronize_device(next(iter(self.kv_caches.values())).device)

    def tensor_checksums(
        self, block_ids_per_group: Sequence[Sequence[int]]
    ) -> dict[int, str]:
        """Hash selected blocks per registered tensor using the server's dtype rules."""
        return {
            index: hashlib.md5(_block_bytes(tensor, dim, block_ids)).hexdigest()
            for index, tensor, dim, block_ids in self._iter_group_tensors(
                block_ids_per_group
            )
        }

    def release(self) -> None:
        """Drop owned references without synchronizing or unregistering."""
        self.kv_caches.clear()
        self._block_dims.clear()

    def _iter_group_tensors(
        self, block_ids_per_group: Sequence[Sequence[int]]
    ) -> Iterator[tuple[int, torch.Tensor, int, Sequence[int]]]:
        """Yield registered tensor indices, tensors, block axes, and bound block IDs."""
        for group, block_ids in zip(
            self.engine_group_infos, block_ids_per_group, strict=True
        ):
            for tensor_index in group.layer_indices:
                tensor_name = f"tensor.{tensor_index}"
                yield (
                    tensor_index,
                    self.kv_caches[tensor_name],
                    self._block_dims[tensor_name],
                    block_ids,
                )
