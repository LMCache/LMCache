# SPDX-License-Identifier: Apache-2.0
"""Private GB10 TP1 retained-view read path (synchronous diagnostic restore)."""

import hashlib
import json
import os
from pathlib import Path
import time


def selected_store(ctx, salt):
    if (
        not salt
        or salt != os.environ.get("LMCACHE_ZC_SALT")
        or not os.getenv("LMCACHE_ZC_DIR")
    ):
        return None
    from lmcache.v1.multiprocess.retained_views import get_store

    store = get_store(ctx)
    return store if store and store.matches_salt(salt) else None


def record(op, **fields):
    root = Path(os.environ["LMCACHE_ZC_DIR"])
    with (root / "replacement.jsonl").open("a") as out:
        out.write(
            json.dumps(dict(op=op, ns=time.monotonic_ns(), pid=os.getpid(), **fields))
            + "\n"
        )


def restore(ctx, key, instance_id, dst, keys, blocks, skip):
    import torch
    from lmcache.v1.multiprocess.retained_views import canonical_planes, encode_key

    locks = ctx.storage_manager.retained_reads
    held, views = locks.claim(id(ctx.session_manager.get(key.request_id)), keys)
    total = 0
    with held:
        try:
            if skip or len(blocks) != 1 or len(blocks[0]) != 16 * len(views):
                raise ValueError(
                    "replacement requires complete TP1 chunks and zero native hits"
                )
            if len(set(blocks[0])) != len(blocks[0]) or min(blocks[0]) <= 0:
                raise ValueError("destination slots must be distinct and valid")
            if len(dst.kv_tensors) != 36:
                raise ValueError("unexpected destination layer count")
            for i, (obj_key, view) in enumerate(zip(keys, views, strict=True)):
                dest_ids = blocks[0][i * 16 : (i + 1) * 16]
                if (
                    view.key != obj_key
                    or view.generation == instance_id
                    or len(view.blocks) != 16
                ):
                    raise ValueError("invalid retained view identity or generation")
                src_hash, dst_hash = hashlib.sha256(), hashlib.sha256()
                size = 0
                layers = []
                old = torch.tensor(view.blocks, dtype=torch.long, device=dst.device)
                new = torch.tensor(dest_ids, dtype=torch.long, device=dst.device)
                for layer, tensor in enumerate(dst.kv_tensors):
                    source = view.planes(layer)
                    target = canonical_planes(tensor, dst.get_engine_kv_format(0))
                    if (
                        target.untyped_storage().data_ptr()
                        != tensor.untyped_storage().data_ptr()
                    ):
                        raise ValueError(
                            "destination normalization must alias actual worker KV"
                        )
                    holder = dst._kv_wrappers[layer]
                    dest_creation = getattr(holder, "creation_id", None)
                    origin = view.layers[layer]
                    if (
                        dest_creation is not None
                        and dest_creation == origin["creation_id"]
                    ):
                        raise ValueError(
                            "replacement aliases original allocator creation"
                        )
                    layers.append(
                        dict(
                            layer=layer,
                            original_export=origin["export_id"],
                            original_creation=origin["creation_id"],
                            destination_creation=dest_creation,
                            destination_ptr=target.data_ptr(),
                            destination_strides=list(target.stride()),
                            destination_offset=target.storage_offset(),
                        )
                    )
                    for plane in (source, target):
                        if (
                            plane.dtype != torch.bfloat16
                            or plane.ndim != 5
                            or plane.shape[0] != 2
                            or tuple(plane.shape[2:]) != (16, 8, 128)
                        ):
                            raise ValueError(
                                "incompatible original or destination BF16 KV layout"
                            )
                    if (
                        min(view.blocks) < 0
                        or max(view.blocks) >= source.shape[1]
                        or max(dest_ids) >= target.shape[1]
                    ):
                        raise ValueError("source or destination slot exceeds backing")
                    for which in (0, 1):
                        original = source[which].index_select(0, old)
                        target[which].index_copy_(0, new, original)
                        restored = target[which].index_select(0, new)
                        for batch, digest in (
                            (original, src_hash),
                            (restored, dst_hash),
                        ):
                            digest.update(
                                batch.contiguous()
                                .view(torch.uint8)
                                .cpu()
                                .numpy()
                                .tobytes()
                            )
                        size += original.numel() * original.element_size()
                if size != view.nbytes or src_hash.digest() != dst_hash.digest():
                    raise ValueError("pre-inference destination KV hash mismatch")
                total += size
                record(
                    "destination_verified",
                    request=key.request_id,
                    key=encode_key(obj_key),
                    original=view.generation,
                    replacement=instance_id,
                    source_blocks=list(view.blocks),
                    destination_blocks=dest_ids,
                    layers=layers,
                    nbytes=size,
                    sha256=dst_hash.hexdigest(),
                    source_sha256=src_hash.hexdigest(),
                )
        finally:
            dst.stream.synchronize()
    return total
