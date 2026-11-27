# SPDX-License-Identifier: Apache-2.0
"""Private TP1 Spark experiment: retain immutable KV via independent mapping leases."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socketserver
import struct
import threading
import time

from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.common.token_codec import pack_token_ids


def canonical_planes(tensor, format_name):
    """Metadata-only [K/V, block, token, head, feature] view of Qwen3 dense KV."""
    name = getattr(format_name, "name", format_name)
    if str(tensor.dtype) != "torch.bfloat16":
        raise ValueError(f"unsupported retained KV dtype {tensor.dtype}")
    if name in ("NL_X_NB_BS_NH_CS", "NL_X_NB_NH_BS_CS"):
        tensor = tensor.unflatten(-1, (2, 128))
        name = name.replace("_CS", "_TWO_HS")
    axes = {
        "NL_X_NB_BS_NH_TWO_HS": (3, 0, 1, 2, 4),
        "NL_X_NB_NH_BS_TWO_HS": (3, 0, 2, 1, 4),
        "NL_X_TWO_NB_BS_NH_HS": (0, 1, 2, 3, 4),
        "NL_X_TWO_NB_NH_BS_HS": (0, 1, 3, 2, 4),
        "NL_X_NB_TWO_BS_NH_HS": (1, 0, 2, 3, 4),
        "NL_X_NB_TWO_NH_BS_HS": (1, 0, 3, 2, 4),
    }
    if name not in axes:
        raise ValueError(f"unsupported retained KV layout {name}")
    view = tensor.permute(axes[name])
    if view.shape[0] != 2 or tuple(view.shape[2:]) != (16, 8, 128):
        raise ValueError(f"unsupported retained KV shape {tuple(view.shape)}")
    return view


def encode_key(key):
    encoded = dict(
        chunk_hash=key.chunk_hash.hex(),
        model_name=key.model_name,
        kv_rank=key.kv_rank,
        object_group_id=key.object_group_id,
        cache_salt=key.cache_salt,
    )
    if key.identity.tags:
        encoded["tags"] = dict(key.identity.tags)
    return encoded


def decode_key(record):
    return ObjectKey(**{**record, "chunk_hash": bytes.fromhex(record["chunk_hash"])})


@dataclass
class RetainedEntry:
    key: ObjectKey
    index: int
    generation: int
    blocks: tuple
    leases: list
    views: list
    layers: list
    nbytes: int = 36 * 1024 * 1024

    def planes(self, layer):
        return self.views[layer]

    def descriptor(self):
        return dict(
            key=encode_key(self.key),
            chunk_index=self.index,
            generation=self.generation,
            block_ids=self.blocks,
            nbytes=self.nbytes,
            layers=self.layers,
            allocation_ids=list(dict.fromkeys(x["creation_id"] for x in self.layers)),
        )


def hash_entry(entry):
    """Diagnostic only. Hash original bytes in layer/plane/logical-token/head order."""
    import torch

    digest = hashlib.sha256()
    size = 0
    for layer in range(len(entry.views)):
        planes = entry.planes(layer)
        for plane in planes.unbind(0):
            data = plane[list(entry.blocks)].contiguous().view(torch.uint8).cpu()
            payload = data.numpy().tobytes()
            digest.update(payload)
            size += len(payload)
    if size != entry.nbytes:
        raise ValueError(f"retained hash byte count {size} != {entry.nbytes}")
    return dict(
        key=encode_key(entry.key),
        chunk_index=entry.index,
        sha256=digest.hexdigest(),
        nbytes=size,
    )


class _Control(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class _Request(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            command = json.loads(self.rfile.readline(32768))
            result = self.server.store.control(command)
            reply = dict(ok=True, result=result)
        except Exception as exc:
            reply = dict(ok=False, error=repr(exc))
        self.wfile.write(json.dumps(reply).encode() + b"\n")


class RetainedStore:
    def __init__(self, folder, salt, tokens, storage, pause_after=None, zero=False):
        if (
            not salt
            or len(tokens) != 4096
            or hashlib.sha256(struct.pack("<4096I", *tokens)).hexdigest()
            != "0fd976c40e819a2dad3e4cef45a9714d563cd649f1f9977caafab95f0ff1dcdd"
        ):
            raise ValueError(
                "retained store requires the exact approved fixture and a fresh salt"
            )
        if pause_after is not None and not 0 <= pause_after < 16:
            raise ValueError("invalid retained publication checkpoint")
        self.folder = Path(folder)
        self.folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.socket_path = self.folder / "daemon.sock"
        self.salt = salt
        self.tokens = pack_token_ids(tokens)
        self.pause_after = pause_after
        self.storage = storage
        self.zero = zero
        self.cv = threading.Condition()
        self.entries = {}
        self.chunks = {}
        self.readers = Counter()
        self.dropping = set()
        self.active = {}
        self.paused = False
        self.source_context = None
        self.sealed = False
        self.pid = os.getpid()
        self.start_ticks = int(
            Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
        )
        self.log_lock = threading.Lock()
        fd = os.open(
            self.folder / "daemon.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        self.log = os.fdopen(fd, "w", buffering=1)
        self.server = _Control(str(self.socket_path), _Request)
        os.chmod(self.socket_path, 0o600)
        self.server.store = self
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.event("open", salt=salt, pause_after=pause_after, zero=zero)

    def event(self, event, **fields):
        with self.log_lock:
            self.log.write(
                json.dumps(
                    dict(
                        event=event,
                        mono_ns=time.monotonic_ns(),
                        pid=self.pid,
                        start_ticks=self.start_ticks,
                        **fields,
                    )
                )
                + "\n"
            )

    def register(self, ctx, instance):
        with self.cv:
            self.active[id(ctx)] = instance
            self.event("registered", instance=instance, context=id(ctx))

    def retire(self, ctx):
        # Serializes with each per-key commit, but a paused writer releases cv.
        with self.cv:
            instance = self.active.pop(id(ctx), None)
            if self.source_context == id(ctx):
                self.sealed = True
            self.event("registration_closing", instance=instance, context=id(ctx))
            self.cv.notify_all()
            return instance

    def matches_salt(self, salt):
        return salt == self.salt

    def contains(self, key):
        with self.cv:
            return key in self.entries and key not in self.dropping

    @contextmanager
    def acquire(self, key):
        with self.cv:
            if not self.contains(key):
                raise KeyError(key)
            self.readers[key] += 1
            entry = self.entries[key]
            self.event("read_acquire", key=encode_key(key), readers=self.readers[key])
        try:
            yield entry
        finally:
            with self.cv:
                self.readers[key] -= 1
                if not self.readers[key]:
                    del self.readers[key]
                self.event(
                    "read_release", key=encode_key(key), readers=self.readers[key]
                )
                self.cv.notify_all()

    def release(self, entry):
        entry.views.clear()
        while entry.leases:
            lease = entry.leases[-1]
            identity = dict(creation_id=lease.creation_id, export_id=lease.export_id)
            lease.close()
            entry.leases.pop()
            self.event("lease_release_returned", key=encode_key(entry.key), **identity)

    def capture(self, key, index, instance, ctx, blocks):
        wrappers = ctx._kv_wrappers
        tensors = ctx.kv_tensors
        if len(wrappers) != 36 or len(tensors) != 36 or len(set(blocks)) != 16:
            raise ValueError("expected 36 registered layers and 16 distinct blocks")
        entry = RetainedEntry(key, index, instance, tuple(blocks), [], [], [])
        try:
            for layer, (wrapper, normalized) in enumerate(
                zip(wrappers, tensors, strict=True)
            ):
                lease = wrapper.acquire_mapping_lease()
                entry.leases.append(lease)
                raw = lease.tensor
                if (
                    raw.data_ptr() != normalized.data_ptr()
                    or raw.dtype != normalized.dtype
                ):
                    raise ValueError(
                        "independent lease does not alias registered layer origin"
                    )
                normalized_alias = raw.as_strided(normalized.shape, normalized.stride())
                planes = canonical_planes(normalized_alias, ctx.get_engine_kv_format(0))
                if min(blocks) <= 0 or max(blocks) >= planes.size(1):
                    raise ValueError("invalid source block")
                entry.views.append(planes)
                meta = dict(
                    layer=layer,
                    creation_id=lease.creation_id,
                    export_id=lease.export_id,
                    allocation_bytes=lease.alloc_size,
                    tensor_offset_bytes=lease.alloc_offset,
                    shape=list(planes.shape),
                    strides=list(planes.stride()),
                    dtype=str(planes.dtype),
                )
                entry.layers.append(meta)
                self.event("lease_acquired", key=encode_key(key), **meta)
            return entry
        except BaseException:
            self.release(entry)
            raise

    def publish(
        self, instance, ctx, backend, handle, request, keys, groups, chunk_size
    ):
        mask = [False] * len(keys)
        with self.cv:
            if self.active.get(id(ctx)) != instance:
                return b"", False, mask
            if (
                chunk_size != 256
                or len(groups) != 1
                or request.world_size != 1
                or request.worker_id != 0
                or request.start % 256
                or request.end % 256
                or not 0 <= request.start < request.end <= 4096
                or request.token_bytes[: len(self.tokens)] != self.tokens
                or len(keys) != (request.end - request.start) // 256
                or len(groups[0]) != len(keys) * 16
            ):
                raise ValueError(
                    "target namespace STORE outside approved prefix or geometry"
                )
            if self.source_context is None:
                self.source_context = id(ctx)
            elif self.source_context != id(ctx) and not self.sealed:
                raise ValueError(
                    "publication already belongs to a different registration"
                )
            producer = backend.import_event(handle, ctx.device)
            backend.synchronize_event(producer, ctx.device)
            del producer
            self.event(
                "producer_ready",
                instance=instance,
                request_id=request.request_id,
                start=request.start,
                end=request.end,
            )
        for i, key in enumerate(keys):
            with self.cv:
                index = request.start // 256 + i
                if key in self.entries:
                    if self.entries[key].index != index:
                        raise ValueError("ambiguous retained key")
                    mask[i] = True
                    continue
                if self.zero:
                    continue
                if self.pause_after == len(self.entries) and not self.sealed:
                    self.paused = True
                    self.event("paused", committed=len(self.entries), next_index=index)
                    self.cv.notify_all()
                    self.cv.wait_for(lambda: self.sealed or id(ctx) not in self.active)
                    self.paused = False
                if self.sealed or self.active.get(id(ctx)) != instance:
                    break
                if index != len(self.chunks) or len(self.entries) >= 16:
                    raise ValueError(
                        "only contiguous leading target chunks can be committed"
                    )
                blocks = groups[0][i * 16 : (i + 1) * 16]
                if any(
                    not set(blocks).isdisjoint(old.blocks)
                    for old in self.entries.values()
                ):
                    raise ValueError(
                        "distinct retained chunks alias a physical token block"
                    )
                entry = self.capture(key, index, instance, ctx, blocks)
                self.entries[key] = entry
                self.chunks[index] = key
                self.event(
                    "commit", request_id=request.request_id, **entry.descriptor()
                )
                mask[i] = True
        with self.cv:
            completion = b""
            if self.active.get(id(ctx)) == instance:
                event = backend.create_event(ctx.device)
                backend.record_event(event, ctx.stream)
                completion = backend.export_event(event, ctx.device)
            self.event(
                "store_ack",
                request_id=request.request_id,
                mask=mask,
                completed_registration=bool(completion),
            )
            return completion, bool(completion) and (all(mask) or self.zero), mask

    def ordinary_payload(self):
        # Only the pinned DRAM implementation is qualified by this PoC.
        resident = staging = 0
        for manager in self.storage.l1_managers():
            with manager._lock:
                resident += sum(
                    state.memory_obj.get_size()
                    for key, state in manager._objects.items()
                    if key.cache_salt == self.salt
                )
                staging += sum(
                    state.memory_obj.get_size()
                    for key, tagged in manager._staging.items()
                    if key.cache_salt == self.salt
                    for state in tagged.values()
                )
        adapters = len(self.storage.l2_adapters())
        return dict(
            l1_resident_bytes=resident,
            l1_staging_bytes=staging,
            l2_adapters=adapters,
            l2_bytes=None if adapters else 0,
        )

    def control(self, command):
        op = command["op"]
        with self.cv:
            if op == "sync":
                import torch

                torch.cuda.synchronize()
                self.event("daemon_device_synchronized")
                return dict(
                    pid=self.pid,
                    gpu=torch.cuda.current_device(),
                    monotonic_ns=time.monotonic_ns(),
                )
            if op == "snapshot":
                return dict(
                    entries=[e.descriptor() for e in self.entries.values()],
                    ordinary_target=self.ordinary_payload(),
                    readers=sum(self.readers.values()),
                    registration_ids=list(self.active.values()),
                    sealed=self.sealed,
                    paused=self.paused,
                    pid=self.pid,
                    start_ticks=self.start_ticks,
                )
            if op == "abort":
                self.sealed = True
                self.cv.notify_all()
                self.event("publication_sealed")
                return True
            keys = (
                [decode_key(command["key"])] if "key" in command else list(self.entries)
            )
            if op == "drop":
                self.sealed = True
                self.cv.notify_all()
                for key in keys:
                    if key not in self.entries:
                        raise KeyError(key)
                    self.dropping.add(key)
                self.cv.wait_for(lambda: all(not self.readers[k] for k in keys))
                for key in keys:
                    entry = self.entries.get(key)
                    if entry is not None:
                        self.release(entry)
                        del self.entries[key], self.chunks[entry.index]
                    self.dropping.discard(key)
                self.event("drop_complete", keys=[encode_key(k) for k in keys])
                return len(keys)
        if op in ("hash", "readback"):
            rows = []
            for key in keys:
                with self.acquire(key) as entry:
                    rows.append(hash_entry(entry))
            return rows
        raise ValueError(f"unknown retained control operation {op}")

    def close(self):
        self.control({"op": "drop"})
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.socket_path.unlink(missing_ok=True)
        self.event("closed")
        self.log.close()


def get_store(ctx):
    return getattr(ctx, "_spark_retained_store", None)


def create_store(ctx):
    folder = os.getenv("LMCACHE_ZC_DIR")
    if not folder:
        return None
    tokens = json.loads(Path(os.environ["LMCACHE_ZC_TOKENS"]).read_text())
    pause = os.getenv("LMCACHE_ZC_PAUSE_AFTER")
    if ctx.storage_manager.l2_adapters():
        raise ValueError("retained Spark mode requires zero L2 adapters")
    store = RetainedStore(
        folder,
        os.environ["LMCACHE_ZC_SALT"],
        tokens,
        ctx.storage_manager,
        int(pause) if pause is not None else None,
        os.getenv("LMCACHE_ZC_ZERO") == "1",
    )
    ctx._spark_retained_store = store
    return store
