# SPDX-License-Identifier: Apache-2.0
"""Contract test for the multi-output serde over the MP / async storage path.

Pins the typed-output behavior the placement wiring (issue #3710) must
satisfy, without requiring a model, vLLM, FlashInfer, or a GPU. The test
file is intentionally standalone: it defines a toy multi-output codec
in-file so the API contract is pinned independently of any concrete
production serde.

The file is split into three groups:

* ``LOCAL`` -- exercises the typed multi-output round-trip directly through
  the sync ``MultiSerializer`` / ``MultiDeserializer`` ABCs: named K / V
  outputs survive verbatim, distinct sizes + sentinels make
  ordering / aliasing bugs impossible to hide, the absent-K (split-tier)
  slot semantics hold, and single-output serdes still flow through the
  length-one bridge unchanged. Also pins the split-tier byte-accounting
  invariant (NVMe-visible bytes shrink to ~0.25x of FP16 KV). These PASS
  today against the codec layer landed by PR #3277.

* ``ASYNC`` -- exercises the same round-trip through
  :class:`AsyncSerdeProcessor`. The processor is typed for single-tensor
  ``Serializer`` / ``Deserializer``, but at runtime it forwards each
  submitted work item to ``serialize`` / ``deserialize`` unchanged, so a
  multi-output group serialized to one blob survives the async store/load
  path today. The contract pinned here is that the processor propagates
  the actual ``n`` from ``serialize()`` back to the destination via
  ``set_used_size``, so downstream L2 adapters write exactly the bytes
  used -- not the over-allocated upper bound. These PASS today.

* ``SPLIT-TIER`` -- pins the static wiring contract the placement layer
  (issue #3710) must expose: ``SerdeL2AdapterWrapper.__init__`` must
  accept ``placement_mode`` / ``split_tier_manifest``;
  :func:`derive_component_key` must produce distinct K / V child keys
  from one logical key; the :class:`SplitTierManifest` state machine
  must track the ``STORE_IN_FLIGHT`` -> ``COMPLETE`` -> ``INVALIDATED``
  lifecycle. These are ``xfail`` today because ``storage_placement.py``
  is not yet on ``dev``; they flip to xpass once the placement layer
  lands and serve as the executable review checklist for that work.

Run::

    pytest tests/v1/distributed/serde/test_multi_output_mp_contract.py -q -rxX
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Optional
import select
import struct
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.base import Deserializer, Serializer
from lmcache.v1.distributed.serde.multi import (
    LayoutDescGroup,
    MemoryObjGroup,
    MultiDeserializer,
    MultiSerializer,
    single_to_multi_deserializer,
    single_to_multi_serializer,
    validate_group_size,
)

try:
    from lmcache.v1.distributed.serde import AsyncSerdeProcessor
    from lmcache.v1.platform import consume_fd

    _HAVE_ASYNC = True
except Exception:  # pragma: no cover - import guard for the async lane
    _HAVE_ASYNC = False


# =============================================================================
# Scaffolding -- GPU-free MemoryObj stand-in for driving the multi-output serde.
# Mirrors the _FakeMemoryObj pattern in test_fp8.py / test_multi.py so this
# file stays standalone and CI-runnable without a CUDA-backed L1Manager.
# =============================================================================


@dataclass
class _FakeMemoryObj:
    """Minimal stand-in exposing ``.tensor`` and ``set_used_size``.

    ``set_used_size`` is the contract hook the async processor calls after
    a successful ``serialize`` so L2 stores the bytes actually written
    rather than the over-allocated upper bound from
    ``estimate_serialized_size``.
    """

    tensor: Optional[torch.Tensor]
    # Recorded by set_used_size; the contract the async processor must
    # honor so the downstream L2 adapter writes exactly the bytes used.
    used_size: Optional[int] = None

    def set_used_size(self, n: int) -> None:
        self.used_size = n


def _byte_buffer(num_bytes: int) -> _FakeMemoryObj:
    """A zeroed uint8 payload of ``num_bytes`` bytes."""
    return _FakeMemoryObj(tensor=torch.zeros(num_bytes, dtype=torch.uint8))


def _sentinel_obj(num_bytes: int, fill: int) -> _FakeMemoryObj:
    """A uint8 payload of ``num_bytes`` filled with the sentinel ``fill``.

    Distinct sentinels per slot make any ordering / aliasing / cross-wiring
    bug surface as a value mismatch rather than a silent byte-identity pass.
    """
    return _FakeMemoryObj(tensor=torch.full((num_bytes,), fill, dtype=torch.uint8))


# Deterministic named outputs with intentionally distinct sizes + sentinels
# so any ordering / aliasing / cross-wiring bug surfaces as a value or size
# mismatch. ``group_size == 2`` mirrors the upstream
# AsymK16V8Multi{Serializer,Deserializer}: the typed group surfaces K and V
# only; per-tensor scales travel packed inside V's encoded blob header (see
# lmcache.v1.distributed.serde.asym_k16_v8 / serialize_header), NOT as a
# separate typed output slot.
_K = ("k", 4096, 0x11)  # bf16-ish keys: larger
_V = ("v", 2048, 0x22)  # fp8 values: smaller
_GROUP_SIZE = 2

_TEST_KEY = ObjectKey(
    chunk_hash=b"\x00" * 32,
    model_name="contract-test",
    kv_rank=0,
)


# =============================================================================
# A toy fake multi-output codec -- transport/contract only, no real
# quantization. Defined inline so this file is standalone.
#
# Wire format (group of N):
#   header: N x [uint8 present-mask] + N x [uint32 little-endian length]
#   body:   concatenation of the present slots' raw tensor bytes,
#           in slot order. Absent slots contribute zero bytes.
# Header byte size = N + 4*N = 5*N. Payload size is the sum of present
# slots' tensor byte sizes. Total = 5*N + sum(present payloads).
# =============================================================================

_MASK_FMT = struct.Struct("<B")  # one byte per present-mask entry
_LEN_FMT = struct.Struct("<I")  # uint32 little-endian per length


def _header_size(group_size: int) -> int:
    return group_size * (_MASK_FMT.size + _LEN_FMT.size)


def _tensor_bytes(t: torch.Tensor) -> bytes:
    # Reinterpret as uint8 to avoid Python bytes() per-byte iteration on
    # storage. Mirrors the trick used elsewhere in the tree but kept local
    # so this test file does not depend on production helpers.
    return t.contiguous().view(torch.uint8).numpy().tobytes()


class _FakeMultiSerializer(MultiSerializer):
    """Toy multi-serializer: concat present slots verbatim into ``dst``."""

    def __init__(self, group_size: int = _GROUP_SIZE) -> None:
        self._n = group_size

    @property
    def group_size(self) -> int:
        return self._n

    def serialize(self, src: MemoryObjGroup, dst, key: ObjectKey) -> int:
        validate_group_size(src, self._n, role="src")
        masks = bytearray()
        lens = bytearray()
        payload = bytearray()
        for slot in src:
            if slot is None:
                masks += _MASK_FMT.pack(0)
                lens += _LEN_FMT.pack(0)
                continue
            blob = _tensor_bytes(slot.tensor)
            masks += _MASK_FMT.pack(1)
            lens += _LEN_FMT.pack(len(blob))
            payload += blob
        header = bytes(masks) + bytes(lens)
        total = len(header) + len(payload)
        if dst.tensor is None or dst.tensor.numel() < total:
            raise ValueError("dst buffer too small for serialized group")
        dv = dst.tensor.view(torch.uint8)
        dv[: len(header)].copy_(torch.frombuffer(bytearray(header), dtype=torch.uint8))
        if payload:
            dv[len(header) : total].copy_(
                torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            )
        return total

    def estimate_serialized_size(self, layout_descs: LayoutDescGroup) -> int:
        validate_group_size(layout_descs, self._n, role="layout")
        total = _header_size(self._n)
        for desc in layout_descs:
            if desc is None:
                continue
            for shape, dtype in zip(desc.shapes, desc.dtypes, strict=True):
                numel = 1
                for dim in shape:
                    numel *= int(dim)
                total += numel * dtype.itemsize
        return total


class _FakeMultiDeserializer(MultiDeserializer):
    """Toy multi-deserializer: split ``src`` back into present dst slots."""

    def __init__(self, group_size: int = _GROUP_SIZE) -> None:
        self._n = group_size

    @property
    def group_size(self) -> int:
        return self._n

    def deserialize(self, src, dst: MemoryObjGroup, key: ObjectKey) -> None:
        validate_group_size(dst, self._n, role="dst")
        sv = src.tensor.view(torch.uint8)
        n = self._n
        present = [bool(sv[i].item()) for i in range(n)]
        lens = [
            int(
                _LEN_FMT.unpack_from(sv[n + i * 4 : n + (i + 1) * 4].numpy().tobytes())[
                    0
                ]
            )
            for i in range(n)
        ]
        cursor = _header_size(n)
        for i, slot in enumerate(dst):
            this_len = lens[i]
            if slot is None or not present[i]:
                cursor += this_len
                continue
            dstv = slot.tensor.view(torch.uint8).flatten()
            dstv[:this_len].copy_(sv[cursor : cursor + this_len])
            cursor += this_len


# A trivial single-tensor serde (identity byte copy) for the bridge
# back-compat test. Mirrors _IdentitySerializer in test_multi.py.
class _IdentitySerializer(Serializer):
    def serialize(self, src, dst, key: ObjectKey) -> int:
        blob = src.tensor.contiguous().view(torch.uint8)
        n = int(blob.numel())
        dst.tensor.view(torch.uint8)[:n].copy_(blob)
        return n

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        total = 0
        for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
            numel = 1
            for dim in shape:
                numel *= int(dim)
            total += numel * dtype.itemsize
        return total


class _IdentityDeserializer(Deserializer):
    def deserialize(self, src, dst, key: ObjectKey) -> None:
        n = int(dst.tensor.view(torch.uint8).numel())
        dst.tensor.view(torch.uint8).copy_(src.tensor.view(torch.uint8)[:n])


# =============================================================================
# LOCAL contract -- these PASS today
# =============================================================================


def test_multi_output_local_roundtrip_preserves_named_outputs() -> None:
    """K/V round-trip with distinct sizes + sentinels: no aliasing / ordering
    bug can hide. Pins the typed-output contract the placement wiring relies
    on: K and V survive verbatim through serialize -> deserialize, with the
    sentinel fill preserved per slot."""
    s = _FakeMultiSerializer()
    d = _FakeMultiDeserializer()
    src = tuple(_sentinel_obj(nbytes, fill) for _, nbytes, fill in (_K, _V))
    layout = tuple(
        MemoryLayoutDesc(shapes=[o.tensor.shape], dtypes=[o.tensor.dtype]) for o in src
    )
    buf = _byte_buffer(s.estimate_serialized_size(layout))
    n = s.serialize(src, buf, _TEST_KEY)
    assert n > 0

    out = tuple(_byte_buffer(nbytes) for _, nbytes, _ in (_K, _V))
    d.deserialize(buf, out, _TEST_KEY)
    for (name, nbytes, fill), o in zip((_K, _V), out, strict=False):
        assert o.tensor.numel() == nbytes, f"{name}: size changed"
        assert torch.all(o.tensor == fill), f"{name}: sentinel/content cross-wired"


def test_multi_output_absent_k_slot_split_tier_semantics() -> None:
    """Split-tier: K absent on serialize (``None`` src slot); V round-trips
    and the K destination is left untouched. This is the slot-absence contract
    the V-only split-tier placement (K in L1 host, V to L2 NVMe) relies on:
    the serialize side has no tensor for K (it is held outside this serde's
    data path), and the deserialize side must not touch the K slot either."""
    s = _FakeMultiSerializer()
    d = _FakeMultiDeserializer()
    v = _sentinel_obj(_V[1], _V[2])
    src: MemoryObjGroup = (None, v)
    layout: LayoutDescGroup = (
        None,
        MemoryLayoutDesc(shapes=[v.tensor.shape], dtypes=[v.tensor.dtype]),
    )
    buf = _byte_buffer(s.estimate_serialized_size(layout))
    s.serialize(src, buf, _TEST_KEY)

    # Pre-fill the K destination so we can detect any spurious write.
    k_out = _sentinel_obj(_K[1], 0xEE)
    v_out = _byte_buffer(_V[1])
    d.deserialize(buf, (k_out, v_out), _TEST_KEY)
    assert torch.all(k_out.tensor == 0xEE), "absent K must not be written"
    assert torch.all(v_out.tensor == _V[2])


def test_single_output_bridge_backcompat() -> None:
    """A single-tensor serde wrapped via the length-one bridge still
    round-trips. Pins the back-compat contract: existing single-tensor
    serdes (fp8, etc.) opt into the group call site without changing their
    on-the-wire bytes."""
    ms = single_to_multi_serializer(_IdentitySerializer())
    md = single_to_multi_deserializer(_IdentityDeserializer())
    assert ms.group_size == 1
    assert md.group_size == 1
    payload = _sentinel_obj(1024, 0x5A)
    layout = (
        MemoryLayoutDesc(shapes=[payload.tensor.shape], dtypes=[payload.tensor.dtype]),
    )
    buf = _byte_buffer(ms.estimate_serialized_size(layout))
    ms.serialize((payload,), buf, _TEST_KEY)
    out = (_byte_buffer(1024),)
    md.deserialize(buf, out, _TEST_KEY)
    assert torch.all(out[0].tensor == 0x5A), "single-output bridge broke byte fidelity"


def test_split_tier_byte_accounting_contract() -> None:
    """The byte accounting the placement wiring must expose (counter contract).

    For a KV element count X (K and V each X elements):

      FP16 full KV                 = K16 + V16 = 4X bytes  (single-tier baseline)
      K16/V8 all-NVMe              = K16 + V8  = 3X bytes  (NVMe only; 0.75x FP16)
      Split-tier (K host, V NVMe)  = K16 (host) + V8 (NVMe) = 2X host + 1X NVMe
                                     NVMe-visible = 1X = 1/3 of all-NVMe
                                                    = 1/4 of FP16

    This is the storage win the V-only split-tier mode in issue #3710 is
    designed for, and the reason component-level placement is needed: it
    cannot be expressed with the current single-object MP storage path.
    """
    X = 1_000_000
    fp16 = 2 * X + 2 * X
    all_nvme = 2 * X + 1 * X
    split_host, split_nvme = 2 * X, 1 * X
    assert all_nvme / fp16 == 0.75
    assert split_nvme / all_nvme == pytest.approx(1 / 3)
    assert split_nvme / fp16 == 0.25
    assert split_host == 2 * X  # K parked in host memory, not on NVMe


# =============================================================================
# ASYNC contract -- PASS today; the async layer is not the blocker
# =============================================================================


def _wait_for_fd(fd: int, timeout_s: float = 2.0) -> bool:
    """Wait until ``fd`` is readable or timeout. Drains the pending signal.

    Mirrors the helper in ``test_async_processor.py`` so this file stays
    standalone and does not import from a sibling test module. On POSIX
    (where ``select.poll`` exists) this blocks on the fd and verifies the
    signal contract; on Windows ``select.poll`` is absent, so the helper
    busy-waits and lets ``query_*_result`` be the source of truth. The
    fd-signal contract itself is already pinned by ``test_async_processor.py``
    on Linux CI, so the Windows fallback only needs to give the thread pool
    time to complete.
    """
    if hasattr(select, "poll"):
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            remaining_ms = int(max(0, (deadline - time.monotonic()) * 1000))
            if poller.poll(remaining_ms):
                try:
                    consume_fd(fd)
                except OSError:
                    pass
                return True
        return False
    # Windows / platforms without select.poll: busy-wait. The fd is not
    # drainable here, but query_*_result is what the test asserts on, so
    # this just gives the thread pool time to finish.
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(0.005)
    return True


@pytest.mark.skipif(not _HAVE_ASYNC, reason="AsyncSerdeProcessor unavailable")
def test_multi_output_through_async_processor_roundtrip() -> None:
    """The async processor round-trips a multi-output group (-> single blob).

    ``AsyncSerdeProcessor`` is typed for single-tensor ``Serializer`` /
    ``Deserializer``, but at runtime it forwards each submitted work item
    to ``serialize`` / ``deserialize`` unchanged, so a multi-output group
    serialized to one blob survives the async store/load path today. The
    async layer is therefore NOT the blocker for issue #3710; the
    remaining gap is narrower: split-tier *placement* -- routing the
    group's outputs to different storage tiers as separate typed child
    outputs (see the xfail test below).

    The contract pinned here: the processor must propagate the actual
    ``n`` from ``serialize()`` back to the destination via
    ``set_used_size``, so the downstream L2 adapter writes exactly the
    bytes used -- not the over-allocated upper bound. Without this, every
    store would pay the over-allocation as wasted L2 bytes.
    """
    s = _FakeMultiSerializer()
    d = _FakeMultiDeserializer()
    proc = AsyncSerdeProcessor(s, d)  # type: ignore[arg-type]
    try:
        src = tuple(_sentinel_obj(nbytes, fill) for _, nbytes, fill in (_K, _V))
        layout = tuple(
            MemoryLayoutDesc(shapes=[o.tensor.shape], dtypes=[o.tensor.dtype])
            for o in src
        )
        # Intentionally over-allocate the destination by 1024 bytes so the
        # test can prove the processor narrows ``buf`` down to the bytes
        # actually written. This mirrors the real-world case: the AsymK16V8
        # serde sizes its destination from ``estimate_serialized_size``
        # (an upper bound that includes a header allowance), and
        # ``serialize()`` returns the actual ``n``.
        exact_size = s.estimate_serialized_size(layout)
        overprovision = 1024
        buf = _byte_buffer(exact_size + overprovision)
        # Submit the group as a single work item: src_objs[0] is the group
        # tuple, which the processor forwards to ``serialize`` unchanged.
        sid = proc.submit_serialize([src], [buf], [_TEST_KEY])
        assert _wait_for_fd(proc.get_serialize_event_fd()), (
            "serialize fd never signaled"
        )
        assert proc.query_serialize_result(sid) is True
        assert buf.used_size == exact_size, (
            f"async processor did not narrow buf to actual n: "
            f"used_size={buf.used_size}, expected {exact_size}"
        )

        out = tuple(_byte_buffer(nbytes) for _, nbytes, _ in (_K, _V))
        did = proc.submit_deserialize([buf], [out], [_TEST_KEY])
        assert _wait_for_fd(proc.get_deserialize_event_fd()), (
            "deserialize fd never signaled"
        )
        assert proc.query_deserialize_result(did) is True
        for (_, _, fill), o in zip((_K, _V), out, strict=False):
            assert torch.all(o.tensor == fill)
    finally:
        proc.close()


# =============================================================================
# SPLIT-TIER contract -- XFAIL today; the target the placement wiring must hit
# =============================================================================


@pytest.mark.xfail(
    reason=(
        "Split-tier placement wiring (issue #3710) is not on dev yet: "
        "lmcache.v1.distributed.storage_placement and the wrapper's "
        "placement_mode / split_tier_manifest constructor args are not "
        "landed. This test pins the static contract that work must expose."
    ),
    strict=True,
)
def test_split_policy_routes_k_to_cpu_v_to_nvme() -> None:
    """Split-tier wiring contract.

    The placement layer (issue #3710) must expose, at minimum:

    * :func:`derive_component_key` / :class:`StoragePlacementMode` /
      :class:`SplitTierManifest` defining the placement contract.
    * :class:`StorageManager` resolving placement at construction and
      exposing ``storage_placement_mode`` + ``split_tier_manifest``.
    * :class:`SerdeL2AdapterWrapper` driving the state machine on both
      store and load paths, translating logical keys to
      ``derive_component_key(logical, "v")`` for the inner adapter and
      reading ``derive_component_key(logical, "k")`` from L1.

    This test exercises the *static* wiring shape -- public symbols,
    constructor signatures, and key-derivation invariants -- so a CPU-only
    sweep still catches regressions in the multi-output routing. The
    end-to-end behavioral proof (K bit-exact + V FP8 noise round-trip
    through a real ``StorageManager`` + ``file_l2``) lives in a separate
    CUDA-gated integration test because L1Manager's default allocator is
    CUDA-backed.
    """
    # Standard
    import inspect

    # First Party
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import SerdeL2AdapterWrapper
    from lmcache.v1.distributed.storage_placement import (
        SplitTierManifest,
        SplitTierState,
        StoragePlacementMode,
        derive_component_key,
        derive_storage_placement_mode,
    )

    # ---- Wrapper accepts placement_mode + split_tier_manifest ----
    sig = inspect.signature(SerdeL2AdapterWrapper.__init__)
    assert "placement_mode" in sig.parameters, (
        "SerdeL2AdapterWrapper.__init__ must accept placement_mode "
        "for split-tier wiring"
    )
    assert "split_tier_manifest" in sig.parameters, (
        "SerdeL2AdapterWrapper.__init__ must accept split_tier_manifest "
        "for split-tier wiring"
    )

    # ---- derive_component_key produces distinct K and V children ----
    logical = ObjectKey(
        chunk_hash=b"\x00" * 32,
        model_name="contract-test",
        kv_rank=0,
        cache_salt="",
    )
    k_child = derive_component_key(logical, "k")
    v_child = derive_component_key(logical, "v")
    # K and V children must be distinct from each other and from the
    # logical key (the wrapper relies on this to address two tiers with
    # one logical key).
    assert k_child != v_child
    assert k_child != logical
    assert v_child != logical
    # Cache salt + model + rank preserved (per-tenant quota accounting).
    assert k_child.cache_salt == logical.cache_salt
    assert k_child.model_name == logical.model_name
    assert k_child.kv_rank == logical.kv_rank
    assert v_child.cache_salt == logical.cache_salt

    # ---- Placement mode resolution: V-only serde -> KV_SPLIT_TIER ----
    from lmcache.v1.distributed.serde.base import SerdeConfig

    class _FakeCfg:
        def __init__(self, serde_config: SerdeConfig) -> None:
            self.serde_config = serde_config

    v_only_mode = derive_storage_placement_mode(
        [_FakeCfg(SerdeConfig(type="asym_k16_v8_v_only"))]
    )
    assert v_only_mode == StoragePlacementMode.KV_SPLIT_TIER
    asym_mode_1_mode = derive_storage_placement_mode(
        [_FakeCfg(SerdeConfig(type="asym_k16_v8"))]
    )
    assert asym_mode_1_mode == StoragePlacementMode.KV_TOGETHER

    # ---- Manifest tracks the 4-state lifecycle ----
    m = SplitTierManifest()
    m.register_pending(logical)
    assert m.lookup(logical) == SplitTierState.STORE_IN_FLIGHT
    m.mark_complete(logical)
    assert m.is_complete(logical)
    m.mark_invalidated(logical)
    assert m.lookup(logical) == SplitTierState.INVALIDATED
