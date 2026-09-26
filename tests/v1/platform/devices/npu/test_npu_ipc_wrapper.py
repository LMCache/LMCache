# SPDX-License-Identifier: Apache-2.0
"""Coverage for the plane-aggregating NPU IPC wrapper.

Device-independent logic runs on CPU; the sharing/reconstruction
mechanics (``_share_npu_`` / ``_new_shared_npu``) are exercised by a
spawn round-trip test gated on real Ascend hardware.
"""

# Standard
import multiprocessing as mp
import pickle

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import (
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.devices.npu import NpuDeviceSpec
from lmcache.v1.platform.devices.npu.ipc_wrapper import NpuIPCWrapper, PlaneRecord

pytestmark = [
    pytest.mark.npu,
    pytest.mark.no_shared_allocator,
]

requires_npu = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="Ascend NPU hardware is required",
)


def _handbuilt_wrapper(
    records: tuple[PlaneRecord, ...], bare: bool = False
) -> NpuIPCWrapper:
    """Build a wrapper without touching NPU storage sharing."""
    wrapper = NpuIPCWrapper.__new__(NpuIPCWrapper)
    wrapper._plane_records = records
    wrapper._bare = bare
    wrapper.device_uuid = "npu-test-0"
    return wrapper


def _patch_cpu_share(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``wrap``/``to_tensor`` runnable on CPU tensors."""

    class _FakeStorage:
        """Stand-in for an NPU tensor's untyped storage."""

        def __init__(self, tensor: torch.Tensor) -> None:
            self._tensor = tensor

        def _share_npu_(self) -> tuple:  # noqa: N802 (torch's naming)
            return (b"npu", self._tensor.data_ptr(), None, None, None)

    monkeypatch.setattr(
        torch.Tensor, "untyped_storage", lambda self: _FakeStorage(self)
    )
    monkeypatch.setattr(
        NpuIPCWrapper, "_get_device_uuid", staticmethod(lambda idx: "npu-test-0")
    )

    def _fake_index(self: NpuIPCWrapper, device_uuid: str) -> str:
        return "cpu"

    monkeypatch.setattr(NpuIPCWrapper, "_get_device_index_from_uuid", _fake_index)

    def _fake_shared_npu(device_index: str, *rest: object) -> torch.UntypedStorage:
        return torch.UntypedStorage(4096)

    monkeypatch.setattr(
        torch.UntypedStorage,
        "_new_shared_npu",
        staticmethod(_fake_shared_npu),
        raising=False,
    )


def _record(marker: int) -> PlaneRecord:
    handle = (b"npu", marker, None, None, None)
    return (handle, torch.float16, (7, 3, 1, 4), (12, 4, 4, 1), 0)


def test_device_spec_binds_npu_ipc_wrapper() -> None:
    """The NPU spec exposes the wrapper through the platform registry."""
    assert NpuDeviceSpec().ipc_wrapper_cls is NpuIPCWrapper
    assert NpuIPCWrapper.device_type == "npu"
    assert issubclass(NpuIPCWrapper, DeviceIPCWrapper)


@pytest.mark.parametrize("n_planes", [1, 2, 3], ids=["bare", "kv", "dsa"])
def test_wrap_classmethod_builds_instance_per_plane(
    n_planes: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``wrap`` keeps one record per plane, in registration order."""
    planes = tuple(torch.zeros(2, 3) for _ in range(n_planes))
    shared_ptrs: list[int] = []

    class _FakeStorage:
        """Stand-in for an NPU tensor's untyped storage."""

        def __init__(self, tensor: torch.Tensor) -> None:
            self._tensor = tensor

        def _share_npu_(self) -> tuple:  # noqa: N802 (torch's naming)
            shared_ptrs.append(self._tensor.data_ptr())
            return (b"npu", shared_ptrs[-1], None, None, None)

    monkeypatch.setattr(
        torch.Tensor, "untyped_storage", lambda self: _FakeStorage(self)
    )
    monkeypatch.setattr(
        NpuIPCWrapper, "_get_device_uuid", staticmethod(lambda idx: "npu-test-0")
    )

    value: torch.Tensor | tuple[torch.Tensor, ...] = (
        planes[0] if n_planes == 1 else planes
    )
    wrapper = NpuIPCWrapper.wrap(value)

    assert wrapper.device_uuid == "npu-test-0"
    assert shared_ptrs == [plane.data_ptr() for plane in planes]
    records = wrapper._plane_records  # noqa: SLF001 (arity under test)
    assert len(records) == n_planes
    assert [r[2] for r in records] == [(2, 3)] * n_planes
    assert [r[1] for r in records] == [torch.float32] * n_planes


def test_wrap_rejects_empty_plane_sequence() -> None:
    with pytest.raises(ValueError, match="at least one plane"):
        NpuIPCWrapper.wrap(())


def test_equality_compares_plane_records() -> None:
    a = _handbuilt_wrapper((_record(1), _record(2)))
    same = _handbuilt_wrapper((_record(1), _record(2)))
    other = _handbuilt_wrapper((_record(1), _record(3)))
    different_uuid = _handbuilt_wrapper((_record(1), _record(2)))
    different_uuid.device_uuid = "npu-test-1"

    assert a == same
    assert a != other
    assert a != different_uuid
    assert a != "not-a-wrapper"
    assert hash(a) == hash(same)


def test_pickle_roundtrip_preserves_plane_records() -> None:
    wrapper = _handbuilt_wrapper((_record(1), _record(2)))

    restored = pickle.loads(pickle.dumps(wrapper))

    assert isinstance(restored, NpuIPCWrapper)
    assert restored == wrapper
    assert restored.device_uuid == "npu-test-0"


def test_record_count_encodes_reconstruction_arity() -> None:
    """A wrapper built from N planes carries N records."""
    single = _handbuilt_wrapper((_record(1),))
    multi = _handbuilt_wrapper((_record(1), _record(2)))

    assert len(single._plane_records) == 1  # noqa: SLF001
    assert len(multi._plane_records) == 2  # noqa: SLF001


def test_wrap_records_registration_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare and 1-tuple registrations share records but keep distinct forms."""
    _patch_cpu_share(monkeypatch)
    tensor = torch.zeros(2, 3)

    bare = NpuIPCWrapper.wrap(tensor)
    tupled = NpuIPCWrapper.wrap((tensor,))

    assert bare._plane_records == tupled._plane_records  # noqa: SLF001
    assert bare._bare is True  # noqa: SLF001
    assert tupled._bare is False  # noqa: SLF001


def test_to_tensor_restores_registration_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``to_tensor`` restores the registered form: bare stays bare, a
    1-tuple stays a 1-tuple."""
    _patch_cpu_share(monkeypatch)
    tensor = torch.zeros(2, 3)

    out_bare = NpuIPCWrapper.wrap(tensor).to_tensor()
    out_tupled = NpuIPCWrapper.wrap((tensor,)).to_tensor()

    assert isinstance(out_bare, torch.Tensor)
    assert isinstance(out_tupled, tuple)
    assert len(out_tupled) == 1
    assert isinstance(out_tupled[0], torch.Tensor)
    assert tuple(out_bare.shape) == (2, 3)
    assert tuple(out_tupled[0].shape) == (2, 3)


def test_equality_distinguishes_registration_form() -> None:
    """Identical records restoring different forms must not compare equal."""
    bare = _handbuilt_wrapper((_record(1),), bare=True)
    tupled = _handbuilt_wrapper((_record(1),))
    same_bare = _handbuilt_wrapper((_record(1),), bare=True)

    assert bare != tupled
    assert bare == same_bare
    assert hash(bare) != hash(tupled)


def test_pickle_roundtrip_preserves_registration_form() -> None:
    wrapper = _handbuilt_wrapper((_record(1),), bare=False)

    restored = pickle.loads(pickle.dumps(wrapper))

    assert restored == wrapper
    assert restored._bare is False  # noqa: SLF001


def _spawn_worker(encoded: bytes, result_queue: mp.Queue) -> None:
    """Decode the wrappers, report restored forms and plane checksums."""
    try:
        torch.npu.init()
        decoder = get_customized_decoder(type=list[NpuIPCWrapper])
        checksums = []
        forms = []
        for wrapper in decoder.decode(encoded):
            value = wrapper.to_tensor()
            forms.append(isinstance(value, torch.Tensor))
            planes = (value,) if isinstance(value, torch.Tensor) else value
            checksums.append(float(sum(p.sum().cpu().item() for p in planes)))
            for plane in planes:
                plane.add_(1)
        result_queue.put(("success", checksums, forms))
    except Exception as e:  # pragma: no cover - reported to the parent
        result_queue.put(("error", str(e), None))


@requires_npu
def test_multiprocess_roundtrip_preserves_registration_form() -> None:
    """Real IPC round-trip keeps the registered form: bare stays bare, a
    1-tuple stays a 1-tuple."""
    # Third Party
    import torch_npu  # noqa: F401

    ctx = mp.get_context("spawn")

    bare = torch.full((2, 3, 1, 4), 1.0, dtype=torch.float32, device="npu")
    single = torch.full((2, 3, 1, 4), 2.0, dtype=torch.float32, device="npu")
    # Read the expected sums before spawning: the worker adds 1 to every
    # plane through the shared IPC memory.
    bare_sum = float(bare.sum().cpu().item())
    single_sum = float(single.sum().cpu().item())
    wrappers = [NpuIPCWrapper.wrap(bare), NpuIPCWrapper.wrap((single,))]

    encoded = get_customized_encoder(type=list[NpuIPCWrapper]).encode(wrappers)
    result_queue = ctx.Queue()
    process = ctx.Process(target=_spawn_worker, args=(encoded, result_queue))
    process.start()
    process.join(timeout=30)

    assert not process.is_alive(), "worker timed out"
    assert process.exitcode == 0, f"worker exit code {process.exitcode}"
    assert not result_queue.empty(), "no result from worker"
    status, checksums, forms = result_queue.get()

    assert status == "success", f"worker error: {checksums}"
    assert forms == [True, False], f"restored forms mismatch: {forms}"
    assert abs(checksums[0] - bare_sum) < 1e-5
    assert abs(checksums[1] - single_sum) < 1e-5
