# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the NPU host-memory pin backend (no Ascend hardware)."""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import os
import sys
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.npu import NpuDeviceSpec
from lmcache.v1.platform.npu import pin_memory as pin_memory_module
from lmcache.v1.platform.npu.pin_memory import NpuPinMemoryBackend

#: Page size, computed exactly like the module under test so alignment
#: expectations hold on any host (e.g. aarch64 64K-page kernels).
try:
    PAGE = os.sysconf("SC_PAGESIZE")
except (AttributeError, ValueError, OSError):
    PAGE = 4096

#: None of these tests allocate through the shared allocator, and the
#: autouse fixture's 5GB pinned allocation fails on hosts where the
#: lmcache_ascend plugin is installed but no ACL context can be established.
pytestmark = pytest.mark.no_shared_allocator


class _FakeAclRt:
    """``acl.rt`` double that records host-registration calls."""

    def __init__(
        self,
        register_code: int = 0,
        unregister_codes: list[int] | None = None,
    ) -> None:
        """Record calls and replay the configured AscendCL status codes.

        Args:
            register_code: Status code every ``host_register`` call returns.
            unregister_codes: Status codes ``host_unregister`` returns in
                order; the last one repeats once the list is exhausted.
        """
        self.register_code = register_code
        self.unregister_codes = unregister_codes if unregister_codes else [0]
        self.register_calls: list[tuple[int, int, int]] = []
        self.unregister_calls: list[int] = []

    def host_register(self, ptr: int, size: int, flags: int) -> tuple[int, int]:
        """Record and emulate ``acl.rt.host_register``."""
        self.register_calls.append((ptr, size, flags))
        return 0xDEAD0000, self.register_code

    def host_unregister(self, ptr: int) -> int:
        """Record and emulate ``acl.rt.host_unregister``."""
        self.unregister_calls.append(ptr)
        index = min(len(self.unregister_calls), len(self.unregister_codes)) - 1
        return self.unregister_codes[index]


def _install_acl_rt(
    monkeypatch: pytest.MonkeyPatch,
    rt: Any,
) -> None:
    """Expose a fake ``acl`` module for one test (``None`` forces ImportError)."""
    module = None if rt is None else SimpleNamespace(rt=rt)
    monkeypatch.setitem(sys.modules, "acl", module)


def _install_fake_npu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake ``torch.npu`` so context setup works without torch_npu."""
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            set_device=lambda device: None,
        ),
        raising=False,
    )


def _make_backend(
    monkeypatch: pytest.MonkeyPatch,
    rt: Any = "default",
) -> NpuPinMemoryBackend:
    """Build a backend against the fake ``acl``/``torch.npu`` seams."""
    if rt == "default":
        rt = _FakeAclRt()
    _install_acl_rt(monkeypatch, rt)
    _install_fake_npu(monkeypatch)
    return NpuPinMemoryBackend()


def test_device_spec_registers_npu_pin_backend() -> None:
    """The NPU device specification registers the NPU pinning adapter."""
    assert NpuDeviceSpec().pin_memory_backend is NpuPinMemoryBackend


def test_pin_widens_region_to_whole_pages_and_unpin_uses_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration widens to page boundaries and unpin reverses the base."""
    rt = _FakeAclRt()
    backend = _make_backend(monkeypatch, rt)

    # An offset region spanning two pages: the registration must cover
    # exactly the page containing the pointer plus the one it spills into.
    ptr, size = 2 * PAGE + 0xABC, PAGE
    assert backend.pin_memory(ptr, size) is True
    assert rt.register_calls == [(2 * PAGE, 2 * PAGE, 0)]

    assert backend.unpin_memory(ptr) is True
    assert rt.unregister_calls == [2 * PAGE]


def test_unpin_of_never_pinned_pointer_is_noop_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pointer that was never pinned unpins as a no-op success."""
    rt = _FakeAclRt()
    backend = _make_backend(monkeypatch, rt)
    backend.pin_memory(PAGE, PAGE)

    assert backend.unpin_memory(0x99999) is True
    assert rt.unregister_calls == []


def test_failed_unregister_keeps_region_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed unregistration stays undoable instead of being dropped."""
    rt = _FakeAclRt(unregister_codes=[507911, 0])
    backend = _make_backend(monkeypatch, rt)
    backend.pin_memory(PAGE, PAGE)

    assert backend.unpin_memory(PAGE) is False
    assert backend.unpin_memory(PAGE) is True
    assert rt.unregister_calls == [PAGE, PAGE]


def test_pin_returns_false_on_acl_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonzero AscendCL registration status is not reported as success."""
    backend = _make_backend(monkeypatch, _FakeAclRt(register_code=507899))

    assert backend.pin_memory(PAGE, PAGE) is False
    assert backend.unpin_memory(PAGE) is True


def test_pin_fails_without_raising_and_latches_when_npu_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing torch.npu degrades pin to False and the gate to unsupported."""
    backend = _make_backend(monkeypatch)
    monkeypatch.delattr(torch, "npu", raising=False)
    assert backend.is_pin_supported is True

    assert backend.pin_memory(PAGE, PAGE) is False
    assert backend.is_pin_supported is False
    assert backend.pin_memory(PAGE, PAGE) is False


def test_unavailable_npu_latches_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction-time unavailability flips the gate without a pin attempt.

    ``is_pin_supported`` must be accurate before any ``pin_memory`` call,
    because callers such as the ``use_lazy`` auto-disable guard consult it
    ahead of the first registration.
    """
    rt = _FakeAclRt()
    _install_acl_rt(monkeypatch, rt)
    monkeypatch.delattr(torch, "npu", raising=False)

    backend = NpuPinMemoryBackend()

    assert backend.is_pin_supported is False
    assert backend.pin_memory(PAGE, PAGE) is False
    assert rt.register_calls == []


def test_unavailable_npu_skips_library_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction never probes for ACL bindings without ``torch.npu``.

    No ACL context can be established without ``torch.npu``, so library
    discovery would be dead work -- and its ``ctypes`` fallback forks
    ``ldconfig`` on every Ascend-less construction.
    """
    load_acl_rt = MagicMock(return_value=None)
    load_libascendcl = MagicMock(return_value=None)
    monkeypatch.setattr(pin_memory_module, "_load_acl_rt", load_acl_rt)
    monkeypatch.setattr(pin_memory_module, "_load_libascendcl", load_libascendcl)
    monkeypatch.delattr(torch, "npu", raising=False)

    backend = NpuPinMemoryBackend()

    assert backend.is_pin_supported is False
    load_acl_rt.assert_not_called()
    load_libascendcl.assert_not_called()


def test_unpin_refuses_after_context_failure_latch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A context-failure latch stops unpin from touching AscendCL.

    The latch clears the runtime binding, so ``unpin_memory`` for an
    earlier registration returns False instead of issuing an unregister
    against the broken runtime; the page-lock is abandoned with it.
    """
    rt = _FakeAclRt()
    backend = _make_backend(monkeypatch, rt)
    assert backend.pin_memory(PAGE, PAGE) is True

    def broken_set_device(device: int) -> None:
        raise RuntimeError("CANN broken")

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            set_device=broken_set_device,
        ),
    )

    # A thread that has never pinned runs the full context-setup path.
    pin_results: list[bool] = []

    def pin_on_fresh_thread() -> None:
        pin_results.append(backend.pin_memory(2 * PAGE, PAGE))

    thread = threading.Thread(target=pin_on_fresh_thread)
    thread.start()
    thread.join()

    assert pin_results == [False]
    assert backend.unpin_memory(PAGE) is False
    assert rt.unregister_calls == []


def test_backend_without_acl_or_lib_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No vendor module and no installable library means no pinning."""
    _install_acl_rt(monkeypatch, None)
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.setattr(pin_memory_module.glob, "glob", lambda pattern: [])
    monkeypatch.setattr(
        pin_memory_module.ctypes.util, "find_library", lambda name: None
    )

    backend = NpuPinMemoryBackend()

    assert backend.is_pin_supported is False
    assert backend.pin_memory(PAGE, PAGE) is False
    assert backend.unpin_memory(PAGE) is False


def test_ctypes_fallback_used_when_acl_module_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the ``acl`` module the ctypes path honors $ASCEND_HOME_PATH."""
    _install_acl_rt(monkeypatch, None)
    _install_fake_npu(monkeypatch)
    monkeypatch.setenv("ASCEND_HOME_PATH", "/opt/ascend/ascend-toolkit")

    register = MagicMock(return_value=0)
    unregister = MagicMock(return_value=0)
    dlopen_paths: list[str] = []

    def fake_cdll(path: str) -> Any:
        dlopen_paths.append(path)
        return SimpleNamespace(
            aclrtHostRegister=register, aclrtHostUnregister=unregister
        )

    monkeypatch.setattr(pin_memory_module.ctypes, "CDLL", fake_cdll)

    backend = NpuPinMemoryBackend()
    assert backend.is_pin_supported is True
    assert dlopen_paths == ["/opt/ascend/ascend-toolkit/lib64/libascendcl.so"]

    assert backend.pin_memory(PAGE, PAGE) is True
    base_arg = register.call_args.args[0]
    size_arg = register.call_args.args[1]
    assert base_arg.value == PAGE
    assert size_arg.value == PAGE

    assert backend.unpin_memory(PAGE) is True
    assert unregister.call_args.args[0].value == PAGE


@pytest.mark.npu
def test_real_npu_runtime_registers_and_unregisters() -> None:
    """Optionally smoke-test one real unaligned registration lifecycle."""
    pytest.importorskip("torch_npu")
    backend = NpuPinMemoryBackend()
    if not backend.is_pin_supported:
        pytest.skip("AscendCL host-registration binding is unavailable")

    size = 1 << 20
    buffer = torch.empty(size + PAGE, dtype=torch.uint8)
    ptr = buffer.data_ptr() + 1  # exercise the page-widening path
    assert backend.pin_memory(ptr, size) is True
    assert backend.unpin_memory(ptr) is True
