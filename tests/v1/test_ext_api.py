# SPDX-License-Identifier: Apache-2.0
"""Native extension API: shipped headers and the cuda_ops._ext_api table.

The GPU test builds ``ext_api_probe/probe.cu`` against ``lmcache.get_include()``
alone, the way a plugin transfer module is built, runs object-group
plans through the exported executor and compares the bytes with
``cuda_ops.execute_object_group_transfer``.
"""

# Standard
from pathlib import Path
import ctypes
import importlib.util
import re

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
import lmcache

REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)


def _ext_api_headers() -> tuple[str, ...]:
    path = REPO_ROOT / "setup_extensions" / "ext_api_headers.py"
    spec = importlib.util.spec_from_file_location("_ext_api_headers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EXT_API_HEADERS


def _header_api_version() -> int:
    header = Path(lmcache.get_include()) / "cuda" / "lmcache_ext_api.h"
    match = re.search(r"#define LMCACHE_EXT_API_VERSION (\d+)", header.read_text())
    assert match is not None
    return int(match.group(1))


def test_shipped_headers_are_closed_under_local_includes():
    csrc = REPO_ROOT / "csrc"
    shipped = set(_ext_api_headers())
    for rel in sorted(shipped):
        header = csrc / rel
        for target in _LOCAL_INCLUDE.findall(header.read_text()):
            # The compiler looks next to the including file first, then in
            # the include root.
            for candidate in (header.parent / target, csrc / target):
                if candidate.is_file():
                    resolved = candidate.resolve().relative_to(csrc).as_posix()
                    break
            else:
                pytest.fail(f'{rel}: #include "{target}" not found')
            assert resolved in shipped, (
                f'{rel} includes "{target}" ({resolved}), which is not in '
                "EXT_API_HEADERS, so extensions built from the wheel cannot "
                "compile"
            )


def test_get_include_has_every_shipped_header():
    include = Path(lmcache.get_include())
    missing = [rel for rel in _ext_api_headers() if not (include / rel).is_file()]
    assert not missing, f"missing under {include}: {missing}"


def _cuda_ops():
    return pytest.importorskip(
        "lmcache.cuda_ops", reason="Requires CUDA extension lmcache.cuda_ops"
    )


def test_ext_api_capsule_matches_headers():
    cuda_ops = _cuda_ops()
    capsule = cuda_ops._ext_api
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.restype = ctypes.c_char_p
    get_name.argtypes = [ctypes.py_object]
    assert get_name(capsule) == b"lmcache.cuda_ops._ext_api"

    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    table = get_pointer(capsule, b"lmcache.cuda_ops._ext_api")
    version, size = (ctypes.c_uint32 * 2).from_address(table)
    assert version == _header_api_version()
    # version, size, four layout sizes, then one function pointer.
    assert size >= 6 * 4 + ctypes.sizeof(ctypes.c_void_p)


# ---------------------------------------------------------------------------
# GPU: a separately built extension driving the exported executor
# ---------------------------------------------------------------------------

NL, NB, BS, NH, HS = 4, 64, 16, 8, 128
CHUNK = 256  # tokens per LMCache object = 16 blocks
NUM_OBJECTS = 2
HOST_ALIGNMENT = 1 << 26


@pytest.fixture(scope="module")
def probe():
    _cuda_ops()
    if not (torch_dev.is_available() and torch_device_type == "cuda"):
        pytest.skip("Requires CUDA")
    # Third Party
    from torch.utils import cpp_extension

    if not cpp_extension.is_ninja_available():
        pytest.skip("Requires ninja to build the probe extension")
    return cpp_extension.load(
        name="lmcache_ext_api_probe",
        sources=[str(Path(__file__).parent / "ext_api_probe" / "probe.cu")],
        extra_include_paths=[lmcache.get_include()],
        verbose=False,
    )


def _plan(paged_ptrs, objects, block_ids, staging):
    # First Party
    import lmcache.cuda_ops as cuda_ops
    import lmcache.lmcache_native as lmcache_native

    shape_desc = lmcache_native.PageBufferShapeDesc()
    shape_desc.kv_size = 2
    shape_desc.nl = NL
    shape_desc.nb = NB
    shape_desc.bs = BS
    shape_desc.nh = NH
    shape_desc.hs = HS
    shape_desc.element_size = 2
    spec = lmcache_native.KernelGroupSpec(
        paged_ptrs.data_ptr(),
        [o.data_ptr() for o in objects],
        shape_desc,
        CHUNK,
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
        block_ids.data_ptr(),
        block_ids.numel(),
    )
    launch = cuda_ops.LaunchVar(0, 0, block_ids.numel(), len(objects), 0)
    return [spec], [cuda_ops.BatchStep(staging, [launch])]


def _staging(cuda_ops, objects, host_objects):
    return [
        cuda_ops.StagingCopy(h.data_ptr(), o.data_ptr(), o.nbytes, 0)
        for o, h in zip(objects, host_objects, strict=True)
    ]


@pytest.mark.cuda
def test_probe_extension_matches_core_executor(probe):
    # First Party
    import lmcache.cuda_ops as cuda_ops
    import lmcache.lmcache_native as lmcache_native

    device = torch.device(torch_device_type, torch_dev.current_device())
    d2h = int(lmcache_native.TransferDirection.D2H)
    h2d = int(lmcache_native.TransferDirection.H2D)
    torch.manual_seed(0)

    paged = [
        torch.randn(2, NB, BS, NH, HS, dtype=torch.bfloat16, device=device)
        for _ in range(NL)
    ]
    paged_ptrs = torch.tensor(
        [t.data_ptr() for t in paged], dtype=torch.int64, device=device
    )
    num_blocks = NUM_OBJECTS * CHUNK // BS
    block_ids = torch.randperm(NB, device=device)[:num_blocks].contiguous()
    object_shape = (2, NL, CHUNK, NH * HS)

    def new_objects():
        return [
            torch.zeros(object_shape, dtype=torch.bfloat16, device=device)
            for _ in range(NUM_OBJECTS)
        ]

    def new_host_objects():
        return [
            torch.zeros(object_shape, dtype=torch.bfloat16).pin_memory()
            for _ in range(NUM_OBJECTS)
        ]

    # Reference: core's own executor and kernel, staged to host.
    ref, ref_host = new_objects(), new_host_objects()
    specs, steps = _plan(paged_ptrs, ref, block_ids, _staging(cuda_ops, ref, ref_host))
    cuda_ops.execute_object_group_transfer(d2h, device, HOST_ALIGNMENT, specs, steps)

    for reverse_layers in (False, True):
        out, out_host = new_objects(), new_host_objects()
        specs, steps = _plan(
            paged_ptrs, out, block_ids, _staging(cuda_ops, out, out_host)
        )
        object_ptrs = torch.tensor(
            [o.data_ptr() for o in out], dtype=torch.int64, device=device
        )
        probe.execute(
            d2h,
            device.index,
            HOST_ALIGNMENT,
            specs,
            steps,
            object_ptrs,
            reverse_layers,
        )
        torch_dev.synchronize(device)
        for got, want in zip(out_host, ref_host, strict=True):
            expected = want.flip(1) if reverse_layers else want
            assert torch.equal(got, expected), f"reverse_layers={reverse_layers}"

    # H2D through the probe restores the paged blocks core saved.
    restored = [torch.zeros_like(t) for t in paged]
    restored_ptrs = torch.tensor(
        [t.data_ptr() for t in restored], dtype=torch.int64, device=device
    )
    specs, steps = _plan(restored_ptrs, ref, block_ids, [])
    object_ptrs = torch.tensor(
        [o.data_ptr() for o in ref], dtype=torch.int64, device=device
    )
    probe.execute(h2d, device.index, HOST_ALIGNMENT, specs, steps, object_ptrs)
    torch_dev.synchronize(device)
    for got, want in zip(restored, paged, strict=True):
        assert torch.equal(got[:, block_ids], want[:, block_ids])
