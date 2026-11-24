# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression test for the Python ``lmcache_memcpy_async`` fallback.

The host buffer is ``cudaHostRegister``'d in ``host_buffer_alignments``-sized
chunks, and one ``cudaMemcpy`` that spans two registered chunks can fail with
``cudaErrorInvalidValue`` (see LMCache/LMCache#6617). The fallback must
therefore split a copy at every ``host_buffer_alignments`` boundary, as the
C++ ``lmcache_memcpy_async`` does. The test replaces libcudart with a recorder,
so it needs no GPU.
"""

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import TransferDirection
from lmcache.v1.platform.torch_ops import mem_kernels

MB = 1 << 20
DEST = 0x10_0000_0000
SRC = 0x20_0000_0000


class _RecordingCudart:
    """Stands in for libcudart and records every ``cudaMemcpy`` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def cudaMemcpy(self, dest, src, nbytes, kind) -> int:  # noqa: N802
        self.calls.append((dest.value, src.value, nbytes.value))
        return 0


@pytest.fixture
def cudart(monkeypatch) -> _RecordingCudart:
    lib = _RecordingCudart()
    monkeypatch.setattr(mem_kernels, "_get_copy_lib", lambda: lib)
    return lib


# (host_buffer_offset, nbytes, host_buffer_alignments)
CASES = [
    pytest.param(0, 64, 64, id="exactly-one-block"),
    pytest.param(32, 32, 64, id="inside-one-block"),
    pytest.param(32, 64, 64, id="crosses-one-boundary"),
    pytest.param(0, 192, 64, id="three-full-blocks"),
    pytest.param(64, 128, 64, id="starts-on-boundary"),
    pytest.param(16, 200, 64, id="unaligned-start-many-boundaries"),
    pytest.param(64 * MB - 4096, 8192, 64 * MB, id="64MB-chunks-straddle"),
]


@pytest.mark.parametrize("direction", [TransferDirection.H2D, TransferDirection.D2H])
@pytest.mark.parametrize("host_buffer_offset,nbytes,alignment", CASES)
def test_no_cudamemcpy_crosses_a_host_register_boundary(
    cudart, direction, host_buffer_offset, nbytes, alignment
):
    mem_kernels.lmcache_memcpy_async(
        DEST, SRC, nbytes, direction, host_buffer_offset, alignment
    )

    # Every copy stays inside one registered chunk of the host buffer.
    copied = 0
    for dest, src, size in cudart.calls:
        assert size > 0
        first = (host_buffer_offset + copied) // alignment
        last = (host_buffer_offset + copied + size - 1) // alignment
        assert first == last, (
            f"cudaMemcpy of {size} bytes at host offset "
            f"{host_buffer_offset + copied} crosses a {alignment}-byte boundary"
        )
        # The pieces tile the original range with no gap or overlap.
        assert dest == DEST + copied
        assert src == SRC + copied
        copied += size

    assert copied == nbytes