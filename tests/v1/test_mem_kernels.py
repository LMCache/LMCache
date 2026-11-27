# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
import lmcache.lmcache_native as lmcache_native

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (torch_dev.is_available() and torch_device_type == "cuda"),
        reason="Requires CUDA backend",
    ),
]

try:
    # First Party
    import lmcache.cuda_ops as cuda_ops
except ImportError:
    cuda_ops = None


def test_lmcache_memcpy_async():
    # Register 4 memory regions and try to launch copy
    chunk_size = 1 << 26
    num_chunks = 4
    dtype = torch.bfloat16
    elements_per_chunk = chunk_size // dtype.itemsize

    big_cpu_tensor = torch.rand(
        elements_per_chunk * num_chunks, dtype=dtype, device="cpu"
    )
    big_gpu_tensor = torch.rand(
        elements_per_chunk * num_chunks, dtype=dtype, device=torch_device_type
    )

    def check_gpu_and_cpu_equal(
        gpu_tensor: torch.Tensor,
        cpu_tensor: torch.Tensor,
    ):
        cpu_tensor_from_gpu = gpu_tensor.to("cpu")
        assert torch.allclose(cpu_tensor_from_gpu, cpu_tensor, atol=1e-5)

    rt = torch_dev.cudart()

    # Register the cpu memory
    ptr = big_cpu_tensor.data_ptr()
    for i in range(num_chunks):
        rt.cudaHostRegister(ptr + i * chunk_size, chunk_size, 0)

    # Launch default cuda copy
    with pytest.raises(RuntimeError):
        big_gpu_tensor.copy_(big_cpu_tensor, non_blocking=True)
        torch_dev.synchronize()

    # Launc default cuda copy for a small page
    big_gpu_tensor[: elements_per_chunk // 2].copy_(
        big_cpu_tensor[: elements_per_chunk // 2], non_blocking=True
    )
    torch_dev.synchronize()

    check_gpu_and_cpu_equal(
        big_gpu_tensor[: elements_per_chunk // 2],
        big_cpu_tensor[: elements_per_chunk // 2],
    )

    # Positions to copy
    # - 0.25 chunk -- 0.75 chunk
    # - 0.75 chunk -- 1.25 chunk
    # - 1.75 chunk -- 3.25 chunk
    # - whole 4 chunks
    starts = [chunk_size // 4, (3 * chunk_size) // 4, (7 * chunk_size) // 4, 0]
    ends = [
        (3 * chunk_size) // 4,
        (5 * chunk_size) // 4,
        (13 * chunk_size) // 4,
        chunk_size * num_chunks,
    ]

    # H2D copy
    for start, end in zip(starts, ends, strict=False):
        # should not be equal before copy
        with pytest.raises(AssertionError):
            check_gpu_and_cpu_equal(
                big_gpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
                big_cpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
            )

        cuda_ops.lmcache_memcpy_async(
            big_gpu_tensor.data_ptr() + start,
            big_cpu_tensor.data_ptr() + start,
            end - start,
            lmcache_native.TransferDirection.H2D,
            start,
            chunk_size,
        )
        torch_dev.synchronize()

        check_gpu_and_cpu_equal(
            big_gpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
            big_cpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
        )

    # Reset the data in gpu
    big_gpu_tensor = torch.rand(
        elements_per_chunk * num_chunks, dtype=dtype, device=torch_device_type
    )
    # D2H copy
    for start, end in zip(starts, ends, strict=False):
        # copy from gpu to cpu
        with pytest.raises(AssertionError):
            check_gpu_and_cpu_equal(
                big_gpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
                big_cpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
            )

        cuda_ops.lmcache_memcpy_async(
            big_cpu_tensor.data_ptr() + start,
            big_gpu_tensor.data_ptr() + start,
            end - start,
            lmcache_native.TransferDirection.D2H,
            start,
            chunk_size,
        )
        torch_dev.synchronize()

        check_gpu_and_cpu_equal(
            big_gpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
            big_cpu_tensor[start // dtype.itemsize : end // dtype.itemsize],
        )

    # Unregister the cpu memory
    ptr = big_cpu_tensor.data_ptr()
    for i in range(num_chunks):
        rt.cudaHostUnregister(ptr + i * chunk_size)


def test_lmcache_memcpy_async_int8_hidden132():
    """Test lmcache_memcpy_async with int8 dtype and hidden_size=132.

    hidden_size=132 means each token occupies 132 bytes, which is NOT a
    multiple of 4 (uint32 size). This tests that the kernel correctly handles
    non-uint32-aligned data sizes when splitting copies across registered
    memory chunk boundaries.

    The KV buffer shape is (2, num_layers, num_tokens, hidden_size), so the
    total bytes = 2 * num_layers * num_tokens * 132. We pick values such that
    the copy spans across at least one PIN_CHUNK_SIZE (64 MB) boundary.
    """
    # Use PIN_CHUNK_SIZE = 64 MB as the registration granularity
    chunk_size = 1 << 26  # 64 MB
    num_chunks = 2
    dtype = torch.int8
    hidden_size = 132  # NOT a multiple of 4 — tests uint32 alignment handling

    # Total bytes must span at least one chunk boundary.
    # We allocate num_chunks * chunk_size bytes worth of int8 elements.
    total_bytes = chunk_size * num_chunks
    total_elements = total_bytes  # int8: 1 byte per element

    cpu_tensor = torch.randint(-128, 127, (total_elements,), dtype=dtype, device="cpu")
    gpu_tensor = torch.zeros(total_elements, dtype=dtype, device=torch_device_type)

    rt = torch_dev.cudart()

    # Register the cpu memory in PIN_CHUNK_SIZE chunks
    ptr = cpu_tensor.data_ptr()
    for i in range(num_chunks):
        rt.cudaHostRegister(ptr + i * chunk_size, chunk_size, 0)

    def check_equal(gpu_t: torch.Tensor, cpu_t: torch.Tensor):
        assert torch.equal(gpu_t.cpu(), cpu_t), "GPU and CPU tensors are not equal"

    # Test cases: copy ranges that cross chunk boundaries with non-4-byte-aligned
    # sizes, simulating real KV cache transfers where hidden_size=132 (int8).
    #
    # Each "KV row" is hidden_size=132 bytes. We pick copy sizes that are
    # multiples of 132 but NOT multiples of 4, so the copy boundary falls on
    # a non-uint32-aligned offset within a registered chunk.
    #
    # num_tokens=3: 3 * 132 = 396 bytes (396 % 4 == 0, but 132 % 4 != 0)
    # num_tokens=1: 1 * 132 = 132 bytes (132 % 4 == 0)
    # num_tokens=5: 5 * 132 = 660 bytes (660 % 4 == 0)
    # Use an offset that puts the copy right across the 64MB boundary.
    boundary = chunk_size  # 64 MB boundary in bytes

    # Place the copy so it straddles the chunk boundary:
    # start = boundary - 2 * hidden_size, length = 4 * hidden_size
    # This means 2 * hidden_size bytes before the boundary and 2 * hidden_size
    # bytes after — each segment is 264 bytes, not a multiple of 4*hidden_size.
    test_cases = [
        # (start_bytes, num_hidden_rows)
        (boundary - 2 * hidden_size, 4),  # straddles boundary, 528 bytes
        (boundary - hidden_size, 3),  # straddles boundary, 396 bytes
        (boundary - 3 * hidden_size, 6),  # straddles boundary, 792 bytes
        (0, 1),  # single row at start, 132 bytes
        (total_bytes - hidden_size, 1),  # single row at end, 132 bytes
    ]

    for start_bytes, num_rows in test_cases:
        nbytes = num_rows * hidden_size
        end_bytes = start_bytes + nbytes
        assert end_bytes <= total_bytes, (
            f"Test case out of bounds: start={start_bytes}, nbytes={nbytes}"
        )

        # Reset gpu_tensor slice to zeros so we can detect the copy
        gpu_tensor[start_bytes:end_bytes].zero_()

        # H2D copy
        cuda_ops.lmcache_memcpy_async(
            gpu_tensor.data_ptr() + start_bytes,
            cpu_tensor.data_ptr() + start_bytes,
            nbytes,
            lmcache_native.TransferDirection.H2D,
            start_bytes,
            chunk_size,
        )
        torch_dev.synchronize()

        check_equal(
            gpu_tensor[start_bytes:end_bytes],
            cpu_tensor[start_bytes:end_bytes],
        )

    # D2H copy: write known values to GPU, copy back to CPU, verify
    gpu_src = torch.randint(
        -128, 127, (total_elements,), dtype=dtype, device=torch_device_type
    )
    cpu_dst = torch.zeros(total_elements, dtype=dtype, device="cpu")

    # Register cpu_dst for D2H
    ptr_dst = cpu_dst.data_ptr()
    for i in range(num_chunks):
        rt.cudaHostRegister(ptr_dst + i * chunk_size, chunk_size, 0)

    for start_bytes, num_rows in test_cases:
        nbytes = num_rows * hidden_size
        end_bytes = start_bytes + nbytes

        cpu_dst[start_bytes:end_bytes].zero_()

        cuda_ops.lmcache_memcpy_async(
            cpu_dst.data_ptr() + start_bytes,
            gpu_src.data_ptr() + start_bytes,
            nbytes,
            lmcache_native.TransferDirection.D2H,
            start_bytes,
            chunk_size,
        )
        torch_dev.synchronize()

        check_equal(
            gpu_src[start_bytes:end_bytes],
            cpu_dst[start_bytes:end_bytes],
        )

    # Unregister memory
    ptr = cpu_tensor.data_ptr()
    for i in range(num_chunks):
        rt.cudaHostUnregister(ptr + i * chunk_size)

    ptr_dst = cpu_dst.data_ptr()
    for i in range(num_chunks):
        rt.cudaHostUnregister(ptr_dst + i * chunk_size)


def test_multi_layer_kernel_offset_past_int32():
    """Regression: a paged pool over 2**31 scalars must not wrap the offset.

    ``page_buffer_offset`` used to compute ``token_idx * scalars_per_token``
    in int32, so slots past 2**31 / scalars_per_token got a negative offset
    and the kernel faulted (seen on a 13831-scalar hybrid aux group at
    ~62k context). The head size is odd so the kernel moves 2-byte units,
    which keeps the single-plane MLA pool at ~4.3 GiB.
    """
    head_size, block_size, num_tokens = 1025, 16, 48
    dtype = torch.bfloat16
    int32_max = 2**31 - 1
    first_slot = int32_max // head_size + 1
    num_blocks = (first_slot + num_tokens) // block_size + 1
    page_buffer_size = num_blocks * block_size
    pool_bytes = page_buffer_size * head_size * dtype.itemsize
    if torch_dev.mem_get_info()[0] < pool_bytes + (1 << 30):
        pytest.skip(f"needs {pool_bytes / 2**30:.1f} GiB free GPU memory")

    device = torch.device(torch_device_type)
    paged = torch.zeros(num_blocks, block_size, head_size, dtype=dtype, device=device)
    rows = paged.view(page_buffer_size, head_size)
    ptrs = torch.tensor([paged.data_ptr()], dtype=torch.long, device=device)
    slots = torch.arange(
        page_buffer_size - num_tokens,
        page_buffer_size,
        dtype=torch.long,
        device=device,
    )
    assert int(slots[0]) * head_size > int32_max
    src = torch.randn(1, 1, num_tokens, head_size, dtype=dtype, device=device)

    for direction, kv in (
        (lmcache_native.TransferDirection.H2D, src),
        (lmcache_native.TransferDirection.D2H, torch.zeros_like(src)),
    ):
        cuda_ops.multi_layer_kv_transfer(
            kv,
            ptrs,
            slots,
            device,
            page_buffer_size,
            direction,
            lmcache_native.EngineKVFormat.NL_X_NB_BS_HS,
            block_size=block_size,
            head_size=head_size,
        )
        torch_dev.synchronize()
        assert torch.equal(rows[slots], src[0, 0])
        assert torch.equal(kv, src)
    # Nothing landed at the wrapped-around (low) addresses.
    assert not rows[:num_tokens].any()
