# SPDX-License-Identifier: Apache-2.0
# Standard
from multiprocessing import Queue
import multiprocessing as mp

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import (
    CudaIPCWrapper,
    IPCCacheEngineKey,
    get_customized_decoder,
    get_customized_encoder,
)


def test_ipc_cache_engine_key_serialization():
    """Test encoding and decoding of IPCCacheEngineKey using msgspec."""
    # Create a sample IPCCacheEngineKey
    original_key = IPCCacheEngineKey.from_int_hash(
        model_name="test_model", world_size=4, worker_id=1, chunk_hash=123456789
    )

    # Encode the key
    encoded = msgspec.msgpack.encode(original_key)

    # Decode the key
    decoded_key = msgspec.msgpack.decode(encoded, type=IPCCacheEngineKey)

    # Verify correctness
    assert original_key == decoded_key, "IPCCacheEngineKeys do not match!"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CudaIPCWrapper tests",
)
def test_cudaipc_wrapper_serialization():
    """Test custom encoder/decoder for single CudaIPCWrapper object."""
    encoder = get_customized_encoder(type=CudaIPCWrapper)
    decoder = get_customized_decoder(type=CudaIPCWrapper)

    # Create a sample tensor
    original_tensor = torch.randn(3, 4, device="cuda")
    wrapper = CudaIPCWrapper(original_tensor)

    # Encode the wrapper
    encoded = encoder.encode(wrapper)

    # Decode the wrapper
    decoded_wrapper = decoder.decode(encoded)
    assert isinstance(decoded_wrapper, CudaIPCWrapper), (
        "Decoded object is not of type CudaIPCWrapper"
    )
    assert decoded_wrapper == wrapper, (
        "Decoded CudaIPCWrapper does not match the original"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CudaIPCWrapper tests",
)
def test_cudaipc_wrapper_list_serialization():
    """Test custom encoder/decoder for list of CudaIPCWrapper objects."""
    wrappers = []
    for _ in range(5):
        tensor = torch.randn(2, 2, device="cuda")
        wrapper = CudaIPCWrapper(tensor)
        wrappers.append(wrapper)

    encoder = get_customized_encoder(type=list[CudaIPCWrapper])
    decoder = get_customized_decoder(type=list[CudaIPCWrapper])

    # Encode the list of wrappers
    encoded = encoder.encode(wrappers)

    # Decode the list of wrappers
    decoded_wrappers = decoder.decode(encoded)

    assert len(decoded_wrappers) == len(wrappers), (
        "Decoded list length does not match original"
    )

    for original, decoded in zip(wrappers, decoded_wrappers, strict=False):
        assert original == decoded, "Decoded CudaIPCWrapper does not match the original"


def _worker_process_deserialize_and_reconstruct(
    encoded_data: bytes, result_queue: Queue
):
    """
    Worker function that runs in a separate process.
    Deserializes CudaIPCWrapper list and reconstructs tensors.
    """
    try:
        # Decode the list of wrappers
        torch.cuda.init()
        decoder = get_customized_decoder(type=list[CudaIPCWrapper])
        decoded_wrappers = decoder.decode(encoded_data)

        # Convert each wrapper back to tensor and compute checksum
        checksums = []
        shapes = []
        for wrapper in decoded_wrappers:
            tensor = wrapper.to_tensor()
            # Compute checksum as sum of all elements
            checksum = float(tensor.sum().cpu().item())
            checksums.append(checksum)
            shapes.append(list(tensor.shape))

            # Do add 1 on the tensor to ensure it's writable
            tensor.add_(1)

        result_queue.put(("success", checksums, shapes))
    except Exception as e:
        result_queue.put(("error", str(e), None))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for CudaIPCWrapper multiprocessing tests",
)
def test_cudaipc_wrapper_multiprocess_serialization():
    """
    Test CudaIPCWrapper serialization across processes using spawn method.
    This verifies that CUDA IPC handles can be properly shared between processes.
    """
    # Set multiprocessing start method to spawn
    ctx = mp.get_context("spawn")

    # Create test tensors and wrappers in the main process
    num_tensors = 3
    tensors = []
    test_data = []
    wrappers = []

    for i in range(num_tensors):
        # Create a tensor with known values
        tensor = torch.full(
            (2, 3), fill_value=float(i + 1), dtype=torch.float32, device="cuda"
        )
        tensors.append(tensor)
        wrapper = CudaIPCWrapper(tensor)
        wrappers.append(wrapper)

        # Store expected checksum and shape
        expected_checksum = float(tensor.sum().cpu().item())
        expected_shape = list(tensor.shape)
        test_data.append((expected_checksum, expected_shape))

    # Serialize the wrappers
    encoder = get_customized_encoder(type=list[CudaIPCWrapper])
    encoded_data = encoder.encode(wrappers)

    # Create a queue for results
    result_queue = ctx.Queue()

    # Start worker process
    process = ctx.Process(
        target=_worker_process_deserialize_and_reconstruct,
        args=(encoded_data, result_queue),
    )
    process.start()

    # Wait for result with timeout
    process.join(timeout=10)

    # Check if process completed successfully
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("Worker process timed out")

    assert process.exitcode == 0, (
        f"Worker process failed with exit code {process.exitcode}"
    )

    # Get result from queue
    assert not result_queue.empty(), "No result received from worker process"
    status, checksums, shapes = result_queue.get()

    assert status == "success", f"Worker process encountered error: {checksums}"
    assert len(checksums) == num_tensors, "Number of checksums does not match"
    assert len(shapes) == num_tensors, "Number of shapes does not match"

    # Verify checksums and shapes match
    for i, (
        (expected_checksum, expected_shape),
        actual_checksum,
        actual_shape,
    ) in enumerate(zip(test_data, checksums, shapes, strict=False)):
        assert actual_shape == expected_shape, (
            f"Tensor {i}: shape mismatch. Expected {expected_shape}, got {actual_shape}"
        )
        assert abs(actual_checksum - expected_checksum) < 1e-5, (
            f"Tensor {i}: checksum mismatch. Expected {expected_checksum}, "
            f"got {actual_checksum}"
        )

    # Verify that the tensors are being modified in the worker process
    for i, (tensor, (expected_checksum, _)) in enumerate(
        zip(tensors, test_data, strict=False)
    ):
        # After adding 1 to each element, the new checksum should be:
        num_elements = tensor.numel()
        new_expected_checksum = expected_checksum + float(num_elements)
        actual_checksum = float(tensor.sum().cpu().item())
        assert abs(actual_checksum - new_expected_checksum) < 1e-5, (
            f"Tensor {i}: post-modification checksum mismatch. "
            f"Expected {new_expected_checksum}, got {actual_checksum}"
        )
