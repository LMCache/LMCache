import torch
import asyncio
import aiofiles
import ctypes

path = "/local/disk_test/local_disk/vllm@test_model@3@123@6ba475e925db8eb63c7325485de0884c6ec75f583c503808de8e5108a0e35008.pt"

dst_device = "cuda"
dtype = torch.bfloat16
shape = (2,32,256,1024)


async def async_save_bytes_to_disk(
    path: str,
    kv_chunk: torch.Tensor,
) -> None:
    """
    Convert KV to bytes and async store bytes to disk.
    """
    num_bytes = kv_chunk.numel() * kv_chunk.element_size()
    ptr = kv_chunk.data_ptr()
    ubyte_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    byte_array = (ctypes.c_ubyte * num_bytes).from_address(
                    ctypes.addressof(ubyte_ptr.contents))
    async with aiofiles.open(path, 'wb') as f:
        await f.write(byte_array)

kv_chunk = torch.randn(shape, dtype=dtype)
kv_chunk = kv_chunk.pin_memory()

asyncio.run(async_save_bytes_to_disk(path, kv_chunk))


with open(path, 'rb') as f:        
    bytes_data = f.read()
kv_chunk = torch.frombuffer(bytes_data, dtype=dtype).view(shape)
kv_chunk = kv_chunk.to(dst_device)