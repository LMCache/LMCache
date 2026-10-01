# MUSA Multiprocess Transfer Pointer Contract

## Goal

Keep the existing multiprocess pointer APIs unchanged. MUSA tensors are opened
through TorchMUSA IPC before transfer setup, and MUSA-specific adaptation stays
inside the MUSA platform implementation.

## Process boundary

Pointers are process-local. The worker exports tensors with
`torch.musa.ipc.export_tensor()`, and the server opens them with
`torch.musa.ipc.open_tensor()`. Only pointers obtained from the server's opened
tensors are passed to transfer operations; raw pointers are never serialized.

The server keeps each IPC owner alive until the transfer stream is synchronized
and the cache context closes.

## Platform contract

The generic path keeps the existing calls:

```python
paged_ptrs = context.get_kernel_group_kv_pointers(group_idx)
staging_ptrs = [context.get_temp_kernel_group_buffer(i, group_idx).data_ptr()]
device_ops.multi_layer_block_kv_transfer(paged_ptrs, staging_ptrs, ...)
```

`MUSACacheContext` returns the same packed `int64` pointer tensor shape used by
the current pointer path. `construct_musa_tensor_from_data_pointer()` rebuilds
a non-owning Tensor view from a process-local pointer and explicit shape,
stride, dtype, device, and storage-offset metadata.

The supported layouts are:

- `NL x [2, NB, BS, NH, HS]`
- `NL x [NB, BS, HS]`

The helper supports explicit strides so padded block layouts can be represented
without copying. The allocation owner must outlive every reconstructed view.

## Stream ordering

Generic completion and event recorders continue to pass an integer stream
pointer. `MusaDeviceOps` wraps that pointer with TorchMUSA's external-stream
API, synchronizes it, and then publishes to the existing Python recorder queue.
The wrapper does not own or destroy the underlying stream.

## Compatibility

No pointer-versus-Tensor branch is added to generic multiprocess code. CUDA
native signatures, callbacks, transfer planning, and wire format are unchanged.
The MUSA block-transfer capability remains disabled until the transfer path
consumes this contract.
