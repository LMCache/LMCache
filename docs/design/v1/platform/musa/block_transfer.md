# MUSA Multiprocess Block Transfer

## Goal

Enable the LMCache-driven MUSA handle path without adding MUSA branches to
generic multiprocess code. The platform backend consumes the existing pointer
operands and supports these validated layouts:

- `NL x [2, NB, BS, NH, HS]`
- `NL x [NB, BS, HS]`

Other layouts fail before transfer.

## Execution

`MusaDeviceSpec` activates `MUSACacheContext` after the memory and event IPC
capabilities pass. The generic transfer loop passes the context's packed KV
pointers and staging pointers to `MusaDeviceOps`.

`MusaDeviceOps` performs the following steps:

1. validate the engine layout;
2. resolve the exact dtype and shape metadata;
3. reconstruct non-owning MUSA Tensor views from process-local pointers;
4. try the optional native MUSA transfer;
5. use the TorchMUSA-compatible torch implementation when native transfer is
   unavailable or incompatible.

Two-byte pointer operands require an exact `shape_desc.dtype`; inferring from
`element_size` would confuse `float16` and `bfloat16`. MLA views preserve
`block_stride_elems` so padded blocks remain correctly addressed.

## Capability

The built-in torch implementation makes block transfer available without the
optional native extension. Full MUSA handle mode still requires explicit
opt-in plus memory IPC and event IPC support. MUSA auto mode remains on the
engine-driven path.
