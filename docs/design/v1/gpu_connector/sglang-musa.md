# SGLang MUSA connector

LMCache selects the SGLang MUSA connector when SGLang runs on a MUSA device.
The in-process connector uses TorchMUSA `index_select` and `index_copy_` so it
does not depend on CUDA transfer kernels.

## Supported layouts

- Non-layerwise MHA: separate K and V layer pools, represented as
  `[2, NL, T, NH * HS]` in LMCache.
- Non-layerwise MLA: one tensor per layer, represented as
  `[NL, T, HS]` in LMCache.
- Layerwise MHA: one layer at a time, represented as `[T, 2, NH * HS]`.

Layerwise MLA and multiprocess MLA are rejected because no validated MUSA
layout exists for them.

## Transfer paths

In-process transfers use SGLang's `slot_mapping`. When SGLang supplies only an
uncached suffix, the connector applies the prefix offset before indexing the
mapping. Invalid ranges and unsupported layouts fail with a clear error.

The multiprocess MHA path uses the MUSA memory/event IPC wrappers and the
Torch block-transfer fallback. Its wire format is the flat pointer sequence
`[K0, ..., K(NL-1), V0, ..., V(NL-1)]`; the server reconstructs the nested K/V
layer pools before transfer. Native MUSA kernels do not claim this layout, so
the Torch fallback remains the correctness path.

Handle transfer is opt-in and fail-closed. It is enabled only when memory IPC,
event IPC, the MUSA cache context, and block transfer are all available. MUSA
tensors are never routed through CUDA IPC or CUDA pointer-copy code.
