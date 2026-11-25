# Consuming KV group content in a serde

A serde sees an L1 object as one or more tensors whose shapes and dtypes come
from the transfer layout. For a Mamba (or other recurrent-state) group that
layout is a synthetic attention-shaped view, typically int8, so neither the
shape nor the dtype says what the bytes are. A serde that transforms data,
such as a quantizer, needs to know:

- whether a tensor holds attention K/V, recurrent state, or something it does
  not recognize, and
- for recurrent state, where each state (e.g. Mamba `conv` and `ssm`) sits
  in a page, with its real dtype and shape.

The worker publishes this at registration as the *group content*, and the
server hands content-aware serdes a resolver for it. This page describes what
is published, where it is kept, how a serde resolves it, how to locate a
state's bytes in an object, and the rules a serde must follow. No built-in
serde reads the content yet.

## What is published

Defined in `lmcache/v1/common/group_view.py`:

| Type | Field | Meaning |
|---|---|---|
| `CacheCategory` | `ATTENTION`, `RECURRENT`, `UNKNOWN` | Engine spec type of a group. The vLLM integration reports Mamba specs (including GDN and linear attention) as `RECURRENT`. `UNKNOWN` is `""`, the proto3 zero value, so a peer that predates the field decodes to it over both msgspec and gRPC. |
| `StateComponentDesc` | `byte_offset` | Offset of the state from the start of one layer's page for one block. The same for every layer and block. |
| | `byte_length` | Bytes of real data, excluding alignment padding. |
| | `dtype_str` | Torch dtype name without the `torch.` prefix, e.g. `"float32"`. |
| | `shape` | Per-block shape of the state. |
| `PartContentDesc` | `cache_category` | As above. |
| | `state_components` | The page's states in vLLM's order. Empty unless the layout is known (see below). |
| | `page_bytes` | Bytes of one layer's page for one block in the object tensor. Non-zero only when `state_components` is non-empty. |

`EngineGroupInfo` carries `cache_category` and `state_components`;
`KernelGroupInfo.content_desc()` builds the `PartContentDesc`, adding
`page_bytes`.

`state_components` is empty, and the bytes must be treated as opaque, when:

- the group is not `RECURRENT`;
- the spec declares no states;
- the layer is not transferred through the unified-state view (the page-view
  and slice-plane views split a page into planes, so a page offset does not
  address them); or
- the states would not fit in the page.

Offsets follow `MambaBase.bind_kv_cache`: states are packed back to back,
each padded to `spec.state_alignment` bytes. Bytes outside every state
(alignment and page padding) are never read by vLLM.

## Where the server keeps it

Both transfer modules register the content with the server context's
`LayoutDescRegistry`, beside each object group's layout
(`group_content_descs`: object group id to one `PartContentDesc | None` per
part):

| Transfer mode | Source of each part's content |
|---|---|
| engine-driven | The worker's `GroupLayout.content`. Each group is its own object group with one part; `None` when the worker predates the field. |
| lmcache-driven | `kernel_groups[k].content_desc()` of the server's `KVLayerGroupsManager`, which it rebuilds from the registered `EngineGroupInfo`s, for each kernel group `k` of the object group in part order. |

The content shares its registry entry's lifecycle: it appears on registration
(and re-registration after a server restart) and goes away with the entry's
last registration. Registrations sharing an entry that describe a group
differently make it unknown instead of letting the latest one win.

## From an object to its parts

A serde receives `serialize(src, dst, key)` / `deserialize(src, dst, key)`.
`key.object_group_id` selects the object group, and `src.get_tensor(i)`
returns part `i` of the object:

- **engine-driven**: every LMCache group is its own object group with one
  part, so part `0` of object group `g` is LMCache group `g`.
- **lmcache-driven**: object group `o` holds one part per kernel group in
  `kv_layer_groups_manager.object_groups[o].kernel_group_indices`, in that
  order, so part `i` is kernel group `kernel_group_indices[i]`.

For a recurrent part with non-empty `state_components`, the unified-state
transfer view produces a single-plane tensor of shape
`[num_layers, slots, hidden]`, where `slots` covers the group's whole window.
If the window spans multiple blocks, each layer's pages are stored back to
back, so each layer holds `slots * hidden * element_size // page_bytes`
pages. Do not assume this layout for other recurrent parts. If the component
descriptions are empty or missing, treat the part as opaque bytes and pass
it through unchanged.

## Locating a state's bytes

```python
import torch

from lmcache.v1.common.group_view import PartContentDesc


def iter_states(part: torch.Tensor, content: PartContentDesc):
    """Yield (layer, page, state_index, view) for every state in a part.

    The views alias ``part``; ``page`` counts blocks within the window.
    """
    num_layers = part.shape[-3]
    layer_bytes = part.reshape(num_layers, -1).view(torch.uint8)
    num_pages = layer_bytes.shape[1] // content.page_bytes
    for layer in range(num_layers):
        for page in range(num_pages):
            base = page * content.page_bytes
            for idx, comp in enumerate(content.state_components):
                start = base + comp.byte_offset
                raw = layer_bytes[layer, start : start + comp.byte_length]
                dtype = getattr(torch, comp.dtype_str)
                yield layer, page, idx, raw.view(dtype).view(comp.shape)
```

Call it only when `content.state_components` is non-empty. The views use the
same byte offsets and dtype reinterpretation as `MambaBase.bind_kv_cache`.
`test_published_state_components_address_each_layer_in_the_chunk` in
`tests/v1/test_vllm_mamba_unified_view_layouts.py` checks the same arithmetic
against vLLM's own binding, over two layers and two blocks per chunk.

## Rules for a serde

1. **Dispatch on `cache_category`, never on shape or dtype.** A recurrent
   part's view can look exactly like fused attention K/V.
2. **Pass through anything you do not understand**: `UNKNOWN`, a missing
   (`None`) content, and `RECURRENT` with empty `state_components` must be
   stored byte for byte. Never treat `UNKNOWN` as attention.
3. **Transform each state in its own dtype.** States in one page usually
   differ (e.g. bf16 `conv`, fp32 `ssm`); quantizing the int8 view, or the
   whole page as one dtype, corrupts them.
4. **Padding carries no data.** Bytes outside every state may be dropped on
   serialize and zero-filled on deserialize, but nothing else may be written
   there.
5. **Make the serialized frame self-describing.** `deserialize` can run when
   no registration context exists, for example when another server, or this
   one after a restart, loads the object from L2 before a worker registers.
   Record in the frame (as `aesgcm` records a version byte) whatever is
   needed to decode it: at least a format version and, for each transformed
   part, the component layout used. Do not look the content up again at load
   time.
6. **Size estimates cannot see the content.** `estimate_serialized_size`
   receives only a `MemoryLayoutDesc`, with no key, so it must return a
   bound that holds for every content the layout can carry, typically the
   untransformed size plus the frame header.

## Resolving content in a serde

A serde that needs the content registers its factory with
`register_content_aware_serde_factory` instead of `register_serde_factory`.
The factory also receives a `PartContentResolver`,
`(key: ObjectKey, part: int) -> PartContentDesc | None`:

```python
from lmcache.v1.distributed.serde import (
    PartContentResolver,
    SerdeProcessor,
    register_content_aware_serde_factory,
)


def _create_my_serde(
    kwargs: dict[str, object], resolver: PartContentResolver
) -> SerdeProcessor:
    ...  # keep ``resolver``; call resolver(key, i) in serialize


register_content_aware_serde_factory("my_serde", _create_my_serde)
```

The MP server context builds its `StorageManager` with a resolver backed by
its `LayoutDescRegistry`, and the storage manager hands it to the serde of
every L2 adapter it builds. Other serdes are unaffected. A storage manager
built without a resolver gives content-aware serdes one that always returns
`None`.

`ObjectKey` carries no world size, so the resolver consults every
registration of `key.model_name` and returns content only where they agree:
registrations with different world sizes may shard state differently. It
returns `None` when the model is not registered, the part is unknown, or
registrations disagree. Rule 2 applies to all of these.

Resolve content only in `serialize`. By rule 5, `deserialize` decodes from the
frame.

## Limits

- Only the vLLM integration publishes content. SGLang and other engines leave
  every group `UNKNOWN`, which rule 2 makes safe.
- Existing lossy serdes (`fp8`, `asym_k16_v8`, `turboquant`) do not consult
  the content and transform every object, so they corrupt Mamba objects in an
  L2 tier they are configured on. Making them pass non-attention parts
  through is separate work.
