# KV Cache Optimization Interface: Design Document

## 1. Overview

The era of agentic AI has arrived. As Jensen Huang highlighted at GTC 2026, the inference inflection point is here — AI agents think, reason, and act, driving massive growth in context length (from 100K to millions of tokens) and token throughput. Agentic systems like Claude Code and OpenClaw spawn sub-agents, iterate on code, and maintain long-running context, all of which pound on KV cache and storage at AI-speed. Optimizing KV cache — reducing its storage footprint and compacting it to fit more context — is essential to scaling agentic inference.

This document defines a modular interface for KV cache optimization in LMCache, focusing on two types of optimization:

1. **KV Cache Compression** (Serde): Reduces the *storage size* and *transfer latency* of KV caches through encoding/decoding (e.g., quantization). The logical shape is preserved — the same number of tokens and layers go in and come out.
2. **KV Cache Compaction** (Token Dropping): Generates a more *compact* KV cache by evicting less-important tokens, reducing both memory footprint and attention compute.

## 2. Functionality Partition

KV cache optimization can be decomposed into three layers:

| Layer | Responsibility |
|-------|---------------|
| **KV Cache Transformation** | Generating a new KV cache from an existing one — either *compressed* (smaller on disk, same shape) or *compacted* (fewer tokens) |
| **Orchestration** | A series of APIs that trigger transformation actions based on specific system conditions |
| **System Support** | Low-level support in LMCache and vLLM (memory layout, tensor management, storage I/O) |

This doc focuses on the **KV Cache Transformation** layer, which concretely takes the form of the two optimizations above: compression via the Serde interface, and compaction via the Token Dropping interface.

---

## 3. Proposed Interfaces

### 3.1 KV Cache Compression (Serde Interface)

The Serde interface handles **compression/decompression** of KV caches during storage I/O — reducing storage size and transfer delay while preserving the logical tensor shape.

```python
@dataclass
class SerdeMeta:
    """Extensible metadata bag for serde operations.
    
    Wrapping metadata in a class keeps the interface stable as 
    new fields are added over time.
    """
    compression_ratio: float = 1.0   # target compression ratio (e.g., 0.25 for 4x compression)
    
    # Future fields (reserved, not yet populated)
    # user_id: Optional[str] = None
    # request_id: Optional[str] = None  
    # token_ids: Optional[torch.Tensor] = None


class Serde(abc.ABC):
    """Interface for KV cache serialization/deserialization.
    
    Invariant:
        serialize(k, v, buf, meta)
        buf.seek(0)
        k_out, v_out = deserialize(buf, meta)
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape
    """

    @abc.abstractmethod
    def serialize(
        self,
        k: torch.Tensor,        # [num_layers, num_tokens, num_heads, head_dim]
        v: torch.Tensor,        # [num_layers, num_tokens, num_heads, head_dim]
        out: BinaryIO,
        meta: SerdeMeta,
    ) -> None:
        """Serialize (and optionally compress) KV tensors to a binary stream."""
        ...

    @abc.abstractmethod
    def deserialize(
        self,
        inp: BinaryIO,
        meta: SerdeMeta,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deserialize (and decompress) KV tensors from a binary stream.
        
        Returns:
            (k, v) with original shapes restored.
        """
        ...
```

#### Where Serde executes in the storage pipeline

```
GPU KV Cache
    │
    ▼
┌──────────────┐
│ GPUConnector  │  ── copy KV from GPU to CPU
└──────┬───────┘
       ▼
┌──────────────┐
│ Serde        │  ── serialize (compress) before storage
│ (serialize)  │     e.g., quantization
└──────┬───────┘
       ▼
┌──────────────┐
│ Storage      │  ── write to CPU/disk/remote
│ Backend      │
└──────┬───────┘
       ▼
   (on read)
       │
┌──────────────┐
│ Serde        │  ── deserialize (decompress) after read
│ (deserialize)│
└──────┬───────┘
       ▼
┌──────────────┐
│ GPUConnector  │  ── copy KV from CPU back to GPU
└──────────────┘
```

### 3.2 KV Cache Compaction (Token Dropping Interface)

The compaction interface generates a **smaller** KV cache by evicting tokens, reducing both memory and compute.

```python
class TokenDropper(abc.ABC):
    """Interface for KV cache compaction via token eviction.
    
    The algorithm receives full q, k, v and returns compacted k, v
    with at most `preserved_num_tokens` tokens per layer.
    
    Constraint: each layer drops the same NUMBER of tokens, 
    but may drop DIFFERENT tokens.
    """

    @abc.abstractmethod
    def compact_kv(
        self,
        q: torch.Tensor,        # [num_layers, num_tokens, num_heads, head_dim]
        k: torch.Tensor,        # [num_layers, num_tokens, num_heads, head_dim]
        v: torch.Tensor,        # [num_layers, num_tokens, num_heads, head_dim]
        preserved_num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            k_out: [num_layers, <= preserved_num_tokens, num_heads, head_dim]
            v_out: [num_layers, <= preserved_num_tokens, num_heads, head_dim]
        """
        ...
```

---

## 4. Example: KV Cache Quantization as Compression

KV cache quantization (e.g., FP16 to INT4) fits naturally as a `Serde` implementation:

```python
class QuantizationSerde(Serde):
    """KV cache quantization (e.g., FP16 -> INT4) as a Serde plugin."""
    
    def __init__(self, key_bits: int = 4, value_bits: int = 4):
        self.key_bits = key_bits
        self.value_bits = value_bits
    
    def serialize(self, k, v, out, meta):
        # Quantize per-channel
        k_quant, k_scales, k_zeros = quantize(k, self.key_bits)
        v_quant, v_scales, v_zeros = quantize(v, self.value_bits)
        
        # Write header: original dtype, shapes
        header = struct.pack("ii", k.shape[0], k.shape[1])  # layers, tokens
        out.write(header)
        
        # Write quantized data + calibration params
        for tensor, scales, zeros in [
            (k_quant, k_scales, k_zeros),
            (v_quant, v_scales, v_zeros),
        ]:
            out.write(tensor.numpy().tobytes())
            out.write(scales.numpy().tobytes())
            out.write(zeros.numpy().tobytes())
    
    def deserialize(self, inp, meta):
        header = inp.read(8)
        num_layers, num_tokens = struct.unpack("ii", header)
        
        k = dequantize(read_quantized(inp), read_scales(inp), read_zeros(inp))
        v = dequantize(read_quantized(inp), read_scales(inp), read_zeros(inp))
        return k, v
```

---

## 6. Roadmap

| Step | Description | Notes |
|------|-------------|-------|
| 1 | Implement `Serde` base class and `SerdeMeta` | Core compression interface |
| 2 | Storage-level serde integration | Integrate into storage pipeline, L2 adapter |
| 3 | CPU-level serde support | Enable compression without GPU round-trips |
| 4 | KV cache quantization as first `Serde` plugin | Validate invariant: deserialize(serialize(k,v)) preserves shape |
| 5 | `TokenDropper` interface + k,v-only baseline | Compaction algorithms using only k and v (no q needed) |
| 6 | Q storage support (`get_qkv` / `put_qkv`) | Single-request prefill q, then multi-request and decode q |
| 7 | ML-side KV cache retrieval API | Let researchers access KV cache for experiments |
| 8 | Full orchestration integration | End-to-end compact + serve pipeline |

## Appendix: Design Constraints

We target algorithms satisfying two constraints:

1. **Shape Constraint (Aligned Token Dimension Across Layers)**: Each layer must retain the same *number* of tokens (though not necessarily the same *set* of tokens). This is a constraint from the vLLM side, which requires aligned token dimensions across layers for efficient memory management.

2. **Data Constraint (q, k, v Only)**: The optimization logic uses only queries (q), keys (k), and values (v) — not model weights, hidden states, or other auxiliary data. This covers the majority of existing KV cache optimization methods in the literature.
