# Examples vLLM + LMCache w. CacheBlend
LMCache should be able to reduce the generation time of the second and following calls (even though the reused KV cache is not a prefix).

## Requirements
When using CacheBlend, you must provide the following configuration:
- `blend_check_layers`: List of layer indices to check for blending (e.g., `[1]`)
- `blend_recompute_ratios`: List of ratios for recomputation (e.g., `[0.15]`)

Example config:
```yaml
enable_blending: true
blend_check_layers: [1]
blend_recompute_ratios: [0.15]
```

## CPU offloading
- `python blend.py` - CacheBlend with CPU as backend
## Disk offloading
- `python blend.py --use-disk` - CachBlend with local disk as backend