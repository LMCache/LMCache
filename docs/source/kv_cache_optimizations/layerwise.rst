# Layerwise KV Transfer


The storage and loading of KV Cache on a layer granularity is a key optimization that allows for forward pass to "stagger" through its computation as each layer's KV Cache is received instead of only waiting to begin after the entire loading
CacheBlend is implemented on top of the layerwise codepath in order to pipeline recompute and loading to mask the latency of loading KV Cache.
Both the vLLM integration and the SGLang integration support the layerwise optimization.


## vLLM Integration:


## SGLang Integration:


## CacheBlend Module:
