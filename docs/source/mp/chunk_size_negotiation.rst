Chunk-Size Negotiation
======================

The vLLM multiprocess connector negotiates a concrete LMCache chunk size at
startup. This keeps chunk boundaries aligned with every vLLM KV cache group,
including hybrid-attention, recurrent-state, and decode-context-parallel
(DCP) layouts.

Final chunk size
----------------

``lmcache server --chunk-size`` is the minimum, not necessarily the final
value. The connector computes the effective token span of every cacheable KV
group, excluding scratch groups, and sends their least common multiple as the
required alignment::

    required_alignment = lcm(group_tokens_per_block...)
    final_chunk_size = smallest multiple of required_alignment
                       that is >= --chunk-size

For attention groups under DCP, the effective span is
``block_size * decode_context_parallel_size``. Recurrent-state groups use
their physical block size because their state is replicated rather than DCP
sharded.

For example, with ``--chunk-size 256``:

* an alignment of 256 resolves to a final chunk size of 256;
* an alignment of 544 resolves to 544;
* an alignment of 544 with ``--chunk-size 600`` resolves to 1088; and
* group spans 32 and 48 have an alignment of 96, so the final size is 288.

One MP server has one immutable final chunk size. After the first client
selects it, another model can share that server only when the selected size is
also a multiple of the new model's required alignment. Use separate MP
servers for incompatible models. When a vLLM deployment uses multiple MP
servers, every server must return the same final size.

Initialization flow
-------------------

#. vLLM resolves its KV cache groups. ``LMCacheMPConnector`` computes the
   required alignment from their effective token spans.
#. The scheduler adapter and each worker adapter send
   ``NEGOTIATE_CHUNK_SIZE`` before issuing normal cache operations.
#. The MP server rounds its configured minimum up, then binds the resolved
   size to the token hasher, session manager, optional CacheBlend matcher,
   telemetry, and coordinator membership.
#. The server returns the final size. The connector verifies that it is
   compatible with every cache group; a multi-server scheduler also verifies
   that all servers agree.
#. Workers register their KV caches using the negotiated size.

The MP server rejects a resize after sessions have started. It also does not
report negotiation success until every chunk-size binding callback succeeds;
a retry resumes at the first callback that did not complete.

Restart recovery
----------------

The scheduler and worker adapters keep the negotiated size in memory. Their
heartbeat threads detect an unhealthy-to-healthy transition after an MP
server restart and perform recovery before marking that server healthy:

#. Re-negotiate using the original required alignment.
#. Verify that the recovered size exactly matches the adapter's cached size.
#. On workers, re-register the KV cache handles and any experimental
   dispatcher state.

If any step fails, the server remains unhealthy from that adapter's point of
view and normal cache operations stay in degraded mode. Recovery is attempted
again after the next successful heartbeat. The heartbeat interval defaults to
10 seconds and can be changed with
``lmcache.mp.heartbeat_interval`` in ``kv_connector_extra_config``.
