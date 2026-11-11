MP recurrent checkpoint restoration
==================================

For a hybrid recurrent model in align mode, a CPU prefix restore can leave the
last state available while the next device prefix lookup needs an earlier state.
The optional history restoration path also restores complete earlier checkpoints
that remain in the MP server's L1 cache.

Add ``"lmcache.mp.restore_recurrent_history": true`` to the
``kv_connector_extra_config`` of ``LMCacheMPConnector`` on both scheduler and
workers. The default is false. Start the MP server from the same source revision;
the feature uses its existing server-module RPC envelope.

The initial supported configuration uses TP=2, DP=1, no PP or context
parallelism, one MP server and one L1 manager, prefix caching, align mode, and
equal hash/page/chunk sizes. The normal prefix restore remains unchanged.
Missing CPU checkpoints are skipped. If all complete historical states do not
fit beside existing local checkpoints, the entire extra restore is skipped.

Historical states use private device blocks and become prefix-cacheable only
after all worker copies have fenced successfully. Cancellation prevents
publication. The feature adds CPU transfers and a synchronous stream fence to
the first restore, so measure the first restore and subsequent reuses together.
It does not guarantee a throughput or end-to-end latency improvement.

For implementation and validation details, see the design document
``docs/design/integration/vllm/mp_recurrent_history.md`` in the source repository.
