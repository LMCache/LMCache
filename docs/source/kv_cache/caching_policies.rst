Using Different Caching Policies
===================================

LMCache supports multiple caching policies.

For example, to use LRU, you can set the environment variable ``LMCACHE_CACHE_POLICY=LRU`` or set it in the configuration file with ``cache_policy="LRU"``.

Currently, LMCache supports "LRU" (Least Recently Used), "LFU" (Least Frequently Used) and "FIFO" (First-In-First-Out) caching policies.