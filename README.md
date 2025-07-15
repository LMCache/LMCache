### Branch

- end-to-end: for main development
- storage-varieties: for figure to show CacheServe performs under different - caching devices
- cacheserve: for Siddhant's graph

### Ablation study

- method:
    Change `lmcache/experimental/token_database.py` line 318 from

    ```python
    if kivi_min_unit_quality_drop <= streaming_min_unit_quality_drop:
    ```

    to

    ```python
    if random.choice([True, False]):
    ```

- device:
    Add to `lmcache/experimental/storage_backend/storage_manager.py` line 165:

    ```python
    if random.choice([True, False]):
        return KVDecision("disk", "kivi", key.metadata.rate), {}
    ```

    So now random choose to save to CPU or disk.

- rate:
    Change `lmcache/experimental/storage_backend/storage_manager.py` line 53 from

    ```python
    for r_cand in candidate_rates:
    ```

    to

    ```python
    if orig_rate == 1:
        r_cands = random.choice(list(candidate_rates))
    else:
        r_cands = [0]
    for r_cand in [r_cands]:
    ```

    Change `lmcache/experimental/storage_backend/storage_manager.py` line 79 from

    ```python
    if rate > baseline_rate:
    ```

    to

    ```python
    if rate != baseline_rate:
    ```
