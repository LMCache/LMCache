.. _dev_doc_LMCache_Engine:

LMCache Engine
------------------------

The LMCache Engine has two mean function : ``store`` and ``retrieve``.


The ``store`` Function :

* Breaks input tokens and KV caches into manageable chunks.
* Store the chunks into dictionary, which is managed by the LMCache Backend
* Uses prefix token hashes (Currently `sha256`) to index the chunks, combined with other arguments (e.g., format) to form the key.
* Efficiently stores the KV caches while avoiding redundancy (if ``skip_existing=True``).
* The store operation can be blocking or non-blocking depending on the ``blocking`` argument.

The ``retrieve`` Function:

* Retrieves KV caches for the input tokens, using the same chunking and hashing mechanism as ``store``.
* Supports partial retrieval via the ``mask`` parameter, allowing retrieval of suffixes or specific portions of the token sequence.
* Concatenates the retrieved KV cache chunks into a usable format for model inference.

The details of ``LMCacheEngine`` class are listed below.

.. automodule:: lmcache.cache_engine
   :members: LMCacheEngine
   :undoc-members:
   :show-inheritance:
   :private-members:

.. Attributes
.. ^^^^^^^^^^^^^

.. * **config** (``LMCacheEngineConfig``): Configuration settings for the cache engine.
.. * **metadata** (``LMCacheEngineMetadata``): Metadata about the language model and cache engine.
.. * **chunk_size** (``int``): The size of each chunk when splitting tokens and KV caches.
.. * **save_decode_cache** (``bool``): Indicates whether to save the decode cache.
.. * **engine_** (``StorageBackend``): The underlying storage backend for the cache engine.

.. Methods
.. ^^^^^^^^^^^^^

.. * ``__init__(self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata)``
  

..   Initializes the ``LMCacheEngine`` with the given configuration and metadata.

..   **Args:**

..   * ``config`` (``LMCacheEngineConfig``): Configuration settings for the cache engine.
..   * ``metadata`` (``LMCacheEngineMetadata``): Metadata about the language model and cache engine.

..   **Raises:**

..   * ``RuntimeError``: If the loaded configuration does not match the current configuration.

.. * ``_make_key(self, chunk_hash: str, fmt: str) -> CacheEngineKey``
  

..   Constructs a ``CacheEngineKey`` for a given chunk hash and format.

..   **Args:**

..   * ``chunk_hash`` (``str``): The hash of the token chunk.
..   * ``fmt`` (``str``): The format of the KV cache (``huggingface`` or ``vllm``).

..   **Returns:**

..   * ``CacheEngineKey``: The constructed cache key.

.. * ``_num_tokens_in_kv(self, kv_tensors: Union[KVCache, torch.Tensor], fmt: str) -> int``
  

..   Determines the number of tokens in the KV cache tensors based on the format.

..   **Args:**

..   * ``kv_tensors`` (``Union[KVCache, torch.Tensor]``): The KV cache tensors.
..   * ``fmt`` (``str``): The format of the KV cache (``huggingface`` or ``vllm``).

..   **Returns:**

..   * ``int``: The number of tokens in the KV cache.

..   **Raises:**

..   * ``ValueError``: If the format is invalid.

.. * ``_get_init_hash(self) -> str``
  

..   Returns the initial hash value for prefix hashing.

..   **Returns:**

..   * ``str``: The initial hash value (empty string).

.. * ``_hash(self, tokens: torch.Tensor, prefix_hash: str) -> str``
  

..   Computes a hash value for a given token chunk and a prefix hash.

..   **Args:**

..   * ``tokens`` (``torch.Tensor``): The token chunk tensor.
..   * ``prefix_hash`` (``str``): The hash of the previous chunk.

..   **Returns:**

..   * ``str``: The computed hash value for the current chunk.

..   **Note:**

..   * Currently uses SHA-256 for hashing. May be replaced with a more efficient hash function in the future.

.. * ``_chunk_tokens(self, tokens: torch.Tensor) -> Iterable[torch.Tensor]``
  

..   Splits the input tokens into chunks of size ``self.chunk_size``.

..   **Args:**

..   * ``tokens`` (``torch.Tensor``): The input tokens tensor of shape ``[seq_len]``.

..   **Yields:**

..   * ``torch.Tensor``: Chunks of tokens, each of shape ``[chunk_size]``.

..   **Note:**

..   * This operation can potentially be parallelized for efficiency.

.. * ``_prefix_hash(self, token_chunks: Iterable[torch.Tensor], num_skip_chunk: Optional[int] = 0) -> List[str]``
  

..   Computes prefix hashes for a sequence of token chunks.

..   **Args:**

..   * ``token_chunks`` (``Iterable[torch.Tensor]``): An iterable of token chunks.
..   * ``num_skip_chunk`` (``Optional[int]``, optional): Number of initial chunks to skip. Defaults to ``0``.

..   **Returns:**

..   * ``List[str]``: A list of prefix hashes starting from ``num_skip_chunk``.

.. * ``_tuple_kv_to_blob(self, kv_tensors: KVCache) -> torch.Tensor``
  

..   Converts a nested tuple of KV tensors into a single tensor blob with two extra dimensions.

..   **Args:**

..   * ``kv_tensors`` (``KVCache``): The KV cache tensors in nested tuple format.

..   **Returns:**

..   * ``torch.Tensor``: The KV tensors flattened into a single tensor.

.. * ``_blob_to_tuple_kv(self, blob: torch.Tensor) -> KVCache``
  

..   Converts a single tensor blob back into a nested tuple of KV tensors.

..   **Args:**

..   * ``blob`` (``torch.Tensor``): The KV tensors blob.

..   **Returns:**

..   * ``KVCache``: The KV cache tensors in nested tuple format.

.. * ``_slice_kv_at(self, start_idx: int, kv_tensors: torch.Tensor, fmt: str) -> List[torch.Tensor]``
  

..   Slices the KV tensors starting from a specific token index.

..   **Args:**

..   * ``start_idx`` (``int``): The starting token index.
..   * ``kv_tensors`` (``torch.Tensor``): The KV cache tensors.
..   * ``fmt`` (``str``): The format of the KV cache (``huggingface`` or ``vllm``).

..   **Returns:**

..   * ``List[torch.Tensor]``: A list of sliced KV tensors.

..   **Raises:**

..   * ``ValueError``: If the format is invalid.

.. * ``_chunk_kv(self, kv_tensors: torch.Tensor, fmt: str) -> Iterable[torch.Tensor]``
  

..   Splits the KV cache tensors into chunks of size ``self.chunk_size``.

..   **Args:**

..   * ``kv_tensors`` (``torch.Tensor``): The KV cache tensors.
..   * ``fmt`` (``str``): The format of the KV cache (``huggingface`` or ``vllm``).

..   **Yields:**

..   * ``torch.Tensor``: Chunks of KV cache tensors.

.. * ``_make_chunks_skip_existing(self, tokens: torch.Tensor, kv_tensors: torch.Tensor, fmt: str) -> Iterable[Tuple[str, torch.Tensor]]``
  

..   Skips existing chunks in the cache and returns the rest of the chunks along with their hashes.

..   **Args:**

..   * ``tokens`` (``torch.Tensor``): The input tokens tensor.
..   * ``kv_tensors`` (``torch.Tensor``): The KV cache tensors.
..   * ``fmt`` (``str``): The format of the KV cache (``huggingface`` or ``vllm``).

..   **Returns:**

..   * ``Iterable[Tuple[str, torch.Tensor]]``: An iterable of ``(chunk_hash, kv_chunk)`` tuples for new chunks.

.. * ``_make_chunks(self, tokens: torch.Tensor, kv_tensors: torch.Tensor, fmt: str, skip_existing=True) -> Iterable[Tuple[str, torch.Tensor]]``
  

..   Creates chunks of tokens and KV caches, optionally skipping existing chunks.

..   **Args:**

..   * ``tokens`` (``torch.Tensor``): The input tokens tensor.
..   * ``kv_tensors`` (``torch.Tensor``): The KV cache tensors.
..   * ``fmt`` (``str``): The format of the KV cache (``huggingface`` or ``vllm``).
..   * ``skip_existing`` (``bool``, optional): Whether to skip chunks that already exist in the cache. Defaults to ``True``.

..   **Returns:**

..   * ``Iterable[Tuple[str, torch.Tensor]]``: An iterable of ``(chunk_hash, kv_chunk)`` tuples.

.. * ``store(self, tokens: torch.Tensor, kv_tensors_raw: KVCache, skip_existing=True, blocking=True) -> None``
  

..   Stores the KV cache of the input tokens into the cache engine.

..   **Args:**

..   * ``tokens`` (``torch.Tensor``): The input tokens tensor of shape ``[seq_len]``.
..   * ``kv_tensors_raw`` (``KVCache``): The KV cache tensors in nested tuple format.
..   * ``skip_existing`` (``bool``, optional): Whether to skip storing chunks that already exist. Defaults to ``True``.
..   * ``blocking`` (``bool``, optional): Whether the store operation should be blocking. Defaults to ``True``.

..   **Returns:**

..   * ``None``

..   **Note:**

..   * The KV cache should not include the batch dimension.
..   * The format is determined by ``self.metadata.fmt`` and should be either ``huggingface`` or ``vllm``.
..   * For ``huggingface``, KV tensors should have shape ``[num_layers, 2, num_kv_heads, num_tokens, head_size]``.
..   * For ``vllm``, KV tensors should have shape ``[num_layers, 2, num_tokens, num_kv_heads, head_size]``.

..   **Raises:**

..   * ``AssertionError``: If the number of tokens in the KV cache does not match the input tokens.

.. * ``retrieve(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[KVCache, torch.Tensor]``
  

..   Retrieves the KV cache of the input tokens from the cache engine.

..   **Args:**

..   * ``tokens`` (``torch.Tensor``): The input tokens tensor of shape ``[seq_len]``.
..   * ``mask`` (``Optional[torch.Tensor]``, optional): A boolean mask indicating which tokens' KV cache should be retrieved. Currently, only
