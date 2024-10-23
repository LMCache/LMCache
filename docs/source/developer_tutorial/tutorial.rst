.. _dev_doc0:

Developer Documentation
================================================

The LMCache project can be decomposed into Three component: LLM Engine and LMCache Backend.

  Current, the LLM Engine (the ``lmcache-vllm`` repository ) is based on ``vllm`` with a customized wrapper.


When a new request arrives, the LLM Engine checks whether the prompt's KV Cache exists. If it does, the LLM Engine fetches the KV Cache from the LMCache Backend. If not, it performs the prefill for the prompt. After finishing the decoding process, the LLM Engine stores the tokens and their corresponding KV Cache in the LMCache Backend.

LLM Engine (``lmcache-vllm``)
----------------------------------------

The core idea is to build a customized vLLM using a wrapper. This wrapper imports all functions and CLI entry points from vLLM, while adding adaptation code to interact with the LM Cache to store or retrieve the KV Cache, and to start or shut down the LMCache Backend when the LLM Engine is initialized or closed.

* ``__init__.py``

  This file sets up a mechanism for dynamically importing modules. The ``ProxyModule`` and ``ModuleFinder`` classes allow the current package to serve as a proxy for the actual ``vllm`` modules, making imports seamless for users.

* ``script.py``

  This file invokes the vLLM's CLI entry point when the Engine client's entry point is called.

* ``vllm_injection.py``

  This file modifies two functions in ``vLLM``: ``vllm.worker.model_runner.ModelRunner.execute_model`` and ``vllm.engine.async_llm_engine._log_task_completion``.
  
  In the ``vllm.worker.model_runner.ModelRunner.execute_model`` module, we add the fetch and store command before and after model execution.

  .. code-block:: python

    # LMCache retrieval
    if lmcache_should_retrieve(model_input, kv_caches):
        model_input = lmcache_retrieve_kv(self.model, model_input, kv_caches)

    #... Model Execution
    
    # LMCache storing
    if lmcache_should_store(model_input, kv_caches):
        lmcache_store_kv(model_executable, model_input, kv_caches)

  Currently, we check whether the LMCache Backend has been initialized each time and initializes it if it hasn't. Later, we will initialize it when vllm initializes the model.

  .. code-block:: python

    init_lmcache_engine(self.model_config, self.parallel_config)


  In the ``vllm.engine.async_llm_engine._log_task_completion`` module, we shut down the LMCache Backend when a task is canceled, which only occurs upon program exit.

  .. code-block:: python

    except asyncio.exceptions.CancelledError:
        # We assume that if the task is cancelled, we are gracefully shutting
        # down. This should only happen on program exit.
        close_lmcache_engine()
        logger.info("Engine is gracefully shutting down.")

* ``vllm_adapter.py``



  The ``vllm_adapter.py`` file serves as the primary bridge between the LLM Engine and LMCache Backend. It defines key functions for interacting with LMCache Backend, including retrieving and storing key-value (KV) caches during the inference process. 

  In the  ``init_lmcache_engine`` function will initialize and return a LMCache engine based on the given model and parallel configuration if the Engine hasn't been initialized yet. Otherwise, it will return None. 

  .. code-block:: python

    def init_lmcache_engine(
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> Optional[LMCacheEngine]:

  The ``close_lmcache_engine`` function gracefully shuts down the LMCache engine by destroying the existing instance. 

  .. code-block:: python

    def close_lmcache_engine() -> None:

  The  ``lmcache_should_retrieve`` function determines whether to retrieve KV caches from LMCache based on the current model input and metadata. It checks if the KV Cache exists and whether the current run is performing a non-profiling prefill operation, and ensures that these conditions are met before retrieving the KV cache.

  .. code-block:: python
    
    def lmcache_should_retrieve(
        model_input: "ModelInputForGPUWithSamplingMetadata", 
        kv_caches: List[torch.Tensor]) -> bool:

  Similar to ``lmcache_should_retrieve``, the ``lmcache_should_store`` function checks if the KV cache should be stored in LMCache after the model execution. It evaluates metadata such as prefill states and ensures the conditions for storing are met.

  .. code-block:: python

    def lmcache_should_store(
        model_input: "ModelInputForGPUWithSamplingMetadata", 
        kv_caches: List[torch.Tensor]) -> bool:

  The ``lmcache_store_kv`` function is responsible for storing the KV cache in LMCache after the model execution. It sends the necessary data (input tokens and KV cache tensors) to the LMCache engine in a non-blocking way, using a CUDA stream for efficiency.

  .. code-block:: python

    def lmcache_store_kv(
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> None:

  The ``lmcache_retrieve_kv`` function retrieves KV caches from LMCache and rebuilds the model input to reflect the retrieved KV data. It integrates the retrieved cache with the current model input, ensuring the decoding process can continue seamlessly with the cached data.

  .. code-block:: python

    def lmcache_retrieve_kv(
        model_executable,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> "ModelInputForGPUWithSamplingMetadata":

  The ``build_partial_prefill_input`` function reconstructs the model input during the prefill stage when a partial prefill operation is needed. It rebuilds key components such as the input tokens, attention metadata, and sampling metadata, ensuring the model input is correctly aligned with the retrieved KV caches.

  .. code-block:: python

    def build_partial_prefill_input(
        model_input: "ModelInputForGPUWithSamplingMetadata",
        input_tokens_list: List[torch.Tensor],
        num_computed_tokens_list: List[int],
        start_pos_list: List[int],
        slot_mapping_flat: torch.Tensor,
        device: torch.device,
    ) -> "ModelInputForGPUWithSamplingMetadata":



  To summarize,when a new inference request arrives, the following steps occur:

  1. The LLM Engine checks if the KV cache for the prompt is already available by calling ``lmcache_should_retrieve``. If the cache is found, it retrieves the cached values using ``lmcache_retrieve_kv``.If the cache is not available, the model proceeds with prefill and decoding operations.

  2. After the decoding, ``lmcache_should_store`` checks if the KV cache should be stored in LMCache. If so, ``lmcache_store_kv`` stores the cache for future use.

  3. Throughout the lifecycle of the LLM Engine, the ``init_lmcache_engine`` and ``close_lmcache_engine`` functions ensure that the LMCache backend is initialized and shut down gracefully.


LMCache Backend (``LMCache``)
----------------------------------------

When LMCache call the Backend's ``lmcache_store_kv``, it give the KV Cache in shape of :math:`[layer\times 2\times tokens\times heads\times head\: size]`. For instance the KV Cache of 1024 tokens of the entire ``mistralai/Mistral-7B-Instruct-v0.2`` model is :math:`[32\times 2\times 1024\times 8\times 128]`.