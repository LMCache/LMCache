.. _offload_kv_cache:

Example: Offload KV cache to CPU
================================

In this example, we will show you how to offload KV cache to CPU memory.

.. note::
    Besides CPU memory, LMCache also supports offloading KV cache to many different destinations.
    See :ref:`getting_started/quickstart/offload_kv_cache:Supported offloading destinations` for more details.

Prerequisites
-------------

Before you begin, make sure you have:

- vLLM v1 with LMCache installed (see :doc:`Installation <../installation>`)
- A GPU that can run a LLM
- `Logged into HuggingFace <https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login>`_ using a token with gated access permission (required for model downloads)


Use CPU offloading in offline inference
---------------------------------------

This section demonstrates how to use CPU memory offloading in offline inference scenarios using LMCache with vLLM.
The example script we use here is available in `vLLM examples <https://github.com/vllm-project/vllm/blob/main/examples/others/lmcache/cpu_offload_lmcache.py>`_.
See the `examples README <https://github.com/vllm-project/vllm/tree/main/examples/others/lmcache#2-cpu-offload-examples>`_ to understand how to run the script for vLLM v1.

First, set up the necessary environment variables for LMCache:

.. code-block:: python

    import os

    # Set token chunk size to 256
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    # Enable CPU memory backend
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    # Set CPU memory limit to 5GB
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"

Next, configure vLLM with LMCache integration:

.. code-block:: python

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    # Configure KV cache transfer to use LMCache
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    # Initialize LLM with LMCache configuration
    # Adjust gpu_memory_utilization based on your GPU memory
    llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
              kv_transfer_config=ktc,
              max_model_len=8000,
              gpu_memory_utilization=0.8)

Now you can run inference with automatic KV cache offloading:

.. code-block:: python

    # Create example prompts with shared prefix
    shared_prompt = "Hello, how are you?" * 1000
    prompts = [
        shared_prompt + "Hello, my name is",
    ]

    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    # Run inference
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")

When the inference is complete, clean up the LMCache backend:

.. code-block:: python

    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.integration.vllm.utils import ENGINE_NAME

    LMCacheEngineBuilder.destroy(ENGINE_NAME)

During inference, LMCache will automatically handle storing and managing KV cache in CPU memory. You can monitor this through the logs, which will show messages like::

    LMCache INFO: Storing KV cache for 6006 out of 6006 tokens for request 0

This indicates that the KV cache has been successfully offloaded to CPU memory.

.. note::
    - Adjust ``gpu_memory_utilization`` based on your GPU's available memory
    - The CPU offloading buffer size can be adjusted through ``LMCACHE_MAX_LOCAL_CPU_SIZE``

Use CPU offloading in online inference
--------------------------------------

This section demonstrates how to use CPU memory offloading in online serving scenarios. The setup involves two main steps: creating a configuration file and launching the vLLM server.

First, create a configuration file named ``lmcache_config.yaml`` with the following content:

.. code-block:: yaml

    # Basic configurations
    chunk_size: 256
    
    # CPU offloading configurations
    local_cpu: true
    max_local_cpu_size: 5.0  # 5GB CPU memory limit
    
Next, launch the vLLM server with LMCache integration. Here's an example command:

.. code-block:: bash

    LMCACHE_CONFIG_FILE=/path/to/lmcache_config.yaml \
    vllm serve \
        meta-llama/Llama-3.1-8B-Instruct \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1",
          "kv_role":"kv_both"
        }'

Key parameters explained:

- ``LMCACHE_CONFIG_FILE``: Path to the LMCache configuration file.
- ``--kv-transfer-config``: Configures LMCache integration
    - ``kv_connector``: Specifies the LMCache connector 
    - ``kv_role``: Set to "kv_both" for both storing and loading KV cache

Once the server is running, you can send requests to it using curl. Here's an example of how to send a request to the vLLM server with LMCache integration:

.. code-block:: bash

    curl http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "<|begin_of_text|><|system|>\nYou are a helpful AI assistant.\n<|user|>\nWhat is the capital of France?\n<|assistant|>",
        "max_tokens": 100,
        "temperature": 0.7
      }'

You should see the following logs:

.. code-block:: text
    :emphasize-lines: 1

    LMCache INFO: Storing KV cache for 31 out of 31 tokens for request cmpl-274bcaa80837444dbf9fbba4155d2620-0 (vllm_v1_adapter.py:497:lmcache.integration.vllm.vllm_v1_adapter)

Once you send the same curl request again, you should see the following logs:

.. code-block:: text
    :emphasize-lines: 1

    LMCache INFO: Reqid: cmpl-4ddf8863a6ac4dc3b6a952f2a107e9b2-0, Total tokens 31, LMCache hit tokens: 30, need to load: 14 (vllm_v1_adapter.py:543:lmcache.integration.vllm.vllm_v1_adapter)


Example: CPU offloading benefits
--------------------------------

This section demonstrates the performance benefits of using CPU offloading with LMCache. We'll use a script that generates multiple prompts and compare the performance with and without LMCache.

Prerequisites (Setup)
~~~~~~~~~~~~~~~~~~~~~~

- At least 24GB GPU memory
- Access to model ``meta-llama/Meta-Llama-3.1-8B-Instruct``
- Sufficient CPU memory (LMCache will use 15 GB by default in this example).

Example script
~~~~~~~~~~~~~~

Save the following script as ``cpu-offloading.py``:

.. code-block:: python

    # SPDX-License-Identifier: Apache-2.0
    """
    This file demonstrates the example usage of cpu offloading
    with LMCache in vLLM v1.

    Note that lmcache needs to be installed to run this example.
    Learn more about LMCache in https://github.com/LMCache/LMCache.
    """
    import os
    import torch
    import argparse
    import time
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    def parse_arguments():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="CPU offloading example with LMCache")
        parser.add_argument("--num-prompts", type=int, default=10,
                          help="Number of prompts to generate (default: 10)")
        parser.add_argument("--num-tokens", type=int, default=10000,
                          help="Number of tokens per prompt (default: 10000)")
        parser.add_argument("--enable-lmcache", action="store_true",
                          help="Enable LMCache for CPU offloading (default: True)")
        return parser.parse_args()

    def setup_lmcache_environment(num_prompts, num_tokens):
        """
        Configure LMCache environment variables.
        Args:
            num_prompts: Number of prompts to process
            num_tokens: Number of tokens per prompt
        """
        cpu_size = num_prompts * num_tokens * 1.5 / 10000  # 1.5GB per 10000 tokens
        
        env_vars = {
            "LMCACHE_CHUNK_SIZE": "256",         # Set tokens per chunk
            "LMCACHE_LOCAL_CPU": "True",         # Enable local CPU backend
            "LMCACHE_MAX_LOCAL_CPU_SIZE": str(cpu_size)  # Dynamic CPU memory limit (GB)
        }
        for key, value in env_vars.items():
            os.environ[key] = value

    def calculate_gpu_utilization(target_memory_gb=24):
        """
        Calculate GPU memory utilization to use exactly target_memory_gb of GPU memory.
        Args:
            target_memory_gb: Target GPU memory usage in gigabytes
        Returns:
            float: GPU memory utilization ratio (0.0 to 1.0)
        Raises:
            RuntimeError: If GPU memory is less than target_memory_gb
        """
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available")
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        if total_memory < target_memory_gb:
            raise RuntimeError(f"GPU memory ({total_memory:.1f}GB) is less than required memory ({target_memory_gb}GB)")
        
        return target_memory_gb / total_memory

    def create_test_prompts(num_prompts=10, num_tokens=1000):
        """
        Create test prompts with index prefix and dummy body.
        Args:
            num_prompts: Number of prompts to generate
            num_tokens: Approximate number of tokens per prompt (using 'Hi ' as token unit)
        Returns:
            list: List of prompts with format '[index] Hi Hi Hi...'
        """
        prompts = []
        dummy_text = "Hi " * num_tokens
        
        for i in range(num_prompts):
            prompt = f"[Prompt {i}] {dummy_text} how are you?"
            prompts.append(prompt)
        
        return prompts

    def initialize_llm(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", max_len=16384, enable_lmcache=True):
        """
        Initialize the LLM with appropriate configurations.
        Args:
            model_name: Name of the model to load
            max_len: Maximum sequence length
        Returns:
            LLM: Configured LLM instance
        """
        ktc = KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role="kv_both",
        ) if enable_lmcache else None
        
        return LLM(
            model=model_name,
            kv_transfer_config=ktc,
            max_model_len=max_len,
            enable_prefix_caching=False,
            gpu_memory_utilization=calculate_gpu_utilization()
        )

    def generate_and_print_output(llm, prompts, sampling_params):
        """
        Generate text and print the results.
        Args:
            llm: LLM instance
            prompts: List of input prompts
            sampling_params: Configured sampling parameters
        Returns:
            float: Time taken for generation in seconds
        """
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()
        
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Generated text: {generated_text!r}")
        
        return end_time - start_time

    def main():
        """Main execution function."""
        # Parse command line arguments
        args = parse_arguments()
        
        # Setup environment if LMCache is enabled
        if args.enable_lmcache:
            setup_lmcache_environment(args.num_prompts, args.num_tokens)
        
        # Create prompts and sampling parameters
        prompts = create_test_prompts(num_prompts=args.num_prompts, num_tokens=args.num_tokens)
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
        
        # Initialize model
        llm = initialize_llm(enable_lmcache=args.enable_lmcache)
        
        # First run
        print("\nFirst run:")
        first_run_time = generate_and_print_output(llm, prompts, sampling_params)
        print(f"First run time: {first_run_time:.2f} seconds")
        
        # Second run
        print("\nSecond run:")
        second_run_time = generate_and_print_output(llm, prompts, sampling_params)
        print(f"Second run time: {second_run_time:.2f} seconds")
        
        # Print speedup
        if first_run_time > 0:
            speedup = first_run_time / second_run_time
            print(f"\nSpeedup (first run / second run): {speedup:.2f}x")
        
        # Cleanup if LMCache was enabled
        if args.enable_lmcache:
            LMCacheEngineBuilder.destroy(ENGINE_NAME)

    if __name__ == "__main__":
        main()

Running the Example
~~~~~~~~~~~~~~~~~~~

1. First, run the script without LMCache:

   .. code-block:: bash

       python cpu-offloading.py 

   You'll see output like:

   .. code-block:: text

       Speedup (first run / second run): 1.00x

   Without LMCache, there's no speedup between runs even if vLLM has prefix caching enabled.
   This is because the KV cache exceeds GPU memory and can't be reused.

2. Now, run with LMCache enabled:

   .. code-block:: bash

       python cpu-offloading.py --enable-lmcache

   You'll see output like:

   .. code-block:: text

       Speedup (first run / second run): 7.43x

The significant speedup in the second case demonstrates how LMCache effectively manages KV cache offloading to CPU memory. 
When the total size of KV cache exceeds GPU memory, LMCache allows you to store and reuse the cache from CPU memory, 
resulting in much faster subsequent generations for prompts with shared prefixes.


Supported offloading destinations
---------------------------------

LMCache now supports offloading KV cache to the following destinations:

- :doc:`CPU memory <../../kv_cache/storage_backends/cpu_ram>`
- :doc:`Local file system <../../kv_cache/storage_backends/local_storage>`
- :doc:`Mooncake Storage <../../kv_cache/storage_backends/mooncake>`
- :doc:`InfiniStore <../../kv_cache/storage_backends/infinistore>`
- :doc:`Redis <../../kv_cache/storage_backends/redis>`
- :doc:`ValKey <../../kv_cache/storage_backends/valkey>`
