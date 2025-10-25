TroubleShoot
============

.. contents::
   :local:
   :depth: 2
   :backlinks: none
   
---------------------------------
🕒 2025-08
---------------------------------

**🧭 Time**
   2025-08-14

**🚨 Issue**
    [xPyD][lmcache0.3.3+vllm0.10.0] "failed to allocate memory for tensor" during benchmark with lmcache xPyD version

**📋 Description**
    When running a vllm benchmark on the xPyD version of lmcache with the model Qwen3-Coder-480B-A35B-Instruct-FP8 in tensor parallel = 8 configuration, the process eventually fails with the following error:
    Failed to allocate memory for tensor(torch.Size([2, 62, 13, 128]), torch.bfloat16)
    because no free blocks is available (memory_management.py:992:lmcache.v1.memory_management)
    
    .. image:: https://private-user-images.githubusercontent.com/43373176/477835691-2937693a-f00b-4a1d-b573-0f9eb4b4bf5a.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjA3OTQ5MDEsIm5iZiI6MTc2MDc5NDYwMSwicGF0aCI6Ii80MzM3MzE3Ni80Nzc4MzU2OTEtMjkzNzY5M2EtZjAwYi00YTFkLWI1NzMtMGY5ZWI0YjRiZjVhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMTglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDE4VDEzMzY0MVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWE5OTNkMGIyM2QzZTkxNDE5ODg1MjIwNWI5MmVjMjFjZTA0ZGM0NDdjOTU1ODUyMTg0NDRiYjBkMzRhNTdmZWEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.wQU8zx5GHVJekIciS8q4ukDD3gXKWHkp0BsO6w47eeo
        :alt: Log information
        :width: 100%
        :align: center

    The Decoder instance eventually fails with the error after processing a cumulative ~100+ requests. The benchmark stream is configured with max-concurrency=2, so the failure is not caused by high instantaneous concurrency. It appears that the Nixl buffer may be filled up？Does the Nixl buffer have a garbage collection (GC) mechanism?

    lmcache config:

    .. code-block:: yaml

        enable_nixl: True
        enable_xpyd: True
        nixl_buffer_size: 1080819712
    
    **Steps to Reproduce**
    Launch prefill process:

    .. code-block:: bash

        LMCACHE_CONFIG_FILE=$prefill_config_file \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        LMCACHE_LOG_LEVEL=DEBUG \
        VLLM_LOGGING_LEVEL=DEBUG \
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        PYTHONHASHSEED=0 \
        NCCL_P2P_DISABLE=1 \
        vllm serve $MODEL \
        --port 8100 \
        --enforce-eager \
        --max-model-len 131072 \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.8 \
        --enable-expert-parallel \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}'
    
    Launch decode process:

    .. code-block:: bash

        python3 lmcache_proxy.py \
        --host 0.0.0.0 \
        --port 9100 \
        --prefiller-host localhost \
        --prefiller-port 8100 \
        --num-prefillers 1 \
        --decoder-host ${decoder_ip} \
        --decoder-port 8200  \
        --decoder-init-port ${decoder_init_port}\
        --decoder-alloc-port ${decoder_alloc_port} \
        --proxy-host 0.0.0.0 \
        --proxy-port 7500 \
        --num-decoders 1
    
    Launch bench

    .. code-block:: bash

        vllm bench serve --port 9100 --seed 122 \
        --model /model \
        --dataset-name random --random-input-len 200 --random-output-len 200 \
        --num-prompts 200 --max-concurrency 2

**🧩 Environment**
    vllm: 0.10.0 v1
    lmcahe: 0.3.3
    Model: Qwen3-Coder-480B-A35B-Instruct-FP8
    P/D: 1P1D

**🟠 Status:**  In Progress

----

**🧭 Time**
   2025-08-15

**🚨 Issue**
    LMCache seems to be using the wrong CUDA devices with Ray+PP

**📋 Description**

    **Setup Context**
    We got 2 nodes. Each node has 8 GPUs. Both nodes and all 16 GPUs are shown as resources under ``ray status``.

    **Reproducer & Error Message**
    Running the following  ``vllm`` command:

    .. code-block:: bash

        VLLM_WORKER_MULTIPROC_METHOD=spawn  LMCACHE_USE_EXPERIMENTAL=True  vllm serve meta-llama/Llama-3.1-70B-Instruct  --gpu-memory-utilization 0.7  --tensor_parallel_size 8  --pipeline_parallel_size 2  --no-enable-chunked-prefill  --no-enable-prefix-caching --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config": {}}' --distributed-executor-backend ray
    
    causes the following error in 8 of the nodes:
    
    .. code-block:: text
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]   File "/home/amg/vllm/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 181, in init_device                                             
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]     init_worker_distributed_environment(self.vllm_config, self.rank,                                                                                          
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]   File "/home/amg/vllm/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 584, in init_worker_distributed_environment                     
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]     ensure_kv_transfer_initialized(vllm_config)                                                                                                               
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]   File "/home/amg/vllm/.venv/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_transfer_state.py", line 64, in ensure_kv_transfer_initialized      
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]     _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector_v1(                                                                                             
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                             
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]   File "/home/amg/vllm/.venv/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/factory.py", line 84, in create_connector_v1              
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]     return connector_cls(config, role)                                                                                                                        
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                                                        
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]   File "/home/amg/vllm/.venv/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py", line 27, in __init__            
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]     self._lmcache_engine = LMCacheConnectorV1Impl(vllm_config, role, self)                                                                                    
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                    
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]   File "/home/amg/amg_stable/LMCache/lmcache/integration/vllm/vllm_v1_adapter.py", line 553, in __init__                                                      
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]     self.lmcache_engine = init_lmcache_engine(
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]                           ^^^^^^^^^^^^^^^^^^^^
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]   File "/home/amg/amg_stable/LMCache/lmcache/integration/vllm/vllm_v1_adapter.py", line 495, in init_lmcache_engine
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]     vllm_gpu_connector = VLLMPagedMemGPUConnectorV2(
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]   File "/home/amg/amg_stable/LMCache/lmcache/v1/gpu_connector.py", line 134, in __init__
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]     self.gpu_buffer = torch.empty(
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619]                       ^^^^^^^^^^^^
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619] RuntimeError: CUDA error: invalid device ordinal
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619] For debugging consider passing CUDA_LAUNCH_BLOCKING=1
        (RayWorkerWrapper pid=3182670, ip=10.192.207.175) ERROR 08-14 17:21:25 [worker_base.py:619] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

    **Potential Root Cause**
    Given that some of the workers were fine while other ones gave the above message made me think that there's something wrong how GPUs are assigned in the second node. 
    So I added the following debug message in ``vllm_v1_adapter.py``:

    .. code-block:: python

            torch.cuda.device(parallel_config.rank)
            logger.info(f"DEBUG---{parallel_config.rank}")
            device = torch.device(f"cuda:{parallel_config.rank}")

    and in the ouput right before the error I'd see things like:

    .. code-block:: text

        DEBUG---8
        ...error...
        DEBUG---9
        ...error...
    I believe the issue is that in the second node the global rank numbers can be 8,9,10,11,12,13,14,15 but the corresponding device IDs should be 0,1,2,3,4,5,6,7,8.

    **Attempt to fix (update: it didn't work)**
    Tried this but it didn't work:

    .. code-block:: python

        local_rank = envs.LOCAL_RANK
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

**🟠 Status:**  In Progress

----

**🧭 Time**
   2025-08-15

**🚨 Issue**
    I've deployed vLLM 0.10.0 integrated with LMCache and Mooncake, DP=4 and TP=2, PD disaggregation.
    I'm encountering frequent errors on the Decode node indicating that it fails to retrieve the kvcache from the Mooncake store.

**📋 Description**
    In the current implementation, storing the kv cache into the Mooncake is an asynchronous operation. The function returns immediately without waiting for the kvcache to be actually written into the Mooncake store. As a result, after the store operation is marked as completed (but before the kv data is truly persisted in Mooncake), the request completion signal is sent up to the vLLM Engine and API server, and eventually back to the proxy server.
    Once the proxy receives the "prefill completed" signal, it sends the request to the Decode node. Then, the Decode node tries to fetch the corresponding kvcache from Mooncake, but it's not yet ready, leading to an error.

    To reproduce:
    - Prefill command

    .. code-block:: bash

        PYTHONHASHSEED=123 \
        LMCACHE_LOG_LEVEL=DEBUG \
        LMCACHE_CONFIG_FILE=/data/lwh/mooncake-config-prefill-TP2DP4.yaml \
        LMCACHE_USE_EXPERIMENTAL=True \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        vllm serve /data/Meta-Llama-3-8B-Instruct \
        --port 8100 \
        --disable-log-requests \
        --enforce-eager \
        --data-parallel-size 4 \
        --tensor-parallel-size 2 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}'

    - mooncake-config-prefill-TP2DP4.yaml

    .. code-block:: yaml

        chunk_size: 256
        local_device: "cpu"
        remote_url: "mooncakestore://200.10.0.22:50051/"
        remote_serde: "naive"
        local_cpu: False
        max_local_cpu_size: 5

        extra_config:
        local_hostname: "200.10.0.22"
        metadata_server: "http://200.10.0.22:8080/metadata"
        protocol: "rdma"
        device_name: "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3"
        master_server_address: "200.10.0.22:50051"
        global_segment_size: 3355443200
        local_buffer_size: 1073741824
        transfer_timeout: 1

    - Decode Command

    .. code-block:: bash

         PYTHONHASHSEED=123 \
        LMCACHE_LOG_LEVEL=DEBUG \
        LMCACHE_CONFIG_FILE=/data/lwh/mooncake-config-decode-TP2DP4.yaml \
        LMCACHE_USE_EXPERIMENTAL=True \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        vllm serve /data/Meta-Llama-3-8B-Instruct \
        --port 8200 \
        --disable-log-requests \
        --enforce-eager \
        --data-parallel-size 4 \
        --tensor-parallel-size 2 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1", "skip_last_n_tokens": 1}}'

    - mooncake-config-decode-TP2DP4.yaml

    .. code-block:: yaml

        chunk_size: 256
        local_device: "cpu"
        remote_url: "mooncakestore://200.10.0.22:50051/"
        remote_serde: "naive"
        local_cpu: False
        max_local_cpu_size: 5
        # external_lookup_client: "mooncakestore://200.10.0.22:50051/"

        extra_config:
        local_hostname: "200.10.0.18"
        metadata_server: "http://200.10.0.22:8080/metadata"
        protocol: "rdma"
        device_name: "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3"
        master_server_address: "200.10.0.22:50051"
        global_segment_size: 33554432000
        local_buffer_size: 10737418240
        transfer_timeout: 1

    - Proxy Command

    .. code-block:: bash
        LMCACHE_LOG_LEVEL=DEBUG python3 /data/lwh/LMCache/examples/disagg_prefill/xpyd_experimental/disagg_proxy_server.py --host localhost --port 9000 --prefiller-host 200.10.0.22 --prefiller-port 8100 --decoder-host 200.10.0.18 --decoder-port 8200

    .. code-block:: python
        :caption: LMCache/examples/disagg_prefill/xpyd_experimental/disagg_proxy_server.py

            @app.post("/v1/completions")
            async def handle_completions(request: Request):
            global counter, stats_calculator
            counter += 1
            req_id = str(counter)  # we use counter as req_id

            st = time.time()
            try:
                req_data = await request.json()

                tokenization_client = round_robin_pick_client(app.state.total_clients, counter)

                tokenize_output = await send_request_to_service(
                    tokenization_client.client, "/tokenize", {"prompt": req_data["prompt"]}
                )
                tokenize_output = tokenize_output.json()

                org_max_tokens = req_data["max_tokens"]
                req_data["prompt"] = tokenize_output["tokens"]
                req_data["max_tokens"] = 1

                # Pick decode client
                decode_client = round_robin_pick_client(app.state.decode_clients, counter)

                disagg_spec = {
                    "req_id": req_id,
                    "receiver_host": decode_client.host,
                    "receiver_init_port": decode_client.init_port,
                    "receiver_alloc_port": decode_client.alloc_port,
                }

                req_data["kv_transfer_params"] = {
                    "ret_first_tok": True,
                    "disagg_spec": disagg_spec,
                }

                req_data["stream"] = False
                stream_options = req_data.pop("stream_options", None)

                # Send request to prefill service round robin, ignore the response
                prefill_client = round_robin_pick_client(app.state.prefill_clients, counter)
                prefill_output = await send_request_to_service(
                    prefill_client.client, "/v1/completions", req_data
                )

                prefill_output = prefill_output.json()

                et = time.time()
                stats_calculator.add(et - st)

                req_data["max_tokens"] = org_max_tokens - 1
                req_data["prompt"].append(prefill_output["kv_transfer_params"]["first_tok"])
                req_data.pop("kv_transfer_params")
                req_data["stream"] = True
                if stream_options is not None:
                    req_data["stream_options"] = stream_options

                # Stream response from decode service
                async def generate_stream():
                    head_chunk = {
                        "id": prefill_output["id"],
                        "object": "text_completion",
                        "created": prefill_output["created"],
                        "model": prefill_output["model"],
                        "choices": [
                            {
                                "index": 0,
                                "text": prefill_output["choices"][0]["text"],
                                "logprobs": None,
                                "finish_reason": None,
                                "stop_reason": None,
                            }
                        ],
                        "usage": None,
                    }
                    yield (
                        "data: " + json.dumps(head_chunk, separators=(",", ":")) + "\n\n"
                    ).encode()

                    # Wait until decode node signals that kv is ready
                    # await wait_decode_kv_ready(req_id)

                    async for chunk in stream_service_response(
                        decode_client.client, "/v1/completions", req_data
                    ):
                        yield chunk

                return StreamingResponse(generate_stream(), media_type="application/json")

                except Exception as e:
                    # Standard
                    import sys
                    import traceback

                    exc_info = sys.exc_info()
                    print("Error occurred in disagg prefill proxy server - completions endpoint")
                    print(e)
                    print("".join(traceback.format_exception(*exc_info)))
                    raise

    -  Bench command
    
    .. code-block:: bash

        curl http://127.0.0.1:9000/v1/completions     -H "Content-Type: application/json"     
        -d '{
        "model": "/data/Meta-Llama-3-8B-Instruct",
        "prompt": "Tell me a story,100 words,Tell me a story,100 words,Tell me a story,100 words",
        "max_tokens": 10
         }'
**🧩 Environment**
    vLLM: 0.10.0
    LMCache: dev branch
    Mooncake: main branch
    P/D: 1P1D
    Prefill: 8 GPU, DP=4, TP=2
    Decode: 8 GPU, DP=4, TP=2

**🔴 Status:** Unresolved

----

**🧭 Time**
   2025-08-16

**🚨 Issue**
     Ref count of MemoryObj -1is negative: -2.Double free occurred somewhere.Setting ref count back to 0 as a hack 

**📋 Description**
    To reproduce 
    - Serving

    .. code-block:: bash

        CUDA_VISIBLE_DEVICES=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
        LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CHUNK_SIZE=512 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=15.0 \
        PYTHONHASHSEED=0 LMCACHE_REMOTE_SERDE=cachegen LMCACHE_REMOTE_URL=lm://localhost:65432 \
        vllm serve \
        meta-llama/Llama-3.1-8B-Instruct \
        --download-dir /mnt/hps/llmcache/models/meta-llama/Llama-3.1-8B-Instruct \
        --port 8000 \
        --gpu-memory-utilization 0.95 \
        --no-enable-chunked-prefill \
        --enforce-eager \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
        --async-scheduling
    
    - Benchmark

    .. code-block:: bash

        vllm bench serve --port 8000 --seed 12345 \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset-name random --random-input-len 8000 --random-output-len 200 \
        --random-prefix-len 2000 \
        --num-prompts 500 --request-rate 3.6 --burstiness 100 --ignore-eos
        
**🧩 Environment**
    lmcache==0.3.3

**🔴 Status:** Unresolved

----

**🧭 Time**
   2025-08-19 

**🚨 Issue**
    Adding a Trace Replayer script in benchmark

**📋 Description**
    A simple trace replayer for performance evaluation using workload traces from `Mooncake Trace Release <https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/traces/conversation_trace.jsonl>`_ (JSONL format). Each trace line represents multiple chunked requests with timestamps, allowing you to simulate realistic request arrival patterns and measure metrics such as time-to-first-token (TTFT), token usage, and throughput.


**⚙️ Steps**


**Step 1: Start a model using vLLM**

.. code-block:: bash

   PYTHONHASHSEED=0 LMCACHE_MAX_LOCAL_CPU_SIZE=3 vllm serve meta-llama/Llama-3.1-8B-Instruct \
       --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'

**Step 2: Run the trace replayer**

By default, the script uses the full Mooncake conversation trace ``conversation_trace.jsonl``.  
Download and run it:

.. code-block:: bash

   wget https://github.com/kvcache-ai/Mooncake/raw/main/FAST25-release/traces/conversation_trace.jsonl -O conversation_trace.jsonl
   python trace_replayer.py

For quick testing, a truncated version (<500 lines) is included as ``sample_trace.jsonl``:

.. code-block:: bash

   python trace_replayer.py --trace_file sample_trace.jsonl


**⚙️ Additional Arguments**

The script allows specifying model, trace file, maximum input length, and replay duration via command-line arguments.

**Example 1: Using a smaller model**

.. code-block:: bash

   # Load and run the model:
   vllm serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

   # Run the trace replayer with max_input_length to avoid exceeding model context length:
   python trace_replayer.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max_input_length 2048

**Example 2: Specifying a custom trace file and max duration**

.. code-block:: bash

   python trace_replayer.py --trace_file <file_name>.jsonl --max_duration 120.0


**📚 Argument Reference**


**--model**
   🧠 Trace with different models.

**--max_input_length**
   ✂️ Truncate input length if the model's max context length is smaller than the trace input.

**--trace_file**
   📄 Specify an alternative trace file.

**--max_duration**
   ⏱️ Maximum duration to replay the trace (in seconds).


**📊 Notes**

- **`--max_input_length`**  
  Optional. Use this to truncate inputs if JSONL requests exceed the model’s maximum context length.  
  Omitting this may cause requests to fail for smaller models.

- **Metrics Collected**  
  TTFT (time-to-first-token), input/output tokens, and request throughput.

- **CSV Output**  
  Detailed request statistics are saved automatically to ``summary-<timestamp>.csv``.


**🏁 Example Output**

When completed successfully, you’ll see an output summary similar to:

.. code-block:: text

   ✅ Trace replay completed.
   Summary file: summary-2025-10-17-1430.csv
   Average TTFT: 1.23s | Throughput: 412 tokens/s | Total requests: 498
----

**🧭 Time**
   2025-08-28

**🚨 Issue**
    Support multiple backends at the same time

**📋 Description**
    does LMCache support heterogeneous multi-tier caching, where tiers span from GPU memory to CPU memory to external backends? Specifically, can we configure multiple heterogeneous backends (e.g., Redis, local disk, etc.) to be used simultaneously, along with GPU and CPU memory, and does LMCache support migration of data across these tiers and backends?

**🛠️ Solution**
    `Added in the q3 roadmap <https://github.com/LMCache/LMCache/issues/1253>`_

----

**🧭 Time**
   2025-08-28

**🚨 Issue**
     Fix crash caused by raised runtime error due to inconsistent number of hit tokens across tp ranks

**📋 Description**
    It is possible for different TP ranks to have different numbers of hit tokens, for example, when using the Mooncake Store backend. Since the Mooncake Store backend is unaware of the TP information associated with keys, during eviction it may remove the KV cache generated by only a subset of TP ranks. This can lead to inconsistencies in the number of hit tokens observed across ranks.
    In previous versions, such inconsistencies would raise a runtime error, which was not properly caught at higher levels, ultimately causing the process to crash.

    .. image:: https://private-user-images.githubusercontent.com/13486004/481593203-5d47a835-23c0-4db8-bbec-4ac0e5c1caf3.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjA3MDM5ODUsIm5iZiI6MTc2MDcwMzY4NSwicGF0aCI6Ii8xMzQ4NjAwNC80ODE1OTMyMDMtNWQ0N2E4MzUtMjNjMC00ZGI4LWJiZWMtNGFjMGU1YzFjYWYzLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMTclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDE3VDEyMjEyNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTUxZTYzODQ3MDJjZGYyY2RiYjkwZjgzOTVhODhkNDA1MTJlOGE3N2M0MzgxZDQxM2Q1ZWFjNmE5YzkyNTU0MWUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.IJsg6GxwgX4kKJSgt3OVil84mrsxvjgC16CujMzP9Q4
        :alt: Log information
        :width: 100%
        :align: center

**🛠️ Solution**
    `Links <https://github.com/LMCache/LMCache/pull/1426>`_ to the PR that fixes this issue.
----

---------------------------------
🕒 2025-09
---------------------------------

**🧭 Time**
    2025-09-01

**🚨 Issue**
    PD Fix config setting

**📋 Description**
    1. nixl_peer_host is used in decoder.
    2. Fix env_converter is not called when needed (e.g., nixl_init_port).
    3. Fix bool error in env_converter.

**🛠️ Solution**
    `Links <https://github.com/LMCache/LMCache/pull/1391>`_ to the PR that fixes this issue.

----

**🧭 Time**
    2025-09-01

**🚨 Issue**
    PD Fix config setting
    Prefiller can start normally, but decoder cannot.

**📋 Description**
    1. nixl_peer_host is used in decoder.
    2. Fix env_converter is not called when needed (e.g., nixl_init_port).
    3. Fix bool error in env_converter.

**🛠️ Solution**
    `Links <https://github.com/LMCache/LMCache/pull/1391>`_ to the PR that fixes this issue.
----
**🧭 Time**
    2025-09-10

**🚨 Issue**
   [Bug] [1P1D Disagg] Facing assertion error while starting Vllm server for assert isinstance(allocator_backend, AllocatorBackendInterface)

**📋 Description**
    I am experimenting with 1P1D configuration and trying to start vllm server with 1P1D configuration but facing the following error:
    
    .. code-block:: bash
        
        Traceback (most recent call last):
        File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 533, in worker_main
            worker = WorkerProc(*args, **kwargs)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 401, in __init__
            self.worker.init_device()
        File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker_base.py", line 603, in init_device
            self.worker.init_device()  # type: ignore
            ^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 192, in init_device
            init_worker_distributed_environment(self.vllm_config, self.rank,
        File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 612, in init_worker_distributed_environment
            ensure_kv_transfer_initialized(vllm_config)
        File "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_transfer_state.py", line 63, in ensure_kv_transfer_initialized
            _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector(
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/factory.py", line 60, in create_connector
            return connector_cls(config, role)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py", line 27, in __init__
            self._lmcache_engine = LMCacheConnectorV1Impl(vllm_config, role, self)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/vllm_v1_adapter.py", line 563, in __init__
            self.lmcache_engine = _init_lmcache_engine(
                                ^^^^^^^^^^^^^^^^^^^^^
        File "/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/vllm_v1_adapter.py", line 507, in _init_lmcache_engine
            engine = LMCacheEngineBuilder.get_or_create(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/usr/local/lib/python3.12/dist-packages/lmcache/v1/cache_engine.py", line 1457, in get_or_create
            engine = LMCacheEngine(
                    ^^^^^^^^^^^^^^
        File "/usr/local/lib/python3.12/dist-packages/lmcache/v1/cache_engine.py", line 130, in __init__
            self.storage_manager = StorageManager(
                                    ^^^^^^^^^^^^^^^
        File "/usr/local/lib/python3.12/dist-packages/lmcache/v1/storage_backend/storage_manager.py", line 114, in __init__
            self.allocator_backend = self._get_allocator_backend(config)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/usr/local/lib/python3.12/dist-packages/lmcache/v1/storage_backend/storage_manager.py", line 153, in _get_allocator_backend
            assert isinstance(allocator_backend, AllocatorBackendInterface)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        AssertionError

**🔍 Details**
    `Links <https://github.com/LMCache/LMCache/issues/1551>`_
----

**🧭 Time**
    2025-09-10

**🚨 Issue**
   [DistServe] The decoder KV cache missed yet the prefiller has transfered it successfully

**📋 Description**
    I have a 1P1D setup with VLLM+ LMCache 0.3.3 (with some fixes) and it works for inference request of 16K input prompt length or lower.
With 32K input length, I see the prefill does successfully transfer KV-cache data over, yet decode performs prefill work again.

**🔍 Details**
    `Links <https://github.com/LMCache/LMCache/issues/1547>`_
----

**🧭 Time**
    2025-09-17

**🚨 Issue**
    blend_kv_v1 with use-disk doesn't work on v0.3.6

**📋 Description**
    blend_kv_v1 with use-disk doesn't work on v0.3.6.
When using the use-disk option, the second request during inference terminates with a key error in local_cpu_backend's hot_cache.

**🔍 Details**
    `Links <https://github.com/LMCache/LMCache/issues/1616>`_

----

**🧭 Time**
    2025-09-23

**🚨 Issue**
    Playing around with lmcache integration with sglang

**📋 Description**
    TypeError: LMCacheLayerwiseConnector.__init__() got an unexpected keyword argument 'tp_group'

**🔴 Status:** Unresolved  

----

**🧭 Time**
    2025-09-24

**🚨 Issue**
    The codes in lmcache/integration folder do not respect the settings in precommit.

**📋 Description**
    The following code passes 80 characters at line 223, and should be identified by pre-commit (ruff)
    
.. code-block:: python
   :caption: LMCache/lmcache/integration/vllm/vllm_v1_adapter.py

    # https://github.com/vllm-project/vllm/commit/ 
    # b029de9902aa3ac58806c8c17776c7074175b6db# 
    # diff-cafd89ce8a698a56acb24ada62831cbc7a980782f78a52d1742ba238031f296cL94 
**🔴 Status:** Unresolved  

----

**🧭 Time**
    2025-09-25

**🚨 Issue**
   ERROR:The number of retrieved tokens is less than the expected number of tokens! This should not happen!

**📋 Description**
    ERROR:The number of retrieved tokens is less than the expected number of tokens!

**🔍 Details**
    `Links <https://github.com/LMCache/LMCache/issues/1686>`_
----

**🧭 Time**
    2025-09-30

**🚨 Issue**
   multi-TP deployment, "exist" key and "put" key in the connector are no same

**📋 Description**
    using multi-tp deployment, the keys of exists and put are different, which leads to the failure to hit the kv cache.

**🔍 Details**
    `Links <https://github.com/LMCache/LMCache/issues/1728>`_
----


---------------------------------
🕒 2025-10
---------------------------------
**🧭 Time**
    2025-10-01

**🚨 Issue**
   ``len(slot_mapping) == len(token_ids)`` Assertion failure with Decode Context Parallelism

**📋 Description**
    When running DeepSeek V3/R1 with ``-tp 4 -dcp 4`` on 4xH200, LMCache fails the assertion ``assert len(slot_mapping) == len(token_ids)``.
    That said, I think this may actually be a bug with vLLM's DCP, regarding its interaction with the KV cache storage/retrieval? I'm not sure.
**🔍 Details**
    `Links <https://github.com/LMCache/LMCache/issues/1732>`_
----

**🧭 Time**
    2025-10-01

**🚨 Issue**
   Lmcache metrics error

**📋 Description**
    For lmcache metrics, what does lmcache:retrieve_hit_rate actually mean? We were testing a scenario with multi-turn chat, the lookup_hit_rate does seem quite high but retrieve_hit_rate is constantly 0. It only shows as one when we put the exact same prompt. Thanks

----
**🧭 Time**
    2025-10-08

**🚨 Issue**
   The native serdes ref_count_up() is preventing MemoryObj from being unpinned and evicted

**📋 Description**
    The extra memory_obj.ref_count_up() is currently blocking the MemoryObj being released if the cache memory is filled up. We got the following WARNINGs which is not recoverable

**🔍 Details**
    `Links <https://github.com/LMCache/LMCache/issues/1756>`_
----

**🧭 Time**
    2025-10-13

**🚨 Issue**
   LMCache + vLLM Version Compatibility Issues Across Multiple Releases

**📋 Description**
    For lmcache metrics, what does lmcache:retrieve_hit_rate actually mean? We were testing a scenario with multi-turn chat, the lookup_hit_rate does seem quite high but retrieve_hit_rate is constantly 0. It only shows as one when we put the exact same prompt. Thanks

**🔍 Details**
    `Links <https://github.com/LMCache/LMCache/issues/1768>`_
----

**🧭 Time**
    2025-10-13

**🚨 Issue**
   Performance Degradation When enable_async_loading: True Under High CPU Memory Usage

**📋 Description**
    When ``enable_async_loading``: True is set, in scenarios with multiple users interacting in multi-turn conversations under high concurrency, if CPU memory usage becomes saturated, the GPU utilization drops sharply from 90% to 60%-70% and overall performance degrades significantly.

    **To Reproduce**
    1. Configure LMCache as follows:

    .. code-block:: yaml

        chunk_size: 256
        local_cpu: True
        max_local_cpu_size: 200
        use_layerwise: true

    2. Start vllm service (H100):

    .. code-block:: bash

        LMCACHE_CONFIG_FILE=example.yaml vllm serve xxx-10B-dense-model --max-model-len 10000 --port 8000 --quantization fp8 --max-num-batched-tokens 1024 --kv-cache-dtype fp8 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
    3. Run a benchmark script (such as `Links <https://github.com/vllm-project/production-stack/blob/main/benchmarks/multi-round-qa/multi-round-qa.py>`_ )
    
    This issue only occurs when enable_async_loading is enabled. In other configurations, GPU utilization remains stable even under high CPU memory usage. See attached performance dashboards and GPU metrics for reference.

**🧩 Environment**
    - Device: Server/Workstation running LMCache
    - OS: Linux (please specify version)
    - GPU: NVIDIA H100 80GB HBM3
    - CUDA Driver: 12.2
    - LMCache version: (please specify)
    - Other info: Driver Version 535.129.03, concurrency level: high

**🔴 Status:** Unresolved  

----
**🧭 Time**
    2025-10-15

**🚨 Issue**
   async mode connector use batched_put OOM

**📋 Description**
    Connector using async batched_put will cause OOM,memoryobj.meta.ref_count not decrease to 0.

    .. image:: https://private-user-images.githubusercontent.com/26517379/501337901-d8b0f4ff-9808-4165-8ffd-141aeff3a63d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjEzNzc0OTMsIm5iZiI6MTc2MTM3NzE5MywicGF0aCI6Ii8yNjUxNzM3OS81MDEzMzc5MDEtZDhiMGY0ZmYtOTgwOC00MTY1LThmZmQtMTQxYWVmZjNhNjNkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDI1VDA3MjYzM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTA1MTg4MDA4NjBmZjUyOGM3ZWFmZmJjOGRjM2VjOTVmY2YxMDlkODUzN2EyNDU3NjQ4MDY3ODA4ZmE2ZmI4N2UmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.h-bOqYqPoRBXscmSmHLvJ7cOccUrwowRF--j_yEKIZw
        :alt: Log information
        :width: 100%
        :align: center
    
    .. image:: https://private-user-images.githubusercontent.com/26517379/501341592-81e9ae13-0609-4ee5-823b-d78d326928eb.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjEzNzc0OTMsIm5iZiI6MTc2MTM3NzE5MywicGF0aCI6Ii8yNjUxNzM3OS81MDEzNDE1OTItODFlOWFlMTMtMDYwOS00ZWU1LTgyM2ItZDc4ZDMyNjkyOGViLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDI1VDA3MjYzM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWE1YjE1NDI0Y2U1MDJkNTYxYjQxNDMzZTFmMDk3ZTdiM2ViMGIyMjI2NWU3YmJhOTcxMTY1ZjQ0NzQ5OTBiODYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.YPxQDFdZumDXKwpc2VNDsARacJr9jkmsuxVfgiUcC9E
        :alt: Log information
        :width: 100%
        :align: center

    **To Reproduce**
        - use async
        - ``support_batched_put`` True

    .. code-block:: yaml

        chunk_size: 256
        local_cpu: False
        max_local_cpu_size: 5
        enable_async_loading: True
        remote_url: "my connector"
    
    server
    .. code-bolck:: bash

        export LMCACHE_CONFIG_FILE=debug_config.yaml \
        export LMCACHE_USE_EXPERIMENTAL=True \
        export VLLM_USE_V1=1

        python3 -m vllm.entrypoints.openai.api_server \
        --port "8000" \
        --uvicorn-log-level warning \
        --model /models/DeepSeek-R1-Distill-Qwen-1.5B \
        --served-model-name DeepSeek-R1-Distill-Qwen-1.5B \
        --trust-remote-code \
        --max-model-len "16384" \
        --disable-log-requests \
        --disable-fastapi-docs \
        --swap-space "0" \
        --no-enable-chunked-prefill \
        --no-enable-prefix-caching \
        --gpu-memory-utilization="0.95" \
        --tensor-parallel-size=2 \
        --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1","kv_role": "kv_both"}' \
        >test.log 2>&1 &
    test

    .. code-bolck:: bash

        curl http://0.0.0.0:8000/v1/chat/completions     -H "Content-Type: application/json" -d '{
            "model": "DeepSeek-R1-Distill-Qwen-1.5B",
            "messages": [
                {"role": "user", "content": "hello world"}
            ]
        }'
**🧩 Environment**
    
    lmcache==0.3.6

----
**🧭 Time**
    2025-10-17

**🚨 Issue**
   Inference crashes when unable to load token

**📋 Description**
    Two inference nodes are connected to the same mooncake. When one node is shut down unexpectedly, the other node will die after receiving the request.

    **To Reproduce**

    Three containers, A, B, and C.
    A: Inference Node 1
    B: ​​Inference Node 2
    C: Mooncake service, including mooncake_master and mooncake_http_metadata_server
    
    1. Both inference nodes use Mooncake as the lmcache storage backend.

    - After startup, they first send a request to node A to store the KV there.
    - They send a request to node B, where they can observe token reuse.

    .. image:: https://private-user-images.githubusercontent.com/43202036/502373331-3b49cbc3-dadc-483f-8622-abeec6ee2061.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjEzNzY5NzAsIm5iZiI6MTc2MTM3NjY3MCwicGF0aCI6Ii80MzIwMjAzNi81MDIzNzMzMzEtM2I0OWNiYzMtZGFkYy00ODNmLTg2MjItYWJlZWM2ZWUyMDYxLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDI1VDA3MTc1MFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTViODY1YzI3MDkwOGQ2YWI4NDJhODYwMzU1MmNmZWY2MjlhNzMzMTc0NmE0ZmQ2ODliZTE0OTE2NWNmOTY5MWQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.wjW4EHTUVvny5oq-62-gud0OgQUEh5SpuemZMbhKOC0
        :alt: Log information
        :width: 100%
        :align: center

    2. Shut down node A directly and send a request to node B again. The inference is stuck and returns 500 after a while.

    .. image:: https://private-user-images.githubusercontent.com/43202036/502373463-b492c6b9-a336-4aaf-822a-a3896eb137d5.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjEzNzY5NzAsIm5iZiI6MTc2MTM3NjY3MCwicGF0aCI6Ii80MzIwMjAzNi81MDIzNzM0NjMtYjQ5MmM2YjktYTMzNi00YWFmLTgyMmEtYTM4OTZlYjEzN2Q1LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDI1VDA3MTc1MFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWI0NzcwY2ZmNDBmZjQ4MzEzNWIzM2QxOGY0OWMyMDVhMmNjNzNkNzYzYzNmNTYyOTBkMWFjYmVkZGRmMzE1NjkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.koJH8h0hH7L4TaVwdCNCkq7pPSt_ugFHUYBqFy6lO5k
        :alt: Log information
        :width: 100%
        :align: center
    Node B engine crashes and exits

    .. image:: https://private-user-images.githubusercontent.com/43202036/502373482-4e2016c9-9352-46ae-8394-4ac3975c1482.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjEzNzY5NzAsIm5iZiI6MTc2MTM3NjY3MCwicGF0aCI6Ii80MzIwMjAzNi81MDIzNzM0ODItNGUyMDE2YzktOTM1Mi00NmFlLTgzOTQtNGFjMzk3NWMxNDgyLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMjUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDI1VDA3MTc1MFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTIzMjY3MDZmM2U3ZGMxY2Y1NDMxZGU5NjRiOGQ0Mjc2OGQ0NzExYzRlYWRjYTI4NGUwYTI4ODU1Nzk5MTdmNTcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.AZnJtMSpUPoUFpmKp-s_NUR18AZWkpkCTv9zVkolIH4
        :alt: Log information
        :width: 100%
        :align: center
    **Expected behavior**
    Fall back to no lmcache when token cannot be loaded

**🧩 Environment**
    
    vllm==0.10.2
    lmcache==0.3.7
    
**🔴 Status:** Unresolved  

----
**🧭 Time**
    2025-10-22

**🚨 Issue**
   Chunk name/CacheEngineKey should contains dtype to avoid multiple vllm cluster cache matched by accident 

**📋 Description**
    There are two vllm cluster with same module, same tp count, but with different kvcache dtype. Under this assumption, there is a risk to have two different chunks belong to different vLLM cluster, but with save CacheEngineKey, if we use shared remote backend, vLLM0 cluster may get the vLLM0 cluster's chunk regard as cache hit, but the content is not correct.

**🛠️ Solution**
    `Links <https://github.com/LMCache/LMCache/pull/1859>`_ to the PR that fixes this issue.

----
**🧭 Time**
    2025-10-23

**🚨 Issue**
   Response is wrong due to async KV loading using wrong memory object

**📋 Description**
    When both vllm's prefix cache and LMCache are hit. ``_async_process_tokens_internal()`` in cache_engine.py appends wrong memory object to chunks and return to caller, causing vLLM generating wrong response

    To Reproduce
    1. Configure a simple cpu backend, enable async loading

    .. code-block:: yaml
        chunk_size: 64
        local_cpu: True
        max_local_cpu_size: 5
        enable_async_loading: True

    2. start vllm with such cpu.yaml

    .. code-block:: bash

        VLLM_LOGGING_LEVEL=DEBUG PYTHONHASHSEED=0 \
        LMCACHE_CONFIG_FILE=cpu.yaml LMCACHE_LOG_LEVEL=DEBUG \
        vllm serve Qwen/Qwen3-4B --served-model-name 'Qwen/Qwen3-4B' --enforce-eager  \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
        --disable-log-stats --disable-log-requests -tp 1
    
    3. prepare a 256 token prompt (for Qwen3-4B) and send

    4. for the first time, response is correct because kv cache is fully computed and written to vllm's GPU cache and LMCache's CPU cache

    5. resend the request, got wrong answer, because token is computed with vllm's GPU cache plus wrong data from LMCache's object

    .. code-block:: bash

        "text": "?\n\nStanford University, formally known as Leland Stanford Junior University, is a prestigious private research university"
    vllm uses cache of 240 tokens from its GPU cache and try to load 15 tokens from LMCache, which triggers LMCache to load the 4th chunk (192-255) in its CPU memory

    .. code-block:: bash

        LMCache INFO: Reqid: cmpl-3f9cb13d8712474ea76302f67af0bb4f-0, Total tokens 256, LMCache hit tokens: 256, need to load: 15
        LMCache DEBUG: Scheduled to load 256 tokens for request cmpl-3f9cb13d8712474ea76302f67af0bb4f-0
        LMCache INFO: Retrieved 64 out of 64 required tokens (from 256 total tokens).
    Somehow the data returned from LMCache is wrong, making vllm generate wrong response

    6. following request will continue to get wrong response

**🔴 Status:** Unresolved  

----

