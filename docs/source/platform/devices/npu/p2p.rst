.. SPDX-License-Identifier: Apache-2.0

P2P KV Cache Sharing on Ascend
==============================

This guide configures Host-to-Host P2P transfers between Ascend MP servers
using either native HIXL or Mooncake Transfer Engine (TE). It covers the
``lmcache_driven`` transfer mode and ordinary DRAM L1. For installation,
container setup and model-specific vLLM arguments, start with
:doc:`lmcache_driven`. For peer discovery and the shared P2P options, see
:doc:`/mp/p2p`.

The inference engine and the P2P transport have different jobs:

.. code-block:: text

   Node A: inference NPU -> local Host L1
                                  |
                       Node B issues a P2P READ
                                  |
   Node B: inference NPU <- local Host L1

Both backends use NPU communication resources even though the P2P payload
is Host memory. Native HIXL does not require Mooncake TE. The Mooncake
recipe below uses Ascend Direct Transport with HIXL client/server (CS)
communication, device RoCE and asynchronous transfers.

Prerequisites
-------------

On both nodes:

* Install compatible LMCache and LMCache-Ascend versions in the same Python
  environment. The Ascend native ``alloc_pinned_ptr`` must provide pure
  ``aclrtMallocHost`` allocation, without an additional Host registration.
  Rebuild LMCache-Ascend after changing its native sources; changing Python
  source paths alone does not update a compiled extension.
* For native HIXL, build and load ``lmcache_ascend.hixl_npu_comms`` against
  the HIXL headers and libraries in the target CANN environment. It must
  provide ``HixlError`` and the ACL context helpers used by this backend.
  LMCache-Ascend automatically enables its HIXL build when it detects CANN
  8.5 or later. ``USE_HIXL=1`` overrides build detection; it does not install
  a compatible HIXL library.
* Both the native HIXL backend and the Mooncake CS recipe select
  ``LocalCommRes`` version ``1.3``. On A2/A3, this requires CANN Toolkit
  9.1 or later and HDK 25.5 or later. The 8.5 build-detection threshold
  above does not establish compatibility with these runtime settings.
* Make the selected NPU devices, driver libraries and the host's
  ``/etc/hccn.conf`` available in the container. Check the selected devices'
  RoCE addresses and link health. The host IP used for listeners is distinct
  from the device RoCE IP in ``hccn.conf``.
* Allow traffic between both nodes' host listeners and device RoCE networks.
  The examples assume host networking. Mooncake also opens dynamic TE and
  communication listeners, so allowing only the ports in LMCache's CLI is
  insufficient on a restricted network.

Check the native HIXL module in each target environment:

.. code-block:: bash

   env LMCACHE_DEVICE_BACKEND=npu python -c 'import lmcache; import lmcache_ascend.hixl_npu_comms as h; from lmcache_ascend import c_ops; print("HIXL:", h.__file__); print("Allocator:", c_ops.__file__); assert all(hasattr(h, name) for name in ("Hixl", "HixlError", "get_current_device", "get_current_context", "set_current_context")); assert hasattr(c_ops, "alloc_pinned_ptr")'
   python -m lmcache.tools.transfer_channel_benchmark --help

An import or ``--help`` check does not initialize a transport or prove that
memory registration succeeds. Run the verified transfer benchmark below
before adding an inference engine.

L1 allocation requirements
--------------------------

For the recipes on this page, use one ordinary, non-lazy DRAM L1 allocation
on each MP server:

.. code-block:: text

   --no-l1-use-lazy --shm-name "" --l1-align-bytes 65536

Keep hugepages disabled and do not select a NUMA-specific allocator, GDS or
Device-DAX L1. Named SHM and lazy pinning use different allocation paths;
they are not substitutes for this ACL Host allocation. In particular,
:doc:`engine_driven` requires named SHM and is outside this recipe.

For a DRAM entry supplied through ``--l1-manager`` JSON, set
``"use_lazy": false``, ``"shm_name": ""`` and ``"use_hugepages": false``
on that entry. Selecting a transfer backend does not override these
settings. No ``LMCACHE_ASCEND_MP_HOST_MEMORY_ALLOCATOR`` environment variable
is needed.

The ordinary Mixed allocator owns the allocation and eventually calls
``aclrtFreeHost``. HIXL or Mooncake TE registers that allocation for
communication and deregisters it during channel cleanup. Keep L1 and the
ACL runtime alive until cleanup finishes. Stop inference requests, local
transfers and peer access before stopping the servers: closing a channel
does not drain outstanding remote access.

Non-lazy L1 reserves the full configured capacity at startup. Leave room
for the model-serving processes and the operating system. Large allocation,
registration and release operations can take substantial time; choose a
small L1 first, then measure startup and shutdown at the intended capacity.

Devices and endpoints
---------------------

Replace ``10.0.0.1`` (node A), ``10.0.0.2`` (node B) and ``eth0`` with your
host addresses and host network interface. Run each service in a separate
terminal using the installed environment. Use the same backend on both
nodes. The commands set their backend configuration explicitly; they assume
no conflicting transport experiment settings are inherited from the shell.

.. list-table:: Example listeners
   :header-rows: 1
   :widths: 40 30 30

   * - Purpose
     - Node A
     - Node B
   * - Native HIXL Engine
     - ``10.0.0.1:6100``
     - ``10.0.0.2:6101``
   * - Benchmark transfer-channel metadata
     - ``10.0.0.1:7600``
     - ``10.0.0.2:7601``
   * - Benchmark catalog / verification
     - ``10.0.0.1:7610``
     - Connects to A
   * - MP coordinator
     - ``10.0.0.1:9300``
     - Connects to A
   * - MP transfer-channel metadata
     - ``10.0.0.1:8500``
     - ``10.0.0.2:8500``
   * - MP peer gRPC / worker ZMQ / HTTP
     - ``9555`` / ``5555`` / ``8080``
     - ``9555`` / ``5555`` / ``8080``

The benchmark needs no coordinator. Stop it before starting the MP servers
so that its native Engine listener is no longer in use. Distinct IPs allow
the two hosts to reuse port numbers. For single-host testing, use different
devices and different ports for every service on the two sides.

``ASCEND_RT_VISIBLE_DEVICES`` selects the process's visible physical devices.
``LMCACHE_HIXL_DEVICE_ID`` is a **logical** index within that list: for
example, ``ASCEND_RT_VISIBLE_DEVICES=6`` with
``LMCACHE_HIXL_DEVICE_ID=0`` selects physical device 6. Native HIXL uses one
logical NPU per MP server. Exposing several devices does not aggregate their
communication bandwidth.

A benchmark process needs only its communication device. An MP server must
also be able to access the devices used by its local inference workers for
IPC transfers. The MP examples expose devices 0 through 7; adjust the mask
to the local model deployment on both the server and vLLM processes.

Native HIXL configuration
-------------------------

.. list-table:: Native backend environment
   :header-rows: 1
   :widths: 43 57

   * - Variable
     - Meaning
   * - ``LMCACHE_HIXL_ENGINE_ID``
     - Required unique, reachable ``IP:port`` for the native Engine.
       It is separate from the transfer-channel metadata endpoint.
   * - ``LMCACHE_HIXL_DEVICE_ID``
     - Logical NPU index; defaults to ``0``.
   * - ``LMCACHE_HIXL_CONNECT_TIMEOUT_MS``
     - Positive connection timeout in milliseconds; defaults to ``10000``.
   * - ``LMCACHE_HIXL_COMM_LISTEN_PORT``
     - Optional HIXL communication listen port (1--65535). Leave unset
       unless the deployment needs an explicit value; it is not the Engine
       or LMCache metadata port.

Mooncake TE configuration
-------------------------

Install a Mooncake package with Ascend Direct Transport support. The
reference configuration below uses the traditional TE Python API from
``mooncake-transfer-engine-npu==0.3.11.post1``. It does not select TENT or
TEV1; do not enable ``MC_USE_TENT`` or ``MC_USE_TEV1`` for this recipe.
Check the module origin if another Mooncake distribution is installed:

.. code-block:: bash

   python -m pip show mooncake-transfer-engine-npu
   python -c 'import mooncake.engine as m; from mooncake.engine import TransferEngine; print(m.__file__)'

For an environment without a conflicting Mooncake installation:

.. code-block:: bash

   python -m pip install 'mooncake-transfer-engine-npu==0.3.11.post1'

The commands below include these transport settings:

.. list-table:: Mooncake Ascend settings
   :header-rows: 1
   :widths: 43 57

   * - Setting
     - Purpose
   * - ``MC_TE_PROTOCOL=ascend``, ``MC_TE_DEVICE=''``
     - Select Ascend Direct Transport. This is different from the generic
       ``rdma`` protocol and its host RDMA device name. ``MC_TE_DEVICE``
       does not select the logical NPU for this recipe.
   * - ``HCCL_SOCKET_IFNAME=eth0``, ``HCCL_INTRA_ROCE_ENABLE=1``
     - Use the selected host interface for sockets and request RoCE,
       including for an intra-host test.
   * - ``ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'``
     - Request HIXL CS communication through the compatible ADXL API.
   * - ``ASCEND_GLOBAL_RESOURCE_CONFIG``
     - Set ``comm_resource_config.protocol_desc`` to ``roce:device`` and
       ``comm_resource_config.listen_port`` to ``26666``.
   * - ``ASCEND_BUFFER_POOL=0:0``
     - Disable the intermediate buffer pool. These numbers describe the
       pool, not NPU indices; a nonzero pool is incompatible with this
       asynchronous recipe.
   * - ``ASCEND_USE_ASYNC_TRANSFER=1``
     - Enable asynchronous native transfers. In the reference TE version,
       the variable's presence enables this mode, even if its value is 0.
   * - ``ASCEND_AUTO_CONNECT=0``, ``ASCEND_USE_SHORT_CONNECTION=0``
     - Use explicit connection management and retain connections.
   * - ``ASCEND_CONNECT_TIMEOUT=10000``, ``ASCEND_TRANSFER_TIMEOUT=60000``
     - Native connect and transfer timeouts, in milliseconds.
   * - ``ASCEND_THREAD_POOL_SIZE=1``, ``ASCEND_BASE_PORT=16000``
     - Fix one dispatch worker and the base for native listener selection.

These are Mooncake-specific settings, not aliases for ``LMCACHE_HIXL_*``.
The traditional transport uses the current NPU device when initializing its
engine; check the selected device in the native logs when comparing it with
HIXL. The benchmark examples expose only physical device 0 on each node.

Transfer-channel benchmark
--------------------------

Use ``lmcache.tools.transfer_channel_benchmark`` with ``--shm-name ""``
on **both** sides to select the ordinary ACL Host allocation path.
Omitting ``--shm-name`` retains the benchmark's default named SHM allocation.
The benchmark defaults to non-lazy allocation: omit ``--use-lazy``. Its CLI
does not accept the MP server's ``--no-l1-use-lazy`` flag.

Each example transfers four 16 MiB objects per iteration (64 MiB total),
with five warmup iterations and 30 measured iterations. The server
registers a 512 MiB buffer containing 16 source objects; the client
registers a 128 MiB buffer (64 MiB payload plus 64 MiB slack).
``--num-source-objects`` must be at least ``--num-objects`` on both sides.
Page size and object size must match.

Use a fresh server process for each client invocation. A server can retain
the previous client's registration metadata; a new client process reusing
the same endpoint may have a different memory address. After a client
finishes, stop the server with Ctrl-C, wait for cleanup, and start it again
before the next run. Save a separate server log for each run.

HIXL benchmark
~~~~~~~~~~~~~~

**Node A, server terminal:**

.. code-block:: bash

   mkdir -p logs/ascend-p2p
   set -o pipefail
   env ASCEND_RT_VISIBLE_DEVICES=0 LMCACHE_DEVICE_BACKEND=npu LMCACHE_HIXL_DEVICE_ID=0 LMCACHE_HIXL_ENGINE_ID=10.0.0.1:6100 LMCACHE_HIXL_CONNECT_TIMEOUT_MS=10000 LMCACHE_LOG_LEVEL=INFO python -u -m lmcache.tools.transfer_channel_benchmark --shm-name "" --role server --transfer-channel-type hixl --url 10.0.0.1:7600 --control-url 10.0.0.1:7610 --buffer-size 512MB --page-size 64KB --object-size 16MB --num-objects 4 --num-source-objects 16 --server-timeout 1800 2>&1 | tee logs/ascend-p2p/hixl-server.log
   echo "server pipeline exit code: $?"

Wait for ``[server] ready``, then run **node B, client terminal**:

.. code-block:: bash

   mkdir -p logs/ascend-p2p
   set -o pipefail
   env ASCEND_RT_VISIBLE_DEVICES=0 LMCACHE_DEVICE_BACKEND=npu LMCACHE_HIXL_DEVICE_ID=0 LMCACHE_HIXL_ENGINE_ID=10.0.0.2:6101 LMCACHE_HIXL_CONNECT_TIMEOUT_MS=10000 LMCACHE_LOG_LEVEL=INFO python -u -m lmcache.tools.transfer_channel_benchmark --shm-name "" --role client --op read --transfer-channel-type hixl --url 10.0.0.1:7600 --listen-url 10.0.0.2:7601 --control-url 10.0.0.1:7610 --buffer-size 512MB --page-size 64KB --object-size 16MB --num-objects 4 --num-source-objects 16 --warmup 5 --iters 30 --verify 2>&1 | tee logs/ascend-p2p/hixl-read.log
   echo "READ pipeline exit code: $?"

After READ finishes, restart A with its server command above, changing the
log filename to ``hixl-server-write.log``. Wait for ``[server] ready``, then
run WRITE on **node B**:

.. code-block:: bash

   mkdir -p logs/ascend-p2p
   set -o pipefail
   env ASCEND_RT_VISIBLE_DEVICES=0 LMCACHE_DEVICE_BACKEND=npu LMCACHE_HIXL_DEVICE_ID=0 LMCACHE_HIXL_ENGINE_ID=10.0.0.2:6101 LMCACHE_HIXL_CONNECT_TIMEOUT_MS=10000 LMCACHE_LOG_LEVEL=INFO python -u -m lmcache.tools.transfer_channel_benchmark --shm-name "" --role client --op write --transfer-channel-type hixl --url 10.0.0.1:7600 --listen-url 10.0.0.2:7601 --control-url 10.0.0.1:7610 --buffer-size 512MB --page-size 64KB --object-size 16MB --num-objects 4 --num-source-objects 16 --warmup 5 --iters 30 --verify 2>&1 | tee logs/ascend-p2p/hixl-write.log
   echo "WRITE pipeline exit code: $?"

Mooncake TE benchmark
~~~~~~~~~~~~~~~~~~~~~

Stop the HIXL benchmark before running this comparison. Use the same devices,
object sizes, iteration counts and memory allocation on both nodes.

**Node A, server terminal:**

.. code-block:: bash

   mkdir -p logs/ascend-p2p
   set -o pipefail
   env ASCEND_RT_VISIBLE_DEVICES=0 LMCACHE_DEVICE_BACKEND=npu LMCACHE_LOG_LEVEL=INFO HCCL_SOCKET_IFNAME=eth0 HCCL_INTRA_ROCE_ENABLE=1 MC_TE_PROTOCOL=ascend MC_TE_DEVICE='' ASCEND_LOCAL_COMM_RES='{"version":"1.3"}' ASCEND_BUFFER_POOL=0:0 ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":"roce:device","comm_resource_config.listen_port":26666}' ASCEND_AUTO_CONNECT=0 ASCEND_USE_SHORT_CONNECTION=0 ASCEND_CONNECT_TIMEOUT=10000 ASCEND_TRANSFER_TIMEOUT=60000 ASCEND_THREAD_POOL_SIZE=1 ASCEND_BASE_PORT=16000 ASCEND_USE_ASYNC_TRANSFER=1 python -u -m lmcache.tools.transfer_channel_benchmark --shm-name "" --role server --transfer-channel-type mooncake_te --url 10.0.0.1:7600 --control-url 10.0.0.1:7610 --buffer-size 512MB --page-size 64KB --object-size 16MB --num-objects 4 --num-source-objects 16 --server-timeout 1800 2>&1 | tee logs/ascend-p2p/mooncake-server.log
   echo "server pipeline exit code: $?"

Wait for ``[server] ready``, then run **node B, client terminal**:

.. code-block:: bash

   mkdir -p logs/ascend-p2p
   set -o pipefail
   env ASCEND_RT_VISIBLE_DEVICES=0 LMCACHE_DEVICE_BACKEND=npu LMCACHE_LOG_LEVEL=INFO HCCL_SOCKET_IFNAME=eth0 HCCL_INTRA_ROCE_ENABLE=1 MC_TE_PROTOCOL=ascend MC_TE_DEVICE='' ASCEND_LOCAL_COMM_RES='{"version":"1.3"}' ASCEND_BUFFER_POOL=0:0 ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":"roce:device","comm_resource_config.listen_port":26666}' ASCEND_AUTO_CONNECT=0 ASCEND_USE_SHORT_CONNECTION=0 ASCEND_CONNECT_TIMEOUT=10000 ASCEND_TRANSFER_TIMEOUT=60000 ASCEND_THREAD_POOL_SIZE=1 ASCEND_BASE_PORT=16000 ASCEND_USE_ASYNC_TRANSFER=1 python -u -m lmcache.tools.transfer_channel_benchmark --shm-name "" --role client --op read --transfer-channel-type mooncake_te --url 10.0.0.1:7600 --listen-url 10.0.0.2:7601 --control-url 10.0.0.1:7610 --buffer-size 512MB --page-size 64KB --object-size 16MB --num-objects 4 --num-source-objects 16 --warmup 5 --iters 30 --verify 2>&1 | tee logs/ascend-p2p/mooncake-read.log
   echo "READ pipeline exit code: $?"

After READ finishes, restart A with its Mooncake server command above,
changing the log filename to ``mooncake-server-write.log``. Wait for
``[server] ready``, then run WRITE on **node B**:

.. code-block:: bash

   mkdir -p logs/ascend-p2p
   set -o pipefail
   env ASCEND_RT_VISIBLE_DEVICES=0 LMCACHE_DEVICE_BACKEND=npu LMCACHE_LOG_LEVEL=INFO HCCL_SOCKET_IFNAME=eth0 HCCL_INTRA_ROCE_ENABLE=1 MC_TE_PROTOCOL=ascend MC_TE_DEVICE='' ASCEND_LOCAL_COMM_RES='{"version":"1.3"}' ASCEND_BUFFER_POOL=0:0 ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":"roce:device","comm_resource_config.listen_port":26666}' ASCEND_AUTO_CONNECT=0 ASCEND_USE_SHORT_CONNECTION=0 ASCEND_CONNECT_TIMEOUT=10000 ASCEND_TRANSFER_TIMEOUT=60000 ASCEND_THREAD_POOL_SIZE=1 ASCEND_BASE_PORT=16000 ASCEND_USE_ASYNC_TRANSFER=1 python -u -m lmcache.tools.transfer_channel_benchmark --shm-name "" --role client --op write --transfer-channel-type mooncake_te --url 10.0.0.1:7600 --listen-url 10.0.0.2:7601 --control-url 10.0.0.1:7610 --buffer-size 512MB --page-size 64KB --object-size 16MB --num-objects 4 --num-source-objects 16 --warmup 5 --iters 30 --verify 2>&1 | tee logs/ascend-p2p/mooncake-write.log
   echo "WRITE pipeline exit code: $?"

Check for ``install AscendDirectTransport``, ``Roce mode is enabled``,
``Set LocalCommRes to:{"version":"1.3"}`` and ``Use async transfer`` in the
reference TE logs. Module import alone does not confirm this transport.

Interpreting benchmark results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

READ must print ``verify OK: 4 objects match expected pattern``. WRITE must
print ``verify OK: server confirms 4 written objects match expected
pattern``. Also check the exit codes and both sides' cleanup logs. WRITE
changes the server's data: restart the server before another verified READ
that expects the original pattern.

The throughput report measures submission through completion for the payload,
excluding connection setup and the final byte check. It reports decimal
GB/s, whereas the example payload is 64 MiB. A 100000 Mb/s port has a raw
line rate of 12.5 GB/s before protocol overhead. Record the actual port rate,
payload, READ/WRITE direction and median as well as mean when comparing
backends. Single-host measurements do not establish cross-host performance.

The benchmark checks transfer-channel data movement. It does not exercise
coordinator discovery, model accuracy or the MP server's optional P2P store
policy (``--enable-p2p-write``).

MP server deployment
--------------------

Stop the benchmark, then start one coordinator and one MP server per node.
The examples use 50 GiB L1 and a 4096-token chunk, matching the DeepSeek-V4
recipe in :doc:`lmcache_driven`; choose a capacity that fits the host and
use matching chunk/layout settings on both nodes. For other models, follow
their deployment guide. The MP server and its local vLLM should use the same
compatible packages, model and KV layout as the peer deployment.

**Node A, coordinator terminal** (shared by either backend):

.. code-block:: bash

   mkdir -p logs/ascend-p2p
   set -o pipefail
   lmcache coordinator --host 10.0.0.1 --port 9300 2>&1 | tee logs/ascend-p2p/coordinator.log

Choose **one** of the following backend pairs. B uses
``--l2-prefetch-policy skip_l1`` for validation: remotely loaded chunks are
temporary, so later loads can exercise P2P again. The normal ``default``
policy retains them in local L1. This flag does not disable vLLM's own prefix
cache or prevent KV computed locally on B from entering L1.

HIXL MP servers
~~~~~~~~~~~~~~~

**Node A, MP terminal:**

.. code-block:: bash

   mkdir -p logs/ascend-p2p
   set -o pipefail
   env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LMCACHE_DEVICE_BACKEND=npu LMCACHE_LOG_LEVEL=DEBUG LMCACHE_HIXL_DEVICE_ID=0 LMCACHE_HIXL_ENGINE_ID=10.0.0.1:6100 LMCACHE_HIXL_CONNECT_TIMEOUT_MS=10000 lmcache server --host 0.0.0.0 --port 5555 --http-port 8080 --instance-id ascend-a --supported-transfer-mode lmcache_driven --chunk-size 4096 --l1-size-gb 50 --l1-align-bytes 65536 --no-l1-use-lazy --shm-name "" --eviction-policy LRU --coordinator-url http://10.0.0.1:9300 --coordinator-advertise-ip 10.0.0.1 --p2p-rpc-port 9555 --p2p-advertise-url 10.0.0.1:8500 --p2p-transfer-engine hixl 2>&1 | tee logs/ascend-p2p/mp-a.log

**Node B, MP terminal:**

.. code-block:: bash

   mkdir -p logs/ascend-p2p
   set -o pipefail
   env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LMCACHE_DEVICE_BACKEND=npu LMCACHE_LOG_LEVEL=DEBUG LMCACHE_HIXL_DEVICE_ID=0 LMCACHE_HIXL_ENGINE_ID=10.0.0.2:6101 LMCACHE_HIXL_CONNECT_TIMEOUT_MS=10000 lmcache server --host 0.0.0.0 --port 5555 --http-port 8080 --instance-id ascend-b --supported-transfer-mode lmcache_driven --chunk-size 4096 --l1-size-gb 50 --l1-align-bytes 65536 --no-l1-use-lazy --shm-name "" --eviction-policy LRU --l2-prefetch-policy skip_l1 --coordinator-url http://10.0.0.1:9300 --coordinator-advertise-ip 10.0.0.2 --p2p-rpc-port 9555 --p2p-advertise-url 10.0.0.2:8500 --p2p-transfer-engine hixl 2>&1 | tee logs/ascend-p2p/mp-b.log

Mooncake TE MP servers
~~~~~~~~~~~~~~~~~~~~~~

**Node A, MP terminal:**

.. code-block:: bash

   mkdir -p logs/ascend-p2p
   set -o pipefail
   env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LMCACHE_DEVICE_BACKEND=npu LMCACHE_LOG_LEVEL=DEBUG HCCL_SOCKET_IFNAME=eth0 HCCL_INTRA_ROCE_ENABLE=1 MC_TE_PROTOCOL=ascend MC_TE_DEVICE='' ASCEND_LOCAL_COMM_RES='{"version":"1.3"}' ASCEND_BUFFER_POOL=0:0 ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":"roce:device","comm_resource_config.listen_port":26666}' ASCEND_AUTO_CONNECT=0 ASCEND_USE_SHORT_CONNECTION=0 ASCEND_CONNECT_TIMEOUT=10000 ASCEND_TRANSFER_TIMEOUT=60000 ASCEND_THREAD_POOL_SIZE=1 ASCEND_BASE_PORT=16000 ASCEND_USE_ASYNC_TRANSFER=1 lmcache server --host 0.0.0.0 --port 5555 --http-port 8080 --instance-id ascend-a --supported-transfer-mode lmcache_driven --chunk-size 4096 --l1-size-gb 50 --l1-align-bytes 65536 --no-l1-use-lazy --shm-name "" --eviction-policy LRU --coordinator-url http://10.0.0.1:9300 --coordinator-advertise-ip 10.0.0.1 --p2p-rpc-port 9555 --p2p-advertise-url 10.0.0.1:8500 --p2p-transfer-engine mooncake_te 2>&1 | tee logs/ascend-p2p/mp-a.log

**Node B, MP terminal:**

.. code-block:: bash

   mkdir -p logs/ascend-p2p
   set -o pipefail
   env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LMCACHE_DEVICE_BACKEND=npu LMCACHE_LOG_LEVEL=DEBUG HCCL_SOCKET_IFNAME=eth0 HCCL_INTRA_ROCE_ENABLE=1 MC_TE_PROTOCOL=ascend MC_TE_DEVICE='' ASCEND_LOCAL_COMM_RES='{"version":"1.3"}' ASCEND_BUFFER_POOL=0:0 ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":"roce:device","comm_resource_config.listen_port":26666}' ASCEND_AUTO_CONNECT=0 ASCEND_USE_SHORT_CONNECTION=0 ASCEND_CONNECT_TIMEOUT=10000 ASCEND_TRANSFER_TIMEOUT=60000 ASCEND_THREAD_POOL_SIZE=1 ASCEND_BASE_PORT=16000 ASCEND_USE_ASYNC_TRANSFER=1 lmcache server --host 0.0.0.0 --port 5555 --http-port 8080 --instance-id ascend-b --supported-transfer-mode lmcache_driven --chunk-size 4096 --l1-size-gb 50 --l1-align-bytes 65536 --no-l1-use-lazy --shm-name "" --eviction-policy LRU --l2-prefetch-policy skip_l1 --coordinator-url http://10.0.0.1:9300 --coordinator-advertise-ip 10.0.0.2 --p2p-rpc-port 9555 --p2p-advertise-url 10.0.0.2:8500 --p2p-transfer-engine mooncake_te 2>&1 | tee logs/ascend-p2p/mp-b.log

Verify peer discovery and remote hits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From a terminal with access to both hosts:

.. code-block:: bash

   curl --noproxy '*' -fsS http://10.0.0.1:9300/instances | python -m json.tool
   curl --noproxy '*' -fsS http://10.0.0.1:8080/status | python -m json.tool
   curl --noproxy '*' -fsS http://10.0.0.2:8080/status | python -m json.tool

Expect ``p2p_state`` to be ``registered`` and ``p2p_peer_count`` to be ``1``
on each server. Then start vLLM on each node using :doc:`lmcache_driven`,
connecting to that node's MP port 5555 through ``LMCacheMPConnector``.
Keep vLLM prefix caching disabled for this check
(``--no-enable-prefix-caching``).

1. Use the same model, tokenizer, chat template, dtype and KV layout on both
   nodes. Send a long, deterministic prompt to A and wait for its request
   and LMCache store to finish.
2. Send the identical token prefix to B, whose local cache has not seen it.
   Use a tokenized prefix longer than one full chunk; for the commands
   above this means more than 4096 tokens, not 4096 characters. Incomplete
   trailing chunks need not produce remote hits.
3. Check the expected answer and B's MP logs together. With only P2P L2
   configured as above, nonzero L2 hit cells establish remote reuse.
   ``Retrieved ... tokens`` or a correct model answer alone is insufficient:
   the request may have hit L1 or recomputed the prompt.
4. For concurrent validation, prefill each distinct prefix on A first,
   then send them concurrently to B. Give each request a different expected
   answer and check every result, alongside nonzero L2 hits and error logs.
   Size L1 for the working set and concurrent prefetch buffers.

On A and B respectively:

.. code-block:: bash

   grep -nE 'Added P2P adapter|Stored [0-9]+ tokens|ERROR|Traceback|HixlError' logs/ascend-p2p/mp-a.log
   grep -nE 'submitted load tasks|hit cells|Retrieved [0-9]+ tokens|ERROR|Traceback|HixlError' logs/ascend-p2p/mp-b.log

For example, ``60 hit cells (0 from L1, 60 from L2)`` on B is evidence of
remote loading in this configuration. The number of cells is not necessarily
the token count: it also depends on the model's object groups and layout.
After validation, restore the default prefetch policy if retaining remote
data in B's L1 is desired, and lower log verbosity before measuring throughput.

Troubleshooting and scope
-------------------------

* **Initialization fails before registration:** check that the Engine IP
  belongs to the host, its port is free, ``/etc/hccn.conf`` is present, and
  the selected NPU's network is healthy. A generic HIXL initialization
  status does not uniquely identify a port conflict; inspect the native
  logs for the underlying failure.
* **Allocation or registration fails:** confirm the actual native module
  and loaded CANN libraries, non-lazy allocation, empty SHM and disabled
  hugepages. Do not reuse an older allocator that additionally maps the
  Host range before communication registration. Allocation success alone
  does not establish registration or transfer success.
* **Correct answers but no L2 hits:** check peer discovery, complete chunk
  lengths, matching token prefixes and layouts, and local caches. Look for
  earlier errors instead of treating fallback recomputation as a P2P pass.
* **DeepSeek-V4 object-group errors:** earlier Ascend end-to-end validation
  used ``--no-separate-object-groups`` on both MP servers. If reproducing
  that configuration, apply it to both sides and record it with the result.
  This is an Ascend layout compatibility condition, not a HIXL transport
  requirement or a general recommendation for other models. In particular,
  follow the separate object-group requirements in :doc:`hybrid_mamba`.
* **Cleanup errors:** preserve both complete logs and exit codes even if
  byte verification passed. Stop peer traffic before teardown and allow
  time for deregistration and Host memory release.

The current HIXL backend transfers through one registered L1 Host buffer.
Additional local regions and direct NPU device-memory transfers are outside
its scope. A successful transfer-channel benchmark, model accuracy test,
and concurrent remote-hit test establish different properties; record the
package versions, configuration and evidence for each. Rebuild and rerun
the relevant checks when the allocator or native bindings change.
