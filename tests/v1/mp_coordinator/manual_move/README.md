# Manual cache-move integrity check

Run a real coordinator, two MP servers with DRAM L1 and P2P, and a small GPU
worker beside each server. No model download or vLLM process is needed. These
scripts use the public MP client and move API; they do not mock storage or
network replies. Run them manually, not through pytest.

## What is checked

For each combination of FP16/BF16 and one/two logical KV ranks:

1. Generate deterministic synthetic KV on the source GPU and STORE it in L1.
2. Fill target GPU buffers with a different value and verify a mismatch.
3. Move A to B. Check the reported counts and wait until the directory shows
   zero source keys and all target keys.
4. RETRIEVE into the target GPU and compare every restored byte with an
   independently generated CPU reference. Compare SHA-256 hashes for every
   rank/layer/chunk with those collected on the source too.
5. Move B to A with `keep_source`, clear A's GPU buffers, and verify the reverse
   retrieval in the same way.
6. Repeat A to B with B already populated. Expect all keys to be reported
   missing and no deletion. Repeated reads of each terminal result must agree.

Source and destination use different block permutations. Unaddressed target
blocks must retain their fill value. Each rank contains 28 layers and six
256-token chunks, with eight KV heads and head size 128. Both logical ranks
run on one physical GPU per worker; this is not a multi-GPU TP inference test.
The largest transfer is 336 MiB. Allow at least 2 GiB of free GPU memory per
worker and 2 GiB of host L1 memory per server, plus runtime overhead.

The byte comparison copies retrieved GPU data to CPU before viewing it as
`uint8`; it does not use approximate floating-point equality. The data are
synthetic, not KV produced by model inference. This is a functional test, not
a performance benchmark or a failure-injection test.

## Requirements

- The same LMCache checkout including the move API and warm-prefetch outcome
  fix on both hosts, installed with working native extensions.
- PyTorch and LMCache built for the installed accelerator, plus NIXL/UCX P2P
  dependencies. Use the same PyTorch version on both hosts for the generated
  reference data. The original harness was verified with NVIDIA GPUs, torch
  2.13.0+cu130, NIXL 1.3.2 and UCX 1.21.0. AMD execution remains to be verified.
- `httpx` in the Python environment running the driver.
- Dedicated MP instances and an isolated coordinator. Existing entries, peers
  or concurrent traffic can invalidate these assertions. The worker HTTP API
  has no authentication; bind only on a trusted test network.
- Reachable coordinator, server HTTP, P2P and worker ports between hosts.
  If using containers, enable the GPU/IPC and RDMA device access needed by
  your LMCache installation. RDMA also needs sufficient locked-memory limits
  (for example, Docker `--ulimit memlock=-1`); an 8 MiB default is insufficient
  for this test. Do not use `python -O`: assertions are the checks.

## Start the two hosts

From the repository root, activate the installed environment. Set `A_IP` and
`B_IP` to the hosts' reachable addresses (IPv4 examples below). Select an idle
GPU before starting **both** the server and its worker. On NVIDIA, use
`CUDA_VISIBLE_DEVICES`; on AMD, use the device visibility settings appropriate
for your ROCm installation. The scripts use LMCache's selected device backend.

On host A:

```bash
export A_IP=<host-a-address>
export CUDA_VISIBLE_DEVICES=<idle-gpu-index>
START_COORDINATOR=1 bash tests/v1/mp_coordinator/manual_move/start.sh \
  "$A_IP" move-a "http://$A_IP:19300"
```

On host B:

```bash
export A_IP=<host-a-address>
export B_IP=<host-b-address>
export CUDA_VISIBLE_DEVICES=<idle-gpu-index>
bash tests/v1/mp_coordinator/manual_move/start.sh \
  "$B_IP" move-b "http://$A_IP:19300"
```

Leave both launchers running. Logs go to `move-logs-move-a/` and
`move-logs-move-b/`. Ctrl-C stops the processes started by that launcher.
The launcher does not install dependencies or select a NIC for you.

Defaults are coordinator 19300, MP client 19555, server HTTP 19755, P2P 19855,
and worker HTTP 19955. Override with `COORDINATOR_PORT`, `SERVER_PORT`,
`HTTP_PORT`, `P2P_PORT`, `WORKER_PORT`, `LOG_DIR`, or `PYTHON`. Match the
coordinator URL and driver endpoints to any overrides. For two instances on
one host, give the second instance distinct server, HTTP, P2P and worker ports.

Before running the driver, check `/healthz` on the coordinator and
`/healthcheck` on each server. Each server's `/status` must show
`p2p_peer_count` equal to 1. Wait for peer discovery to complete.

## Run the checks

From any host that can reach the coordinator and workers:

```bash
python3 tests/v1/mp_coordinator/manual_move/drive.py \
  --coordinator "http://$A_IP:19300" \
  --source-worker "http://$A_IP:19955" \
  --target-worker "http://$B_IP:19955" \
  --output move-results
```

Use `--source-instance` and `--target-instance` if you changed the instance
names. The output directory must not already exist, so reruns cannot silently
mix old and new evidence. Success ends with four `CASE PASS` lines and
`RESULT PASS`. A failed assertion or HTTP operation exits nonzero. Each case
writes a JSON record with move responses, compared byte counts, per-chunk
hashes and the negative-control mismatch count. Worker sessions are closed
on driver exit; stop the dedicated servers afterward to release L1 entries.

## RDMA versus P2P

A successful run proves the selected P2P path and data integrity. It does not
by itself prove which network transport NIXL selected. For a host-memory RoCE
check, configure UCX on both hosts before starting the launchers, for example:

```bash
export UCX_TLS=rc
export UCX_NET_DEVICES=<rdma-device>:<port>
export UCX_IB_GID_INDEX=<gid-index>
export UCX_LOG_LEVEL=info
export UCX_PROTO_INFO=y
```

Choose values for your NIC and network. Preserve logs identifying the selected
inter-node transport (the original run showed `rma(rc_mlx5/...)`). This tests
RDMA between host-memory L1 caches, not GPUDirect RDMA between GPU buffers.
GPU STORE/RETRIEVE uses the existing local transfer path.
