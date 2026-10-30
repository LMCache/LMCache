# Single-node P2P (two instances on one host)

Use this to exercise the full P2P path on one multi-GPU machine; latencies are not
representative (loopback). Everything else matches `docs/source/mp/p2p.rst`, "Running on a
single node".

```bash
S=.claude/skills/lmcache-p2p-launch/scripts
$S/gen_env.sh --single-node a --gpu 0 --model Qwen/Qwen3-8B --l1-gb 40   # ~/lmcache-p2p/node-a/env.sh
$S/gen_env.sh --single-node b --gpu 1 --model Qwen/Qwen3-8B --l1-gb 40   # ~/lmcache-p2p/node-b/env.sh
P2P_ENV=~/lmcache-p2p/node-a/env.sh $S/start_node.sh --with-coordinator
P2P_ENV=~/lmcache-p2p/node-b/env.sh $S/start_node.sh
P2P_ENV=~/lmcache-p2p/node-a/env.sh $S/wait_ready.sh && P2P_ENV=~/lmcache-p2p/node-b/env.sh $S/wait_ready.sh
P2P_ENV=~/lmcache-p2p/node-a/env.sh $S/status.sh
```

The two instances differ in every port (MP, HTTP, P2P, vLLM) and in `--instance-id`;
`--single-node` sets `NODE_IP=PEER_IP=127.0.0.1` and disables the RDMA UCX block (loopback
uses `sm`/TCP). The test skill works unchanged with either env file: `VLLM_URL` is the
instance itself and `PEER_VLLM_URL` the other one.
