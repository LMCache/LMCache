# Flags and ports used by the launch scripts

| env.sh variable | default | used for |
|---|---|---|
| `COORD_IP`, `COORD_PORT` | -, 12500 | `lmcache coordinator --host 0.0.0.0 --port`; servers register at `COORD_URL` |
| `MP_PORT` | 6700 | `lmcache server --port`: vLLM's connector (`lmcache.mp.port`) and peers' lookup/unlock RPCs |
| `HTTP_PORT` | 7700 | `--http-port`: `/status`, `/metrics` (Prometheus), `lmcache kvcache clear --url` |
| `P2P_PORT` | 8700 | `--p2p-advertise-url NODE_IP:P2P_PORT`, `--p2p-listen-url 0.0.0.0:P2P_PORT`: nixl transfer endpoint |
| `VLLM_PORT` | 8300 | `vllm serve --port`; also serves `/metrics` |
| `NODE_IP` | first `hostname -I` | advertised to the coordinator and peers; must be routable from them |
| `INSTANCE_ID` | `p2p-<hostname>` | `--instance-id`; shows up in `/instances`, `/status` and logs |
| `L1_GB`, `GPU` | 80, 0 | `--l1-size-gb`; `CUDA_VISIBLE_DEVICES` for vLLM |
| `MAX_MODEL_LEN`, `GPU_MEM_UTIL` | 32768, 0.85 | vLLM |
| `PEER_IP`, `PEER_SSH`, `PEER_*_PORT` | - | the other node, for the test and diagnose skills |

Server flags fixed by `launch_lmcache.sh`: `--max-workers 4`, `--eviction-policy LRU`,
`--l1-align-bytes 65536`, `--chunk-size 256`, `--http-host 0.0.0.0`. Anything else can be
appended: `start_node.sh` passes no extras, so run `launch_lmcache.sh --l2-adapter ...`
yourself or edit the script for a deployment that also has an L2.

P2P flags with their meaning are listed in `docs/source/mp/p2p.rst`; the coordinator flags in
`docs/source/mp/coordinator.rst`. Multiple servers on one host must differ in **every** port.

`lmcache kvcache clear --url http://<node>:7700` empties a node's L1 between experiments.
