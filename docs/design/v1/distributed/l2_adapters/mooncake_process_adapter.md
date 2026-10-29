# Opt-in Mooncake process integration

This integration combines the process transport, admission guard, L1 transfer
pins, prefetch/store drain, and GPU transfer shutdown fixes. Applying only the
process/guard component commits does not provide these controller guarantees.

Use a Linux, non-lazy, named shared L1 arena and generate a fresh SHM name for
every service start. The allocator supplies the canonical backing name; the
transport verifies its device/inode, mapping, size and access. Anonymous/lazy
arenas, multiple L1 regions and serde temporary buffers are unsupported.
Never concurrently force-clear/delete keys or close L1 while I/O is active.

Example L2 adapter JSON (addresses are placeholders):

```json
{
  "type": "mooncake_store",
  "local_hostname": "NODE_IP",
  "metadata_server": "http://MASTER_IP:18080/metadata",
  "master_server_addr": "MASTER_IP:50051",
  "protocol": "rdma",
  "rdma_devices": "mlx5_0",
  "global_segment_size": "0",
  "local_buffer_size": "1073741824",
  "num_workers": 8,
  "process": {
    "operation_timeout": 8.0,
    "startup_timeout": 15.0,
    "kill_timeout": 10.0,
    "restart_interval": 2.0,
    "max_pending": 16,
    "guard": {
      "lookup_timeout": 2.0,
      "cooldown": 15.0,
      "probe_timeout": 2.0,
      "recovery_successes": 2
    }
  }
}
```

`process` and its nested `guard` fields are validated locally and never forwarded
to the SDK. Omitting `process` preserves the in-process adapter. Native SDK
initialization runs in the child; the parent factory returns after checking the
SHM mapping. Every data submission exception is either a pre-admission rejection
or follows confirmed native quiescence. If quiescence cannot be confirmed, the
parent exits with status 78 instead of releasing possibly active buffers.

Shutdown drains prefetch and store controllers, then closes the adapter and L1.
Each controller retains its I/O pins until terminal completion. The concrete
adapter also owns the process before the asynchronous guard adopts it, so an
immediate startup/close cannot leave an orphan worker. The deployment launcher
must stop model request admission before closing GPU transfer modules.

An open circuit produces cache misses and bypassed writes; L1 remains usable.
The first affected request can wait for a deadline and recomputation adds latency.
Successful independent PUT/GET/byte-check/delete probes, native readiness and
drained previous operations are required before admission resumes. Bridge health
and remote availability are separate status fields. Provider/driver failure
semantics and performance must be validated on the deployment's actual hardware.
