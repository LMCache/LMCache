# VMM fd delivery (`vmm_fd`)

Delivers the POSIX fds of CUDA VMM KV caches from engine workers to the MP
server. Code: `lmcache/v1/multiprocess/vmm_fd.py`, and its server owner
`VmmFdChannelModule` in `lmcache/v1/multiprocess/modules/vmm_fd_channel.py`.
Companion of [vmm_cuda_ipc.md](../platform/devices/cuda/vmm_cuda_ipc.md),
whose wrapper exposes the hooks this module plugs into.

## Why it exists

A VMM allocation whose `requestedHandleTypes` include
`POSIX_FILE_DESCRIPTOR` is exported as a POSIX fd (the wrapper prefers it
over fabric); on GPUs without IMEX it is the only exportable form. An fd is
an index into the exporting process's own file table, so the number is
meaningless to the server. The kernel has to install the same file in the
server's table, which only `SCM_RIGHTS` over an `AF_UNIX` socket does; a
ZMQ/gRPC message carries bytes only, so the fd cannot ride the
`REGISTER_KV_CACHE` request. The server imports each wrapper inside the
register handler, so the fd must reach the server before that request.

## Mechanism

`VmmCudaIPCWrapper` calls two process-wide hooks and does not know how fds
travel: `set_vmm_fd_sender` on the worker (from `wrap()`) and
`set_vmm_fd_resolver` on the server (from `to_tensor()`). This module
provides what fills them.

```
MP server startup (--enable-vmm-fd-channel)
  VmmFdChannelModule: resolver = VmmFdResolver(/dev/shm/lmcache_fd_<port>.sock)
                      set_vmm_fd_resolver(resolver.resolve)

vLLM worker startup (lmcache.mp.use_vmm_api)
  set_vmm_fd_sender(partial(send_fd, fd_channel_path_for_server_url(url)))

every registration (first REGISTER and heartbeat re-registration)
  worker: VmmCudaIPCWrapper.wrap(tensor), POSIX-fd kind
            send_fd(path, export_id, fd)   # connect, check peer uid, send
            close the exported fd
  server accept thread: check peer uid, keep fd under export_id
  worker: REGISTER_KV_CACHE(wrappers)      # descriptors carry export_id
  server: to_tensor() -> resolve(export_id) -> cuMemImportFromShareableHandle
                                           # skipped if already registered
```

- **Sender installation.** The vLLM adapter installs the sender whenever
  `lmcache.mp.use_vmm_api` is set, whether or not the node has IMEX: the
  wrapper picks fabric or POSIX fd per allocation at wrap time, from the
  allocation's `requestedHandleTypes`, so the worker cannot know in advance.
  The sender holds only the socket path; `send_fd` runs, and connects, only
  for a POSIX-fd wrapper. Fabric-kind wrappers never reach it.
- **Path.** One socket per server, `/dev/shm/lmcache_fd_<port>.sock`, from the
  server's `--port`. The worker derives it from its server URL; a URL without
  a port (`ipc://`, `grpc+unix://`) gets no sender and a warning.
- **Message.** `SOCK_SEQPACKET`, one connection per fd. Each message is the
  16-byte export id plus exactly one fd; anything else is closed and the
  connection dropped.
- **Ordering.** `sendmsg` on a Unix socket queues the fd on the server before
  it returns, and wrapping finishes before REGISTER is sent. `resolve()` waits
  up to 5 s only for the accept thread to drain the queue.
- **Ownership.** `wrap` closes its fd after `send_fd` returns, whether delivery
  succeeded or not; a retry re-wraps and re-exports. `resolve()` hands the fd
  to the wrapper, which closes it after import.
- **Unresolved fds.** An fd nobody resolves pins the exporter's device memory.
  It happens when REGISTER is a no-op (instance already registered) or fails.
  The resolver closes fds unresolved for 300 s (the default worker
  `mq_timeout`), checking every 30 s.

## Security

The socket lives in world-writable `/dev/shm`.

- Mode `0600`, set before `listen()`.
- Both ends check the peer uid with `SO_PEERCRED`. The worker side stops a
  listener planted by another user from collecting KV-cache fds.
- An fd is matched only by its export id, a random UUID4 (122 random bits)
  that the worker sends only to the server, over the request channel and
  the owner-only fd channel.

## Failure handling

- If the socket cannot be opened, or the server is not Linux CUDA, the server
  logs a warning and runs without the channel; only POSIX-fd VMM
  registrations then fail, with a hint to start the server with
  `--enable-vmm-fd-channel`.
- A socket left by a crashed server is unlinked and bound anew only if it
  refuses connections. A live listener or a non-socket file is left alone.

## Limitations

- The path comes from the port the worker sees. If that differs from the
  server's `--port` (e.g. behind a port-remapping Service), the worker cannot
  reach the channel.
- POSIX fds do not cross nodes, so a worker whose server runs on another
  node cannot register POSIX-fd pools. If a flagged server on the worker's
  own node uses the same port, `send_fd` succeeds against it: the fds expire
  there unresolved, and the error surfaces on the remote server's REGISTER
  instead of as the worker's "Cannot reach" message.
- Isolated deployments (no shared `/dev/shm`) are not supported: the wrapper
  still rejects POSIX-fd allocations under `isolated_ipc`.
- vLLM sleep mode, which replaces physical pages under live imports, needs a
  retire/re-register lifecycle that is not implemented.
