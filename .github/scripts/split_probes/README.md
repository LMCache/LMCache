# Split probes

Runtime checks for a client-only install. `split_packaging_gate.sh probe`
installs `lmcache[grpc]` alone in one venv and `lmcache-server` in another,
copies this directory, writes `client-manifest.json` from the client wheel,
and runs `run_probes.py` once per ZMQ endpoint (`none`, `ipc`, `tcp`):

- `server_probe.py` starts real ZMQ and gRPC request servers from the server
  venv with a few test handlers.
- `client_probe.py` runs in the client venv with an import hook that refuses
  any `lmcache` module the client wheel does not ship. It imports the engine
  entry points, exercises CPU engine-driven gather/scatter, the inline tensor
  codec, POSIX SHM across processes, and real ZMQ/gRPC requests, including the
  server's error reply to an operation it does not handle.
- `transport_checks.py` checks that the ZMQ server logged that request,
  shut down cleanly and released its endpoint.

These are CPU checks; they move no KV between devices.
