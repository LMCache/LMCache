# SPDX-License-Identifier: Apache-2.0
"""ZMQ endpoint helpers and the transport verdict used by run_probes.py."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import json
import os
import socket

HERE = Path(__file__).resolve().parent
MISSING_HANDLER_LOG = "Operation 'query_prefetch_status' is not enabled on this server"


def zmq_bind_url(choice: str, directory: str) -> str:
    if choice == "ipc":
        return f"ipc://{directory}/zmq.sock"
    if choice == "tcp":
        return "tcp://127.0.0.1:*"
    return choice


def endpoint_after_exit(url: str) -> dict:
    if url.startswith("ipc://"):
        # libzmq 4.3 keeps a named ipc file after close (libzmq_ipc_cleanup.py);
        # released means nobody accepts on it and the path can be bound again.
        path = url[len("ipc://") :]
        state: dict[str, bool | str] = {"ipc_socket_file_exists": os.path.exists(path)}
        raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            raw.connect(path)
            state["unix_connect"] = "accepted"
        except ConnectionRefusedError:
            state["unix_connect"] = "refused"
        except FileNotFoundError:
            state["unix_connect"] = "no-file"
        finally:
            raw.close()
        # Third Party
        import zmq

        ctx = zmq.Context()
        rebind = ctx.socket(zmq.ROUTER)
        rebind.bind(url)
        rebind.close(linger=0)
        ctx.term()
        state["rebind_same_path"] = "ok"
        return state
    host, port = url[len("tcp://") :].rsplit(":", 1)
    try:
        socket.create_connection((host, int(port)), timeout=1).close()
    except ConnectionRefusedError:
        return {"tcp_connect": "refused"}
    return {"tcp_connect": "accepted"}


def zmq_verdict(
    out: Path, transport: dict, server_returncode, server_stdout: str
) -> bool:
    """A missing-handler error counts only if the server logged that request."""
    probe_file = out / "client-probe.json"
    probe = (
        json.loads(probe_file.read_text())
        if probe_file.exists() and probe_file.read_text()
        else {}
    )
    check = next(
        (c for c in probe.get("checks", []) if c["name"] == "zmq-real-request-client"),
        None,
    )
    shutdown = [
        json.loads(line)["shutdown"]
        for line in server_stdout.splitlines()
        if line.startswith('{"shutdown"')
    ]
    server_log = (out / "transport-server.stderr").read_text() + server_stdout
    connected = transport.get("connected_endpoint")
    verdict = dict(transport)
    verdict["server_process"] = {
        "returncode": server_returncode,
        "shutdown": shutdown[-1] if shutdown else None,
    }
    verdict["server_logged_missing_handler"] = MISSING_HANDLER_LOG in server_log
    verdict["client_check"] = check
    verdict["after_exit"] = endpoint_after_exit(connected) if connected else None
    after = verdict["after_exit"] or {}
    released = after.get("tcp_connect") == "refused" or (
        after.get("unix_connect") in ("refused", "no-file")
        and after.get("rebind_same_path") == "ok"
    )
    down = verdict["server_process"]["shutdown"] or {}
    ok = (
        check is not None
        and check["status"] == "pass"
        and verdict["server_logged_missing_handler"]
        and transport.get("client_process", {}).get("returncode") == 0
        and server_returncode == 0
        and down.get("context_terminated") is True
        and down.get("zmq_socket_closed") is True
        and down.get("zmq_worker_alive") is False
        and released
        and transport.get("ipc_socket_file_exists_while_serving", True)
    )
    verdict["status"] = "pass" if ok else "fail"
    (out / "zmq-transport.json").write_text(json.dumps(verdict, indent=2) + "\n")
    return ok
