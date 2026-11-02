# SPDX-License-Identifier: Apache-2.0
"""Run client_probe.py from a client-only venv against server_probe.py.

usage: run_probes.py CLIENT_PYTHON SERVER_PYTHON --zmq-endpoint none|ipc|tcp
    --out-dir DIR
server_probe.py runs from the server venv. Each probe's root is its venv's
site-packages; client-manifest.json lists the client's files.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import argparse
import hashlib
import json
import os
import select
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Third Party
import transport_checks as rp  # noqa: E402


def site_packages(python: str) -> Path:
    out = subprocess.run(
        [
            python,
            "-I",
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("client_python")
    parser.add_argument("server_python")
    parser.add_argument("--zmq-endpoint", default="none")
    parser.add_argument("--out-dir", type=Path, required=True)
    options = parser.parse_args()
    client_root, server_root = (
        site_packages(options.client_python),
        site_packages(options.server_python),
    )
    out = options.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "client-manifest.json").read_text())
    protected = [
        folder / item["path"]
        for folder in (server_root, client_root)
        for item in manifest["files"]
    ]
    before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected
    }
    env = dict(os.environ, LMCACHE_LOG_LEVEL="ERROR", PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="lmcache-split-") as directory:
        endpoint = (
            "none"
            if options.zmq_endpoint == "none"
            else rp.zmq_bind_url(options.zmq_endpoint, directory)
        )
        transport = {
            "requested": options.zmq_endpoint,
            "bind_url": endpoint,
            "client_root": str(client_root),
            "server_root": str(server_root),
        }
        error_file = (out / "transport-server.stderr").open("w")
        server = subprocess.Popen(
            [
                options.server_python,
                "-I",
                "-B",
                str(HERE / "server_probe.py"),
                str(server_root),
                endpoint,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=error_file,
            text=True,
            env=env,
        )
        server_stdout = ""
        assert server.stdout is not None  # stdout=PIPE
        try:
            ready, _, _ = select.select([server.stdout], [], [], 60)
            if not ready:
                raise RuntimeError(
                    "transport server did not report readiness within 60s"
                )
            line = server.stdout.readline()
            if not line:
                raise RuntimeError(
                    "transport server exited; see transport-server.stderr"
                )
            endpoints = json.loads(line)
            args = [
                options.client_python,
                "-I",
                "-B",
                str(HERE / "client_probe.py"),
                str(client_root),
                "--grpc",
                endpoints["grpc"],
            ]
            if endpoints["zmq"]:
                args += ["--zmq", endpoints["zmq"]]
                transport["connected_endpoint"] = endpoints["zmq"]
                if endpoints["zmq"].startswith("ipc://"):
                    transport["ipc_socket_file_exists_while_serving"] = os.path.exists(
                        endpoints["zmq"][len("ipc://") :]
                    )
            started = time.monotonic()
            result = subprocess.run(
                args, env=env, capture_output=True, text=True, timeout=120
            )
            transport["client_process"] = {
                "returncode": result.returncode,
                "seconds": round(time.monotonic() - started, 3),
            }
            (out / "client-probe.json").write_text(result.stdout)
            (out / "client-probe.stderr").write_text(result.stderr)
            if result.returncode:
                raise RuntimeError("client probe failed; see client-probe.json")
            after = {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in protected
            }
            changed = sorted(path for path in before if before[path] != after[path])
            (out / "probe-source-identity.json").write_text(
                json.dumps(
                    {"checked_files": len(protected), "changed_files": changed},
                    indent=2,
                )
                + "\n"
            )
            if changed:
                raise RuntimeError("Installed files changed during probe")
            data = json.loads(result.stdout)
            print(
                json.dumps(
                    {
                        "pass": data["pass"],
                        "fail": data["fail"],
                        "unexpected_boundary_attempts": data[
                            "unexpected_boundary_attempts"
                        ],
                    }
                )
            )
        finally:
            if server.poll() is None:
                server_stdout, _ = server.communicate("stop\n", timeout=30)
            (out / "transport-server.stdout").write_text(server_stdout)
            error_file.close()
            if endpoint != "none" and not rp.zmq_verdict(
                out, transport, server.returncode, server_stdout
            ):
                raise SystemExit(
                    f"ZMQ transport verdict failed; see {out / 'zmq-transport.json'}"
                )
            (out / "transport.json").write_text(json.dumps(transport, indent=2) + "\n")


if __name__ == "__main__":
    main()
