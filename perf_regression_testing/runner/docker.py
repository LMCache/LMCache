# SPDX-License-Identifier: Apache-2.0
"""Thin wrappers around the docker CLI for the runner's containers.

Containers are named ``prt-<run>-<role>``, labelled ``prt.run=<run>``, and
run as the invoking host user so their files can be wiped without root.
"""

# Standard
from pathlib import Path
from typing import Any
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request


def run_args(
    name: str,
    run_id: str,
    image: str,
    container: dict[str, Any],
    mounts: list[dict[str, Any]],
    env: dict[str, str],
    gpus: list[int] | None,
    devices: list[str],
    cap_add: list[str],
    seccomp: str | None = None,
) -> list[str]:
    """Build the ``docker run`` options shared by every runner container.

    Args:
        name: Container name.
        run_id: Run id, stored as label ``prt.run``.
        image: Image reference or id.
        container: The ``[container]`` table.
        mounts: Mounts in ``[container].mounts`` form, placeholders resolved.
        env: Environment variables.
        gpus: GPU indices to expose, or None for no GPU.
        devices: Host device paths to pass through (``--device``).
        cap_add: Linux capabilities to add (``--cap-add``).
        seccomp: ``--security-opt seccomp=`` value, or None for docker's
            default profile.

    Returns:
        ``docker run`` arguments up to and including the image.
    """
    args = [
        "docker",
        "run",
        "--name",
        name,
        "--label",
        f"prt.run={run_id}",
        f"--network={container['network']}",
        f"--ipc={container['ipc']}",
        "--ulimit",
        f"memlock={container['ulimit_memlock']}",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
    ]
    if gpus is not None:
        args += ["--gpus", '"device=' + ",".join(map(str, gpus)) + '"']
    for d in devices:
        args += ["--device", d]
    for c in cap_add:
        args += ["--cap-add", c]
    if seccomp is not None:
        args += ["--security-opt", f"seccomp={seccomp}"]
    for m in mounts:
        args += ["-v", f"{m['path']}:{m['path']}" + (":ro" if m.get("ro") else "")]
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    return args + [image]


def start(args: list[str], cmd: list[str], log_path: Path) -> subprocess.Popen[bytes]:
    """Start ``docker run`` attached in a child process, logging to a file.

    Args:
        args: Output of :func:`run_args`.
        cmd: Command and arguments run inside the container.
        log_path: File that receives the container's stdout and stderr.

    Returns:
        The ``docker run`` client process.
    """
    with open(log_path, "w") as log:
        return subprocess.Popen(args + cmd, stdout=log, stderr=subprocess.STDOUT)


def status(name: str) -> str | None:
    """Return container ``name``'s state (``created``, ``running``, ``exited``...).

    Args:
        name: Container name.

    Returns:
        Docker's ``State.Status``, or None when the container does not exist.
    """
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", name],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip() if r.returncode == 0 else None


def cgroup_dir(name: str) -> Path:
    """Return the cgroup v2 directory of running container ``name``.

    Args:
        name: Name of a running container.

    Returns:
        The container's directory under ``/sys/fs/cgroup``.

    Raises:
        subprocess.CalledProcessError: the container does not exist.
    """
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Pid}}", name],
        capture_output=True,
        text=True,
        check=True,
    )
    with open(f"/proc/{int(out.stdout)}/cgroup") as f:
        (line,) = (x for x in f if x.startswith("0::"))
    return Path("/sys/fs/cgroup" + line[3:].strip())


def root_free_gb() -> float:
    """Return free GB on the filesystem holding docker's data root.

    Returns:
        Free space in GB (1e9 bytes).

    Raises:
        subprocess.CalledProcessError: the docker daemon is unreachable.
    """
    out = subprocess.run(
        ["docker", "info", "-f", "{{.DockerRootDir}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return shutil.disk_usage(out.stdout.strip()).free / 1e9


def image_id(image: str) -> str:
    """Return the content id of a local image.

    Args:
        image: Image tag or id.

    Returns:
        The id, ``sha256:...``.

    Raises:
        subprocess.CalledProcessError: the image does not exist locally.
    """
    out = subprocess.run(
        ["docker", "image", "inspect", "-f", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def image_label(image: str, key: str) -> str | None:
    """Return one label of a local image.

    Args:
        image: Image tag or id.
        key: Label name.

    Returns:
        The label's value, or None if the image has no such label.

    Raises:
        subprocess.CalledProcessError: the image does not exist locally.
    """
    out = subprocess.run(
        ["docker", "image", "inspect", "-f", "{{json .Config.Labels}}", image],
        capture_output=True,
        text=True,
        check=True,
    )
    return (json.loads(out.stdout) or {}).get(key)


def remove(names: list[str], timeout_s: float = 600) -> list[str]:
    """Stop and remove containers, retrying until they are gone.

    ``docker rm -f`` fails while a killed container is still dying, and the
    leftover name would block the next run, hence the retries.

    Args:
        names: Container names; ones that do not exist are skipped.
        timeout_s: Give up retrying after this many seconds.

    Returns:
        The names that still exist after ``timeout_s``.
    """
    subprocess.run(["docker", "stop", "-t", "120", *names], capture_output=True)
    deadline = time.time() + timeout_s
    left = names
    while True:
        subprocess.run(["docker", "rm", "-f", *left], capture_output=True)
        left = [n for n in left if status(n) is not None]
        if not left or time.time() > deadline:
            return left
        time.sleep(5)


def wait_http(url: str, timeout_s: float, name: str) -> bool:
    """Poll ``url`` until it answers 200, the container exits, or time runs out.

    Args:
        url: URL to GET.
        timeout_s: Give up after this many seconds.
        name: Container that serves ``url``; an exit ends the wait early.

    Returns:
        True when the URL answered, False otherwise.
    """
    start_t = time.time()
    deadline = start_t + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        # It may not exist yet, or stay "created" while the image unpacks.
        st = status(name)
        if st in ("exited", "dead") or (st is None and time.time() - start_t > 30):
            return False
        time.sleep(5)
    return False
