# SPDX-License-Identifier: Apache-2.0
"""Load the P2P deployment env.sh (written by lmcache-p2p-launch/scripts/gen_env.sh).

Python clients call ``load()`` to get the variables as a dict and derive defaults
(``VLLM_URL``, ``PEER_VLLM_URL``, ``MODEL``, ...). The file is resolved like the bash
scripts do: ``$P2P_ENV``, else ``$P2P_HOME/env.sh``, else ``~/lmcache-p2p/env.sh``.
"""

# Standard
import os
import subprocess


def env_path() -> str:
    """Return the env.sh path the shell scripts would use."""
    if os.environ.get("P2P_ENV"):
        return os.environ["P2P_ENV"]
    home = os.environ.get("P2P_HOME", os.path.expanduser("~/lmcache-p2p"))
    return os.path.join(home, "env.sh")


def load() -> dict[str, str]:
    """Source env.sh in a subshell and return its exported variables.

    Returns an empty dict when the file does not exist, so scripts can still run
    with explicit ``--a/--b`` URLs.
    """
    path = env_path()
    if not os.path.isfile(path):
        return {}
    out = subprocess.run(
        ["bash", "-c", f'source "{path}" >/dev/null 2>&1 && env -0'],
        capture_output=True,
        check=True,
    ).stdout.decode()
    return dict(kv.split("=", 1) for kv in out.split("\0") if "=" in kv)


def url_defaults(env: dict[str, str]) -> tuple[str, str, str, str]:
    """Return (vllm_a, vllm_b, http_a, http_b): node A is this env, node B its peer."""
    return (
        env.get("VLLM_URL", "http://127.0.0.1:8300"),
        env.get("PEER_VLLM_URL", "http://127.0.0.1:8301"),
        env.get("HTTP_URL", "http://127.0.0.1:7700"),
        env.get("PEER_HTTP_URL", "http://127.0.0.1:7701"),
    )
