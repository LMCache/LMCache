#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Try to create a NIXL agent with the UCX backend under the current environment.

Run it with the same UCX_* variables you plan to give the LMCache server; it fails
in the same way the server would at startup (``NIXL_ERR_BACKEND``), in about two
seconds and without touching a running deployment. Pass ``--tls`` to test candidate
UCX_TLS values.

usage: nixl_probe.py [--tls rc,sm,self,cuda_copy] [--env KEY=VALUE ...]
exit 0 when the backend comes up, 1 otherwise.
"""

# Standard
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tls")
    ap.add_argument("--env", nargs="*", default=[])
    a = ap.parse_args()
    if a.tls:
        os.environ["UCX_TLS"] = a.tls
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    shown = {k: v for k, v in os.environ.items() if k.startswith(("UCX_", "NIXL_"))}
    print("env:", shown or "(no UCX_/NIXL_ variables)")
    try:
        # Third Party
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as e:
        print(
            f"FAIL: nixl not importable ({e}); "
            "install with `uv pip install -r requirements/nixl.txt`"
        )
        return 1
    try:
        agent = nixl_agent("probe", nixl_agent_config(backends=["UCX"]))
    except Exception as e:  # noqa: BLE001 - report whatever the backend raised
        print(f"FAIL: {type(e).__name__}: {e}")
        print(
            "hint: on GPU hosts nixl>=1.5 requires cuda_copy in UCX_TLS, "
            "e.g. rc,sm,self,cuda_copy; "
            "the exact reason is in the E...config.cpp line printed above"
        )
        return 1
    print("OK: UCX backend instantiated; backends =", list(agent.backends))
    return 0


if __name__ == "__main__":
    sys.exit(main())
