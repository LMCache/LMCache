# SPDX-License-Identifier: Apache-2.0
"""Export the coordinator's OpenAPI schema as a stable JSON document.

The committed copy at ``docs/design/v1/mp_coordinator/openapi.json`` is the
coordinator's REST contract: any alternative implementation (e.g. a native
coordinator) is API-compatible exactly when it serves this contract, and
``tests/v1/mp_coordinator/test_openapi_contract.py`` fails whenever the
FastAPI app drifts from the committed file. After an intentional API change,
regenerate with::

    python -m lmcache.v1.mp_coordinator.utils.export_openapi \\
        docs/design/v1/mp_coordinator/openapi.json
"""

# Standard
from pathlib import Path
from typing import Any
import argparse
import json
import sys

# First Party
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig


def generate_spec() -> dict[str, Any]:
    """Build the coordinator app and return its OpenAPI schema.

    Background loops are disabled via config; only the route table and
    request/response models matter for schema generation.

    Returns:
        The OpenAPI schema as a JSON-serializable dict.
    """
    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        enable_startup_resync=False,
    )
    return create_app(config).openapi()


def render_spec(spec: dict[str, Any]) -> str:
    """Serialize an OpenAPI schema to canonical JSON.

    Sorted keys and fixed indentation keep regeneration deterministic, so
    contract changes always show up as minimal diffs.

    Args:
        spec: The OpenAPI schema to serialize.

    Returns:
        The canonical JSON text, newline-terminated.
    """
    return json.dumps(spec, indent=2, sort_keys=True) + "\n"


def main() -> None:
    """Write the coordinator's OpenAPI schema to a file or stdout."""
    parser = argparse.ArgumentParser(
        description="Export the coordinator's OpenAPI schema."
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Destination path. Writes to stdout when omitted.",
    )
    args = parser.parse_args()

    text = render_spec(generate_spec())
    if args.output is None:
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
