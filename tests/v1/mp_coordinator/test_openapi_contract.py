# SPDX-License-Identifier: Apache-2.0
"""Guards the committed OpenAPI contract against drift from the FastAPI app."""

# Standard
from pathlib import Path
import json

# First Party
from lmcache.v1.mp_coordinator.utils.export_openapi import generate_spec, render_spec

SPEC_PATH = Path(__file__).parents[3] / "docs/design/v1/mp_coordinator/openapi.json"
REGEN_CMD = (
    "python -m lmcache.v1.mp_coordinator.utils.export_openapi "
    "docs/design/v1/mp_coordinator/openapi.json"
)


def test_committed_spec_matches_app():
    assert SPEC_PATH.is_file(), f"{SPEC_PATH} is missing; generate it with: {REGEN_CMD}"
    committed = json.loads(SPEC_PATH.read_text())
    assert committed == generate_spec(), (
        "openapi.json no longer matches the coordinator app. If the API "
        f"change is intentional, review the diff and regenerate with: {REGEN_CMD}"
    )


def test_committed_spec_is_canonically_formatted():
    text = SPEC_PATH.read_text()
    assert text == render_spec(json.loads(text)), (
        f"openapi.json is not in canonical form; regenerate with: {REGEN_CMD}"
    )
