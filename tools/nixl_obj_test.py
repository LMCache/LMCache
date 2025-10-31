#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Utility for validating NIXL-backed S3 connectivity."""

# Standard
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

# Third Party
from nixl._api import nixl_agent as NixlAgent
from nixl._api import nixl_agent_config as NixlAgentConfig
from nixl._api import nixlBind

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig

LOGGER = init_logger("tools.nixl_s3_probe")


def _load_engine_config(config_path: Path) -> LMCacheEngineConfig:
    """Load an :class:`LMCacheEngineConfig` from a YAML file."""
    try:
        config = LMCacheEngineConfig.from_file(str(config_path))
    except FileNotFoundError as exc:
        raise SystemExit(f"Configuration file not found: {config_path}") from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        raise SystemExit(f"Failed to load configuration: {exc}") from exc

    return config


def _extract_nixl_backend_info(
    config: LMCacheEngineConfig,
) -> tuple[str, dict[str, str]]:
    """Extract the NIXL backend information from the config."""
    extra_config = config.extra_config or {}

    if not extra_config.get("enable_nixl_object"):
        raise SystemExit(
            "Configuration does not enable the NIXL object backend. "
            "Set extra_config.enable_nixl_object to true."
        )

    backend = extra_config.get("nixl_object_backend")
    if not backend:
        raise SystemExit("Missing 'nixl_object_backend' in extra_config.")

    backend_params = extra_config.get("nixl_object_backend_params") or {}
    if not isinstance(backend_params, dict):
        raise SystemExit("'nixl_object_backend_params' must be a mapping.")

    # Ensure all parameters are strings for the NIXL API
    normalized_params = {str(key): str(value) for key, value in backend_params.items()}

    LOGGER.debug(
        "Using NIXL backend %s with params: %s", backend, json.dumps(normalized_params)
    )

    return backend, normalized_params


def _create_nixl_agent(backend: str, backend_params: dict[str, str]) -> NixlAgent:
    """Instantiate a temporary NIXL agent for probing."""
    agent_name = "NixlObjectAgent_" + str(uuid.uuid4())
    agent_config = NixlAgentConfig(backends=[])
    agent = NixlAgent(agent_name, agent_config)
    agent.create_backend(backend, backend_params)
    return agent

def parse_s3_object_name(s3_uri: str) -> str:
    if not s3_uri.startswith("s3://"):
        raise ValueError("S3 URI must start with 's3://'")
    
    # Remove s3:// prefix
    uri_without_prefix = s3_uri.removeprefix("s3://")
    
    # Split by first slash to separate bucket from object path
    parts = uri_without_prefix.split("/", 1)
    
    if len(parts) < 2:
        raise ValueError("S3 URI must contain both bucket and object path")
    
    # Return only the object name part (everything after bucket/)
    return parts[1]

def _check_object_exists(agent: NixlAgent, object_uri: str) -> bool:
    """Query the NIXL backend to determine if the object exists."""

    reg_list = [(0, 0, 0, parse_s3_object_name(object_uri))]

    try:
        response = agent.query_memory(reg_list, "OBJ", mem_type="OBJ")
    except nixlBind.nixlBackendError as exc:
        LOGGER.error("NIXL backend reported an error: %s", exc)
        raise

    if not response:
        return False

    return response[0] is not None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate connectivity to a NIXL-backed S3 object store using an LMCache "
            "configuration file."
        )
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the LMCache YAML configuration file.",
    )
    parser.add_argument(
        "s3_uri",
        type=str,
        help="S3 URI to check (for example: s3://bucket/path/to/object).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    config = _load_engine_config(args.config)
    backend, backend_params = _extract_nixl_backend_info(config)

    LOGGER.info("Initializing NIXL agent for backend '%s'", backend)

    try:
        agent = _create_nixl_agent(backend, backend_params)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.error("Failed to create NIXL agent: %s", exc)
        return 2

    try:
        exists = _check_object_exists(agent, args.s3_uri)
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.error(
            "Connectivity test failed for URI '%s'. See logs for details.", args.s3_uri
        )
        return 2
    finally:
        shutdown = getattr(agent, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.debug("Ignoring error during agent shutdown", exc_info=True)

    if exists:
        LOGGER.info("S3 object '%s' exists.", args.s3_uri)
        return 0

    LOGGER.warning("S3 object '%s' does not exist.", args.s3_uri)
    return 1


if __name__ == "__main__":
    sys.exit(main())
