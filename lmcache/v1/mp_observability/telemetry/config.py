# SPDX-License-Identifier: Apache-2.0

"""Configuration for the telemetry event system."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import argparse
import json

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.telemetry.processors.base import (
    _PROCESSOR_CONFIG_REGISTRY,
    TelemetryProcessorConfig,
    get_registered_telemetry_processor_types,
)

logger = init_logger(__name__)


@dataclass
class TelemetryConfig:
    """Configuration for the telemetry event system.

    Attributes:
        enabled: Whether telemetry is enabled.
        max_queue_size: Maximum number of events in the queue before tail-drop.
        processor_configs: Ordered list of processor configs.
    """

    enabled: bool = False
    max_queue_size: int = 10000
    processor_configs: list[TelemetryProcessorConfig] = field(default_factory=list)


DEFAULT_TELEMETRY_CONFIG = TelemetryConfig()

_TELEMETRY_PROCESSOR_ARG_DEST = "telemetry_processor"


def add_telemetry_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add telemetry configuration arguments to an existing parser.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        The same parser with telemetry arguments added.
    """
    # Force registration of built-in processors so help text is accurate
    # First Party
    import lmcache.v1.mp_observability.telemetry.processors  # noqa: F401

    group = parser.add_argument_group(
        "Telemetry", "Configuration for the telemetry event system"
    )
    group.add_argument(
        "--enable-telemetry",
        action="store_true",
        default=False,
        help="Enable the telemetry event system.",
    )
    group.add_argument(
        "--telemetry-max-queue-size",
        type=int,
        default=10000,
        help="Maximum number of events in the telemetry queue before "
        "tail-drop. Default is 10000.",
    )
    group.add_argument(
        "--telemetry-processor",
        dest=_TELEMETRY_PROCESSOR_ARG_DEST,
        action="append",
        default=[],
        type=str,
        metavar="JSON",
        help='Processor spec as JSON with a "type" field and processor-specific '
        'configs, e.g. \'{"type":"logging"}\'. '
        "Repeat for multiple processors. "
        "Supported processors: ["
        + ", ".join(sorted(get_registered_telemetry_processor_types()))
        + "].",
    )
    return parser


def parse_args_to_telemetry_config(
    args: argparse.Namespace,
) -> TelemetryConfig:
    """Build TelemetryConfig from parsed command-line arguments.

    Args:
        args: Parsed arguments (e.g. from ``parser.parse_args()``).

    Returns:
        TelemetryConfig with one processor config per ``--telemetry-processor``.

    Raises:
        ValueError: If JSON is invalid or a processor type is unknown.
    """
    raw_list = getattr(args, _TELEMETRY_PROCESSOR_ARG_DEST, None)
    if raw_list is None:
        raw_list = []

    processor_configs: list[TelemetryProcessorConfig] = []
    for i, raw in enumerate(raw_list):
        try:
            d = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON for --telemetry-processor #{i + 1}: {e}"
            ) from e

        if not isinstance(d, dict):
            raise ValueError(
                f"--telemetry-processor #{i + 1}: expected a JSON object, "
                f"got {type(d).__name__}"
            )

        type_name = d.get("type")
        if type_name is None:
            raise ValueError(f"--telemetry-processor #{i + 1}: missing 'type' field")
        if type_name not in _PROCESSOR_CONFIG_REGISTRY:
            known = ", ".join(sorted(_PROCESSOR_CONFIG_REGISTRY)) or "(none)"
            raise ValueError(
                f"--telemetry-processor #{i + 1}: unknown processor type "
                f"{type_name!r}. Known: {known}"
            )

        config_cls = _PROCESSOR_CONFIG_REGISTRY[type_name]
        try:
            processor_configs.append(config_cls.from_dict(d))
        except (TypeError, ValueError) as e:
            logger.error(
                "Error parsing --telemetry-processor #%d (type %r): %s",
                i + 1,
                type_name,
                e,
            )
            logger.error(
                "Processor config help for %s processor:\n"
                "---------------------\n"
                "%s\n"
                "---------------------\n\n",
                type_name,
                config_cls.help(),
            )
            raise ValueError(
                f"--telemetry-processor #{i + 1} ({type_name!r}): {e}"
            ) from e

    return TelemetryConfig(
        enabled=args.enable_telemetry,
        max_queue_size=args.telemetry_max_queue_size,
        processor_configs=processor_configs,
    )
