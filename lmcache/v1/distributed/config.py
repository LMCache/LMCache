# SPDX-License-Identifier: Apache-2.0

"""
Configuration for distributed storage manager
"""

# Standard
from dataclasses import dataclass, field
from typing import Literal
import argparse

# First Party
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdaptersConfig,
    add_l2_adapters_args,
    parse_args_to_l2_adapters_config,
)


@dataclass
class L1MemoryManagerConfig:
    """
    The configuration for L1 memory manager.
    """

    size_in_bytes: int
    """ The size of L1 memory in bytes. """

    use_lazy: bool
    """ Whether to use lazy initialization for L1 memory. """

    init_size_in_bytes: int = field(default=20 << 30)
    """ The initial size when using lazy allocation. Default is 20GB. """

    align_bytes: int = field(default=0x1000)
    """ The alignment size in bytes. Default is 4KB. """

    def __post_init__(self):
        self.init_size_in_bytes = min(self.init_size_in_bytes, self.size_in_bytes)


@dataclass
class L1ManagerConfig:
    """
    Special config for the L1 Object/Key manager
    """

    memory_config: L1MemoryManagerConfig
    """ The memory manager configuration for L1 cache. """

    write_ttl_seconds: int = field(default=600)
    """ Time to live for each object's write lock. Default is 600s (10 minutes). """

    read_ttl_seconds: int = field(default=300)
    """ Time to live for each object's read lock. Default is 300s (5 minutes). """


@dataclass
class EvictionConfig:
    """
    The configuration for eviction policies.
    """

    eviction_policy: Literal["LRU"]
    """ The eviction policy to use. Currently only 'LRU' is supported. """

    trigger_watermark: float = field(default=0.8)
    """ The memory usage watermark to trigger eviction (0.0 to 1.0). """

    eviction_ratio: float = field(default=0.2)
    """ The fraction of *allocated* memory to evict when triggered (0.0 to 1.0). """


@dataclass
class StorageManagerConfig:
    """
    The configuration for the distributed storage manager.
    """

    l1_manager_config: L1ManagerConfig
    """ The configuration for the L1 manager. """

    eviction_config: EvictionConfig
    """ The configuration for eviction policies. """

    l2_adapter_config: L2AdaptersConfig = field(
        default_factory=lambda: L2AdaptersConfig([])
    )
    """ The configuration for L2 adapters. """


def add_storage_manager_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Add storage manager configuration arguments to an existing parser.

    This function allows other modules to integrate storage manager arguments
    into their own argument parsers. Arguments are organized into groups to
    avoid naming conflicts with other modules.

    Args:
        parser: The argument parser to add arguments to.

    Returns:
        argparse.ArgumentParser: The same parser with storage manager
            arguments added.

    Example:
        >>> # In another module that needs its own arguments
        >>> parser = argparse.ArgumentParser(description="My Application")
        >>> parser.add_argument("--my-arg", type=str)
        >>> add_storage_manager_args(parser)
        >>> args = parser.parse_args()
        >>> config = parse_args_to_config(args)
    """
    # L1 Memory Manager Config
    memory_group = parser.add_argument_group(
        "L1 Memory Manager", "Configuration for L1 memory manager"
    )
    memory_group.add_argument(
        "--l1-size-gb",
        type=float,
        required=True,
        help="The size of L1 memory in GB.",
    )
    memory_group.add_argument(
        "--l1-use-lazy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use lazy loading for L1 memory. (Default is True)",
    )
    memory_group.add_argument(
        "--l1-init-size-gb",
        type=int,
        default=20,
        help="The initial size (GB) when using lazy allocation. Default is 20.",
    )
    memory_group.add_argument(
        "--l1-align-bytes",
        type=int,
        default=4096,
        help="The alignment size in bytes. Default is 4KB (4096 bytes).",
    )

    # L1 Manager Config (TTL settings)
    ttl_group = parser.add_argument_group(
        "L1 Manager TTL", "TTL configuration for L1 manager locks"
    )
    ttl_group.add_argument(
        "--l1-write-ttl-seconds",
        type=int,
        default=600,
        help="Time to live for each object's write lock. Default is 600s.",
    )
    ttl_group.add_argument(
        "--l1-read-ttl-seconds",
        type=int,
        default=300,
        help="Time to live for each object's read lock. Default is 300s.",
    )

    # Eviction Config
    eviction_group = parser.add_argument_group(
        "Eviction Policy", "Configuration for eviction policies"
    )
    eviction_group.add_argument(
        "--eviction-policy",
        type=str,
        choices=["LRU"],
        required=True,
        help="The eviction policy to use. Currently only 'LRU' is supported.",
    )
    eviction_group.add_argument(
        "--eviction-trigger-watermark",
        type=float,
        default=0.8,
        help="The memory usage watermark to trigger eviction (0.0 to 1.0). "
        "Default is 0.8.",
    )
    eviction_group.add_argument(
        "--eviction-ratio",
        type=float,
        default=0.2,
        help="The fraction of memory to evict when triggered (0.0 to 1.0). "
        "Default is 0.2.",
    )

    # Adapter config
    add_l2_adapters_args(parser)
    return parser


def get_arg_parser() -> argparse.ArgumentParser:
    """
    Get a standalone argument parser for storage manager configuration.

    This creates a new parser with only storage manager arguments.
    For integrating with other modules' parsers, use add_storage_manager_args()
    instead.

    Returns:
        argparse.ArgumentParser: The argument parser with all storage manager
            configuration options.
    """
    parser = argparse.ArgumentParser(
        description="Distributed Storage Manager Configuration"
    )
    return add_storage_manager_args(parser)


def parse_args_to_config(
    args: argparse.Namespace,
) -> StorageManagerConfig:
    """
    Convert parsed command line arguments to a StorageManagerConfig.

    Args:
        args: Parsed arguments from the argument parser.

    Returns:
        StorageManagerConfig: The configuration object.
    """
    memory_config = L1MemoryManagerConfig(
        size_in_bytes=int(args.l1_size_gb * (1 << 30)),
        use_lazy=args.l1_use_lazy,
        init_size_in_bytes=int(args.l1_init_size_gb * (1 << 30)),
        align_bytes=args.l1_align_bytes,
    )

    l1_manager_config = L1ManagerConfig(
        memory_config=memory_config,
        write_ttl_seconds=args.l1_write_ttl_seconds,
        read_ttl_seconds=args.l1_read_ttl_seconds,
    )

    eviction_config = EvictionConfig(
        eviction_policy=args.eviction_policy,
        trigger_watermark=args.eviction_trigger_watermark,
        eviction_ratio=args.eviction_ratio,
    )

    l2_adapter_config = parse_args_to_l2_adapters_config(args)

    return StorageManagerConfig(
        l1_manager_config=l1_manager_config,
        eviction_config=eviction_config,
        l2_adapter_config=l2_adapter_config,
    )


def parse_args(args: list[str] | None = None) -> StorageManagerConfig:
    """
    Parse command line arguments and return a StorageManagerConfig.

    This is a convenience function that combines get_arg_parser() and
    parse_args_to_config().

    Args:
        args: Optional list of arguments to parse. If None, uses sys.argv.

    Returns:
        StorageManagerConfig: The configuration object.
    """
    parser = get_arg_parser()
    parsed_args = parser.parse_args(args)
    return parse_args_to_config(parsed_args)
