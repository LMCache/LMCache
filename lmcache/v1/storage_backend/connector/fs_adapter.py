# SPDX-License-Identifier: Apache-2.0
# Standard
from urllib.parse import urlparse

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import (
    ConnectorAdapter,
    ConnectorContext,
    extract_plugin_type,
)
from lmcache.v1.storage_backend.connector.base_connector import (
    RemoteConnector,
)

logger = init_logger(__name__)

PLUGIN_TYPE = "fs"


def _extract_base_paths(url: str) -> str:
    """Extract the filesystem path component of an ``fs://`` URL.

    The authority is optional, as the scheme is documented:
    ``fs://[host:port]/path``. Both ``fs:///var/lmcache`` and
    ``fs://host:0/var/lmcache`` name ``/var/lmcache``.

    FSConnector stores through the local filesystem, so it needs a path and
    nothing else. This deliberately does not use ``parse_remote_url``, which
    requires a host and a port that this connector would discard.

    Args:
        url: An ``fs://`` URL, with or without an authority component.

    Returns:
        The path component of the URL. This may name several comma-separated
        paths, which FSConnector splits.

    Raises:
        ValueError: If the URL has no path component.
    """
    path = urlparse(url).path
    if not path:
        raise ValueError(
            f"Invalid fs URL '{url}': no path. Expected fs://[host:port]/path, "
            "for example fs:///var/lmcache"
        )
    return path


class FsConnectorAdapter(ConnectorAdapter):
    """Adapter for Filesystem connectors."""

    def __init__(self) -> None:
        super().__init__("fs://")

    def can_parse(self, url: str) -> bool:
        if url.startswith(self.schema):
            return True
        if url.startswith("plugin://"):
            pname = url[len("plugin://") :]
            return extract_plugin_type(pname) == PLUGIN_TYPE
        return False

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .fs_connector import FSConnector

        logger.info("Creating FS connector")

        # Legacy URL mode: extract base_path from URL
        base_paths_str = None
        if context.plugin_name is None:
            base_paths_str = _extract_base_paths(context.url)

        return FSConnector(
            context.loop,
            context.local_cpu_backend,
            context.config,
            plugin_name=context.plugin_name,
            base_paths_str=base_paths_str,
        )
