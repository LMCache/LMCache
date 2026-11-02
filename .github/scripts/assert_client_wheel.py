# SPDX-License-Identifier: Apache-2.0
"""Validate the installed lmcache-client wheel from outside the source tree."""

# Standard
from importlib import metadata, util

# First Party
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
import lmcache
import lmcache.lmcache_native


def main() -> None:
    """Assert client metadata, imports, and extension boundaries."""
    distribution = metadata.distribution("lmcache-client")
    assert distribution.metadata["Name"] == "lmcache-client"
    assert lmcache.__version__ != "unknown"
    assert RequestClientFactory is not None
    assert util.find_spec("lmcache.lmcache_redis") is None
    assert util.find_spec("lmcache.lmcache_fs") is None
    assert not any(
        entry_point.group == "console_scripts" and entry_point.name == "lmcache"
        for entry_point in distribution.entry_points
    )

    requirements = distribution.requires or []
    forbidden = ("fastapi", "google-cloud-bigtable", "redis", "uvicorn")
    assert not any(
        requirement.lower().startswith(forbidden) for requirement in requirements
    )


if __name__ == "__main__":
    main()
