# SPDX-License-Identifier: Apache-2.0
"""Check the installed client wheel outside the source checkout.

Run with the clean installation's interpreter and -I to prevent checkout imports.
"""

# Standard
from importlib import import_module, metadata
from pathlib import Path

# Third Party
import torch  # noqa: F401 -- native extensions require torch to be loaded first

# First Party
from lmcache.v1.client.transport.factory import RequestClientFactory
import lmcache


def main() -> None:
    """Verify wheel ownership, the native extension and client imports."""
    distribution = metadata.distribution("lmcache-client")
    init = distribution.locate_file("lmcache/__init__.py")
    assert Path(lmcache.__file__).samefile(str(init)), lmcache.__file__
    for module in (
        "lmcache.lmcache_native",
        "lmcache.sdk",
        "lmcache.v1.client.worker_transfer",
        "lmcache.v1.platform.devices.cpu.cache_context",
        "lmcache.v1.platform.devices.gcu.cache_context",
        "lmcache.cli.metrics",
    ):
        import_module(module)
        print("Imported", module)
    for url in ("127.0.0.1:1", "grpc://127.0.0.1:1", "tcp://127.0.0.1:1"):
        RequestClientFactory.create(url).close()
        print("Created and closed", url)
    print("Client wheel smoke passed:", distribution.version, lmcache.__file__)


if __name__ == "__main__":
    main()
