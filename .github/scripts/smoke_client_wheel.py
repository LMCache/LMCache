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
    """Verify wheel ownership, native/module boundaries and client imports."""
    distribution = metadata.distribution("lmcache-client")
    init = distribution.locate_file("lmcache/__init__.py")
    assert Path(lmcache.__file__).samefile(str(init)), lmcache.__file__
    files = {str(path) for path in distribution.files or ()}
    assert any(path.startswith("lmcache/lmcache_native.") for path in files)
    assert not any(
        path.startswith(("lmcache/lmcache_redis.", "lmcache/lmcache_fs."))
        for path in files
    )
    assert "lmcache/v1/multiprocess/server.py" not in files
    assert not any(
        path.startswith(
            (
                "lmcache/inprocess/",
                "lmcache/v1/gpu_connector/",
                "lmcache/v1/distributed/",
                "lmcache/v1/mp_coordinator/",
                "lmcache/v1/mp_observability/",
            )
        )
        for path in files
    )
    for module in (
        "lmcache.sdk",
        "lmcache.v1.client.worker_transfer",
        "lmcache.v1.platform.devices.cpu.cache_context",
        "lmcache.v1.platform.devices.gcu.cache_context",
        "lmcache.v1.multiprocess.custom_types",
        "lmcache.v1.multiprocess.group_view",
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
