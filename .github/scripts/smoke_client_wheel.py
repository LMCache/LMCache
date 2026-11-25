# SPDX-License-Identifier: Apache-2.0
"""Check the installed lmcache-client wheel; run with -I outside the checkout."""

# Standard
from importlib import metadata
from pathlib import Path

# Third Party
import torch  # noqa: F401 -- the native extension needs torch loaded first

# First Party
from lmcache.v1.client.transport.factory import RequestClientFactory
import lmcache
import lmcache.lmcache_native  # noqa: F401 -- lmcache/__init__.py hides a failed load

# Server and in-process paths that the full lmcache wheel keeps.
FULL_WHEEL_ONLY = (
    "lmcache/cli/",
    "lmcache/v1/multiprocess/",
    "lmcache/integration/tensorrt_llm/tensorrt_adapter.py",
)


def main() -> None:
    """Check file ownership and exclusions, then create a request client."""
    distribution = metadata.distribution("lmcache-client")
    init = distribution.locate_file("lmcache/__init__.py")
    assert Path(lmcache.__file__).samefile(str(init)), lmcache.__file__
    leaked = [
        str(f) for f in distribution.files or () if str(f).startswith(FULL_WHEEL_ONLY)
    ]
    assert not leaked, leaked
    RequestClientFactory.create("grpc://127.0.0.1:1").close()
    print("Client wheel smoke passed:", distribution.version, lmcache.__file__)


if __name__ == "__main__":
    main()
