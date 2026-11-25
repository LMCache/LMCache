# SPDX-License-Identifier: Apache-2.0
"""Tests for the native extensions the lmcache-client wheel builds."""

# Third Party
import pytest

# First Party
from setup_extensions import BuildPolicy
from setup_extensions.storage_backend_profiles.nixl import NixlStorageBackend


@pytest.fixture
def nixl_requested_without_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Request the NIXL backend on a machine that has no NIXL SDK."""
    for name in ("NIXL_INCLUDE_DIR", "NIXL_LIBRARY_DIR", "NIXL_PREFIX"):
        monkeypatch.delenv(name, raising=False)
    for name in ("NO_NATIVE_EXT", "NO_CUDA_EXT"):
        monkeypatch.delenv(name, raising=False)
    if NixlStorageBackend().detect():
        pytest.skip("the NIXL SDK is installed on this machine")
    monkeypatch.setenv("BUILD_WITH_NIXL", "1")


@pytest.mark.usefixtures("nixl_requested_without_sdk")
def test_client_build_does_not_call_storage_backend_builders() -> None:
    """BUILD_WITH_NIXL=1 fails a full build without the SDK, not a client build."""
    with pytest.raises(RuntimeError, match="BUILD_WITH_NIXL=1 requires NIXL"):
        BuildPolicy.collect_extensions(None)

    ext_modules, cmdclass = BuildPolicy.collect_client_extensions(None)

    assert [ext.name for ext in ext_modules] == ["lmcache.lmcache_native"]
    assert "build_ext" in cmdclass
