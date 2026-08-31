# SPDX-License-Identifier: Apache-2.0
"""Tests for the k3 CacheBlend plugin compatibility patch."""

# Standard
from importlib import util
from pathlib import Path
import types

# Third Party
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PATCH_SCRIPT = (
    REPO_ROOT / ".buildkite/k3_tests/blend/scripts/patch-cacheblend-plugin.py"
)


def _load_patch_module() -> types.ModuleType:
    """Load the patch script as a test module."""
    spec = util.spec_from_file_location("patch_cacheblend_plugin", PATCH_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_cacheblend_plugin_rewrites_legacy_request_type(
    tmp_path: Path,
) -> None:
    """Old CacheBlend RequestType calls are rewritten to descriptor RPC names."""
    module = _load_patch_module()
    source = tmp_path / "lmcache_cacheblend" / "mp_adapter.py"
    source.parent.mkdir()
    source.write_text(
        "\n".join(
            [
                "from lmcache.v1.multiprocess.protocol import RequestType",
                "",
                "def submit(request_type: RequestType) -> RequestType:",
                "    return request_type",
                "",
                "METHODS = [",
                "    RequestType.PING,",
                "    RequestType.CB_REGISTER_ROPE_V3,",
                "    RequestType.CB_UNIFIED_LOOKUP,",
                "]",
                "",
            ]
        )
    )

    patched = module.patch_cacheblend_plugin(tmp_path)

    assert patched == [source]
    assert source.read_text() == "\n".join(
        [
            "from lmcache.v1.multiprocess.transport.grpc_impl.protocol import RPC, "
            "RpcMethod",
            "",
            "def submit(request_type: RpcMethod) -> RpcMethod:",
            "    return request_type",
            "",
            "METHODS = [",
            "    RPC.Ping,",
            "    RPC.CbRegisterRope,",
            "    RPC.CbUnifiedLookup,",
            "]",
            "",
        ]
    )


def test_patch_cacheblend_plugin_preserves_other_protocol_imports(
    tmp_path: Path,
) -> None:
    """Mixed protocol imports keep non-RequestType names intact."""
    module = _load_patch_module()
    source = tmp_path / "lmcache_cacheblend" / "mp_adapter.py"
    source.parent.mkdir()
    source.write_text(
        "from lmcache.v1.multiprocess.protocol import RequestType, KeyType\n"
        "KEY = RequestType.LOOKUP\n"
    )

    module.patch_cacheblend_plugin(tmp_path)

    assert source.read_text() == (
        "from lmcache.v1.multiprocess.protocol import KeyType\n"
        "from lmcache.v1.multiprocess.transport.grpc_impl.protocol import RPC, "
        "RpcMethod\n"
        "KEY = RPC.Lookup\n"
    )


def test_patch_cacheblend_plugin_rejects_unknown_members(tmp_path: Path) -> None:
    """Unknown legacy names fail during setup instead of during vLLM startup."""
    module = _load_patch_module()
    source = tmp_path / "lmcache_cacheblend" / "mp_adapter.py"
    source.parent.mkdir()
    source.write_text(
        "from lmcache.v1.multiprocess.protocol import RequestType\n"
        "KEY = RequestType.DOES_NOT_EXIST\n"
    )

    with pytest.raises(RuntimeError, match="Unsupported RequestType member"):
        module.patch_cacheblend_plugin(tmp_path)
