# SPDX-License-Identifier: Apache-2.0
"""Patch the external CacheBlend plugin for the current LMCache RPC names."""

# Standard
from pathlib import Path
import re
import sys

LEGACY_REQUEST_TYPE_TO_RPC = {
    "REGISTER_KV_CACHE": "RegisterKvCache",
    "REGISTER_Q_CACHE": "RegisterQCache",
    "UNREGISTER_KV_CACHE": "UnregisterKvCache",
    "UNREGISTER_Q_CACHE": "UnregisterQCache",
    "STORE_Q": "StoreQ",
    "STORE": "Store",
    "RETRIEVE": "Retrieve",
    "LOOKUP": "Lookup",
    "QUERY_PREFETCH_STATUS": "QueryPrefetchStatus",
    "WAIT_PREFETCH_STATUS": "WaitPrefetchStatus",
    "QUERY_PREFETCH_LOOKUP_HITS": "QueryPrefetchLookupHits",
    "FREE_LOOKUP_LOCKS": "FreeLookupLocks",
    "END_SESSION": "EndSession",
    "REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT": "RegisterKvCacheEngineDrivenContext",
    "UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT": "UnregisterKvCacheEngineDrivenContext",
    "PREPARE_STORE": "PrepareStore",
    "COMMIT_STORE": "CommitStore",
    "PREPARE_RETRIEVE": "PrepareRetrieve",
    "COMMIT_RETRIEVE": "CommitRetrieve",
    "CLEAR": "Clear",
    "GET_CHUNK_SIZE": "GetChunkSize",
    "GET_EXPERIMENTAL": "GetExperimental",
    "PING": "Ping",
    "REPORT_BLOCK_ALLOCATION": "ReportBlockAllocation",
    "NOOP": "Noop",
    "CB_REGISTER_ROPE": "CbRegisterRope",
    "CB_UNREGISTER_ROPE": "CbUnregisterRope",
    "CB_RETRIEVE_PRE_COMPUTED": "CbRetrievePreComputed",
    "CB_UNIFIED_LOOKUP": "CbUnifiedLookup",
    "CB_REGISTER_ROPE_V3": "CbRegisterRope",
    "CB_UNREGISTER_ROPE_V3": "CbUnregisterRope",
    "CB_RETRIEVE_PRE_COMPUTED_V3": "CbRetrievePreComputed",
    "P2P_LOOKUP_AND_LOCK": "P2PLookupAndLock",
    "P2P_QUERY_LOOKUP_RESULTS": "P2PQueryLookupResults",
    "P2P_UNLOCK_OBJECTS": "P2PUnlockObjects",
}

PROTOCOL_IMPORT = re.compile(
    r"^from lmcache\.v1\.multiprocess\.protocol import (?P<imports>.+)$"
)
REQUEST_TYPE_ACCESS = re.compile(r"\bRequestType\.([A-Z0-9_]+)\b")


def patch_cacheblend_plugin(plugin_dir: Path) -> list[Path]:
    """Patch CacheBlend plugin Python files to use descriptor RPC tokens.

    Args:
        plugin_dir: Root of the cloned ``cacheblend-plugin`` checkout.

    Returns:
        The list of files modified by the patch.

    Raises:
        RuntimeError: If a file still contains unsupported ``RequestType``
            references after applying known replacements.
    """
    if not plugin_dir.is_dir():
        raise RuntimeError(f"CacheBlend plugin directory not found: {plugin_dir}")

    patched_files = []
    for path in plugin_dir.rglob("*.py"):
        original = path.read_text()
        patched = _patch_protocol_imports(original)
        patched = _patch_request_type_accesses(patched, path)
        patched = _patch_remaining_request_type_names(patched)

        if patched == original:
            continue

        path.write_text(patched)
        patched_files.append(path)

    return patched_files


def _patch_protocol_imports(source: str) -> str:
    """Replace single-line RequestType imports from LMCache protocol."""
    lines = []
    for line in source.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        match = PROTOCOL_IMPORT.match(body)
        if match is None:
            lines.append(line)
            continue

        names = [name.strip() for name in match.group("imports").split(",")]
        if "RequestType" not in names:
            lines.append(line)
            continue

        names = ["RPC" if name == "RequestType" else name for name in names]
        if "RpcMethod" not in names:
            names.append("RpcMethod")
        lines.append(
            f"from lmcache.v1.multiprocess.protocol import {', '.join(names)}{newline}"
        )

    return "".join(lines)


def _patch_request_type_accesses(source: str, path: Path) -> str:
    """Replace known ``RequestType.X`` accesses with ``RPC.CamelCase``."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        rpc_name = LEGACY_REQUEST_TYPE_TO_RPC.get(name)
        if rpc_name is None:
            raise RuntimeError(f"Unsupported RequestType member {name} in {path}")
        return f"RPC.{rpc_name}"

    return REQUEST_TYPE_ACCESS.sub(replace, source)


def _patch_remaining_request_type_names(source: str) -> str:
    """Replace bare legacy ``RequestType`` references with ``RpcMethod``."""
    return re.sub(r"\bRequestType\b", "RpcMethod", source)


def main(argv: list[str]) -> int:
    """Run the CacheBlend plugin patch from the command line."""
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <cacheblend-plugin-dir>", file=sys.stderr)
        return 2

    patched_files = patch_cacheblend_plugin(Path(argv[1]))
    if patched_files:
        print("Patched CacheBlend plugin files for LMCache descriptor RPC names:")
        for path in patched_files:
            print(f"  {path}")
    else:
        print("CacheBlend plugin already uses LMCache descriptor RPC names.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
