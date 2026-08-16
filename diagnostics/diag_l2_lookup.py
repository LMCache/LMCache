#!/usr/bin/env python3
"""Diagnostic: check if LOOKUP chunk_hashes match DDN filenames.

This script:
1. Lists all .data files on DDN (including hash-prefix subdirectories) and extracts their chunk_hashes
2. Computes chunk_hashes for a test prompt using TokenHasher
3. Checks if FSL2Adapter._object_key_to_filename produces correct paths
4. Verifies _key_to_path returns True for existing files

Run inside the sgl-lmcache container on node 43.
"""
import asyncio
import os
import sys
import glob

# Add LMCache to path
sys.path.insert(0, "/LMCache")

from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.distributed.api import ObjectKey

# --- 1. List DDN files and extract chunk_hashes ---

DDN_PATH = "/ddn/glm5.2.lmcache-dsa-test"

print("=" * 80)
print("STEP 1: List DDN files and extract chunk_hashes")
print("=" * 80)

# Files are now stored under hash-prefix subdirectories: base_path/{hash[:2]}/{filename}
files = sorted(glob.glob(os.path.join(DDN_PATH, "**", "*.data"), recursive=True))
print(f"Found {len(files)} files on DDN")

ddn_chunk_hashes = set()
ddn_kv_ranks = set()
ddn_object_groups = set()

for f in files:
    basename = os.path.basename(f)
    subdir = os.path.basename(os.path.dirname(f))
    print(f"  {subdir}/{basename}")

    # Parse filename: 0x<kv_rank>@<obj_group>@<chunk_hash>[@<cache_salt>].data
    parts = basename.replace(".data", "").split("@")
    if len(parts) >= 3:
        kv_rank_hex = parts[0]  # 0x08000800
        obj_group_hex = parts[1]  # 0
        chunk_hash_hex = parts[2]  # hex string

        ddn_chunk_hashes.add(chunk_hash_hex)
        ddn_kv_ranks.add(kv_rank_hex)
        ddn_object_groups.add(obj_group_hex)

print(f"\nDistinct chunk_hashes on DDN: {len(ddn_chunk_hashes)}")
for h in ddn_chunk_hashes:
    print(f"  {h}")
print(f"\nDistinct kv_ranks on DDN: {sorted(ddn_kv_ranks)}")
print(f"Distinct object_groups on DDN: {sorted(ddn_object_groups)}")

# --- 2. Test FSL2Adapter path construction ---

print("\n" + "=" * 80)
print("STEP 2: Test FSL2Adapter._object_key_to_filename path construction")
print("=" * 80)

from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    _object_key_to_filename,
    _KEY_SEP,
    _FILE_EXT,
    _SUBDIR_HASH_PREFIX_LEN,
)

# Take a known chunk_hash from DDN and construct an ObjectKey
if ddn_chunk_hashes:
    test_chunk_hash = bytes.fromhex(list(ddn_chunk_hashes)[0])
else:
    test_chunk_hash = b"\x00" * 32
test_kv_rank = 0x08000800  # rank 0
test_model = "/data/models/ZhipuAI/GLM-5.2-FP8"  # the model_name used by sglang

test_key = ObjectKey(
    chunk_hash=test_chunk_hash,
    model_name=test_model,
    kv_rank=test_kv_rank,
    object_group_id=0,
    cache_salt="",
)

constructed_filename = _object_key_to_filename(test_key)
# Build path with hash-prefix subdirectory
chunk_hex = test_chunk_hash.hex()
subdir = chunk_hex[:_SUBDIR_HASH_PREFIX_LEN]
constructed_path = os.path.join(DDN_PATH, subdir, constructed_filename)
print(f"Model name: {test_model} (not embedded in filename)")
print(f"chunk_hash (hex): {test_chunk_hash.hex()}")
print(f"kv_rank: {hex(test_kv_rank)}")
print(f"object_group_id: 0")
print(f"Subdir (hash prefix): {subdir}")
print(f"Constructed filename: {constructed_filename}")
print(f"Constructed path: {constructed_path}")
print(f"File exists: {os.path.exists(constructed_path)}")

print(f"\n_KEY_SEP: {_KEY_SEP!r}")
print(f"_FILE_EXT: {_FILE_EXT!r}")
print(f"_SUBDIR_HASH_PREFIX_LEN: {_SUBDIR_HASH_PREFIX_LEN}")

# Check if the constructed filename matches any actual file
actual_basenames = [os.path.basename(f) for f in files]
if constructed_filename in actual_basenames:
    print(f"\n✅ Constructed filename MATCHES an actual DDN file!")
else:
    print(f"\n❌ Constructed filename does NOT match any DDN file!")
    print("   Looking for partial matches...")
    for bn in actual_basenames:
        if test_chunk_hash.hex() in bn:
            print(f"   Found file with matching chunk_hash: {bn}")

# --- 3. Test with all kv_ranks ---

print("\n" + "=" * 80)
print("STEP 3: Test path construction for all 8 kv_ranks")
print("=" * 80)

for rank in range(8):
    kv_rank = ObjectKey.ComputeKVRank(
        world_size=8,
        global_rank=rank,
        local_world_size=8,
        local_rank=rank,
    )
    test_key = ObjectKey(
        chunk_hash=test_chunk_hash,
        model_name=test_model,
        kv_rank=kv_rank,
        object_group_id=0,
        cache_salt="",
    )
    fname = _object_key_to_filename(test_key)
    chunk_hex = test_chunk_hash.hex()
    subdir = chunk_hex[:_SUBDIR_HASH_PREFIX_LEN]
    fpath = os.path.join(DDN_PATH, subdir, fname)
    exists = os.path.exists(fpath)
    status = "✅ EXISTS" if exists else "❌ MISSING"
    print(f"  rank {rank}: kv_rank={hex(kv_rank)} → {subdir}/{fname} [{status}]")

# --- 4. Compute chunk_hash for a test prompt ---

print("\n" + "=" * 80)
print("STEP 4: Compute chunk_hash for test prompts using TokenHasher")
print("=" * 80)

hasher = TokenHasher(chunk_size=256, hash_algorithm="blake3")

# Test with a simple prompt (just to verify the hasher works)
test_tokens = list(range(256))  # 256 dummy tokens
hashes = hasher.compute_chunk_hashes(test_tokens)
print(f"Test with 256 dummy tokens: {len(hashes)} chunks")
if hashes:
    print(f"  chunk 0 hash: {hashes[0].hex()}")

# Test with 512 tokens (2 chunks)
test_tokens_512 = list(range(512))
hashes_512 = hasher.compute_chunk_hashes(test_tokens_512)
print(f"Test with 512 dummy tokens: {len(hashes_512)} chunks")
if hashes_512:
    print(f"  chunk 0 hash: {hashes_512[0].hex()}")
    print(f"  chunk 1 hash: {hashes_512[1].hex()}")

# --- 5. Summary ---

print("\n" + "=" * 80)
print("STEP 5: Summary")
print("=" * 80)

print(f"DDN has {len(files)} files with {len(ddn_chunk_hashes)} distinct chunk_hashes")
print(f"DDN has {len(ddn_kv_ranks)} distinct kv_ranks: {sorted(ddn_kv_ranks)}")
print(f"DDN has {len(ddn_object_groups)} distinct object_groups: {sorted(ddn_object_groups)}")

# Verify all 8 ranks exist for each chunk_hash
for ch in ddn_chunk_hashes:
    matching_files = [f for f in files if ch in os.path.basename(f)]
    print(f"\n  chunk_hash {ch[:16]}...: {len(matching_files)} files (expected 8 for 8 GPUs)")

print("\n✅ Diagnostic complete. If path construction matches and files exist,")
print("   the L2 lookup issue is in the LOOKUP request itself (wrong token_ids or")
print("   chunk_hash computed during LOOKUP).")
print("\n   Next step: add debug logging to lookup.py to log computed chunk_hashes")
print("   during an actual LOOKUP request and compare against DDN filenames.")
