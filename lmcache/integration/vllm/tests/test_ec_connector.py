# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import torch

from lmcache.integration.vllm.lmcache_ec_connector import LMCacheECConnector


class _FakeTransferConfig:
    def __init__(self, storage_path: str):
        self._storage_path = storage_path

    def get_from_extra_config(self, key: str, default=None):
        if key == "shared_storage_path":
            return self._storage_path
        return default


class _FakeVllmConfig:
    def __init__(self, storage_path: str):
        self.ec_transfer_config = _FakeTransferConfig(storage_path)


class _FakeRole:
    name = "WORKER"


def test_ec_roundtrip_save_then_load():
    with tempfile.TemporaryDirectory() as td:
        vllm_config = _FakeVllmConfig(td)
        conn = LMCacheECConnector(vllm_config=vllm_config, role=_FakeRole())

        # Mimic producer role
        conn.is_producer = True

        mm_hash = "hash_abc"
        x = torch.randn(7, 13, dtype=torch.float16)
        encoder_cache = {mm_hash: x}

        conn.save_caches(encoder_cache, mm_hash)

        fn = os.path.join(td, mm_hash, "encoder_cache.safetensors")
        assert os.path.exists(fn)

        # Minimal fake metadata plumbing: monkeypatch _get_connector_metadata
        from lmcache.integration.vllm.vllm_ec_adapter import (
            LMCacheECConnectorMetadata,
            MMMeta,
        )

        meta = LMCacheECConnectorMetadata()
        meta.add_mm_data(MMMeta.make_meta(mm_hash, x.shape[0]))
        conn._get_connector_metadata = lambda: meta  # type: ignore

        encoder_cache2 = {}
        conn.start_load_caches(encoder_cache2)

        assert mm_hash in encoder_cache2
        assert encoder_cache2[mm_hash].shape == x.shape
        assert encoder_cache2[mm_hash].dtype == x.dtype
