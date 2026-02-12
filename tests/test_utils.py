# SPDX-License-Identifier: Apache-2.0
"""Unit tests for lmcache.utils module."""
# Third Party
import pytest
import torch

# First Party
from lmcache.utils import (
    CacheEngineKey,
    LayerCacheEngineKey,
    parse_cache_key,
)


class TestCacheKeyParsing:
    """Tests for CacheEngineKey and LayerCacheEngineKey parsing."""

    def test_parse_cache_key_without_tags(self):
        """Test parsing CacheEngineKey without tags."""
        key_str = "model@2@0@abc123@bfloat16"
        result = parse_cache_key(key_str)
        
        assert isinstance(result, CacheEngineKey)
        assert not isinstance(result, LayerCacheEngineKey)
        assert result.model_name == "model"
        assert result.world_size == 2
        assert result.worker_id == 0
        assert result.chunk_hash == 0xabc123
        assert result.dtype == torch.bfloat16

    def test_parse_cache_key_with_tags(self):
        """Test parsing CacheEngineKey with tags - demonstrates the bug."""
        key_str = "model@2@0@abc123@bfloat16@mytag%myvalue@othertag%othervalue"
        result = parse_cache_key(key_str)
        
        # This should be CacheEngineKey, not LayerCacheEngineKey
        assert isinstance(result, CacheEngineKey)
        assert not isinstance(result, LayerCacheEngineKey)
        assert result.model_name == "model"
        assert result.world_size == 2
        assert result.worker_id == 0
        assert result.chunk_hash == 0xabc123
        assert result.dtype == torch.bfloat16
        assert result.tags is not None
        assert ("mytag", "myvalue") in result.tags
        assert ("othertag", "othervalue") in result.tags

    def test_parse_layer_cache_key_without_tags(self):
        """Test parsing LayerCacheEngineKey without tags."""
        key_str = "model@2@0@abc123@bfloat16@5"
        result = parse_cache_key(key_str)
        
        assert isinstance(result, LayerCacheEngineKey)
        assert result.model_name == "model"
        assert result.world_size == 2
        assert result.worker_id == 0
        assert result.chunk_hash == 0xabc123
        assert result.dtype == torch.bfloat16
        assert result.layer_id == 5

    def test_parse_layer_cache_key_with_tags(self):
        """Test parsing LayerCacheEngineKey with tags - demonstrates the bug."""
        key_str = "model@2@0@abc123@bfloat16@5@mytag%myvalue"
        result = parse_cache_key(key_str)
        
        # This should be LayerCacheEngineKey with layer_id=5 and tags
        assert isinstance(result, LayerCacheEngineKey)
        assert result.model_name == "model"
        assert result.world_size == 2
        assert result.worker_id == 0
        assert result.chunk_hash == 0xabc123
        assert result.dtype == torch.bfloat16
        assert result.layer_id == 5
        assert result.tags is not None
        assert ("mytag", "myvalue") in result.tags

    def test_roundtrip_cache_engine_key_without_tags(self):
        """Test that CacheEngineKey serialization round-trips correctly."""
        original = CacheEngineKey(
            model_name="test-model",
            world_size=4,
            worker_id=1,
            chunk_hash=0xdeadbeef,
            dtype=torch.float16,
            request_configs=None,
        )
        
        key_str = original.to_string()
        parsed = parse_cache_key(key_str)
        
        assert isinstance(parsed, CacheEngineKey)
        assert not isinstance(parsed, LayerCacheEngineKey)
        assert parsed == original

    def test_roundtrip_cache_engine_key_with_tags(self):
        """Test that CacheEngineKey with tags serialization round-trips correctly."""
        original = CacheEngineKey(
            model_name="test-model",
            world_size=4,
            worker_id=1,
            chunk_hash=0xdeadbeef,
            dtype=torch.float32,
            request_configs={
                "lmcache.tag.user": "alice",
                "lmcache.tag.session": "xyz789",
            },
        )
        
        key_str = original.to_string()
        parsed = parse_cache_key(key_str)
        
        assert isinstance(parsed, CacheEngineKey)
        assert not isinstance(parsed, LayerCacheEngineKey)
        assert parsed == original

    def test_roundtrip_layer_cache_engine_key_without_tags(self):
        """Test that LayerCacheEngineKey serialization round-trips correctly."""
        original = LayerCacheEngineKey(
            model_name="test-model",
            world_size=4,
            worker_id=1,
            chunk_hash=0xdeadbeef,
            dtype=torch.bfloat16,
            request_configs=None,
            layer_id=7,
        )
        
        key_str = original.to_string()
        parsed = parse_cache_key(key_str)
        
        assert isinstance(parsed, LayerCacheEngineKey)
        assert parsed == original
        assert parsed.layer_id == 7

    def test_roundtrip_layer_cache_engine_key_with_tags(self):
        """Test that LayerCacheEngineKey with tags serialization round-trips correctly."""
        original = LayerCacheEngineKey(
            model_name="test-model",
            world_size=4,
            worker_id=1,
            chunk_hash=0xdeadbeef,
            dtype=torch.bfloat16,
            request_configs={
                "lmcache.tag.user": "bob",
                "lmcache.tag.priority": "high",
            },
            layer_id=3,
        )
        
        key_str = original.to_string()
        parsed = parse_cache_key(key_str)
        
        assert isinstance(parsed, LayerCacheEngineKey)
        assert parsed == original
        assert parsed.layer_id == 3

    def test_bug_demonstration_dtype_as_digit(self):
        """Demonstrate the bug: if dtype were numeric, it might be misidentified."""
        # While actual dtypes aren't pure digits, the logic is flawed
        # The bug is that parts[4] is checked for isdigit(), but parts[4] is dtype
        # This test shows the conceptual bug in the parsing logic
        
        # A CacheEngineKey with tags where the first tag happens to be numeric
        key_str = "model@2@0@abc@bfloat16@123%value"  # tag key is "123"
        result = parse_cache_key(key_str)
        
        # Should be CacheEngineKey (it has a tag), not LayerCacheEngineKey
        # But the current buggy code checks parts[4] which is "bfloat16"
        assert isinstance(result, CacheEngineKey)
        assert not isinstance(result, LayerCacheEngineKey)

    def test_bug_key_formats(self):
        """Document the actual string formats to clarify the bug."""
        # Format breakdown:
        # CacheEngineKey:      model@ws@wid@hash@dtype[@tag%val...]
        # LayerCacheEngineKey: model@ws@wid@hash@dtype@layer[@tag%val...]
        #                      [0]   [1][2] [3]  [4]   [5]    [6+]
        
        # The bug: parse_cache_key checks parts[4].isdigit()
        # But parts[4] is always dtype (e.g., "bfloat16"), not layer_id
        # layer_id is at parts[5] for LayerCacheEngineKey
        
        cache_key = CacheEngineKey(
            model_name="m", world_size=1, worker_id=0,
            chunk_hash=0xff, dtype=torch.float16, request_configs=None
        )
        assert cache_key.to_string() == "m@1@0@ff@half"
        
        layer_key = LayerCacheEngineKey(
            model_name="m", world_size=1, worker_id=0,
            chunk_hash=0xff, dtype=torch.float16, request_configs=None, layer_id=9
        )
        assert layer_key.to_string() == "m@1@0@ff@half@9"
