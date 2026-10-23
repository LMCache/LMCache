# SPDX-License-Identifier: Apache-2.0
"""Unit tests for lmcache.utils module."""

# Standard
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import os
import stat

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import (
    CacheEngineKey,
    DaxDeviceGeometry,
    LayerCacheEngineKey,
    _append_range_or_elements,
    cdiv,
    check_dax_range,
    compress_slot_mapping,
    convert_tokens_to_list,
    decompress_slot_mapping,
    get_dax_device_geometry,
    get_version,
    parse_cache_key,
    parse_mixed_slot_mapping,
    round_down,
    round_up,
)
import lmcache.utils as lmcache_utils


# ============================================================
# Math utility functions
# ============================================================
class TestCdiv:
    def test_exact_division(self):
        assert cdiv(10, 5) == 2

    def test_ceiling_division(self):
        assert cdiv(7, 3) == 3

    def test_one_element(self):
        assert cdiv(1, 1) == 1

    def test_numerator_smaller_than_denominator(self):
        assert cdiv(1, 3) == 1

    def test_large_values(self):
        assert cdiv(1000000001, 1000000000) == 2


class TestRoundDown:
    def test_exact_multiple(self):
        assert round_down(10, 5) == 10

    def test_not_exact_multiple(self):
        assert round_down(7, 3) == 6

    def test_smaller_than_y(self):
        assert round_down(2, 5) == 0

    def test_zero(self):
        assert round_down(0, 5) == 0


class TestRoundUp:
    def test_exact_multiple(self):
        assert round_up(10, 5) == 10

    def test_not_exact_multiple(self):
        assert round_up(7, 3) == 9

    def test_smaller_than_y(self):
        assert round_up(2, 5) == 5

    def test_zero(self):
        assert round_up(0, 5) == 0


# ============================================================
# Slot mapping compression / decompression
# ============================================================
class TestCompressSlotMapping:
    def test_empty_list(self):
        assert compress_slot_mapping([]) == []

    def test_single_element(self):
        assert compress_slot_mapping([5]) == [5]

    def test_two_consecutive(self):
        # Two consecutive elements should NOT be compressed
        assert compress_slot_mapping([3, 4]) == [3, 4]

    def test_three_or_more_consecutive(self):
        assert compress_slot_mapping([1, 2, 3]) == [[1, 3]]
        assert compress_slot_mapping([1, 2, 3, 4, 5]) == [[1, 5]]

    def test_mixed(self):
        result = compress_slot_mapping([1, 2, 3, 4, 5, 7, 8])
        assert result == [[1, 5], 7, 8]

    def test_non_consecutive_order_preserved(self):
        # 1,2 is only 2 consecutive -> not compressed
        # 2,4 not consecutive -> separate
        result = compress_slot_mapping([5, 3, 1, 2, 4])
        assert result == [5, 3, 1, 2, 4]

    def test_multiple_ranges(self):
        result = compress_slot_mapping([1, 2, 3, 4, 5, 9, 10, 11, 12])
        assert result == [[1, 5], [9, 12]]

    def test_all_non_consecutive(self):
        result = compress_slot_mapping([1, 3, 5, 7])
        assert result == [1, 3, 5, 7]


class TestAppendRangeOrElements:
    def test_single_element(self):
        result = []
        _append_range_or_elements(result, 5, 5)
        assert result == [5]

    def test_two_elements(self):
        result = []
        _append_range_or_elements(result, 5, 6)
        assert result == [5, 6]

    def test_three_or_more_elements(self):
        result = []
        _append_range_or_elements(result, 1, 5)
        assert result == [[1, 5]]


class TestDecompressSlotMapping:
    def test_empty(self):
        assert decompress_slot_mapping([]) == []

    def test_single_int(self):
        assert decompress_slot_mapping([5]) == [5]

    def test_single_range(self):
        assert decompress_slot_mapping([[1, 5]]) == [1, 2, 3, 4, 5]

    def test_mixed(self):
        assert decompress_slot_mapping([[1, 5], 7, 8]) == [1, 2, 3, 4, 5, 7, 8]

    def test_roundtrip(self):
        """Compress then decompress should return original."""
        original = [1, 2, 3, 4, 5, 9, 10, 11, 12]
        compressed = compress_slot_mapping(original)
        assert decompress_slot_mapping(compressed) == original


class TestParseMixedSlotMapping:
    def test_simple_numbers(self):
        slots, err = parse_mixed_slot_mapping("1,2,3,17,19")
        assert err is None
        assert slots == [1, 2, 3, 17, 19]

    def test_range_format(self):
        slots, err = parse_mixed_slot_mapping("[9,12]")
        assert err is None
        assert slots == [9, 10, 11, 12]

    def test_mixed_format(self):
        slots, err = parse_mixed_slot_mapping("1,2,3,[9,12],17,19")
        assert err is None
        assert slots == [1, 2, 3, 9, 10, 11, 12, 17, 19]

    def test_whitespace_handling(self):
        slots, err = parse_mixed_slot_mapping(" 1 , 2 , [3 , 5] ")
        assert err is None
        assert slots == [1, 2, 3, 4, 5]

    def test_nested_brackets_error(self):
        slots, err = parse_mixed_slot_mapping("[[1,2]]")
        assert slots is None
        assert err is not None
        assert "error" in err

    def test_unmatched_closing_bracket(self):
        slots, err = parse_mixed_slot_mapping("1,2]")
        assert slots is None
        assert err is not None

    def test_unclosed_bracket(self):
        slots, err = parse_mixed_slot_mapping("[1,2")
        assert slots is None
        assert err is not None

    def test_invalid_range_start_gt_end(self):
        slots, err = parse_mixed_slot_mapping("[5,2]")
        assert slots is None
        assert err is not None

    def test_invalid_format(self):
        slots, err = parse_mixed_slot_mapping("abc")
        assert slots is None
        assert err is not None

    def test_empty_string(self):
        slots, err = parse_mixed_slot_mapping("")
        assert err is None
        assert slots == []


# ============================================================
# Version
# ============================================================
class TestGetVersion:
    def test_version_with_values(self):
        with (
            patch("lmcache.utils.VERSION", "1.0.0"),
            patch("lmcache.utils.COMMIT_ID", "abc123"),
        ):
            assert get_version() == "1.0.0-abc123"

    def test_version_empty(self):
        with patch("lmcache.utils.VERSION", ""), patch("lmcache.utils.COMMIT_ID", ""):
            assert get_version() == "NA-NA"

    def test_version_partial(self):
        with (
            patch("lmcache.utils.VERSION", "2.0"),
            patch("lmcache.utils.COMMIT_ID", ""),
        ):
            assert get_version() == "2.0-NA"


# ============================================================
# convert_tokens_to_list
# ============================================================
class TestConvertTokensToList:
    def test_none_input(self):
        assert convert_tokens_to_list(None, 0, 5) == []

    def test_tensor_input(self):
        t = torch.tensor([10, 20, 30, 40, 50])
        result = convert_tokens_to_list(t, 1, 3)
        assert result == [20, 30, 40]

    def test_list_input(self):
        tokens = [10, 20, 30, 40, 50]
        result = convert_tokens_to_list(tokens, 0, 2)
        assert result == [10, 20, 30]

    def test_full_range(self):
        t = torch.tensor([1, 2, 3])
        result = convert_tokens_to_list(t, 0, 2)
        assert result == [1, 2, 3]


# ============================================================
# CacheEngineKey
# ============================================================
class TestCacheEngineKey:
    @pytest.fixture()
    def basic_key(self):
        return CacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=0xABCD,
            dtype=torch.float16,
        )

    @pytest.fixture()
    def key_with_tags(self):
        return CacheEngineKey(
            model_name="model_a",
            world_size=2,
            worker_id=1,
            chunk_hash=0xFF,
            dtype=torch.bfloat16,
            request_configs={
                "lmcache.tag.lora": "adapter1",
                "lmcache.tag.version": "v2",
            },
        )

    def test_hash_and_eq(self, basic_key):
        key2 = CacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=0xABCD,
            dtype=torch.float16,
        )
        assert basic_key == key2
        assert hash(basic_key) == hash(key2)

    def test_not_equal_different_hash(self, basic_key):
        key2 = CacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=0x1234,
            dtype=torch.float16,
        )
        assert basic_key != key2

    def test_not_equal_different_type(self, basic_key):
        assert basic_key != "not_a_key"

    def test_to_string_basic(self, basic_key):
        s = basic_key.to_string()
        assert "test_model" in s
        assert "abcd" in s
        assert "half" in s

    def test_to_string_with_tags(self, key_with_tags):
        s = key_with_tags.to_string()
        assert "lora%adapter1" in s
        assert "version%v2" in s

    def test_from_string_roundtrip(self, basic_key):
        s = basic_key.to_string()
        restored = CacheEngineKey.from_string(s)
        assert restored == basic_key

    def test_from_string_with_tags_roundtrip(self, key_with_tags):
        s = key_with_tags.to_string()
        restored = CacheEngineKey.from_string(s)
        assert restored == key_with_tags

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            CacheEngineKey.from_string("invalid")

    def test_from_string_invalid_tag_format(self):
        with pytest.raises(ValueError):
            CacheEngineKey.from_string("model@1@0@abcd@half@badtag")

    def test_to_dict_and_from_dict(self, basic_key):
        d = basic_key.to_dict()
        assert d["__type__"] == "CacheEngineKey"
        restored = CacheEngineKey.from_dict(d)
        assert restored == basic_key

    def test_to_dict_with_tags(self, key_with_tags):
        d = key_with_tags.to_dict()
        assert "request_configs" in d

    def test_from_dict_invalid_tag(self):
        d = {
            "model_name": "m",
            "world_size": 1,
            "worker_id": 0,
            "chunk_hash": 0xAB,
            "dtype": "half",
            "request_configs": ["badformat"],
        }
        with pytest.raises(ValueError):
            CacheEngineKey.from_dict(d)

    def test_with_new_worker_id(self, basic_key):
        new_key = basic_key.with_new_worker_id(42)
        assert new_key.worker_id == 42
        assert new_key.model_name == basic_key.model_name
        assert new_key.chunk_hash == basic_key.chunk_hash

    def test_split_layers(self, basic_key):
        layers = basic_key.split_layers(3)
        assert len(layers) == 3
        for i, layer_key in enumerate(layers):
            assert isinstance(layer_key, LayerCacheEngineKey)
            assert layer_key.layer_id == i
            assert layer_key.model_name == "test_model"

    def test_get_first_layer(self, basic_key):
        first = basic_key.get_first_layer()
        assert isinstance(first, LayerCacheEngineKey)
        assert first.layer_id == 0
        assert first.model_name == basic_key.model_name

    def test_chunk_hash_hex(self, basic_key):
        assert basic_key.chunk_hash_hex == "abcd"

    def test_chunk_hash_hex_bytes(self):
        key = CacheEngineKey(
            model_name="m",
            world_size=1,
            worker_id=0,
            chunk_hash=0xFF,
            dtype=torch.float16,
        )
        # Manually set chunk_hash to bytes for coverage
        object.__setattr__(key, "chunk_hash", b"\xab\xcd")
        assert key.chunk_hash_hex == "abcd"

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            CacheEngineKey(
                model_name="m",
                world_size=1,
                worker_id=0,
                chunk_hash=0x1,
                dtype=torch.complex64,
            )

    def test_no_tags_when_no_tag_prefix(self):
        key = CacheEngineKey(
            model_name="m",
            world_size=1,
            worker_id=0,
            chunk_hash=0x1,
            dtype=torch.float16,
            request_configs={"some_config": "value"},
        )
        assert key.tags is None


# ============================================================
# LayerCacheEngineKey
# ============================================================
class TestLayerCacheEngineKey:
    @pytest.fixture()
    def layer_key(self):
        return LayerCacheEngineKey(
            model_name="model_b",
            world_size=1,
            worker_id=0,
            chunk_hash=0xBEEF,
            dtype=torch.float16,
            layer_id=5,
        )

    def test_hash_includes_layer_id(self, layer_key):
        other = LayerCacheEngineKey(
            model_name="model_b",
            world_size=1,
            worker_id=0,
            chunk_hash=0xBEEF,
            dtype=torch.float16,
            layer_id=6,
        )
        assert layer_key != other
        assert hash(layer_key) != hash(other)

    def test_eq_same_layer(self, layer_key):
        same = LayerCacheEngineKey(
            model_name="model_b",
            world_size=1,
            worker_id=0,
            chunk_hash=0xBEEF,
            dtype=torch.float16,
            layer_id=5,
        )
        assert layer_key == same

    def test_eq_different_base_key(self, layer_key):
        different = LayerCacheEngineKey(
            model_name="other_model",
            world_size=1,
            worker_id=0,
            chunk_hash=0xBEEF,
            dtype=torch.float16,
            layer_id=5,
        )
        assert layer_key != different

    def test_to_string(self, layer_key):
        s = layer_key.to_string()
        assert "model_b" in s
        assert "beef" in s
        assert "@5" in s

    def test_to_string_with_tags(self):
        key = LayerCacheEngineKey(
            model_name="m",
            world_size=1,
            worker_id=0,
            chunk_hash=0xAB,
            dtype=torch.float16,
            layer_id=2,
            request_configs={"lmcache.tag.foo": "bar"},
        )
        s = key.to_string()
        assert "foo%bar" in s

    def test_from_string_roundtrip(self, layer_key):
        s = layer_key.to_string()
        restored = LayerCacheEngineKey.from_string(s)
        assert restored == layer_key

    def test_from_string_with_tags(self):
        key = LayerCacheEngineKey(
            model_name="m",
            world_size=1,
            worker_id=0,
            chunk_hash=0xAB,
            dtype=torch.float16,
            layer_id=3,
            request_configs={"lmcache.tag.t1": "v1"},
        )
        s = key.to_string()
        restored = LayerCacheEngineKey.from_string(s)
        assert restored == key

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            LayerCacheEngineKey.from_string("a@b")

    def test_from_string_invalid_tag(self):
        with pytest.raises(ValueError):
            LayerCacheEngineKey.from_string("m@1@0@ab@half@3@badtag")

    def test_split_layers(self, layer_key):
        layers = layer_key.split_layers(4)
        assert len(layers) == 4
        for i, lk in enumerate(layers):
            assert lk.layer_id == i


# ============================================================
# parse_cache_key
# ============================================================
class TestCacheKeyParsing:
    def test_parse_cache_engine_key(self):
        key = CacheEngineKey(
            model_name="m",
            world_size=1,
            worker_id=0,
            chunk_hash=0xABC,
            dtype=torch.float16,
        )
        s = key.to_string()
        parsed = parse_cache_key(s)
        assert isinstance(parsed, CacheEngineKey)
        assert parsed == key

    @pytest.mark.xfail(
        reason="parse_cache_key bug: cannot parse LayerCacheEngineKey with dtype"
    )
    def test_parse_layer_cache_engine_key(self):
        key = LayerCacheEngineKey(
            model_name="m",
            world_size=1,
            worker_id=0,
            chunk_hash=0xABC,
            dtype=torch.float16,
            layer_id=7,
        )
        s = key.to_string()
        parsed = parse_cache_key(s)
        assert isinstance(parsed, LayerCacheEngineKey)
        assert parsed == key

    """
    Tests for CacheEngineKey and LayerCacheEngineKey parsing.
    """

    def test_parse_cache_key_without_tags(self):
        """
        Test parsing CacheEngineKey without tags.
        """
        key_str = "model@2@0@abc123@bfloat16"
        result = parse_cache_key(key_str)

        assert isinstance(result, CacheEngineKey)
        assert not isinstance(result, LayerCacheEngineKey)
        assert result.model_name == "model"
        assert result.world_size == 2
        assert result.worker_id == 0
        assert result.chunk_hash == 0xABC123
        assert result.dtype == torch.bfloat16

    def test_parse_cache_key_with_tags(self):
        """
        Test parsing CacheEngineKey with tags.
        """
        key_str = "model@2@0@abc123@bfloat16@mytag%myvalue@othertag%othervalue"
        result = parse_cache_key(key_str)

        # This should be CacheEngineKey, not LayerCacheEngineKey
        assert isinstance(result, CacheEngineKey)
        assert not isinstance(result, LayerCacheEngineKey)
        assert result.model_name == "model"
        assert result.world_size == 2
        assert result.worker_id == 0
        assert result.chunk_hash == 0xABC123
        assert result.dtype == torch.bfloat16
        assert result.tags is not None
        assert ("mytag", "myvalue") in result.tags
        assert ("othertag", "othervalue") in result.tags

    def test_parse_layer_cache_key_without_tags(self):
        """
        Test parsing LayerCacheEngineKey without tags.
        """
        key_str = "model@2@0@abc123@bfloat16@5"
        result = parse_cache_key(key_str)

        assert isinstance(result, LayerCacheEngineKey)
        assert result.model_name == "model"
        assert result.world_size == 2
        assert result.worker_id == 0
        assert result.chunk_hash == 0xABC123
        assert result.dtype == torch.bfloat16
        assert result.layer_id == 5

    def test_parse_layer_cache_key_with_tags(self):
        """
        Test parsing LayerCacheEngineKey with tags - demonstrates the bug.
        """
        key_str = "model@2@0@abc123@bfloat16@5@mytag%myvalue"
        result = parse_cache_key(key_str)

        # This should be LayerCacheEngineKey with layer_id=5 and tags
        assert isinstance(result, LayerCacheEngineKey)
        assert result.model_name == "model"
        assert result.world_size == 2
        assert result.worker_id == 0
        assert result.chunk_hash == 0xABC123
        assert result.dtype == torch.bfloat16
        assert result.layer_id == 5
        assert result.tags is not None
        assert ("mytag", "myvalue") in result.tags

    def test_roundtrip_cache_engine_key_without_tags(self):
        """
        Test that CacheEngineKey serialization round-trips correctly.
        """
        original = CacheEngineKey(
            model_name="test-model",
            world_size=4,
            worker_id=1,
            chunk_hash=0xDEADBEEF,
            dtype=torch.float16,
            request_configs=None,
        )

        key_str = original.to_string()
        parsed = parse_cache_key(key_str)

        assert isinstance(parsed, CacheEngineKey)
        assert not isinstance(parsed, LayerCacheEngineKey)
        assert parsed == original

    def test_roundtrip_cache_engine_key_with_tags(self):
        """
        Test that CacheEngineKey with tags serialization round-trips correctly.
        """
        original = CacheEngineKey(
            model_name="test-model",
            world_size=4,
            worker_id=1,
            chunk_hash=0xDEADBEEF,
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
        """
        Test that LayerCacheEngineKey serialization round-trips correctly.
        """
        original = LayerCacheEngineKey(
            model_name="test-model",
            world_size=4,
            worker_id=1,
            chunk_hash=0xDEADBEEF,
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
        """
        Test that LayerCacheEngineKey with tags serialization round-trips correctly.
        """
        original = LayerCacheEngineKey(
            model_name="test-model",
            world_size=4,
            worker_id=1,
            chunk_hash=0xDEADBEEF,
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


# ============================================================
# DAX device geometry
# ============================================================
_FAKE_RDEV = os.makedev(252, 7)


def _fake_devdax_node(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backing: Path,
    *,
    size: int | None,
    align: int = 2 * 1024 * 1024,
) -> None:
    """Make ``backing`` look like a Device-DAX node with the given sysfs attrs.

    ``fstat`` on the file reports a character device, and the sysfs node for
    its ``major:minor`` lives under ``tmp_path``. ``size=None`` leaves the node
    without a ``size`` attribute, as sysfs does for non-DAX character devices.
    """
    node = tmp_path / "sysfs" / f"{os.major(_FAKE_RDEV)}:{os.minor(_FAKE_RDEV)}"
    node.mkdir(parents=True)
    if size is not None:
        (node / "size").write_text(f"{size}\n")
    (node / "align").write_text(f"{align}\n")
    monkeypatch.setattr(lmcache_utils, "DAX_SYSFS_CHAR_ROOT", str(tmp_path / "sysfs"))

    real_fstat = os.fstat
    identity = (backing.stat().st_dev, backing.stat().st_ino)

    def fake_fstat(fd: int) -> object:
        info = real_fstat(fd)
        if (info.st_dev, info.st_ino) == identity:
            return SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=_FAKE_RDEV)
        return info

    monkeypatch.setattr(os, "fstat", fake_fstat)


class TestDaxDeviceGeometry:
    def test_regular_file_reports_size_and_no_alignment(self, tmp_path: Path):
        backing = tmp_path / "dax.bin"
        backing.write_bytes(b"\0" * 8192)
        fd = os.open(backing, os.O_RDWR)
        try:
            assert get_dax_device_geometry(fd) == DaxDeviceGeometry(
                capacity=8192, align=1
            )
            check_dax_range(str(backing), fd, 4096, 4096)
            with pytest.raises(RuntimeError, match="exceeds .* capacity"):
                check_dax_range(str(backing), fd, 4096, 4097)
        finally:
            os.close(fd)

    def test_character_device_reads_sysfs(self, tmp_path: Path, monkeypatch):
        backing = tmp_path / "dax0.0"
        backing.touch()
        _fake_devdax_node(monkeypatch, tmp_path, backing, size=8 << 20, align=1 << 20)
        fd = os.open(backing, os.O_RDWR)
        try:
            assert get_dax_device_geometry(fd) == DaxDeviceGeometry(
                capacity=8 << 20, align=1 << 20
            )
            check_dax_range("/dev/dax0.0", fd, 6 << 20, 2 << 20)
        finally:
            os.close(fd)

    def test_range_starting_at_capacity_is_rejected(self, tmp_path: Path, monkeypatch):
        # mmap of this range succeeds on real hardware and SIGBUSes on first
        # access, so it must be caught here.
        backing = tmp_path / "dax0.0"
        backing.touch()
        _fake_devdax_node(monkeypatch, tmp_path, backing, size=8 << 20)
        fd = os.open(backing, os.O_RDWR)
        try:
            with pytest.raises(RuntimeError, match="exceeds /dev/dax0.0 capacity"):
                check_dax_range("/dev/dax0.0", fd, 8 << 20, 2 << 20)
        finally:
            os.close(fd)

    @pytest.mark.parametrize("offset,size", [(1 << 20, 2 << 20), (0, 3 << 20)])
    def test_misaligned_range_is_rejected(
        self, tmp_path: Path, monkeypatch, offset: int, size: int
    ):
        backing = tmp_path / "dax0.0"
        backing.touch()
        _fake_devdax_node(monkeypatch, tmp_path, backing, size=8 << 20)
        fd = os.open(backing, os.O_RDWR)
        try:
            with pytest.raises(RuntimeError, match="device alignment"):
                check_dax_range("/dev/dax0.0", fd, offset, size)
        finally:
            os.close(fd)

    def test_character_device_without_sysfs_size_fails_closed(
        self, tmp_path: Path, monkeypatch
    ):
        backing = tmp_path / "null"
        backing.touch()
        _fake_devdax_node(monkeypatch, tmp_path, backing, size=None)
        fd = os.open(backing, os.O_RDWR)
        try:
            with pytest.raises(RuntimeError, match="cannot read Device-DAX size"):
                check_dax_range("/dev/null", fd, 0, 2 << 20)
        finally:
            os.close(fd)
