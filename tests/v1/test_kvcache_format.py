# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.check.utils import _get_default_metadata
from lmcache.v1.gpu_connector import VLLMPagedMemGPUConnectorV2
from lmcache.v1.kvcache_format import (
    KV_FORMAT_SCHEMA_VERSION,
    KVCacheFormat,
    KVLayerGroupSpec,
    L0LayoutSpec,
    build_dense_format_single_group,
    deserialize_kvcache_format,
    serialize_kvcache_format,
    validate_kvcache_format,
)
from lmcache.v1.metadata import LMCacheMetadata


def test_round_trip_serialization():
    fmt = build_dense_format_single_group(
        num_layers=2,
        dtype=torch.float16,
        hidden_dim=128,
        block_size=256,
        use_mla=False,
        separation="packed",
    )
    encoded = serialize_kvcache_format(fmt)
    decoded = deserialize_kvcache_format(encoded)
    assert decoded == fmt


def test_validation_rejects_invalid_block_size():
    fmt = KVCacheFormat(
        layout="MHA_DENSE_PACKED",
        l0=L0LayoutSpec(
            addressing="gpu_block_ids",
            block_size=0,  # invalid
        ),
        layer_groups=[
            KVLayerGroupSpec(
                start_layer=0,
                num_layers=1,
                dtype="torch.float16",
                hidden_dim=128,
            )
        ],
        format_id="mha_dense/packed/v1",
    )
    with pytest.raises(ValueError):
        validate_kvcache_format(fmt)


def test_validation_rejects_overlapping_groups():
    fmt = KVCacheFormat(
        layout="MHA_DENSE_PACKED",
        l0=L0LayoutSpec(
            addressing="gpu_block_ids",
            block_size=256,
        ),
        layer_groups=[
            KVLayerGroupSpec(
                start_layer=0,
                num_layers=2,
                dtype="torch.float16",
                hidden_dim=128,
            ),
            KVLayerGroupSpec(
                start_layer=1,  # overlaps previous group
                num_layers=1,
                dtype="torch.float16",
                hidden_dim=128,
            ),
        ],
        format_id="mha_dense/packed/v1",
    )
    with pytest.raises(ValueError):
        validate_kvcache_format(fmt)


def test_schema_version_mismatch_rejected():
    fmt = KVCacheFormat(
        layout="MHA_DENSE_PACKED",
        l0=L0LayoutSpec(
            addressing="gpu_block_ids",
            block_size=256,
        ),
        layer_groups=[
            KVLayerGroupSpec(
                start_layer=0,
                num_layers=1,
                dtype="torch.float16",
                hidden_dim=128,
            )
        ],
        format_id="mha_dense/packed/v1",
        schema_version=KV_FORMAT_SCHEMA_VERSION + 1,
    )
    with pytest.raises(ValueError):
        validate_kvcache_format(fmt)


def test_mla_builder_uses_mla_canonical_and_hidden_dim():
    fmt = build_dense_format_single_group(
        num_layers=4,
        dtype=torch.bfloat16,
        hidden_dim=64,
        block_size=128,
        use_mla=True,
        separation="packed",
    )
    assert fmt.layout == "MLA_LATENT_PACKED"
    assert fmt.family == "MLA_LATENT"
    assert fmt.canonical == "KV_MLA_FMT"
    assert fmt.layer_groups[0].hidden_dim == 64


def test_builder_uses_dtype_canonicalization():
    fmt = build_dense_format_single_group(
        num_layers=2,
        dtype=torch.bfloat16,
        hidden_dim=32,
        block_size=64,
        use_mla=False,
        separation="packed",
    )
    assert fmt.layer_groups[0].dtype == str(torch.bfloat16)


def test_vllm_connector_from_metadata_uses_kv_format():
    kv_dtype = torch.float16
    kv_shape = (4, 2, 256, 8, 16)  # hidden_dim = 128
    kv_format = build_dense_format_single_group(
        num_layers=4,
        dtype=kv_dtype,
        hidden_dim=128,
        block_size=256,
        use_mla=False,
        separation="packed",
    )
    metadata = LMCacheMetadata(
        model_name="test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        kv_format=kv_format,
        chunk_size=256,
    )

    connector = VLLMPagedMemGPUConnectorV2.from_metadata(
        metadata, use_gpu=False, device=None
    )
    assert connector.hidden_dim_size == 128
    assert connector.num_layers == 4


def test_vllm_connector_rejects_inconsistent_block_size():
    kv_dtype = torch.float16
    kv_shape = (4, 2, 128, 8, 16)  # chunk_size mismatch vs kv_format
    kv_format = build_dense_format_single_group(
        num_layers=4,
        dtype=kv_dtype,
        hidden_dim=128,
        block_size=256,
        use_mla=False,
        separation="packed",
    )
    metadata = LMCacheMetadata(
        model_name="test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        kv_format=kv_format,
        chunk_size=128,
    )

    with pytest.raises(ValueError):
        VLLMPagedMemGPUConnectorV2.from_metadata(metadata, use_gpu=False, device=None)


def test_get_kv_layout_from_metadata_without_format():
    kv_dtype = torch.float16
    kv_shape = (2, 2, 128, 4, 16)  # hidden_dim = 64
    metadata = LMCacheMetadata(
        model_name="test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        chunk_size=128,
    )

    layout, fmt = metadata.get_kv_layout("for test")
    assert fmt is None
    assert layout["block_size"] == 128
    assert layout["hidden_dim"] == 64
    assert layout["num_layers"] == 2
    assert layout["use_mla"] is False


def test_default_metadata_includes_kv_format():
    metadata = _get_default_metadata("test-model")
    assert metadata.kv_format is not None
    layout, fmt = metadata.get_kv_layout("for test")
    assert fmt is not None
    assert layout["block_size"] == metadata.kv_shape[2]
