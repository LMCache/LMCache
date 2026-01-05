# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
from unittest.mock import MagicMock
import asyncio
import inspect
import random
import socket
import string
import threading
import uuid

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector import VLLMPagedMemGPUConnectorV2


def recover_engine_states(engine):
    engine.gpu_connector.kv_cache_pointers_on_gpu = {}


def recover_gpu_connector_states(gpu_connector):
    gpu_connector.kv_cache_pointers_on_gpu = {}


def dumb_metadata(fmt="vllm", kv_shape=(32, 2, 256, 8, 128)):
    return LMCacheEngineMetadata("test_model", 3, 123, fmt, torch.bfloat16, kv_shape)


def dumb_metadata_with_model_name(
    model_name: str, fmt="vllm", kv_shape=(32, 2, 256, 8, 128)
):
    return LMCacheEngineMetadata(model_name, 3, 123, fmt, torch.bfloat16, kv_shape)


def dumb_cache_engine_key(id: int = 0) -> CacheEngineKey:
    return CacheEngineKey("vllm", "test_model", 3, 123, id, torch.bfloat16)


def random_string(N):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=N))


def init_asyncio_loop():
    async_loop = asyncio.new_event_loop()
    async_thread = threading.Thread(target=async_loop.run_forever)
    async_thread.start()
    return async_loop, async_thread


def close_asyncio_loop(async_loop, async_thread):
    if async_loop.is_running():
        async_loop.call_soon_threadsafe(async_loop.stop)
    if async_thread.is_alive():
        async_thread.join()


def get_available_port(host: str = "127.0.0.1") -> int:
    """
    Get an available port dynamically by binding to port 0.

    Args:
        host: The host address to bind to. Default is "127.0.0.1".

    Returns:
        An available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def get_available_ports(count: int, host: str = "127.0.0.1") -> list[int]:
    """
    Get multiple available ports dynamically.

    Args:
        count: Number of ports to get.
        host: The host address to bind to. Default is "127.0.0.1".

    Returns:
        A list of available port numbers.
    """
    ports = []
    for _ in range(count):
        ports.append(get_available_port(host))
    return ports


def generate_kv_cache(num_tokens, fmt, device):
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = (
        [num_tokens, num_heads, head_size]
        if fmt == "vllm"
        else [num_heads, num_tokens, head_size]
    )
    dtype = torch.bfloat16 if fmt == "vllm" else torch.float16

    for i in range(num_layers):
        k = torch.rand(shape, dtype=dtype, device=device)
        v = torch.rand(shape, dtype=dtype, device=device)
        ret.append((k, v))

    return tuple(ret)


def generate_kv_cache_paged_list_tensors(
    num_blocks, device, block_size=16, dtype=torch.bfloat16, use_mla=False
):
    """
    Instead of Tuple[Tuple[Tensor, Tensor]], return List[Tensor]
    where KV are in the same tensor
    """
    ret = []
    num_layers = 32
    num_heads = 1 if use_mla else 8
    head_size = 128
    shape = (
        [num_blocks, block_size, head_size]
        if use_mla
        else [2, num_blocks, block_size, num_heads, head_size]
    )

    for i in range(num_layers):
        kv = torch.rand(shape, dtype=dtype, device=device)
        ret.append(kv)

    return ret


def generate_sglang_kv_cache_paged_list_tensors(
    num_layers,
    num_blocks,
    block_size,
    num_heads,
    head_size,
    use_mla=False,
    device="cuda",
    dtype=torch.bfloat16,
):
    """
    Instead of Tuple[Tuple[Tensor, Tensor]], return List[Tensor]
    where KV are in the same tensor
    """
    shape = (
        [num_blocks * block_size, 1, head_size]
        if use_mla
        else [num_blocks * block_size, num_heads, head_size]
    )
    if use_mla:
        kv_cache = [
            torch.rand(shape, dtype=dtype, device=device) for i in range(num_layers)
        ]
    else:
        k_cache = [
            torch.rand(shape, dtype=dtype, device=device) for i in range(num_layers)
        ]
        v_cache = [
            torch.rand(shape, dtype=dtype, device=device) for i in range(num_layers)
        ]
        kv_cache = k_cache + v_cache
    return kv_cache


def generate_mla_kv_cache_paged_list_tensors(
    num_blocks,
    device,
    block_size=64,
    dtype=torch.bfloat16,
    num_layers=32,
    head_size=576,
):
    """
    return KV cache of MLA
    """
    ret = []
    shape = [num_blocks, block_size, head_size]

    for i in range(num_layers):
        kv = torch.rand(shape, dtype=dtype, device=device)
        ret.append(kv)

    return ret


def generate_kv_cache_paged(num_blocks, device, block_size=16, dtype=torch.bfloat16):
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = [num_blocks, block_size, num_heads, head_size]

    for i in range(num_layers):
        k = torch.rand(shape, dtype=dtype, device=device)
        v = torch.rand(shape, dtype=dtype, device=device)
        ret.append((k, v))

    return tuple(ret)


def generate_tokens(num_tokens, device, fixed=False):
    if fixed:
        return torch.tensor([-1] * num_tokens).to(device)
    else:
        # random tokens
        return torch.randint(0, 10000, size=[num_tokens]).to(device)


def concatenate_kv_caches(kv_chunks, fmt):
    dim = 1 if fmt == "huggingface" else 0
    ret = []
    for kv_layer in zip(*kv_chunks, strict=False):
        klist, vlist = zip(*kv_layer, strict=False)
        klayer = torch.cat(klist, dim=dim)
        vlayer = torch.cat(vlist, dim=dim)
        ret.append((klayer, vlayer))
    return tuple(ret)


def check_mem_obj_equal(left, right, use_mla: bool = False):
    """
    check whether two memory objects are the same
    """
    for left_mem_obj, right_mem_obj in zip(left, right, strict=False):
        left_tensor_size = left_mem_obj.tensor.size()
        right_tensor_size = right_mem_obj.tensor.size()
        if use_mla:
            assert left_tensor_size[0] == 1
            assert right_tensor_size[0] == 1

            left_kv, right_kv = left_mem_obj.tensor[0], right_mem_obj.tensor[0]
            right_kv = right_kv.to(left_kv.device)

            assert len(left_kv.shape) == 3
            assert len(right_kv.shape) == 3

            assert (left_kv[:, :, :] == right_kv[:, :, :]).all()
        else:
            assert left_tensor_size[0] == 2
            assert right_tensor_size[0] == 2

            left_kv, right_kv = left_mem_obj.tensor, right_mem_obj.tensor
            left_k, left_v = left_kv[0], left_kv[1]
            right_k, right_v = right_kv[0], right_kv[1]
            right_k = right_k.to(left_k.device)
            right_v = right_v.to(left_v.device)

            assert len(left_k.shape) == 3
            assert len(left_v.shape) == 3
            assert len(right_k.shape) == 3
            assert len(right_v.shape) == 3

            assert (left_k[:, :, :] == right_k[:, :, :]).all()
            assert (left_v[:, :, :] == right_v[:, :, :]).all()


def check_paged_kv_cache_equal(left, right, slot_mapping, num_heads=8, head_size=128):
    """
    check whether two paged kv caches are the same at slot_mapping
    """
    token_dim = 0
    num_tokens = slot_mapping.shape[0]
    for left_kv, right_kv in zip(left, right, strict=False):
        left_k = left_kv[0].reshape(-1, num_heads, head_size)
        left_v = left_kv[1].reshape(-1, num_heads, head_size)
        right_k = right_kv[0].reshape(-1, num_heads, head_size)
        right_v = right_kv[1].reshape(-1, num_heads, head_size)

        assert len(left_k.shape) == 3
        assert len(left_v.shape) == 3
        assert len(right_k.shape) == 3
        assert len(right_v.shape) == 3

        assert left_k.shape[token_dim] >= num_tokens
        assert left_v.shape[token_dim] >= num_tokens
        assert right_k.shape[token_dim] >= num_tokens
        assert right_v.shape[token_dim] >= num_tokens

        assert (left_k[slot_mapping, :, :] == right_k[slot_mapping, :, :]).all()
        assert (left_v[slot_mapping, :, :] == right_v[slot_mapping, :, :]).all()


def check_sglang_paged_kv_cache_equal(
    left, right, slot_mapping, num_heads=8, head_size=128
):
    """
    check whether two paged kv caches are the same at slot_mapping
    """
    token_dim = 0
    num_tokens = slot_mapping.shape[0]
    for left_kv, right_kv in zip(left, right, strict=False):
        _left_kv = left_kv.reshape(-1, num_heads, head_size)
        _right_kv = right_kv.reshape(-1, num_heads, head_size)

        assert len(_left_kv.shape) == 3
        assert len(_right_kv.shape) == 3

        assert _left_kv.shape[token_dim] >= num_tokens
        assert _right_kv.shape[token_dim] >= num_tokens

        assert (_left_kv[slot_mapping, :, :] == _right_kv[slot_mapping, :, :]).all()


def check_paged_kv_cache_equal_with_mla(left, right, slot_mapping, head_size=128):
    """
    check whether two paged kv caches are the same at slot_mapping when use mla
    """
    token_dim = 0
    num_tokens = slot_mapping.shape[0]
    for left_kv, right_kv in zip(left, right, strict=False):
        new_left_kv = left_kv.reshape(-1, head_size)
        new_right_kv = right_kv.reshape(-1, head_size)

        assert len(new_left_kv.shape) == 2
        assert len(new_right_kv.shape) == 2

        assert new_left_kv.shape[token_dim] >= num_tokens
        assert new_right_kv.shape[token_dim] >= num_tokens

        assert (new_left_kv[slot_mapping, :] == new_right_kv[slot_mapping, :]).all()


def check_kv_cache_device(kvs, device):
    for kv in kvs:
        k, v = kv
        assert k.device == torch.device(device)
        assert v.device == torch.device(device)


def create_gpu_connector(hidden_dim, num_layers):
    return VLLMPagedMemGPUConnectorV2(hidden_dim, num_layers)


def get_all_methods_from_base(base_class):
    """
    Get all public methods defined in the base class (excluding inherited from object).
    """
    methods = set()
    for name in dir(base_class):
        # Skip private and special methods
        if name.startswith("_"):
            continue
        attr = getattr(base_class, name)
        if callable(attr):
            methods.add(name)
    return methods


def get_methods_implemented_in_class(cls, base_class=None):
    """
    Get methods that are actually implemented in the class itself.
    Args:
        cls: The class to inspect
        base_class: Optional base class to stop at. If None, stops at
            abstract base classes.
    """
    implemented = set()

    # Check the class's own __dict__ for methods
    for name in cls.__dict__:
        if name.startswith("_"):
            continue
        attr = cls.__dict__[name]
        # Check if it's callable (function, method, etc.)
        if callable(attr):
            implemented.add(name)

    # Also check using getattr to catch any dynamically added methods
    for name in dir(cls):
        if name.startswith("_"):
            continue
        if name in implemented:
            continue  # Already found
        try:
            attr = getattr(cls, name)
            if callable(attr):
                # Verify it's not inherited from base class
                # by checking if it exists in the class's MRO
                for base in cls.__mro__:
                    # Stop when we hit the specified base class
                    if base_class is not None and base is base_class:
                        break
                    # Or stop when we hit an abstract base class
                    if base_class is None and inspect.isabstract(base):
                        break
                    if name in base.__dict__:
                        implemented.add(name)
                        break
        except AttributeError:
            pass

    return implemented


def get_abstract_methods(cls):
    """
    Get all abstract methods from a class.
    """
    abstract_methods = set()
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if getattr(method, "__isabstractmethod__", False):
            abstract_methods.add(name)
    return abstract_methods


def check_method_signatures(base_class, impl_class):
    """
    Check if method signatures in implementation class match the base class.
    Returns a list of mismatches.
    """
    base_methods = get_all_methods_from_base(base_class)
    signature_mismatches = []

    for method_name in base_methods:
        base_method = getattr(base_class, method_name)
        impl_method = getattr(impl_class, method_name, None)

        if impl_method is None:
            continue

        try:
            base_sig = inspect.signature(base_method)
            impl_sig = inspect.signature(impl_method)

            # Compare parameter names (excluding 'self')
            base_params = [p for p in base_sig.parameters.keys() if p != "self"]
            impl_params = [p for p in impl_sig.parameters.keys() if p != "self"]

            if base_params != impl_params:
                signature_mismatches.append(
                    {
                        "method": method_name,
                        "base_params": base_params,
                        "impl_params": impl_params,
                    }
                )
        except (ValueError, TypeError):
            # Some methods might not have inspectable signatures
            pass

    return signature_mismatches


class DummyLMCacheAsyncLookupServer:
    def __init__(self):
        pass

    def send_response_to_scheduler(
        self,
        lookup_id: str,
        retrieved_length: int,
    ) -> None:
        pass


class MockAdapter:
    """
    Mock adapter to provide config and lmcache_engine to InternalAPIServer.
    """

    def __init__(self, engine, config):
        self.lmcache_engine = engine
        self.config = config


def create_test_metadata(
    worker_id: int = 0,
    world_size: int = 1,
    kv_shape: tuple = (4, 2, 256, 8, 128),
) -> LMCacheEngineMetadata:
    """Create test metadata for LMCacheEngine."""
    return LMCacheEngineMetadata(
        model_name="test_model",
        world_size=world_size,
        worker_id=worker_id,
        fmt="vllm",
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
    )


def create_test_config(
    chunk_size: int = 256,
    local_cpu: bool = True,
    max_local_cpu_size: float = 1.0,
    rpc_port: int = 0,
    extra_config: Optional[dict] = None,
    instance_id: Optional[str] = None,
) -> LMCacheEngineConfig:
    """Create test configuration for LMCacheEngine."""
    if instance_id is None:
        instance_id = f"test_instance_{uuid.uuid4().hex[:8]}"
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=local_cpu,
        max_local_cpu_size=max_local_cpu_size,
        lmcache_instance_id=instance_id,
    )
    config.extra_config = extra_config.copy() if extra_config else {}
    config.extra_config["lmcache_rpc_port"] = rpc_port
    return config


def create_mock_vllm_config(
    rank: int = 0, world_size: int = 1, rpc_port: int = 0
) -> MagicMock:
    """Create a mock VllmConfig for testing."""
    vllm_config = MagicMock()

    # Mock model_config
    vllm_config.model_config = MagicMock()
    vllm_config.model_config.model = "test_model"
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.model_config.get_num_layers = MagicMock(return_value=4)
    vllm_config.model_config.get_num_kv_heads = MagicMock(return_value=8)
    vllm_config.model_config.get_head_size = MagicMock(return_value=128)
    vllm_config.model_config.hf_config = MagicMock()
    vllm_config.model_config.hf_config.model_type = "llama"

    # Mock parallel_config
    vllm_config.parallel_config = MagicMock()
    vllm_config.parallel_config.rank = rank
    vllm_config.parallel_config.world_size = world_size
    vllm_config.parallel_config.tensor_parallel_size = world_size
    vllm_config.parallel_config.pipeline_parallel_size = 1

    # Mock cache_config
    vllm_config.cache_config = MagicMock()
    vllm_config.cache_config.cache_dtype = torch.bfloat16

    # Mock kv_transfer_config with engine_id
    vllm_config.kv_transfer_config = MagicMock()
    vllm_config.kv_transfer_config.engine_id = "test_engine"
    vllm_config.kv_transfer_config.get_from_extra_config = MagicMock(
        side_effect=lambda key, default: (
            rpc_port if key == "lmcache_rpc_port" else default
        )
    )

    return vllm_config
