# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch


# TODO: write unit tests protecting this class for the type assumptions it makes
class VllmState:
    """
    The goal of this class is to expose the internal state of the vllm engine
    and to provide a set of utilities for extracting vllm-agnostic attributes
    from vllm data structures (e.g. Request)

    From the caller perspective, this class is opaque but if state is accessed
    from this class, they should know we are hiding some vllm-relevant logic/states

    This class does NOT allow us AVOIDING importing data structures in other lmcache
    modules from vllm (e.g. Request, KVConnectorBase_V1) and only consolidates
    vllm-related reused state and logic

    The benefits are:
    - easy to debug breaking integration changes (all logic in one place)
    - version dependent logic all handled in one place
    - can reuse and pass around in any lmcache modules

    IMPORTANT NOTE:
    All purely transformational (non-stateful) logic should be static

    Keep everything in functions so that branching logic (e.g. with hasattr)
    can be extended in the future (including import logic)

    e.g.set_vllm_config() is in its own function in case the import locations
    change in the future
    """

    def __init__(self):
        self.set_vllm_config()
        self.set_version()
        self.set_platform()
        self.set_parallel_methods()

        self.set_none_values()

    # -- setters (add branching import logic if needed) --
    def set_vllm_config(self):
        # Third Party
        from vllm.config import VllmConfig

        self.vllm_config = VllmConfig
        self.model_config = self.vllm_config.model_config
        self.parallel_config = self.vllm_config.parallel_config
        self.cache_config = self.vllm_config.cache_config

    def set_version(self):
        # Third Party
        from vllm.version import __version__ as VLLM_VERSION

        self.vllm_version = VLLM_VERSION

    def set_platform(self):
        # Third Party
        from vllm.platforms import current_platform

        self.platform = current_platform

    def set_none_values(self):
        self._get_kv_cache_torch_dtype = None

    def set_parallel_methods(self):
        # Third Party
        from vllm.distributed.parallel_state import (
            get_pp_group,
            get_tensor_model_parallel_rank,
            get_tp_group,
        )

        self.get_pp_group = get_pp_group
        self.get_tensor_model_parallel_rank = get_tensor_model_parallel_rank
        self.get_tp_group = get_tp_group

    # -- one to one function pass throughs (only in this class for import protection) --
    def get_pp_group(self, *args, **kwargs):
        return self.get_pp_group(*args, **kwargs)

    def get_tensor_model_parallel_rank(self, *args, **kwargs):
        return self.get_tensor_model_parallel_rank(*args, **kwargs)

    def get_tp_group(self, *args, **kwargs):
        return self.get_tp_group(*args, **kwargs)

    # -- utilities --
    def get_kv_cache_torch_dtype(self) -> torch.dtype:
        if self._get_kv_cache_torch_dtype is None:
            # Try to import from old location before merged https://github.com/vllm-project/vllm/pull/26908
            try:
                # Third Party
                from vllm.utils.torch_utils import get_kv_cache_torch_dtype
            except ImportError:
                # Third Party
                from vllm.utils import get_kv_cache_torch_dtype

            self._get_kv_cache_torch_dtype = get_kv_cache_torch_dtype

        return self._get_kv_cache_torch_dtype(
            self.cache_config.cache_dtype, self.model_config.dtype
        )

    # -- related to speculative decoding --
    def speculative_config_method(self) -> str:
        """
        example return value: "deepseek_mtp"
        """
        return self.vllm_config.speculative_config.method

    def use_eagle(self) -> bool:
        return self.vllm_config.speculative_config.use_eagle()

    def num_draft_layers(self) -> int:
        draft_model_config = self.vllm_config.speculative_config.draft_model_config
        num_draft_layers = draft_model_config.get_num_layers(
            self.vllm_config.parallel_config
        )
        return num_draft_layers
