# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch

# First Party
import lmcache.lmcache_native as lmcache_native
import lmcache.v1.gpu_connector.neuron_nixl_staging as staging_mod


class FakeNixlAgent:
    def __init__(self):
        self.registered = []
        self.xfer_descs = []
        self.prepped = []
        self.transfers = []
        self.released = []
        self.deregistered = []

    def register_memory(self, reg_list, mem_type=None):
        token = ("reg", len(self.registered), mem_type)
        self.registered.append((reg_list, mem_type))
        return token

    def get_xfer_descs(self, descs, mem_type=None):
        self.xfer_descs.append((descs, mem_type))
        return descs

    def prep_xfer_dlist(self, name, descs, mem_type=None):
        handle = ("prep", len(self.prepped), name, mem_type)
        self.prepped.append((name, descs, mem_type, handle))
        return handle

    def make_prepped_xfer(
        self, direction, local_handle, local_ids, remote_handle, remote_ids
    ):
        return (
            direction,
            local_handle,
            tuple(local_ids),
            remote_handle,
            tuple(remote_ids),
        )

    def transfer(self, handle):
        self.transfers.append(handle)
        return "DONE"

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        self.released.append(handle)

    def deregister_memory(self, reg_descs):
        self.deregistered.append(reg_descs)


class FakeNixlWrapper:
    def __init__(self, name):
        self.name = name
        self.reg_descs = []
        self.registered = []
        self.xfer_descs = []
        self.remote_agents = []
        self.prepped = []
        self.xfers = []
        self.released_xfers = []
        self.released_dlists = []
        self.removed_agents = []
        self.deregistered = []

    def get_reg_descs(self, reg_list, mem_type=None):
        descs = ("reg_descs", len(self.reg_descs), mem_type, tuple(reg_list))
        self.reg_descs.append(descs)
        return descs

    def register_memory(self, descs, backends=None):
        self.registered.append((descs, tuple(backends or [])))
        return ("registered", len(self.registered))

    def get_xfer_descs(self, descs, mem_type=None):
        tagged = ("xfer_descs", len(self.xfer_descs), mem_type, tuple(descs))
        self.xfer_descs.append(tagged)
        return tagged

    def get_agent_metadata(self):
        return b"agent-bytes"

    def add_remote_agent(self, metadata):
        self.remote_agents.append(metadata)
        return "remote-agent"

    def remove_remote_agent(self, name):
        self.removed_agents.append(name)

    def prep_xfer_dlist(self, name, descs):
        handle = ("prep", len(self.prepped), name)
        self.prepped.append((name, descs, handle))
        return handle

    def make_prepped_xfer(
        self, direction, local_handle, local_ids, remote_handle, remote_ids, notif
    ):
        handle = (
            direction,
            local_handle,
            tuple(local_ids),
            remote_handle,
            tuple(remote_ids),
            notif,
        )
        self.xfers.append(handle)
        return handle

    def transfer(self, handle):
        return "DONE"

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        self.released_xfers.append(handle)

    def release_dlist_handle(self, handle):
        self.released_dlists.append(handle)

    def deregister_memory(self, descs):
        self.deregistered.append(descs)


def test_compact_slot_mapping_remaps_blocks_and_preserves_invalid_slots():
    stager = staging_mod.NeuronNixlBlockStager(backends=["LIBFABRIC"])
    slots = torch.tensor([-1, 4, 5, 12, 13], dtype=torch.long)

    selected_blocks, compact = stager._compact_slot_mapping(slots, block_size=4)

    assert selected_blocks == [1, 3]
    assert compact.tolist() == [-1, 0, 1, 4, 5]


def test_regions_for_hnd_two_major_split_k_and_v_planes():
    stager = staging_mod.NeuronNixlBlockStager(backends=["LIBFABRIC"])
    tensor = torch.zeros((2, 4, 3, 2, 5), dtype=torch.float32)

    regions = stager._regions_for_tensor(
        tensor,
        [1, 3],
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        block_size=2,
    )

    assert len(regions) == 4
    expected_size = 3 * 2 * 5 * tensor.element_size()
    assert all(region.size_bytes == expected_size for region in regions)
    assert regions[0].addr == staging_mod._slice_addr(tensor, (0, 1, 0, 0, 0))
    assert regions[1].addr == staging_mod._slice_addr(tensor, (1, 1, 0, 0, 0))
    assert regions[2].addr == staging_mod._slice_addr(tensor, (0, 3, 0, 0, 0))
    assert regions[3].addr == staging_mod._slice_addr(tensor, (1, 3, 0, 0, 0))


def test_transfer_into_key_value_uses_compact_slot_mapping_and_cpu_staging(monkeypatch):
    fake_agent = FakeNixlAgent()
    recorded = {}

    monkeypatch.setattr(
        staging_mod,
        "_load_nixl_api",
        lambda: (lambda _name, _cfg: fake_agent, lambda **kwargs: kwargs),
    )

    def fake_multi_layer_kv_transfer(
        key_value,
        key_value_ptrs,
        slot_mapping,
        paged_memory_device,
        page_buffer_size,
        direction,
        engine_kv_format,
        block_size=0,
        head_size=0,
        skip_prefix_n_tokens=0,
        block_stride_elems=0,
    ):
        recorded["key_value_shape"] = tuple(key_value.shape)
        recorded["stage_shapes"] = [tuple(t.shape) for t in key_value_ptrs]
        recorded["slot_mapping"] = slot_mapping.tolist()
        recorded["page_buffer_size"] = page_buffer_size
        recorded["paged_memory_device"] = str(paged_memory_device)
        recorded["direction"] = int(direction)
        recorded["fmt"] = int(engine_kv_format)
        recorded["block_size"] = block_size
        recorded["head_size"] = head_size

    monkeypatch.setattr(
        staging_mod.device_ops, "multi_layer_kv_transfer", fake_multi_layer_kv_transfer
    )

    stager = staging_mod.NeuronNixlBlockStager(backends=["LIBFABRIC"])
    key_value = torch.empty((2, 1, 4, 6), dtype=torch.float32)
    layer = torch.empty((2, 4, 3, 2, 2), dtype=torch.float32)
    slots = torch.tensor([2, 3, 6, 7], dtype=torch.long)

    stager.transfer_into_key_value(
        key_value=key_value,
        layer_tensors=[layer],
        slot_mapping=slots,
        engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        block_size=2,
        head_size=2,
    )

    assert recorded["key_value_shape"] == (2, 1, 4, 6)
    assert recorded["stage_shapes"] == [(2, 2, 3, 2, 2)]
    assert recorded["slot_mapping"] == [0, 1, 2, 3]
    assert recorded["page_buffer_size"] == 4
    assert recorded["paged_memory_device"] == "cpu"
    assert recorded["direction"] == int(lmcache_native.TransferDirection.D2H)
    assert recorded["fmt"] == int(lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS)
    assert recorded["block_size"] == 2
    assert recorded["head_size"] == 2

    assert [mem_type for _, mem_type in fake_agent.registered] == ["VRAM", "DRAM"]
    assert len(fake_agent.transfers) == 1
    assert len(fake_agent.released) == 1
    assert len(fake_agent.deregistered) == 2


def test_vllm_wrapper_copy_uses_two_agent_remote_source_handle(monkeypatch):
    created_wrappers = []

    monkeypatch.setattr(
        staging_mod,
        "_load_vllm_nixl_wrapper",
        lambda: (
            lambda name, _cfg: created_wrappers.append(FakeNixlWrapper(name))
            or created_wrappers[-1],
            lambda **kwargs: kwargs,
        ),
    )

    stager = staging_mod.NeuronNixlBlockStager(backends=["LIBFABRIC"])
    src = torch.empty((2, 4, 3, 2, 2), dtype=torch.float32)
    dst = torch.empty((2, 2, 3, 2, 2), dtype=torch.float32)

    stager._copy_layer_blocks_via_vllm_wrapper(
        src_layer=src,
        dst_layer=dst,
        selected_blocks=[1, 3],
        engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        block_size=2,
    )

    assert len(created_wrappers) == 2
    src_wrapper, dst_wrapper = created_wrappers
    assert src_wrapper.remote_agents == []
    assert dst_wrapper.remote_agents == [b"agent-bytes"]
    assert dst_wrapper.prepped[0][0] == "remote-agent"
    assert dst_wrapper.prepped[1][0] == "NIXL_INIT_AGENT"
    xfer = dst_wrapper.xfers[0]
    assert xfer[0] == "READ"
    assert xfer[2] == (0, 1, 2, 3)
    assert xfer[4] == (0, 1, 2, 3)
    assert xfer[5] == b"lmcache-neuron-staging"
    assert len(dst_wrapper.released_xfers) == 1
    assert len(dst_wrapper.released_dlists) == 2
    assert dst_wrapper.removed_agents == ["remote-agent"]
    assert len(src_wrapper.deregistered) == 1
    assert len(dst_wrapper.deregistered) == 1
