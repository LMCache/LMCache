# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import os
import shutil
import subprocess
import sys

# Third Party
import pytest
import torch

RESULTS_DIR = Path("test_results")

_is_child = "LMC_TEST_MODE" in os.environ
_cuda_available = torch.cuda.is_available()


# ==========================================
# 1. Core Logic
# ==========================================


def get_test_context():
    mode = os.getenv("LMC_TEST_MODE", "NON_CUDA")
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")

    cuda_status = "cuda_ready" if cuda_visible != "" else "no_cuda"
    backend = "cuda_ops" if mode == "CUDA_OPS" else "non_cuda"

    if backend == "cuda_ops":
        print(f">>> Importing lmcache.c_ops as ops (Mode: {mode})")
        # First Party
        import lmcache.c_ops as ops
    else:
        print(f">>> Importing lmcache.non_cuda_equivalents as ops (Mode: {mode})")
        # First Party
        import lmcache.non_cuda_equivalents as ops

    return ops, f"{backend}_{cuda_status}"


def save_result(func_name, data):
    _, scene = get_test_context()
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(data, RESULTS_DIR / f"{func_name}@{scene}.pt")


# ==========================================
# 2. Scenario functions
# ==========================================


def scenario_get_gpu_pci_bus_id():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    res = ops.get_gpu_pci_bus_id(0)

    if is_cuda_backend and torch.cuda.is_available():
        assert res is not None, "get_gpu_pci_bus_id returned None"
        assert isinstance(res, str) and len(res) > 0

    # Save 1 = PASS (call succeeded without crash)
    save_result(
        "get_gpu_pci_bus_id",
        torch.tensor([1], dtype=torch.int32),
    )


def scenario_calculate_cdf():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    num_bins_list = [1, 2, 5, 11, 15, 31, 32, 63]

    for num_bins in num_bins_list:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        input_tensor = torch.randint(0, num_bins, (1, 1000, 1), dtype=torch.int8)

        if is_cuda_backend:
            target_dev = f"cuda:{torch.cuda.current_device()}"
            input_tensor = input_tensor.to(target_dev)

        raw_output = ops.calculate_cdf(input_tensor, num_bins)
        out_cpu = raw_output.flatten().cpu()

        if is_cuda_backend:
            out_int32 = out_cpu.to(torch.int32)
            out_uint16 = torch.where(out_int32 < 0, out_int32 + 65536, out_int32)
            final_result = out_uint16.float() / 65536.0
        else:
            final_result = out_cpu.float()

        save_result(f"calculate_cdf_bins{num_bins}", final_result)


def scenario_rotary_embedding_k_fused():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Setup Dimensions
    num_tokens = 128
    num_kv_heads = 32
    head_size = 128
    max_position = 2048
    rotary_dim = head_size

    # 2. Generate Inputs
    old_positions = torch.randint(0, 1000, (num_tokens,), dtype=torch.long)
    new_positions = old_positions + 1

    cos_sin_cache = torch.randn(max_position, rotary_dim, dtype=torch.float32)

    # Test both is_neox=True (NeoX-style, contiguous halves) and
    # is_neox=False (GPT-J-style, interleaved)
    for is_neox in [True, False]:
        # Reset seed for consistent key tensor across both tests
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.float32)

        if is_cuda_backend:
            target_dev = f"cuda:{torch.cuda.current_device()}"
            old_positions_dev = old_positions.to(target_dev)
            new_positions_dev = new_positions.to(target_dev)
            key = key.to(target_dev)
            cos_sin_cache_dev = cos_sin_cache.to(target_dev)
        else:
            old_positions_dev = old_positions
            new_positions_dev = new_positions
            cos_sin_cache_dev = cos_sin_cache

        # 3. Execute (in-place update on key)
        ops.rotary_embedding_k_fused(
            old_positions_dev,
            new_positions_dev,
            key,
            head_size,
            cos_sin_cache_dev,
            is_neox,
        )

        # 4. Save with is_neox suffix to distinguish the two test cases
        neox_suffix = "neox" if is_neox else "gptj"
        save_result(f"rotary_embedding_k_fused_{neox_suffix}", key.cpu())


def scenario_lmcache_memcpy_async():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Setup dimensions and mock data (4KB)
    nbytes = 1024 * 4
    src_host = torch.randint(1, 255, (nbytes,), dtype=torch.uint8)
    gpu_buffer = torch.zeros(nbytes, dtype=torch.uint8)

    if torch.cuda.is_available():
        dst_host = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
    else:
        dst_host = torch.zeros(nbytes, dtype=torch.uint8)

    # 2. Assign directions and device locations
    if is_cuda_backend:
        gpu_buffer = gpu_buffer.to(f"cuda:{torch.cuda.current_device()}")

    h2d_dir = ops.TransferDirection.H2D
    d2h_dir = ops.TransferDirection.D2H

    # --- PART A: H2D (Host to Device) ---
    ops.lmcache_memcpy_async(
        gpu_buffer.data_ptr(),
        src_host.data_ptr(),
        nbytes,
        h2d_dir,
        0,
        16,
    )

    if is_cuda_backend:
        torch.cuda.synchronize()

    # --- PART B: D2H (Device to Host) ---
    ops.lmcache_memcpy_async(
        dst_host.data_ptr(),
        gpu_buffer.data_ptr(),
        nbytes,
        d2h_dir,
        0,
        16,
    )

    if is_cuda_backend:
        torch.cuda.synchronize()

    # 3. Internal sanity check
    final_result = dst_host.cpu()
    assert torch.equal(final_result, src_host), (
        f"Data corrupted during H2D→D2H loop in {scene_info}, "
        f"max diff = {(final_result.float() - src_host.float()).abs().max().item()}"
    )

    # 4. Save
    save_result("lmcache_memcpy_async", final_result)


def scenario_load_and_reshape_flash():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Standard Params
    src_device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"
    dst_device = "cpu"

    num_blocks = 100
    block_size = 16
    num_heads = 8
    head_size = 128
    num_layers = 32
    num_tokens = 256
    chunk_size = 256
    dtype = torch.bfloat16

    # 2. Setup Data (Deterministic Pattern)
    total_elements = num_blocks * block_size * num_heads * head_size

    kv_cache_cpu = []
    for i in range(num_layers):
        base_tensor = torch.linspace(i, i + 1, total_elements, dtype=torch.float32)
        base_tensor = base_tensor.reshape(
            num_blocks, block_size, num_heads, head_size
        ).to(dtype)
        k = base_tensor
        v = base_tensor + 0.5
        kv_cache_cpu.append([k, v])

    kv_cache = [
        [layer[0].to(src_device), layer[1].to(src_device)] for layer in kv_cache_cpu
    ]

    # Slot mapping: deterministic strided selection
    step = (num_blocks * block_size) // num_tokens
    slot_indices = list(range(0, num_blocks * block_size, step))[:num_tokens]
    slot_mapping = torch.tensor(slot_indices, device=src_device, dtype=torch.int64)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    # 3. Extract (to CPU pinned)
    extracted_chunks = []
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        mem_obj_shape = (2, num_layers, len(slot_mapping_temp), num_heads * head_size)
        mem_obj_tensor = torch.zeros(mem_obj_shape, dtype=dtype, device=dst_device)

        if is_cuda_backend:
            mem_obj_tensor = mem_obj_tensor.pin_memory()

        for layer_id in range(num_layers):
            ops.load_and_reshape_flash(
                mem_obj_tensor,
                kv_cache[layer_id][0],
                kv_cache[layer_id][1],
                slot_mapping_temp,
                layer_id,
            )
        extracted_chunks.append(mem_obj_tensor)

    if is_cuda_backend:
        torch.cuda.synchronize()

    # 4. Verify: compare extracted data against original kv_cache
    #    mem_obj_tensor layout:
    #       [2, num_layers, num_tokens_in_chunk, num_heads * head_size]
    #    dim 0: K=0, V=1
    #    Original kv_cache layout: [num_blocks, block_size, num_heads, head_size]
    #    slot_mapping tells us which (block, offset) each token comes from
    for chunk_id, slot_mapping_temp in enumerate(slot_mapping_chunked):
        slots = slot_mapping_temp.cpu()
        extracted = extracted_chunks[chunk_id].cpu()

        for layer_id in range(num_layers):
            orig_k = kv_cache_cpu[layer_id][
                0
            ]  # [num_blocks, block_size, num_heads, head_size]
            orig_v = kv_cache_cpu[layer_id][1]

            for tok_idx, slot in enumerate(slots):
                block_idx = slot.item() // block_size
                offset = slot.item() % block_size

                # Expected: flattened [num_heads * head_size]
                expected_k = orig_k[block_idx, offset].reshape(-1)
                expected_v = orig_v[block_idx, offset].reshape(-1)

                # Extracted
                got_k = extracted[0, layer_id, tok_idx]
                got_v = extracted[1, layer_id, tok_idx]

                k_diff = (got_k.float() - expected_k.float()).abs().max().item()
                assert torch.equal(got_k, expected_k), (
                    f"K mismatch layer={layer_id}, slot={slot.item()}, "
                    f"max diff={k_diff}"
                )

                v_diff = (got_v.float() - expected_v.float()).abs().max().item()
                assert torch.equal(got_v, expected_v), (
                    f"V mismatch layer={layer_id}, slot={slot.item()}, "
                    f"max diff={v_diff}"
                )

    # 5. Save extracted data for cross-scenario comparison
    save_result("load_and_reshape_flash", extracted_chunks[0].cpu())


def scenario_reshape_and_cache_back_flash():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Environment Setup
    src_device = "cpu"
    dst_device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    num_blocks = 100
    block_size = 16
    num_heads = 8
    head_size = 128
    num_layers = 32
    num_tokens = 256
    chunk_size = 256
    dtype = torch.bfloat16

    # 2. Prepare Source Data (CPU Buffer)
    # Shape: [2, num_layers, num_tokens, num_heads * head_size]
    mem_obj_shape = (2, num_layers, num_tokens, num_heads * head_size)
    src_buffer = torch.zeros(mem_obj_shape, dtype=dtype, device=src_device)

    # Data Pattern: Odd numbers (1.0, 3.0, 5.0, ...)
    for i in range(num_tokens):
        val = 1.0 + (i * 2.0)
        src_buffer[0, :, i, :] = val  # Key
        src_buffer[1, :, i, :] = val + 0.5  # Value

    if is_cuda_backend:
        src_buffer = src_buffer.pin_memory()

    # 3. Prepare Destination (Empty Cache)
    kv_cache = [
        [
            torch.zeros(
                num_blocks,
                block_size,
                num_heads,
                head_size,
                device=dst_device,
                dtype=dtype,
            ),
            torch.zeros(
                num_blocks,
                block_size,
                num_heads,
                head_size,
                device=dst_device,
                dtype=dtype,
            ),
        ]
        for _ in range(num_layers)
    ]

    # 4. Slot Mapping (Continuous: Token 0 → Slot 0, Token 1 → Slot 1, ...)
    slot_indices = list(range(num_tokens))
    slot_mapping = torch.tensor(slot_indices, device=dst_device, dtype=torch.int64)
    slot_mapping_chunked = torch.split(slot_mapping, chunk_size)

    # 5. Execute Operator (Load Back)
    current_token_offset = 0
    for chunk_id, slot_chunk in enumerate(slot_mapping_chunked):
        chunk_len = len(slot_chunk)

        buffer_chunk = src_buffer[
            :, :, current_token_offset : current_token_offset + chunk_len, :
        ]
        if not buffer_chunk.is_contiguous():
            buffer_chunk = buffer_chunk.contiguous()

        for layer_id in range(num_layers):
            ops.reshape_and_cache_back_flash(
                buffer_chunk,
                kv_cache[layer_id][0],
                kv_cache[layer_id][1],
                slot_chunk,
                layer_id,
            )
        current_token_offset += chunk_len

    if is_cuda_backend:
        torch.cuda.synchronize()

    # 6. Verify: check written values against source pattern
    for layer_id in range(num_layers):
        k_cache = kv_cache[layer_id][
            0
        ].cpu()  # [num_blocks, block_size, num_heads, head_size]
        v_cache = kv_cache[layer_id][1].cpu()

        for tok_idx, slot in enumerate(slot_indices):
            block_idx = slot // block_size
            offset = slot % block_size

            expected_k_val = 1.0 + (tok_idx * 2.0)
            expected_v_val = expected_k_val + 0.5

            got_k = k_cache[block_idx, offset]
            got_v = v_cache[block_idx, offset]

            expected_k = torch.full_like(got_k, expected_k_val)
            expected_v = torch.full_like(got_v, expected_v_val)

            assert torch.allclose(got_k.float(), expected_k.float(), atol=0.1), (
                f"K mismatch at layer={layer_id}, slot={slot}, "
                f"expected={expected_k_val}, got={got_k[0, 0].item()}"
            )
            assert torch.allclose(got_v.float(), expected_v.float(), atol=0.1), (
                f"V mismatch at layer={layer_id}, slot={slot}, "
                f"expected={expected_v_val}, got={got_v[0, 0].item()}"
            )

    # 7. Save first block of layer 0 key cache for cross-scenario comparison
    save_result("reshape_and_cache_back_flash", kv_cache[0][0][0].cpu())


def scenario_encode_fast_new():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Hyperparameters
    nlayers = 2
    nchannels = 4
    ntokens = 128
    alphabet_size = 16
    max_buf_len = ntokens * 2

    src_device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    # 2. Construct Data
    # A. CDF: uniform distribution, strictly increasing
    step = 100 // alphabet_size
    base_cdf = torch.arange(0, 100, step, dtype=torch.int32)
    base_cdf = base_cdf[:alphabet_size]

    cdf_cpu = (
        base_cdf.unsqueeze(0).unsqueeze(0).expand(nlayers, nchannels, -1).contiguous()
    )
    cdf = cdf_cpu.to(dtype=torch.int16, device=src_device)

    # B. Input symbols: cycling 0..14
    total_syms = nlayers * ntokens * nchannels
    input_cpu = torch.arange(total_syms, dtype=torch.float32)
    input_cpu = (input_cpu % (alphabet_size - 1)).to(torch.int8)
    input_cpu = input_cpu.reshape(nlayers, ntokens, nchannels)
    input_sym = input_cpu.to(device=src_device)

    # 3. Prepare Outputs
    output_buffer = torch.zeros(
        (nlayers, nchannels, max_buf_len),
        dtype=torch.uint8,
        device=src_device,
    )
    output_lengths = torch.zeros(
        (nlayers, nchannels),
        dtype=torch.int32,
        device=src_device,
    )

    # 4. Execute
    ops.encode_fast_new(
        cdf,
        input_sym,
        output_buffer,
        output_lengths,
    )

    if is_cuda_backend:
        torch.cuda.synchronize()

    # 5. Verify
    lengths_cpu = output_lengths.cpu()

    assert (lengths_cpu > 0).all(), "Encoding produced zero-length output!"
    assert (lengths_cpu <= max_buf_len).all(), "Buffer overflow detected!"

    # 6. Save: first 20 bytes of layer 0, channel 0
    valid_len = lengths_cpu[0, 0].item()
    res = output_buffer[0, 0, : min(valid_len, 20)].cpu()
    save_result("encode_fast_new", res)


def scenario_decode_fast_new():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Config
    nlayers = 2
    nchannels = 4
    ntokens = 128
    alphabet_size = 16
    max_buf_len = ntokens * 2

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    # 2. Data Generation
    cdf = torch.randint(
        1,
        100,
        (nlayers, nchannels, alphabet_size),
        dtype=torch.int32,
    )
    cdf = torch.cumsum(cdf, dim=-1).to(device).to(torch.int16)

    input_sym = torch.randint(
        0,
        alphabet_size - 2,
        (nlayers, ntokens, nchannels),
        dtype=torch.int8,
    ).to(device)

    # 3. Encode first (need encoded data to test decode)
    encoded_buffer = torch.zeros(
        (nlayers, nchannels, max_buf_len),
        dtype=torch.uint8,
        device=device,
    )
    encoded_lengths = torch.zeros(
        (nlayers, nchannels),
        dtype=torch.int32,
        device=device,
    )

    ops.encode_fast_new(
        cdf,
        input_sym,
        encoded_buffer,
        encoded_lengths,
    )
    if is_cuda_backend:
        torch.cuda.synchronize()

    # 4. Decode
    decoded_sym = torch.zeros_like(input_sym, dtype=torch.uint8)

    ops.decode_fast_new(
        cdf,
        encoded_buffer,
        encoded_lengths,
        decoded_sym,
    )
    if is_cuda_backend:
        torch.cuda.synchronize()

    # 5. Verify: decoded must match original
    input_uint8 = input_sym.to(torch.uint8)
    mismatch = (input_uint8 != decoded_sym).sum().item()
    if mismatch > 0:
        mask = input_uint8 != decoded_sym
        ly, t, c = mask.nonzero()[0].tolist()
        pytest.fail(
            f"Decode mismatch: {mismatch} errors. "
            f"First diff at L{ly}T{t}C{c}: "
            f"orig={input_uint8[ly, t, c].item()} "
            f"decoded={decoded_sym[ly, t, c].item()}"
        )

    # 6. Save decoded slice for cross-scenario comparison
    save_result("decode_fast_new", decoded_sym[0, :20, 0].cpu())


def scenario_decode_fast_prefsum():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 1. Configuration
    nlayers = 2
    nchannels = 4
    ntokens = 128
    alphabet_size = 16
    max_buf_len = ntokens * 2

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    # 2. Data Generation (Normalized CDF)
    cdf = torch.randint(
        1,
        100,
        (nlayers, nchannels, alphabet_size),
        dtype=torch.int32,
    )
    cdf = torch.cumsum(cdf, dim=-1).float()
    cdf = (cdf / cdf[..., -1:] * 65536).to(torch.int32)
    cdf[..., -1] = 65536
    cdf = cdf.to(device).to(torch.int16).contiguous()

    input_sym = torch.randint(
        0,
        alphabet_size - 2,
        (nlayers, ntokens, nchannels),
        dtype=torch.int8,
    ).to(device)

    # 3. Encode to get variable lengths
    tmp_buf = torch.zeros(
        (nlayers, nchannels, max_buf_len),
        dtype=torch.uint8,
        device=device,
    )
    tmp_len = torch.zeros(
        (nlayers, nchannels),
        dtype=torch.int32,
        device=device,
    )
    ops.encode_fast_new(cdf, input_sym, tmp_buf, tmp_len)
    if is_cuda_backend:
        torch.cuda.synchronize()

    # 4. Pack into 1D dense bytestream
    lens_flat = tmp_len.cpu().flatten().tolist()
    bufs_flat = tmp_buf.cpu().reshape(-1, max_buf_len).numpy()
    all_bytes = []
    for i, length in enumerate(lens_flat):
        all_bytes.extend(bufs_flat[i, :length].tolist())

    bytestream_1d = torch.tensor(
        all_bytes,
        dtype=torch.uint8,
        device=device,
    ).contiguous()

    # 5. Offsets (end-position via cumsum)
    lengths_prefsum = (
        tmp_len.flatten().cumsum(0).reshape(tmp_len.shape).to(torch.int64).to(device)
    ).contiguous()

    # 6. Decode
    decoded_sym = (
        torch.zeros_like(
            input_sym,
            dtype=torch.uint8,
        )
        .to(device)
        .contiguous()
    )

    ops.decode_fast_prefsum(
        cdf,
        bytestream_1d,
        lengths_prefsum,
        decoded_sym,
    )
    if is_cuda_backend:
        torch.cuda.synchronize()

    # 7. Verify roundtrip
    input_ref = input_sym.to(torch.uint8)
    mismatch = (input_ref != decoded_sym).sum().item()
    if mismatch > 0:
        mask = input_ref != decoded_sym
        ly, t, c = mask.nonzero()[0].tolist()
        pytest.fail(
            f"Prefsum mismatch: {mismatch} errors. "
            f"First diff at L{ly}T{t}C{c}: "
            f"orig={input_ref[ly, t, c].item()} "
            f"decoded={decoded_sym[ly, t, c].item()}"
        )

    # 8. Save
    save_result(
        "decode_fast_prefsum",
        decoded_sym[0, :20, 0].cpu(),
    )


def scenario_single_layer_kv_transfer():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    num_tokens = 64
    num_blocks = 256
    block_size = 16
    num_heads = 12
    head_size = 64
    hidden_size = num_heads * head_size

    slot_mapping = torch.arange(
        0,
        num_tokens * 2,
        2,
        device=device,
    ).to(torch.int64)

    # (gpu_kv_format, is_mla, token_major, direction)
    # direction: False = LMC→vLLM (H2D), True = vLLM→LMC (D2H)
    test_cases = [
        # flash attn: [2, NB, BS, NH, HS] — two_major
        (ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, True, False),
        (ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, False, False),
        (ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, True, True),
        # flash infer: [NB, 2, BS, NH, HS]
        (ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS, False, True, False),
        (ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS, False, False, False),
        (ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS, False, True, True),
        # vLLM MLA: [NB, BS, HS]
        (ops.GPUKVFormat.NL_X_NB_BS_HS, True, True, False),
        (ops.GPUKVFormat.NL_X_NB_BS_HS, True, True, True),
    ]

    for gpu_kv_format, is_mla, token_major, direction in test_cases:
        dir_tag = "v2l" if direction else "l2v"
        is_two_major = gpu_kv_format == ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        case_desc = (
            f"fmt={gpu_kv_format}, MLA={is_mla}, TM={token_major}, Dir={dir_tag}"
        )

        # ── 1. Setup Shapes ──
        if is_mla:
            lmc_shape = (num_tokens, hidden_size)
            vllm_shape = (num_blocks, block_size, hidden_size)
        else:
            lmc_shape = (
                (num_tokens, 2, hidden_size)
                if token_major
                else (2, num_tokens, hidden_size)
            )
            if is_two_major:
                # flash attn: [2, num_blocks, block_size, num_heads, head_size]
                vllm_shape = (2, num_blocks, block_size, num_heads, head_size)
            else:
                # flash infer: [num_blocks, 2, block_size, num_heads, head_size]
                vllm_shape = (num_blocks, 2, block_size, num_heads, head_size)

        # ── 2. Deterministic Data ──
        lmc_size = 1
        for s in lmc_shape:
            lmc_size *= s
        vllm_size = 1
        for s in vllm_shape:
            vllm_size *= s

        lmc_tensor = (
            (torch.arange(lmc_size, device=device) % 1000)
            .to(torch.float16)
            .reshape(lmc_shape)
        )
        vllm_tensor = (
            (torch.arange(vllm_size, device=device) % 1000)
            .to(torch.float16)
            .reshape(vllm_shape)
        )

        # ── 3. Golden Reference ──
        lmc_ref = lmc_tensor.clone()
        vllm_ref = vllm_tensor.clone()
        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        if not direction:  # LMC → vLLM
            if is_mla:
                vllm_ref[block_indices, block_offsets, :] = lmc_ref
            else:
                src = lmc_ref if token_major else lmc_ref.permute(1, 0, 2)
                src = src.view(num_tokens, 2, num_heads, head_size)
                if is_two_major:
                    # [2, NB, BS, NH, HS]
                    vllm_ref[0, block_indices, block_offsets] = src[:, 0, :, :]
                    vllm_ref[1, block_indices, block_offsets] = src[:, 1, :, :]
                else:
                    # [NB, 2, BS, NH, HS]
                    vllm_ref[block_indices, 0, block_offsets] = src[:, 0, :, :]
                    vllm_ref[block_indices, 1, block_offsets] = src[:, 1, :, :]
        else:  # vLLM → LMC
            if is_mla:
                lmc_ref = vllm_ref[block_indices, block_offsets, :]
            else:
                if is_two_major:
                    k = vllm_ref[0, block_indices, block_offsets]
                    v = vllm_ref[1, block_indices, block_offsets]
                else:
                    k = vllm_ref[block_indices, 0, block_offsets]
                    v = vllm_ref[block_indices, 1, block_offsets]
                combined = torch.stack(
                    [k, v],
                    dim=1,
                ).view(num_tokens, 2, hidden_size)
                lmc_ref = combined if token_major else combined.permute(1, 0, 2)

        # ── 4. Execute ──
        xfer_dir = ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
        ops.single_layer_kv_transfer(
            lmc_tensor,
            vllm_tensor,
            slot_mapping,
            xfer_dir,
            gpu_kv_format,
            token_major,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

        # ── 5. Verify ──
        if not direction:
            torch.testing.assert_close(
                vllm_tensor,
                vllm_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"Mismatch in {case_desc}",
            )
        else:
            torch.testing.assert_close(
                lmc_tensor,
                lmc_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"Mismatch in {case_desc}",
            )

    # ── 6. Save canonical results for cross-backend comparison ──
    # Use flash attn (two_major) format to match original file names
    canonical_cases = [
        (False, True, False),  # l2v, non-MLA
        (False, True, True),  # v2l, non-MLA
        (True, True, False),  # l2v, MLA
        (True, True, True),  # v2l, MLA
    ]

    for is_mla, token_major, direction in canonical_cases:
        dir_tag = "v2l" if direction else "l2v"

        if is_mla:
            lmc_shape = (num_tokens, hidden_size)
            vllm_shape = (num_blocks, block_size, hidden_size)
            fmt = ops.GPUKVFormat.NL_X_NB_BS_HS
        else:
            lmc_shape = (num_tokens, 2, hidden_size)
            vllm_shape = (2, num_blocks, block_size, num_heads, head_size)
            fmt = ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS

        lmc_size = 1
        for s in lmc_shape:
            lmc_size *= s
        vllm_size = 1
        for s in vllm_shape:
            vllm_size *= s

        lmc_tensor = (
            (torch.arange(lmc_size, device=device) % 1000)
            .to(torch.float16)
            .reshape(lmc_shape)
        )
        vllm_tensor = (
            (torch.arange(vllm_size, device=device) % 1000)
            .to(torch.float16)
            .reshape(vllm_shape)
        )

        xfer_dir = ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
        ops.single_layer_kv_transfer(
            lmc_tensor,
            vllm_tensor,
            slot_mapping,
            xfer_dir,
            fmt,
            token_major,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

        result = lmc_tensor.cpu() if direction else vllm_tensor.cpu()
        save_result(
            f"single_layer_kv_transfer_{dir_tag}_mla_{is_mla}",
            result,
        )


def scenario_single_layer_kv_transfer_sgl():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    num_tokens = 32
    num_blocks = 128
    block_size = 16
    num_heads = 8
    head_size = 64
    hidden_size = num_heads * head_size

    slot_mapping = torch.arange(
        0,
        num_tokens * 3,
        3,
        device=device,
    ).to(torch.int64)

    # (token_major, direction)
    # direction: False = LMC→SGL, True = SGL→LMC
    test_cases = [
        (True, False),
        (False, False),
        (True, True),
        (False, True),
    ]

    for token_major, direction in test_cases:
        dir_tag = "s2l" if direction else "l2s"

        # 1. Setup Shapes
        lmc_shape = (
            (num_tokens, 2, hidden_size)
            if token_major
            else (2, num_tokens, hidden_size)
        )
        sgl_shape = (
            num_blocks,
            block_size,
            num_heads,
            head_size,
        )

        # 2. Deterministic Data
        lmc_size = 1
        for s in lmc_shape:
            lmc_size *= s
        sgl_size = 1
        for s in sgl_shape:
            sgl_size *= s

        lmc_tensor = (
            (torch.arange(lmc_size, device=device) % 500)
            .to(torch.float16)
            .reshape(lmc_shape)
        )
        sgl_k_tensor = (
            (torch.arange(sgl_size, device=device) % 500 + 500)
            .to(torch.float16)
            .reshape(sgl_shape)
        )
        sgl_v_tensor = (
            (torch.arange(sgl_size, device=device) % 500 + 1000)
            .to(torch.float16)
            .reshape(sgl_shape)
        )

        # 3. Golden Reference
        lmc_ref = lmc_tensor.clone()
        sgl_k_ref = sgl_k_tensor.clone()
        sgl_v_ref = sgl_v_tensor.clone()

        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        if not direction:  # LMC → SGL
            src = lmc_ref if token_major else lmc_ref.permute(1, 0, 2)
            src_k = src[:, 0, :].view(
                num_tokens,
                num_heads,
                head_size,
            )
            src_v = src[:, 1, :].view(
                num_tokens,
                num_heads,
                head_size,
            )
            sgl_k_ref[block_indices, block_offsets] = src_k
            sgl_v_ref[block_indices, block_offsets] = src_v
        else:  # SGL → LMC
            k_data = sgl_k_ref[block_indices, block_offsets].reshape(
                num_tokens, hidden_size
            )
            v_data = sgl_v_ref[block_indices, block_offsets].reshape(
                num_tokens, hidden_size
            )

            combined = torch.stack(
                [k_data, v_data],
                dim=1,
            )  # [N, 2, H]
            lmc_ref = combined if token_major else combined.permute(1, 0, 2)

        # 4. Execute
        ops.single_layer_kv_transfer_sgl(
            lmc_tensor,
            sgl_k_tensor,
            sgl_v_tensor,
            slot_mapping,
            ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D,
            token_major,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

        # 5. Verify
        case_desc = f"TM={token_major}, Dir={dir_tag}"
        if not direction:
            torch.testing.assert_close(
                sgl_k_tensor,
                sgl_k_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"K mismatch in {case_desc}",
            )
            torch.testing.assert_close(
                sgl_v_tensor,
                sgl_v_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"V mismatch in {case_desc}",
            )
        else:
            torch.testing.assert_close(
                lmc_tensor,
                lmc_ref,
                rtol=1e-3,
                atol=1e-3,
                msg=f"Mismatch in {case_desc}",
            )

        # 6. Save each case separately
        result = lmc_tensor.cpu() if direction else sgl_k_tensor.cpu()
        save_result(
            f"single_layer_kv_transfer_sgl_{dir_tag}_tm_{token_major}",
            result,
        )


def scenario_multi_layer_kv_transfer():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    num_layers = 2
    num_tokens = 4
    head_size = 16
    page_buffer_size = 10
    block_size = 5
    dtype = torch.float32

    slot_mapping = torch.tensor(
        [0, 2, 5, 9],
        dtype=torch.int64,
        device=device,
    )

    # ── Format-specific test cases ──
    # Each: (gpu_kv_format, is_mla, block_size_arg)
    format_cases = [
        (ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS, False, 1),  # vLLM cross layer
        (ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False, 1),  # flash attn
        (ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS, False, block_size),  # flash infer
        (ops.GPUKVFormat.NL_X_NB_BS_HS, True, 1),  # vLLM MLA
        (ops.GPUKVFormat.NL_X_NBBS_ONE_HS, True, 1),  # SGLang MLA
    ]

    for gpu_kv_format, is_mla, bs_arg in format_cases:
        k_or_v_size = 1 if is_mla else 2

        for direction in [True, False]:
            dir_tag = "paged2lmc" if direction else "lmc2paged"
            fmt_name = str(gpu_kv_format).split(".")[-1]

            # ── 1. LMCache Tensor ──
            lmc_shape = (k_or_v_size, num_layers, num_tokens, head_size)
            key_value = torch.zeros(lmc_shape, dtype=dtype, device=device)

            if not direction:  # LMC → Paged
                for kv in range(k_or_v_size):
                    for ly in range(num_layers):
                        for t in range(num_tokens):
                            val = (
                                kv * 5000
                                + ly * 1000
                                + t * 10
                                + torch.arange(head_size, device=device)
                            ).to(dtype)
                            key_value[kv, ly, t] = val

            # ── 2. Paged Buffers (one per layer) ──
            page_buffers = []
            for ly in range(num_layers):
                if gpu_kv_format == ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
                    num_blocks = page_buffer_size // bs_arg
                    pb = torch.zeros(
                        (num_blocks, 2, bs_arg, head_size),
                        dtype=dtype,
                        device=device,
                    )
                elif is_mla:
                    pb = torch.zeros(
                        (page_buffer_size, head_size),
                        dtype=dtype,
                        device=device,
                    )
                else:
                    # Handles NB_NL_TWO_BS_NH_HS and NL_X_TWO_NB_BS_NH_HS
                    pb = torch.zeros(
                        (2, page_buffer_size, head_size),
                        dtype=dtype,
                        device=device,
                    )

                if direction:  # Paged → LMC
                    for s in range(page_buffer_size):
                        for kv in range(k_or_v_size):
                            val = (
                                kv * 7000
                                + ly * 2000
                                + s * 10
                                + torch.arange(head_size, device=device)
                            ).to(dtype)
                            if gpu_kv_format == ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
                                blk_idx = s // bs_arg
                                blk_off = s % bs_arg
                                pb[blk_idx, kv, blk_off] = val
                            elif is_mla:
                                pb[s] = val
                            else:
                                # Handles NB_NL_TWO_BS_NH_HS and NL_X_TWO_NB_BS_NH_HS
                                pb[kv, s] = val

                page_buffers.append(pb)

            # ── 3. Pointer Tensor ──
            key_value_ptrs = torch.tensor(
                [pb.data_ptr() for pb in page_buffers],
                dtype=torch.int64,
                device=device,
            )

            # ── 4. Execute ──
            xfer_dir = (
                ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
            )
            ops.multi_layer_kv_transfer(
                key_value,
                key_value_ptrs,
                slot_mapping,
                torch.device(device),
                page_buffer_size,
                xfer_dir,
                gpu_kv_format,
                bs_arg,
            )
            if is_cuda_backend:
                torch.cuda.synchronize()

            # ── 5. Verify (internal, per-format) ──
            for t_id in range(num_tokens):
                s_idx = slot_mapping[t_id].item()
                for ly in range(num_layers):
                    for kv in range(k_or_v_size):
                        lmc_val = key_value[kv, ly, t_id]

                        if gpu_kv_format == ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
                            blk_idx = s_idx // bs_arg
                            blk_off = s_idx % bs_arg
                            paged_val = page_buffers[ly][blk_idx, kv, blk_off]
                        elif is_mla:
                            paged_val = page_buffers[ly][s_idx]
                        else:
                            # Handles NB_NL_TWO_BS_NH_HS and NL_X_TWO_NB_BS_NH_HS
                            paged_val = page_buffers[ly][kv, s_idx]

                        torch.testing.assert_close(
                            lmc_val,
                            paged_val,
                            msg=(
                                f"Mismatch: {fmt_name} {dir_tag}, "
                                f"kv={kv}, layer={ly}, token={t_id}"
                            ),
                        )

    # ── 6. Save ONE canonical result for cross-backend comparison ──
    # Use flash attn format (NL_X_TWO_NB_BS_NH_HS), direction=True (paged→lmc)
    # Re-run the canonical case to get a clean result
    canonical_format = ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
    for direction in [True, False]:
        dir_tag = "paged2lmc" if direction else "lmc2paged"
        lmc_shape = (2, num_layers, num_tokens, head_size)
        key_value = torch.zeros(lmc_shape, dtype=dtype, device=device)

        if not direction:
            for ly in range(num_layers):
                for t in range(num_tokens):
                    val = (
                        ly * 1000 + t * 10 + torch.arange(head_size, device=device)
                    ).to(dtype)
                    key_value[0, ly, t] = val
                    key_value[1, ly, t] = val + 500

        page_buffers = []
        for ly in range(num_layers):
            pb = torch.zeros(
                (2, page_buffer_size, head_size),
                dtype=dtype,
                device=device,
            )
            if direction:
                for s in range(page_buffer_size):
                    val = (
                        ly * 2000 + s * 10 + torch.arange(head_size, device=device)
                    ).to(dtype)
                    pb[0, s] = val
                    pb[1, s] = val + 700
            page_buffers.append(pb)

        key_value_ptrs = torch.tensor(
            [pb.data_ptr() for pb in page_buffers],
            dtype=torch.int64,
            device=device,
        )

        xfer_dir = ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
        ops.multi_layer_kv_transfer(
            key_value,
            key_value_ptrs,
            slot_mapping,
            torch.device(device),
            page_buffer_size,
            xfer_dir,
            canonical_format,
            1,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

        save_result(
            f"multi_layer_kv_transfer_{dir_tag}",
            key_value.cpu(),
        )


def scenario_multi_layer_kv_transfer_unilateral():
    ops, scene_info = get_test_context()
    is_cuda_backend = scene_info.startswith("cuda_ops")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = f"cuda:{torch.cuda.current_device()}" if is_cuda_backend else "cpu"

    num_layers = 2
    num_tokens = 4
    head_size = 16
    page_buffer_size = 10
    dtype = torch.float32

    slot_mapping = torch.tensor(
        [1, 3, 4, 7],
        dtype=torch.int64,
        device=device,
    )

    # ── Test cases: (gpu_kv_format, is_mla) ──
    format_cases = [
        (ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False),  # SGLang MHA (unilateral path)
        (
            ops.GPUKVFormat.NL_X_NB_BS_HS,
            True,
        ),  # vLLM MLA (delegates to multi_layer_kv_transfer)
        (
            ops.GPUKVFormat.NL_X_NBBS_ONE_HS,
            True,
        ),  # SGLang MLA (delegates to multi_layer_kv_transfer)
    ]

    for gpu_kv_format, is_mla in format_cases:
        k_or_v_size = 1 if is_mla else 2

        for direction in [True, False]:
            dir_tag = "p2l" if direction else "l2p"

            # ── 1. LMCache Tensor ──
            lmc_shape = (k_or_v_size, num_layers, num_tokens, head_size)
            lmc_tensor = torch.zeros(lmc_shape, dtype=dtype, device=device)

            if not direction:  # LMC → Paged
                for kv in range(k_or_v_size):
                    for ly in range(num_layers):
                        for t in range(num_tokens):
                            val = (
                                kv * 5000
                                + ly * 1000
                                + t * 10
                                + torch.arange(head_size, device=device)
                            ).to(dtype)
                            lmc_tensor[kv, ly, t] = val

            # ── 2. Paged Buffers ──
            if is_mla:
                # MLA delegates to multi_layer_kv_transfer
                # ptrs: [layer0, layer1, ...], each -> [page_buffer_size, head_size]
                page_buffers = []
                for ly in range(num_layers):
                    pb = torch.zeros(
                        (page_buffer_size, head_size),
                        dtype=dtype,
                        device=device,
                    )
                    if direction:  # Paged → LMC
                        for s in range(page_buffer_size):
                            val = (
                                ly * 2000
                                + s * 10
                                + torch.arange(head_size, device=device)
                            ).to(dtype)
                            pb[s] = val
                    page_buffers.append(pb)

                key_value_ptrs = torch.tensor(
                    [pb.data_ptr() for pb in page_buffers],
                    dtype=torch.int64,
                    device=device,
                )
            else:
                # Non-MLA unilateral: separate K/V buffers
                # ptrs: [K_l0, K_l1, ..., V_l0, V_l1, ...]
                # each -> [page_buffer_size, head_size]
                buffers = {}
                for kv in range(2):
                    for ly in range(num_layers):
                        pb = torch.zeros(
                            (page_buffer_size, head_size),
                            dtype=dtype,
                            device=device,
                        )
                        if direction:  # Paged → LMC
                            for s in range(page_buffer_size):
                                val = (
                                    kv * 7000
                                    + ly * 2000
                                    + s * 10
                                    + torch.arange(head_size, device=device)
                                ).to(dtype)
                                pb[s] = val
                        buffers[(kv, ly)] = pb

                ptr_list = []
                for ly in range(num_layers):
                    ptr_list.append(buffers[(0, ly)].data_ptr())
                for ly in range(num_layers):
                    ptr_list.append(buffers[(1, ly)].data_ptr())

                key_value_ptrs = torch.tensor(
                    ptr_list,
                    dtype=torch.int64,
                    device=device,
                ).contiguous()

            # ── 3. Execute ──
            xfer_dir = (
                ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
            )
            ops.multi_layer_kv_transfer_unilateral(
                lmc_tensor,
                key_value_ptrs,
                slot_mapping,
                torch.device(device),
                page_buffer_size,
                xfer_dir,
                gpu_kv_format,
            )
            if is_cuda_backend:
                torch.cuda.synchronize()

            # ── 4. Verify ──
            for t_id in range(num_tokens):
                s_idx = slot_mapping[t_id].item()
                for ly in range(num_layers):
                    for kv in range(k_or_v_size):
                        lmc_val = lmc_tensor[kv, ly, t_id]

                        if is_mla:
                            paged_val = page_buffers[ly][s_idx]
                        else:
                            paged_val = buffers[(kv, ly)][s_idx]

                        torch.testing.assert_close(
                            lmc_val,
                            paged_val,
                            msg=(
                                f"Mismatch: {gpu_kv_format} {dir_tag}, "
                                f"KV={kv}, layer={ly}, "
                                f"token={t_id}, slot={s_idx}"
                            ),
                        )

    # ── 5. Save canonical result for cross-backend comparison ──
    # Use non-MLA unilateral (the primary use case of this function)
    for direction in [True, False]:
        dir_tag = "p2l" if direction else "l2p"

        lmc_shape = (2, num_layers, num_tokens, head_size)
        lmc_tensor = torch.zeros(lmc_shape, dtype=dtype, device=device)

        if not direction:
            for kv in range(2):
                for ly in range(num_layers):
                    for t in range(num_tokens):
                        val = (
                            kv * 5000
                            + ly * 1000
                            + t * 10
                            + torch.arange(head_size, device=device)
                        ).to(dtype)
                        lmc_tensor[kv, ly, t] = val

        buffers = {}
        for kv in range(2):
            for ly in range(num_layers):
                pb = torch.zeros(
                    (page_buffer_size, head_size),
                    dtype=dtype,
                    device=device,
                )
                if direction:
                    for s in range(page_buffer_size):
                        val = (
                            kv * 7000
                            + ly * 2000
                            + s * 10
                            + torch.arange(head_size, device=device)
                        ).to(dtype)
                        pb[s] = val
                buffers[(kv, ly)] = pb

        ptr_list = []
        for ly in range(num_layers):
            ptr_list.append(buffers[(0, ly)].data_ptr())
        for ly in range(num_layers):
            ptr_list.append(buffers[(1, ly)].data_ptr())

        key_value_ptrs = torch.tensor(
            ptr_list,
            dtype=torch.int64,
            device=device,
        ).contiguous()

        xfer_dir = ops.TransferDirection.D2H if direction else ops.TransferDirection.H2D
        ops.multi_layer_kv_transfer_unilateral(
            lmc_tensor,
            key_value_ptrs,
            slot_mapping,
            torch.device(device),
            page_buffer_size,
            xfer_dir,
            ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        )
        if is_cuda_backend:
            torch.cuda.synchronize()

        save_result(
            f"multi_layer_kv_transfer_unilateral_{dir_tag}",
            lmc_tensor.cpu(),
        )


def scenario_alloc_free_pinned_ptr():
    ops, scene_info = get_test_context()

    alloc_size = 4096
    flags = 0  # cudaHostAllocDefault

    # 1. Allocate
    ptr = ops.alloc_pinned_ptr(alloc_size, flags)
    assert isinstance(ptr, int), f"Expected int, got {type(ptr)}"
    assert ptr != 0, "alloc_pinned_ptr returned null"

    # 2. Free
    ops.free_pinned_ptr(ptr)

    # 3. Save: 1 = PASS
    save_result(
        "alloc_free_pinned_ptr",
        torch.tensor([1], dtype=torch.int32),
    )


def scenario_alloc_free_numa_ptr():
    ops, scene_info = get_test_context()

    alloc_size = 4096
    node = 0  # NUMA node 0 (always exists)

    # 1. Allocate
    ptr = ops.alloc_numa_ptr(alloc_size, node)
    assert isinstance(ptr, int), f"Expected int, got {type(ptr)}"
    assert ptr != 0, "alloc_numa_ptr returned null"

    # 2. Free (must pass same size as alloc)
    ops.free_numa_ptr(ptr, alloc_size)

    # 3. Save: 1 = PASS
    save_result(
        "alloc_free_numa_ptr",
        torch.tensor([1], dtype=torch.int32),
    )


def scenario_alloc_free_pinned_numa_ptr():
    ops, scene_info = get_test_context()

    alloc_size = 4096
    node = 0  # NUMA node 0

    # 1. Allocate (NUMA + cudaHostRegister)
    ptr = ops.alloc_pinned_numa_ptr(alloc_size, node)
    assert isinstance(ptr, int), f"Expected int, got {type(ptr)}"
    assert ptr != 0, "alloc_pinned_numa_ptr returned null"

    # 2. Free (cudaHostUnregister + munmap)
    ops.free_pinned_numa_ptr(ptr, alloc_size)

    # 3. Save: 1 = PASS
    save_result(
        "alloc_free_pinned_numa_ptr",
        torch.tensor([1], dtype=torch.int32),
    )


def scenario_alloc_free_shm_pinned_ptr():
    ops, scene_info = get_test_context()

    alloc_size = 4096
    shm_name = "/test_lmcache_shm"

    # 1. Allocate
    ptr = ops.alloc_shm_pinned_ptr(alloc_size, shm_name)
    assert isinstance(ptr, int), f"Expected int, got {type(ptr)}"
    assert ptr != 0, "alloc_shm_pinned_ptr returned null"

    # 2. Free
    ops.free_shm_pinned_ptr(ptr, alloc_size, shm_name)

    # 3. Save: 1 = PASS
    save_result(
        "alloc_free_shm_pinned_ptr",
        torch.tensor([1], dtype=torch.int32),
    )


def scenario_transfer_direction_enum():
    ops, scene_info = get_test_context()

    # 1. Verify enum members exist
    td = ops.TransferDirection
    assert hasattr(td, "H2D"), "Missing TransferDirection.H2D"
    assert hasattr(td, "D2H"), "Missing TransferDirection.D2H"

    # 2. Verify values are distinct
    assert td.H2D != td.D2H, "H2D and D2H should be distinct"

    # 3. Extract int value (compatible with both
    #    pybind11 enum and Python enum)
    h2d = td.H2D
    d2h = td.D2H
    h2d_val = h2d.value if hasattr(h2d, "value") else int(h2d)
    d2h_val = d2h.value if hasattr(d2h, "value") else int(d2h)

    # 4. Save for cross-backend comparison
    save_result(
        "transfer_direction_enum",
        torch.tensor(
            [h2d_val, d2h_val],
            dtype=torch.int32,
        ),
    )


# ==========================================
# 3. Registry
# ==========================================
# cover pybind list in csrc/pybind.cpp
SCENARIO_REGISTRY = {
    "transfer_direction_enum": scenario_transfer_direction_enum,
    "multi_layer_kv_transfer": scenario_multi_layer_kv_transfer,
    "multi_layer_kv_transfer_unilateral": scenario_multi_layer_kv_transfer_unilateral,
    "single_layer_kv_transfer": scenario_single_layer_kv_transfer,
    "single_layer_kv_transfer_sgl": scenario_single_layer_kv_transfer_sgl,
    "load_and_reshape_flash": scenario_load_and_reshape_flash,
    "reshape_and_cache_back_flash": scenario_reshape_and_cache_back_flash,
    "lmcache_memcpy_async": scenario_lmcache_memcpy_async,
    "encode_fast_new": scenario_encode_fast_new,
    "decode_fast_new": scenario_decode_fast_new,
    "decode_fast_prefsum": scenario_decode_fast_prefsum,
    "calculate_cdf": scenario_calculate_cdf,
    "rotary_embedding_k_fused": scenario_rotary_embedding_k_fused,
    "alloc_free_pinned_ptr": scenario_alloc_free_pinned_ptr,
    "alloc_free_pinned_numa_ptr": scenario_alloc_free_pinned_numa_ptr,
    "alloc_free_numa_ptr": scenario_alloc_free_numa_ptr,
    "alloc_free_shm_pinned_ptr": scenario_alloc_free_shm_pinned_ptr,
    "get_gpu_pci_bus_id": scenario_get_gpu_pci_bus_id,
}


# ==========================================
# 4. Subprocess launcher
# ==========================================


def run_scenario(mode, cuda_visible):
    env = os.environ.copy()
    env["LMC_TEST_MODE"] = mode
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible

    print(
        f"\n>>> Launching Scenario: MODE={mode}, CUDA_VISIBLE_DEVICES='{cuda_visible}'"
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-s", "-q"],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result


# ==========================================
# 5. The test functions pytest sees
# ==========================================

if _is_child:
    # --- Child process: each scenario is its own test case ---

    @pytest.mark.parametrize("name", list(SCENARIO_REGISTRY.keys()))
    def test_scenario(name):
        SCENARIO_REGISTRY[name]()

else:
    # --- Top-level: launch all children once, then compare each function ---

    @pytest.fixture(scope="module")
    def run_all_children():
        """Launch child processes. Runs once for the entire module.

        If CUDA is available: launches 3 child processes for comparison:
        - CUDA_OPS with GPU visible
        - NON_CUDA with GPU visible (for comparison)
        - NON_CUDA without GPU visible

        If CUDA is not available: launches 1 child process:
        - NON_CUDA without GPU visible (no-crash test only)
        """
        if RESULTS_DIR.exists():
            shutil.rmtree(RESULTS_DIR)

        if _cuda_available:
            # CUDA available: run all scenarios for comparison
            scenarios = [("CUDA_OPS", "0"), ("NON_CUDA", "0"), ("NON_CUDA", "")]
        else:
            # CUDA not available: only run non-CUDA without GPU
            scenarios = [("NON_CUDA", "")]

        for mode, cuda_vis in scenarios:
            r = run_scenario(mode, cuda_vis)
            assert r.returncode == 0, (
                f"Scenario {mode}/CUDA_VISIBLE_DEVICES='{cuda_vis}' failed:\n"
                f"{r.stdout}\n{r.stderr}"
            )

    @pytest.mark.parametrize("name", list(SCENARIO_REGISTRY.keys()))
    def test_compare(run_all_children, name):
        """Each scenario function gets its own PASS/FAIL.

        If CUDA is available: compares results from all scenarios.
        If CUDA is not available: just verifies the scenario ran without crashing.
        """
        # Match: exact name or name as prefix (e.g. calculate_cdf → calculate_cdf_bins*)
        exact_files = sorted(RESULTS_DIR.glob(f"{name}@*.pt"))
        prefix_files = sorted(RESULTS_DIR.glob(f"{name}_*@*.pt"))
        all_files = sorted(set(exact_files + prefix_files))

        if _cuda_available:
            # CUDA available: compare results from 3 scenarios
            assert len(all_files) >= 3, (
                f"{name}: expected at least 3 results, found {len(all_files)}"
            )

            # Group by sub-function name
            sub_funcs = sorted(set(f.name.split("@")[0] for f in all_files))

            for sub in sub_funcs:
                sub_files = sorted(RESULTS_DIR.glob(f"{sub}@*.pt"))
                assert len(sub_files) == 3, (
                    f"{sub}: expected 3 results, found {len(sub_files)}"
                )

                data = {
                    f.name.split("@")[1].replace(".pt", ""): torch.load(
                        f, weights_only=False
                    )
                    for f in sub_files
                }

                scenes = list(data.keys())
                base_scene = scenes[0]
                base_val = data[base_scene]

                for scene in scenes:
                    val = data[scene]

                    if isinstance(val, torch.Tensor):
                        v_current = val.detach().cpu().float()
                        v_base = base_val.detach().cpu().float()
                        is_match = torch.allclose(
                            v_current, v_base, rtol=1e-4, atol=1e-4
                        )
                        if not is_match:
                            max_diff = (v_current - v_base).abs().max().item()
                            pytest.fail(
                                f"{sub}: {scene} vs {base_scene} mismatch, "
                                f"max diff = {max_diff:.2e}"
                            )
                    else:
                        if val != base_val:
                            pytest.fail(
                                f"{sub}: {scene}={val} != {base_scene}={base_val}"
                            )
        else:
            # CUDA not available: just verify the scenario ran (result files exist)
            assert len(all_files) >= 1, (
                f"{name}: expected at least 1 result (no-crash test)"
                f"found {len(all_files)}"
            )
            # No comparison needed
            # the fact that we have result files means it didn't crash
