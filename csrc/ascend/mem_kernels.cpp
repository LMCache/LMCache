#include "mem_kernels.h"
#include <ATen/ATen.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>
#include "utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <pybind11/pybind11.h>
#include <Python.h>

template <typename T, typename TENSOR_TYPE>
T* get_kernel_ptr(TENSOR_TYPE& tensor) {
    torch::Device device = tensor.device();
    // NPU should be using PrivateUse1
    if (device.is_privateuseone() || device.is_cuda()) {
        return static_cast<T*>(tensor.data_ptr());
    } else if (device.is_cpu()) {
        // find device ptr based on the host pinned ptr
        // because acl does not currently support HostGetDevicePointer API
        void* devPtr = get_device_ptr(tensor.data_ptr());
        TORCH_CHECK(devPtr != nullptr, "Unable to retrieve device ptr, is this a host registered pointer ?");
        return reinterpret_cast<T*>(devPtr);
    } else {
        TORCH_CHECK(false, "Invalid device. Device must be ascend (PrivateUseOne) or pinned cpu.");
    }
}

/**
 * Quickly offload KV cache from vLLM paged memory to the offloading buffer
 * Processes all the layers at the same time
 *
 * Each layer in vLLM's KV buffer has a shape of
 * [2, PAGE_BUFFER_SIZE, num_heads*head_size]
 *
 * Each AIV Core processes the copy for a token
 *
 * Therefore:
 *  AIV Core - token
 *
 * The function does:
 * slot_id = slot_mapping[tokenId]
 * ptrs[mem_offset(kv, layer, tokenId, hiddenDims)] = key_value[mem_offset(kv, layer, pages, pageSize, slot_id, hiddenDims)]
 *
 * Param:
 *  - direction: false  means LMCache to PagedBuffer, true  means PagedBuffer to
 * LMCache
 */
void multi_layer_kv_transfer(torch::Tensor& key_value, // [kv, num_layer, num_tokens, hidden]
                             const torch::Tensor& key_value_ptrs, // [num_layers]
                             const torch::Tensor& slot_mapping, // [num_tokens]
                             const torch::Device& paged_memory_device,
                             const int page_buffer_size, const bool direction,
                             const bool use_mla) {
    uint8_t* key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);
    // it is actually a uint8_t**. we will reinterpret it inside the kernel
    uint8_t* page_buffer_ptrs = get_kernel_ptr<uint8_t, const torch::Tensor>(key_value_ptrs);
    uint8_t* slot_mapping_ptr = get_kernel_ptr<uint8_t, const torch::Tensor>(slot_mapping);

    int num_layers = key_value.size(1);
    int num_tokens = slot_mapping.size(0);
    int hidden_dims = key_value.size(-1);
    int kv_size = 2;
    if (use_mla) {
        kv_size = 1;
    }
    
    const c10::OptionalDeviceGuard device_guard(paged_memory_device);
    // we require the kv ptr list to be on the device too
    const c10::OptionalDeviceGuard kv_device_guard(device_of(key_value_ptrs));

    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at::ScalarType scalar_type = key_value.scalar_type();
    at::ScalarType slot_type = slot_mapping.scalar_type();
    const char* socName = aclrtGetSocName();
    
    at_npu::native::OpCommand cmd;
    cmd.Name("multi_layer_kv_transfer_kernel");
    cmd.SetCustomHandler([scalar_type, slot_type, socName, stream, page_buffer_ptrs, key_value_ptr,
                          slot_mapping_ptr, hidden_dims, kv_size, num_layers, page_buffer_size,
                          num_tokens, direction]()->int{
        auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
        auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socName);
        uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
        lmc::multi_layer_kv_transfer_kernel(dtype_num, slot_num, aiv_num, stream, page_buffer_ptrs, key_value_ptr,
                                        slot_mapping_ptr, hidden_dims, kv_size, num_layers, page_buffer_size,
                                        num_tokens, direction);
        return 0;
    });
    cmd.Run();
    return ;
};


void multi_layer_kv_transfer_unilateral(torch::Tensor& key_value,
                                        const torch::Tensor& key_ptrs,
                                        const torch::Tensor& value_ptrs,
                                        const torch::Tensor& slot_mapping,
                                        const torch::Device& paged_memory_device,
                                        const int page_buffer_size,
                                        const bool direction){
       // TODO:
    PyErr_SetString(PyExc_NotImplementedError, "Please contact LMCache Ascend.");
    throw py::error_already_set();
};


void single_layer_kv_transfer(torch::Tensor& lmc_key_value_cache, // [num_tokens, 2, num_heads*head_size]
                                                                  // or
                                                                  // [2, num_tokens, num_heads*head_size]
                              torch::Tensor& vllm_key_cache, // [num_blocks, block_size, num_heads, head_size]
                              torch::Tensor& vllm_value_cache, // [....]
                              torch::Tensor& slot_mapping, // [num_tokens]
                              const bool direction, // false: LMCache to PagedBuffer, true: PagedBuffer to LMCache
                              const bool token_major // true: lmc_key_value_cache is [num_tokens, 2, num_heads*head_size]
                                                     // false: otherwise
) {
    uint8_t *lmc_key_value_cache_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(lmc_key_value_cache);
    uint8_t *vllm_key_cache_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(vllm_key_cache);
    uint8_t *vllm_value_cache_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(vllm_value_cache);
    uint8_t *slot_mapping_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(slot_mapping);

    int num_tokens = slot_mapping.size(0);
    int hidden_dims = lmc_key_value_cache.size(-1);

    const c10::OptionalDeviceGuard device_guard(device_of(vllm_key_cache));
    const c10::OptionalDeviceGuard slot_device_guard(device_of(slot_mapping));
    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    at::ScalarType scalar_type = vllm_key_cache.scalar_type();
    at::ScalarType slot_type = slot_mapping.scalar_type();
    
    const char* socName = aclrtGetSocName();

    at_npu::native::OpCommand cmd;
    cmd.Name("single_layer_kv_transfer_kernel");
    cmd.SetCustomHandler([scalar_type, slot_type, socName, stream, lmc_key_value_cache_ptr,
                          vllm_key_cache_ptr, vllm_value_cache_ptr, slot_mapping_ptr,
                          hidden_dims, num_tokens, direction, token_major]() -> int {
        auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
        auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socName);
        uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
        // TODO: We will add the isMLA argument once the signature have support for the MLA.
        lmc::single_layer_kv_transfer_kernel(dtype_num, slot_num, aiv_num, stream, lmc_key_value_cache_ptr,
                                         vllm_key_cache_ptr, vllm_value_cache_ptr, slot_mapping_ptr,
                                         hidden_dims, num_tokens, direction, token_major, false);
        return 0;
    });
    cmd.Run();
    return ;
};

void load_and_reshape_flash(
    torch::Tensor& key_value, // [2, num_layer, num_tokens, num_heads*head_size]
                              // must be one gpu / pinned cpu
    torch::Tensor& key_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& value_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& slot_mapping, // [num_tokens],
    const int layer_idx) {
    
    uint8_t* key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);
    uint8_t* key_cache_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_cache);
    uint8_t* value_cache_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(value_cache);

    uint8_t* slot_mapping_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(slot_mapping);

    int num_tokens = slot_mapping.size(0);
    int num_layers = key_value.size(1);
    int block_size = key_cache.size(1);
    int num_blocks = key_cache.size(0);
    int hidden_dims = key_value.size(-1);
    const c10::OptionalDeviceGuard device_guard(device_of(key_cache));
    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    at::ScalarType scalar_type = key_value.scalar_type();
    at::ScalarType slot_type = slot_mapping.scalar_type();
    const char* socName = aclrtGetSocName();

    at_npu::native::OpCommand cmd;
    cmd.Name("load_and_reshape_flash_kernel");
    cmd.SetCustomHandler([scalar_type, slot_type, socName, stream, key_value_ptr,
                          key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
                          hidden_dims, num_blocks, block_size,
                          num_tokens, num_layers, layer_idx]()->int {
        auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
        auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socName);
        uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
        lmc::load_and_reshape_flash_kernel(dtype_num, slot_num, aiv_num, stream, key_value_ptr,
                                       key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
                                       hidden_dims, num_blocks, block_size,
                                       num_tokens, num_layers, layer_idx, true);
        return 0;
    });
    cmd.Run();
    return;
};

void reshape_and_cache_back_flash(
    torch::Tensor& key_value, // [2, num_layer, num_tokens, num_heads*head_size]
                              // must be one gpu / pinned cpu
    torch::Tensor& key_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& value_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& slot_mapping, // [num_tokens],
    const int layer_idx) {
    
    uint8_t* key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);
    uint8_t* key_cache_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_cache);
    uint8_t* value_cache_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(value_cache);

    uint8_t* slot_mapping_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(slot_mapping);

    int num_tokens = slot_mapping.size(0);
    int num_layers = key_value.size(1);
    int block_size = key_cache.size(1);
    int num_blocks = key_cache.size(0);
    int hidden_dims = key_value.size(-1);
    const c10::OptionalDeviceGuard device_guard(device_of(key_cache));
    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    at::ScalarType scalar_type = key_value.scalar_type();
    at::ScalarType slot_type = slot_mapping.scalar_type();
    
    const char* socName = aclrtGetSocName();

    at_npu::native::OpCommand cmd;
    cmd.Name("reshape_and_cache_back_flash");
    cmd.SetCustomHandler([scalar_type, slot_type, socName, stream, key_value_ptr,
                          key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
                          hidden_dims, num_blocks, block_size,
                          num_tokens, num_layers, layer_idx]() -> int {
        auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
        auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socName);
        uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
        lmc::load_and_reshape_flash_kernel(dtype_num, slot_num, aiv_num, stream, key_value_ptr,
                                       key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
                                       hidden_dims, num_blocks, block_size,
                                       num_tokens, num_layers, layer_idx, false);
        return 0;
    });
    cmd.Run();
    return;
};
