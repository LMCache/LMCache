#pragma once
#include <torch/torch.h>
#include <torch/extension.h>
#include <acl/acl.h>
#include "managed_mem.h"

void unregisterPtr(void* ptr) {
    if (ptr) {
        auto& hmm = lmc::HostRegisteredMemoryManager::GetInstance();
        hmm.unregisterMemory(ptr);
    }
}

torch::Tensor create_pinned_host_registered_tensor(size_t bufferSize) {
    torch::TensorOptions tensorOpsCpu = torch::TensorOptions()
                                                .dtype(torch::kUInt8)
                                                .device(torch::kCPU)
                                                .pinned_memory(true);
    TORCH_CHECK(bufferSize > 0, "Buffer size must be greater than zero. Got: " + std::to_string(bufferSize));
    
    // unlikely this would be greater than int64_t
    int64_t numel = static_cast<int64_t>(bufferSize);
    
    void* hostPtr;
    aclError err = aclrtMallocHost((void**)&hostPtr, bufferSize);
    TORCH_CHECK(err == 0, "Unable to malloc host buffer, error: " + std::to_string(err));

    auto& hmm = lmc::HostRegisteredMemoryManager::GetInstance();
    hmm.registerHostPtr(hostPtr, bufferSize);
    
    std::vector<int64_t> dims = {numel};
    return torch::from_blob(hostPtr, dims, unregisterPtr, tensorOpsCpu);
}