#pragma once
#include <shared_mutex>
#include <map>
#include <torch/torch.h>
#include <torch/extension.h>

namespace lmc {

struct RegisteredMemoryRecord {
    uintptr_t ptr;
    uintptr_t devptr;
    size_t buffSize;
};

/* 
* We are not responsible for acl init and ctx initialization,
* we assume the user responsible for ctx initialization
*/
class HostRegisteredMemoryManager {
private:
    HostRegisteredMemoryManager();

    // Delete copy constructor and assignment operator
    HostRegisteredMemoryManager(const HostRegisteredMemoryManager&) = delete;
    HostRegisteredMemoryManager& operator=(const HostRegisteredMemoryManager&) = delete;
    HostRegisteredMemoryManager(HostRegisteredMemoryManager&&) = delete;
    HostRegisteredMemoryManager& operator=(HostRegisteredMemoryManager&&) = delete;

    std::map<void*, RegisteredMemoryRecord> allocatedMap;
    mutable std::shared_mutex mux;
    
public:
    static HostRegisteredMemoryManager& GetInstance()
    {
        static HostRegisteredMemoryManager instance;
        return instance;
    }
    ~HostRegisteredMemoryManager();
    
    // Register a pointer through high level APIs (aclrt) return devPtr
    // Returns an already existing RegisteredMemoryRecord or the newly created one
    // Inputs: 
    // -hostPtr: host pointer of the allocated memory area to register on device
    // -bufferSize: size of the allocated memory area to register on device
    RegisteredMemoryRecord  registerHostPtr(void* hostPtr, size_t bufferSize); //torch::Tensor& tensor); // 
    // Register a pointer through low level APIs (hal)
    // This should be used for driver versions, where cannot rely on aclrtHostRegister()
    // Returns the created RegisteredMemoryRecord
    // Inputs: 
    // -bufferSize: size of the allocated memory area to register on device
    RegisteredMemoryRecord  halRegisterHostPtr(size_t bufferSize);
    void                    unregisterMemory(void* hostPtr);
    void*                   getDevicePtr(void* hostPtr);
    size_t                  getRecordSize(void* hostPtr);
    void                    unregisterAll();
};
} // namespace lmc

// Register a tensor on the current device
// Inputs: 
// -tensor: The tensor to register on the device
// Returns the device ptr for that tensor
void* register_memory(torch::Tensor& tensor);
// Reverse of register
// Inputs: 
// -tensor: The tensor to register on the device
void  unregister_memory(torch::Tensor& tensor);
// Takes in input a host pointer, returns the corresponding device pointer
void* get_device_ptr(void* ptr);
