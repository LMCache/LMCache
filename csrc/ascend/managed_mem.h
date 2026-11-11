// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <map>
#include <shared_mutex>
#include <string>

namespace lmc {

struct RegisteredMemoryRecord {
  uintptr_t ptr;
  uintptr_t devptr;
  size_t buffSize;
  int32_t device;
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
  HostRegisteredMemoryManager& operator=(const HostRegisteredMemoryManager&) =
      delete;
  HostRegisteredMemoryManager(HostRegisteredMemoryManager&&) = delete;
  HostRegisteredMemoryManager& operator=(HostRegisteredMemoryManager&&) =
      delete;

  std::map<void*, RegisteredMemoryRecord> allocatedMap;
  mutable std::shared_mutex mux;

 public:
  static HostRegisteredMemoryManager& GetInstance() {
    static HostRegisteredMemoryManager instance;
    return instance;
  }
  ~HostRegisteredMemoryManager();

  // Register a pointer through high level APIs (aclrt) return devPtr
  // Returns an already existing RegisteredMemoryRecord or the newly created one
  // Inputs:
  // -hostPtr: host pointer of the allocated memory area to register on device
  // -bufferSize: size of the allocated memory area to register on device
  RegisteredMemoryRecord* registerHostPtr(
      void* hostPtr,
      size_t bufferSize);  // torch::Tensor& tensor); //
  // Register a pointer through low level APIs (hal)
  // This should be used for driver versions, where cannot rely on
  // aclrtHostRegister() Returns the created RegisteredMemoryRecord Inputs:
  // -hostPtr: host pointer of the allocated memory area to register on device
  // -bufferSize: size of the allocated memory area to register on device
  RegisteredMemoryRecord* halRegisterHostPtr(void* hostPtr, size_t bufferSize);
  RegisteredMemoryRecord* registerMappedMem(void* hostPtr, void* devPtr,
                                            size_t bufferSize);
  int aclUnregisterHostPtr(void* hostPtr);
  int halUnregisterHostPtr(void* hostPtr);
  void* getDevicePtr(void* hostPtr);
  size_t getRecordSize(void* hostPtr);
  void unregisterAll();
};

std::string get_driver_version();
bool is_version_at_least_25(const std::string& version_str);
// Uregisters the malloced hostPtr
void hal_host_unregister_ptr(void* ptr);

}  // namespace lmc

void* register_ptr(void* ptr, size_t size);
int unregister_ptr(void* ptr);
void* register_mapping(void* hostPtr, void* devPtr, size_t size);

// Takes in input a host pointer, returns the corresponding device pointer
void* get_device_ptr(void* ptr);
