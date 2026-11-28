// SPDX-License-Identifier: Apache-2.0

#include "managed_mem.h"
#include "driver/ascend_hal.h"
#include "driver/ascend_hal_define.h"
#include <acl/acl.h>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <sys/mman.h>

// Only required for old driver version (look at registerHostPtr)
#ifdef PROF_ERROR
  // You can add a pragma message to see this in your build log if you want:
  // #pragma message("Undefining PROF_ERROR from ascend_hal.h before NPU
  // headers")
  #undef PROF_ERROR
#endif
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "torch/extension.h"

namespace lmc {
// Signatures for internal helper functions

// Gets the current device offsetting on ASCEND_RT_VISIBLE_DEVICES when needed
int get_device();

// Class implementations

HostRegisteredMemoryManager::HostRegisteredMemoryManager() {};

HostRegisteredMemoryManager::~HostRegisteredMemoryManager() {
  this->unregisterAll();
};

void HostRegisteredMemoryManager::unregisterAll() {
  const std::unique_lock<std::shared_mutex> guard(this->mux);

  // Iterate through each key-value pair in the map.
  for (const auto& pair : this->allocatedMap) {
    void* hostPtr = pair.first;
    aclrtHostUnregister(hostPtr);
  }

  // After unregistering all pointers, clear the map completely.
  this->allocatedMap.clear();
};

// Register a pointer through high level APIs (aclrt) return devPtr
// Returns the created RegisteredMemoryRecord
RegisteredMemoryRecord* HostRegisteredMemoryManager::registerHostPtr(
    void* hostPtr, size_t bufferSize) {  // torch::Tensor& tensor){

  TORCH_CHECK(
      !(hostPtr == nullptr || bufferSize == 0),
      "Error: hostPtr cannot be null and bufferSize must be greater than 0.");
  const std::unique_lock<std::shared_mutex> guard(this->mux);

  // Check if the host pointer is already registered
  if (this->allocatedMap.count(hostPtr)) {
    return &this->allocatedMap[hostPtr];
  }

  void* devPtr;
  aclError err = aclrtHostRegister(hostPtr, static_cast<uint64_t>(bufferSize),
                                   ACL_HOST_REGISTER_MAPPED, (void**)&devPtr);

  if (err != ACL_SUCCESS) {
    std::cerr << "Unable to aclrtHostRegister, errcode: " << err << std::endl;
    return nullptr;
  }

  this->allocatedMap.emplace(
      hostPtr, RegisteredMemoryRecord{reinterpret_cast<uintptr_t>(hostPtr),
                                      reinterpret_cast<uintptr_t>(devPtr),
                                      bufferSize, -1});

  return &this->allocatedMap[hostPtr];
};

// Register an existing host-device pointer mapping to the memory manager
// Returns the created RegisteredMemoryRecord
RegisteredMemoryRecord* HostRegisteredMemoryManager::registerMappedMem(
    void* hostPtr, void* devPtr, size_t bufferSize) {
  TORCH_CHECK(
      !(hostPtr == nullptr || devPtr == nullptr || bufferSize == 0),
      "Error: hostPtr and devPtr cannot be null and bufferSize must be greater "
      "than 0.");
  const std::unique_lock<std::shared_mutex> guard(this->mux);

  // Check if the host pointer is already registered
  TORCH_CHECK(!(this->allocatedMap.count(hostPtr)),
              "Error: hostPtr already registered to host memory manager.");

  this->allocatedMap.emplace(
      hostPtr, RegisteredMemoryRecord{reinterpret_cast<uintptr_t>(hostPtr),
                                      reinterpret_cast<uintptr_t>(devPtr),
                                      bufferSize, -1});

  return &this->allocatedMap[hostPtr];
};

// Register a pointer through low level APIs (HAL). Allocates a new pinned host
// memory This should be used for driver versions, where cannot rely on
// aclrtHostRegister() Returns the created RegisteredMemoryRecord
RegisteredMemoryRecord* HostRegisteredMemoryManager::halRegisterHostPtr(
    void* hostPtr, size_t bufferSize) {
  // We allocate a new chunk of memory, register it, and replace the tensor.
  // Essentially, the halHostRegister function requires a ptr given by mmap.
  TORCH_CHECK((bufferSize >= 0), "Error: bufferSize must be greater than 0.");
  const std::unique_lock<std::shared_mutex> guard(this->mux);

  void* devPtr;
  int device = get_device();
  auto drvRet = halHostRegister((void*)hostPtr, static_cast<UINT64>(bufferSize),
                                HOST_MEM_MAP_DEV_PCIE_TH, (UINT32)device,
                                (void**)&devPtr);
  if (drvRet != 0) {
    std::cerr << "Unable to halHostRegister: " << drvRet
              << " . Please ensure your driver version >= 24.1.0" << std::endl;
    return nullptr;
  }

  // Lock the memory and fail if impossible to lock
  auto lockErr = mlock(reinterpret_cast<void*>(hostPtr), bufferSize);
  if (lockErr == -1) {
    // This can happen in non-privileged mode or not enough rlimit,
    // let's not proceed since we wanted to guarantee pinned
    // because we already alloced, let's free
    auto ret = halHostUnregisterEx(reinterpret_cast<void*>(hostPtr),
                                   static_cast<UINT32>(device),
                                   HOST_MEM_MAP_DEV_PCIE_TH);
    TORCH_CHECK(
        ret == 0,
        "Unable to pin host memory, unable to unregister. Error code: " +
            std::to_string(ret))
    auto mret = munmap(reinterpret_cast<void*>(hostPtr), bufferSize);
    TORCH_CHECK(false, "Unable to pin host memory with error code: " +
                           std::to_string(lockErr))
  }

  this->allocatedMap.emplace(
      hostPtr,
      RegisteredMemoryRecord{reinterpret_cast<uintptr_t>(hostPtr),
                             reinterpret_cast<uintptr_t>(devPtr), bufferSize,
                             static_cast<int32_t>(device)});

  return &this->allocatedMap[hostPtr];
};

int HostRegisteredMemoryManager::aclUnregisterHostPtr(void* hostPtr) {
  TORCH_CHECK(hostPtr != nullptr, "Error: hostPtr cannot be null.");

  // we don't actually mind if it doesn't unregister,
  // at context destroy it should be unregister anyway.
  const std::unique_lock<std::shared_mutex> guard(this->mux);
  if (this->allocatedMap.count(hostPtr) == 0) {
    // we probably did not register anyway
    return 0;
  }
  aclError err = aclrtHostUnregister(hostPtr);
  this->allocatedMap.erase(hostPtr);
  return static_cast<int>(err);
};

int HostRegisteredMemoryManager::halUnregisterHostPtr(void* hostPtr) {
  TORCH_CHECK(hostPtr != nullptr, "Error: hostPtr cannot be null.");
  const std::unique_lock<std::shared_mutex> guard(this->mux);
  if (this->allocatedMap.count(hostPtr) == 0) {
    // we probably did not register anyway
    return 0;
  }
  auto record = this->allocatedMap[hostPtr];
  auto err = halHostUnregisterEx(reinterpret_cast<void*>(hostPtr),
                                 static_cast<UINT32>(record.device),
                                 HOST_MEM_MAP_DEV_PCIE_TH);
  return static_cast<int>(err);
}

/*
 *    For now we only do a linear search as we probably won't have a long list
 * of ptrs we go through each record and check whether we are in range, if so we
 * calculate the offset from the host ptr and apply to the device ptr finally we
 * return the device ptr.
 */
void* HostRegisteredMemoryManager::getDevicePtr(void* hostPtr) {
  if (hostPtr == nullptr) {
    return nullptr;
  }
  const std::shared_lock<std::shared_mutex> guard(this->mux);

  const uintptr_t hostAddrPtr = reinterpret_cast<uintptr_t>(hostPtr);

  for (const auto& pair : this->allocatedMap) {
    const RegisteredMemoryRecord& record = pair.second;

    if (hostAddrPtr >= record.ptr &&
        hostAddrPtr < (record.ptr + record.buffSize)) {
      const size_t offset = hostAddrPtr - record.ptr;

      const uintptr_t deviceAddrPtr = record.devptr + offset;

      return reinterpret_cast<void*>(deviceAddrPtr);
    }
  }

  return nullptr;
};

size_t HostRegisteredMemoryManager::getRecordSize(void* hostPtr) {
  if (hostPtr == nullptr) {
    return 0;
  }
  const std::shared_lock<std::shared_mutex> guard(this->mux);

  const uintptr_t hostAddrPtr = reinterpret_cast<uintptr_t>(hostPtr);

  for (const auto& pair : this->allocatedMap) {
    const RegisteredMemoryRecord& record = pair.second;

    if (hostAddrPtr >= record.ptr &&
        hostAddrPtr < (record.ptr + record.buffSize)) {
      return record.buffSize;
    }
  }
  return 0;
};

std::string get_driver_version() {
  void* handle = nullptr;
  int (*dsmi_get_version)(int, char*, unsigned int, unsigned int*) = nullptr;
  std::string result;

  handle = dlopen("libdrvdsmi_host.so", RTLD_LAZY);
  if (!handle) {
    TORCH_CHECK(false,
                std::string("Error opening libdrvdsmi_host.so: ") + dlerror());
    return result;
  }
  dlerror();

  // Load the function
  *(void**)(&dsmi_get_version) = dlsym(handle, "dsmi_get_version");
  const char* dlsym_error = dlerror();
  if (dlsym_error) {
    dlclose(handle);
    TORCH_CHECK(false,
                std::string("Error loading dsmi_get_version: ") + dlsym_error);
    return result;
  }

  // Call the function
  int device_id = c10_npu::getCurrentNPUStream().device_index();
  const unsigned int buffer_size = 256;
  std::vector<char> version_buffer(buffer_size);
  unsigned int ret_len = 0;
  int ret =
      dsmi_get_version(device_id, version_buffer.data(), buffer_size, &ret_len);
  if (ret == 0) {
    if (ret_len > 0 && ret_len <= buffer_size) {
      version_buffer[ret_len] = '\0';  // Ensure null-termination
      result = version_buffer.data();
    } else {
      TORCH_CHECK(false,
                  "Error: Invalid length returned: " + std::to_string(ret_len));
    }
  } else {
    TORCH_CHECK(false,
                "Error: dsmi_get_version returned " + std::to_string(ret));
  }

  dlclose(handle);

  return result;
}

// To be on the safe side, returns false in case of uncertainties
bool is_version_at_least_25(const std::string& version_str) {
  if (version_str.empty()) {
    return false;
  }

  size_t num_end = 0;
  long major_version = 0;

  try {
    major_version = std::stol(version_str, &num_end);
  } catch (const std::invalid_argument&) {
    // No valid number at start
    return false;
  } catch (const std::out_of_range&) {
    // Should never happen, here for robustness
    return false;
  }
  return major_version >= 25;
}

int get_device() {
  int device = c10_npu::getCurrentNPUStream().device_index();
  const char* env_visible_devices_p = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
  // If we are using a custom list of visible devices, the index refers to that
  if (env_visible_devices_p != nullptr) {
    std::string env_visible_devices = env_visible_devices_p;
    std::vector<uint32_t> list_visible_devices;
    std::stringstream ss(env_visible_devices);
    std::string item;
    while (std::getline(ss, item, ',')) {
      list_visible_devices.push_back(std::stoi(item));
    }
    std::sort(list_visible_devices.begin(), list_visible_devices.end());
    // Here two cases are possible:
    // 1. no hccl, we just use current_device, even though we have specify the
    // ASCEND_RT_VISIBLE_DEVICES
    // 2. hccl, and we use current_device that seems to be correct
    // for case 2, since the current_device would have been correct anyway,
    // obtaining from the list would be fine. for case 1, we have shifted the
    // device to the RT_VISIBLE_DEVICES, so it should be corrected.
    device = list_visible_devices[device];
  }
  return device;
}

void hal_host_unregister_ptr(void* ptr) {
  if (ptr) {
    int device = get_device();
    auto& hmm = HostRegisteredMemoryManager::GetInstance();
    size_t bufferSize = hmm.getRecordSize(ptr);
    auto ret = halHostUnregisterEx(reinterpret_cast<void*>(ptr),
                                   static_cast<UINT32>(device),
                                   HOST_MEM_MAP_DEV_PCIE_TH);
    if (ret != 0) {
      std::cout << "Unable to hal host unregister: " << ret << std::endl;
    }
    auto mret = munmap(reinterpret_cast<void*>(ptr), bufferSize);
    if (mret != 0) {
      std::cout << "Unable to unmap memory: " << ret << std::endl;
    }
  }
}

}  // namespace lmc

/*
 * Caller should check whether this return nullptr for error handling.
 */
void* register_ptr(void* ptr, size_t size) {
  // assumed this is a host ptr
  TORCH_CHECK(ptr != nullptr, "ptr is a nullptr.");
  auto& hmm = lmc::HostRegisteredMemoryManager::GetInstance();
  std::string verString = lmc::get_driver_version();
  lmc::RegisteredMemoryRecord* record;

  if (lmc::is_version_at_least_25(
          verString)) {  // New driver version, supports aclrtHostRegister()
    record = hmm.registerHostPtr(ptr, size);
  } else {
    record = hmm.halRegisterHostPtr(ptr, size);
  }

  if (record == nullptr) {
    return nullptr;
  }

  return reinterpret_cast<void*>(record->devptr);
}

void* register_mapping(void* hostPtr, void* devPtr, size_t size) {
  TORCH_CHECK(hostPtr != nullptr, "hostPtr is a nullptr.");
  TORCH_CHECK(devPtr != nullptr, "devPtr is a nullptr.");
  auto& hmm = lmc::HostRegisteredMemoryManager::GetInstance();
  lmc::RegisteredMemoryRecord* record =
      hmm.registerMappedMem(hostPtr, devPtr, size);

  if (record == nullptr) {
    return nullptr;
  }

  return reinterpret_cast<void*>(record->devptr);
}

int unregister_ptr(void* ptr) {
  TORCH_CHECK(ptr != nullptr, "ptr is a nullptr.");
  auto& hmm = lmc::HostRegisteredMemoryManager::GetInstance();
  std::string verString = lmc::get_driver_version();
  if (lmc::is_version_at_least_25(verString)) {
    return hmm.aclUnregisterHostPtr(ptr);
  } else {
    return hmm.halUnregisterHostPtr(ptr);
  }
}

void* get_device_ptr(void* ptr) {
  auto& hmm = lmc::HostRegisteredMemoryManager::GetInstance();
  return hmm.getDevicePtr(ptr);
};
