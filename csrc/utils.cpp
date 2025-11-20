#include <stdexcept>
#include "utils.h"

#ifndef USE_XPU

#include <cuda_runtime.h>

std::string get_gpu_pci_bus_id(int device) {
  char pciBusId[13];  // 13 bytes per CUDA doc
  cudaError_t err = cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaDeviceGetPCIBusId failed: ") +
                             cudaGetErrorString(err));
  }
  return std::string(pciBusId);
}

#else

#include <sycl/sycl.hpp>
#include <iostream>

std::string get_gpu_pci_bus_id(int device) {
    sycl::queue q;
    std::vector<sycl::device> devices = sycl::device::get_devices(sycl::info::device_type::gpu);

    if (device >= 0 && device < static_cast<int>(devices.size())) {
        q = sycl::queue(devices[device]);
    } else {
        q = sycl::queue(sycl::default_selector_v);
    }

    auto d = q.get_device();

    auto ext_pci_props = d.get_info<sycl::ext::intel::info::device::pci_address>();
    return ext_pci_props.substr(5,2);
}

#endif

