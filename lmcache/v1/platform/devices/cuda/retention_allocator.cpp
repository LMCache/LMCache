// SPDX-License-Identifier: Apache-2.0
// PyTorch CUDAPluggableAllocator callbacks. No CUDA runtime or torch ABI dependency.
#include <cuda.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>

namespace {
struct Extent { size_t size; uint64_t seq; int device; };
std::mutex mu;
std::map<CUdeviceptr, Extent> mapped;
uint64_t sequence = 0;
int log_fd = -1;
char generation[128];
bool log_event(const char* name, uint64_t seq, size_t size, CUdeviceptr ptr, int rc, uint64_t handle = 0) {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return dprintf(log_fd, "{\"event\":\"%s\",\"generation\":\"%s\",\"creation_id\":\"%s.%llu\","
        "\"handle\":%llu,\"bytes\":%zu,\"base\":%llu,\"rc\":%d,\"mono_ns\":%llu}\n",
        name, generation, generation, (unsigned long long)seq, (unsigned long long)handle, size,
        (unsigned long long)ptr, rc, (unsigned long long)t.tv_sec*1000000000ULL+t.tv_nsec)>0;
}
}

extern "C" int zc_init(const char* path, const char* gen) {
    std::lock_guard<std::mutex> guard(mu);
    if (log_fd != -1 || strlen(gen) >= sizeof(generation)) return -1;
    log_fd = open(path, O_APPEND | O_WRONLY | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (log_fd < 0) return -1;
    strcpy(generation, gen);
    return 0;
}

extern "C" void* zc_malloc(ssize_t requested, int device, CUstream) {
    std::lock_guard<std::mutex> guard(mu);
    if (requested <= 0 || log_fd < 0) return nullptr;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t gran = 0;
    if (cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || !gran)
        return nullptr;
    const size_t size = ((size_t(requested) + gran - 1) / gran) * gran;
    CUmemGenericAllocationHandle handle{};
    CUdeviceptr base = 0;
    auto rc = cuMemCreate(&handle, size, &prop, 0);
    const auto seq = ++sequence;
    bool recorded = log_event("create", seq, size, 0, rc, rc == CUDA_SUCCESS ? handle : 0);
    if (rc != CUDA_SUCCESS) return nullptr;
    bool reserved = false, did_map = false;
    if (recorded && (rc = cuMemAddressReserve(&base, size, 0, 0, 0)) == CUDA_SUCCESS) {
        reserved = true;
        rc = cuMemMap(base, size, 0, handle, 0);
        did_map = rc == CUDA_SUCCESS;
        recorded = log_event("map", seq, size, base, rc, handle);
        if (did_map && recorded) {
            CUmemAccessDesc desc{};
            desc.location = prop.location;
            desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            rc = cuMemSetAccess(base, size, &desc, 1);
        }
    }
    auto release = cuMemRelease(handle);
    recorded = log_event("creator_handle_release", seq, size, base, release, handle) && recorded;
    if (recorded && did_map && rc == CUDA_SUCCESS && release == CUDA_SUCCESS) {
        mapped.emplace(base, Extent{size, seq, device});
        return reinterpret_cast<void*>(base);
    }
    if (did_map) log_event("rollback_unmap", seq, size, base, cuMemUnmap(base,size));
    if (reserved) log_event("rollback_va_free", seq, size, base, cuMemAddressFree(base,size));
    return nullptr;
}

extern "C" int zc_lookup(uint64_t ptr, size_t span, uint64_t* base, uint64_t* size, uint64_t* seq) {
    std::lock_guard<std::mutex> guard(mu);
    auto it = mapped.upper_bound(ptr);
    if (it == mapped.begin()) return -1;
    --it;
    auto offset = ptr - it->first;
    if (offset >= it->second.size || span > it->second.size - offset) return -1;
    *base = it->first; *size = it->second.size; *seq = it->second.seq;
    return 0;
}

extern "C" void zc_free(void* ptr, ssize_t, int, CUstream) {
    std::lock_guard<std::mutex> guard(mu);
    auto base = reinterpret_cast<CUdeviceptr>(ptr);
    auto it = mapped.find(base);
    if (it == mapped.end()) return;
    auto e = it->second;
    // The caching allocator retires blocks before releasing its segment.
    auto rc = cuMemUnmap(base,e.size);
    log_event("worker_unmap", e.seq, e.size, base, rc);
    if (rc == CUDA_SUCCESS) {
        mapped.erase(it);
        log_event("worker_va_free", e.seq, e.size, base, cuMemAddressFree(base,e.size));
    }
}
