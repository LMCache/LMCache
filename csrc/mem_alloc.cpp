#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>            // for strerror
#include <linux/mempolicy.h>  // for MPOL_BIND, MPOL_MF_MOVE, MPOL_MF_STRICT
#include <sched.h>            // for sched_setaffinity
#include <cstdio>             // for FILE, fopen, fgets, fclose
#include <cstdlib>            // for atoi
#include <sys/stat.h>         // for access, F_OK
#include "mem_alloc.h"

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags) {
  void* ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&ptr, size, flags);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaHostAlloc failed: " + std::to_string(err));
  }
  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_ptr(uintptr_t ptr) {
  cudaError_t err = cudaFreeHost(reinterpret_cast<void*>(ptr));
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaFreeHost failed: " + std::to_string(err));
  }
}

static void first_touch(void* p, size_t size) {
  const long ps = sysconf(_SC_PAGESIZE);
  for (size_t off = 0; off < size; off += ps) {
    volatile char* c = (volatile char*)p + off;
    *c = 0;
  }
}

static inline int mbind_sys(void* addr, unsigned long len, int mode,
                            const unsigned long* nodemask,
                            unsigned long maxnode, unsigned int flags) {
  long rc = syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
  return (rc == -1) ? -errno : 0;
}

uintptr_t alloc_pinned_numa_ptr(size_t size, int node) {
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED)
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));

  // Get actual NUMA node count dynamically
  int max_numa_nodes = get_numa_node_count();
  if (node >= max_numa_nodes) {
    munmap(ptr, size);
    throw std::runtime_error(std::string("Invalid NUMA node: ") +
                             std::to_string(node) +
                             ", max nodes: " + std::to_string(max_numa_nodes));
  }

  // Check if node ID is too large for unsigned long mask (64 bits)
  if (node >= (sizeof(unsigned long) * 8)) {
    munmap(ptr, size);
    throw std::runtime_error(
        std::string("Node ID ") + std::to_string(node) +
        " is too large. This build only supports up to 64 NUMA nodes.");
  }

  unsigned long mask = 1UL << node;
  long maxnode = max_numa_nodes;
  if (mbind_sys(ptr, size, MPOL_BIND, &mask, maxnode,
                MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    int err = errno;
    munmap(ptr, size);
    throw std::runtime_error(std::string("mbind failed: ") + strerror(err));
  }

  first_touch(ptr, size);

  cudaError_t st = cudaHostRegister(ptr, size, 0);
  if (st != cudaSuccess) {
    munmap(ptr, size);
    throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                             cudaGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_numa_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);
  // Unpin first, then unmap.
  cudaError_t st = cudaHostUnregister(p);
  if (st != cudaSuccess) {
    munmap(p, size);
    throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                             cudaGetErrorString(st));
  }
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}

// CPU NUMA functions implementation
int set_cpu_affinity(const int* cpu_list, int cpu_count) {
  if (!cpu_list || cpu_count <= 0) {
    return -1;
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  for (int i = 0; i < cpu_count; i++) {
    if (cpu_list[i] >= 0) {
      CPU_SET(cpu_list[i], &cpuset);
    }
  }

  pid_t pid = getpid();
  int result = sched_setaffinity(pid, sizeof(cpu_set_t), &cpuset);

  return (result == 0) ? 0 : -errno;
}

int set_memory_policy(int policy, const int* nodes, int node_count) {
  if (!nodes || node_count <= 0) {
    return -1;
  }

  // Convert policy string to numeric value
  int policy_value;
  switch (policy) {
    case 0:  // local
      policy_value = MPOL_LOCAL;
      break;
    case 1:  // preferred
      policy_value = MPOL_PREFERRED;
      break;
    case 2:  // bind
      policy_value = MPOL_BIND;
      break;
    case 3:  // interleave
      policy_value = MPOL_INTERLEAVE;
      break;
    default:
      return -1;
  }

  // Get actual NUMA node count dynamically
  int max_numa_nodes = get_numa_node_count();

  // Create nodemask
  unsigned long nodemask = 0;
  for (int i = 0; i < node_count; i++) {
    // Check if node ID is too large for unsigned long mask (64 bits)
    if (nodes[i] >= (sizeof(unsigned long) * 8)) {
      return -1;  // Node ID too large for nodemask
    }
    if (nodes[i] >= 0 && nodes[i] < max_numa_nodes) {
      nodemask |= (1UL << nodes[i]);
    } else {
      return -1;  // Invalid node number
    }
  }

  // Set memory policy for current process using syscall
  long result =
      syscall(SYS_set_mempolicy, policy_value, &nodemask, max_numa_nodes);

  return (result == 0) ? 0 : -errno;
}

int get_numa_node_count() {
  // Try to read NUMA node count from /sys/devices/system/node/online
  FILE* file = fopen("/sys/devices/system/node/online", "r");
  if (file) {
    char line[256];
    if (fgets(line, sizeof(line), file)) {
      fclose(file);
      // Parse "0-2" format to count nodes
      int max_node = 0;
      char* saveptr;
      char* token = strtok_r(line, ",", &saveptr);
      while (token) {
        char* dash = strchr(token, '-');
        if (dash) {
          // Range format: "0-2"
          int start = atoi(token);
          int end = atoi(dash + 1);
          if (end > max_node) max_node = end;
        } else {
          // Single node format: "0"
          int node = atoi(token);
          if (node > max_node) max_node = node;
        }
        token = strtok_r(nullptr, ",", &saveptr);
      }
      return max_node + 1;  // +1 because nodes are 0-indexed
    }
    fclose(file);
  }
  
  // Fallback: try to count nodes in /sys/devices/system/node/
  // Use a more reasonable upper limit based on system architecture
  int max_possible_nodes = 1024;  // Increased from 256 for modern systems
  int node_count = 0;
  for (int i = 0; i < max_possible_nodes; i++) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", i);
    if (access(path, F_OK) == 0) {
      node_count = i + 1;
    } else {
      break;
    }
  }
  
  return (node_count > 0) ? node_count : 1;  // Default to 1 if all else fails
}

int get_cpu_count() {
  return sysconf(_SC_NPROCESSORS_ONLN);
}