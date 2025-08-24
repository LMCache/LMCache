# [Core] Add CPU NUMA affinity support with dynamic system detection

## 📋 **PR Summary**

This PR implements CPU NUMA affinity functionality for LMCache, addressing the **GPU NUMA-affinity placement** enhancement mentioned in [LMCache Q3 Roadmap #1253](https://github.com/LMCache/LMCache/issues/1253). The implementation follows the existing architectural pattern established in [PR #1409](https://github.com/LMCache/LMCache/pull/1409) and provides comprehensive NUMA-aware memory and CPU placement for both GPU and CPU backends.

## 🎯 **Related Issues**

- **Fixes**: [LMCache Q3 Roadmap #1253](https://github.com/LMCache/LMCache/issues/1253) - GPU NUMA-affinity placement
- **Related**: [PR #1409](https://github.com/LMCache/LMCache/pull/1409) - Existing GPU NUMA functionality

## 🚀 **New Features**

### **CPU NUMA Affinity Support**
- **CPU Affinity Binding**: Bind processes to specific CPU cores using `sched_setaffinity`
- **Memory Policy Management**: Set NUMA memory allocation policies (local, preferred, bind, interleave)
- **Dynamic System Detection**: Automatically detect actual NUMA topology without hardcoded limits
- **Configuration Integration**: Seamless integration with existing LMCache configuration system

### **Enhanced NUMA Architecture**
- **Unified NUMA Interface**: Consistent API for both GPU and CPU NUMA operations
- **Dynamic Limits**: Removes hardcoded 64/256 NUMA node limits, supports up to 1024 nodes
- **Fallback Mechanisms**: Robust error handling and system compatibility
- **Performance Optimization**: C++ implementation in `c_ops` module for high-performance operations

## 🔧 **Technical Implementation**

### **C++ Core Functions** (`csrc/mem_alloc.cpp`)
```cpp
// CPU NUMA functions with dynamic system detection
int set_cpu_affinity(const int* cpu_list, int cpu_count);
int set_memory_policy(int policy, const int* nodes, int node_count);
int get_numa_node_count();  // Dynamic detection from /sys/devices/system/node/
int get_cpu_count();        // Dynamic detection from sysconf
```

### **Python Integration** (`lmcache/v1/storage_backend/local_cpu_backend.py`)
- Automatic CPU NUMA settings application during backend initialization
- Environment variable support: `LMCACHE_CPU_AFFINITY`, `LMCACHE_NUMA_POLICY`
- Configuration-driven NUMA placement

### **Configuration Schema** (`lmcache/v1/config.py`)
```python
# New CPU NUMA configuration options
"cpu_numa_nodes": Optional[Union[str, list[int]]]      # NUMA nodes for CPU operations
"cpu_numa_policy": str                                 # Memory policy (local, preferred, bind, interleave)
"cpu_affinity": Optional[Union[str, list[int]]]       # CPU core affinity binding
```

## 📊 **System Compatibility**

### **Tested Environments**
- **NUMA Topology**: 3 NUMA nodes (0-2) with 128 CPU cores
- **Memory Distribution**: 
  - Node 0: 515GB, CPUs 0-31, 64-95
  - Node 1: 516GB, CPUs 32-63, 96-127  
  - Node 2: 129GB (memory-only)
- **Architecture**: x86_64 with modern multi-socket configuration

### **Performance Results**
- **CPU Count Detection**: 128 cores ✅
- **NUMA Node Detection**: 3 nodes ✅ (previously hardcoded to 1)
- **Dynamic Limits**: Supports up to 1024 NUMA nodes (extensible)

## 🔄 **Architecture Integration**

### **Existing GPU NUMA Support**
- **Maintains Compatibility**: All existing GPU NUMA functionality preserved
- **Unified Interface**: Consistent API patterns between GPU and CPU NUMA
- **Shared Infrastructure**: Leverages existing `c_ops` module architecture

### **LocalCPUBackend Enhancement**
- **Automatic NUMA Application**: CPU NUMA settings applied during backend initialization
- **Error Handling**: Graceful fallback when NUMA operations fail
- **Logging**: Comprehensive logging for debugging and monitoring

## 📝 **Usage Examples**

### **Configuration via Environment Variables**
```bash
export LMCACHE_CPU_AFFINITY="0-31,64-95"      # Bind to NUMA node 0 CPUs
export LMCACHE_NUMA_POLICY="bind"              # Bind memory to specific nodes
export LMCACHE_NUMA_NODES="0"                  # Use NUMA node 0
```

### **Configuration via LMCacheEngineConfig**
```python
config = LMCacheEngineConfig(
    cpu_affinity=[0, 1, 2, 3, 64, 65, 66, 67],  # Specific CPU cores
    cpu_numa_policy="preferred",                   # Memory policy
    cpu_numa_nodes=[0]                            # Preferred NUMA node
)
```

### **Programmatic Usage**
```python
from lmcache.c_ops import set_cpu_affinity, set_memory_policy

# Set CPU affinity to cores 0-31 (NUMA node 0)
set_cpu_affinity(list(range(32)), 32)

# Set memory policy to bind to NUMA node 0
set_memory_policy(2, [0], 1)  # 2 = MPOL_BIND
```

## 🧪 **Testing and Validation**

### **Unit Tests**
- C++ function compilation and linking ✅
- Python module import and function calls ✅
- Configuration schema validation ✅

### **Integration Tests**
- LocalCPUBackend NUMA integration ✅
- Dynamic system detection accuracy ✅
- Error handling and fallback mechanisms ✅

### **System Validation**
- **Actual Hardware**: 3 NUMA nodes, 128 CPU cores
- **NUMA Detection**: Accurate topology parsing from `/sys/devices/system/node/`
- **Performance**: No performance regression in existing functionality

## 🔍 **Code Quality**

### **Architecture Compliance**
- **PR #1409 Pattern**: Follows established GPU NUMA implementation pattern
- **C++ Integration**: Proper integration with existing `c_ops` module
- **Error Handling**: Comprehensive error handling and logging
- **Documentation**: Clear function documentation and usage examples

### **Maintainability**
- **Dynamic Detection**: No hardcoded system limits
- **Modular Design**: Clean separation of concerns
- **Extensible**: Easy to add new NUMA policies or CPU affinity methods

## 🚀 **Future Enhancements**

### **Planned Features**
- **Advanced CPU Affinity**: Support for CPU sets and complex affinity patterns
- **NUMA Balancing**: Automatic NUMA-aware load balancing
- **Performance Monitoring**: NUMA performance metrics and optimization
- **Multi-Node Support**: Enhanced support for large-scale NUMA systems

### **Integration Opportunities**
- **CacheBlend**: NUMA-aware cache placement for hybrid workloads
- **vLLM Integration**: Enhanced NUMA support for vLLM deployments
- **Distributed Systems**: NUMA-aware distributed cache coordination

## 📋 **Checklist**

- [x] **Code Quality**: Well-documented, follows LMCache coding standards
- [x] **Unit Tests**: Core functionality tested and validated
- [x] **Integration Tests**: Backend integration verified
- [x] **System Compatibility**: Tested on actual multi-NUMA hardware
- [x] **Performance**: No regression in existing functionality
- [x] **Documentation**: Comprehensive usage examples and API documentation
- [x] **Architecture**: Follows established LMCache patterns

## 🔗 **Related Links**

- **Issue**: [LMCache Q3 Roadmap #1253](https://github.com/LMCache/LMCache/issues/1253)
- **Related PR**: [PR #1409 - GPU NUMA functionality](https://github.com/LMCache/LMCache/pull/1409)
- **Architecture**: Follows existing `c_ops` module pattern

---

**This PR addresses the GPU NUMA-affinity placement enhancement from the Q3 roadmap while providing comprehensive CPU NUMA support for LMCache's growing multi-socket and NUMA-aware deployment scenarios.** 