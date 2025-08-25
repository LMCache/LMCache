#!/bin/bash

# Build script for LMCache with hugepage support
# This script compiles the C++ extensions and runs tests

set -e

echo "🚀 Building LMCache with hugepage support..."

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "❌ Error: Please run this script from the LMCache root directory"
    exit 1
fi

# Check if hugepage support is available
echo "📋 Checking hugepage support availability..."

if [ -d "/sys/kernel/mm/hugepages" ]; then
    echo "✅ Hugepage filesystem found"
    
    # Check for 2MB hugepages
    if [ -d "/sys/kernel/mm/hugepages/hugepages-2048kB" ]; then
        echo "✅ 2MB hugepages available"
        cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
        cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages
    fi
    
    # Check for 1GB hugepages
    if [ -d "/sys/kernel/mm/hugepages/hugepages-1048576kB" ]; then
        echo "✅ 1GB hugepages available"
        cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
        cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages
    fi
else
    echo "⚠️  Hugepage filesystem not found"
    echo "   To enable hugepages, you may need to:"
    echo "   1. Add 'hugepagesz=2M hugepages=1024' to /etc/default/grub"
    echo "   2. Reboot the system"
    echo "   3. Mount hugepages: mount -t hugetlbfs none /dev/hugepages"
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/
find . -name "*.so" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Build the extension
echo "🔨 Building C++ extension with hugepage support..."
python setup.py build_ext --inplace

# Check if the extension was built successfully
if [ -f "lmcache/c_ops.cpython-*.so" ]; then
    echo "✅ C++ extension built successfully"
else
    echo "❌ Failed to build C++ extension"
    exit 1
fi

# Install in development mode
echo "📦 Installing in development mode..."
pip install -e .

# Run basic tests
echo "🧪 Running basic tests..."

# Test hugepage availability check
python -c "
import lmcache.c_ops as lmc_ops
print('Testing hugepage support...')
print(f'Hugepages available: {lmcache.c_ops.is_hugepage_available()}')
if lmcache.c_ops.is_hugepage_available():
    print(f'Hugepage size: {lmcache.c_ops.get_hugepage_size() / (1024*1024):.1f} MB')
    print(f'Available count: {lmcache.c_ops.get_available_hugepage_count()}')
print('✅ Basic import test passed')
"

# Run the example script
echo "📚 Running example script..."
python examples/hugepage_usage.py

# Run pytest if available
if command -v pytest &> /dev/null; then
    echo "🧪 Running pytest tests..."
    pytest tests/test_hugepage_memory.py -v
else
    echo "⚠️  pytest not found, skipping automated tests"
fi

echo ""
echo "🎉 Build completed successfully!"
echo ""
echo "To use hugepage support in your code:"
echo "1. from lmcache.v1.hugepage_memory import create_hugepage_allocator"
echo "2. allocator = create_hugepage_allocator(size)"
echo "3. memory_obj = allocator.allocate(shape, dtype)"
echo "4. allocator.close()"
echo ""
echo "For more examples, see: examples/hugepage_usage.py" 