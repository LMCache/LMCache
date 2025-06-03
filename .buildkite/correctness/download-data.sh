#!/bin/bash
set -euo pipefail

echo "📥 Downloading MMLU data..."

# Check if data already exists
if [ -d "data" ] && [ -d "data/test" ] && [ -d "data/dev" ]; then
    echo "✅ MMLU data already exists, skipping download"
    exit 0
fi

# Download the data
echo "🌐 Downloading data.tar..."
wget -q --show-progress https://people.eecs.berkeley.edu/~hendrycks/data.tar

# Extract the data
echo "📦 Extracting data.tar..."
tar xf data.tar

# Verify extraction
if [ ! -d "data" ]; then
    echo "❌ ERROR: data directory not created after extraction"
    exit 1
fi

if [ ! -d "data/test" ]; then
    echo "❌ ERROR: data/test directory not found"
    exit 1
fi

if [ ! -d "data/dev" ]; then
    echo "❌ ERROR: data/dev directory not found"
    exit 1
fi

# Count files to verify
test_files=$(find data/test -name "*.csv" | wc -l)
dev_files=$(find data/dev -name "*.csv" | wc -l)

echo "✅ MMLU data downloaded successfully"
echo "   - Test files: $test_files"
echo "   - Dev files: $dev_files"

# Clean up
rm -f data.tar

echo "🧹 Cleaned up data.tar"