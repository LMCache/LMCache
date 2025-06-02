#!/bin/bash

echo "🧹 Cleaning up test environment..."

# Kill any running Docker containers
echo "🐳 Stopping Docker containers..."
sudo docker ps -q | xargs -r sudo docker kill || true

# Clean up Docker system
echo "🗑️ Cleaning Docker system..."
sudo docker system prune -f || true

# Remove test results (optional - comment out if you want to keep them)
# echo "📁 Removing test results..."
# rm -rf mmlu-results/ compare-results/ data/ || true

echo "✅ Cleanup completed!"