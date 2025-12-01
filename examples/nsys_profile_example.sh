#!/bin/bash
# 使用 nsys 对 LMCache 进行性能分析的示例脚本

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== LMCache nsys Profiling 示例 ===${NC}\n"

# 检查 nsys 是否安装
if ! command -v nsys &> /dev/null; then
    echo -e "${YELLOW}警告: nsys 未安装${NC}"
    echo "请安装 NVIDIA Nsight Systems:"
    echo "  conda install -c nvidia nsight-systems"
    echo "  或从 https://developer.nvidia.com/nsight-systems 下载"
    exit 1
fi

# 输出文件名
OUTPUT_FILE="lmcache_profile_$(date +%Y%m%d_%H%M%S).nsys-rep"

echo -e "${GREEN}开始 profiling...${NC}"
echo "输出文件: ${OUTPUT_FILE}"
echo ""

# 基本 profiling 命令
# 你可以根据实际需求修改参数

nsys profile \
    --output="${OUTPUT_FILE}" \
    --trace=cuda,nvtx,osrt \
    --gpu-metrics-device=0 \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --duration=60 \
    "$@"

echo ""
echo -e "${GREEN}Profiling 完成!${NC}"
echo ""
echo "查看结果的方式:"
echo "  1. GUI 方式: nsys-ui ${OUTPUT_FILE}"
echo "  2. 命令行统计: nsys stats --report nvtx ${OUTPUT_FILE}"
echo "  3. 导出 CSV: nsys export --type=csv --output=profile.csv ${OUTPUT_FILE}"
echo ""

# 可选：自动打开 GUI（如果可用）
if command -v nsys-ui &> /dev/null; then
    read -p "是否现在打开 nsys-ui 查看结果? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        nsys-ui "${OUTPUT_FILE}" &
    fi
fi

