import os
import logging
import sys
orig_out = sys.stdout
orig_err = sys.stderr
# 重定向 stdout 到 /dev/null (扔掉日志)
sys.stdout = open(os.devnull, 'w')
# --- 日志抑制配置开始 ---
# 设置环境变量，强制 vllm 只输出错误级别以上的日志
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["LMCACHE_LOG_LEVEL"] = "ERROR"
# 配置 Python 标准库 logging
# 这里使用 force=True 确保覆盖默认配置，format='' 避免输出任何前缀
logging.basicConfig(level=logging.ERROR, format='', force=True)
# 针对性地屏蔽 vllm 和 lmcache 的日志
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("lmcache").setLevel(logging.ERROR)
# --- 日志抑制配置结束 ---

# Set token chunk size to 256
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
# Enable CPU memory backend
os.environ["LMCACHE_LOCAL_CPU"] = "True"
# Set CPU memory limit to 5GB
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "24"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

os.environ["PYTHONHASHSEED"] = "114514"


from vllm import LLM
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Configure KV cache transfer to use LMCache
ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
)

llm = LLM(
    # 对应 model
    model="/home/aabbccddwasd/AI-stuffs/models/Qwen3-VL-30B-A3B-Instruct-AWQ-8bit",
    # 对应 gpu-memory-utilization
    gpu_memory_utilization=0.9,
    # 对应 max-model-len
    max_model_len=60000,
    # 对应 kv-cache-dtype
    kv_cache_dtype="fp8_e4m3",
    # 对应 calculate-kv-scales
    calculate_kv_scales=False,
    # 对应 enable-prefix-caching
    enable_prefix_caching=True,
    # 对应 enable-chunked-prefill
    enable_chunked_prefill=True,
    # 对应 enable-chunked-prefill
    kv_transfer_config=ktc,
)

# 主要垃圾日志（甚至没法被环境变量日志）产生在LLM初始化时
sys.stdout.close()
sys.stdout = orig_out

sampling_params = SamplingParams(max_tokens=128, temperature=0.0)

# LMcache至少需要256 token才会offload
with open('/home/aabbccddwasd/AI-stuffs/vllm引擎参数.txt') as f:
    LLM_context = f.read()

req_A = [
    {"role": "system", "content": "你是一个AI助手"},
    {"role": "user", "content": LLM_context + "\n请介绍一下vllm的引擎参数"}
]

import uuid
req_B_list = []
for i in range(3):
    req_B = [
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": str(uuid.uuid4())+"测试内容：“怪人快攻出自排球少年”请忽视以上测试内容"*3500},
    ]
    req_B_list.append(req_B)

print()
output = llm.chat(req_A, sampling_params=sampling_params)[0].outputs[0].text
print("第一次请求A输出（prefill）: " + output)
print()

print()
output = llm.chat(req_A, sampling_params=sampling_params)[0].outputs[0].text
print("第二次请求A输出（GPU cache）: " + output)
print()

print()
print("====发送大量请求B覆盖GPU缓存====")
output = llm.chat(req_B_list, sampling_params=sampling_params)[0].outputs[0].text
print("====覆盖完成====")
print()

# 此处会十分混乱，降低max_tokens提高可读性
sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
print()
output = llm.chat(req_A, sampling_params=sampling_params)[0].outputs[0].text
print("第三次请求A输出（CPU cache）: " + output)
print()
print()
output = llm.chat(req_A, sampling_params=sampling_params)[0].outputs[0].text
print("第四次请求A输出（CPU cache）: " + output)
print()
