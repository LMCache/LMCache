import os
import sys

# ==========================================
# 1. 环境变量配置
# ==========================================

# 【关键修改】不要禁用 tqdm，否则 vllm 计算速度时会除零崩溃
# os.environ["TQDM_DISABLE"] = "1"  <-- 删除这行
# 设置 transformers 和 vllm 的日志级别

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["LMCACHE_LOG_LEVEL"] = "ERROR"

# 其他环境变量
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "24"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTHONHASHSEED"] = "114514"

# 导入必要的库
import logging

logging.basicConfig(level=logging.ERROR, format='')

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


# ==========================================
# 2. 修复后的重定向逻辑 (核心修改)
# ==========================================
class OutputSilencer:
    def __init__(self):
        self.saved_stdout_fd = None
        self.saved_stderr_fd = None
        self.devnull_fd = None

    def silence(self):
        # 1. 备份原始的 FD 1 (stdout) 和 FD 2 (stderr)
        # os.dup() 会返回一个新的未使用的 FD 编号 (如 3, 4)
        self.saved_stdout_fd = os.dup(1)
        self.saved_stderr_fd = os.dup(2)
        # 2. 打开 /dev/null
        self.devnull_fd = os.open(os.devnull, os.O_RDWR)
        # 3. 重定向 FD 1 和 FD 2 指向 /dev/null
        os.dup2(self.devnull_fd, 1)
        os.dup2(self.devnull_fd, 2)
        # 4. 同步 Python 层的对象，防止 print 报错
        sys.stdout = open(1, 'w')
        sys.stderr = open(2, 'w')

    def restore(self):
        if self.saved_stdout_fd is None:
            return
        # 1. 关闭当前的 /dev/null FD 引用 (可选，dup2会自动关闭目标)

        # 2. 将备份的 FD (3, 4) 复制回 FD 1 和 FD 2
        os.dup2(self.saved_stdout_fd, 1)
        os.dup2(self.saved_stderr_fd, 2)
        # 3. 关闭备份的临时 FD (释放资源)
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)
        # 4. 恢复 Python 的 sys 对象引用
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # 重置状态
        self.saved_stdout_fd = None


# ==========================================
# 3. 执行逻辑
# ==========================================
silencer = OutputSilencer()
print(">>> [INFO] 开始静默初始化 LLM...")
# 开始静默
silencer.silence()

try:
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    llm = LLM(
        model="/home/aabbccddwasd/AI-stuffs/models/Qwen3-VL-30B-A3B-Instruct-AWQ-8bit",
        gpu_memory_utilization=0.9,
        max_model_len=60000,
        kv_cache_dtype="fp8_e4m3",
        calculate_kv_scales=False,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        kv_transfer_config=ktc,
    )
finally:
    # 无论初始化成功与否，都恢复输出，否则你看不到报错信息
    silencer.restore()

print(">>> [SUCCESS] LLM 初始化完成，输出已恢复！")
print()

# --- 后续推理代码 ---

sampling_params = SamplingParams(max_tokens=128, temperature=0.0)

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
        {"role": "user", "content": str(uuid.uuid4()) + "测试内容：“怪人快攻出自排球少年”请忽视以上测试内容" * 3500},
    ]
    req_B_list.append(req_B)

print("正在发送请求 A (Prefill)...")
output = llm.chat(req_A, sampling_params=sampling_params)[0].outputs[0].text
print("第一次请求A输出（prefill）: " + output)
print()

print("正在发送请求 A (GPU Cache)...")
output = llm.chat(req_A, sampling_params=sampling_params)[0].outputs[0].text
print("第二次请求A输出（GPU cache）: " + output)
print()

print("====发送大量请求B覆盖GPU缓存====")
output = llm.chat(req_B_list, sampling_params=sampling_params)[0].outputs[0].text
print("====覆盖完成====")
print()

sampling_params = SamplingParams(max_tokens=64, temperature=0.0)

print("正在发送请求 A (CPU Cache - 第1次)...")
output = llm.chat(req_A, sampling_params=sampling_params)[0].outputs[0].text
print("第三次请求A输出（CPU cache）: " + output)
print()

print("正在发送请求 A (CPU Cache - 第2次)...")
output = llm.chat(req_A, sampling_params=sampling_params)[0].outputs[0].text
print("第四次请求A输出（CPU cache）: " + output)
print()
