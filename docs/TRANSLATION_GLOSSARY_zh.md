# LMCache 文档翻译术语表

本术语表旨在帮助翻译者保持翻译的一致性。

## 📌 核心概念

| 英文 | 中文 | 说明 |
|------|------|------|
| LMCache | LMCache | 项目名称，不翻译 |
| vLLM | vLLM | 项目名称，不翻译 |
| KV Cache | KV 缓存 | 键值缓存 |
| Cache | 缓存 | |
| Prefill | 预填充 | |
| Decode | 解码 | |
| Token | Token / 词元 | 根据上下文选择 |
| Prompt | 提示词 | |
| Inference | 推理 | |
| Serving | 服务 | |
| Backend | 后端 | |
| Storage Backend | 存储后端 | |
| Chunk | 块 / 分块 | 根据上下文 |

## 🏗️ 架构相关

| 英文 | 中文 | 说明 |
|------|------|------|
| Disaggregated Prefill | 分离式预填充 | |
| Offload | 卸载 | |
| Sharing | 共享 | |
| Compression | 压缩 | |
| Blending | 融合 | |
| Layer-wise | 分层 | |
| Encoder | 编码器 | |
| Decoder | 解码器 | |
| Connector | 连接器 | |
| Plugin | 插件 | |
| Endpoint | 端点 / 接口 | 根据上下文 |

## 💾 存储相关

| 英文 | 中文 | 说明 |
|------|------|------|
| Redis | Redis | 不翻译 |
| Local Storage | 本地存储 | |
| CPU RAM | CPU 内存 | |
| GPU RAM | GPU 显存 | |
| S3 | S3 | AWS服务，不翻译 |
| Remote Storage | 远程存储 | |
| Distributed Storage | 分布式存储 | |

## 🔧 技术术语

| 英文 | 中文 | 说明 |
|------|------|------|
| Latency | 延迟 | |
| Throughput | 吞吐量 | |
| TTFT (Time To First Token) | 首个词元时间 / TTFT | 可保留缩写 |
| Batch | 批次 | |
| Request | 请求 | |
| Query | 查询 | |
| API | API | 不翻译 |
| Configuration | 配置 | |
| Deployment | 部署 | |
| Production | 生产环境 | |
| Docker | Docker | 不翻译 |
| Kubernetes | Kubernetes | 不翻译，可简称 K8s |

## 📊 性能指标

| 英文 | 中文 | 说明 |
|------|------|------|
| Performance | 性能 | |
| Throughput | 吞吐量 | |
| Latency | 延迟 | |
| Benchmark | 基准测试 | |
| Metric | 指标 | |
| Observability | 可观测性 | |
| Monitoring | 监控 | |
| Profiling | 性能分析 | |

## 🛠️ 开发相关

| 英文 | 中文 | 说明 |
|------|------|------|
| Developer Guide | 开发者指南 | |
| API Reference | API 参考 | |
| Getting Started | 快速开始 / 入门指南 | |
| Installation | 安装 | |
| Quickstart | 快速开始 | |
| Tutorial | 教程 | |
| Example | 示例 | |
| Use Case | 使用场景 | |
| Troubleshooting | 故障排除 / 问题排查 | |
| FAQ | 常见问题 / FAQ | |

## 📝 文档相关

| 英文 | 中文 | 说明 |
|------|------|------|
| Documentation | 文档 | |
| Note | 注意 | |
| Warning | 警告 | |
| Tip | 提示 | |
| Example | 示例 | |
| Code Block | 代码块 | |
| Figure | 图 | |
| Table | 表格 | |

## 🔤 常用动词

| 英文 | 中文 | 说明 |
|------|------|------|
| Install | 安装 | |
| Configure | 配置 | |
| Deploy | 部署 | |
| Run | 运行 | |
| Execute | 执行 | |
| Build | 构建 | |
| Test | 测试 | |
| Debug | 调试 | |
| Enable | 启用 | |
| Disable | 禁用 | |
| Initialize | 初始化 | |
| Set up | 设置 / 配置 | |

## 🌟 特殊说明

### 不翻译的内容

1. **命令和代码**
   ```bash
   # 保持原样
   pip install lmcache
   make html
   ```

2. **配置参数名**
   ```python
   # 参数名不翻译
   lm_config = LMCacheEngineConfig(...)
   ```

3. **文件名和路径**
   ```
   # 保持原样
   /path/to/config.yaml
   requirements.txt
   ```

4. **技术缩写**
   - API, CLI, SDK, HTTP, REST
   - GPU, CPU, RAM, SSD
   - JSON, YAML, XML

### 翻译原则

1. **一致性**: 同一术语在整个文档中使用相同翻译
2. **准确性**: 优先使用行业标准译法
3. **可读性**: 在准确的前提下，使用通俗易懂的表达
4. **保留性**: 重要术语可采用"中文（English）"的方式

### 示例

✅ 好的翻译：
```
"KV Cache compression reduces memory usage"
→ "KV 缓存压缩可以减少内存使用"
```

❌ 不好的翻译：
```
"KV Cache compression reduces memory usage"
→ "键值缓存压缩减少记忆体使用" （使用了不常见的"记忆体"）
```

## 🔄 更新此术语表

如果你发现需要添加新术语或修改现有翻译，请：

1. 提交 Issue 讨论
2. 在 Pull Request 中一并更新此文件
3. 确保团队达成共识

## 参考资源

- [Microsoft 术语库](https://www.microsoft.com/zh-cn/language)
- [Google 开发者文档风格指南](https://developers.google.com/style)
- [中文技术文档写作风格指南](https://github.com/yikeke/zh-style-guide) 