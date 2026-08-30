# note-rag

对个人 Obsidian 笔记库的 RAG 问答机器人：用自然语言提问，从全库笔记中检索相关段落，由大模型生成**附出处**的回答。

> 为什么做这个：RAG 是 AI 应用落地的主流形态。通过一个天天使用的真实工具，完整实践「切块 → 向量化 → 检索 → 生成」各环节的工程取舍。

## 功能

**已实现（开发至第 4 步）**：

- 笔记库全库切块（标题感知切分，块带标题链上下文）
- BGE-M3 本地向量化（GPU 加速）+ Chroma 索引
- 语义检索：top-k + 相似度门槛，低于门槛判「没找到」
- 出处信息随块存储（来源文件 + 标题链）

**进行中**：

- [ ] DeepSeek 生成答案 + 出处编号（第 5 步）
- [ ] 网页界面（原型 B：检索卡式）+ 手机访问（第 6-7 步）
- [ ] 增量更新（第 8 步）

## 技术栈

- Python 3.11 + FastAPI（Web 服务，第 6 步起）
- BGE-M3 本地向量化模型（data/models/，魔搭下载，GPU/CPU 自适应）
- Chroma 向量库（data/chroma/，嵌入式）
- DeepSeek API（生成答案，key 存本地 .env）

## 如何运行

```bash
# 安装依赖（虚拟环境在项目内 .venv/）
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# 首次：从 ModelScope 下载 BGE-M3 到 data/models/bge-m3
./.venv/Scripts/python.exe -m modelscope download --model BAAI/bge-m3 --local-dir data/models/bge-m3

# 配置（config.yaml 已被 gitignore，复制示例后填写）
cp config.example.yaml config.yaml

# 常用命令
./.venv/Scripts/python.exe -m app.main config   # 查看配置
./.venv/Scripts/python.exe -m app.main stats    # 笔记与切块统计
./.venv/Scripts/python.exe -m app.main index    # 建索引
./.venv/Scripts/python.exe -m app.main search "问题"  # 检索
```

## 文档索引

| 文档 | 内容 |
|---|---|
| [01-需求记录.md](01-需求记录.md) | 需求与已确认决策 |
| [02-功能规划.md](02-功能规划.md) | 功能清单与 MVP 取舍 |
| [03-方案设计.md](03-方案设计.md) | 技术选型与架构 |
| [04-开发计划.md](04-开发计划.md) | 分步开发与验收标准 |

## 进度

- [x] 立项：需求记录
- [x] 功能规划
- [x] 方案设计
- [x] 分步开发：第 1-4 步（骨架/切块/索引/检索）
- [ ] 分步开发：第 5-9 步（生成/网页/手机/增量/打磨）
- [ ] 迭代打磨
