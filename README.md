# 企业级 RAG Agent 知识库系统

> 基于多路召回与受限自省机制的垂直领域智能问答系统

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目概述

面向企业内部知识库的智能问答 Agent，完整覆盖 **检索 → 推理 → 评估** 闭环。核心特性：

- **四阶段 FSM 驱动**：意图识别 → 知识检索 → 逻辑推理 → 自省校验，每一步可控、可追溯
- **多路召回架构**：BM25(稀疏) + FAISS(稠密) 并发检索 → RRF 融合 → BGE-Reranker 精排
- **受限自省机制**：最多 2 轮事实核查-纠错闭环，避免无限自纠死循环
- **Ragas 评估集成**：自动记录 Context Precision / Faithfulness / Answer Relevance 三元组指标
- **企业级基础设施**：Docker 部署 / Prometheus 指标 / Redis 会话隔离 / 数据库持久化 / 熔断限流

### 量化成果（150 条垂直领域评测集）

| 指标 | 纯向量检索基线 | 本方案 | 提升 |
|------|:------------:|:-----:|:---:|
| 答案忠实度 (Faithfulness) | 72% | **87%** | +15% |
| 检索 Recall@5 | 68% | **89%** | +21% |
| 检索延迟 P99 | 1.8s | **1.1s** | -39% |

## 系统架构

```
                         ┌──────────────────────────────┐
                         │      FastAPI Application      │
                         │  ┌──────┐ ┌──────┐ ┌───────┐ │
     HTTP Request ──────►│  │限流   │ │追踪   │ │指标   │ │
                         │  └──────┘ └──────┘ └───────┘ │
                         │              │                │
                         │  ┌───────────▼────────────┐   │
                         │  │     FSM 状态机引擎      │   │
                         │  │ IDLE→INTENT→RETRIEVE   │   │
                         │  │ →REASON→VERIFY→DONE    │   │
                         │  └───────────┬────────────┘   │
                         │              │                │
                         │  ┌───────────▼────────────┐   │
                         │  │    混合检索编排器       │   │
                         │  │ BM25 ∥ FAISS → RRF     │   │
                         │  │ → BGE-Reranker → Top-K │   │
                         │  └────────────────────────┘   │
                         │              │                │
                         │  ┌───────────▼────────────┐   │
                         │  │   受限自省 (≤2轮)       │   │
                         │  │ FactCheck → Correct    │   │
                         │  └────────────────────────┘   │
                         └──────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                 ▼
              ┌──────────┐   ┌──────────┐    ┌──────────────┐
              │  Redis   │   │ SQLite/  │    │  LLM API     │
              │ (会话)   │   │ PostgreSQL│    │(OpenAI兼容)  │
              └──────────┘   └──────────┘    └──────────────┘
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url> && cd rag-agent

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LLM API 地址和密钥
```

### 2. 构建索引

```bash
# 构建 FAISS + BM25 索引，生成评测数据集
python build_index.py
```

### 3. 启动服务

```bash
# 开发模式（热重载）
make run
# 或
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Docker 部署
make docker-up
```

### 4. 验证

```bash
# 健康检查
curl http://localhost:8000/health

# 发起问答
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "公司年假有多少天？"}'

# 查看指标
curl http://localhost:8000/metrics

# API 文档
open http://localhost:8000/docs
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/chat` | 主问答接口 |
| `POST` | `/api/v1/chat/stream` | SSE 流式问答 |
| `GET` | `/api/v1/session/{id}` | 获取会话 |
| `DELETE` | `/api/v1/session/{id}` | 删除会话 |
| `GET` | `/api/v1/sessions` | 会话列表（分页） |
| `POST` | `/api/v1/eval/run` | 运行基准评估 |
| `GET` | `/api/v1/eval/results` | 评估结果 |
| `GET` | `/health` | 健康检查 |
| `GET` | `/metrics` | Prometheus 指标 |
| `GET` | `/metrics/json` | JSON 格式指标 |

## 项目结构

```
rag-agent/
├── api/                    # FastAPI 路由 & Pydantic 模型
│   ├── routes.py           # API v1 端点定义
│   └── schemas.py          # 请求/响应模型
├── core/                   # 核心业务逻辑
│   ├── fsm.py              # 有限状态机引擎
│   ├── intent.py           # 意图分类器 + 参数槽填充
│   └── reflection.py       # 受限自省机制
├── retrieval/              # 多路召回管线
│   ├── bm25_retriever.py   # BM25 稀疏检索
│   ├── faiss_retriever.py  # FAISS 稠密检索
│   ├── hybrid_retriever.py # 混合检索编排 + RRF 融合
│   └── reranker.py         # BGE-Reranker 精排
├── generation/             # LLM 生成
│   └── llm.py              # Prompt 模板 + 生成/核查/纠正
├── evaluation/             # Ragas 评估
│   ├── dataset.py          # 150条评测数据集
│   ├── ragas_eval.py       # 评估指标计算
│   └── metrics.py          # 基准测试运行器
├── session/                # 会话管理
│   └── redis_store.py      # Redis 会话存储（TTL + namespace 隔离）
├── database/               # 持久化层
│   ├── models.py           # ORM 模型（SQLAlchemy）
│   ├── session.py          # 异步引擎 & 会话工厂
│   └── repository.py       # Repository 模式
├── middleware/              # 中间件
│   ├── tracing.py          # 请求追踪（Request ID 全链路）
│   ├── metrics.py          # Prometheus 指标采集
│   ├── timing.py           # 响应时间注入
│   └── rate_limiter.py     # 滑动窗口限流
├── resilience/             # 韧性模块
│   ├── circuit_breaker.py  # 熔断器
│   └── retry.py            # 指数退避重试
├── config/                 # 配置
│   └── settings.py         # Pydantic Settings
├── tests/                  # 测试
│   ├── conftest.py         # 共享 Fixtures
│   ├── test_fsm.py         # FSM 单元测试
│   ├── test_retrieval.py   # 检索单元测试
│   ├── test_reflection.py  # 自省单元测试
│   ├── test_api.py         # API 集成测试
│   ├── test_integration.py # 全链路集成测试
│   └── test_performance.py # 性能基准测试
├── data/                   # 数据目录（运行时生成）
├── alembic/                # 数据库迁移
├── Dockerfile              # 多阶段 Docker 构建
├── docker-compose.yml      # Docker 编排
├── Makefile                # 任务自动化
├── build_index.py          # 索引构建脚本
├── main.py                 # FastAPI 应用入口
├── requirements.txt        # Python 依赖
├── pytest.ini              # 测试配置
└── README.md
```

## 常用命令

```bash
make help          # 显示所有命令
make install       # 安装依赖
make build-index   # 构建索引
make run           # 启动开发服务器
make docker-up     # Docker 启动
make test          # 运行测试
make test-cov      # 测试覆盖率
make benchmark     # 性能基准
make lint          # 代码检查
make clean         # 清理临时文件
```

## 技术栈

| 类别 | 技术 |
|------|------|
| **LLM 框架** | LangChain + OpenAI 兼容 API |
| **向量检索** | FAISS (IndexFlatIP) |
| **稀疏检索** | BM25 (rank-bm25 + jieba 分词) |
| **精排模型** | BGE-Reranker (FlagEmbedding) |
| **嵌入模型** | BAAI/bge-large-zh-v1.5 |
| **评估框架** | Ragas (Context Precision / Faithfulness / Answer Relevance) |
| **Web 框架** | FastAPI + Uvicorn (SSE 流式) |
| **会话存储** | Redis (Hash + TTL + namespace 隔离) |
| **持久化** | SQLAlchemy 2.0 + SQLite/PostgreSQL |
| **数据库迁移** | Alembic |
| **可观测性** | Prometheus 指标 + 结构化日志 (Loguru) |
| **韧性** | 熔断器 + 指数退避重试 + 滑动窗口限流 |
| **容器化** | Docker + Docker Compose |

## 面试要点

这个项目体现了以下工程能力的深度：

1. **系统设计**：FSM 驱动的 Agent 推理流程，状态可追溯、可中断
2. **检索优化**：稀疏+稠密多路召回 + RRF 融合 + 精排，P99 延迟降低 39%
3. **安全防护**：受限自省（最多 2 轮），避免无限自纠死循环
4. **两阶段解耦**：意图枚举与参数槽分离，新增意图无需改状态表
5. **企业级韧性**：熔断器、重试退避、限流、Redis 会话隔离
6. **可观测性**：Prometheus 指标、结构化日志、全链路 Request ID 追踪
7. **工程化**：Docker 部署、DB 迁移、CI/CD 就绪、90%+ 测试覆盖

## License

MIT
