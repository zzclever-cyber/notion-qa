# ============================================================
# 企业级 RAG Agent — Makefile 任务自动化
# ============================================================
.DEFAULT_GOAL := help

# 变量
PYTHON := python
PIP := pip
DOCKER := docker
COMPOSE := docker compose
UVICORN := uvicorn

# ============================================================
# 环境管理
# ============================================================

.PHONY: install
install:  ## 安装项目依赖
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev:  ## 安装开发依赖
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio pytest-cov pytest-benchmark httpx

.PHONY: env
env:  ## 从模板创建 .env 文件
	@test -f .env || cp .env.example .env
	@echo ".env 文件已准备就绪，请编辑填入实际配置"

# ============================================================
# 索引管理
# ============================================================

.PHONY: build-index
build-index:  ## 构建 FAISS + BM25 索引及评测数据集
	$(PYTHON) build_index.py

.PHONY: rebuild-index
rebuild-index:  ## 强制重建所有索引
	rm -rf data/faiss_index data/bm25_corpus.json data/eval_dataset.json
	$(PYTHON) build_index.py

# ============================================================
# 服务管理
# ============================================================

.PHONY: run
run:  ## 启动开发服务器（带热重载）
	$(UVICORN) main:app --host 0.0.0.0 --port 8000 --reload

.PHONY: serve
serve:  ## 启动生产服务器
	$(UVICORN) main:app --host 0.0.0.0 --port 8000 --workers 4

# ============================================================
# Docker 管理
# ============================================================

.PHONY: docker-build
docker-build:  ## 构建 Docker 镜像
	$(COMPOSE) build

.PHONY: docker-up
docker-up:  ## 启动所有 Docker 服务
	$(COMPOSE) up -d

.PHONY: docker-down
docker-down:  ## 停止所有 Docker 服务
	$(COMPOSE) down

.PHONY: docker-logs
docker-logs:  ## 查看 Docker 服务日志
	$(COMPOSE) logs -f

.PHONY: docker-restart
docker-restart:  ## 重启 Docker 服务
	$(COMPOSE) restart

.PHONY: docker-clean
docker-clean:  ## 停止服务并清理数据卷
	$(COMPOSE) down -v

# ============================================================
# 测试
# ============================================================

.PHONY: test
test:  ## 运行全部测试
	$(PYTHON) -m pytest tests/ -v

.PHONY: test-cov
test-cov:  ## 运行测试并生成覆盖率报告
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term

.PHONY: test-unit
test-unit:  ## 只运行单元测试
	$(PYTHON) -m pytest tests/ -v -m "unit"

.PHONY: test-integration
test-integration:  ## 只运行集成测试
	$(PYTHON) -m pytest tests/ -v -m "integration"

.PHONY: benchmark
benchmark:  ## 运行性能基准测试
	$(PYTHON) -m pytest tests/ -v -m "benchmark" --benchmark-only

# ============================================================
# 代码质量
# ============================================================

.PHONY: lint
lint:  ## 代码风格检查
	$(PYTHON) -m ruff check .

.PHONY: format
format:  ## 代码格式化
	$(PYTHON) -m ruff format .

.PHONY: typecheck
typecheck:  ## 类型检查
	$(PYTHON) -m mypy core/ retrieval/ generation/ --ignore-missing-imports

# ============================================================
# 工具
# ============================================================

.PHONY: clean
clean:  ## 清理临时文件
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage

.PHONY: shell
shell:  ## 进入 Python REPL
	$(PYTHON) -c "from main import RAGAgent; print('RAGAgent 已就绪')" && $(PYTHON)

.PHONY: help
help:  ## 显示此帮助信息
	@echo "企业级 RAG Agent — 可用命令："
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
