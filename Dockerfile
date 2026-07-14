# ============================================================
# 企业级 RAG Agent — 多阶段 Docker 构建
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（全局安装，多阶段构建兼容）
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/install -r requirements.txt

# ============================================================
# 运行阶段
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制已安装的包（全局目录，app 用户可读）
COPY --from=builder /install /usr/local/lib/python3.11/site-packages

# 复制应用代码
COPY . .

# 创建非 root 用户（后于 COPY，保证文件归属正确）
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# 健康检查（用 $PORT，兼容 Railway 注入端口）
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:'+os.getenv('PORT','8000')+'/api/v1/health')"

EXPOSE 8000

# Railway 等平台通过 $PORT 注入端口；本地默认 8000。用 shell 形式让 ${PORT} 展开
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
