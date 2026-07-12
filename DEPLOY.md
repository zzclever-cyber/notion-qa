# 部署指南 — Notion-QA 全栈知识库

三段式部署：**前端 Vercel · 后端 Railway · 数据库 Neon(PostgreSQL)**。
本机没有 gh/vercel/railway CLI，因此下面用各平台网页控制台操作；每一步都标了「谁来做」。

---

## 0. 准备账号（你来做）
- [GitHub](https://github.com)（放代码，Vercel/Railway 从这里拉）
- [Neon](https://neon.tech)（免费 PostgreSQL）
- [Railway](https://railway.app)（跑后端 Docker）
- [Vercel](https://vercel.com)（跑 Next.js 前端）
- 可选：[Upstash](https://upstash.com)（免费 Redis；不配也能跑，会话降级到内存）

---

## 1. 推代码到 GitHub（你来做）
项目已在本地 `git init` 并提交（见仓库根）。在 GitHub 新建一个**空**仓库（不要勾 README），然后：

```bash
cd "E:/ai 全栈/RAG Agent"
git remote add origin https://github.com/<你的用户名>/notion-qa.git
git branch -M main
git push -u origin main
```

> `.env`、`.venv/`、`node_modules/`、上传文件、索引、个人笔记都已在 `.gitignore` 里，**密钥不会被推上去**。

---

## 2. Neon 数据库（你来做）
1. Neon 控制台 → New Project → 记下 **Connection string**（形如
   `postgresql://user:pw@ep-xxx.aws.neon.tech/neondb?sslmode=require`）。
2. **不用手动建表 / 不用跑 alembic**：后端启动时 `init_db()` 会 `create_all` 自动建全部表。
3. 这串 URL 直接粘给 Railway 的 `DATABASE_URL` 即可 —— 后端已内置规范化
   （自动转 `postgresql+asyncpg://`、剥离 `sslmode/channel_binding`、开启 SSL）。

---

## 3. 后端 → Railway（你来做，我可帮你核对）
1. Railway → New Project → **Deploy from GitHub repo** → 选你的仓库。
2. Railway 会自动识别根目录的 `Dockerfile` 并构建（`CMD` 已用 `$PORT`，无需改）。
3. **Variables** 里加环境变量：

   | 变量 | 值 |
   |---|---|
   | `DATABASE_URL` | 第 2 步的 Neon 连接串（原样粘贴） |
   | `JWT_SECRET` | 随机 48+ 位串：`python -c "import secrets;print(secrets.token_urlsafe(48))"` |
   | `LLM_API_BASE` | `https://api.deepseek.com` |
   | `LLM_API_KEY` | 你的 DeepSeek key |
   | `LLM_MODEL_NAME` | `deepseek-v4-pro` |
   | `EMBEDDING_API_BASE` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
   | `EMBEDDING_API_KEY` | 你的阿里 dashscope key |
   | `EMBEDDING_API_MODEL` | `text-embedding-v3` |
   | `REDIS_URL`（可选） | Upstash 的 redis URL；不填则用内存会话 |

4. Settings → Networking → **Generate Domain**，拿到公网地址，如
   `https://notion-qa-production.up.railway.app`。
5. 访问 `<域名>/api/v1/health` 应返回 `{"status":"ok",...}`。

> ⚠️ **索引持久化**：上传后的 FAISS/BM25 索引写在容器本地磁盘（`data/kb_indexes/`、`uploads/`），
> Railway 重新部署/重启会清空 → 需重新上传。要持久化就在 Railway 加一个 **Volume**，挂载到 `/app/data`
> （和 `/app/uploads`）。演示场景可以不挂，现场重传即可。

> 💡 镜像较大（含 torch，因 FlagEmbedding 精排依赖）。当前精排默认关闭，若想瘦身可在 requirements 里
> 去掉 `FlagEmbedding` 和 `sentence-transformers`（检索仍走 BM25+FAISS+RRF）。

---

## 4. 前端 → Vercel（你来做）
1. Vercel → Add New Project → 导入同一个 GitHub 仓库。
2. **Root Directory** 选 `frontend`（关键！仓库根是后端）。
3. Framework 自动识别 Next.js。**Environment Variables** 加：

   | 变量 | 值 |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | 第 3 步 Railway 的公网域名（不带结尾斜杠） |

4. Deploy → 拿到前端地址，如 `https://notion-qa.vercel.app`。
5. 打开它 → 注册/登录 → 建知识库 → 传 PDF → 问答。

> 前端不用 NextAuth/GitHub OAuth（当前是邮箱密码方案），所以**无需** `NEXTAUTH_*`、`GITHUB_*`。

---

## 5. 收尾校验（我可帮你跑）
- `GET <railway>/api/v1/health` → ok
- 前端注册 → 建库 → 传文档 → 状态 ready → 流式问答 + 引用 + 自省 badge
- CORS 已放开（`allow_origins=["*"]`，Bearer 鉴权无需 cookie），跨域直接可用

---

## 常见坑
- **DATABASE_URL**：直接粘 Neon 原串即可，后端会规范化；若手动写，用
  `postgresql+asyncpg://user:pw@host/db?ssl=require`（不要 `sslmode`）。
- **Vercel 拉到的是后端**：一定把 Root Directory 设成 `frontend`。
- **问答报 500 / 检索为空**：确认 Railway 的 LLM/EMBEDDING key 填对，且已上传文档且状态 ready。
- **重启后文档没了**：Railway 临时磁盘，见第 3 步的 Volume 说明。
