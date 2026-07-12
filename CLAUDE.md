# Notion-QA: AI 知识库 SaaS 平台 — 全栈改造

## 改造目标

将当前纯后端的 RAG Agent 系统改造成一个**完整可演示的 AI 知识库 SaaS 产品**，新增 Next.js 前端 + 鉴权 + 文件上传管线。
改造后定位：打"全栈基本功"——证明能独立把一个 AI 产品从零搭到上线。

## 核心原则

- **每个功能只做到"能演示 + 能讲清"**，不过度追求商业完整度
- **不做 Stripe 支付、不做多租户团队管理、不做 tRPC**
- **复用现有 AI 推理链路（FSM + 三级检索 + 自省），不重写后端核心**

---

## 一、保留的现有代码（不动）

以下模块直接复用，不要重写：
- `core/fsm.py` — FSM 推理引擎（6 状态 + 显式转移表）
- `core/intent.py` — 意图分类 + 参数槽填充
- `core/reflection.py` — 受限自省机制
- `retrieval/hybrid_retriever.py` — BM25 + FAISS + Reranker 三级检索
- `retrieval/bm25_retriever.py` / `faiss_retriever.py` / `reranker.py`
- `generation/llm.py` — LLM 生成器
- `resilience/circuit_breaker.py` — 熔断器
- `middleware/tracing.py` / `metrics.py` / `rate_limiter.py`
- `session/redis_store.py` — Redis 会话存储
- `database/` — SQLAlchemy 异步 ORM + Alembic
- `evaluation/ragas_eval.py` — 评估框架
- `config/settings.py` — 配置系统
- `Dockerfile` / `docker-compose.yml` — 容器化部署

## 二、需要改造的后端部分

### 2.1 新增 API 路由（`api/` 目录下扩展）

```
POST   /api/v1/auth/register        → 用户注册（email + password）
POST   /api/v1/auth/login           → 用户登录，返回 JWT access_token
GET    /api/v1/auth/me              → 获取当前用户信息
POST   /api/v1/upload               → 上传 PDF/Word/Markdown，自动切片+建索引
GET    /api/v1/knowledge-bases      → 列出用户的知识库
POST   /api/v1/knowledge-bases      → 创建知识库
GET    /api/v1/knowledge-bases/{id} → 知识库详情（含文档列表）
DELETE /api/v1/knowledge-bases/{id} → 删除知识库
DELETE /api/v1/documents/{id}       → 删除文档 + 从索引中移除
POST   /api/v1/chat                 → RAG 问答（已有，需关联 kb_id + user_id）
POST   /api/v1/chat/stream          → SSE 流式问答（已有，需关联 kb_id + user_id）
GET    /api/v1/usage/stats          → 用户用量统计（文档数、问答次数、token 消耗）
```

### 2.2 新增数据库表（Alembic 迁移）

在 `database/models.py` 中新增：

```python
# 用户表
class User(Base):
    id: UUID
    email: str (unique)
    hashed_password: str
    display_name: str
    created_at: datetime

# 知识库表
class KnowledgeBase(Base):
    id: UUID
    user_id: FK → User
    name: str
    description: str
    created_at: datetime

# 文档表
class Document(Base):
    id: UUID
    kb_id: FK → KnowledgeBase
    filename: str
    file_type: str  # pdf / docx / md / txt
    chunk_count: int
    file_size_bytes: int
    status: str  # processing / ready / failed
    created_at: datetime
```

### 2.3 新增文件处理模块（新建 `processing/` 目录）

```
processing/
├── __init__.py
├── parser.py        → PyMuPDF (fitz) 解析 PDF + python-docx 解析 Word + markdown
├── chunker.py       → 句级切片（复用现有逻辑，保证句边界对齐）
└── indexer.py       → 文档切片 → embedding → FAISS 索引增量更新
```

### 2.4 新增 JWT 鉴权中间件（`middleware/auth.py`）

```python
# 从 Authorization: Bearer <token> 中提取 user_id
# 注入到 request.state.user_id
# 所有 /api/v1/ 路由（除 auth/* 外）都需要验证
```

### 2.5 改造现有 ChatRecord 表

在现有 `ChatRecord` 表上新增字段：`user_id` (FK → User)、`kb_id` (FK → KnowledgeBase)。

---

## 三、前端（Next.js 14 项目）

在当前项目根目录下新建 `frontend/` 目录，用 `create-next-app` 初始化。

### 3.1 技术栈

```
Next.js 14 App Router + TypeScript + Tailwind CSS
shadcn/ui (组件库) + Lucide React (图标)
Vercel AI SDK (ai + @ai-sdk/react) — Chat 页流式对话（★ 全栈岗高频技术点）
React Query (TanStack Query) — Dashboard 页面数据请求
next-auth v5 — GitHub OAuth 鉴权
Recharts — 用量统计图表
```

> **为什么用 Vercel AI SDK 而不是手写 SSE？**
> AI SDK 的 `useChat` hook 封装了 EventSource 生命周期管理、断线重连、消息解析、loading/error 状态。
> 但注意：后端是 FastAPI 而非 Next.js API Route，所以需要在 FastAPI 端
> 把 SSE 输出格式对齐 AI SDK 的数据流协议（见 3.4 节）。

### 3.2 页面结构（3 个核心页面 + 1 个 Layout）

```
Layout (Navbar + UserMenu + Sidebar)
│
├── /login
│   └── GitHub OAuth 登录按钮（NextAuth.js）
│
├── /dashboard
│   ├── 知识库卡片列表（名称、文档数、创建时间）
│   ├── "创建知识库" 按钮 → Dialog 弹窗
│   ├── 每个卡片可点击进入 → /kb/[id]
│   └── 底部：用量统计概览（总文档数、总问答次数、Token 消耗）
│
├── /kb/[id]
│   ├── 文档列表（表格：文件名、类型、切块数、状态、上传时间）
│   ├── 上传按钮 → UploadDialog（拖拽上传 + 进度条）
│   ├── 每行文档可删除
│   └── "开始对话" 按钮 → 跳转 /kb/[id]/chat
│
└── /kb/[id]/chat
    ├── ChatMessages（对话气泡：用户问题 + AI 回答）
    │   └── AI 回答中：引用来源卡片（点击可展开查看原文）
    │   └── 自省结果标记（绿色/黄色/红色 badge）
    ├── ChatInput（底部 textarea，自动增高，Enter 发送）
    └── 流式效果：AI 回答逐 token 出现（SSE EventSource）
```

### 3.3 前端关键交互细节

**Chat 页面的流式效果（Vercel AI SDK 方案）：**

```tsx
// frontend/src/app/kb/[id]/chat/page.tsx
'use client';

import { useChat } from '@ai-sdk/react';

export default function ChatPage({ params }: { params: { id: string } }) {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: `${process.env.NEXT_PUBLIC_API_URL}/api/v1/chat/stream?kb_id=${params.id}`,
    // headers: { Authorization: `Bearer ${token}` },  // 鉴权
  });

  return (
    <div className="flex flex-col h-screen">
      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map(m => (
          <div key={m.id} className={m.role === 'user' ? 'text-right' : 'text-left'}>
            <div className="inline-block max-w-[80%] p-3 rounded-lg">
              {m.content}
            </div>
            {/* annotations 里放引用来源 + 自省结果 */}
            {m.annotations?.map((a: any, i: number) =>
              a.type === 'citation' ? <CitationCard key={i} source={a} /> :
              a.type === 'reflection' ? <ReflectionBadge key={i} result={a} /> : null
            )}
          </div>
        ))}
      </div>

      {/* 输入框 */}
      <form onSubmit={handleSubmit} className="p-4 border-t">
        <input
          value={input}
          onChange={handleInputChange}
          placeholder="输入问题..."
          disabled={isLoading}
        />
      </form>
    </div>
  );
}
```

**后端 FastAPI SSE 格式对齐：**

AI SDK 的 `useChat` 期望标准 SSE 流，每条消息格式为：
```
data: {"choices":[{"delta":{"content":"token文本"}}]}
```

但由于你的后端已有自定义 named events（intent/retrieve/verify/done），直接让 FastAPI 对齐 AI SDK 格式会丢失这些中间状态。有两种做法：

- **方案 A（推荐）**：Chat 页用原生 `EventSource` 处理完整的 named events（意图/检索/生成/自省），只在生成阶段逐 token 渲染。这样能展示完整的 RAG 流程，Demo 更直观。
- **方案 B**：后端新增一个 AI SDK 兼容端点 `/api/v1/chat/ai-sdk`，只返回标准 SSE token 流。Dashboard 的聊天用这个，方便集成。

**建议两个都做**：Chat 页面默认用原生 EventSource 展示完整流程（演示用），同时在代码里注释说明"生产可切换 AI SDK"。面试时主动说："我理解 AI SDK 是 Next.js 生态的标准做法，代码里已预留了切换接口。"

**文件上传：**
1. 前端：react-dropzone 拖拽区域 + 上传进度条（axios onUploadProgress）
2. 后端：接收文件 → 存入 `uploads/` → 后台任务解析+切片+建索引 → 更新文档状态

**鉴权流程（NextAuth.js ↔ FastAPI JWT 桥接）：**

这是一个关键设计决策。NextAuth.js v5（Auth.js）默认用自己签发的 JWT（存在 cookie 里），FastAPI 后端用 PyJWT 签发自己的 JWT。两种 token 不互通。桥接方案：

```
用户点击 "Sign in with GitHub"
    │
    ▼
NextAuth.js (前端 /api/auth/[...nextauth])
    │ 1. GitHub OAuth 跳转 → 用户授权 → 回调
    │ 2. NextAuth 拿到 GitHub access_token + 用户信息
    │ 3. 在 NextAuth 的 jwt callback 中：
    │    调用后端 POST /api/v1/auth/oauth-callback
    │    传入 { github_token, github_user }
    ▼
FastAPI /api/v1/auth/oauth-callback
    │ 1. 用 GitHub token 验证身份
    │ 2. 查找或创建 User 记录
    │ 3. 用 PyJWT 签发 access_token（后端自己的 JWT）
    │ 4. 返回 { backend_token }
    ▼
NextAuth jwt callback
    │ 将 backend_token 存入 NextAuth JWT 的 backendToken 字段
    ▼
前端 API 请求
    │ 从 NextAuth session 中读取 backendToken
    │ 设置 Authorization: Bearer <backendToken>
    │ → 中间件 backend/middleware/auth.py 验证
```

**代码实现要点：**
```typescript
// frontend/src/auth.ts — NextAuth 配置
export const { handlers, auth, signIn } = NextAuth({
  callbacks: {
    async jwt({ token, account, profile }) {
      if (account?.access_token) {
        // GitHub 登录成功 → 调用后端换取 backend JWT
        const res = await fetch(`${BACKEND_URL}/api/v1/auth/oauth-callback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            github_token: account.access_token,
            github_user: {
              login: profile?.login,
              email: profile?.email,
              avatar_url: profile?.avatar_url,
            },
          }),
        });
        const { backend_token } = await res.json();
        token.backendToken = backend_token;
      }
      return token;
    },
    async session({ session, token }) {
      session.backendToken = token.backendToken as string;
      return session;
    },
  },
});
```

```python
# backend/api/auth.py — 新增 OAuth 回调端点
@router.post("/auth/oauth-callback")
async def oauth_callback(data: OAuthCallbackRequest):
    # 1. 用 GitHub token 调 GitHub API 验证用户身份
    # 2. 查找或创建 User → 从 github_login 匹配
    # 3. 签发 JWT access_token（含 user_id, exp）
    # 4. 返回 { backend_token: str }
```

**面试话术：**
> "鉴权我用了两层 token。NextAuth 管理前端 session（存在 httpOnly cookie 里），后端自己签发 JWT 做 API 鉴权。登录时 NextAuth 的 jwt callback 调后端 OAuth 回调接口，用 GitHub token 换后端 JWT，后端 JWT 存到 session 里。这样前端 session 和后端 API 鉴权是解耦的——以后换 Google OAuth 或加 API Key 鉴权不需要改 NextAuth 层。"

### 3.4 前端项目初始化命令

```bash
cd frontend/
npx create-next-app@latest . --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"
npx shadcn@latest init
npx shadcn@latest add button input card dialog table textarea badge dropdown-menu avatar
npm install @tanstack/react-query next-auth@beta recharts react-dropzone lucide-react
npm install ai @ai-sdk/react @ai-sdk/openai  # Vercel AI SDK
npm install openapi-typescript  # 从 FastAPI /openapi.json 自动生成前端类型
```

---

## 四、部署

### 前端部署（Vercel）
1. GitHub 仓库连接 Vercel
2. 环境变量：`NEXTAUTH_URL`、`NEXTAUTH_SECRET`、`GITHUB_CLIENT_ID`、`GITHUB_CLIENT_SECRET`、`NEXT_PUBLIC_API_URL`（后端地址）

### 后端部署（Railway）
1. 推送 `main.py` + `Dockerfile` 到 Railway
2. 环境变量：`DATABASE_URL`（Neon）、`REDIS_URL`（Upstash）、`JWT_SECRET`、`DEEPSEEK_API_KEY`

### 数据库（Neon PostgreSQL）
- 三个项目共用一个 Neon 实例（免费 0.5GB），用不同的 schema 隔离

---

## 五、验收标准

改完后，以下流程必须能从头跑到尾：
1. 用户用 GitHub 账号登录
2. 创建一个知识库 "公司制度"
3. 上传 1 个 PDF 文件（比如 12 篇企业文档的 PDF）
4. 等待索引完成（文档状态变为 ready）
5. 在 Chat 页面提问 "年假怎么申请？"
6. 看到逐 token 流式输出回答
7. 回答下方显示引用了哪些文档片段
8. 看到自省 badge（绿色"已核验"）

完成后录制 2 分钟 Demo 视频。

---

## 六、开发顺序（按这个来）

1. **后端改造**：新增数据库表 → 鉴权中间件 → 文件处理模块 → API 路由
2. **前端搭建**：Next.js 初始化 → shadcn/ui 配置 → Layout + 登录页 → Dashboard → 知识库详情页 → Chat 页
3. **前后端联调**：鉴权 → 上传 → 问答 → 流式
4. **部署**：Vercel（前端）+ Railway（后端）+ Neon（数据库）
