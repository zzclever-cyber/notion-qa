// 后端 API 客户端 —— 统一附加 JWT、错误处理、上传进度
import type {
  TokenResponse,
  User,
  KnowledgeBase,
  KnowledgeBaseDetail,
  DocumentItem,
  UsageStats,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const TOKEN_KEY = "notion_qa_token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}
export function setToken(token: string): void {
  if (typeof window !== "undefined") window.localStorage.setItem(TOKEN_KEY, token);
}
export function clearToken(): void {
  if (typeof window !== "undefined") window.localStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...(init.body ? { "Content-Type": "application/json" } : {}),
    ...((init.headers as Record<string, string>) || {}),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = `请求失败 (${res.status})`;
    try {
      const body = await res.json();
      detail = body?.detail || detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// ---------- 鉴权 ----------
export const authApi = {
  register: (email: string, password: string, display_name?: string) =>
    request<TokenResponse>("/api/v1/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password, display_name }),
    }),
  login: (email: string, password: string) =>
    request<TokenResponse>("/api/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  me: () => request<User>("/api/v1/auth/me"),
};

// ---------- 知识库 ----------
export const kbApi = {
  list: () => request<KnowledgeBase[]>("/api/v1/knowledge-bases"),
  create: (name: string, description?: string) =>
    request<KnowledgeBase>("/api/v1/knowledge-bases", {
      method: "POST",
      body: JSON.stringify({ name, description }),
    }),
  get: (id: string) =>
    request<KnowledgeBaseDetail>(`/api/v1/knowledge-bases/${id}`),
  remove: (id: string) =>
    request<{ status: string; id: string }>(`/api/v1/knowledge-bases/${id}`, {
      method: "DELETE",
    }),
};

// ---------- 文档 ----------
export const docApi = {
  remove: (id: string) =>
    request<{ status: string; id: string }>(`/api/v1/documents/${id}`, {
      method: "DELETE",
    }),
  /** 上传文件，走 XHR 以拿到上传进度 */
  upload: (
    kbId: string,
    file: File,
    onProgress?: (percent: number) => void
  ): Promise<DocumentItem> =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const form = new FormData();
      form.append("kb_id", kbId);
      form.append("file", file);
      xhr.open("POST", `${API_BASE}/api/v1/upload`);
      const token = getToken();
      if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText) as DocumentItem);
        } else {
          let detail = `上传失败 (${xhr.status})`;
          try {
            detail = JSON.parse(xhr.responseText)?.detail || detail;
          } catch {
            /* ignore */
          }
          reject(new ApiError(xhr.status, detail));
        }
      };
      xhr.onerror = () => reject(new ApiError(0, "网络错误，上传失败"));
      xhr.send(form);
    }),
};

// ---------- 用量 ----------
export const usageApi = {
  stats: () => request<UsageStats>("/api/v1/usage/stats"),
};
