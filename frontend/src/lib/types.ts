// 与后端 api/schemas.py 对齐的前端类型

export interface User {
  id: string;
  email: string;
  display_name: string;
  avatar_url: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export type DocStatus = "processing" | "ready" | "failed";

export interface DocumentItem {
  id: string;
  kb_id: string;
  filename: string;
  file_type: string;
  chunk_count: number;
  file_size_bytes: number;
  status: DocStatus;
  error_message: string;
  created_at: string | null;
}

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string;
  document_count: number;
  created_at: string | null;
}

export interface KnowledgeBaseDetail {
  id: string;
  name: string;
  description: string;
  created_at: string | null;
  documents: DocumentItem[];
}

export interface UsageStats {
  kb_count: number;
  document_count: number;
  chat_count: number;
  total_tokens: number;
}

// 流式检索来源（对应 SSE retrieve 事件的 sources）
export interface Source {
  doc_id: string;
  title: string;
  category: string;
  snippet: string;
  score: number;
}

// 自省结果（对应 SSE verify 事件）
export interface Reflection {
  consistent: boolean;
  rounds: number;
  contradictions: number;
}
