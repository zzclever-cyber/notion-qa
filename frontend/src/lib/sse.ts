// 流式问答 —— 用 fetch + ReadableStream 消费后端 SSE named events
// 后端事件流: intent → retrieve(含 sources) → generate_start → data{token} → verify_start → verify → done
import { API_BASE, getToken } from "./api";

/* eslint-disable @typescript-eslint/no-explicit-any */
export interface StreamHandlers {
  onEvent: (name: string, data: any) => void;
  onError?: (err: Error) => void;
  onDone?: () => void;
}

export interface StreamParams {
  query: string;
  kbId?: string;
  sessionId?: string;
  enableReflection?: boolean;
  signal?: AbortSignal;
}

export async function streamChat(
  params: StreamParams,
  handlers: StreamHandlers
): Promise<void> {
  const { query, kbId, sessionId, enableReflection = true, signal } = params;
  const token = getToken();
  const qs = kbId ? `?kb_id=${encodeURIComponent(kbId)}` : "";

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/v1/chat/stream${qs}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({
        query,
        session_id: sessionId,
        enable_reflection: enableReflection,
      }),
      signal,
    });
  } catch (e) {
    handlers.onError?.(e as Error);
    return;
  }

  if (!res.ok || !res.body) {
    handlers.onError?.(new Error(`流式请求失败 (${res.status})`));
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        parseBlock(block, handlers);
      }
    }
    if (buffer.trim()) parseBlock(buffer, handlers);
    handlers.onDone?.();
  } catch (e) {
    if ((e as Error).name !== "AbortError") handlers.onError?.(e as Error);
  }
}

function parseBlock(block: string, handlers: StreamHandlers) {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return;
  const raw = dataLines.join("\n");
  let data: any;
  try {
    data = JSON.parse(raw);
  } catch {
    data = { raw };
  }
  handlers.onEvent(event, data);
}
