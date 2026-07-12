"use client";
import { useState, useRef, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  Send,
  Bot,
  ChevronDown,
  ShieldCheck,
  ShieldAlert,
  ShieldQuestion,
  FileText,
} from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { streamChat } from "@/lib/sse";
import { kbApi } from "@/lib/api";
import type { Source, Reflection } from "@/lib/types";

interface Msg {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  reflection?: Reflection;
  stage?: string; // retrieving | generating | verifying | done
  error?: string;
}

const STAGE_LABEL: Record<string, string> = {
  intent: "识别意图…",
  retrieving: "检索知识库…",
  generating: "生成回答…",
  verifying: "事实核查…",
};

function ReflectionBadge({ r }: { r: Reflection }) {
  if (r.consistent)
    return (
      <Badge tone="green">
        <ShieldCheck size={12} />
        已核验一致
      </Badge>
    );
  if (r.contradictions > 0)
    return (
      <Badge tone="red">
        <ShieldAlert size={12} />
        发现 {r.contradictions} 处冲突
      </Badge>
    );
  return (
    <Badge tone="yellow">
      <ShieldQuestion size={12} />
      部分信息存疑
    </Badge>
  );
}

function Citations({ sources }: { sources: Source[] }) {
  const [openIdx, setOpenIdx] = useState<number | null>(null);
  return (
    <div className="mt-3 space-y-1.5">
      <div className="text-xs font-medium text-slate-400">
        引用来源 · {sources.length}
      </div>
      {sources.map((s, i) => (
        <div key={i} className="overflow-hidden rounded-md border border-slate-200 bg-slate-50 text-xs">
          <button
            onClick={() => setOpenIdx(openIdx === i ? null : i)}
            className="flex w-full items-center justify-between px-2.5 py-1.5 text-left"
          >
            <span className="flex min-w-0 items-center gap-1.5 text-slate-700">
              <FileText size={12} className="shrink-0" />
              <span className="truncate">{s.title || s.doc_id}</span>
              {s.category && <span className="shrink-0 text-slate-400">· {s.category}</span>}
            </span>
            <span className="flex shrink-0 items-center gap-1 text-slate-400">
              相关度 {s.score}
              <ChevronDown size={12} className={openIdx === i ? "rotate-180" : ""} />
            </span>
          </button>
          {openIdx === i && (
            <div className="border-t border-slate-200 px-2.5 py-2 leading-relaxed text-slate-600">
              {s.snippet}
              {s.snippet.length >= 200 ? "…" : ""}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function ChatInner({ id }: { id: string }) {
  const kbQ = useQuery({ queryKey: ["kb", id], queryFn: () => kbApi.get(id) });
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const sessionRef = useRef<string | undefined>(undefined);
  const endRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /* eslint-disable @typescript-eslint/no-explicit-any */
  const patchLast = useCallback((fn: (m: Msg) => Msg) => {
    setMessages((prev) => {
      if (!prev.length) return prev;
      const next = prev.slice();
      next[next.length - 1] = fn(next[next.length - 1]);
      return next;
    });
  }, []);

  const send = useCallback(() => {
    const q = input.trim();
    if (!q || streaming) return;
    setInput("");
    if (taRef.current) taRef.current.style.height = "auto";
    setMessages((prev) => [
      ...prev,
      { role: "user", content: q },
      { role: "assistant", content: "", stage: "intent" },
    ]);
    setStreaming(true);

    streamChat(
      { query: q, kbId: id, sessionId: sessionRef.current, enableReflection: true },
      {
        onEvent: (name, data: any) => {
          if (name === "intent") {
            patchLast((m) => ({ ...m, stage: "retrieving" }));
          } else if (name === "retrieve") {
            patchLast((m) => ({ ...m, sources: data.sources || [], stage: "generating" }));
          } else if (name === "generate_start") {
            patchLast((m) => ({ ...m, stage: "generating" }));
          } else if (name === "message" && typeof data.token === "string") {
            patchLast((m) => ({ ...m, content: m.content + data.token, stage: "generating" }));
          } else if (name === "verify_start") {
            patchLast((m) => ({ ...m, stage: "verifying" }));
          } else if (name === "verify") {
            patchLast((m) => ({
              ...m,
              reflection: {
                consistent: !!data.consistent,
                rounds: data.rounds ?? 0,
                contradictions: data.contradictions ?? 0,
              },
            }));
          } else if (name === "done") {
            if (data.session_id) sessionRef.current = data.session_id;
            patchLast((m) => ({ ...m, stage: "done" }));
          } else if (name === "error") {
            patchLast((m) => ({ ...m, error: data.error || "生成出错", stage: "done" }));
          }
        },
        onError: (e) =>
          patchLast((m) => ({ ...m, error: e.message, stage: "done" })),
        onDone: () => setStreaming(false),
      }
    );
  }, [input, streaming, id, patchLast]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const onInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  return (
    <div className="flex h-[calc(100vh-8rem)] flex-col">
      <div className="mb-3 flex items-center gap-2">
        <Link
          href={`/kb/${id}`}
          className="inline-flex items-center gap-1 text-sm text-slate-500 hover:text-slate-700"
        >
          <ArrowLeft size={16} />
          {kbQ.data?.name ?? "知识库"}
        </Link>
      </div>

      {/* 消息区 */}
      <div className="flex-1 space-y-5 overflow-y-auto rounded-xl border border-slate-200 bg-white p-4">
        {messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-slate-400">
            <Bot size={32} />
            <p className="text-sm">向「{kbQ.data?.name ?? "知识库"}」提问，答案会逐字流式返回并附上引用来源</p>
          </div>
        )}

        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-slate-900 px-4 py-2.5 text-sm text-white">
                {m.content}
              </div>
            </div>
          ) : (
            <div key={i} className="flex justify-start">
              <div className="max-w-[85%]">
                <div className="rounded-2xl rounded-bl-sm border border-slate-200 bg-slate-50 px-4 py-2.5 text-sm text-slate-800">
                  {/* 流式阶段提示 */}
                  {streaming && i === messages.length - 1 && !m.content && !m.error && (
                    <div className="flex items-center gap-2 text-slate-400">
                      <Spinner className="h-3.5 w-3.5" />
                      {STAGE_LABEL[m.stage ?? "intent"] ?? "处理中…"}
                    </div>
                  )}
                  {m.content && <div className="whitespace-pre-wrap leading-relaxed">{m.content}</div>}
                  {m.error && <div className="text-red-600">⚠️ {m.error}</div>}

                  {/* 生成中且已开始出字，底部显示核查状态 */}
                  {streaming &&
                    i === messages.length - 1 &&
                    m.content &&
                    m.stage === "verifying" && (
                      <div className="mt-2 flex items-center gap-1.5 text-xs text-slate-400">
                        <Spinner className="h-3 w-3" />
                        事实核查中…
                      </div>
                    )}

                  {m.reflection && (
                    <div className="mt-2">
                      <ReflectionBadge r={m.reflection} />
                    </div>
                  )}
                </div>

                {m.sources && m.sources.length > 0 && <Citations sources={m.sources} />}
              </div>
            </div>
          )
        )}
        <div ref={endRef} />
      </div>

      {/* 输入区 */}
      <div className="mt-3 flex items-end gap-2 rounded-xl border border-slate-200 bg-white p-2">
        <textarea
          ref={taRef}
          value={input}
          onChange={onInput}
          onKeyDown={onKeyDown}
          rows={1}
          placeholder="输入问题，Enter 发送，Shift+Enter 换行"
          disabled={streaming}
          className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-slate-400 disabled:opacity-60"
        />
        <Button size="icon" onClick={send} disabled={streaming || !input.trim()} title="发送">
          {streaming ? <Spinner className="h-4 w-4" /> : <Send size={16} />}
        </Button>
      </div>
    </div>
  );
}

export default function ChatPage() {
  const params = useParams<{ id: string }>();
  return (
    <AppShell>
      <ChatInner id={params.id} />
    </AppShell>
  );
}
