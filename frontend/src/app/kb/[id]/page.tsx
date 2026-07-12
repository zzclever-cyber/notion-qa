"use client";
import { useState, useCallback } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useDropzone } from "react-dropzone";
import {
  ArrowLeft,
  Upload,
  Trash2,
  MessageSquare,
  UploadCloud,
  FileText,
} from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Spinner } from "@/components/ui/spinner";
import { Card } from "@/components/ui/card";
import { Table, THead, TBody, TR, TH, TD } from "@/components/ui/table";
import { kbApi, docApi, ApiError } from "@/lib/api";
import { formatBytes, formatDate } from "@/lib/utils";
import type { DocStatus } from "@/lib/types";

function StatusBadge({ status }: { status: DocStatus }) {
  if (status === "ready") return <Badge tone="green">已就绪</Badge>;
  if (status === "failed") return <Badge tone="red">失败</Badge>;
  return (
    <Badge tone="yellow">
      <Spinner className="h-3 w-3" />
      处理中
    </Badge>
  );
}

function KbDetailInner({ id }: { id: string }) {
  const qc = useQueryClient();
  const kbQ = useQuery({
    queryKey: ["kb", id],
    queryFn: () => kbApi.get(id),
    // 有文档处理中时每 2s 轮询状态
    refetchInterval: (q) =>
      q.state.data?.documents.some((d) => d.status === "processing") ? 2000 : false,
  });

  const [open, setOpen] = useState(false);
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [err, setErr] = useState("");

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["kb", id] });
    qc.invalidateQueries({ queryKey: ["usage"] });
    qc.invalidateQueries({ queryKey: ["kbs"] });
  };

  const handleUpload = useCallback(
    async (file: File) => {
      setErr("");
      setUploading(true);
      setProgress(0);
      try {
        await docApi.upload(id, file, setProgress);
        refresh();
        setOpen(false);
      } catch (e) {
        setErr(e instanceof ApiError ? e.message : "上传失败");
      } finally {
        setUploading(false);
      }
    },
    [id] // eslint-disable-line react-hooks/exhaustive-deps
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    multiple: false,
    disabled: uploading,
    onDrop: (files) => files[0] && handleUpload(files[0]),
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
      "text/markdown": [".md"],
      "text/plain": [".txt"],
    },
  });

  const delMut = useMutation({
    mutationFn: (docId: string) => docApi.remove(docId),
    onSuccess: refresh,
  });

  const kb = kbQ.data;

  return (
    <div className="space-y-6">
      <Link
        href="/dashboard"
        className="inline-flex items-center gap-1 text-sm text-slate-500 hover:text-slate-700"
      >
        <ArrowLeft size={16} />
        返回
      </Link>

      {kbQ.isLoading ? (
        <div className="flex justify-center py-16 text-slate-400">
          <Spinner className="h-6 w-6" />
        </div>
      ) : kb ? (
        <>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">{kb.name}</h1>
              <p className="mt-1 text-sm text-slate-500">
                {kb.description || "暂无描述"}
              </p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setOpen(true)}>
                <Upload size={16} />
                上传文档
              </Button>
              <Link href={`/kb/${id}/chat`}>
                <Button>
                  <MessageSquare size={16} />
                  开始对话
                </Button>
              </Link>
            </div>
          </div>

          <Card>
            {kb.documents.length === 0 ? (
              <div className="flex flex-col items-center gap-3 py-16 text-center">
                <FileText size={32} className="text-slate-300" />
                <p className="text-sm text-slate-500">还没有文档，上传一个 PDF / Word / Markdown 试试</p>
                <Button size="sm" variant="outline" onClick={() => setOpen(true)}>
                  <Upload size={16} />
                  上传文档
                </Button>
              </div>
            ) : (
              <Table>
                <THead>
                  <TR>
                    <TH>文件名</TH>
                    <TH>类型</TH>
                    <TH>切块数</TH>
                    <TH>大小</TH>
                    <TH>状态</TH>
                    <TH>上传时间</TH>
                    <TH className="text-right">操作</TH>
                  </TR>
                </THead>
                <TBody>
                  {kb.documents.map((d) => (
                    <TR key={d.id}>
                      <TD className="max-w-[220px] truncate font-medium text-slate-900" title={d.filename}>
                        {d.filename}
                      </TD>
                      <TD className="uppercase">{d.file_type}</TD>
                      <TD>{d.chunk_count}</TD>
                      <TD>{formatBytes(d.file_size_bytes)}</TD>
                      <TD>
                        <span title={d.status === "failed" ? d.error_message : undefined}>
                          <StatusBadge status={d.status} />
                        </span>
                      </TD>
                      <TD className="text-slate-400">{formatDate(d.created_at)}</TD>
                      <TD className="text-right">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => delMut.mutate(d.id)}
                          disabled={delMut.isPending}
                          title="删除文档"
                        >
                          <Trash2 size={16} className="text-red-500" />
                        </Button>
                      </TD>
                    </TR>
                  ))}
                </TBody>
              </Table>
            )}
          </Card>
        </>
      ) : (
        <p className="text-slate-500">知识库不存在或无权访问。</p>
      )}

      {/* 上传对话框 */}
      <Dialog open={open} onClose={() => !uploading && setOpen(false)} title="上传文档">
        <div
          {...getRootProps()}
          className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-8 text-center transition-colors ${
            isDragActive ? "border-slate-900 bg-slate-50" : "border-slate-300"
          } ${uploading ? "pointer-events-none opacity-60" : ""}`}
        >
          <input {...getInputProps()} />
          <UploadCloud size={28} className="text-slate-400" />
          {uploading ? (
            <div className="w-full">
              <div className="mb-1 text-sm text-slate-600">上传中… {progress}%</div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                <div
                  className="h-full bg-slate-900 transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          ) : (
            <>
              <p className="text-sm text-slate-600">
                {isDragActive ? "松手上传" : "拖拽文件到此处，或点击选择"}
              </p>
              <p className="text-xs text-slate-400">支持 PDF / Word(.docx) / Markdown / txt</p>
            </>
          )}
        </div>
        {err && <p className="mt-3 text-sm text-red-600">{err}</p>}
        <p className="mt-3 text-xs text-slate-400">
          上传后后台会自动解析、切片、建索引，状态变为「已就绪」即可对话。
        </p>
      </Dialog>
    </div>
  );
}

export default function KbDetailPage() {
  const params = useParams<{ id: string }>();
  return (
    <AppShell>
      <KbDetailInner id={params.id} />
    </AppShell>
  );
}
