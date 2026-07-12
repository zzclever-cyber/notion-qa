"use client";
import { useState } from "react";
import Link from "next/link";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { Plus, FileText, MessageSquare, Coins, Library } from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog } from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { Card } from "@/components/ui/card";
import { kbApi, usageApi, ApiError } from "@/lib/api";
import { formatDate } from "@/lib/utils";

function StatTile({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: number | string;
}) {
  return (
    <Card className="flex items-center gap-3 p-4">
      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-100 text-slate-600">
        {icon}
      </div>
      <div>
        <div className="text-2xl font-semibold text-slate-900">{value}</div>
        <div className="text-xs text-slate-500">{label}</div>
      </div>
    </Card>
  );
}

function DashboardInner() {
  const qc = useQueryClient();
  const kbsQ = useQuery({ queryKey: ["kbs"], queryFn: kbApi.list });
  const usageQ = useQuery({ queryKey: ["usage"], queryFn: usageApi.stats });

  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [err, setErr] = useState("");

  const createMut = useMutation({
    mutationFn: () => kbApi.create(name.trim(), desc.trim() || undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["kbs"] });
      qc.invalidateQueries({ queryKey: ["usage"] });
      setOpen(false);
      setName("");
      setDesc("");
      setErr("");
    },
    onError: (e) => setErr(e instanceof ApiError ? e.message : "创建失败"),
  });

  const usage = usageQ.data;
  const chartData = usage
    ? [
        { name: "知识库", value: usage.kb_count },
        { name: "文档", value: usage.document_count },
        { name: "问答", value: usage.chat_count },
      ]
    : [];

  return (
    <div className="space-y-8">
      {/* 用量概览 */}
      <section>
        <h2 className="mb-3 text-lg font-semibold text-slate-900">用量概览</h2>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <StatTile icon={<Library size={18} />} label="知识库" value={usage?.kb_count ?? "-"} />
          <StatTile icon={<FileText size={18} />} label="文档总数" value={usage?.document_count ?? "-"} />
          <StatTile icon={<MessageSquare size={18} />} label="问答次数" value={usage?.chat_count ?? "-"} />
          <StatTile icon={<Coins size={18} />} label="Token 消耗(估算)" value={usage?.total_tokens ?? "-"} />
        </div>
        {chartData.length > 0 && (
          <Card className="mt-3 p-4">
            <div className="mb-2 text-sm font-medium text-slate-600">资源分布</div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={chartData} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 12, fill: "#64748b" }} axisLine={false} tickLine={false} />
                <YAxis allowDecimals={false} tick={{ fontSize: 12, fill: "#64748b" }} axisLine={false} tickLine={false} />
                <Tooltip cursor={{ fill: "#f8fafc" }} />
                <Bar dataKey="value" fill="#475569" radius={[4, 4, 0, 0]} maxBarSize={64} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        )}
      </section>

      {/* 知识库列表 */}
      <section>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-900">我的知识库</h2>
          <Button size="sm" onClick={() => setOpen(true)}>
            <Plus size={16} />
            创建知识库
          </Button>
        </div>

        {kbsQ.isLoading ? (
          <div className="flex justify-center py-16 text-slate-400">
            <Spinner className="h-6 w-6" />
          </div>
        ) : kbsQ.data && kbsQ.data.length > 0 ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {kbsQ.data.map((kb) => (
              <Link key={kb.id} href={`/kb/${kb.id}`}>
                <Card className="h-full p-5 transition-shadow hover:shadow-md">
                  <div className="mb-2 flex items-center gap-2">
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-900 text-white">
                      <Library size={16} />
                    </div>
                    <div className="font-semibold text-slate-900">{kb.name}</div>
                  </div>
                  <p className="line-clamp-2 min-h-[2.5rem] text-sm text-slate-500">
                    {kb.description || "暂无描述"}
                  </p>
                  <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
                    <span>{kb.document_count} 篇文档</span>
                    <span>{formatDate(kb.created_at)}</span>
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        ) : (
          <Card className="flex flex-col items-center gap-3 py-16 text-center">
            <Library size={32} className="text-slate-300" />
            <p className="text-sm text-slate-500">还没有知识库，创建一个开始吧</p>
            <Button size="sm" onClick={() => setOpen(true)}>
              <Plus size={16} />
              创建知识库
            </Button>
          </Card>
        )}
      </section>

      {/* 创建对话框 */}
      <Dialog open={open} onClose={() => setOpen(false)} title="创建知识库">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (name.trim()) createMut.mutate();
          }}
          className="space-y-4"
        >
          <div>
            <Label htmlFor="kb-name">名称</Label>
            <Input
              id="kb-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="如：公司制度"
              autoFocus
              required
            />
          </div>
          <div>
            <Label htmlFor="kb-desc">描述（可选）</Label>
            <Input
              id="kb-desc"
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              placeholder="这个知识库放什么内容"
            />
          </div>
          {err && <p className="text-sm text-red-600">{err}</p>}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={() => setOpen(false)}>
              取消
            </Button>
            <Button type="submit" disabled={createMut.isPending || !name.trim()}>
              {createMut.isPending && <Spinner className="h-4 w-4" />}
              创建
            </Button>
          </div>
        </form>
      </Dialog>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <AppShell>
      <DashboardInner />
    </AppShell>
  );
}
