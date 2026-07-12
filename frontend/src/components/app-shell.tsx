"use client";
import { useEffect, type ReactNode } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { BookOpen, LogOut } from "lucide-react";
import { useAuth } from "@/context/auth";
import { Spinner } from "@/components/ui/spinner";
import { Button } from "@/components/ui/button";

/** 受保护页面外壳：未登录重定向到 /login，已登录渲染顶栏 + 内容 */
export function AppShell({ children }: { children: ReactNode }) {
  const { user, loading, logout } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) router.replace("/login");
  }, [loading, user, router]);

  if (loading || !user) {
    return (
      <div className="flex h-screen items-center justify-center text-slate-400">
        <Spinner className="h-6 w-6" />
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
          <Link href="/dashboard" className="flex items-center gap-2 font-semibold text-slate-900">
            <BookOpen size={20} className="text-slate-700" />
            Notion-QA
          </Link>
          <div className="flex items-center gap-3">
            <span className="hidden text-sm text-slate-500 sm:inline">
              {user.display_name || user.email}
            </span>
            <Button variant="ghost" size="sm" onClick={logout} title="退出登录">
              <LogOut size={16} />
              <span className="hidden sm:inline">退出</span>
            </Button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">{children}</main>
    </div>
  );
}
