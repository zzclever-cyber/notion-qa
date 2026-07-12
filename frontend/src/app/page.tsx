"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { getToken } from "@/lib/api";
import { Spinner } from "@/components/ui/spinner";

export default function Home() {
  const router = useRouter();
  useEffect(() => {
    router.replace(getToken() ? "/dashboard" : "/login");
  }, [router]);
  return (
    <div className="flex h-screen items-center justify-center text-slate-400">
      <Spinner className="h-6 w-6" />
    </div>
  );
}
