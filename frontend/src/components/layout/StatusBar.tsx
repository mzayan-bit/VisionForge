"use client";

import React from "react";
import { usePathname } from "next/navigation";
import { Circle, Server, Activity, GitBranch } from "lucide-react";

export const StatusBar: React.FC = () => {
  const pathname = usePathname();

  return (
    <footer className="h-7 border-t border-white/10 bg-[#090909] px-3 flex items-center justify-between text-[11px] font-mono text-neutral-400 shrink-0 select-none">
      {/* Left side: System status & active route */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 text-emerald-400">
          <Circle className="w-2 h-2 fill-emerald-400 animate-pulse" />
          <span className="font-medium text-[10px]">SYSTEM: OPERATIONAL</span>
        </div>

        <span className="text-neutral-700">|</span>

        <div className="flex items-center gap-1 text-neutral-300">
          <Server className="w-3 h-3 text-neutral-500" />
          <span>Route: <span className="text-blue-400">{pathname}</span></span>
        </div>
      </div>

      {/* Right side: Latency, environment, branch */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1 text-neutral-400">
          <Activity className="w-3 h-3 text-emerald-400" />
          <span>Latency: <span className="text-neutral-200">24ms</span></span>
        </div>

        <span className="text-neutral-700">|</span>

        <div className="flex items-center gap-1 text-neutral-400">
          <GitBranch className="w-3 h-3 text-neutral-500" />
          <span>main</span>
        </div>

        <span className="text-neutral-700">|</span>

        <span className="text-neutral-400 uppercase text-[10px] bg-neutral-800/60 px-1.5 py-0.2 rounded border border-white/10">
          ENV: DEVELOPMENT
        </span>
      </div>
    </footer>
  );
};
