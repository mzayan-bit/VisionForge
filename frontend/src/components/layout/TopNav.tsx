"use client";

import React from "react";
import Link from "next/link";
import { Search, Terminal, BookOpen, Activity, Command, ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/Badge";

export interface TopNavProps {
  onOpenCommandPalette: () => void;
}

export const TopNav: React.FC<TopNavProps> = ({ onOpenCommandPalette }) => {
  return (
    <header className="h-14 border-b border-white/10 bg-[#0d0d0d]/80 backdrop-blur-md px-4 flex items-center justify-between shrink-0 sticky top-0 z-30 select-none">
      {/* Brand & Logo */}
      <div className="flex items-center gap-3">
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400 group-hover:border-blue-400/50 transition-colors shadow-sm">
            <Terminal className="w-4 h-4" />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold font-geist tracking-tight text-white flex items-center gap-1.5">
              VisionForge
              <span className="text-[10px] font-mono px-1 py-0.2 rounded bg-neutral-800 text-neutral-400 border border-white/10">
                v0.1.0
              </span>
            </span>
            <span className="text-[10px] text-neutral-400 font-mono">Computer Vision Workbench</span>
          </div>
        </Link>
      </div>

      {/* Global Command Palette Trigger */}
      <div className="hidden md:flex items-center flex-1 max-w-md mx-8">
        <button
          onClick={onOpenCommandPalette}
          className="w-full flex items-center justify-between px-3 py-1.5 rounded-lg bg-neutral-900/80 border border-white/10 text-neutral-400 hover:text-neutral-200 hover:border-white/20 hover:bg-neutral-900 transition-all text-xs cursor-pointer group"
        >
          <div className="flex items-center gap-2">
            <Search className="w-3.5 h-3.5 text-neutral-500 group-hover:text-blue-400 transition-colors" />
            <span>Search workbench or type command...</span>
          </div>
          <div className="flex items-center gap-1 font-mono text-[10px] text-neutral-400 bg-neutral-800 px-1.5 py-0.5 rounded border border-white/10">
            <Command className="w-3 h-3" />
            <span>K</span>
          </div>
        </button>
      </div>

      {/* Right Quick Actions */}
      <div className="flex items-center gap-3">
        <Badge variant="success" dot size="sm">
          READY
        </Badge>
        <div className="h-4 w-px bg-white/10 hidden sm:block" />

        <Link
          href="/documentation"
          className="p-1.5 rounded-md text-neutral-400 hover:text-white hover:bg-neutral-800/80 transition-colors"
          title="Documentation"
        >
          <BookOpen className="w-4 h-4" />
        </Link>

        <Link
          href="/settings"
          className="p-1.5 rounded-md text-neutral-400 hover:text-white hover:bg-neutral-800/80 transition-colors"
          title="System Diagnostics"
        >
          <Activity className="w-4 h-4" />
        </Link>

        <div className="h-4 w-px bg-white/10 hidden sm:block" />

        <div className="flex items-center gap-2 pl-1">
          <div className="w-6 h-6 rounded-full bg-blue-500/20 border border-blue-400/30 flex items-center justify-center text-blue-300 font-mono text-[10px]">
            MZ
          </div>
          <span className="text-xs text-neutral-300 font-mono hidden sm:inline-block">Lab Admin</span>
        </div>
      </div>
    </header>
  );
};
