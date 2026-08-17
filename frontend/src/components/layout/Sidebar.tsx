"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home,
  Layout,
  Cpu,
  Fingerprint,
  Compass,
  Search,
  BarChart2,
  FlaskConical,
  Database,
  BookOpen,
  Settings,
  ChevronLeft,
  ChevronRight,
  GitBranch,
  Sparkles,
  Layers,
  Video,
  FileSearch,
  Activity,
} from "lucide-react";

export interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
}

interface NavGroup {
  title: string;
  items: {
    label: string;
    href: string;
    icon: React.ReactNode;
    badge?: string;
  }[];
}

export const Sidebar: React.FC<SidebarProps> = ({ collapsed, onToggleCollapse }) => {
  const pathname = usePathname();

  const navGroups: NavGroup[] = [
    {
      title: "Workspace",
      items: [
        { label: "Overview", href: "/", icon: <Home className="w-4 h-4 text-blue-400" /> },
        { label: "Pipeline Studio", href: "/pipeline", icon: <GitBranch className="w-4 h-4 text-rose-400" /> },
        { label: "Workspace Studio", href: "/workspace", icon: <Layout className="w-4 h-4 text-indigo-400" /> },
      ],
    },
    {
      title: "Vision & Video",
      items: [
        { label: "Vision Lab", href: "/vision-lab", icon: <Layers className="w-4 h-4 text-cyan-400" /> },
        { label: "Video Lab", href: "/video-lab", icon: <Video className="w-4 h-4 text-emerald-400" /> },
        { label: "Visual Search", href: "/search", icon: <Search className="w-4 h-4 text-purple-400" /> },
        { label: "Dataset Explorer", href: "/explorer", icon: <Compass className="w-4 h-4 text-amber-400" /> },
      ],
    },
    {
      title: "Data & Active Learning",
      items: [
        { label: "Datasets", href: "/datasets", icon: <Database className="w-4 h-4 text-emerald-400" /> },
        { label: "Active Learning", href: "/active-learning", icon: <Activity className="w-4 h-4 text-amber-400" /> },
        { label: "Embeddings", href: "/embeddings", icon: <Fingerprint className="w-4 h-4 text-blue-400" /> },
      ],
    },
    {
      title: "Models & Evaluation",
      items: [
        { label: "Model Registry", href: "/models", icon: <Cpu className="w-4 h-4 text-indigo-400" /> },
        { label: "Training Lab", href: "/training", icon: <Cpu className="w-4 h-4 text-cyan-400" /> },
        { label: "Evaluation", href: "/evaluation", icon: <BarChart2 className="w-4 h-4 text-emerald-400" /> },
        { label: "Explainability", href: "/explainability", icon: <Fingerprint className="w-4 h-4 text-purple-400" /> },
      ],
    },
    {
      title: "Research Lab",
      items: [
        { label: "Experiments & Ablations", href: "/experiments", icon: <FlaskConical className="w-4 h-4 text-purple-400" /> },
        { label: "Research Workflows", href: "/workflow", icon: <GitBranch className="w-4 h-4 text-emerald-400" /> },
        { label: "Ask VisionForge", href: "/ask", icon: <Sparkles className="w-4 h-4 text-cyan-400" /> },
      ],
    },
    {
      title: "System",
      items: [
        { label: "Settings & Diagnostics", href: "/settings", icon: <Settings className="w-4 h-4 text-neutral-400" /> },
        { label: "Documentation", href: "/documentation", icon: <BookOpen className="w-4 h-4 text-neutral-400" /> },
      ],
    },
  ];

  return (
    <aside
      className={`border-r border-white/10 bg-[#0a0a0c] flex flex-col justify-between transition-all duration-200 shrink-0 select-none ${
        collapsed ? "w-16" : "w-64"
      }`}
    >
      <div className="p-3 space-y-5 overflow-y-auto max-h-[calc(100vh-3.5rem)]">
        {navGroups.map((group) => (
          <div key={group.title} className="space-y-1">
            {!collapsed && (
              <div className="px-3 mb-1.5 text-[9px] font-semibold text-neutral-500 uppercase tracking-widest font-mono">
                {group.title}
              </div>
            )}
            <nav className="space-y-0.5">
              {group.items.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    title={collapsed ? item.label : undefined}
                    className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                      isActive
                        ? "bg-white/10 text-white font-semibold shadow-sm border border-white/10"
                        : "text-neutral-400 hover:text-white hover:bg-white/5"
                    } ${collapsed ? "justify-center px-0" : ""}`}
                  >
                    <span className="shrink-0">{item.icon}</span>
                    {!collapsed && <span className="truncate">{item.label}</span>}
                  </Link>
                );
              })}
            </nav>
          </div>
        ))}
      </div>

      {/* Collapse Toggle Footer */}
      <div className="p-3 border-t border-white/10 flex items-center justify-between bg-[#08080a]">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-[11px] font-mono text-neutral-400">VisionForge v1.0.0</span>
          </div>
        )}
        <button
          onClick={onToggleCollapse}
          className="p-1.5 rounded-lg text-neutral-400 hover:text-white hover:bg-white/5 transition-colors mx-auto"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>
    </aside>
  );
};
