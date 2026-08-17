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
} from "lucide-react";

export interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ collapsed, onToggleCollapse }) => {
  const pathname = usePathname();

  const navItems = [
    { label: "Home", href: "/", icon: <Home className="w-4 h-4" /> },
    { label: "Ask VisionForge", href: "/ask", icon: <Sparkles className="w-4 h-4 text-cyan-400" /> },
    { label: "Pipeline Studio", href: "/pipeline", icon: <GitBranch className="w-4 h-4 text-rose-400" /> },
    { label: "Workspace", href: "/workspace", icon: <Layout className="w-4 h-4" /> },
    { label: "Models", href: "/models", icon: <Cpu className="w-4 h-4" /> },
    { label: "Embeddings", href: "/embeddings", icon: <Fingerprint className="w-4 h-4 text-blue-400" /> },
    { label: "Explorer", href: "/explorer", icon: <Compass className="w-4 h-4 text-purple-400" /> },
    { label: "Visual Search", href: "/search", icon: <Search className="w-4 h-4 text-emerald-400" /> },
    { label: "Evaluation", href: "/evaluation", icon: <BarChart2 className="w-4 h-4 text-emerald-400" /> },
    { label: "Explainability", href: "/explainability", icon: <Fingerprint className="w-4 h-4 text-purple-400" /> },
    { label: "Experiments", href: "/experiments", icon: <FlaskConical className="w-4 h-4" /> },
    { label: "Research Workflow", href: "/workflow", icon: <GitBranch className="w-4 h-4 text-emerald-400" /> },
    { label: "Datasets", href: "/datasets", icon: <Database className="w-4 h-4 text-emerald-400" /> },
    { label: "Training Lab", href: "/training", icon: <Cpu className="w-4 h-4 text-indigo-400" /> },
    { label: "Vision Lab", href: "/vision-lab", icon: <FlaskConical className="w-4 h-4 text-cyan-400" /> },
    { label: "Video Lab", href: "/video-lab", icon: <Compass className="w-4 h-4 text-cyan-400" /> },
    { label: "Active Learning", href: "/active-learning", icon: <Compass className="w-4 h-4 text-amber-400" /> },
  ];

  const systemItems = [
    { label: "Documentation", href: "/documentation", icon: <BookOpen className="w-4 h-4" /> },
    { label: "Settings", href: "/settings", icon: <Settings className="w-4 h-4" /> },
  ];

  return (
    <aside
      className={`border-r border-white/10 bg-[#0a0a0a] flex flex-col justify-between transition-all duration-200 shrink-0 select-none ${
        collapsed ? "w-16" : "w-60"
      }`}
    >
      <div className="p-3 space-y-6">
        {/* Core Navigation Section */}
        <div>
          {!collapsed && (
            <div className="px-3 mb-2 text-[10px] font-semibold text-neutral-400 uppercase tracking-widest font-geist">
              Workbench
            </div>
          )}
          <nav className="space-y-1">
            {navItems.map((item) => {
              const isActive =
                item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  title={collapsed ? item.label : undefined}
                  className={`flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                    isActive
                      ? "bg-blue-600/15 text-blue-400 border border-blue-500/30 shadow-sm"
                      : "text-neutral-400 hover:text-white hover:bg-neutral-900"
                  }`}
                >
                  <span className={isActive ? "text-blue-400" : "text-neutral-400"}>
                    {item.icon}
                  </span>
                  {!collapsed && <span className="truncate">{item.label}</span>}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* System Section */}
        <div>
          {!collapsed && (
            <div className="px-3 mb-2 text-[10px] font-semibold text-neutral-400 uppercase tracking-widest font-geist">
              System
            </div>
          )}
          <nav className="space-y-1">
            {systemItems.map((item) => {
              const isActive = pathname.startsWith(item.href);

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  title={collapsed ? item.label : undefined}
                  className={`flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                    isActive
                      ? "bg-blue-600/15 text-blue-400 border border-blue-500/30 shadow-sm"
                      : "text-neutral-400 hover:text-white hover:bg-neutral-900"
                  }`}
                >
                  <span className={isActive ? "text-blue-400" : "text-neutral-400"}>
                    {item.icon}
                  </span>
                  {!collapsed && <span className="truncate">{item.label}</span>}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Collapse Toggle Footer */}
      <div className="p-3 border-t border-white/10 flex items-center justify-between">
        <button
          onClick={onToggleCollapse}
          className="w-full flex items-center justify-center p-2 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-900 transition-colors cursor-pointer text-xs font-medium gap-2"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <>
              <ChevronLeft className="w-4 h-4" />
              <span className="truncate">Collapse Sidebar</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
};
