"use client";

import React, { useState } from "react";
import { Info, Activity, Cpu, PanelRightClose, PanelRightOpen, Shield } from "lucide-react";
import { Tabs } from "@/components/ui/Tabs";
import { Badge } from "@/components/ui/Badge";

export interface RightContextPanelProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
}

export const RightContextPanel: React.FC<RightContextPanelProps> = ({
  collapsed,
  onToggleCollapse,
}) => {
  const [activeTab, setActiveTab] = useState("overview");

  if (collapsed) {
    return (
      <div className="border-l border-white/10 bg-[#0a0a0a] w-10 flex flex-col items-center py-3 shrink-0">
        <button
          onClick={onToggleCollapse}
          className="p-1.5 rounded text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors cursor-pointer"
          title="Expand Context Panel"
        >
          <PanelRightOpen className="w-4 h-4" />
        </button>
      </div>
    );
  }

  const contextTabs = [
    { id: "overview", label: "Details", icon: <Info className="w-3.5 h-3.5" /> },
    { id: "system", label: "Runtime", icon: <Cpu className="w-3.5 h-3.5" /> },
    { id: "metrics", label: "Health", icon: <Activity className="w-3.5 h-3.5" /> },
  ];

  return (
    <aside className="w-72 border-l border-white/10 bg-[#0c0c0c] flex flex-col justify-between shrink-0 select-none text-xs">
      {/* Context Header */}
      <div>
        <div className="p-3 border-b border-white/10 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-blue-400" />
            <span className="font-semibold text-white font-geist">Context Inspector</span>
          </div>
          <button
            onClick={onToggleCollapse}
            className="p-1 rounded text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors cursor-pointer"
            title="Collapse Context Panel"
          >
            <PanelRightClose className="w-4 h-4" />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="px-2 pt-2">
          <Tabs tabs={contextTabs} activeTab={activeTab} onChange={setActiveTab} />
        </div>

        {/* Tab Content */}
        <div className="p-4 space-y-4">
          {activeTab === "overview" && (
            <div className="space-y-3">
              <div>
                <span className="text-[10px] text-neutral-400 font-mono uppercase">Active Workspace</span>
                <p className="text-sm font-medium text-white font-geist">VisionForge Workbench Core</p>
              </div>

              <div className="pt-2 border-t border-white/10 space-y-2">
                <div className="flex items-center justify-between text-neutral-400">
                  <span>Architecture</span>
                  <span className="text-neutral-200 font-mono">Decoupled Fast / Next</span>
                </div>
                <div className="flex items-center justify-between text-neutral-400">
                  <span>Theme Protocol</span>
                  <Badge variant="info" size="sm">Dark Mode First</Badge>
                </div>
                <div className="flex items-center justify-between text-neutral-400">
                  <span>State Layer</span>
                  <span className="text-neutral-200 font-mono">SSR Ready</span>
                </div>
              </div>
            </div>
          )}

          {activeTab === "system" && (
            <div className="space-y-3 font-mono text-[11px]">
              <div className="p-2.5 rounded bg-neutral-900 border border-white/10 space-y-1.5">
                <div className="flex justify-between text-neutral-400">
                  <span>REST API:</span>
                  <span className="text-emerald-400">/api/v1</span>
                </div>
                <div className="flex justify-between text-neutral-400">
                  <span>FastAPI Core:</span>
                  <span className="text-neutral-200">Online</span>
                </div>
                <div className="flex justify-between text-neutral-400">
                  <span>Next App Router:</span>
                  <span className="text-blue-400">React 19</span>
                </div>
              </div>
            </div>
          )}

          {activeTab === "metrics" && (
            <div className="space-y-3">
              <div className="p-3 rounded bg-neutral-900/60 border border-white/10 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-neutral-400 text-[11px]">System Status</span>
                  <Badge variant="success" dot size="sm">OPERATIONAL</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-neutral-400 text-[11px]">API Latency</span>
                  <span className="font-mono text-emerald-400">24ms</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};
