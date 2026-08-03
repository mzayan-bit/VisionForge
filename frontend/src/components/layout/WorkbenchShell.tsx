"use client";

import React, { useState, useEffect } from "react";
import { TopNav } from "./TopNav";
import { Sidebar } from "./Sidebar";
import { RightContextPanel } from "./RightContextPanel";
import { StatusBar } from "./StatusBar";
import { CommandPalette } from "@/components/ui/CommandPalette";

export interface WorkbenchShellProps {
  children: React.ReactNode;
}

export const WorkbenchShell: React.FC<WorkbenchShellProps> = ({ children }) => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [contextPanelCollapsed, setContextPanelCollapsed] = useState(false);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);

  // Global Keyboard Shortcut Listener (⌘K or Ctrl+K)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-[#0a0a0a] text-neutral-100 font-sans antialiased overflow-hidden">
      {/* Top Navigation */}
      <TopNav onOpenCommandPalette={() => setIsCommandPaletteOpen(true)} />

      {/* Main 3-Pane Body Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        />

        {/* Main Workspace Viewport */}
        <main className="flex-1 overflow-y-auto p-4 md:p-6 bg-[#0a0a0a] custom-scrollbar">
          <div className="max-w-7xl mx-auto">{children}</div>
        </main>

        {/* Right Context Panel */}
        <RightContextPanel
          collapsed={contextPanelCollapsed}
          onToggleCollapse={() => setContextPanelCollapsed(!contextPanelCollapsed)}
        />
      </div>

      {/* Bottom Status Bar */}
      <StatusBar />

      {/* Global Command Palette Modal */}
      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
      />
    </div>
  );
};
