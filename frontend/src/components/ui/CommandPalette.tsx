"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Search, Home, Layout, Cpu, BarChart2, FlaskConical, Database, BookOpen, Settings } from "lucide-react";
import { Modal } from "./Modal";

export interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ isOpen, onClose }) => {
  const router = Router();
  const [query, setQuery] = useState("");

  function Router() {
    return useRouter();
  }

  const commands = [
    { id: "home", label: "Navigate to Home", route: "/", icon: <Home className="w-4 h-4" /> },
    { id: "workspace", label: "Navigate to Workspace", route: "/workspace", icon: <Layout className="w-4 h-4" /> },
    { id: "models", label: "Navigate to Models", route: "/models", icon: <Cpu className="w-4 h-4" /> },
    { id: "benchmarks", label: "Navigate to Benchmarks", route: "/benchmarks", icon: <BarChart2 className="w-4 h-4" /> },
    { id: "experiments", label: "Navigate to Experiments", route: "/experiments", icon: <FlaskConical className="w-4 h-4" /> },
    { id: "datasets", label: "Navigate to Datasets", route: "/datasets", icon: <Database className="w-4 h-4" /> },
    { id: "documentation", label: "Navigate to Documentation", route: "/documentation", icon: <BookOpen className="w-4 h-4" /> },
    { id: "settings", label: "Navigate to Settings", route: "/settings", icon: <Settings className="w-4 h-4" /> },
  ];

  const filteredCommands = commands.filter((cmd) =>
    cmd.label.toLowerCase().includes(query.toLowerCase())
  );

  const handleSelect = (route: string) => {
    router.push(route);
    onClose();
    setQuery("");
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} maxWidth="lg">
      <div className="flex items-center gap-3 pb-3 border-b border-white/10 text-neutral-300">
        <Search className="w-5 h-5 text-blue-400 shrink-0" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Type a command or search workbench..."
          className="w-full bg-transparent text-sm text-white placeholder-neutral-500 focus:outline-none"
          autoFocus
        />
        <kbd className="text-[10px] font-mono px-1.5 py-0.5 bg-neutral-800 text-neutral-400 rounded border border-white/10">
          ESC
        </kbd>
      </div>

      <div className="mt-3 max-h-64 overflow-y-auto space-y-1">
        {filteredCommands.length === 0 ? (
          <div className="py-6 text-center text-xs text-neutral-500 font-mono">
            No matching workbench commands found
          </div>
        ) : (
          filteredCommands.map((cmd) => (
            <button
              key={cmd.id}
              onClick={() => handleSelect(cmd.route)}
              className="w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm text-neutral-300 hover:text-white hover:bg-neutral-800/80 transition-colors text-left group cursor-pointer"
            >
              <div className="flex items-center gap-3">
                <span className="text-neutral-400 group-hover:text-blue-400 transition-colors">
                  {cmd.icon}
                </span>
                <span>{cmd.label}</span>
              </div>
              <span className="text-xs font-mono text-neutral-500 group-hover:text-neutral-300">
                Jump to
              </span>
            </button>
          ))
        )}
      </div>
    </Modal>
  );
};
