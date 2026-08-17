"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import {
  Search,
  Home,
  Layout,
  Cpu,
  BarChart2,
  FlaskConical,
  Database,
  BookOpen,
  Settings,
  GitBranch,
  Video,
  Layers,
  Sparkles,
  Fingerprint,
  Compass,
} from "lucide-react";
import { Modal } from "./Modal";

export interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ isOpen, onClose }) => {
  const router = useRouter();
  const [query, setQuery] = useState("");

  const commands = [
    { id: "home", label: "Navigate to Overview", route: "/", icon: <Home className="w-4 h-4 text-blue-400" />, section: "Workspace" },
    { id: "workflow", label: "Open Research Workflows", route: "/workflow", icon: <GitBranch className="w-4 h-4 text-emerald-400" />, section: "Research" },
    { id: "experiments", label: "Open Experiments & Ablations", route: "/experiments", icon: <FlaskConical className="w-4 h-4 text-purple-400" />, section: "Research" },
    { id: "ask", label: "Ask VisionForge (Multimodal V-L)", route: "/ask", icon: <Sparkles className="w-4 h-4 text-cyan-400" />, section: "Research" },
    { id: "vision-lab", label: "Open Vision Lab (Image Inference)", route: "/vision-lab", icon: <Layers className="w-4 h-4 text-cyan-400" />, section: "Vision" },
    { id: "video-lab", label: "Open Video Lab (Temporal Events)", route: "/video-lab", icon: <Video className="w-4 h-4 text-indigo-400" />, section: "Vision" },
    { id: "search", label: "Visual & Vector Search", route: "/search", icon: <Search className="w-4 h-4 text-purple-400" />, section: "Vision" },
    { id: "datasets", label: "Manage Datasets & Versions", route: "/datasets", icon: <Database className="w-4 h-4 text-emerald-400" />, section: "Data" },
    { id: "active-learning", label: "Active Learning Studio", route: "/active-learning", icon: <Compass className="w-4 h-4 text-amber-400" />, section: "Data" },
    { id: "models", label: "Model Registry", route: "/models", icon: <Cpu className="w-4 h-4 text-indigo-400" />, section: "Models" },
    { id: "training", label: "Training Lab", route: "/training", icon: <Cpu className="w-4 h-4 text-cyan-400" />, section: "Models" },
    { id: "evaluation", label: "Evaluation & Benchmarks", route: "/evaluation", icon: <BarChart2 className="w-4 h-4 text-emerald-400" />, section: "Models" },
    { id: "explainability", label: "Explainability & Diagnostics", route: "/explainability", icon: <Fingerprint className="w-4 h-4 text-purple-400" />, section: "Models" },
    { id: "settings", label: "Settings & System Diagnostics", route: "/settings", icon: <Settings className="w-4 h-4 text-neutral-400" />, section: "System" },
    { id: "docs", label: "Architecture Documentation", route: "/documentation", icon: <BookOpen className="w-4 h-4 text-neutral-400" />, section: "System" },
  ];

  const filteredCommands = commands.filter(
    (cmd) =>
      cmd.label.toLowerCase().includes(query.toLowerCase()) ||
      cmd.section.toLowerCase().includes(query.toLowerCase())
  );

  const handleSelect = (route: string) => {
    router.push(route);
    onClose();
    setQuery("");
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} maxWidth="lg">
      <div className="flex items-center gap-3 pb-3 border-b border-white/10 text-neutral-300">
        <Search className="w-5 h-5 text-purple-400 shrink-0" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Type a command or jump to workspace..."
          className="w-full bg-transparent text-sm text-white placeholder-neutral-500 focus:outline-none"
          autoFocus
        />
        <kbd className="text-[10px] font-mono px-1.5 py-0.5 bg-neutral-800 text-neutral-400 rounded border border-white/10">
          ESC
        </kbd>
      </div>

      <div className="mt-3 max-h-80 overflow-y-auto space-y-1">
        {filteredCommands.length === 0 ? (
          <div className="py-6 text-center text-xs text-neutral-500 font-mono">
            No matching workbench destinations found
          </div>
        ) : (
          filteredCommands.map((cmd) => (
            <button
              key={cmd.id}
              onClick={() => handleSelect(cmd.route)}
              className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs text-neutral-300 hover:text-white hover:bg-neutral-800/80 transition-colors text-left group cursor-pointer"
            >
              <div className="flex items-center gap-3">
                <span className="shrink-0">{cmd.icon}</span>
                <span className="font-medium text-neutral-200 group-hover:text-white">{cmd.label}</span>
              </div>
              <span className="text-[10px] font-mono text-neutral-500 uppercase px-1.5 py-0.5 rounded bg-neutral-900 border border-white/5">
                {cmd.section}
              </span>
            </button>
          ))
        )}
      </div>
    </Modal>
  );
};
