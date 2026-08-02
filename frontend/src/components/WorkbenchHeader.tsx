import React from "react";

export function WorkbenchHeader() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20">
            VF
          </div>
          <div>
            <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
              VisionForge
            </span>
            <span className="ml-2 px-2 py-0.5 text-xs font-mono rounded bg-slate-800/80 text-cyan-400 border border-cyan-500/20">
              v0.1.0-alpha
            </span>
          </div>
        </div>

        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-400">
          <a
            href="#workbench"
            className="text-slate-100 hover:text-cyan-400 transition-colors"
          >
            Workbench
          </a>
          <a
            href="#architecture"
            className="hover:text-cyan-400 transition-colors"
          >
            Architecture
          </a>
          <a
            href="#foundation"
            className="hover:text-cyan-400 transition-colors"
          >
            Foundation Models
          </a>
          <a href="#docs" className="hover:text-cyan-400 transition-colors">
            Docs
          </a>
        </nav>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs text-slate-300">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>Core Engine Ready</span>
          </div>
        </div>
      </div>
    </header>
  );
}
