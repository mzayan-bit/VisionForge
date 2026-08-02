import React from "react";
import { WorkbenchHeader } from "@/components/WorkbenchHeader";
import { StatusCard } from "@/components/StatusCard";

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200">
      <WorkbenchHeader />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
        {/* Hero Section */}
        <section className="relative rounded-2xl bg-gradient-to-b from-slate-900/90 to-slate-950 border border-slate-800 p-8 sm:p-12 overflow-hidden shadow-2xl">
          <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl -z-10 pointer-events-none"></div>
          <div className="max-w-3xl space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-xs font-mono text-cyan-400">
              <span className="h-1.5 w-1.5 rounded-full bg-cyan-400"></span>
              Visual AI Laboratory & Workbench Platform
            </div>

            <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight leading-tight">
              Engineered for Modern{" "}
              <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400 bg-clip-text text-transparent">
                Computer Vision Research
              </span>
            </h1>

            <p className="text-lg text-slate-300 leading-relaxed">
              VisionForge is a modular workbench where researchers and engineers
              can integrate, benchmark, visualize, and experiment with state-of-the-art
              foundation models.
            </p>

            <div className="pt-2 flex flex-wrap items-center gap-4">
              <a
                href="#architecture"
                className="px-5 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-sm transition-all shadow-lg shadow-cyan-500/25"
              >
                Explore Architecture
              </a>
              <a
                href="https://github.com/mzayan-bit/VisionForge"
                target="_blank"
                rel="noreferrer"
                className="px-5 py-2.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 font-medium text-sm transition-all"
              >
                GitHub Repository
              </a>
            </div>
          </div>
        </section>

        {/* System Diagnostics & Foundation Status */}
        <section id="architecture" className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-slate-100 tracking-tight">
                System Diagnostics & Architecture
              </h2>
              <p className="text-sm text-slate-400">
                Core foundation services initialized and verified.
              </p>
            </div>
            <span className="text-xs font-mono text-slate-500">
              Environment: Development
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <StatusCard
              id="status-backend"
              title="Python Backend Service"
              subtitle="FastAPI REST API & async engine"
              status="Online"
              statusType="success"
              iconText="PY"
            />

            <StatusCard
              id="status-package-manager"
              title="Package Management"
              subtitle="uv high-performance resolver"
              status="Active"
              statusType="info"
              iconText="UV"
            />

            <StatusCard
              id="status-frontend"
              title="Workbench Dashboard"
              subtitle="Next.js App Router & TypeScript"
              status="Rendered"
              statusType="success"
              iconText="TS"
            />
          </div>
        </section>

        {/* Workbench Modules Preview */}
        <section id="workbench" className="space-y-6">
          <div>
            <h2 className="text-xl font-bold text-slate-100 tracking-tight">
              Workbench Core Modules
            </h2>
            <p className="text-sm text-slate-400">
              Extensible architectural building blocks for model evaluation and visualization.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-3">
              <div className="h-8 w-8 rounded bg-cyan-500/10 text-cyan-400 flex items-center justify-center font-mono text-sm font-bold">
                01
              </div>
              <h3 className="font-semibold text-slate-200">Model Integration Layer</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Unified abstract interfaces for wrapping vision foundation architectures cleanly.
              </p>
            </div>

            <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-3">
              <div className="h-8 w-8 rounded bg-blue-500/10 text-blue-400 flex items-center justify-center font-mono text-sm font-bold">
                02
              </div>
              <h3 className="font-semibold text-slate-200">Benchmarking Engine</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Comprehensive latency, memory, accuracy, and throughput profiling suite.
              </p>
            </div>

            <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-3">
              <div className="h-8 w-8 rounded bg-indigo-500/10 text-indigo-400 flex items-center justify-center font-mono text-sm font-bold">
                03
              </div>
              <h3 className="font-semibold text-slate-200">Visual Inspection Canvas</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Interactive spatial visualization for bounding boxes, segmentations, and heatmaps.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t border-slate-800/80 bg-slate-950 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <p>© 2026 VisionForge Open Source Project. All rights reserved.</p>
          <div className="flex items-center gap-6">
            <span>Clean Architecture</span>
            <span>Research Quality</span>
            <span>Modular Design</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
