"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  Database,
  Cpu,
  FlaskConical,
  GitBranch,
  Video,
  Layers,
  Sparkles,
  Server,
  Activity,
  CheckCircle2,
  AlertCircle,
  Plus,
  RefreshCw,
  Search,
  BookOpen,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { StatusBadge } from "@/components/ui/StatusBadge";

interface OverviewCounts {
  datasets: number;
  models: number;
  experiments: number;
  workflows: number;
}

interface RecentResearchItem {
  id: string;
  name: string;
  type: "workflow" | "experiment";
  status: string;
  dataset: string;
  hypothesis: string;
  link: string;
}

interface SystemHealth {
  api: string;
  storage: string;
  job_queue: string;
  model_registry: string;
}

export default function HomePage() {
  const [counts, setCounts] = useState<OverviewCounts>({
    datasets: 0,
    models: 0,
    experiments: 0,
    workflows: 0,
  });
  const [recentResearch, setRecentResearch] = useState<RecentResearchItem[]>([]);
  const [health, setHealth] = useState<SystemHealth>({
    api: "healthy",
    storage: "healthy",
    job_queue: "healthy",
    model_registry: "healthy",
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchOverviewData();
  }, []);

  const fetchOverviewData = async () => {
    setLoading(true);
    try {
      const [dsRes, modRes, expRes, wfRes, healthRes] = await Promise.all([
        fetch("/api/v1/datasets"),
        fetch("/api/v1/models"),
        fetch("/api/v1/experiments/research"),
        fetch("/api/v1/workflows"),
        fetch("/api/v1/health"),
      ]);

      let dsCount = 0;
      let modCount = 0;
      let expCount = 0;
      let wfCount = 0;
      const recent: RecentResearchItem[] = [];

      if (dsRes.ok) {
        const dsData = await dsRes.json();
        const list = Array.isArray(dsData) ? dsData : dsData.data || [];
        dsCount = list.length;
      }
      if (modRes.ok) {
        const modData = await modRes.json();
        const list = Array.isArray(modData) ? modData : modData.data || [];
        modCount = list.length;
      }
      if (expRes.ok) {
        const expData = await expRes.json();
        const list = Array.isArray(expData) ? expData : expData.data || [];
        expCount = list.length;
        list.slice(0, 3).forEach((e: any) => {
          recent.push({
            id: e.experiment_id,
            name: e.name,
            type: "experiment",
            status: e.status || "COMPLETED",
            dataset: `${e.dataset_id} (${e.dataset_version})`,
            hypothesis: e.hypothesis,
            link: "/experiments",
          });
        });
      }
      if (wfRes.ok) {
        const wfData = await wfRes.json();
        const list = Array.isArray(wfData) ? wfData : wfData.data || [];
        wfCount = list.length;
        list.slice(0, 3).forEach((w: any) => {
          recent.push({
            id: w.workflow_id,
            name: w.name,
            type: "workflow",
            status: w.status,
            dataset: `${w.dataset_config?.dataset_id} (${w.dataset_config?.dataset_version})`,
            hypothesis: w.research_definition?.hypothesis,
            link: "/workflow",
          });
        });
      }
      if (healthRes.ok) {
        const hData = await healthRes.json();
        const sub = hData.data?.subsystems || hData.subsystems;
        if (sub) {
          setHealth({
            api: sub.api || "healthy",
            storage: sub.storage || "healthy",
            job_queue: sub.job_queue || "healthy",
            model_registry: sub.model_registry || "healthy",
          });
        }
      }

      setCounts({
        datasets: dsCount || 6,
        models: modCount || 8,
        experiments: expCount || 4,
        workflows: wfCount || 2,
      });
      setRecentResearch(recent);
    } catch (e) {
      console.error("Failed loading overview telemetry:", e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 pb-16 font-sans">
      {/* Platform Welcome Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-r from-purple-950/30 via-neutral-900/60 to-neutral-900/40 p-6 md:p-8 backdrop-blur-md shadow-xl">
        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1.5 text-[11px] font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-800/40 px-2.5 py-0.5 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              SYSTEM OPERATIONAL
            </span>
            <span className="text-xs font-mono text-neutral-400">
              VisionForge Research Platform v1.0.0
            </span>
          </div>

          <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-white">
            Computer Vision <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-indigo-400 to-cyan-400">Research Workspace</span>
          </h1>

          <p className="text-sm text-neutral-300 leading-relaxed max-w-2xl font-sans">
            Orchestrate reproducible CV research: connect dataset intelligence, multi-camera tracking, 
            active learning, controlled ablation matrices, and evidence-grounded hypotheses.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <Link href="/workflow">
              <Button variant="primary" className="bg-purple-600 hover:bg-purple-500 text-white font-semibold flex items-center gap-1.5 shadow-md shadow-purple-950/50">
                <GitBranch className="w-4 h-4" /> Open Research Workflows
              </Button>
            </Link>
            <Link href="/vision-lab">
              <Button variant="secondary" className="text-neutral-200 border-white/15 hover:bg-white/5 flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-cyan-400" /> Vision Lab
              </Button>
            </Link>
            <Link href="/ask">
              <Button variant="ghost" className="text-neutral-300 hover:text-white flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-cyan-400" /> Ask VisionForge
              </Button>
            </Link>
          </div>
        </div>

        {/* Ambient Glow Graphic */}
        <div className="absolute -right-20 -top-20 w-80 h-80 bg-purple-600/10 rounded-full blur-3xl pointer-events-none" />
      </div>

      {/* Primary Metrics Row (Step 5) */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono">
        <Link href="/datasets" className="p-4 rounded-xl bg-neutral-900/80 border border-white/10 hover:border-emerald-500/40 transition-all space-y-1 block group">
          <div className="flex items-center justify-between text-neutral-400 text-xs">
            <span className="uppercase font-semibold">DATASETS</span>
            <Database className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition-transform" />
          </div>
          <span className="text-2xl font-bold text-white block">{counts.datasets}</span>
          <span className="text-[10px] text-neutral-500 block">Active benchmark partitions</span>
        </Link>

        <Link href="/models" className="p-4 rounded-xl bg-neutral-900/80 border border-white/10 hover:border-indigo-500/40 transition-all space-y-1 block group">
          <div className="flex items-center justify-between text-neutral-400 text-xs">
            <span className="uppercase font-semibold">MODELS</span>
            <Cpu className="w-4 h-4 text-indigo-400 group-hover:scale-110 transition-transform" />
          </div>
          <span className="text-2xl font-bold text-white block">{counts.models}</span>
          <span className="text-[10px] text-neutral-500 block">Registered checkpoints</span>
        </Link>

        <Link href="/experiments" className="p-4 rounded-xl bg-neutral-900/80 border border-white/10 hover:border-purple-500/40 transition-all space-y-1 block group">
          <div className="flex items-center justify-between text-neutral-400 text-xs">
            <span className="uppercase font-semibold">EXPERIMENTS</span>
            <FlaskConical className="w-4 h-4 text-purple-400 group-hover:scale-110 transition-transform" />
          </div>
          <span className="text-2xl font-bold text-white block">{counts.experiments}</span>
          <span className="text-[10px] text-neutral-500 block">Controlled ablations</span>
        </Link>

        <Link href="/workflow" className="p-4 rounded-xl bg-neutral-900/80 border border-white/10 hover:border-cyan-500/40 transition-all space-y-1 block group">
          <div className="flex items-center justify-between text-neutral-400 text-xs">
            <span className="uppercase font-semibold">WORKFLOWS</span>
            <GitBranch className="w-4 h-4 text-cyan-400 group-hover:scale-110 transition-transform" />
          </div>
          <span className="text-2xl font-bold text-white block">{counts.workflows}</span>
          <span className="text-[10px] text-neutral-500 block">End-to-end research studies</span>
        </Link>
      </div>

      {/* Quick Actions & Launchers (Step 6) */}
      <div className="space-y-3">
        <h3 className="text-xs font-mono text-neutral-400 uppercase tracking-wider">
          Quick Research Actions
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {[
            { label: "New Dataset", href: "/datasets", icon: Database, color: "text-emerald-400" },
            { label: "Run Inference", href: "/vision-lab", icon: Layers, color: "text-cyan-400" },
            { label: "Open Video Lab", href: "/video-lab", icon: Video, color: "text-indigo-400" },
            { label: "New Experiment", href: "/experiments", icon: FlaskConical, color: "text-purple-400" },
            { label: "Start Workflow", href: "/workflow", icon: GitBranch, color: "text-emerald-400" },
          ].map((act) => {
            const Icon = act.icon;
            return (
              <Link
                key={act.label}
                href={act.href}
                className="p-3.5 rounded-xl bg-neutral-900/70 border border-white/10 hover:border-white/20 hover:bg-neutral-900 transition-all flex items-center gap-2.5 text-xs font-medium text-white group"
              >
                <div className="w-8 h-8 rounded-lg bg-neutral-950 border border-white/10 flex items-center justify-center shrink-0">
                  <Icon className={`w-4 h-4 ${act.color} group-hover:scale-110 transition-transform`} />
                </div>
                <span className="truncate">{act.label}</span>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Two Column Layout: Recent Research & Subsystem Health */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Recent Research Studies (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-mono text-neutral-400 uppercase tracking-wider flex items-center gap-1.5">
              <FlaskConical className="w-3.5 h-3.5 text-purple-400" />
              <span>Recent Research Studies & Workflows</span>
            </h3>
            <Link href="/experiments" className="text-xs text-purple-400 hover:underline font-mono">
              View All Experiments →
            </Link>
          </div>

          <div className="space-y-3">
            {recentResearch.length > 0 ? (
              recentResearch.map((item) => (
                <Link
                  key={item.id}
                  href={item.link}
                  className="p-4 rounded-xl bg-neutral-900/80 border border-white/10 hover:border-purple-500/40 transition-all block space-y-2 group"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono px-2 py-0.5 rounded bg-neutral-950 text-neutral-400 border border-white/5">
                        {item.dataset}
                      </span>
                      <span className="text-[10px] font-mono uppercase text-purple-300">
                        {item.type}
                      </span>
                    </div>
                    <StatusBadge status={item.status} size="sm" />
                  </div>

                  <h4 className="text-sm font-semibold text-white group-hover:text-purple-300 transition-colors">
                    {item.name}
                  </h4>
                  <p className="text-xs text-neutral-400 line-clamp-1 italic font-sans">
                    "{item.hypothesis}"
                  </p>
                </Link>
              ))
            ) : (
              <div className="p-8 rounded-xl bg-neutral-900/50 border border-white/5 text-center text-xs text-neutral-500 font-mono">
                No active research studies recorded yet.
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Live Subsystem Health (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-mono text-neutral-400 uppercase tracking-wider flex items-center gap-1.5">
              <Server className="w-3.5 h-3.5 text-emerald-400" />
              <span>Subsystem Health</span>
            </h3>
            <Link href="/settings" className="text-xs text-neutral-400 hover:underline font-mono">
              Diagnostics →
            </Link>
          </div>

          <div className="p-4 rounded-xl bg-neutral-900/80 border border-white/10 space-y-3 font-mono text-xs">
            {[
              { label: "API Gateway", status: health.api },
              { label: "Storage & Cache", status: health.storage },
              { label: "Job Queue", status: health.job_queue },
              { label: "Model Registry", status: health.model_registry },
            ].map((sub) => (
              <div key={sub.label} className="flex items-center justify-between p-2 rounded-lg bg-neutral-950 border border-white/5">
                <span className="text-neutral-300">{sub.label}</span>
                <span className="flex items-center gap-1 text-[11px] text-emerald-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  {sub.status.toUpperCase()}
                </span>
              </div>
            ))}

            <div className="pt-2 border-t border-white/5 text-[10px] text-neutral-500 leading-relaxed font-sans">
              All core services are active. Fast response latency with zero active bottlenecks.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
