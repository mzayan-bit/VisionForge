"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Settings,
  Server,
  Shield,
  Cpu,
  Activity,
  CheckCircle2,
  AlertCircle,
  Clock,
  RefreshCw,
  Layers,
  Database,
  Terminal,
  Zap,
  Info,
  Radio,
  ExternalLink,
  Sliders,
  Search,
  Video,
  FileText,
  AlertTriangle,
  XCircle,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";

// ─── Interfaces ───────────────────────────────────────────────────

interface SubsystemsHealth {
  api: string;
  storage: string;
  job_queue: string;
  model_registry: string;
  visual_memory?: string;
}

interface HealthData {
  status: string;
  version: string;
  service: string;
  environment: string;
  uptime_seconds: number;
  subsystems: SubsystemsHealth;
  ai_core: Record<string, any>;
}

interface DependencyItem {
  name: string;
  status: string;
  category: string;
  configured: boolean;
  detail: string;
}

interface DependencyHealthReport {
  overall_status: string;
  service: string;
  timestamp: string;
  dependencies: Record<string, DependencyItem>;
}

interface FailureRecord {
  failure_id: string;
  timestamp: string;
  service: string;
  error_code: string;
  message: string;
  request_id?: string;
  job_id?: string;
  details: Record<string, any>;
}

interface JobRecord {
  job_id: string;
  job_type: string;
  name: string;
  status: "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED";
  created_at: string;
  started_at?: string;
  completed_at?: string;
  duration_seconds?: number;
  progress_pct: number;
  error_code?: string;
  error_summary?: string;
  request_id?: string;
  metadata: Record<string, any>;
}

interface CVOperationalMetrics {
  total_inferences: number;
  avg_inference_latency_ms: number;
  total_search_queries: number;
  avg_search_latency_ms: number;
  total_video_frames_processed: number;
  total_active_models_loaded: number;
}

interface SystemDiagnosticsSnapshot {
  timestamp: string;
  uptime_seconds: number;
  total_requests: number;
  total_errors: number;
  error_rate_pct: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  active_jobs_count: number;
  queued_jobs_count: number;
  failed_jobs_count: number;
  storage_healthy: boolean;
  cv_metrics: CVOperationalMetrics;
  recent_jobs: JobRecord[];
  recent_failures: FailureRecord[];
}

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [depReport, setDepReport] = useState<DependencyHealthReport | null>(null);
  const [diagnostics, setDiagnostics] = useState<SystemDiagnosticsSnapshot | null>(null);
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedFailure, setSelectedFailure] = useState<FailureRecord | null>(null);

  useEffect(() => {
    fetchDiagnostics();
  }, []);

  const fetchDiagnostics = async () => {
    setLoading(true);
    try {
      const [healthRes, depRes, diagRes, jobsRes] = await Promise.all([
        fetch("/api/v1/health"),
        fetch("/api/v1/health/dependencies"),
        fetch("/api/v1/system/diagnostics"),
        fetch("/api/v1/system/jobs"),
      ]);

      if (healthRes.ok) {
        const hData = await healthRes.json();
        setHealth(hData.data || hData);
      }
      if (depRes.ok) {
        const dpData = await depRes.json();
        setDepReport(dpData.data || dpData);
      }
      if (diagRes.ok) {
        const dData = await diagRes.json();
        setDiagnostics(dData.data || dData);
      }
      if (jobsRes.ok) {
        const jData = await jobsRes.json();
        setJobs(jData.data || jData || []);
      }
    } catch (e) {
      console.error("Failed loading diagnostics:", e);
    } finally {
      setLoading(false);
    }
  };

  const formatUptime = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hrs}h ${mins}m ${secs}s`;
  };

  const getStatusBadge = (status: string) => {
    const s = status.toLowerCase();
    if (s === "healthy" || s === "ok" || s === "completed" || s === "ready") {
      return (
        <span className="flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full text-emerald-400 bg-emerald-950/50 border border-emerald-800/40">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          {status.toUpperCase()}
        </span>
      );
    }
    if (s === "running" || s === "queued") {
      return (
        <span className="flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full text-sky-400 bg-sky-950/50 border border-sky-800/40 animate-pulse">
          <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
          {status.toUpperCase()}
        </span>
      );
    }
    if (s === "disabled") {
      return (
        <span className="flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full text-neutral-400 bg-neutral-900 border border-neutral-800">
          <span className="w-1.5 h-1.5 rounded-full bg-neutral-500" />
          DISABLED
        </span>
      );
    }
    return (
      <span className="flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full text-rose-400 bg-rose-950/50 border border-rose-800/40">
        <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
        {status.toUpperCase()}
      </span>
    );
  };

  return (
    <div className="space-y-6 pb-16 font-sans">
      {/* Header with Quick Actions */}
      <div className="flex items-center justify-between">
        <PageHeader
          title="System Observability & Reliability Center"
          description="Workbench platform health probes, background job monitoring, CV telemetry, and subsystem diagnostics."
          breadcrumbs={["VisionForge", "Observability"]}
        />
        <div className="flex items-center gap-3">
          <Link href="/metrics" target="_blank">
            <Button
              variant="outline"
              size="sm"
              className="border-neutral-700 bg-neutral-900/60 hover:bg-neutral-800 text-neutral-300 flex items-center gap-1.5"
            >
              <Radio className="w-3.5 h-3.5 text-indigo-400" />
              Prometheus Metrics
              <ExternalLink className="w-3 h-3 ml-0.5 text-neutral-500" />
            </Button>
          </Link>
          <Button
            variant="ghost"
            size="sm"
            onClick={fetchDiagnostics}
            className="text-neutral-400 hover:text-white flex items-center gap-1.5"
            title="Refresh Telemetry"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} /> Refresh
          </Button>
        </div>
      </div>

      {/* System Status Banner */}
      <div className="p-4 rounded-xl bg-gradient-to-r from-neutral-900 via-neutral-900/90 to-indigo-950/30 border border-white/10 flex flex-wrap items-center justify-between gap-4 font-mono text-xs">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <Server className="w-4 h-4 text-indigo-400" />
            <span className="text-neutral-400">Service:</span>
            <span className="text-white font-semibold">{health?.service || "visionforge-backend"}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-neutral-400">Version:</span>
            <span className="text-neutral-200">v{health?.version || "0.1.0"}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-neutral-400">Env:</span>
            <Badge variant="info" size="sm">
              {health?.environment || "development"}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-emerald-400" />
            <span className="text-neutral-400">Target Device:</span>
            <span className="text-emerald-400 font-medium uppercase">{health?.ai_core?.optimal_device || "auto"}</span>
          </div>
        </div>
        <div className="flex items-center gap-2 text-neutral-400">
          <Clock className="w-3.5 h-3.5 text-neutral-500" />
          <span>Uptime:</span>
          <span className="text-neutral-200 font-semibold">{health ? formatUptime(health.uptime_seconds) : "—"}</span>
        </div>
      </div>

      {/* Dependency Health Grid */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white font-geist flex items-center gap-2">
            <Shield className="w-4 h-4 text-emerald-400" />
            Subsystem & Dependency Health Matrix
          </h3>
          <span className="text-xs font-mono text-neutral-400">
            Overall: {getStatusBadge(depReport?.overall_status || health?.status || "healthy")}
          </span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3">
          {depReport && depReport.dependencies ? (
            Object.entries(depReport.dependencies).map(([key, dep]) => (
              <div
                key={key}
                className="p-3.5 rounded-xl bg-neutral-900/80 border border-white/10 space-y-1.5"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-medium text-neutral-300 truncate">
                    {dep.name}
                  </span>
                  {getStatusBadge(dep.status)}
                </div>
                <p className="text-[11px] font-mono text-neutral-400 line-clamp-2">
                  {dep.detail || (dep.configured ? "Configured & Active" : "Disabled")}
                </p>
              </div>
            ))
          ) : (
            [
              { label: "API Gateway", status: "healthy", detail: "FastAPI REST Server" },
              { label: "Storage & Cache", status: "healthy", detail: "Persistent Data Volume" },
              { label: "Job Queue", status: "healthy", detail: "Thread-Safe Async Queue" },
              { label: "Model Registry", status: "healthy", detail: "Local Checkpoint Store" },
              { label: "Visual Memory", status: "healthy", detail: "768D NumPy Vector Matrix" },
            ].map((item) => (
              <div key={item.label} className="p-3.5 rounded-xl bg-neutral-900/80 border border-white/10 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-medium text-neutral-300">{item.label}</span>
                  {getStatusBadge(item.status)}
                </div>
                <p className="text-[11px] font-mono text-neutral-400">{item.detail}</p>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Operational Telemetry & CV Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono">
        <div className="p-4 rounded-xl bg-neutral-900/80 border border-white/10">
          <span className="text-[10px] text-neutral-500 block uppercase">TOTAL API REQUESTS</span>
          <span className="text-xl font-semibold text-white">
            {diagnostics?.total_requests?.toLocaleString() ?? "0"}
          </span>
          <span className="text-[10px] text-neutral-400 block mt-1">
            Uptime: {diagnostics ? formatUptime(diagnostics.uptime_seconds) : "—"}
          </span>
        </div>

        <div className="p-4 rounded-xl bg-neutral-900/80 border border-white/10">
          <span className="text-[10px] text-neutral-500 block uppercase">ERROR RATE</span>
          <span
            className={`text-xl font-semibold ${
              (diagnostics?.error_rate_pct ?? 0) > 1.0 ? "text-rose-400" : "text-emerald-400"
            }`}
          >
            {diagnostics?.error_rate_pct ?? 0.0}%
          </span>
          <span className="text-[10px] text-neutral-400 block mt-1">
            {diagnostics?.total_errors ?? 0} errors recorded
          </span>
        </div>

        <div className="p-4 rounded-xl bg-neutral-900/80 border border-white/10">
          <span className="text-[10px] text-neutral-500 block uppercase">AVG API LATENCY</span>
          <span className="text-xl font-semibold text-white">
            {diagnostics?.avg_latency_ms ?? 0} ms
          </span>
          <span className="text-[10px] text-neutral-400 block mt-1">P95: {diagnostics?.p95_latency_ms ?? 0} ms</span>
        </div>

        <div className="p-4 rounded-xl bg-neutral-900/80 border border-white/10">
          <span className="text-[10px] text-neutral-500 block uppercase">CV INFERENCES</span>
          <span className="text-xl font-semibold text-indigo-400">
            {diagnostics?.cv_metrics?.total_inferences ?? 0}
          </span>
          <span className="text-[10px] text-neutral-400 block mt-1">
            Avg: {diagnostics?.cv_metrics?.avg_inference_latency_ms ?? 0} ms
          </span>
        </div>
      </div>

      {/* Background Jobs Observatory */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-sky-400" />
              <h3 className="text-sm font-semibold text-white font-geist">Background Jobs & Workload Observatory</h3>
            </div>
            <span className="text-xs font-mono text-neutral-400">
              {jobs.length} recorded workloads
            </span>
          </div>
        </CardHeader>
        <CardBody className="p-0">
          {jobs.length > 0 ? (
            <div className="overflow-x-auto font-mono text-xs">
              <table className="w-full text-left">
                <thead className="border-b border-white/10 bg-white/[0.02] text-neutral-400">
                  <tr>
                    <th className="py-2.5 px-4">Job ID</th>
                    <th className="py-2.5 px-4">Name / Operation</th>
                    <th className="py-2.5 px-4">Type</th>
                    <th className="py-2.5 px-4">Status</th>
                    <th className="py-2.5 px-4">Duration</th>
                    <th className="py-2.5 px-4">Progress</th>
                    <th className="py-2.5 px-4">Error / Notes</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {jobs.map((j) => (
                    <tr key={j.job_id} className="hover:bg-white/[0.02]">
                      <td className="py-2.5 px-4 text-neutral-300 font-semibold">{j.job_id}</td>
                      <td className="py-2.5 px-4 text-neutral-200">{j.name}</td>
                      <td className="py-2.5 px-4 text-neutral-400 uppercase text-[11px]">{j.job_type}</td>
                      <td className="py-2.5 px-4">{getStatusBadge(j.status)}</td>
                      <td className="py-2.5 px-4 text-neutral-400">
                        {j.duration_seconds ? `${j.duration_seconds}s` : "—"}
                      </td>
                      <td className="py-2.5 px-4 text-neutral-300">
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-neutral-800 rounded-full h-1.5 overflow-hidden">
                            <div
                              className="bg-indigo-500 h-full rounded-full"
                              style={{ width: `${j.progress_pct}%` }}
                            />
                          </div>
                          <span className="text-[10px]">{Math.round(j.progress_pct)}%</span>
                        </div>
                      </td>
                      <td className="py-2.5 px-4 text-neutral-400 max-w-[200px] truncate">
                        {j.error_summary ? (
                          <span className="text-rose-400">{j.error_summary}</span>
                        ) : (
                          <span className="text-neutral-500">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="p-8 text-center text-neutral-500 font-mono text-xs">
              No background jobs have executed in this session.
            </div>
          )}
        </CardBody>
      </Card>

      {/* Recent Failures & Diagnostic Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-rose-400" />
              <h3 className="text-sm font-semibold text-white font-geist">Recent Subsystem Failures & Error Log</h3>
            </div>
            <span className="text-xs font-mono text-neutral-400">
              {diagnostics?.recent_failures?.length ?? 0} events
            </span>
          </div>
        </CardHeader>
        <CardBody className="p-0">
          {diagnostics && diagnostics.recent_failures.length > 0 ? (
            <div className="overflow-x-auto font-mono text-xs">
              <table className="w-full text-left">
                <thead className="border-b border-white/10 bg-white/[0.02] text-neutral-400">
                  <tr>
                    <th className="py-2.5 px-4">Time</th>
                    <th className="py-2.5 px-4">Service</th>
                    <th className="py-2.5 px-4">Error Code</th>
                    <th className="py-2.5 px-4">Request ID</th>
                    <th className="py-2.5 px-4">Message</th>
                    <th className="py-2.5 px-4">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {diagnostics.recent_failures.map((f) => (
                    <tr key={f.failure_id} className="hover:bg-white/[0.02]">
                      <td className="py-2.5 px-4 text-neutral-400">
                        {f.timestamp.split("T")[1]?.slice(0, 8)}
                      </td>
                      <td className="py-2.5 px-4 text-neutral-300 uppercase text-[11px]">{f.service}</td>
                      <td className="py-2.5 px-4 text-rose-400 font-semibold">{f.error_code}</td>
                      <td className="py-2.5 px-4 text-neutral-400 font-mono text-[11px]">
                        {f.request_id ? f.request_id.slice(0, 12) + "..." : "—"}
                      </td>
                      <td className="py-2.5 px-4 text-neutral-300 max-w-[280px] truncate">{f.message}</td>
                      <td className="py-2.5 px-4">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setSelectedFailure(f)}
                          className="text-[11px] h-6 px-2 text-indigo-400 hover:text-indigo-300"
                        >
                          Inspect
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="p-8 text-center text-neutral-500 font-mono text-xs">
              Zero system failures recorded. Platform running optimally.
            </div>
          )}
        </CardBody>
      </Card>

      {/* Failure Diagnostic Modal */}
      {selectedFailure && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-neutral-900 border border-neutral-700 rounded-xl max-w-xl w-full p-6 space-y-4 font-mono text-xs shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <div className="flex items-center gap-2 text-rose-400">
                <AlertCircle className="w-4 h-4" />
                <span className="font-semibold text-sm">Diagnostic Failure Detail</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedFailure(null)}
                className="text-neutral-400 hover:text-white"
              >
                Close
              </Button>
            </div>

            <div className="space-y-2 text-neutral-300">
              <div>
                <span className="text-neutral-500 block">FAILURE ID</span>
                <span>{selectedFailure.failure_id}</span>
              </div>
              <div>
                <span className="text-neutral-500 block">ERROR CODE</span>
                <span className="text-rose-400 font-bold">{selectedFailure.error_code}</span>
              </div>
              <div>
                <span className="text-neutral-500 block">REQUEST ID</span>
                <span className="text-sky-400">{selectedFailure.request_id || "N/A"}</span>
              </div>
              <div>
                <span className="text-neutral-500 block">MESSAGE</span>
                <span>{selectedFailure.message}</span>
              </div>
              <div>
                <span className="text-neutral-500 block">METADATA / DETAILS</span>
                <pre className="p-3 bg-black/60 rounded border border-white/5 overflow-x-auto text-[11px] text-neutral-300">
                  {JSON.stringify(selectedFailure.details, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
