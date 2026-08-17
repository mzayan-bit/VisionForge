"use client";

import React, { useEffect, useState } from "react";
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
  recent_failures: FailureRecord[];
}

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [diagnostics, setDiagnostics] = useState<SystemDiagnosticsSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedFailure, setSelectedFailure] = useState<FailureRecord | null>(null);

  useEffect(() => {
    fetchDiagnostics();
  }, []);

  const fetchDiagnostics = async () => {
    setLoading(true);
    try {
      const [healthRes, diagRes] = await Promise.all([
        fetch("/api/v1/health"),
        fetch("/api/v1/system/diagnostics"),
      ]);

      if (healthRes.ok) {
        const hData = await healthRes.json();
        setHealth(hData.data || hData);
      }
      if (diagRes.ok) {
        const dData = await diagRes.json();
        setDiagnostics(dData.data || dData);
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

  return (
    <div className="space-y-6 pb-16 font-sans">
      <div className="flex items-center justify-between">
        <PageHeader
          title="Settings & System Diagnostics"
          description="Workbench platform health, operational metrics, and subsystem telemetry."
          breadcrumbs={["VisionForge", "Settings"]}
        />
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

      {/* Subsystem Health Grid (Step 60) */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          {
            label: "API Gateway",
            status: health?.subsystems?.api || "healthy",
            detail: `v${health?.version || "1.0.0"} • ${health?.environment || "development"}`,
            icon: Server,
          },
          {
            label: "Storage & Cache",
            status: health?.subsystems?.storage || "healthy",
            detail: `${health?.ai_core?.cache_size_mb ?? 0} MB cached`,
            icon: Database,
          },
          {
            label: "Job Queue",
            status: health?.subsystems?.job_queue || "healthy",
            detail: `${diagnostics?.active_jobs_count ?? 0} active • ${diagnostics?.queued_jobs_count ?? 0} queued`,
            icon: Cpu,
          },
          {
            label: "Model Registry",
            status: health?.subsystems?.model_registry || "healthy",
            detail: `${health?.ai_core?.registered_models ?? 0} registered models`,
            icon: Layers,
          },
        ].map((item) => {
          const Icon = item.icon;
          const isHealthy = item.status === "healthy";

          return (
            <div
              key={item.label}
              className="p-4 rounded-xl bg-neutral-900/80 border border-white/10 space-y-2"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-neutral-400 flex items-center gap-1.5">
                  <Icon className="w-3.5 h-3.5 text-neutral-400" />
                  {item.label}
                </span>
                <span
                  className={`flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full ${
                    isHealthy
                      ? "text-emerald-400 bg-emerald-950/50 border border-emerald-800/40"
                      : "text-amber-300 bg-amber-950/50 border border-amber-800/40"
                  }`}
                >
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      isHealthy ? "bg-emerald-400" : "bg-amber-400"
                    }`}
                  />
                  {item.status.toUpperCase()}
                </span>
              </div>
              <p className="text-xs font-mono text-neutral-300">{item.detail}</p>
            </div>
          );
        })}
      </div>

      {/* Operational Metrics (Step 49 & 60) */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono">
        <div className="p-4 rounded-xl bg-neutral-900/80 border border-white/10">
          <span className="text-[10px] text-neutral-500 block uppercase">TOTAL REQUESTS</span>
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
          <span className="text-[10px] text-neutral-400 block mt-1">Rolling window (500 reqs)</span>
        </div>

        <div className="p-4 rounded-xl bg-neutral-900/80 border border-white/10">
          <span className="text-[10px] text-neutral-500 block uppercase">P95 LATENCY</span>
          <span className="text-xl font-semibold text-neutral-200">
            {diagnostics?.p95_latency_ms ?? 0} ms
          </span>
          <span className="text-[10px] text-neutral-400 block mt-1">95th percentile</span>
        </div>
      </div>

      {/* Recent Failures & Diagnostic Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-rose-400" />
              <h3 className="text-sm font-semibold text-white font-geist">Recent Subsystem Failures</h3>
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
                <thead className="bg-neutral-950 border-b border-white/10 text-neutral-400 text-[11px]">
                  <tr>
                    <th className="p-3">Timestamp</th>
                    <th className="p-3">Service</th>
                    <th className="p-3">Error Code</th>
                    <th className="p-3">Message</th>
                    <th className="p-3">Request ID</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {diagnostics.recent_failures.map((f) => (
                    <tr
                      key={f.failure_id}
                      onClick={() => setSelectedFailure(f)}
                      className="hover:bg-neutral-800/50 cursor-pointer transition-colors"
                    >
                      <td className="p-3 text-neutral-400 text-[11px]">
                        {new Date(f.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="p-3 text-purple-300 font-semibold">{f.service}</td>
                      <td className="p-3">
                        <span className="px-1.5 py-0.5 rounded bg-rose-950/60 text-rose-300 border border-rose-800/40 text-[10px]">
                          {f.error_code}
                        </span>
                      </td>
                      <td className="p-3 text-neutral-200 truncate max-w-xs">{f.message}</td>
                      <td className="p-3 text-neutral-500 text-[10px] truncate max-w-[120px]">
                        {f.request_id || "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="p-8 text-center text-xs text-neutral-400 font-mono space-y-1">
              <CheckCircle2 className="w-6 h-6 text-emerald-400 mx-auto mb-2" />
              <p className="text-white font-medium">Zero Failures Recorded</p>
              <p className="text-neutral-500">All subsystem operations are executing cleanly.</p>
            </div>
          )}
        </CardBody>
      </Card>

      {/* Failure Inspection Modal */}
      {selectedFailure && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-neutral-900 border border-white/20 rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl font-mono text-xs">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-rose-400" />
                <h3 className="text-sm font-semibold text-white font-sans">Failure Diagnostics</h3>
              </div>
              <button onClick={() => setSelectedFailure(null)} className="text-neutral-400 hover:text-white">
                ✕
              </button>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-neutral-400">
                <span>Service:</span>
                <span className="text-white">{selectedFailure.service}</span>
              </div>
              <div className="flex justify-between text-neutral-400">
                <span>Error Code:</span>
                <span className="text-rose-400 font-semibold">{selectedFailure.error_code}</span>
              </div>
              <div className="flex justify-between text-neutral-400">
                <span>Request ID:</span>
                <span className="text-neutral-300">{selectedFailure.request_id || "N/A"}</span>
              </div>
              <div className="flex justify-between text-neutral-400">
                <span>Timestamp:</span>
                <span className="text-neutral-300">{selectedFailure.timestamp}</span>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-neutral-950 border border-white/10 text-neutral-300 whitespace-pre-wrap font-sans text-xs leading-relaxed">
              {selectedFailure.message}
            </div>

            <div className="flex justify-end pt-2">
              <Button variant="ghost" size="sm" onClick={() => setSelectedFailure(null)}>
                Close
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
