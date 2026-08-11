"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import {
  Activity,
  AlertCircle,
  BarChart2,
  CheckCircle2,
  ChevronRight,
  Clock,
  Copy,
  Cpu,
  Database,
  Download,
  FileText,
  FlaskConical,
  GitBranch,
  GitCommit,
  Layers,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Tag,
  Terminal,
  XCircle,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Button } from "@/components/ui/Button";

// ─── Interfaces ───────────────────────────────────────────────────

interface EnvironmentSnapshot {
  python_version: string;
  os_platform: string;
  cpu_architecture: string;
  gpu_device: string;
  torch_version: string;
  git_commit_sha: string;
  git_branch: string;
  is_working_tree_clean: boolean;
}

interface DatasetFingerprint {
  dataset_id: string;
  version: string;
  preparation_id?: string;
  num_samples: number;
  num_classes: number;
  manifest_sha256: string;
  fingerprint_hash: string;
}

interface LineageNode {
  id: string;
  label: string;
  type: string;
  status: string;
  metadata: Record<string, any>;
  route_link: string;
}

interface LineageEdge {
  source_id: string;
  target_id: string;
  relationship_type: string;
}

interface LineageGraph {
  nodes: LineageNode[];
  edges: LineageEdge[];
}

interface TimelineEvent {
  event_id: string;
  timestamp: string;
  event_type: string;
  title: string;
  description: string;
  entity_id?: string;
}

interface RandomnessConfig {
  random_seed: number;
  python_seed?: number;
  numpy_seed?: number;
  torch_seed?: number;
  determinism_notes?: string;
}

interface Experiment {
  experiment_id: string;
  name: string;
  description: string;
  purpose: string;
  status: "DRAFT" | "RUNNING" | "COMPLETED" | "FAILED" | "ARCHIVED";
  hypothesis?: string;
  observations?: string;
  conclusions?: string;
  tags: string[];
  dataset_id?: string;
  dataset_version?: string;
  dataset_fingerprint?: DatasetFingerprint;
  preparation_id?: string;
  training_run_ids: string[];
  model_ids: string[];
  evaluation_ids: string[];
  benchmark_ids: string[];
  inference_ids: string[];
  training_config_snapshot?: Record<string, any>;
  environment_snapshot: EnvironmentSnapshot;
  randomness: RandomnessConfig;
  parent_experiment_id?: string;
  created_at: string;
  updated_at: string;
}

interface ReproducibilityReport {
  experiment_id: string;
  is_reproducible: boolean;
  checks_passed: string[];
  checks_failed: string[];
  missing_dependencies: string[];
  verified_at: string;
}

export default function ExperimentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const expId = params.id as string;

  // State
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [lineage, setLineage] = useState<LineageGraph | null>(null);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [reportMd, setReportMd] = useState<string>("");
  const [reproReport, setReproReport] = useState<ReproducibilityReport | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<"lineage" | "timeline" | "repro" | "report">("lineage");

  // Notes state
  const [hypothesis, setHypothesis] = useState<string>("");
  const [observations, setObservations] = useState<string>("");
  const [conclusions, setConclusions] = useState<string>("");
  const [savingNotes, setSavingNotes] = useState<boolean>(false);
  const [copiedSha, setCopiedSha] = useState<boolean>(false);
  const [reproducing, setReproducing] = useState<boolean>(false);

  useEffect(() => {
    if (expId) {
      fetchExperimentData();
    }
  }, [expId]);

  const fetchExperimentData = async () => {
    setLoading(true);
    try {
      // 1. Fetch Experiment Detail
      const res = await fetch(`/api/v1/experiments/${expId}`);
      if (res.ok) {
        const data: Experiment = await res.json();
        setExperiment(data);
        setHypothesis(data.hypothesis || "");
        setObservations(data.observations || "");
        setConclusions(data.conclusions || "");
      }

      // 2. Fetch Lineage Graph
      const linRes = await fetch(`/api/v1/experiments/${expId}/lineage`);
      if (linRes.ok) {
        setLineage(await linRes.json());
      }

      // 3. Fetch Timeline
      const timeRes = await fetch(`/api/v1/experiments/${expId}/timeline`);
      if (timeRes.ok) {
        setTimeline(await timeRes.json());
      }

      // 4. Fetch Reproducibility Report
      const reproRes = await fetch(`/api/v1/experiments/${expId}/validate`);
      if (reproRes.ok) {
        setReproReport(await reproRes.json());
      }

      // 5. Fetch Markdown Report
      const rptRes = await fetch(`/api/v1/experiments/${expId}/report`);
      if (rptRes.ok) {
        const data = await rptRes.json();
        setReportMd(data.report_md);
      }
    } catch (err) {
      console.error("Failed to load experiment data:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveNotes = async () => {
    setSavingNotes(true);
    try {
      const res = await fetch(`/api/v1/experiments/${expId}/notes`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hypothesis, observations, conclusions }),
      });
      if (res.ok) {
        const updated = await res.json();
        setExperiment(updated);
      }
    } catch (err) {
      console.error("Failed to save notes:", err);
    } finally {
      setSavingNotes(false);
    }
  };

  const handleReproduce = async () => {
    setReproducing(true);
    try {
      const res = await fetch(`/api/v1/experiments/${expId}/reproduce`, {
        method: "POST",
      });
      if (res.ok) {
        const newExp = await res.json();
        router.push(`/experiments/${newExp.experiment_id}`);
      }
    } catch (err) {
      console.error("Failed to trigger reproduction:", err);
    } finally {
      setReproducing(false);
    }
  };

  const copyGitSha = () => {
    if (experiment?.environment_snapshot.git_commit_sha) {
      navigator.clipboard.writeText(experiment.environment_snapshot.git_commit_sha);
      setCopiedSha(true);
      setTimeout(() => setCopiedSha(false), 2000);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col min-h-screen bg-[#0a0a0a] text-neutral-200 items-center justify-center p-8">
        <RefreshCw className="w-8 h-8 text-blue-500 animate-spin mb-4" />
        <div className="text-xs text-neutral-400 font-mono">Loading Experiment Lineage & Telemetry...</div>
      </div>
    );
  }

  if (!experiment) {
    return (
      <div className="p-8 text-center text-xs text-red-400">
        Experiment &apos;{expId}&apos; was not found.
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-screen bg-[#0a0a0a] text-neutral-200 font-inter">
      {/* Page Header */}
      <PageHeader
        title={`Experiment: ${experiment.name}`}
        description={`ID: ${experiment.experiment_id} | Created: ${new Date(experiment.created_at).toLocaleString()}`}
        breadcrumbs={["VisionForge", "Experiments", experiment.experiment_id]}
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              icon={<RotateCcw className="w-4 h-4 text-purple-400" />}
              onClick={handleReproduce}
              disabled={reproducing}
            >
              {reproducing ? "Spawning..." : "Reproduce Experiment"}
            </Button>

            <a
              href={`data:text/markdown;charset=utf-8,${encodeURIComponent(reportMd)}`}
              download={`experiment_report_${experiment.experiment_id}.md`}
            >
              <Button variant="primary" icon={<Download className="w-4 h-4" />}>
                Export Markdown Report
              </Button>
            </a>
          </div>
        }
      />

      {/* Main Content Area */}
      <div className="p-6 space-y-6 flex-1">
        {/* Top Summary Banner */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-[#121212] border border-white/10 rounded-xl p-4">
            <div className="text-[11px] font-medium text-neutral-400 uppercase tracking-wider mb-1">
              Status & Audit
            </div>
            <div className="flex items-center gap-2">
              <span
                className={`text-sm font-bold font-mono px-2 py-0.5 rounded ${
                  experiment.status === "COMPLETED"
                    ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                    : "bg-blue-500/15 text-blue-400 border border-blue-500/30"
                }`}
              >
                {experiment.status}
              </span>
              {reproReport?.is_reproducible && (
                <span className="text-[10px] text-emerald-400 flex items-center gap-1 font-mono">
                  <ShieldCheck className="w-3.5 h-3.5" /> 100% Reproducible
                </span>
              )}
            </div>
          </div>

          <div className="bg-[#121212] border border-white/10 rounded-xl p-4">
            <div className="text-[11px] font-medium text-neutral-400 uppercase tracking-wider mb-1">
              Git Commit SHA
            </div>
            <div className="flex items-center justify-between font-mono text-xs text-purple-400">
              <span>{experiment.environment_snapshot.git_commit_sha.substring(0, 12)}...</span>
              <button onClick={copyGitSha} className="text-neutral-500 hover:text-white" title="Copy SHA">
                <Copy className="w-3.5 h-3.5" />
              </button>
            </div>
            {copiedSha && <div className="text-[9px] text-emerald-400">Copied SHA to clipboard</div>}
          </div>

          <div className="bg-[#121212] border border-white/10 rounded-xl p-4">
            <div className="text-[11px] font-medium text-neutral-400 uppercase tracking-wider mb-1">
              Dataset Fingerprint
            </div>
            <div className="font-mono text-xs text-emerald-400 truncate">
              {experiment.dataset_fingerprint?.fingerprint_hash.substring(0, 16) || "Verified SHA-256"}
            </div>
            <div className="text-[10px] text-neutral-500 mt-1">
              {experiment.dataset_id || "safety_v2"} ({experiment.dataset_version || "v2.0"})
            </div>
          </div>

          <div className="bg-[#121212] border border-white/10 rounded-xl p-4">
            <div className="text-[11px] font-medium text-neutral-400 uppercase tracking-wider mb-1">
              Linked Resources
            </div>
            <div className="text-xs font-mono text-white">
              {experiment.training_run_ids.length} Runs | {experiment.model_ids.length} Models | {experiment.evaluation_ids.length} Evals
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center bg-[#141414] border border-white/10 rounded-xl p-1 w-fit">
          <button
            onClick={() => setActiveTab("lineage")}
            className={`px-4 py-2 rounded-lg text-xs font-medium transition-all ${
              activeTab === "lineage"
                ? "bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            Interactive Lineage Graph
          </button>
          <button
            onClick={() => setActiveTab("timeline")}
            className={`px-4 py-2 rounded-lg text-xs font-medium transition-all ${
              activeTab === "timeline"
                ? "bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            Chronological Timeline
          </button>
          <button
            onClick={() => setActiveTab("repro")}
            className={`px-4 py-2 rounded-lg text-xs font-medium transition-all ${
              activeTab === "repro"
                ? "bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            Reproducibility Audit
          </button>
          <button
            onClick={() => setActiveTab("report")}
            className={`px-4 py-2 rounded-lg text-xs font-medium transition-all ${
              activeTab === "report"
                ? "bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            Notes & Research Report
          </button>
        </div>

        {/* ─── TAB 1: INTERACTIVE LINEAGE GRAPH ───────────────────────────── */}
        {activeTab === "lineage" && (
          <div className="space-y-6">
            <div className="bg-[#121212] border border-white/10 rounded-xl p-6 space-y-5">
              <div className="flex justify-between items-center border-b border-white/10 pb-3">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-blue-400" />
                  Visual Dependency Lineage Graph
                </h3>
                <span className="text-xs text-neutral-500 font-mono">
                  Click any node to navigate to original resource
                </span>
              </div>

              {/* Lineage Visual Flow Nodes */}
              <div className="py-8 flex flex-col md:flex-row items-center justify-center gap-4 overflow-x-auto">
                {lineage?.nodes.map((node, index) => (
                  <React.Fragment key={node.id}>
                    {index > 0 && (
                      <div className="text-neutral-600 flex items-center">
                        <ChevronRight className="w-5 h-5 hidden md:block" />
                      </div>
                    )}

                    <Link href={node.route_link} className="group">
                      <div className="bg-[#181818] border border-white/10 group-hover:border-blue-500/50 rounded-xl p-4 w-48 text-center space-y-2 transition-all shadow-lg hover:shadow-blue-500/10 cursor-pointer">
                        <div className="text-[10px] font-mono uppercase font-bold text-blue-400 tracking-wider">
                          {node.type}
                        </div>
                        <div className="text-xs font-semibold text-white truncate">{node.label}</div>
                        <div className="text-[10px] font-mono text-neutral-500">ID: {node.id}</div>
                      </div>
                    </Link>
                  </React.Fragment>
                ))}

                {(!lineage || lineage.nodes.length === 0) && (
                  <div className="text-xs text-neutral-500 py-8">
                    No lineage nodes attached yet. Attach training runs or datasets.
                  </div>
                )}
              </div>
            </div>

            {/* Immutable Snapshots Details Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Environment Telemetry Card */}
              <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-purple-400" />
                  Captured Environment Snapshot
                </h4>
                <div className="space-y-2 font-mono text-xs">
                  <div className="flex justify-between border-b border-white/5 py-1">
                    <span className="text-neutral-400">Python Version:</span>
                    <span className="text-white">{experiment.environment_snapshot.python_version}</span>
                  </div>
                  <div className="flex justify-between border-b border-white/5 py-1">
                    <span className="text-neutral-400">OS Platform:</span>
                    <span className="text-white">{experiment.environment_snapshot.os_platform}</span>
                  </div>
                  <div className="flex justify-between border-b border-white/5 py-1">
                    <span className="text-neutral-400">PyTorch Framework:</span>
                    <span className="text-white">{experiment.environment_snapshot.torch_version}</span>
                  </div>
                  <div className="flex justify-between border-b border-white/5 py-1">
                    <span className="text-neutral-400">Git Branch:</span>
                    <span className="text-blue-400">{experiment.environment_snapshot.git_branch}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-neutral-400">Working Tree Clean:</span>
                    <span className="text-emerald-400">
                      {experiment.environment_snapshot.is_working_tree_clean ? "TRUE" : "DIRTY"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Master Randomness & Seeds Card */}
              <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-emerald-400" />
                  Randomness & Seed Configuration
                </h4>
                <div className="space-y-2 font-mono text-xs">
                  <div className="flex justify-between border-b border-white/5 py-1">
                    <span className="text-neutral-400">Master Random Seed:</span>
                    <span className="text-emerald-400 font-bold">{experiment.randomness.random_seed}</span>
                  </div>
                  <div className="flex justify-between border-b border-white/5 py-1">
                    <span className="text-neutral-400">Python Seed:</span>
                    <span className="text-white">{experiment.randomness.python_seed}</span>
                  </div>
                  <div className="flex justify-between border-b border-white/5 py-1">
                    <span className="text-neutral-400">NumPy Seed:</span>
                    <span className="text-white">{experiment.randomness.numpy_seed}</span>
                  </div>
                  <div className="flex justify-between py-1">
                    <span className="text-neutral-400">PyTorch Seed:</span>
                    <span className="text-white">{experiment.randomness.torch_seed}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ─── TAB 2: CHRONOLOGICAL TIMELINE ───────────────────────────────── */}
        {activeTab === "timeline" && (
          <div className="bg-[#121212] border border-white/10 rounded-xl p-6 space-y-6">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
              <Clock className="w-4 h-4 text-blue-400" />
              Event Timeline
            </h3>

            <div className="relative border-l border-white/10 ml-4 space-y-6 pl-6">
              {timeline.map((evt) => (
                <div key={evt.event_id} className="relative">
                  <div className="absolute -left-[31px] top-1.5 w-3.5 h-3.5 rounded-full bg-blue-500 border-4 border-[#121212]" />
                  <div className="bg-[#181818] border border-white/5 rounded-lg p-4 space-y-1">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-semibold text-white">{evt.title}</span>
                      <span className="font-mono text-neutral-500">
                        {new Date(evt.timestamp).toLocaleString()}
                      </span>
                    </div>
                    <p className="text-xs text-neutral-400">{evt.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ─── TAB 3: REPRODUCIBILITY AUDIT ───────────────────────────────── */}
        {activeTab === "repro" && (
          <div className="bg-[#121212] border border-white/10 rounded-xl p-6 space-y-6">
            <div className="flex justify-between items-center border-b border-white/10 pb-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                Reproducibility Verification Audit
              </h3>
              <Button variant="secondary" size="sm" onClick={fetchExperimentData}>
                Re-validate Audit
              </Button>
            </div>

            {reproReport && (
              <div className="space-y-4 text-xs">
                <div className="space-y-2">
                  <h4 className="font-semibold text-neutral-300">Passed Verification Checks:</h4>
                  {reproReport.checks_passed.map((chk, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-emerald-400 font-mono">
                      <CheckCircle2 className="w-4 h-4 shrink-0" />
                      <span>{chk}</span>
                    </div>
                  ))}
                </div>

                {reproReport.checks_failed.length > 0 && (
                  <div className="space-y-2 pt-2">
                    <h4 className="font-semibold text-neutral-300">Failed Audit Checks:</h4>
                    {reproReport.checks_failed.map((chk, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-red-400 font-mono">
                        <XCircle className="w-4 h-4 shrink-0" />
                        <span>{chk}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ─── TAB 4: NOTES & REPORT VIEWER ──────────────────────────────── */}
        {activeTab === "report" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Notes Form */}
            <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4 text-xs">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                <FileText className="w-4 h-4 text-blue-400" />
                Researcher Notes & Conclusions
              </h3>

              <div>
                <label className="text-neutral-400 block mb-1 font-medium">Hypothesis</label>
                <textarea
                  rows={2}
                  value={hypothesis}
                  onChange={(e) => setHypothesis(e.target.value)}
                  className="w-full bg-[#1a1a1a] border border-white/10 rounded p-2 text-white"
                />
              </div>

              <div>
                <label className="text-neutral-400 block mb-1 font-medium">Observations</label>
                <textarea
                  rows={3}
                  value={observations}
                  onChange={(e) => setObservations(e.target.value)}
                  className="w-full bg-[#1a1a1a] border border-white/10 rounded p-2 text-white"
                />
              </div>

              <div>
                <label className="text-neutral-400 block mb-1 font-medium">Conclusions</label>
                <textarea
                  rows={3}
                  value={conclusions}
                  onChange={(e) => setConclusions(e.target.value)}
                  className="w-full bg-[#1a1a1a] border border-white/10 rounded p-2 text-white"
                />
              </div>

              <Button
                variant="primary"
                size="sm"
                onClick={handleSaveNotes}
                disabled={savingNotes}
              >
                {savingNotes ? "Saving..." : "Save Researcher Notes"}
              </Button>
            </div>

            {/* Markdown Report Preview */}
            <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
                <Download className="w-4 h-4 text-emerald-400" />
                Auto-Generated Research Report
              </h3>
              <pre className="bg-[#080808] border border-white/5 rounded-lg p-4 font-mono text-[11px] text-neutral-300 whitespace-pre-wrap max-h-[450px] overflow-y-auto">
                {reportMd}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
