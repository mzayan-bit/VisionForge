"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Activity,
  AlertCircle,
  BarChart2,
  CheckCircle2,
  ChevronRight,
  Clock,
  Cpu,
  Database,
  FileText,
  Filter,
  FlaskConical,
  GitCommit,
  Layers,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
  Tag,
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
  parent_experiment_id?: string;
  created_at: string;
  updated_at: string;
}

interface ExperimentComparison {
  experiment_a_id: string;
  experiment_b_id: string;
  config_diff: Record<string, [any, any]>;
  metric_diff: Record<string, [any, any]>;
  summary_notes: string;
}

export default function ExperimentsPage() {
  const router = useRouter();

  // State
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [statusFilter, setStatusFilter] = useState<string>("ALL");
  const [selectedTag, setSelectedTag] = useState<string>("ALL");

  // Create Modal State
  const [showCreateModal, setShowCreateModal] = useState<boolean>(false);
  const [createForm, setCreateForm] = useState({
    name: "",
    purpose: "",
    hypothesis: "",
    dataset_id: "safety_v2",
    dataset_version: "v2.0",
    preparation_id: "prep_12",
    random_seed: 42,
    tags: "baseline, yolo11",
  });
  const [creating, setCreating] = useState<boolean>(false);

  // Compare Modal State
  const [showCompareModal, setShowCompareModal] = useState<boolean>(false);
  const [compareExpA, setCompareExpA] = useState<string>("");
  const [compareExpB, setCompareExpB] = useState<string>("");
  const [comparisonResult, setComparisonResult] = useState<ExperimentComparison | null>(null);
  const [comparing, setComparing] = useState<boolean>(false);

  useEffect(() => {
    fetchExperiments();
  }, []);

  const fetchExperiments = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/v1/experiments");
      if (res.ok) {
        const data = await res.json();
        setExperiments(data);
        if (data.length >= 2) {
          setCompareExpA(data[0].experiment_id);
          setCompareExpB(data[1].experiment_id);
        }
      }
    } catch (err) {
      console.error("Failed to fetch experiments:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateExperiment = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);

    try {
      const tagsList = createForm.tags.split(",").map((t) => t.trim()).filter(Boolean);

      const res = await fetch("/api/v1/experiments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: createForm.name,
          purpose: createForm.purpose,
          hypothesis: createForm.hypothesis,
          tags: tagsList,
          dataset_id: createForm.dataset_id,
          dataset_version: createForm.dataset_version,
          preparation_id: createForm.preparation_id,
          random_seed: createForm.random_seed,
        }),
      });

      if (res.ok) {
        const newExp = await res.json();
        setShowCreateModal(false);
        setCreateForm({
          name: "",
          purpose: "",
          hypothesis: "",
          dataset_id: "safety_v2",
          dataset_version: "v2.0",
          preparation_id: "prep_12",
          random_seed: 42,
          tags: "baseline, yolo11",
        });
        fetchExperiments();
        router.push(`/experiments/${newExp.experiment_id}`);
      }
    } catch (err) {
      console.error("Failed to create experiment:", err);
    } finally {
      setCreating(false);
    }
  };

  const handleCompareSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!compareExpA || !compareExpB) return;

    setComparing(true);
    try {
      const res = await fetch("/api/v1/experiments/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          experiment_a_id: compareExpA,
          experiment_b_id: compareExpB,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setComparisonResult(data);
      }
    } catch (err) {
      console.error("Failed to compare experiments:", err);
    } finally {
      setComparing(false);
    }
  };

  // Filter experiments
  const filteredExperiments = experiments.filter((exp) => {
    if (statusFilter !== "ALL" && exp.status !== statusFilter) return false;
    if (selectedTag !== "ALL" && !exp.tags.includes(selectedTag)) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      return (
        exp.name.toLowerCase().includes(q) ||
        exp.experiment_id.toLowerCase().includes(q) ||
        (exp.dataset_id && exp.dataset_id.toLowerCase().includes(q))
      );
    }
    return true;
  });

  const allTags = Array.from(new Set(experiments.flatMap((e) => e.tags)));

  return (
    <div className="flex flex-col min-h-screen bg-[#0a0a0a] text-neutral-200 font-inter">
      <PageHeader
        title="Experiment Tracker & Lineage System"
        description="Trace research experiments from Dataset Version -> Preparation -> Training Run -> Model -> Evaluation -> Benchmark -> Inference."
        breadcrumbs={["VisionForge", "Experiments"]}
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              icon={<BarChart2 className="w-4 h-4 text-purple-400" />}
              onClick={() => setShowCompareModal(true)}
            >
              Compare Experiments
            </Button>
            <Button
              variant="primary"
              icon={<Plus className="w-4 h-4" />}
              onClick={() => setShowCreateModal(true)}
            >
              Create Experiment
            </Button>
          </div>
        }
      />

      <div className="p-6 space-y-6 flex-1">
        {/* Filters & Search Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-4 bg-[#121212] border border-white/10 rounded-xl p-4">
          <div className="flex items-center gap-3 flex-1 min-w-[280px]">
            <div className="relative flex-1">
              <Search className="w-4 h-4 text-neutral-500 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search experiments by name, ID, or dataset..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg pl-9 pr-4 py-2 text-xs text-white placeholder-neutral-500 focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Status Filter */}
            <div className="flex items-center gap-2 text-xs text-neutral-400">
              <Filter className="w-3.5 h-3.5 text-blue-400" />
              <span>Status:</span>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white"
              >
                <option value="ALL">All Statuses</option>
                <option value="COMPLETED">Completed</option>
                <option value="RUNNING">Running</option>
                <option value="DRAFT">Draft</option>
                <option value="FAILED">Failed</option>
              </select>
            </div>

            {/* Tag Filter */}
            {allTags.length > 0 && (
              <div className="flex items-center gap-2 text-xs text-neutral-400">
                <Tag className="w-3.5 h-3.5 text-purple-400" />
                <span>Tag:</span>
                <select
                  value={selectedTag}
                  onChange={(e) => setSelectedTag(e.target.value)}
                  className="bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white"
                >
                  <option value="ALL">All Tags</option>
                  {allTags.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </div>

        {/* Experiments Grid / List */}
        <div className="bg-[#121212] border border-white/10 rounded-xl overflow-hidden">
          <div className="p-4 border-b border-white/10 flex justify-between items-center bg-[#161616]">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 flex items-center gap-2">
              <FlaskConical className="w-4 h-4 text-blue-400" />
              Tracked Research Experiments ({filteredExperiments.length})
            </h3>
            <span className="text-[11px] text-neutral-500 font-mono">
              Immutable Configuration Snapshots Active
            </span>
          </div>

          <div className="divide-y divide-white/5">
            {filteredExperiments.map((exp) => (
              <div
                key={exp.experiment_id}
                className="p-5 hover:bg-[#161616] transition-colors flex flex-col md:flex-row md:items-center justify-between gap-4"
              >
                <div className="space-y-2 flex-1">
                  <div className="flex items-center gap-3">
                    <Link
                      href={`/experiments/${exp.experiment_id}`}
                      className="text-sm font-semibold text-white hover:text-blue-400 transition-colors"
                    >
                      {exp.name}
                    </Link>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white/5 border border-white/10 text-neutral-400">
                      {exp.experiment_id}
                    </span>

                    {/* Status Badge */}
                    <span
                      className={`text-[10px] font-semibold px-2 py-0.5 rounded uppercase font-mono ${
                        exp.status === "COMPLETED"
                          ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                          : exp.status === "RUNNING"
                          ? "bg-blue-500/15 text-blue-400 border border-blue-500/30 animate-pulse"
                          : "bg-neutral-800 text-neutral-400"
                      }`}
                    >
                      {exp.status}
                    </span>

                    {exp.parent_experiment_id && (
                      <span className="text-[10px] px-2 py-0.5 rounded bg-purple-500/15 text-purple-400 border border-purple-500/30 font-mono">
                        Reproduction of {exp.parent_experiment_id}
                      </span>
                    )}
                  </div>

                  <p className="text-xs text-neutral-400 line-clamp-1">
                    {exp.purpose || exp.description || "No description provided."}
                  </p>

                  <div className="flex flex-wrap items-center gap-4 text-xs text-neutral-500 font-mono pt-1">
                    <span className="flex items-center gap-1.5 text-neutral-400">
                      <Database className="w-3.5 h-3.5 text-emerald-400" />
                      {exp.dataset_id || "safety_v2"} ({exp.dataset_version || "v2.0"})
                    </span>
                    <span className="flex items-center gap-1.5 text-neutral-400">
                      <GitCommit className="w-3.5 h-3.5 text-purple-400" />
                      {exp.environment_snapshot.git_commit_sha.substring(0, 8)}
                    </span>
                    <span className="flex items-center gap-1.5 text-neutral-400">
                      <Clock className="w-3.5 h-3.5 text-blue-400" />
                      {new Date(exp.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>

                {/* Right Action Trigger */}
                <div className="flex items-center gap-3 shrink-0">
                  <div className="flex gap-1.5">
                    {exp.tags.map((t) => (
                      <span
                        key={t}
                        className="text-[10px] bg-[#1a1a1a] text-neutral-400 px-2 py-1 rounded border border-white/5 font-mono"
                      >
                        #{t}
                      </span>
                    ))}
                  </div>

                  <Link href={`/experiments/${exp.experiment_id}`}>
                    <Button
                      variant="secondary"
                      size="sm"
                      icon={<ChevronRight className="w-3.5 h-3.5" />}
                    >
                      Lineage & Telemetry
                    </Button>
                  </Link>
                </div>
              </div>
            ))}

            {!loading && filteredExperiments.length === 0 && (
              <div className="p-12 text-center text-xs text-neutral-500 space-y-3">
                <FlaskConical className="w-8 h-8 mx-auto text-neutral-600" />
                <div>No matching experiments found.</div>
                <Button variant="secondary" size="sm" onClick={() => setShowCreateModal(true)}>
                  Create First Experiment
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ─── CREATE EXPERIMENT MODAL ─────────────────────────────────────── */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[#121212] border border-white/10 rounded-xl p-6 w-full max-w-xl space-y-5">
            <div className="flex justify-between items-center border-b border-white/10 pb-3">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <FlaskConical className="w-4 h-4 text-blue-400" />
                Initialize Research Experiment
              </h3>
              <button
                onClick={() => setShowCreateModal(false)}
                className="text-neutral-500 hover:text-white"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCreateExperiment} className="space-y-4 text-xs">
              <div>
                <label className="text-neutral-400 block mb-1 font-medium">Experiment Name</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. YOLO11s Safety Helmet Baseline"
                  value={createForm.name}
                  onChange={(e) => setCreateForm({ ...createForm, name: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-2 text-white"
                />
              </div>

              <div>
                <label className="text-neutral-400 block mb-1 font-medium">Research Goal / Purpose</label>
                <input
                  type="text"
                  placeholder="e.g. Compare YOLO11s baseline against RT-DETR-L on safety_v2"
                  value={createForm.purpose}
                  onChange={(e) => setCreateForm({ ...createForm, purpose: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-2 text-white"
                />
              </div>

              <div>
                <label className="text-neutral-400 block mb-1 font-medium">Hypothesis</label>
                <textarea
                  rows={2}
                  placeholder="e.g. Vision Transformer global attention will reduce false negatives in occluded helmet detection."
                  value={createForm.hypothesis}
                  onChange={(e) => setCreateForm({ ...createForm, hypothesis: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-2 text-white"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-neutral-400 block mb-1">Dataset ID</label>
                  <input
                    type="text"
                    value={createForm.dataset_id}
                    onChange={(e) => setCreateForm({ ...createForm, dataset_id: e.target.value })}
                    className="w-full bg-[#1a1a1a] border border-white/10 rounded px-3 py-1.5 text-white font-mono"
                  />
                </div>
                <div>
                  <label className="text-neutral-400 block mb-1">Dataset Version</label>
                  <input
                    type="text"
                    value={createForm.dataset_version}
                    onChange={(e) =>
                      setCreateForm({ ...createForm, dataset_version: e.target.value })
                    }
                    className="w-full bg-[#1a1a1a] border border-white/10 rounded px-3 py-1.5 text-white font-mono"
                  />
                </div>
              </div>

              <div>
                <label className="text-neutral-400 block mb-1 font-medium">Tags (comma separated)</label>
                <input
                  type="text"
                  value={createForm.tags}
                  onChange={(e) => setCreateForm({ ...createForm, tags: e.target.value })}
                  className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-2 text-white font-mono"
                />
              </div>

              <div className="flex justify-end gap-3 pt-3 border-t border-white/10">
                <Button variant="secondary" size="sm" onClick={() => setShowCreateModal(false)}>
                  Cancel
                </Button>
                <Button variant="primary" size="sm" disabled={creating}>
                  {creating ? "Initializing..." : "Create Experiment"}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ─── COMPARE EXPERIMENTS MODAL ──────────────────────────────────── */}
      {showCompareModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[#121212] border border-white/10 rounded-xl p-6 w-full max-w-2xl space-y-5">
            <div className="flex justify-between items-center border-b border-white/10 pb-3">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-purple-400" />
                Compare Research Experiments & Config Diff
              </h3>
              <button
                onClick={() => setShowCompareModal(false)}
                className="text-neutral-500 hover:text-white"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCompareSubmit} className="space-y-4 text-xs">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-neutral-400 block mb-1 font-medium">Experiment A</label>
                  <select
                    value={compareExpA}
                    onChange={(e) => setCompareExpA(e.target.value)}
                    className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-2 text-white"
                  >
                    {experiments.map((e) => (
                      <option key={`ca_${e.experiment_id}`} value={e.experiment_id}>
                        {e.name} ({e.experiment_id})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-neutral-400 block mb-1 font-medium">Experiment B</label>
                  <select
                    value={compareExpB}
                    onChange={(e) => setCompareExpB(e.target.value)}
                    className="w-full bg-[#1a1a1a] border border-white/10 rounded-lg px-3 py-2 text-white"
                  >
                    {experiments.map((e) => (
                      <option key={`cb_${e.experiment_id}`} value={e.experiment_id}>
                        {e.name} ({e.experiment_id})
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <Button variant="primary" size="sm" className="w-full" disabled={comparing}>
                {comparing ? "Comparing..." : "Run Config Diff & Comparison"}
              </Button>
            </form>

            {comparisonResult && (
              <div className="space-y-3 pt-3 border-t border-white/10 text-xs">
                <div className="bg-[#181818] p-3 rounded font-mono text-blue-400">
                  {comparisonResult.summary_notes}
                </div>

                <div className="space-y-2">
                  <h4 className="font-semibold text-white">Parameter Differences (Config Diff):</h4>
                  <div className="bg-[#080808] border border-white/5 rounded p-3 font-mono text-neutral-300">
                    {Object.entries(comparisonResult.config_diff).map(([key, vals]) => (
                      <div key={key} className="flex justify-between py-1 border-b border-white/5">
                        <span className="text-neutral-400">{key}:</span>
                        <span>
                          <span className="text-blue-400">{String(vals[0])}</span> vs{" "}
                          <span className="text-purple-400">{String(vals[1])}</span>
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
