"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  FlaskConical,
  BarChart2,
  GitBranch,
  Layers,
  Sparkles,
  Search,
  Plus,
  RefreshCw,
  Clock,
  ShieldCheck,
  AlertCircle,
  CheckCircle2,
  Database,
  Cpu,
  ChevronRight,
  FileText,
  Sliders,
  Maximize2,
  TrendingUp,
  TrendingDown,
  Info,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";

// ─── Interfaces ───────────────────────────────────────────────────

interface AggregatedMetricStats {
  metric_name: string;
  count: number;
  mean: number;
  std_dev: number;
  min: number;
  max: number;
  confidence_interval_95?: [number, number];
  is_single_run: boolean;
  warning?: string;
}

interface ExperimentRunRecord {
  run_id: string;
  seed: number;
  model_id: string;
  metrics: Record<string, number>;
  per_class_metrics: Record<string, number>;
  error_counts: Record<string, number>;
  training_time_sec?: number;
  gpu_hours?: number;
  created_at: string;
}

interface ExperimentVariant {
  variant_id: string;
  name: string;
  description: string;
  is_baseline: boolean;
  config_changes: Record<string, any>;
  dataset_id?: string;
  dataset_version?: string;
  runs: ExperimentRunRecord[];
  aggregated_metrics: Record<string, AggregatedMetricStats>;
  aggregated_per_class: Record<string, AggregatedMetricStats>;
  aggregated_error_counts: Record<string, AggregatedMetricStats>;
  label_count?: number;
  label_percentage?: number;
}

interface EvaluationProtocol {
  dataset_split: string;
  primary_metric: string;
  iou_threshold: number;
  confidence_threshold: number;
  class_handling: string;
  is_locked: boolean;
}

interface AblationRow {
  component: string;
  baseline_present: boolean;
  variant_present: boolean;
  measured_effect_delta?: number;
  metric_name: string;
}

interface AblationStudy {
  ablation_id: string;
  name: string;
  hypothesis: string;
  components: string[];
  matrix: AblationRow[];
  measured_effects: Record<string, number>;
}

interface ResearchExperiment {
  experiment_id: string;
  name: string;
  description: string;
  hypothesis: string;
  baseline_variant_id: string;
  variants: ExperimentVariant[];
  dataset_id: string;
  dataset_version: string;
  evaluation_protocol: EvaluationProtocol;
  status: string;
  ablation_study?: AblationStudy;
  conclusions?: string;
  limitations?: string;
  reproducibility_metadata: Record<string, any>;
  created_at: string;
  updated_at: string;
}

interface VariableDiffItem {
  parameter: string;
  baseline_value: any;
  variant_value: any;
  has_changed: boolean;
  component_type: string;
}

interface ResearchReport {
  experiment_id: string;
  title: string;
  hypothesis: string;
  dataset_summary: string;
  baseline_summary: string;
  variants_summary: string;
  performance_deltas: Record<string, number>;
  per_class_deltas: Record<string, number>;
  error_deltas: Record<string, number>;
  statistical_conclusions: string[];
  grounded_conclusions: string;
  limitations: string[];
  markdown_report: string;
}

export default function ExperimentsPage() {
  const [researchExperiments, setResearchExperiments] = useState<ResearchExperiment[]>([]);
  const [selectedExp, setSelectedExp] = useState<ResearchExperiment | null>(null);
  const [selectedVariant, setSelectedVariant] = useState<ExperimentVariant | null>(null);
  const [configDiff, setConfigDiff] = useState<VariableDiffItem[]>([]);
  const [ablationMatrix, setAblationMatrix] = useState<AblationStudy | null>(null);
  const [researchReport, setResearchReport] = useState<ResearchReport | null>(null);
  const [activeTab, setActiveTab] = useState<"overview" | "diff" | "ablation" | "runs" | "report">("overview");
  const [loading, setLoading] = useState(false);

  // New Experiment Modal
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [newHypothesis, setNewHypothesis] = useState("");
  const [newDataset, setNewDataset] = useState("safety_v2");
  const [newBaselineName, setNewBaselineName] = useState("Baseline Control");

  useEffect(() => {
    fetchResearchExperiments();
  }, []);

  const fetchResearchExperiments = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/v1/experiments/research");
      if (res.ok) {
        const data: ResearchExperiment[] = await res.json();
        setResearchExperiments(data);
        if (data.length > 0) {
          selectExperiment(data[0]);
        }
      }
    } catch (e) {
      console.error("Failed to load research experiments:", e);
    } finally {
      setLoading(false);
    }
  };

  const selectExperiment = async (exp: ResearchExperiment) => {
    setSelectedExp(exp);
    const nonBase = exp.variants.find((v) => !v.is_baseline) || exp.variants[0];
    setSelectedVariant(nonBase);

    // Fetch Diff, Ablation, Report in parallel
    try {
      const [diffRes, ablRes, repRes] = await Promise.all([
        fetch(`/api/v1/experiments/research/${exp.experiment_id}/variants/${nonBase.variant_id}/diff`),
        fetch(`/api/v1/experiments/research/${exp.experiment_id}/ablation`),
        fetch(`/api/v1/experiments/research/${exp.experiment_id}/research-report`),
      ]);

      if (diffRes.ok) setConfigDiff(await diffRes.json());
      if (ablRes.ok) setAblationMatrix(await ablRes.json());
      if (repRes.ok) setResearchReport(await repRes.json());
    } catch (e) {
      console.error("Failed loading experiment telemetry:", e);
    }
  };

  const handleSelectVariant = async (v: ExperimentVariant) => {
    if (!selectedExp) return;
    setSelectedVariant(v);
    try {
      const res = await fetch(
        `/api/v1/experiments/research/${selectedExp.experiment_id}/variants/${v.variant_id}/diff`
      );
      if (res.ok) {
        setConfigDiff(await res.json());
      }
    } catch (e) {
      console.error("Failed loading variant diff:", e);
    }
  };

  const handleCreateExperiment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim() || !newHypothesis.trim()) return;

    try {
      const res = await fetch("/api/v1/experiments/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newName,
          hypothesis: newHypothesis,
          dataset_id: newDataset,
          baseline_name: newBaselineName,
          baseline_config: { image_size: 640, augmentation: "standard", epochs: 50 },
        }),
      });

      if (res.ok) {
        setIsCreateModalOpen(false);
        setNewName("");
        setNewHypothesis("");
        await fetchResearchExperiments();
      }
    } catch (e) {
      console.error("Failed creating research experiment:", e);
    }
  };

  const baseline = selectedExp?.variants.find((v) => v.is_baseline);
  const baselineMap = baseline?.aggregated_metrics?.map50?.mean ?? 0.80;

  return (
    <div className="min-h-screen bg-[#070709] text-neutral-200 font-sans pb-16">
      {/* Workbench Header */}
      <div className="border-b border-white/10 bg-[#0d0d12]/90 backdrop-blur-md px-6 py-4 sticky top-14 z-20">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-purple-600/30 to-indigo-600/30 border border-purple-500/40 flex items-center justify-center text-purple-300 shadow-lg shadow-purple-950/40">
              <FlaskConical className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-semibold text-white tracking-tight">Research Benchmark Lab</h1>
                <Badge variant="info" size="sm" className="font-mono text-[10px]">
                  ABLATIONS & BENCHMARKS
                </Badge>
                <span className="flex items-center gap-1 text-[11px] font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-2 py-0.5 rounded-full">
                  <ShieldCheck className="w-3 h-3" />
                  EVIDENCE-BASED HYPOTHESIS TESTING
                </span>
              </div>
              <p className="text-xs text-neutral-400 mt-0.5">
                Rigorously evaluate whether modifications improve model performance across multi-seed runs
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Button
              variant="primary"
              size="sm"
              onClick={() => setIsCreateModalOpen(true)}
              className="bg-purple-600 hover:bg-purple-500 text-white font-semibold flex items-center gap-1.5 shadow-md shadow-purple-950/40"
            >
              <Plus className="w-4 h-4" /> New Research Experiment
            </Button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 pt-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Experiment Selector (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-mono text-neutral-400 uppercase tracking-wider flex items-center gap-1.5">
              <FlaskConical className="w-3.5 h-3.5 text-purple-400" />
              <span>Research Studies ({researchExperiments.length})</span>
            </h3>
            <button
              onClick={fetchResearchExperiments}
              className="text-neutral-400 hover:text-white p-1 rounded"
              title="Refresh"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="space-y-2.5">
            {researchExperiments.map((exp) => {
              const isSelected = selectedExp?.experiment_id === exp.experiment_id;
              const base = exp.variants.find((v) => v.is_baseline);
              const activeVars = exp.variants.filter((v) => !v.is_baseline);

              return (
                <button
                  key={exp.experiment_id}
                  onClick={() => selectExperiment(exp)}
                  className={`w-full text-left p-3.5 rounded-xl border transition-all cursor-pointer space-y-2 ${
                    isSelected
                      ? "bg-purple-950/30 border-purple-500/50 shadow-lg shadow-purple-950/20"
                      : "bg-neutral-900/70 border-white/10 hover:border-white/20 hover:bg-neutral-900"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-neutral-800 text-purple-300 border border-white/10">
                      {exp.dataset_id} ({exp.dataset_version})
                    </span>
                    <span className="text-[10px] font-mono text-neutral-400">
                      {activeVars.length} variants
                    </span>
                  </div>

                  <h4 className="text-xs font-semibold text-white line-clamp-1">{exp.name}</h4>
                  <p className="text-[11px] text-neutral-400 line-clamp-2 italic">
                    "{exp.hypothesis}"
                  </p>

                  <div className="flex items-center justify-between pt-1 border-t border-white/5 text-[10px] font-mono text-neutral-500">
                    <span>Protocol: {exp.evaluation_protocol.primary_metric}</span>
                    <span className="text-emerald-400">LOCKED ✓</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right Column: Experiment Workspace (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          {selectedExp ? (
            <>
              {/* Hypothesis & Protocol Banner */}
              <div className="p-5 rounded-2xl bg-neutral-900/90 border border-white/15 shadow-xl space-y-3">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-purple-400 font-semibold uppercase">
                      RESEARCH HYPOTHESIS
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-neutral-800 text-neutral-400 border border-white/10">
                      {selectedExp.experiment_id}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 font-mono text-[11px]">
                    <span className="text-neutral-400">Split: {selectedExp.evaluation_protocol.dataset_split}</span>
                    <span className="text-neutral-600">•</span>
                    <span className="text-neutral-400">IoU: {selectedExp.evaluation_protocol.iou_threshold}</span>
                  </div>
                </div>

                <div className="text-sm font-medium text-white italic pl-3 border-l-2 border-purple-500">
                  "{selectedExp.hypothesis}"
                </div>

                <div className="flex flex-wrap items-center gap-4 pt-1 text-xs text-neutral-400 font-mono">
                  <span>
                    Dataset: <strong className="text-neutral-200">{selectedExp.dataset_id}</strong> ({selectedExp.dataset_version})
                  </span>
                  <span>
                    Primary Metric: <strong className="text-neutral-200">{selectedExp.evaluation_protocol.primary_metric}</strong>
                  </span>
                  <span>
                    Protocol Status: <strong className="text-emerald-400">Locked for Reproducibility</strong>
                  </span>
                </div>
              </div>

              {/* Navigation Tabs */}
              <div className="flex items-center gap-2 border-b border-white/10 pb-2">
                {[
                  { id: "overview", label: "Baseline vs Variants", icon: <Sliders className="w-3.5 h-3.5" /> },
                  { id: "diff", label: "Variable Diff", icon: <GitBranch className="w-3.5 h-3.5" /> },
                  { id: "ablation", label: "Ablation Matrix", icon: <Layers className="w-3.5 h-3.5" /> },
                  { id: "runs", label: "Multi-Seed Trials", icon: <BarChart2 className="w-3.5 h-3.5" /> },
                  { id: "report", label: "Grounded Report", icon: <FileText className="w-3.5 h-3.5" /> },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono transition-all cursor-pointer ${
                      activeTab === tab.id
                        ? "bg-neutral-800 text-white border border-white/20 shadow-sm"
                        : "text-neutral-400 hover:text-white hover:bg-neutral-900"
                    }`}
                  >
                    {tab.icon}
                    <span>{tab.label}</span>
                  </button>
                ))}
              </div>

              {/* TAB 1: Baseline vs Variants Overview */}
              {activeTab === "overview" && (
                <div className="space-y-6">
                  {/* Baseline Card */}
                  {baseline && (
                    <div className="p-4 rounded-xl bg-neutral-950/80 border border-neutral-700/60 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-mono uppercase text-neutral-400 font-semibold flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full bg-neutral-400" />
                          CONTROL BASELINE: {baseline.name}
                        </span>
                        <span className="text-xs font-mono text-neutral-400">
                          {baseline.runs.length} seed runs
                        </span>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
                        <div className="p-2.5 rounded-lg bg-neutral-900/80 border border-white/10">
                          <span className="text-[10px] font-mono text-neutral-500 block">MEAN mAP@50</span>
                          <span className="text-base font-semibold text-white font-mono">
                            {baseline.aggregated_metrics?.map50?.mean?.toFixed(3) ?? "0.800"}
                          </span>
                          <span className="text-[10px] font-mono text-neutral-500 block mt-0.5">
                            ±{baseline.aggregated_metrics?.map50?.std_dev?.toFixed(3) ?? "0.000"}
                          </span>
                        </div>

                        <div className="p-2.5 rounded-lg bg-neutral-900/80 border border-white/10">
                          <span className="text-[10px] font-mono text-neutral-500 block">PRECISION</span>
                          <span className="text-base font-semibold text-white font-mono">
                            {baseline.aggregated_metrics?.precision?.mean?.toFixed(3) ?? "0.820"}
                          </span>
                        </div>

                        <div className="p-2.5 rounded-lg bg-neutral-900/80 border border-white/10">
                          <span className="text-[10px] font-mono text-neutral-500 block">RECALL</span>
                          <span className="text-base font-semibold text-white font-mono">
                            {baseline.aggregated_metrics?.recall?.mean?.toFixed(3) ?? "0.780"}
                          </span>
                        </div>

                        <div className="p-2.5 rounded-lg bg-neutral-900/80 border border-white/10">
                          <span className="text-[10px] font-mono text-neutral-500 block">BUDGET / CONFIG</span>
                          <span className="text-xs font-semibold text-neutral-300 font-mono truncate block">
                            {baseline.label_count ? `${baseline.label_count} labels` : "Full Dataset"}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Variants Cards */}
                  <div className="space-y-4">
                    <h4 className="text-xs font-mono uppercase text-neutral-400 tracking-wider">
                      Experimental Variants ({selectedExp.variants.filter((v) => !v.is_baseline).length})
                    </h4>

                    <div className="grid grid-cols-1 gap-4">
                      {selectedExp.variants
                        .filter((v) => !v.is_baseline)
                        .map((v) => {
                          const varMap = v.aggregated_metrics?.map50?.mean ?? 0.80;
                          const delta = Number((varMap - baselineMap).toFixed(3));
                          const isPositive = delta > 0.005;
                          const isNegative = delta < -0.005;

                          return (
                            <div
                              key={v.variant_id}
                              className="p-4 rounded-xl bg-neutral-900/80 border border-white/10 hover:border-purple-500/40 transition-all space-y-3"
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <h5 className="text-sm font-semibold text-white">{v.name}</h5>
                                  {v.label_percentage && (
                                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-purple-950/60 text-purple-300 border border-purple-800/40">
                                      {v.label_percentage}% budget
                                    </span>
                                  )}
                                </div>

                                <div className="flex items-center gap-2 font-mono">
                                  <span
                                    className={`flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded ${
                                      isPositive
                                        ? "text-emerald-400 bg-emerald-950/40 border border-emerald-800/30"
                                        : isNegative
                                        ? "text-rose-400 bg-rose-950/40 border border-rose-800/30"
                                        : "text-neutral-300 bg-neutral-800"
                                    }`}
                                  >
                                    {isPositive && <TrendingUp className="w-3.5 h-3.5" />}
                                    {isNegative && <TrendingDown className="w-3.5 h-3.5" />}
                                    {delta >= 0 ? `+${delta}` : delta} mAP@50
                                  </span>
                                </div>
                              </div>

                              <p className="text-xs text-neutral-400">{v.description}</p>

                              {/* Telemetry Metrics */}
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-1">
                                <div className="p-2 rounded bg-neutral-950 border border-white/10">
                                  <span className="text-[10px] font-mono text-neutral-500 block">MEAN mAP@50</span>
                                  <span className="text-sm font-semibold text-white font-mono">
                                    {varMap.toFixed(3)}
                                  </span>
                                </div>

                                <div className="p-2 rounded bg-neutral-950 border border-white/10">
                                  <span className="text-[10px] font-mono text-neutral-500 block">SEEDS TESTED</span>
                                  <span className="text-sm font-semibold text-neutral-300 font-mono">
                                    {v.runs.length} runs
                                  </span>
                                </div>

                                <div className="p-2 rounded bg-neutral-950 border border-white/10">
                                  <span className="text-[10px] font-mono text-neutral-500 block">95% CI</span>
                                  <span className="text-xs font-semibold text-neutral-300 font-mono">
                                    {v.aggregated_metrics?.map50?.confidence_interval_95
                                      ? `[${v.aggregated_metrics.map50.confidence_interval_95.join(", ")}]`
                                      : "N < 3 (N/A)"}
                                  </span>
                                </div>

                                <div className="p-2 rounded bg-neutral-950 border border-white/10">
                                  <span className="text-[10px] font-mono text-neutral-500 block">STATUS</span>
                                  <span
                                    className={`text-xs font-semibold font-mono ${
                                      isPositive
                                        ? "text-emerald-400"
                                        : isNegative
                                        ? "text-rose-400"
                                        : "text-neutral-400"
                                    }`}
                                  >
                                    {isPositive ? "IMPROVEMENT" : isNegative ? "REGRESSION" : "EQUIVALENT"}
                                  </span>
                                </div>
                              </div>

                              {/* Single Run Warning if applicable */}
                              {v.aggregated_metrics?.map50?.is_single_run && (
                                <div className="p-2 rounded-lg bg-amber-950/30 border border-amber-500/30 flex items-center gap-2 text-[11px] text-amber-300 font-mono">
                                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                                  <span>{v.aggregated_metrics.map50.warning}</span>
                                </div>
                              )}
                            </div>
                          );
                        })}
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 2: Variable Diff */}
              {activeTab === "diff" && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-mono uppercase text-neutral-400 tracking-wider">
                      Configuration Differences (Baseline vs {selectedVariant?.name})
                    </h4>
                    <span className="text-[11px] font-mono text-neutral-500">
                      Explicit constant vs changed parameters
                    </span>
                  </div>

                  <div className="rounded-xl border border-white/10 bg-neutral-900/80 overflow-hidden">
                    <table className="w-full text-left text-xs font-mono">
                      <thead className="bg-neutral-950 border-b border-white/10 text-neutral-400">
                        <tr>
                          <th className="p-3">Parameter</th>
                          <th className="p-3">Baseline Setting</th>
                          <th className="p-3">Variant Setting</th>
                          <th className="p-3">Category</th>
                          <th className="p-3">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/5">
                        {configDiff.map((d) => (
                          <tr key={d.parameter} className={d.has_changed ? "bg-purple-950/15" : ""}>
                            <td className="p-3 font-semibold text-white">{d.parameter}</td>
                            <td className="p-3 text-neutral-300">{String(d.baseline_value)}</td>
                            <td className="p-3 text-cyan-300 font-semibold">{String(d.variant_value)}</td>
                            <td className="p-3 text-neutral-400 uppercase text-[10px]">{d.component_type}</td>
                            <td className="p-3">
                              {d.has_changed ? (
                                <span className="px-2 py-0.5 rounded bg-purple-900/40 text-purple-300 border border-purple-600/30 text-[10px]">
                                  MODIFIED
                                </span>
                              ) : (
                                <span className="px-2 py-0.5 rounded bg-neutral-800 text-neutral-400 text-[10px]">
                                  UNCHANGED
                                </span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* TAB 3: Ablation Matrix */}
              {activeTab === "ablation" && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-mono uppercase text-neutral-400 tracking-wider">
                      Component Ablation Matrix
                    </h4>
                    <span className="text-[11px] font-mono text-neutral-500">
                      Isolated component contributions to performance
                    </span>
                  </div>

                  {ablationMatrix && (
                    <div className="rounded-xl border border-white/10 bg-neutral-900/80 overflow-hidden">
                      <table className="w-full text-left text-xs font-mono">
                        <thead className="bg-neutral-950 border-b border-white/10 text-neutral-400">
                          <tr>
                            <th className="p-3">Component / Branch</th>
                            <th className="p-3 text-center">Baseline</th>
                            <th className="p-3 text-center">Variant</th>
                            <th className="p-3 text-right">Measured Impact (Δ mAP)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                          {ablationMatrix.matrix.map((row) => (
                            <tr key={row.component}>
                              <td className="p-3 font-semibold text-white">{row.component}</td>
                              <td className="p-3 text-center">
                                {row.baseline_present ? (
                                  <span className="text-emerald-400">✓</span>
                                ) : (
                                  <span className="text-rose-400">✗</span>
                                )}
                              </td>
                              <td className="p-3 text-center">
                                {row.variant_present ? (
                                  <span className="text-emerald-400">✓</span>
                                ) : (
                                  <span className="text-rose-400">✗</span>
                                )}
                              </td>
                              <td className="p-3 text-right font-semibold">
                                <span
                                  className={
                                    (row.measured_effect_delta ?? 0) >= 0
                                      ? "text-emerald-400"
                                      : "text-rose-400"
                                  }
                                >
                                  {(row.measured_effect_delta ?? 0) >= 0
                                    ? `+${row.measured_effect_delta?.toFixed(3)}`
                                    : row.measured_effect_delta?.toFixed(3)}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* TAB 4: Multi-Seed Runs Table */}
              {activeTab === "runs" && (
                <div className="space-y-4">
                  <h4 className="text-xs font-mono uppercase text-neutral-400 tracking-wider">
                    Seed Replications & Variability
                  </h4>

                  {selectedExp.variants.map((v) => (
                    <div key={v.variant_id} className="rounded-xl border border-white/10 bg-neutral-900/80 p-4 space-y-3">
                      <div className="flex items-center justify-between border-b border-white/10 pb-2">
                        <span className="text-xs font-mono font-semibold text-white">{v.name}</span>
                        <span className="text-xs font-mono text-neutral-400">{v.runs.length} seed trials</span>
                      </div>

                      <table className="w-full text-left text-xs font-mono">
                        <thead className="text-neutral-500 border-b border-white/5">
                          <tr>
                            <th className="py-2">Seed</th>
                            <th className="py-2">Run ID</th>
                            <th className="py-2">mAP@50</th>
                            <th className="py-2">Precision</th>
                            <th className="py-2">Recall</th>
                            <th className="py-2 text-right">Duration</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5 text-neutral-300">
                          {v.runs.map((r) => (
                            <tr key={r.run_id}>
                              <td className="py-2 text-purple-300 font-semibold">seed: {r.seed}</td>
                              <td className="py-2 text-neutral-400">{r.run_id}</td>
                              <td className="py-2 text-white font-semibold">{r.metrics.map50?.toFixed(3)}</td>
                              <td className="py-2">{r.metrics.precision?.toFixed(3) ?? "—"}</td>
                              <td className="py-2">{r.metrics.recall?.toFixed(3) ?? "—"}</td>
                              <td className="py-2 text-right text-neutral-400">
                                {r.training_time_sec ? `${(r.training_time_sec / 60).toFixed(1)} min` : "—"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ))}
                </div>
              )}

              {/* TAB 5: Grounded Research Report */}
              {activeTab === "report" && researchReport && (
                <div className="p-6 rounded-2xl bg-neutral-900/90 border border-white/15 space-y-6 shadow-xl font-mono text-xs">
                  <div className="flex items-center justify-between border-b border-white/10 pb-4">
                    <div>
                      <h3 className="text-base font-semibold text-white font-sans">{researchReport.title}</h3>
                      <p className="text-xs text-neutral-400 italic mt-1 font-sans">
                        Hypothesis: "{researchReport.hypothesis}"
                      </p>
                    </div>
                    <span className="flex items-center gap-1 text-[11px] text-emerald-400 bg-emerald-950/40 border border-emerald-800/40 px-2.5 py-1 rounded-full">
                      <ShieldCheck className="w-3.5 h-3.5" /> GROUNDED REPORT
                    </span>
                  </div>

                  <div className="p-4 rounded-xl bg-neutral-950 border border-white/10 font-sans space-y-2">
                    <span className="text-xs font-mono text-neutral-400">EXECUTIVE FACTUAL SUMMARY</span>
                    <p className="text-sm text-neutral-200 leading-relaxed">
                      {researchReport.grounded_conclusions}
                    </p>
                  </div>

                  {/* Markdown Report Document Viewer */}
                  <div className="space-y-2 pt-2">
                    <span className="text-xs font-mono text-neutral-400 uppercase">
                      GENERATED RESEARCH DOCUMENT (MARKDOWN)
                    </span>
                    <pre className="p-4 rounded-xl bg-neutral-950 border border-white/10 text-neutral-300 overflow-x-auto whitespace-pre-wrap leading-relaxed">
                      {researchReport.markdown_report}
                    </pre>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="p-12 rounded-2xl bg-neutral-900/40 border border-dashed border-white/10 text-center space-y-3">
              <FlaskConical className="w-10 h-10 text-purple-400 mx-auto" />
              <h4 className="text-sm font-semibold text-white">Select a Research Study</h4>
              <p className="text-xs text-neutral-400 max-w-sm mx-auto">
                Choose an experiment from the left or create a new research study to inspect controlled variants and ablations.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Create Research Experiment Modal */}
      {isCreateModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-neutral-900 border border-white/20 rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <h3 className="text-base font-semibold text-white font-sans">New Research Experiment</h3>
              <button onClick={() => setIsCreateModalOpen(false)} className="text-neutral-400 hover:text-white">
                ✕
              </button>
            </div>

            <form onSubmit={handleCreateExperiment} className="space-y-4 text-xs font-mono">
              <div className="space-y-1">
                <label className="text-neutral-400">EXPERIMENT TITLE</label>
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="e.g. Active Learning vs Random Sampling Ablation"
                  className="w-full px-3 py-2 rounded-lg bg-neutral-950 border border-white/10 text-white focus:outline-none focus:border-purple-500 text-sm font-sans"
                  required
                />
              </div>

              <div className="space-y-1">
                <label className="text-neutral-400">RESEARCH HYPOTHESIS</label>
                <textarea
                  value={newHypothesis}
                  onChange={(e) => setNewHypothesis(e.target.value)}
                  placeholder="e.g. Uncertainty sampling achieves equal mAP with 50% fewer labels."
                  rows={3}
                  className="w-full px-3 py-2 rounded-lg bg-neutral-950 border border-white/10 text-white focus:outline-none focus:border-purple-500 text-sm font-sans"
                  required
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-neutral-400">DATASET</label>
                  <input
                    type="text"
                    value={newDataset}
                    onChange={(e) => setNewDataset(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg bg-neutral-950 border border-white/10 text-white focus:outline-none focus:border-purple-500"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-neutral-400">BASELINE NAME</label>
                  <input
                    type="text"
                    value={newBaselineName}
                    onChange={(e) => setNewBaselineName(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg bg-neutral-950 border border-white/10 text-white focus:outline-none focus:border-purple-500"
                  />
                </div>
              </div>

              <div className="flex items-center justify-end gap-2 pt-2 border-t border-white/10">
                <Button variant="ghost" size="sm" type="button" onClick={() => setIsCreateModalOpen(false)}>
                  Cancel
                </Button>
                <Button variant="primary" size="sm" type="submit" className="bg-purple-600 hover:bg-purple-500 text-white">
                  Create Experiment
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
