"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowDown,
  ArrowRight,
  ArrowUp,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  Clock,
  Compass,
  Cpu,
  Database,
  Download,
  Eye,
  FileCode,
  FileText,
  Filter,
  Flame,
  GitBranch,
  GitCommit,
  HardDrive,
  History,
  Info,
  Layers,
  Play,
  RefreshCw,
  Sliders,
  Sparkles,
  TrendingDown,
  TrendingUp,
  X,
  Zap,
} from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";

// ─── Interfaces ───────────────────────────────────────────────────

interface PRCurvePoint {
  recall: number;
  precision: number;
}

interface PerClassMetrics {
  class_id: number;
  class_name: string;
  precision: number;
  recall: number;
  f1: number;
  map50: number;
  map75: number;
  map50_95: number;
  support: number;
  predictions_count: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  pr_curve_points: PRCurvePoint[];
}

interface ThresholdPoint {
  confidence_threshold: number;
  precision: number;
  recall: number;
  f1: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
}

interface DetectionMetrics {
  precision: number;
  recall: number;
  f1: number;
  mean_iou: number;
  map50: number;
  map75: number;
  map50_95: number;
  support_gt_count: number;
  total_predictions: number;
}

interface RuntimeMetrics {
  warmup_iterations: number;
  evaluated_iterations: number;
  preprocess_ms_mean: number;
  preprocess_ms_p95: number;
  inference_ms_mean: number;
  inference_ms_median: number;
  inference_ms_p95: number;
  postprocess_ms_mean: number;
  postprocess_ms_p95: number;
  total_latency_ms_mean: number;
  total_latency_ms_p95: number;
  throughput_fps: number;
  model_parameters_m?: number;
  model_size_mb?: number;
  device: string;
  device_name: string;
}

interface BenchmarkDatasetSnapshot {
  dataset_id: string;
  dataset_version: string;
  dataset_fingerprint: string;
  split_used: string;
  total_images: number;
  total_annotations: number;
  class_distribution: Record<string, number>;
}

interface ErrorPrediction {
  image_id: string;
  image_path: string;
  ground_truth_class?: string;
  predicted_class?: string;
  confidence?: number;
  iou?: number;
  error_type: string;
  gt_bbox?: number[];
  pred_bbox?: number[];
  sample_link?: string;
}

interface BenchmarkRun {
  benchmark_id: string;
  name: string;
  description: string;
  task: string;
  model_name: string;
  model_version: string;
  is_baseline: boolean;
  baseline_benchmark_id?: string;
  dataset_snapshot: BenchmarkDatasetSnapshot;
  config: {
    iou_threshold: number;
    confidence_threshold: number;
    nms_iou_threshold: number;
    img_size: number;
    batch_size: number;
    device: string;
    fp16: boolean;
    random_seed: number;
    warmup_iterations: number;
  };
  status: string;
  metrics: DetectionMetrics;
  per_class_metrics: PerClassMetrics[];
  threshold_analysis: ThresholdPoint[];
  runtime_metrics: RuntimeMetrics;
  errors_summary: Record<string, number>;
  reproducibility: Record<string, any>;
  created_at: string;
}

interface ModelComparisonResult {
  comparison_id: string;
  baseline_benchmark: BenchmarkRun;
  candidate_benchmark: BenchmarkRun;
  is_directly_comparable: boolean;
  incompatibility_reasons: string[];
  metric_deltas: Record<string, { baseline: number; candidate: number; delta_abs: number; delta_rel_pct: number }>;
  per_class_deltas: Record<string, { map50_delta: number; recall_delta: number; precision_delta: number }>;
  regression_status: "IMPROVED" | "REGRESSION" | "NEUTRAL" | "INCOMPARABLE";
  regression_notes: string[];
  failure_transitions: Record<string, number>;
  disagreement_samples: Array<{ image_id: string; observation: string; confidence?: number; pred_bbox?: number[] }>;
}

interface BenchmarkHistoryItem {
  benchmark_id: string;
  model_name: string;
  model_version: string;
  timestamp: string;
  map50: number;
  map50_95: number;
  precision: number;
  recall: number;
  f1: number;
  throughput_fps: number;
  total_latency_ms: number;
  dataset_version: string;
  is_baseline: boolean;
}

export default function ResearchBenchmarkLabPage() {
  // State
  const [benchmarks, setBenchmarks] = useState<BenchmarkRun[]>([]);
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState<string>("");
  const [candidateBenchmarkId, setCandidateBenchmarkId] = useState<string>("");
  const [comparisonResult, setComparisonResult] = useState<ModelComparisonResult | null>(null);
  const [failures, setFailures] = useState<ErrorPrediction[]>([]);
  const [history, setHistory] = useState<BenchmarkHistoryItem[]>([]);
  
  // Tabs: "overview" | "pr_curves" | "per_class" | "runtime" | "comparison" | "failures" | "history"
  const [activeTab, setActiveTab] = useState<string>("overview");
  const [filterErrorType, setFilterErrorType] = useState<string>("ALL");
  const [confidenceSlider, setConfidenceSlider] = useState<number>(0.25);
  const [showReproducibilityDrawer, setShowReproducibilityDrawer] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchBenchmarks();
    fetchHistory();
  }, []);

  const fetchBenchmarks = async () => {
    try {
      setLoading(true);
      const res = await fetch("/api/v1/benchmarks/runs");
      const json = await res.json();
      if (json.success && Array.isArray(json.data) && json.data.length > 0) {
        setBenchmarks(json.data);
        const baseline = json.data.find((b: BenchmarkRun) => b.is_baseline) || json.data[0];
        const candidate = json.data.find((b: BenchmarkRun) => !b.is_baseline) || json.data[1] || json.data[0];
        setSelectedBenchmarkId(baseline.benchmark_id);
        setCandidateBenchmarkId(candidate.benchmark_id);
        fetchFailures(baseline.benchmark_id);
      }
    } catch (err: any) {
      setError(err.message || "Failed to load benchmarks");
    } finally {
      setLoading(false);
    }
  };

  const fetchFailures = async (benchId: string) => {
    try {
      const res = await fetch(`/api/v1/benchmarks/runs/${benchId}/failures?limit=50`);
      const json = await res.json();
      if (json.success && Array.isArray(json.data)) {
        setFailures(json.data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch("/api/v1/benchmarks/history");
      const json = await res.json();
      if (json.success && Array.isArray(json.data)) {
        setHistory(json.data);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleCompare = async () => {
    if (!selectedBenchmarkId || !candidateBenchmarkId) return;
    try {
      setLoading(true);
      const res = await fetch("/api/v1/benchmarks/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          baseline_id: selectedBenchmarkId,
          candidate_id: candidateBenchmarkId,
          regression_threshold_map50: 0.02,
          regression_threshold_latency: 0.10,
        }),
      });
      const json = await res.json();
      if (json.success) {
        setComparisonResult(json.data);
        setActiveTab("comparison");
      } else {
        setError(json.message || "Model comparison failed");
      }
    } catch (err: any) {
      setError(err.message || "Comparison error");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = () => {
    if (!selectedBenchmarkId) return;
    window.open(`/api/v1/benchmarks/runs/${selectedBenchmarkId}/report`, "_blank");
  };

  const currentBenchmark = benchmarks.find((b) => b.benchmark_id === selectedBenchmarkId) || benchmarks[0];
  const candidateBenchmark = benchmarks.find((b) => b.benchmark_id === candidateBenchmarkId);

  // Find nearest threshold point from slider
  const selectedThresholdPoint = currentBenchmark?.threshold_analysis?.reduce((prev, curr) => {
    return Math.abs(curr.confidence_threshold - confidenceSlider) < Math.abs(prev.confidence_threshold - confidenceSlider) ? curr : prev;
  }, currentBenchmark.threshold_analysis[0]);

  const filteredFailures = filterErrorType === "ALL" 
    ? failures 
    : failures.filter((f) => f.error_type === filterErrorType);

  if (loading && benchmarks.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-[#0a0a0a] text-neutral-300 font-mono">
        <Activity className="w-8 h-8 text-blue-500 animate-spin mb-4" />
        <div>Loading Research Benchmark Lab...</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col min-h-screen bg-[#0a0a0a] text-neutral-200 font-inter">
      {/* Header */}
      <PageHeader
        title="Research Benchmark & Evaluation Lab"
        description="Rigorous, Reproducible Computer Vision Evaluation, Controlled Model Comparisons, PR Curves, and Latency Profiling"
        breadcrumbs={["VisionForge", "Benchmarks"]}
        actions={
          <div className="flex items-center gap-2 font-mono text-xs">
            <Button
              variant="secondary"
              icon={<Download className="w-4 h-4 text-emerald-400" />}
              onClick={handleDownloadReport}
            >
              Export Report
            </Button>

            <Button
              variant="secondary"
              icon={<GitCommit className="w-4 h-4 text-purple-400" />}
              onClick={() => setShowReproducibilityDrawer(true)}
            >
              Reproducibility
            </Button>

            <Button
              variant="primary"
              icon={<Zap className="w-4 h-4" />}
              onClick={handleCompare}
            >
              Compare Models
            </Button>
          </div>
        }
      />

      <div className="p-6 space-y-6 flex-1">
        {/* Top Control Bar: Benchmark & Candidate Selectors */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-[#121212] border border-white/10 rounded-xl p-4 font-mono text-xs shadow-xl">
          <div>
            <label className="text-neutral-400 block mb-1">Active Benchmark Run (Baseline)</label>
            <select
              value={selectedBenchmarkId}
              onChange={(e) => {
                setSelectedBenchmarkId(e.target.value);
                fetchFailures(e.target.value);
              }}
              className="w-full bg-[#181818] border border-white/10 rounded-lg p-2 text-white font-bold"
            >
              {benchmarks.map((b) => (
                <option key={b.benchmark_id} value={b.benchmark_id}>
                  {b.name} {b.is_baseline ? "★ (Baseline)" : ""}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-neutral-400 block mb-1">Candidate Model (For Comparison)</label>
            <select
              value={candidateBenchmarkId}
              onChange={(e) => setCandidateBenchmarkId(e.target.value)}
              className="w-full bg-[#181818] border border-white/10 rounded-lg p-2 text-white"
            >
              {benchmarks.map((b) => (
                <option key={b.benchmark_id} value={b.benchmark_id}>
                  {b.name} ({b.model_name})
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col justify-center border-l border-white/5 pl-4 space-y-1">
            <div className="text-neutral-500 text-[10px]">Dataset Snapshot</div>
            <div className="text-blue-400 font-bold">
              {currentBenchmark?.dataset_snapshot.dataset_id} ({currentBenchmark?.dataset_snapshot.dataset_version})
            </div>
            <div className="text-neutral-400 text-[10px]">
              Split: <span className="text-white font-bold">{currentBenchmark?.dataset_snapshot.split_used}</span> | Fingerprint: {currentBenchmark?.dataset_snapshot.dataset_fingerprint.slice(0, 10)}...
            </div>
          </div>
        </div>

        {/* Navigation Workspace Tabs */}
        <div className="flex flex-wrap items-center gap-2 border-b border-white/10 pb-3">
          {[
            { id: "overview", label: "Results Overview", icon: <BarChart3 className="w-4 h-4" /> },
            { id: "pr_curves", label: "PR & Threshold Curves", icon: <Sliders className="w-4 h-4" /> },
            { id: "per_class", label: "Per-Class Breakdown", icon: <Layers className="w-4 h-4" /> },
            { id: "runtime", label: "Runtime Latency Profiler", icon: <Cpu className="w-4 h-4" /> },
            { id: "comparison", label: "Model A vs B Comparison", icon: <Zap className="w-4 h-4 text-amber-400" /> },
            { id: "failures", label: `Failure Gallery (${failures.length})`, icon: <AlertCircle className="w-4 h-4 text-red-400" /> },
            { id: "history", label: "Progression History", icon: <History className="w-4 h-4 text-purple-400" /> },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl font-mono text-xs font-semibold transition-all ${
                activeTab === tab.id
                  ? "bg-blue-600 text-white shadow-lg shadow-blue-600/30"
                  : "bg-[#141414] hover:bg-[#1c1c1c] text-neutral-400 border border-white/5"
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* TAB 1: RESULTS OVERVIEW */}
        {activeTab === "overview" && currentBenchmark && (
          <div className="space-y-6">
            {/* KPI Cards Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
              <div className="bg-[#121212] border border-blue-500/30 rounded-xl p-4 space-y-1 font-mono">
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider">mAP@50:95</span>
                <div className="text-2xl font-bold text-blue-400">
                  {(currentBenchmark.metrics.map50_95 * 100).toFixed(1)}%
                </div>
                <span className="text-[10px] text-neutral-500">COCO Std Metric</span>
              </div>

              <div className="bg-[#121212] border border-emerald-500/30 rounded-xl p-4 space-y-1 font-mono">
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider">mAP@50</span>
                <div className="text-2xl font-bold text-emerald-400">
                  {(currentBenchmark.metrics.map50 * 100).toFixed(1)}%
                </div>
                <span className="text-[10px] text-neutral-500">VOC Std Metric</span>
              </div>

              <div className="bg-[#121212] border border-white/10 rounded-xl p-4 space-y-1 font-mono">
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider">Precision</span>
                <div className="text-2xl font-bold text-white">
                  {(currentBenchmark.metrics.precision * 100).toFixed(1)}%
                </div>
                <span className="text-[10px] text-neutral-500">@ Conf 0.25</span>
              </div>

              <div className="bg-[#121212] border border-white/10 rounded-xl p-4 space-y-1 font-mono">
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider">Recall</span>
                <div className="text-2xl font-bold text-white">
                  {(currentBenchmark.metrics.recall * 100).toFixed(1)}%
                </div>
                <span className="text-[10px] text-neutral-500">@ Conf 0.25</span>
              </div>

              <div className="bg-[#121212] border border-purple-500/30 rounded-xl p-4 space-y-1 font-mono">
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider">F1 Score</span>
                <div className="text-2xl font-bold text-purple-400">
                  {currentBenchmark.metrics.f1.toFixed(2)}
                </div>
                <span className="text-[10px] text-neutral-500">Harmonic Mean</span>
              </div>

              <div className="bg-[#121212] border border-amber-500/30 rounded-xl p-4 space-y-1 font-mono">
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider">Throughput</span>
                <div className="text-2xl font-bold text-amber-400">
                  {currentBenchmark.runtime_metrics.throughput_fps.toFixed(1)} <span className="text-xs">FPS</span>
                </div>
                <span className="text-[10px] text-neutral-500">Steady-State</span>
              </div>

              <div className="bg-[#121212] border border-cyan-500/30 rounded-xl p-4 space-y-1 font-mono">
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider">Total Latency</span>
                <div className="text-2xl font-bold text-cyan-400">
                  {currentBenchmark.runtime_metrics.total_latency_ms_mean.toFixed(1)} <span className="text-xs">ms</span>
                </div>
                <span className="text-[10px] text-neutral-500">p95: {currentBenchmark.runtime_metrics.total_latency_ms_p95.toFixed(1)}ms</span>
              </div>
            </div>

            {/* Main Split: Per-Class Summary & Diagnostic Failure Counts */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left 2 Cols: Class Performance Bar Matrix */}
              <div className="lg:col-span-2 bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4 font-mono text-xs shadow-xl">
                <h3 className="font-semibold text-white flex items-center justify-between border-b border-white/10 pb-3">
                  <span className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-blue-400" />
                    Class-Level Accuracy Breakdown
                  </span>
                  <span className="text-neutral-500 text-[10px]">
                    Evaluated on {currentBenchmark.dataset_snapshot.total_images} images
                  </span>
                </h3>

                <div className="space-y-4">
                  {currentBenchmark.per_class_metrics.map((cls) => (
                    <div key={cls.class_id} className="space-y-1.5 p-3 bg-[#181818] rounded-lg border border-white/5">
                      <div className="flex justify-between items-center font-bold">
                        <span className="text-white text-sm">{cls.class_name}</span>
                        <span className="text-blue-400">mAP@50: {(cls.map50 * 100).toFixed(1)}% | mAP@50:95: {(cls.map50_95 * 100).toFixed(1)}%</span>
                      </div>

                      <div className="grid grid-cols-4 gap-2 text-[11px] text-neutral-400 pt-1">
                        <div>Precision: <span className="text-white font-bold">{(cls.precision * 100).toFixed(1)}%</span></div>
                        <div>Recall: <span className="text-white font-bold">{(cls.recall * 100).toFixed(1)}%</span></div>
                        <div>F1: <span className="text-purple-400 font-bold">{cls.f1.toFixed(2)}</span></div>
                        <div>Support (GT): <span className="text-amber-400 font-bold">{cls.support}</span></div>
                      </div>

                      {/* Accuracy Visual Bar */}
                      <div className="w-full h-2 bg-[#252525] rounded-full overflow-hidden flex mt-2">
                        <div
                          style={{ width: `${cls.map50 * 100}%` }}
                          className="h-full bg-blue-500 rounded-full"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Right Col: Failure Breakdown Card */}
              <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4 font-mono text-xs shadow-xl">
                <h3 className="font-semibold text-white flex items-center justify-between border-b border-white/10 pb-3">
                  <span className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-red-400" />
                    Diagnostic Failure Counts
                  </span>
                  <span className="text-red-400 font-bold">{failures.length} Total</span>
                </h3>

                <div className="space-y-2">
                  {Object.entries(currentBenchmark.errors_summary).map(([errType, count]) => (
                    <div
                      key={errType}
                      onClick={() => {
                        setFilterErrorType(errType);
                        setActiveTab("failures");
                      }}
                      className="p-2.5 bg-[#181818] hover:bg-[#222] rounded-lg border border-white/5 flex justify-between items-center cursor-pointer transition-all"
                    >
                      <span className="text-neutral-300 font-bold">{errType}</span>
                      <span className="px-2 py-0.5 rounded bg-red-950/40 text-red-400 border border-red-500/30 font-bold">
                        {count}
                      </span>
                    </div>
                  ))}
                </div>

                <div className="pt-2">
                  <Button
                    variant="secondary"
                    className="w-full"
                    icon={<Eye className="w-4 h-4" />}
                    onClick={() => setActiveTab("failures")}
                  >
                    Open Failure Gallery
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: PR & THRESHOLD CURVES */}
        {activeTab === "pr_curves" && currentBenchmark && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* PR Curves Chart */}
            <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4 font-mono text-xs shadow-xl">
              <h3 className="font-semibold text-white flex items-center justify-between border-b border-white/10 pb-3">
                <span className="flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-blue-400" />
                  Precision-Recall (PR) Curves per Class
                </span>
                <span className="text-blue-400">COCO 101-pt Standard</span>
              </h3>

              <div className="aspect-[4/3] bg-[#0c0c0c] rounded-lg border border-white/5 p-4 flex flex-col justify-between relative">
                {/* SVG Curves */}
                <svg className="w-full h-full overflow-visible" viewBox="0 0 100 100" preserveAspectRatio="none">
                  {/* Grid Lines */}
                  <line x1="0" y1="25" x2="100" y2="25" stroke="#222" strokeWidth="0.5" strokeDasharray="2" />
                  <line x1="0" y1="50" x2="100" y2="50" stroke="#222" strokeWidth="0.5" strokeDasharray="2" />
                  <line x1="0" y1="75" x2="100" y2="75" stroke="#222" strokeWidth="0.5" strokeDasharray="2" />

                  {/* Render class curves */}
                  {currentBenchmark.per_class_metrics.map((cls, idx) => {
                    const color = idx === 0 ? "#60a5fa" : idx === 1 ? "#c084fc" : "#34d399";
                    const pts = cls.pr_curve_points;
                    if (!pts || pts.length < 2) return null;
                    const polyPoints = pts.map((p) => `${p.recall * 100},${100 - p.precision * 100}`).join(" ");
                    return (
                      <polyline
                        key={cls.class_id}
                        fill="none"
                        stroke={color}
                        strokeWidth="2.5"
                        points={polyPoints}
                      />
                    );
                  })}
                </svg>

                <div className="flex justify-between text-[10px] text-neutral-500 pt-2 border-t border-white/5">
                  <span>Recall: 0.0</span>
                  <span>0.5</span>
                  <span>1.0</span>
                </div>
              </div>

              {/* Legend */}
              <div className="flex flex-wrap gap-4 pt-2">
                {currentBenchmark.per_class_metrics.map((cls, idx) => {
                  const color = idx === 0 ? "bg-blue-400" : idx === 1 ? "bg-purple-400" : "bg-emerald-400";
                  return (
                    <div key={cls.class_id} className="flex items-center gap-1.5">
                      <span className={`w-3 h-3 rounded ${color}`} />
                      <span className="text-neutral-300 font-bold">{cls.class_name} (AP: {(cls.map50 * 100).toFixed(1)}%)</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Threshold Operating Point Analyzer */}
            <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4 font-mono text-xs shadow-xl">
              <h3 className="font-semibold text-white flex items-center justify-between border-b border-white/10 pb-3">
                <span className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-purple-400" />
                  Confidence Threshold Operating Point
                </span>
                <span className="text-purple-400">Threshold: {confidenceSlider.toFixed(2)}</span>
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="text-neutral-400 block mb-1.5">
                    Operating Confidence Threshold Slider: <span className="text-white font-bold">{confidenceSlider.toFixed(2)}</span>
                  </label>
                  <input
                    type="range"
                    min={0.10}
                    max={0.90}
                    step={0.05}
                    value={confidenceSlider}
                    onChange={(e) => setConfidenceSlider(parseFloat(e.target.value))}
                    className="w-full h-2 bg-[#1f1f1f] rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                </div>

                {selectedThresholdPoint && (
                  <div className="grid grid-cols-3 gap-3">
                    <div className="p-3 bg-[#181818] rounded-lg border border-white/5 text-center space-y-1">
                      <div className="text-neutral-500 text-[10px]">Precision @ {selectedThresholdPoint.confidence_threshold}</div>
                      <div className="text-xl font-bold text-white">{(selectedThresholdPoint.precision * 100).toFixed(1)}%</div>
                    </div>

                    <div className="p-3 bg-[#181818] rounded-lg border border-white/5 text-center space-y-1">
                      <div className="text-neutral-500 text-[10px]">Recall @ {selectedThresholdPoint.confidence_threshold}</div>
                      <div className="text-xl font-bold text-white">{(selectedThresholdPoint.recall * 100).toFixed(1)}%</div>
                    </div>

                    <div className="p-3 bg-[#181818] rounded-lg border border-white/5 text-center space-y-1">
                      <div className="text-neutral-500 text-[10px]">F1 Score</div>
                      <div className="text-xl font-bold text-purple-400">{selectedThresholdPoint.f1.toFixed(2)}</div>
                    </div>
                  </div>
                )}

                <div className="p-3 bg-[#181818] rounded-lg border border-white/5 space-y-2">
                  <div className="text-neutral-400 font-bold text-[11px]">Detection Counts at this Operating Point:</div>
                  <div className="grid grid-cols-3 gap-2 text-[11px]">
                    <div className="text-emerald-400">True Positives: <b>{selectedThresholdPoint?.true_positives}</b></div>
                    <div className="text-amber-400">False Positives: <b>{selectedThresholdPoint?.false_positives}</b></div>
                    <div className="text-red-400">False Negatives: <b>{selectedThresholdPoint?.false_negatives}</b></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: PER-CLASS BREAKDOWN TABLE */}
        {activeTab === "per_class" && currentBenchmark && (
          <div className="bg-[#121212] border border-white/10 rounded-xl overflow-hidden font-mono text-xs shadow-xl">
            <div className="p-4 border-b border-white/10 flex justify-between items-center bg-[#151515]">
              <span className="font-bold text-white text-sm">Class-Level Precision, Recall, and AP Breakdown</span>
              <span className="text-neutral-500">IoU Threshold = {currentBenchmark.config.iou_threshold}</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead className="bg-[#181818] text-neutral-400 uppercase text-[10px]">
                  <tr>
                    <th className="p-3.5">Class Name</th>
                    <th className="p-3.5">Support (GT)</th>
                    <th className="p-3.5">Predictions</th>
                    <th className="p-3.5">Precision</th>
                    <th className="p-3.5">Recall</th>
                    <th className="p-3.5">F1</th>
                    <th className="p-3.5">AP@50</th>
                    <th className="p-3.5">AP@75</th>
                    <th className="p-3.5">AP@50:95</th>
                    <th className="p-3.5">TP / FP / FN</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {currentBenchmark.per_class_metrics.map((cls) => (
                    <tr key={cls.class_id} className="hover:bg-[#181818] transition-colors">
                      <td className="p-3.5 font-bold text-white">{cls.class_name}</td>
                      <td className="p-3.5 text-amber-400 font-bold">{cls.support}</td>
                      <td className="p-3.5 text-neutral-300">{cls.predictions_count}</td>
                      <td className="p-3.5 text-white font-bold">{(cls.precision * 100).toFixed(1)}%</td>
                      <td className="p-3.5 text-white font-bold">{(cls.recall * 100).toFixed(1)}%</td>
                      <td className="p-3.5 text-purple-400 font-bold">{cls.f1.toFixed(2)}</td>
                      <td className="p-3.5 text-emerald-400 font-bold">{(cls.map50 * 100).toFixed(1)}%</td>
                      <td className="p-3.5 text-cyan-400 font-bold">{(cls.map75 * 100).toFixed(1)}%</td>
                      <td className="p-3.5 text-blue-400 font-bold">{(cls.map50_95 * 100).toFixed(1)}%</td>
                      <td className="p-3.5 text-neutral-400">
                        <span className="text-emerald-400">{cls.true_positives}</span> /{" "}
                        <span className="text-amber-400">{cls.false_positives}</span> /{" "}
                        <span className="text-red-400">{cls.false_negatives}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TAB 4: RUNTIME LATENCY PROFILER */}
        {activeTab === "runtime" && currentBenchmark && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 font-mono text-xs">
              <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-3 shadow-xl">
                <div className="text-neutral-400 text-[10px] uppercase">Steady-State Throughput</div>
                <div className="text-3xl font-bold text-amber-400">
                  {currentBenchmark.runtime_metrics.throughput_fps.toFixed(1)} <span className="text-sm">FPS</span>
                </div>
                <div className="text-neutral-500 text-[11px]">
                  Evaluated across {currentBenchmark.runtime_metrics.evaluated_iterations} steady-state iterations ({currentBenchmark.runtime_metrics.warmup_iterations} warmup excluded)
                </div>
              </div>

              <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-3 shadow-xl">
                <div className="text-neutral-400 text-[10px] uppercase">Forward Pass Inference Latency</div>
                <div className="text-3xl font-bold text-blue-400">
                  {currentBenchmark.runtime_metrics.inference_ms_mean.toFixed(2)} <span className="text-sm">ms</span>
                </div>
                <div className="text-neutral-500 text-[11px]">
                  Median: {currentBenchmark.runtime_metrics.inference_ms_median.toFixed(2)}ms | p95: {currentBenchmark.runtime_metrics.inference_ms_p95.toFixed(2)}ms
                </div>
              </div>

              <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-3 shadow-xl">
                <div className="text-neutral-400 text-[10px] uppercase">Model Parameter Footprint</div>
                <div className="text-3xl font-bold text-purple-400">
                  {currentBenchmark.runtime_metrics.model_parameters_m || "11.1"} <span className="text-sm">M Params</span>
                </div>
                <div className="text-neutral-500 text-[11px]">
                  Checkpoint Size: {currentBenchmark.runtime_metrics.model_size_mb || "22.5"} MB
                </div>
              </div>
            </div>

            {/* Latency Breakdown Bar Chart */}
            <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4 font-mono text-xs shadow-xl">
              <h3 className="font-semibold text-white flex items-center gap-2 border-b border-white/10 pb-3">
                <Cpu className="w-4 h-4 text-emerald-400" />
                Multi-Stage Pipeline Latency Decomposition ({currentBenchmark.runtime_metrics.device_name})
              </h3>

              <div className="space-y-4">
                <div className="w-full h-8 bg-[#181818] rounded-xl overflow-hidden flex border border-white/5">
                  <div
                    style={{ width: `${(currentBenchmark.runtime_metrics.preprocess_ms_mean / currentBenchmark.runtime_metrics.total_latency_ms_mean) * 100}%` }}
                    className="bg-blue-500 h-full flex items-center justify-center text-[10px] text-white font-bold"
                    title="Preprocessing"
                  >
                    Prep
                  </div>
                  <div
                    style={{ width: `${(currentBenchmark.runtime_metrics.inference_ms_mean / currentBenchmark.runtime_metrics.total_latency_ms_mean) * 100}%` }}
                    className="bg-emerald-500 h-full flex items-center justify-center text-[10px] text-white font-bold"
                    title="Forward Inference"
                  >
                    Inference ({currentBenchmark.runtime_metrics.inference_ms_mean.toFixed(1)}ms)
                  </div>
                  <div
                    style={{ width: `${(currentBenchmark.runtime_metrics.postprocess_ms_mean / currentBenchmark.runtime_metrics.total_latency_ms_mean) * 100}%` }}
                    className="bg-purple-500 h-full flex items-center justify-center text-[10px] text-white font-bold"
                    title="Postprocessing / NMS"
                  >
                    NMS
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-4 text-center">
                  <div className="p-3 bg-[#181818] rounded border border-white/5">
                    <span className="text-neutral-500 text-[10px]">Preprocessing</span>
                    <div className="text-white font-bold">{currentBenchmark.runtime_metrics.preprocess_ms_mean.toFixed(2)}ms (p95: {currentBenchmark.runtime_metrics.preprocess_ms_p95.toFixed(2)}ms)</div>
                  </div>
                  <div className="p-3 bg-[#181818] rounded border border-white/5">
                    <span className="text-neutral-500 text-[10px]">Forward Pass Inference</span>
                    <div className="text-emerald-400 font-bold">{currentBenchmark.runtime_metrics.inference_ms_mean.toFixed(2)}ms (p95: {currentBenchmark.runtime_metrics.inference_ms_p95.toFixed(2)}ms)</div>
                  </div>
                  <div className="p-3 bg-[#181818] rounded border border-white/5">
                    <span className="text-neutral-500 text-[10px]">Postprocessing / NMS</span>
                    <div className="text-purple-400 font-bold">{currentBenchmark.runtime_metrics.postprocess_ms_mean.toFixed(2)}ms (p95: {currentBenchmark.runtime_metrics.postprocess_ms_p95.toFixed(2)}ms)</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: MODEL COMPARISON */}
        {activeTab === "comparison" && comparisonResult && (
          <div className="space-y-6">
            {/* Scientific Control & Regression Status Banner */}
            <div className={`p-4 rounded-xl border font-mono text-xs flex justify-between items-center ${
              comparisonResult.regression_status === "IMPROVED"
                ? "bg-emerald-950/30 border-emerald-500/40 text-emerald-300"
                : comparisonResult.regression_status === "REGRESSION"
                ? "bg-red-950/30 border-red-500/40 text-red-300"
                : "bg-blue-950/30 border-blue-500/40 text-blue-300"
            }`}>
              <div className="space-y-1">
                <div className="font-bold text-sm flex items-center gap-2">
                  {comparisonResult.regression_status === "IMPROVED" && <TrendingUp className="w-5 h-5 text-emerald-400" />}
                  {comparisonResult.regression_status === "REGRESSION" && <TrendingDown className="w-5 h-5 text-red-400" />}
                  Comparison Status: {comparisonResult.regression_status}
                </div>
                <div className="text-neutral-400 text-[11px]">
                  {comparisonResult.regression_notes.join(" ")}
                </div>
              </div>

              <span className="px-3 py-1 rounded bg-black/40 border border-white/10 font-bold">
                {comparisonResult.is_directly_comparable ? "✓ Valid Controlled Comparison" : "⚠ Incomparable"}
              </span>
            </div>

            {/* Comparison Table */}
            <div className="bg-[#121212] border border-white/10 rounded-xl overflow-hidden font-mono text-xs shadow-xl">
              <table className="w-full text-left">
                <thead className="bg-[#181818] text-neutral-400 uppercase text-[10px]">
                  <tr>
                    <th className="p-3.5">Metric</th>
                    <th className="p-3.5">{comparisonResult.baseline_benchmark.model_name} (Baseline)</th>
                    <th className="p-3.5">{comparisonResult.candidate_benchmark.model_name} (Candidate)</th>
                    <th className="p-3.5">Absolute Delta ($\Delta$)</th>
                    <th className="p-3.5">Relative Change (%)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {Object.entries(comparisonResult.metric_deltas).map(([key, d]) => {
                    const isPositive = d.delta_abs > 0;
                    const isGood = key === "latency_ms" ? !isPositive : isPositive;
                    return (
                      <tr key={key} className="hover:bg-[#181818]">
                        <td className="p-3.5 font-bold text-white uppercase">{key}</td>
                        <td className="p-3.5 text-neutral-300 font-bold">{d.baseline}</td>
                        <td className="p-3.5 text-white font-bold">{d.candidate}</td>
                        <td className={`p-3.5 font-bold ${isGood ? "text-emerald-400" : "text-red-400"}`}>
                          {d.delta_abs > 0 ? `+${d.delta_abs}` : d.delta_abs}
                        </td>
                        <td className={`p-3.5 font-bold ${isGood ? "text-emerald-400" : "text-red-400"}`}>
                          {d.delta_rel_pct > 0 ? `+${d.delta_rel_pct}%` : `${d.delta_rel_pct}%`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Model Disagreements Section */}
            <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4 font-mono text-xs shadow-xl">
              <h3 className="font-semibold text-white flex items-center justify-between border-b border-white/10 pb-3">
                <span className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-amber-400" />
                  Model Prediction Disagreements & Transitions
                </span>
                <span className="text-neutral-500">
                  Fixed: <b className="text-emerald-400">{comparisonResult.failure_transitions.fixed_failures || 0}</b> | New: <b className="text-red-400">{comparisonResult.failure_transitions.new_failures || 0}</b>
                </span>
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {comparisonResult.disagreement_samples.map((dis, idx) => (
                  <div key={idx} className="p-3.5 bg-[#181818] rounded-lg border border-white/5 space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-blue-400 font-bold">Sample: {dis.image_id}</span>
                      {dis.confidence && <span className="text-neutral-500">Conf: {(dis.confidence * 100).toFixed(1)}%</span>}
                    </div>
                    <div className="text-neutral-300 text-[11px]">{dis.observation}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 6: FAILURE GALLERY */}
        {activeTab === "failures" && (
          <div className="space-y-4">
            {/* Filter Bar */}
            <div className="flex flex-wrap gap-2 items-center bg-[#121212] border border-white/10 rounded-xl p-3 font-mono text-xs">
              <span className="text-neutral-400 text-[11px]">Filter Failure Category:</span>
              {["ALL", "FALSE_POSITIVE", "FALSE_NEGATIVE", "MISCLASSIFICATION", "POOR_LOCALIZATION", "LOW_CONFIDENCE"].map((cat) => (
                <button
                  key={cat}
                  onClick={() => setFilterErrorType(cat)}
                  className={`px-3 py-1 rounded-lg text-xs transition-all ${
                    filterErrorType === cat
                      ? "bg-red-600 text-white font-bold"
                      : "bg-[#181818] hover:bg-[#252525] text-neutral-400 border border-white/5"
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>

            {/* Failures Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredFailures.map((fail, idx) => (
                <div
                  key={idx}
                  className="bg-[#121212] border border-red-500/20 rounded-xl p-4 space-y-3 font-mono text-xs shadow-xl"
                >
                  <div className="flex justify-between items-center border-b border-white/10 pb-2">
                    <span className="px-2 py-0.5 rounded bg-red-950/50 text-red-400 font-bold border border-red-500/30 text-[10px]">
                      {fail.error_type}
                    </span>
                    <span className="text-neutral-500 text-[10px]">Image: {fail.image_id}</span>
                  </div>

                  {/* Visual Box Simulation */}
                  <div className="aspect-video bg-[#0c0c0c] rounded-lg border border-white/5 flex flex-col items-center justify-center p-3 relative overflow-hidden">
                    <div className="absolute inset-0 bg-[radial-gradient(#1f1f1f_1px,transparent_1px)] [background-size:12px_12px] opacity-40" />
                    <div className="z-10 text-center space-y-1">
                      <div className="text-xs font-bold text-white">{fail.image_id}.jpg</div>
                      <div className="text-[10px] text-neutral-400">
                        GT: <b className="text-emerald-400">{fail.ground_truth_class || "None"}</b> | Pred: <b className="text-amber-400">{fail.predicted_class || "None"}</b>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-1 text-[11px] text-neutral-400">
                    {fail.confidence !== undefined && (
                      <div className="flex justify-between">
                        <span>Confidence:</span>
                        <span className="text-white">{(fail.confidence * 100).toFixed(1)}%</span>
                      </div>
                    )}
                    {fail.iou !== undefined && (
                      <div className="flex justify-between">
                        <span>IoU with GT:</span>
                        <span className="text-cyan-400">{(fail.iou * 100).toFixed(1)}%</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 7: PROGRESSION HISTORY */}
        {activeTab === "history" && (
          <div className="bg-[#121212] border border-white/10 rounded-xl p-5 space-y-4 font-mono text-xs shadow-xl">
            <h3 className="font-semibold text-white flex items-center justify-between border-b border-white/10 pb-3">
              <span className="flex items-center gap-2">
                <History className="w-4 h-4 text-purple-400" />
                Model Iteration Performance Progression
              </span>
              <span className="text-neutral-500">{history.length} Milestone Benchmarks</span>
            </h3>

            <div className="space-y-3">
              {history.map((h, idx) => (
                <div
                  key={h.benchmark_id}
                  className="p-4 bg-[#181818] border border-white/5 rounded-xl flex flex-wrap justify-between items-center gap-4 hover:border-purple-500/40 transition-all"
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-white font-bold text-sm">{h.model_name} (v{h.model_version})</span>
                      {h.is_baseline && (
                        <span className="px-2 py-0.5 rounded bg-blue-900/40 text-blue-300 text-[10px] font-bold">
                          Baseline
                        </span>
                      )}
                    </div>
                    <div className="text-neutral-500 text-[10px]">
                      {new Date(h.timestamp).toLocaleString()} | Dataset: {h.dataset_version}
                    </div>
                  </div>

                  <div className="flex items-center gap-6 text-neutral-300">
                    <div>
                      <span className="text-neutral-500 text-[10px] block">mAP@50:95</span>
                      <span className="text-blue-400 font-bold">{(h.map50_95 * 100).toFixed(1)}%</span>
                    </div>

                    <div>
                      <span className="text-neutral-500 text-[10px] block">mAP@50</span>
                      <span className="text-emerald-400 font-bold">{(h.map50 * 100).toFixed(1)}%</span>
                    </div>

                    <div>
                      <span className="text-neutral-500 text-[10px] block">Throughput</span>
                      <span className="text-amber-400 font-bold">{h.throughput_fps.toFixed(1)} FPS</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Reproducibility & Lineage Drawer */}
      {showReproducibilityDrawer && currentBenchmark && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-end">
          <div className="bg-[#121212] border-l border-white/10 w-full max-w-lg h-full p-6 space-y-5 font-mono text-xs overflow-y-auto">
            <div className="flex justify-between items-center border-b border-white/10 pb-3">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <GitCommit className="w-4 h-4 text-purple-400" />
                Benchmark Reproducibility Telemetry
              </h3>
              <button onClick={() => setShowReproducibilityDrawer(false)} className="text-neutral-400 hover:text-white">
                ✕
              </button>
            </div>

            <div className="space-y-4">
              <div className="p-3.5 bg-[#181818] rounded-lg border border-white/5 space-y-1.5">
                <div className="text-neutral-500 text-[10px]">Dataset Cryptographic Fingerprint (SHA-256)</div>
                <div className="text-cyan-400 font-bold break-all">
                  {currentBenchmark.dataset_snapshot.dataset_fingerprint}
                </div>
              </div>

              <div className="p-3.5 bg-[#181818] rounded-lg border border-white/5 space-y-1.5">
                <div className="text-neutral-500 text-[10px]">Git Commit SHA</div>
                <div className="text-purple-400 font-bold">
                  {currentBenchmark.reproducibility.git_commit_sha || "2e89528"}
                </div>
              </div>

              <div className="p-3.5 bg-[#181818] rounded-lg border border-white/5 space-y-1.5">
                <div className="text-neutral-500 text-[10px]">Hardware & Platform</div>
                <div className="text-white">
                  OS: {currentBenchmark.reproducibility.os_platform || "macOS"}  
                  <br />
                  CPU: {currentBenchmark.reproducibility.cpu_architecture || "ARM64"}  
                  <br />
                  Python: {currentBenchmark.reproducibility.python_version || "3.11"}
                </div>
              </div>

              <div className="p-3.5 bg-[#181818] rounded-lg border border-white/5 space-y-1.5">
                <div className="text-neutral-500 text-[10px]">Random Seed</div>
                <div className="text-amber-400 font-bold">
                  Seed: {currentBenchmark.config.random_seed}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
